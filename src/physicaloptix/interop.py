"""Live propagation behind the shared coronagraph image contract.

Each adapter is built at one spectrum. Serving wavelengths select no new
propagation; use one adapter per band in MultiBandCoronagraph for that purpose.
Time is in days from the scene epoch; this static model ignores its value.
"""

import dataclasses
import enum
import hashlib
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.leaves import HostGrid
from hwoutils.precision import like
from hwoutils.provenance import (
    Digest,
    StampMismatchError,
    Stamps,
    build_stamps,
    compare_stamps,
    qualify,
    stamp_value,
)
from optixstuff.apertures import DEFAULT_APERTURE_RULE, Circular, HalfMaximum
from optixstuff.coronagraph import (
    AbstractImagingCoronagraph,
    ProductGrids,
    SourcePSFs,
    on_grid,
)
from optixstuff.grids import Image, PixelGrid
from optixstuff.performance import SampledProfile, performance_curves
from optixstuff.spectral import Band, Quadrature, same_wavelength

from physicaloptix._sky import (
    _require_inside_pupil_nyquist,
    entrance_energy,
    uniform_sky_image,
)
from physicaloptix._version import __version__
from physicaloptix.core import Field, PlaneKind
from physicaloptix.sources import _fixed_angle_camera

CACHED_READERS = (
    "uniform_sky_profile",
    "performance_record",
    "curves",
    "uniform_sky_response",
    "uniform_sky_map",
)
LIVE_READERS = ("stellar_map", "off_axis_psf", "source_psfs")

_FROZEN_FIELDS = ("core_path", "input_field")


def _native_psf(core_path, science, source, energy):
    """Propagate a source to the native sky-angle grid.

    Args:
        core_path: Optical train ending at the Lyot pupil.
        science: Camera sampling fixed sky angles on the native focal grid.
        source: Entrance field with the source tilt applied.
        energy: Incident band-averaged power of the untilted entrance field.

    Returns:
        Band-limited point density in incident flux per square reference lambda/D.
    """
    lyot, _ = core_path.propagate(source)
    power = science.forward(lyot).intensity()
    return replace(power, data=power.data / energy, quantity="stellar_flux_fraction")


def _record_in_band_frame(record, scale, profile, rule):
    """Express native aperture measurements in the build band's lambda/D.

    Args:
        record: Performance curves measured on the native grid.
        scale: Reference wavelength divided by the build band centre.
        profile: Uniform-sky profile already expressed in the build frame.
        rule: Aperture rule as requested, in build-centre lambda/D.

    Returns:
        The same aperture photometry with converted axes, areas and densities.
    """
    area_scale = scale**2
    return replace(
        record,
        frame=profile.frame,
        rule=rule,
        separation_lod=record.separation_lod * scale,
        stellar_diameters_lod=record.stellar_diameters_lod * scale,
        iwa_lod=record.iwa_lod * scale,
        data_extent_lod=record.data_extent_lod * scale,
        core_area_lod2=record.core_area_lod2 * area_scale,
        core_area_variance_lod2=record.core_area_variance_lod2 * area_scale,
        leakage_density_lod2=record.leakage_density_lod2 / area_scale,
        leakage_density_variance_lod2=record.leakage_density_variance_lod2 / area_scale,
        uniform_sky=profile,
    )


class PathCalibration(eqx.Module):
    """Snapshot of a train's aperture photometry and uniform-sky response.

    Attributes:
        records: Performance curves in build-centre lambda/D for a tagged field.
        sky: Dimensionless uniform-sky transmission image.
        uniform_sky_profile: Annular means of the sky transmission.
        reads: Owner-qualified model paths the snapshot was measured through,
            relative to the owner named ``coronagraph``; empty when unknown.
        built_at: Build facts of the measurement, made with
            ``hwoutils.provenance.build_stamps``; empty when unknown.
    """

    records: tuple
    sky: Image
    uniform_sky_profile: SampledProfile
    reads: tuple[str, ...] = eqx.field(static=True, default=())
    built_at: Stamps = eqx.field(static=True, default=())

    @classmethod
    def _measure(
        cls,
        core_path,
        input_field,
        grid,
        *,
        low_nm,
        max_separation_lod,
        separation_step_lod,
        rules,
        leakage_rule,
        n_sky_screens,
        seed,
        built_at=(),
    ):
        """Measure aperture photometry and uniform-sky transmission.

        Every record carries the radial sky profile, so a record alone gives
        the sky transmission of extended sources.

        Args:
            core_path: Optical train ending at the Lyot pupil.
            input_field: Untilted entrance field with its build spectrum.
            grid: Native science grid in reference-wavelength lambda/D.
            low_nm: Shortest active wavelength in nanometers.
            max_separation_lod: Largest requested separation in lambda/D.
            separation_step_lod: Spacing of the source offsets in lambda/D.
            rules: Aperture rules to measure at the source offsets.
            leakage_rule: Reduction applied to the stellar map.
            n_sky_screens: Number of independent sky illumination screens.
            seed: Seed for drawing those screens.
            built_at: Build facts of this measurement, kept on the snapshot with
                the reads that produced it.

        Returns:
            The train's performance records, sky map and radial sky profile.
        """
        science, place = _fixed_angle_camera(
            input_field,
            grid,
            reference_wavelength_nm=grid.frame.reference_wavelength_nm,
            low_nm=low_nm,
        )
        energy = entrance_energy(input_field)
        step = separation_step_lod
        seps = np.arange(0.0, max_separation_lod + 2 * step, step)
        images = {
            s: _native_psf(core_path, science, place((s, 0.0)), energy) for s in seps
        }
        # A uniform radiance illuminates solid angle. The angular area per
        # native lambda/D cell scales as wavelength squared at each node.
        sky_data = input_field.data / jnp.sqrt(energy)
        if input_field.spectrum is not None:
            wavelengths = input_field.spectrum.nodes_nm[0, :, None, None]
            sky_data = sky_data * (wavelengths / grid.frame.reference_wavelength_nm)
        sky_field = Field(
            sky_data, input_field.grid, input_field.plane, input_field.spectrum
        )
        sky = uniform_sky_image(
            core_path,
            science,
            sky_field,
            n_screens=n_sky_screens,
            seed=seed,
        )
        profile = SampledProfile.from_uniform_sky(sky)
        scale = 1.0
        if input_field.spectrum is not None:
            center_nm = input_field.spectrum.bands[0].center_nm
            scale = grid.frame.reference_wavelength_nm / center_nm

        def native(rule):
            # Rules are stated in build-centre lambda/D; measure on the native grid.
            if scale != 1.0 and isinstance(rule, Circular):
                return replace(rule, radius_lod=rule.radius_lod / scale)
            return rule

        records = tuple(
            performance_curves(
                images.__getitem__,
                images[0.0],
                native(rule),
                separations_lod=seps,
                leakage_rule=leakage_rule,
                stellar_diameter_lod=0.0,
                uniform_sky=profile,
            )
            for rule in rules
        )
        if scale != 1.0:
            profile = replace(
                profile,
                separation_lod=profile.separation_lod * scale,
                frame=replace(grid.frame, reference_wavelength_nm=center_nm),
            )
            records = tuple(
                _record_in_band_frame(record, scale, profile, rule)
                for record, rule in zip(records, rules, strict=True)
            )
        return cls(
            jax.tree.map(jnp.asarray, records),
            sky,
            jax.tree.map(jnp.asarray, profile),
            reads=qualify("coronagraph", _FROZEN_FIELDS),
            built_at=built_at,
        )


def _tree_digest(tree, what) -> Digest:
    """Digest the concrete array leaves of a tree, in tree order.

    Each leaf enters as its dtype string, its shape and its C-order bytes, so
    equal bytes under another dtype or shape give another digest.

    Args:
        tree: Pytree whose leaves are arrays or numbers.
        what: Name of the tree for the error message.

    Returns:
        The SHA-256 digest of the leaves.

    Raises:
        TypeError: A leaf is traced; staleness under trace is decided by the
            declared reads, never by a digest.
    """
    state = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.core.Tracer):
            raise TypeError(
                f"{what} holds a traced leaf; digests need concrete values, so "
                "decide staleness from the declared reads"
            )
        array = np.ascontiguousarray(np.asarray(leaf))
        state.update(f"{array.dtype.str}:{array.shape}:".encode())
        state.update(array.tobytes())
    return Digest(state.hexdigest())


def _structure(value) -> str:
    """Encode the operator-defining structure of a value as deterministic text.

    Dataclass and Equinox fields enter by name, including every static field,
    and a grid enters through its full affine description. An array enters by
    dtype and shape only: its values belong to the numerical digests.

    Args:
        value: Path element, grid, field metadata or scalar.

    Returns:
        The encoding.

    Raises:
        TypeError: The value has no deterministic encoding.
    """
    if value is None or isinstance(value, (bool, str)):
        return repr(value)
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"float:{float(value).hex()}"
    if isinstance(value, complex):
        return f"complex:{value.real.hex()}:{value.imag.hex()}"
    if isinstance(value, enum.Enum):
        return f"{type(value).__qualname__}.{value.name}"
    if isinstance(value, (jax.Array, np.ndarray)):
        return f"array:{np.dtype(value.dtype).str}:{tuple(value.shape)}"
    if isinstance(value, HostGrid):
        return stamp_value(value)
    if isinstance(value, (tuple, list)):
        return "(" + ",".join(_structure(item) for item in value) + ")"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        items = ",".join(
            f"{f.name}={_structure(getattr(value, f.name))}"
            for f in dataclasses.fields(value)
        )
        cls = type(value)
        return f"{cls.__module__}.{cls.__qualname__}({items})"
    if isinstance(value, (np.dtype, type)) or callable(value):
        name = getattr(value, "__qualname__", None) or str(value)
        return f"{getattr(value, '__module__', '')}.{name}"
    raise TypeError(f"no deterministic structure for {type(value).__name__}")


def _field_structure(field) -> str:
    """Encode an entrance field without its values or spectrum nodes.

    Args:
        field: The entrance field.

    Returns:
        Its class, grid, plane, leading axes, spectrum class, band metadata and
        quadrature layout.
    """
    spectrum = field.spectrum
    bands = () if spectrum is None else spectrum.bands
    spectrum_class = "none" if spectrum is None else type(spectrum).__qualname__
    return (
        f"{type(field).__qualname__}(grid={_structure(field.grid)},"
        f"plane={_structure(field.plane)},axes={_structure(field.leading_axes)},"
        f"spectrum={spectrum_class},bands={_structure(bands)},"
        f"pieces={_structure(() if spectrum is None else spectrum.pieces)})"
    )


def _grid_facts(grid) -> HostGrid:
    """The shape, centre, basis and frame numbers of a grid as one host vector."""
    frame = grid.frame
    frame_numbers = (
        () if frame is None else (frame.reference_wavelength_nm, frame.diameter_m)
    )
    return HostGrid(
        [*grid.shape, grid.center_x, grid.center_y]
        + [x for row in grid.basis for x in row]
        + list(frame_numbers)
    )


def _calibration_stamps(
    core_path,
    input_field,
    grid,
    *,
    wavelength_nm,
    low_nm,
    max_separation_lod,
    separation_step_lod,
    fov_margin_lod,
    rules,
    leakage_rule,
    n_sky_screens,
    seed,
    pupil_id,
) -> Stamps:
    """Build facts of everything a calibration depends on.

    Raises:
        TypeError: A leaf of the path or the field is traced.
    """
    spectrum = input_field.spectrum
    bands = () if spectrum is None else spectrum.bands
    return build_stamps(
        producer="physicaloptix.PathCoronagraph",
        producer_version=__version__,
        core_path=_tree_digest(core_path, "core_path"),
        input_field=_tree_digest(input_field.data, "input_field"),
        path_structure=Digest(
            hashlib.sha256(
                f"{_structure(core_path)}|{_field_structure(input_field)}|"
                f"{_structure(grid)}".encode()
            ).hexdigest()
        ),
        spectrum="none"
        if spectrum is None
        else _tree_digest((spectrum.nodes_nm, spectrum.weights_nm), "spectrum"),
        bands=tuple(float(x) for band in bands for x in (band.low_nm, band.high_nm)),
        wavelength_nm=wavelength_nm,
        low_nm=low_nm,
        grid=_grid_facts(grid),
        max_separation_lod=max_separation_lod,
        separation_step_lod=separation_step_lod,
        fov_margin_lod=fov_margin_lod,
        rules=tuple(repr(rule) for rule in rules),
        leakage_rule=leakage_rule,
        n_sky_screens=n_sky_screens,
        seed=seed,
        pupil_id=pupil_id,
    )


def _has_tracer(value) -> bool:
    """Whether a value or any nested dataclass field or sequence entry is traced."""
    if isinstance(value, jax.core.Tracer):
        return True
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return any(
            _has_tracer(getattr(value, field.name))
            for field in dataclasses.fields(value)
        )
    if isinstance(value, (tuple, list)):
        return any(_has_tracer(item) for item in value)
    return False


def _inside_nyquist(offset, limit, field):
    """Test the open pupil-tilt interval before casting offsets to the field."""
    if not any(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(offset)):
        value = np.asarray(offset)
        return np.abs(value.astype(np.result_type(value, float), copy=False)) < limit
    value = jnp.asarray(offset)
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = like(value, field)
    magnitude = jnp.abs(value)
    edge = magnitude.dtype.type(limit)
    # If the host limit rounds down, that representable value is still inside.
    return magnitude <= edge if float(edge) < limit else magnitude < edge


class PathCoronagraph(AbstractImagingCoronagraph):
    """Live optical train with a fixed sky-angle grid and a calibration snapshot.

    Attributes:
        core_path: Optical path from the entrance field to the Lyot pupil plane.
        input_field: Entrance field, optionally with a one-band spectrum.
        grid: Native sky-angle grid in reference-wavelength lambda/D.
        wavelength_nm: Monochromatic build wavelength when supplied.
        low_nm: Shortest active build wavelength used to check native sampling,
            in nm. Single-wavelength models also use it to check serving wavelengths.
        max_separation_lod: Largest separation requested for aperture photometry.
        separation_step_lod: Spacing of the calibrated source offsets in lambda/D.
        fov_margin_lod: Extra half field of view beyond the largest separation.
        rules: Aperture rules prepared in the calibration.
        leakage_rule: Method for reducing stellar leakage to a curve.
        n_sky_screens: Number of independent screens in the sky estimate.
        seed: Seed used to reproduce the sky illumination.
        calibrated_pupil_id: Identifier of the pupil prescription the model was
            built on, or None when that pupil declared none.
        calibration: Snapshot of the train's performance records and sky products,
            with the reads and build facts it froze.
    """

    core_path: object
    input_field: Field
    grid: PixelGrid
    wavelength_nm: float | None = eqx.field(static=True)
    low_nm: float = eqx.field(static=True)
    max_separation_lod: float = eqx.field(static=True)
    separation_step_lod: float = eqx.field(static=True)
    fov_margin_lod: float = eqx.field(static=True)
    rules: tuple = eqx.field(static=True)
    leakage_rule: str = eqx.field(static=True)
    n_sky_screens: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    calibrated_pupil_id: str | None = eqx.field(static=True)
    calibration: PathCalibration

    @property
    def bandpass(self) -> Band | None:
        """Entrance wavelength band, or None for an untagged field."""
        spectrum = self.input_field.spectrum
        return None if spectrum is None else spectrum.bands[0]

    @property
    def band_averaged(self) -> bool:
        """Return whether the build spectrum spans a nonzero wavelength interval."""
        band = self.bandpass
        return band is not None and band.width_nm > 0

    def scale_wavelength_nm(self, wavelength_nm, band_center_nm):
        """Scale the reference grid by the observing-to-build wavelength ratio.

        A mono response scales with the node wavelength. A band average scales
        once with the observing band's center relative to its build band center.

        Args:
            wavelength_nm: Node wavelength in nm.
            band_center_nm: Center of the exposure's observing band in nm.

        Returns:
            The wavelength setting the angular scale of the native lambda/D grid.
        """
        scale_nm = super().scale_wavelength_nm(wavelength_nm, band_center_nm)
        band = self.bandpass
        if band is None:
            return scale_nm
        return scale_nm * (self.frame.reference_wavelength_nm / band.center_nm)

    @classmethod
    def from_path(
        cls,
        core_path,
        input_field,
        *,
        pupil,
        max_separation_lod,
        reference_wavelength_nm,
        pixel_scale_lod=0.25,
        separation_step_lod=0.5,
        fov_margin_lod=4.0,
        wavelength_nm=None,
        rules=(),
        n_sky_screens=1000,
        seed=0,
        leakage_rule="aperture_mean",
    ):
        """Compile a point-star model from an entrance-to-Lyot path.

        pupil.frame(reference_wavelength_nm) defines the native angular
        grid, and max_separation_lod sets how far out the records are sampled.
        wavelength_nm binds monochromatic OPD screens when supplied.
        DEFAULT_APERTURE_RULE and HalfMaximum(0.5) are always prepared; rules adds
        further immutable records. Finite stellar diameters are not modeled.
        n_sky_screens independent complex pupil screens (drawn with seed)
        estimate the uniform-sky map and its annular means.

        Args:
            core_path: Optical path from the entrance field to the Lyot pupil plane; its
                last stage must end in the pupil plane.
            input_field: Entrance field; a field without a spectrum is made
                monochromatic at wavelength_nm.
            pupil: Pupil geometry; its frame at the reference wavelength defines the
                native angular grid, and its ``pupil_id`` is recorded as the
                prescription the model was calibrated on.
            max_separation_lod: Largest separation in lambda/D at which the records are
                sampled.
            reference_wavelength_nm: Reference wavelength in nanometers of the angular
                frame.
            pixel_scale_lod: Focal pixel size in lambda/D.
            separation_step_lod: Spacing in lambda/D of the sampled source separations.
            fov_margin_lod: Extra half field of view in lambda/D beyond
                max_separation_lod.
            wavelength_nm: Wavelength in nanometers that binds monochromatic OPD screens
                when given.
            rules: Extra aperture rules to prepare, in addition to the default and
                half-maximum rules. With a build spectrum, circular radii and all
                record geometry use build-centre lambda/D; otherwise the native
                reference lambda/D.
            n_sky_screens: Number of independent screens used to estimate the
                uniform-sky map.
            seed: Seed of the screen sequence.
            leakage_rule: Rule that reduces leakage to a curve, passed to the
                performance curves.

        Returns:
            The compiled point-star model.

        Raises:
            ValueError: The path does not end in a pupil plane, or the native grid
                exceeds the pupil Nyquist radius or intensity sampling limit.
        """
        if wavelength_nm is not None and input_field.spectrum is None:
            input_field = Field(
                input_field.data[None],
                input_field.grid,
                input_field.plane,
                Quadrature.monochromatic(wavelength_nm),
            )
        last = core_path.stages[-1].op
        plane = getattr(last, "plane_out", None) or getattr(last, "plane", None)
        if plane is not PlaneKind.PUPIL:
            raise ValueError("core_path must end in the pupil (Lyot) plane")
        npix = 2 * int(np.ceil((max_separation_lod + fov_margin_lod) / pixel_scale_lod))
        grid = PixelGrid.focal(
            npix, pixel_scale_lod, pupil.frame(reference_wavelength_nm)
        )
        spectrum = input_field.spectrum
        low_nm = float(reference_wavelength_nm)
        if spectrum is not None:
            active = np.asarray(spectrum.active[0])
            low_nm = float(np.min(np.asarray(spectrum.nodes_nm[0])[active]))
        _require_inside_pupil_nyquist(grid, input_field, low_nm=low_nm)
        extent = input_field.grid.shape[-1] * input_field.grid.step
        max_pitch = low_nm / (2 * extent * reference_wavelength_nm)
        if pixel_scale_lod > max_pitch:
            raise ValueError(
                f"native pitch must be at most {max_pitch:.17g} lambda/D at "
                f"{low_nm:g} nm (intensity Nyquist limit)"
            )

        options = dict(
            low_nm=low_nm,
            max_separation_lod=float(max_separation_lod),
            separation_step_lod=float(separation_step_lod),
            rules=tuple(
                dict.fromkeys((DEFAULT_APERTURE_RULE, HalfMaximum(0.5), *rules))
            ),
            leakage_rule=leakage_rule,
            n_sky_screens=n_sky_screens,
            seed=seed,
        )
        facts = dict(
            wavelength_nm=None if wavelength_nm is None else float(wavelength_nm),
            fov_margin_lod=float(fov_margin_lod),
            pupil_id=pupil.pupil_id,
        )
        built_at = _calibration_stamps(core_path, input_field, grid, **facts, **options)
        return cls(
            core_path=core_path,
            input_field=input_field,
            grid=grid,
            wavelength_nm=facts["wavelength_nm"],
            fov_margin_lod=facts["fov_margin_lod"],
            calibrated_pupil_id=facts["pupil_id"],
            calibration=PathCalibration._measure(
                core_path, input_field, grid, built_at=built_at, **options
            ),
            **options,
        )

    def recalibrated(self):
        """Recompute the calibration from the current train and stored seed.

        Returns:
            A copy with freshly measured aperture curves and sky transmission.
        """
        calibration = PathCalibration._measure(
            self.core_path,
            self.input_field,
            self.grid,
            low_nm=self.low_nm,
            max_separation_lod=self.max_separation_lod,
            separation_step_lod=self.separation_step_lod,
            rules=self.rules,
            leakage_rule=self.leakage_rule,
            n_sky_screens=self.n_sky_screens,
            seed=self.seed,
            built_at=self._current_stamps(),
        )
        return eqx.tree_at(lambda model: model.calibration, self, calibration)

    @property
    def pupil_id(self) -> str | None:
        """Identifier of the pupil prescription the model was calibrated on."""
        return self.calibrated_pupil_id

    def for_telescope(self, telescope) -> "PathCoronagraph":
        """Serve a telescope without rebinding the path.

        A complete path contains its own entrance pupil, so it serves the
        calibrated baseline and nothing is rebound. The checks are the default
        ones: the frame diameter must equal the pupil diameter, and two known,
        different pupil identifiers refuse the pairing. The call does no array
        work, so it is legal under ``jit``, ``vmap`` and ``grad``.

        Args:
            telescope: The ``Telescope`` that supplies the pupil.

        Returns:
            This model.

        Raises:
            ValueError: The frame diameter differs from the pupil diameter.
            StampMismatchError: Both pupil identifiers are known and differ.
        """
        super().for_telescope(telescope)
        return self

    def _current_stamps(self) -> Stamps:
        """Build facts of the path and settings as they are now (host only)."""
        return _calibration_stamps(
            self.core_path,
            self.input_field,
            self.grid,
            wavelength_nm=self.wavelength_nm,
            low_nm=self.low_nm,
            max_separation_lod=self.max_separation_lod,
            separation_step_lod=self.separation_step_lod,
            fov_margin_lod=self.fov_margin_lod,
            rules=self.rules,
            leakage_rule=self.leakage_rule,
            n_sky_screens=self.n_sky_screens,
            seed=self.seed,
            pupil_id=self.calibrated_pupil_id,
        )

    def calibration_stamps(self) -> Stamps:
        """The build facts the calibration froze.

        Returns:
            Sorted build stamps, empty when they are unknown.
        """
        return self.calibration.built_at

    def frozen_reads(self, reader: str) -> tuple[str, ...]:
        """Owner-qualified paths a reader takes from the calibration snapshot.

        A cached reader returns the snapshot's reads; a live reader follows the
        path and returns an empty tuple. Pass the result to
        ``hwoutils.provenance.bind_reads`` and ``stale_reads``.

        Args:
            reader: One of ``CACHED_READERS`` or ``LIVE_READERS``.

        Returns:
            The reads, relative to the owner named ``coronagraph``.

        Raises:
            ValueError: The reader is not known.
        """
        if reader in LIVE_READERS:
            return ()
        if reader in CACHED_READERS:
            return self.calibration.reads
        raise ValueError(
            f"unknown reader {reader!r}; cached readers are {CACHED_READERS}, "
            f"live readers are {LIVE_READERS}"
        )

    def stale_calibration(self) -> tuple[str, ...]:
        """Labels whose present value differs from what the calibration froze.

        Host only. A calibration without stamps, or a label unknown on either
        side, is unverified and never reported.

        Returns:
            Sorted mismatched labels, empty when the snapshot matches the path.

        Raises:
            TypeError: A leaf of the path or the entrance field is traced.
        """
        comparison = compare_stamps(self.calibration_stamps(), self._current_stamps())
        return tuple(label for label, _, _ in comparison.mismatched)

    def require_fresh_calibration(self, *, reader: str) -> None:
        """Refuse a cached read of a calibration that no longer matches the path.

        A live reader follows the current path and always passes.

        Args:
            reader: One of ``CACHED_READERS`` or ``LIVE_READERS``.

        Raises:
            ValueError: The reader is not known.
            StampMismatchError: A cached reader is asked of a stale calibration;
                the message names the stale labels and ``recalibrated()``.
        """
        if not self.frozen_reads(reader):
            return
        stale = self.stale_calibration()
        if stale:
            raise StampMismatchError(
                f"{reader} reads a calibration measured on a different path; "
                f"stale facts: {', '.join(stale)}. Call recalibrated() to "
                "measure the current path."
            )

    @property
    def uniform_sky_profile(self):
        """Annular means from the current calibration snapshot."""
        return self.calibration.uniform_sky_profile

    @property
    def performance_record(self):
        """Photometry for the default aperture in the calibration snapshot."""
        return self.calibration.records[0]

    def curves(self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule="aperture_mean"):
        """Select an eagerly built record; unprepared rules raise.

        Args:
            rule: Prepared aperture rule, as given at construction.
            leakage_rule: Leakage rule of the record.

        Returns:
            The prepared performance record for that rule pair.

        Raises:
            ValueError: The aperture and leakage rule pair was not prepared.
        """
        for prepared, record in zip(self.rules, self.calibration.records, strict=True):
            if prepared == rule and record.leakage_rule == leakage_rule:
                return record
        raise ValueError("No prepared record for this rule and leakage_rule")

    @property
    def frame(self):
        """Angular frame of the native image grid."""
        return self.grid.frame

    def coronagraph_at(self, wavelength_nm):
        """Select the live model at a serving wavelength.

        Args:
            wavelength_nm: Serving wavelength in nanometers.

        Returns:
            This model, whose propagation uses its build spectrum.

        Raises:
            ValueError: A single-wavelength model is served below low_nm.
        """
        self._check_wavelength(wavelength_nm)
        return self

    def at_wavelength(self, wavelength_nm):
        """Return the product grids of the build spectrum.

        Args:
            wavelength_nm: Serving wavelength in nanometers.

        Returns:
            Product grids, with the native grid for the stellar and source images and
            the sky map grid for the uniform sky.

        Raises:
            ValueError: A single-wavelength model is served below low_nm.
        """
        self._check_wavelength(wavelength_nm)
        return ProductGrids(self.grid, self.grid, self.calibration.sky.grid)

    def uniform_sky_response(self, separation_lod):
        """Annular mean of the uniform-sky map in the performance frame.

        Args:
            separation_lod: Numeric separations in the performance record's lambda/D.

        Returns:
            Annular mean of the uniform-sky map at those separations, dimensionless.

        Raises:
            TypeError: The separations carry units.
        """
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return self.uniform_sky_profile.evaluate(
            like(separation_lod, self.input_field.data)
        )

    def _camera(self):
        """Prepare propagation in the native reference frame.

        Returns:
            The camera and the function placing sources in that frame.
        """
        return _fixed_angle_camera(
            self.input_field,
            self.grid,
            reference_wavelength_nm=self.frame.reference_wavelength_nm,
            low_nm=self.low_nm,
        )

    def _check_wavelength(self, wavelength_nm):
        """Check the lower sampling limit of a single-wavelength model.

        Args:
            wavelength_nm: Requested serving wavelength in nanometers.

        Raises:
            ValueError: A single-wavelength model is served below low_nm.
        """
        if (
            not self.band_averaged
            and not isinstance(wavelength_nm, jax.core.Tracer)
            and float(wavelength_nm) < self.low_nm
            and not same_wavelength(wavelength_nm, self.low_nm)
        ):
            raise ValueError(
                f"wavelength {float(wavelength_nm):g} nm is below the shortest "
                f"active wavelength {self.low_nm:g} nm"
            )

    def _psf(self, science, source, energy, grid):
        """Read a live source on its native grid or over detector footprints.

        Args:
            science: Native fixed-angle camera.
            source: Tilted entrance field.
            energy: Incident band-averaged entrance power.
            grid: Detector grid, or None for the native point density.

        Returns:
            Fraction of the incident flux per native area or per detector cell.
        """
        native = _native_psf(self.core_path, science, source, energy)
        return native if grid is None else on_grid(native, grid)

    def stellar_map(self, wavelength_nm, stellar_diameter_lod, *, grid=None):
        """Point-star flux fraction; reject concrete or mark traced finite stars NaN.

        Args:
            wavelength_nm: Serving wavelength in nanometers.
            stellar_diameter_lod: Stellar angular diameter in lambda/D; only zero is
                supported, and a traced nonzero value yields NaN.
            grid: Target grid; None keeps the native grid.

        Returns:
            Stellar flux-fraction image of the on-axis star, native or resampled onto
            the target grid.

        Raises:
            ValueError: A single-wavelength model is served below low_nm, or the
                concrete stellar diameter is nonzero.
        """
        if (
            not isinstance(stellar_diameter_lod, jax.core.Tracer)
            and float(stellar_diameter_lod) != 0
        ):
            raise ValueError("finite stellar diameter requires a prepared stellar map")
        self._check_wavelength(wavelength_nm)
        science, place = self._camera()
        image = self._psf(
            science, place((0.0, 0.0)), entrance_energy(self.input_field), grid
        )
        if not isinstance(stellar_diameter_lod, jax.core.Tracer):
            return image
        return replace(
            image,
            data=jnp.where(
                stellar_diameter_lod == 0, image.data, like(jnp.nan, image.data)
            ),
        )

    def off_axis_psf(self, wavelength_nm, separation_lod, grid: PixelGrid) -> Image:
        """Read one source along the positive x axis in detector cells.

        Args:
            wavelength_nm: Serving wavelength in nanometers.
            separation_lod: Source separation in lambda/D.
            grid: Target detector grid.

        Returns:
            Flux fraction per cell, zero for a source outside the pupil domain.
        """
        image = self.source_psfs(wavelength_nm, separation_lod, 0, grid=grid).image
        return replace(image, data=image.data[0], leading_axes=())

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
        """Native point densities or target pixel fractions for each pupil tilt.

        A tilt at or beyond the pupil Nyquist frequency on either axis is
        ambiguous with a different source position: that source has ``valid``
        False and an all-zero image, and the other sources are unaffected.

        Args:
            wavelength_nm: Serving wavelength in nanometers.
            x_lod: Source offsets along x in lambda/D, broadcast with y_lod.
            y_lod: Source offsets along y in lambda/D, broadcast with x_lod.
            grid: Target grid; None keeps the native grid.

        Returns:
            Images of each source and a validity flag per source; invalid sources have
            all-zero images.

        Raises:
            ValueError: A single-wavelength model is served below low_nm.
        """
        self._check_wavelength(wavelength_nm)
        science, place = self._camera()
        energy = entrance_energy(self.input_field)

        def one(x, y):
            return self._psf(science, place((x, y)), energy, grid)

        nyquist_lod = (
            0.5
            / self.input_field.grid.step
            * (self.low_nm / self.frame.reference_wavelength_nm)
        )
        valid = jnp.atleast_1d(
            _inside_nyquist(x_lod, nyquist_lod, self.input_field.data)
            & _inside_nyquist(y_lod, nyquist_lod, self.input_field.data)
        )
        x, y = jnp.broadcast_arrays(
            jnp.atleast_1d(like(x_lod, self.input_field.data)),
            jnp.atleast_1d(like(y_lod, self.input_field.data)),
        )
        images = jax.vmap(one)(jnp.where(valid, x, 0), jnp.where(valid, y, 0))
        keep = valid.reshape((-1,) + (1,) * (images.data.ndim - 1))
        images = replace(
            images, data=jnp.where(keep, images.data, 0), leading_axes=("source",)
        )
        return SourcePSFs(images, valid)

    def uniform_sky_map(self, wavelength_nm, *, grid):
        """Dimensionless extended-illumination transmission on the requested grid.

        Args:
            wavelength_nm: Serving wavelength in nanometers.
            grid: Target grid of the map.

        Returns:
            Dimensionless uniform-sky transmission image on the target grid.

        Raises:
            ValueError: A single-wavelength model is served below low_nm.
        """
        self._check_wavelength(wavelength_nm)
        sky = self.calibration.sky
        # Grid equality is a host comparison; a traced grid is resampled.
        if not _has_tracer(grid) and sky.grid == grid:
            return sky
        return on_grid(sky, grid)
