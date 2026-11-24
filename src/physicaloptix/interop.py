"""Live propagation behind the shared coronagraph image contract.

Each adapter is built at one spectrum. Serving wavelengths select no new
propagation; use one adapter per band in MultiBandCoronagraph for that purpose.
Time is in days from the scene epoch; this static model ignores its value.
"""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from optixstuff.apertures import DEFAULT_APERTURE_RULE, HalfMaximum
from optixstuff.coronagraph import (
    AbstractImagingCoronagraph,
    ProductGrids,
    SourcePSFs,
    on_grid,
)
from optixstuff.grids import Image, PixelGrid
from optixstuff.performance import SampledProfile, performance_curves
from optixstuff.spectral import Band, Quadrature

from physicaloptix._sky import (
    _require_inside_pupil_nyquist,
    entrance_energy,
    uniform_sky_image,
)
from physicaloptix.core import Field, PlaneKind
from physicaloptix.sources import _fixed_angle_camera


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


class PathCalibration(eqx.Module):
    """Snapshot of a train's aperture photometry and uniform-sky response.

    Attributes:
        records: Performance curves for the prepared aperture rules.
        sky: Dimensionless uniform-sky transmission image.
        uniform_sky_profile: Annular means of the sky transmission.
    """

    records: tuple
    sky: Image
    uniform_sky_profile: SampledProfile

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
        profile = jax.tree.map(jnp.asarray, SampledProfile.from_uniform_sky(sky))
        records = tuple(
            performance_curves(
                images.__getitem__,
                images[0.0],
                rule,
                separations_lod=seps,
                leakage_rule=leakage_rule,
                stellar_diameter_lod=0.0,
                uniform_sky=profile,
            )
            for rule in rules
        )
        return cls(jax.tree.map(jnp.asarray, records), sky, profile)


class PathCoronagraph(AbstractImagingCoronagraph):
    """Live optical train with a fixed sky-angle grid and a calibration snapshot.

    Attributes:
        core_path: Optical path from the entrance field to the Lyot pupil plane.
        input_field: Entrance field, optionally with a one-band spectrum.
        grid: Native sky-angle grid in reference-wavelength lambda/D.
        wavelength_nm: Monochromatic build wavelength when supplied.
        low_nm: Shortest wavelength admitted by the native sampling, in nm.
        max_separation_lod: Largest separation requested for aperture photometry.
        separation_step_lod: Spacing of the calibrated source offsets in lambda/D.
        fov_margin_lod: Extra half field of view beyond the largest separation.
        rules: Aperture rules prepared in the calibration.
        leakage_rule: Method for reducing stellar leakage to a curve.
        n_sky_screens: Number of independent screens in the sky estimate.
        seed: Seed used to reproduce the sky illumination.
        calibration: Snapshot of the train's performance records and sky products.
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
    calibration: PathCalibration

    @property
    def bandpass(self) -> Band | None:
        """Entrance wavelength band, or None for an untagged field."""
        spectrum = self.input_field.spectrum
        return None if spectrum is None else spectrum.bands[0]

    @classmethod
    def from_path(
        cls,
        core_path,
        input_field,
        *,
        primary,
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

        primary.frame(reference_wavelength_nm) defines the native angular
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
            primary: Primary geometry; its frame at the reference wavelength defines the
                native angular grid.
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
                half-maximum rules.
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
            npix, pixel_scale_lod, primary.frame(reference_wavelength_nm)
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
        return cls(
            core_path=core_path,
            input_field=input_field,
            grid=grid,
            wavelength_nm=None if wavelength_nm is None else float(wavelength_nm),
            fov_margin_lod=float(fov_margin_lod),
            calibration=PathCalibration._measure(
                core_path, input_field, grid, **options
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
        )
        return eqx.tree_at(lambda model: model.calibration, self, calibration)

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
            rule: Aperture rule of the record.
            leakage_rule: Leakage rule of the record.

        Returns:
            The prepared performance record for that rule pair.

        Raises:
            ValueError: The aperture and leakage rule pair was not prepared.
        """
        for record in self.calibration.records:
            if record.rule == rule and record.leakage_rule == leakage_rule:
                return record
        raise ValueError("No prepared record for this rule and leakage_rule")

    @property
    def frame(self):
        """Angular frame for performance separation queries."""
        return self.grid.frame

    def coronagraph_at(self, wavelength_nm):
        """Select the live model at a serving wavelength.

        Args:
            wavelength_nm: Serving wavelength in nanometers.

        Returns:
            This model, whose propagation uses its build spectrum.

        Raises:
            ValueError: The concrete wavelength is shorter than low_nm.
        """
        self._check_wavelength(wavelength_nm)
        return self

    def at_wavelength(self, wavelength_nm):
        """Return the product grids of the build spectrum.

        Args:
            wavelength_nm: Serving wavelength in nanometers, at least low_nm.

        Returns:
            Product grids, with the native grid for the stellar and source images and
            the sky map grid for the uniform sky.

        Raises:
            ValueError: The concrete wavelength is shorter than low_nm.
        """
        self._check_wavelength(wavelength_nm)
        return ProductGrids(self.grid, self.grid, self.calibration.sky.grid)

    def uniform_sky_response(self, separation_lod):
        """Annular mean of the uniform-sky map in the performance frame.

        Args:
            separation_lod: Separations in lambda/D as dimensionless numeric values.

        Returns:
            Annular mean of the uniform-sky map at those separations, dimensionless.

        Raises:
            TypeError: The separations carry units.
        """
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return self.uniform_sky_profile.evaluate(separation_lod)

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
        """Check the sampling limit when the serving wavelength is concrete.

        Args:
            wavelength_nm: Requested serving wavelength in nanometers.

        Raises:
            ValueError: The wavelength falls below the shortest active node.
        """
        if (
            not isinstance(wavelength_nm, jax.core.Tracer)
            and float(wavelength_nm) < self.low_nm
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
            wavelength_nm: Serving wavelength in nanometers, at least low_nm.
            stellar_diameter_lod: Stellar angular diameter in lambda/D; only zero is
                supported, and a traced nonzero value yields NaN.
            grid: Target grid; None keeps the native grid.

        Returns:
            Stellar flux-fraction image of the on-axis star, native or resampled onto
            the target grid.

        Raises:
            ValueError: The concrete wavelength is shorter than low_nm or the
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
            data=jnp.where(jnp.asarray(stellar_diameter_lod) == 0, image.data, jnp.nan),
        )

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
        """Native point densities or target pixel fractions for each pupil tilt.

        A tilt at or beyond the pupil Nyquist frequency on either axis is
        ambiguous with a different source position: that source has ``valid``
        False and an all-zero image, and the other sources are unaffected.

        Args:
            wavelength_nm: Serving wavelength in nanometers, at least low_nm.
            x_lod: Source offsets along x in lambda/D, broadcast with y_lod.
            y_lod: Source offsets along y in lambda/D, broadcast with x_lod.
            grid: Target grid; None keeps the native grid.

        Returns:
            Images of each source and a validity flag per source; invalid sources have
            all-zero images.

        Raises:
            ValueError: The concrete wavelength is shorter than low_nm.
        """
        self._check_wavelength(wavelength_nm)
        science, place = self._camera()
        energy = entrance_energy(self.input_field)

        def one(x, y):
            return self._psf(science, place((x, y)), energy, grid)

        x, y = jnp.broadcast_arrays(
            jnp.atleast_1d(jnp.asarray(x_lod)), jnp.atleast_1d(jnp.asarray(y_lod))
        )
        nyquist_lod = (
            0.5
            / self.input_field.grid.step
            * (self.low_nm / self.frame.reference_wavelength_nm)
        )
        valid = (jnp.abs(x) < nyquist_lod) & (jnp.abs(y) < nyquist_lod)
        images = jax.vmap(one)(jnp.where(valid, x, 0), jnp.where(valid, y, 0))
        keep = valid.reshape((-1,) + (1,) * (images.data.ndim - 1))
        images = replace(
            images, data=jnp.where(keep, images.data, 0.0), leading_axes=("source",)
        )
        return SourcePSFs(images, valid)

    def uniform_sky_map(self, wavelength_nm, *, grid):
        """Dimensionless extended-illumination transmission on the requested grid.

        Args:
            wavelength_nm: Serving wavelength in nanometers, at least low_nm.
            grid: Target grid of the map.

        Returns:
            Dimensionless uniform-sky transmission image on the target grid.

        Raises:
            ValueError: The concrete wavelength is shorter than low_nm.
        """
        self._check_wavelength(wavelength_nm)
        sky = self.calibration.sky
        return sky if sky.grid == grid else on_grid(sky, grid)
