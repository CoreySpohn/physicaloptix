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
from optixstuff.grids import Image, Measure, PixelGrid
from optixstuff.performance import SampledProfile, performance_curves
from optixstuff.spectral import Band, Quadrature

from physicaloptix._sky import (
    _require_inside_pupil_nyquist,
    entrance_energy,
    uniform_sky_image,
    unit_energy,
)
from physicaloptix.core import Field, PlaneKind
from physicaloptix.core.grid import _grid_axis
from physicaloptix.transforms import Fraunhofer


def _tilted_xy(field, x_lod, y_lod):
    coords = jnp.asarray(_grid_axis(field.grid))
    tilt = (
        jnp.exp(2j * jnp.pi * x_lod * coords)[None, :]
        * jnp.exp(2j * jnp.pi * y_lod * coords)[:, None]
    )
    return Field(field.data * tilt, field.grid, field.plane, field.spectrum)


class PathCoronagraph(AbstractImagingCoronagraph):
    """Cached on-axis propagation and eagerly prepared aperture-rule records.

    Attributes:
        core_path: Optical path from the entrance field to the Lyot pupil plane.
        input_field: Entrance field, with a one-band spectrum.
        lyot_field: On-axis field at the Lyot plane, propagated once at construction.
        input_energy: Total energy of the entrance field, used as the stellar flux
            normalization.
        grid: Native focal grid of the products, in lambda/D of the reference
            wavelength.
        uniform_sky_profile: Annular mean profile of the uniform-sky map, sampled on
            separation.
        wavelength_nm: Wavelength in nanometers that binds monochromatic screens, or
            None.
    """

    core_path: object
    input_field: Field
    lyot_field: Field
    input_energy: jax.Array
    grid: PixelGrid
    _records: tuple
    _sky: Image
    uniform_sky_profile: SampledProfile
    wavelength_nm: float | None = eqx.field(static=True)

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
        _require_inside_pupil_nyquist(grid, input_field)
        energy = entrance_energy(input_field)
        lyot, _ = core_path.propagate(input_field)
        science = Fraunhofer(input_field.grid, grid)

        def psf(field):
            return Image(
                science.forward(field).intensity().integrated().data / energy,
                grid,
                (),
                Measure.integral(),
                "stellar_flux_fraction",
                reconstruction="band_limited",
            )

        stellar = psf(lyot)
        seps = np.arange(
            0.0, max_separation_lod + 2 * separation_step_lod, separation_step_lod
        )
        images = tuple(
            psf(core_path.propagate(_tilted_xy(input_field, s, 0.0))[0]) for s in seps
        )
        by_separation = dict(zip(seps, images, strict=True))
        records = tuple(
            performance_curves(
                by_separation.__getitem__,
                stellar,
                rule,
                separations_lod=seps,
                leakage_rule=leakage_rule,
                stellar_diameter_lod=0.0,
            )
            for rule in dict.fromkeys((DEFAULT_APERTURE_RULE, HalfMaximum(0.5), *rules))
        )
        # Store coefficients as JAX leaves for immutable jit/vmap evaluation.
        records = jax.tree.map(jnp.asarray, records)
        sky = uniform_sky_image(
            core_path,
            science,
            unit_energy(input_field),
            n_screens=n_sky_screens,
            seed=seed,
        )
        sky_profile = jax.tree.map(jnp.asarray, SampledProfile.from_uniform_sky(sky))
        return cls(
            core_path,
            input_field,
            lyot,
            energy,
            grid,
            records,
            sky,
            sky_profile,
            None if wavelength_nm is None else float(wavelength_nm),
        )

    def curves(self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule="aperture_mean"):
        """Select an eagerly built record; unprepared rules raise.

        Args:
            rule: Aperture rule of the record.
            leakage_rule: Leakage rule of the record.

        Returns:
            The prepared performance record for that rule pair.
        """
        for record in self._records:
            if record.rule == rule and record.leakage_rule == leakage_rule:
                return record
        raise ValueError("No prepared record for this rule and leakage_rule")

    @property
    def frame(self):
        """Angular frame for performance separation queries."""
        return self.grid.frame

    def at_wavelength(self, wavelength_nm):
        """Return the product grids of the build spectrum.

        Args:
            wavelength_nm: Wavelength in nanometers; unused, because the grids are those
                of the build spectrum.

        Returns:
            Product grids, with the native grid for the stellar and source images and
            the sky map grid for the uniform sky.
        """
        return ProductGrids(self.grid, self.grid, self._sky.grid)

    def uniform_sky_response(self, separation_lod):
        """Annular mean of the uniform-sky map in the performance frame.

        Args:
            separation_lod: Separations in lambda/D as dimensionless numeric values.

        Returns:
            Annular mean of the uniform-sky map at those separations, dimensionless.
        """
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return self.uniform_sky_profile.evaluate(separation_lod)

    def _psf(self, field, grid):
        direct = (
            grid is not None
            and grid.unit == "lod"
            and grid.frame == self.grid.frame
            and grid.shape[0] == grid.shape[1]
            and grid.is_geometric
            and grid.basis[0][1] == grid.basis[1][0] == 0
            and grid.basis[0][0] == grid.basis[1][1] > 0
        )
        render_grid = grid if direct else self.grid
        science = Fraunhofer(self.input_field.grid, render_grid)
        power = science.forward(field).intensity()
        if grid is not None:
            power = power.integrated()
        native = replace(
            power,
            data=power.data / self.input_energy,
            quantity="stellar_flux_fraction",
        )
        return native if grid is None or native.grid == grid else on_grid(native, grid)

    def stellar_map(self, wavelength_nm, stellar_diameter_lod, *, grid=None):
        """Point-star flux fraction; reject concrete or mark traced finite stars NaN.

        Args:
            wavelength_nm: Wavelength in nanometers; unused, because the model holds its
                build spectrum.
            stellar_diameter_lod: Stellar angular diameter in lambda/D; only zero is
                supported, and a traced nonzero value yields NaN.
            grid: Target grid; None keeps the native grid.

        Returns:
            Stellar flux-fraction image of the on-axis star, native or resampled onto
            the target grid.
        """
        if (
            not isinstance(stellar_diameter_lod, jax.core.Tracer)
            and float(stellar_diameter_lod) != 0
        ):
            raise ValueError("finite stellar diameter requires a prepared stellar map")
        image = self._psf(self.lyot_field, grid)
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
            wavelength_nm: Wavelength in nanometers; unused, because the model holds its
                build spectrum.
            x_lod: Source offsets along x in lambda/D, broadcast with y_lod.
            y_lod: Source offsets along y in lambda/D, broadcast with x_lod.
            grid: Target grid; None keeps the native grid.

        Returns:
            Images of each source and a validity flag per source; invalid sources have
            all-zero images.
        """

        def one(x, y):
            lyot, _ = self.core_path.propagate(_tilted_xy(self.input_field, x, y))
            return self._psf(lyot, grid)

        x, y = jnp.broadcast_arrays(
            jnp.atleast_1d(jnp.asarray(x_lod)), jnp.atleast_1d(jnp.asarray(y_lod))
        )
        nyquist_lod = 0.5 / self.input_field.grid.step
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
            wavelength_nm: Wavelength in nanometers; unused, because the model holds its
                build spectrum.
            grid: Target grid of the map.

        Returns:
            Dimensionless uniform-sky transmission image on the target grid.
        """
        return self._sky if self._sky.grid == grid else on_grid(self._sky, grid)
