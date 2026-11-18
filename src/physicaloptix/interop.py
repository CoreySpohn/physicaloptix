"""Live propagation behind the shared coronagraph image and photometry contract.

Each adapter is built at one spectrum. Serving wavelengths select no new
propagation; use one adapter per band in MultiBandCoronagraph for that purpose.
Time is in days from the scene epoch; this static model ignores its value.
"""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.grids import Image, PixelGrid
from hwoutils.spectral import Band, Quadrature
from optixstuff.apertures import DEFAULT_APERTURE_RULE, HalfMaximum
from optixstuff.coronagraph import (
    AbstractImagingCoronagraph,
    ProductGrids,
    SourcePSFs,
    _on_grid,
)
from optixstuff.performance import SampledProfile, performance_curves

from physicaloptix._sky import uniform_sky_image, unit_energy
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
    """Cached on-axis propagation and eagerly prepared aperture-rule records."""

    core_path: object
    input_field: Field
    lyot_field: Field
    input_energy: jax.Array
    grid: PixelGrid
    _records: tuple
    _sky: Image
    _sky_profile: SampledProfile
    outer_working_angle_lod: float = eqx.field(static=True)
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
        outer_working_angle_lod,
        reference_wavelength_nm,
        pixel_scale_lod=0.25,
        sep_step_lod=0.5,
        fov_margin_lod=4.0,
        wavelength_nm=None,
        rules=(),
        n_sky_screens=1000,
        seed=0,
        leakage_rule="aperture_mean",
    ):
        """Compile a point-star model from an entrance-to-Lyot path.

        primary.frame(reference_wavelength_nm) defines the native angular
        grid. wavelength_nm binds monochromatic OPD screens when supplied.
        DEFAULT_APERTURE_RULE and HalfMaximum(0.5) are always prepared; rules adds
        further immutable records. Finite stellar diameters are not modeled.
        n_sky_screens independent complex pupil screens (drawn with seed)
        estimate the uniform-sky map and its annular means.
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
        npix = 2 * int(
            np.ceil((outer_working_angle_lod + fov_margin_lod) / pixel_scale_lod)
        )
        grid = PixelGrid.focal(
            npix, pixel_scale_lod, primary.frame(reference_wavelength_nm)
        )
        energy = jnp.sum(input_field.intensity().data) * input_field.grid.pixel_area
        lyot, _ = core_path.propagate(input_field)
        science = Fraunhofer(input_field.grid, grid)

        def psf(field, outside="zero"):
            return Image(
                science.forward(field).intensity().data * grid.pixel_area / energy,
                grid,
                (),
                "per_pixel",
                "stellar_flux_fraction",
                reconstruction="band_limited",
                outside=outside,
            )

        # The stellar halo continues past the field; off-axis cores do not.
        stellar = psf(lyot, "reflect")
        seps = np.arange(0.0, outer_working_angle_lod + 2 * sep_step_lod, sep_step_lod)
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
                stellar_diameter=0.0,
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
            float(outer_working_angle_lod),
            None if wavelength_nm is None else float(wavelength_nm),
        )

    def curves(self, rule=DEFAULT_APERTURE_RULE, *, leakage_rule="aperture_mean"):
        """Select an eagerly built record; unprepared rules raise."""
        for record in self._records:
            if record.rule == rule and record.leakage_rule == leakage_rule:
                return record
        raise ValueError("No prepared record for this rule and leakage_rule")

    @property
    def frame(self):
        """Angular frame for performance separation queries."""
        return self.grid.frame

    def at_wavelength(self, wavelength_nm):
        """Return the product grids of the build spectrum."""
        return ProductGrids(self.grid, self.grid, self._sky.grid)

    def uniform_sky_response(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Annular mean of the uniform-sky map in the performance frame."""
        if hasattr(separation_lod, "unit"):
            raise TypeError("separation_lod must be dimensionless numeric values")
        return self._sky_profile.evaluate(separation_lod)

    def _psf(self, field, grid, outside="zero"):
        direct = (
            grid.unit == "lod"
            and grid.frame == self.grid.frame
            and grid.shape[0] == grid.shape[1]
            and grid.is_geometric
            and grid.basis[0][1] == grid.basis[1][0] == 0
            and grid.basis[0][0] == grid.basis[1][1] > 0
        )
        render_grid = grid if direct else self.grid
        science = Fraunhofer(self.input_field.grid, render_grid)
        native = Image(
            science.forward(field).intensity().data
            * render_grid.pixel_area
            / self.input_energy,
            render_grid,
            (),
            "per_pixel",
            "stellar_flux_fraction",
            reconstruction="band_limited",
            outside=outside,
        )
        return native if native.grid == grid else _on_grid(native, grid)

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, grid=None):
        """Point-star flux fraction; reject concrete or mark traced finite stars NaN."""
        if (
            not isinstance(stellar_diam_lod, jax.core.Tracer)
            and float(stellar_diam_lod) != 0
        ):
            raise ValueError("finite stellar diameter requires a prepared stellar map")
        # The stellar halo continues past the field.
        image = self._psf(
            self.lyot_field, self.grid if grid is None else grid, outside="reflect"
        )
        if not isinstance(stellar_diam_lod, jax.core.Tracer):
            return image
        return replace(
            image,
            data=jnp.where(jnp.asarray(stellar_diam_lod) == 0, image.data, jnp.nan),
        )

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid=None):
        """Per-pixel source fractions and support of each sampled pupil tilt.

        A tilt at or beyond the pupil Nyquist frequency on either axis is
        ambiguous with a different source position.
        """
        grid = self.grid if grid is None else grid

        def one(x, y):
            lyot, _ = self.core_path.propagate(_tilted_xy(self.input_field, x, y))
            return self._psf(lyot, grid)

        x, y = jnp.broadcast_arrays(
            jnp.atleast_1d(jnp.asarray(x_lod)), jnp.atleast_1d(jnp.asarray(y_lod))
        )
        nyquist_lod = 0.5 / self.input_field.grid.step
        valid = (jnp.abs(x) < nyquist_lod) & (jnp.abs(y) < nyquist_lod)
        images = jax.vmap(one)(jnp.where(valid, x, 0), jnp.where(valid, y, 0))
        return SourcePSFs(replace(images, leading_axes=("source",)), valid)

    def uniform_sky_map(self, wavelength_nm, *, grid):
        """Dimensionless extended-illumination transmission on the requested grid."""
        return self._sky if self._sky.grid == grid else _on_grid(self._sky, grid)
