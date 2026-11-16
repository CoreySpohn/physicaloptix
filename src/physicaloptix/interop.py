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
from hwoutils.grids import Image, PixelGrid, resample
from optixstuff.coronagraph import AbstractCoronagraph
from yippy import Circular, HalfMaximum, performance_curves

from physicaloptix.core import Field, PlaneKind, Spectrum
from physicaloptix.core.grid import _grid_axis
from physicaloptix.transforms import Fraunhofer

_DEFAULT_RULE = Circular(0.7)


def _tilted_xy(field, x_lod, y_lod):
    coords = jnp.asarray(_grid_axis(field.grid))
    tilt = (
        jnp.exp(2j * jnp.pi * x_lod * coords)[None, :]
        * jnp.exp(2j * jnp.pi * y_lod * coords)[:, None]
    )
    return Field(field.data * tilt, field.grid, field.plane, field.spectrum)


def _on_grid(image, grid):
    """Reconcile angular frames before the shared flux-preserving resampler."""
    if image.grid == grid:
        return image
    source, target = image.grid, grid
    if source.unit != target.unit or (
        source.unit == "lod" and source.frame != target.frame
    ):
        if source.unit == "lod":
            source = source.to_arcsec(source.frame.reference_wavelength_nm)
        if target.unit == "lod":
            target = target.to_arcsec(target.frame.reference_wavelength_nm)
    return replace(resample(replace(image, grid=source), target), grid=grid)


class PathCoronagraph(AbstractCoronagraph):
    """Cached on-axis propagation and eagerly prepared aperture-rule records."""

    core_path: object
    input_field: Field
    lyot_field: Field
    input_energy: jax.Array
    grid: PixelGrid
    _records: tuple
    curve_seps: jax.Array
    curve_occulter_transmission: jax.Array
    outer_working_angle_lod: float = eqx.field(static=True)
    wavelength_nm: float | None = eqx.field(static=True)

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
        leakage_rule="aperture_mean",
    ):
        """Compile a point-star model from an entrance-to-Lyot path.

        primary.frame(reference_wavelength_nm) defines the native angular
        grid. wavelength_nm binds monochromatic OPD screens when supplied.
        Circular(0.7) and HalfMaximum(0.5) are always prepared; rules adds
        further immutable records. Finite stellar diameters are not modeled.
        """
        if wavelength_nm is not None and input_field.spectrum is None:
            input_field = Field(
                input_field.data[None],
                input_field.grid,
                input_field.plane,
                Spectrum([float(wavelength_nm)], [1.0]),
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

        def psf(field):
            return Image(
                science(field).intensity().data * grid.pixel_area / energy,
                grid,
                (),
                "per_pixel",
                "stellar_flux_fraction",
            )

        stellar = psf(lyot)
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
            for rule in dict.fromkeys((Circular(0.7), HalfMaximum(0.5), *rules))
        )
        # Store coefficients as JAX leaves for immutable jit/vmap evaluation.
        records = jax.tree.map(jnp.asarray, records)
        return cls(
            core_path,
            input_field,
            lyot,
            energy,
            grid,
            records,
            jnp.asarray(seps),
            jnp.asarray([jnp.sum(im.data) for im in images]),
            float(outer_working_angle_lod),
            None if wavelength_nm is None else float(wavelength_nm),
        )

    def curves(self, rule=_DEFAULT_RULE, *, leakage_rule="aperture_mean"):
        """Select an eagerly built record; unprepared rules raise."""
        for record in self._records:
            if record.rule == rule and record.leakage_rule == leakage_rule:
                return record
        raise ValueError("No prepared record for this rule and leakage_rule")

    def occulter_transmission(self, separation_lod, wavelength_nm=None, *, time_d=0.0):
        """Total off-axis energy fraction at the build spectrum."""
        return jnp.interp(
            jnp.asarray(separation_lod),
            self.curve_seps,
            self.curve_occulter_transmission,
        )

    def _psf(self, field, grid):
        direct = (
            grid.unit == "lod"
            and grid.frame == self.grid.frame
            and grid.shape[0] == grid.shape[1]
            and grid.center_x == (grid.shape[1] - 1) / 2
            and grid.center_y == (grid.shape[0] - 1) / 2
            and grid.basis[0][1] == grid.basis[1][0] == 0
            and grid.basis[0][0] == grid.basis[1][1] > 0
        )
        render_grid = grid if direct else self.grid
        science = Fraunhofer(self.input_field.grid, render_grid)
        native = Image(
            science(field).intensity().data
            * render_grid.pixel_area
            / self.input_energy,
            render_grid,
            (),
            "per_pixel",
            "stellar_flux_fraction",
        )
        return _on_grid(native, grid)

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, grid):
        """Point-star per-pixel flux fraction; finite diameters remain unsupported."""
        return self._psf(self.lyot_field, grid)

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, grid):
        """Per-pixel source flux fractions with leading axis 'source'."""

        def one(x, y):
            lyot, _ = self.core_path.propagate(_tilted_xy(self.input_field, x, y))
            return self._psf(lyot, grid).data

        data = jax.vmap(one)(jnp.atleast_1d(x_lod), jnp.atleast_1d(y_lod))
        return Image(data, grid, ("source",), "per_pixel", "stellar_flux_fraction")

    def background_transmission(self, wavelength_nm, *, grid):
        """Sample dimensionless sky transmission without a pixel-area factor."""
        x, y = self.grid.coordinates()
        native = Image(
            self.occulter_transmission(jnp.hypot(x, y)),
            self.grid,
            (),
            "per_unit_area",
            "transmission",
        )
        return _on_grid(native, grid)

    def extended_scene(self, scene_map, wavelength_nm, *, grid, rotation_deg=0.0):
        """Require a table model for per-scene-pixel propagation."""
        raise NotImplementedError(
            "PathCoronagraph does not render extended scenes; "
            "use a table-backed coronagraph"
        )
