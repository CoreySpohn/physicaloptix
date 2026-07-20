"""PathCoronagraph: an OpticalPath behind optixstuff's AbstractCoronagraph.

The adapter is a compiled view of a propagation model. At build time it runs
the coronagraph core once on-axis (caching the Lyot-plane field), sweeps a
separation grid off-axis, and derives the scalar performance curves from the
propagated PSFs -- the inner working angle is the rising half-max crossing of
the throughput curve, never a declared constant, and the outer working angle
is design metadata. At serve time the scalar methods interpolate those
curves, and the image interface costs one matrix Fourier transform on-axis
(the cached Lyot field) or one core propagation off-axis.

Conventions match the AbstractCoronagraph contract and the table-backed
implementations: PSFs are per-pixel maps normalized to unit stellar flux
entering the coronagraph; ``core_mean_intensity`` is an intensity density
in (lambda/D)^-2 over the photometric core (the region of the off-axis PSF
at or above half its peak, whose area is ``core_area``);
``occulter_transmission`` is the total transmitted off-axis energy fraction.

Wavelength: each instance is a compiled view at its build spectrum.
``from_path(..., wavelength_nm=...)`` tags the input field with a
one-sample spectrum so OPD phase screens in the core path convert at the
TRUE wavelength during every propagation (build-time curves and serve-time
maps alike); an already-chromatic input field yields band-averaged curves
and maps. The serve-time ``wavelength_nm`` arguments are accepted for
interface conformance but do not re-propagate at a different wavelength --
stack per-wavelength instances in ``optixstuff.MultiBandCoronagraph`` for
a broadband model. ``time_s`` is accepted for interface conformance (the
model is static).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import arcsec_to_lambda_d, rad_to_arcsec
from jaxtyping import Array
from optixstuff.coronagraph import AbstractCoronagraph

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.transforms import Fraunhofer


def _tilted(field, separation_lod):
    """The input field tilted to place a source at +x separation."""
    x = jnp.asarray(field.grid.coords)
    tilt = jnp.exp(2j * jnp.pi * separation_lod * x)[None, :]
    return Field(
        data=field.data * tilt,
        grid=field.grid,
        plane=field.plane,
        spectrum=field.spectrum,
    )


def _tilted_xy(field, x_lod, y_lod):
    """The input field tilted to place a source at a 2D focal position."""
    coords = jnp.asarray(field.grid.coords)
    tilt = (
        jnp.exp(2j * jnp.pi * x_lod * coords)[None, :]
        * jnp.exp(2j * jnp.pi * y_lod * coords)[:, None]
    )
    return Field(
        data=field.data * tilt,
        grid=field.grid,
        plane=field.plane,
        spectrum=field.spectrum,
    )


class PathCoronagraph(AbstractCoronagraph):
    """Coronagraph performance model backed by live OpticalPath propagation.

    Build with :meth:`from_path` from a coronagraph core path (entrance
    pupil to Lyot plane, ending in the PUPIL plane; the science-plane
    transform is applied by the adapter at the requested sampling).
    """

    core_path: object
    input_field: Field
    lyot_field: Field
    input_energy: Array
    curve_seps: Array
    curve_throughput: Array
    curve_core_area: Array
    curve_core_mean_intensity: Array
    curve_occulter_transmission: Array
    diameter_m: float = eqx.field(static=True)
    pixel_scale_lod: float = eqx.field(static=True)
    IWA: float = eqx.field(static=True)
    OWA: float = eqx.field(static=True)
    wavelength_nm: float | None = eqx.field(static=True, default=None)

    @classmethod
    def from_path(
        cls,
        core_path,
        input_field,
        *,
        diameter_m,
        owa_lod,
        pixel_scale_lod=0.25,
        sep_step_lod=0.5,
        fov_margin_lod=4.0,
        wavelength_nm=None,
    ):
        """Build the adapter and derive its performance curves.

        Args:
            core_path: ``OpticalPath`` from the entrance pupil to the Lyot
                plane (must end in the PUPIL plane).
            input_field: The unperturbed entrance-pupil field. May be
                chromatic, in which case the derived curves and served
                maps are band-averaged over its spectrum.
            diameter_m: Telescope diameter, for pixel-scale conversion.
            owa_lod: Outer working angle in lambda/D (design metadata: the
                dark-zone outer radius).
            pixel_scale_lod: Native sampling for the internal curve sweep.
            sep_step_lod: Separation grid step for the curve sweep.
            fov_margin_lod: Native-grid margin beyond the OWA.
            wavelength_nm: Build wavelength. A monochromatic
                ``input_field`` is promoted to a one-sample spectrum at
                this wavelength, so OPD phase screens in the core path
                convert at the true wavelength in every propagation.
                ``None`` (default) leaves the field untouched: phase
                screens then convert at their own construction
                wavelength, and the compiled view is achromatic-core.

        Returns:
            A ready ``PathCoronagraph``.
        """
        if wavelength_nm is not None and input_field.spectrum is None:
            input_field = Field(
                data=input_field.data[jnp.newaxis],
                grid=input_field.grid,
                plane=input_field.plane,
                spectrum=Spectrum(
                    wavelengths_nm=jnp.asarray([float(wavelength_nm)]),
                    weights=jnp.asarray([1.0]),
                ),
            )
        last_plane = core_path.stages[-1].op
        plane_out = getattr(last_plane, "plane_out", None) or getattr(
            last_plane, "plane", None
        )
        if plane_out is not PlaneKind.PUPIL:
            raise ValueError(
                "core_path must end in the pupil (Lyot) plane; the adapter "
                "applies the science-plane transform itself, got "
                f"{plane_out}"
            )

        pupil_grid = input_field.grid
        half_width = owa_lod + fov_margin_lod
        npix = 2 * int(np.ceil(half_width / pixel_scale_lod))
        native_grid = Grid.focal(npix, pixel_scale_lod)
        science = Fraunhofer(grid_in=pupil_grid, grid_out=native_grid)

        # weight-summed over the spectrum, matching Field.intensity()
        abs2 = input_field.data.real**2 + input_field.data.imag**2
        if input_field.spectrum is not None:
            abs2 = jnp.tensordot(input_field.spectrum.weights, abs2, axes=1)
        input_energy = jnp.sum(abs2) * pupil_grid.weights
        lyot_field, _ = core_path.propagate(input_field)
        cell = native_grid.weights

        def density(field):
            return science(field).intensity() / input_energy

        on_axis = density(lyot_field)

        seps = np.arange(0.0, owa_lod + 2.0 * sep_step_lod, sep_step_lod)
        throughput, core_area, core_mean, occ_trans = [], [], [], []
        for sep in seps:
            off = density(core_path.propagate(_tilted(input_field, sep))[0])
            core = off >= 0.5 * jnp.max(off)
            throughput.append(float(jnp.sum(jnp.where(core, off, 0.0)) * cell))
            core_area.append(float(jnp.sum(core) * cell))
            core_mean.append(
                float(jnp.sum(jnp.where(core, on_axis, 0.0)) / jnp.sum(core))
            )
            occ_trans.append(float(jnp.sum(off) * cell))

        throughput = np.asarray(throughput)
        half = throughput.max() / 2.0
        above = np.nonzero(throughput >= half)[0]
        first = int(above[0]) if above.size else len(seps) - 1
        if first == 0:
            iwa = float(seps[0])
        else:
            lo, hi = throughput[first - 1], throughput[first]
            frac = (half - lo) / (hi - lo)
            iwa = float(seps[first - 1] + frac * sep_step_lod)

        return cls(
            core_path=core_path,
            input_field=input_field,
            lyot_field=lyot_field,
            input_energy=input_energy,
            curve_seps=jnp.asarray(seps),
            curve_throughput=jnp.asarray(throughput),
            curve_core_area=jnp.asarray(core_area),
            curve_core_mean_intensity=jnp.asarray(core_mean),
            curve_occulter_transmission=jnp.asarray(occ_trans),
            diameter_m=float(diameter_m),
            pixel_scale_lod=float(pixel_scale_lod),
            IWA=iwa,
            OWA=float(owa_lod),
            wavelength_nm=None if wavelength_nm is None else float(wavelength_nm),
        )

    # -- scalar interface (interpolated compiled views) -------------------

    def _interp(self, curve, separation_lod):
        return jnp.interp(jnp.asarray(separation_lod), self.curve_seps, curve)

    def throughput(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Core throughput from the propagated off-axis PSFs."""
        return self._interp(self.curve_throughput, separation_lod)

    def core_area(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Photometric core area (above-half-max region) in (lambda/D)^2."""
        return self._interp(self.curve_core_area, separation_lod)

    def core_mean_intensity(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Mean stellar intensity density over the core, in (lambda/D)^-2."""
        return self._interp(self.curve_core_mean_intensity, separation_lod)

    def occulter_transmission(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Total transmitted off-axis energy fraction."""
        return self._interp(self.curve_occulter_transmission, separation_lod)

    # -- image interface ---------------------------------------------------

    def _requested_grid(self, wavelength_nm, pixel_scale_rad, npixels):
        pixel_scale_lod = arcsec_to_lambda_d(
            rad_to_arcsec(float(pixel_scale_rad)),
            float(wavelength_nm),
            self.diameter_m,
        )
        return Grid.focal(int(npixels), float(pixel_scale_lod))

    def _psf(self, field, grid):
        science = Fraunhofer(grid_in=self.input_field.grid, grid_out=grid)
        return science(field).intensity() * grid.weights / self.input_energy

    def on_axis_psf(self, wavelength_nm, pixel_scale_rad, npixels):
        """On-axis PSF per pixel, unit pre-coronagraph stellar flux.

        Costs one matrix Fourier transform: the Lyot-plane field is cached
        at construction.
        """
        grid = self._requested_grid(wavelength_nm, pixel_scale_rad, npixels)
        return self._psf(self.lyot_field, grid)

    def off_axis_psf(self, wavelength_nm, separation_lod, pixel_scale_rad, npixels):
        """Off-axis (planet) PSF at +x separation, unit stellar flux.

        Re-propagates the coronagraph core per call; for serving loops,
        freeze to a table instead (the builder-not-server rule).
        """
        grid = self._requested_grid(wavelength_nm, pixel_scale_rad, npixels)
        lyot, _ = self.core_path.propagate(_tilted(self.input_field, separation_lod))
        return self._psf(lyot, grid)

    # -- sampling-explicit contract (native: render directly at the
    # -- requested target grid, no intermediate resample) ------------------

    def _target_grid(self, pixel_scale_lod, shape):
        """A square render grid covering the requested (ny, nx) target.

        The matrix Fourier transform renders on a square grid; for a
        non-square target the render covers the larger dimension and
        ``_center_crop`` trims it. Cropping is exact -- each MFT output
        pixel is independent.
        """
        ny, nx = int(shape[0]), int(shape[1])
        n = max(ny, nx)
        return Grid.focal(n, float(pixel_scale_lod)), (ny, nx)

    @staticmethod
    def _center_crop(image, shape):
        n_y, n_x = image.shape[-2], image.shape[-1]
        ny, nx = shape
        y0, x0 = (n_y - ny) // 2, (n_x - nx) // 2
        return image[..., y0 : y0 + ny, x0 : x0 + nx]

    def stellar_map(self, wavelength_nm, stellar_diam_lod, *, pixel_scale_lod, shape):
        """On-axis stellar map rendered at the target sampling.

        One matrix Fourier transform of the cached Lyot field. The view
        is compiled at the build spectrum: the serve-time
        ``wavelength_nm`` enters only through the lambda/D sampling the
        caller already computed (stack per-wavelength instances in a
        ``MultiBandCoronagraph`` for a chromatic model). Point-source
        model: ``stellar_diam_lod`` is ignored.
        """
        del wavelength_nm, stellar_diam_lod
        grid, shape = self._target_grid(pixel_scale_lod, shape)
        return self._center_crop(self._psf(self.lyot_field, grid), shape)

    def extended_scene(
        self,
        scene_map,
        map_pixel_scale_lod,
        wavelength_nm,
        *,
        pixel_scale_lod,
        shape,
        rotation_deg=0.0,
    ):
        """Extended scenes are not served by the live adapter.

        A per-source-position render would cost one core propagation per
        scene pixel; freeze the design to a table (the YIP emitter) and
        use a table-backed coronagraph for disk work.
        """
        raise NotImplementedError(
            "PathCoronagraph does not render extended scenes (one core "
            "propagation per scene pixel); bake the design into a table "
            "(e.g. the YIP emitter) and use a table-backed coronagraph"
        )

    def source_psfs(self, wavelength_nm, x_lod, y_lod, *, pixel_scale_lod, shape):
        """Off-axis PSFs at 2D positions, rendered at the target sampling.

        One core propagation per source; the science-plane transform
        lands directly on the target grid.
        """
        del wavelength_nm
        grid, shape = self._target_grid(pixel_scale_lod, shape)
        xs = jnp.atleast_1d(jnp.asarray(x_lod))
        ys = jnp.atleast_1d(jnp.asarray(y_lod))

        def one(x, y):
            lyot, _ = self.core_path.propagate(_tilted_xy(self.input_field, x, y))
            return self._psf(lyot, grid)

        return self._center_crop(jax.vmap(one)(xs, ys), shape)

    def background_transmission(self, wavelength_nm, *, pixel_scale_lod, shape):
        """Uniform-background transmission evaluated on the target grid.

        The derived ``occulter_transmission`` curve at each target
        pixel's separation -- value semantics, no resample involved.
        """
        del wavelength_nm
        grid, shape = self._target_grid(pixel_scale_lod, shape)
        coords = jnp.asarray(grid.coords)
        r = jnp.hypot(coords[None, :], coords[:, None])
        return self._center_crop(
            self._interp(self.curve_occulter_transmission, r), shape
        )
