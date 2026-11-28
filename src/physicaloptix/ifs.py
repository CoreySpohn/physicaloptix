"""Wave-optics lenslet IFS chain: physical PSFlet templates (builder mode).

A lenslet-array IFS is a pupil imager: each lenslet sits at the telescope
focal plane and forms a micro-pupil (a demagnified image of the telescope
pupil) at its back focal plane, which the spectrograph relays onto the
detector. The PSFlet is therefore the pupil image convolved with the Fourier
transform of the lenslet aperture -- broad sinc^2 wings for a square lenslet
-- optionally truncated by a pinhole at the micro-pupil plane (the BIGRE
design of Antichi et al. 2009, used by SPHERE/CHARIS/Roman CGI) and by the
spectrograph stop. Rizzo et al. 2017 (the WFIRST IFS simulator) is the
template-level precedent.

``LensletChain`` composes three continuous-FT MFTs in normalized units:

1. telescope exit pupil -> the lenslet tile. Tile coordinates are in lenslet
   pitches (the tile spans [-1/2, 1/2]); the canonical template illumination
   is the on-axis PSF landing centered on the lenslet, which is what carries
   the telescope-pupil Fourier content that forms the pupil image.
2. windowed tile -> micro-pupil plane, in lenslet diffraction units
   (lambda f_lenslet / pitch; the window sinc's first zero sits at 1). The
   pupil image has diameter ``pitch_lod(lambda)`` in these units. Optional
   pinhole mask here.
3. micro-pupil -> spectrograph stop (conjugate to the lenslet, so back in
   pitch units; stop mask, optional aberration OPD, optional dispersion
   tilt) -> detector, in diffraction units mapped to detector pixels.

The relay geometry fixes the pupil-image diameter on the detector at
``micropupil_px`` pixels for every wavelength, so one diffraction unit is
``micropupil_px / pitch_lod(lambda)`` pixels: diffractive structure grows
with wavelength while the geometric core does not. That chromatic morphology
is exactly what tabulated templates exist to capture (an analytic profile
scaled with wavelength cannot represent it).

The chain is a BUILDER, not a runtime operator: ``psflet_pack`` tabulates
pixel-integrated templates on a sub-pixel offset grid for downstream
spectral-extraction codes.
"""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import two_pi
from hwoutils.leaves import is_traced
from hwoutils.numerics import same
from hwoutils.precision import like, precision_of
from jax import lax
from jaxtyping import Array
from optixstuff import PsfletPack
from optixstuff.grids import (
    Image,
    Measure,
    PixelGrid,
    footprint_overlap,
    integrate_footprints,
)
from optixstuff.psflet_pack import TEMPLATE_ORIGIN

from physicaloptix.core.grid import grid_axis
from physicaloptix.elements.base import _amplitude_leaf
from physicaloptix.transforms.cmft import cmft_fwd


class LensletChain(eqx.Module):
    """Single-lenslet wave chain from the telescope exit pupil to the detector.

    Attributes:
        pupil_field: Complex exit-pupil field, ``(n, n)`` shared across
            wavelengths or ``(n_lam, n, n)`` per-wavelength (matched against
            the pack wavelengths at build time).
        pupil_grid: PixelGrid of ``pupil_field`` (pupil diameters).
        stop_opd_nm: Optional spectrograph aberration OPD map [nm] on the
            stop grid, ``(n_stop, n_stop)``; ``None`` for an ideal relay.
        pitch_lod_ref: Lenslet pitch in lambda/D at ``lam_ref_nm``.
        lam_ref_nm: Wavelength at which ``pitch_lod_ref`` is quoted.
        micropupil_px: Geometric pupil-image diameter on the detector [px]
            (wavelength-independent relay geometry).
        n_tile: Samples across the lenslet tile.
        n_mp: Samples across the micro-pupil grid.
        n_stop: Samples across the stop grid.
        mp_half_extent: Micro-pupil grid half-width [diffraction units]. The
            sampling knob: it must hold enough sinc wings for the fluxes to
            converge (a pinhole physically truncates the wings and relaxes
            it).
        stop_halfwidth: Spectrograph stop half-width [pitch units]. The
            geometric lenslet-image beam spans [-1/2, 1/2]; 0.5 passes it
            exactly.
        stop_kind: Stop shape, ``"square"`` or ``"circular"``.
        pinhole_radius: BIGRE pinhole radius at the micro-pupil
            [diffraction units]; ``None`` for no pinhole.
        fill_factor: Transmissive fraction of the lenslet pitch per axis.
        illumination: Canonical template illumination: ``"psf"`` (the
            telescope PSF centered on the lenslet; the physical convention)
            or ``"flat"`` (uniform tile; the analytic sinc^2 reference).
    """

    pupil_field: Array = eqx.field(converter=_amplitude_leaf)
    pupil_grid: PixelGrid
    stop_opd_nm: Array | None
    pitch_lod_ref: float = eqx.field(static=True)
    lam_ref_nm: float = eqx.field(static=True)
    micropupil_px: float = eqx.field(static=True)
    n_tile: int = eqx.field(static=True)
    n_mp: int = eqx.field(static=True)
    n_stop: int = eqx.field(static=True)
    mp_half_extent: float = eqx.field(static=True)
    stop_halfwidth: float = eqx.field(static=True)
    stop_kind: str = eqx.field(static=True)
    pinhole_radius: float | None = eqx.field(static=True)
    fill_factor: float = eqx.field(static=True)
    illumination: str = eqx.field(static=True)

    def __init__(
        self,
        pupil_field,
        pupil_grid,
        *,
        pitch_lod_ref,
        lam_ref_nm,
        micropupil_px,
        n_tile=64,
        n_mp=256,
        n_stop=128,
        mp_half_extent=8.0,
        stop_halfwidth=0.75,
        stop_kind="square",
        pinhole_radius=None,
        fill_factor=1.0,
        illumination="psf",
        stop_opd_nm=None,
    ):
        """Build the chain configuration (see the class docstring).

        Args:
            pupil_field: Complex exit-pupil field, (n, n) shared across wavelengths or
                (n_lam, n, n) per wavelength.
            pupil_grid: PixelGrid of pupil_field, in pupil diameters.
            pitch_lod_ref: Lenslet pitch in lambda/D at lam_ref_nm.
            lam_ref_nm: Wavelength in nanometers at which pitch_lod_ref is quoted.
            micropupil_px: Geometric pupil-image diameter on the detector in pixels.
            n_tile: Samples across the lenslet tile.
            n_mp: Samples across the micro-pupil grid.
            n_stop: Samples across the stop grid.
            mp_half_extent: Micro-pupil grid half-width in diffraction units.
            stop_halfwidth: Spectrograph stop half-width in pitch units.
            stop_kind: Stop shape, 'square' or 'circular'.
            pinhole_radius: Pinhole radius at the micro-pupil in diffraction units; None
                for no pinhole.
            fill_factor: Transmissive fraction of the lenslet pitch per axis, in (0, 1].
            illumination: Template illumination, 'psf' (telescope PSF centered on the
                lenslet) or 'flat' (uniform tile).
            stop_opd_nm: Optional spectrograph aberration OPD map in nanometers on the
                stop grid, (n_stop, n_stop); None for an ideal relay.
        """
        self.pupil_field = pupil_field
        self.pupil_grid = pupil_grid
        self.stop_opd_nm = None if stop_opd_nm is None else jnp.asarray(stop_opd_nm)
        self.pitch_lod_ref = float(pitch_lod_ref)
        self.lam_ref_nm = float(lam_ref_nm)
        self.micropupil_px = float(micropupil_px)
        self.n_tile = int(n_tile)
        self.n_mp = int(n_mp)
        self.n_stop = int(n_stop)
        self.mp_half_extent = float(mp_half_extent)
        self.stop_halfwidth = float(stop_halfwidth)
        self.stop_kind = stop_kind
        self.pinhole_radius = None if pinhole_radius is None else float(pinhole_radius)
        self.fill_factor = float(fill_factor)
        self.illumination = illumination

    def __check_init__(self):
        """Validate configuration and the intermediate-grid Nyquist budgets."""
        if self.illumination not in ("psf", "flat"):
            raise ValueError(
                f"illumination must be 'psf' or 'flat', got {self.illumination!r}"
            )
        if self.stop_kind not in ("square", "circular"):
            raise ValueError(
                f"stop_kind must be 'square' or 'circular', got {self.stop_kind!r}"
            )
        if self.pupil_field.ndim not in (2, 3):
            raise ValueError(
                "pupil_field must be (n, n) or (n_lam, n, n), got shape "
                f"{self.pupil_field.shape}"
            )
        npix = self.pupil_grid.shape[0]
        if self.pupil_field.shape[-2:] != (npix, npix):
            raise ValueError(
                f"pupil_field shape {self.pupil_field.shape} does not match "
                f"pupil_grid ({npix}, {npix})"
            )
        if not 0.0 < self.fill_factor <= 1.0:
            raise ValueError(f"fill_factor must be in (0, 1], got {self.fill_factor}")
        if self.stop_opd_nm is not None and self.stop_opd_nm.shape != (
            self.n_stop,
            self.n_stop,
        ):
            raise ValueError(
                f"stop_opd_nm shape {self.stop_opd_nm.shape} does not match "
                f"the stop grid ({self.n_stop}, {self.n_stop})"
            )
        # tile -> micro-pupil kernel: tile spacing 1/n_tile resolves
        # |u_mp| < n_tile / 2 unambiguously.
        if self.mp_half_extent > self.n_tile / 2:
            raise ValueError(
                f"mp_half_extent {self.mp_half_extent} exceeds the tile grid's "
                f"Nyquist half-band {self.n_tile / 2}; raise n_tile"
            )
        # micro-pupil -> stop kernel: mp spacing resolves
        # |x_s| < n_mp / (4 * mp_half_extent).
        if self.stop_halfwidth > self.n_mp / (4.0 * self.mp_half_extent):
            raise ValueError(
                f"stop_halfwidth {self.stop_halfwidth} exceeds the micro-pupil "
                f"grid's Nyquist half-band "
                f"{self.n_mp / (4.0 * self.mp_half_extent)}; raise n_mp"
            )

    @property
    def tile_grid(self):
        """The lenslet tile grid: one pitch across, half-pixel-offset."""
        return PixelGrid.pupil(self.n_tile)

    @property
    def mp_grid(self):
        """The micro-pupil grid [diffraction units]."""
        return PixelGrid.regular(
            (self.n_mp, self.n_mp),
            2.0 * self.mp_half_extent / self.n_mp,
            "pupil_diameter",
        )

    @property
    def stop_grid(self):
        """The spectrograph stop grid [pitch units], spanning the stop."""
        return PixelGrid.regular(
            (self.n_stop, self.n_stop),
            2.0 * self.stop_halfwidth / self.n_stop,
            "pupil_diameter",
        )

    def pitch_lod(self, wavelength_nm):
        """Lenslet pitch in lambda/D at ``wavelength_nm`` (fixed sky angle).

        Args:
            wavelength_nm: Wavelength in nanometers.

        Returns:
            Lenslet pitch in lambda/D at that wavelength.
        """
        return self.pitch_lod_ref * self.lam_ref_nm / wavelength_nm

    def px_per_diffraction(self, wavelength_nm):
        """Detector pixels per lenslet diffraction unit at ``wavelength_nm``.

        Args:
            wavelength_nm: Wavelength in nanometers.

        Returns:
            Detector pixels per lenslet diffraction unit at that wavelength.
        """
        return self.micropupil_px / self.pitch_lod(wavelength_nm)

    def _axis(self, grid):
        return like(grid_axis(grid), self)

    def _window(self):
        c = self._axis(self.tile_grid)
        half = 0.5 * self.fill_factor
        inside = (jnp.abs(c[:, None]) <= half) & (jnp.abs(c[None, :]) <= half)
        return inside.astype(self.pupil_field.real.dtype)

    def local_field(self, wavelength_nm, pupil=None):
        """Windowed complex field over the lenslet tile.

        Args:
            wavelength_nm: Wavelength [nm].
            pupil: Exit-pupil slice to illuminate with; defaults to
                ``pupil_field`` (which must then be 2D).

        Returns:
            Complex tile field, ``(n_tile, n_tile)``, aperture window applied.
        """
        if pupil is None:
            if self.pupil_field.ndim != 2:
                raise ValueError(
                    "per-wavelength pupil_field needs an explicit pupil slice"
                )
            pupil = self.pupil_field
        tile = self._axis(self.tile_grid)
        if self.illumination == "flat":
            field = jnp.ones((self.n_tile, self.n_tile), dtype=pupil.dtype)
        else:
            u_tile = tile * like(self.pitch_lod(wavelength_nm), tile)
            field = cmft_fwd(pupil, self._axis(self.pupil_grid), u_tile)
        return field * self._window()

    def micropupil_field(self, wavelength_nm, pupil=None):
        """Complex field at the micro-pupil plane, pinhole applied if any.

        Args:
            wavelength_nm: Wavelength in nanometers.
            pupil: Exit-pupil slice to illuminate with; defaults to pupil_field, which
                must then be 2D.

        Returns:
            Complex field on the micro-pupil grid, (n_mp, n_mp), with the pinhole
            applied if there is one.
        """
        windowed = self.local_field(wavelength_nm, pupil)
        u_mp = self._axis(self.mp_grid)
        field = cmft_fwd(windowed, self._axis(self.tile_grid), u_mp)
        if self.pinhole_radius is not None:
            rr2 = u_mp[:, None] ** 2 + u_mp[None, :] ** 2
            field = field * (rr2 <= self.pinhole_radius**2)
        return field

    def _relay_to_stop(self, tile_field, wavelength_nm, shift_diffraction=(0.0, 0.0)):
        """Windowed tile field -> field just after the spectrograph stop."""
        u_mp = self._axis(self.mp_grid)
        field = cmft_fwd(tile_field, self._axis(self.tile_grid), u_mp)
        if self.pinhole_radius is not None:
            rr2 = u_mp[:, None] ** 2 + u_mp[None, :] ** 2
            field = field * (rr2 <= self.pinhole_radius**2)
        x_s = self._axis(self.stop_grid)
        field = cmft_fwd(field, u_mp, x_s)
        if self.stop_kind == "square":
            inside = (jnp.abs(x_s[:, None]) <= self.stop_halfwidth) & (
                jnp.abs(x_s[None, :]) <= self.stop_halfwidth
            )
        else:
            inside = x_s[:, None] ** 2 + x_s[None, :] ** 2 <= self.stop_halfwidth**2
        field = field * inside
        if self.stop_opd_nm is not None:
            field = field * jnp.exp(
                1j * (two_pi * (self.stop_opd_nm / like(wavelength_nm, tile_field)))
            )
        dx, dy = shift_diffraction
        if is_traced(dx) or is_traced(dy) or dx != 0.0 or dy != 0.0:
            phase = x_s[None, :] * like(dx, x_s) + x_s[:, None] * like(dy, x_s)
            field = field * jnp.exp(2j * jnp.pi * phase)
        return field

    def stop_field(self, wavelength_nm, pupil=None, shift_diffraction=(0.0, 0.0)):
        """Complex field just after the spectrograph stop.

        ``shift_diffraction`` is the dispersion hook: a ``(dx, dy)`` tilt
        phase in diffraction units that translates the detector image by the
        shift theorem (a disperser is exactly this, with a
        wavelength-dependent shift).

        Args:
            wavelength_nm: Wavelength in nanometers.
            pupil: Exit-pupil slice to illuminate with; defaults to pupil_field, which
                must then be 2D.
            shift_diffraction: (dx, dy) tilt in diffraction units that translates the
                detector image.

        Returns:
            Complex field just after the stop, on the stop grid, (n_stop, n_stop).
        """
        windowed = self.local_field(wavelength_nm, pupil)
        return self._relay_to_stop(windowed, wavelength_nm, shift_diffraction)

    def tile_fields(self, wavelength_nm, centers_pitch, pupil=None):
        """Windowed local fields at every lenslet, one two-axis MFT each.

        The full-array move: the upstream path ends at a pupil and the MFT
        has free output sampling, so each lenslet's local field is one MFT
        from the pupil onto that lenslet's tile -- no global fine focal grid
        is ever materialized, and mutual coherence between tiles is carried
        exactly (all tiles view one complex pupil field).

        Args:
            wavelength_nm: Wavelength [nm].
            centers_pitch: ``(n_lenslets, 2)`` lenslet centers ``(x, y)`` in
                units of the lenslet pitch on sky (wavelength-independent;
                rotation already applied).
            pupil: Optional exit-pupil slice (chromatic ``pupil_field``).

        Returns:
            Complex tile fields, ``(n_lenslets, n_tile, n_tile)``, aperture
            window applied.
        """
        if self.illumination != "psf":
            raise ValueError("array tile fields need illumination='psf'")
        if pupil is None:
            if self.pupil_field.ndim != 2:
                raise ValueError(
                    "per-wavelength pupil_field needs an explicit pupil slice"
                )
            pupil = self.pupil_field
        tile = self._axis(self.tile_grid)
        p_lod = like(self.pitch_lod(wavelength_nm), tile)
        window = self._window()
        xp = self._axis(self.pupil_grid)

        def one(center):
            uy = (tile + center[1]) * p_lod
            ux = (tile + center[0]) * p_lod
            return cmft_fwd(pupil, xp, uy, ux) * window

        return jax.vmap(one)(like(centers_pitch, tile))

    def detector_patch(
        self, tile_field, wavelength_nm, off_y, off_x, shift_diffraction=(0.0, 0.0)
    ):
        """Complex detector field of one lenslet at the given pixel offsets.

        ``off_y`` / ``off_x`` are 1D offsets [px] from the lenslet's PSFlet
        centroid; fractional offsets are evaluated exactly by the MFT (no
        interpolation).

        Args:
            tile_field: Windowed complex tile field, (n_tile, n_tile), as returned by
                local_field.
            wavelength_nm: Wavelength in nanometers.
            off_y: 1D row offsets in pixels from the lenslet PSFlet centroid.
            off_x: 1D column offsets in pixels from the lenslet PSFlet centroid.
            shift_diffraction: (dx, dy) tilt in diffraction units that translates the
                detector image.

        Returns:
            Complex detector field of one lenslet, shape (len(off_y), len(off_x)).
        """
        stop = self._relay_to_stop(tile_field, wavelength_nm, shift_diffraction)
        ppd = like(self.px_per_diffraction(wavelength_nm), stop)
        x_s = self._axis(self.stop_grid)
        return cmft_fwd(
            stop, x_s, like(off_y, tile_field) / ppd, like(off_x, tile_field) / ppd
        )

    def detector_field(
        self, wavelength_nm, px_coords, pupil=None, shift_diffraction=(0.0, 0.0)
    ):
        """Complex detector-plane field sampled at ``px_coords`` [px].

        ``px_coords`` is the 1D pixel-offset axis shared by rows (dy) and
        columns (dx); the origin is the unshifted PSFlet center.

        Args:
            wavelength_nm: Wavelength in nanometers.
            px_coords: 1D pixel-offset axis in pixels shared by rows and columns; the
                origin is the unshifted PSFlet center.
            pupil: Exit-pupil slice to illuminate with; defaults to pupil_field, which
                must then be 2D.
            shift_diffraction: (dx, dy) tilt in diffraction units that translates the
                detector image.

        Returns:
            Complex detector-plane field, (len(px_coords), len(px_coords)), rows y.
        """
        stop = self.stop_field(wavelength_nm, pupil, shift_diffraction)
        u_det = like(px_coords, stop) / like(
            self.px_per_diffraction(wavelength_nm), stop
        )
        return cmft_fwd(stop, self._axis(self.stop_grid), u_det)

    def psflet_intensity(
        self, wavelength_nm, px_coords, pupil=None, shift_diffraction=(0.0, 0.0)
    ):
        """Detector-plane PSFlet intensity at ``px_coords`` [px], ``(y, x)``.

        Args:
            wavelength_nm: Wavelength in nanometers.
            px_coords: 1D pixel-offset axis in pixels shared by rows and columns.
            pupil: Exit-pupil slice to illuminate with; defaults to pupil_field, which
                must then be 2D.
            shift_diffraction: (dx, dy) tilt in diffraction units that translates the
                detector image.

        Returns:
            Squared magnitude of the detector field, (y, x), in the field power units of
            the pupil field.
        """
        field = self.detector_field(wavelength_nm, px_coords, pupil, shift_diffraction)
        return field.real**2 + field.imag**2

    def energies(self, wavelength_nm, pupil=None):
        """Capture diagnostics: energy after the window, micro-pupil, stop.

        Returns the tuple ``(tile, micropupil, stop)`` of plane energies on
        each plane's own cell measure; ratios are the capture fractions the
        sampling knobs are validated with.

        Args:
            wavelength_nm: Wavelength in nanometers.
            pupil: Exit-pupil slice to illuminate with; defaults to pupil_field, which
                must then be 2D.

        Returns:
            Tuple (tile, micropupil, stop) of scalar plane energies, each on its own
            plane cell measure.
        """
        windowed = self.local_field(wavelength_nm, pupil)
        tile = jnp.sum(jnp.abs(windowed) ** 2) * self.tile_grid.pixel_area
        mp = self.micropupil_field(wavelength_nm, pupil)
        e_mp = jnp.sum(jnp.abs(mp) ** 2) * self.mp_grid.pixel_area
        stop = self.stop_field(wavelength_nm, pupil)
        e_stop = jnp.sum(jnp.abs(stop) ** 2) * self.stop_grid.pixel_area
        return tile, e_mp, e_stop


def pixel_integrate(intensity, n_quad, stride):
    """Box-average a fine intensity grid into pixel-integrated samples.

    ``intensity`` is sampled with ``n_quad`` points per detector pixel
    (midpoint rule); the unit-pixel integral centered on every ``stride``-th
    fine sample is the ``n_quad x n_quad`` window mean. Returns the
    ``(n_out, n_out)`` array with ``n_out = (n_fine - n_quad) // stride + 1``.

    Args:
        intensity: Fine intensity grid, 2D, sampled with n_quad points per detector
            pixel.
        n_quad: Fine samples per detector pixel along each axis.
        stride: Spacing in fine samples between successive pixel centers.

    Returns:
        Pixel-integrated array of shape (n_out, n_out), in units of intensity times
        pixel area; empty along an axis with too few fine samples.
    """
    ny, nx = (max(0, (n - n_quad) // stride + 1) for n in intensity.shape)
    if not ny or not nx:
        return jnp.zeros((ny, nx), dtype=jnp.result_type(intensity, 1.0))
    # Unit detector pixels surround n_quad fine samples in each dimension.
    # For overlapping windows or gaps, reuse one pixel's translated footprint.
    shape = (ny, nx) if stride == n_quad else (1, 1)
    fine = PixelGrid.regular(tuple(n * n_quad for n in shape), 1 / n_quad, "pixel")
    pixels = PixelGrid.regular(shape, 1, "pixel")
    overlap = footprint_overlap(fine, pixels)

    def collect(window):
        image = Image(window, fine, (), Measure.density("grid"), "field_power")
        return integrate_footprints(image, overlap).data

    if stride == n_quad:
        return collect(intensity[: ny * n_quad, : nx * n_quad])

    def integrate_window(y, x):
        return collect(lax.dynamic_slice(intensity, (y, x), fine.shape))[0, 0]

    return jax.vmap(
        lambda y: jax.vmap(lambda x: integrate_window(y, x))(jnp.arange(nx) * stride)
    )(jnp.arange(ny) * stride)


def _fine_axis(offsets, n_quad):
    """The fine sampling axis whose windowed means are the pixel integrals.

    For every offset ``t`` the pixel integral needs midpoint sub-samples
    ``t - 1/2 + (k + 1/2) / n_quad``; with the offset step a multiple of
    ``1 / n_quad`` all sub-samples of all offsets share one uniform axis.
    """
    step = float(offsets[1] - offsets[0])
    fine = 1.0 / n_quad
    stride = step / fine
    if not same(stride, round(stride)):
        raise ValueError(
            f"offset step {step} must be an integer multiple of 1/n_quad "
            f"= {fine} for exact pixel integration"
        )
    stride = round(stride)
    n_fine = (len(offsets) - 1) * stride + n_quad
    start = float(offsets[0]) - 0.5 + 0.5 * fine
    return start + fine * np.arange(n_fine), stride


def _check_stop_band(chain, wavelength_nm, fine, cx, cy, ppd):
    """Check the shifted endpoints on both detector axes.

    Args:
        chain: The ``LensletChain`` that defines the stop grid.
        wavelength_nm: Wavelength [nm].
        fine: Ascending axis of fine sample offsets [px].
        cx: Accumulated x shift [px].
        cy: Accumulated y shift [px].
        ppd: Detector pixels per diffraction unit.

    Raises:
        ValueError: A shifted endpoint exceeds the stop grid's Nyquist half-band.
    """
    lo, hi = float(fine[0]), float(fine[-1])
    u_max = max(abs(lo + cx), abs(hi + cx), abs(lo + cy), abs(hi + cy)) / float(ppd)
    u_nyquist = chain.n_stop / (4.0 * chain.stop_halfwidth)
    if u_max > u_nyquist:
        raise ValueError(
            f"shifted render at {wavelength_nm:g} nm samples out to {u_max:.2f} "
            "diffraction units, beyond the stop grid's Nyquist half-band "
            f"{u_nyquist:.2f}. Raise n_stop or shrink half_extent."
        )


def psflet_template(chain, wavelength_nm, offsets, pupil=None, n_quad=8):
    """Pixel-integrated PSFlet template plane and its intensity centroid.

    Args:
        chain: The ``LensletChain``.
        wavelength_nm: Wavelength [nm].
        offsets: Uniform ascending pixel-offset axis (shared dy/dx).
        pupil: Optional exit-pupil slice (chromatic ``pupil_field``).
        n_quad: Midpoint sub-samples per pixel per axis.

    Returns:
        ``(template, centroid)``: the ``(n_off, n_off)`` template plane
        (rows dy, columns dx) about its own centroid and the accumulated
        offset from the geometric origin in detector pixels, ordered (cx, cy).

    Raises:
        RuntimeError: The centroid did not converge in five renders.
        ValueError: The offset step is incompatible with ``n_quad``, a required
            pupil slice is missing, or a shifted render exceeds the stop grid's
            Nyquist half-band.

    The centroid iteration runs on the host, so the wavelength and offsets
    must be concrete (the function cannot be traced by ``jit`` or ``vmap``),
    and ``(cx, cy)`` are NumPy scalars.
    """
    host_fine, stride = _fine_axis(np.asarray(offsets), n_quad)
    fine = like(host_fine, chain.pupil_field)
    offsets = like(offsets, chain.pupil_field)
    stop = chain.stop_field(wavelength_nm, pupil)
    x_stop = chain._axis(chain.stop_grid)
    host_ppd = chain.px_per_diffraction(wavelength_nm)
    ppd = like(host_ppd, stop)
    cx = cy = 0.0
    for render in range(5):
        _check_stop_band(chain, wavelength_nm, host_fine, cx, cy, host_ppd)
        field = cmft_fwd(stop, x_stop, (fine + cy) / ppd, (fine + cx) / ppd)
        template = pixel_integrate(field.real**2 + field.imag**2, n_quad, stride)
        # Measure on the stored pixel integrals in their local coordinates.
        moment = (
            jnp.stack((template.sum(axis=0) @ offsets, template.sum(axis=1) @ offsets))
            / template.sum()
        )
        dx, dy = np.asarray(moment)
        if render and np.hypot(dx, dy) < 1e-3:
            return template, (cx, cy)
        cx += dx
        cy += dy

    raise RuntimeError(
        f"PSFlet centroid at {wavelength_nm:g} nm did not converge in five passes"
    )


def psflet_pack(chain, wavelengths_nm, *, half_extent=6.0, step=0.125, n_quad=8):
    """Tabulate the chain into a PSFlet template pack.

    Each template is centered on its own centroid, and ``centroids`` gives its
    offset from the dispersion-model centroid in detector pixels.
    The pack declares this origin as ``template_origin="centroid"``.

    Write the pack with ``optixstuff.save_psflet_pack(path, pack)``.

    One field anchor at the lenslet-grid origin (field dependence enters by
    emitting packs per field region as upstream fields become available).

    Args:
        chain: The ``LensletChain``.
        wavelengths_nm: Ascending tabulation wavelengths. A chromatic
            ``chain.pupil_field`` must carry one slice per wavelength.
        half_extent: Tabulated offset extent [px].
        step: Offset grid spacing [px]; must be a multiple of ``1/n_quad``.
        n_quad: Midpoint sub-samples per pixel per axis.

    Returns:
        An :class:`optixstuff.PsfletPack` whose ``meta_json`` provenance
        carries the chain configuration and per-wavelength capture fractions.

    Raises:
        RuntimeError: A template's centroid did not converge.
        ValueError: The wavelength grid or chromatic pupil slice count is invalid,
            the offset step is incompatible with ``n_quad``, or the detector
            samples exceed the stop grid's Nyquist half-band.
    """
    lams = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    if lams.ndim != 1 or (len(lams) > 1 and not np.all(np.diff(lams) > 0)):
        raise ValueError("wavelengths_nm must be 1D ascending")
    if chain.pupil_field.ndim == 3 and chain.pupil_field.shape[0] != len(lams):
        raise ValueError(
            f"chromatic pupil_field has {chain.pupil_field.shape[0]} slices "
            f"but {len(lams)} wavelengths were requested"
        )
    # The shortest wavelength has the fewest pixels per diffraction unit,
    # so the full pixel extent spans the most diffraction units there.
    shortest_wavelength_nm = lams.min()
    u_det_max = (half_extent + 0.5) / float(
        chain.px_per_diffraction(shortest_wavelength_nm)
    )
    u_det_nyquist = chain.n_stop / (4.0 * chain.stop_halfwidth)
    if u_det_max > u_det_nyquist:
        raise ValueError(
            f"detector extent {u_det_max:.2f} diffraction units at "
            f"{shortest_wavelength_nm:g} nm exceeds the stop grid's Nyquist "
            f"half-band {u_det_nyquist:.2f}. Raise n_stop or shrink half_extent."
        )
    n_off = round(2.0 * half_extent / step) + 1
    offsets = np.linspace(-half_extent, half_extent, n_off)

    planes, cents, capture = [], [], []
    for i, lam in enumerate(lams):
        pupil = chain.pupil_field[i] if chain.pupil_field.ndim == 3 else None
        template, (cx, cy) = psflet_template(
            chain, lam, offsets, pupil=pupil, n_quad=n_quad
        )
        planes.append(np.asarray(template))
        cents.append([float(cx), float(cy)])
        e_tile, e_mp, e_stop = chain.energies(lam, pupil)
        capture.append(
            {
                "wavelength_nm": float(lam),
                "micropupil_capture": float(e_mp / e_tile),
                "stop_capture": float(e_stop / e_tile),
            }
        )

    meta = {
        "generator": "physicaloptix.ifs.LensletChain",
        "pitch_lod_ref": chain.pitch_lod_ref,
        "lam_ref_nm": chain.lam_ref_nm,
        "micropupil_px": chain.micropupil_px,
        "mp_half_extent": chain.mp_half_extent,
        "stop_halfwidth": chain.stop_halfwidth,
        "stop_kind": chain.stop_kind,
        "pinhole_radius": chain.pinhole_radius,
        "fill_factor": chain.fill_factor,
        "illumination": chain.illumination,
        "aberrated": chain.stop_opd_nm is not None,
        "n_quad": n_quad,
        "capture": capture,
    }
    return PsfletPack(
        templates=np.stack(planes)[None],
        offsets=offsets,
        wavelengths_nm=lams,
        field_xy=np.zeros((1, 2)),
        template_origin=TEMPLATE_ORIGIN,
        centroids=np.asarray(cents, dtype=float)[None],
        meta_json=json.dumps(meta),
    )


def detector_scene(
    chain,
    centers_pitch,
    xc_px,
    yc_px,
    wavelengths_nm,
    *,
    det_shape,
    weights=None,
    oversample=4,
    patch_half_px=6.0,
):
    """Full-array COHERENT detector scene, pixel-integrated.

    The wave view of the array that the sparse-H model approximates: every
    lenslet's complex detector patch is accumulated on one fine detector
    lattice per wavelength -- adjacent lenslets sample mutually coherent
    field, so their overlapping PSFlet wings interfere, the term an
    incoherent template add cannot represent (Antichi et al. 2009's
    coherent-vs-incoherent crosstalk distinction). The accumulated field is
    squared, midpoint pixel-integrated (the pack convention), and summed
    incoherently over wavelength.

    Geometry comes in as data so the wave view and a sparse-template view
    can share it exactly: ``centers_pitch`` are the lenslet centers on sky
    and ``xc_px`` / ``yc_px`` the per-(lenslet, wavelength) detector
    centroids of the caller's dispersion model.

    Args:
        chain: The ``LensletChain`` (``illumination="psf"``).
        centers_pitch: ``(n_lenslets, 2)`` lenslet centers ``(x, y)`` in
            lenslet pitches on sky.
        xc_px: ``(n_lenslets, n_lam)`` detector centroid columns [px].
        yc_px: ``(n_lenslets, n_lam)`` detector centroid rows [px].
        wavelengths_nm: 1D wavelength samples [nm]. A chromatic
            ``chain.pupil_field`` must carry one slice per sample.
        det_shape: Detector ``(ny, nx)`` [px]; pixel centers at integer
            coordinates.
        weights: Per-wavelength incoherent weights (default uniform mean).
        oversample: Fine samples per pixel per axis (the midpoint rule).
        patch_half_px: Half-width of each lenslet's accumulation window
            [px]; the window must stay inside the detector.

    Returns:
        The ``(ny, nx)`` pixel-integrated detector intensity.
    """
    lams = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    xc = np.asarray(xc_px, dtype=float)
    yc = np.asarray(yc_px, dtype=float)
    n_lenslets = np.asarray(centers_pitch).shape[0]
    if xc.shape != (n_lenslets, len(lams)) or yc.shape != (n_lenslets, len(lams)):
        raise ValueError(
            f"centroid tables must be (n_lenslets, n_lam) = "
            f"({n_lenslets}, {len(lams)}), got {xc.shape} and {yc.shape}"
        )
    ny, nx = det_shape
    ovs = int(oversample)
    n_win = 2 * round(patch_half_px * ovs)
    w = (
        np.full(len(lams), 1.0 / len(lams))
        if weights is None
        else np.asarray(weights, dtype=float)
    )

    # fine index f covers pixel i = f // ovs at coordinate (f + 0.5)/ovs - 0.5;
    # a patch starts at the fine index putting its window center on the centroid.
    fy0 = np.round((yc + 0.5) * ovs - 0.5).astype(int) - n_win // 2
    fx0 = np.round((xc + 0.5) * ovs - 0.5).astype(int) - n_win // 2
    if (
        fy0.min() < 0
        or fx0.min() < 0
        or fy0.max() > ny * ovs - n_win
        or fx0.max() > nx * ovs - n_win
    ):
        raise ValueError(
            "a lenslet accumulation window leaves the detector; grow det_shape "
            "or shrink patch_half_px"
        )

    def first_offset(start, centroid):
        # Subtract the centroid from the whole pixel before adding the
        # subpixel midpoint, so translating the patch preserves its samples.
        pixel, subpixel = np.divmod(start, ovs)
        return (pixel - centroid) + ((subpixel + 0.5) / ovs - 0.5)

    first_y = first_offset(fy0, yc)
    first_x = first_offset(fx0, xc)
    total = jnp.zeros((ny, nx), dtype=precision_of(chain))
    ramp = jnp.arange(n_win, dtype=total.dtype) / ovs
    for i, lam in enumerate(lams):
        pupil = chain.pupil_field[i] if chain.pupil_field.ndim == 3 else None
        tiles = chain.tile_fields(lam, centers_pitch, pupil)

        def add_patch(canvas, per_lenslet, lam=lam):
            tile, y0, x0, dy, dx = per_lenslet
            patch = chain.detector_patch(tile, lam, dy + ramp, dx + ramp)
            block = lax.dynamic_slice(canvas, (y0, x0), (n_win, n_win))
            return lax.dynamic_update_slice(canvas, block + patch, (y0, x0)), None

        canvas = jnp.zeros((ny * ovs, nx * ovs), dtype=tiles.dtype)
        canvas, _ = lax.scan(
            add_patch,
            canvas,
            (
                tiles,
                jnp.asarray(fy0[:, i], dtype=jnp.int32),
                jnp.asarray(fx0[:, i], dtype=jnp.int32),
                like(first_y[:, i], tiles),
                like(first_x[:, i], tiles),
            ),
        )
        fine = jnp.abs(canvas) ** 2
        pixels = pixel_integrate(fine, ovs, ovs)
        total = total + like(w[i], pixels) * pixels
    return total
