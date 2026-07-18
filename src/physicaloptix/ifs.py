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
pixel-integrated templates on a sub-pixel offset grid and ``save_psflet_pack``
writes the documented npz pack format (``format_version`` 1: ``templates``
``(n_field, n_lam, n_off, n_off)``, shared ``offsets``, ``wavelengths_nm``,
``field_xy``, optional ``centroids`` wavecal corrections, ``meta_json``)
consumed by downstream spectral-extraction codes via their template mode.
"""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from physicaloptix.core import Grid
from physicaloptix.transforms.cmft import cmft_fwd

PACK_FORMAT_VERSION = 1


def _cmft2(f, x, uy, ux):
    """Two-axis continuous-FT MFT: separate row (uy) and column (ux) outputs.

    The library ``cmft_fwd`` special case has ``uy == ux``; per-lenslet tiles
    need independent axis offsets (a lenslet center displaces both axes).
    """
    ky = jnp.exp(-2j * jnp.pi * jnp.outer(uy, x))
    kx = jnp.exp(-2j * jnp.pi * jnp.outer(ux, x))
    dx = x[1] - x[0]
    return dx**2 * (ky @ f @ kx.T)


class LensletChain(eqx.Module):
    """Single-lenslet wave chain from the telescope exit pupil to the detector.

    Attributes:
        pupil_field: Complex exit-pupil field, ``(n, n)`` shared across
            wavelengths or ``(n_lam, n, n)`` per-wavelength (matched against
            the pack wavelengths at build time).
        pupil_grid: Grid of ``pupil_field`` (pupil diameters).
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

    pupil_field: Array = eqx.field(converter=jnp.asarray)
    pupil_grid: Grid
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
        """Build the chain configuration (see the class docstring)."""
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
        npix = self.pupil_grid.npix
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
        return Grid.pupil(self.n_tile)

    @property
    def mp_grid(self):
        """The micro-pupil grid [diffraction units]."""
        return Grid(npix=self.n_mp, dx=2.0 * self.mp_half_extent / self.n_mp)

    @property
    def stop_grid(self):
        """The spectrograph stop grid [pitch units], spanning the stop."""
        return Grid(npix=self.n_stop, dx=2.0 * self.stop_halfwidth / self.n_stop)

    def pitch_lod(self, wavelength_nm):
        """Lenslet pitch in lambda/D at ``wavelength_nm`` (fixed sky angle)."""
        return self.pitch_lod_ref * self.lam_ref_nm / wavelength_nm

    def px_per_diffraction(self, wavelength_nm):
        """Detector pixels per lenslet diffraction unit at ``wavelength_nm``."""
        return self.micropupil_px / self.pitch_lod(wavelength_nm)

    def _window(self):
        c = jnp.asarray(self.tile_grid.coords)
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
        tile = jnp.asarray(self.tile_grid.coords)
        if self.illumination == "flat":
            field = jnp.ones((self.n_tile, self.n_tile), dtype=pupil.dtype)
        else:
            u_tile = tile * self.pitch_lod(wavelength_nm)
            field = cmft_fwd(pupil, jnp.asarray(self.pupil_grid.coords), u_tile)
        return field * self._window()

    def micropupil_field(self, wavelength_nm, pupil=None):
        """Complex field at the micro-pupil plane, pinhole applied if any."""
        windowed = self.local_field(wavelength_nm, pupil)
        u_mp = jnp.asarray(self.mp_grid.coords)
        field = cmft_fwd(windowed, jnp.asarray(self.tile_grid.coords), u_mp)
        if self.pinhole_radius is not None:
            rr2 = u_mp[:, None] ** 2 + u_mp[None, :] ** 2
            field = field * (rr2 <= self.pinhole_radius**2)
        return field

    def _relay_to_stop(self, tile_field, wavelength_nm, shift_diffraction=(0.0, 0.0)):
        """Windowed tile field -> field just after the spectrograph stop."""
        u_mp = jnp.asarray(self.mp_grid.coords)
        field = cmft_fwd(tile_field, jnp.asarray(self.tile_grid.coords), u_mp)
        if self.pinhole_radius is not None:
            rr2 = u_mp[:, None] ** 2 + u_mp[None, :] ** 2
            field = field * (rr2 <= self.pinhole_radius**2)
        x_s = jnp.asarray(self.stop_grid.coords)
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
                2j * jnp.pi * self.stop_opd_nm / jnp.asarray(wavelength_nm)
            )
        dx, dy = shift_diffraction
        if dx != 0.0 or dy != 0.0:
            phase = x_s[None, :] * dx + x_s[:, None] * dy
            field = field * jnp.exp(2j * jnp.pi * phase)
        return field

    def stop_field(self, wavelength_nm, pupil=None, shift_diffraction=(0.0, 0.0)):
        """Complex field just after the spectrograph stop.

        ``shift_diffraction`` is the dispersion hook: a ``(dx, dy)`` tilt
        phase in diffraction units that translates the detector image by the
        shift theorem (a disperser is exactly this, with a
        wavelength-dependent shift).
        """
        windowed = self.local_field(wavelength_nm, pupil)
        return self._relay_to_stop(windowed, wavelength_nm, shift_diffraction)

    def tile_fields(self, wavelength_nm, centers_pitch, pupil=None):
        """Windowed local fields at every lenslet, one two-axis MFT each.

        The full-array move: the upstream train ends at a pupil and the MFT
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
        tile = jnp.asarray(self.tile_grid.coords)
        p_lod = self.pitch_lod(wavelength_nm)
        window = self._window()
        xp = jnp.asarray(self.pupil_grid.coords)

        def one(center):
            uy = (tile + center[1]) * p_lod
            ux = (tile + center[0]) * p_lod
            return _cmft2(pupil, xp, uy, ux) * window

        return jax.vmap(one)(jnp.asarray(centers_pitch))

    def detector_patch(
        self, tile_field, wavelength_nm, off_y, off_x, shift_diffraction=(0.0, 0.0)
    ):
        """Complex detector field of one lenslet at the given pixel offsets.

        ``off_y`` / ``off_x`` are 1D offsets [px] from the lenslet's PSFlet
        centroid; fractional offsets are evaluated exactly by the MFT (no
        interpolation).
        """
        stop = self._relay_to_stop(tile_field, wavelength_nm, shift_diffraction)
        ppd = self.px_per_diffraction(wavelength_nm)
        x_s = jnp.asarray(self.stop_grid.coords)
        return _cmft2(stop, x_s, jnp.asarray(off_y) / ppd, jnp.asarray(off_x) / ppd)

    def detector_field(
        self, wavelength_nm, px_coords, pupil=None, shift_diffraction=(0.0, 0.0)
    ):
        """Complex detector-plane field sampled at ``px_coords`` [px].

        ``px_coords`` is the 1D pixel-offset axis shared by rows (dy) and
        columns (dx); the origin is the unshifted PSFlet center.
        """
        stop = self.stop_field(wavelength_nm, pupil, shift_diffraction)
        u_det = jnp.asarray(px_coords) / self.px_per_diffraction(wavelength_nm)
        return cmft_fwd(stop, jnp.asarray(self.stop_grid.coords), u_det)

    def psflet_intensity(
        self, wavelength_nm, px_coords, pupil=None, shift_diffraction=(0.0, 0.0)
    ):
        """Detector-plane PSFlet intensity at ``px_coords`` [px], ``(y, x)``."""
        field = self.detector_field(wavelength_nm, px_coords, pupil, shift_diffraction)
        return field.real**2 + field.imag**2

    def energies(self, wavelength_nm, pupil=None):
        """Capture diagnostics: energy after the window, micro-pupil, stop.

        Returns the tuple ``(tile, micropupil, stop)`` of plane energies on
        each plane's own cell measure; ratios are the capture fractions the
        sampling knobs are validated with.
        """
        windowed = self.local_field(wavelength_nm, pupil)
        tile = jnp.sum(jnp.abs(windowed) ** 2) * self.tile_grid.weights
        mp = self.micropupil_field(wavelength_nm, pupil)
        e_mp = jnp.sum(jnp.abs(mp) ** 2) * self.mp_grid.weights
        stop = self.stop_field(wavelength_nm, pupil)
        e_stop = jnp.sum(jnp.abs(stop) ** 2) * self.stop_grid.weights
        return tile, e_mp, e_stop


def pixel_integrate(intensity, n_quad, stride):
    """Box-average a fine intensity grid into pixel-integrated samples.

    ``intensity`` is sampled with ``n_quad`` points per detector pixel
    (midpoint rule); the unit-pixel integral centered on every ``stride``-th
    fine sample is the ``n_quad x n_quad`` window mean. Returns the
    ``(n_out, n_out)`` array with ``n_out = (n_fine - n_quad) // stride + 1``.
    """
    window = (n_quad, n_quad)
    strides = (stride, stride)
    zero = jnp.zeros((), dtype=intensity.dtype)
    summed = lax.reduce_window(intensity, zero, lax.add, window, strides, "VALID")
    return summed / (n_quad * n_quad)


def _fine_axis(offsets, n_quad):
    """The fine sampling axis whose windowed means are the pixel integrals.

    For every offset ``t`` the pixel integral needs midpoint sub-samples
    ``t - 1/2 + (k + 1/2) / n_quad``; with the offset step a multiple of
    ``1 / n_quad`` all sub-samples of all offsets share one uniform axis.
    """
    step = float(offsets[1] - offsets[0])
    fine = 1.0 / n_quad
    stride = step / fine
    if abs(stride - round(stride)) > 1e-9:
        raise ValueError(
            f"offset step {step} must be an integer multiple of 1/n_quad "
            f"= {fine} for exact pixel integration"
        )
    stride = round(stride)
    n_fine = (len(offsets) - 1) * stride + n_quad
    start = float(offsets[0]) - 0.5 + 0.5 * fine
    return start + fine * np.arange(n_fine), stride


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
        (rows dy, columns dx) and the ``(cx, cy)`` intensity centroid [px]
        measured on the fine grid (the wavecal residual when aberrations
        shift the PSFlet off the geometric centroid).
    """
    fine, stride = _fine_axis(np.asarray(offsets), n_quad)
    intensity = chain.psflet_intensity(wavelength_nm, jnp.asarray(fine), pupil)
    template = pixel_integrate(intensity, n_quad, stride)
    total = jnp.sum(intensity)
    fx = jnp.asarray(fine)
    cx = jnp.sum(intensity * fx[None, :]) / total
    cy = jnp.sum(intensity * fx[:, None]) / total
    return template, (cx, cy)


def psflet_pack(chain, wavelengths_nm, *, half_extent=6.0, step=0.125, n_quad=8):
    """Tabulate the chain into a PSFlet template pack payload.

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
        Dict with the pack arrays (``templates``, ``offsets``,
        ``wavelengths_nm``, ``field_xy``, ``centroids``) and ``meta_json``
        provenance carrying the chain configuration and per-wavelength
        capture fractions.
    """
    lams = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    if lams.ndim != 1 or (len(lams) > 1 and not np.all(np.diff(lams) > 0)):
        raise ValueError("wavelengths_nm must be 1D ascending")
    if chain.pupil_field.ndim == 3 and chain.pupil_field.shape[0] != len(lams):
        raise ValueError(
            f"chromatic pupil_field has {chain.pupil_field.shape[0]} slices "
            f"but {len(lams)} wavelengths were requested"
        )
    # stop -> detector kernel: the hard aliasing bound of the stop-grid sum
    # is |u_det| < n_stop / (4 * stop_halfwidth); the reddest wavelength has
    # the smallest px_per_diffraction, so it needs the widest u_det extent.
    u_det_max = (half_extent + 0.5) / float(chain.px_per_diffraction(lams.max()))
    u_det_nyquist = chain.n_stop / (4.0 * chain.stop_halfwidth)
    if u_det_max > u_det_nyquist:
        raise ValueError(
            f"detector extent {u_det_max:.2f} diffraction units exceeds the "
            f"stop grid's Nyquist half-band {u_det_nyquist:.2f}; raise n_stop "
            "or shrink half_extent"
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
    return {
        "templates": np.stack(planes)[None],
        "offsets": offsets,
        "wavelengths_nm": lams,
        "field_xy": np.zeros((1, 2)),
        "centroids": np.asarray(cents, dtype=float)[None],
        "meta_json": json.dumps(meta),
    }


def save_psflet_pack(path, pack):
    """Write a pack payload to the documented npz format (version 1)."""
    np.savez(
        path,
        format_version=np.int64(PACK_FORMAT_VERSION),
        templates=pack["templates"],
        offsets=pack["offsets"],
        wavelengths_nm=pack["wavelengths_nm"],
        field_xy=pack["field_xy"],
        centroids=pack["centroids"],
        meta_json=np.asarray(pack["meta_json"]),
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

    total = jnp.zeros((ny, nx))
    win = jnp.arange(n_win)
    for i, lam in enumerate(lams):
        pupil = chain.pupil_field[i] if chain.pupil_field.ndim == 3 else None
        tiles = chain.tile_fields(lam, centers_pitch, pupil)

        def add_patch(canvas, per_lenslet, lam=lam):
            tile, y0, x0, cy, cx = per_lenslet
            off_y = (y0 + win + 0.5) / ovs - 0.5 - cy
            off_x = (x0 + win + 0.5) / ovs - 0.5 - cx
            patch = chain.detector_patch(tile, lam, off_y, off_x)
            block = lax.dynamic_slice(canvas, (y0, x0), (n_win, n_win))
            return lax.dynamic_update_slice(canvas, block + patch, (y0, x0)), None

        canvas = jnp.zeros((ny * ovs, nx * ovs), dtype=tiles.dtype)
        canvas, _ = lax.scan(
            add_patch,
            canvas,
            (
                tiles,
                jnp.asarray(fy0[:, i]),
                jnp.asarray(fx0[:, i]),
                jnp.asarray(yc[:, i]),
                jnp.asarray(xc[:, i]),
            ),
        )
        fine = jnp.abs(canvas) ** 2
        pixels = fine.reshape(ny, ovs, nx, ovs).mean(axis=(1, 3))
        total = total + w[i] * pixels
    return total
