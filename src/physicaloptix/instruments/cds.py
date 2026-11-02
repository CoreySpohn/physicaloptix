"""Direct propagation of the public CDS APLC-A design (compact-mask form).

The apodized-pupil Lyot coronagraph is evaluated with the Babinet (compact
mask) decomposition of Soummer et al. (2007): the Lyot field is
``LS * (Q - B[(1 - T) F[Q]])`` where ``Q`` is the apodized, tilted pupil,
``T`` the focal-plane mask transmission, ``F`` / ``B`` a matrix Fourier
transform pair, and ``LS`` the Lyot stop. Only the 250 x 250 window that holds
the grayscale mask is ever sampled, so one image costs about 6 GFLOP.

The detector step is ``D / (1j lambda/lambda0)`` times the continuous-FT
pair with focal coordinates divided by ``lambda/lambda0``.

Normalization and conventions follow the CDS design (pupil divided by
``sqrt(sum |P|^2 dA)`` with ``dA`` in square meters, tilt
``exp(2j pi (lambda0 / lambda) (sx x + sy y) / D)`` with columns along x and
rows along y, detector field in sqrt(photons) per lambda0/D), so an
intensity map is in photons per (lambda0/D)^2 per entrance photon.

Precision: the large products run in the dtype of the prepared kernels
(complex64 or complex128). Kernels and tilt phases are always built in
float64 first, so a complex64 run only rounds the products. Float64 pupil
coordinates require ``jax_enable_x64``.
"""

import importlib.util
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits

from physicaloptix.transforms.cmft import PreparedMFT, cmft_bwd, cmft_fwd
from physicaloptix.transforms.czt import CZT

_DESIGN = "designs/aplc/LUVOIR-A_delivery_May2019"
_PUPIL = "inputs/TelAp_LUVOIR_gap_pad01_bw_ovsamp04_N1000.fits.gz"
_LYOT = "inputs/LS_LUVOIR_ID0120_OD0982_no_struts_gy_ovsamp4_N1000.fits.gz"
_APODIZER = (
    "luvoir_stdt_baseline_bw10/medium_fpm/solutions/"
    "0_LUVOIR_N1000_FPM682M0250_IWA0672_OWA02372_C10_BW10_Nlam5_LS_IDD0120_OD0982_"
    "no_ls_struts.fits.gz"
)

APLC_A_FPM_RADIUS = 6.82  # lambda0/D
APLC_A_FPM_PIXELS = 250
APLC_A_DIAMETER_M = 15.0
APLC_A_PIXEL_M = 0.015


def gray_disk(radius, n, oversample=8):
    """Grayscale disk on an ``n x n`` window of width ``2 * radius``.

    Cell centers sit at ``(j + 0.5 - n / 2) * 2 radius / n``; each cell is the
    mean of an ``oversample x oversample`` grid of sub-cell samples with an
    inclusive boundary, which is how the CDS design builds its mask.

    Returns:
        The disk, shape ``(n, n)`` (rows y, columns x), and the 1D centers.
    """
    delta = 2.0 * radius / n
    c = (np.arange(n) + 0.5 - n / 2.0) * delta
    sub = ((np.arange(oversample) + 0.5) / oversample - 0.5) * delta
    acc = np.zeros((n, n))
    for dy in sub:
        for dx in sub:
            acc += (c[None, :] + dx) ** 2 + (c[:, None] + dy) ** 2 <= radius**2
    return acc / oversample**2, c


class APLCInputs(eqx.Module):
    """Host-side (float64) inputs of an APLC with a compact focal-plane mask.

    Attributes:
        pupil: Normalized telescope pupil (CDS normalization), ``(n, n)``.
        apodizer: Pupil-plane apodizer, ``(n, n)``.
        lyot: Lyot stop, ``(n, n)``.
        fpm_open: One minus the mask transmission (the part the Lyot stage
            removes), ``(m, m)``; 1 inside the opaque disk.
        x: Pupil coordinates in pupil diameters, ``(n,)``.
        u: Mask-plane coordinates in lambda0/D, ``(m,)``.
        diameter_m: Pupil diameter in meters.
    """

    pupil: np.ndarray
    apodizer: np.ndarray
    lyot: np.ndarray
    fpm_open: np.ndarray
    x: np.ndarray
    u: np.ndarray
    diameter_m: float = eqx.field(static=True)


def _design_root(root):
    if root is not None:
        return Path(root)
    spec = importlib.util.find_spec("cds_pipeline")
    if spec is None or spec.origin is None:
        msg = "cds_pipeline is not installed: pass root= the APLC delivery folder"
        raise FileNotFoundError(msg)
    return Path(spec.origin).parent / _DESIGN


def load_aplc_a(root=None):
    """Load the public CDS LUVOIR-A APLC (medium mask) as ``APLCInputs``.

    Args:
        root: Folder holding ``inputs/`` and ``luvoir_stdt_baseline_bw10/``;
            defaults to the copy shipped inside the installed ``cds_pipeline``.
    """
    root = _design_root(root)
    pupil = fits.getdata(root / _PUPIL).astype(np.float64)
    apod = fits.getdata(root / _APODIZER).astype(np.float64)
    lyot = fits.getdata(root / _LYOT).astype(np.float64)
    n = pupil.shape[0]
    photons = np.sum(np.abs(pupil) ** 2 * APLC_A_PIXEL_M**2)
    x = (np.arange(n) - (n - 1) / 2.0) * APLC_A_PIXEL_M / APLC_A_DIAMETER_M
    disk, u = gray_disk(APLC_A_FPM_RADIUS, APLC_A_FPM_PIXELS)
    return APLCInputs(
        pupil=pupil / np.sqrt(photons),
        apodizer=apod,
        lyot=lyot,
        fpm_open=disk,
        x=x,
        u=u,
        diameter_m=APLC_A_DIAMETER_M,
    )


def detector_axis(n=128, pitch=0.25):
    """Detector coordinates in lambda0/D, centered between the middle pixels."""
    return (np.arange(n) - (n - 1) / 2.0) * pitch


def lyot_field_reference(inputs, sx, sy, wavelength_ratio=1.0):
    """Lyot-plane field from the library's own ``cmft`` pair (rebuilds kernels).

    Complex128 reference path used for parity and as the unoptimized timing
    baseline. ``wavelength_ratio`` is lambda / lambda0.
    """
    x = jnp.asarray(inputs.x)
    u = jnp.asarray(inputs.u) / wavelength_ratio
    alpha = 1.0 / wavelength_ratio
    tilt = jnp.exp(2j * jnp.pi * alpha * (sy * x[:, None] + sx * x[None, :]))
    q = jnp.asarray(inputs.pupil * inputs.apodizer) * tilt
    focal = cmft_fwd(q, x, u)
    back = cmft_bwd(jnp.asarray(inputs.fpm_open) * focal, x, u)
    return jnp.asarray(inputs.lyot) * (q - back)


def detector_field_reference(inputs, lyot, v, wavelength_ratio=1.0):
    """Detector field from a Lyot field with the library's ``cmft_fwd``."""
    x = jnp.asarray(inputs.x)
    c = inputs.diameter_m / (1j * wavelength_ratio)
    return c * cmft_fwd(lyot, x, jnp.asarray(v) / wavelength_ratio)


def _real_times_complex(q, k, split):
    """``q @ k`` for real ``q``; optionally as two real products."""
    if split:
        return jax.lax.complex(q @ k.real, q @ k.imag)
    return q.astype(k.dtype) @ k


def _tilts(x, ax, ay, cdt):
    tx = jnp.exp(2j * jnp.pi * ax[:, None] * x[None, :]).astype(cdt)
    ty = jnp.exp(2j * jnp.pi * ay[:, None] * x[None, :]).astype(cdt)
    return tx, ty


def _lyot_chunk(q, ls, disk, kf, cb, tx, ty, split):
    """Lyot field for a chunk of ``S`` sources, layout ``(n, S, n)``.

    All sources share each matrix product (the source index is folded into
    the right-hand side), so ``q`` and the kernels are read once per chunk.
    """
    s, n = tx.shape
    m = kf.shape[0]
    kx = (tx.T[:, :, None] * kf.T[:, None, :]).reshape(n, s * m)
    t = _real_times_complex(q, kx, split).reshape(n, s, m) * ty.T[:, :, None]
    a = (kf @ t.reshape(n, s * m)).reshape(m, s, m)
    g = disk[:, None, :] * a
    kc = jnp.conj(kf)
    h = (g.reshape(m * s, m) @ kc).reshape(m, s, n)
    back = (kc.T @ h.reshape(m, s * n)).reshape(n, s, n)
    qs = q[:, None, :] * ty.T[:, :, None] * tx[None, :, :]
    return ls[:, None, :] * (qs - cb * back)


def _detector_chunk(lyot, kd, kdy, cdet):
    n, s, _ = lyot.shape
    vx = kd.shape[0]
    vy = kdy.shape[0]
    p = (lyot.reshape(n * s, n) @ kd.T).reshape(n, s * vx)
    d = (kdy @ p).reshape(vy, s, vx)
    return cdet * jnp.transpose(d, (1, 0, 2))


def _lyot_chunk_czt(q, ls, disk, fwd, bwd, cb, tx, ty):
    """Lyot field for ``S`` sources by chirp-z transforms, layout ``(S, n, n)``."""
    qs = q[None] * ty[:, :, None] * tx[:, None, :]
    a = fwd.apply(fwd.apply(qs, -1), 1)
    g = bwd.apply(bwd.apply(disk[None] * a, -1), 1)
    return ls[None] * (qs - cb * g)


def _detector_chunk_czt(lyot, dx, dy, cdet):
    return cdet * dy.apply(dx.apply(lyot, -1), 1)


class DirectAPLC(eqx.Module):
    """Prepared direct propagator: masks and kernels for fixed wavelengths.

    Build with ``DirectAPLC.prepare``. Every method is a pure function of the
    stored arrays and of array-valued source positions, so it can be wrapped
    in a single ``jax.jit`` and called with new positions without recompiling.

    Attributes:
        q: Apodized normalized pupil, real, ``(n, n)``.
        lyot: Lyot stop, real, ``(n, n)``.
        disk: Opaque part of the mask (one minus transmission), ``(m, m)``.
        x: Pupil coordinates (float64), ``(n,)``.
        kf: Mask-plane kernels per wavelength, ``(nw, m, n)``.
        kd: Detector column (x) kernels per wavelength, ``(nw, vx, n)``.
        kdy: Detector row (y) kernels per wavelength, ``(nw, vy, n)``.
        cb: Babinet weight ``dx^2 du^2 / ratio^2`` per wavelength, real.
        cdet: Detector constant per wavelength, complex.
        sigma: ``lambda0 / lambda`` per wavelength (float64).
    """

    q: jnp.ndarray
    lyot: jnp.ndarray
    disk: jnp.ndarray
    x: jnp.ndarray
    kf: jnp.ndarray
    kd: jnp.ndarray
    kdy: jnp.ndarray
    cb: jnp.ndarray
    cdet: jnp.ndarray
    sigma: jnp.ndarray
    czt: tuple | None = None

    @classmethod
    def prepare(
        cls, inputs, wavelength_ratios, v, dtype=jnp.complex128, vy=None, czt=False
    ):
        """Cache kernels for the given wavelength ratios and detector axis.

        Args:
            inputs: ``APLCInputs``.
            wavelength_ratios: lambda / lambda0, scalar or 1D.
            v: 1D detector column (x) coordinates in lambda0/D.
            dtype: ``complex64`` or ``complex128`` for the products.
            vy: 1D detector row (y) coordinates; defaults to ``v``.
            czt: Also build the chirp-z constants (all grids must be uniform).
        """
        cdt = jnp.dtype(dtype)
        rdt = jnp.float32 if cdt == jnp.complex64 else jnp.float64
        w = np.atleast_1d(np.asarray(wavelength_ratios, dtype=np.float64))
        fpm = PreparedMFT.build(inputs.x, inputs.u, scale=w, dtype=cdt)
        det = PreparedMFT.build(inputs.x, np.asarray(v, float), scale=w, dtype=cdt)
        dety = (
            det
            if vy is None
            else PreparedMFT.build(inputs.x, np.asarray(vy, float), scale=w, dtype=cdt)
        )
        cdet = np.asarray(det.dx2) * inputs.diameter_m / (1j * w)
        cz = None
        if czt:
            x, u = np.asarray(inputs.x, float), np.asarray(inputs.u, float)
            vx = np.asarray(v, float)
            vyy = vx if vy is None else np.asarray(vy, float)
            dx, du = x[1] - x[0], (u[1] - u[0])
            build = lambda *a: CZT.build(*a, dtype=cdt)  # noqa: E731
            cz = (
                build(x[0], dx, x.size, u[0] / w, du / w, u.size, -1.0),
                build(u[0] / w, du / w, u.size, x[0], dx, x.size, 1.0),
                build(x[0], dx, x.size, vx[0] / w, (vx[1] - vx[0]) / w, vx.size, -1.0),
                build(
                    x[0], dx, x.size, vyy[0] / w, (vyy[1] - vyy[0]) / w, vyy.size, -1.0
                ),
            )
        return cls(
            q=jnp.asarray(inputs.pupil * inputs.apodizer, dtype=rdt),
            lyot=jnp.asarray(inputs.lyot, dtype=rdt),
            disk=jnp.asarray(inputs.fpm_open, dtype=rdt),
            x=jnp.asarray(inputs.x, dtype=jnp.float64),
            kf=fpm.kernel,
            kd=det.kernel,
            kdy=dety.kernel,
            cb=jnp.asarray(fpm.dx2 * fpm.du2, dtype=rdt),
            cdet=jnp.asarray(cdet, dtype=jnp.complex128),
            sigma=jnp.asarray(1.0 / w),
            czt=cz,
        )

    @property
    def cdtype(self):
        """Complex dtype of the cached kernels."""
        return self.kf.dtype

    def _slice(self, iw):
        cz = None if self.czt is None else jax.tree.map(lambda a: a[iw], self.czt)
        return (
            self.kf[iw],
            self.kd[iw],
            self.kdy[iw],
            self.cb[iw],
            self.cdet[iw],
            self.sigma[iw],
            cz,
        )

    def _slices(self):
        return (self.kf, self.kd, self.kdy, self.cb, self.cdet, self.sigma, self.czt)

    def _lyot(self, wl, sx, sy, method, split):
        """Lyot fields for sources at one wavelength slice; (n, S, n) or (S, n, n)."""
        kf, _, _, cb, _, sigma, cz = wl
        ax = jnp.atleast_1d(jnp.asarray(sx, jnp.float64)) * sigma
        ay = jnp.atleast_1d(jnp.asarray(sy, jnp.float64)) * sigma
        tx, ty = _tilts(self.x, ax, ay, self.cdtype)
        if method == "czt":
            return _lyot_chunk_czt(
                self.q, self.lyot, self.disk, cz[0], cz[1], cb, tx, ty
            )
        return _lyot_chunk(self.q, self.lyot, self.disk, kf, cb, tx, ty, split)

    def _field(self, wl, sx, sy, method, split):
        """Detector fields ``(S, vy, vx)`` of sources at one wavelength slice."""
        _, kd, kdy, _, cdet, _, cz = wl
        lyot = self._lyot(wl, sx, sy, method, split)
        cdet = cdet.astype(self.cdtype)
        if method == "czt":
            return _detector_chunk_czt(lyot, cz[2], cz[3], cdet)
        return _detector_chunk(lyot, kd, kdy, cdet)

    def lyot_field(self, sx, sy, iw=0, split=False, method="gemm"):
        """Lyot-plane field ``(n, n)`` of one source at wavelength index ``iw``."""
        out = self._lyot(self._slice(iw), sx, sy, method, split)
        return out[0] if method == "czt" else out[:, 0, :]

    def field(self, sx, sy, iw=0, split=False, method="gemm"):
        """Detector fields ``(S, vy, vx)`` for sources ``sx``, ``sy`` (length S)."""
        return self._field(self._slice(iw), sx, sy, method, split)

    def intensity(
        self, sx, sy, src_weights, wl_weights, chunk=4, split=False, method="gemm"
    ):
        """Weighted incoherent sum of detector intensities.

        ``sum_w wl_weights[w] sum_s src_weights[s] |E(s, w)|^2`` with ``chunk``
        sources per product (the source count is padded with zero weights); the
        intensities are summed in float64 even when the products run in complex64.
        Wavelengths run in a ``lax.scan`` so only one wavelength's kernels are
        live at a time. ``method`` is ``"gemm"`` (dense matrix products, sources
        folded into the right-hand side) or ``"czt"`` (chirp-z, needs
        ``prepare(..., czt=True)``).
        """
        sx = jnp.asarray(sx, jnp.float64)
        sy = jnp.asarray(sy, jnp.float64)
        sw = jnp.asarray(src_weights, jnp.float64)
        pad = (-sx.shape[0]) % chunk
        sx = jnp.pad(sx, (0, pad)).reshape(-1, chunk)
        sy = jnp.pad(sy, (0, pad)).reshape(-1, chunk)
        sw = jnp.pad(sw, (0, pad)).reshape(-1, chunk)
        shape = (self.kdy.shape[1], self.kd.shape[1])

        def per_wavelength(acc, xs):
            wl, ww = xs

            def per_chunk(acc_w, ys):
                cx, cy, cw = ys
                e = self._field(wl, cx, cy, method, split)
                inten = (e.real**2 + e.imag**2).astype(jnp.float64)
                return acc_w + jnp.tensordot(cw, inten, axes=1), None

            img, _ = jax.lax.scan(per_chunk, jnp.zeros(shape), (sx, sy, sw))
            return acc + ww * img, None

        wl_w = jnp.asarray(wl_weights, jnp.float64)
        out, _ = jax.lax.scan(per_wavelength, jnp.zeros(shape), (self._slices(), wl_w))
        return out
