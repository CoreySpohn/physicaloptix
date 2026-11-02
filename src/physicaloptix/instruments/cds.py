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

Options: ``rank_tol`` factors the 1D mask and detector transforms through
shared real bases of the finite Fourier transform (numerical rank about 30
and 55 of 250 and 128 rows, Landau-Pollak), ``contract`` additionally folds
the Babinet correction and the Lyot stop into one fixed real tensor, and
``crop`` skips the exactly-zero margins of the pupil and the Lyot stop.
``mode_fields`` returns the (exactly linear) responses of complex pupil-field
modes, the building block of wavefront time series.

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

from physicaloptix.transforms.cmft import (
    PreparedMFT,
    cmft_bwd,
    cmft_fwd,
    fold_kernel,
    fold_left,
    fold_right,
    matmul,
    resolve_precision,
    symmetric_grid,
)
from physicaloptix.transforms.czt import CZT
from physicaloptix.transforms.lowrank import fourier_basis, kernel_window

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

# Largest Babinet cancellation depth (``DirectAPLC.babinet_depth``) at which complex64
# stayed within 4.3e-6 of the local source peak against complex128 on APLC-A
# (315 positions from 0 to 16 lambda0/D, three wavelengths over the 10 percent band,
# rank_tol 1e-8, full float32 products): a margin of 2.3 against the 1e-5 target.
# Empirical for that design; recheck with ``babinet_depth`` for another one.
COMPLEX64_DEPTH_LIMIT = 20.0


def prefers_complex64(depth, limit=COMPLEX64_DEPTH_LIMIT):
    """Whether a source with Babinet cancellation ``depth`` can run in complex64."""
    return np.asarray(depth) <= limit


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


def _real_times_complex(q, k, split, prec):
    """``q @ k`` for real ``q``; optionally as two real products."""
    if split:
        return jax.lax.complex(matmul(q, k.real, prec), matmul(q, k.imag, prec))
    return matmul(q.astype(k.dtype), k, prec)


def _complex_times_real(f, v, split, prec):
    """``f @ v`` for complex ``f`` and real ``v``; optionally as two real products."""
    if split:
        return jax.lax.complex(matmul(f.real, v, prec), matmul(f.imag, v, prec))
    return matmul(f, v.astype(f.dtype), prec)


def _tilts(x, ax, ay, cdt):
    tx = jnp.exp(2j * jnp.pi * ax[:, None] * x[None, :]).astype(cdt)
    ty = jnp.exp(2j * jnp.pi * ay[:, None] * x[None, :]).astype(cdt)
    return tx, ty


def _support_box(a):
    """Bounding box ``((row0, row1), (col0, col1))`` of the nonzero entries of ``a``."""
    nz = np.asarray(a) != 0
    rows = np.flatnonzero(nz.any(axis=1))
    cols = np.flatnonzero(nz.any(axis=0))
    if rows.size == 0:
        msg = "cannot crop an array that is zero everywhere"
        raise ValueError(msg)
    return (int(rows[0]), int(rows[-1]) + 1), (int(cols[0]), int(cols[-1]) + 1)


def _sl(r):
    return slice(r[0], r[1])


def _lyot_chunk(
    q, ls, disk, kf, cb, tx, ty, split, prec, kb=None, pbox=None, lbox=None
):
    """Lyot field for a chunk of ``S`` sources, layout ``(rows, S, cols)``.

    All sources share each matrix product (the source index is folded into
    the right-hand side), so ``q`` and the kernels are read once per chunk.
    With ``kb`` (the folded back-transform kernel) the two products that
    return from the mask plane use the symmetry of the mask grid. ``pbox``
    and ``lbox`` are the exact-zero support boxes of ``q`` and of the Lyot
    stop; everything outside them is skipped and the result covers ``lbox``.
    """
    s, n = tx.shape
    m = kf.shape[0]
    pr, pc = ((0, n), (0, n)) if pbox is None else pbox
    lr, lc = ((0, n), (0, n)) if lbox is None else lbox
    kp = kf[:, _sl(pc)]
    kx = (tx[:, _sl(pc)].T[:, :, None] * kp.T[:, None, :]).reshape(-1, s * m)
    qp = q[_sl(pr), _sl(pc)]
    nr = qp.shape[0]
    t = _real_times_complex(qp, kx, split, prec).reshape(nr, s, m)
    t = t * ty[:, _sl(pr)].T[:, :, None]
    a = matmul(kf[:, _sl(pr)], t.reshape(nr, s * m), prec).reshape(m, s, m)
    g = disk[:, None, :] * a
    nlr, nlc = lr[1] - lr[0], lc[1] - lc[0]
    if kb is None:
        h = matmul(g.reshape(m * s, m), jnp.conj(kf[:, _sl(lc)]), prec)
        h = h.reshape(m, s * nlc)
        back = matmul(jnp.conj(kf[:, _sl(lr)]).T, h, prec).reshape(nlr, s, nlc)
    else:
        h = fold_right(g.reshape(m * s, m), kb, prec).reshape(m, s, n)
        back = fold_left(kb, h.reshape(m, s * n), prec).reshape(n, s, n)
    qs = q[_sl(lr), _sl(lc)][:, None, :]
    qs = qs * ty[:, _sl(lr)].T[:, :, None] * tx[:, _sl(lc)][None, :, :]
    return ls[_sl(lr), _sl(lc)][:, None, :] * (qs - cb * back)


def _detector_chunk(lyot, kd, kdy, cdet, prec):
    n, s, _ = lyot.shape
    vx = kd.shape[0]
    vy = kdy.shape[0]
    p = matmul(lyot.reshape(n * s, lyot.shape[2]), kd.T, prec).reshape(n, s * vx)
    d = matmul(kdy, p, prec).reshape(vy, s, vx)
    return cdet * jnp.transpose(d, (1, 0, 2))


def _mask_coefficients(q, disk, ktm, vm, tx, ty, pbox, split, prec):
    """Mask-plane return coefficients ``R`` (``(Rm, S, Rm)``) in the basis ``vm``.

    The Babinet correction of the Lyot field is ``vm R vm^T``, with
    ``R = conj(kt)^T (disk * (kt C kt^T)) conj(kt)`` and
    ``C = vm^T (tilted q) vm`` for the mask kernel ``kf = kt vm^T``.
    """
    s = tx.shape[0]
    m, r = ktm.shape
    pr, pc = pbox
    vp = vm[_sl(pc)]
    kx = (tx[:, _sl(pc)].T[:, :, None] * vp[:, None, :]).reshape(-1, s * r)
    nr = pr[1] - pr[0]
    t = _real_times_complex(q[_sl(pr), _sl(pc)], kx, split, prec).reshape(nr, s, r)
    t = (t * ty[:, _sl(pr)].T[:, :, None]).reshape(nr, s * r)
    c = _real_times_complex(vm[_sl(pr)].T, t, split, prec)
    a = matmul(ktm, c.reshape(r, s * r), prec).reshape(m * s, r)
    a = matmul(a, ktm.T, prec).reshape(m, s, m)
    g = disk[:, None, :] * a
    ktc = jnp.conj(ktm)
    w = matmul(ktc.T, g.reshape(m, s * m), prec).reshape(r * s, m)
    return matmul(w, ktc, prec).reshape(r, s, r)


def _lyot_ranked(q, ls, disk, ktm, vm, cb, tx, ty, split, prec, pbox, lbox):
    """Lyot field of ``S`` sources through the rank-factored mask transforms."""
    s = tx.shape[0]
    r = ktm.shape[1]
    rc = _mask_coefficients(q, disk, ktm, vm, tx, ty, pbox, split, prec)
    lr, lc = lbox
    nlr, nlc = lr[1] - lr[0], lc[1] - lc[0]
    h = _real_times_complex(vm[_sl(lr)], rc.reshape(r, s * r), split, prec)
    h = h.reshape(nlr * s, r)
    back = _complex_times_real(h, vm[_sl(lc)].T, split, prec).reshape(nlr, s, nlc)
    qs = q[_sl(lr), _sl(lc)][:, None, :]
    qs = qs * ty[:, _sl(lr)].T[:, :, None] * tx[:, _sl(lc)][None, :, :]
    return ls[_sl(lr), _sl(lc)][:, None, :] * (qs - cb * back)


def _detector_ranked(lyot, ktx, kty, vd, lbox, cdet, split, prec):
    """Detector field from a Lyot field (box layout) through the basis ``vd``."""
    nlr, s, nlc = lyot.shape
    rd = vd.shape[1]
    lr, lc = lbox
    p = _complex_times_real(lyot.reshape(nlr * s, nlc), vd[_sl(lc)], split, prec)
    p = p.reshape(nlr, s * rd)
    c = _real_times_complex(vd[_sl(lr)].T, p, split, prec)
    return _expand_detector(c.reshape(rd, s, rd), ktx, kty, cdet, prec)


def _expand_detector(c, ktx, kty, cdet, prec):
    """Detector field ``(S, vy, vx)`` from coefficients ``c``, shape ``(Rd, S, Rd)``."""
    rd, s, _ = c.shape
    d = matmul(kty, c.reshape(rd, s * rd), prec).reshape(kty.shape[0] * s, rd)
    d = matmul(d, ktx.T, prec).reshape(kty.shape[0], s, ktx.shape[0])
    return cdet * jnp.transpose(d, (1, 0, 2))


def _field_contracted(
    q, lq, disk, ktm, ktx, kty, vm, vd, theta, cb, cdet, tx, ty, split, prec, pbox, lbox
):
    """Detector field with the Babinet correction contracted through the Lyot stop.

    ``vd^T (L * vm R vm^T) vd`` is the fixed real tensor ``theta`` applied to
    ``R``, so the Lyot-plane back-propagation and its detector projection are
    never formed; the direct term projects ``L q`` alone.
    """
    s = tx.shape[0]
    rd = vd.shape[1]
    lr, lc = lbox
    rc = _mask_coefficients(q, disk, ktm, vm, tx, ty, pbox, split, prec)
    kx = (tx[:, _sl(lc)].T[:, :, None] * vd[_sl(lc)][:, None, :]).reshape(-1, s * rd)
    nlr = lr[1] - lr[0]
    t = _real_times_complex(lq[_sl(lr), _sl(lc)], kx, split, prec).reshape(nlr, s, rd)
    t = (t * ty[:, _sl(lr)].T[:, :, None]).reshape(nlr, s * rd)
    c = _real_times_complex(vd[_sl(lr)].T, t, split, prec).reshape(rd, s, rd)
    flat = jnp.transpose(rc, (1, 0, 2)).reshape(s, -1)
    corr = _complex_times_real(flat, theta.T, split, prec).reshape(s, rd, rd)
    c = c - cb * jnp.transpose(corr, (1, 0, 2))
    return _expand_detector(c, ktx, kty, cdet, prec)


def _lyot_tensor(vm, vd, lyot):
    """Real tensor ``theta[(a, b), (i, j)]`` of the Lyot stop in the two bases.

    ``sum_yx vd[y,a] vm[y,i] L[y,x] vd[x,b] vm[x,j]``.
    """
    n, rm = vm.shape
    rd = vd.shape[1]
    ab = (vd[:, :, None] * vm[:, None, :]).reshape(n, rd * rm)
    th = ab.T @ (lyot @ ab)
    return th.reshape(rd, rm, rd, rm).transpose(0, 2, 1, 3).reshape(rd * rd, rm * rm)


def _lyot_chunk_czt(q, ls, disk, fwd, bwd, cb, tx, ty):
    """Lyot field for ``S`` sources by chirp-z transforms, layout ``(S, n, n)``."""
    qs = q[None] * ty[:, :, None] * tx[:, None, :]
    a = fwd.apply(fwd.apply(qs, -1), 1)
    g = bwd.apply(bwd.apply(disk[None] * a, -1), 1)
    return ls[None] * (qs - cb * g)


def _detector_chunk_czt(lyot, dx, dy, cdet):
    return cdet * dy.apply(dx.apply(lyot, -1), 1)


class LowRank(eqx.Module):
    """Shared real bases and small factors of the rank-factored transforms.

    The mask kernel is ``kf = kt vm^T`` and the detector kernels are
    ``kd = ktx vd^T`` and ``kdy = kty vd^T``, with ``vm`` and ``vd`` real,
    orthonormal and shared by all wavelengths and sources.

    Attributes:
        vm, vd: Bases ``(n, Rm)`` and ``(n, Rd)``.
        ktm, ktx, kty: ``kf vm``, ``kd vd``, ``kdy vd`` per wavelength.
        theta: Real tensor ``(Rd^2, Rm^2)`` that contracts the Babinet
            correction through the Lyot stop, or ``None``.
        lq: ``lyot * q``, real, or ``None`` when ``theta`` is ``None``.
        rank_mask, rank_detector: ``Rm`` and ``Rd`` (static).
        tol: The singular-value ratio the ranks were chosen at (static).
    """

    vm: jnp.ndarray
    vd: jnp.ndarray
    ktm: jnp.ndarray
    ktx: jnp.ndarray
    kty: jnp.ndarray
    theta: jnp.ndarray | None
    lq: jnp.ndarray | None
    rank_mask: int = eqx.field(static=True)
    rank_detector: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)

    def slice(self, iw):
        """Per-wavelength factors at index ``iw``."""
        return (self.ktm[iw], self.ktx[iw], self.kty[iw])

    def per_wavelength(self):
        """Per-wavelength factors, stacked over wavelengths (for ``lax.scan``)."""
        return (self.ktm, self.ktx, self.kty)


def _build_lowrank(inputs, w, v, vy, cdt, rdt, tol, contract):
    """Bases from the finite-Fourier windows, factored kernels and optional theta."""
    x = np.asarray(inputs.x, float)
    vx = np.asarray(v, float)
    vyy = vx if vy is None else np.asarray(vy, float)
    vm = fourier_basis(x, kernel_window(inputs.u, w), tol)
    window_d = max(kernel_window(vx, w), kernel_window(vyy, w))
    vd = fourier_basis(x, window_d, tol)

    def factor(basis, coords):
        phase = -2.0 * np.pi * (np.asarray(coords)[None, :, None] / w[:, None, None])
        return np.exp(1j * phase * x[None, None, :]) @ basis

    ktm = factor(vm, inputs.u)
    ktx = factor(vd, vx)
    kty = factor(vd, vyy)
    theta = lq = None
    if contract:
        theta = jnp.asarray(_lyot_tensor(vm, vd, inputs.lyot), dtype=rdt)
        lq = jnp.asarray(inputs.pupil * inputs.apodizer * inputs.lyot, dtype=rdt)
    return LowRank(
        vm=jnp.asarray(vm, dtype=rdt),
        vd=jnp.asarray(vd, dtype=rdt),
        ktm=jnp.asarray(ktm, dtype=cdt),
        ktx=jnp.asarray(ktx, dtype=cdt),
        kty=jnp.asarray(kty, dtype=cdt),
        theta=theta,
        lq=lq,
        rank_mask=int(vm.shape[1]),
        rank_detector=int(vd.shape[1]),
        tol=float(tol),
    )


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
        kback: Real half-kernel for the return from the mask plane
            (``fold_kernel`` of ``conj(kf).T``), ``(nw, n, ne)``, or ``None``.
        lowrank: ``None`` for the dense transforms, else a ``LowRank`` holding
            the shared real bases and the small wavelength-dependent factors.
        pbox, lbox: Exact-zero support boxes ``((row0, row1), (col0, col1))``
            of the pupil and of the Lyot stop (static); the full array when
            cropping is off.
        sigma: ``lambda0 / lambda`` per wavelength (float64).
        precision: Matmul precision of every product (static).
        chunk: Default sources per product in ``intensity`` (static).
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
    kback: jnp.ndarray | None = None
    lowrank: object = None
    pbox: tuple = eqx.field(static=True, default=None)
    lbox: tuple = eqx.field(static=True, default=None)
    precision: object = eqx.field(static=True, default=jax.lax.Precision.HIGHEST)
    chunk: int = eqx.field(static=True, default=1)

    @classmethod
    def prepare(
        cls,
        inputs,
        wavelength_ratios,
        v,
        dtype=jnp.complex128,
        vy=None,
        czt=False,
        precision=None,
        chunk=None,
        fold=False,
        rank_tol=None,
        contract=False,
        crop=False,
    ):
        """Cache kernels for the given wavelength ratios and detector axis.

        Args:
            inputs: ``APLCInputs``.
            wavelength_ratios: lambda / lambda0, scalar or 1D.
            v: 1D detector column (x) coordinates in lambda0/D.
            dtype: ``complex64`` or ``complex128`` for the products.
            vy: 1D detector row (y) coordinates; defaults to ``v``.
            czt: Also build the chirp-z constants (all grids must be uniform).
            precision: Matmul precision for every product (``Precision``,
                ``DotAlgorithmPreset`` or a name). ``None`` resolves the
                default policy now: full precision for complex128, and for
                complex64 on a GPU ``BF16_BF16_F32_X3`` (dense transforms) or
                ``TF32_TF32_F32_X3`` (with ``rank_tol``), full float32 elsewhere.
            chunk: Default sources per product in ``intensity``; ``None`` is 16
                on a GPU and 1 elsewhere.
            fold: Use the symmetry of the pupil and mask grids (both must be
                symmetric about zero) to halve the work of the two products
                that return from the mask plane. Exact up to rounding.
            rank_tol: ``None`` keeps the dense transforms. A number factors the
                1D mask and detector transforms through shared real bases whose
                modes are kept down to this singular-value ratio of the sampled
                finite-Fourier window (about 1e-12 for complex128 work at 1e-11
                of the field maximum; 1e-8 is ample for complex64). The ranks
                are in ``lowrank.rank_mask`` and ``lowrank.rank_detector``.
            contract: With ``rank_tol``, also contract the Babinet correction
                through the Lyot stop (a fixed real tensor), so the Lyot-plane
                back-propagation is never formed in ``field`` and ``intensity``
                (``lyot_field`` still forms it).
            crop: Skip the exactly-zero margins of the pupil and of the Lyot
                stop (exact). Not available with ``fold``.
        """
        cdt = jnp.dtype(dtype)
        rdt = jnp.float32 if cdt == jnp.complex64 else jnp.float64
        prec = resolve_precision(precision, cdt, factored=rank_tol is not None)
        if chunk is None:
            chunk = 16 if jax.default_backend() == "gpu" else 1
        w = np.atleast_1d(np.asarray(wavelength_ratios, dtype=np.float64))
        fpm = PreparedMFT.build(inputs.x, inputs.u, scale=w, dtype=cdt, precision=prec)
        det = PreparedMFT.build(
            inputs.x, np.asarray(v, float), scale=w, dtype=cdt, precision=prec
        )
        dety = (
            det
            if vy is None
            else PreparedMFT.build(
                inputs.x, np.asarray(vy, float), scale=w, dtype=cdt, precision=prec
            )
        )
        full = ((0, inputs.x.size), (0, inputs.x.size))
        pbox, lbox = full, full
        if crop:
            if fold:
                msg = "crop=True does not combine with fold=True"
                raise ValueError(msg)
            pbox = _support_box(inputs.pupil * inputs.apodizer)
            lbox = _support_box(inputs.lyot)
        if contract and rank_tol is None:
            msg = "contract=True needs rank_tol"
            raise ValueError(msg)
        lowrank = None
        if rank_tol is not None:
            lowrank = _build_lowrank(inputs, w, v, vy, cdt, rdt, rank_tol, contract)
        kback = None
        if fold:
            if not (symmetric_grid(inputs.x) and symmetric_grid(inputs.u)):
                msg = "fold=True needs pupil and mask grids symmetric about zero"
                raise ValueError(msg)
            kback = fold_kernel(jnp.conj(jnp.swapaxes(fpm.kernel, -1, -2)))
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
            kback=kback,
            lowrank=lowrank,
            pbox=pbox,
            lbox=lbox,
            precision=prec,
            chunk=int(chunk),
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
            None if self.kback is None else self.kback[iw],
            None if self.lowrank is None else self.lowrank.slice(iw),
        )

    def _slices(self):
        return (
            self.kf,
            self.kd,
            self.kdy,
            self.cb,
            self.cdet,
            self.sigma,
            self.czt,
            self.kback,
            None if self.lowrank is None else self.lowrank.per_wavelength(),
        )

    def _lyot(self, wl, sx, sy, method, split):
        """Lyot fields for sources at one wavelength slice, box layout ``(r, S, c)``.

        The ``czt`` method returns the full ``(S, n, n)`` layout instead.
        """
        kf, _, _, cb, _, sigma, cz, kb, lr = wl
        ax = jnp.atleast_1d(jnp.asarray(sx, jnp.float64)) * sigma
        ay = jnp.atleast_1d(jnp.asarray(sy, jnp.float64)) * sigma
        tx, ty = _tilts(self.x, ax, ay, self.cdtype)
        if method == "czt":
            return _lyot_chunk_czt(
                self.q, self.lyot, self.disk, cz[0], cz[1], cb, tx, ty
            )
        if self.lowrank is not None:
            return _lyot_ranked(
                self.q,
                self.lyot,
                self.disk,
                lr[0],
                self.lowrank.vm,
                cb,
                tx,
                ty,
                split,
                self.precision,
                self.pbox,
                self.lbox,
            )
        return _lyot_chunk(
            self.q,
            self.lyot,
            self.disk,
            kf,
            cb,
            tx,
            ty,
            split,
            self.precision,
            kb,
            self.pbox,
            self.lbox,
        )

    def _field(self, wl, sx, sy, method, split):
        """Detector fields ``(S, vy, vx)`` of sources at one wavelength slice."""
        _, kd, kdy, cb, cdet, sigma, cz, _, lr = wl
        cdet = cdet.astype(self.cdtype)
        lbox = self.lbox
        if method == "czt":
            lyot = self._lyot(wl, sx, sy, method, split)
            return _detector_chunk_czt(lyot, cz[2], cz[3], cdet)
        if self.lowrank is not None and self.lowrank.theta is not None:
            ax = jnp.atleast_1d(jnp.asarray(sx, jnp.float64)) * sigma
            ay = jnp.atleast_1d(jnp.asarray(sy, jnp.float64)) * sigma
            tx, ty = _tilts(self.x, ax, ay, self.cdtype)
            lk = self.lowrank
            return _field_contracted(
                self.q,
                lk.lq,
                self.disk,
                lr[0],
                lr[1],
                lr[2],
                lk.vm,
                lk.vd,
                lk.theta,
                cb,
                cdet,
                tx,
                ty,
                split,
                self.precision,
                self.pbox,
                lbox,
            )
        lyot = self._lyot(wl, sx, sy, method, split)
        if self.lowrank is not None:
            return _detector_ranked(
                lyot, lr[1], lr[2], self.lowrank.vd, lbox, cdet, split, self.precision
            )
        return _detector_chunk(
            lyot, kd[:, _sl(lbox[1])], kdy[:, _sl(lbox[0])], cdet, self.precision
        )

    def lyot_field(self, sx, sy, iw=0, split=False, method="gemm"):
        """Lyot-plane field ``(n, n)`` of one source at wavelength index ``iw``."""
        out = self._lyot(self._slice(iw), sx, sy, method, split)
        if method == "czt":
            return out[0]
        full = jnp.zeros(self.q.shape, out.dtype)
        return full.at[_sl(self.lbox[0]), _sl(self.lbox[1])].set(out[:, 0, :])

    def field(self, sx, sy, iw=0, split=False, method="gemm"):
        """Detector fields ``(S, vy, vx)`` for sources ``sx``, ``sy`` (length S)."""
        return self._field(self._slice(iw), sx, sy, method, split)

    def mode_fields(self, sx, sy, modes, iw=0, method="gemm"):
        """Detector fields of complex pupil-field modes ``(S, J, vy, vx)``.

        Mode ``j`` is the pupil field ``q * modes[j]`` (before the source tilt)
        and the fields are linear in it, so for any pupil field
        ``q * (m0 + sum_j c_j modes[j])`` the detector field is
        ``E0 + sum_j c_j E_j`` with ``E0`` the response of ``m0`` (``m0 = 1`` is
        ``field``). Exact up to rounding: use it once per source and wavelength
        to turn a perturbation of the pupil (``i * phase_mode`` for a small
        wavefront change, an amplitude change, an actuator influence function)
        into a stack of responses, then combine them with the time-dependent
        coefficients. Modes are processed one at a time; real modes use the
        faster real-pupil products (multiply the response by ``1j`` for a phase
        mode). ``split`` is not offered: complex modes need the unsplit products.

        Args:
            sx: Source x positions in lambda0/D (length S).
            sy: Source y positions (length S).
            modes: Complex (or real) multipliers of the pupil, ``(J, n, n)``.
            iw: Wavelength index.
            method: ``"gemm"`` or ``"czt"``, as in ``field``.
        """
        modes = jnp.asarray(modes)
        complex_modes = jnp.iscomplexobj(modes)
        if self.lowrank is not None and self.lowrank.theta is not None:
            lq = self.lowrank.lq
        else:
            lq = None

        def one(m):
            m = m.astype(self.cdtype if complex_modes else self.q.dtype)
            q = self.q * m
            if lq is None:
                mod = eqx.tree_at(lambda s: s.q, self, q)
            else:
                mod = eqx.tree_at(
                    lambda s: (s.q, s.lowrank.lq), self, (q, self.lowrank.lq * m)
                )
            return mod._field(mod._slice(iw), sx, sy, method, False)

        return jnp.swapaxes(jax.lax.map(one, modes), 0, 1)

    def babinet_depth(self, sx, sy, iw=0):
        """Cancellation depth of the Babinet subtraction for each source.

        The detector field is the difference of the direct term (the
        Lyot-stopped apodized pupil alone) and the Babinet correction. The ratio
        of the largest direct-term amplitude on the detector grid to the largest
        coronagraphic amplitude (``1`` far from the mask, large inside it) is
        how many digits the subtraction loses, so the rounding error of a
        complex64 run grows with it (see ``COMPLEX64_DEPTH_LIMIT``). The grid
        should contain the source (the maximum is then the source's own peak).
        Costs about two ``field`` calls.

        Args:
            sx: Source x positions in lambda0/D (length S).
            sy: Source y positions (length S).
            iw: Wavelength index.

        Returns:
            Array ``(S,)``.
        """
        bare = eqx.tree_at(lambda s: s.cb, self, jnp.zeros_like(self.cb))
        direct = jnp.abs(bare.field(sx, sy, iw))
        total = jnp.abs(self.field(sx, sy, iw))
        return direct.max(axis=(1, 2)) / total.max(axis=(1, 2))

    def intensity(
        self, sx, sy, src_weights, wl_weights, chunk=None, split=False, method="gemm"
    ):
        """Weighted incoherent sum of detector intensities.

        ``sum_w wl_weights[w] sum_s src_weights[s] |E(s, w)|^2`` with ``chunk``
        sources per product (default ``self.chunk``; the source count is padded
        with zero weights); the intensities are summed in float64 even when the
        products run in complex64.
        Wavelengths run in a ``lax.scan`` so only one wavelength's kernels are
        live at a time. ``method`` is ``"gemm"`` (dense matrix products, sources
        folded into the right-hand side) or ``"czt"`` (chirp-z, needs
        ``prepare(..., czt=True)``).
        """
        chunk = self.chunk if chunk is None else chunk
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
                return acc_w + jnp.tensordot(
                    cw, inten, axes=1, precision=jax.lax.Precision.HIGHEST
                ), None

            img, _ = jax.lax.scan(per_chunk, jnp.zeros(shape), (sx, sy, sw))
            return acc + ww * img, None

        wl_w = jnp.asarray(wl_weights, jnp.float64)
        out, _ = jax.lax.scan(per_wavelength, jnp.zeros(shape), (self._slices(), wl_w))
        return out
