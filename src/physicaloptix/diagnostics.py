"""Diagnostics: sampling gates on static grids, and model-level audits.

The sampling metrics run in plain Python at build time -- a chain that would
alias fails before the first propagation, at zero traced cost. Formulas
adapted from abcdLux's advisory metrics (Desdoigts), promoted here to
construction-time gates.

:func:`quadrature_audit` is a different kind of check: it asks whether a
measured *statistic* is a property of the optics or of the mode basis chosen
to parameterize the drift.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


def mft_sampling_parameter(x_in, x_out):
    """Nyquist ratio of the MFT kernel ``exp(-2j pi outer(x_out, x_in))``.

    The kernel's fastest input-direction phase step is
    ``2 pi * max|x_out| * dx_in`` (and symmetrically for the output
    direction); the Nyquist criterion is a step of at most pi. Returns
    ``p = min(p_in, p_out)``:

    - ``p >= 1``: the kernel is Nyquist-sampled in both directions.
    - ``p < 1``: aliasing is expected somewhere in the kernel.

    Args:
        x_in: 1D input coordinates.
        x_out: 1D output coordinates.

    Returns:
        The scalar Nyquist ratio.
    """
    x_in = np.asarray(x_in, float)
    x_out = np.asarray(x_out, float)
    dx_in = abs(x_in[1] - x_in[0])
    dx_out = abs(x_out[1] - x_out[0])
    p_in = 1.0 / (2.0 * np.max(np.abs(x_out)) * dx_in)
    p_out = 1.0 / (2.0 * np.max(np.abs(x_in)) * dx_out)
    return float(min(p_in, p_out))


def fresnel_sampling_parameter(grid, alpha):
    """Nyquist ratio of the paraxial angular-spectrum chirp ``exp(-i pi alpha nu^2)``.

    The transfer function's fastest fringe sits at the grid's Nyquist frequency;
    sampling it without aliasing on the ``npix``-point frequency grid gives

        p = (npix * dx) / sqrt(npix * alpha)     [ == D / sqrt(N lambda z) ]

    the dimensionless form of abcdLux's ``asm_sampling_parameter`` (Desdoigts;
    the same advisory-metric lineage as :func:`mft_sampling_parameter`). ``p``
    scales as the square root of the grid-point count, so real-domain padding
    improves it (see :func:`fresnel_pad_factor`).

    - ``p >= 1``: the transfer function is Nyquist-sampled.
    - ``p < 1``: aliasing is expected (a worst-case OPERATOR bound, so it is
      conservative for band-limited fields).

    Args:
        grid: The (same in/out) propagation grid.
        alpha: The dimensionless Fresnel parameter ``lambda * z / D^2``.

    Returns:
        The scalar Nyquist ratio (``inf`` when ``alpha <= 0``).
    """
    if alpha <= 0.0:
        return float("inf")
    extent = grid.npix * grid.dx
    return float(extent / np.sqrt(grid.npix * alpha))


def fresnel_pad_factor(grid, alpha):
    """Real-domain zero-pad factor that brings the Fresnel gate to ``p >= 1``.

    ``p`` grows as ``sqrt`` of the point count, so padding by ``npad`` scales it
    by ``sqrt(npad)``; the smallest integer factor reaching Nyquist is
    ``ceil(1 / p^2)``.

    Args:
        grid: The propagation grid.
        alpha: The dimensionless Fresnel parameter.

    Returns:
        The recommended integer pad factor (``1`` when already well sampled).
    """
    p = fresnel_sampling_parameter(grid, alpha)
    if not np.isfinite(p):
        return 1
    return max(1, int(np.ceil(1.0 / p**2)))


def quadrature_partner(modes: Array) -> Array:
    """The quarter-period partner of each real pupil mode, ``(m, y, x)``.

    A half-plane Hilbert operator: for a mode
    ``cos(2 pi k.x + psi)`` it returns ``sin(2 pi k.x + psi)``, and it acts
    linearly, so a superposition maps to the superposition of the partners.
    Applied twice it negates, which is the defining property of a quadrature
    operator and what makes "the missing quadrature" well posed for a basis
    that is not built from clean ripples.

    The construction is spectral: the mode's transform is multiplied by
    ``-i`` on one half-plane and ``+i`` on the other (DC untouched), which is
    a quarter-cycle rotation of every spatial frequency in the same sense.

    An aperture-truncated mode is not a clean ripple -- its spectrum is the
    ripple convolved with the aperture transform -- so the partner of a
    truncated cosine carries a small component no ripple can represent. That
    leakage is a few percent in norm and does not change the verdict
    :func:`quadrature_audit` returns, but it is why the audit compares two
    statistics rather than thresholding a reconstruction residual.

    Args:
        modes: Real mode stack, shape ``(m, y, x)``.

    Returns:
        The partner stack, same shape and dtype-family as ``modes``.
    """
    modes = jnp.asarray(modes)
    if modes.ndim != 3:
        raise ValueError(f"modes must be a 3D stack (m, y, x), got {modes.shape}")
    ny, nx = modes.shape[-2:]
    kx = jnp.fft.fftfreq(nx)[None, :] * nx
    ky = jnp.fft.fftfreq(ny)[:, None] * ny
    upper = (kx > 0) | ((kx == 0) & (ky > 0))
    dc = (kx == 0) & (ky == 0)
    rot = jnp.where(upper, -1j, 0.0) + jnp.where(~upper & ~dc, 1j, 0.0)
    return jnp.real(
        jnp.fft.ifft2(jnp.fft.fft2(modes, axes=(-2, -1)) * rot, axes=(-2, -1))
    )


def _eta(g: Array, rms: Array, mask=None) -> float:
    """Median degree of impropriety ``|P| / Gamma`` of a Jacobian stack.

    Matches :meth:`physicaloptix.SpeckleMoments.impropriety`, which is the
    same formula reached through a built process; kept standalone here so an
    audit needs only ``(G, rms)`` and not a nominal field or a photometric
    reference (``eta`` depends on neither).
    """
    rms2 = jnp.asarray(rms, dtype=float) ** 2
    gamma = jnp.einsum("k,kyx->yx", rms2, jnp.abs(g) ** 2)
    p = jnp.einsum("k,kyx->yx", rms2, g**2)
    live = gamma > 0
    eta = jnp.where(live, jnp.abs(p) / jnp.where(live, gamma, 1.0), 0.0)
    sel = live if mask is None else (live & jnp.asarray(mask, dtype=bool))
    return float(jnp.median(eta[sel]))


@dataclass(frozen=True)
class QuadratureAudit:
    """Whether a basis's impropriety survives completing its quadratures.

    ``ratio = eta_completed / eta_as_built`` is the number to read. A basis
    carrying one quadrature per spatial frequency cannot cancel the
    pseudo-covariance, so completing it collapses ``eta`` and the ratio falls
    far below 1. A basis that already spans both quadratures has nothing to
    gain, and the ratio sits near 1.
    """

    eta_as_built: float
    eta_completed: float
    ratio: float
    n_modes: int
    n_modes_completed: int
    verdict: str  # "locked" or "complete"

    def __str__(self) -> str:
        """A few lines fit to print next to a reported impropriety."""
        return "\n".join(
            [
                f"modes               {self.n_modes} -> {self.n_modes_completed}",
                f"eta as built        {self.eta_as_built:.4f}",
                f"eta completed       {self.eta_completed:.4f}",
                f"ratio               {self.ratio:.3f}",
                f"verdict             {self.verdict}",
                (
                    "  the reported impropriety is a property of the BASIS, "
                    "not the optics"
                    if self.verdict == "locked"
                    else "  the basis spans both quadratures; eta is physical"
                ),
            ]
        )


def quadrature_audit_from_jacobians(
    g_as_built: Array,
    g_partners: Array,
    *,
    per_mode_rms=1.0,
    mask=None,
    locked_below: float = 0.5,
) -> QuadratureAudit:
    """The quadrature audit from Jacobians a caller already has.

    :func:`quadrature_audit` materializes the mode stack and its partners,
    which is fine at notebook scale and impossible at production scale: a
    220-mode basis on a 2048-pixel pupil is about 7 GB per stack. Builders
    that propagate mode by mode (the usual pattern for exactly that reason)
    should propagate the partners the same way and hand both Jacobians here.

    Args:
        g_as_built: Complex Jacobian ``(m, y, x)`` of the basis under audit.
        g_partners: Complex Jacobian ``(m, y, x)`` of the quadrature partners,
            in the same mode order -- propagate
            :func:`quadrature_partner` output through the same chain.
        per_mode_rms: Scalar or ``(m,)`` drift rms; partners inherit theirs.
        mask: Optional boolean focal-plane mask to reduce over.
        locked_below: Ratio under which the verdict is ``"locked"``.

    Returns:
        A :class:`QuadratureAudit` report.
    """
    g_as_built = jnp.asarray(g_as_built)
    g_partners = jnp.asarray(g_partners)
    if g_as_built.shape != g_partners.shape:
        raise ValueError(
            f"g_partners has shape {g_partners.shape}; expected "
            f"{g_as_built.shape} to match the basis under audit"
        )
    n = g_as_built.shape[0]
    rms_in = jnp.asarray(per_mode_rms, dtype=float)
    if rms_in.ndim > 0 and rms_in.shape != (n,):
        raise ValueError(
            f"per_mode_rms has shape {rms_in.shape}; expected ({n},) "
            "to match the mode stack"
        )
    rms = jnp.broadcast_to(rms_in, (n,))

    eta_built = _eta(g_as_built, rms, mask)
    eta_completed = _eta(
        jnp.concatenate([g_as_built, g_partners]), jnp.concatenate([rms, rms]), mask
    )
    ratio = float("inf") if eta_built == 0.0 else eta_completed / eta_built
    return QuadratureAudit(
        eta_as_built=eta_built,
        eta_completed=eta_completed,
        ratio=ratio,
        n_modes=n,
        n_modes_completed=2 * n,
        verdict="locked" if ratio < locked_below else "complete",
    )


def quadrature_audit(
    modes: Array,
    propagate,
    *,
    per_mode_rms=1.0,
    mask=None,
    locked_below: float = 0.5,
) -> QuadratureAudit:
    """Audit whether a basis's impropriety is physical or a parameterization.

    The degree of impropriety ``eta = |P| / Gamma`` is a joint statement about
    the optics and the mode basis. A drift basis with one ripple per spatial
    frequency pins each frequency to a single direction in the complex plane,
    which forces ``eta`` upward however the hardware actually behaves -- and a
    large ``eta`` read off such a basis says nothing about the instrument.

    This runs the counterfactual. It builds each mode's missing quadrature
    with :func:`quadrature_partner`, propagates the doubled basis through the
    same chain, and compares the two impropriety statistics. Giving each
    partner its mode's rms makes the counterfactual precisely the homogeneous
    (translation-invariant) drift hypothesis: equal variance in both
    quadratures of every spatial frequency.

    The cost is one extra propagation of ``m`` modes, so this is a run-once
    audit rather than a per-iteration check.

    Args:
        modes: The real pupil mode stack ``(m, y, x)`` that produced the
            Jacobian under audit.
        propagate: Callable mapping a mode stack ``(m, y, x)`` to its complex
            focal-plane Jacobian ``(m, y, x)`` -- the same chain used to build
            the process being audited.
        per_mode_rms: Scalar or ``(m,)`` drift rms. Partners inherit their
            mode's value. ``eta`` is invariant to an overall scale, so this
            matters only when the modes are unequally weighted.
        mask: Optional boolean focal-plane mask (a dark zone) to reduce over.
        locked_below: Ratio under which the verdict is ``"locked"``.

    Returns:
        A :class:`QuadratureAudit` report.
    """
    modes = jnp.asarray(modes)
    if modes.ndim != 3:
        raise ValueError(f"modes must be a 3D stack (m, y, x), got {modes.shape}")
    n = modes.shape[0]
    rms_in = jnp.asarray(per_mode_rms, dtype=float)
    if rms_in.ndim > 0 and rms_in.shape != (n,):
        raise ValueError(
            f"per_mode_rms has shape {rms_in.shape}; expected ({n},) "
            "to match the mode stack"
        )
    rms = jnp.broadcast_to(rms_in, (n,))

    return quadrature_audit_from_jacobians(
        jnp.asarray(propagate(modes)),
        jnp.asarray(propagate(quadrature_partner(modes))),
        per_mode_rms=rms,
        mask=mask,
        locked_below=locked_below,
    )
