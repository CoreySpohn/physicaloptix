"""Compensated binary32 arithmetic for exact angular-spectrum phase.

A carrier or spatial phase can contain millions of cycles. A float32 pair
retains the residual needed for range reduction while keeping intermediates
in the field's precision. These routines operate on normal-range operands.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _split_host(value):
    """Split a static binary64 value into two binary32-compatible parts."""
    high = np.asarray(value, dtype=np.float32).astype(np.float64)
    return high, value - high


def _add(a, b):
    """Renormalize a sum with its rounding residual (TwoSum)."""
    a_hi, a_lo = a
    b_hi, b_lo = b
    high = jax.lax.optimization_barrier(a_hi + b_hi)
    b_part = jax.lax.optimization_barrier(high - a_hi)
    a_part = jax.lax.optimization_barrier(high - b_part)
    low = (a_hi - a_part) + (b_hi - b_part) + a_lo + b_lo
    return _normalize(high, low)


def _normalize(high, low):
    """Renormalize the pair when the first term dominates (FastTwoSum)."""
    value = jax.lax.optimization_barrier(high + low)
    return value, low - jax.lax.optimization_barrier(value - high)


def _neg(a):
    """Negate both parts without rounding."""
    return -a[0], -a[1]


def _upper(x):
    """Take the upper 12 significant bits for an exact partial product."""
    return (x.view(jnp.uint32) & np.uint32(0xFFFFF000)).view(x.dtype)


def _mul(a, b):
    """Retain the rounded product and the Dekker product residual."""
    a_hi, a_lo = a
    b_hi, b_lo = b
    high = jax.lax.optimization_barrier(a_hi * b_hi)
    a_upper, b_upper = _upper(a_hi), _upper(b_hi)
    a_lower, b_lower = a_hi - a_upper, b_hi - b_upper
    hh, hl, lh = jax.lax.optimization_barrier(
        (a_upper * b_upper, a_upper * b_lower, a_lower * b_upper)
    )
    low = ((hh - high) + hl + lh) + (a_lower * b_lower + a_hi * b_lo + a_lo * b_hi)
    return _normalize(high, low)


def _div(a, b):
    """Refine a rounded quotient using its compensated residual."""
    q = jax.lax.optimization_barrier(a[0] / b[0])
    residual = _add(a, _neg(_mul((q, jnp.zeros_like(q)), b)))
    return _normalize(q, (residual[0] + residual[1]) / b[0])


def _sqrt(a):
    """Refine a rounded square root with one Newton correction."""
    root = jnp.sqrt(a[0])
    residual = _add(
        a, _neg(_mul((root, jnp.zeros_like(root)), (root, jnp.zeros_like(root))))
    )
    return _normalize(root, (residual[0] + residual[1]) / (2 * root))


def _spatial_cycles(wavelength, cutoff, path_per_cutoff, near_grazing, non_dc):
    """Evaluate the reduced spatial offset or the direct phase near cutoff."""
    wl = (wavelength, jnp.zeros_like(wavelength))
    relative = _div(wl, cutoff)
    # Form the small difference first. The split cutoff retains detuning
    # even when cutoff and wavelength round to the same binary32 value.
    delta = _add(cutoff, _neg(wl))
    detuning = delta[0] + delta[1]
    propagating = (detuning > 0) | ~non_dc
    evanescent = (detuning < 0) & non_dc
    one = (jnp.ones_like(relative[0]), jnp.zeros_like(relative[0]))
    radicand = _mul(_div(delta, cutoff), _add(one, relative))
    live = non_dc & propagating
    root = _sqrt(
        (
            jnp.where(live, radicand[0], 1),
            jnp.where(live, radicand[1], 0),
        )
    )

    # The direct expression is smaller near cutoff; elsewhere retaining the
    # carrier separately also resolves phases too large for a binary32 pair.
    use_direct = near_grazing & live
    direct = _div(
        _mul(path_per_cutoff, root),
        (jnp.where(use_direct, relative[0], 1), jnp.where(use_direct, relative[1], 0)),
    )
    offset = _neg(_div(_mul(path_per_cutoff, relative), _add(one, root)))
    high = jnp.where(near_grazing, direct[0], offset[0])
    low = jnp.where(near_grazing, direct[1], offset[1])
    cycles = jnp.where(live, jnp.fmod(high, 1) + jnp.fmod(low, 1), 0)
    return cycles, radicand[0], propagating, evanescent
