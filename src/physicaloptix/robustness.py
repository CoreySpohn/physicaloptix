"""Sensitivity and robustness products built from a dark-zone Jacobian.

Where :mod:`physicaloptix.speckle` draws realizations of a drifting wavefront,
this module answers the question a realization cannot: given a per-mode
wavefront-error budget, what dark-zone contrast does it BUY, and which modes
are buying it? The answer is a set of small mode-by-mode matrices contracted
against the per-mode variance vector, so a tolerancing sweep costs a matrix
product rather than a Monte-Carlo campaign.

Everything here is a contraction of the SAME two objects the linearization
already provides -- the nominal dark-zone field ``c`` and the dark-zone
Jacobian ``g = d(E_dz)/d(mode)`` -- so the products differentiate with
respect to the design through the propagation that produced them. That is
the capability a non-differentiable propagation cannot offer: the gradient of
a contrast budget with respect to the optical prescription.

The central object is the PASTIS matrix ``Re(g g^H) / n_dz``, whose quadratic
form is the classical mode-mode contrast sensitivity. The annulus variance
budget extends it to the IMPROPER (non-circular) case, where the residual
speckle field's pseudo-covariance does not vanish: the exact dark-zone-mean
variance is then a sum of four contractions, an algebraic identity rather
than an approximation (see :func:`annulus_variance_budget`).
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

__all__ = [
    "SensitivityBudget",
    "SensitivityOperators",
    "pastis_matrix",
]


def _dark_zone(e_nom, G, mask):
    """The nominal field and Jacobian restricted to the dark zone."""
    if mask is None:
        return jnp.ravel(e_nom), G.reshape(G.shape[0], -1)
    mask = jnp.asarray(mask, dtype=bool)
    return e_nom[mask], G[:, mask]


def pastis_matrix(e_nom, G, mask=None):
    """The PASTIS mode-mode contrast-sensitivity matrix.

    ``P = Re(g g^H) / n_dz`` for the dark-zone Jacobian ``g``, so that a
    wavefront-error vector ``a`` produces mean dark-zone intensity
    ``a^T P a`` in the incoherent (speckle-speckle) limit. Diagonal entries
    are each mode's own contrast cost; off-diagonal entries are the
    interference between modes, which is what makes a budget non-separable.

    Args:
        e_nom: Complex nominal focal field, shape ``(y, x)``. Present for
            signature symmetry with the rest of the module; the PASTIS matrix
            itself depends only on ``G``.
        G: Complex sensitivity ``d(E_focal)/d(mode)``, shape ``(m, y, x)``.
        mask: Optional boolean dark-zone mask over the focal grid. ``None``
            uses every pixel.

    Returns:
        The real symmetric ``(m, m)`` matrix, in intensity units per squared
        mode unit.
    """
    _, g = _dark_zone(e_nom, G, mask)
    n_dz = g.shape[1]
    return jnp.real(g @ jnp.conj(g).T) / n_dz


class SensitivityOperators(eqx.Module):
    """The mode-by-mode contractions a dark-zone variance budget is built from.

    Built once from ``(e_nom, G)`` by :meth:`build`, then contracted against
    any number of per-mode variance vectors. The whole point is that the
    expensive object (the Jacobian over a full focal plane) collapses ONCE
    into ``(m, m)`` matrices and ``(m,)`` vectors, after which sweeping a
    tolerancing budget is linear algebra in the number of modes rather than
    the number of pixels.

    Attributes:
        pastis: ``Re(g g^H) / n_dz``, the classical sensitivity matrix.
        heterodyne: ``|g|^2 I_C / n_dz``, the nominal-field beat term.
        imbalance: ``mean(Re(conj(c)^2 g^2))``, the improper (non-circular)
            partner of ``heterodyne``.
        speckle: ``|g|^2 |g|^2^T / n_dz``, the speckle-speckle term.
        pseudocov: ``Re(g^2 (g^2)^H) / n_dz``, its improper partner.
        mean_intensity: The dark-zone mean of ``|e_nom|^2``.
        n_dz: Number of dark-zone pixels.
    """

    pastis: Array
    heterodyne: Array
    imbalance: Array
    speckle: Array
    pseudocov: Array
    mean_intensity: Array
    n_dz: int = eqx.field(static=True)

    @classmethod
    def build(cls, e_nom, G, mask=None):
        """Collapse a nominal field and Jacobian into the mode-space operators.

        Args:
            e_nom: Complex nominal focal field, shape ``(y, x)``.
            G: Complex sensitivity ``d(E_focal)/d(mode)``, shape
                ``(m, y, x)``.
            mask: Optional boolean dark-zone mask; ``None`` uses every pixel.

        Returns:
            A :class:`SensitivityOperators`.
        """
        c, g = _dark_zone(e_nom, G, mask)
        n_dz = g.shape[1]
        i_c = jnp.abs(c) ** 2
        abs_g2 = jnp.abs(g) ** 2
        g2 = g**2
        return cls(
            pastis=jnp.real(g @ jnp.conj(g).T) / n_dz,
            heterodyne=abs_g2 @ i_c / n_dz,
            imbalance=jnp.mean(jnp.real(jnp.conj(c)[None, :] ** 2 * g2), axis=1),
            speckle=abs_g2 @ abs_g2.T / n_dz,
            pseudocov=jnp.real(g2 @ jnp.conj(g2).T) / n_dz,
            mean_intensity=jnp.mean(i_c),
            n_dz=n_dz,
        )

    def budget(self, per_mode_variance) -> "SensitivityBudget":
        """The exact dark-zone-mean variance budget for a variance vector.

        Args:
            per_mode_variance: Per-mode wavefront-error VARIANCE (the square
                of the per-mode rms), shape ``(m,)``.

        Returns:
            A :class:`SensitivityBudget` whose ``total`` is the exact
            dark-zone-mean contrast variance.
        """
        r2 = jnp.asarray(per_mode_variance)
        return SensitivityBudget(
            heterodyne=2.0 * self.heterodyne @ r2,
            imbalance=2.0 * self.imbalance @ r2,
            speckle=r2 @ self.speckle @ r2,
            pseudocov=r2 @ self.pseudocov @ r2,
        )

    def circular_budget(self, per_mode_variance):
        """The budget a CIRCULAR (proper-Gaussian) model would predict.

        The textbook form: ``2 I_C Gamma + Gamma^2`` for the mean residual
        power ``Gamma``. It drops both improper terms, so comparing it with
        :meth:`budget` measures exactly what the circularity assumption costs
        -- the gap is not a small correction when the dark zone is deep, and
        it enters a requirements inversion as an over-allowance on the
        wavefront-error budget.

        Args:
            per_mode_variance: Per-mode wavefront-error variance, shape
                ``(m,)``.

        Returns:
            The scalar variance the circular model predicts.
        """
        r2 = jnp.asarray(per_mode_variance)
        gamma_bar = self.mean_residual_power(r2)
        return 2.0 * self.mean_intensity * gamma_bar + gamma_bar**2

    def mean_residual_power(self, per_mode_variance):
        """Dark-zone-mean residual power ``mean_p sum_k r2_k |g_kp|^2``.

        This is the PASTIS diagonal contracted with the variance vector: each
        mode's own mean contribution, with no interference between modes.
        """
        return jnp.diagonal(self.pastis) @ jnp.asarray(per_mode_variance)


class SensitivityBudget(eqx.Module):
    """The four contractions of an exact improper dark-zone variance budget.

    ``total`` is an algebraic IDENTITY, not a model: for a linear response to
    Gaussian mode coefficients it equals the dark-zone mean of the exact
    per-pixel variance, with no approximation to check. The split is what
    carries the physics -- ``heterodyne`` and ``imbalance`` are the terms that
    beat against the nominal field (dominant in a shallow zone), ``speckle``
    and ``pseudocov`` the terms quadratic in the residual (dominant once the
    nominal field is dug away).

    ``imbalance`` and ``pseudocov`` are the IMPROPER terms: they vanish only
    if the residual field is circularly symmetric, which a real coronagraph
    dark zone is not.
    """

    heterodyne: Array
    imbalance: Array
    speckle: Array
    pseudocov: Array

    @property
    def total(self):
        """The exact dark-zone-mean contrast variance."""
        return self.heterodyne + self.imbalance + self.speckle + self.pseudocov

    @property
    def improper_fraction(self):
        """Share of the total carried by the two non-circular terms."""
        return (self.imbalance + self.pseudocov) / self.total
