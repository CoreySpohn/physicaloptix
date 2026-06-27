"""Analytic speckle field built from precomputed field ingredients.

``AnalyticSpeckleField`` packages the frozen ingredients of the linear speckle
generator -- a complex nominal focal field ``E_nom``, a complex wavefront-error
sensitivity ``G = d(E_focal)/d(mode)``, and a temporal model for the drifting mode
coefficients ``eps(t)`` -- behind optixstuff's :class:`AbstractSpeckleField`. It
realizes ``I(t) = |E_nom + G eps(t)|^2`` as a contrast map for coronagraphoto's
speckle path.

The ingredients come from a physical-optics propagation of a specific design (e.g.
the EAC-1 AAVC): ``G`` carries the coronagraphic PSF morphology because it is
propagated through the coronagraph, and ``E_nom`` is complex, so the
speckle-pinning cross term is available (set ``coherent=True``) -- which an
intensity-only YIP cannot provide. The temporal coefficients are a spectral
synthesis ``eps_k(t) = sum_j a_kj cos(2 pi f_j t + phi_kj)`` with the randomness
fixed at construction, so :meth:`realize` is deterministic in time and
differentiable, and temporal correlation survives a roll sequence.

This class owns no file I/O: a caller builds the arrays (e.g. from a cached
export) and constructs the field.
"""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from optixstuff.speckle import AbstractSpeckleField

J2000_JD = 2451545.0


class AnalyticSpeckleField(AbstractSpeckleField):
    """Time-driven speckle field from frozen ``E_nom`` / ``G`` / ``eps(t)``.

    :meth:`realize` returns the contrast delta -- the wavefront-error excess
    over the deterministic floor, i.e. ``(I(t) - |E_nom|^2) / normalization``
    -- never the floor itself, so it adds cleanly on top of the coronagraph's
    ``stellar_intens`` in ``coronagraphoto.speckle_rate``. With
    ``coherent=False`` (default) it returns the strictly positive incoherent
    halo ``|G eps|^2 / normalization`` (no pinning); with ``coherent=True`` it
    adds the cross term ``2 Re(E_nom* . G eps)``, which carries the bright-tail
    speckle pinning and needs the complex ``E_nom``.

    The field is monochromatic in v1: ``G`` / ``E_nom`` are at the design
    wavelength and ``realize`` ignores its ``wavelength_nm`` argument (kept for
    interface conformance). The deep-contrast cross term needs float64 inputs.
    """

    e_nom: Array  # complex (y, x): nominal focal field
    G: Array  # complex (m, y, x): d(E_focal)/d(mode)
    amplitudes: Array  # float (m, f): per-mode spectral amplitudes a_kj
    frequencies_hz: Array  # float (f,): temporal frequencies f_j
    phases: Array  # float (m, f): per-mode random phases phi_kj
    normalization: float
    pixel_scale_lod: float
    epoch_jd: float
    coherent: bool = eqx.field(static=True)

    def __init__(
        self,
        e_nom,
        G,
        amplitudes,
        frequencies_hz,
        phases,
        normalization,
        *,
        pixel_scale_lod=0.25,
        epoch_jd=J2000_JD,
        coherent=False,
    ):
        """Build a speckle field from precomputed ingredients.

        Args:
            e_nom: Complex nominal focal field, shape ``(y, x)``.
            G: Complex sensitivity ``d(E_focal)/d(mode)``, shape ``(m, y, x)``.
            amplitudes: Per-mode spectral amplitudes ``a_kj``, shape ``(m, f)``.
            frequencies_hz: Temporal frequencies ``f_j`` in Hz, shape ``(f,)``.
            phases: Per-mode random phases ``phi_kj``, shape ``(m, f)``.
            normalization: Intensity that maps to unit contrast (the telescope
                PSF peak the focal field is referenced to).
            pixel_scale_lod: Native pixel scale in lambda/D per pixel. Must
                equal the coronagraph's plate scale for the speckle path.
            epoch_jd: Julian Date mapping to ``time_s = 0``. Default J2000.
            coherent: Include the pinning cross term. Default ``False``
                (incoherent halo).
        """
        self.e_nom = e_nom
        self.G = G
        self.amplitudes = amplitudes
        self.frequencies_hz = frequencies_hz
        self.phases = phases
        self.normalization = normalization
        self.pixel_scale_lod = pixel_scale_lod
        self.epoch_jd = epoch_jd
        self.coherent = coherent

    def _eps(self, time_s):
        """Mode coefficients ``eps(t)`` by spectral synthesis, shape ``(m,)``."""
        t = jnp.asarray(time_s)
        phase = 2.0 * jnp.pi * self.frequencies_hz * t + self.phases
        return jnp.sum(self.amplitudes * jnp.cos(phase), axis=-1)

    def realize(self, *, wavelength_nm, time_s=0.0):
        """Speckle contrast delta at ``time_s`` (see class docstring)."""
        eps = self._eps(time_s)
        g_eps = jnp.tensordot(eps, self.G, axes=1)
        if self.coherent:
            delta = jnp.abs(self.e_nom + g_eps) ** 2 - jnp.abs(self.e_nom) ** 2
        else:
            delta = jnp.abs(g_eps) ** 2
        return delta / self.normalization
