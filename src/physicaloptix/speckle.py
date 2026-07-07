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
import jax
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
            # The stable form of |E_nom + g_eps|^2 - |E_nom|^2: computing the
            # cross term directly avoids subtracting two floor-magnitude numbers
            # (catastrophic cancellation in the bright regime).
            delta = 2.0 * jnp.real(jnp.conj(self.e_nom) * g_eps) + jnp.abs(g_eps) ** 2
        else:
            delta = jnp.abs(g_eps) ** 2
        return delta / self.normalization


class SpeckleProcess(eqx.Module):
    """One parameter set for the linear speckle process; views derive from it.

    Holds the spatial ingredients (``E_nom``, ``G``) together with the
    per-mode temporal PSD specification (knee + slope + per-mode rms), so the
    generator view and any future inference view (the state-space ``(A, Q)``
    of a filter) derive from the SAME parameters and cannot drift apart ("one
    parameter set, two views").

    :meth:`draw` samples one frozen realization -- spectral-synthesis
    amplitudes from the PSD and uniform random phases -- and returns it as an
    :class:`AnalyticSpeckleField` (the sampled/generator view, unchanged).
    Ensembles are many draws with different keys; each draw's per-mode rms is
    exact (the random amplitudes are renormalized mode-by-mode), so the WFE
    budget is honored draw by draw rather than only in expectation.

    The PSD is the SCoOB-style knee form ``(1 + (f / knee)^2)^(slope / 2)``,
    evaluated on a fixed log-spaced frequency grid straddling the knee.
    ``per_mode_rms`` is in the same mode units as ``G``'s mode coordinate.
    """

    e_nom: Array  # complex (y, x): nominal focal field
    G: Array  # complex (m, y, x): d(E_focal)/d(mode)
    per_mode_rms: Array  # float (m,): rms drift per mode
    knee_hz: float
    slope: float
    normalization: float
    pixel_scale_lod: float
    epoch_jd: float
    coherent: bool = eqx.field(static=True)
    n_freq: int = eqx.field(static=True)
    decades_below: float = eqx.field(static=True)
    decades_above: float = eqx.field(static=True)

    def __init__(
        self,
        e_nom,
        G,
        per_mode_rms,
        knee_hz,
        normalization,
        *,
        slope=-2.0,
        pixel_scale_lod=0.25,
        epoch_jd=J2000_JD,
        coherent=False,
        n_freq=64,
        decades_below=0.7,
        decades_above=2.3,
    ):
        """Build the process parameter object.

        Args:
            e_nom: Complex nominal focal field, shape ``(y, x)``.
            G: Complex sensitivity ``d(E_focal)/d(mode)``, shape ``(m, y, x)``.
            per_mode_rms: Per-mode rms drift, scalar (broadcast to every
                mode) or shape ``(m,)``.
            knee_hz: Temporal PSD knee frequency in Hz
                (``1 / (2 pi tau)`` for a decorrelation time ``tau``).
            normalization: Intensity that maps to unit contrast.
            slope: High-frequency PSD power-law slope. Default -2.
            pixel_scale_lod: Native pixel scale in lambda/D per pixel.
            epoch_jd: Julian Date mapping to ``time_s = 0``. Default J2000.
            coherent: Drawn fields include the pinning cross term.
            n_freq: Number of spectral-synthesis frequencies.
            decades_below: Frequency-grid extent below the knee (decades).
            decades_above: Frequency-grid extent above the knee (decades).
        """
        self.e_nom = e_nom
        self.G = G
        self.per_mode_rms = jnp.broadcast_to(
            jnp.asarray(per_mode_rms, dtype=float), (G.shape[0],)
        )
        self.knee_hz = knee_hz
        self.slope = slope
        self.normalization = normalization
        self.pixel_scale_lod = pixel_scale_lod
        self.epoch_jd = epoch_jd
        self.coherent = coherent
        self.n_freq = n_freq
        self.decades_below = decades_below
        self.decades_above = decades_above

    def __check_init__(self):
        """Validate that per_mode_rms matches G's mode axis."""
        if self.per_mode_rms.shape != (self.G.shape[0],):
            raise ValueError(
                f"per_mode_rms has shape {self.per_mode_rms.shape}; expected "
                f"({self.G.shape[0]},) to match G's mode axis"
            )

    @classmethod
    def from_decorrelation(
        cls,
        e_nom,
        G,
        *,
        decorr_hours,
        total_rms,
        normalization,
        **kwargs,
    ):
        """Parameterize by decorrelation time and a total WFE budget.

        The knee is ``1 / (2 pi tau)`` so the field decorrelates over roughly
        ``decorr_hours``, and the budget is split evenly over the modes
        (``per_mode_rms = total_rms / sqrt(m)``; rms adds in quadrature).
        """
        tau_s = decorr_hours * 3600.0
        m = G.shape[0]
        return cls(
            e_nom,
            G,
            total_rms / jnp.sqrt(float(m)),
            1.0 / (2.0 * jnp.pi * tau_s),
            normalization,
            **kwargs,
        )

    def frequencies_hz(self) -> Array:
        """The fixed log-spaced spectral-synthesis frequency grid."""
        log_knee = jnp.log10(self.knee_hz)
        return jnp.logspace(
            log_knee - self.decades_below, log_knee + self.decades_above, self.n_freq
        )

    def psd(self, frequencies_hz) -> Array:
        """Temporal PSD (knee form) evaluated at ``frequencies_hz``."""
        return (1.0 + (frequencies_hz / self.knee_hz) ** 2) ** (self.slope / 2.0)

    def draw(self, key) -> AnalyticSpeckleField:
        """Sample one frozen realization of the process.

        Amplitudes are PSD-shaped Gaussian draws renormalized so each mode's
        synthesized ``eps_k(t)`` has exactly ``per_mode_rms[k]`` rms
        (``Var[eps_k] = 0.5 sum_j a_kj^2``); phases are uniform on
        ``[0, 2 pi)``. Independent realizations come from independent keys.
        """
        k_amp, k_phase = jax.random.split(key)
        m = self.G.shape[0]
        f = self.frequencies_hz()
        amp = jnp.sqrt(self.psd(f))[None, :] * jax.random.normal(
            k_amp, (m, self.n_freq)
        )
        amp = amp * (
            self.per_mode_rms[:, None]
            / jnp.sqrt(0.5 * jnp.sum(amp**2, axis=1, keepdims=True))
        )
        phases = jax.random.uniform(
            k_phase, (m, self.n_freq), minval=0.0, maxval=2.0 * jnp.pi
        )
        return AnalyticSpeckleField(
            self.e_nom,
            self.G,
            amp,
            f,
            phases,
            normalization=self.normalization,
            pixel_scale_lod=self.pixel_scale_lod,
            epoch_jd=self.epoch_jd,
            coherent=self.coherent,
        )
