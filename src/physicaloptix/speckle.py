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


def lambda_scaled_channels(e_nom, G, reference_wavelength_nm, wavelengths_nm):
    """Per-wavelength ``(e_nom, G)`` stacks under the lambda-scaling approximation.

    The standard chromatic model for a speckle field generated at one
    reference wavelength: the lambda/D morphology is wavelength-independent
    (set by the WFE spatial frequencies; the radial dilation on a fixed
    angular detector falls out of the consumer's wavelength-aware lambda/D
    conversion), while the OPD sensitivity carries the phase factor
    ``i 2 pi / lambda``, so ``G(lambda) = G(lambda0) * (lambda0/lambda)``
    -- the incoherent halo then scales as ``(lambda0/lambda)^2``. The
    nominal field ``e_nom`` is held fixed (the design leakage's own
    chromaticity is NOT modeled; propagate per sub-band for that).

    Args:
        e_nom: Complex nominal focal field, shape ``(y, x)``.
        G: Complex sensitivity ``d(E_focal)/d(mode)``, shape ``(m, y, x)``.
        reference_wavelength_nm: The wavelength ``G`` was generated at.
        wavelengths_nm: Channel wavelengths, shape ``(w,)``.

    Returns:
        ``(e_stack, g_stack)`` of shapes ``(w, y, x)`` and ``(w, m, y, x)``.
    """
    wavelengths = jnp.asarray(wavelengths_nm, dtype=float)
    scale = reference_wavelength_nm / wavelengths
    e_stack = jnp.broadcast_to(e_nom, (wavelengths.shape[0], *e_nom.shape))
    g_stack = G[jnp.newaxis] * scale[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    return e_stack, g_stack


def _check_chromatic_layout(e_nom, G, normalization, wavelengths_nm):
    """Validate mono ``(y,x)/(m,y,x)`` or chromatic ``(w,...)`` ingredients."""
    if wavelengths_nm is None:
        if e_nom.ndim != 2 or G.ndim != 3:
            raise ValueError(
                "monochromatic ingredients must be e_nom (y, x) and G "
                f"(m, y, x); got {e_nom.shape} and {G.shape} (set "
                "wavelengths_nm for a chromatic field)"
            )
        if normalization.ndim != 0:
            raise ValueError(
                "monochromatic normalization must be a scalar, got shape "
                f"{normalization.shape}"
            )
        return
    w = wavelengths_nm.shape[0]
    if e_nom.ndim != 3 or G.ndim != 4 or e_nom.shape[0] != w or G.shape[0] != w:
        raise ValueError(
            f"chromatic ingredients must be e_nom (w, y, x) and G "
            f"(w, m, y, x) with w == {w}; got {e_nom.shape} and {G.shape}"
        )
    if normalization.ndim not in (0, 1) or (
        normalization.ndim == 1 and normalization.shape[0] != w
    ):
        raise ValueError(
            f"chromatic normalization must be a scalar or shape ({w},); "
            f"got {normalization.shape}"
        )


def _select_channel(e_nom, G, normalization, wavelengths_nm, wavelength_nm):
    """The ``(e_nom, G, normalization)`` of the channel nearest a wavelength."""
    if wavelengths_nm is None:
        return e_nom, G, normalization
    index = jnp.argmin(jnp.abs(wavelengths_nm - jnp.asarray(wavelength_nm)))
    norm = jnp.broadcast_to(normalization, wavelengths_nm.shape)[index]
    return e_nom[index], G[index], norm


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

    Monochromatic by default: ``G`` / ``E_nom`` are at the design wavelength
    and ``realize`` ignores its ``wavelength_nm`` argument. With
    ``wavelengths_nm`` set, ``e_nom`` / ``G`` (and optionally
    ``normalization``) carry a leading channel axis and ``realize`` selects
    the channel nearest the requested wavelength while the mode trajectory
    stays shared across channels (a wavefront error in nanometres is
    achromatic). Build the stacks per sub-band for an exact model, or with
    :func:`lambda_scaled_channels` / :meth:`broadened` for the standard
    lambda-scaling approximation. The deep-contrast cross term needs
    float64 inputs.
    """

    e_nom: Array  # complex (y, x) or (w, y, x): nominal focal field
    G: Array  # complex (m, y, x) or (w, m, y, x): d(E_focal)/d(mode)
    amplitudes: Array  # float (m, f): per-mode spectral amplitudes a_kj
    frequencies_hz: Array  # float (f,): temporal frequencies f_j
    phases: Array  # float (m, f): per-mode random phases phi_kj
    normalization: Array
    pixel_scale_lod: float
    epoch_jd: float
    wavelengths_nm: Array | None
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
        wavelengths_nm=None,
    ):
        """Build a speckle field from precomputed ingredients.

        Args:
            e_nom: Complex nominal focal field, shape ``(y, x)`` -- or
                ``(w, y, x)`` with ``wavelengths_nm`` set.
            G: Complex sensitivity ``d(E_focal)/d(mode)``, shape
                ``(m, y, x)`` -- or ``(w, m, y, x)`` with
                ``wavelengths_nm`` set.
            amplitudes: Per-mode spectral amplitudes ``a_kj``, shape ``(m, f)``.
            frequencies_hz: Temporal frequencies ``f_j`` in Hz, shape ``(f,)``.
            phases: Per-mode random phases ``phi_kj``, shape ``(m, f)``.
            normalization: Intensity that maps to unit contrast (the telescope
                PSF peak the focal field is referenced to); a scalar, or one
                value per channel for a chromatic field.
            pixel_scale_lod: Native pixel scale in lambda/D per pixel
                (shared by every channel: the maps live in lambda/D units,
                where the morphology is achromatic).
            epoch_jd: Julian Date mapping to ``time_s = 0``. Default J2000.
            coherent: Include the pinning cross term. Default ``False``
                (incoherent halo).
            wavelengths_nm: Channel wavelengths, shape ``(w,)``, enabling
                the chromatic layout above. ``None`` (default) for a
                monochromatic field.
        """
        self.e_nom = e_nom
        self.G = G
        self.amplitudes = amplitudes
        self.frequencies_hz = frequencies_hz
        self.phases = phases
        self.normalization = jnp.asarray(normalization, dtype=float)
        self.pixel_scale_lod = pixel_scale_lod
        self.epoch_jd = epoch_jd
        self.wavelengths_nm = (
            None if wavelengths_nm is None else jnp.asarray(wavelengths_nm, dtype=float)
        )
        self.coherent = coherent

    def __check_init__(self):
        """Validate the (chromatic) ingredient layout."""
        _check_chromatic_layout(
            self.e_nom, self.G, self.normalization, self.wavelengths_nm
        )

    def _eps(self, time_s):
        """Mode coefficients ``eps(t)`` by spectral synthesis, shape ``(m,)``."""
        t = jnp.asarray(time_s)
        phase = 2.0 * jnp.pi * self.frequencies_hz * t + self.phases
        return jnp.sum(self.amplitudes * jnp.cos(phase), axis=-1)

    def realize(self, *, wavelength_nm, time_s=0.0):
        """Speckle contrast delta at ``time_s`` (see class docstring)."""
        e_nom, g, normalization = _select_channel(
            self.e_nom, self.G, self.normalization, self.wavelengths_nm, wavelength_nm
        )
        g_eps = jnp.tensordot(self._eps(time_s), g, axes=1)
        if self.coherent:
            # The stable form of |E_nom + g_eps|^2 - |E_nom|^2: computing the
            # cross term directly avoids subtracting two floor-magnitude numbers
            # (catastrophic cancellation in the bright regime).
            delta = 2.0 * jnp.real(jnp.conj(e_nom) * g_eps) + jnp.abs(g_eps) ** 2
        else:
            delta = jnp.abs(g_eps) ** 2
        return delta / normalization

    def broadened(self, *, reference_wavelength_nm, wavelengths_nm):
        """A chromatic copy under the lambda-scaling approximation.

        See :func:`lambda_scaled_channels` for the physics and its limits.

        Args:
            reference_wavelength_nm: The wavelength this field's ``G`` was
                generated at.
            wavelengths_nm: Channel wavelengths for the broadened field.

        Returns:
            A chromatic ``AnalyticSpeckleField`` sharing this field's
            temporal realization.
        """
        if self.wavelengths_nm is not None:
            raise ValueError("field is already chromatic")
        e_stack, g_stack = lambda_scaled_channels(
            self.e_nom, self.G, reference_wavelength_nm, wavelengths_nm
        )
        return AnalyticSpeckleField(
            e_stack,
            g_stack,
            self.amplitudes,
            self.frequencies_hz,
            self.phases,
            self.normalization,
            pixel_scale_lod=self.pixel_scale_lod,
            epoch_jd=self.epoch_jd,
            coherent=self.coherent,
            wavelengths_nm=wavelengths_nm,
        )


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
