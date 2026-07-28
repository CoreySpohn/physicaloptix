"""Analytic speckle field built from precomputed field ingredients.

``AnalyticSpeckleField`` packages the frozen ingredients of the linear speckle
generator -- a complex nominal focal field ``E_nom``, a complex wavefront-error
sensitivity ``G = d(E_focal)/d(mode)``, and a temporal model for the drifting mode
coefficients ``eps(t)`` -- behind optixstuff's :class:`AbstractSpeckleField`. It
realizes ``I(t) = |E_nom + G eps(t)|^2`` as a per-pixel flux-fraction map for
coronagraphoto's speckle path, carrying the photometric primitives (input pupil
energy, output pixel scale) rather than a pre-derived scalar; peak-referenced
contrast is a derived view (:meth:`AnalyticSpeckleField.peak_contrast` with
:func:`telescope_peak`).

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

from physicaloptix.transforms.fraunhofer import Fraunhofer

J2000_JD = 2451545.0


def telescope_peak(field, grid_out):
    """Peak intensity density of the unocculted telescope PSF.

    Propagates ``field`` (monochromatic, pupil plane) through a bare
    Fraunhofer transform to ``grid_out`` and returns the maximum intensity
    density -- the caller-side reference that converts flux-fraction maps to
    peak-referenced contrast (:meth:`AnalyticSpeckleField.peak_contrast`).
    Computed on the grid in use because the sampled peak value depends on the
    pixel scale; do not cache it across grids.

    Args:
        field: The unocculted aperture ``Field`` at the pupil plane.
        grid_out: The focal ``Grid`` the peak is sampled on.

    Returns:
        The peak intensity density as a float.
    """
    transform = Fraunhofer(field.grid, grid_out)
    return float(jnp.max(transform(field).intensity()))


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


def _check_chromatic_layout(e_nom, G, input_energy, wavelengths_nm):
    """Validate mono ``(y,x)/(m,y,x)`` or chromatic ``(w,...)`` ingredients."""
    if wavelengths_nm is None:
        if e_nom.ndim != 2 or G.ndim != 3:
            raise ValueError(
                "monochromatic ingredients must be e_nom (y, x) and G "
                f"(m, y, x); got {e_nom.shape} and {G.shape} (set "
                "wavelengths_nm for a chromatic field)"
            )
        if input_energy.ndim != 0:
            raise ValueError(
                "monochromatic input_energy must be a scalar, got shape "
                f"{input_energy.shape}"
            )
        return
    w = wavelengths_nm.shape[0]
    if e_nom.ndim != 3 or G.ndim != 4 or e_nom.shape[0] != w or G.shape[0] != w:
        raise ValueError(
            f"chromatic ingredients must be e_nom (w, y, x) and G "
            f"(w, m, y, x) with w == {w}; got {e_nom.shape} and {G.shape}"
        )
    if input_energy.ndim not in (0, 1) or (
        input_energy.ndim == 1 and input_energy.shape[0] != w
    ):
        raise ValueError(
            f"chromatic input_energy must be a scalar or shape ({w},); "
            f"got {input_energy.shape}"
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

    :meth:`realize` returns the per-pixel flux-fraction delta -- the
    wavefront-error excess over the deterministic floor, i.e.
    ``(I(t) - |E_nom|^2) * du^2 / E_in`` -- never the floor itself, so it adds
    cleanly on top of the coronagraph's ``stellar_intens`` (itself a
    flux-fraction map) in ``coronagraphoto.speckle_rate``, honoring the
    optixstuff :class:`AbstractSpeckleField` contract. The photometric
    primitives are stored (``input_energy`` = ``E_in``, ``pixel_scale_lod`` =
    ``du``) and the divisor ``normalization = input_energy /
    pixel_scale_lod**2`` is derived once at construction, so the hot path
    carries no unit branching; peak-referenced contrast is the derived view
    :meth:`peak_contrast`. With ``coherent=False`` (default) the delta is the
    strictly positive incoherent halo ``|G eps|^2`` (no pinning); with
    ``coherent=True`` it adds the cross term ``2 Re(E_nom* . G eps)``, which
    carries the bright-tail speckle pinning and needs the complex ``E_nom``.

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
    frequencies_hz: Array  # float (f,) shared or (m, f) per-mode: frequencies f_j
    phases: Array  # float (m, f): per-mode random phases phi_kj
    input_energy: Array
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
        *,
        input_energy,
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
            frequencies_hz: Temporal frequencies ``f_j`` in Hz, shape ``(f,)``
                (shared across modes) or ``(m, f)`` (per-mode grids, which
                ``_eps`` broadcasts against the ``(m, f)`` phases unchanged).
            phases: Per-mode random phases ``phi_kj``, shape ``(m, f)``.
            input_energy: Total energy of the field handed to the coronagraph
                train at the pre-coronagraph reference plane
                (``field.energy()``: ``sum(|E|^2)`` times the pupil cell
                area); a scalar, or one value per channel for a chromatic
                field. ``linearize`` records it as
                ``Linearization.input_energy``.
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
        self.input_energy = jnp.asarray(input_energy, dtype=float)
        self.pixel_scale_lod = pixel_scale_lod
        # Derived once, eagerly: realize() divides by one stored value, so the
        # jitted hot path carries no unit branching, and the primitives stay
        # recorded for export / the peak_contrast view. tree_at on
        # input_energy does not re-derive this; rebuild the field to change
        # photometry.
        self.normalization = self.input_energy / pixel_scale_lod**2
        self.epoch_jd = epoch_jd
        self.wavelengths_nm = (
            None if wavelengths_nm is None else jnp.asarray(wavelengths_nm, dtype=float)
        )
        self.coherent = coherent

    def __check_init__(self):
        """Validate the (chromatic) ingredient layout."""
        _check_chromatic_layout(
            self.e_nom, self.G, self.input_energy, self.wavelengths_nm
        )

    def _eps(self, time_s):
        """Mode coefficients ``eps(t)`` by spectral synthesis, shape ``(m,)``."""
        t = jnp.asarray(time_s)
        phase = 2.0 * jnp.pi * self.frequencies_hz * t + self.phases
        return jnp.sum(self.amplitudes * jnp.cos(phase), axis=-1)

    def realize(self, *, wavelength_nm, time_s=0.0):
        """Per-pixel flux-fraction delta at ``time_s`` (see class docstring)."""
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

    def peak_contrast(self, *, telescope_peak, wavelength_nm, time_s=0.0):
        """The :meth:`realize` map as peak-referenced contrast (a view).

        Contrast curves are quoted against the unocculted telescope PSF peak;
        this rescales the per-pixel flux fraction by
        ``normalization / telescope_peak`` (the stored primitives make the
        conversion exact). Get ``telescope_peak`` from
        :func:`telescope_peak` on the grid actually in use -- the peak-pixel
        value is sampling-dependent, which is why it is never stored.

        Args:
            telescope_peak: Peak intensity density of the unocculted
                telescope PSF on this field's grid.
            wavelength_nm: Wavelength in nanometres (chromatic fields select
                the nearest channel, as in :meth:`realize`).
            time_s: Time since ``epoch_jd`` in seconds.

        Returns:
            2D contrast-delta array (dimensionless, peak-referenced).
        """
        _, _, norm = _select_channel(
            self.e_nom,
            self.G,
            self.normalization,
            self.wavelengths_nm,
            wavelength_nm,
        )
        delta = self.realize(wavelength_nm=wavelength_nm, time_s=time_s)
        return delta * norm / telescope_peak

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
            input_energy=self.input_energy,
            pixel_scale_lod=self.pixel_scale_lod,
            epoch_jd=self.epoch_jd,
            coherent=self.coherent,
            wavelengths_nm=wavelengths_nm,
        )


class SpeckleMoments(eqx.Module):
    """Closed-form ensemble moments of a :class:`SpeckleProcess` contrast field.

    The frozen output of :meth:`SpeckleProcess.moments`: the per-pixel mean and
    variance maps of the flux-fraction delta (the units :meth:`realize`
    returns), the raw kernels they are built from (``Gamma``, the complex
    pseudo-covariance ``P``, and the pinning-quadrature variance ``Var(X)``,
    in the field's intensity units), and -- when a dark-zone mask is supplied
    -- the mask-averaged mean and variance.
    """

    mean_map: Array  # E[delta] per pixel (flux-fraction units)
    var_map: Array  # Var[delta] per pixel (flux-fraction units^2)
    gamma_map: Array  # Gamma = sum_k rms_k^2 |g_k|^2 (real, >= 0)
    p_map: Array  # P = sum_k rms_k^2 g_k^2 (complex pseudo-covariance)
    var_x_map: Array  # Var(X), the pinning-quadrature variance
    annulus_mean: Array | None = None  # mask-averaged E[delta]
    annulus_var: Array | None = None  # mask-averaged Var[delta]


class CrossBandMoments(eqx.Module):
    """Closed-form joint band-pair moments of a chromatic :class:`SpeckleProcess`.

    The frozen output of :meth:`SpeckleProcess.cross_band_moments`: per pixel,
    the exact second-order description of the flux-fraction deltas across the
    process's wavelength channels, which share one real mode trajectory. The
    field-level kernels are

        Gamma_ij = sum_k rms_k^2 rho_k(tau) G_k(i) conj(G_k(j))
        P_ij     = sum_k rms_k^2 rho_k(tau) G_k(i) G_k(j)

    (``P`` is the pseudo-covariance -- the phase-sensitive correlation of the
    improper complex Gaussian field; it vanishes for circular statistics but
    not in general) and the intensity-level covariance is the complex Gaussian
    moment theorem, exact and noncentral:

        N_i N_j Cov[delta_i, delta_j] = |Gamma_ij|^2 + |P_ij|^2
            + 2 Re[conj(A_i) A_j Gamma_ij + conj(A_i) conj(A_j) P_ij]

    with ``A_i = e_nom[i]`` and the heterodyne bracket present only for a
    ``coherent`` process. ``tau_s`` is the lag both kernels are damped by
    (``rho_k`` is the synthesis autocorrelation); the equal-time container has
    ``tau_s = 0.0`` and its band-diagonal reproduces the per-channel
    :meth:`SpeckleProcess.moments` exactly. ``mean_map`` is time-independent.
    """

    mean_map: Array  # (w, y, x): E[delta] per channel (flux-fraction units)
    cov_map: Array  # (w, w, y, x): Cov[delta_i(t), delta_j(t + tau_s)]
    gamma_map: Array  # (w, w, y, x) complex: Gamma_ij
    p_map: Array  # (w, w, y, x) complex: P_ij (pseudo-covariance)
    tau_s: float = eqx.field(static=True)
    annulus_mean: Array | None = None  # (w,): mask-averaged mean
    annulus_cov: Array | None = None  # (w, w): mask-averaged covariance


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

    Monochromatic by default: ``e_nom`` / ``G`` carry no channel axis. With
    ``wavelengths_nm`` set, ``e_nom`` / ``G`` carry a leading channel axis
    (``(w, y, x)`` / ``(w, m, y, x)``, the layout :func:`linearize` records
    for a chromatic field) and every :meth:`draw` carries the same
    wavelengths through to its :class:`AnalyticSpeckleField`, which then
    selects the channel nearest a requested wavelength; the mode trajectory
    itself is shared across channels. :meth:`moments` is monochromatic only
    (the joint band statistics live in :meth:`cross_band_moments`) and raises
    on a chromatic process -- select a channel's ``(e_nom, G)`` first.

    The PSD is the SCoOB-style knee form ``(1 + (f / knee)^2)^(slope / 2)``,
    evaluated on a log-spaced frequency grid straddling the knee. ``knee_hz``
    and ``slope`` are scalars shared by every mode (one shared grid) or
    ``(m,)`` for per-mode timescales (one grid per mode, so the modes drift on
    different timescales); ``per_mode_rms`` is in the same mode units as
    ``G``'s mode coordinate.

    Spectral lines carry the quadrature weight ``S(f_j) df_j`` (trapezoid
    widths, ``df_weighted=True``), which is what makes the synthesized process
    approximate the continuous PSD it names: the realized autocorrelation
    :meth:`autocorrelation` then converges to the PSD's transform, so a
    ``slope=-2`` (Lorentzian) process decorrelates as
    ``exp(-2 pi knee tau)`` -- exactly the ``tau`` that
    :meth:`from_decorrelation` was asked for. Weighting by ``S(f_j)`` alone
    (``df_weighted=False``, the pre-2026-07-25 behavior, kept so existing
    ensembles stay reproducible) instead synthesizes ``S(f) / f`` on a log
    grid, which decorrelates 2-3x too slowly. Equal-time statistics are
    identical either way: the weights are normalized to the per-mode rms, so
    only the temporal kernel changes.
    """

    e_nom: Array  # complex (y, x) or (w, y, x): nominal focal field
    G: Array  # complex (m, y, x) or (w, m, y, x): d(E_focal)/d(mode)
    per_mode_rms: Array  # float (m,): rms drift per mode
    knee_hz: Array  # float (m,): per-mode temporal PSD knee
    slope: Array  # float (m,): per-mode high-frequency PSD slope
    input_energy: Array
    normalization: Array
    pixel_scale_lod: float
    epoch_jd: float
    wavelengths_nm: Array | None
    coherent: bool = eqx.field(static=True)
    n_freq: int = eqx.field(static=True)
    decades_below: float = eqx.field(static=True)
    decades_above: float = eqx.field(static=True)
    per_mode_freq: bool = eqx.field(static=True)
    df_weighted: bool = eqx.field(static=True)

    def __init__(
        self,
        e_nom,
        G,
        per_mode_rms,
        knee_hz,
        *,
        input_energy,
        slope=-2.0,
        pixel_scale_lod=0.25,
        epoch_jd=J2000_JD,
        coherent=False,
        n_freq=64,
        decades_below=1.7,
        decades_above=2.3,
        df_weighted=True,
        wavelengths_nm=None,
    ):
        """Build the process parameter object.

        Args:
            e_nom: Complex nominal focal field, shape ``(y, x)`` -- or
                ``(w, y, x)`` with ``wavelengths_nm`` set.
            G: Complex sensitivity ``d(E_focal)/d(mode)``, shape
                ``(m, y, x)`` -- or ``(w, m, y, x)`` with ``wavelengths_nm``
                set.
            per_mode_rms: Per-mode rms drift, scalar (broadcast to every
                mode) or shape ``(m,)``.
            knee_hz: Temporal PSD knee frequency in Hz
                (``1 / (2 pi tau)`` for a decorrelation time ``tau``);
                scalar (shared by every mode) or shape ``(m,)`` for
                per-mode timescales.
            input_energy: Total energy of the field handed to the coronagraph
                train at the pre-coronagraph reference plane
                (``field.energy()``); the flux-fraction normalization
                ``input_energy / pixel_scale_lod**2`` is derived once at
                construction. A scalar, or one value per channel for a
                chromatic process.
            slope: High-frequency PSD power-law slope. Default -2; scalar
                or shape ``(m,)`` for per-mode slopes.
            pixel_scale_lod: Native pixel scale in lambda/D per pixel
                (shared by every channel: the maps live in lambda/D units,
                where the morphology is achromatic).
            epoch_jd: Julian Date mapping to ``time_s = 0``. Default J2000.
            coherent: Drawn fields include the pinning cross term.
            n_freq: Number of spectral-synthesis frequencies.
            decades_below: Frequency-grid extent below the knee (decades).
                Default 1.7, which puts the lowest line about 50x below the
                knee so lags out to several decorrelation times do not ring
                off the end of the grid.
            decades_above: Frequency-grid extent above the knee (decades).
            df_weighted: Weight each spectral line by ``S(f_j) df_j``
                (trapezoid quadrature) rather than ``S(f_j)`` alone, so the
                synthesized temporal kernel is the PSD's transform. Default
                ``True``; pass ``False`` (with ``decades_below=0.7``) to
                reproduce ensembles drawn before 2026-07-25.
            wavelengths_nm: Channel wavelengths, shape ``(w,)``, enabling
                the chromatic layout above and carried through :meth:`draw`
                into the returned :class:`AnalyticSpeckleField`. ``None``
                (default) for a monochromatic process.
        """
        self.e_nom = e_nom
        self.G = G
        m = G.shape[-3]
        self.per_mode_rms = jnp.broadcast_to(
            jnp.asarray(per_mode_rms, dtype=float), (m,)
        )
        self.knee_hz = jnp.broadcast_to(jnp.asarray(knee_hz, dtype=float), (m,))
        self.slope = jnp.broadcast_to(jnp.asarray(slope, dtype=float), (m,))
        # Per-mode frequency grids/PSDs are only needed when the modes drift
        # on different timescales or with different slopes; when they agree
        # (the scalar case) keep the single shared 1D grid, so a scalar-knee
        # process stays bit-identical to the pre-per-mode behavior.
        self.per_mode_freq = not (
            bool(jnp.all(self.knee_hz == self.knee_hz[0]))
            and bool(jnp.all(self.slope == self.slope[0]))
        )
        self.input_energy = jnp.asarray(input_energy, dtype=float)
        # Derived once, eagerly, from the stored primitives (see
        # AnalyticSpeckleField.__init__); tree_at on input_energy does not
        # re-derive it.
        self.normalization = self.input_energy / pixel_scale_lod**2
        self.pixel_scale_lod = pixel_scale_lod
        self.epoch_jd = epoch_jd
        self.wavelengths_nm = (
            None if wavelengths_nm is None else jnp.asarray(wavelengths_nm, dtype=float)
        )
        self.coherent = coherent
        self.n_freq = n_freq
        self.decades_below = decades_below
        self.decades_above = decades_above
        self.df_weighted = df_weighted

    def __check_init__(self):
        """Validate the per-mode parameters and (chromatic) ingredient layout."""
        m = (self.G.shape[-3],)
        for name, value in (
            ("per_mode_rms", self.per_mode_rms),
            ("knee_hz", self.knee_hz),
            ("slope", self.slope),
        ):
            if value.shape != m:
                raise ValueError(
                    f"{name} has shape {value.shape}; expected {m} to match "
                    "G's mode axis"
                )
        _check_chromatic_layout(
            self.e_nom, self.G, self.input_energy, self.wavelengths_nm
        )

    @classmethod
    def from_decorrelation(
        cls,
        e_nom,
        G,
        *,
        decorr_hours,
        total_rms,
        input_energy,
        **kwargs,
    ):
        """Parameterize by decorrelation time and a total WFE budget.

        The knee is ``1 / (2 pi tau)`` so the field decorrelates over roughly
        ``decorr_hours``, and the budget is split evenly over the modes
        (``per_mode_rms = total_rms / sqrt(m)``; rms adds in quadrature).
        """
        tau_s = decorr_hours * 3600.0
        m = G.shape[-3]
        return cls(
            e_nom,
            G,
            total_rms / jnp.sqrt(float(m)),
            1.0 / (2.0 * jnp.pi * tau_s),
            input_energy=input_energy,
            **kwargs,
        )

    def frequencies_hz(self) -> Array:
        """The log-spaced spectral-synthesis frequency grid.

        Shape ``(f,)`` when every mode shares one knee and slope (one grid
        straddling the common knee, the default), or ``(m, f)`` when the
        modes carry per-mode timescales (one grid straddling each mode's own
        knee).
        """
        if not self.per_mode_freq:
            log_knee = jnp.log10(self.knee_hz[0])
            return jnp.logspace(
                log_knee - self.decades_below,
                log_knee + self.decades_above,
                self.n_freq,
            )
        log_knee = jnp.log10(self.knee_hz)  # (m,)
        return jax.vmap(
            lambda lk: jnp.logspace(
                lk - self.decades_below, lk + self.decades_above, self.n_freq
            )
        )(log_knee)  # (m, f)

    def psd(self, frequencies_hz) -> Array:
        """Temporal PSD (knee form) evaluated at ``frequencies_hz``.

        Follows the grid rank: a ``(f,)`` grid uses the shared knee/slope
        and returns ``(f,)``; an ``(m, f)`` grid uses each mode's own
        knee/slope and returns ``(m, f)``.
        """
        f = jnp.asarray(frequencies_hz)
        if f.ndim >= 2:
            knee = self.knee_hz[:, None]
            slope = self.slope[:, None]
        else:
            knee = self.knee_hz[0]
            slope = self.slope[0]
        return (1.0 + (f / knee) ** 2) ** (slope / 2.0)

    def line_weights(self) -> Array:
        """Per-line spectral power weights, shape ``(m, f)``.

        The spectral-synthesis coefficient of line ``j`` carries mean power
        proportional to this weight, so it -- not :meth:`psd` -- is the
        quantity the draw and the realized autocorrelation are built from.
        With ``df_weighted=True`` the weight is the trapezoid quadrature
        element ``S(f_j) df_j`` of ``int S(f) df``; with ``df_weighted=False``
        it is the bare ``S(f_j)``, which on a log grid is instead a
        quadrature of ``int S(f) / f df``.

        Returned unnormalized (callers divide by the sum), and always
        broadcast to the full ``(m, f)`` mode axis.
        """
        f = self.frequencies_hz()
        weights = self.psd(f)
        if self.df_weighted:
            half = 0.5 * jnp.diff(f, axis=-1)
            zero = jnp.zeros_like(half[..., :1])
            df = jnp.concatenate([half, zero], axis=-1) + jnp.concatenate(
                [zero, half], axis=-1
            )
            weights = weights * df
        if weights.ndim == 1:
            weights = jnp.broadcast_to(weights, (self.G.shape[-3], self.n_freq))
        return weights

    def autocorrelation(self, lag_s) -> Array:
        """Realized modal autocorrelation ``rho(tau)`` of the synthesis.

        The spectral synthesis makes each mode's realized autocorrelation
        ``rho_k(tau) = sum_j w_kj cos(2 pi f_kj tau) / sum_j w_kj`` for the
        line weights ``w`` of :meth:`line_weights`. This is the generator's
        temporal kernel AS BUILT -- the quantity that sets decorrelation
        times, ADI floors, and any two-time statistic -- so it is worth
        checking against the continuous transform of the PSD it names rather
        than assuming they agree.

        For ``slope=-2`` (a Lorentzian PSD) the continuous target is the
        Ornstein-Uhlenbeck kernel ``exp(-2 pi knee tau)``, which the
        ``df_weighted=True`` synthesis reproduces over the lags its grid
        spans.

        Args:
            lag_s: Lag ``tau`` in seconds; scalar or array, broadcast against
                the mode axis.

        Returns:
            ``rho`` with shape ``(m,) + jnp.shape(lag_s)``.
        """
        tau = jnp.asarray(lag_s, dtype=float)
        f = self.frequencies_hz()
        if f.ndim == 1:
            f = jnp.broadcast_to(f, (self.G.shape[-3], self.n_freq))
        w = self.line_weights()
        flat = tau.reshape(-1)  # (t,)
        phase = 2.0 * jnp.pi * f[:, :, None] * flat[None, None, :]  # (m, f, t)
        rho = jnp.sum(w[:, :, None] * jnp.cos(phase), axis=1) / jnp.sum(
            w, axis=1, keepdims=True
        )
        return rho.reshape((f.shape[0], *tau.shape))

    def exposure_neff(self, exposure_s) -> Array:
        """Independent realizations a mode averages over in one exposure.

        A detector integrates, so the quantity a frame carries is the
        exposure-averaged coefficient, not an instantaneous one. Averaging a
        stationary process over a window of length ``T`` suppresses its
        variance by the factor

            1 / N_eff = (2 / T^2) int_0^T (T - tau) rho(tau) dtau,

        which for this synthesis' line spectrum is exactly
        ``sum_j w_kj sinc^2(f_kj T) / sum_j w_kj`` (``sinc(x) =
        sin(pi x) / (pi x)``) -- no quadrature needed, since each line
        integrates in closed form.

        Read it as a regime test rather than a correction factor. ``N_eff``
        near 1 means the exposure is short against the mode's decorrelation
        time, the field is effectively frozen, and one instantaneous
        ``realize`` IS the exposure. Large ``N_eff`` means the exposure
        averages over many independent speckle realizations, so a snapshot
        overstates the fluctuation by ``sqrt(N_eff)`` and
        ``realize_average`` (with enough sub-steps to resolve the fastest
        line) is the honest sampler.

        This is the exact ``N_eff`` of the synthesis AS BUILT, and it inherits
        the same limit :meth:`autocorrelation` documents: the finite line sum
        stops decaying past a few decorrelation times, so against the
        Lorentzian a ``slope=-2`` process names -- exact factor
        ``2(u - 1 + e^-u) / u^2`` at ``u = T / tau`` -- it agrees to a percent
        out to about one ``tau``, to ten percent at ten, and is about twice
        optimistic by a hundred. For exposures that long, generate an explicit
        trajectory whose kernel is exact at every lag rather than trusting
        either number.

        Args:
            exposure_s: Exposure length in seconds; scalar or array,
                broadcast against the mode axis.

        Returns:
            ``N_eff`` with shape ``(m,) + jnp.shape(exposure_s)``.
        """
        t_exp = jnp.asarray(exposure_s, dtype=float)
        f = self.frequencies_hz()
        if f.ndim == 1:
            f = jnp.broadcast_to(f, (self.G.shape[-3], self.n_freq))
        w = self.line_weights()
        flat = t_exp.reshape(-1)  # (t,)
        # jnp.sinc is the normalized sin(pi x) / (pi x), which is exactly the
        # window transform here, and is 1 at x = 0 without a special case.
        window = jnp.sinc(f[:, :, None] * flat[None, None, :]) ** 2  # (m, f, t)
        reduction = jnp.sum(w[:, :, None] * window, axis=1) / jnp.sum(
            w, axis=1, keepdims=True
        )
        return (1.0 / reduction).reshape((f.shape[0], *t_exp.shape))

    def draw(self, key, *, renormalize=True) -> AnalyticSpeckleField:
        """Sample one frozen realization of the process.

        Phases are uniform on ``[0, 2 pi)`` and independent realizations come
        from independent keys. ``renormalize`` selects the draw statistics:

        - ``True`` (default): PSD-shaped Gaussian amplitudes renormalized so
          each mode's synthesized ``eps_k(t)`` has EXACTLY ``per_mode_rms[k]``
          rms per draw (``Var[eps_k] = 0.5 sum_j a_kj^2``). The WFE budget is
          honored draw by draw, but the per-draw renormalization makes the
          fourth moment sub-Gaussian by order ``1 / N_eff`` (see
          :meth:`moments`).
        - ``False``: a circularly-symmetric complex-normal spectrum (Rayleigh
          amplitudes, PSD-weighted so the ENSEMBLE ``Var[eps_k] = rms_k^2``).
          The equal-time modal coefficients are then exactly Gaussian, so the
          ensemble is the exact improper-Gaussian process :meth:`moments`
          describes -- the oracle-exact draw.
        """
        m = self.G.shape[-3]
        f = self.frequencies_hz()
        # The line weight S(f_j) df_j, not the bare PSD ordinate, is what the
        # synthesis coefficients carry; see line_weights.
        power = self.line_weights()
        rms = self.per_mode_rms[:, None]
        if renormalize:
            k_amp, k_phase = jax.random.split(key)
            amp = jnp.sqrt(power) * jax.random.normal(k_amp, (m, self.n_freq))
            amp = amp * (rms / jnp.sqrt(0.5 * jnp.sum(amp**2, axis=1, keepdims=True)))
            phases = jax.random.uniform(
                k_phase, (m, self.n_freq), minval=0.0, maxval=2.0 * jnp.pi
            )
        else:
            # weights sum to 2 over frequency, the amplitude-squared convention
            # that gives ensemble Var[eps_k] = rms_k^2 (matches the correlated
            # spectral draw in the tiptilt family, here for diagonal covariance).
            weights = 2.0 * power / jnp.sum(power, axis=1, keepdims=True)
            k_real, k_imag = jax.random.split(key)
            z = (
                jax.random.normal(k_real, (m, self.n_freq))
                + 1j * jax.random.normal(k_imag, (m, self.n_freq))
            ) / jnp.sqrt(2.0)
            c = jnp.sqrt(weights) * rms * z
            amp = jnp.abs(c)
            phases = jnp.angle(c)
        return AnalyticSpeckleField(
            self.e_nom,
            self.G,
            amp,
            f,
            phases,
            input_energy=self.input_energy,
            pixel_scale_lod=self.pixel_scale_lod,
            epoch_jd=self.epoch_jd,
            coherent=self.coherent,
            wavelengths_nm=self.wavelengths_nm,
        )

    def renormalization_kurtosis(self, n_points=4000) -> Array:
        """Closed-form excess kurtosis of the ``renormalize=True`` modal draw.

        The renormalized draw of mode ``k`` is
        ``eps_k = rms_k sqrt(2) sum_j omega_kj cos(theta_kj)`` with ``omega``
        the unit-normalized vector of weight-shaped Gaussian amplitudes, so
        the per-draw rms constraint makes the marginal sub-Gaussian with
        excess kurtosis ``kappa_k = -(3/2) E[sum_j omega_kj^4]`` where, for
        line weights ``s_j`` (:meth:`line_weights`, any overall scale),

            E[sum_j omega^4] = 3 int_0^inf t prod_l (1 + 2 s_l t)^{-1/2}
                                 sum_j s_j^2 (1 + 2 s_j t)^{-2} dt,

        evaluated here as a 1D log-grid quadrature. Equal weights recover
        the uniform-sphere value ``-9 / (2 (F + 2))`` for ``F`` frequencies.
        The draw's odd moments vanish (the phases are symmetric), so this is
        the only non-Gaussian correction :meth:`moments` needs for a
        ``renormalize=True`` ensemble.

        Args:
            n_points: Quadrature points for the log-grid integral.

        Returns:
            Excess kurtosis per mode, shape ``(m,)`` (identical entries when
            every mode shares one frequency grid).
        """
        weights = self.line_weights()

        def kappa_of(s):
            s = s / jnp.max(s)
            t = jnp.logspace(-8.0, 10.0, n_points)
            st = s[:, None] * t[None, :]
            log_prod = -0.5 * jnp.sum(jnp.log1p(2.0 * st), axis=0)
            inner = jnp.sum(s[:, None] ** 2 / (1.0 + 2.0 * st) ** 2, axis=0)
            integrand = t**2 * jnp.exp(log_prod) * inner  # dt = t dlog(t)
            return -1.5 * 3.0 * jnp.trapezoid(integrand, jnp.log(t))

        m = self.G.shape[-3]
        if not self.per_mode_freq:
            # Every mode shares one grid, so one quadrature serves them all.
            return jnp.broadcast_to(kappa_of(weights[0]), (m,))
        return jax.lax.map(kappa_of, weights)

    def moments(self, *, mask=None, renormalized=False) -> "SpeckleMoments":
        """Closed-form ensemble moments of the delta ``realize`` returns.

        Implements the improper (non-circular) complex-Gaussian moment
        theorem for the diagonal modal covariance ``C_a = diag(rms_k^2)``:
        per focal-plane pixel, with ``Gamma = sum_k rms_k^2 |g_k|^2``,
        ``P = sum_k rms_k^2 g_k^2``, ``I_C = |e_nom|^2``,
        ``phi_C = angle(e_nom)`` and pinning-quadrature variance
        ``Var(X) = (Gamma + Re[P e^{-2 i phi_C}]) / 2``,

            E[delta]   = Gamma / norm
            Var[delta] = (4 I_C Var(X) + Gamma^2 + |P|^2) / norm^2

        with the heterodyne term ``4 I_C Var(X)`` dropped when
        ``coherent=False``. These are EXACT for a ``renormalize=False``
        ensemble (Gaussian modal coefficients). A ``renormalize=True``
        ensemble is sub-Gaussian; ``renormalized=True`` adds its closed-form
        correction ``sum_k kappa_k rms_k^4 |g_k|^4 / norm^2`` to the
        speckle-speckle term, with ``kappa_k`` from
        :meth:`renormalization_kurtosis` (no ensemble needed).

        Args:
            mask: Optional boolean dark-zone mask over the focal grid; when
                given the result also carries the annulus-averaged mean and
                variance (mask-weighted means of the maps).
            renormalized: Predict a ``renormalize=True`` ensemble (per-draw
                rms exact) instead of the exact-Gaussian ``renormalize=False``
                ensemble. Default ``False``.

        Returns:
            A :class:`SpeckleMoments` with the per-pixel maps (and the annulus
            reductions when ``mask`` is given).
        """
        if self.wavelengths_nm is not None:
            raise NotImplementedError(
                "moments is monochromatic; select a channel's (e_nom, G) first"
            )
        rms2 = self.per_mode_rms**2
        gamma = jnp.einsum("k,kyx->yx", rms2, jnp.abs(self.G) ** 2)
        p = jnp.einsum("k,kyx->yx", rms2, self.G**2)
        i_c = jnp.abs(self.e_nom) ** 2
        phi_c = jnp.angle(self.e_nom)
        var_x = 0.5 * (gamma + jnp.real(p * jnp.exp(-2j * phi_c)))
        norm = self.normalization
        mean_map = gamma / norm
        heterodyne = 4.0 * i_c * var_x if self.coherent else 0.0
        var_map = (heterodyne + gamma**2 + jnp.abs(p) ** 2) / norm**2
        if renormalized:
            kappa = self.renormalization_kurtosis()
            var_map = var_map + (
                jnp.einsum("k,kyx->yx", kappa * rms2**2, jnp.abs(self.G) ** 4) / norm**2
            )
        annulus_mean = annulus_var = None
        if mask is not None:
            w = jnp.asarray(mask, dtype=float)
            denom = jnp.sum(w)
            annulus_mean = jnp.sum(mean_map * w) / denom
            annulus_var = jnp.sum(var_map * w) / denom
        return SpeckleMoments(
            mean_map=mean_map,
            var_map=var_map,
            gamma_map=gamma,
            p_map=p,
            var_x_map=var_x,
            annulus_mean=annulus_mean,
            annulus_var=annulus_var,
        )

    def cross_band_moments(
        self, *, mask=None, tau_s=0.0, renormalized=False
    ) -> "CrossBandMoments":
        """Exact joint band-pair moments of a chromatic process.

        Because every wavelength channel is driven by the SAME real mode
        trajectory, the channels are one jointly improper complex Gaussian
        field; this returns its complete per-pixel second-order description
        (see :class:`CrossBandMoments` for the formulas). The result scales
        as ``w^2 * y * x`` in memory -- about 40 MB of complex kernels for
        six channels on a 256 x 256 grid; pass wide-band, fine grids through
        ``mask``-reduced or per-pair workflows if that grows too large.

        Args:
            mask: Optional boolean dark-zone mask over the focal grid; when
                given the result also carries the mask-averaged mean ``(w,)``
                and covariance ``(w, w)``.
            tau_s: Two-time lag in seconds. The returned ``cov_map`` is then
                ``Cov[delta_i(t), delta_j(t + tau_s)]``: both kernels carry
                the per-mode synthesis autocorrelation ``rho_k(tau_s)``
                (:meth:`autocorrelation`), which is exactly 1 at lag 0.
            renormalized: Add the closed-form sub-Gaussian correction for a
                ``renormalize=True`` ensemble (per-draw rms exact), the
                cross-band mirror of ``moments(renormalized=True)``:
                ``sum_k kappa_k rms_k^4 |G_k(i)|^2 |G_k(j)|^2``. Equal-time
                only; raises with a nonzero ``tau_s``. Default ``False``
                (the exact-Gaussian ``renormalize=False`` ensemble).

        Returns:
            A :class:`CrossBandMoments` with the per-pixel maps (and the
            annulus reductions when ``mask`` is given).

        Raises:
            ValueError: On a monochromatic process (use :meth:`moments`).
            NotImplementedError: For ``renormalized=True`` with ``tau_s != 0``.
        """
        if self.wavelengths_nm is None:
            raise ValueError(
                "cross_band_moments needs a chromatic process (wavelengths_nm "
                "set); use moments for a monochromatic one"
            )
        if renormalized and tau_s != 0.0:
            raise NotImplementedError(
                "the renormalization correction is equal-time only; pass "
                "tau_s=0.0 or renormalized=False"
            )
        rms2 = self.per_mode_rms**2
        damped = rms2 * self.autocorrelation(tau_s)
        gamma = jnp.einsum("m,imyx,jmyx->ijyx", damped, self.G, jnp.conj(self.G))
        p = jnp.einsum("m,imyx,jmyx->ijyx", damped, self.G, self.G)
        w = self.wavelengths_nm.shape[0]
        norms = jnp.broadcast_to(self.normalization, (w,))
        denom = norms[:, None, None, None] * norms[None, :, None, None]
        cov = jnp.abs(gamma) ** 2 + jnp.abs(p) ** 2
        if self.coherent:
            a_i = jnp.conj(self.e_nom)[:, None]
            a_j = self.e_nom[None, :]
            cov = cov + 2.0 * jnp.real(a_i * a_j * gamma + a_i * jnp.conj(a_j) * p)
        if renormalized:
            cov = cov + jnp.einsum(
                "m,imyx,jmyx->ijyx",
                self.renormalization_kurtosis() * rms2**2,
                jnp.abs(self.G) ** 2,
                jnp.abs(self.G) ** 2,
            )
        cov = cov / denom
        mean = (
            jnp.einsum("m,wmyx->wyx", rms2, jnp.abs(self.G) ** 2) / norms[:, None, None]
        )
        annulus_mean = annulus_cov = None
        if mask is not None:
            weights = jnp.asarray(mask, dtype=float)
            total = jnp.sum(weights)
            annulus_mean = jnp.einsum("wyx,yx->w", mean, weights) / total
            annulus_cov = jnp.einsum("ijyx,yx->ij", cov, weights) / total
        return CrossBandMoments(
            mean_map=mean,
            cov_map=cov,
            gamma_map=gamma,
            p_map=p,
            tau_s=float(tau_s),
            annulus_mean=annulus_mean,
            annulus_cov=annulus_cov,
        )
