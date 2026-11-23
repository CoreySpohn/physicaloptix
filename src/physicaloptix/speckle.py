"""Spectral-synthesis speckles on the shared linear per-pixel field law.

FocalResponse stores the density amplitudes and converts them to per-pixel
amplitudes for the shared excess intensity law. Random spectral amplitudes
and phases are frozen at construction.
"""

from dataclasses import dataclass, replace
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from hwoutils.constants import d2s
from jaxtyping import Array
from optixstuff import FocalResponse, LinearSpeckleField
from optixstuff.grids import PixelGrid
from optixstuff.spectral import trapezoid_weights

from physicaloptix.transforms.fraunhofer import Fraunhofer


def ou_exposure_neff(timescales_s, exposure_s):
    """Independent realizations an OU mode averages over in one exposure.

    A detector integrates, so averaging a mode over a window of length ``T``
    suppresses its variance by ``1 / N_eff``. For the exponential kernel this
    integrates in closed form,

        1 / N_eff = 2 (u - 1 + e^-u) / u^2,     u = T / tau,

    with no quadrature, no ensemble, and no spectral grid -- and unlike the
    spectral synthesis' version it is exact at EVERY exposure length, because
    the kernel it integrates is exact at every lag. It runs from 1 for a
    frozen field (``T`` far below ``tau``) to the familiar ``T / (2 tau)``
    once the exposure spans many decorrelation times.

    Use it to decide whether an exposure needs
    ``LinearSpeckleField.realize_average`` at all: at ``N_eff`` near 1 an
    instantaneous realization IS the exposure, and at large ``N_eff`` a
    snapshot overstates the speckle fluctuation by ``sqrt(N_eff)``.

    Args:
        timescales_s: Per-mode decorrelation timescale in seconds, a scalar
            or shape ``(m,)``.
        exposure_s: Exposure length in seconds, a scalar or an array
            broadcast against the mode axis.

    Returns:
        ``N_eff`` with shape ``(m,) + jnp.shape(exposure_s)``, matching
        ``physicaloptix.SpeckleProcess.exposure_neff``. A zero-length exposure
        gives exactly 1.

    Raises:
        ValueError: If a timescale is not strictly positive.
    """
    tau = np.atleast_1d(np.asarray(timescales_s, dtype=float))
    if not np.all(tau > 0.0):
        raise ValueError("timescales_s must be positive")
    t_exp = jnp.asarray(exposure_s, dtype=float)
    u = t_exp.reshape(-1)[None, :] / jnp.asarray(tau)[:, None]  # (m, t)
    # Two branches around u ~ 1e-4. Above it, evaluate directly, writing
    # u - 1 + exp(-u) as u + expm1(-u) so the leading cancellation is exact.
    # Below it even that loses the answer: the numerator vanishes as u^2 / 2,
    # so subtracting two O(u) terms leaves only a few digits. There the
    # Maclaurin series is the accurate form, and its own truncation error is
    # O(u^4). Both branches are evaluated, so each must stay finite to keep
    # the gradient clean -- hence the guarded arguments.
    small = u < 1e-4
    safe = jnp.where(small, 1.0, u)
    direct = 2.0 * (safe + jnp.expm1(-safe)) / safe**2
    tiny = jnp.where(small, u, 0.0)
    series = 1.0 - tiny / 3.0 + tiny**2 / 12.0 - tiny**3 / 60.0
    reduction = jnp.where(small, series, direct)
    return (1.0 / reduction).reshape((tau.size, *t_exp.shape))


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
        grid_out: The focal ``PixelGrid`` the peak is sampled on.

    Returns:
        The peak intensity density as a float.
    """
    transform = Fraunhofer(field.grid, grid_out)
    return float(jnp.max(transform.forward(field).intensity().data))


def _check_wavelength_layout(response, wavelengths_nm):
    """Match the response's wavelength axis to the interpolation nodes."""
    if wavelengths_nm is None:
        if response.e_nom.ndim != 2:
            raise ValueError(
                "monochromatic ingredients require a monochromatic response"
            )
    elif wavelengths_nm.ndim != 1 or response.e_nom.shape[:-2] != wavelengths_nm.shape:
        raise ValueError("chromatic ingredients must match the wavelength nodes")


class AnalyticSpeckleField(LinearSpeckleField):
    """Frozen spectral synthesis using the shared per-pixel field law.

    A pure halo uses a zero nominal response. The inherited chromatic
    model holds e_nom fixed and scales G by reference_wavelength_nm /
    wavelength_nm.

    Attributes:
        amplitudes: Spectral-synthesis amplitudes, shape (m, f), in mode-coefficient
            units.
        frequencies_hz: Synthesis frequencies in Hz, shape (f,) shared by all modes or
            (m, f) per mode.
        phases: Spectral-synthesis phases in radians, shape (m, f).
    """

    amplitudes: Array
    frequencies_hz: Array
    phases: Array

    def coefficients(self, time_d):
        """Synthesize ``(*time_shape, n_modes)`` coefficients at absolute days.

        Args:
            time_d: Absolute times in days; any shape.

        Returns:
            Mode coefficients of shape (*time_shape, n_modes), in the mode units of the
            response.
        """
        t = (jnp.asarray(time_d, dtype=float) - self.epoch_d) * d2s
        phase = 2.0 * jnp.pi * self.frequencies_hz * t[..., None, None] + self.phases
        return jnp.sum(self.amplitudes * jnp.cos(phase), axis=-1)

    def peak_contrast(self, *, telescope_peak, wavelength_nm, time_d=0.0):
        """Peak-referenced delta; telescope_peak is an intensity density.

        Args:
            telescope_peak: Peak intensity density of the unocculted telescope PSF on
                the same grid, as returned by telescope_peak.
            wavelength_nm: Wavelength in nanometers.
            time_d: Absolute time in days.

        Returns:
            Delta map referenced to the telescope peak (dimensionless contrast), on the
            grid of the response.
        """
        delta = self.realize(wavelength_nm=wavelength_nm, time_d=time_d)
        return (
            delta.data
            * self.response.input_energy
            / (telescope_peak * self.grid.pixel_area)
        )


def _interpolate_nodes(wavelength_nm, wavelengths_nm, values):
    """Linear complex-amplitude interpolation, clamped to endpoint tables."""
    if wavelengths_nm.size == 1:
        return values[0]
    index = jnp.clip(
        jnp.searchsorted(wavelengths_nm, wavelength_nm), 1, wavelengths_nm.size - 1
    )
    lower, upper = wavelengths_nm[index - 1], wavelengths_nm[index]
    weight = jnp.clip((wavelength_nm - lower) / (upper - lower), 0.0, 1.0)
    return (1.0 - weight) * values[index - 1] + weight * values[index]


class ChromaticAnalyticSpeckleField(AnalyticSpeckleField):
    """Propagated chromatic amplitudes with the shared intensity law.

    Stored node amplitudes are densities. Each query linearly interpolates
    the complex e_nom and G and the input energy (endpoints are clamped).
    This replaces reference/lambda scaling, retaining propagated morphology,
    dispersion and nominal leakage exactly at stored nodes. Between nodes
    it is an amplitude interpolation approximation, not new propagation.
    Tables share one native lod grid/frame; angular dilation remains the
    consumer's responsibility. broadened and realize_average are inherited.

    Attributes:
        wavelengths_nm: Node wavelengths in nanometers, shape (w,), matching the leading
            axis of the response tables.
    """

    wavelengths_nm: Array
    _response_ndim: ClassVar[int] = 3

    def __check_init__(self):
        """Validate the propagated node table."""
        _check_wavelength_layout(self.response, self.wavelengths_nm)
        if self.wavelengths_nm.size < 1:
            raise ValueError("chromatic fields need at least one wavelength node")
        if not isinstance(self.wavelengths_nm, jax.core.Tracer):
            nodes = np.asarray(self.wavelengths_nm)
            if not (
                np.all(np.isfinite(nodes))
                and np.all(nodes > 0)
                and np.all(np.diff(nodes) > 0)
            ):
                raise ValueError(
                    "wavelength nodes must be positive and strictly increasing"
                )

    def spectral_breaks_nm(self) -> np.ndarray:
        """Nodes of the interpolated complex-amplitude and energy tables [nm].

        Returns:
            Node wavelengths in nanometers of the interpolated tables.
        """
        return np.asarray(self.wavelengths_nm)

    def at_wavelength(self, wavelength_nm):
        """A mono field whose shared law equals the interpolated node law.

        The sensitivity is precompensated for the parent's phase scaling;
        no phase factor is applied twice. The returned e_nom is also the
        correct nominal floor for visualization at this wavelength.

        Args:
            wavelength_nm: Wavelength in nanometers at which the tables are
                interpolated.

        Returns:
            Monochromatic AnalyticSpeckleField whose response is the interpolated node
            response at that wavelength.
        """
        response = self.response
        e = _interpolate_nodes(wavelength_nm, self.wavelengths_nm, response.e_nom)
        g = _interpolate_nodes(wavelength_nm, self.wavelengths_nm, response.G)
        energies = jnp.broadcast_to(response.input_energy, self.wavelengths_nm.shape)
        energy = _interpolate_nodes(wavelength_nm, self.wavelengths_nm, energies)
        return AnalyticSpeckleField(
            response=FocalResponse(
                e,
                g * (wavelength_nm / response.reference_wavelength_nm),
                self.grid,
                energy,
                response.reference_wavelength_nm,
            ),
            epoch_d=self.epoch_d,
            amplitudes=self.amplitudes,
            frequencies_hz=self.frequencies_hz,
            phases=self.phases,
        )

    def realize(self, *, wavelength_nm, time_d=0.0):
        """Evaluate every requested node through LinearSpeckleField.realize.

        Args:
            wavelength_nm: Wavelength in nanometers: a scalar, or an array whose axes
                become leading axes of the result.
            time_d: Absolute time in days.

        Returns:
            Flux-fraction delta image on the response grid, with the wavelength axes
            leading.
        """
        wavelengths = jnp.asarray(wavelength_nm)

        def node(wavelength):
            return self.at_wavelength(wavelength).realize(
                wavelength_nm=wavelength, time_d=time_d
            )

        if wavelengths.ndim == 0:
            return node(wavelengths)
        images = jax.vmap(node)(wavelengths.reshape(-1))
        axes = (
            ("band", "quadrature")
            if wavelengths.ndim == 2
            else ("wavelength",)
            if wavelengths.ndim == 1
            else tuple(f"wavelength_{i}" for i in range(wavelengths.ndim))
        )
        return replace(
            images,
            data=images.data.reshape((*wavelengths.shape, *images.data.shape[1:])),
            leading_axes=axes + images.leading_axes,
        )

    def peak_contrast(self, *, telescope_peak, wavelength_nm, time_d=0.0):
        """Peak contrast at a scalar wavelength using its interpolated energy.

        Args:
            telescope_peak: Peak intensity density of the unocculted telescope PSF on
                the same grid.
            wavelength_nm: Scalar wavelength in nanometers.
            time_d: Absolute time in days.

        Returns:
            Delta map referenced to the telescope peak (dimensionless contrast), on the
            grid of the response.
        """
        return self.at_wavelength(jnp.asarray(wavelength_nm)).peak_contrast(
            telescope_peak=telescope_peak, wavelength_nm=wavelength_nm, time_d=time_d
        )


class SpeckleMoments(eqx.Module):
    """Closed-form ensemble moments of a :class:`SpeckleProcess` contrast field.

    The frozen output of :meth:`SpeckleProcess.moments`: the per-pixel mean and
    variance maps of the flux-fraction delta (the units :meth:`realize`
    returns), the raw kernels they are built from (``Gamma``, the complex
    pseudo-covariance ``P``, and the pinning-quadrature variance ``Var(X)``,
    in energy-per-pixel units), and -- when a dark-zone mask is supplied
    -- the mask-averaged mean and variance.

    Attributes:
        grid: Focal grid of the maps.
        mean_map: E[delta] per pixel, in flux-fraction units.
        var_map: Var[delta] per pixel, in flux-fraction units squared.
        gamma_pixel_map: Gamma = sum_k rms_k^2 |g_k|^2 per pixel, real and nonnegative.
        p_pixel_map: P = sum_k rms_k^2 g_k^2 per pixel, the complex pseudo-covariance.
        var_x_pixel_map: Var(X) per pixel, the pinning-quadrature variance.
        annulus_mean: Mask-averaged E[delta], or None when no mask was given.
        annulus_var: Mask-averaged Var[delta], or None when no mask was given.
    """

    grid: PixelGrid
    mean_map: Array  # E[delta] per pixel (flux-fraction units)
    var_map: Array  # Var[delta] per pixel (flux-fraction units^2)
    gamma_pixel_map: Array  # Gamma = sum_k rms_k^2 |g_k|^2 (real, >= 0)
    p_pixel_map: Array  # P = sum_k rms_k^2 g_k^2 (complex pseudo-covariance)
    var_x_pixel_map: Array  # Var(X), the pinning-quadrature variance
    annulus_mean: Array | None = None  # mask-averaged E[delta]
    annulus_var: Array | None = None  # mask-averaged Var[delta]

    def impropriety(self) -> Array:
        """Degree of impropriety ``eta = |P| / Gamma`` per pixel, ``(y, x)``.

        The single-band counterpart of :meth:`CrossBandMoments.impropriety`:
        how far the drifting field is from circular at each pixel, bounded in
        ``[0, 1]`` by Cauchy-Schwarz on the modal sum.

        The two poles are worth naming, because they bracket every real case:

        - ``eta = 0`` is the classical circular limit, where the
          modified-Rician intensity law is exactly right. It is reached when
          the drift is statistically homogeneous (translation-invariant),
          because translation invariance pairs each spatial frequency's two
          quadratures and cancels ``P`` term by term.
        - ``eta = 1`` is the opposite pole, where the field is confined to a
          line in the complex plane rather than filling an ellipse. A single
          real-coefficient mode does this exactly, as does any
          definite-parity family through a parity-preserving train.

        A large ``eta`` is therefore not automatically a statement about the
        optics: a mode basis carrying one quadrature per spatial frequency
        forces ``eta`` upward by construction, whatever the hardware does.
        Read this together with how the basis was built.

        Pixels with no modal response (``Gamma == 0``) return ``0.0`` rather
        than NaN; the doubled ``where`` also keeps reverse-mode gradients
        finite there.

        Returns:
            Degree of impropriety per pixel, (y, x), dimensionless in [0, 1].
        """
        live = self.gamma_pixel_map > 0
        safe = jnp.where(live, self.gamma_pixel_map, 1.0)
        return jnp.where(live, jnp.abs(self.p_pixel_map) / safe, 0.0)


@dataclass(frozen=True)
class Photometry:
    """The photometric contract of a :class:`SpeckleProcess`, with guards.

    A plain frozen report, deliberately not a PyTree: it is meant to be
    printed and eyeballed on the host, not carried through a transformation.

    Stores input energy and the actual grid. contrast_scale, when a telescope
    peak density is supplied, is input_energy / (peak * grid.pixel_area).
    No cached density divisor is carried into the field law.

    Attributes:
        input_energy: Input energy, a scalar or (w,) for a chromatic process.
        grid: Focal grid of the process.
        e_nom_flux_fraction: Energy in e_nom as a fraction of input_energy.
        contrast_scale: input_energy divided by the peak-pixel energy, or None when no
            telescope peak was supplied.
        warnings: Messages describing photometric inconsistencies; empty when there are
            none.
    """

    input_energy: Array  # scalar, or (w,) for a chromatic process
    grid: PixelGrid
    e_nom_flux_fraction: Array  # energy in e_nom, as a fraction of input_energy
    contrast_scale: Array | None  # input_energy / peak-pixel energy
    warnings: tuple[str, ...]

    def __str__(self) -> str:
        """One field per line, with any warnings appended.

        Returns:
            One line per field, with any warnings appended.
        """
        lines = [
            f"input_energy        {np.asarray(self.input_energy)}",
            f"grid                {self.grid}",
            f"e_nom flux fraction {np.asarray(self.e_nom_flux_fraction)}",
        ]
        if self.contrast_scale is not None:
            lines.append(f"contrast_scale      {np.asarray(self.contrast_scale)}")
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


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
    moment theorem, exact and noncentral::

        N_i N_j Cov[delta_i, delta_j] = |Gamma_ij|^2 + |P_ij|^2
            + 2 Re[conj(A_i) A_j Gamma_ij + conj(A_i) conj(A_j) P_ij]

    with ``A_i = e_nom[i] * sqrt(grid.pixel_area)``, ``N_i = input_energy[i]``,
    and the heterodyne bracket present only for a
    ``coherent`` process. ``tau_s`` is the lag both kernels are damped by
    (``rho_k`` is the synthesis autocorrelation); the equal-time container has
    ``tau_s = 0.0`` and its band-diagonal reproduces the per-channel
    :meth:`SpeckleProcess.moments` exactly. ``mean_map`` is time-independent.

    Attributes:
        grid: Focal grid of the maps.
        mean_map: E[delta] per channel, (w, y, x), in flux-fraction units.
        cov_map: Cov[delta_i(t), delta_j(t + tau_s)], (w, w, y, x), in flux-fraction
            units squared.
        gamma_pixel_map: Complex Gamma_ij, (w, w, y, x).
        p_pixel_map: Complex pseudo-covariance P_ij, (w, w, y, x).
        tau_s: Lag in seconds by which both kernels are damped; zero for the equal-time
            container.
        annulus_mean: Mask-averaged mean, (w,), or None when no mask was given.
        annulus_cov: Mask-averaged covariance, (w, w), or None when no mask was given.
    """

    grid: PixelGrid
    mean_map: Array  # (w, y, x): E[delta] per channel (flux-fraction units)
    cov_map: Array  # (w, w, y, x): Cov[delta_i(t), delta_j(t + tau_s)]
    gamma_pixel_map: Array  # (w, w, y, x) complex: Gamma_ij
    p_pixel_map: Array  # (w, w, y, x) complex: P_ij (pseudo-covariance)
    tau_s: float = eqx.field(static=True)
    annulus_mean: Array | None = None  # (w,): mask-averaged mean
    annulus_cov: Array | None = None  # (w, w): mask-averaged covariance

    def _require_equal_time(self, name):
        """Raise unless this is an equal-time (tau_s == 0) container."""
        if self.tau_s != 0.0:
            raise ValueError(
                f"{name} is defined on the equal-time container; rebuild with tau_s=0.0"
            )

    def correlation(self) -> Array:
        """Band-pair intensity correlation ``rho_ij`` per pixel, ``(w, w, y, x)``.

        ``cov_map`` normalized by the per-band standard deviations; the
        diagonal is exactly 1. Equal-time only (a lagged container's diagonal
        is not the variance).

        Returns:
            Band-pair intensity correlation, (w, w, y, x), dimensionless.
        """
        self._require_equal_time("correlation")
        d = jnp.sqrt(jnp.einsum("iiyx->iyx", self.cov_map))
        return self.cov_map / (d[:, None] * d[None, :])

    def n_eff(self) -> Array:
        """Effective independent spectral channels per pixel, ``(y, x)``.

        The participation ratio ``(tr S)^2 / tr(S^2)`` of the band-band
        intensity covariance at each pixel: 1 when the channels are perfectly
        correlated (the achromatic limit), the channel count when they are
        independent. Bounded by the covariance rank, which the shared mode
        vector caps at ``m (m + 3) / 2`` regardless of how finely the band is
        sampled. Equal-time only.

        Returns:
            Effective number of independent spectral channels per pixel, (y, x).
        """
        self._require_equal_time("n_eff")
        trace = jnp.einsum("iiyx->yx", self.cov_map)
        return trace**2 / jnp.einsum("ijyx,jiyx->yx", self.cov_map, self.cov_map)

    def impropriety(self) -> Array:
        """Cross-band degree of impropriety ``|P_ij| / sqrt(Gamma_ii Gamma_jj)``.

        The band-pair generalization of the single-band ``|P| / Gamma``; it is
        bounded by 1 (Cauchy-Schwarz) and vanishes for statistically
        homogeneous (translation-invariant) residuals, whose quadrature pairing
        cancels the pseudo-covariance at every pair of wavelengths. Shape
        ``(w, w, y, x)``. Equal-time only.

        Returns:
            Cross-band degree of impropriety, (w, w, y, x), dimensionless.
        """
        self._require_equal_time("impropriety")
        g = jnp.real(jnp.einsum("iiyx->iyx", self.gamma_pixel_map))
        return jnp.abs(self.p_pixel_map) / jnp.sqrt(g[:, None] * g[None, :])


class SpeckleProcess(eqx.Module):
    """One parameter set for the linear speckle process; views derive from it.

    Holds a density-amplitude FocalResponse together with the
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
    wavelengths through to a :class:`ChromaticAnalyticSpeckleField`, which
    interpolates complex amplitudes at each requested node. The mode
    trajectory is shared across channels. :meth:`moments` is monochromatic only
    (the joint band statistics live in :meth:`cross_band_moments`) and raises
    on a chromatic process -- select a channel's ``(e_nom, G)`` first.

    The PSD is the SCoOB-style knee form ``(1 + (f / knee)^2)^(slope / 2)``,
    evaluated on a log-spaced frequency grid straddling the knee. ``knee_hz``
    and ``slope`` are scalars shared by every mode (one shared grid) or
    ``(m,)`` for per-mode timescales (one grid per mode, so the modes drift on
    different timescales); ``per_mode_rms`` is in the same mode units as
    ``G``'s mode coordinate.

    Spectral lines carry the quadrature weight ``S(f_j) df_j`` with
    trapezoid widths, so the synthesized autocorrelation approximates the
    continuous PSD's transform. A ``slope=-2`` (Lorentzian) process
    decorrelates as ``exp(-2 pi knee tau)``, matching the time requested
    through :meth:`from_decorrelation`.

    Attributes:
        response: Complex density amplitudes and their photometric reference.
        per_mode_rms: RMS drift per mode, shape (m,), in the mode units of the response.
        knee_hz: Temporal PSD knee frequency per mode in Hz, shape (m,).
        slope: High-frequency PSD slope per mode, shape (m,).
        epoch_d: Days from the scene epoch at which elapsed time is zero.
        wavelengths_nm: Channel wavelengths in nanometers, shape (w,), or None for a
            monochromatic process.
        coherent: Whether drawn fields include the pinning cross term.
        n_freq: Number of spectral-synthesis frequencies.
        decades_below: Frequency-grid extent below the knee, in decades.
        decades_above: Frequency-grid extent above the knee, in decades.
        per_mode_freq: True when modes have different knees or slopes, so each has its
            own frequency grid.
    """

    response: FocalResponse
    per_mode_rms: Array  # float (m,): rms drift per mode
    knee_hz: Array  # float (m,): per-mode temporal PSD knee
    slope: Array  # float (m,): per-mode high-frequency PSD slope
    epoch_d: float = eqx.field(static=True)
    wavelengths_nm: Array | None
    coherent: bool = eqx.field(static=True)
    n_freq: int = eqx.field(static=True)
    decades_below: float = eqx.field(static=True)
    decades_above: float = eqx.field(static=True)
    per_mode_freq: bool = eqx.field(static=True)

    def __init__(
        self,
        response: FocalResponse,
        per_mode_rms,
        knee_hz,
        *,
        slope=-2.0,
        epoch_d=0.0,
        coherent=False,
        n_freq=64,
        decades_below=1.7,
        decades_above=2.3,
        wavelengths_nm=None,
    ):
        """Build the process parameter object.

        Args:
            response: Complex density amplitudes and their photometric reference.
            per_mode_rms: Per-mode rms drift, scalar (broadcast to every
                mode) or shape ``(m,)``.
            knee_hz: Temporal PSD knee frequency in Hz
                (``1 / (2 pi tau)`` for a decorrelation time ``tau``);
                scalar (shared by every mode) or shape ``(m,)`` for
                per-mode timescales.
            slope: High-frequency PSD power-law slope. Default -2; scalar
                or shape ``(m,)`` for per-mode slopes.
            epoch_d: Days from the scene epoch at which elapsed time is zero.
                Default 0.0.
            coherent: Drawn fields include the pinning cross term.
            n_freq: Number of spectral-synthesis frequencies.
            decades_below: Frequency-grid extent below the knee (decades).
                Default 1.7, which puts the lowest line about 50x below the
                knee so lags out to several decorrelation times do not ring
                off the end of the grid.
            decades_above: Frequency-grid extent above the knee (decades).
            wavelengths_nm: Channel wavelengths, shape ``(w,)``, enabling
                the chromatic layout above and carried through :meth:`draw`
                into the returned :class:`AnalyticSpeckleField`. ``None``
                (default) for a monochromatic process.
        """
        self.response = response
        m = response.G.shape[-3]
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
        self.epoch_d = float(epoch_d)
        self.wavelengths_nm = (
            None if wavelengths_nm is None else jnp.asarray(wavelengths_nm, dtype=float)
        )
        self.coherent = coherent
        self.n_freq = n_freq
        self.decades_below = decades_below
        self.decades_above = decades_above

    def __check_init__(self):
        """Validate mode parameters, native geometry and ingredient layout."""
        if self.response.grid.unit != "lod":
            raise ValueError("speckle ingredients require their native lod grid")
        m = (self.response.G.shape[-3],)
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
        _check_wavelength_layout(self.response, self.wavelengths_nm)

    @classmethod
    def from_decorrelation(
        cls,
        response: FocalResponse,
        *,
        decorr_hours,
        total_rms,
        **kwargs,
    ):
        """Parameterize by decorrelation time and a total WFE budget.

        The knee is ``1 / (2 pi tau)`` so the field decorrelates over roughly
        ``decorr_hours``, and the budget is split evenly over the modes
        (``per_mode_rms = total_rms / sqrt(m)``; rms adds in quadrature).

        Args:
            response: Complex density amplitudes and their photometric reference.
            decorr_hours: Decorrelation time in hours.
            total_rms: Total WFE budget, split evenly over the modes, in the mode units
                of the response.
            **kwargs: Forwarded to the SpeckleProcess constructor.

        Returns:
            The process parameter object.
        """
        tau_s = decorr_hours * 3600.0
        m = response.G.shape[-3]
        return cls(
            response,
            total_rms / jnp.sqrt(float(m)),
            1.0 / (2.0 * jnp.pi * tau_s),
            **kwargs,
        )

    def frequencies_hz(self) -> Array:
        """The log-spaced spectral-synthesis frequency grid.

        Shape ``(f,)`` when every mode shares one knee and slope (one grid
        straddling the common knee, the default), or ``(m, f)`` when the
        modes carry per-mode timescales (one grid straddling each mode's own
        knee).

        Returns:
            Log-spaced synthesis frequencies in Hz, shape (f,) or (m, f).
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

        Args:
            frequencies_hz: Frequencies in Hz, shape (f,) or (m, f).

        Returns:
            Temporal PSD values at those frequencies, with the shape of the input.
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
        The weight is the trapezoid quadrature element ``S(f_j) df_j``
        of ``int S(f) df``.

        Returned unnormalized (callers divide by the sum), and always
        broadcast to the full ``(m, f)`` mode axis.

        Returns:
            Unnormalized per-line spectral power weights, shape (m, f).
        """
        f = self.frequencies_hz()
        weights = self.psd(f) * trapezoid_weights(f)
        if weights.ndim == 1:
            weights = jnp.broadcast_to(
                weights, (self.response.G.shape[-3], self.n_freq)
            )
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
        synthesis reproduces over the lags its grid
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
            f = jnp.broadcast_to(f, (self.response.G.shape[-3], self.n_freq))
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
            f = jnp.broadcast_to(f, (self.response.G.shape[-3], self.n_freq))
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

        Args:
            key: JAX random key; independent keys give independent realizations.
            renormalize: True renormalizes each mode to exactly its per-mode rms per
                draw; False draws the exact improper-Gaussian ensemble.

        Returns:
            Frozen realization of the process; a ChromaticAnalyticSpeckleField when
            wavelengths_nm is set.
        """
        m = self.response.G.shape[-3]
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
        response = self.response
        if not self.coherent:
            response = replace(response, e_nom=jnp.zeros_like(response.e_nom))
        common = dict(
            response=response,
            epoch_d=self.epoch_d,
            amplitudes=amp,
            frequencies_hz=f,
            phases=phases,
        )
        if self.wavelengths_nm is None:
            return AnalyticSpeckleField(**common)
        return ChromaticAnalyticSpeckleField(
            wavelengths_nm=self.wavelengths_nm, **common
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

        m = self.response.G.shape[-3]
        if not self.per_mode_freq:
            # Every mode shares one grid, so one quadrature serves them all.
            return jnp.broadcast_to(kappa_of(weights[0]), (m,))
        return jax.lax.map(kappa_of, weights)

    def photometry(self, *, telescope_peak=None, unity_tol=1e-6) -> "Photometry":
        """Report the photometric primitives, the derived divisor, and guards.

        A host-side diagnostic, **not** jit-compatible: it inspects values to
        decide whether to warn. Call it once when wiring a process up, not on
        a hot path.

        The failure mode this exists to catch is silent. ``input_energy`` and
        a peak intensity are both "a number describing how bright the star
        is", they differ by a factor of order ten to a hundred, and a process
        built with the wrong one runs perfectly happily while every contrast
        it reports is wrong by that factor. Two checks are applied:

        - **Energy conservation** (rigorous). The energy in ``e_nom``, which
          is ``sum |e_nom|^2 * grid.pixel_area``, cannot exceed the energy
          that entered. A ratio above 1 proves ``input_energy`` is not a total
          energy. For a coronagraph the honest value is far below 1, since
          the stop discards most of the starlight and the stored grid is a
          crop of the focal plane.
        - **Degenerate contrast scale** (heuristic, needs ``telescope_peak``).
          If ``input_energy / (telescope_peak * grid.pixel_area)`` is exactly 1, then
          ``input_energy`` was almost certainly obtained as
          ``telescope_peak * grid.pixel_area`` rather than measured. That
          reproduces peak-referenced contrast out of :meth:`realize`, which
          silently breaks any consumer expecting a flux fraction.

        Args:
            telescope_peak: Peak intensity density of the unocculted
                telescope PSF on this field's grid, from
                :func:`telescope_peak`. Enables ``contrast_scale`` and the
                second guard.
            unity_tol: Relative tolerance for the degenerate-scale check.

        Returns:
            A :class:`Photometry` report; ``warnings`` is empty when both
            checks pass.
        """
        du2 = self.response.grid.pixel_area
        e_energy = jnp.sum(jnp.abs(self.response.e_nom) ** 2, axis=(-2, -1)) * du2
        flux_fraction = e_energy / self.response.input_energy
        contrast_scale = (
            None
            if telescope_peak is None
            else self.response.input_energy
            / (self.response.grid.pixel_area * telescope_peak)
        )

        warnings: list[str] = []
        worst = float(jnp.max(flux_fraction))
        if worst > 1.0:
            warnings.append(
                f"the nominal field already carries {worst:.3g}x the stated "
                "input_energy, which is not physically possible: input_energy "
                "must be the TOTAL energy of the field entering the "
                "coronagraph (Linearization.input_energy), not a peak intensity"
            )
        if contrast_scale is not None and bool(
            jnp.all(jnp.abs(contrast_scale - 1.0) <= unity_tol)
        ):
            warnings.append(
                "input_energy / peak-pixel energy is 1, so input_energy looks "
                "like telescope_peak * grid.pixel_area rather than a "
                "measured energy; realize() would then return peak-referenced "
                "contrast while its consumers expect a flux fraction. Pass the "
                "real input energy and use peak_contrast(telescope_peak=...) "
                "wherever contrast is wanted"
            )

        return Photometry(
            input_energy=self.response.input_energy,
            grid=self.response.grid,
            e_nom_flux_fraction=flux_fraction,
            contrast_scale=contrast_scale,
            warnings=tuple(warnings),
        )

    def moments(self, *, mask=None, renormalized=False) -> "SpeckleMoments":
        """Closed-form ensemble moments of the delta ``realize`` returns.

        Implements the improper (non-circular) complex-Gaussian moment
        theorem for the diagonal modal covariance ``C_a = diag(rms_k^2)``:
        per focal-plane pixel, with ``Gamma = sum_k rms_k^2 |g_pixel_k|^2``,
        ``P = sum_k rms_k^2 g_pixel_k^2``, ``I_C = |e_nom|^2 * grid.pixel_area``,
        ``phi_C = angle(e_nom)`` and pinning-quadrature variance
        ``Var(X) = (Gamma + Re[P e^{-2 i phi_C}]) / 2``::

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
                "moments is monochromatic; select a channel's (e_nom, G) "
                "first, or use cross_band_moments for the joint band "
                "statistics"
            )
        e_pixel, g_pixel = self.response.per_pixel()
        rms2 = self.per_mode_rms**2
        gamma = jnp.einsum("k,kyx->yx", rms2, jnp.abs(g_pixel) ** 2)
        p = jnp.einsum("k,kyx->yx", rms2, g_pixel**2)
        i_c = jnp.abs(e_pixel) ** 2
        phi_c = jnp.angle(e_pixel)
        var_x = 0.5 * (gamma + jnp.real(p * jnp.exp(-2j * phi_c)))
        norm = self.response.input_energy
        mean_map = gamma / norm
        heterodyne = 4.0 * i_c * var_x if self.coherent else 0.0
        var_map = (heterodyne + gamma**2 + jnp.abs(p) ** 2) / norm**2
        if renormalized:
            kappa = self.renormalization_kurtosis()
            var_map = var_map + (
                jnp.einsum("k,kyx->yx", kappa * rms2**2, jnp.abs(g_pixel) ** 4)
                / norm**2
            )
        annulus_mean = annulus_var = None
        if mask is not None:
            w = jnp.asarray(mask, dtype=float)
            denom = jnp.sum(w)
            annulus_mean = jnp.sum(mean_map * w) / denom
            annulus_var = jnp.sum(var_map * w) / denom
        return SpeckleMoments(
            grid=self.response.grid,
            mean_map=mean_map,
            var_map=var_map,
            gamma_pixel_map=gamma,
            p_pixel_map=p,
            var_x_pixel_map=var_x,
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
        e_pixel, g_pixel = self.response.per_pixel()
        rms2 = self.per_mode_rms**2
        damped = rms2 * self.autocorrelation(tau_s)
        gamma = jnp.einsum("m,imyx,jmyx->ijyx", damped, g_pixel, jnp.conj(g_pixel))
        p = jnp.einsum("m,imyx,jmyx->ijyx", damped, g_pixel, g_pixel)
        w = self.wavelengths_nm.shape[0]
        norms = jnp.broadcast_to(self.response.input_energy, (w,))
        denom = norms[:, None, None, None] * norms[None, :, None, None]
        cov = jnp.abs(gamma) ** 2 + jnp.abs(p) ** 2
        if self.coherent:
            a_i = jnp.conj(e_pixel)[:, None]
            a_j = e_pixel[None, :]
            cov = cov + 2.0 * jnp.real(a_i * a_j * gamma + a_i * jnp.conj(a_j) * p)
        if renormalized:
            cov = cov + jnp.einsum(
                "m,imyx,jmyx->ijyx",
                self.renormalization_kurtosis() * rms2**2,
                jnp.abs(g_pixel) ** 2,
                jnp.abs(g_pixel) ** 2,
            )
        cov = cov / denom
        mean = (
            jnp.einsum("m,wmyx->wyx", rms2, jnp.abs(g_pixel) ** 2)
            / norms[:, None, None]
        )
        annulus_mean = annulus_cov = None
        if mask is not None:
            weights = jnp.asarray(mask, dtype=float)
            total = jnp.sum(weights)
            annulus_mean = jnp.einsum("wyx,yx->w", mean, weights) / total
            annulus_cov = jnp.einsum("ijyx,yx->ij", cov, weights) / total
        return CrossBandMoments(
            grid=self.response.grid,
            mean_map=mean,
            cov_map=cov,
            gamma_pixel_map=gamma,
            p_pixel_map=p,
            tau_s=float(tau_s),
            annulus_mean=annulus_mean,
            annulus_cov=annulus_cov,
        )

    def joint_covariance(self, mask, *, tau_s=0.0) -> Array:
        """The full joint covariance over (band, pixel) pairs on a pixel set.

        The per-pixel maps of :meth:`cross_band_moments` are the coincident
        diagonal of the complete two-point object; spatio-spectral statistics
        (band-weighted aperture sums, matched filters across channels) need
        the cross-pixel terms too. This returns
        ``Cov[delta(band i, pixel p, t), delta(band j, pixel q, t + tau_s)]``
        with shape ``(w, p, w, p)`` over the ``mask`` pixels in raster order.
        Memory scales as ``(w * p)^2``, so the selection is capped at
        ``w * p <= 4096``. Like :meth:`cross_band_moments` with
        ``renormalized=False`` (the default), this describes the exact-Gaussian
        ``renormalize=False`` ensemble; there is no renormalized-ensemble
        variant of this method.

        Args:
            mask: Boolean array over the focal grid selecting the pixels;
                must be concrete (the selection happens host-side, so this
                method is not jittable). Must match the focal grid shape.
            tau_s: Two-time lag in seconds, as in :meth:`cross_band_moments`.

        Returns:
            The real covariance array, shape ``(w, p, w, p)``.

        Raises:
            ValueError: On a monochromatic process, when ``w * p`` exceeds
                4096, or when ``mask``'s shape does not match the focal grid.
        """
        if self.wavelengths_nm is None:
            raise ValueError(
                "joint_covariance needs a chromatic process (wavelengths_nm "
                "set); use moments for a monochromatic one"
            )
        rows, cols = np.nonzero(np.asarray(mask))
        w = self.wavelengths_nm.shape[0]
        if w * rows.size > 4096:
            raise ValueError(
                f"w * p = {w * rows.size} exceeds the 4096 cap; the "
                "(w p) x (w p) covariance would be too large -- select fewer "
                "pixels"
            )
        if tuple(np.shape(mask)) != tuple(self.response.G.shape[-2:]):
            raise ValueError(
                f"mask shape {tuple(np.shape(mask))} does not match the focal "
                f"grid {tuple(self.response.G.shape[-2:])}"
            )
        e, g = self.response.per_pixel(pixels=(rows, cols))
        damped = self.per_mode_rms**2 * self.autocorrelation(tau_s)
        gamma = jnp.einsum("m,imp,jmq->ipjq", damped, g, jnp.conj(g))
        p_kern = jnp.einsum("m,imp,jmq->ipjq", damped, g, g)
        cov = jnp.abs(gamma) ** 2 + jnp.abs(p_kern) ** 2
        if self.coherent:
            a_i = jnp.conj(e)[:, :, None, None]
            a_j = e[None, None, :, :]
            cov = cov + 2.0 * jnp.real(a_i * a_j * gamma + a_i * jnp.conj(a_j) * p_kern)
        norms = jnp.broadcast_to(self.response.input_energy, (w,))
        return cov / (norms[:, None, None, None] * norms[None, None, :, None])
