"""The plane-aware near-field (Fresnel) propagator over the angular spectrum.

The near-field companion to :class:`Fraunhofer`: it carries a field a physical
distance ``distance_m`` between two planes at the SAME sampling (pupil to an
out-of-pupil intermediate, or intermediate to intermediate) by the paraxial
angular-spectrum transfer function. In the dimensionless core the whole hop
collapses to one number, ``alpha = lambda * z / D^2`` with ``D`` the beam
diameter at the plane, and the transfer function is ``H(nu) = exp(-i pi alpha
nu^2)`` on the dimensionless frequency grid ``nu`` (cycles per beam diameter).

No fftshift of the field or kernel is needed: a frequency-domain multiply is a
cyclic (shift-invariant) convolution, so with the same grid in and out the
half-pixel origin cancels for any npix. The FFT pair is orthonormal, so with a
unit-modulus kernel the hop conserves energy and ``backward`` (the conjugate
kernel) is the exact adjoint, equal to propagation by ``-z``. A construction-time
sampling gate (:func:`physicaloptix.diagnostics.fresnel_sampling_parameter`,
adopted from abcdLux) flags an undersampled chirp.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp

from physicaloptix.core import Field, Grid, PlaneKind, validate_field
from physicaloptix.diagnostics import fresnel_sampling_parameter


class Fresnel(eqx.Module):
    """Same-grid near-field propagation by the angular-spectrum transfer function.

    ``forward`` maps a ``plane_in`` field on ``grid`` to ``plane_out`` on the
    same grid; ``backward`` is the adjoint (propagation by ``-z``). The chirp's
    Nyquist ratio is evaluated once at construction on the static grid and
    handled per ``on_undersampled`` ("raise", "warn", or "record").
    """

    grid: Grid
    plane_in: PlaneKind = eqx.field(static=True)
    plane_out: PlaneKind = eqx.field(static=True)
    distance_m: float = eqx.field(static=True)
    beam_diameter_m: float = eqx.field(static=True)
    wavelength_nm: float = eqx.field(static=True)
    method: str = eqx.field(static=True)
    npad: int = eqx.field(static=True)
    on_undersampled: str = eqx.field(static=True)
    sampling_parameter: float = eqx.field(static=True)

    def __init__(
        self,
        grid,
        *,
        distance_m,
        beam_diameter_m,
        wavelength_nm,
        plane_in=PlaneKind.PUPIL,
        plane_out=PlaneKind.INTERMEDIATE,
        method="fresnel",
        npad=1,
        on_undersampled="warn",
        max_wavelength_nm=None,
    ):
        """Build the propagator and evaluate its sampling gate.

        Args:
            grid: The (same in/out) propagation grid.
            distance_m: Physical propagation distance (metres); may be negative.
            beam_diameter_m: Physical beam diameter AT this plane (the
                demagnified relay beam, not the primary), which forms ``alpha``.
            wavelength_nm: Design wavelength (nanometres); mono fields use it,
                chromatic fields scale each slice by its own wavelength.
            plane_in: Plane the input field must be in.
            plane_out: Plane the output field is tagged with.
            on_undersampled: Policy when the chirp Nyquist ratio is below 1:
                "raise" fails construction, "warn" emits a warning, "record"
                only stores the metric. The gate is a conservative operator
                bound, so band-limited fields are often fine below 1.
            method: "fresnel" (paraxial, default) or "exact" (non-paraxial
                angular spectrum with a decaying evanescent branch).
            npad: Real-domain zero-pad factor to suppress FFT wrap-around; the
                field is embedded in an ``npad * npix`` grid and cropped back
                (see :func:`physicaloptix.diagnostics.fresnel_pad_factor`).
            max_wavelength_nm: Red end of a chromatic band for the gate (alpha
                grows with lambda, so the reddest slice is worst-sampled);
                defaults to ``wavelength_nm``.
        """
        if method not in ("fresnel", "exact"):
            raise ValueError(f"method must be fresnel/exact, got {method!r}")
        if int(npad) < 1:
            raise ValueError(f"npad must be a positive integer, got {npad}")
        if on_undersampled not in ("raise", "warn", "record"):
            raise ValueError(
                f"on_undersampled must be raise/warn/record, got {on_undersampled!r}"
            )
        self.method = method
        self.npad = int(npad)
        self.grid = grid
        self.plane_in = plane_in
        self.plane_out = plane_out
        self.distance_m = float(distance_m)
        self.beam_diameter_m = float(beam_diameter_m)
        self.wavelength_nm = float(wavelength_nm)
        self.on_undersampled = on_undersampled
        gate_wavelength = max_wavelength_nm if max_wavelength_nm else wavelength_nm
        alpha = abs(self._alpha(gate_wavelength))
        self.sampling_parameter = fresnel_sampling_parameter(grid, alpha)
        if self.sampling_parameter < 1.0:
            message = (
                f"Fresnel transfer function undersampled: sampling parameter "
                f"{self.sampling_parameter:.3g} < 1 for grid {grid} at "
                f"alpha={alpha:.3g}"
            )
            if on_undersampled == "raise":
                raise ValueError(message)
            if on_undersampled == "warn":
                warnings.warn(message, stacklevel=2)

    def _alpha(self, wavelength_nm):
        """The dimensionless Fresnel parameter ``lambda * z / D^2``."""
        return (wavelength_nm * 1e-9) * self.distance_m / self.beam_diameter_m**2

    def _nu2(self, npix):
        """Squared dimensionless frequency on an ``npix`` grid (cycles/diameter).

        The pitch ``dx`` is unchanged by padding, so a larger ``npix`` only
        refines the frequency grid (which is why padding improves sampling).
        """
        nu = jnp.fft.fftfreq(npix, d=self.grid.dx)
        return nu[:, jnp.newaxis] ** 2 + nu[jnp.newaxis, :] ** 2

    def _transfer(self, wavelength_nm, npix):
        """The angular-spectrum transfer function for one wavelength on ``npix``.

        Paraxial (``method="fresnel"``): ``exp(-i pi alpha nu^2)``. Exact
        (``method="exact"``): ``exp(i 2 pi (gamma/beta) sqrt(1 - beta^2 nu^2))``
        with ``beta = lambda/D``, ``gamma = z/D``; beyond the evanescent cutoff
        (``beta^2 nu^2 > 1``) the wave DECAYS (magnitude set by ``|z|``, so the
        branch attenuates and never amplifies), rather than clamping to unity.
        """
        nu2 = self._nu2(npix)
        if self.method == "fresnel":
            return jnp.exp(-1j * jnp.pi * self._alpha(wavelength_nm) * nu2)
        lam_m = wavelength_nm * 1e-9
        beta = lam_m / self.beam_diameter_m
        gamma = self.distance_m / self.beam_diameter_m
        radicand = 1.0 - beta**2 * nu2
        propagating = jnp.sqrt(jnp.clip(radicand, 0.0, None))
        evanescent = jnp.sqrt(jnp.clip(-radicand, 0.0, None))
        phase = 2.0 * jnp.pi * (gamma / beta) * propagating
        decay = -2.0 * jnp.pi * (abs(gamma) / beta) * evanescent
        return jnp.exp(1j * phase + decay)

    def _apply_data(self, data, transfer):
        """Filter one 2D slice by its transfer function (orthonormal FFT pair).

        With ``npad > 1`` the slice is zero-padded into a larger grid (a guard
        band against cyclic wrap-around), filtered, and cropped back.
        """
        if self.npad == 1:
            spectrum = jnp.fft.fft2(data, norm="ortho")
            return jnp.fft.ifft2(spectrum * transfer, norm="ortho")
        npix = self.grid.npix
        n = npix * self.npad
        lo = (n - npix) // 2
        window = jnp.zeros((n, n), dtype=data.dtype)
        padded = window.at[lo : lo + npix, lo : lo + npix].set(data)
        spectrum = jnp.fft.fft2(padded, norm="ortho")
        filtered = jnp.fft.ifft2(spectrum * transfer, norm="ortho")
        return filtered[lo : lo + npix, lo : lo + npix]

    def _propagate(self, field, conjugate):
        """Apply the (optionally conjugated) transfer function per wavelength."""
        n = self.grid.npix * self.npad
        if field.spectrum is None:
            transfer = self._transfer(self.wavelength_nm, n)
            if conjugate:
                transfer = jnp.conj(transfer)
            return self._apply_data(field.data, transfer)
        transfers = jax.vmap(lambda wl: self._transfer(wl, n))(
            field.spectrum.wavelengths_nm
        )
        if conjugate:
            transfers = jnp.conj(transfers)
        return jax.vmap(self._apply_data)(field.data, transfers)

    def forward(self, field):
        """Propagate ``plane_in`` -> ``plane_out`` by ``+distance_m``."""
        validate_field(field, plane=self.plane_in, grid=self.grid, context="Fresnel")
        return Field(
            data=self._propagate(field, conjugate=False),
            grid=self.grid,
            plane=self.plane_out,
            spectrum=field.spectrum,
        )

    def backward(self, field):
        """Adjoint propagation ``plane_out`` -> ``plane_in`` (propagation by ``-z``)."""
        validate_field(field, plane=self.plane_out, grid=self.grid, context="Fresnel")
        return Field(
            data=self._propagate(field, conjugate=True),
            grid=self.grid,
            plane=self.plane_in,
            spectrum=field.spectrum,
        )

    def __call__(self, field):
        """Alias for :meth:`forward` (the OpticalPath fold convention)."""
        return self.forward(field)
