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
import numpy as np
from hwoutils.constants import m2nm, nm2m
from hwoutils.numerics import COARSEST
from hwoutils.precision import complex_of
from optixstuff.grids import PixelGrid

from physicaloptix.core import Field, PlaneKind, validate_field
from physicaloptix.diagnostics import fresnel_sampling_parameter
from physicaloptix.transforms._phase import _spatial_cycles, _split_host


class Fresnel(eqx.Module):
    """Same-grid near-field propagation by the angular-spectrum transfer function.

    ``forward`` maps a ``plane_in`` field on ``grid`` to ``plane_out`` on the
    same grid; ``backward`` is the adjoint (propagation by ``-z``). The chirp's
    Nyquist ratio is evaluated once at construction on the static grid and
    handled per ``on_undersampled`` ("raise", "warn", or "record").

    Attributes:
        grid: Propagation grid shared by the input and output fields.
        plane_in: Plane the input field must be in.
        plane_out: Plane the output field is tagged with.
        distance_m: Propagation distance in meters; may be negative.
        beam_diameter_m: Physical beam diameter at this plane in meters.
        wavelength_nm: Design wavelength in nanometers.
        spectral_nodes_nm: Optional original host quadrature nodes for an exact hop.
        method: 'fresnel' (paraxial) or 'exact' (non-paraxial angular spectrum).
        npad: Real-domain zero-pad factor against FFT wrap-around.
        on_undersampled: Policy applied at construction when the sampling parameter is
            below 1: 'raise', 'warn' or 'record'.
        sampling_parameter: Chirp Nyquist ratio evaluated at construction; values below
            1 mean the transfer function is undersampled.
    """

    grid: PixelGrid
    plane_in: PlaneKind = eqx.field(static=True)
    plane_out: PlaneKind = eqx.field(static=True)
    distance_m: float = eqx.field(static=True)
    beam_diameter_m: float = eqx.field(static=True)
    wavelength_nm: float = eqx.field(static=True)
    spectral_nodes_nm: tuple[float, ...] | None = eqx.field(static=True)
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
        high_nm=None,
        spectral_nodes_nm=None,
    ):
        """Build the propagator and evaluate its sampling gate.

        Args:
            grid: The (same in/out) propagation grid.
            distance_m: Physical propagation distance (meters); may be negative.
            beam_diameter_m: Physical beam diameter AT this plane (the demagnified relay
                beam, not the primary), which forms alpha.
            wavelength_nm: Design wavelength (nanometers); mono fields use it, chromatic
                fields scale each slice by its own wavelength.
            plane_in: Plane the input field must be in.
            plane_out: Plane the output field is tagged with.
            method: 'fresnel' (paraxial, default) or 'exact' (non-paraxial angular
                spectrum with a decaying evanescent branch).
            npad: Real-domain zero-pad factor to suppress FFT wrap-around; the field is
                embedded in an npad * npix grid and cropped back (see
                :func:`physicaloptix.diagnostics.fresnel_pad_factor`).
            on_undersampled: Policy when the chirp Nyquist ratio is below 1: 'raise'
                fails construction, 'warn' emits a warning, 'record' only stores the
                metric. The gate is a conservative operator bound, so band-limited
                fields are often fine below 1.
            high_nm: Red end of a chromatic band for the gate in nanometers (alpha grows
                with lambda, so the reddest slice is worst-sampled); defaults to
                wavelength_nm.
            spectral_nodes_nm: Original host wavelengths for the one-band field's
                node axis, in its existing order. Supply these when the exact hop
                needs detuning or phase information below the spacing of the stored
                quadrature nodes. The unrounded values bind both propagation
                directions. When omitted, a one-node monochromatic band supplies
                its host tag and other quadratures use the dynamic nodes.

        Raises:
            ValueError: If the method, sampling policy or spectral nodes are invalid.
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
        self.spectral_nodes_nm = (
            None if spectral_nodes_nm is None else tuple(map(float, spectral_nodes_nm))
        )
        if self.spectral_nodes_nm is not None and (
            method != "exact"
            or not self.spectral_nodes_nm
            or not all(np.isfinite(w) and w > 0 for w in self.spectral_nodes_nm)
        ):
            raise ValueError(
                "spectral_nodes_nm requires positive finite nodes and method='exact'"
            )
        self.on_undersampled = on_undersampled
        gate_wavelength = high_nm if high_nm else wavelength_nm
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

    def spectral_breaks_nm(self) -> np.ndarray:
        """Evanescent cutoffs [nm] of the exact propagator; Fresnel is smooth.

        Returns:
            Evanescent cutoff wavelengths in nanometers, ascending, for the exact
            method; empty for the paraxial method or zero distance.
        """
        if self.method == "fresnel" or self.distance_m == 0:
            return np.empty(0)
        nu2 = np.unique(self._nu2(self.grid.shape[0] * self.npad))
        return (self.beam_diameter_m * m2nm / np.sqrt(nu2[nu2 > 0]))[::-1]

    def _alpha(self, wavelength_nm):
        """The dimensionless Fresnel parameter ``lambda * z / D^2``."""
        return (wavelength_nm * nm2m) * self.distance_m / self.beam_diameter_m**2

    def _nu2(self, npix, dtype=None):
        """Squared dimensionless frequency on an ``npix`` grid (cycles/diameter).

        The pitch ``dx`` is unchanged by padding, so a larger ``npix`` only
        refines the frequency grid (which is why padding improves sampling).
        """
        nu = np.fft.fftfreq(npix, d=self.grid.step)
        if dtype is not None:
            nu = jnp.asarray(nu, dtype=dtype)
        return nu[:, None] ** 2 + nu[None, :] ** 2

    def _transfer(self, wavelength_nm, npix, dtype=None):
        """The angular-spectrum transfer function for one wavelength on ``npix``.

        Paraxial (``method="fresnel"``): ``exp(-i pi alpha nu^2)``. Exact
        (``method="exact"``): ``exp(i 2 pi (gamma/beta) sqrt(1 - beta^2 nu^2))``
        with ``beta = lambda/D``, ``gamma = z/D``; beyond the evanescent cutoff
        (``beta^2 nu^2 > 1``) the wave DECAYS (magnitude set by ``|z|``, so the
        branch attenuates and never amplifies), rather than clamping to unity.
        """
        if dtype is None:
            dtype = jnp.result_type(wavelength_nm, 0.0)
        if self.method == "fresnel":
            nu2 = self._nu2(npix, dtype=dtype)
            return jnp.exp(-1j * jnp.pi * self._alpha(wavelength_nm) * nu2)

        # For a monochromatic design the whole kernel is static. In particular,
        # resolve the optical path and the detuning from each cutoff before a
        # float32 field can narrow either one. Spectral nodes may be traced.
        static = isinstance(wavelength_nm, float | np.floating)
        xp = np if static else jnp
        nu2 = self._nu2(npix)
        diameter_nm = self.beam_diameter_m * m2nm
        cutoff = diameter_nm / np.sqrt(np.where(nu2 == 0, 1.0, nu2))
        # Only the equivalent phase formulas share a decision grid. The
        # physical detuning on either side of cutoff must retain its sign
        # and magnitude even within one cell of that grid.
        decision_wavelength = xp.asarray(wavelength_nm, COARSEST)
        non_dc = nu2 != 0
        switch = np.minimum(np.sqrt(0.75) * cutoff, np.finfo(COARSEST).max)
        near_grazing = non_dc & (decision_wavelength > xp.asarray(switch, COARSEST))
        # Scale by a power of two before converting meters to nanometers;
        # even a finite path outside the single-precision nanometer range
        # then has representable high and low parts.
        _, exponent = np.frexp(self.distance_m)
        shift = max(
            0,
            int(exponent) + int(np.ceil(np.log2(m2nm))) - np.finfo(COARSEST).maxexp + 1,
        )
        distance = float(np.ldexp(self.distance_m, -shift) * m2nm)
        scale = float(np.ldexp(1.0, -shift))
        wavelength = wavelength_nm * scale
        if static:
            remainder = np.fmod(distance, wavelength)
        else:
            high = float(COARSEST(distance))
            residual = distance - high
            middle = float(COARSEST(residual))
            low = residual - middle
            remainder = (
                jnp.fmod(jnp.asarray(high, dtype=dtype), wavelength)
                + jnp.fmod(jnp.asarray(middle, dtype=dtype), wavelength)
                + jnp.fmod(jnp.asarray(low, dtype=dtype), wavelength)
            )
        carrier = remainder / wavelength
        path_cycles = distance / wavelength
        if not static and wavelength_nm.dtype == COARSEST:
            # Scale each cutoff to a normal mantissa. The wavelength is
            # scaled by the same exact power of two before differencing.
            cutoff_mantissa, cutoff_exponent = np.frexp(cutoff)
            cutoff_pair = tuple(
                jnp.asarray(x, wavelength.dtype) for x in _split_host(cutoff_mantissa)
            )
            path_pair = tuple(
                jnp.asarray(x, wavelength.dtype)
                for x in _split_host(distance / (cutoff * scale))
            )
            spatial, radicand, propagating, evanescent = _spatial_cycles(
                jnp.ldexp(wavelength_nm, -jnp.asarray(cutoff_exponent)),
                cutoff_pair,
                path_pair,
                near_grazing,
                jnp.asarray(non_dc),
            )
        else:
            delta = cutoff - wavelength_nm
            radicand = xp.where(
                non_dc, (delta / cutoff) * (1 + wavelength_nm / cutoff), 1.0
            )
            propagating = radicand > 0
            evanescent = radicand < 0
            root = xp.sqrt(xp.where(propagating, radicand, 1.0))
            # Rationalize sqrt(1 - s) - 1 before multiplying by the carrier.
            offset = -self._alpha(wavelength_nm) * nu2 / (1 + root)
            spatial = xp.fmod(xp.where(near_grazing, path_cycles * root, offset), 1)
        cycles = xp.where(near_grazing, spatial, carrier + spatial)
        decay = (
            -2
            * xp.pi
            * xp.abs(path_cycles)
            * xp.sqrt(xp.where(evanescent, -radicand, 1))
        )
        return jnp.asarray(
            xp.exp(
                2j * xp.pi * xp.where(propagating, cycles, 0)
                + xp.where(evanescent, decay, 0)
            ),
            dtype=complex_of(dtype),
        )

    def _apply_data(self, data, transfer):
        """Filter one 2D slice by its transfer function (orthonormal FFT pair).

        With ``npad > 1`` the slice is zero-padded into a larger grid (a guard
        band against cyclic wrap-around), filtered, and cropped back.
        """
        if self.npad == 1:
            spectrum = jnp.fft.fft2(data, norm="ortho")
            return jnp.fft.ifft2(spectrum * transfer, norm="ortho")
        npix = self.grid.shape[0]
        n = npix * self.npad
        lo = (n - npix) // 2
        window = jnp.zeros((n, n), dtype=data.dtype)
        padded = window.at[lo : lo + npix, lo : lo + npix].set(data)
        spectrum = jnp.fft.fft2(padded, norm="ortho")
        filtered = jnp.fft.ifft2(spectrum * transfer, norm="ortho")
        return filtered[lo : lo + npix, lo : lo + npix]

    def _propagate(self, field, conjugate):
        """Apply the (optionally conjugated) transfer function per wavelength."""
        n = self.grid.shape[0] * self.npad
        if field.spectrum is None:
            transfer = self._transfer(self.wavelength_nm, n, field.data.real.dtype)
            if conjugate:
                transfer = jnp.conj(transfer)
            return self._apply_data(field.data, transfer)
        nodes = field.spectrum.nodes_nm[0]
        host_nodes = self.spectral_nodes_nm
        band = field.spectrum.bands[0]
        if (
            host_nodes is None
            and self.method == "exact"
            and band.width_nm == 0
            and nodes.size == 1
        ):
            host_nodes = (band.low_nm,)
        if host_nodes is None:
            transfers = jax.vmap(lambda wl: self._transfer(wl, n))(nodes)
        else:
            if len(host_nodes) != nodes.size:
                raise ValueError("spectral_nodes_nm must match the field's node axis")
            transfers = jnp.stack(
                [self._transfer(wl, n, field.data.real.dtype) for wl in host_nodes]
            )
            transfers = eqx.error_if(
                transfers,
                jnp.any(nodes.astype(COARSEST) != jnp.asarray(host_nodes, COARSEST)),
                "spectral_nodes_nm must match the field's stored wavelength nodes",
            )
        if conjugate:
            transfers = jnp.conj(transfers)
        return jax.vmap(self._apply_data)(field.data, transfers)

    def forward(self, field):
        """Propagate ``plane_in`` -> ``plane_out`` by ``+distance_m``.

        Args:
            field: Field in plane_in on the propagation grid, mono or chromatic.

        Returns:
            Field in plane_out on the same grid, propagated by +distance_m.
        """
        validate_field(field, plane=self.plane_in, grid=self.grid, context="Fresnel")
        return Field(
            data=self._propagate(field, conjugate=False),
            grid=self.grid,
            plane=self.plane_out,
            spectrum=field.spectrum,
        )

    def backward(self, field):
        """Adjoint propagation ``plane_out`` -> ``plane_in`` (propagation by ``-z``).

        Args:
            field: Field in plane_out on the propagation grid, mono or chromatic.

        Returns:
            Field in plane_in on the same grid, propagated by -distance_m.
        """
        validate_field(field, plane=self.plane_out, grid=self.grid, context="Fresnel")
        return Field(
            data=self._propagate(field, conjugate=True),
            grid=self.grid,
            plane=self.plane_in,
            spectrum=field.spectrum,
        )
