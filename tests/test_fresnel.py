"""Tests for the near-field Fresnel propagator and its sampling gate."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.diagnostics import fresnel_pad_factor, fresnel_sampling_parameter
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import broadcast_to_spectrum
from physicaloptix.transforms import Fraunhofer, Fresnel

WL = 500.0
LAM_M = WL * 1e-9
DIAM_M = 0.02  # a centimetre-scale relay beam (the regime the design targets)


def _coords(npix, dx):
    return (np.arange(npix) - npix / 2 + 0.5) * dx


def _gaussian(npix, w0, carrier_nu0=0.0):
    """A dimensionless-waist Gaussian, optionally with a phase carrier (a tilt)."""
    x = _coords(npix, 1.0 / npix)
    xg, yg = np.meshgrid(x, x)
    field = np.exp(-(xg**2 + yg**2) / w0**2).astype(complex)
    if carrier_nu0:
        field = field * np.exp(2j * np.pi * carrier_nu0 * xg)
    return jnp.asarray(field)


def _asymmetric_field(npix):
    """A field whose intensity is not flip-symmetric (guards shift handling)."""
    x = _coords(npix, 1.0 / npix)
    xg, yg = np.meshgrid(x, x)
    blob_a = np.exp(-((xg - 0.15) ** 2 + (yg - 0.1) ** 2) / 0.05**2)
    blob_b = 0.4 * np.exp(-((xg + 0.2) ** 2 + (yg + 0.05) ** 2) / 0.08**2)
    carrier = np.exp(2j * np.pi * (2.0 * xg + 1.0 * yg))
    return jnp.asarray(((blob_a + blob_b) * carrier).astype(complex))


def _centroid_x(data, dx):
    npix = data.shape[-1]
    xg, _ = np.meshgrid(_coords(npix, dx), _coords(npix, dx))
    inten = np.abs(np.asarray(data)) ** 2
    return float((inten * xg).sum() / inten.sum())


def _rms_radius(data, dx):
    npix = data.shape[-1]
    xg, yg = np.meshgrid(_coords(npix, dx), _coords(npix, dx))
    inten = np.abs(np.asarray(data)) ** 2
    return float(np.sqrt((inten * (xg**2 + yg**2)).sum() / inten.sum()))


def _direct_fresnel(data, dx, alpha):
    """Independent paraxial Fresnel by direct chirp convolution (centered coords)."""
    x = _coords(data.shape[-1], dx)
    d = x[:, None] - x[None, :]
    kernel = np.sqrt(1.0 / (1j * alpha)) * np.exp(1j * np.pi * d**2 / alpha) * dx
    return kernel @ np.asarray(data) @ kernel.T


def _alpha_of(distance_m, diameter_m=DIAM_M, wavelength_nm=WL):
    return wavelength_nm * 1e-9 * distance_m / diameter_m**2


def _distance_for_alpha(alpha, diameter_m=DIAM_M, wavelength_nm=WL):
    return alpha * diameter_m**2 / (wavelength_nm * 1e-9)


def _fresnel(grid, alpha, *, plane_out=PlaneKind.INTERMEDIATE, **kw):
    # Physics tests run in the conservative-but-correct p < 1 band by design, so
    # record the metric rather than warn; the gate policy has its own test.
    kw.setdefault("on_undersampled", "record")
    return Fresnel(
        grid=grid,
        distance_m=_distance_for_alpha(alpha),
        beam_diameter_m=DIAM_M,
        wavelength_nm=WL,
        plane_in=PlaneKind.PUPIL,
        plane_out=plane_out,
        **kw,
    )


def _pupil(data, npix):
    return Field(data=data, grid=Grid.pupil(npix), plane=PlaneKind.PUPIL)


class TestFresnelSamplingGate:
    def test_matches_physical_length_ratio(self):
        """p equals the physical D / sqrt(N lambda z) for a consistent geometry."""
        npix = 128
        grid = Grid.pupil(npix)  # unit-diameter beam grid, extent L = 1
        d_m, z_m, lam_m = 0.027, 0.5, 500e-9
        alpha = lam_m * z_m / d_m**2
        p = fresnel_sampling_parameter(grid, alpha)
        p_physical = d_m / np.sqrt(npix * lam_m * z_m)
        np.testing.assert_allclose(p, p_physical, rtol=1e-12)

    def test_polarity_larger_alpha_is_worse(self):
        grid = Grid.pupil(256)
        assert fresnel_sampling_parameter(grid, 1e-4) > fresnel_sampling_parameter(
            grid, 1e-2
        )

    def test_zero_distance_is_perfectly_sampled(self):
        assert np.isinf(fresnel_sampling_parameter(Grid.pupil(64), 0.0))

    def test_padding_improves_sampling_as_sqrt_of_points(self):
        """A 4x-padded grid (same dx) gains exactly 2x in the sampling ratio."""
        alpha = 1e-2
        p1 = fresnel_sampling_parameter(Grid.pupil(128), alpha)
        padded = Grid(npix=512, dx=1.0 / 128)  # 4x points, same pitch
        p4 = fresnel_sampling_parameter(padded, alpha)
        np.testing.assert_allclose(p4 / p1, 2.0, rtol=1e-12)

    def test_pad_factor_reaches_nyquist(self):
        grid = Grid.pupil(256)
        alpha = 5e-3  # undersampled: p < 1
        assert fresnel_sampling_parameter(grid, alpha) < 1.0
        npad = fresnel_pad_factor(grid, alpha)
        assert npad >= 2
        padded = Grid(npix=256 * npad, dx=grid.dx)
        assert fresnel_sampling_parameter(padded, alpha) >= 1.0

    def test_pad_factor_is_one_when_well_sampled(self):
        assert fresnel_pad_factor(Grid.pupil(512), 1e-6) == 1


class TestFresnelParaxial:
    def test_forward_backward_round_trip(self):
        """Backward (the adjoint = propagate -z) inverts forward, to roundoff."""
        npix = 256
        field = _pupil(_asymmetric_field(npix), npix)
        prop = _fresnel(Grid.pupil(npix), 3e-3)
        back = prop.backward(prop.forward(field))
        np.testing.assert_allclose(
            np.asarray(back.data), np.asarray(field.data), atol=1e-12
        )

    def test_round_trip_odd_npix(self):
        """The no-fftshift form must be exact for odd npix too (shift guard)."""
        npix = 255
        field = _pupil(_asymmetric_field(npix), npix)
        prop = _fresnel(Grid.pupil(npix), 3e-3)
        back = prop.backward(prop.forward(field))
        np.testing.assert_allclose(
            np.asarray(back.data), np.asarray(field.data), atol=1e-12
        )

    def test_conserves_energy(self):
        npix = 256
        field = _pupil(_asymmetric_field(npix), npix)
        prop = _fresnel(Grid.pupil(npix), 5e-3)
        e_in = float(jnp.sum(jnp.abs(field.data) ** 2))
        e_out = float(jnp.sum(jnp.abs(prop.forward(field).data) ** 2))
        assert abs(e_out - e_in) / e_in < 1e-12

    def test_zero_distance_is_identity(self):
        npix = 128
        field = _pupil(_asymmetric_field(npix), npix)
        prop = Fresnel(
            grid=Grid.pupil(npix),
            distance_m=0.0,
            beam_diameter_m=DIAM_M,
            wavelength_nm=WL,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.INTERMEDIATE,
        )
        out = prop.forward(field)
        np.testing.assert_allclose(
            np.asarray(out.data), np.asarray(field.data), atol=1e-14
        )

    def test_tilt_shifts_beam_by_alpha_nu0(self):
        """A pupil phase carrier (invisible in input intensity) becomes a
        position shift of exactly alpha * nu0 after near-field propagation."""
        npix = 256
        nu0 = 3.0
        alpha = 5e-3
        field = _pupil(_gaussian(npix, w0=0.12, carrier_nu0=nu0), npix)
        # The input intensity is centered (the carrier is pure phase).
        assert abs(_centroid_x(field.data, 1.0 / npix)) < 1e-6
        out = _fresnel(Grid.pupil(npix), alpha).forward(field)
        shift = _centroid_x(out.data, 1.0 / npix)
        np.testing.assert_allclose(shift, alpha * nu0, rtol=2e-3)

    def test_gaussian_beam_broadening_law(self):
        """The 2nd-moment radius follows W(alpha) = W0 sqrt(1 + (alpha/alpha_R)^2),
        alpha_R = pi W0^2 -- an external analytic prediction."""
        npix = 512
        w0 = 0.08
        alpha_r = np.pi * w0**2
        field = _pupil(_gaussian(npix, w0), npix)
        rms0 = _rms_radius(field.data, 1.0 / npix)
        for ratio in (0.5, 1.0):
            alpha = ratio * alpha_r
            out = _fresnel(Grid.pupil(npix), alpha).forward(field)
            broaden = _rms_radius(out.data, 1.0 / npix) / rms0
            np.testing.assert_allclose(broaden, np.sqrt(1.0 + ratio**2), rtol=1e-3)

    def test_matches_direct_chirp_convolution(self):
        """The FFT propagator equals an independent direct chirp convolution."""
        npix = 256
        alpha = 4e-3
        field = _pupil(_gaussian(npix, w0=0.1, carrier_nu0=2.0), npix)
        out = _fresnel(Grid.pupil(npix), alpha).forward(field)
        ref = _direct_fresnel(field.data, 1.0 / npix, alpha)
        rel = np.linalg.norm(np.asarray(out.data) - ref) / np.linalg.norm(ref)
        assert rel < 5e-3

    def test_rejects_wrong_plane(self):
        npix = 64
        focal = Field(
            data=jnp.ones((npix, npix), complex),
            grid=Grid.pupil(npix),
            plane=PlaneKind.FOCAL,
        )
        with pytest.raises(ValueError):
            _fresnel(Grid.pupil(npix), 1e-3).forward(focal)

    def test_rejects_grid_mismatch(self):
        npix = 64
        field = _pupil(_asymmetric_field(npix), npix)
        prop = _fresnel(Grid.pupil(128), 1e-3)
        with pytest.raises(ValueError, match="grid"):
            prop.forward(field)

    def test_composes_as_a_relay_in_optical_path(self):
        """Two Fresnel relay hops then a Fraunhofer reach the focal plane."""
        npix = 64
        pupil = Grid.pupil(npix)
        focal = Grid.focal(32, 0.5)
        path = OpticalPath(
            stages=(
                Stage("out", _fresnel(pupil, 1e-3)),
                Stage(
                    "back",
                    Fresnel(
                        grid=pupil,
                        distance_m=-_distance_for_alpha(1e-3),
                        beam_diameter_m=DIAM_M,
                        wavelength_nm=WL,
                        plane_in=PlaneKind.INTERMEDIATE,
                        plane_out=PlaneKind.PUPIL,
                    ),
                ),
                Stage("science", Fraunhofer(grid_in=pupil, grid_out=focal)),
            )
        )
        out, _ = path.propagate(_pupil(_gaussian(npix, 0.2), npix))
        assert out.plane is PlaneKind.FOCAL
        assert out.data.shape == (32, 32)

    def test_is_differentiable_through_a_phase_parameter(self):
        """A defocus coefficient on the input yields a finite, nonzero gradient."""
        npix = 64
        pupil = Grid.pupil(npix)
        x = _coords(npix, 1.0 / npix)
        xg, yg = np.meshgrid(x, x)
        base = jnp.asarray(_gaussian(npix, 0.2))
        quad = jnp.asarray((xg**2 + yg**2).astype(float))
        prop = _fresnel(pupil, 2e-3)

        def loss(coeff):
            data = base * jnp.exp(1j * coeff * quad)
            out = prop.forward(_pupil(data, npix))
            return jnp.sum(jnp.abs(out.data[40:60, 40:60]) ** 2)

        grad = jax.grad(loss)(0.3)
        assert jnp.isfinite(grad)
        assert grad != 0.0

    def test_gate_warns_by_default_and_can_raise(self):
        """An undersampled configuration warns by default and raises on policy."""
        grid = Grid.pupil(256)
        big_alpha = 5e-2  # p ~ 0.28
        with pytest.warns(UserWarning, match="undersampled"):
            prop = _fresnel(grid, big_alpha, on_undersampled="warn")
        assert prop.sampling_parameter < 1.0
        with pytest.raises(ValueError, match="sampl"):
            _fresnel(grid, big_alpha, on_undersampled="raise")

    def test_rejects_bad_undersampled_policy(self):
        with pytest.raises(ValueError, match="raise/warn/record"):
            _fresnel(Grid.pupil(64), 1e-3, on_undersampled="explode")


class TestFresnelChromatic:
    def test_matches_per_slice_mono(self):
        """Each chromatic slice propagates with its own lambda-scaled alpha."""
        npix = 128
        spectrum = Spectrum.tophat(500.0, 0.2, 3)  # 450, 500, 550 nm
        base = _pupil(_gaussian(npix, 0.15, carrier_nu0=2.0), npix)
        chrom = broadcast_to_spectrum(base, spectrum)
        distance_m = _distance_for_alpha(2e-3)
        prop = Fresnel(
            grid=Grid.pupil(npix),
            distance_m=distance_m,
            beam_diameter_m=DIAM_M,
            wavelength_nm=500.0,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.INTERMEDIATE,
            on_undersampled="record",
        )
        out = prop.forward(chrom)
        assert out.data.shape == (3, npix, npix)
        for k, wl in enumerate(np.asarray(spectrum.wavelengths_nm)):
            mono = Fresnel(
                grid=Grid.pupil(npix),
                distance_m=distance_m,
                beam_diameter_m=DIAM_M,
                wavelength_nm=float(wl),
                plane_in=PlaneKind.PUPIL,
                plane_out=PlaneKind.INTERMEDIATE,
                on_undersampled="record",
            ).forward(base)
            np.testing.assert_allclose(
                np.asarray(out.data[k]), np.asarray(mono.data), atol=1e-12
            )

    def test_chromatic_round_trip(self):
        npix = 128
        spectrum = Spectrum.tophat(600.0, 0.15, 4)
        chrom = broadcast_to_spectrum(_pupil(_gaussian(npix, 0.15), npix), spectrum)
        prop = Fresnel(
            grid=Grid.pupil(npix),
            distance_m=_distance_for_alpha(2e-3),
            beam_diameter_m=DIAM_M,
            wavelength_nm=600.0,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.INTERMEDIATE,
            on_undersampled="record",
        )
        back = prop.backward(prop.forward(chrom))
        np.testing.assert_allclose(
            np.asarray(back.data), np.asarray(chrom.data), atol=1e-12
        )

    def test_gate_uses_reddest_wavelength(self):
        """The construction gate keys on max_wavelength_nm (alpha grows with lambda)."""
        grid = Grid.pupil(256)
        distance_m = _distance_for_alpha(2e-2, wavelength_nm=550.0)
        common = dict(
            grid=grid,
            distance_m=distance_m,
            beam_diameter_m=DIAM_M,
            wavelength_nm=550.0,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.INTERMEDIATE,
            on_undersampled="record",
        )
        p_red = Fresnel(**common, max_wavelength_nm=700.0).sampling_parameter
        p_blue = Fresnel(**common, max_wavelength_nm=400.0).sampling_parameter
        assert p_red < p_blue


class TestFresnelExact:
    def test_exact_matches_paraxial_at_beam_scale(self):
        """At the design beam scale (beta << 1) exact and paraxial agree."""
        npix = 128
        field = _pupil(_gaussian(npix, 0.12, carrier_nu0=2.0), npix)
        alpha = 3e-3
        paraxial = _fresnel(Grid.pupil(npix), alpha, method="fresnel").forward(field)
        exact = _fresnel(Grid.pupil(npix), alpha, method="exact").forward(field)
        rel = np.linalg.norm(
            np.asarray(exact.data) - np.asarray(paraxial.data)
        ) / np.linalg.norm(np.asarray(paraxial.data))
        assert rel < 1e-8

    def test_exact_diverges_from_paraxial_at_wide_angle(self):
        """A small beam (large beta) makes the non-paraxial term measurable."""
        npix = 128
        pupil = Grid.pupil(npix)
        field = _pupil(_gaussian(npix, 0.12, carrier_nu0=6.0), npix)
        # A ~5 micron beam: beta = lambda / D is no longer negligible.
        kw = dict(
            grid=pupil,
            distance_m=2e-4,
            beam_diameter_m=5e-6,
            wavelength_nm=WL,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.INTERMEDIATE,
            on_undersampled="record",
        )
        paraxial = Fresnel(**kw, method="fresnel").forward(field)
        exact = Fresnel(**kw, method="exact").forward(field)
        rel = np.linalg.norm(
            np.asarray(exact.data) - np.asarray(paraxial.data)
        ) / np.linalg.norm(np.asarray(paraxial.data))
        assert rel > 1e-3

    def test_exact_evanescent_attenuates_never_amplifies(self):
        """Beyond the evanescent cutoff the transfer function decays (|H| < 1),
        it does not clamp to |H| = 1 or amplify."""
        npix = 64
        prop = Fresnel(
            grid=Grid.pupil(npix),
            distance_m=1e-4,
            beam_diameter_m=1e-5,  # ~10 micron beam: beta * nu_max > 1
            wavelength_nm=WL,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.INTERMEDIATE,
            on_undersampled="record",
            method="exact",
        )
        transfer = np.asarray(prop._transfer(WL, npix))
        mag = np.abs(transfer)
        assert mag.max() <= 1.0 + 1e-9  # never amplifies
        assert mag.min() < 0.9  # an evanescent (attenuating) region exists

    def test_rejects_bad_method(self):
        with pytest.raises(ValueError, match="method"):
            _fresnel(Grid.pupil(64), 1e-3, method="quantum")


class TestFresnelPadding:
    def test_padding_preserves_a_well_contained_result(self):
        """For a field that decays before the edge, padding does not change the
        (already wrap-free) result."""
        npix = 128
        field = _pupil(_gaussian(npix, 0.08, carrier_nu0=2.0), npix)
        unpadded = _fresnel(Grid.pupil(npix), 3e-3, npad=1).forward(field)
        padded = _fresnel(Grid.pupil(npix), 3e-3, npad=2).forward(field)
        rel = np.linalg.norm(
            np.asarray(padded.data) - np.asarray(unpadded.data)
        ) / np.linalg.norm(np.asarray(unpadded.data))
        assert rel < 1e-4

    def test_padding_equals_native_larger_grid(self):
        """npad = N is exactly a native N-times-larger-grid propagation, cropped
        (the mechanism proof: embed, filter, crop)."""
        npix, npad = 64, 4
        alpha = 8e-3
        field_small = _gaussian(npix, 0.1, carrier_nu0=3.0)
        padded = _fresnel(Grid.pupil(npix), alpha, npad=npad).forward(
            _pupil(field_small, npix)
        )
        n = npix * npad
        lo = (n - npix) // 2
        big = np.zeros((n, n), complex)
        big[lo : lo + npix, lo : lo + npix] = np.asarray(field_small)
        big_grid = Grid(npix=n, dx=1.0 / npix)
        native = Fresnel(
            grid=big_grid,
            distance_m=_distance_for_alpha(alpha),
            beam_diameter_m=DIAM_M,
            wavelength_nm=WL,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.INTERMEDIATE,
            on_undersampled="record",
        ).forward(Field(data=jnp.asarray(big), grid=big_grid, plane=PlaneKind.PUPIL))
        native_crop = np.asarray(native.data)[lo : lo + npix, lo : lo + npix]
        np.testing.assert_allclose(np.asarray(padded.data), native_crop, atol=1e-12)

    def test_rejects_bad_npad(self):
        with pytest.raises(ValueError, match="npad"):
            _fresnel(Grid.pupil(64), 1e-3, npad=0)
