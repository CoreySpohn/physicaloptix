"""Tests for the near-field Fresnel propagator and its sampling gate."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TEST_FRAME
from hwoutils.constants import m2nm
from hwoutils.numerics import eps
from hwoutils.testing import ulps
from optixstuff.grids import PixelGrid
from optixstuff.spectral import Band, Quadrature

from physicaloptix.core import Field, PlaneKind
from physicaloptix.diagnostics import fresnel_pad_factor, fresnel_sampling_parameter
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import broadcast_to_spectrum
from physicaloptix.transforms import Fraunhofer, Fresnel

WL = 500.0
LAM_M = WL * 1e-9
DIAM_M = 0.02  # a centimeter-scale relay beam (the regime the design targets)


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
    return Field(data=data, grid=PixelGrid.pupil(npix), plane=PlaneKind.PUPIL)


class TestFresnelSamplingGate:
    def test_matches_physical_length_ratio(self):
        """p equals the physical D / sqrt(N lambda z) for a consistent geometry."""
        npix = 128
        grid = PixelGrid.pupil(npix)  # unit-diameter beam grid, extent L = 1
        d_m, z_m, lam_m = 0.027, 0.5, 500e-9
        alpha = lam_m * z_m / d_m**2
        p = fresnel_sampling_parameter(grid, alpha)
        p_physical = d_m / np.sqrt(npix * lam_m * z_m)
        np.testing.assert_allclose(p, p_physical, rtol=1e-12)

    def test_polarity_larger_alpha_is_worse(self):
        grid = PixelGrid.pupil(256)
        assert fresnel_sampling_parameter(grid, 1e-4) > fresnel_sampling_parameter(
            grid, 1e-2
        )

    def test_zero_distance_is_perfectly_sampled(self):
        assert np.isinf(fresnel_sampling_parameter(PixelGrid.pupil(64), 0.0))

    def test_padding_improves_sampling_as_sqrt_of_points(self):
        """A 4x-padded grid (same dx) gains exactly 2x in the sampling ratio."""
        alpha = 1e-2
        p1 = fresnel_sampling_parameter(PixelGrid.pupil(128), alpha)
        padded = PixelGrid.regular(
            (512, 512), 1.0 / 128, "pupil_diameter"
        )  # 4x points, same pitch
        p4 = fresnel_sampling_parameter(padded, alpha)
        np.testing.assert_allclose(p4 / p1, 2.0, rtol=1e-12)

    def test_pad_factor_reaches_nyquist(self):
        grid = PixelGrid.pupil(256)
        alpha = 5e-3  # undersampled: p < 1
        assert fresnel_sampling_parameter(grid, alpha) < 1.0
        npad = fresnel_pad_factor(grid, alpha)
        assert npad >= 2
        padded = PixelGrid.regular(
            (256 * npad, 256 * npad), grid.step, "pupil_diameter"
        )
        assert fresnel_sampling_parameter(padded, alpha) >= 1.0

    def test_pad_factor_is_one_when_well_sampled(self):
        assert fresnel_pad_factor(PixelGrid.pupil(512), 1e-6) == 1


class TestFresnelParaxial:
    def test_forward_backward_round_trip(self):
        """Backward (the adjoint = propagate -z) inverts forward, to roundoff.

        Four 2D FFTs allow six roundings per axis and butterfly stage.
        """
        npix = 256
        field = _pupil(_asymmetric_field(npix), npix)
        prop = _fresnel(PixelGrid.pupil(npix), 3e-3)
        back = prop.backward(prop.forward(field))
        np.testing.assert_allclose(
            np.asarray(back.data),
            np.asarray(field.data),
            rtol=0,
            atol=float(
                ulps(48 * int(np.ceil(np.log2(npix))), jnp.max(jnp.abs(field.data)))
            ),
        )

    def test_round_trip_odd_npix(self):
        """The no-fftshift form must be exact for odd npix too (shift guard).

        Four 2D FFTs allow six roundings per axis and butterfly stage.
        """
        npix = 255
        field = _pupil(_asymmetric_field(npix), npix)
        prop = _fresnel(PixelGrid.pupil(npix), 3e-3)
        back = prop.backward(prop.forward(field))
        np.testing.assert_allclose(
            np.asarray(back.data),
            np.asarray(field.data),
            rtol=0,
            atol=float(
                ulps(48 * int(np.ceil(np.log2(npix))), jnp.max(jnp.abs(field.data)))
            ),
        )

    def test_conserves_energy(self):
        """Parseval allows two 2D FFTs, squaring the field and a pairwise pixel sum."""
        npix = 256
        field = _pupil(_asymmetric_field(npix), npix)
        prop = _fresnel(PixelGrid.pupil(npix), 5e-3)
        e_in = jnp.sum(jnp.abs(field.data) ** 2)
        e_out = jnp.sum(jnp.abs(prop.forward(field).data) ** 2)
        assert abs(e_out - e_in) < float(ulps(50 * int(np.log2(npix)) + 3, e_in))

    def test_zero_distance_is_identity(self):
        """Two 2D FFTs allow six roundings per axis and butterfly stage."""
        npix = 128
        field = _pupil(_asymmetric_field(npix), npix)
        prop = Fresnel(
            grid=PixelGrid.pupil(npix),
            distance_m=0.0,
            beam_diameter_m=DIAM_M,
            wavelength_nm=WL,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.INTERMEDIATE,
        )
        out = prop.forward(field)
        np.testing.assert_allclose(
            np.asarray(out.data),
            np.asarray(field.data),
            rtol=0,
            atol=float(
                ulps(24 * int(np.ceil(np.log2(npix))), jnp.max(jnp.abs(field.data)))
            ),
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
        out = _fresnel(PixelGrid.pupil(npix), alpha).forward(field)
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
            out = _fresnel(PixelGrid.pupil(npix), alpha).forward(field)
            broaden = _rms_radius(out.data, 1.0 / npix) / rms0
            np.testing.assert_allclose(broaden, np.sqrt(1.0 + ratio**2), rtol=1e-3)

    def test_matches_direct_chirp_convolution(self):
        """The FFT propagator equals an independent direct chirp convolution."""
        npix = 256
        alpha = 4e-3
        field = _pupil(_gaussian(npix, w0=0.1, carrier_nu0=2.0), npix)
        out = _fresnel(PixelGrid.pupil(npix), alpha).forward(field)
        ref = _direct_fresnel(field.data, 1.0 / npix, alpha)
        rel = np.linalg.norm(np.asarray(out.data) - ref) / np.linalg.norm(ref)
        assert rel < 5e-3

    def test_rejects_wrong_plane(self):
        npix = 64
        focal = Field(
            data=jnp.ones((npix, npix), complex),
            grid=PixelGrid.pupil(npix),
            plane=PlaneKind.FOCAL,
        )
        with pytest.raises(ValueError):
            _fresnel(PixelGrid.pupil(npix), 1e-3).forward(focal)

    def test_rejects_grid_mismatch(self):
        npix = 64
        field = _pupil(_asymmetric_field(npix), npix)
        prop = _fresnel(PixelGrid.pupil(128), 1e-3)
        with pytest.raises(ValueError, match="grid"):
            prop.forward(field)

    def test_composes_as_a_relay_in_optical_path(self):
        """Two Fresnel relay hops then a Fraunhofer reach the focal plane."""
        npix = 64
        pupil = PixelGrid.pupil(npix)
        focal = PixelGrid.focal(32, 0.5, TEST_FRAME)
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
        pupil = PixelGrid.pupil(npix)
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
        grid = PixelGrid.pupil(256)
        big_alpha = 5e-2  # p ~ 0.28
        with pytest.warns(UserWarning, match="undersampled"):
            prop = _fresnel(grid, big_alpha, on_undersampled="warn")
        assert prop.sampling_parameter < 1.0
        with pytest.raises(ValueError, match="sampl"):
            _fresnel(grid, big_alpha, on_undersampled="raise")

    def test_rejects_bad_undersampled_policy(self):
        with pytest.raises(ValueError, match="raise/warn/record"):
            _fresnel(PixelGrid.pupil(64), 1e-3, on_undersampled="explode")


class TestPlaneWaveEigenfunction:
    """Plane waves are eigenfunctions of the angular-spectrum transfer.

    A carrier exp(2 pi i (nx x + ny y)) with (nx, ny) on the frequency
    lattice must emerge multiplied by exactly H(nx, ny) -- the
    machine-precision pin of the fftfreq-vs-coordinate convention (the
    Talbot and tilt tests bound the frequency metric only at the 1e-2/1e-3
    level). Asymmetric (nx, ny) with a sign catches transposes and axis
    flips; odd npix guards the no-fftshift bookkeeping."""

    @pytest.mark.parametrize("npix", [128, 127])
    @pytest.mark.parametrize("method", ["fresnel", "exact"])
    def test_carrier_picks_up_exactly_the_transfer_phase(self, npix, method):
        """A unit carrier passes through two 2D FFTs with six roundings per butterfly
        per axis.

        The exact reference also rounds eight operations at its full double-precision
        phase.

        Args:
            npix: Grid samples along one axis.
            method: Angular spectrum approximation under test.
        """
        nx, ny = 3.0, -2.0
        alpha = 2e-3
        x = _coords(npix, 1.0 / npix)
        xg, yg = np.meshgrid(x, x)
        carrier = jnp.asarray(np.exp(2j * np.pi * (nx * xg + ny * yg)))
        prop = _fresnel(PixelGrid.pupil(npix), alpha, method=method)
        out = prop.forward(_pupil(carrier, npix))
        nu2 = nx**2 + ny**2
        if method == "fresnel":
            transfer = np.exp(-1j * np.pi * alpha * nu2)
        else:
            lam_m = WL * 1e-9
            beta = lam_m / DIAM_M
            gamma = _distance_for_alpha(alpha) / DIAM_M
            transfer = np.exp(
                2j * np.pi * (gamma / beta) * np.sqrt(1.0 - beta**2 * nu2)
            )
        budget = 24 * int(np.ceil(np.log2(npix))) * eps(carrier.dtype)
        if method == "exact":
            # The independent double reference rounds the unreduced carrier angle.
            budget += 8 * eps(np.float64) * (2 * np.pi * abs(gamma / beta))
        np.testing.assert_allclose(
            np.asarray(out.data), np.asarray(carrier) * transfer, rtol=0, atol=budget
        )

    def test_backward_applies_the_conjugate_phase(self):
        """A unit carrier passes through two 2D FFTs with six roundings per butterfly
        per axis.
        """
        npix = 128
        nx, ny = 5.0, 1.0
        alpha = 3e-3
        x = _coords(npix, 1.0 / npix)
        xg, yg = np.meshgrid(x, x)
        carrier = jnp.asarray(np.exp(2j * np.pi * (nx * xg + ny * yg)))
        prop = _fresnel(PixelGrid.pupil(npix), alpha)
        out = prop.backward(
            Field(
                data=carrier, grid=PixelGrid.pupil(npix), plane=PlaneKind.INTERMEDIATE
            )
        )
        transfer = np.exp(1j * np.pi * alpha * (nx**2 + ny**2))
        np.testing.assert_allclose(
            np.asarray(out.data),
            np.asarray(carrier) * transfer,
            rtol=0,
            atol=24 * int(np.ceil(np.log2(npix))) * eps(carrier.dtype),
        )


class TestFresnelChromatic:
    def test_matches_per_slice_mono(self):
        """Each chromatic slice propagates with its own lambda-scaled alpha.

        Four 2D FFTs allow six roundings per axis and butterfly stage.
        """
        npix = 128
        spectrum = Quadrature.build(
            (Band(500.0 * (1 - 0.2 / 2), 500.0 * (1 + 0.2 / 2)),), order=3
        )  # 450, 500, 550 nm
        base = _pupil(_gaussian(npix, 0.15, carrier_nu0=2.0), npix)
        chrom = broadcast_to_spectrum(base, spectrum)
        distance_m = _distance_for_alpha(2e-3)
        prop = Fresnel(
            grid=PixelGrid.pupil(npix),
            distance_m=distance_m,
            beam_diameter_m=DIAM_M,
            wavelength_nm=500.0,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.INTERMEDIATE,
            on_undersampled="record",
        )
        out = prop.forward(chrom)
        assert out.data.shape == (3, npix, npix)
        for k, wl in enumerate(np.asarray(spectrum.nodes_nm[0])):
            mono = Fresnel(
                grid=PixelGrid.pupil(npix),
                distance_m=distance_m,
                beam_diameter_m=DIAM_M,
                wavelength_nm=float(wl),
                plane_in=PlaneKind.PUPIL,
                plane_out=PlaneKind.INTERMEDIATE,
                on_undersampled="record",
            ).forward(base)
            np.testing.assert_allclose(
                np.asarray(out.data[k]),
                np.asarray(mono.data),
                rtol=0,
                atol=float(
                    ulps(48 * int(np.ceil(np.log2(npix))), jnp.max(jnp.abs(mono.data)))
                ),
            )

    def test_chromatic_round_trip(self):
        """Four 2D FFTs allow six roundings per axis and butterfly stage."""
        npix = 128
        spectrum = Quadrature.build(
            (Band(600.0 * (1 - 0.15 / 2), 600.0 * (1 + 0.15 / 2)),), order=4
        )
        chrom = broadcast_to_spectrum(_pupil(_gaussian(npix, 0.15), npix), spectrum)
        prop = Fresnel(
            grid=PixelGrid.pupil(npix),
            distance_m=_distance_for_alpha(2e-3),
            beam_diameter_m=DIAM_M,
            wavelength_nm=600.0,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.INTERMEDIATE,
            on_undersampled="record",
        )
        back = prop.backward(prop.forward(chrom))
        np.testing.assert_allclose(
            np.asarray(back.data),
            np.asarray(chrom.data),
            rtol=0,
            atol=float(
                ulps(48 * int(np.ceil(np.log2(npix))), jnp.max(jnp.abs(chrom.data)))
            ),
        )

    def test_gate_uses_reddest_wavelength(self):
        """The construction gate keys on high_nm (alpha grows with lambda)."""
        grid = PixelGrid.pupil(256)
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
        p_red = Fresnel(**common, high_nm=700.0).sampling_parameter
        p_blue = Fresnel(**common, high_nm=400.0).sampling_parameter
        assert p_red < p_blue


class TestFresnelExact:
    def test_exact_matches_paraxial_at_beam_scale(self):
        """At the design beam scale (beta << 1) exact and paraxial agree.

        The analytic 1e-8 truncation band adds roundoff from the two FFT pairs.
        """
        npix = 128
        field = _pupil(_gaussian(npix, 0.12, carrier_nu0=2.0), npix)
        alpha = 3e-3
        paraxial = _fresnel(PixelGrid.pupil(npix), alpha, method="fresnel").forward(
            field
        )
        exact = _fresnel(PixelGrid.pupil(npix), alpha, method="exact").forward(field)
        rel = np.linalg.norm(
            np.asarray(exact.data) - np.asarray(paraxial.data)
        ) / np.linalg.norm(np.asarray(paraxial.data))
        assert rel < 1e-8 + 48 * int(np.ceil(np.log2(npix))) * eps(exact.data.dtype)

    def test_exact_diverges_from_paraxial_at_wide_angle(self):
        """A small beam (large beta) makes the non-paraxial term measurable."""
        npix = 128
        pupil = PixelGrid.pupil(npix)
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
        it does not clamp to |H| = 1 or amplify.

        Complex exponential and magnitude allow four roundings at unit modulus.
        """
        npix = 64
        prop = Fresnel(
            grid=PixelGrid.pupil(npix),
            distance_m=1e-4,
            beam_diameter_m=1e-5,  # ~10 micron beam: beta * nu_max > 1
            wavelength_nm=WL,
            plane_in=PlaneKind.PUPIL,
            plane_out=PlaneKind.INTERMEDIATE,
            on_undersampled="record",
            method="exact",
        )
        # At axial spatial mode 1 the exact propagating/evanescent switch
        # occurs at lambda = beam diameter (the longest cutoff on this grid).
        breaks = prop.spectral_breaks_nm()
        assert breaks[-1] == prop.beam_diameter_m * m2nm
        assert breaks[0] < WL < breaks[-1]
        transfer = np.asarray(prop._transfer(WL, npix))
        mag = np.abs(transfer)
        assert mag.max() <= 1.0 + 4 * eps(mag.dtype)  # never amplifies
        assert mag.min() < 0.9  # an evanescent (attenuating) region exists

    @pytest.mark.parametrize("cutoff", [500.0, 500.00001])
    def test_grazing_mode_static_and_traced(self, cutoff):
        """Compare either side of cutoff to the binary64 angular spectrum.

        Subtraction is exact near cutoff by Sterbenz's lemma. Allow sixteen
        roundings for the radicand, root and transfer, scaled by the
        unwrapped phase or attenuation exponent.

        Args:
            cutoff: Axial cutoff, either exactly stored or split in float32.
        """
        n = 8
        spacing = float(np.spacing(np.float32(cutoff)))
        prop = Fresnel(
            PixelGrid.pupil(n),
            distance_m=0.001,
            beam_diameter_m=cutoff / m2nm,
            wavelength_nm=cutoff,
            method="exact",
            on_undersampled="record",
        )
        cutoff = prop.spectral_breaks_nm()[-1]

        def mode(wl):
            return prop._transfer(wl, n)[0, 1]

        compiled = jax.jit(mode)
        static_compiled = jax.jit(mode, static_argnums=0)
        for wl in (
            cutoff - spacing / 4,
            cutoff,
            cutoff + spacing / 4,
            500.0,
            500.0 + spacing,
        ):
            node = jnp.asarray(wl)
            for represented, actual in (
                (wl, mode(wl)),
                (wl, static_compiled(wl)),
                (float(node), mode(node)),
                (float(node), compiled(node)),
            ):
                # Factor the difference of squares without rounding a ratio near 1.
                vertical = np.sqrt(
                    complex((cutoff - represented) * (cutoff + represented))
                )
                phase = (
                    2
                    * np.pi
                    * prop.distance_m
                    * m2nm
                    * vertical
                    / (cutoff * represented)
                )
                np.testing.assert_allclose(
                    actual,
                    np.exp(1j * phase),
                    rtol=0,
                    atol=16 * eps(actual.dtype) * (1 + abs(phase)),
                )

    @pytest.mark.parametrize("tagged_mono", [True, False])
    def test_spectral_construction_preserves_host_grazing_detuning(self, tagged_mono):
        """Host spectral metadata preserves phase and decay through a cutoff.

        The host phase and the two length-eight FFTs contribute eight
        arithmetic roundings and six radix-two stages at unit field scale.

        Args:
            tagged_mono: Read one exact band tag or bind two original host nodes.
        """
        cutoff = 500.0
        spacing = float(np.spacing(np.float32(cutoff)))
        nodes = (
            (cutoff + spacing / 4,)
            if tagged_mono
            else (
                cutoff - spacing / 4,
                cutoff + spacing / 4,
            )
        )
        prop = Fresnel(
            PixelGrid.pupil(8),
            distance_m=0.001,
            beam_diameter_m=cutoff / m2nm,
            wavelength_nm=cutoff,
            method="exact",
            on_undersampled="record",
            **({} if tagged_mono else {"spectral_nodes_nm": nodes}),
        )
        rule = (
            Quadrature.monochromatic(nodes[0])
            if tagged_mono
            else Quadrature((Band(499.0, 501.0),), [nodes], [[1.0, 1.0]])
        )
        mono = _pupil(
            jnp.broadcast_to(jnp.exp(2j * jnp.pi * jnp.arange(8) / 8), (8, 8)), 8
        )
        field = broadcast_to_spectrum(mono, rule)
        wl = np.asarray(nodes)
        root = np.sqrt(((cutoff - wl) * (cutoff + wl)).astype(complex))
        expected = np.exp(2j * np.pi * prop.distance_m * m2nm * root / (cutoff * wl))
        for evaluate in (prop.forward, jax.jit(lambda f: prop.forward(f))):
            out = evaluate(field)
            reference = expected[:, None, None] * np.asarray(mono.data)
            np.testing.assert_allclose(
                out.data,
                reference,
                rtol=0,
                atol=ulps(8 + 6 * 4, out.data.real.dtype.type(1)),
            )

    def test_exact_phase_at_evaluation_switch(self):
        """Resolve a non-DC phase after millions of carrier cycles.

        The host reference reduces binary64 cycles before evaluating the
        exponential. Its budget is eight binary64 roundings of the original
        path cycles (including the root with condition number four). The
        traced float32 budget counts 32 second-order roundings from the
        compensated phase arithmetic, plus eight at the reduced angle.
        The analytic wavelength tangent also budgets eight roundings for
        the squared wavelength, root and scale.
        """
        n = 8
        switch = 512.0
        cutoff = switch / np.sqrt(0.75)
        prop = Fresnel(
            PixelGrid.pupil(n),
            distance_m=2.123456789,
            beam_diameter_m=cutoff / m2nm,
            wavelength_nm=switch,
            method="exact",
            on_undersampled="record",
        )
        cutoff = prop.spectral_breaks_nm()[-1]

        def mode(wl):
            return prop._transfer(wl, n)[0, 1]

        def reference(wl, unit):
            path_cycles = prop.distance_m * m2nm / wl
            root = np.sqrt((1 - wl / cutoff) * (1 + wl / cutoff))
            cycles = path_cycles * root
            phasor = np.exp(2j * np.pi * np.fmod(cycles, 1))
            atol = (
                2
                * np.pi
                * (
                    (8 * np.finfo(np.float64).eps + 32 * unit**2) * path_cycles
                    + 8 * unit
                )
            )
            derivative = -2j * np.pi * path_cycles / (wl * root) * phasor
            return phasor, derivative, atol

        compiled = jax.jit(mode)
        static_compiled = jax.jit(mode, static_argnums=0)
        spacing = float(np.spacing(np.float32(switch)))
        for wl in (
            float(np.nextafter(np.float32(switch), np.float32(0))),
            switch - spacing / 16,
            switch,
            switch + spacing / 16,
            float(np.nextafter(np.float32(switch), np.float32(np.inf))),
        ):
            node = jnp.asarray(wl)
            unit = eps(node.dtype)
            expected, _, atol = reference(wl, unit)
            for static_fn in (mode, static_compiled):
                np.testing.assert_allclose(static_fn(wl), expected, rtol=0, atol=atol)
            expected, tangent_reference, atol = reference(float(node), unit)
            for fn in (mode, compiled):
                actual, tangent = jax.jvp(fn, (node,), (jnp.ones_like(node),))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)
                np.testing.assert_allclose(
                    tangent, tangent_reference, rtol=atol + 8 * unit, atol=0
                )

    def test_chromatic_carrier_retains_fractional_cycle(self):
        """A traced spectral node resolves a subcycle after millions of cycles.

        Wavelengths are exactly representable in both lanes. At zero spatial
        frequency the host remainder is an independent float64 reference. The
        reduced remainders, division, angle and phasor allow eight roundings.
        The analytic wavelength tangent adds a square and a scale to that
        phase budget.
        """
        prop = Fresnel(
            PixelGrid.pupil(8),
            distance_m=2.123456789,
            beam_diameter_m=DIAM_M,
            wavelength_nm=WL,
            method="exact",
            on_undersampled="record",
        )
        wavelengths = np.array([499.0, 500.0, 512.5])
        reference = np.exp(
            2j * np.pi * np.fmod(prop.distance_m * m2nm, wavelengths) / wavelengths
        )
        carrier = jax.vmap(lambda wl: prop._transfer(wl, 8)[0, 0])
        for fn in (carrier, jax.jit(carrier)):
            actual = fn(jnp.asarray(wavelengths))
            np.testing.assert_allclose(
                actual,
                reference,
                atol=8 * eps(actual.dtype) * 2 * np.pi,
                rtol=0,
            )

        nodes = jnp.asarray(wavelengths)
        _, tangent = jax.jvp(carrier, (nodes,), (jnp.ones_like(nodes),))
        expected_tangent = (
            -2j * np.pi * prop.distance_m * m2nm / wavelengths**2 * reference
        )
        np.testing.assert_allclose(
            tangent,
            expected_tangent,
            rtol=(8 * 2 * np.pi + 2) * eps(tangent.dtype),
            atol=0,
        )

    def test_chromatic_path_above_single_precision_nanometer_range(self):
        """A finite path can exceed the storage range in nanometers.

        A binary wavelength gives an integer number of cycles for this path.
        """
        prop = Fresnel(
            PixelGrid.pupil(4),
            distance_m=2.0**100,
            beam_diameter_m=2.0**50,
            wavelength_nm=2.0**28,
            method="exact",
            on_undersampled="record",
        )
        node = jnp.asarray(prop.wavelength_nm)
        assert prop._transfer(node, 4)[0, 0] == 1
        assert jax.jit(lambda w: prop._transfer(w, 4)[0, 0])(node) == 1

    def test_rejects_bad_method(self):
        with pytest.raises(ValueError, match="method"):
            _fresnel(PixelGrid.pupil(64), 1e-3, method="quantum")


class TestFresnelPadding:
    def test_padding_preserves_a_well_contained_result(self):
        """For a field that decays before the edge, padding does not change the
        (already wrap-free) result."""
        npix = 128
        field = _pupil(_gaussian(npix, 0.08, carrier_nu0=2.0), npix)
        unpadded = _fresnel(PixelGrid.pupil(npix), 3e-3, npad=1).forward(field)
        padded = _fresnel(PixelGrid.pupil(npix), 3e-3, npad=2).forward(field)
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
        padded = _fresnel(PixelGrid.pupil(npix), alpha, npad=npad).forward(
            _pupil(field_small, npix)
        )
        n = npix * npad
        lo = (n - npix) // 2
        big = np.zeros((n, n), complex)
        big[lo : lo + npix, lo : lo + npix] = np.asarray(field_small)
        big_grid = PixelGrid.regular((n, n), 1.0 / npix, "pupil_diameter")
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
            _fresnel(PixelGrid.pupil(64), 1e-3, npad=0)
