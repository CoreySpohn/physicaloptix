"""Analytic anchors for the coating transfer-matrix module."""

import jax.numpy as jnp
import numpy as np

from physicaloptix.coatings import multilayer_response, sellmeier, thickness_kernel

WL = jnp.linspace(400.0, 700.0, 31)


class TestAnalyticAnchors:
    def test_bare_interface_fresnel(self):
        """No layers: r = (n0 - ns) / (n0 + ns), the Fresnel amplitude."""
        r, _ = multilayer_response(WL, [], [], n_substrate=1.5)
        np.testing.assert_allclose(r, (1.0 - 1.5) / (1.0 + 1.5), rtol=1e-12)

    def test_matched_slab_positive_phase(self):
        """THE sign-convention anchor: an index-matched slab is pure
        propagation, r == 0 and t == exp(+i 2 pi n d / lambda) in the
        library's e^{+ikz} convention. The textbook-convention (conjugate)
        matrix fails this with t == exp(-i ...)."""
        n, d = 1.5, 1234.5
        wl = jnp.array([550.0])
        r, t = multilayer_response(wl, [n], [d], n_incident=n, n_substrate=n)
        np.testing.assert_allclose(jnp.abs(r[0]), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            t[0], jnp.exp(1j * 2.0 * jnp.pi * n * d / 550.0), rtol=1e-10
        )

    def test_quarter_wave_layer_closed_form(self):
        """Single quarter-wave layer at wl0: r = (n0 ns - n1^2) / (n0 ns + n1^2)."""
        wl0, n1, ns = 550.0, 1.38, 1.5
        d = wl0 / (4.0 * n1)
        r, _ = multilayer_response(jnp.array([wl0]), [n1], [d], n_substrate=ns)
        expected = (1.0 * ns - n1**2) / (1.0 * ns + n1**2)
        np.testing.assert_allclose(r[0], expected, rtol=1e-10)

    def test_quarter_wave_stack_peak_reflectance(self):
        """N (high, low) pairs at wl0: R from the admittance closed form.

        Each quarter-wave layer transforms the admittance as Y -> n^2 / Y;
        iterating from the substrate through the (HL)^N stack gives
        Y = (n_h / n_l)^(2N) * ns, and R = ((n0 - Y) / (n0 + Y))^2.
        """
        wl0, n_h, n_l, ns, pairs = 550.0, 2.35, 1.38, 1.5, 6
        indices, thicknesses = [], []
        for _ in range(pairs):
            indices += [n_h, n_l]
            thicknesses += [wl0 / (4.0 * n_h), wl0 / (4.0 * n_l)]
        r, _ = multilayer_response(
            jnp.array([wl0]), indices, thicknesses, n_substrate=ns
        )
        admittance = (n_h / n_l) ** (2 * pairs) * ns
        expected_r_squared = ((1.0 - admittance) / (1.0 + admittance)) ** 2
        np.testing.assert_allclose(jnp.abs(r[0]) ** 2, expected_r_squared, rtol=1e-8)

    def test_energy_conservation_lossless(self):
        """|r|^2 + (ns/n0)|t|^2 == 1 at every wavelength for real indices."""
        r, t = multilayer_response(
            WL, [2.35, 1.38, 2.35], [60.0, 95.0, 60.0], n_substrate=1.5
        )
        np.testing.assert_allclose(
            jnp.abs(r) ** 2 + 1.5 * jnp.abs(t) ** 2, 1.0, rtol=1e-10
        )


class TestSellmeier:
    def test_fused_silica_at_reference_wavelength(self):
        """Fused silica (Malitson 1965 coefficients): n(587.6 nm) ~= 1.4585."""
        b = jnp.array([0.6961663, 0.4079426, 0.8974794])
        c = jnp.array([0.0684043**2, 0.1162414**2, 9.896161**2])
        n = sellmeier(jnp.array([587.6]), b, c)
        np.testing.assert_allclose(n[0], 1.4585, atol=2e-4)

    def test_sellmeier_layer_composes(self):
        """The intended data path: n(lambda) array feeding a layer index."""
        b = jnp.array([0.6961663, 0.4079426, 0.8974794])
        c = jnp.array([0.0684043**2, 0.1162414**2, 9.896161**2])
        n_layer = sellmeier(WL, b, c)  # (w,) array-valued layer index
        r, t = multilayer_response(WL, [n_layer], [100.0], n_substrate=1.5)
        np.testing.assert_allclose(
            jnp.abs(r) ** 2 + 1.5 * jnp.abs(t) ** 2, 1.0, rtol=1e-10
        )


class TestThicknessKernel:
    def test_matches_finite_differences(self):
        wl = jnp.linspace(500.0, 600.0, 5)
        layers, thicknesses = [2.35, 1.38], [60.0, 95.0]
        kern = thickness_kernel(wl, layers, thicknesses, layer=0, n_substrate=1.5)
        h = 1e-3
        r_plus, _ = multilayer_response(
            wl, layers, [thicknesses[0] + h, thicknesses[1]], n_substrate=1.5
        )
        r_minus, _ = multilayer_response(
            wl, layers, [thicknesses[0] - h, thicknesses[1]], n_substrate=1.5
        )
        fd = (jnp.log(r_plus) - jnp.log(r_minus)) / (2.0 * h)
        np.testing.assert_allclose(kern, fd, rtol=1e-5)

    def test_matched_slab_kernel_is_opd_law(self):
        """Sign + unit anchor tying the kernel to the OPD law: each nm of
        matched-slab thickness adds n nm of optical path, so
        d(log t)/dd == +i 2 pi n / lambda exactly."""
        n = 1.5
        wl = jnp.linspace(500.0, 600.0, 3)
        kern = thickness_kernel(
            wl, [n], [100.0], layer=0, output="t", n_incident=n, n_substrate=n
        )
        np.testing.assert_allclose(kern, 1j * 2.0 * jnp.pi * n / wl, rtol=1e-8)
