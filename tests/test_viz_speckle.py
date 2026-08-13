"""plot_process / plot_speckle_ensemble / plot_field_ellipse contracts."""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

# The base install is deliberately eyepiece-free, so this module is only
# collectible when the viz extra is present.
pytest.importorskip("eyepiece")

from physicaloptix.speckle import SpeckleMoments, SpeckleProcess
from physicaloptix.viz import plot_process, plot_speckle_ensemble

# Deliberately far from 1. A contrast delta and a raw intensity differ by
# exactly this factor, so any law overlaid without BOTH halves of the change
# of variables is wrong by six orders of magnitude -- not by a tolerance.
_NORM = 1.0e6
_I_C = 4.0
_GAMMA = 1.0


def _circular_draws(n=200_000, seed=5):
    """Contrast deltas from an exactly circular (P = 0) speckle field.

    Built from the physics, not from the laws being tested: a complex
    Gaussian of variance ``gamma / 2`` per quadrature added to a
    deterministic ``E_nom`` gives a total intensity that is modified Rician
    with ``(Ic, Is) = (i_c, gamma)`` by construction.
    """
    rng = np.random.default_rng(seed)
    e_nom = np.sqrt(_I_C)
    noise = rng.normal(scale=np.sqrt(_GAMMA / 2.0), size=(n, 2))
    intensity = (e_nom + noise[:, 0]) ** 2 + noise[:, 1] ** 2
    return (intensity - _I_C) / _NORM


_CIRCULAR_PARAMS = {
    "i_c": _I_C,
    "gamma": _GAMMA,
    "p": 0.0 + 0.0j,
    "phi_c": 0.0,
    "normalization": _NORM,
}


def _process(npix=8, m=3, knee_hz=1e-4, slope=-2.0, **kwargs):
    rng = np.random.default_rng(3)
    e_nom = jnp.asarray(
        rng.normal(size=(npix, npix)) + 1j * rng.normal(size=(npix, npix))
    )
    g = jnp.asarray(
        0.01
        * (rng.normal(size=(m, npix, npix)) + 1j * rng.normal(size=(m, npix, npix)))
    )
    return SpeckleProcess(
        e_nom,
        g,
        per_mode_rms=1.0,
        knee_hz=knee_hz,
        slope=slope,
        input_energy=1.0,
        pixel_scale_lod=0.25,
        **kwargs,
    )


def test_drawn_psd_matches_the_process_at_the_plotted_frequencies():
    """Read the abscissa back off the artist, then ask the process itself.

    Recomputing the source's own expression would pass even if the panel
    plotted the wrong frequency grid; deriving the expectation FROM the
    plotted x values is what makes a wrong grid fail.
    """
    process = _process()
    result = plot_process(process)
    line = result.artists["psd"]
    f_plotted = np.asarray(line.get_xdata())
    y_plotted = np.asarray(line.get_ydata())

    from_process = np.asarray(process.psd(jnp.asarray(f_plotted)))
    np.testing.assert_allclose(
        y_plotted, from_process / from_process.sum(), rtol=1e-10, atol=0.0
    )
    plt.close("all")


def test_line_weights_curve_is_the_processes_own_weights():
    process = _process()
    result = plot_process(process)
    y_plotted = np.asarray(result.artists["weights"].get_ydata())

    weights = np.asarray(process.line_weights())[0]
    np.testing.assert_allclose(y_plotted, weights / weights.sum(), rtol=1e-10, atol=0.0)
    plt.close("all")


def test_weights_and_psd_shares_actually_differ():
    """Guard the panel's whole point: df weighting is not a no-op.

    If these two curves coincided the panel would be decoration. They differ
    because the grid is logarithmic, which is the pre-2026-07-25 bug story.
    """
    process = _process()
    result = plot_process(process)
    psd_share = np.asarray(result.artists["psd"].get_ydata())
    weight_share = np.asarray(result.artists["weights"].get_ydata())
    assert not np.allclose(psd_share, weight_share, rtol=1e-3, atol=0.0)
    plt.close("all")


def test_autocorrelation_decays_and_the_realized_curve_tracks_the_target():
    process = _process(slope=-2.0)
    result = plot_process(process)
    target = np.asarray(result.artists["target"].get_ydata())
    realized = np.asarray(result.artists["realized"].get_ydata())

    # The OU target starts at 1 and decays monotonically with lag.
    assert target[0] == pytest.approx(1.0)
    assert np.all(np.diff(target) < 0.0)
    assert target[-1] < 0.05
    # The slope -2 synthesis reproduces it to a few percent over the lags the
    # grid spans (the guide quotes ~4% at the default 64-line grid).
    assert realized[0] == pytest.approx(1.0, abs=1e-6)
    assert np.max(np.abs(realized - target)) < 0.06
    plt.close("all")


def test_target_is_heavy_and_realized_is_dashed():
    """The line-weight hierarchy is normative, not decorative."""
    process = _process()
    result = plot_process(process)
    for target_key, realized_key in (("psd", "weights"), ("target", "realized")):
        target = result.artists[target_key]
        realized = result.artists[realized_key]
        assert target.get_linewidth() > realized.get_linewidth()
        assert target.get_linestyle() == "-"
        assert realized.get_linestyle() == "--"
    plt.close("all")


def test_caller_axes_are_used_and_shape_is_enforced():
    process = _process()
    _, axes = plt.subplots(1, 2)
    result = plot_process(process, axes=axes)
    assert result.axes[0] is axes[0]
    assert result.axes[1] is axes[1]
    plt.close("all")

    _, three = plt.subplots(1, 3)
    with pytest.raises(ValueError, match=r"expected axes shape \(2,\), got \(3,\)"):
        plot_process(process, axes=three)
    plt.close("all")

    _, grid = plt.subplots(2, 2)
    with pytest.raises(ValueError, match=r"expected axes shape \(2,\), got \(2, 2\)"):
        plot_process(process, axes=grid)
    plt.close("all")


def _integral(line):
    x = np.asarray(line.get_xdata())
    y = np.asarray(line.get_ydata())
    return float(np.trapezoid(y, x))


def test_histogram_and_laws_integrate_to_the_same_total():
    """The units gate.

    A density=True histogram integrates to 1 over its own span. A law in the
    SAME units must too. A law in raw-intensity units instead of delta units
    would integrate to 1/_NORM (or _NORM), missing by six orders of
    magnitude -- which is why this asserts a tight relative tolerance rather
    than "the curves look alike".
    """
    draws = _circular_draws()
    _, ax = plt.subplots()
    result = plot_speckle_ensemble(
        draws, laws=("exact", "rician"), ax=ax, **_CIRCULAR_PARAMS
    )

    counts, edges = np.histogram(draws, bins=60, density=True)
    hist_total = float(np.sum(counts * np.diff(edges)))
    assert hist_total == pytest.approx(1.0, rel=1e-9)

    for line in result.artists["lines"]:
        assert _integral(line) == pytest.approx(hist_total, rel=0.03)
    plt.close("all")


def test_dropping_the_jacobian_would_fail_the_units_gate():
    """Pin that the gate above has teeth, not just a generous tolerance."""
    draws = _circular_draws()
    _, ax = plt.subplots()
    result = plot_speckle_ensemble(draws, laws="rician", ax=ax, **_CIRCULAR_PARAMS)
    drawn = _integral(result.artists["lines"][0])
    # The two classic mistakes, both of which produce a smooth plausible curve.
    assert not (0.97 < drawn / _NORM < 1.03)
    assert not (0.97 < drawn * _NORM < 1.03)
    plt.close("all")


def test_rician_is_the_p_to_zero_limit_of_the_exact_law():
    draws = _circular_draws(n=1000)
    _, ax = plt.subplots()
    result = plot_speckle_ensemble(
        draws, laws=("exact", "rician"), ax=ax, **_CIRCULAR_PARAMS
    )
    exact, rician = result.artists["lines"]
    ex = np.asarray(exact.get_ydata())
    ri = np.asarray(rician.get_ydata())
    live = ex > ex.max() * 1e-6
    np.testing.assert_allclose(ex[live], ri[live], rtol=2e-3)
    plt.close("all")


def test_an_improper_field_departs_from_the_rician():
    """|P| > 0 must actually change the exact law, or the overlay says nothing."""
    draws = _circular_draws(n=1000)
    params = dict(_CIRCULAR_PARAMS, p=0.9 * _GAMMA * np.exp(1j * 0.7))
    _, ax = plt.subplots()
    result = plot_speckle_ensemble(draws, laws=("exact", "rician"), ax=ax, **params)
    exact, rician = result.artists["lines"]
    ex = np.asarray(exact.get_ydata())
    ri = np.asarray(rician.get_ydata())
    live = ri > ri.max() * 1e-3
    assert np.max(np.abs(ex[live] / ri[live] - 1.0)) > 0.1
    plt.close("all")


def test_a_sequence_of_laws_draws_distinct_line_weights():
    draws = _circular_draws(n=1000)
    _, ax = plt.subplots()
    result = plot_speckle_ensemble(
        draws, laws=("exact", "rician"), ax=ax, **_CIRCULAR_PARAMS
    )
    exact, rician = result.artists["lines"]
    assert len(result.artists["lines"]) == 2
    assert exact.get_linewidth() > rician.get_linewidth()
    assert exact.get_linestyle() == "-"
    assert rician.get_linestyle() == "--"
    plt.close("all")


def _capturing_law():
    captured = {}

    def law(x_grid, params):
        captured.update(params)
        return np.zeros_like(x_grid)

    return law, captured


def _hand_moments(process, *, gamma, p):
    """A SpeckleMoments whose kernels differ unmistakably from the process's."""
    shape = np.asarray(process.e_nom).shape
    ones = jnp.ones(shape)
    return SpeckleMoments(
        mean_map=ones,
        var_map=ones,
        gamma_map=ones * gamma,
        p_map=(ones * p).astype(complex),
        var_x_map=ones,
    )


def test_process_beats_moments():
    process = _process()
    moments = _hand_moments(process, gamma=123.0, p=7.0)
    law, captured = _capturing_law()
    _, ax = plt.subplots()
    plot_speckle_ensemble(
        _circular_draws(n=500),
        laws=law,
        process=process,
        moments=moments,
        pixel=(2, 3),
        ax=ax,
    )
    from_process = process.moments()
    assert captured["gamma"] == pytest.approx(
        float(np.asarray(from_process.gamma_map)[2, 3])
    )
    assert captured["gamma"] != pytest.approx(123.0)
    plt.close("all")


def test_explicit_scalars_beat_the_process():
    process = _process()
    law, captured = _capturing_law()
    _, ax = plt.subplots()
    plot_speckle_ensemble(
        _circular_draws(n=500),
        laws=law,
        process=process,
        pixel=(2, 3),
        i_c=999.0,
        ax=ax,
    )
    assert captured["i_c"] == pytest.approx(999.0)
    # Precedence is per parameter: the rest still come from the process.
    assert captured["gamma"] == pytest.approx(
        float(np.asarray(process.moments().gamma_map)[2, 3])
    )
    plt.close("all")


def test_moments_supply_gamma_and_p_but_not_the_coherent_pair():
    process = _process()
    moments = _hand_moments(process, gamma=123.0, p=7.0)
    law, captured = _capturing_law()
    _, ax = plt.subplots()
    plot_speckle_ensemble(
        _circular_draws(n=500),
        laws=law,
        moments=moments,
        pixel=(2, 3),
        i_c=_I_C,
        phi_c=0.0,
        normalization=_NORM,
        ax=ax,
    )
    assert captured["gamma"] == pytest.approx(123.0)
    assert captured["i_c"] == pytest.approx(_I_C)
    plt.close("all")


def test_missing_parameters_name_the_moment_matching_recipe():
    with pytest.raises(ValueError, match="moment-match"):
        plot_speckle_ensemble(_circular_draws(n=100))


def test_map_sources_require_a_pixel():
    process = _process()
    with pytest.raises(ValueError, match="pixel="):
        plot_speckle_ensemble(_circular_draws(n=100), process=process)


def test_unknown_law_name_raises():
    with pytest.raises(ValueError, match="unknown law"):
        plot_speckle_ensemble(
            _circular_draws(n=100), laws="beckmann", **_CIRCULAR_PARAMS
        )


# A deliberately IMPROPER, NON-AXIS-ALIGNED case. angle(P) = 1.1 rad, so the
# ellipse must sit at 0.55 rad (31.5 deg). Using angle(P) instead of its half
# would give 63 deg -- a mistake no axis-aligned fixture can see.
_ELL_GAMMA = 1.0
_ELL_P = 0.6 * np.exp(1j * 1.1)


def _improper_draws(gamma, p, n=400_000, seed=17):
    """Complex samples with EXACTLY the requested (Gamma, P), by construction.

    Built in P's own principal frame: independent quadratures of variance
    ``(Gamma +- |P|) / 2`` rotated by ``angle(P) / 2``. Then
    ``E|z|^2 = Gamma`` and ``E[z^2] = P`` identically, so this fixture is an
    independent statement of the physics rather than a replay of the code
    under test.
    """
    rng = np.random.default_rng(seed)
    alpha = 0.5 * np.angle(p)
    w1 = rng.normal(scale=np.sqrt((gamma + abs(p)) / 2.0), size=n)
    w2 = rng.normal(scale=np.sqrt((gamma - abs(p)) / 2.0), size=n)
    return (w1 + 1j * w2) * np.exp(1j * alpha)


def _moments_at(gamma, p, shape=(4, 4)):
    ones = jnp.ones(shape)
    return SpeckleMoments(
        mean_map=ones,
        var_map=ones,
        gamma_map=ones * gamma,
        p_map=(ones * p).astype(complex),
        var_x_map=ones,
    )


def _geometry(ellipse):
    return ellipse.get_width() / 2.0, ellipse.get_height() / 2.0, ellipse.angle


def test_moments_ellipse_matches_the_closed_form_including_the_half_angle():
    from physicaloptix.viz import plot_field_ellipse

    _, ax = plt.subplots()
    result = plot_field_ellipse((1, 1), moments=_moments_at(_ELL_GAMMA, _ELL_P), ax=ax)
    semi_major, semi_minor, angle_deg = _geometry(result.artists["ellipse"])

    assert semi_major == pytest.approx(np.sqrt((_ELL_GAMMA + abs(_ELL_P)) / 2.0))
    assert semi_minor == pytest.approx(np.sqrt((_ELL_GAMMA - abs(_ELL_P)) / 2.0))
    assert angle_deg % 180.0 == pytest.approx(
        np.degrees(0.5 * np.angle(_ELL_P)) % 180.0
    )
    # The trap, stated as an assertion: the full angle is a different number.
    assert angle_deg % 180.0 != pytest.approx(np.degrees(np.angle(_ELL_P)) % 180.0)
    plt.close("all")


def test_moments_ellipse_and_draws_ellipse_agree():
    from physicaloptix.viz import plot_field_ellipse

    draws = _improper_draws(_ELL_GAMMA, _ELL_P)
    _, ax1 = plt.subplots()
    analytic = _geometry(
        plot_field_ellipse(
            (1, 1), moments=_moments_at(_ELL_GAMMA, _ELL_P), ax=ax1
        ).artists["ellipse"]
    )
    _, ax2 = plt.subplots()
    empirical = _geometry(
        plot_field_ellipse(None, draws=draws, ax=ax2).artists["ellipse"]
    )

    assert empirical[0] == pytest.approx(analytic[0], rel=0.02)
    assert empirical[1] == pytest.approx(analytic[1], rel=0.02)
    assert empirical[2] % 180.0 == pytest.approx(analytic[2] % 180.0, abs=1.5)
    plt.close("all")


def test_a_proper_field_gives_a_circle():
    from physicaloptix.viz import plot_field_ellipse

    _, ax = plt.subplots()
    result = plot_field_ellipse((1, 1), moments=_moments_at(_ELL_GAMMA, 0.0j), ax=ax)
    semi_major, semi_minor, _ = _geometry(result.artists["ellipse"])
    eccentricity = np.sqrt(1.0 - (semi_minor / semi_major) ** 2)
    assert eccentricity == pytest.approx(0.0, abs=1e-9)
    plt.close("all")


def test_cloud_arrow_and_false_alarm_circle():
    from physicaloptix.viz import plot_field_ellipse

    e_nom = 2.0 + 1.0j
    i_c = abs(e_nom) ** 2
    threshold, norm = 3.0, 5.0
    draws = _improper_draws(_ELL_GAMMA, _ELL_P, n=2000)

    _, ax = plt.subplots()
    result = plot_field_ellipse(
        (1, 1),
        moments=_moments_at(_ELL_GAMMA, _ELL_P),
        draws=draws,
        e_nom=e_nom,
        normalization=norm,
        threshold=threshold,
        ax=ax,
    )

    # The cloud is the TOTAL field, so it sits on E_nom's tip, not the origin.
    cloud = result.artists["cloud"].get_offsets()
    assert cloud.mean(axis=0)[0] == pytest.approx(e_nom.real, abs=0.05)
    assert cloud.mean(axis=0)[1] == pytest.approx(e_nom.imag, abs=0.05)
    # The ellipse is centered there too.
    assert result.artists["ellipse"].get_center() == pytest.approx(
        (e_nom.real, e_nom.imag)
    )
    # The arrow runs from the origin to E_nom.
    start, end = result.artists["arrow"]._posA_posB
    assert start == pytest.approx((0.0, 0.0))
    assert end == pytest.approx((e_nom.real, e_nom.imag))
    # The pinned false-alarm convention: origin-centered, total-intensity radius.
    circle = result.artists["circle"]
    assert circle.get_center() == pytest.approx((0.0, 0.0))
    assert circle.get_radius() == pytest.approx(np.sqrt(i_c + threshold * norm))
    plt.close("all")


def test_map_valued_inputs_are_indexed_by_pixel():
    from physicaloptix.viz import plot_field_ellipse

    gamma_map = jnp.asarray(np.arange(16, dtype=float).reshape(4, 4) + 1.0)
    ones = jnp.ones((4, 4))
    moments = SpeckleMoments(
        mean_map=ones,
        var_map=ones,
        gamma_map=gamma_map,
        p_map=jnp.zeros((4, 4), dtype=complex),
        var_x_map=ones,
    )
    _, ax = plt.subplots()
    result = plot_field_ellipse((2, 3), moments=moments, ax=ax)
    semi_major, _, _ = _geometry(result.artists["ellipse"])
    assert semi_major == pytest.approx(np.sqrt(float(gamma_map[2, 3]) / 2.0))
    plt.close("all")

    with pytest.raises(ValueError, match="pixel="):
        plot_field_ellipse(None, moments=moments)


def test_field_ellipse_input_and_axes_validation():
    from physicaloptix.viz import plot_field_ellipse

    with pytest.raises(ValueError, match="moments= or draws="):
        plot_field_ellipse((0, 0))
    with pytest.raises(ValueError, match="normalization="):
        plot_field_ellipse((1, 1), moments=_moments_at(1.0, 0.0j), threshold=1.0)
    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match=r"single ax.*\(2,\)"):
        plot_field_ellipse((1, 1), moments=_moments_at(1.0, 0.0j), ax=axes)
    plt.close("all")


def test_viz_speckle_module_never_touches_the_private_eps():
    """The public accessor is delta_e; _eps is off limits to the viz tier."""
    import pathlib

    import physicaloptix.viz.speckle as module

    source = pathlib.Path(module.__file__).read_text()
    assert source.count("_eps") == 0


def test_field_ellipse_consumes_delta_e_from_a_real_field():
    """End-to-end: the documented cloud recipe runs on a real speckle field."""
    from physicaloptix.viz import plot_field_ellipse

    process = _process()
    field = process.draw(__import__("jax").random.key(0))
    times = np.linspace(0.0, 3600.0, 64)
    draws = np.stack(
        [np.asarray(field.delta_e(wavelength_nm=500.0, time_s=t)) for t in times]
    )
    _, ax = plt.subplots()
    result = plot_field_ellipse(
        (2, 3), draws=draws, e_nom=np.asarray(process.e_nom), ax=ax
    )
    assert result.artists["cloud"].get_offsets().shape == (len(times), 2)
    assert result.artists["ellipse"].get_width() > 0.0
    plt.close("all")
