"""Analytic anchors for the Roman compact coronagraph train."""

import jax
import numpy as np
import pytest

from physicaloptix.instruments.roman import RomanCompact, proper_trim

jax.config.update("jax_enable_x64", True)

N_SMALL, N_BIG, D_PIX = 256, 192, 100.0


def _disk(n, diam_pix):
    idx = np.arange(n) - n // 2
    return (np.hypot(*np.meshgrid(idx, idx)) <= diam_pix / 2).astype(float)


def _clear_model(**kwargs):
    defaults = dict(
        pupil=_disk(N_SMALL, D_PIX),
        lyot=np.ones((N_SMALL, N_SMALL)),
        pupil_mask=None,
        fpm=None,
        kind="spc",
        n_small=N_SMALL,
        n_big=N_BIG,
        pupil_diam_pix=D_PIX,
        fpm_sampling_lod=0.1,
        fpm_lam0_nm=825.0,
        lam0_nm=825.0,
    )
    defaults.update(kwargs)
    return RomanCompact(**defaults)


@pytest.mark.parametrize(
    "shape,n,expected_origin",
    [
        ((5, 5), 3, (1, 1)),
        ((6, 6), 3, (2, 2)),
        ((5, 5), 4, (0, 0)),
        ((4, 4), 7, (-1, -1)),
        ((3, 3), 6, (-2, -2)),
    ],
)
def test_proper_trim_is_integer_centered(shape, n, expected_origin):
    a = np.arange(np.prod(shape), dtype=float).reshape(shape)
    out = np.asarray(proper_trim(a, n))
    oy, ox = expected_origin
    for j in range(n):
        for i in range(n):
            src_j, src_i = j + oy, i + ox
            inside = 0 <= src_j < shape[0] and 0 <= src_i < shape[1]
            assert out[j, i] == (a[src_j, src_i] if inside else 0.0)


def test_entrance_is_normalized_to_unit_total_intensity():
    stages = _clear_model().propagate(825.0, 0.0, 0.0, output_dim=65)
    assert np.sum(np.abs(stages["entrance"]) ** 2) == pytest.approx(1.0, rel=1e-12)


def test_clear_train_conserves_energy_over_the_full_image():
    n = 257  # odd full grid: the image transform is then unitary on the full frame
    model = _clear_model(pupil=_disk(n, D_PIX), lyot=np.ones((n, n)), n_small=n)
    stages = model.propagate(825.0, 0.0, 0.0, output_dim=n)
    assert np.sum(np.abs(stages["image"]) ** 2) == pytest.approx(1.0, rel=1e-9)


def _centroid(img, cx0, cy0, radius=4):
    idx = np.arange(img.shape[0]) - img.shape[0] // 2
    x, y = np.meshgrid(idx, idx)
    w = img * (np.hypot(x - cx0, y - cy0) <= radius)
    return (w * x).sum() / w.sum(), (w * y).sum() / w.sum()


def test_source_x_offset_lands_on_output_columns_after_the_transpose():
    stages = _clear_model().propagate(825.0, 3.0, 0.0, output_dim=65)
    expected = 3.0 * N_SMALL / D_PIX
    cx, cy = _centroid(np.abs(stages["image"]) ** 2, expected, 0.0)
    assert cx == pytest.approx(expected, abs=0.02)
    assert cy == pytest.approx(0.0, abs=1e-9)


def test_source_y_offset_lands_on_output_rows():
    stages = _clear_model().propagate(825.0, 0.0, -2.0, output_dim=65)
    expected = -2.0 * N_SMALL / D_PIX
    cx, cy = _centroid(np.abs(stages["image"]) ** 2, 0.0, expected)
    assert cy == pytest.approx(expected, abs=0.02)
    assert cx == pytest.approx(0.0, abs=1e-9)


def test_source_offset_scales_with_lam0_over_lambda():
    stages = _clear_model().propagate(700.0, 3.0, 0.0, output_dim=65)
    # the image grid is fixed in lambda/D at the propagated wavelength
    expected = 3.0 * 825.0 / 700.0 * N_SMALL / D_PIX
    cx, _ = _centroid(np.abs(stages["image"]) ** 2, expected, 0.0)
    assert cx == pytest.approx(expected, abs=0.02)


def test_fresnel_round_trip_returns_the_entrance_field():
    stages = _clear_model().propagate(825.0, 1.0, 0.5, output_dim=65)
    back = stages["back_to_dm1"]
    np.testing.assert_allclose(
        back, np.asarray(proper_trim(stages["entrance"], N_BIG)), atol=1e-12
    )


def _mft2(field_in, dout, diam, nout, direction):
    """Numpy statement of the prescription's matrix Fourier transform (xc = yc = 0)."""
    n = field_in.shape[1]
    x = np.arange(n) - n // 2
    u = (np.arange(nout) - nout // 2) * (dout / diam)
    expxu = dout / diam * np.exp(direction * 2j * np.pi * np.outer(x, u))
    expyv = np.exp(direction * 2j * np.pi * np.outer(x, u)).T
    return expyv @ field_in @ expxu


def _asymmetric_mask(m):
    idx = np.arange(m) - m // 2
    x, y = np.meshgrid(idx, idx)
    return ((x > 2) | ((y > 5) & (x > -4))).astype(float)


def test_spc_focal_mask_matches_the_prescription_transform_pair():
    m, dout = 41, 0.5
    fpm = _asymmetric_mask(m)
    stages = _clear_model(fpm=fpm, fpm_sampling_lod=dout).propagate(
        825.0, 0.4, -0.3, output_dim=65
    )
    at_mask = _mft2(np.asarray(stages["back_to_dm1"]), dout, D_PIX, m, +1)
    exit_ = _mft2(at_mask * fpm, dout, D_PIX, int(D_PIX), -1)
    np.testing.assert_allclose(
        stages["fpm"], at_mask, atol=1e-12 * np.abs(at_mask).max()
    )
    expected = np.asarray(proper_trim(exit_, N_SMALL))
    np.testing.assert_allclose(
        stages["fpm_exit"], expected, atol=1e-12 * np.abs(exit_).max()
    )


@pytest.mark.parametrize("n_big,diam", [(191, 100.0), (192, 101.0), (192, 100.5)])
def test_spc_mask_rejects_a_return_grid_of_other_parity(n_big, diam):
    # the return grid is pupil_diam_pix samples wide; a parity mismatch with
    # n_big leaves an uncancelled half-pixel ramp in the exit field
    with pytest.raises(ValueError, match="pupil_diam_pix"):
        _clear_model(
            fpm=_asymmetric_mask(41),
            n_big=n_big,
            pupil_diam_pix=diam,
            pupil=_disk(N_SMALL, diam),
        )


@pytest.mark.parametrize("name", ["dm1_surface_m", "dm2_surface_m"])
def test_dm_surface_must_match_the_pupil_grid(name):
    with pytest.raises(ValueError, match=name):
        _clear_model(**{name: np.zeros((N_SMALL - 1, N_SMALL - 1))})


def test_hlc_patterned_complex_mask_matches_the_prescription_babinet_form():
    m, dout = 41, 0.5
    idx = np.arange(m) - m // 2
    x, y = np.meshgrid(idx, idx)
    clear = 0.95 * np.exp(0.1j)
    fpm = np.full((m, m), clear, dtype=complex)
    spot = (np.hypot(x - 1, y + 2) < 6) | ((x > 4) & (np.abs(y) < 2))
    fpm[spot] = 0.2 * np.exp(0.7j + 0.05j * x[spot])
    model = _clear_model(kind="hlc", fpm=fpm, fpm_sampling_lod=dout, n_big=N_SMALL)
    stages = model.propagate(825.0, 0.4, -0.3, output_dim=65)
    field = np.asarray(stages["back_to_dm1"]) * clear
    at_mask = _mft2(field, dout, D_PIX, m, +1)
    region = np.real(fpm) != np.real(clear)
    corrected = field + _mft2(at_mask * region * (fpm - 1.0), dout, D_PIX, N_SMALL, -1)
    np.testing.assert_allclose(
        stages["fpm"], at_mask, atol=1e-12 * np.abs(at_mask).max()
    )
    np.testing.assert_allclose(
        stages["fpm_exit"], corrected, atol=1e-12 * np.abs(corrected).max()
    )


def _prescription_focus(lyot_field, output_dim):
    """Numpy form of the prescription's last step (n * ffts(-1), .T, trim)."""
    n = lyot_field.shape[0]
    w = np.roll(np.roll(n * np.asarray(lyot_field), -n // 2, 0), -n // 2, 1)
    w = np.fft.fft2(w) / w.size
    w = np.roll(np.roll(w, n // 2, 0), n // 2, 1).T
    return np.asarray(proper_trim(w, output_dim))


def test_image_matches_the_prescription_focus_including_phase():
    stages = _clear_model().propagate(825.0, 1.3, -0.7, output_dim=65)
    expected = _prescription_focus(stages["lyot"], 65)
    np.testing.assert_allclose(
        stages["image"], expected, atol=1e-12 * np.abs(expected).max()
    )


def test_entrance_tilt_matches_the_prescription_integer_centered_phase():
    sx, sy, wl = 1.3, -0.7, 700.0
    stages = _clear_model().propagate(wl, sx, sy, output_dim=65)
    n = N_SMALL
    pupil = _disk(n, D_PIX)
    pupil = pupil / np.sqrt(np.sum(pupil**2))
    r = (np.arange(n) - n // 2) / (D_PIX / 2.0)
    x = np.tile(r, (n, 1))
    xtilt, ytilt = sy * 825.0 / wl, sx * 825.0 / wl
    expected = pupil * np.exp(1j * np.pi * (xtilt * x + ytilt * x.T))
    np.testing.assert_allclose(stages["entrance"], expected, atol=1e-14)


def test_dm_surface_piston_applies_twice_the_surface_as_phase():
    surface = np.full((N_SMALL, N_SMALL), 20e-9)
    wl = 825.0
    plain = _clear_model().propagate(wl, 0.0, 0.0, output_dim=65)
    with_dm = _clear_model(dm1_surface_m=surface).propagate(wl, 0.0, 0.0, output_dim=65)
    ratio = (
        with_dm["dm1"][N_SMALL // 2, N_SMALL // 2]
        / plain["entrance"][N_SMALL // 2, N_SMALL // 2]
    )
    assert ratio == pytest.approx(
        np.exp(2j * np.pi * 2 * 20e-9 / (wl * 1e-9)), rel=1e-12
    )


def test_dm2_surface_is_applied_at_the_dm2_plane():
    rng = np.random.default_rng(0)
    surface = 5e-9 * rng.standard_normal((N_SMALL, N_SMALL))
    wl = 825.0
    plain = _clear_model().propagate(wl, 0.0, 0.0, output_dim=65)
    with_dm = _clear_model(dm2_surface_m=surface).propagate(wl, 0.0, 0.0, output_dim=65)
    phase = np.exp(2j * np.pi * 2 * surface / (wl * 1e-9))
    np.testing.assert_allclose(with_dm["dm2"], plain["dm2"] * phase, atol=1e-15)


def test_hlc_mask_with_no_patterned_region_scales_by_the_clear_transmission():
    c = 0.9 * np.exp(0.3j)
    fpm = np.full((41, 41), c, dtype=complex)
    model = _clear_model(kind="hlc", fpm=fpm, n_big=N_SMALL)
    plain = _clear_model(n_big=N_SMALL).propagate(825.0, 0.0, 0.0, output_dim=65)
    out = model.propagate(825.0, 0.0, 0.0, output_dim=65)
    np.testing.assert_allclose(out["image"], c * plain["image"], atol=1e-14)
