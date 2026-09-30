"""Analytic anchors for the Roman compact coronagraph train."""

import jax
import numpy as np
import pytest

from physicaloptix.instruments.roman import (
    RomanCompact,
    dm_median_volts,
    dm_strokes_from_volts,
    dm_surface,
    proper_trim,
    volts_to_stroke,
)

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


@pytest.mark.parametrize(
    "name", ["dm1_surface_m", "dm2_surface_m", "dm1_wfe_m", "dm2_wfe_m"]
)
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


# ------------------------------------------------------------------ deformable mirror

INF_DX, INF_PITCH = 1e-4, 1e-3  # synthetic influence function: 10 samples per actuator
# FFT convolution leaves rounding-level (1e-16 relative) residue, not exact zeros
FLOOR = 1e-15


def _gaussian_influence(n=91, sigma_px=8.0, lobe=0.0):
    idx = np.arange(n) - n // 2
    g = np.exp(-(idx[None, :] ** 2 + idx[:, None] ** 2) / (2 * sigma_px**2))
    # optional off-center lobe makes the function left-right asymmetric
    return g + lobe * np.exp(-((idx[None, :] - 12) ** 2 + idx[:, None] ** 2) / 18.0)


def _dm(strokes, **kwargs):
    args = dict(
        influence=_gaussian_influence(),
        influence_dx_m=INF_DX,
        influence_pitch_m=INF_PITCH,
        pitch_m=INF_PITCH,
        center_act=(3.5, 3.5),
        grid_npix=256,
        grid_dx_m=INF_DX,
    )
    args.update(kwargs)
    return np.asarray(dm_surface(strokes, **args))


def _poke(y, x, height=2e-9, shape=(8, 8)):
    strokes = np.zeros(shape)
    strokes[y, x] = height  # row index is y, column index is x
    return strokes


def test_dm_with_zero_strokes_is_flat():
    assert np.all(_dm(np.zeros((8, 8))) == 0.0)


def test_dm_surface_is_linear_in_the_strokes():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((2, 8, 8)) * 1e-9
    both = _dm(a + b)
    np.testing.assert_allclose(both, _dm(a) + _dm(b), atol=1e-14 * np.abs(both).max())


def _patch_at(surface, row, col, half=45):
    return surface[row - half : row + half + 1, col - half : col + half + 1]


def test_dm_poke_reproduces_the_influence_function_on_a_commensurate_grid():
    surface = _dm(_poke(5, 2))
    # actuator (x=2, y=5) sits (2 - 3.5, 5 - 3.5) pitches from the axis: 15 samples
    row, col = 128 + 15, 128 - 15
    patch = _patch_at(surface, row, col)
    np.testing.assert_allclose(patch, 2e-9 * _gaussian_influence(), atol=FLOOR * 2e-9)
    outside = surface.copy()
    _patch_at(outside, row, col)[:] = 0.0
    assert np.abs(outside).max() < FLOOR * 2e-9


def test_dm_scales_the_influence_function_to_the_actuator_pitch():
    # 1.25x the tabulated pitch: the function stretches by 1.25, so a grid at
    # 1.25x the tabulated sampling again lands on its samples, 15 samples out
    pitch = 1.25 * INF_PITCH
    surface = _dm(_poke(5, 2), pitch_m=pitch, grid_dx_m=1.25 * INF_DX)
    patch = _patch_at(surface, 128 + 15, 128 - 15)
    np.testing.assert_allclose(patch, 2e-9 * _gaussian_influence(), atol=FLOOR * 2e-9)


def test_dm_half_sample_offset_interpolates_with_keys_weights():
    # shifting the axis by 0.05 actuator (half a sample) puts every output sample
    # halfway between influence samples along x: Keys a = -0.5 gives
    # (-1/16, 9/16, 9/16, -1/16) on the four neighbors
    surface = _dm(_poke(5, 2), center_act=(3.55, 3.5))
    inf = _gaussian_influence()
    weights = np.array([-1.0, 9.0, 9.0, -1.0]) / 16.0
    expected = np.zeros_like(inf)
    for k, w in zip(range(-1, 3), weights, strict=True):
        expected[:, 1:-2] += w * inf[:, 1 + k : inf.shape[1] - 2 + k]
    # output column c samples influence index c - 67.5 (the actuator sits 15.5
    # samples left of the axis, the influence center at index 45), so expected
    # column j (index j + 0.5) is output column j + 68
    patch = surface[128 + 15 - 45 : 128 + 15 + 46, 69 : 69 + expected.shape[1] - 3]
    np.testing.assert_allclose(patch, 2e-9 * expected[:, 1:-2], atol=FLOOR * 2e-9)


def test_dm_flip_lr_mirrors_actuators_and_influence_but_not_the_center():
    rng = np.random.default_rng(2)
    strokes = rng.standard_normal((8, 8)) * 1e-9
    lopsided = _gaussian_influence(lobe=0.3)
    center = (3.3, 3.6)  # asymmetric: a flip that also moved the center would show
    flipped = _dm(strokes, influence=lopsided, center_act=center, flip_lr=True)
    mirrored = _dm(strokes[:, ::-1], influence=lopsided[:, ::-1], center_act=center)
    np.testing.assert_allclose(flipped, mirrored, atol=1e-14 * np.abs(mirrored).max())


def test_dm_flip_lr_about_a_symmetric_center_mirrors_the_surface():
    rng = np.random.default_rng(3)
    strokes = rng.standard_normal((8, 8)) * 1e-9
    plain = _dm(strokes, grid_npix=255)
    flipped = _dm(strokes, grid_npix=255, flip_lr=True)
    np.testing.assert_allclose(
        flipped, plain[:, ::-1], atol=1e-14 * np.abs(plain).max()
    )


def _centroid_xy(surface):
    idx = np.arange(surface.shape[0]) - surface.shape[0] // 2
    w = np.abs(surface)
    return (w * idx[None, :]).sum() / w.sum(), (w * idx[:, None]).sum() / w.sum()


def test_dm_x_tilt_foreshortens_y_by_the_cosine():
    tilt = 20.0
    # actuator (x=3, y=7): 0.5 pitches in -x, 3.5 in +y
    cx, cy = _centroid_xy(_dm(_poke(7, 3), tilt_deg=(tilt, 0.0, 0.0)))
    assert cy == pytest.approx(35.0 * np.cos(np.radians(tilt)), abs=0.02)
    assert cx == pytest.approx(-5.0, abs=0.02)


def test_dm_y_tilt_foreshortens_x_by_the_cosine():
    tilt = 20.0
    # actuator (x=7, y=4): 3.5 pitches in +x, 0.5 in +y
    cx, cy = _centroid_xy(_dm(_poke(4, 7), tilt_deg=(0.0, tilt, 0.0)))
    assert cx == pytest.approx(35.0 * np.cos(np.radians(tilt)), abs=0.02)
    assert cy == pytest.approx(5.0, abs=0.02)


def test_dm_z_tilt_is_a_left_handed_rotation_about_the_axis():
    # actuator (x=7, y=3): 3.5 pitches along +x, 0.5 along -y; +90 deg about z
    # (left-handed, the prescription's convention) carries +x onto -y
    cx, cy = _centroid_xy(_dm(_poke(3, 7), tilt_deg=(0.0, 0.0, 90.0)))
    assert cx == pytest.approx(-5.0, abs=0.05)
    assert cy == pytest.approx(-35.0, abs=0.05)


def test_dm_is_jittable_and_differentiable_in_the_strokes_and_tilts():
    kw = dict(
        influence=_gaussian_influence(),
        influence_dx_m=INF_DX,
        influence_pitch_m=INF_PITCH,
        pitch_m=INF_PITCH,
        center_act=(3.5, 3.5),
        grid_npix=64,
        grid_dx_m=2 * INF_DX,
    )

    def surface(strokes, tilt):
        return dm_surface(strokes, tilt_deg=(tilt, 0.0, 0.0), **kw)

    strokes = np.zeros((8, 8))
    jac = jax.jit(jax.jacfwd(surface))(strokes, 9.65)
    # the Jacobian column of actuator k is the surface of a unit poke at k
    np.testing.assert_allclose(
        jac[:, :, 4, 2], surface(_poke(4, 2, height=1.0), 9.65), atol=1e-14
    )
    dtilt = jax.grad(lambda t: jax.numpy.sum(surface(_poke(6, 2, 1.0), t) ** 2))(9.65)
    assert np.isfinite(dtilt) and dtilt != 0.0


def test_dm_rejects_an_even_influence_function():
    with pytest.raises(ValueError, match="odd"):
        _dm(np.zeros((8, 8)), influence=np.ones((90, 90)))


def test_dm_rejects_a_non_integer_influence_magnification():
    with pytest.raises(ValueError, match="integer"):
        _dm(np.zeros((8, 8)), influence_dx_m=1.1e-4)


def test_dm_rejects_an_influence_function_wider_than_its_margin():
    with pytest.raises(ValueError, match="margin"):
        _dm(np.zeros((8, 8)), influence=_gaussian_influence(n=201))


# ------------------------------------------------------------------ DM voltages


def _calibration(shape=(4, 4), nv=11, slope=2e-9, coupling=0.0):
    volts = np.arange(nv) * 10.0
    table = np.broadcast_to((slope * volts)[:, None, None], (nv, *shape)).copy()
    kernel = np.full((nv, *shape, 3, 3), coupling)
    return dict(
        stroke_volts=volts, stroke_table_m=table, coupling_volts=volts, coupling=kernel
    )


def _v2s(v, cal):
    return np.asarray(volts_to_stroke(v, **cal))


def test_volts_to_stroke_follows_each_actuator_table():
    cal = _calibration()
    cal["stroke_table_m"][:, 1, 2] *= 3.0  # one actuator three times as responsive
    stroke = _v2s(np.full((4, 4), 25.0), cal)
    # actuators move toward the DM with voltage: the stroke is negative
    np.testing.assert_allclose(stroke[0, 0], -2e-9 * 25.0, rtol=1e-12)
    np.testing.assert_allclose(stroke[1, 2], -3 * 2e-9 * 25.0, rtol=1e-12)


def test_volts_to_stroke_clamps_at_the_ends_of_the_table():
    stroke = _v2s(np.array([[-20.0, 130.0]]), _calibration(shape=(1, 2)))
    np.testing.assert_allclose(stroke, [[0.0, -2e-9 * 100.0]], atol=1e-30)


def test_volts_to_stroke_accepts_a_rectangular_array():
    v = np.arange(15.0).reshape(3, 5) * 4.0
    np.testing.assert_allclose(
        _v2s(v, _calibration(shape=(3, 5))), -2e-9 * v, rtol=1e-12
    )


def test_volts_to_stroke_couples_to_the_neighbors_and_trims_the_edge():
    v = np.zeros((4, 4))
    v[0, 1] = 50.0  # an edge actuator: its outer neighbors fall off the array
    stroke = _v2s(v, _calibration(coupling=0.1))
    s0 = -2e-9 * 50.0
    expected = np.zeros((4, 4))
    expected[0, 1] = s0
    for dy, dx in ((0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        expected[0 + dy, 1 + dx] += 0.1 * s0
    np.testing.assert_allclose(stroke, expected, rtol=1e-12, atol=1e-30)


def test_volts_to_stroke_interpolates_the_coupling_in_voltage():
    cal = _calibration()
    # right-neighbor coupling rises with voltage
    cal["coupling"][:, 2, 2, 1, 2] = np.linspace(0.0, 0.2, len(cal["coupling_volts"]))
    v = np.zeros((4, 4))
    v[2, 2] = 55.0
    stroke = _v2s(v, cal)
    assert stroke[2, 3] == pytest.approx(0.2 * 55.0 / 100.0 * (-2e-9 * 55.0), rel=1e-12)


def test_volts_to_stroke_rejects_malformed_calibrations():
    cal = _calibration()
    with pytest.raises(ValueError, match="stroke_table_m"):
        _v2s(np.zeros((4, 5)), cal)
    decreasing = dict(cal, stroke_volts=cal["stroke_volts"][::-1].copy())
    with pytest.raises(ValueError, match="increasing"):
        _v2s(np.zeros((4, 4)), decreasing)


def test_dm_strokes_quantize_and_remove_the_median_of_live_actuators():
    v = np.array([[10.3, 20.0, 30.0, 40.0]] * 4)
    live = np.ones((4, 4), bool)
    live[0, 0] = False
    heights = np.asarray(
        dm_strokes_from_volts(v, **_calibration(), volt_quantum=0.5, live=live)
    )
    # linear table: height = slope * (quantized volts - their live-actuator median)
    vq = np.floor(v / 0.5) * 0.5
    np.testing.assert_allclose(heights, 2e-9 * (vq - np.median(vq[live])), rtol=1e-12)


def test_dm_strokes_pass_gradients_through_the_quantization():
    v = np.full((4, 4), 23.7)
    live = np.ones((4, 4), bool)
    w = np.arange(16.0).reshape(4, 4)

    def objective(volts, quantum):
        heights = dm_strokes_from_volts(
            volts, **_calibration(), volt_quantum=quantum, live=live
        )
        return jax.numpy.sum(w * heights)

    coarse = jax.grad(objective)(v, 0.5)
    fine = jax.grad(objective)(v, 1e-9)
    assert np.abs(coarse).max() > 0.0
    np.testing.assert_allclose(coarse, fine, rtol=1e-12)


def test_dm_strokes_reject_a_live_mask_of_the_wrong_shape():
    with pytest.raises(ValueError, match="live"):
        dm_strokes_from_volts(
            np.zeros((4, 4)), **_calibration(), volt_quantum=0.5, live=np.ones(4, bool)
        )


@pytest.mark.parametrize("k", [1, 2])
def test_dm_wavefront_term_is_added_once_after_the_surface(k):
    rng = np.random.default_rng(k)
    surface = 3e-9 * rng.standard_normal((N_SMALL, N_SMALL))
    wfe = 7e-9 * rng.standard_normal((N_SMALL, N_SMALL))
    wl, stage = 825.0, f"dm{k}"
    base = _clear_model(**{f"dm{k}_surface_m": surface})
    both = _clear_model(**{f"dm{k}_surface_m": surface, f"dm{k}_wfe_m": wfe})
    a = base.propagate(wl, 0.0, 0.0, output_dim=65)[stage]
    b = both.propagate(wl, 0.0, 0.0, output_dim=65)[stage]
    np.testing.assert_allclose(
        b, a * np.exp(2j * np.pi * wfe / (wl * 1e-9)), atol=1e-15
    )


def test_dm_median_volts_quantizes_then_takes_the_live_median():
    q = 110.0 / 2**16
    v = np.array([[1.0, 2.00001], [3.0, 100.0]])
    live = np.array([[True, True], [True, False]])
    expect = np.median(np.floor(np.array([1.0, 2.00001, 3.0]) / q) * q)
    got = float(dm_median_volts(v, volt_quantum=q, live=live))
    assert got == pytest.approx(expect, rel=1e-15)


def test_dm_median_volts_rejects_a_live_mask_of_the_wrong_shape():
    with pytest.raises(ValueError, match="live"):
        dm_median_volts(np.zeros((4, 4)), volt_quantum=0.5, live=np.ones(4, bool))
