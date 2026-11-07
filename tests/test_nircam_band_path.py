"""``build_nircam_band``: the band as one late-bound path with shared seams.

Truth sources and tolerance bases:

- Identity with the per-node construction. With every seam at zero, the fold
  must reproduce ``nircam_band_image`` (the same stage arithmetic, one path
  per node) for a configuration with the focal mask, Lyot stop, an SI OPD
  cube and an off-axis source. Eager evaluation is asserted bitwise
  (exact-algebra: the same expressions on the same arrays; a zero Lyot shift
  and zero offset screens are exact multiplications by one); the jitted fold
  differs only by XLA fusion: measured 5e-15 relative, asserted at rtol 1e-13
  (floating-point).
- Linearity of the node sum: the gradient of a shared leaf through the band
  equals the weight-summed gradients of the one-node bands (exact-algebra).
- Checkpoint on and off are one computation: equal values and gradients.
- The detector +y convention: a source offset toward +y by m output samples
  displaces the image by m rows toward +row (shift theorem of the continuous
  Fourier transform; a sign error moves it the other way).
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

from physicaloptix.instruments import (
    BandLimitedRoundMask,
    NIRCamBand,
    NIRCamConfig,
    NIRCamInputs,
    build_nircam_band,
    nircam_band_image,
)
from physicaloptix.polychromatic import PolychromaticPath
from physicaloptix.sources import PointSource

DIAMETER_M = 6.5
NPUP = 32
NODES_NM = (3000.0, 3600.0)
RAW_WEIGHTS = (2.0, 3.0)
SCALE = 0.0624
DET_NPIX = 9
OVERSAMPLE = 2
J1_ZERO2 = float(scipy.special.jn_zeros(1, 2)[1])
CORE_SIGMA = J1_ZERO2 / 0.6


def _coords():
    return (np.arange(NPUP) - (NPUP - 1) / 2) / NPUP


def _pupil():
    x = _coords()
    xx, yy = np.meshgrid(x, x)
    amp = ((xx**2 + yy**2) <= 0.25).astype(float)
    amp[(np.abs(xx) < 0.04) & (yy > 0)] = 0.0
    amp[(xx > 0.1) & (yy < -0.1) & (amp > 0)] = 0.5
    return amp


def _ote_opd():
    xx, yy = np.meshgrid(_coords(), _coords())
    return 300.0 * yy + 500.0 * xx * yy + 800.0 * xx**3


def _lyot():
    xx, yy = np.meshgrid(_coords(), _coords())
    lyot = ((xx**2 + yy**2) <= 0.42**2).astype(float)
    lyot[(xx > 0.3) & (np.abs(yy) < 0.05)] = 0.0
    return lyot


def _si_cube():
    xx, yy = np.meshgrid(_coords(), _coords())
    base = 400.0 * xx**2 - 250.0 * xx * yy + 600.0 * yy**3
    return np.stack([base, 0.7 * base[::-1, :] + 30.0 * xx])


def _support():
    _, yy = np.meshgrid(_coords(), _coords())
    support = np.ones((NPUP, NPUP))
    support[(yy < -0.25) & (yy > -0.35)] = 0.0
    return support


def _inputs():
    mask = BandLimitedRoundMask(
        sigma_per_arcsec=CORE_SIGMA, regions=((0.1, 0.5, 0.2, None, 0.3),)
    )
    return NIRCamInputs(
        pupil_amplitude=jnp.asarray(_pupil()),
        pupil_diameter_m=DIAMETER_M,
        ote_opd_nm=jnp.asarray(_ote_opd()),
        focal_mask=mask,
        lyot_amplitude=jnp.asarray(_lyot()),
        si_support=jnp.asarray(_support()),
    )


def _config(**extra):
    kwargs = {
        "ote_opd": True,
        "focal_mask": True,
        "lyot_stop": True,
        "si_wfe": True,
        "mask_oversample": 2,
        "source_position_arcsec": (0.13, -0.07),
    }
    kwargs.update(extra)
    return NIRCamConfig(
        wavelength_nm=NODES_NM[0],
        pixel_scale_arcsec=SCALE,
        detector_npix=DET_NPIX,
        oversample=OVERSAMPLE,
        **kwargs,
    )


def _band(nodes=NODES_NM, weights=RAW_WEIGHTS, cube="default"):
    cube = _si_cube() if isinstance(cube, str) else cube
    return NIRCamBand(wavelengths_nm=nodes, weights=weights, si_opd_nm=cube)


def test_zero_seam_band_equals_nircam_band_image():
    config, inputs, band = _config(), _inputs(), _band()
    reference = np.asarray(nircam_band_image(config, inputs, band))
    model, source = build_nircam_band(config, inputs, band)
    eager = np.asarray(model.image((source,)))
    np.testing.assert_array_equal(eager, reference)
    assert reference.max() > 0.0

    jitted = np.asarray(eqx.filter_jit(lambda m, s: m.image((s,)))(model, source))
    np.testing.assert_allclose(jitted, reference, rtol=1e-13, atol=0)


def test_stage_names_and_zero_seams():
    model, source = build_nircam_band(_config(), _inputs(), _band())
    assert [stage.name for stage in model.path.stages] == [
        "entrance_pupil",
        "ote_opd",
        "ote_offset",
        "inversion",
        "focal_mask",
        "lyot_stop",
        "si_wfe",
        "si_offset",
        "science_focal",
    ]
    assert model.source_parity == (1, -1)
    assert not np.any(np.asarray(model.path.stage("ote_offset").opd_nm))
    assert not np.any(np.asarray(model.path.stage("si_offset").opd_nm))
    assert not np.any(np.asarray(model.path.stage("lyot_stop").shift_px))
    np.testing.assert_array_equal(source.position_arcsec, [0.13, -0.07])


def test_optional_stages_follow_the_config_toggles():
    config = _config(ote_opd=False, focal_mask=False, lyot_stop=False, si_wfe=False)
    model, _ = build_nircam_band(config, _inputs(), _band(cube=None))
    assert [stage.name for stage in model.path.stages] == [
        "entrance_pupil",
        "ote_offset",
        "inversion",
        "si_offset",
        "science_focal",
    ]


def _with_seams(model, offset, shift):
    model = eqx.tree_at(lambda m: m.path.stage("ote_offset").opd_nm, model, offset)
    return eqx.tree_at(lambda m: m.path.stage("lyot_stop").shift_px, model, shift)


def _seam_point():
    rng = np.random.default_rng(0)
    return jnp.asarray(5.0 * rng.standard_normal((NPUP, NPUP))), jnp.asarray(
        [0.3, -0.4]
    )


def test_seams_change_the_image():
    model, source = build_nircam_band(_config(), _inputs(), _band())
    base = np.asarray(model.image((source,)))
    offset, shift = _seam_point()
    moved = np.asarray(_with_seams(model, offset, shift).image((source,)))
    assert np.abs(moved - base).max() > 1e-3 * base.max()


def _probe():
    n = DET_NPIX * OVERSAMPLE
    return jnp.asarray(np.random.default_rng(1).standard_normal((n, n)))


def test_shared_seam_gradient_is_the_sum_of_per_node_gradients():
    weights = np.asarray(RAW_WEIGHTS) / np.sum(RAW_WEIGHTS)
    cube = _si_cube()
    config, inputs, probe = _config(), _inputs(), _probe()
    offset, shift = _seam_point()

    def grads(band):
        model, source = build_nircam_band(config, inputs, band)

        def loss(offset, shift):
            return jnp.sum(probe * _with_seams(model, offset, shift).image((source,)))

        return jax.grad(loss, argnums=(0, 1))(offset, shift)

    full = grads(_band())
    parts = [
        grads(_band(nodes=(wl,), weights=(1.0,), cube=cube[k : k + 1]))
        for k, wl in enumerate(NODES_NM)
    ]
    for i in range(2):
        summed = sum(w * part[i] for w, part in zip(weights, parts, strict=True))
        scale = float(jnp.max(jnp.abs(summed)))
        assert scale > 0.0
        np.testing.assert_allclose(full[i], summed, rtol=0, atol=1e-10 * scale)


def test_checkpoint_on_and_off_agree_in_value_and_gradient():
    config, inputs, probe = _config(), _inputs(), _probe()
    offset, shift = _seam_point()
    model, source = build_nircam_band(config, inputs, _band())

    def run(checkpoint):
        tuned = dataclasses.replace(model, checkpoint=checkpoint)

        def loss(offset, shift, position):
            src = eqx.tree_at(lambda s: s.position_arcsec, source, position)
            return jnp.sum(probe * _with_seams(tuned, offset, shift).image((src,)))

        point = (offset, shift, source.position_arcsec)
        return loss(*point), jax.grad(loss, argnums=(0, 1, 2))(*point)

    value_on, grads_on = run(True)
    value_off, grads_off = run(False)
    np.testing.assert_allclose(value_on, value_off, rtol=1e-13, atol=0)
    for g_on, g_off in zip(grads_on, grads_off, strict=True):
        scale = float(jnp.max(jnp.abs(g_off)))
        assert scale > 0.0
        np.testing.assert_allclose(g_on, g_off, rtol=0, atol=1e-10 * scale)


def test_detector_plus_y_lands_on_plus_rows():
    """No mask, no OPD: a y-asymmetric pupil image displaced by m output
    samples toward +y moves m rows toward +row."""
    m = 3
    config = _config(ote_opd=False, focal_mask=False, lyot_stop=False, si_wfe=False)
    model, _ = build_nircam_band(config, _inputs(), _band(cube=None))
    pitch = SCALE / OVERSAMPLE
    on_axis = np.asarray(model.image((PointSource([0.0, 0.0]),)))
    shifted = np.asarray(model.image((PointSource([0.0, m * pitch]),)))
    np.testing.assert_allclose(
        shifted[m:], on_axis[:-m], rtol=0, atol=1e-10 * on_axis.max()
    )
    assert isinstance(model, PolychromaticPath)
