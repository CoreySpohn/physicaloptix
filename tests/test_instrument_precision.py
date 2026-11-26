"""Working-precision closure of public NIRCam and Roman propagation paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.leaves import canonical
from hwoutils.precision import PrecisionError, complex_of, float_dtype
from hwoutils.testing import assert_precision_closed, precision_pair, ulps

from physicaloptix.instruments import (
    BandLimitedRoundMask,
    NIRCamBand,
    NIRCamConfig,
    NIRCamInputs,
    RomanCompact,
    RomanFull,
    build_nircam,
    build_nircam_band,
    compile_train,
    integrate_detector_pixels,
    mask_plane_field,
    nircam_band_image,
)
from physicaloptix.instruments.roman import (
    dm_median_volts,
    dm_strokes_from_volts,
    dm_surface,
    volts_to_stroke,
)
from physicaloptix.instruments.roman_full import detector_image, noll_z6, source_tilt


def _nircam(npup, detector_npix):
    x = (np.arange(npup) - (npup - 1) / 2) / npup
    xx, yy = np.meshgrid(x, x)
    radius = np.hypot(xx, yy)
    pupil = (radius <= 0.5) & (np.abs(xx) > 0.01)
    opd = 80 * (xx**2 - yy**2) + 20 * yy
    wavelengths = (4050.0, 4450.0, 4850.0)
    config = NIRCamConfig(
        np.float64(wavelengths[0]),
        0.062424185,
        detector_npix=detector_npix,
        oversample=2,
        mask_oversample=2,
        source_position_arcsec=(0.35, -0.2),
        ote_opd=True,
        focal_mask=True,
        lyot_stop=True,
        si_wfe=True,
    )
    inputs = NIRCamInputs(
        pupil,
        6.603464,
        ote_opd_nm=opd,
        focal_mask=BandLimitedRoundMask(3.2927866),
        lyot_amplitude=radius < 0.42,
        si_support=pupil.astype(float),
        si_opd_nm=opd.T,
        si_wavelength_nm=wavelengths[0],
    )
    band = NIRCamBand(
        wavelengths, (1, 2, 1), np.stack([a * opd.T for a in (1, 1.1, 1.2)])
    )
    return config, inputs, band


def _nircam_mono(config, inputs, band):
    path, entrance = build_nircam(config, inputs)
    science, taps = path.propagate(entrance, taps=("inversion",))
    return science.data, mask_plane_field(path, taps).data


def _nircam_independent(config, inputs, band):
    return integrate_detector_pixels(nircam_band_image(config, inputs, band), 2).data


def _nircam_shared(config, inputs, band):
    model, source = build_nircam_band(config, inputs, band)
    image = model.image(source)
    return image.data, integrate_detector_pixels(image, config.oversample).data


def _roman_compact(kind):
    n = 16
    x = np.arange(n) - n // 2
    radius = np.hypot(x[None, :], x[:, None])
    fpm = np.ones((7, 7), dtype=complex_of(float_dtype()))
    fpm[2:5, 2:5] = 0.2 + 0.1j
    model = RomanCompact(
        pupil=radius <= 4,
        lyot=radius < 4,
        pupil_mask=None,
        fpm=fpm,
        kind=kind,
        n_small=n,
        n_big=n,
        pupil_diam_pix=np.float64(8),
        fpm_sampling_lod=np.float64(0.5),
        fpm_lam0_nm=np.float64(825),
        lam0_nm=np.float64(825),
        dm1_surface_m=radius * 1e-9,
    )
    return (model,)


def _compact_image(model):
    return model.propagate(
        np.float64(825), np.float64(0.2), np.float64(-0.15), output_dim=7
    )


def _roman_full():
    n = 32
    steps = [
        ("multiply", "pupil"),
        ("normalize",),
        ("lens", 0.5),
        ("propagate", 0.5, "focus", True),
        ("propagate", 0.1, "relay", False),
        ("wavefront", "opd"),
        ("hlc_fpm", "mask", "region", 1e-5),
        ("spc_fpm", "mask", 1e-5, n),
    ]
    x = np.arange(n) - n // 2
    radius = np.hypot(x[None, :], x[:, None])
    mask = np.ones((7, 7))
    mask[3, 3] = 0.5
    plan = compile_train(
        steps, lam_m=550e-9, n=n, beam_diameter_m=1e-3, pupil_diam_pix=16
    )
    model = RomanFull(
        {"pupil": radius < 8, "opd": radius * 1e-9, "mask": mask, "region": mask < 1},
        plan.program,
        n,
        np.float64(550e-9),
        np.float64(16),
        plan.final.dx,
    )
    return (model,)


def _full_image(model):
    field = model()["end"]
    return detector_image(
        field,
        dx_m=np.float64(model.final_dx_m),
        lam_m=model.lam_m,
        defocus_c=np.float64(0.2),
        output_dim=7,
        mag=np.float64(1.25),
    )


def _roman_dm(volts, nodes, table, coupling):
    cal = dict(
        stroke_volts=nodes,
        coupling_volts=nodes,
        stroke_table_m=table,
        coupling=coupling,
    )
    live = jnp.ones_like(volts, dtype=bool)
    strokes = dm_strokes_from_volts(
        volts, **cal, live=live, volt_quantum=np.float64(0.25)
    )
    surface = dm_surface(
        strokes,
        jnp.ones((3, 3), dtype=volts.dtype),
        influence_dx_m=np.float64(0.1),
        influence_pitch_m=np.float64(0.2),
        pitch_m=np.float64(0.2),
        center_act=(np.float64(1), np.float64(1)),
        grid_npix=8,
        grid_dx_m=np.float64(0.1),
        tilt_deg=(5.0, 3.0, 0.0),
    )
    return (
        volts_to_stroke(volts, **cal),
        dm_median_volts(volts, live=live, volt_quantum=np.float64(0.25)),
        surface,
    )


def _dm_inputs():
    volts = jnp.arange(9, dtype=float_dtype()).reshape(3, 3) + 0.6
    nodes = jnp.asarray([0.0, 10.0, 20.0])
    return (
        volts,
        nodes,
        jnp.broadcast_to(nodes[:, None, None] * 1e-9, (3, 3, 3)),
        jnp.full((3, 3, 3, 3, 3), 0.01),
    )


def _roman_host_commands(step):
    volts = np.array([[2.5, 5.0]])
    live = np.ones_like(volts, dtype=bool)
    cal = dict(
        stroke_volts=jnp.stack((step * 0, step * 16)),
        stroke_table_m=np.array([[[0, 0]], [[1e-7, 2e-7]]]),
        coupling_volts=[0, 10],
        coupling=np.zeros((2, 1, 2, 3, 3)),
    )
    surface = dm_surface(
        volts,
        jnp.ones((1, 1), dtype=step.dtype),
        influence_dx_m=1,
        influence_pitch_m=1,
        pitch_m=1,
        center_act=(0, 0),
        grid_npix=5,
        grid_dx_m=1,
    )
    detector = detector_image(
        np.eye(8, dtype=complex),
        dx_m=step * 1e-5,
        lam_m=550e-9,
        defocus_c=0.2,
        output_dim=7,
    )
    return (
        volts_to_stroke(volts, **cal),
        dm_strokes_from_volts(volts, **cal, live=live, volt_quantum=0.25),
        dm_median_volts(volts, live=live, volt_quantum=step),
        surface,
        detector,
    )


def _roman_phase(offset):
    return (
        source_tilt(
            16, np.float64(8), offset, -offset, np.float64(550e-9), np.float64(575e-9)
        ),
        noll_z6(16, offset, np.float64(1)),
    )


@pytest.mark.parametrize(
    ("forward", "make_inputs", "roundings"),
    [
        pytest.param(
            _nircam_mono, lambda: _nircam(32, 9), 4 * 32, id="nircam-mono-mask-tap"
        ),
        pytest.param(
            _nircam_independent,
            lambda: _nircam(32, 9),
            8 * 32,
            id="nircam-band-detector",
        ),
        pytest.param(
            _nircam_shared,
            lambda: _nircam(256, 80),
            8 * 256,
            id="nircam-mask335r-f444w",
            marks=pytest.mark.medium(
                reason="three full-band FFTs and focal transforms in two precisions"
            ),
        ),
        pytest.param(
            _compact_image, lambda: _roman_compact("spc"), 8 * 16, id="roman-spc"
        ),
        pytest.param(
            _compact_image, lambda: _roman_compact("hlc"), 8 * 16, id="roman-hlc"
        ),
        pytest.param(_full_image, _roman_full, 12 * 32, id="roman-full-detector"),
        pytest.param(_roman_dm, _dm_inputs, 128, id="roman-dm-calibration"),
        pytest.param(
            _roman_host_commands,
            lambda: (jnp.asarray(0.5, dtype=float_dtype()),),
            128,
            id="roman-host-commands",
        ),
        pytest.param(
            _roman_phase,
            lambda: (jnp.asarray(0.25, dtype=float_dtype()),),
            16,
            id="roman-tilt-and-z6",
        ),
    ],
)
def test_instrument_forward_precision(forward, make_inputs, roundings, precision):
    """Allow roundoff at the peak amplitude or intensity of each output plane.

    Each 2D transform includes two one-dimensional sums. Budget four pupil
    widths for the masked monochromatic field and eight for the band intensity
    or compact model with its additional transforms. The full relay adds two
    focal mask transform pairs and a detector resampling, for twelve widths.
    The DM calibration interpolates up to nine coupled actuators, then
    convolves and interpolates on a 24-sample grid with 16 cubic taps; allow
    128 roundings. The tilt and astigmatism evaluate at most 16 arithmetic and
    trigonometric operations on coordinates of order one.

    Args:
        forward: Public instrument propagation or a sequence of public stages.
        make_inputs: Construction of this path's example inputs.
        roundings: Single-precision rounding budget at each output's peak.
        precision: Session working dtype.
    """
    args = make_inputs()
    if precision == np.float32:
        for leaf in jax.tree.leaves(forward(*args)):
            assert np.finfo(leaf.dtype).dtype == precision
    else:
        assert_precision_closed(forward, *canonical(args, precision="float32"))
        precision_pair(forward, args, bound=float(ulps(roundings, np.float32(1))))


def test_live_voltage_median_at_float_limit(precision):
    """Fails if the median overflows or halves small voltages into subnormals.

    The extrema and the midpoint between one and two minimum normals are exact
    in the working precision.

    Args:
        precision: Session working dtype.
    """
    limits = np.finfo(precision)
    for a, b, quantum, expected in (
        (limits.max, limits.max, 1.0, limits.max),
        (limits.tiny, np.nan, limits.tiny, limits.tiny),
        (limits.tiny, 2 * limits.tiny, limits.tiny, 1.5 * limits.tiny),
    ):
        volts = jnp.asarray([a, -a, b, np.nan], dtype=precision)
        live = jnp.asarray([True, False, True, True])
        for median in (dm_median_volts, jax.jit(dm_median_volts)):
            result = median(volts, volt_quantum=quantum, live=live)
            np.testing.assert_array_equal(result, expected)


@pytest.mark.float64_only(reason="mixed strong precisions require x64")
def test_dm_rejects_mixed_command_and_calibration_precision():
    """Fails if calibration silently narrows a strong voltage or voltage step."""
    volts = jnp.ones((1, 1), dtype=np.float32)
    with pytest.raises(PrecisionError, match="Mixed floating precisions"):
        dm_median_volts(
            volts,
            volt_quantum=jnp.ones((), dtype=float_dtype()),
            live=np.ones((1, 1), dtype=bool),
        )
    with pytest.raises(ValueError, match="do not fit"):
        dm_median_volts(
            [[1e100]], volt_quantum=volts[0, 0], live=np.ones((1, 1), dtype=bool)
        )
