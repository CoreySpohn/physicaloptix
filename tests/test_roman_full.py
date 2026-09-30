"""Step compiler and the full-train interpreter on a small synthetic relay."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.instruments.roman_full import (
    RomanFull,
    compile_train,
    detector_image,
    polarization_maps,
    source_tilt,
)

LAM, N, D, DPIX = 550e-9, 64, 0.01, 32.0


def _steps():
    return [
        ("multiply", "tilt"),
        ("multiply", "pupil"),
        ("normalize",),
        ("lens", 0.5),
        ("propagate", 0.5, "focus", True),
        ("propagate", 0.25, "relay", False),
        ("lens", 0.25),
        ("propagate", 0.25, "pupil2", False),
        ("wavefront", "map"),
        ("propagate", 0.1, "pupil2", False),
    ]


def _arrays():
    x = (np.arange(N) - N // 2) / (DPIX / 2)
    pupil = (x[None, :] ** 2 + x[:, None] ** 2 <= 1.0).astype(float)
    return {
        "tilt": source_tilt(N, DPIX, 0.0, 0.0, LAM, LAM),
        "pupil": jnp.asarray(pupil),
        "map": jnp.zeros((N, N)),
    }


def _model():
    plan = compile_train(
        _steps(), lam_m=LAM, n=N, beam_diameter_m=D, pupil_diam_pix=DPIX
    )
    model = RomanFull(
        arrays=_arrays(),
        program=plan.program,
        n=N,
        lam_m=LAM,
        pupil_diam_pix=DPIX,
        final_dx_m=plan.final.dx,
    )
    return plan, model


def test_compile_records_beams_and_unique_taps():
    plan, _ = _model()
    assert plan.taps == ("focus", "relay", "pupil2", "pupil2#2")
    assert len(plan.beams) == len(_steps())
    assert plan.beams[4].dx == plan.beams[0].dx


def test_run_conserves_energy_and_returns_taps():
    _, model = _model()
    out = eqx.filter_jit(lambda m: m(taps=("focus", "pupil2")))(model)
    assert set(out) == {"focus", "pupil2", "end"}
    for k in out:
        assert float(jnp.sum(jnp.abs(out[k]) ** 2)) == pytest.approx(1.0, rel=1e-12)


def test_gradient_through_a_map():
    _, model = _model()

    def loss(opd):
        m = eqx.tree_at(lambda mm: mm.arrays["map"], model, opd)
        return jnp.sum(jnp.abs(m()["end"][:8, :8]) ** 2)

    g = jax.grad(loss)(model.arrays["map"])
    assert bool(jnp.all(jnp.isfinite(g)))


def test_unknown_step_rejected():
    with pytest.raises(ValueError, match="unknown step"):
        compile_train(
            [("bogus",)], lam_m=LAM, n=N, beam_diameter_m=D, pupil_diam_pix=DPIX
        )


def test_runs_without_x64():
    jax.config.update("jax_enable_x64", False)
    try:
        _, model = _model()
        end = model()["end"]
        assert end.dtype == jnp.complex64
        assert bool(jnp.all(jnp.isfinite(end)))
    finally:
        jax.config.update("jax_enable_x64", True)


def test_detector_image_orientation_and_zoom():
    f = jnp.zeros((64, 64), complex).at[32, 40].set(1.0)
    img = detector_image(f, dx_m=1e-5, lam_m=LAM, defocus_c=0.0, output_dim=21, mag=1.0)
    ref = np.roll(np.rot90(np.fliplr(np.asarray(f)), 3), (1, 1), axis=(0, 1))
    ref = ref[32 - 10 : 32 + 11, 32 - 10 : 32 + 11]
    assert np.allclose(np.abs(np.asarray(img)), np.abs(ref), atol=1e-12)


def test_polarization_average_of_unit_piston_is_flat():
    zamp = np.zeros((2, 2, 6, 22))
    zamp[..., 0] = 1.0
    zpha = np.zeros((2, 2, 6, 22))
    amp, pha = polarization_maps(zamp, zpha, 575e-9, 309.0, 10)
    assert amp.shape == (340, 340)
    assert np.allclose(amp, 1.0)
    assert np.allclose(pha, 0.0)


def test_polarization_rejects_unsupported_condition():
    z = np.zeros((2, 2, 6, 22))
    with pytest.raises(ValueError, match="condition"):
        polarization_maps(z, z, 575e-9, 309.0, 5)


def test_build_helpers_are_public_on_roman_full():
    from physicaloptix.instruments import roman_full

    for name in (
        "PilotBeam",
        "plan_propagate",
        "plan_lens",
        "resample_map",
        "ellipse_mask",
        "noll_z6",
    ):
        assert name in roman_full.__all__
        assert callable(getattr(roman_full, name))
