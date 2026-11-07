"""``PolychromaticPath`` on a 32 px, 2-node toy path.

The path holds every shared seam once: an OPD offset (``SampledPhaseScreen``),
a stop with a shift (``SampledOptic.shift_px``) and a per-node science grid
(``ChromaticTable``). Truth sources and tolerance bases:

- Per-node independent paths built the direct way (each node its own plain
  ``PhaseScreen`` and ``Fraunhofer``), summed with unit-sum weights: the
  late-bound fold is the same arithmetic, so the images agree to the
  floating-point floor (measured bitwise; asserted at rtol 1e-14).
- Linearity of the sum: the gradient of a shared leaf through the whole fold
  equals the weight-summed gradients of the one-node folds (exact-algebra,
  asserted at rtol 1e-10).
- ``jax.test_util.check_grads`` per seam (finite-difference basis: central
  difference at eps 1e-6, truncation error about 1e-9 relative, measured
  2e-5 relative at the default eps 1e-4 for the source position; rtol 1e-6);
  the Lyot shift is checked at the non-integer shift (0.3, -0.4) because
  order-1 interpolation has a gradient kink at integer shifts.
- Checkpoint on and off are the same computation: equal values and gradients.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from physicaloptix.core import ChromaticTable, Field, Grid, PlaneKind
from physicaloptix.elements import (
    ModeBasis,
    PhaseScreen,
    SampledOptic,
    SampledPhaseScreen,
)
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.polychromatic import PolychromaticPath
from physicaloptix.sources import PointSource
from physicaloptix.transforms import Fraunhofer

NPUP = 32
NODES = (1800.0, 2400.0)
RAW_WEIGHTS = (1.0, 3.0)
DIAMETER_M = 6.5
ARCSEC_PER_SAMPLE_AT_REF = 0.04
REF_NM = 2400.0
NFOCAL = 24


def _grids():
    return Grid.pupil(NPUP), {wl: Grid.focal(NFOCAL, 0.5 * wl / REF_NM) for wl in NODES}


def _pupil_amplitude():
    x = np.asarray(Grid.pupil(NPUP).coords)
    xx, yy = np.meshgrid(x, x)
    amp = ((xx**2 + yy**2) <= 0.25).astype(float)
    amp[(np.abs(xx) < 0.05) & (yy > 0)] = 0.0
    return jnp.asarray(amp)


def _fixed_opd_basis():
    rng = np.random.default_rng(0)
    return ModeBasis(
        B=jnp.asarray(20.0 * rng.standard_normal((1, NPUP, NPUP))), coeffs=jnp.ones(1)
    )


def _stop():
    x = np.asarray(Grid.pupil(NPUP).coords)
    xx, yy = np.meshgrid(x, x)
    return jnp.asarray(((xx**2 + yy**2) <= 0.2).astype(float))


def _entrance():
    amp = _pupil_amplitude()
    grid = Grid.pupil(NPUP)
    energy = jnp.sum(amp**2) * grid.weights
    return Field(
        data=jnp.full((NPUP, NPUP), 1.0 / jnp.sqrt(energy), dtype=complex),
        grid=grid,
        plane=PlaneKind.PUPIL,
    )


def _fold(checkpoint=True, nodes=NODES, weights=RAW_WEIGHTS):
    pupil, focal = _grids()
    stages = (
        Stage(
            "entrance_pupil", SampledOptic(_pupil_amplitude(), pupil, PlaneKind.PUPIL)
        ),
        Stage(
            "fixed_opd", PhaseScreen(_fixed_opd_basis(), pupil, wavelength_nm=nodes[0])
        ),
        Stage("ote_offset", SampledPhaseScreen(jnp.zeros((NPUP, NPUP)), pupil)),
        Stage(
            "lyot_stop",
            SampledOptic(_stop(), pupil, PlaneKind.PUPIL, shift_px=jnp.zeros(2)),
        ),
        Stage(
            "science_focal",
            ChromaticTable(
                tuple(
                    Fraunhofer(pupil, focal[wl], on_undersampled="raise")
                    for wl in nodes
                ),
                nodes,
            ),
        ),
    )
    return PolychromaticPath(
        OpticalPath(stages),
        _entrance(),
        nodes,
        weights,
        DIAMETER_M,
        checkpoint=checkpoint,
    )


def _sources():
    return (
        PointSource(jnp.asarray([0.021, -0.013])),
        PointSource(
            jnp.asarray([-0.05, 0.03]),
            flux=0.3,
            spectral_weights=jnp.asarray([2.0, 0.5]),
        ),
    )


def _with_seams(model, offset=None, shift=None):
    if offset is not None:
        model = eqx.tree_at(lambda m: m.path.stage("ote_offset").opd_nm, model, offset)
    if shift is not None:
        model = eqx.tree_at(lambda m: m.path.stage("lyot_stop").shift_px, model, shift)
    return model


def _direct_node_image(wavelength_nm, source, offset, shift):
    """One node the direct way: plain per-node stages, no late binding."""
    pupil, focal = _grids()
    stop = SampledOptic(_stop(), pupil, PlaneKind.PUPIL, shift_px=shift)
    stages = (
        Stage(
            "entrance_pupil", SampledOptic(_pupil_amplitude(), pupil, PlaneKind.PUPIL)
        ),
        Stage(
            "fixed_opd",
            PhaseScreen(_fixed_opd_basis(), pupil, wavelength_nm=wavelength_nm),
        ),
        Stage(
            "ote_offset",
            SampledPhaseScreen(offset, pupil, wavelength_nm=wavelength_nm),
        ),
        Stage("lyot_stop", stop),
        Stage("science_focal", Fraunhofer(pupil, focal[wavelength_nm])),
    )
    field = source.entrance_field(_entrance(), wavelength_nm, DIAMETER_M, (1, 1))
    out, _ = OpticalPath(stages).propagate(field)
    return out.intensity() * out.grid.weights


def test_image_equals_independent_per_node_paths_summed():
    offset = jnp.asarray(np.random.default_rng(2).standard_normal((NPUP, NPUP)))
    shift = jnp.asarray([0.3, -0.4])
    model = _with_seams(_fold(), offset, shift)
    sources = _sources()
    image = model.image(sources)
    weights = np.asarray(RAW_WEIGHTS) / np.sum(RAW_WEIGHTS)
    expected = 0.0
    for k, wl in enumerate(NODES):
        for source in sources:
            spectral = (
                1.0 if source.spectral_weights is None else source.spectral_weights[k]
            )
            expected = expected + (
                weights[k] * source.flux * spectral
            ) * _direct_node_image(wl, source, offset, shift)
    np.testing.assert_allclose(
        np.asarray(image), np.asarray(expected), rtol=1e-14, atol=0
    )
    assert float(jnp.max(image)) > 0.0


def test_a_single_source_may_be_passed_bare():
    source = _sources()[0]
    model = _fold()
    np.testing.assert_array_equal(
        np.asarray(model.image(source)), np.asarray(model.image((source,)))
    )


def _loss_fn(checkpoint):
    sources = _sources()
    weights = jnp.asarray(np.random.default_rng(1).standard_normal((NFOCAL, NFOCAL)))

    def loss(offset, shift, position, flux):
        model = _with_seams(_fold(checkpoint), offset, shift)
        src = (
            eqx.tree_at(
                lambda s: (s.position_arcsec, s.flux), sources[0], (position, flux)
            ),
            sources[1],
        )
        return jnp.sum(weights * model.image(src))

    return loss


_POINT = (
    jnp.asarray(np.random.default_rng(3).standard_normal((NPUP, NPUP))),
    jnp.asarray([0.3, -0.4]),
    jnp.asarray([0.021, -0.013]),
    jnp.asarray(1.2),
)


def test_checkpoint_on_and_off_agree_in_value_and_gradient():
    on, off = _loss_fn(True), _loss_fn(False)
    np.testing.assert_allclose(on(*_POINT), off(*_POINT), rtol=1e-13, atol=0)
    grads_on = jax.grad(on, argnums=(0, 1, 2, 3))(*_POINT)
    grads_off = jax.grad(off, argnums=(0, 1, 2, 3))(*_POINT)
    for g_on, g_off in zip(grads_on, grads_off, strict=True):
        scale = float(jnp.max(jnp.abs(g_off)))
        assert scale > 0.0
        np.testing.assert_allclose(g_on, g_off, rtol=0, atol=1e-10 * scale)


def test_shared_leaf_gradient_is_the_sum_of_per_node_gradients():
    """The two-node gradient of the shared offset and shift equals the
    unit-weight-normalized sum of the one-node gradients."""
    weights = np.asarray(RAW_WEIGHTS) / np.sum(RAW_WEIGHTS)
    source = _sources()[0]
    probe = jnp.asarray(np.random.default_rng(1).standard_normal((NFOCAL, NFOCAL)))
    offset, shift = _POINT[0], _POINT[1]

    def loss(model, offset, shift):
        return jnp.sum(probe * _with_seams(model, offset, shift).image((source,)))

    full = jax.grad(loss, argnums=(1, 2))(_fold(), offset, shift)
    parts = [
        jax.grad(loss, argnums=(1, 2))(
            _fold(nodes=(wl,), weights=(1.0,)), offset, shift
        )
        for wl in NODES
    ]
    for i in range(2):
        summed = sum(w * part[i] for w, part in zip(weights, parts, strict=True))
        scale = float(jnp.max(jnp.abs(summed)))
        assert scale > 0.0
        np.testing.assert_allclose(full[i], summed, rtol=0, atol=1e-10 * scale)


@pytest.mark.parametrize(
    "seam", [0, 1, 2, 3], ids=["offset", "lyot", "position", "flux"]
)
def test_check_grads_per_seam(seam):
    loss = _loss_fn(True)

    def one(x):
        args = list(_POINT)
        args[seam] = x
        return loss(*args)

    check_grads(
        one, (_POINT[seam],), order=1, modes=["rev"], eps=1e-6, atol=1e-6, rtol=1e-6
    )


def test_validation():
    path = _fold().path
    entrance = _entrance()
    with pytest.raises(ValueError, match="one weight per node"):
        PolychromaticPath(path, entrance, NODES, (1.0,), DIAMETER_M)
    with pytest.raises(ValueError, match="non-negative"):
        PolychromaticPath(path, entrance, NODES, (1.0, -1.0), DIAMETER_M)
    with pytest.raises(KeyError, match="not a table node"):
        PolychromaticPath(path, entrance, (1800.0, 2399.0), (1.0, 1.0), DIAMETER_M)
    model = _fold()
    with pytest.raises(ValueError, match="at least one source"):
        model.image(())
    with pytest.raises(ValueError, match="spectral_weights"):
        model.image((PointSource([0.0, 0.0], spectral_weights=[1.0, 1.0, 1.0]),))
