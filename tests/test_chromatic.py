"""Late-bound wavelength: ``at_wavelength`` on stages, paths and tables.

Truth source: identity. A path bound late with ``at_wavelength(wl)`` must be
the path built directly at ``wl``, so the propagated fields are equal bit for
bit (the same expression is evaluated on the same arrays). Tolerance basis:
exact-algebra.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from physicaloptix.core import AbstractChromaticOp, Field, Grid, PlaneKind
from physicaloptix.elements import (
    DispersiveScreen,
    ModeBasis,
    PhaseScreen,
    SampledOptic,
)
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer

NPIX = 16


def _opd_basis():
    rng = np.random.default_rng(0)
    return ModeBasis(
        B=jnp.asarray(30.0 * rng.standard_normal((1, NPIX, NPIX))), coeffs=jnp.ones(1)
    )


def _field():
    grid = Grid.pupil(NPIX)
    return Field(
        data=jnp.ones((NPIX, NPIX), dtype=complex), grid=grid, plane=PlaneKind.PUPIL
    )


def _path(wavelength_nm):
    grid = Grid.pupil(NPIX)
    stop = SampledOptic(jnp.ones((NPIX, NPIX)), grid, PlaneKind.PUPIL)
    screen = PhaseScreen(_opd_basis(), grid, wavelength_nm=wavelength_nm)
    return OpticalPath(
        (
            Stage("stop", stop),
            Stage("screen", screen),
            Stage("science", Fraunhofer(grid, Grid.focal(24, 0.5))),
        )
    )


class TestAtWavelength:
    def test_phase_screen_is_chromatic_and_rebinds_only_the_wavelength(self):
        screen = _path(500.0).stage("screen")
        assert isinstance(screen, AbstractChromaticOp)
        bound = screen.at_wavelength(700.0)
        assert bound.wavelength_nm == 700.0
        assert bound.basis is screen.basis
        assert screen.wavelength_nm == 500.0

    def test_dispersive_screen_rebinds_the_monochromatic_wavelength(self):
        grid = Grid.pupil(NPIX)
        kernel = jnp.asarray([[1j, 1j]])
        screen = DispersiveScreen(
            _opd_basis(), kernel, [400.0, 800.0], grid, wavelength_nm=500.0
        )
        assert isinstance(screen, AbstractChromaticOp)
        assert screen.at_wavelength(650.0).wavelength_nm == 650.0

    def test_late_bound_path_equals_the_path_built_at_that_wavelength(self):
        late = _path(500.0).at_wavelength(700.0)
        direct = _path(700.0)
        out_late, _ = late.propagate(_field())
        out_direct, _ = direct.propagate(_field())
        np.testing.assert_array_equal(out_late.data, out_direct.data)

    def test_non_chromatic_stages_are_kept_and_names_preserved(self):
        path = _path(500.0)
        bound = path.at_wavelength(700.0)
        assert [s.name for s in bound.stages] == [s.name for s in path.stages]
        assert bound.stage("stop") is path.stage("stop")

    def test_gradient_flows_to_the_shared_leaf_through_the_bound_path(self):
        """Central difference on the OPD coefficient, step 1e-4 on a quantity
        of order 1 (basis in nm, so a 1e-4 step is a 1e-4 nm change): the
        truncation error is of order step**2 times the third derivative, so
        rtol 1e-6. Tolerance basis: finite-difference error.
        """
        path = _path(500.0)

        def loss(coeff, wavelength_nm):
            tree = eqx.tree_at(lambda p: p.stage("screen").basis.coeffs, path, coeff)
            out, _ = tree.at_wavelength(wavelength_nm).propagate(_field())
            return jnp.sum(out.intensity() * jnp.arange(out.data.size).reshape(24, 24))

        coeff = jnp.asarray([0.7])
        grad = jax.grad(loss)(coeff, 650.0)
        step = 1e-4
        fd = (loss(coeff + step, 650.0) - loss(coeff - step, 650.0)) / (2 * step)
        np.testing.assert_allclose(grad[0], fd, rtol=1e-6)


def test_unknown_wavelength_bind_is_a_noop_on_a_path_without_chromatic_stages():
    grid = Grid.pupil(NPIX)
    path = OpticalPath(
        (Stage("stop", SampledOptic(jnp.ones((NPIX, NPIX)), grid, PlaneKind.PUPIL)),)
    )
    assert path.at_wavelength(900.0).stage("stop") is path.stage("stop")
