"""Tests for OpticalPath.linearize: the unified (E_nom, G) entry point."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import ModeBasis, SampledOptic
from physicaloptix.linearize import linearity_residual
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.speckle import SpeckleProcess
from physicaloptix.transforms import Fraunhofer

WL_NM = 500.0


@pytest.fixture
def setup():
    npup = 24
    pupil_grid = Grid.pupil(npup)
    focal_grid = Grid.focal(32, 0.5)
    x = np.asarray(pupil_grid.coords)
    xx, yy = np.meshgrid(x, x)
    disk = ((xx**2 + yy**2) <= 0.25).astype(float)
    path = OpticalPath(
        stages=(
            Stage(
                "stop",
                SampledOptic(
                    transmission=jnp.asarray(disk),
                    grid=pupil_grid,
                    plane=PlaneKind.PUPIL,
                ),
            ),
            Stage("science", Fraunhofer(grid_in=pupil_grid, grid_out=focal_grid)),
        )
    )
    field = Field(
        data=jnp.asarray(disk).astype(complex),
        grid=pupil_grid,
        plane=PlaneKind.PUPIL,
    )
    rng = np.random.default_rng(0)
    basis = ModeBasis(
        B=jnp.asarray(rng.standard_normal((4, npup, npup))),
        coeffs=jnp.zeros(4),
    )
    return path, field, basis


class TestMethodsAgree:
    def test_analytic_matches_jvp(self, setup):
        path, field, basis = setup
        lin_a = path.linearize(field, basis, wavelength_nm=WL_NM, method="analytic")
        lin_j = path.linearize(field, basis, wavelength_nm=WL_NM, method="jvp")
        np.testing.assert_allclose(
            np.asarray(lin_a.G), np.asarray(lin_j.G), rtol=0, atol=1e-12
        )

    def test_jacfwd_matches_jvp(self, setup):
        path, field, basis = setup
        lin_f = path.linearize(field, basis, wavelength_nm=WL_NM, method="jacfwd")
        lin_j = path.linearize(field, basis, wavelength_nm=WL_NM, method="jvp")
        np.testing.assert_allclose(
            np.asarray(lin_f.G), np.asarray(lin_j.G), rtol=0, atol=1e-12
        )

    def test_chunked_streaming_matches_batched(self, setup):
        path, field, basis = setup
        full = path.linearize(field, basis, wavelength_nm=WL_NM, method="analytic")
        chunked = path.linearize(
            field, basis, wavelength_nm=WL_NM, method="analytic", chunk_size=1
        )
        np.testing.assert_array_equal(np.asarray(full.G), np.asarray(chunked.G))


class TestLinearizationProduct:
    def test_e_nom_matches_plain_propagation(self, setup):
        path, field, basis = setup
        lin = path.linearize(field, basis, wavelength_nm=WL_NM)
        out, _ = path.propagate(field)
        np.testing.assert_allclose(
            np.asarray(lin.e_nom), np.asarray(out.data), rtol=0, atol=1e-15
        )

    def test_shapes_and_meta(self, setup):
        path, field, basis = setup
        lin = path.linearize(field, basis, wavelength_nm=WL_NM)
        assert lin.G.shape == (4, 32, 32)
        assert lin.e_nom.shape == (32, 32)
        assert lin.n_modes == 4
        assert lin.kind == "opd"
        assert lin.wavelength_nm == WL_NM
        assert lin.method in ("analytic", "jvp", "jacfwd")

    def test_auto_resolves_by_memory_budget(self, setup):
        path, field, basis = setup
        lin = path.linearize(field, basis, wavelength_nm=WL_NM, memory_budget_bytes=1)
        # A one-byte budget forces the streaming path; result is unchanged.
        full = path.linearize(field, basis, wavelength_nm=WL_NM)
        np.testing.assert_array_equal(np.asarray(lin.G), np.asarray(full.G))

    def test_rejects_unknown_basis_kind(self, setup):
        _, _, basis = setup
        with pytest.raises(ValueError, match="kind"):
            ModeBasis(B=basis.B, coeffs=basis.coeffs, kind="banana")


class TestLinearity:
    def test_residual_is_small_and_quadratic(self, setup):
        """The first-order model's residual scales as eps^2 (ratio 4 at 2x)."""
        path, field, basis = setup
        lin = path.linearize(field, basis, wavelength_nm=WL_NM)
        rng = np.random.default_rng(0)
        direction = jnp.asarray(rng.standard_normal(4))
        eps1 = 1e-3 * direction  # nm-scale OPD against a 500 nm wavelength
        eps2 = 2e-3 * direction
        r1 = linearity_residual(path, field, basis, lin, eps1)
        r2 = linearity_residual(path, field, basis, lin, eps2)
        assert r1 < 1e-4
        np.testing.assert_allclose(r2 / r1, 4.0, rtol=0.05)


class TestSpeckleBridge:
    def test_to_speckle_process_round_trip(self, setup):
        path, field, basis = setup
        lin = path.linearize(field, basis, wavelength_nm=WL_NM)
        process = lin.to_speckle_process(
            normalization=1.0, decorr_hours=1.0, total_rms=0.01
        )
        assert isinstance(process, SpeckleProcess)
        np.testing.assert_array_equal(np.asarray(process.G), np.asarray(lin.G))
        speckle_field = process.draw(jax.random.PRNGKey(0))
        delta = speckle_field.realize(wavelength_nm=WL_NM, time_s=0.0)
        assert delta.shape == lin.e_nom.shape


class TestAmplitudeKind:
    def test_analytic_matches_jvp_for_amplitude_modes(self, setup):
        path, field, basis = setup
        amp = ModeBasis(B=basis.B, coeffs=basis.coeffs, kind="amplitude")
        lin_a = path.linearize(field, amp, wavelength_nm=WL_NM, method="analytic")
        lin_j = path.linearize(field, amp, wavelength_nm=WL_NM, method="jvp")
        np.testing.assert_allclose(
            np.asarray(lin_a.G), np.asarray(lin_j.G), rtol=0, atol=1e-12
        )
        assert lin_a.kind == "amplitude"

    def test_amplitude_columns_are_achromatic(self, setup):
        """Amplitude modes carry no phase factor: G is wavelength-free."""
        path, field, basis = setup
        amp = ModeBasis(B=basis.B, coeffs=basis.coeffs, kind="amplitude")
        lin_500 = path.linearize(field, amp, wavelength_nm=500.0)
        lin_1000 = path.linearize(field, amp, wavelength_nm=1000.0)
        np.testing.assert_array_equal(np.asarray(lin_500.G), np.asarray(lin_1000.G))

    def test_amplitude_linear_model_is_exact(self, setup):
        """E(eps) = E (1 + B.eps) is affine: the linear model is exact."""
        path, field, basis = setup
        amp = ModeBasis(B=basis.B, coeffs=basis.coeffs, kind="amplitude")
        lin = path.linearize(field, amp, wavelength_nm=WL_NM)
        rng = np.random.default_rng(0)
        eps = 0.05 * jnp.asarray(rng.standard_normal(4))
        assert linearity_residual(path, field, amp, lin, eps) < 1e-12
