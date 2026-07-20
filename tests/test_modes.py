"""Tests for the mode-basis constructors (Zernike, segment piston/tip/tilt)."""

import typing

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.apertures import eac1_primary, rasterize_primary, rasterize_segments
from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import ModeBasis
from physicaloptix.elements.modes import segment_ptt_basis, zernike_basis
from physicaloptix.linearize import linearity_residual, linearize
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer


@pytest.fixture(scope="module")
def primary():
    return eac1_primary()


def _aperture(grid, diameter=1.0):
    coords = np.asarray(grid.coords)
    x_grid, y_grid = np.meshgrid(coords, coords)
    return np.sqrt(x_grid**2 + y_grid**2) <= diameter / 2.0


class TestSegmentPTT:
    def test_three_modes_per_segment(self, primary):
        basis = segment_ptt_basis(primary, Grid.pupil(48))
        assert isinstance(basis, ModeBasis)
        assert basis.n_modes == 3 * primary.n_segments

    def test_kind_is_opd(self, primary):
        assert segment_ptt_basis(primary, Grid.pupil(48)).kind == "opd"

    def test_modes_are_segment_local(self, primary):
        """Each of the centre segment's three modes is zero outside the
        centre hexagon, so a segment-local error cannot leak."""
        npix = 48
        basis = segment_ptt_basis(primary, Grid.pupil(npix))
        centre_support = rasterize_segments(primary, npix)[0] > 0
        for k in range(3):  # piston, tip, tilt of the centre segment
            mode = np.asarray(basis.B[k])
            assert np.all(mode[~centre_support] == 0.0)

    def test_piston_honours_the_nm_unit_contract(self, primary):
        """A piston coefficient of 1 gives ptt_nm of area-weighted RMS over
        its segment: B is delivered in nanometres, not a dimensionless mode."""
        npix = 48
        ptt_nm = 7.0
        basis = segment_ptt_basis(primary, Grid.pupil(npix), ptt_nm=ptt_nm)
        area = rasterize_segments(primary, npix)[0].sum()
        piston = np.asarray(basis.B[0])
        rms = np.sqrt((piston**2).sum() / area)
        np.testing.assert_allclose(rms, ptt_nm, rtol=1e-12)

    def test_tip_carries_no_piston(self, primary):
        """The tip mode is zero-mean over its segment (hexagon symmetry), so
        it is orthogonal to that segment's piston."""
        basis = segment_ptt_basis(primary, Grid.pupil(48))
        tip = np.asarray(basis.B[1])
        assert abs(tip.sum()) < 1e-9 * np.abs(tip).sum()


class TestZernike:
    def test_mode_count_and_kind(self):
        basis = zernike_basis(Grid.pupil(64), 10)
        assert basis.n_modes == 10
        assert basis.kind == "opd"

    def test_piston_is_constant_over_the_aperture(self):
        """Noll mode 1 is a constant (the aperture), zero outside it."""
        grid = Grid.pupil(64)
        piston = np.asarray(zernike_basis(grid, 4, rms_nm=3.0).B[0])
        aperture = _aperture(grid)
        assert np.all(piston[~aperture] == 0.0)
        vals = piston[aperture]
        np.testing.assert_allclose(vals, vals[0], rtol=1e-12)

    def test_unit_contract_rms_over_aperture(self):
        """Every mode carries rms_nm of RMS over the aperture (B is in nm)."""
        grid = Grid.pupil(64)
        rms_nm = 5.0
        basis = zernike_basis(grid, 8, rms_nm=rms_nm)
        npupil = _aperture(grid).sum()
        for j in range(8):
            mode = np.asarray(basis.B[j])
            rms = np.sqrt((mode**2).sum() / npupil)
            np.testing.assert_allclose(rms, rms_nm, rtol=1e-12)

    def test_modes_are_orthogonal_over_the_aperture(self):
        """Distinct unit-RMS Zernikes have a small cross-correlation."""
        grid = Grid.pupil(96)
        basis = zernike_basis(grid, 6)
        npupil = _aperture(grid).sum()
        stack = np.asarray(basis.B)
        for i in range(6):
            for j in range(i + 1, 6):
                inner = (stack[i] * stack[j]).sum() / npupil
                assert abs(inner) < 5e-2, f"modes {i},{j}: {inner:.3e}"


class TestNollIdentity:
    """Mode j must BE Noll mode j, not merely an orthonormal basis member.

    The RMS/orthogonality/piston tests all survive an index permutation or a
    sin/cos swap in the (n, m) mapping; a wrong Noll order silently corrupts
    every downstream WFE budget labeled "Z4 defocus". Normalized correlation
    against the explicit polynomials pins identity and sign; the swapped-
    parity control pins Noll's odd-j-is-sine rule.
    """

    _FORMS: typing.ClassVar = {
        2: lambda rho, th: 2.0 * rho * np.cos(th),
        3: lambda rho, th: 2.0 * rho * np.sin(th),
        4: lambda rho, th: np.sqrt(3.0) * (2.0 * rho**2 - 1.0),
        5: lambda rho, th: np.sqrt(6.0) * rho**2 * np.sin(2.0 * th),
        6: lambda rho, th: np.sqrt(6.0) * rho**2 * np.cos(2.0 * th),
        11: lambda rho, th: np.sqrt(5.0) * (6.0 * rho**4 - 6.0 * rho**2 + 1.0),
    }

    @staticmethod
    def _polar(grid):
        coords = np.asarray(grid.coords)
        x_grid, y_grid = np.meshgrid(coords, coords)
        return 2.0 * np.hypot(x_grid, y_grid), np.arctan2(y_grid, x_grid)

    def test_modes_match_the_explicit_noll_polynomials(self):
        grid = Grid.pupil(128)
        basis = zernike_basis(grid, 11, rms_nm=1.0)
        rho, theta = self._polar(grid)
        aperture = rho <= 1.0
        for j, form in self._FORMS.items():
            mode = np.asarray(basis.B[j - 1])[aperture]
            ref = form(rho, theta)[aperture]
            corr = (mode * ref).sum() / np.sqrt((mode**2).sum() * (ref**2).sum())
            assert corr > 1.0 - 1e-9, f"Noll j={j}: correlation {corr:.12f}"

    def test_parity_control_rejects_a_sin_cos_swap(self):
        """Noll j=5 is the SINE astigmatism: its correlation with the cosine
        form must vanish (a swapped mapping would score ~1 above and ~0
        here)."""
        grid = Grid.pupil(128)
        basis = zernike_basis(grid, 6, rms_nm=1.0)
        rho, theta = self._polar(grid)
        aperture = rho <= 1.0
        mode5 = np.asarray(basis.B[4])[aperture]
        wrong = (np.sqrt(6.0) * rho**2 * np.cos(2.0 * theta))[aperture]
        corr = (mode5 * wrong).sum() / np.sqrt((mode5**2).sum() * (wrong**2).sum())
        assert abs(corr) < 0.05


def _science_path(pupil, npix_focal=32):
    focal = Grid.focal(npix_focal, 0.5)
    return OpticalPath(
        stages=(Stage("science", Fraunhofer(grid_in=pupil, grid_out=focal)),)
    )


class TestLinearizeIntegration:
    def test_zernike_basis_linearizes_cleanly(self):
        """A Zernike basis feeds linearize: G has the right shape and the
        linear model's error scales quadratically with the coefficient."""
        pupil = Grid.pupil(48)
        aperture = _aperture(pupil).astype(float)
        path = _science_path(pupil)
        field = Field(
            data=jnp.asarray(aperture, dtype=complex),
            grid=pupil,
            plane=PlaneKind.PUPIL,
        )
        basis = zernike_basis(pupil, 6, rms_nm=1.0)
        lin = linearize(path, field, basis, wavelength_nm=500.0)
        assert lin.G.shape == (6, 32, 32)
        direction = jnp.asarray(np.random.default_rng(0).standard_normal(6))
        r1 = linearity_residual(path, field, basis, lin, 1e-3 * direction)
        r2 = linearity_residual(path, field, basis, lin, 2e-3 * direction)
        assert r1 < 1e-4
        assert r2 / r1 == pytest.approx(4.0, rel=0.3)

    def test_segment_ptt_basis_linearizes_cleanly(self, primary):
        """The segment PTT basis feeds linearize on the real EAC pupil."""
        pupil = Grid.pupil(48)
        pupil_arr = rasterize_primary(primary, 48, supersample=4)
        path = _science_path(pupil)
        field = Field(
            data=jnp.asarray(pupil_arr, dtype=complex),
            grid=pupil,
            plane=PlaneKind.PUPIL,
        )
        basis = segment_ptt_basis(primary, pupil, ptt_nm=1.0, supersample=4)
        lin = linearize(path, field, basis, wavelength_nm=500.0)
        assert lin.G.shape == (3 * primary.n_segments, 32, 32)
        direction = jnp.asarray(
            np.random.default_rng(1).standard_normal(3 * primary.n_segments)
        )
        r1 = linearity_residual(path, field, basis, lin, 1e-3 * direction)
        r2 = linearity_residual(path, field, basis, lin, 2e-3 * direction)
        assert r1 < 1e-4
        assert r2 / r1 == pytest.approx(4.0, rel=0.3)
