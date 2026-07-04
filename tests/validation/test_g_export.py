"""Validation gate: linearize reproduces the dense-basis (E_nom, G) export.

The reference export was produced by the linear-response builder on the full
EAC-1 AAVC chain (charge-6 multi-scale vortex, cap_num_airy0=512, design
wavelength 1000 nm) with a seeded dense random Fourier ripple basis. The rng
sequence is reproducible, and the stored kx/ky arrays cross-check that the
regenerated basis is the same one. ``linearize(method="analytic")`` must then
reproduce E_nom and the G columns to floating-point roundoff.

Tolerances are cross-build roundoff, not physics: transcendental dispatch
(cos/sqrt/atan2/exp) differs by ulps between library builds, so regenerated
values match the export to ~1e-15 relative on the basis and ~1e-10
relative-to-peak through the full chain (measured), not bitwise.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import ModeBasis, MultiScaleVortex, SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer, cmft_fwd

WL_NM = 1000.0
KR_LO, KR_HI = 3.0, 25.0
N_MODES = 220
CHECK_COLUMNS = (0, 7, 119, 219)


@pytest.fixture(scope="session")
def dense_export(dense_speckle_export):
    return dense_speckle_export


@pytest.fixture(scope="session")
def aavc_speckle_path(eac1_cache):
    """The EAC-1 AAVC path exactly as the speckle chain used it (cap 512)."""
    z = eac1_cache
    npup = int(z["meta"][0])
    dims, pix_lod = 256, 0.25
    pupil_grid = Grid.pupil(npup)
    science_grid = Grid.focal(dims, pix_lod)

    def pupil_optic(name):
        return SampledOptic(
            transmission=jnp.asarray(z[name], float),
            grid=pupil_grid,
            plane=PlaneKind.PUPIL,
        )

    path = OpticalPath(
        stages=(
            Stage("pupil_stop", pupil_optic("pupil_stop")),
            Stage("apodizer", pupil_optic("apodizer")),
            Stage(
                "vortex",
                MultiScaleVortex.build(
                    charge=6, npup=npup, cap_num_airy0=512, band_subtract=True
                ),
            ),
            Stage("lyot", pupil_optic("lyot")),
            Stage("science", Fraunhofer(grid_in=pupil_grid, grid_out=science_grid)),
        )
    )
    pupil = jnp.asarray(z["pupil"], float).astype(complex)
    field = Field(data=pupil, grid=pupil_grid, plane=PlaneKind.PUPIL)
    return path, field, pupil_grid, science_grid


def regenerate_basis(npup, pupil_amplitude, pupil_grid):
    """Rebuild the script's seeded dense random ripple basis (rng(1))."""
    rng = np.random.default_rng(1)
    kr = np.sqrt(rng.uniform(KR_LO**2, KR_HI**2, N_MODES))
    ang = rng.uniform(0.0, 2 * np.pi, N_MODES)
    psi = rng.uniform(0.0, 2 * np.pi, N_MODES)
    kx, ky = kr * np.cos(ang), kr * np.sin(ang)
    x = np.asarray(pupil_grid.coords)
    xp, yp = np.meshgrid(x, x)

    def mode(k):
        ripple = np.cos(2 * np.pi * (kx[k] * xp + ky[k] * yp) + psi[k])
        return jnp.asarray(ripple) * pupil_amplitude

    return kx, ky, mode


class TestGExportReproduction:
    def test_regenerated_basis_matches_stored_frequencies(
        self, dense_export, aavc_speckle_path
    ):
        _, _, pupil_grid, _ = aavc_speckle_path
        kx, ky, _ = regenerate_basis(pupil_grid.npix, jnp.ones((1, 1)), pupil_grid)
        # Same seeded stream; ulp-level libm differences between builds.
        np.testing.assert_allclose(kx, dense_export["kx"], rtol=1e-12)
        np.testing.assert_allclose(ky, dense_export["ky"], rtol=1e-12)

    def test_e_nom_matches_export(self, dense_export, aavc_speckle_path):
        path, field, _, _ = aavc_speckle_path
        out, _ = path.propagate(field)
        scale = float(np.abs(dense_export["e_nom"]).max())
        # Measured cross-build agreement: 9.5e-11 relative to peak.
        np.testing.assert_allclose(
            np.asarray(out.data),
            dense_export["e_nom"],
            rtol=0,
            atol=1e-9 * scale,
        )

    def test_tele_peak_matches_export(self, dense_export, aavc_speckle_path):
        _, field, pupil_grid, science_grid = aavc_speckle_path
        tele = (
            jnp.abs(
                cmft_fwd(
                    field.data,
                    jnp.asarray(pupil_grid.coords),
                    jnp.asarray(science_grid.coords),
                )
            )
            ** 2
        )
        np.testing.assert_allclose(
            float(tele.max()), float(dense_export["tele_peak"]), rtol=1e-12
        )

    def test_linearize_reproduces_g_columns(self, dense_export, aavc_speckle_path):
        path, field, pupil_grid, _ = aavc_speckle_path
        amp = jnp.abs(field.data)
        _, _, mode = regenerate_basis(pupil_grid.npix, amp, pupil_grid)
        b = jnp.stack([mode(k) for k in CHECK_COLUMNS])
        basis = ModeBasis(B=b, coeffs=jnp.zeros(len(CHECK_COLUMNS)))
        lin = path.linearize(field, basis, wavelength_nm=WL_NM, method="analytic")
        for row, col in enumerate(CHECK_COLUMNS):
            ref = dense_export["G"][col]
            scale = float(np.abs(ref).max())
            np.testing.assert_allclose(
                np.asarray(lin.G[row]),
                ref,
                rtol=0,
                atol=1e-10 * scale,
                err_msg=f"G column {col}",
            )
