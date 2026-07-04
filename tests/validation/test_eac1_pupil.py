"""Validation gate: the synthesized EAC-1 pupil vs the design-survey FITS.

The owned rasterizer must reproduce the HWO Coronagraph Design Survey
baseline pupil (2048 px across 7.2 m, 16x supersampled gray pixels) from the
bundled geometry YAML, and the unit-energy-normalized pupil must match the
cached reference the acceptance gates run on. Finally the full AAVC chain is
re-run on the synthesized pupil: because the design masks were optimized for
this exact pupil, the gate thresholds are unchanged.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.apertures import (
    eac1_primary,
    normalize_unit_energy,
    rasterize_primary,
)
from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import MultiScaleVortex, SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer, cmft_fwd

_FITS_CANDIDATES = (
    Path(__file__).parents[1] / "data" / "pupil_eac1_2048.fits.gz",
    # Development-workspace clone of the design-survey pipeline.
    Path(__file__).parents[3]
    / "cds_pipeline/cds_pipeline/baseline/data"  # internal-ref-ok
    / "pupil_eac1_2048.fits.gz",
)


@pytest.fixture(scope="session")
def cds_pupil_fits():
    fits = pytest.importorskip("astropy.io.fits")
    for path in _FITS_CANDIDATES:
        if path.exists():
            return np.asarray(fits.getdata(path), dtype=float)
    pytest.skip("design-survey EAC-1 pupil FITS not found")


@pytest.fixture(scope="session")
def synthesized_pupil():
    return rasterize_primary(eac1_primary(), 2048, supersample=16)


class TestPupilReproduction:
    def test_matches_design_survey_fits(self, synthesized_pupil, cds_pupil_fits):
        diff = synthesized_pupil - cds_pupil_fits
        differing = np.abs(diff) > 0
        # Every difference is a single supersample count on a gray edge
        # pixel (928 pixels measured; edge-tie arithmetic, see module doc).
        assert float(np.abs(diff).max()) <= 1.0 / 256 + 1e-15
        assert int(differing.sum()) < 2000, f"{int(differing.sum())} differing pixels"
        # A tie can push a pixel to fully open in one raster and 255/256 in
        # the other, so "edge" means gray in either raster.
        edge = ((cds_pupil_fits > 0) & (cds_pupil_fits < 1)) | (
            (synthesized_pupil > 0) & (synthesized_pupil < 1)
        )
        assert bool(np.all(edge[differing]))

    def test_normalized_matches_gate_cache(self, synthesized_pupil, eac1_cache):
        normalized = normalize_unit_energy(synthesized_pupil, 7.2 / 2048)
        cached = eac1_cache["pupil"].astype(float)
        # Bounded by one supersample count at the cache scale (edge ties);
        # fully-open pixels agree to the energy-rescale level those tie
        # pixels induce (~1e-6 relative).
        scale = float(cached.max())
        np.testing.assert_allclose(normalized, cached, rtol=0, atol=scale / 256 * 1.01)
        np.testing.assert_allclose(float(normalized.max()), scale, rtol=1e-5)


@pytest.fixture(scope="module")
def chain(synthesized_pupil, eac1_cache):
    z = eac1_cache
    npup, dims = int(z["meta"][0]), int(z["meta"][7])
    charge, ps_lod = int(z["meta"][4]), float(z["meta"][6])
    pupil_grid = Grid.pupil(npup)
    science_grid = Grid.focal(dims, ps_lod)

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
                    charge=charge, npup=npup, cap_num_airy0=npup // 2
                ),
            ),
            Stage("lyot", pupil_optic("lyot")),
            Stage("science", Fraunhofer(grid_in=pupil_grid, grid_out=science_grid)),
        )
    )
    pupil = jnp.asarray(normalize_unit_energy(synthesized_pupil, 7.2 / npup)).astype(
        complex
    )
    field = Field(data=pupil, grid=pupil_grid, plane=PlaneKind.PUPIL)
    return path, field, pupil_grid, science_grid


class TestGatesOnSynthesizedPupil:
    def test_gates_hold_on_synthesized_pupil(self, chain, eac1_cache):
        path, field, pupil_grid, science_grid = chain
        z = eac1_cache
        x = jnp.asarray(pupil_grid.coords)
        u = jnp.asarray(science_grid.coords)

        tele = np.abs(np.asarray(cmft_fwd(field.data, x, u))) ** 2

        def scale_fit_residual(ours, reference):
            a = np.asarray(ours, float)
            b = np.asarray(reference, float)
            s = float((a * b).sum() / (a * a).sum())
            return float(np.linalg.norm(b - s * a) / np.linalg.norm(b))

        assert scale_fit_residual(tele, z["tele_psf"]) < 1e-6

        on, _ = path.propagate(field)
        psf_on = np.asarray(on.intensity())
        floor = psf_on.max() / tele.max()
        cds_floor = float(z["onaxis_psf"].max() / z["tele_psf"].max())
        assert floor < 5e-11
        assert abs(floor - cds_floor) / cds_floor < 0.05

        tilt = jnp.exp(2j * jnp.pi * 7.0 * x)[None, :]
        off = Field(data=field.data * tilt, grid=pupil_grid, plane=PlaneKind.PUPIL)
        psf_off, _ = path.propagate(off)
        resid = scale_fit_residual(np.asarray(psf_off.intensity()), z["offax_7"])
        assert resid < 1e-4
