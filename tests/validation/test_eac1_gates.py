"""EAC-1 AAVC acceptance gates: the owned chain vs the cds_pipeline cache.

The gates:

- Gate 0: the bare telescope PSF matches cds/HCIPy at machine precision.
- Gate 1: off-axis (planet) PSFs through the full AAVC chain match to ~1e-5.
- Multi-scale: the on-axis stellar null floor matches the cds floor (~3.05e-11
  measured against 3.043e-11, 0.2 percent).

They need the cds reference cache (see ``eac1_cache`` in conftest) and run the
full 2048-pixel chain, so they take minutes on CPU; everything expensive is
computed once in a session fixture.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.elements import MultiScaleVortex, SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.transforms import Fraunhofer, cmft_fwd


def scale_fit_residual(ours, reference):
    """Least-squares scale of ours onto reference and the relative residual."""
    a = np.asarray(ours, float)
    b = np.asarray(reference, float)
    s = float((a * b).sum() / (a * a).sum())
    resid = float(np.linalg.norm(b - s * a) / np.linalg.norm(b))
    return s, resid


@pytest.fixture(scope="session")
def aavc(eac1_cache):
    """The full EAC-1 AAVC train and its propagated PSFs, computed once."""
    z = eac1_cache
    npup, _, _, _, charge, _, pixscale_lod, dims = z["meta"]
    npup, dims, charge = int(npup), int(dims), int(charge)

    pupil_grid = Grid.pupil(npup)
    science_grid = Grid.focal(dims, float(pixscale_lod))

    def pupil_optic(name):
        return SampledOptic(
            transmission=jnp.asarray(z[name], float),
            grid=pupil_grid,
            plane=PlaneKind.PUPIL,
        )

    vortex = MultiScaleVortex.build(
        charge=charge, npup=npup, cap_num_airy0=npup // 2, band_subtract=True
    )
    train = OpticalPath(
        stages=(
            Stage("pupil_stop", pupil_optic("pupil_stop")),
            Stage("apodizer", pupil_optic("apodizer")),
            Stage("vortex", vortex),
            Stage("lyot", pupil_optic("lyot")),
            Stage("science", Fraunhofer(grid_in=pupil_grid, grid_out=science_grid)),
        )
    )

    pupil = jnp.asarray(z["pupil"], float).astype(complex)
    x = jnp.asarray(pupil_grid.coords)

    def source(separation_lod=0.0):
        tilt = jnp.exp(2j * jnp.pi * separation_lod * x)[None, :]
        return Field(data=pupil * tilt, grid=pupil_grid, plane=PlaneKind.PUPIL)

    def image(separation_lod=0.0, taps=()):
        out, tapped = train.propagate(source(separation_lod), taps=taps)
        return np.asarray(out.intensity()), tapped

    tele = np.abs(np.asarray(cmft_fwd(pupil, x, jnp.asarray(science_grid.coords)))) ** 2

    psf_on, _ = image()
    psf_off7, _ = image(7.0)
    psf_off15, _ = image(15.0)

    return {
        "z": z,
        "train": train,
        "source": source,
        "tele": tele,
        "psf_on": psf_on,
        "psf_off7": psf_off7,
        "psf_off15": psf_off15,
    }


class TestGate0Telescope:
    def test_telescope_psf_matches_cds_at_machine_precision(self, aavc):
        s, resid = scale_fit_residual(aavc["tele"], aavc["z"]["tele_psf"])
        assert resid < 1e-9, f"gate 0 residual {resid:.3e} (scale {s:.6g})"


class TestGate1OffAxis:
    def test_offaxis_7_lod(self, aavc):
        s, resid = scale_fit_residual(aavc["psf_off7"], aavc["z"]["offax_7"])
        assert resid < 1e-4, f"off-axis 7 residual {resid:.3e} (scale {s:.6g})"

    def test_offaxis_15_lod(self, aavc):
        s, resid = scale_fit_residual(aavc["psf_off15"], aavc["z"]["offax_15"])
        assert resid < 1e-4, f"off-axis 15 residual {resid:.3e} (scale {s:.6g})"


class TestMultiScaleNull:
    def test_onaxis_raw_contrast_floor_matches_cds(self, aavc):
        floor = float(aavc["psf_on"].max() / aavc["tele"].max())
        z = aavc["z"]
        cds_floor = float(z["onaxis_psf"].max() / z["tele_psf"].max())
        rel = abs(floor - cds_floor) / cds_floor
        assert floor < 5e-11, f"on-axis floor {floor:.3e}"
        assert rel < 0.05, f"floor {floor:.3e} vs cds {cds_floor:.3e} (rel {rel:.2%})"


class TestTrainMechanics:
    def test_taps_do_not_perturb_the_gate_chain(self, aavc):
        """Taps-on returns bit-identical science data on the real chain."""
        out_plain, _ = aavc["train"].propagate(aavc["source"]())
        out_tapped, taps = aavc["train"].propagate(aavc["source"](), taps=("lyot",))
        np.testing.assert_array_equal(
            np.asarray(out_plain.data), np.asarray(out_tapped.data)
        )
        assert taps["lyot"].plane is PlaneKind.PUPIL
