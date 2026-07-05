"""Validation: broadband EAC-1 AAVC propagation on the fixed angular grid.

The mask-only chain is achromatic in the dimensionless core, so an on-axis
broadband propagation is one core propagation broadcast over the band with a
per-wavelength science projection; the deep null must survive band-averaging
(each slice samples the same native null). A chromatic off-axis source needs
per-wavelength core propagation, and its peak must sit at a fixed angle.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.elements import MultiScaleVortex, SampledOptic
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import broadcast_to_spectrum, point_source
from physicaloptix.transforms import Fraunhofer, cmft_fwd

REF_NM = 1000.0


@pytest.fixture(scope="module")
def aavc_core(eac1_cache):
    z = eac1_cache
    npup = int(z["meta"][0])
    charge = int(z["meta"][4])
    pupil_grid = Grid.pupil(npup)

    def pupil_optic(name):
        return SampledOptic(
            transmission=jnp.asarray(z[name], float),
            grid=pupil_grid,
            plane=PlaneKind.PUPIL,
        )

    core = OpticalPath(
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
        )
    )
    pupil = jnp.asarray(z["pupil"], float).astype(complex)
    field = Field(data=pupil, grid=pupil_grid, plane=PlaneKind.PUPIL)
    dims, ps_lod = int(z["meta"][7]), float(z["meta"][6])
    science = Fraunhofer(
        grid_in=pupil_grid,
        grid_out=Grid.focal(dims, ps_lod),
        reference_wavelength_nm=REF_NM,
        min_wavelength_nm=0.9 * REF_NM,
    )
    return core, field, science


class TestBroadbandNull:
    def test_band_averaged_floor_holds_the_mono_null(self, aavc_core):
        """Band-averaging an achromatic null can only dilute its hot pixel.

        Each slice samples the same native residual field at breathing
        coordinates, so only the reference slice hits the mono hot pixel
        exactly and the band-averaged max sits at or below the mono floor
        (measured: 7.8e-12 vs 3.05e-11, a 4x dilution of the max statistic).
        """
        core, field, science = aavc_core
        band = Spectrum.tophat(REF_NM, 0.1, 5)

        lyot_mono, _ = core.propagate(field)
        mono_psf = np.asarray(science(lyot_mono).intensity())

        lyot_band = broadcast_to_spectrum(lyot_mono, band)
        out = science(lyot_band)
        slices = np.asarray(out.data.real**2 + out.data.imag**2)
        broadband = np.asarray(out.intensity())

        # The reference-wavelength slice IS the mono propagation.
        np.testing.assert_allclose(slices[2], mono_psf, rtol=1e-10)

        x = jnp.asarray(field.grid.coords)
        tele = (
            np.abs(
                np.asarray(
                    cmft_fwd(field.data, x, jnp.asarray(science.grid_out.coords))
                )
            )
            ** 2
        )
        floor_mono = mono_psf.max() / tele.max()
        floor_band = broadband.max() / tele.max()
        assert floor_band < 5e-11
        assert floor_band <= floor_mono * 1.05
        assert floor_band > floor_mono * 0.02


class TestBroadbandPlanet:
    def test_chromatic_planet_sits_at_fixed_angle(self, aavc_core):
        core, field, science = aavc_core
        band = Spectrum.tophat(REF_NM, 0.1, 3)
        source = point_source(
            field,
            spectrum=band,
            separation_lod=7.875,  # a pixel center of the 0.25-px grid
            reference_wavelength_nm=REF_NM,
        )
        lyot, _ = core.propagate(source)
        out = science(lyot)
        intensity = np.asarray(out.data.real**2 + out.data.imag**2)
        peaks = [
            np.unravel_index(np.argmax(intensity[i]), intensity[i].shape)
            for i in range(len(band))
        ]
        assert len({tuple(p) for p in peaks}) == 1
        coords = np.asarray(out.grid.coords)
        peak_x = coords[peaks[0][1]]
        assert abs(peak_x - 7.875) < 0.25
