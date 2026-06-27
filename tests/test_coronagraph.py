"""DLuxCoronagraph: a dLux-backed optixstuff coronagraph.

The interop pattern under test: build it from an optixstuff primary and call the
optixstuff AbstractCoronagraph interface to get PSFs -- no dLux object in sight.
"""

import dLux as dl
import numpy as np
import pytest
from conftest import nyquist_rad
from optixstuff.coronagraph import AbstractCoronagraph

import physicaloptix as po


class TestInteropSurface:
    def test_is_optixstuff_coronagraph_not_a_dlux_object(self, eac5_primary):
        coro = po.DLuxCoronagraph.from_primary(eac5_primary)
        assert isinstance(coro, AbstractCoronagraph)
        assert not isinstance(coro, dl.AngularOpticalSystem)

    def test_one_liner_facade(self, eac5_primary):
        out = po.psf(eac5_primary, 600.0, nyquist_rad(eac5_primary.diameter_m), 64)
        assert np.asarray(out).shape == (64, 64)


class TestOnAxisPSF:
    def test_shape_and_normalisation(self, eac5_primary):
        coro = po.DLuxCoronagraph.from_primary(eac5_primary)
        out = np.asarray(
            coro.on_axis_psf(600.0, nyquist_rad(eac5_primary.diameter_m), 128)
        )
        assert out.shape == (128, 128)
        # no mask -> telescope PSF integrates to ~unit flux
        assert out.sum() > 0.97

    def test_interface_honours_requested_sampling(self, eac5_primary):
        coro = po.DLuxCoronagraph.from_primary(eac5_primary)
        ps = nyquist_rad(eac5_primary.diameter_m)
        assert np.asarray(coro.on_axis_psf(600.0, ps, 64)).shape == (64, 64)
        assert np.asarray(coro.on_axis_psf(600.0, ps, 200)).shape == (200, 200)


class TestOffAxisPSF:
    def test_off_axis_shifts_the_peak(self, eac5_primary):
        coro = po.DLuxCoronagraph.from_primary(eac5_primary)
        ps = nyquist_rad(eac5_primary.diameter_m)
        on = np.asarray(coro.on_axis_psf(600.0, ps, 128))
        off = np.asarray(coro.off_axis_psf(600.0, 10.0, ps, 128))  # 10 lambda/D
        on_peak = np.unravel_index(int(on.argmax()), on.shape)
        off_peak = np.unravel_index(int(off.argmax()), off.shape)
        assert on_peak[1] == pytest.approx(64, abs=2)
        assert off_peak[1] > on_peak[1] + 20


class TestScalarInterface:
    def test_scalar_props_passthrough(self, eac5_primary):
        coro = po.DLuxCoronagraph.from_primary(
            eac5_primary, core_throughput=0.2, raw_contrast=1e-10
        )
        assert float(coro.throughput(5.0, 600.0)) == pytest.approx(0.2)
        assert float(coro.core_mean_intensity(5.0, 600.0)) == pytest.approx(1e-10)


class TestAdapterDispatch:
    def test_simple_primary_routes_to_circular(self, simple_primary):
        coro = po.DLuxCoronagraph.from_primary(simple_primary)
        out = np.asarray(coro.on_axis_psf(600.0, nyquist_rad(6.0), 64))
        assert out.shape == (64, 64)
        assert out.sum() > 0.95
