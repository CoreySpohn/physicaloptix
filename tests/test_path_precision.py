"""Precision closure of optical paths, beam splitting, and sky illumination."""

import numpy as np
import pytest
from hwoutils.leaves import canonical
from hwoutils.testing import assert_precision_closed, precision_pair, ulps

from physicaloptix._sky import entrance_energy, uniform_sky_image, unit_energy
from physicaloptix.core import Field
from physicaloptix.path import OpticalPath
from physicaloptix.system import BeamSplitter, Branch, OpticalSystem, SplitterPort
from physicaloptix.transforms import Fraunhofer


def _band_sky(system, field):
    camera = Fraunhofer(
        field.grid,
        system.branches[0].path.stages[0].op.grid_out,
        reference_wavelength_nm=500.0,
    )
    mono = Field(field.data[0], field.grid, field.plane)
    return uniform_sky_image(
        system.trunk, camera, mono, n_screens=2, seed=0, sky_band_nm=(475.0, 525.0)
    )


@pytest.mark.float64_only(reason="trace a float32 path with x64 available")
@pytest.mark.parametrize(
    "forward",
    [
        lambda s, f: s.trunk.propagate(f),
        lambda s, f: s.split(f),
        lambda s, f: SplitterPort(s.split, "reflect")(f),
        lambda s, f: s.propagate(f),
        lambda s, f: entrance_energy(f),
        lambda s, f: unit_energy(f),
        lambda s, f: uniform_sky_image(
            s.trunk, s.branches[0].path.stages[0].op, f, n_screens=2, seed=0
        ),
        _band_sky,
    ],
    ids=[
        "path",
        "split",
        "port",
        "system",
        "energy",
        "normalized",
        "sky",
        "continuous-band-sky",
    ],
)
def test_path_precision(small_path, chromatic_field, forward):
    """Fails if spectral interpolation, a pupil screen, or a camera widens.

    For a 24-sample pupil each Fourier axis has 24 complex products. Allow
    four real roundings per product and accumulation, two axes, and twice
    the field error for intensity. Compare at each output leaf's peak.
    """
    split = BeamSplitter.dichroic(
        wavelengths_nm=[400.0, 600.0],
        transmittance=[0.25, 0.75],
        grid=chromatic_field.grid,
        plane=chromatic_field.plane,
    )
    system = OpticalSystem(
        OpticalPath(small_path.stages[:1]),
        split,
        (Branch("science", "transmit", OpticalPath(small_path.stages[1:])),),
    )
    args = system, chromatic_field
    assert_precision_closed(forward, *canonical(args, precision="float32"))
    precision_pair(forward, args, bound=ulps(16 * 24, np.float32(1)))
