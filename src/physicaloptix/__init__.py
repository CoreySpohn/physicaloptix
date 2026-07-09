"""physicaloptix -- physical optics (PSFs and diffraction) for the HWO suite.

A downstream consumer of optixstuff, parallel to coronagraphoto (image sim)
and jaxedith (ETC): optixstuff stays free of physical optics; diffraction
lives here.

The propagation core is owned: ``Grid`` /
``PlaneKind`` / ``Field`` data model, the continuous-FT MFT pair
(``cmft_fwd`` / ``cmft_bwd``) with the plane-aware ``Fraunhofer`` wrapper and
construction-time sampling gates, ``SampledOptic`` / ``ModeBasis`` and the
``MultiScaleVortex`` ladder, and the ``OpticalPath`` fold with static taps
and ``linearize`` (the unified (E_nom, G) entry point feeding the speckle
layer and ``physicaloptix.stats``) -- validated against the cds_pipeline
EAC-1 AAVC (acceptance gates in ``tests/validation/``).

``PathCoronagraph`` wraps an ``OpticalPath`` behind optixstuff's
``AbstractCoronagraph`` with performance curves derived from the propagated
PSFs. The speckle layer (``SpeckleProcess`` / ``AnalyticSpeckleField``) is
the linear speckle generator (E_nom, G) and is backend-free.
"""

from physicaloptix._version import __version__
from physicaloptix.apertures import (
    eac1_primary,
    load_primary_yaml,
    normalize_unit_energy,
    rasterize_primary,
    rasterize_segments,
)
from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.detector import read_detector
from physicaloptix.diagnostics import mft_sampling_parameter
from physicaloptix.diff import diff_spec
from physicaloptix.elements import (
    ModeBasis,
    MultiScaleVortex,
    PhaseScreen,
    SampledOptic,
    ZernikeWavefrontSensor,
    fourier_dm_basis,
    segment_ptt_basis,
    zernike_basis,
)
from physicaloptix.interop import PathCoronagraph
from physicaloptix.linearize import Linearization, linearity_residual, linearize
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import broadcast_to_spectrum, point_source
from physicaloptix.speckle import AnalyticSpeckleField, SpeckleProcess
from physicaloptix.system import BeamSplitter
from physicaloptix.transforms import Fraunhofer, Fresnel, cmft_bwd, cmft_fwd
from physicaloptix.viz import render_path

__all__ = [
    "AnalyticSpeckleField",
    "BeamSplitter",
    "Field",
    "Fraunhofer",
    "Fresnel",
    "Grid",
    "Linearization",
    "ModeBasis",
    "MultiScaleVortex",
    "OpticalPath",
    "PathCoronagraph",
    "PhaseScreen",
    "PlaneKind",
    "SampledOptic",
    "SpeckleProcess",
    "Spectrum",
    "Stage",
    "ZernikeWavefrontSensor",
    "__version__",
    "broadcast_to_spectrum",
    "cmft_bwd",
    "cmft_fwd",
    "diff_spec",
    "eac1_primary",
    "fourier_dm_basis",
    "linearity_residual",
    "linearize",
    "load_primary_yaml",
    "mft_sampling_parameter",
    "normalize_unit_energy",
    "point_source",
    "rasterize_primary",
    "rasterize_segments",
    "read_detector",
    "render_path",
    "segment_ptt_basis",
    "zernike_basis",
]
