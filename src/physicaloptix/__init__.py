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
)
from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.diagnostics import mft_sampling_parameter
from physicaloptix.diff import diff_spec
from physicaloptix.elements import ModeBasis, MultiScaleVortex, SampledOptic
from physicaloptix.interop import PathCoronagraph
from physicaloptix.linearize import Linearization, linearity_residual, linearize
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.speckle import AnalyticSpeckleField, SpeckleProcess
from physicaloptix.transforms import Fraunhofer, cmft_bwd, cmft_fwd
from physicaloptix.viz import render_path

__all__ = [
    "AnalyticSpeckleField",
    "Field",
    "Fraunhofer",
    "Grid",
    "Linearization",
    "ModeBasis",
    "MultiScaleVortex",
    "OpticalPath",
    "PathCoronagraph",
    "PlaneKind",
    "SampledOptic",
    "SpeckleProcess",
    "Spectrum",
    "Stage",
    "__version__",
    "cmft_bwd",
    "cmft_fwd",
    "diff_spec",
    "eac1_primary",
    "linearity_residual",
    "linearize",
    "load_primary_yaml",
    "mft_sampling_parameter",
    "normalize_unit_energy",
    "rasterize_primary",
    "render_path",
]
