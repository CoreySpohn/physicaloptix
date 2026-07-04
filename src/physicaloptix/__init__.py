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

``DLuxCoronagraph`` (with ``to_dlux_aperture`` and the ``psf`` facade) is the
legacy dLux-backed path behind optixstuff's ``AbstractCoronagraph``, kept
until the path-backed adapter replaces it. The speckle layer
(``SpeckleProcess`` / ``AnalyticSpeckleField``) is the linear speckle
generator (E_nom, G) and is backend-free.
"""

from physicaloptix._version import __version__
from physicaloptix.apertures import to_dlux_aperture
from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.coronagraph import DLuxCoronagraph, psf
from physicaloptix.diagnostics import mft_sampling_parameter
from physicaloptix.diff import diff_spec
from physicaloptix.elements import ModeBasis, MultiScaleVortex, SampledOptic
from physicaloptix.linearize import Linearization, linearity_residual, linearize
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.speckle import AnalyticSpeckleField, SpeckleProcess
from physicaloptix.transforms import Fraunhofer, cmft_bwd, cmft_fwd

__all__ = [
    "AnalyticSpeckleField",
    "DLuxCoronagraph",
    "Field",
    "Fraunhofer",
    "Grid",
    "Linearization",
    "ModeBasis",
    "MultiScaleVortex",
    "OpticalPath",
    "PlaneKind",
    "SampledOptic",
    "SpeckleProcess",
    "Spectrum",
    "Stage",
    "__version__",
    "cmft_bwd",
    "cmft_fwd",
    "diff_spec",
    "linearity_residual",
    "linearize",
    "mft_sampling_parameter",
    "psf",
    "to_dlux_aperture",
]
