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
from physicaloptix.coatings import multilayer_response, sellmeier, thickness_kernel
from physicaloptix.core import Field, Grid, PlaneKind, Spectrum
from physicaloptix.detector import read_detector
from physicaloptix.diagnostics import mft_sampling_parameter
from physicaloptix.diff import diff_spec
from physicaloptix.elements import (
    DispersiveScreen,
    ModeBasis,
    MultiScaleVortex,
    PhaseScreen,
    SampledOptic,
    ZernikeWavefrontSensor,
    fourier_dm_basis,
    segment_ptt_basis,
    zernike_basis,
)
from physicaloptix.ifs import (
    LensletChain,
    detector_scene,
    psflet_pack,
    save_psflet_pack,
)
from physicaloptix.interop import PathCoronagraph
from physicaloptix.linearize import (
    Linearization,
    linearity_residual,
    linearize,
    linearize_stages,
)
from physicaloptix.multichannel import (
    ChannelLinearization,
    MultiChannelLinearization,
    linearize_shared,
    ncpa_differential_opd,
)
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.robustness import (
    SensitivityBudget,
    SensitivityOperators,
    pastis_matrix,
)
from physicaloptix.sources import broadcast_to_spectrum, point_source
from physicaloptix.speckle import (
    AnalyticSpeckleField,
    CrossBandMoments,
    SpeckleMoments,
    SpeckleProcess,
    lambda_scaled_channels,
    telescope_peak,
)
from physicaloptix.system import BeamSplitter, Branch, OpticalSystem, SplitterPort
from physicaloptix.trains import (
    REFLECTION_OPD_FACTOR,
    MirrorSpec,
    build_mirror_train,
    load_train_yaml,
    synthesize_psd_surface,
)
from physicaloptix.transforms import Fraunhofer, Fresnel, cmft_bwd, cmft_fwd
from physicaloptix.viz import render_path

__all__ = [
    "REFLECTION_OPD_FACTOR",
    "AnalyticSpeckleField",
    "BeamSplitter",
    "Branch",
    "ChannelLinearization",
    "CrossBandMoments",
    "DispersiveScreen",
    "Field",
    "Fraunhofer",
    "Fresnel",
    "Grid",
    "LensletChain",
    "Linearization",
    "MirrorSpec",
    "ModeBasis",
    "MultiChannelLinearization",
    "MultiScaleVortex",
    "OpticalPath",
    "OpticalSystem",
    "PathCoronagraph",
    "PhaseScreen",
    "PlaneKind",
    "SampledOptic",
    "SensitivityBudget",
    "SensitivityOperators",
    "SpeckleMoments",
    "SpeckleProcess",
    "Spectrum",
    "SplitterPort",
    "Stage",
    "ZernikeWavefrontSensor",
    "__version__",
    "broadcast_to_spectrum",
    "build_mirror_train",
    "cmft_bwd",
    "cmft_fwd",
    "detector_scene",
    "diff_spec",
    "eac1_primary",
    "fourier_dm_basis",
    "lambda_scaled_channels",
    "linearity_residual",
    "linearize",
    "linearize_shared",
    "linearize_stages",
    "load_primary_yaml",
    "load_train_yaml",
    "mft_sampling_parameter",
    "multilayer_response",
    "ncpa_differential_opd",
    "normalize_unit_energy",
    "pastis_matrix",
    "point_source",
    "psflet_pack",
    "rasterize_primary",
    "rasterize_segments",
    "read_detector",
    "render_path",
    "save_psflet_pack",
    "segment_ptt_basis",
    "sellmeier",
    "synthesize_psd_surface",
    "telescope_peak",
    "thickness_kernel",
    "zernike_basis",
]
