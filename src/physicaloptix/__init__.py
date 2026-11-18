"""physicaloptix -- physical optics (PSFs and diffraction) for the HWO suite.

A downstream consumer of optixstuff, parallel to coronagraphoto (image sim)
and jaxedith (ETC): optixstuff stays free of physical optics; diffraction
lives here.

The propagation core uses shared ``hwoutils.grids.PixelGrid`` geometry,
with its own ``PlaneKind`` / ``Field`` data model, the continuous-FT MFT pair
(``cmft_fwd`` / ``cmft_bwd``) with the plane-aware ``Fraunhofer`` wrapper and
construction-time sampling gates, ``SampledOptic`` / ``ModeBasis`` and the
``MultiScaleVortex`` ladder, and the ``OpticalPath`` fold with static taps
and ``linearize`` (the unified (E_nom, G) entry point feeding the speckle
layer and ``physicaloptix.stats``) -- validated against the cds_pipeline
EAC-1 AAVC (acceptance gates in ``tests/validation/``).

``PathCoronagraph`` wraps an ``OpticalPath`` behind optixstuff's
``AbstractImagingCoronagraph`` with performance curves derived from the propagated
PSFs. The speckle layer (``SpeckleProcess`` / ``AnalyticSpeckleField``) is
the linear speckle generator (E_nom, G) and is backend-free.
"""

from physicaloptix import stats
from physicaloptix._version import __version__
from physicaloptix.apertures import (
    eac1_primary,
    load_primary_yaml,
    normalize_unit_energy,
    rasterize_primary,
    rasterize_segments,
)
from physicaloptix.coatings import multilayer_response, sellmeier, thickness_kernel
from physicaloptix.command import CommandResponse, command_response
from physicaloptix.core import (
    AbstractChromaticOp,
    ChromaticTable,
    Field,
    PlaneKind,
    Spectrum,
)
from physicaloptix.detector import ReparameterizedDetector
from physicaloptix.diagnostics import (
    QuadratureAudit,
    mft_sampling_parameter,
    quadrature_audit,
    quadrature_audit_from_jacobians,
    quadrature_partner,
)
from physicaloptix.diff import diff_spec
from physicaloptix.elements import (
    AbstractCommandableElement,
    DispersiveScreen,
    ModeBasis,
    MultiScaleVortex,
    PhaseScreen,
    SampledOptic,
    SampledPhaseScreen,
    ZernikeWavefrontSensor,
    fourier_dm_basis,
    noll_to_nm,
    segment_ptt_basis,
    zernike_basis,
    zernike_name,
)
from physicaloptix.ifs import (
    LensletChain,
    detector_scene,
    psflet_pack,
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
from physicaloptix.polychromatic import PolychromaticPath
from physicaloptix.robustness import (
    SensitivityBudget,
    SensitivityOperators,
    pastis_matrix,
)
from physicaloptix.sources import PointSource, broadcast_to_spectrum, point_source
from physicaloptix.speckle import (
    AnalyticSpeckleField,
    ChromaticAnalyticSpeckleField,
    CrossBandMoments,
    Photometry,
    SpeckleMoments,
    SpeckleProcess,
    ou_exposure_neff,
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

__all__ = [
    "REFLECTION_OPD_FACTOR",
    "AbstractChromaticOp",
    "AbstractCommandableElement",
    "AnalyticSpeckleField",
    "BeamSplitter",
    "Branch",
    "ChannelLinearization",
    "ChromaticAnalyticSpeckleField",
    "ChromaticTable",
    "CommandResponse",
    "CrossBandMoments",
    "DispersiveScreen",
    "Field",
    "Fraunhofer",
    "Fresnel",
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
    "Photometry",
    "PlaneKind",
    "PointSource",
    "PolychromaticPath",
    "QuadratureAudit",
    "ReparameterizedDetector",
    "SampledOptic",
    "SampledPhaseScreen",
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
    "command_response",
    "detector_scene",
    "diff_spec",
    "eac1_primary",
    "fourier_dm_basis",
    "linearity_residual",
    "linearize",
    "linearize_shared",
    "linearize_stages",
    "load_primary_yaml",
    "load_train_yaml",
    "mft_sampling_parameter",
    "multilayer_response",
    "ncpa_differential_opd",
    "noll_to_nm",
    "normalize_unit_energy",
    "ou_exposure_neff",
    "pastis_matrix",
    "point_source",
    "psflet_pack",
    "quadrature_audit",
    "quadrature_audit_from_jacobians",
    "quadrature_partner",
    "rasterize_primary",
    "rasterize_segments",
    "segment_ptt_basis",
    "sellmeier",
    "stats",
    "synthesize_psd_surface",
    "telescope_peak",
    "thickness_kernel",
    "zernike_basis",
    "zernike_name",
]


def __getattr__(name):
    # physicaloptix.viz is a lazy namespace that imports no eyepiece itself,
    # so attribute access after a bare `import physicaloptix` loads it here.
    if name == "viz":
        import importlib

        return importlib.import_module("physicaloptix.viz")
    msg = f"module 'physicaloptix' has no attribute '{name}'"
    raise AttributeError(msg)
