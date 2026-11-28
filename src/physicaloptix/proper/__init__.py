"""PROPER-style propagation operations: array helpers and pilot-beam planning.

``ops`` holds the centered transforms, resampling weights and masks; ``beam``
holds the pilot beam, which resolves a propagation schedule in NumPy when a
model is built, and the JAX executor of the resulting operation list.
"""

from physicaloptix.proper.beam import (
    Op,
    PilotBeam,
    apply_op,
    plan_lens,
    plan_propagate,
    run_ops,
)
from physicaloptix.proper.ops import (
    cubic_conv_weights,
    ellipse_mask,
    fft_backward,
    fft_forward,
    ffts,
    mft2,
    noll_z6,
    proper_trim,
    resample_map,
    shift_center,
    szoom_weights,
)

__all__ = [
    "Op",
    "PilotBeam",
    "apply_op",
    "cubic_conv_weights",
    "ellipse_mask",
    "fft_backward",
    "fft_forward",
    "ffts",
    "mft2",
    "noll_z6",
    "plan_lens",
    "plan_propagate",
    "proper_trim",
    "resample_map",
    "run_ops",
    "shift_center",
    "szoom_weights",
]
