"""Instrument-specific optical-path builders."""

from physicaloptix.instruments.nircam import (
    BandLimitedRoundMask,
    NIRCamBand,
    NIRCamConfig,
    NIRCamInputs,
    build_nircam,
    mask_plane_field,
    nircam_band_image,
    pixel_integrate,
)

__all__ = [
    "BandLimitedRoundMask",
    "NIRCamBand",
    "NIRCamConfig",
    "NIRCamInputs",
    "build_nircam",
    "mask_plane_field",
    "nircam_band_image",
    "pixel_integrate",
]
