"""Instrument-specific optical-path builders."""

from physicaloptix.instruments.nircam import (
    BandLimitedRoundMask,
    NIRCamBand,
    NIRCamConfig,
    NIRCamInputs,
    build_nircam,
    integrate_detector_pixels,
    mask_plane_field,
    nircam_band_image,
)

__all__ = [
    "BandLimitedRoundMask",
    "NIRCamBand",
    "NIRCamConfig",
    "NIRCamInputs",
    "build_nircam",
    "integrate_detector_pixels",
    "mask_plane_field",
    "nircam_band_image",
]
