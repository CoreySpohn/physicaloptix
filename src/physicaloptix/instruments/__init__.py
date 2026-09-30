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
from physicaloptix.instruments.roman import RomanCompact
from physicaloptix.instruments.roman_full import RomanFull, compile_train

__all__ = [
    "BandLimitedRoundMask",
    "NIRCamBand",
    "NIRCamConfig",
    "NIRCamInputs",
    "RomanCompact",
    "RomanFull",
    "build_nircam",
    "compile_train",
    "integrate_detector_pixels",
    "mask_plane_field",
    "nircam_band_image",
]
