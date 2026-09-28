"""JWST NIRCam imaging path built from an exported optical prescription.

The builder reproduces the plane sequence STPSF resolves for a NIRCam
calculation without a focal-plane mask: the entrance pupil (amplitude and,
optionally, the fixed OTE OPD), the coordinate inversion in y that STPSF
places after the telescope, and a matrix Fourier transform to the science
focal plane sampled at the detector pixel scale times ``oversample``. The
output grid matches STPSF's ``OVERSAMP`` product: ``detector_npix *
oversample`` samples per side with the optical axis at index ``(n - 1) / 2``.

Frames and signs:

- The pupil arrays are in the entrance-pupil frame (+x = +V2, +y = +V3,
  rows are y). The inversion reflects them in y, so the detector frame is
  the entrance frame with y reversed.
- Source positions are given in the detector-oriented output frame (x right,
  y up), as STPSF's ``source_offset_x`` / ``source_offset_y``. The tilt is
  applied at the entrance pupil, reflected through the inversion, so the
  image lands at ``+position``.
- OPDs held by ``NIRCamInputs`` are in this library's convention: nanometers
  of accumulated path, phasor ``exp(+2j pi opd / lambda)`` with the
  ``exp(-2j pi u x)`` forward kernel. ``NIRCamInputs.from_bundle`` converts
  an STPSF OPD into it once, at the import boundary.

Normalization: the source field is scaled so the field just after the
entrance pupil carries unit energy (``Field.energy() == 1``), the
entrance-pupil normalization STPSF applies with ``normalize='first'``. No
later renormalization happens, so ``out.intensity() * out.grid.weights`` is
the fraction of entrance-pupil energy per output sample and finite-field
losses are kept.
"""

import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import arcsec_to_lambda_d
from jaxtyping import Array

from physicaloptix.core import Field, Grid, PlaneKind, validate_field
from physicaloptix.elements import ModeBasis, PhaseScreen, SampledOptic
from physicaloptix.elements.base import Element
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import point_source
from physicaloptix.transforms import Fraunhofer

M_TO_NM = 1e9
STPSF_PHASOR = "exp(+i*2*pi*OPD/lambda)"


def _fits():
    try:
        from astropy.io import fits
    except ImportError as err:
        raise ImportError(
            "reading a prescription bundle needs astropy (pip install astropy)"
        ) from err
    return fits


class CoordinateInversion(Element):
    """Reflect the field about the optical axis along one axis.

    On the symmetric half-pixel grid, reversing the array along an axis maps
    each coordinate to its negative exactly, so the reflection is lossless.
    """

    grid: Grid
    axis: str = eqx.field(static=True, default="y")
    plane: PlaneKind = eqx.field(static=True, default=PlaneKind.PUPIL)

    def __check_init__(self):
        """Validate the axis name."""
        if self.axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {self.axis!r}")

    def __call__(self, field):
        """Reverse the field along ``axis`` (leading axes broadcast)."""
        validate_field(
            field, plane=self.plane, grid=self.grid, context="CoordinateInversion"
        )
        data = field.data[..., ::-1, :] if self.axis == "y" else field.data[..., ::-1]
        return Field(
            data=data, grid=field.grid, plane=field.plane, spectrum=field.spectrum
        )


class NIRCamConfig(eqx.Module):
    """What to compute: wavelength, source position, stages and sampling.

    Attributes:
        wavelength_nm: Monochromatic wavelength in nanometers.
        pixel_scale_arcsec: Detector pixel scale in arcseconds per pixel.
        detector_npix: Output field size in detector pixels per side.
        oversample: Output samples per detector pixel along each axis.
        source_position_arcsec: ``(x, y)`` source offset in arcseconds in the
            detector-oriented output frame.
        ote_opd: Apply the fixed OTE OPD in the entrance pupil.
    """

    wavelength_nm: float = eqx.field(static=True)
    pixel_scale_arcsec: float = eqx.field(static=True)
    detector_npix: int = eqx.field(static=True, default=81)
    oversample: int = eqx.field(static=True, default=4)
    source_position_arcsec: tuple[float, float] = eqx.field(
        static=True, default=(0.0, 0.0), converter=tuple
    )
    ote_opd: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        """Reject non-physical values at construction."""
        if not self.wavelength_nm > 0:
            raise ValueError(
                f"wavelength_nm must be positive, got {self.wavelength_nm}"
            )
        if not self.pixel_scale_arcsec > 0:
            raise ValueError(
                f"pixel_scale_arcsec must be positive, got {self.pixel_scale_arcsec}"
            )
        if self.detector_npix < 1 or self.oversample < 1:
            raise ValueError(
                "detector_npix and oversample must be at least 1, got "
                f"{self.detector_npix} and {self.oversample}"
            )
        if len(self.source_position_arcsec) != 2:
            raise ValueError(
                "source_position_arcsec must be (x, y), got "
                f"{self.source_position_arcsec}"
            )


class NIRCamInputs(eqx.Module):
    """The optical prescription arrays, their physical sampling and provenance.

    Attributes:
        pupil_amplitude: Entrance-pupil electric-field amplitude transmission,
            square ``(n, n)``, entrance-pupil frame (+x = +V2, +y = +V3, rows
            are y), optical axis at index ``(n - 1) / 2``.
        pupil_diameter_m: Physical side length of the pupil array in meters;
            it sets the lambda/D unit of the output grid.
        ote_opd_nm: Optional fixed OTE OPD on the same grid, in nanometers in
            this library's sign convention (see the module docstring).
        provenance: Where the arrays came from (file names, checksums).
    """

    pupil_amplitude: Array
    pupil_diameter_m: float = eqx.field(static=True)
    ote_opd_nm: Array | None = None
    provenance: dict = eqx.field(default_factory=dict)

    def __check_init__(self):
        """Validate array shapes and the pupil diameter."""
        shape = self.pupil_amplitude.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"pupil_amplitude must be square 2D, got shape {shape}")
        if self.ote_opd_nm is not None and self.ote_opd_nm.shape != shape:
            raise ValueError(
                f"ote_opd_nm shape {self.ote_opd_nm.shape} does not match the "
                f"pupil shape {shape}"
            )
        if not self.pupil_diameter_m > 0:
            raise ValueError(
                f"pupil_diameter_m must be positive, got {self.pupil_diameter_m}"
            )

    @classmethod
    def from_bundle(cls, path):
        """Read an STPSF-exported NIRCam prescription bundle.

        Expected layout of the bundle directory::

            manifest.json          {"bundle_name": ...,
                                    "bundle_files": {
                                        "pupil_amplitude": {"file", "sha256"},
                                        "ote_opd": {"file", "sha256"}, ...}}
            pupil_amplitude.fits   amplitude transmission (primary HDU)
            ote_opd.fits           OPD in meters (primary HDU), optional

        Each array file is square, sampled on the entrance-pupil grid, and
        carries ``CONTENT``, ``BUNIT``, ``PIXSCALE`` with ``PIXUNIT = 'm/pix'``,
        ``NPIX``, ``CENTERX`` / ``CENTERY`` (0-based optical axis, which must
        be ``(NPIX - 1) / 2``) and ``PHASECNV`` (the STPSF phasor
        ``exp(+i*2*pi*OPD/lambda)``). Every file is checked against its
        manifest sha256 before use.

        STPSF propagates with POPPY's ``exp(+2j pi u x)`` forward kernel, the
        complex conjugate of this library's ``exp(-2j pi u x)``; with the same
        ``exp(+2j pi OPD / lambda)`` phasor on both sides, the two fields are
        complex conjugates of each other when the OPD enters here with its
        sign reversed. The OPD is therefore converted once, here:
        ``ote_opd_nm = -1e9 * opd_m``. Intensities are unchanged by the
        conjugation; array orientation is not touched.

        Args:
            path: The bundle directory (containing ``manifest.json``).

        Returns:
            The ``NIRCamInputs`` with provenance recording the bundle name,
            the manifest hashes and each file's header.

        Raises:
            FileNotFoundError: A manifest or listed file is missing.
            ValueError: A checksum, unit, grid or convention does not match.
        """
        fits = _fits()
        root = Path(path)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest.json in bundle directory {root}")
        manifest = json.loads(manifest_path.read_text())
        entries = manifest.get("bundle_files", {})
        if "pupil_amplitude" not in entries:
            raise ValueError(f"{manifest_path} lists no pupil_amplitude file")

        files = {}

        def read(key, *, content, bunit):
            entry = entries[key]
            file_path = root / entry["file"]
            if not file_path.exists():
                raise FileNotFoundError(
                    f"bundle file {entry['file']} listed in {manifest_path} is missing"
                )
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if digest != entry.get("sha256"):
                raise ValueError(
                    f"{entry['file']}: sha256 {digest} does not match the "
                    f"manifest value {entry.get('sha256')}"
                )
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                data = np.array(hdul[0].data, dtype=float)
            npix = data.shape[0]
            checks = {
                "CONTENT": (header.get("CONTENT"), content),
                "BUNIT": (header.get("BUNIT"), bunit),
                "PIXUNIT": (header.get("PIXUNIT"), "m/pix"),
                "PHASECNV": (header.get("PHASECNV"), STPSF_PHASOR),
                "CENTERX": (header.get("CENTERX"), (npix - 1) / 2),
                "CENTERY": (header.get("CENTERY"), (npix - 1) / 2),
            }
            for card, (found, wanted) in checks.items():
                if found != wanted:
                    raise ValueError(
                        f"{entry['file']}: header {card} = {found!r}, expected "
                        f"{wanted!r}"
                    )
            if data.ndim != 2 or data.shape != (npix, npix):
                raise ValueError(
                    f"{entry['file']}: array must be square, got {data.shape}"
                )
            files[key] = {
                "file": entry["file"],
                "sha256": digest,
                "header": {
                    k: header[k] for k in header if k not in ("COMMENT", "HISTORY", "")
                },
            }
            return data, float(header["PIXSCALE"])

        amplitude, pixscale = read(
            "pupil_amplitude", content="amplitude_transmission", bunit=""
        )
        opd_nm = None
        if "ote_opd" in entries:
            opd_m, opd_pixscale = read("ote_opd", content="opd", bunit="m")
            if opd_m.shape != amplitude.shape or opd_pixscale != pixscale:
                raise ValueError(
                    "ote_opd grid does not match the pupil grid: shape "
                    f"{opd_m.shape} vs {amplitude.shape}, PIXSCALE {opd_pixscale} "
                    f"vs {pixscale}"
                )
            opd_nm = jnp.asarray(-M_TO_NM * opd_m)

        provenance = {
            "bundle_name": manifest.get("bundle_name"),
            "bundle_path": str(root.resolve()),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "config_sha256": manifest.get("config_sha256"),
            "inputs_sha256": manifest.get("inputs_sha256"),
            "files": files,
        }
        return cls(
            pupil_amplitude=jnp.asarray(amplitude),
            pupil_diameter_m=pixscale * amplitude.shape[0],
            ote_opd_nm=opd_nm,
            provenance=provenance,
        )


def build_nircam(
    config: NIRCamConfig, inputs: NIRCamInputs
) -> tuple[OpticalPath, Field]:
    """Build the NIRCam path and its source field.

    Stages, in order: ``entrance_pupil`` (amplitude), ``ote_opd`` (only when
    ``config.ote_opd``), ``inversion`` (reflection in y) and
    ``science_focal`` (MFT to the oversampled output grid). Any of them can
    be tapped by name in ``OpticalPath.propagate``.

    Args:
        config: Wavelength, source position, stage toggles and sampling.
        inputs: The prescription arrays and pupil diameter.

    Returns:
        ``(path, field)``: the path and the entrance-plane source field to
        propagate through it. ``path.propagate(field)[0]`` is the focal field
        on ``detector_npix * oversample`` samples per side in lambda/D.

    Raises:
        ValueError: ``config.ote_opd`` is set but ``inputs`` has no OPD, or
            the output grid undersamples the MFT kernel.
    """
    wavelength_nm = config.wavelength_nm
    diameter_m = inputs.pupil_diameter_m
    amplitude = inputs.pupil_amplitude
    npup = amplitude.shape[0]
    pupil_grid = Grid.pupil(npup)
    sample_lod = arcsec_to_lambda_d(
        config.pixel_scale_arcsec / config.oversample, wavelength_nm, diameter_m
    )
    focal_grid = Grid.focal(config.detector_npix * config.oversample, sample_lod)

    stages = [
        Stage(
            "entrance_pupil",
            SampledOptic(
                transmission=amplitude, grid=pupil_grid, plane=PlaneKind.PUPIL
            ),
        )
    ]
    if config.ote_opd:
        if inputs.ote_opd_nm is None:
            raise ValueError(
                "config.ote_opd is set but the inputs carry no OTE OPD map"
            )
        basis = ModeBasis(B=inputs.ote_opd_nm[jnp.newaxis], coeffs=jnp.ones(1))
        stages.append(
            Stage(
                "ote_opd", PhaseScreen(basis, pupil_grid, wavelength_nm=wavelength_nm)
            )
        )
    stages.append(Stage("inversion", CoordinateInversion(grid=pupil_grid, axis="y")))
    stages.append(
        Stage(
            "science_focal",
            Fraunhofer(pupil_grid, focal_grid, on_undersampled="raise"),
        )
    )
    path = OpticalPath(stages=tuple(stages))

    energy = jnp.sum(amplitude**2) * pupil_grid.weights
    flat = Field(
        data=jnp.full((npup, npup), 1.0 / jnp.sqrt(energy), dtype=complex),
        grid=pupil_grid,
        plane=PlaneKind.PUPIL,
    )
    sx, sy = config.source_position_arcsec
    # The tilt is applied upstream of the y inversion, so its y component is
    # reflected here to land at +sy in the detector frame.
    position_lod = (
        float(arcsec_to_lambda_d(sx, wavelength_nm, diameter_m)),
        -float(arcsec_to_lambda_d(sy, wavelength_nm, diameter_m)),
    )
    field = point_source(flat, position_lod=position_lod)
    return path, field
