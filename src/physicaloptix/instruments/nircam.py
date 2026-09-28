"""JWST NIRCam imaging and round-mask coronagraph path from an exported prescription.

The builder reproduces the plane sequence STPSF resolves for a NIRCam
calculation: the entrance pupil (amplitude and, optionally, the fixed OTE
OPD), the coordinate inversion in y that STPSF places after the telescope,
optionally a round band-limited focal-plane mask and the Lyot stop, optionally
the instrument (SI) wavefront error in the exit pupil after the Lyot stop, and
a matrix Fourier transform to the science focal plane sampled at the detector
pixel scale times ``oversample``. The output grid matches STPSF's
``OVERSAMP`` product: ``detector_npix * oversample`` samples per side with the
optical axis at index ``(n - 1) / 2``.

Plane order and placement: the OTE OPD acts in the entrance pupil, BEFORE the
focal mask, and the SI WFE acts in the exit pupil AFTER the Lyot stop. The two
are separate optics; folding them into one entrance screen changes the
coronagraphic image, because only light that reaches the mask plane with the
OTE error is redistributed by the occulter.

Focal-plane mask propagation (``FocalPlaneMask``): the pupil field ``E`` is
carried to a mask-plane window by the continuous-FT MFT, and the field
returned to the Lyot plane is ``E + B[(M - 1) F E]`` (a Babinet form), with
``F`` / ``B`` the MFT pair and ``M`` the amplitude transmission sampled on the
window. The unperturbed field ``E`` is carried through exactly on any window,
and only ``(M - 1)`` outside the window is dropped: a window that contains a
compactly supported perturbation loses nothing. The NIRCam round masks are NOT
compactly supported: besides the occulting core, the holder edge (opaque for
y > 10 arcsec) and the glass edge (a strip at -13 < y < -11.5 arcsec, all x)
run to the edge of any band, so every window narrower than the full band drops
part of the mask. The default window is therefore the pupil raster's full
Nyquist band (``npup`` lambda/D wide, ``npup * mask_oversample`` samples): it
is the raster's information limit (a sampled pupil's far field is periodic
beyond it) and the band a padded-FFT propagation of the same raster covers. On
it ``B F`` is the identity to roundoff and the Babinet form equals the direct
propagation ``B[M F E]``: every part of the mask the sampled pupil can resolve
is applied. Narrower windows (``mask_extent_lod``) are for convergence
studies, not for masks with extended regions.

Frames and signs:

- The pupil arrays are in the entrance-pupil frame (+x = +V2, +y = +V3,
  rows are y). The inversion reflects them in y, so the exit pupil (Lyot
  stop, SI WFE) and the mask plane are in the detector frame: the entrance
  frame with y reversed. Lyot and SI arrays are therefore given in the exit
  frame, as STPSF applies them.
- Source positions are given in the detector-oriented output frame (x right,
  y up), as STPSF's ``source_offset_x`` / ``source_offset_y``. The tilt is
  applied at the entrance pupil, reflected through the inversion, so the
  image lands at ``+position``; the mask is centered at its own
  ``center_arcsec`` in the same frame, independent of the source.
- OPDs held by ``NIRCamInputs`` are in this library's convention: nanometers
  of accumulated optical path (a positive OPD is a delayed wavefront), phasor
  ``exp(+2j pi opd / lambda)``, forward kernel ``exp(-2j pi u x)``, an implied
  ``exp(-i omega t)`` time dependence. STPSF (POPPY) instead writes OPDs in the
  Wyant-Creath sense, a positive OPD being an ADVANCED wavefront, and
  represents fields with an ``exp(+i omega t)`` time dependence (forward
  kernel ``exp(+2j pi u x)``, the same ``exp(+2j pi OPD / lambda)`` phasor).
  The same physical surface error therefore carries opposite OPD signs in the
  two codes, and the same physical field is represented by complex-conjugate
  amplitudes; intensities agree sample by sample. ``NIRCamInputs.from_bundle``
  converts each STPSF OPD once, at the import boundary: ``opd_nm = -1e9 *
  opd_m``. Real amplitude transmissions (pupil, mask, Lyot, SI support) are
  conjugation-invariant and are used as given.

Normalization: the source field is scaled so the field just after the
entrance pupil carries unit energy (``Field.energy() == 1``), the
entrance-pupil normalization STPSF applies with ``normalize='first'``. No
later renormalization happens, so ``out.intensity() * out.grid.weights`` is
the fraction of entrance-pupil energy per output sample and finite-field,
mask and Lyot losses are kept.

Band and pixels: ``nircam_band_image`` propagates each wavelength node of a
``NIRCamBand`` independently (its own lambda/D scale, OPD phase, source tilt,
mask sampling and SI OPD) onto the one angular output grid set by the pixel
scale, and sums the photon-weighted energy images. ``pixel_integrate`` then
integrates detector pixels once, by summing the ``oversample x oversample``
samples inside each pixel (the composite midpoint rule, STPSF's ``DET_SAMP``
rule); no further pixel kernel may be applied afterwards.

Taps: ``OpticalPath.propagate(field, taps=...)`` records any stage output by
name (``entrance_pupil``, ``ote_opd``, ``inversion``, ``focal_mask`` = the
field arriving at the Lyot plane, ``lyot_stop`` = just after the Lyot stop,
``si_wfe``). The mask-plane field (after the mask) is not a stage output of
the pupil-to-pupil mask stage; ``mask_plane_field(path, taps)`` computes it
from the ``inversion`` tap.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from hwoutils.conversions import arcsec_to_lambda_d, lambda_d_to_arcsec
from hwoutils.transforms import downsample_psf
from jaxtyping import Array

from physicaloptix.core import Field, Grid, PlaneKind, validate_field
from physicaloptix.elements import ModeBasis, PhaseScreen, SampledOptic
from physicaloptix.elements.base import Element
from physicaloptix.path import OpticalPath, Stage
from physicaloptix.sources import point_source
from physicaloptix.transforms import Fraunhofer

M_TO_NM = 1e9
STPSF_PHASOR = "exp(+i*2*pi*OPD/lambda)"
# Second positive zero of the Bessel function J1: the round band-limited
# profile is truncated there, after its first sidelobe, to match the hardware.
J1_SECOND_ZERO = 7.015586669815619
# Largest |analytic - sampled| accepted when a bundle's sampled focal mask is
# checked against its recorded analytic parameters (float64 evaluation noise).
MASK_CHECK_ATOL = 1e-12


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


class BandLimitedRoundMask(eqx.Module):
    """A round band-limited occulter (NIRCam MASK210R / MASK335R / MASK430R).

    Amplitude (electric-field) transmission, real and non-negative, after
    Krist et al.'s NIRCam design with STPSF's truncation::

        t(r) = 1 - (2 J1(s) / s)**2,  s = clip(sigma * r, tiny, clip),  t(0) = 0

    (Krist et al. give the intensity transmission; the amplitude lacks the
    outer square.) ``regions`` are then applied in order, each an open box
    ``(x_lo, x_hi, y_lo, y_hi, amplitude)`` in arcsec relative to the mask
    center, ``None`` meaning unbounded on that side: the ND squares, the
    opaque holder edge and the glass edge. Later regions override earlier
    ones. Coordinates are in the mask-plane frame, which has the orientation
    of the detector-oriented output frame (x right, y up).

    Attributes:
        sigma_per_arcsec: Band-limit parameter sigma in 1/arcsec.
        clip: Value of ``sigma * r`` at which the profile is truncated
            (default: the second zero of J1).
        regions: Rectangular override regions, in application order.
        center_arcsec: ``(x, y)`` mask center in arcsec (STPSF's
            ``shift_x`` / ``shift_y``).
    """

    sigma_per_arcsec: float = eqx.field(static=True)
    clip: float = eqx.field(static=True, default=J1_SECOND_ZERO)
    regions: tuple = eqx.field(static=True, default=(), converter=tuple)
    center_arcsec: tuple[float, float] = eqx.field(
        static=True, default=(0.0, 0.0), converter=tuple
    )

    def __check_init__(self):
        """Validate the profile parameters and region layout."""
        if not self.sigma_per_arcsec > 0 or not self.clip > 0:
            raise ValueError(
                "sigma_per_arcsec and clip must be positive, got "
                f"{self.sigma_per_arcsec} and {self.clip}"
            )
        if len(self.center_arcsec) != 2:
            raise ValueError(f"center_arcsec must be (x, y), got {self.center_arcsec}")
        for region in self.regions:
            if len(region) != 5:
                raise ValueError(
                    "each region must be (x_lo, x_hi, y_lo, y_hi, amplitude), "
                    f"got {region}"
                )

    def transmission(self, x_arcsec, y_arcsec):
        """Amplitude transmission at mask-plane positions (host-side NumPy).

        Args:
            x_arcsec: x positions in arcsec (broadcast against ``y_arcsec``).
            y_arcsec: y positions in arcsec.

        Returns:
            The real amplitude transmission, float64, broadcast shape.
        """
        from scipy.special import j1

        xr = np.asarray(x_arcsec, dtype=float) - self.center_arcsec[0]
        yr = np.asarray(y_arcsec, dtype=float) - self.center_arcsec[1]
        xr, yr = np.broadcast_arrays(xr, yr)
        r = np.sqrt(xr**2 + yr**2)
        s = np.clip(self.sigma_per_arcsec * r, np.finfo(float).tiny, self.clip)
        t = 1.0 - (2.0 * j1(s) / s) ** 2
        t[r == 0] = 0.0
        for x_lo, x_hi, y_lo, y_hi, amplitude in self.regions:
            inside = np.ones(t.shape, dtype=bool)
            for value, lo, hi in ((xr, x_lo, x_hi), (yr, y_lo, y_hi)):
                if lo is not None:
                    inside &= value > lo
                if hi is not None:
                    inside &= value < hi
            t[inside] = amplitude
        return t


class FocalPlaneMask(eqx.Module):
    """Pupil -> focal-plane mask -> Lyot pupil, in the Babinet form.

    ``__call__`` returns ``E + B[(M - 1) F E]``: ``F`` / ``B`` are the
    continuous-FT MFT pair between the pupil grid and the mask-plane window
    (``transform``), ``M`` the amplitude transmission sampled on the window.
    The unperturbed field passes through exactly on any window, and only
    ``(M - 1)`` outside the window is dropped, so a narrower window is exact
    only for a compactly supported perturbation (not for masks with extended
    regions such as a holder half-plane). On the full Nyquist band of the
    pupil raster (window of ``npup`` lambda/D) ``B F`` is the identity and the
    result equals ``B[M F E]``. A composite pupil-to-pupil
    stage, so it carries ``plane_in`` / ``plane_out`` like a propagator.

    Attributes:
        transform: The ``Fraunhofer`` pair, pupil grid -> mask-plane window.
        transmission: Mask amplitude transmission on the window, ``(n, n)``,
            rows y; sampled at one wavelength, so the stage is monochromatic.
    """

    transform: Fraunhofer
    transmission: Array
    plane_in: PlaneKind = eqx.field(static=True, default=PlaneKind.PUPIL)
    plane_out: PlaneKind = eqx.field(static=True, default=PlaneKind.PUPIL)

    def __check_init__(self):
        """Validate that the transmission matches the window grid."""
        npix = self.transform.grid_out.npix
        if self.transmission.shape != (npix, npix):
            raise ValueError(
                f"transmission shape {self.transmission.shape} does not match "
                f"the mask-plane window ({npix}, {npix})"
            )

    def _check_mono(self, field):
        if field.spectrum is not None:
            raise ValueError(
                "FocalPlaneMask is sampled at one wavelength; propagate each "
                "wavelength through its own path"
            )

    def mask_plane(self, field):
        """The field just after the mask, on the mask-plane window (focal)."""
        self._check_mono(field)
        focal = self.transform.forward(field)
        return Field(
            data=focal.data * self.transmission,
            grid=focal.grid,
            plane=focal.plane,
            spectrum=focal.spectrum,
        )

    def __call__(self, field):
        """Return the field arriving at the Lyot plane (same pupil grid)."""
        self._check_mono(field)
        focal = self.transform.forward(field)
        perturbation = Field(
            data=focal.data * (self.transmission - 1.0),
            grid=focal.grid,
            plane=focal.plane,
            spectrum=focal.spectrum,
        )
        returned = self.transform.backward(perturbation)
        return Field(
            data=field.data + returned.data,
            grid=field.grid,
            plane=self.plane_out,
            spectrum=field.spectrum,
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
        ote_opd: Apply the fixed OTE OPD in the entrance pupil (pre-mask).
        focal_mask: Apply the focal-plane mask (``inputs.focal_mask``).
        lyot_stop: Apply the Lyot stop (``inputs.lyot_amplitude``).
        si_wfe: Apply the SI WFE and its support in the exit pupil, after the
            Lyot stop (post-mask).
        mask_oversample: Mask-plane samples per lambda/D (D = the pupil array
            side), i.e. a pitch of ``1 / mask_oversample`` lambda/D.
        mask_extent_lod: Full width of the mask-plane window in lambda/D;
            ``None`` is the pupil raster's full Nyquist band (``npup``), the
            only exact choice for masks whose ``M - 1`` is not compact.
    """

    wavelength_nm: float = eqx.field(static=True)
    pixel_scale_arcsec: float = eqx.field(static=True)
    detector_npix: int = eqx.field(static=True, default=81)
    oversample: int = eqx.field(static=True, default=4)
    source_position_arcsec: tuple[float, float] = eqx.field(
        static=True, default=(0.0, 0.0), converter=tuple
    )
    ote_opd: bool = eqx.field(static=True, default=False)
    focal_mask: bool = eqx.field(static=True, default=False)
    lyot_stop: bool = eqx.field(static=True, default=False)
    si_wfe: bool = eqx.field(static=True, default=False)
    mask_oversample: int = eqx.field(static=True, default=4)
    mask_extent_lod: float | None = eqx.field(static=True, default=None)

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
        if self.mask_oversample < 1:
            raise ValueError(
                f"mask_oversample must be at least 1, got {self.mask_oversample}"
            )
        if self.mask_extent_lod is not None and not self.mask_extent_lod > 0:
            raise ValueError(
                f"mask_extent_lod must be positive, got {self.mask_extent_lod}"
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
        focal_mask: Optional focal-plane mask: any module with a
            ``transmission(x_arcsec, y_arcsec)`` method returning the
            amplitude transmission in the mask-plane frame (for example
            ``BandLimitedRoundMask``).
        lyot_amplitude: Optional Lyot-stop amplitude transmission on the pupil
            grid, exit-pupil frame.
        si_opd_nm: Optional SI OPD on the pupil grid, exit-pupil frame, in this
            library's sign convention, valid at ``si_wavelength_nm``.
        si_support: Optional SI amplitude support (defaults to ones).
        si_wavelength_nm: Wavelength the SI OPD applies at (it is
            wavelength-dependent); required with ``si_opd_nm``.
        provenance: Where the arrays came from (file names, checksums).
    """

    pupil_amplitude: Array
    pupil_diameter_m: float = eqx.field(static=True)
    ote_opd_nm: Array | None = None
    focal_mask: eqx.Module | None = None
    lyot_amplitude: Array | None = None
    si_opd_nm: Array | None = None
    si_support: Array | None = None
    si_wavelength_nm: float | None = eqx.field(static=True, default=None)
    provenance: dict = eqx.field(default_factory=dict)

    def __check_init__(self):
        """Validate array shapes and the pupil diameter."""
        shape = self.pupil_amplitude.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"pupil_amplitude must be square 2D, got shape {shape}")
        for name in ("ote_opd_nm", "lyot_amplitude", "si_opd_nm", "si_support"):
            array = getattr(self, name)
            if array is not None and array.shape != shape:
                raise ValueError(
                    f"{name} shape {array.shape} does not match the pupil shape {shape}"
                )
        if self.si_opd_nm is not None and not (
            self.si_wavelength_nm is not None and self.si_wavelength_nm > 0
        ):
            raise ValueError(
                "si_opd_nm needs the positive wavelength it applies at "
                f"(si_wavelength_nm), got {self.si_wavelength_nm}"
            )
        if not self.pupil_diameter_m > 0:
            raise ValueError(
                f"pupil_diameter_m must be positive, got {self.pupil_diameter_m}"
            )

    @classmethod
    def from_bundle(cls, path):
        """Read an STPSF-exported NIRCam prescription bundle.

        Expected layout of the bundle directory (only the pupil is required;
        every other entry is read when the manifest lists it)::

            manifest.json          {"bundle_name": ...,
                                    "bundle_files": {
                                        "pupil_amplitude": {"file", "sha256"},
                                        "ote_opd": {...}, "lyot_stop": {...},
                                        "si_wfe_opd": {...},
                                        "focal_mask": {"file", "sha256",
                                                       "analytic": {...}}}}
            pupil_amplitude.fits   amplitude transmission (primary HDU)
            ote_opd.fits           OPD in meters, entrance frame, pre-mask
            lyot_stop.fits         amplitude transmission, exit frame
            si_wfe_opd.fits        OPD in meters, exit frame, post-mask, at
                                   the wavelength in header WAVELEN [m];
                                   extension SUPPORT = amplitude support
            focal_mask.fits        sampled mask amplitude, image plane

        Pupil-grid files are square, sampled on the entrance-pupil grid, and
        carry ``CONTENT``, ``BUNIT``, ``PIXSCALE`` with ``PIXUNIT = 'm/pix'``,
        ``NPIX``, ``CENTERX`` / ``CENTERY`` (0-based optical axis, which must
        be ``(NPIX - 1) / 2``) and ``PHASECNV`` (the STPSF phasor
        ``exp(+i*2*pi*OPD/lambda)``). Every file is checked against its
        manifest sha256 before use.

        The focal mask is taken from its ``analytic`` manifest record (kind
        ``nircamcircular``: ``sigma_per_arcsec``, ``j1_zero2``,
        ``regions_in_application_order``, ``shift_x`` / ``shift_y``), so it
        can be evaluated at any wavelength. The sampled file (``PIXUNIT =
        'arcsec/pix'``, zero angle at ``CENTERX = CENTERY = NPIX / 2``, header
        ``SIGMA`` and ``J1ZERO2``) is the mask as STPSF applied it on its own
        grid; the analytic mask is re-evaluated on that grid and must
        reproduce it to ``MASK_CHECK_ATOL``.

        OPDs are converted once, here, from STPSF's convention to this
        library's (see the module docstring): ``opd_nm = -1e9 * opd_m``.
        Amplitude arrays and orientations are used as given.

        Args:
            path: The bundle directory (containing ``manifest.json``).

        Returns:
            The ``NIRCamInputs`` with provenance recording the bundle name,
            the manifest hashes, each file's header and the focal-mask check.

        Raises:
            FileNotFoundError: A manifest or listed file is missing.
            ValueError: A checksum, unit, grid, convention or the analytic
                mask check does not match.
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

        def read(key, *, content, bunit, pupil=True, extensions=()):
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
                extra = {}
                for name in extensions:
                    if name not in hdul:
                        raise ValueError(f"{entry['file']}: no {name} extension")
                    extra[name] = (
                        np.array(hdul[name].data, dtype=float),
                        hdul[name].header,
                    )
            npix = data.shape[0]
            center = (npix - 1) / 2 if pupil else npix / 2
            checks = {
                "CONTENT": (header.get("CONTENT"), content),
                "BUNIT": (header.get("BUNIT"), bunit),
                "PIXUNIT": (header.get("PIXUNIT"), "m/pix" if pupil else "arcsec/pix"),
                "CENTERX": (header.get("CENTERX"), center),
                "CENTERY": (header.get("CENTERY"), center),
            }
            if pupil:
                checks["PHASECNV"] = (header.get("PHASECNV"), STPSF_PHASOR)
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
            return data, header, extra

        amplitude, header, _ = read(
            "pupil_amplitude", content="amplitude_transmission", bunit=""
        )
        pixscale = float(header["PIXSCALE"])

        def read_pupil_grid(key, **kwargs):
            data, head, extra = read(key, **kwargs)
            if data.shape != amplitude.shape or float(head["PIXSCALE"]) != pixscale:
                raise ValueError(
                    f"{key} grid does not match the pupil grid: shape "
                    f"{data.shape} vs {amplitude.shape}, PIXSCALE "
                    f"{head['PIXSCALE']} vs {pixscale}"
                )
            return data, head, extra

        opd_nm = None
        if "ote_opd" in entries:
            opd_m, _, _ = read_pupil_grid("ote_opd", content="opd", bunit="m")
            opd_nm = jnp.asarray(-M_TO_NM * opd_m)

        lyot = None
        if "lyot_stop" in entries:
            lyot_data, _, _ = read_pupil_grid(
                "lyot_stop", content="amplitude_transmission", bunit=""
            )
            lyot = jnp.asarray(lyot_data)

        si_opd_nm = si_support = si_wavelength_nm = None
        if "si_wfe_opd" in entries:
            si_m, si_header, extra = read_pupil_grid(
                "si_wfe_opd", content="opd", bunit="m", extensions=("SUPPORT",)
            )
            support, support_header = extra["SUPPORT"]
            if support_header.get("CONTENT") != "amplitude_transmission":
                raise ValueError(
                    "si_wfe_opd SUPPORT extension: CONTENT = "
                    f"{support_header.get('CONTENT')!r}, expected "
                    "'amplitude_transmission'"
                )
            if support.shape != si_m.shape:
                raise ValueError(
                    f"si_wfe_opd SUPPORT shape {support.shape} does not match "
                    f"the OPD shape {si_m.shape}"
                )
            if "WAVELEN" not in si_header:
                raise ValueError("si_wfe_opd: header WAVELEN [m] is missing")
            si_opd_nm = jnp.asarray(-M_TO_NM * si_m)
            si_support = jnp.asarray(support)
            si_wavelength_nm = float(si_header["WAVELEN"]) * M_TO_NM

        focal_mask = None
        mask_check = None
        if "focal_mask" in entries:
            focal_mask, mask_check = _read_focal_mask(read, entries["focal_mask"])

        provenance = {
            "bundle_name": manifest.get("bundle_name"),
            "bundle_path": str(root.resolve()),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "config_sha256": manifest.get("config_sha256"),
            "inputs_sha256": manifest.get("inputs_sha256"),
            "files": files,
        }
        if mask_check is not None:
            provenance["focal_mask_check"] = mask_check
        return cls(
            pupil_amplitude=jnp.asarray(amplitude),
            pupil_diameter_m=pixscale * amplitude.shape[0],
            ote_opd_nm=opd_nm,
            focal_mask=focal_mask,
            lyot_amplitude=lyot,
            si_opd_nm=si_opd_nm,
            si_support=si_support,
            si_wavelength_nm=si_wavelength_nm,
            provenance=provenance,
        )


def _read_focal_mask(read, entry):
    """Build the analytic round mask from a bundle entry and verify it.

    The analytic mask is evaluated on the sampled file's own grid (x along
    axis 1, y along axis 0, zero angle at ``CENTERX`` / ``CENTERY``) and must
    reproduce the sampled amplitude to ``MASK_CHECK_ATOL``.
    """
    analytic = entry.get("analytic")
    if not analytic:
        raise ValueError("focal_mask entry has no analytic record")
    if analytic.get("kind") != "nircamcircular":
        raise ValueError(
            f"focal_mask kind {analytic.get('kind')!r} is not supported "
            "(only 'nircamcircular')"
        )
    if analytic.get("rotation") not in (None, 0, 0.0):
        raise ValueError(
            f"focal_mask rotation {analytic.get('rotation')} is not supported"
        )
    regions = []
    for region in analytic.get("regions_in_application_order", []):
        x_range = region.get("x_range") or (None, None)
        y_range = region.get("y_range") or (None, None)
        regions.append(
            (x_range[0], x_range[1], y_range[0], y_range[1], float(region["amplitude"]))
        )
    mask = BandLimitedRoundMask(
        sigma_per_arcsec=float(analytic["sigma_per_arcsec"]),
        clip=float(analytic["j1_zero2"]),
        regions=tuple(regions),
        center_arcsec=(
            float(analytic.get("shift_x") or 0.0),
            float(analytic.get("shift_y") or 0.0),
        ),
    )
    sampled, header, _ = read(
        "focal_mask", content="amplitude_transmission", bunit="", pupil=False
    )
    for card, value in (("SIGMA", mask.sigma_per_arcsec), ("J1ZERO2", mask.clip)):
        if card in header and float(header[card]) != value:
            raise ValueError(
                f"focal_mask header {card} = {header[card]} does not match the "
                f"analytic record {value}"
            )
    step = float(header["PIXSCALE"])
    npix = sampled.shape[0]
    x = (np.arange(npix) - float(header["CENTERX"])) * step
    y = (np.arange(npix) - float(header["CENTERY"])) * step
    diff = float(
        np.abs(mask.transmission(x[np.newaxis, :], y[:, np.newaxis]) - sampled).max()
    )
    if not diff <= MASK_CHECK_ATOL:
        raise ValueError(
            f"the analytic focal mask does not reproduce the sampled mask: "
            f"max |analytic - sampled| = {diff:.3e} > {MASK_CHECK_ATOL:.0e}"
        )
    check = {
        "max_abs_diff_analytic_vs_sampled": diff,
        "sampled_npix": npix,
        "sampled_pixscale_arcsec": step,
        "sampled_wavelength_m": header.get("WAVELEN"),
        "atol": MASK_CHECK_ATOL,
    }
    return mask, check


def _focal_mask_stage(config, inputs, pupil_grid):
    """The mask-plane window, its sampled transmission and the MFT pair."""
    npup = pupil_grid.npix
    extent = npup if config.mask_extent_lod is None else config.mask_extent_lod
    n_mask = round(extent * config.mask_oversample)
    if n_mask < 1:
        raise ValueError(
            f"mask-plane window of {extent} lambda/D at {config.mask_oversample} "
            "samples per lambda/D holds no sample"
        )
    window = Grid.focal(n_mask, 1.0 / config.mask_oversample)
    transform = Fraunhofer(pupil_grid, window, on_undersampled="raise")
    theta = np.asarray(
        lambda_d_to_arcsec(
            window.coords, config.wavelength_nm, inputs.pupil_diameter_m
        ),
        dtype=float,
    )
    transmission = inputs.focal_mask.transmission(
        theta[np.newaxis, :], theta[:, np.newaxis]
    )
    return FocalPlaneMask(transform=transform, transmission=jnp.asarray(transmission))


def build_nircam(
    config: NIRCamConfig, inputs: NIRCamInputs
) -> tuple[OpticalPath, Field]:
    """Build the NIRCam path and its source field.

    Stages, in order (optional ones only when their toggle is set):
    ``entrance_pupil`` (amplitude), ``ote_opd`` (pre-mask OPD),
    ``inversion`` (reflection in y), ``focal_mask`` (pupil -> mask ->
    Lyot-plane pupil, see ``FocalPlaneMask``), ``lyot_stop``, ``si_wfe``
    (SI support times the post-mask OPD phasor) and ``science_focal`` (MFT to
    the oversampled output grid). Any of them can be tapped by name in
    ``OpticalPath.propagate``; ``mask_plane_field`` gives the mask-plane
    field. With every optional stage off, the path is the no-mask path.

    Args:
        config: Wavelength, source position, stage toggles and sampling.
        inputs: The prescription arrays and pupil diameter.

    Returns:
        ``(path, field)``: the path and the entrance-plane source field to
        propagate through it. ``path.propagate(field)[0]`` is the focal field
        on ``detector_npix * oversample`` samples per side in lambda/D.

    Raises:
        ValueError: A toggle is set but ``inputs`` lacks its array, the SI
            OPD is for another wavelength, or a grid undersamples its MFT
            kernel (including a mask window wider than the full band).
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
    if config.focal_mask:
        if inputs.focal_mask is None:
            raise ValueError(
                "config.focal_mask is set but the inputs carry no focal mask"
            )
        stages.append(
            Stage("focal_mask", _focal_mask_stage(config, inputs, pupil_grid))
        )
    if config.lyot_stop:
        if inputs.lyot_amplitude is None:
            raise ValueError(
                "config.lyot_stop is set but the inputs carry no Lyot stop"
            )
        stages.append(
            Stage(
                "lyot_stop",
                SampledOptic(
                    transmission=inputs.lyot_amplitude,
                    grid=pupil_grid,
                    plane=PlaneKind.PUPIL,
                ),
            )
        )
    if config.si_wfe:
        if inputs.si_opd_nm is None:
            raise ValueError("config.si_wfe is set but the inputs carry no SI OPD")
        if not np.isclose(inputs.si_wavelength_nm, wavelength_nm, rtol=1e-12, atol=0):
            raise ValueError(
                f"the SI OPD applies at {inputs.si_wavelength_nm} nm, not at the "
                f"configured wavelength {wavelength_nm} nm"
            )
        support = (
            jnp.ones_like(inputs.si_opd_nm)
            if inputs.si_support is None
            else inputs.si_support
        )
        phasor = support * jnp.exp(2j * jnp.pi * inputs.si_opd_nm / wavelength_nm)
        stages.append(
            Stage(
                "si_wfe",
                SampledOptic(
                    transmission=phasor, grid=pupil_grid, plane=PlaneKind.PUPIL
                ),
            )
        )
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


def mask_plane_field(path, taps):
    """The field just after the focal mask, on the mask-plane window.

    Args:
        path: A path from ``build_nircam`` with the ``focal_mask`` stage.
        taps: The tapped fields of ``path.propagate``; must include
            ``"inversion"``, the field entering the focal mask.

    Returns:
        A focal-plane ``Field`` on the mask-plane window (lambda/D, rows y);
        ``intensity() * grid.weights`` is the fraction of entrance-pupil
        energy per window sample.

    Raises:
        ValueError: The path has no focal mask or the tap is missing.
    """
    ops = {stage.name: stage.op for stage in path.stages}
    if "focal_mask" not in ops:
        raise ValueError("the path has no focal_mask stage")
    if "inversion" not in taps:
        raise ValueError(
            "tap 'inversion' (the field entering the focal mask) is needed to "
            "form the mask-plane field"
        )
    return ops["focal_mask"].mask_plane(taps["inversion"])


def _node_tuple(values):
    return tuple(float(v) for v in values)


def _optional_array(value):
    return None if value is None else jnp.asarray(value)


class NIRCamBand(eqx.Module):
    """Wavelength nodes and photon weights of a band, with per-node SI OPDs.

    The weights are relative photon weights of a stated source spectrum
    through the filter; ``nircam_band_image`` normalizes them to unit sum, so
    the band image stays a fraction of entrance-pupil energy. They do not
    establish absolute throughput.

    Attributes:
        wavelengths_nm: Node wavelengths in nanometers.
        weights: Non-negative relative weight of each node.
        si_opd_nm: Optional SI OPD at each node, ``(n_nodes, n, n)``, exit
            frame, this library's sign convention (the SI WFE is
            wavelength-dependent, so every node carries its own map).
        provenance: Where the nodes came from (source spectrum, rule, hashes).
    """

    wavelengths_nm: tuple[float, ...] = eqx.field(static=True, converter=_node_tuple)
    weights: tuple[float, ...] = eqx.field(static=True, converter=_node_tuple)
    si_opd_nm: Array | None = eqx.field(default=None, converter=_optional_array)
    provenance: dict = eqx.field(default_factory=dict)

    def __check_init__(self):
        """Validate node counts, wavelengths, weights and the SI cube."""
        n_nodes = len(self.wavelengths_nm)
        if n_nodes < 1:
            raise ValueError("a band needs at least one wavelength node")
        if len(self.weights) != n_nodes:
            raise ValueError(
                f"need one weight per node: {n_nodes} nodes, "
                f"{len(self.weights)} weights"
            )
        if not all(w > 0 for w in self.wavelengths_nm):
            raise ValueError(
                f"node wavelengths must be positive, got {self.wavelengths_nm}"
            )
        if any(w < 0 for w in self.weights):
            raise ValueError(f"node weights must not be negative, got {self.weights}")
        if not sum(self.weights) > 0:
            raise ValueError("node weights must have a positive sum")
        if self.si_opd_nm is not None and (
            self.si_opd_nm.ndim != 3 or self.si_opd_nm.shape[0] != n_nodes
        ):
            raise ValueError(
                f"si_opd_nm must hold one (n, n) slice per node ({n_nodes}), "
                f"got shape {self.si_opd_nm.shape}"
            )

    @classmethod
    def from_bundle(cls, path, nlambda):
        """Read one node set (and its SI OPD cube) from a prescription bundle.

        Bundle layout (the manifest of ``NIRCamInputs.from_bundle`` plus)::

            manifest.json  {"configuration": {
                               "filter": "<FILTER>",
                               "<filter>_nodes": {
                                   "source_spectrum": ..., "rule": ...,
                                   "sets": {"<nlambda>": {
                                       "wavelengths_m": [...],
                                       "weights": [...]}}}}}
            si_wfe_opd.fits  extension ``OPD_<FILTER>_N<nlambda>``: the SI
                             OPD cube in meters (STPSF convention), one slice
                             per node, header ``WAVE<k>`` [m] per slice

        The cube is optional (no ``si_wfe_opd`` entry gives ``si_opd_nm =
        None``); when present, its file must match the manifest sha256 and
        every ``WAVE<k>`` must equal the node wavelength. OPDs are converted
        once, as in ``NIRCamInputs.from_bundle``: ``opd_nm = -1e9 * opd_m``.

        Args:
            path: The bundle directory (containing ``manifest.json``).
            nlambda: Number of nodes of the set to read.

        Returns:
            The ``NIRCamBand`` with provenance (source spectrum, node rule,
            cube extension and file hash).

        Raises:
            FileNotFoundError: The manifest or the SI file is missing.
            ValueError: The node set is absent, or the cube does not match
                its checksum, unit, shape or node wavelengths.
        """
        root = Path(path)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest.json in bundle directory {root}")
        manifest = json.loads(manifest_path.read_text())
        configuration = manifest.get("configuration", {})
        filter_name = str(configuration.get("filter", ""))
        record = configuration.get(f"{filter_name.lower()}_nodes", {})
        sets = record.get("sets", {})
        if str(nlambda) not in sets:
            raise ValueError(
                f"{manifest_path} has no {filter_name} node set for nlambda = "
                f"{nlambda} (available: {sorted(sets)})"
            )
        node_set = sets[str(nlambda)]
        wavelengths_m = [float(w) for w in node_set["wavelengths_m"]]
        provenance = {
            "bundle_name": manifest.get("bundle_name"),
            "filter": filter_name,
            "nlambda": int(nlambda),
            "source_spectrum": record.get("source_spectrum"),
            "rule": record.get("rule"),
            "note": record.get("note"),
        }
        si_opd_nm = None
        entry = manifest.get("bundle_files", {}).get("si_wfe_opd")
        if entry is not None:
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
            extname = f"OPD_{filter_name.upper()}_N{nlambda}"
            fits = _fits()
            with fits.open(file_path) as hdul:
                if extname not in hdul:
                    raise ValueError(f"{entry['file']}: no {extname} extension")
                header = hdul[extname].header
                cube_m = np.array(hdul[extname].data, dtype=float)
            if header.get("BUNIT") != "m":
                raise ValueError(
                    f"{entry['file']} {extname}: header BUNIT = "
                    f"{header.get('BUNIT')!r}, expected 'm'"
                )
            if cube_m.ndim != 3 or cube_m.shape[0] != len(wavelengths_m):
                raise ValueError(
                    f"{entry['file']} {extname}: shape {cube_m.shape} does not "
                    f"hold {len(wavelengths_m)} node slices"
                )
            for k, wavelength_m in enumerate(wavelengths_m):
                found = header.get(f"WAVE{k}")
                if found is None or not np.isclose(
                    float(found), wavelength_m, rtol=1e-12, atol=0
                ):
                    raise ValueError(
                        f"{entry['file']} {extname}: header WAVE{k} = {found} "
                        f"does not match node {k} at {wavelength_m} m"
                    )
            si_opd_nm = jnp.asarray(-M_TO_NM * cube_m)
            provenance["si_wfe_opd_extension"] = extname
            provenance["si_wfe_opd_sha256"] = digest
        return cls(
            wavelengths_nm=[w * M_TO_NM for w in wavelengths_m],
            weights=node_set["weights"],
            si_opd_nm=si_opd_nm,
            provenance=provenance,
        )


def nircam_band_image(config, inputs, band):
    """Photon-weighted band image, each node propagated independently.

    Every node gets its own path from ``build_nircam``: ``config`` with the
    node wavelength (so the lambda/D scale, the OPD phase, the source tilt
    and the mask sampling all follow the node) and, when the band carries
    SI OPDs, ``inputs`` with the node's SI OPD. All nodes land on the same
    angular output grid (``detector_npix * oversample`` samples at
    ``pixel_scale_arcsec / oversample``), and their energy images are summed
    with the weights normalized to unit sum.

    Args:
        config: Everything but the wavelength, which each node replaces.
        inputs: The prescription; its SI OPD is replaced per node when
            ``band.si_opd_nm`` is given (otherwise an SI stage at a node
            other than ``inputs.si_wavelength_nm`` fails at build).
        band: The nodes and weights.

    Returns:
        The band image as a fraction of entrance-pupil energy per output
        sample (unit-sum weights), ``(n, n)`` with ``n = detector_npix *
        oversample``, rows y.
    """
    weights = np.asarray(band.weights) / np.sum(band.weights)
    total = None
    for k, (wavelength_nm, weight) in enumerate(
        zip(band.wavelengths_nm, weights, strict=True)
    ):
        node_config = dataclasses.replace(config, wavelength_nm=wavelength_nm)
        node_inputs = inputs
        if band.si_opd_nm is not None:
            node_inputs = dataclasses.replace(
                inputs, si_opd_nm=band.si_opd_nm[k], si_wavelength_nm=wavelength_nm
            )
        path, field = build_nircam(node_config, node_inputs)
        out, _ = path.propagate(field)
        image = weight * out.intensity() * out.grid.weights
        total = image if total is None else total + image
    return total


def pixel_integrate(image, oversample):
    """Integrate an oversampled energy image over detector pixels, once.

    Each detector pixel is the sum of its ``oversample x oversample``
    samples: with samples at the sub-pixel centers and values that are
    fractions of energy per sample, this is the composite midpoint rule for
    the pixel integral (error falling as ``oversample**-2``) and STPSF's
    ``DET_SAMP`` rule. The result is the integrated detector image; applying
    another pixel kernel afterwards would integrate twice.

    Args:
        image: Square energy image, ``(n, n)`` with ``n`` a multiple of
            ``oversample``, as returned by ``nircam_band_image`` or
            ``out.intensity() * out.grid.weights``.
        oversample: Samples per detector pixel along each axis.

    Returns:
        The ``(n / oversample, n / oversample)`` detector image.

    Raises:
        ValueError: The image is not square 2D or does not tile into pixels.
    """
    image = jnp.asarray(image)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"image must be square 2D, got shape {image.shape}")
    if oversample < 1 or image.shape[0] % oversample:
        raise ValueError(
            f"a {image.shape[0]}-sample image does not tile into pixels of "
            f"{oversample} samples"
        )
    n_pix = image.shape[0] // oversample
    pixels, _ = downsample_psf(image, 1.0, (n_pix, n_pix))
    return pixels
