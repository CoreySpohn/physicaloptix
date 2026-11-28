"""The Babinet-form focal-plane mask and its full-band FFT selection.

``FocalPlaneMask`` is a pupil -> focal-plane mask -> pupil stage for any
sampled amplitude transmission. ``mask_window`` builds the mask-plane window
grid (by default the pupil raster's full Nyquist band) and
``select_mask_transform`` picks the transform pair that carries the pupil field
to it: the zero-padded FFT (``FullBandFraunhofer``) on the full band, the dense
MFT (``Fraunhofer``) on any other window.
"""

import equinox as eqx
import numpy as np
from hwoutils.precision import like
from jaxtyping import Array
from optixstuff.grids import PixelGrid

from physicaloptix.core import Field, PlaneKind
from physicaloptix.transforms import Fraunhofer, FullBandFraunhofer

__all__ = [
    "BandLimitedRoundMask",
    "FocalPlaneMask",
    "mask_window",
    "select_mask_transform",
]

# Second positive zero of the Bessel function J1: the round band-limited
# profile is truncated there, after its first sidelobe.
J1_SECOND_ZERO = 7.015586669815619


def mask_window(pupil_grid, oversample, *, frame, extent_lod=None):
    """The mask-plane window grid for a pupil grid.

    Args:
        pupil_grid: The pupil grid (``npix`` samples spanning ``npix * dx``).
        oversample: Window samples per lambda/D (a pitch of ``1 / oversample``).
        frame: Primary frame defining the mask-plane lambda/D unit.
        extent_lod: Full window width in lambda/D; ``None`` is the pupil
            raster's full Nyquist band, ``pupil_grid.shape[0]`` lambda/D.

    Returns:
        The focal ``PixelGrid`` of the window.

    Raises:
        ValueError: The window holds no sample.
    """
    extent = pupil_grid.shape[0] if extent_lod is None else extent_lod
    n_mask = round(extent * oversample)
    if n_mask < 1:
        raise ValueError(
            f"mask-plane window of {extent} lambda/D at {oversample} "
            "samples per lambda/D holds no sample"
        )
    return PixelGrid.focal(n_mask, 1.0 / oversample, frame)


def select_mask_transform(pupil_grid, window, oversample, mask_transform="auto"):
    """The transform pair from the pupil grid to the mask-plane window.

    Args:
        pupil_grid: The pupil grid.
        window: The window grid from ``mask_window``.
        oversample: The window samples per lambda/D used to build it.
        mask_transform: ``"auto"`` (FFT on the full band, MFT otherwise),
            ``"fft"`` (full band only) or ``"mft"`` (any window).

    Returns:
        A ``FullBandFraunhofer`` or a ``Fraunhofer`` (sampling gate "raise").

    Raises:
        ValueError: ``mask_transform`` is unknown, or ``"fft"`` is requested
            for a window that is not the full band or an oversample that is
            not an integer.
    """
    if mask_transform not in ("auto", "fft", "mft"):
        raise ValueError(
            f"mask_transform must be auto, fft or mft, got {mask_transform!r}"
        )
    full_band = window.shape[0] == pupil_grid.shape[0] * oversample
    integer = float(oversample).is_integer()
    if mask_transform == "fft" and not (full_band and integer):
        raise ValueError(
            "mask_transform='fft' needs the full-band window (extent "
            f"{pupil_grid.shape[0]} lambda/D) and an integer oversample, got a "
            f"window of {window.shape[0]} samples at oversample {oversample}"
        )
    if mask_transform == "fft" or (mask_transform == "auto" and full_band and integer):
        return FullBandFraunhofer(pupil_grid, window.frame, int(oversample))
    return Fraunhofer(pupil_grid, window, on_undersampled="raise")


class FocalPlaneMask(eqx.Module):
    """Pupil -> focal-plane mask -> Lyot pupil, in the Babinet form.

    ``__call__`` returns ``E + B[(M - 1) F E]``: ``F`` / ``B`` are the
    continuous-FT pair between the pupil grid and the mask-plane window
    (``transform``: the MFT ``Fraunhofer``, or ``FullBandFraunhofer`` on the full
    band), ``M`` the amplitude transmission sampled on the window.
    The unperturbed field passes through exactly on any window, and only
    ``(M - 1)`` outside the window is dropped, so a narrower window is exact
    only for a compactly supported perturbation (not for masks with extended
    regions such as a holder half-plane). On the full Nyquist band of the
    pupil raster (window of ``npup`` lambda/D) ``B F`` is the identity and the
    result equals ``B[M F E]``. A composite pupil-to-pupil
    stage, so it carries ``plane_in`` / ``plane_out`` like a propagator.

    Attributes:
        transform: The Fraunhofer (MFT) or FullBandFraunhofer (FFT) pair, pupil grid ->
            mask-plane window.
        transmission: Mask amplitude transmission on the window, (n, n), rows y; sampled
            at one wavelength, so the stage is monochromatic.
        plane_in: Plane of the incoming field (default pupil).
        plane_out: Plane of the returned field (default pupil).
    """

    transform: Fraunhofer | FullBandFraunhofer
    transmission: Array
    plane_in: PlaneKind = eqx.field(static=True, default=PlaneKind.PUPIL)
    plane_out: PlaneKind = eqx.field(static=True, default=PlaneKind.PUPIL)

    def spectral_breaks_nm(self) -> np.ndarray:
        """The sampled mask is fixed; its transform supplies any breaks.

        Returns:
            Wavelength breaks in nanometers declared by the transform.
        """
        return self.transform.spectral_breaks_nm()

    def __check_init__(self):
        """Validate that the transmission matches the window grid."""
        npix = self.transform.grid_out.shape[0]
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
        """The field just after the mask, on the mask-plane window (focal).

        Args:
            field: Monochromatic pupil field on the pupil grid of the transform.

        Returns:
            Field on the mask-plane window, in the focal plane, multiplied by the
            transmission.
        """
        self._check_mono(field)
        focal = self.transform.forward(field)
        return Field(
            data=focal.data * self.transmission,
            grid=focal.grid,
            plane=focal.plane,
            spectrum=focal.spectrum,
        )

    def __call__(self, field):
        """Return the field arriving at the Lyot plane (same pupil grid).

        Args:
            field: Monochromatic pupil field on the pupil grid of the transform.

        Returns:
            Field at the Lyot plane on the same pupil grid, in the plane_out plane.
        """
        self._check_mono(field)
        focal = self.transform.forward(field)
        perturbation = Field(
            data=focal.data * (self.transmission - like(1.0, focal.data)),
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


class BandLimitedRoundMask(eqx.Module):
    """A round band-limited occulter with an optional layout of override regions.

    Amplitude (electric-field) transmission, real and non-negative, after
    Krist et al.'s band-limited design, truncated at ``clip``::

        t(r) = 1 - (2 J1(s) / s)**2,  s = clip(sigma * r, tiny, clip),  t(0) = 0

    (Krist et al. give the intensity transmission; the amplitude lacks the
    outer square.) ``regions`` are then applied in order, each an open box
    ``(x_lo, x_hi, y_lo, y_hi, amplitude)`` in arcsec relative to the mask
    center, ``None`` meaning unbounded on that side: for example neutral
    density squares and opaque or glass edges. Later regions override earlier
    ones. Coordinates are in the mask-plane frame (x right, y up).

    Attributes:
        sigma_per_arcsec: Band-limit parameter sigma in 1/arcsec.
        clip: Value of ``sigma * r`` at which the profile is truncated
            (default: the second zero of J1).
        regions: Rectangular override regions, in application order.
        center_arcsec: ``(x, y)`` mask center in arcsec.
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
        t = np.array(1.0 - (2.0 * j1(s) / s) ** 2)
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
