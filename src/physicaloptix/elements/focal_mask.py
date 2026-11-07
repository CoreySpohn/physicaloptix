"""The Babinet-form focal-plane mask and its full-band FFT selection.

``FocalPlaneMask`` is a pupil -> focal-plane mask -> pupil stage for any
sampled amplitude transmission. ``mask_window`` builds the mask-plane window
grid (by default the pupil raster's full Nyquist band) and
``select_mask_transform`` picks the transform pair that carries the pupil field
to it: the zero-padded FFT (``FullBandFraunhofer``) on the full band, the dense
MFT (``Fraunhofer``) on any other window.
"""

import equinox as eqx
from jaxtyping import Array

from physicaloptix.core import Field, Grid, PlaneKind
from physicaloptix.transforms import Fraunhofer, FullBandFraunhofer

__all__ = ["FocalPlaneMask", "mask_window", "select_mask_transform"]


def mask_window(pupil_grid, oversample, *, extent_lod=None):
    """The mask-plane window grid for a pupil grid.

    Args:
        pupil_grid: The pupil grid (``npix`` samples spanning ``npix * dx``).
        oversample: Window samples per lambda/D (a pitch of ``1 / oversample``).
        extent_lod: Full window width in lambda/D; ``None`` is the pupil
            raster's full Nyquist band, ``pupil_grid.npix`` lambda/D.

    Returns:
        The focal ``Grid`` of the window.

    Raises:
        ValueError: The window holds no sample.
    """
    extent = pupil_grid.npix if extent_lod is None else extent_lod
    n_mask = round(extent * oversample)
    if n_mask < 1:
        raise ValueError(
            f"mask-plane window of {extent} lambda/D at {oversample} "
            "samples per lambda/D holds no sample"
        )
    return Grid.focal(n_mask, 1.0 / oversample)


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
    full_band = window.npix == pupil_grid.npix * oversample
    integer = float(oversample).is_integer()
    if mask_transform == "fft" and not (full_band and integer):
        raise ValueError(
            "mask_transform='fft' needs the full-band window (extent "
            f"{pupil_grid.npix} lambda/D) and an integer oversample, got a "
            f"window of {window.npix} samples at oversample {oversample}"
        )
    if mask_transform == "fft" or (mask_transform == "auto" and full_band and integer):
        return FullBandFraunhofer(pupil_grid, int(oversample))
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
        transform: The ``Fraunhofer`` (MFT) or ``FullBandFraunhofer`` (FFT) pair,
            pupil grid -> mask-plane window.
        transmission: Mask amplitude transmission on the window, ``(n, n)``,
            rows y; sampled at one wavelength, so the stage is monochromatic.
    """

    transform: Fraunhofer | FullBandFraunhofer
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
