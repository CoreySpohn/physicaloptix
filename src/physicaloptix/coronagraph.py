"""A dLux-backed optixstuff coronagraph.

``DLuxCoronagraph`` satisfies optixstuff's ``AbstractCoronagraph`` interface, so
coronagraphoto / jaxedith consume it as any other coronagraph -- the dLux optical
system is hidden inside. This is the sibling of yippy's sampled-YIP coronagraph:
yippy interpolates a precomputed PSF table, this propagates live (freeze-to-table
is a planned bridge).
"""

import dLux as dl
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from optixstuff.coronagraph import AbstractCoronagraph

from physicaloptix.apertures import to_dlux_aperture

ARCSEC = 180.0 / np.pi * 3600.0  # arcsec per radian


class DLuxCoronagraph(AbstractCoronagraph):
    """An optixstuff coronagraph whose PSFs come from live dLux propagation.

    Build it from an optixstuff primary with :meth:`from_primary`; it then
    satisfies the ``AbstractCoronagraph`` interface (``on_axis_psf`` /
    ``off_axis_psf`` plus the scalar ETC methods).

    No coronagraph mask is modelled yet, so ``on_axis_psf`` is presently the
    telescope PSF -- a suppression-free degenerate coronagraph. Adding an
    occulter / Lyot-stop layer is the next step.
    """

    _aperture: eqx.Module
    _diameter_m: float
    _wf_npixels: int = eqx.field(static=True)
    pixel_scale_lod: float
    IWA: float
    OWA: float
    _raw_contrast: float
    _core_throughput: float

    def __init__(
        self,
        aperture: eqx.Module,
        diameter_m: float,
        *,
        wf_npixels: int = 256,
        pixel_scale_lod: float = 0.25,
        IWA: float = 3.0,
        OWA: float = 30.0,
        raw_contrast: float = 1e-10,
        core_throughput: float = 0.2,
    ) -> None:
        """Wrap a dLux aperture as an optixstuff coronagraph."""
        self._aperture = aperture
        self._diameter_m = diameter_m
        self._wf_npixels = wf_npixels
        self.pixel_scale_lod = pixel_scale_lod
        self.IWA = IWA
        self.OWA = OWA
        self._raw_contrast = raw_contrast
        self._core_throughput = core_throughput

    @classmethod
    def from_primary(cls, primary, **kwargs) -> "DLuxCoronagraph":
        """Build from an optixstuff primary (the optixstuff -> dLux seam)."""
        return cls(to_dlux_aperture(primary), float(primary.diameter_m), **kwargs)

    def _optics(self, pixel_scale_rad, npixels):
        return dl.AngularOpticalSystem(
            wf_npixels=self._wf_npixels,
            diameter=self._diameter_m,
            layers=[("pupil", self._aperture)],
            psf_npixels=int(npixels),
            psf_pixel_scale=float(pixel_scale_rad) * ARCSEC,
            oversample=1,
        )

    # -- image interface (consumed by coronagraphoto) ---------------------
    def on_axis_psf(self, wavelength_nm, pixel_scale_rad, npixels):
        """On-axis PSF via dLux (telescope PSF until a mask is added)."""
        optics = self._optics(pixel_scale_rad, npixels)
        return optics.propagate(jnp.array([wavelength_nm * 1e-9]))

    def off_axis_psf(self, wavelength_nm, separation_lod, pixel_scale_rad, npixels):
        """Off-axis (planet) PSF, placed along +x by convention."""
        optics = self._optics(pixel_scale_rad, npixels)
        wl_m = wavelength_nm * 1e-9
        offset = jnp.array([separation_lod * wl_m / self._diameter_m, 0.0])
        return optics.propagate(jnp.array([wl_m]), offset)

    # -- scalar interface (consumed by jaxedith) -- placeholders ----------
    def throughput(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Core throughput (constant eta_p placeholder)."""
        return self._core_throughput

    def core_area(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Photometric core area in (lambda/D)^2 (placeholder)."""
        return 1.0

    def core_mean_intensity(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Mean stellar leakage (constant raw_contrast placeholder)."""
        return self._raw_contrast

    def occulter_transmission(self, separation_lod, wavelength_nm, *, time_s=0.0):
        """Off-axis sky transmission (no occulter modelled -> 1)."""
        return 1.0

    def __repr__(self):
        """One-line summary."""
        return (
            f"DLuxCoronagraph(D={self._diameter_m:.3g} m, "
            f"wf_npix={self._wf_npixels}, no mask [telescope PSF])"
        )


def psf(primary, wavelength_nm, pixel_scale_rad, npixels, **kwargs):
    """One-liner facade: optixstuff primary -> on-axis PSF, no visible dLux."""
    coro = DLuxCoronagraph.from_primary(primary, **kwargs)
    return coro.on_axis_psf(wavelength_nm, pixel_scale_rad, npixels)
