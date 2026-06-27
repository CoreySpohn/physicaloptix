"""End-to-end seam tests: optixstuff primary -> physicaloptix PSF.

Checks the normalisation convention (unocculted PSF integrates to ~unit flux)
and that the optixstuff primary correctly drives diffraction (1/D PSF scaling),
through the public ``physicaloptix.psf`` facade.
"""

import numpy as np
import optixstuff as ox

import physicaloptix as po

WL = 600.0  # nm


def _nyquist_rad(diameter_m, wavelength_m=600e-9):
    return (wavelength_m / diameter_m) / 4.0


def _ee_radius_pix(psf, frac=0.5):
    """Sub-pixel encircled-energy radius (pixels) for `frac` of the energy."""
    psf = np.asarray(psf)
    cy, cx = np.unravel_index(int(np.argmax(psf)), psf.shape)
    yy, xx = np.mgrid[0 : psf.shape[0], 0 : psf.shape[1]]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
    order = np.argsort(r)
    cum = np.cumsum(psf.ravel()[order]) / psf.sum()
    return float(np.interp(frac, cum, r[order]))


class TestNormalisation:
    def test_unocculted_psf_integrates_to_unity_wide_fov(self):
        primary = ox.SimplePrimary(diameter_m=6.0)
        out = np.asarray(po.psf(primary, WL, _nyquist_rad(6.0), 256))
        # matches optixstuff on_axis_psf "unit stellar flux before coronagraph"
        assert out.sum() > 0.99

    def test_narrow_fov_truncates_below_unity(self):
        primary = ox.SimplePrimary(diameter_m=6.0)
        ps = _nyquist_rad(6.0)
        wide = np.asarray(po.psf(primary, WL, ps, 256)).sum()
        narrow = np.asarray(po.psf(primary, WL, ps, 48)).sum()
        assert narrow < wide
        assert narrow < 1.0


class TestPrimaryDrivesDiffraction:
    def test_larger_diameter_gives_tighter_psf(self):
        # fixed, fine angular sampling; only the primary diameter changes
        ps = _nyquist_rad(6.0) / 3.0
        small = po.psf(ox.SimplePrimary(diameter_m=6.0), WL, ps, 256)
        large = po.psf(ox.SimplePrimary(diameter_m=12.0), WL, ps, 256)
        ee_small = _ee_radius_pix(small)
        ee_large = _ee_radius_pix(large)
        # PSF angular size scales as 1/D: doubling D halves the EE50 radius
        assert ee_large < ee_small
        assert abs(ee_large / ee_small - 0.5) < 0.06
