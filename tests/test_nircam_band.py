"""Band integration and finite-pixel anchors for the NIRCam builder (data-free).

Band integration must propagate every wavelength node independently onto one
fixed angular output grid: at a fixed source angle the image centroid stays
put, the diffraction pattern widens in proportion to the wavelength, and a
fixed OPD produces a phase that falls as 1 / wavelength. A shortcut that
reuses one monochromatic pupil field (its OPD phase frozen at one wavelength)
and only rescales the transform, or that converts the source angle at one
wavelength, fails these tests. Every expectation is computed here from
physical primitives (meters, radians, nanometers) with an independent NumPy
direct Fourier sum, never through the builder's own unit conversion.

Detector pixels are integrated once, by summing the ``oversample x
oversample`` output samples inside each pixel (the composite midpoint rule,
and STPSF's ``DET_SAMP`` rule). The check is against pixels integrated
independently by Gauss-Legendre quadrature of the direct Fourier sum: the
midpoint error must fall at the formal second order as the oversampling
doubles, which a second (double) pixel integration or a missing one would
not do.

Tolerances: identities between two exact finite sums use the floating-point
floor 1e-10 of the peak (measured residuals are below 1e-14). The
pixel-integration order band [1.8, 2.2] brackets the formal order 2 of the
midpoint rule; the measured orders are listed in that test.
"""

import hashlib
import json

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits

from physicaloptix.ifs import pixel_integrate as ifs_pixel_integrate
from physicaloptix.instruments import (
    NIRCamBand,
    NIRCamConfig,
    NIRCamInputs,
    build_nircam,
    integrate_detector_pixels,
    nircam_band_image,
)

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
NM_TO_M = 1e-9
DIAMETER_M = 6.5
NPUP = 48
FLOOR = 1e-10


def _pupil_coords(npup):
    """Sample centers in pupil diameters, axis at (npup - 1) / 2."""
    return (np.arange(npup) - (npup - 1) / 2) / npup


def _asymmetric_pupil(npup=NPUP):
    x = _pupil_coords(npup)
    xx, yy = np.meshgrid(x, x)  # rows are y, ascending
    amp = ((xx**2 + yy**2) <= 0.25).astype(float)
    amp[(np.abs(xx) < 0.04) & (yy > 0)] = 0.0  # strut on +y only
    amp[(xx > 0.1) & (yy < -0.1) & (amp > 0)] = 0.5  # gray patch in +x, -y
    return amp


def _asymmetric_opd_nm(npup=NPUP):
    """Non-symmetric OPD, |OPD| < 375 nm (no phase wrap above 750 nm)."""
    x = _pupil_coords(npup)
    xx, yy = np.meshgrid(x, x)
    return 300.0 * yy + 500.0 * xx * yy + 800.0 * xx**3


def _inputs(*, opd_nm=None, **extra):
    return NIRCamInputs(
        pupil_amplitude=jnp.asarray(_asymmetric_pupil()),
        pupil_diameter_m=DIAMETER_M,
        ote_opd_nm=None if opd_nm is None else jnp.asarray(opd_nm),
        **extra,
    )


def _config(wavelength_nm, det_npix, oversample, scale, **extra):
    return NIRCamConfig(
        wavelength_nm=wavelength_nm,
        pixel_scale_arcsec=scale,
        detector_npix=det_npix,
        oversample=oversample,
        **extra,
    )


def _propagate(config, inputs, taps=()):
    path, field = build_nircam(config, inputs)
    return path.propagate(field, taps=taps)


def _fraction_image(config, inputs):
    """Fraction of entrance-pupil energy per output sample."""
    out, _ = _propagate(config, inputs)
    return np.asarray(out.intensity()) * out.grid.weights


def _direct_field(amp, opd_nm, wavelength_nm, theta_x_rad, theta_y_rad, source):
    """Independent direct Fourier sum in physical units at arbitrary angles.

    The entrance-frame arrays reach the detector reflected in y; a source at
    sky offset ``s`` images at ``theta - s``. The field is scaled to unit
    entrance-pupil energy, so ``|E|**2`` is the energy density per
    (lambda/D)**2 at this wavelength.
    """
    npup = amp.shape[0]
    lam_m = wavelength_nm * NM_TO_M
    x_m = _pupil_coords(npup) * DIAMETER_M
    sx, sy = np.asarray(source) * ARCSEC_TO_RAD
    kx = np.exp(-2j * np.pi * np.outer(np.asarray(theta_x_rad) - sx, x_m) / lam_m)
    ky = np.exp(-2j * np.pi * np.outer(np.asarray(theta_y_rad) - sy, x_m) / lam_m)
    pupil = amp[::-1, :] * np.exp(2j * np.pi * opd_nm[::-1, :] * NM_TO_M / lam_m)
    norm = 1.0 / np.sqrt(np.sum(amp**2) / npup**2)
    return norm * (ky @ pupil @ kx.T) / npup**2


def _output_angles(det_npix, oversample, scale):
    n_out = det_npix * oversample
    step_rad = scale / oversample * ARCSEC_TO_RAD
    return (np.arange(n_out) - (n_out - 1) / 2) * step_rad, step_rad


def _expected_fraction(wavelength_nm, det_npix, oversample, scale, source, opd_nm):
    theta, step_rad = _output_angles(det_npix, oversample, scale)
    field = _direct_field(
        _asymmetric_pupil(), opd_nm, wavelength_nm, theta, theta, source
    )
    cell_lod = step_rad * DIAMETER_M / (wavelength_nm * NM_TO_M)
    return np.abs(field) ** 2 * cell_lod**2


class TestTwoWavelengths:
    """Two wavelengths at one fixed source angle, propagated independently."""

    SCALE = 0.06  # arcsec per detector pixel

    def test_centroid_stays_at_the_source_angle(self):
        """No OPD, so each image is centrosymmetric about the source. The
        source sits (+3, -2) output samples off axis on an odd grid, so a box
        centered on that sample holds a symmetric image and its centroid is
        the source angle at every wavelength. Converting the source offset
        at one wavelength only would move the centroid at the other."""
        det_npix, oversample = 11, 3
        step = self.SCALE / oversample
        source = (3 * step, -2 * step)
        center = (det_npix * oversample - 1) // 2
        cx, cy = center + 3, center - 2
        half = 10
        yy, xx = np.mgrid[cy - half : cy + half + 1, cx - half : cx + half + 1]
        images = []
        for wavelength_nm in (1800.0, 3600.0):
            config = _config(
                wavelength_nm,
                det_npix,
                oversample,
                self.SCALE,
                source_position_arcsec=source,
            )
            image = _fraction_image(config, _inputs())
            box = image[cy - half : cy + half + 1, cx - half : cx + half + 1]
            centroid = (
                ((box * xx).sum() / box.sum() - center) * step,
                ((box * yy).sum() / box.sum() - center) * step,
            )
            np.testing.assert_allclose(centroid, source, rtol=0, atol=FLOOR * step)
            images.append(image)
        assert np.abs(images[0] - images[1]).max() > 0.1 * images[0].max()

    def test_diffraction_width_scales_with_wavelength(self):
        """With no OPD the energy density per (lambda/D)**2 is one function
        of theta * D / lambda. At twice the wavelength, sample k of a grid
        with pitch s / 2 is therefore sample k + n / 2 of the half-wavelength
        image on a grid with pitch s / 4 (the same angle divided by two)."""
        det_npix = 10
        long_cfg = _config(3600.0, det_npix, 2, self.SCALE)
        short_cfg = _config(1800.0, det_npix, 4, self.SCALE)
        out_long, _ = _propagate(long_cfg, _inputs())
        out_short, _ = _propagate(short_cfg, _inputs())
        density_long = np.asarray(out_long.intensity())
        density_short = np.asarray(out_short.intensity())
        n = density_long.shape[0]
        matched = density_short[n // 2 : n // 2 + n, n // 2 : n // 2 + n]
        np.testing.assert_allclose(
            density_long, matched, rtol=0, atol=FLOOR * density_long.max()
        )

    def test_opd_phase_falls_as_one_over_wavelength(self):
        """The fixed OPD gives phase 2 pi OPD / lambda in the entrance pupil
        (|OPD| < 375 nm, so no wrap at 1800 nm)."""
        opd_nm = _asymmetric_opd_nm()
        lit = _asymmetric_pupil() > 0
        phases = []
        for wavelength_nm in (1800.0, 3600.0):
            config = _config(wavelength_nm, 10, 2, self.SCALE, ote_opd=True)
            _, taps = _propagate(config, _inputs(opd_nm=opd_nm), taps=("ote_opd",))
            phase = np.angle(np.asarray(taps["ote_opd"].data))[lit]
            np.testing.assert_allclose(
                phase * wavelength_nm / (2 * np.pi), opd_nm[lit], rtol=0, atol=1e-9
            )
            phases.append(phase)
        np.testing.assert_allclose(phases[0], 2 * phases[1], rtol=0, atol=1e-12)


class TestBandImage:
    SCALE = 0.0624
    NODES_NM = (3000.0, 3350.0, 3700.0)
    RAW_WEIGHTS = (2.0, 5.0, 3.0)  # normalized to unit sum by the band image
    SOURCE = (0.13, -0.07)

    @pytest.mark.parametrize(("det_npix", "oversample"), [(11, 3), (8, 2)])
    def test_band_is_the_weighted_sum_of_independent_propagations(
        self, det_npix, oversample
    ):
        """Off-axis source, asymmetric OPD, three nodes: the band image equals
        the weighted sum of independent physical-unit Fourier sums, each at
        its own wavelength, as a fraction of entrance-pupil energy per
        sample. The same sum with the pupil phase frozen at one node (a
        cached monochromatic field) differs by far more than the floor
        (measured 3e-3 of the peak; a monochromatic image 7e-3)."""
        opd_nm = 3.0 * _asymmetric_opd_nm()
        band = NIRCamBand(wavelengths_nm=self.NODES_NM, weights=self.RAW_WEIGHTS)
        config = _config(
            3350.0,
            det_npix,
            oversample,
            self.SCALE,
            source_position_arcsec=self.SOURCE,
            ote_opd=True,
        )
        image = np.asarray(nircam_band_image(config, _inputs(opd_nm=opd_nm), band))
        weights = np.asarray(self.RAW_WEIGHTS) / np.sum(self.RAW_WEIGHTS)
        expected = sum(
            w
            * _expected_fraction(
                lam, det_npix, oversample, self.SCALE, self.SOURCE, opd_nm
            )
            for lam, w in zip(self.NODES_NM, weights, strict=True)
        )
        np.testing.assert_allclose(image, expected, rtol=0, atol=FLOOR * expected.max())
        frozen = sum(
            w
            * _expected_fraction(
                lam,
                det_npix,
                oversample,
                self.SCALE,
                self.SOURCE,
                opd_nm * lam / self.NODES_NM[1],
            )
            for lam, w in zip(self.NODES_NM, weights, strict=True)
        )
        assert np.abs(frozen - expected).max() > 1e-3 * expected.max()
        mono = _expected_fraction(
            self.NODES_NM[1], det_npix, oversample, self.SCALE, self.SOURCE, opd_nm
        )
        assert np.abs(mono - expected).max() > 1e-3 * expected.max()

    def test_each_node_uses_its_own_si_opd(self):
        """Per-node post-mask SI OPDs: node k carries a tilt that moves its
        image by m_k output samples in +x (library convention, exit frame).
        The band image must be the weighted sum of the node images each
        shifted by its own m_k; reusing one node's OPD shifts all alike."""
        det_npix, oversample = 10, 2
        nodes = (3000.0, 3600.0)
        shifts = (1, 3)
        weights = (0.4, 0.6)
        step_rad = self.SCALE / oversample * ARCSEC_TO_RAD
        x_m = _pupil_coords(NPUP) * DIAMETER_M
        xx, _ = np.meshgrid(x_m, x_m)
        cube = np.stack([m * step_rad * xx / NM_TO_M for m in shifts])
        band = NIRCamBand(wavelengths_nm=nodes, weights=weights, si_opd_nm=cube)
        config = _config(3350.0, det_npix, oversample, self.SCALE, si_wfe=True)
        inputs = _inputs(si_opd_nm=jnp.zeros((NPUP, NPUP)), si_wavelength_nm=3350.0)
        image = np.asarray(nircam_band_image(config, inputs, band))
        n = image.shape[0]
        top = max(shifts)
        expected = np.zeros((n, n - top))
        for lam, w, m in zip(nodes, weights, shifts, strict=True):
            flat = _fraction_image(
                _config(lam, det_npix, oversample, self.SCALE), inputs
            )
            expected += w * flat[:, top - m : n - m]
        np.testing.assert_allclose(
            image[:, top:], expected, rtol=0, atol=FLOOR * expected.max()
        )

    def test_monochromatic_si_opd_is_refused_at_other_nodes(self):
        band = NIRCamBand(wavelengths_nm=(3000.0, 3600.0), weights=(0.5, 0.5))
        config = _config(3350.0, 10, 2, self.SCALE, si_wfe=True)
        inputs = _inputs(si_opd_nm=jnp.zeros((NPUP, NPUP)), si_wavelength_nm=3350.0)
        with pytest.raises(ValueError, match="applies at"):
            nircam_band_image(config, inputs, band)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"wavelengths_nm": (), "weights": ()}, "at least one"),
            ({"wavelengths_nm": (3000.0,), "weights": (0.5, 0.5)}, "one weight"),
            ({"wavelengths_nm": (0.0,), "weights": (1.0,)}, "positive"),
            ({"wavelengths_nm": (3000.0, 3100.0), "weights": (1.0, -0.1)}, "negative"),
            ({"wavelengths_nm": (3000.0,), "weights": (0.0,)}, "sum"),
            (
                {
                    "wavelengths_nm": (3000.0, 3100.0),
                    "weights": (0.5, 0.5),
                    "si_opd_nm": np.zeros((3, 4, 4)),
                },
                "slice per node",
            ),
        ],
    )
    def test_band_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            NIRCamBand(**kwargs)


class TestPixelIntegration:
    """Detector pixels integrated once, checked against Gauss-Legendre."""

    WAVELENGTH_NM = 3350.0
    SCALE = 0.0624  # 0.6 lambda/D at 3350 nm for D = 6.5 m
    DET_NPIX = 9
    SOURCE = (0.05, -0.03)

    def _gauss_legendre_pixels(self, order):
        opd_nm = _asymmetric_opd_nm()
        t, w = np.polynomial.legendre.leggauss(order)
        centers = (np.arange(self.DET_NPIX) - (self.DET_NPIX - 1) / 2) * self.SCALE
        half = self.SCALE / 2
        theta = (
            (centers[:, np.newaxis] + half * t[np.newaxis, :]) * ARCSEC_TO_RAD
        ).ravel()
        field = _direct_field(
            _asymmetric_pupil(),
            opd_nm,
            self.WAVELENGTH_NM,
            theta,
            theta,
            self.SOURCE,
        )
        lod_per_rad = DIAMETER_M / (self.WAVELENGTH_NM * NM_TO_M)
        weight = np.tile(w, self.DET_NPIX) * half * ARCSEC_TO_RAD * lod_per_rad
        density = np.abs(field) ** 2 * weight[:, np.newaxis] * weight[np.newaxis, :]
        n = self.DET_NPIX
        return density.reshape(n, order, n, order).sum(axis=(1, 3))

    def _pixels(self, oversample):
        config = _config(
            self.WAVELENGTH_NM,
            self.DET_NPIX,
            oversample,
            self.SCALE,
            source_position_arcsec=self.SOURCE,
            ote_opd=True,
        )
        image = _fraction_image(config, _inputs(opd_nm=_asymmetric_opd_nm()))
        pixels = np.asarray(integrate_detector_pixels(image, oversample))
        np.testing.assert_allclose(pixels.sum(), image.sum(), rtol=1e-14)
        return pixels

    def test_midpoint_pixels_converge_at_second_order(self):
        """Measured (max-norm error / max pixel): q = 2: 3.0e-2, 4: 7.3e-3,
        8: 1.8e-3, 16: 4.5e-4; observed orders 2.04, 2.01, 2.00. The
        quadrature reference is converged: orders 12 and 16 differ by 6e-16
        of the peak pixel. A second pixel integration (a box blur before the
        block sum) or none (point samples) converges to a different image,
        so the error would stall and the order would collapse."""
        reference = self._gauss_legendre_pixels(12)
        check = self._gauss_legendre_pixels(16)
        assert np.abs(reference - check).max() < 1e-12 * reference.max()
        errors = [
            np.abs(self._pixels(q) - reference).max() / reference.max()
            for q in (2, 4, 8, 16)
        ]
        orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
        assert np.all((orders > 1.8) & (orders < 2.2)), (errors, orders)
        assert errors[-1] < 1e-3

    def test_block_sum_is_q_squared_times_the_ifs_window_mean(self):
        """The detector rule sums energies; ``ifs.pixel_integrate`` averages
        an intensity. On aligned, non-overlapping windows they differ by the
        sample count per pixel exactly (to rounding)."""
        q = 4
        image = np.random.default_rng(3).random((12 * q, 12 * q))
        summed = np.asarray(integrate_detector_pixels(image, q))
        mean = np.asarray(ifs_pixel_integrate(jnp.asarray(image), q, q))
        np.testing.assert_allclose(summed, q**2 * mean, rtol=1e-14)
        np.testing.assert_allclose(
            summed, image.reshape(12, q, 12, q).sum(axis=(1, 3)), rtol=1e-14
        )

    @pytest.mark.parametrize(("shape", "oversample"), [((9, 9), 2), ((8, 6), 2)])
    def test_rejects_grids_that_do_not_tile_into_pixels(self, shape, oversample):
        with pytest.raises(ValueError):
            integrate_detector_pixels(np.ones(shape), oversample)


# --- Reading the band nodes, weights and per-node SI OPDs from a bundle ----


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


NODES_M = (3.2e-6, 3.35e-6, 3.5e-6)
WEIGHTS = (0.3, 0.45, 0.25)


def _write_band_bundle(directory, *, waves_in_header=NODES_M, si=True):
    cube_m = np.stack(
        [np.full((NPUP, NPUP), 1e-8 * (k + 1)) for k in range(len(NODES_M))]
    )
    files = {}
    if si:
        primary = fits.PrimaryHDU(np.zeros((NPUP, NPUP)))
        primary.header["CONTENT"] = "opd"
        primary.header["BUNIT"] = "m"
        support = fits.ImageHDU(np.ones((NPUP, NPUP)), name="SUPPORT")
        support.header["CONTENT"] = "amplitude_transmission"
        cube = fits.ImageHDU(cube_m, name="OPD_F335M_N3")
        cube.header["BUNIT"] = "m"
        for k, wave in enumerate(waves_in_header):
            cube.header[f"WAVE{k}"] = wave
        path = directory / "si_wfe_opd.fits"
        fits.HDUList([primary, support, cube]).writeto(path)
        files["si_wfe_opd"] = {"file": path.name, "sha256": _sha256(path)}
    manifest = {
        "bundle_name": "synthetic-band",
        "configuration": {
            "filter": "F335M",
            "f335m_nodes": {
                "source_spectrum": "synthetic",
                "sets": {
                    "3": {
                        "nlambda": 3,
                        "wavelengths_m": list(NODES_M),
                        "weights": list(WEIGHTS),
                    }
                },
            },
        },
        "bundle_files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, cube_m


class TestBandFromBundle:
    def test_reads_nodes_weights_and_si_cube(self, tmp_path):
        bundle, cube_m = _write_band_bundle(tmp_path)
        band = NIRCamBand.from_bundle(bundle, 3)
        np.testing.assert_allclose(band.wavelengths_nm, np.asarray(NODES_M) / NM_TO_M)
        np.testing.assert_allclose(band.weights, WEIGHTS)
        np.testing.assert_allclose(np.asarray(band.si_opd_nm), -cube_m / NM_TO_M)
        assert band.provenance["source_spectrum"] == "synthetic"
        assert band.provenance["si_wfe_opd_sha256"] == _sha256(
            tmp_path / "si_wfe_opd.fits"
        )

    def test_bundle_without_si_gives_no_cube(self, tmp_path):
        bundle, _ = _write_band_bundle(tmp_path, si=False)
        assert NIRCamBand.from_bundle(bundle, 3).si_opd_nm is None

    def test_cube_wavelength_mismatch_fails(self, tmp_path):
        bundle, _ = _write_band_bundle(
            tmp_path, waves_in_header=(3.2e-6, 3.36e-6, 3.5e-6)
        )
        with pytest.raises(ValueError, match="WAVE1"):
            NIRCamBand.from_bundle(bundle, 3)

    def test_missing_node_set_fails(self, tmp_path):
        bundle, _ = _write_band_bundle(tmp_path)
        with pytest.raises(ValueError, match="nlambda = 5"):
            NIRCamBand.from_bundle(bundle, 5)

    def test_checksum_mismatch_fails(self, tmp_path):
        bundle, _ = _write_band_bundle(tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["bundle_files"]["si_wfe_opd"]["sha256"] = "0" * 64
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="sha256"):
            NIRCamBand.from_bundle(bundle, 3)
