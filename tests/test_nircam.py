"""Convention anchors for the NIRCam pupil-path builder (data-free).

Every expectation is computed here from physical primitives (meters, radians,
nanometers) with an independent NumPy direct Fourier sum or an exact Fourier
shift, never through the builder's own unit conversion: a wrong wavelength
unit, pupil diameter, source-offset sign, pupil parity or OPD sign changes
the answer. The synthetic pupil is asymmetric in both x and y and the OPD is
non-symmetric, so a missing or doubled y reflection, an x/y transpose or a
phase conjugation fails at least one test. Even and odd output sizes are
exercised because the optical axis sits between pixels on one and on a
pixel on the other.

Tolerances are floating-point floors: both sides of each comparison are
exact finite sums of the same quadrature (or an exact integer-pixel Fourier
shift), so the only difference is roundoff. Measured (x64): complex residual
<= 8.3e-16 of the peak, shifted-intensity residual <= 1.4e-15 of the peak,
energy error <= 3.7e-15. The 1e-10 floor fails every deliberate mutation
tried (tilt not reflected, inversion dropped or on x, OPD sign or unit
wrong at the reader, wavelength off by 0.1 percent, normalization dropped).
"""

import hashlib
import json

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from hwoutils.conversions import arcsec_to_rad

from physicaloptix.core import PlaneKind
from physicaloptix.instruments import NIRCamConfig, NIRCamInputs, build_nircam

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
NM_TO_M = 1e-9
DIAMETER_M = 6.5
WAVELENGTH_NM = 3350.0
PIXEL_SCALE_ARCSEC = 0.0624
NPUP = 64
FLOOR = 1e-10

# (detector pixels, oversample): even output, odd output with oversampling,
# odd output at detector sampling.
GRIDS = [(16, 2), (11, 3), (15, 1)]


def test_arcsec_constant_parity():
    assert ARCSEC_TO_RAD == pytest.approx(float(arcsec_to_rad(1.0)), rel=1e-15)


def _pupil_coords(npup):
    """Sample centers in pupil diameters, axis at (npup - 1) / 2."""
    return (np.arange(npup) - (npup - 1) / 2) / npup


def _asymmetric_pupil(npup):
    x = _pupil_coords(npup)
    xx, yy = np.meshgrid(x, x)  # rows are y, ascending
    amp = ((xx**2 + yy**2) <= 0.25).astype(float)
    amp[(np.abs(xx) < 0.04) & (yy > 0)] = 0.0  # strut on +y only
    amp[(xx > 0.1) & (yy < -0.1) & (amp > 0)] = 0.5  # gray patch in +x, -y
    return amp


def _asymmetric_opd_nm(npup):
    x = _pupil_coords(npup)
    xx, yy = np.meshgrid(x, x)
    return 300.0 * yy + 500.0 * xx * yy + 800.0 * xx**3


def _inputs(npup=NPUP, *, opd_nm=None):
    return NIRCamInputs(
        pupil_amplitude=jnp.asarray(_asymmetric_pupil(npup)),
        pupil_diameter_m=DIAMETER_M,
        ote_opd_nm=None if opd_nm is None else jnp.asarray(opd_nm),
    )


def _config(det_npix, oversample, *, source=(0.0, 0.0), ote_opd=False, scale=None):
    return NIRCamConfig(
        wavelength_nm=WAVELENGTH_NM,
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC if scale is None else scale,
        detector_npix=det_npix,
        oversample=oversample,
        source_position_arcsec=source,
        ote_opd=ote_opd,
    )


def _propagate(config, inputs, taps=()):
    path, field = build_nircam(config, inputs)
    return path.propagate(field, taps=taps)


def _image(config, inputs):
    out, _ = _propagate(config, inputs)
    return np.asarray(out.intensity())


def _expected_field(amp, opd_nm, det_npix, oversample, source_arcsec):
    """Independent direct Fourier sum in physical units.

    The entrance-frame arrays reach the detector reflected in y (the one
    coordinate inversion of the prescription); a source at sky offset s
    images at theta - s in the detector frame. The field is scaled so the
    entrance-pupil energy is one in pupil-diameter units.
    """
    npup = amp.shape[0]
    lam_m = WAVELENGTH_NM * NM_TO_M
    x_m = _pupil_coords(npup) * DIAMETER_M
    n_out = det_npix * oversample
    step_rad = PIXEL_SCALE_ARCSEC / oversample * ARCSEC_TO_RAD
    theta = (np.arange(n_out) - (n_out - 1) / 2) * step_rad
    sx, sy = np.asarray(source_arcsec) * ARCSEC_TO_RAD
    kx = np.exp(-2j * np.pi * np.outer(theta - sx, x_m) / lam_m)
    ky = np.exp(-2j * np.pi * np.outer(theta - sy, x_m) / lam_m)
    pupil = amp[::-1, :] * np.exp(2j * np.pi * opd_nm[::-1, :] * NM_TO_M / lam_m)
    norm = 1.0 / np.sqrt(np.sum(amp**2) / npup**2)
    return norm * (ky @ pupil @ kx.T) / npup**2


class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"wavelength_nm": 0.0},
            {"wavelength_nm": -3350.0},
            {"pixel_scale_arcsec": 0.0},
            {"detector_npix": 0},
            {"oversample": 0},
            {"source_position_arcsec": (1.0,)},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        base = {
            "wavelength_nm": WAVELENGTH_NM,
            "pixel_scale_arcsec": PIXEL_SCALE_ARCSEC,
        }
        with pytest.raises(ValueError):
            NIRCamConfig(**{**base, **kwargs})

    def test_ote_opd_without_an_opd_map_fails_at_build(self):
        with pytest.raises(ValueError, match="OPD"):
            build_nircam(_config(16, 2, ote_opd=True), _inputs())

    def test_inputs_reject_mismatched_opd(self):
        with pytest.raises(ValueError, match="shape"):
            NIRCamInputs(
                pupil_amplitude=jnp.ones((8, 8)),
                pupil_diameter_m=DIAMETER_M,
                ote_opd_nm=jnp.zeros((8, 9)),
            )


class TestPathStructure:
    def test_named_taps_and_planes(self):
        opd = _asymmetric_opd_nm(NPUP)
        out, taps = _propagate(
            _config(16, 2, ote_opd=True),
            _inputs(opd_nm=opd),
            taps=("entrance_pupil", "ote_opd", "inversion"),
        )
        assert set(taps) == {"entrance_pupil", "ote_opd", "inversion"}
        assert all(t.plane is PlaneKind.PUPIL for t in taps.values())
        assert out.plane is PlaneKind.FOCAL
        assert out.data.shape == (32, 32)

    def test_no_opd_stage_when_disabled(self):
        path, _ = build_nircam(_config(16, 2), _inputs(opd_nm=np.zeros((NPUP, NPUP))))
        assert "ote_opd" not in [stage.name for stage in path.stages]


class TestUnits:
    @pytest.mark.parametrize("oversample", [1, 4])
    def test_output_sampling_in_lambda_over_d(self, oversample):
        """Output spacing = (pixel scale / oversample) in radians over
        lambda / D, from nm and meters typed here."""
        path, _ = build_nircam(_config(9, oversample), _inputs())
        lod_rad = WAVELENGTH_NM * NM_TO_M / DIAMETER_M
        expected = PIXEL_SCALE_ARCSEC / oversample * ARCSEC_TO_RAD / lod_rad
        grid = path.stages[-1].op.grid_out
        assert grid.npix == 9 * oversample
        assert grid.dx == pytest.approx(expected, rel=1e-14)


class TestIndependentFourierSum:
    @pytest.mark.parametrize(("det_npix", "oversample"), GRIDS)
    def test_complex_field_matches_physical_unit_dft(self, det_npix, oversample):
        """Complex focal field (not intensity) against a direct sum in meters
        and radians: pins arcsec and nm units, signed off-axis position, the
        y reflection, the OPD phase sign and the absolute scale together."""
        amp = _asymmetric_pupil(NPUP)
        opd = _asymmetric_opd_nm(NPUP)
        source = (0.37, -0.21)
        out, _ = _propagate(
            _config(det_npix, oversample, source=source, ote_opd=True),
            _inputs(opd_nm=opd),
        )
        expected = _expected_field(amp, opd, det_npix, oversample, source)
        diff = np.abs(np.asarray(out.data) - expected).max()
        assert diff <= FLOOR * np.abs(expected).max()

    def test_no_opd_matches_flat_dft(self):
        amp = _asymmetric_pupil(NPUP)
        out, _ = _propagate(_config(15, 1, source=(-0.2, 0.45)), _inputs())
        expected = _expected_field(amp, np.zeros_like(amp), 15, 1, (-0.2, 0.45))
        diff = np.abs(np.asarray(out.data) - expected).max()
        assert diff <= FLOOR * np.abs(expected).max()


class TestSignedDisplacement:
    @pytest.mark.parametrize(("det_npix", "oversample"), GRIDS)
    def test_source_offset_is_a_signed_pixel_shift(self, det_npix, oversample):
        """A source at (+3, -2) output samples images as the on-axis PSF
        shifted by +3 columns and -2 rows (exact Fourier shift)."""
        step = PIXEL_SCALE_ARCSEC / oversample
        inputs = _inputs()
        on = _image(_config(det_npix, oversample), inputs)
        off = _image(
            _config(det_npix, oversample, source=(3 * step, -2 * step)), inputs
        )
        n = on.shape[0]
        np.testing.assert_allclose(
            off[: n - 2, 3:], on[2:, : n - 3], rtol=0, atol=FLOOR * on.max()
        )

    @pytest.mark.parametrize(("det_npix", "oversample"), GRIDS)
    def test_opd_ramp_moves_image_along_reflected_axes(self, det_npix, oversample):
        """A positive-path OPD ramp of +2 samples along entrance +x and +3
        along entrance +y moves the image +2 columns and -3 rows: +x maps to
        +x, entrance +y reaches the detector as -y."""
        step_rad = PIXEL_SCALE_ARCSEC / oversample * ARCSEC_TO_RAD
        x_m = _pupil_coords(NPUP) * DIAMETER_M
        xx, yy = np.meshgrid(x_m, x_m)
        ramp_nm = (2 * step_rad * xx + 3 * step_rad * yy) / NM_TO_M
        config = _config(det_npix, oversample, ote_opd=True)
        flat = _image(config, _inputs(opd_nm=np.zeros_like(ramp_nm)))
        tilted = _image(config, _inputs(opd_nm=ramp_nm))
        n = flat.shape[0]
        np.testing.assert_allclose(
            tilted[: n - 3, 2:], flat[3:, : n - 2], rtol=0, atol=FLOOR * flat.max()
        )


class TestEnergy:
    @pytest.mark.parametrize(
        ("npup", "det_npix", "oversample", "du_lod"),
        [(32, 32, 2, 0.5), (33, 33, 1, 1.0)],
    )
    def test_energy_conserved_on_complete_grid(
        self, npup, det_npix, oversample, du_lod
    ):
        """Entrance-pupil energy is one and, on a complete conjugate output
        grid (N du = npup), all of it reaches the focal field (Parseval),
        with an off-axis source and an OPD present. Energies are summed here
        from the declared cell sizes, before any normalization of images."""
        lod_arcsec = WAVELENGTH_NM * NM_TO_M / DIAMETER_M / ARCSEC_TO_RAD
        config = _config(
            det_npix,
            oversample,
            source=(0.3, -0.2),
            ote_opd=True,
            scale=du_lod * lod_arcsec * oversample,
        )
        out, taps = _propagate(
            config, _inputs(npup, opd_nm=_asymmetric_opd_nm(npup)), ("entrance_pupil",)
        )
        pupil_energy = np.sum(np.abs(np.asarray(taps["entrance_pupil"].data)) ** 2)
        focal_energy = np.sum(np.abs(np.asarray(out.data)) ** 2)
        assert pupil_energy / npup**2 == pytest.approx(1.0, abs=1e-12)
        assert focal_energy * du_lod**2 == pytest.approx(1.0, abs=1e-10)


# --- STPSF-exported bundle reader ------------------------------------------


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_array(directory, name, data, *, content, bunit, pixscale):
    npix = data.shape[0]
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=float))
    header = hdu.header
    header["CONTENT"] = content
    header["BUNIT"] = bunit
    header["PLANE"] = "entrance_pupil"
    header["PIXSCALE"] = pixscale
    header["PIXUNIT"] = "m/pix"
    header["NPIX"] = npix
    header["CENTERX"] = (npix - 1) / 2
    header["CENTERY"] = (npix - 1) / 2
    header["PHASECNV"] = "exp(+i*2*pi*OPD/lambda)"
    path = directory / name
    hdu.writeto(path)
    return {"file": name, "sha256": _sha256(path)}


def _write_bundle(directory, amp, opd_m, *, opd_bunit="m"):
    pixscale = DIAMETER_M / amp.shape[0]
    files = {
        "pupil_amplitude": _write_array(
            directory,
            "pupil_amplitude.fits",
            amp,
            content="amplitude_transmission",
            bunit="",
            pixscale=pixscale,
        ),
        "ote_opd": _write_array(
            directory,
            "ote_opd.fits",
            opd_m,
            content="opd",
            bunit=opd_bunit,
            pixscale=pixscale,
        ),
    }
    manifest = {"bundle_name": "synthetic", "bundle_files": files}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


class TestFromBundle:
    def test_reads_arrays_diameter_and_provenance(self, tmp_path):
        amp = _asymmetric_pupil(NPUP)
        opd_m = _asymmetric_opd_nm(NPUP) * NM_TO_M
        inputs = NIRCamInputs.from_bundle(_write_bundle(tmp_path, amp, opd_m))
        np.testing.assert_array_equal(np.asarray(inputs.pupil_amplitude), amp)
        assert inputs.pupil_diameter_m == pytest.approx(DIAMETER_M, rel=1e-15)
        assert inputs.provenance["bundle_name"] == "synthetic"
        assert inputs.provenance["files"]["ote_opd"]["sha256"] == _sha256(
            tmp_path / "ote_opd.fits"
        )

    def test_stpsf_opd_ramp_moves_image_toward_minus_x(self, tmp_path):
        """STPSF (POPPY) propagates with an e^{+2 pi i u x} forward kernel:
        a positive OPD increasing toward +x moves its image toward -x (the
        sign POPPY documents for its tilt). A +2-sample ramp in meters must
        therefore move this builder's image by -2 columns: pins the m -> nm
        conversion and the sign reversal at the import boundary."""
        step_rad = PIXEL_SCALE_ARCSEC / 2 * ARCSEC_TO_RAD
        x_m = _pupil_coords(NPUP) * DIAMETER_M
        xx, _ = np.meshgrid(x_m, x_m)
        ramp_m = 2 * step_rad * xx
        amp = _asymmetric_pupil(NPUP)
        inputs = NIRCamInputs.from_bundle(_write_bundle(tmp_path, amp, ramp_m))
        np.testing.assert_allclose(
            np.asarray(inputs.ote_opd_nm), -ramp_m / NM_TO_M, rtol=1e-15
        )
        config = _config(16, 2, ote_opd=True)
        flat = _image(config, _inputs(opd_nm=np.zeros_like(ramp_m)))
        tilted = _image(config, inputs)
        n = flat.shape[0]
        np.testing.assert_allclose(
            tilted[:, : n - 2], flat[:, 2:], rtol=0, atol=FLOOR * flat.max()
        )

    def test_missing_file_fails_explicitly(self, tmp_path):
        amp = _asymmetric_pupil(NPUP)
        _write_bundle(tmp_path, amp, np.zeros_like(amp))
        (tmp_path / "ote_opd.fits").unlink()
        with pytest.raises(FileNotFoundError, match=r"ote_opd\.fits"):
            NIRCamInputs.from_bundle(tmp_path)

    def test_wrong_opd_unit_fails(self, tmp_path):
        amp = _asymmetric_pupil(NPUP)
        _write_bundle(tmp_path, amp, np.zeros_like(amp), opd_bunit="micron")
        with pytest.raises(ValueError, match="BUNIT"):
            NIRCamInputs.from_bundle(tmp_path)

    def test_checksum_mismatch_fails(self, tmp_path):
        amp = _asymmetric_pupil(NPUP)
        _write_bundle(tmp_path, amp, np.zeros_like(amp))
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest["bundle_files"]["pupil_amplitude"]["sha256"] = "0" * 64
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="sha256"):
            NIRCamInputs.from_bundle(tmp_path)
