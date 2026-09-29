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

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
from astropy.io import fits
from hwoutils.conversions import arcsec_to_rad

from physicaloptix.core import PlaneKind
from physicaloptix.instruments import (
    BandLimitedRoundMask,
    NIRCamConfig,
    NIRCamInputs,
    build_nircam,
    mask_plane_field,
)

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


def _inputs(npup=NPUP, *, opd_nm=None, **extra):
    return NIRCamInputs(
        pupil_amplitude=jnp.asarray(_asymmetric_pupil(npup)),
        pupil_diameter_m=DIAMETER_M,
        ote_opd_nm=None if opd_nm is None else jnp.asarray(opd_nm),
        **extra,
    )


def _config(
    det_npix, oversample, *, source=(0.0, 0.0), ote_opd=False, scale=None, **extra
):
    return NIRCamConfig(
        wavelength_nm=WAVELENGTH_NM,
        pixel_scale_arcsec=PIXEL_SCALE_ARCSEC if scale is None else scale,
        detector_npix=det_npix,
        oversample=oversample,
        source_position_arcsec=source,
        ote_opd=ote_opd,
        **extra,
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


# --- Coronagraphic path: focal mask, Lyot stop, post-mask SI WFE -----------
#
# The masked-path anchors below use the same synthetic asymmetric pupil. The
# occulter is the round band-limited profile with a core radius of 0.6 arcsec
# (about 5.6 lambda/D here), so on the full-band mask-plane window of
# NPUP * mask_oversample samples its perturbation (M - 1) is compactly
# supported well inside the window. Every tolerance is a floating-point
# floor for the same reason as above: both sides are exact finite sums.

LOD_RAD = WAVELENGTH_NM * NM_TO_M / DIAMETER_M
LOD_ARCSEC = LOD_RAD / ARCSEC_TO_RAD
J1_ZERO2 = float(scipy.special.jn_zeros(1, 2)[1])
CORE_RADIUS_ARCSEC = 0.6
CORE_SIGMA = J1_ZERO2 / CORE_RADIUS_ARCSEC
BOX_REGION = ((0.1, 0.5, 0.2, None, 0.3),)


class _UniformMask(eqx.Module):
    """A focal mask of constant amplitude transmission everywhere."""

    amplitude: float = eqx.field(static=True)

    def transmission(self, x_arcsec, y_arcsec):
        shape = np.broadcast_shapes(np.shape(x_arcsec), np.shape(y_arcsec))
        return np.full(shape, self.amplitude)


def _reference_mask(x, y, *, center=(0.0, 0.0), regions=()):
    """Independent round band-limited amplitude profile (arcsec in)."""
    xr = np.asarray(x, float) - center[0]
    yr = np.asarray(y, float) - center[1]
    xr, yr = np.broadcast_arrays(xr, yr)
    r = np.hypot(xr, yr)
    s = np.clip(CORE_SIGMA * r, np.finfo(float).tiny, J1_ZERO2)
    t = np.where(r == 0, 0.0, 1.0 - (2.0 * scipy.special.j1(s) / s) ** 2)
    for x_lo, x_hi, y_lo, y_hi, amplitude in regions:
        sel = np.ones(t.shape, bool)
        for value, lo, hi in ((xr, x_lo, x_hi), (yr, y_lo, y_hi)):
            if lo is not None:
                sel &= value > lo
            if hi is not None:
                sel &= value < hi
        t = np.where(sel, amplitude, t)
    return t


def _exit_lyot(npup):
    """Asymmetric Lyot stop in the exit-pupil frame (notch on +x only)."""
    x = _pupil_coords(npup)
    xx, yy = np.meshgrid(x, x)
    lyot = ((xx**2 + yy**2) <= 0.42**2).astype(float)
    lyot[(xx > 0.3) & (np.abs(yy) < 0.05)] = 0.0
    return lyot


def _exit_support(npup):
    """SI support with a stripe removed at -y (exit frame)."""
    x = _pupil_coords(npup)
    _, yy = np.meshgrid(x, x)
    support = np.ones((npup, npup))
    support[(yy < -0.25) & (yy > -0.35)] = 0.0
    return support


def _exit_si_opd_nm(npup):
    x = _pupil_coords(npup)
    xx, yy = np.meshgrid(x, x)
    return 400.0 * xx**2 - 250.0 * xx * yy + 600.0 * yy**3


def _occulter(**kwargs):
    return BandLimitedRoundMask(sigma_per_arcsec=CORE_SIGMA, **kwargs)


def _masked_inputs(*, mask=None, opd_nm=None, si_opd_nm=None, support=None):
    lyot = _exit_lyot(NPUP)
    return _inputs(
        opd_nm=opd_nm,
        focal_mask=_occulter() if mask is None else mask,
        lyot_amplitude=jnp.asarray(lyot),
        si_opd_nm=None if si_opd_nm is None else jnp.asarray(si_opd_nm),
        si_support=None if support is None else jnp.asarray(support),
        si_wavelength_nm=None if si_opd_nm is None else WAVELENGTH_NM,
    )


def _scale_for_pitch(oversample, pitch_lod):
    """Detector pixel scale [arcsec] giving output samples of pitch_lod."""
    return oversample * pitch_lod * LOD_ARCSEC


def _expected_masked_field(
    amp,
    opd_nm,
    lyot,
    si_opd_nm,
    support,
    mask_fn,
    mask_oversample,
    det_npix,
    oversample,
    source_arcsec,
    scale,
):
    """Independent direct Fourier sums in meters and radians, full band.

    The exit-frame field (entrance arrays reflected in y, source as a tilt)
    is summed to the mask-plane samples at theta_j = (j - (N - 1)/2) * lambda
    / (D * mask_oversample), multiplied by the mask, summed back to the pupil
    samples (the direct form, not the Babinet form), multiplied by the Lyot
    stop and the SI support and phasor, and summed to the detector samples.
    """
    npup = amp.shape[0]
    lam_m = WAVELENGTH_NM * NM_TO_M
    x_m = _pupil_coords(npup) * DIAMETER_M
    xx, yy = np.meshgrid(x_m, x_m)
    sx, sy = np.asarray(source_arcsec) * ARCSEC_TO_RAD
    norm = 1.0 / np.sqrt(np.sum(amp**2) / npup**2)
    field = (
        norm
        * amp[::-1, :]
        * np.exp(2j * np.pi * opd_nm[::-1, :] * NM_TO_M / lam_m)
        * np.exp(2j * np.pi * (sx * xx + sy * yy) / lam_m)
    )
    n_mask = npup * mask_oversample
    theta = (
        (np.arange(n_mask) - (n_mask - 1) / 2) * lam_m / (DIAMETER_M * mask_oversample)
    )
    k_fwd = np.exp(-2j * np.pi * np.outer(theta, x_m) / lam_m)
    focal = k_fwd @ field @ k_fwd.T / npup**2
    theta_arcsec = theta / ARCSEC_TO_RAD
    focal = focal * mask_fn(theta_arcsec[np.newaxis, :], theta_arcsec[:, np.newaxis])
    k_bwd = np.exp(2j * np.pi * np.outer(x_m, theta) / lam_m)
    lyot_field = k_bwd @ focal @ k_bwd.T / mask_oversample**2
    lyot_field = lyot_field * lyot * support
    lyot_field = lyot_field * np.exp(2j * np.pi * si_opd_nm * NM_TO_M / lam_m)
    n_out = det_npix * oversample
    step_rad = scale / oversample * ARCSEC_TO_RAD
    theta_out = (np.arange(n_out) - (n_out - 1) / 2) * step_rad
    k_out = np.exp(-2j * np.pi * np.outer(theta_out, x_m) / lam_m)
    return k_out @ lyot_field @ k_out.T / npup**2


class TestMaskedConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mask_oversample": 0},
            {"mask_extent_lod": 0.0},
            {"mask_extent_lod": -4.0},
        ],
    )
    def test_rejects_invalid_mask_sampling(self, kwargs):
        with pytest.raises(ValueError):
            NIRCamConfig(
                wavelength_nm=WAVELENGTH_NM,
                pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("toggle", "match"),
        [("focal_mask", "focal mask"), ("lyot_stop", "Lyot"), ("si_wfe", "SI")],
    )
    def test_missing_input_fails_at_build(self, toggle, match):
        with pytest.raises(ValueError, match=match):
            build_nircam(_config(16, 2, **{toggle: True}), _inputs())

    def test_si_wavelength_mismatch_fails_at_build(self):
        inputs = _inputs(
            si_opd_nm=jnp.zeros((NPUP, NPUP)),
            si_wavelength_nm=WAVELENGTH_NM + 10.0,
        )
        with pytest.raises(ValueError, match="wavelength"):
            build_nircam(_config(16, 2, si_wfe=True), inputs)

    def test_window_beyond_full_band_fails_at_build(self):
        with pytest.raises(ValueError, match="undersampled"):
            build_nircam(
                _config(16, 2, focal_mask=True, mask_extent_lod=2.0 * NPUP),
                _masked_inputs(),
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lyot_amplitude": jnp.ones((NPUP, NPUP + 1))},
            {"si_opd_nm": jnp.zeros((NPUP - 1, NPUP)), "si_wavelength_nm": 1.0},
            {"si_opd_nm": jnp.zeros((NPUP, NPUP))},
        ],
    )
    def test_inputs_reject_bad_post_mask_arrays(self, kwargs):
        with pytest.raises(ValueError):
            _inputs(**kwargs)


class TestMaskedStructure:
    def test_stage_order_and_taps(self):
        opd = _asymmetric_opd_nm(NPUP)
        inputs = _masked_inputs(opd_nm=opd, si_opd_nm=_exit_si_opd_nm(NPUP))
        config = _config(
            16, 2, ote_opd=True, focal_mask=True, lyot_stop=True, si_wfe=True
        )
        path, field = build_nircam(config, inputs)
        assert [stage.name for stage in path.stages] == [
            "entrance_pupil",
            "ote_opd",
            "inversion",
            "focal_mask",
            "lyot_stop",
            "si_wfe",
            "science_focal",
        ]
        taps = ("entrance_pupil", "inversion", "focal_mask", "lyot_stop", "si_wfe")
        out, tapped = path.propagate(field, taps=taps)
        assert all(tapped[name].plane is PlaneKind.PUPIL for name in taps)
        mask_plane = mask_plane_field(path, tapped)
        assert mask_plane.plane is PlaneKind.FOCAL
        assert mask_plane.data.shape == (NPUP * 4, NPUP * 4)
        assert mask_plane.grid.dx == pytest.approx(0.25, rel=1e-15)
        assert out.data.shape == (32, 32)

    def test_mask_plane_tap_needs_the_pre_mask_field(self):
        path, field = build_nircam(_config(16, 2, focal_mask=True), _masked_inputs())
        _, tapped = path.propagate(field, taps=("focal_mask",))
        with pytest.raises(ValueError, match="inversion"):
            mask_plane_field(path, tapped)


class TestUnityMask:
    @pytest.mark.parametrize("extent_lod", [None, 24.0, 8.0])
    @pytest.mark.parametrize(("det_npix", "oversample"), GRIDS)
    def test_unity_mask_is_the_identity(self, extent_lod, det_npix, oversample):
        """A unity focal mask returns the unperturbed pupil field on any
        mask-plane window, including a small stamp (8 lambda/D): the image
        equals the same path without a focal mask, Lyot stop on, OPD and an
        off-axis source present."""
        opd = _asymmetric_opd_nm(NPUP)
        source = (0.21, -0.13)
        inputs = _masked_inputs(mask=_UniformMask(1.0), opd_nm=opd)
        common = {"source": source, "ote_opd": True, "lyot_stop": True}
        masked, taps = _propagate(
            _config(
                det_npix,
                oversample,
                focal_mask=True,
                mask_oversample=2,
                mask_extent_lod=extent_lod,
                **common,
            ),
            inputs,
            taps=("inversion", "focal_mask"),
        )
        reference, _ = _propagate(_config(det_npix, oversample, **common), inputs)
        np.testing.assert_allclose(
            np.asarray(taps["focal_mask"].data),
            np.asarray(taps["inversion"].data),
            rtol=0,
            atol=FLOOR * np.abs(np.asarray(taps["inversion"].data)).max(),
        )
        ref = np.asarray(reference.data)
        np.testing.assert_allclose(
            np.asarray(masked.data), ref, rtol=0, atol=FLOOR * np.abs(ref).max()
        )


class TestKnownAttenuation:
    @pytest.mark.parametrize("mask_oversample", [2, 3])
    def test_half_amplitude_mask_passes_a_quarter_of_the_energy(self, mask_oversample):
        """A uniform 0.5 amplitude mask over the full band halves the field
        at the Lyot plane and passes 0.25 of the entrance-pupil energy, both
        at the Lyot plane and on a complete conjugate output grid."""
        config = _config(
            64,
            1,
            source=(0.3, -0.2),
            focal_mask=True,
            mask_oversample=mask_oversample,
            scale=_scale_for_pitch(1, 1.0),
        )
        out, taps = _propagate(
            config,
            _masked_inputs(mask=_UniformMask(0.5)),
            taps=("inversion", "focal_mask"),
        )
        pre = np.asarray(taps["inversion"].data)
        post = np.asarray(taps["focal_mask"].data)
        np.testing.assert_allclose(
            post, 0.5 * pre, rtol=0, atol=FLOOR * np.abs(pre).max()
        )
        assert float(taps["focal_mask"].energy()) == pytest.approx(0.25, abs=1e-12)
        mask_plane = mask_plane_field(
            build_nircam(config, _masked_inputs(mask=_UniformMask(0.5)))[0], taps
        )
        assert float(mask_plane.energy()) == pytest.approx(0.25, abs=1e-12)
        focal_energy = np.sum(np.abs(np.asarray(out.data)) ** 2) * out.grid.weights
        assert focal_energy == pytest.approx(0.25, abs=1e-10)


class TestMaskSupport:
    def test_result_is_window_independent_once_the_support_is_inside(self):
        """COMPACT perturbation only (a regionless occulter, (M - 1) within
        0.6 arcsec = 5.6 lambda/D): every mask-plane window that contains it
        gives the full-band result; a window that cuts it (+-4 lambda/D) does
        not. Masks with extended regions are covered by the next test."""

        def image(extent):
            config = _config(
                16,
                2,
                source=(0.05, 0.02),
                focal_mask=True,
                lyot_stop=True,
                mask_oversample=2,
                mask_extent_lod=extent,
            )
            return _image(config, _masked_inputs())

        full = image(None)
        for extent in (16.0, 24.0, 32.0):
            np.testing.assert_allclose(
                image(extent), full, rtol=0, atol=FLOOR * full.max()
            )
        assert np.abs(image(8.0) - full).max() > 1e-3 * full.max()

    @pytest.mark.parametrize("extent_lod", [16.0, 32.0, 48.0])
    def test_non_compact_mask_needs_the_full_band(self, extent_lod):
        """A mask with an opaque half-plane (y > 1 arcsec, running to the band
        edge, like the NIRCam holder) has a non-compact (M - 1). On the full
        band the Lyot-plane field equals the direct B[M F E] computed
        independently, and (square DFT window, one sample per lambda/D, where
        Parseval is exact) the returned energy equals the post-mask
        mask-plane energy: the half-plane's light is removed. Every truncated
        window keeps some of it."""
        half_plane = ((None, None, 1.0, None, 0.0),)
        mask = _occulter(regions=half_plane)
        amp = _asymmetric_pupil(NPUP)
        common = {"source": (0.1, 0.05), "focal_mask": True, "mask_oversample": 1}
        inputs = _masked_inputs(mask=mask)

        def lyot_field(extent):
            _, taps = _propagate(
                _config(16, 2, mask_extent_lod=extent, **common),
                inputs,
                taps=("inversion", "focal_mask"),
            )
            return taps

        full = lyot_field(None)
        flat = np.zeros_like(amp)
        expected = _expected_masked_field(
            amp,
            flat,
            np.ones_like(amp),
            flat,
            np.ones_like(amp),
            lambda x, y: _reference_mask(x, y, regions=half_plane),
            1,
            NPUP,
            1,
            common["source"],
            LOD_ARCSEC,
        )
        # Detector samples on a complete conjugate grid (1 lambda/D, npup
        # samples): the direct-form reference is compared through them.
        out, _ = _propagate(_config(NPUP, 1, scale=LOD_ARCSEC, **common), inputs)
        np.testing.assert_allclose(
            np.asarray(out.data), expected, rtol=0, atol=FLOOR * np.abs(expected).max()
        )
        path, _ = build_nircam(_config(16, 2, **common), inputs)
        post_mask_energy = float(mask_plane_field(path, full).energy())
        assert float(full["focal_mask"].energy()) == pytest.approx(
            post_mask_energy, rel=1e-12
        )
        truncated = lyot_field(extent_lod)
        e_full = float(full["focal_mask"].energy())
        assert float(truncated["focal_mask"].energy()) > e_full * (1 + 1e-3)


class TestMaskedIndependentFourierSum:
    @pytest.mark.parametrize(("det_npix", "oversample"), GRIDS)
    def test_complex_field_matches_physical_unit_dft(self, det_npix, oversample):
        """Complex output field of the full coronagraphic path against direct
        sums in meters and radians: pins the mask pitch, extent, arcsec scale
        and orientation, Babinet == direct propagation on the full band, the
        Lyot and SI arrays in the exit-pupil frame, the SI support and the
        post-mask OPD sign."""
        amp = _asymmetric_pupil(NPUP)
        opd = _asymmetric_opd_nm(NPUP)
        lyot = _exit_lyot(NPUP)
        si_opd = _exit_si_opd_nm(NPUP)
        support = _exit_support(NPUP)
        source = (0.37, -0.21)
        center = (0.12, -0.08)
        mask = _occulter(center_arcsec=center, regions=BOX_REGION)
        inputs = _masked_inputs(
            mask=mask, opd_nm=opd, si_opd_nm=si_opd, support=support
        )
        config = _config(
            det_npix,
            oversample,
            source=source,
            ote_opd=True,
            focal_mask=True,
            lyot_stop=True,
            si_wfe=True,
            mask_oversample=2,
        )
        out, _ = _propagate(config, inputs)
        expected = _expected_masked_field(
            amp,
            opd,
            lyot,
            si_opd,
            support,
            lambda x, y: _reference_mask(x, y, center=center, regions=BOX_REGION),
            2,
            det_npix,
            oversample,
            source,
            PIXEL_SCALE_ARCSEC,
        )
        diff = np.abs(np.asarray(out.data) - expected).max()
        assert diff <= FLOOR * np.abs(expected).max()


class TestMaskDisplacement:
    def test_image_is_mask_intensity_times_unmasked_image(self):
        """Without a Lyot stop, with a square-DFT mask-plane window (full band
        at one sample per lambda/D, where F and B are exact inverses on both
        sides) and the output samples on the mask-plane samples, the image is
        |M|^2 times the unmasked image at every sample: an off-center occulter
        with an asymmetric region is imaged where the mask prescription puts
        it, x right and y up, in arcsec, as an amplitude (not intensity)
        transmission. Even output parity (32 samples)."""
        center = (0.3, -0.2)
        mask = _occulter(center_arcsec=center, regions=BOX_REGION)
        scale = _scale_for_pitch(2, 1.0)
        source = (0.15, -0.1)
        base = {"source": source, "scale": scale, "mask_oversample": 1}
        masked = _image(
            _config(16, 2, focal_mask=True, **base), _masked_inputs(mask=mask)
        )
        unmasked = _image(_config(16, 2, **base), _masked_inputs(mask=mask))
        theta = (np.arange(32) - 15.5) * LOD_ARCSEC
        m = _reference_mask(
            theta[np.newaxis, :],
            theta[:, np.newaxis],
            center=center,
            regions=BOX_REGION,
        )
        np.testing.assert_allclose(
            masked, m**2 * unmasked, rtol=0, atol=FLOOR * unmasked.max()
        )
        assert m.min() < 0.1  # the occulter core lies inside the output field
        assert np.any(m == 0.3)  # and so does part of the box region

    @pytest.mark.parametrize(("det_npix", "oversample"), GRIDS)
    def test_source_shift_equals_opposite_mask_shift(self, det_npix, oversample):
        """Moving the source by s under a fixed occulter gives the image of a
        centered source with the occulter moved by -s, shifted by +s: here
        s = (+3, -2) samples, with the Lyot stop in, so the occulted image
        moves consistently and with the correct sign."""
        pitch = 0.5
        scale = _scale_for_pitch(oversample, pitch)
        step = pitch * LOD_ARCSEC
        s = (3 * step, -2 * step)
        base = {
            "scale": scale,
            "focal_mask": True,
            "lyot_stop": True,
            "mask_oversample": 2,
        }
        moved_source = _image(
            _config(det_npix, oversample, source=s, **base), _masked_inputs()
        )
        moved_mask = _image(
            _config(det_npix, oversample, **base),
            _masked_inputs(mask=_occulter(center_arcsec=(-s[0], -s[1]))),
        )
        n = moved_source.shape[0]
        np.testing.assert_allclose(
            moved_source[: n - 2, 3:],
            moved_mask[2:, : n - 3],
            rtol=0,
            atol=FLOOR * moved_source.max(),
        )


class TestPrePostMaskOPD:
    def _pair(self, mask, det_npix=16, oversample=2, scale=None):
        opd = _asymmetric_opd_nm(NPUP)
        common = {
            "focal_mask": True,
            "lyot_stop": True,
            "mask_oversample": 2,
            "scale": scale,
        }
        pre, _ = _propagate(
            _config(det_npix, oversample, ote_opd=True, **common),
            _masked_inputs(mask=mask, opd_nm=opd),
        )
        post, _ = _propagate(
            _config(det_npix, oversample, si_wfe=True, **common),
            _masked_inputs(
                mask=mask, si_opd_nm=opd[::-1, :], support=np.ones((NPUP, NPUP))
            ),
        )
        none, _ = _propagate(
            _config(det_npix, oversample, **common), _masked_inputs(mask=mask)
        )
        return np.asarray(pre.data), np.asarray(post.data), np.asarray(none.data)

    def test_same_opd_either_side_of_a_unity_mask_is_identical(self):
        """The same physical OPD (reflected into the exit frame) placed
        before or after a unity focal mask gives the same complex field."""
        pre, post, _ = self._pair(_UniformMask(1.0))
        np.testing.assert_allclose(post, pre, rtol=0, atol=FLOOR * np.abs(pre).max())

    def test_pre_and_post_mask_opd_differ_with_an_occulter(self):
        """With the occulter in, the pre-mask OPD scatters light past it and
        the post-mask OPD only reshapes what the stop passes: the images
        differ, and on a complete output grid the post-mask OPD leaves the
        transmitted energy unchanged (floor 1e-10) while the pre-mask OPD
        changes it (measured 1e-2 here; bound 1e-3)."""
        pre, post, _ = self._pair(_occulter())
        pre_i, post_i = np.abs(pre) ** 2, np.abs(post) ** 2
        assert np.abs(pre_i - post_i).max() > 0.05 * pre_i.max()
        pre, post, none = self._pair(
            _occulter(), det_npix=64, oversample=1, scale=_scale_for_pitch(1, 1.0)
        )
        e_pre, e_post, e_none = (np.sum(np.abs(f) ** 2) for f in (pre, post, none))
        assert e_post == pytest.approx(e_none, rel=1e-10)
        assert abs(e_pre - e_none) > 1e-3 * e_none


class TestRoundMaskProfile:
    def test_profile_center_clip_regions_and_shift(self):
        center = (0.4, -0.3)
        regions = ((None, None, 0.5, None, 0.0), (-0.2, 0.2, 0.4, 0.6, 0.5))
        mask = _occulter(center_arcsec=center, regions=regions)
        x = np.array([0.4, 0.4 + 0.1, 0.4 + 5.0, 0.4, 0.4, 0.4, 0.4 + 0.3])
        y = np.array(
            [-0.3, -0.3, -0.3, -0.3 + 0.55, -0.3 + 0.7, -0.3 + 0.5, -0.3 + 0.1]
        )
        got = mask.transmission(x, y)
        want = _reference_mask(x, y, center=center, regions=regions)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)
        assert got[0] == 0.0  # exactly on the (shifted) center
        assert got[2] == 1.0  # beyond the clip radius
        assert got[3] == 0.5  # later region overrides the earlier one
        assert got[4] == 0.0  # inside the half-plane region only
        assert got[5] == pytest.approx(want[5])  # boundary is open: not in region

    def test_default_clip_is_the_second_zero_of_j1(self):
        assert _occulter().clip == pytest.approx(J1_ZERO2, rel=1e-15)


# --- Reader: focal mask, Lyot stop and SI WFE from the bundle ---------------


def _write_header_array(path, data, cards, extensions=()):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=float))
    for key, value in cards.items():
        hdu.header[key] = value
    hdus = [hdu]
    for name, ext_data, ext_cards in extensions:
        ext = fits.ImageHDU(np.asarray(ext_data, dtype=float), name=name)
        for key, value in ext_cards.items():
            ext.header[key] = value
        hdus.append(ext)
    fits.HDUList(hdus).writeto(path)
    return _sha256(path)


def _pupil_cards(npix, *, content, bunit, plane):
    return {
        "CONTENT": content,
        "BUNIT": bunit,
        "PLANE": plane,
        "PIXSCALE": DIAMETER_M / npix,
        "PIXUNIT": "m/pix",
        "NPIX": npix,
        "CENTERX": (npix - 1) / 2,
        "CENTERY": (npix - 1) / 2,
        "PHASECNV": "exp(+i*2*pi*OPD/lambda)",
    }


MASK_NPIX = 64
MASK_PIXSCALE = 0.05


def _sampled_mask():
    theta = (np.arange(MASK_NPIX) - MASK_NPIX / 2) * MASK_PIXSCALE
    return _reference_mask(
        theta[np.newaxis, :], theta[:, np.newaxis], regions=BOX_REGION
    )


def _write_masked_bundle(directory, *, si_opd_m=None, sampled_mask=None):
    amp = _asymmetric_pupil(NPUP)
    _write_bundle(directory, amp, _asymmetric_opd_nm(NPUP) * NM_TO_M)
    manifest = json.loads((directory / "manifest.json").read_text())
    files = manifest["bundle_files"]
    files["lyot_stop"] = {
        "file": "lyot_stop.fits",
        "sha256": _write_header_array(
            directory / "lyot_stop.fits",
            _exit_lyot(NPUP),
            _pupil_cards(
                NPUP, content="amplitude_transmission", bunit="", plane="lyot_pupil"
            ),
        ),
    }
    si_cards = _pupil_cards(
        NPUP, content="opd", bunit="m", plane="exit_pupil_post_lyot"
    )
    si_cards["WAVELEN"] = WAVELENGTH_NM * NM_TO_M
    files["si_wfe_opd"] = {
        "file": "si_wfe_opd.fits",
        "sha256": _write_header_array(
            directory / "si_wfe_opd.fits",
            _exit_si_opd_nm(NPUP) * NM_TO_M if si_opd_m is None else si_opd_m,
            si_cards,
            extensions=(
                ("SUPPORT", _exit_support(NPUP), {"CONTENT": "amplitude_transmission"}),
            ),
        ),
    }
    mask_cards = {
        "CONTENT": "amplitude_transmission",
        "BUNIT": "",
        "PLANE": "image_plane",
        "PIXSCALE": MASK_PIXSCALE,
        "PIXUNIT": "arcsec/pix",
        "NPIX": MASK_NPIX,
        "CENTERX": MASK_NPIX / 2,
        "CENTERY": MASK_NPIX / 2,
        "SIGMA": CORE_SIGMA,
        "J1ZERO2": J1_ZERO2,
    }
    files["focal_mask"] = {
        "file": "focal_mask.fits",
        "sha256": _write_header_array(
            directory / "focal_mask.fits",
            _sampled_mask() if sampled_mask is None else sampled_mask,
            mask_cards,
        ),
        "analytic": {
            "kind": "nircamcircular",
            "sigma_per_arcsec": CORE_SIGMA,
            "j1_zero2": J1_ZERO2,
            "regions_in_application_order": [
                {
                    "name": "box",
                    "x_range": [0.1, 0.5],
                    "y_range": [0.2, None],
                    "amplitude": 0.3,
                }
            ],
            "shift_x": None,
            "shift_y": None,
            "rotation": None,
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


class TestFromBundleMasked:
    def test_reads_mask_lyot_and_si(self, tmp_path):
        inputs = NIRCamInputs.from_bundle(_write_masked_bundle(tmp_path))
        np.testing.assert_array_equal(
            np.asarray(inputs.lyot_amplitude), _exit_lyot(NPUP)
        )
        np.testing.assert_allclose(
            np.asarray(inputs.si_opd_nm), -_exit_si_opd_nm(NPUP), rtol=1e-14, atol=1e-12
        )
        np.testing.assert_array_equal(
            np.asarray(inputs.si_support), _exit_support(NPUP)
        )
        assert inputs.si_wavelength_nm == pytest.approx(WAVELENGTH_NM, rel=1e-15)
        mask = inputs.focal_mask
        assert mask.sigma_per_arcsec == CORE_SIGMA
        assert mask.clip == J1_ZERO2
        assert mask.regions == BOX_REGION
        assert mask.center_arcsec == (0.0, 0.0)
        check = inputs.provenance["focal_mask_check"]
        assert check["max_abs_diff_analytic_vs_sampled"] <= 1e-14

    def test_sampled_mask_that_disagrees_with_analytic_fails(self, tmp_path):
        sampled = _sampled_mask()
        sampled[10, 12] += 1e-6
        with pytest.raises(ValueError, match="analytic"):
            NIRCamInputs.from_bundle(
                _write_masked_bundle(tmp_path, sampled_mask=sampled)
            )

    def test_stpsf_si_ramp_moves_image_toward_minus_x_and_minus_y(self, tmp_path):
        """STPSF's SI OPD sits after the inversion: a positive STPSF OPD ramp
        of +2 samples toward +x and +3 toward +y (exit frame, meters) moves
        the image by -2 columns and -3 rows. Pins the SI sign reversal, the
        m -> nm factor and the absence of a reflection after the mask."""
        step_rad = PIXEL_SCALE_ARCSEC / 2 * ARCSEC_TO_RAD
        x_m = _pupil_coords(NPUP) * DIAMETER_M
        xx, yy = np.meshgrid(x_m, x_m)
        ramp_m = 2 * step_rad * xx + 3 * step_rad * yy
        inputs = NIRCamInputs.from_bundle(
            _write_masked_bundle(tmp_path, si_opd_m=ramp_m)
        )
        flat_inputs = eqx.tree_at(
            lambda i: i.si_opd_nm, inputs, jnp.zeros_like(inputs.si_opd_nm)
        )
        flat_inputs = eqx.tree_at(
            lambda i: i.si_support, flat_inputs, jnp.ones_like(inputs.si_support)
        )
        ones_inputs = eqx.tree_at(
            lambda i: i.si_support, inputs, jnp.ones_like(inputs.si_support)
        )
        config = _config(16, 2, si_wfe=True)
        flat = _image(config, flat_inputs)
        tilted = _image(config, ones_inputs)
        n = flat.shape[0]
        np.testing.assert_allclose(
            tilted[: n - 3, : n - 2], flat[3:, 2:], rtol=0, atol=FLOOR * flat.max()
        )
