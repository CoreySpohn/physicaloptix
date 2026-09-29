"""Cross-code benchmark: the NIRCam builder against STPSF 2.2.0 references.

Reference-required. The prescription bundle and the STPSF references are
found through ``PHYSICALOPTIX_NIRCAM_BUNDLE`` (the bundle directory) and
``PHYSICALOPTIX_NIRCAM_REFERENCE`` (the monochromatic ``<case>.fits`` files
and ``band/manifest.json`` with the F335M band references). With
``PHYSICALOPTIX_REQUIRE_REFERENCES=1`` an absent bundle or reference is a
failure and so is a run that executed no case; otherwise it skips with a
reason. Every reference array is checked against its manifest hash first.

Truth source: STPSF 2.2.0 (``cross-code-benchmark``). physicaloptix reads only
the exported prescription and never calls STPSF. Products compared: the ideal
optical OVERSAMP image and the matched DET_SAMP (pixel-integrated once by a
4 x 4 block sum in both codes), normalized once to unit entrance-pupil energy.
``U`` is the STPSF unocculted peak of the same product and wavelength set.

Tolerances (basis: ``spike`` from refinement, not from the cross-code
residual; PROPOSED, not frozen): the no-mask cases share the pupil raster,
the output samples and the finite sums, so they must agree to a
floating-point floor (1e-12). The masked cases differ only in the mask-plane
sampling; their bounds are the physicaloptix mask-pitch envelope plus the
reference's own mask-plane sampling envelope, measured by refining each
setting, for c-mask335r at (0, 0) and (+1.0, +0.5) and c-mask335r-ote-si at
(0, 0) (monochromatic 3350 nm) and for the 3-node F335M band of the first two.
The remaining masked cases take the bound of the refined case with the same
leakage source (the 0.01 arcsec pointing cases and the SI-only case take the
centered bound, the OTE-only case the OTE + SI bound): coverage by analogy.
DET_SAMP bounds follow from ``max|d_det| <= 16 max|d_over|``.

The masked checks are regression gates whose containment holds by
construction, not independent evidence of agreement: the cross-code residual
is the same mask-plane sampling difference the bound is built from, so each
bound contains it to first order (the triangle inequality on the two
envelopes). The tightest case, c-mask335r at (+1.0, +0.5) at 3350 nm
(``peak_rel``, measured at 0.95 of its bound), is that first-order
containment, not a near failure; it is also the case most sensitive to any
numerical drift.

Negative controls: the entrance pupil flipped top to bottom, the source
displaced by half an OVERSAMP sample and the mask attenuation applied twice
(amplitude transmission squared) must each exceed a bound. The reserved band
case (OTE + SI WFE through the band) is never compared here.
"""

import dataclasses
import hashlib
import json
import math

import equinox as eqx
import numpy as np
import pytest

from physicaloptix.instruments import (
    NIRCamBand,
    NIRCamConfig,
    NIRCamInputs,
    build_nircam,
    integrate_detector_pixels,
    nircam_band_image,
)

fits = pytest.importorskip("astropy.io.fits")

NO_MASK = {"max_abs_diff_over_U": 1e-12, "total_rel": 1e-12, "peak_rel": 1e-12}
CENTERED = {"max_abs_diff_over_U": 3e-9, "total_rel": 5e-4, "peak_rel": 6e-4}
OFF_AXIS = {"max_abs_diff_over_U": 2e-7, "total_rel": 2e-6, "peak_rel": 3e-6}
OTE_SI = {"max_abs_diff_over_U": 6e-9, "total_rel": 2e-5, "peak_rel": 2e-4}
BAND_CENTERED = {"max_abs_diff_over_U": 2e-9, "total_rel": 3e-4, "peak_rel": 4e-4}
BAND_OFF_AXIS = {"max_abs_diff_over_U": 3e-7, "total_rel": 4e-6, "peak_rel": 7e-6}

CORON = {"focal_mask": True, "lyot_stop": True}
STAGES = {
    "a-pupil": {},
    "b-pupil-ote": {"ote_opd": True},
    "c-mask335r": CORON,
    "c-mask335r-ote": {**CORON, "ote_opd": True},
    "c-mask335r-si": {**CORON, "si_wfe": True},
    "c-mask335r-ote-si": {**CORON, "ote_opd": True, "si_wfe": True},
}
MONO_CASES = [
    ("a-pupil", (0.0, 0.0), NO_MASK),
    ("a-pupil", (1.0, 0.5), NO_MASK),
    ("a-pupil", (0.01, 0.0), NO_MASK),
    ("a-pupil", (0.0, 0.01), NO_MASK),
    ("b-pupil-ote", (0.0, 0.0), NO_MASK),
    ("c-mask335r", (0.0, 0.0), CENTERED),
    ("c-mask335r", (1.0, 0.5), OFF_AXIS),
    ("c-mask335r", (0.01, 0.0), CENTERED),
    ("c-mask335r", (0.0, 0.01), CENTERED),
    ("c-mask335r-ote", (0.0, 0.0), OTE_SI),
    ("c-mask335r-si", (0.0, 0.0), CENTERED),
    ("c-mask335r-ote-si", (0.0, 0.0), OTE_SI),
]
BAND_CASES = [
    ("a-pupil", (0.0, 0.0), NO_MASK),
    ("c-mask335r", (0.0, 0.0), BAND_CENTERED),
    ("c-mask335r", (1.0, 0.5), BAND_OFF_AXIS),
]
BAND_NODES = 9


def _case_id(stage, source):
    return f"{stage}_x{source[0]:+.3f}_y{source[1]:+.3f}"


def _sha256_array(arr):
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _read_checked(path, expected_hash):
    over = np.asarray(fits.getdata(path, extname="OVERSAMP"), dtype=np.float64)
    det = np.asarray(fits.getdata(path, extname="DET_SAMP"), dtype=np.float64)
    found = _sha256_array(over) + ":" + _sha256_array(det)
    assert found == expected_hash, (
        f"{path}: reference arrays do not match their manifest"
    )
    return over, det


@pytest.fixture(scope="session")
def bench(nircam_bundle_dir, nircam_reference_dir):
    manifest = json.loads((nircam_bundle_dir / "manifest.json").read_text())
    band_manifest = json.loads(
        (nircam_reference_dir / "band" / "manifest.json").read_text()
    )
    return {
        "manifest": manifest,
        "band_manifest": band_manifest,
        "inputs": NIRCamInputs.from_bundle(nircam_bundle_dir),
        "band": NIRCamBand.from_bundle(nircam_bundle_dir, BAND_NODES),
        "reference_dir": nircam_reference_dir,
    }


def _config(bench, stage, source, **overrides):
    cfg = bench["manifest"]["configuration"]
    return NIRCamConfig(
        wavelength_nm=cfg["wavelength_nm"],
        pixel_scale_arcsec=cfg["pixelscale_arcsec"],
        detector_npix=cfg["fov_pixels"],
        oversample=cfg["detector_oversample"],
        source_position_arcsec=source,
        **{**STAGES[stage], **overrides},
    )


def _mono_image(config, inputs):
    path, field = build_nircam(config, inputs)
    out, _ = path.propagate(field)
    return np.asarray(out.intensity()) * out.grid.weights


def _mono_reference(bench, stage, source):
    case_id = _case_id(stage, source)
    entry = bench["manifest"]["reference_psfs"].get(case_id)
    assert entry is not None, f"bundle manifest lists no reference {case_id}"
    return _read_checked(
        bench["reference_dir"] / f"{case_id}.fits", entry["array_sha256"]
    )


def _band_reference(bench, stage, source):
    case_id = f"{_case_id(stage, source)}_f335m_n{BAND_NODES}"
    entry = bench["band_manifest"]["cases"].get(case_id)
    assert entry is not None, f"band manifest lists no reference {case_id}"
    assert not entry["reserved"], f"{case_id} is reserved and must not be scored"
    return _read_checked(
        bench["reference_dir"] / "band" / entry["path"], entry["array_sha256"]
    )


def _metric(po, ref, unocculted_peak):
    return {
        "max_abs_diff_over_U": float(np.abs(po - ref).max() / unocculted_peak),
        "total_rel": float(abs(po.sum() / ref.sum() - 1.0)),
        "peak_rel": float(abs(po.max() / ref.max() - 1.0)),
    }


def _det_bound(bound, u_over, u_det):
    scaled = 16 * bound["max_abs_diff_over_U"] * u_over / u_det
    exponent = math.floor(math.log10(scaled))
    return {
        "max_abs_diff_over_U": math.ceil(scaled / 10**exponent - 1e-9) * 10**exponent,
        "total_rel": bound["total_rel"],
    }


def _ratios(metric, bound):
    return {k: metric[k] / bound[k] for k in bound}


def _check(case_id, po, ref, u_over, det_po, det_ref, u_det, bound, record_case):
    over = _metric(po, ref, u_over)
    det = _metric(det_po, det_ref, u_det)
    det_bound = _det_bound(bound, u_over, u_det)
    ratio_over = _ratios(over, bound)
    ratio_det = _ratios(det, det_bound)
    worst = max(*ratio_over.values(), *ratio_det.values())
    record_case(
        case_id,
        f"OVERSAMP max|d|/U {over['max_abs_diff_over_U']:.1e} total "
        f"{over['total_rel']:.1e}; DET_SAMP max|d|/U {det['max_abs_diff_over_U']:.1e}; "
        f"largest ratio to proposed bound {worst:.2f}",
    )
    for name, ratio in (("OVERSAMP", ratio_over), ("DET_SAMP", ratio_det)):
        assert all(r <= 1.0 for r in ratio.values()), f"{case_id} {name}: {ratio}"


@pytest.mark.parametrize(
    ("stage", "source", "bound"),
    MONO_CASES,
    ids=[_case_id(s, p) for s, p, _ in MONO_CASES],
)
def test_monochromatic_case(bench, stage, source, bound, record_case):
    ref_over, ref_det = _mono_reference(bench, stage, source)
    u_over, u_det = (
        float(a.max()) for a in _mono_reference(bench, "a-pupil", (0.0, 0.0))
    )
    config = _config(bench, stage, source)
    po = _mono_image(config, bench["inputs"])
    det = np.asarray(integrate_detector_pixels(po, config.oversample))
    _check(
        f"{_case_id(stage, source)} (3350 nm)",
        po,
        ref_over,
        u_over,
        det,
        ref_det,
        u_det,
        bound,
        record_case,
    )


@pytest.mark.parametrize(
    ("stage", "source", "bound"),
    BAND_CASES,
    ids=[_case_id(s, p) for s, p, _ in BAND_CASES],
)
def test_f335m_band_case(bench, stage, source, bound, record_case):
    ref_over, ref_det = _band_reference(bench, stage, source)
    u_over, u_det = (
        float(a.max()) for a in _band_reference(bench, "a-pupil", (0.0, 0.0))
    )
    config = _config(bench, stage, source)
    po = np.asarray(nircam_band_image(config, bench["inputs"], bench["band"]))
    det = np.asarray(integrate_detector_pixels(po, config.oversample))
    _check(
        f"{_case_id(stage, source)} (F335M, {BAND_NODES} nodes)",
        po,
        ref_over,
        u_over,
        det,
        ref_det,
        u_det,
        bound,
        record_case,
    )


class _SquaredMask(eqx.Module):
    mask: eqx.Module

    def transmission(self, x_arcsec, y_arcsec):
        return self.mask.transmission(x_arcsec, y_arcsec) ** 2


def _flipped(inputs):
    return dataclasses.replace(inputs, pupil_amplitude=inputs.pupil_amplitude[::-1, :])


def _squared(inputs):
    return dataclasses.replace(inputs, focal_mask=_SquaredMask(inputs.focal_mask))


CONTROLS = [
    ("a-pupil", "pupil flipped top-bottom", _flipped, (0.0, 0.0), NO_MASK),
    ("a-pupil", "source displaced half a sample", None, (0.5, 0.0), NO_MASK),
    ("c-mask335r", "pupil flipped top-bottom", _flipped, (0.0, 0.0), CENTERED),
    ("c-mask335r", "source displaced half a sample", None, (0.5, 0.0), CENTERED),
    ("c-mask335r", "mask attenuation applied twice", _squared, (0.0, 0.0), CENTERED),
]


@pytest.mark.parametrize(
    ("stage", "name", "edit", "shift_samples", "bound"),
    CONTROLS,
    ids=[f"{s}: {n}" for s, n, *_ in CONTROLS],
)
def test_negative_control_is_detected(
    bench, stage, name, edit, shift_samples, bound, record_case
):
    """A deliberately wrong input must exceed the proposed bound on the
    correct reference (the metric is not blind to it)."""
    ref_over, _ = _mono_reference(bench, stage, (0.0, 0.0))
    u_over = float(_mono_reference(bench, "a-pupil", (0.0, 0.0))[0].max())
    sample = bench["manifest"]["configuration"]["pixelscale_arcsec"] / 4
    source = (shift_samples[0] * sample, shift_samples[1] * sample)
    inputs = bench["inputs"] if edit is None else edit(bench["inputs"])
    po = _mono_image(_config(bench, stage, source), inputs)
    ratios = _ratios(_metric(po, ref_over, u_over), bound)
    margin = max(ratios.values())
    worst = max(ratios, key=ratios.get)
    record_case(
        f"control {stage}: {name}",
        f"detected at {margin:.2e} x the proposed bound ({worst})",
    )
    assert margin > 1.0, f"{stage}: {name} is not detected: {ratios}"
