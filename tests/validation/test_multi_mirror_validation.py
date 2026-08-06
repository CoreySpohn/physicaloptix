"""Whole-train physics gates for the multi-mirror Fresnel chain.

Anchors: the analytic Talbot conversion law sin(pi alpha nu^2); the
invariance of RAW speckle power under quadrature redistribution; the
post-correction floor an achromatic phase conjugation cannot remove from
out-of-pupil errors (while removing a lumped pupil error exactly).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from physicaloptix import (
    Field,
    Fraunhofer,
    Grid,
    PlaneKind,
    Spectrum,
    broadcast_to_spectrum,
)
from physicaloptix.trains import (
    REFLECTION_OPD_FACTOR,
    MirrorSpec,
    build_mirror_train,
    synthesize_psd_surface,
)

NPIX, WL0, NLAM = 128, 500.0, 3
NFOC, PSCALE, IWA, OWA = 96, 0.28, 4.0, 12.0
ALPHAS = {"m_near": 6.9e-5, "m_mid": 4.0e-4, "m_far": 9.4e-4}


@pytest.fixture(scope="module")
def setup():
    grid = Grid.pupil(NPIX)
    coords = grid.coords
    xx, yy = np.meshgrid(coords, coords)
    aperture = (np.hypot(xx, yy) <= 0.5).astype(complex)
    surfaces = {
        name: synthesize_psd_surface(seed, grid, rms_nm=1.0, k_max=40)
        for seed, name in enumerate(["m_pupil", *ALPHAS], start=30)
    }
    specs = tuple(
        [MirrorSpec(name="m_pupil", alpha=0.0, surface_nm=surfaces["m_pupil"])]
        + [
            MirrorSpec(name=n, alpha=a, surface_nm=surfaces[n])
            for n, a in ALPHAS.items()
        ]
    )
    spectrum = Spectrum.tophat(WL0, 0.2, NLAM)
    field = broadcast_to_spectrum(
        Field(data=jnp.asarray(aperture), grid=grid, plane=PlaneKind.PUPIL), spectrum
    )
    honest = build_mirror_train(specs, grid, wavelength_nm=WL0, beam_diameter_m=0.085)
    lumped_opd = REFLECTION_OPD_FACTOR * sum(surfaces.values())
    aperture_mask = np.asarray(aperture) != 0
    return grid, aperture, field, spectrum, honest, aperture_mask, lumped_opd, surfaces


def _focal_contrast(grid, field, e_out, mask):
    focal = Grid.focal(NFOC, PSCALE)
    prop = Fraunhofer(grid_in=grid, grid_out=focal)
    peak = np.abs(np.asarray(prop(field).data)) ** 2
    peak = peak.max(axis=(1, 2))
    delta = Field(
        data=e_out.data - field.data,
        grid=grid,
        plane=PlaneKind.PUPIL,
        spectrum=field.spectrum,
    )
    df = np.abs(np.asarray(prop(delta).data)) ** 2 / peak[:, None, None]
    return df[:, mask].mean(axis=1)


def _annulus(nfoc, pscale):
    fx = (np.arange(nfoc) - nfoc / 2 + 0.5) * pscale
    r = np.hypot(*np.meshgrid(fx, fx))
    return (r >= IWA) & (r <= OWA)


class TestMultiMirrorPhysics:
    def test_talbot_conversion_law_through_the_train(self, setup):
        # A uniform field is a Fresnel eigenfunction, so the inbound hop
        # leaves it unchanged; the ripple applied AT the mirror converts on
        # the RETURN hop (-alpha), arriving at the exit pupil with amplitude
        # fraction |sin(pi alpha nu^2)| -- measure there, on the full train.
        grid, aperture, *_ = setup
        alpha, nu = 4.0e-4, 20.0
        coords = grid.coords
        xx = np.meshgrid(coords, coords)[0]
        ripple = 0.5 * np.cos(2 * np.pi * nu * xx)  # nm surface, no aperture
        spec = (MirrorSpec(name="m", alpha=alpha, surface_nm=ripple),)
        path = build_mirror_train(spec, grid, wavelength_nm=WL0, beam_diameter_m=0.085)
        ones = Field(
            data=jnp.ones_like(jnp.asarray(aperture)), grid=grid, plane=PlaneKind.PUPIL
        )
        out, _ = path.propagate(ones)
        amp = np.abs(np.asarray(out.data)) - 1.0
        eps = 2 * np.pi * REFLECTION_OPD_FACTOR * 0.5 / WL0
        measured = amp.std() * np.sqrt(2) / eps
        expected = abs(np.sin(np.pi * alpha * nu**2))
        assert measured == pytest.approx(expected, rel=0.05)
        # phase quadrature carries the complementary cos fraction
        phase = np.angle(np.asarray(out.data))
        measured_cos = phase.std() * np.sqrt(2) / eps
        assert measured_cos == pytest.approx(
            abs(np.cos(np.pi * alpha * nu**2)), rel=0.05
        )

    def test_raw_contrast_matches_lumped(self, setup):
        grid, _aperture, field, spectrum, honest, _ap, lumped_opd, _ = setup
        mask = _annulus(NFOC, PSCALE)
        e_h, _ = honest.propagate(field)
        wl = np.asarray(spectrum.wavelengths_nm)[:, None, None]
        e_l = Field(
            data=field.data * jnp.exp(1j * 2 * jnp.pi * jnp.asarray(lumped_opd) / wl),
            grid=grid,
            plane=PlaneKind.PUPIL,
            spectrum=spectrum,
        )
        c_h = _focal_contrast(grid, field, e_h, mask)
        c_l = _focal_contrast(grid, field, e_l, mask)
        np.testing.assert_allclose(c_h, c_l, rtol=0.02)

    def test_achromatic_phase_conjugation_floor(self, setup):
        grid, _aperture, field, spectrum, honest, ap, lumped_opd, _ = setup
        mask = _annulus(NFOC, PSCALE)
        i0 = NLAM // 2
        wl = np.asarray(spectrum.wavelengths_nm)

        def corrected(e_out):
            ratio0 = np.asarray(e_out.data[i0]) / np.where(
                ap, np.asarray(field.data[i0]), 1.0
            )
            opd = np.where(ap, np.angle(ratio0), 0.0) * wl[i0] / (2 * np.pi)
            data = e_out.data * jnp.exp(
                -1j * 2 * jnp.pi * jnp.asarray(opd) / wl[:, None, None]
            )
            return Field(data=data, grid=grid, plane=PlaneKind.PUPIL, spectrum=spectrum)

        e_h, _ = honest.propagate(field)
        e_l = Field(
            data=field.data
            * jnp.exp(1j * 2 * jnp.pi * jnp.asarray(lumped_opd) / wl[:, None, None]),
            grid=grid,
            plane=PlaneKind.PUPIL,
            spectrum=spectrum,
        )
        raw = _focal_contrast(grid, field, e_h, mask)
        floor_h = _focal_contrast(grid, field, corrected(e_h), mask)
        floor_l = _focal_contrast(grid, field, corrected(e_l), mask)
        # lumped: exactly correctable at every wavelength
        assert floor_l.max() < 1e-25
        # honest: a floor survives, a few percent of raw, at every wavelength
        assert 0.005 < float((floor_h / raw).min())
        assert float((floor_h / raw).max()) < 0.10
