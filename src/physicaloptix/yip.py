"""The yield-input-package (YIP) emitter: freeze a path to YIP tables.

Emits the standard coronagraph yield data package -- ``stellar_intens`` (band
images versus stellar angular diameter), ``offax_psf`` (band images versus
source offset), and ``sky_trans`` (extended-source transmission) -- in the
FITS layout the table-backed readers consume. Recipes follow the design-survey
reference implementation exactly:

- Band images are the quadrature-weighted mean of per-wavelength intensities on ONE
  fixed angular grid (reference-wavelength lambda/D).
- A stellar diameter is 50 pointings drawn area-uniformly over the stellar
  disk (radius ``sqrt(U)/2`` in diameter units, uniform azimuth, seeded per
  diameter), each applied as a fixed-angle 2D tilt, incoherently averaged.
- Sky transmission is stochastic: complex-Gaussian pupil screens at uniform
  random in-band wavelengths, the field scaled by ``lambda / lambda_ref``,
  intensities accumulated and normalized by ``npix^2 / n_screens / 2`` --
  which makes an open telescope transmit exactly one.

The native science grid samples sky angle in reference-wavelength lambda/D.
The emitter normalizes the entrance field to unit energy and returns point
densities for the star and off-axis PSFs. The FITS encoding multiplies each
density by the native pixel area. A caller requesting an explicit detector
grid from the table reader or the live adapter receives a flux integral over
each pixel, including at native pitch. Sky transmission is dimensionless.
A YIP freezes the train on disk. The live adapter keeps a calibration snapshot
for its performance and sky products. After editing the train, call
``recalibrated()`` to obtain an adapter with a refreshed snapshot.
"""

from dataclasses import replace

import numpy as np
from optixstuff.grids import Frame, Image, Measure
from yippy import write_yip
from yippy.writer import yip_header

from physicaloptix._sky import (
    _require_inside_pupil_nyquist,
    uniform_sky_image,
    unit_energy,
)
from physicaloptix.sources import _fixed_angle_camera

DEFAULT_STELLAR_DIAMETERS = np.concatenate([[0.0], np.logspace(-3, 0, 11)])


def _band_image(core_path, science, source):
    """Area-scaled point intensity on the fixed angular science grid.

    The YIP encodes each pointing as a centre sample times cell area,
    before the incoherent average over a finite star's pointings.
    """
    lyot, _ = core_path.propagate(source)
    return np.asarray(science.forward(lyot).intensity().integrated().data)


def emit_yip(
    output_dir,
    core_path,
    input_field,
    *,
    science_grid,
    spectrum,
    reference_wavelength_nm,
    offsets_x_lod,
    diameter_m,
    stellar_diameters_lod=None,
    n_pointings=50,
    n_sky_screens=1000,
    sky_band_nm=None,
    seed=0,
    inscribed_diameter_m=None,
    obscured=0.0,
    design_name="physicaloptix",
    time_multiplier_detection=1.0,
    time_multiplier_characterization=1.0,
):
    """Freeze a coronagraph path to a yield data package on disk.

    Args:
        output_dir: Directory to write the package into (created if needed).
        core_path: ``OpticalPath`` from entrance pupil to Lyot plane.
        input_field: The entrance-pupil field (normalized internally to unit
            pre-coronagraph energy).
        science_grid: The fixed angular science grid, in
            reference-wavelength lambda/D.
        spectrum: A one-band Quadrature, for example
            ``Quadrature.build((band,), core_path, order=4)``. Its dimensional
            weights define the mean; an explicit midpoint rule reproduces the
            survey sampling.
        reference_wavelength_nm: Band-center wavelength defining the grid.
        offsets_x_lod: 1D array of off-axis source offsets along +x, in
            reference lambda/D.
        diameter_m: Circumscribed telescope diameter (header metadata).
        stellar_diameters_lod: Stellar angular diameters; defaults to the survey
            list ``[0] + logspace(-3, 0, 11)``.
        n_pointings: Point sources per resolved stellar diameter.
        n_sky_screens: Stochastic screens for the sky transmission map.
        sky_band_nm: ``(min, max)`` wavelength range the sky screens draw
            from. The survey convention draws over the FULL band
            (``center * (1 +- bandwidth / 2)``), which is wider than the
            midpoint-sampled spectrum; defaults to the spectrum's range.
        seed: Seed for the (deterministic) pointing and screen draws.
        inscribed_diameter_m: Inscribed diameter (header; defaults to
            ``diameter_m``).
        obscured: Obscured aperture-area fraction (header).
        design_name: Design name (header).
        time_multiplier_detection: Header metadata.
        time_multiplier_characterization: Header metadata.

    Returns:
        Point densities, transmission and sample axes under the YIPPackage names.

    Raises:
        ValueError: The quadrature has more than one band, or the science grid
            extends beyond the pupil Nyquist radius at the shortest active node.
    """
    science_grid = replace(
        science_grid, frame=Frame(reference_wavelength_nm, diameter_m, "circumscribed")
    )

    if stellar_diameters_lod is None:
        stellar_diameters_lod = DEFAULT_STELLAR_DIAMETERS
    stellar_diameters_lod = np.asarray(stellar_diameters_lod, dtype=float)
    offsets_x_lod = np.asarray(offsets_x_lod, dtype=float)

    if len(spectrum.bands) != 1:
        raise ValueError("a yield package requires a one-band quadrature")
    active = np.asarray(spectrum.active[0])
    nodes_nm = np.asarray(spectrum.nodes_nm[0])[active]
    base = unit_energy(input_field)
    low_nm = float(nodes_nm.min())
    _require_inside_pupil_nyquist(science_grid, base, low_nm=low_nm)
    science, place = _fixed_angle_camera(
        base,
        science_grid,
        spectrum=spectrum,
        reference_wavelength_nm=reference_wavelength_nm,
        low_nm=low_nm,
    )

    # -- off-axis PSFs ------------------------------------------------------
    offax = np.stack(
        [
            _band_image(core_path, science, place((offset, 0.0)))
            for offset in offsets_x_lod
        ]
    )

    # -- stellar intensity vs angular diameter ------------------------------
    stellar = []
    for diameter in stellar_diameters_lod:
        if diameter == 0.0:
            stellar.append(_band_image(core_path, science, place((0.0, 0.0))))
            continue
        rng = np.random.default_rng(seed)  # per-diameter, as the survey does
        radius = np.sqrt(rng.uniform(0, 1, n_pointings)) / 2
        theta = rng.uniform(0, 2 * np.pi, n_pointings)
        image = 0.0
        for r, t in zip(radius, theta, strict=True):
            position = (
                float(r * np.cos(t) * diameter),
                float(r * np.sin(t) * diameter),
            )
            image = image + _band_image(core_path, science, place(position))
        stellar.append(image / n_pointings)
    stellar = np.stack(stellar)

    # -- stochastic sky transmission ----------------------------------------
    band = spectrum.bands[0]
    science_band_nm = (band.low_nm, band.high_nm)
    sky_band_nm = science_band_nm if sky_band_nm is None else sky_band_nm
    sky = uniform_sky_image(
        core_path,
        science,
        base,
        n_screens=n_sky_screens,
        seed=seed,
        sky_band_nm=sky_band_nm,
    )

    # -- write the package ---------------------------------------------------
    products = {
        "stellar_intensity": Image(
            stellar,
            science_grid,
            ("stellar_diameter",),
            Measure.integral(),
            "stellar_flux_fraction",
            reconstruction="band_limited",
        ).density("grid"),
        "stellar_diameters_lod": stellar_diameters_lod,
        "offaxis_psfs": Image(
            offax,
            science_grid,
            ("source",),
            Measure.integral(),
            "stellar_flux_fraction",
            reconstruction="band_limited",
        ).density("grid"),
        "offsets_xy_lod": np.column_stack(
            (offsets_x_lod, np.zeros_like(offsets_x_lod))
        ),
        "sky_transmission": sky,
    }

    def product_header(band_nm, n_wavelengths):
        return yip_header(
            science_grid,
            design=design_name,
            inscribed_diameter_m=diameter_m
            if inscribed_diameter_m is None
            else inscribed_diameter_m,
            low_nm=band_nm[0],
            high_nm=band_nm[1],
            obscured=obscured,
            jitter_mas=0.0,
            n_wavelengths=n_wavelengths,
            n_pointings=n_pointings,
            time_multiplier_detection=time_multiplier_detection,
            time_multiplier_characterization=time_multiplier_characterization,
        )

    headers = dict.fromkeys(products, product_header(science_band_nm, nodes_nm.size))
    headers["sky_transmission"] = product_header(sky_band_nm, n_sky_screens)
    # Keep the original area-scaled samples on disk. Re-encoding the native
    # density after averaging resolved-star pointings can round a second time.
    write_yip(
        output_dir,
        {
            **products,
            "stellar_intensity": stellar,
            "offaxis_psfs": offax,
            "headers": headers,
            "sample_kind": "point",
        },
    )
    return products
