"""Speckle-statistics shapes: the process, the ensemble, the field ellipse.

Three domain pictures over machinery that already exists. ``physicaloptix.
speckle`` owns the process and its closed forms; ``physicaloptix.stats``
owns the delta-referenced per-pixel laws; eyepiece owns the drawing
primitives. What is added here is the domain vocabulary and, above all, the
UNITS CONTRACT -- which is the whole reason these functions exist.

The units trap, stated once because every function below is shaped by it:
:meth:`AnalyticSpeckleField.realize` returns a per-pixel FLUX-FRACTION
DELTA, the wavefront-error excess over the deterministic floor. The promoted
``pixel_pdf``/``pixel_sf`` laws are delta-referenced to match it (total
intensity ``I = i_c + delta * normalization``). The legacy
``modified_rician_pdf``, by contrast, is a RAW-INTENSITY law. Plotting a
histogram of one against a density of the other is the exact failure that
makes independent hand-rolled versions of this figure disagree, and it is
both curves are smooth, both look plausible, and only their integrals
disagree. ``plot_speckle_ensemble`` therefore converts explicitly and its
tests gate on the integral, not on appearance.

Note this module is ``physicaloptix.viz.speckle``, distinct from the
top-level ``physicaloptix.speckle`` it draws. The dependency runs one way:
the viz module imports the library module, never the reverse.
"""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch

from physicaloptix.stats import modified_rician_pdf, pixel_pdf
from physicaloptix.viz import _require

# Lag axis for the autocorrelation panel, in units of the mode's own
# decorrelation time tau = 1 / (2 pi knee). Four tau shows the decay into the
# floor while staying inside the lags the synthesis grid actually spans: the
# lowest spectral line sits about 50x below the knee at the default
# decades_below=1.7, and lags approaching 1 / (2 pi f_min) ring off the end of
# the grid rather than decaying.
_LAG_TAUS = 4.0
_LAG_POINTS = 200
_PSD_MODE = 0


def _validate_axes(axes, expected, name):
    """Return ``axes`` as an array of the expected shape, or raise.

    Raises:
        ValueError: ``axes`` does not have shape ``expected``, naming both
            the expected and the received shape.
    """
    arr = np.asarray(axes, dtype=object)
    if arr.shape != expected:
        msg = f"{name}: expected axes shape {expected}, got {arr.shape}"
        raise ValueError(msg)
    return arr


def plot_process(process, *, axes=None, line_kw=None):
    """Draw a ``SpeckleProcess``'s spectrum and its realized temporal kernel.

    Two panels, both making the same point in different domains: what the
    process was ASKED for versus what the spectral synthesis actually BUILT.

    Panel 0 (loglog) is the spectrum. Both curves are normalized to unit sum
    over the synthesis frequency grid, which is what makes them comparable at
    all: ``psd`` is a spectral DENSITY ``S(f)`` while ``line_weights`` carries
    the quadrature element ``S(f_j) df_j``, so the two have different units
    and only their shares of the total are the same kind of quantity. Read as
    shares, the panel shows where the process's variance actually sits. This
    is the panel that exposes the classic synthesis bug: weighting lines by the
    bare ordinate on a LOG grid synthesizes ``S(f) / f``, handing the lowest
    frequencies far more power than their slice of spectrum deserves, and
    slow power is exactly what stretches decorrelation.

    Panel 1 is the temporal kernel: the realized ``autocorrelation`` against
    the Ornstein-Uhlenbeck closed form ``exp(-2 pi knee tau)`` that a
    slope -2 (Lorentzian) PSD implies. Verifying the kernel as built, rather
    than assuming it, is the habit that caught that bug; the OU reference is
    exact only for slope -2, and is drawn for any slope precisely so a
    process that does NOT match its named target says so on the page.

    Line-weight hierarchy: the target (the continuous PSD, the OU kernel) is
    heavy, the realized quantity (the line weights, the synthesized
    autocorrelation) is dashed.

    Only public ``SpeckleProcess`` surface is read: ``frequencies_hz()``,
    ``psd()``, ``line_weights()``, ``autocorrelation()``, and the ``knee_hz``
    parameter.

    Args:
        process: A ``physicaloptix.speckle.SpeckleProcess``.
        axes: A length-2 sequence of Axes, ``[spectrum, autocorrelation]``.
            None creates a new 1x2 figure.
        line_kw: Extra kwargs applied last to every ``plot`` call in both
            panels.

    Returns:
        A ``MosaicResult`` with ``axes`` shape ``(2,)``.
        ``artists["psd"]``/``artists["weights"]`` are panel 0's target and
        realized ``Line2D``; ``artists["target"]``/``artists["realized"]``
        are panel 1's.

    Raises:
        ValueError: ``axes`` is given with a shape other than ``(2,)``.
    """
    ep = _require.eyepiece()
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), layout="constrained")
    axes = _validate_axes(axes, (2,), "plot_process")
    spectrum_ax, acf_ax = axes[0], axes[1]

    # Resolved at call time, never bound at import: the active hwostyle mode
    # is a property of the moment the figure is drawn.
    model_color = hwostyle.roles.model
    measured_color = hwostyle.roles.measured
    kw = dict(line_kw or {})

    frequencies = np.asarray(process.frequencies_hz())
    if frequencies.ndim > 1:
        frequencies = frequencies[_PSD_MODE]
    psd = np.asarray(process.psd(process.frequencies_hz()))
    if psd.ndim > 1:
        psd = psd[_PSD_MODE]
    weights = np.asarray(process.line_weights())[_PSD_MODE]

    (psd_line,) = spectrum_ax.loglog(
        frequencies,
        psd / psd.sum(),
        color=model_color,
        lw=2.0,
        label="PSD share, $S(f_j)$",
        **kw,
    )
    (weight_line,) = spectrum_ax.loglog(
        frequencies,
        weights / weights.sum(),
        color=measured_color,
        ls="--",
        lw=1.4,
        marker="o",
        ms=3,
        label="line weights as built",
        **kw,
    )
    spectrum_ax.set_xlabel("frequency [Hz]")
    spectrum_ax.set_ylabel("share of total power")

    knee = float(np.asarray(process.knee_hz)[_PSD_MODE])
    tau_s = 1.0 / (2.0 * np.pi * knee)
    lags = np.linspace(0.0, _LAG_TAUS * tau_s, _LAG_POINTS)
    realized = np.asarray(process.autocorrelation(lags))[_PSD_MODE]
    ou_target = np.exp(-2.0 * np.pi * knee * lags)

    (target_line,) = acf_ax.plot(
        lags / tau_s,
        ou_target,
        color=model_color,
        lw=2.0,
        label=r"OU target, $e^{-2\pi f_{knee}\tau}$",
        **kw,
    )
    (realized_line,) = acf_ax.plot(
        lags / tau_s,
        realized,
        color=measured_color,
        ls="--",
        lw=1.4,
        label="`autocorrelation`, as built",
        **kw,
    )
    acf_ax.set_xlabel(r"lag $\tau$ [$1 / 2\pi f_{knee}$]")
    acf_ax.set_ylabel("autocorrelation")

    artists = {
        "psd": psd_line,
        "weights": weight_line,
        "target": target_line,
        "realized": realized_line,
    }
    return ep.MosaicResult(axes=axes, artists=artists)


def _exact_delta_pdf(x_grid, params):
    """The delta-referenced exact per-pixel law, in delta units already.

    ``stats.pixel_pdf`` carries the ``norm *`` change-of-variables Jacobian
    itself, so this is a straight pass-through -- the point of the
    delta-referenced form.
    """
    return pixel_pdf(
        x_grid,
        params["i_c"],
        params["gamma"],
        params["p"],
        params["phi_c"],
        params["normalization"],
    )


def _rician_delta_pdf(x_grid, params):
    """The modified Rician, CONVERTED from raw intensity into delta units.

    This is the units trap in one function. ``stats.modified_rician_pdf`` is
    a law over the total intensity ``I``, not over the contrast delta the
    histogram is binned in. Overlaying it needs BOTH halves of the change of
    variables ``I = i_c + x * norm``:

    - the ARGUMENT must be the total intensity ``i_c + x * norm``, not ``x``;
    - the VALUE must be multiplied by the Jacobian ``dI/dx = norm``.

    Dropping either one leaves a curve that is still smooth, still
    bell-shaped, and still plausible next to the histogram -- and wrong by a
    factor of ``norm``, which for a real coronagraph is many orders of
    magnitude. Only the integral gives it away, which is why the units test
    gates on the integral rather than on appearance.
    """
    intensity = params["i_c"] + x_grid * params["normalization"]
    density = np.asarray(modified_rician_pdf(intensity, params["i_c"], params["gamma"]))
    return density * params["normalization"]


# Each named law carries its role in the line-weight hierarchy: the exact law
# is heavy, the limiting law it converges to is dashed and thinner.
_LAWS = {
    "exact": (_exact_delta_pdf, {"lw": 2.0, "ls": "-"}),
    "rician": (_rician_delta_pdf, {"lw": 1.4, "ls": "--"}),
}
_CALLABLE_STYLE = {"lw": 2.0, "ls": "-"}
_LAW_POINTS = 400


def _resolve_from_process(process, pixel):
    """Per-pixel law parameters read off a ``SpeckleProcess``."""
    if pixel is None:
        msg = "plot_speckle_ensemble: process= needs pixel=(iy, ix) to pick a pixel"
        raise ValueError(msg)
    moments = process.moments()
    e_nom = np.asarray(process.e_nom)[tuple(pixel)]
    return {
        "i_c": float(abs(e_nom) ** 2),
        "phi_c": float(np.angle(e_nom)),
        "gamma": float(np.asarray(moments.gamma_map)[tuple(pixel)]),
        "p": complex(np.asarray(moments.p_map)[tuple(pixel)]),
        "normalization": float(np.asarray(process.normalization)),
    }


def _resolve_from_moments(moments, pixel):
    """The two parameters a ``SpeckleMoments`` can supply on its own.

    ``i_c`` and ``phi_c`` describe the DETERMINISTIC field ``E_nom``, which
    ``SpeckleMoments`` does not carry, so they stay the caller's to supply
    explicitly, rather than being papered over with a guess.
    """
    if pixel is None:
        msg = "plot_speckle_ensemble: moments= needs pixel=(iy, ix) to pick a pixel"
        raise ValueError(msg)
    return {
        "gamma": float(np.asarray(moments.gamma_map)[tuple(pixel)]),
        "p": complex(np.asarray(moments.p_map)[tuple(pixel)]),
    }


def _resolve_params(explicit, process, moments, pixel):
    """Merge the three parameter sources in their pinned precedence.

    Explicit scalars win, then ``process=`` (+ ``pixel=``), then ``moments=``
    (+ ``pixel=``). Precedence is applied PER PARAMETER, so naming a single
    ``i_c`` explicitly overrides just that one and leaves the rest coming
    from the process.

    Raises:
        ValueError: A required parameter is supplied by no source.
    """
    resolved = {}
    if moments is not None:
        resolved.update(_resolve_from_moments(moments, pixel))
    if process is not None:
        resolved.update(_resolve_from_process(process, pixel))
    resolved.update({k: v for k, v in explicit.items() if v is not None})

    missing = [
        k for k in ("i_c", "gamma", "p", "phi_c", "normalization") if k not in resolved
    ]
    if missing:
        msg = (
            f"plot_speckle_ensemble: no source for {missing}. Supply them "
            "explicitly, or pass process=/moments= with pixel=. With draws "
            "alone, moment-match the Rician first: for m = mean(I) and "
            "v = var(I) in INTENSITY units, i_s = m - sqrt(m^2 - v) and "
            "i_c = sqrt(m^2 - v), then pass gamma=i_s."
        )
        raise ValueError(msg)
    return resolved


def _law_entries(laws):
    """Normalize ``laws`` into a list of ``(label, function, style)``."""
    if isinstance(laws, str) or callable(laws):
        laws = [laws]
    entries = []
    for law in laws:
        if callable(law):
            entries.append((getattr(law, "__name__", "law"), law, _CALLABLE_STYLE))
            continue
        if law not in _LAWS:
            msg = f"unknown law: {law!r}; use {sorted(_LAWS)}, or pass a callable"
            raise ValueError(msg)
        function, style = _LAWS[law]
        entries.append((law, function, style))
    return entries


def plot_speckle_ensemble(
    draws,
    *,
    laws="rician",
    i_c=None,
    gamma=None,
    p=None,
    phi_c=None,
    normalization=None,
    moments=None,
    process=None,
    pixel=None,
    ax=None,
    log=True,
    hist_kw=None,
):
    """Histogram an ensemble of contrast deltas against one or more laws.

    The flagship comparison: the exact delta-referenced per-pixel law and the
    modified Rician it reduces to, overlaid on one panel over the SAME
    histogram, in the SAME units.

    **Units.** ``draws`` are per-pixel flux-fraction contrast deltas -- what
    ``AnalyticSpeckleField.realize()`` returns -- and the abscissa is in
    those units throughout. ``stats.pixel_pdf`` is already delta-referenced
    and passes through; ``stats.modified_rician_pdf`` is a RAW-INTENSITY law
    and is converted here, argument and Jacobian both (see
    ``_rician_delta_pdf``). Hand-rolled versions of this figure disagree
    because that conversion is done inconsistently or not at all, and the
    error is invisible by eye. This function exists to make it once, in one
    place, correctly.

    **Parameter precedence**, applied per parameter: explicit scalars beat
    ``process=`` (+ ``pixel=``), which beats ``moments=`` (+ ``pixel=``).
    ``SpeckleMoments`` carries ``gamma`` and ``p`` but not ``i_c``/``phi_c``
    -- those describe ``E_nom``, which it does not hold -- so with
    ``moments=`` alone the coherent parameters remain the caller's to supply.

    **Data only.** With an ensemble and no model, moment-match the Rician:
    for the ensemble mean ``m`` and variance ``v`` of the TOTAL INTENSITY,
    ``i_s = m - sqrt(m^2 - v)`` and ``i_c = sqrt(m^2 - v)``; pass the result
    as ``i_c=`` and ``gamma=``. That estimator is the 12_apriori use, and it
    is deliberately left to the caller rather than run silently: a fit is a
    claim about the data, not a plotting default.

    Args:
        draws: 1D array-like of per-pixel contrast deltas.
        laws: A law name ("exact", "rician"), a callable
            ``law(x_grid, params) -> density in delta units``, or a sequence
            of either to overlay. The exact law draws heavy, the limiting
            Rician dashed.
        i_c: Coherent intensity at the pixel.
        gamma: Halo Gamma (incoherent intensity) at the pixel.
        p: Complex pseudo-covariance P at the pixel.
        phi_c: Coherent phase at the pixel.
        normalization: Flux-fraction divisor, ``I = i_c + delta * norm``.
        moments: A ``SpeckleMoments``; supplies ``gamma`` and ``p`` at
            ``pixel``.
        process: A ``SpeckleProcess``; supplies all five parameters at
            ``pixel``.
        pixel: ``(iy, ix)`` index selecting the pixel from map-valued
            sources. Required with ``process=`` or ``moments=``.
        ax: Axes to draw into. None creates a new figure and axes.
        log: Whether to set a log y-scale (densities span decades).
        hist_kw: Extra kwargs passed to ``ax.hist``, applied last.

    Returns:
        A ``PlotResult``. ``artists["hist"]`` is the histogram's patch
        container, ``artists["lines"]`` the list of law ``Line2D`` in
        ``laws`` order.

    Raises:
        ValueError: An unknown law name; ``process=``/``moments=`` without
            ``pixel=``; or a required law parameter supplied by no source.
    """
    ep = _require.eyepiece()
    entries = _law_entries(laws)
    params = _resolve_params(
        {
            "i_c": i_c,
            "gamma": gamma,
            "p": p,
            "phi_c": phi_c,
            "normalization": normalization,
        },
        process,
        moments,
        pixel,
    )

    samples = np.asarray(draws, dtype=float).ravel()
    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    hkw = {"bins": 60, "density": True, "alpha": 0.55, **(hist_kw or {})}
    _, edges, patches = ax.hist(samples, **hkw)

    # The laws are evaluated over exactly the histogram's own span, so "the
    # histogram and the law integrate to the same total over the plotted
    # range" is a statement about one range, not two.
    lower = max(float(edges[0]), -params["i_c"] / params["normalization"])
    x_grid = np.linspace(lower, float(edges[-1]), _LAW_POINTS)

    lines = []
    for label, function, style in entries:
        density = np.asarray(function(x_grid, params), dtype=float)
        (line,) = ax.plot(x_grid, density, label=label, **style)
        lines.append(line)

    if log:
        ax.set_yscale("log")
    ax.set_xlabel(r"contrast delta $\Delta$")
    ax.set_ylabel("probability density")

    return ep.PlotResult(ax=ax, artists={"hist": patches, "lines": lines})


def _at_pixel(value, pixel, *, sample_axis=False):
    """Select ``pixel`` from a map-valued input; pass a scalar/1D through.

    Args:
        value: A scalar, a ``(y, x)`` map, a ``(n,)`` sample vector, or an
            ``(n, y, x)`` stack of maps.
        pixel: ``(iy, ix)`` index, or None.
        sample_axis: True when ``value`` carries a leading sample axis, so a
            3D input indexes as ``value[:, iy, ix]`` and a 1D input is
            already the per-pixel samples.

    Raises:
        ValueError: A map-valued input was given with no ``pixel``.
    """
    arr = np.asarray(value)
    map_valued = arr.ndim == 3 if sample_axis else arr.ndim == 2
    if not map_valued:
        return arr
    if pixel is None:
        msg = "plot_field_ellipse: a map-valued input needs pixel=(iy, ix)"
        raise ValueError(msg)
    iy, ix = pixel
    return arr[:, iy, ix] if sample_axis else arr[iy, ix]


def _ellipse_from_moments(gamma, p):
    """Semi-axes and orientation of the 1-sigma impropriety ellipse.

    The quadrature covariance of an improper complex Gaussian diagonalizes in
    the frame rotated by ``angle(P) / 2``, with variances
    ``(Gamma +- |P|) / 2`` along and across P's principal axis -- the same
    rotated frame ``stats.pixel_sf`` integrates in, so the picture and the
    law agree by construction rather than by coincidence.

    The HALF-angle is the whole subtlety: ``P`` is a pseudo-covariance, a
    second-order quantity in the field, so its phase winds twice as fast as
    the axis it describes. Using ``angle(P)`` would rotate the ellipse by
    twice the right amount -- a mistake that is invisible on any
    axis-aligned test case.

    Returns:
        A ``(semi_major, semi_minor, angle_rad)`` triple.
    """
    magnitude = abs(p)
    semi_major = np.sqrt((gamma + magnitude) / 2.0)
    semi_minor = np.sqrt(max(gamma - magnitude, 0.0) / 2.0)
    return semi_major, semi_minor, 0.5 * np.angle(p)


def _ellipse_from_draws(samples):
    """Semi-axes and orientation from the empirical quadrature covariance.

    The measured counterpart of :func:`_ellipse_from_moments`: an
    eigendecomposition of the real 2x2 covariance of ``(Re, Im)``. For a
    zero-mean improper Gaussian the two agree in the large-sample limit,
    which is the parity a test pins.

    Returns:
        A ``(semi_major, semi_minor, angle_rad)`` triple.
    """
    quadratures = np.stack([samples.real, samples.imag])
    cov = np.cov(quadratures)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    major = eigenvectors[:, order[0]]
    return (
        np.sqrt(max(eigenvalues[0], 0.0)),
        np.sqrt(max(eigenvalues[1], 0.0)),
        np.arctan2(major[1], major[0]),
    )


def plot_field_ellipse(
    pixel,
    *,
    moments=None,
    draws=None,
    e_nom=None,
    normalization=None,
    threshold=None,
    ax=None,
):
    """Draw one pixel's speckle field in the complex plane.

    Four elements, each answering a different question about the same pixel:
    the deterministic field ``E_nom`` as an arrow from the origin (where the
    pixel sits with no drift); the realized cloud of total field values
    ``E_nom + delta_E`` (what the drift actually does); the 1-sigma
    impropriety ellipse (what the second-order statistics SAY the drift
    does); and, with ``threshold=``, the false-alarm circle a detection test
    would draw.

    The ellipse is the point of the picture. A circular (proper) field gives
    a circle and the modified Rician is exactly right; an improper field
    (``|P| > 0``) gives a genuine ellipse, and the further from circular, the
    further the intensity law departs from the Rician. Cause and effect share
    color with the rest of the document: the analytic ellipse takes
    ``roles.model``, the empirical cloud ``roles.measured``.

    The cloud is fed from ``AnalyticSpeckleField.delta_e`` -- the public
    complex-increment accessor -- and this module never reaches into the
    field's private mode-coefficient state, which is the reason that public
    accessor was added. (The private name is deliberately not written
    anywhere in this module, so a bare source grep for it is a valid gate.)
    Build ``draws`` as, for example, ``np.stack([field.delta_e(
    wavelength_nm=w, time_s=t) for t in times])``.

    The false-alarm circle is origin-centered with radius
    ``sqrt(i_c + threshold * normalization)``: it is the contour of TOTAL
    intensity ``|E|^2 = i_c + threshold * normalization``, so ``threshold``
    is read in the same contrast-delta units as ``plot_speckle_ensemble``'s
    abscissa, and a sample outside the circle is a delta above threshold.

    Args:
        pixel: ``(iy, ix)`` index selecting the pixel from any map-valued
            input. May be None when every input is already per-pixel.
        moments: A ``SpeckleMoments``; supplies the analytic ellipse from
            ``gamma_map``/``p_map`` at ``pixel``.
        draws: Complex field increments at the pixel -- ``(n,)``, or
            ``(n, y, x)`` indexed by ``pixel``. Drawn as the cloud, and used
            for the ellipse when ``moments`` is absent.
        e_nom: The complex nominal field, scalar or a ``(y, x)`` map. None
            places the cloud and ellipse at the origin and omits the arrow.
        normalization: Flux-fraction divisor; required with ``threshold``.
        threshold: Contrast-delta detection threshold. None omits the circle.
        ax: Axes to draw into. None creates a new figure and axes.

    Returns:
        A ``PlotResult``. ``artists["ellipse"]`` is the impropriety
        ``Ellipse``; ``artists["cloud"]`` the scatter (absent without
        ``draws``); ``artists["arrow"]`` the ``E_nom`` ``FancyArrowPatch``
        (absent without ``e_nom``); ``artists["circle"]`` the false-alarm
        ``Circle`` (absent without ``threshold``).

    Raises:
        ValueError: Neither ``moments`` nor ``draws`` given; a map-valued
            input with no ``pixel``; ``threshold`` without
            ``normalization``; or an axes ARRAY passed as ``ax``.
    """
    ep = _require.eyepiece()
    if moments is None and draws is None:
        msg = "plot_field_ellipse: pass moments= or draws= (or both)"
        raise ValueError(msg)
    if threshold is not None and normalization is None:
        msg = "plot_field_ellipse: threshold= needs normalization= to set the radius"
        raise ValueError(msg)
    if ax is not None and not isinstance(ax, plt.Axes):
        shape = np.asarray(ax, dtype=object).shape
        msg = f"plot_field_ellipse: expected a single ax, got axes shape {shape}"
        raise ValueError(msg)

    center = complex(_at_pixel(e_nom, pixel)) if e_nom is not None else 0.0 + 0.0j
    samples = None if draws is None else _at_pixel(draws, pixel, sample_axis=True)

    if moments is not None:
        gamma = float(_at_pixel(moments.gamma_map, pixel))
        p = complex(_at_pixel(moments.p_map, pixel))
        semi_major, semi_minor, angle = _ellipse_from_moments(gamma, p)
    else:
        semi_major, semi_minor, angle = _ellipse_from_draws(samples)

    if ax is None:
        _, ax = plt.subplots(layout="constrained")

    model_color = hwostyle.roles.model
    measured_color = hwostyle.roles.measured
    artists = {}

    if samples is not None:
        total = center + samples
        artists["cloud"] = ax.scatter(
            total.real, total.imag, s=6, alpha=0.35, color=measured_color, zorder=2
        )

    ellipse = Ellipse(
        (center.real, center.imag),
        width=2.0 * semi_major,
        height=2.0 * semi_minor,
        angle=np.degrees(angle),
        fill=False,
        lw=2.0,
        color=model_color,
        zorder=3,
    )
    ax.add_patch(ellipse)
    artists["ellipse"] = ellipse

    if e_nom is not None:
        arrow = FancyArrowPatch(
            (0.0, 0.0),
            (center.real, center.imag),
            arrowstyle="-|>",
            mutation_scale=12,
            color=hwostyle.roles.star,
            zorder=4,
        )
        ax.add_patch(arrow)
        artists["arrow"] = arrow

    if threshold is not None:
        radius = np.sqrt(abs(center) ** 2 + threshold * normalization)
        circle = Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            ls="--",
            lw=1.2,
            color=hwostyle.roles.alert,
            zorder=3,
        )
        ax.add_patch(circle)
        artists["circle"] = circle

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}\,E$")
    ax.set_ylabel(r"$\mathrm{Im}\,E$")
    ax.autoscale_view()
    return ep.PlotResult(ax=ax, artists=artists)
