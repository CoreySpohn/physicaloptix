"""Speckle boiling in time: the epoch strip and the animation.

Two views of one thing -- a dark hole whose speckles drift -- typed on the
SPECKLE-FIELD PROTOCOL rather than on any concrete class. Anything with
``realize(wavelength_nm=..., time_s=...) -> (y, x)`` and a
``pixel_scale_lod`` drives both functions: the analytic generator, a
replayed testbed trajectory, a fitted reduced-order model, a maintained
residual, or a ten-line fake in a test. A precomputed ``(n_t, y, x)`` cube
enters through the same door.

**One norm across time, always.** Both functions build a single norm from
the whole cube and hold it fixed for every panel and every frame. This is
not a stylistic preference: renormalizing per epoch rescales each frame to
fill the same color range, so a dark hole that is quietly boiling and one
that is degrading by an order of magnitude render identically. The shared
norm is the measurement.

**What ``realize`` does and does not include.** The protocol contract is
that ``realize`` returns the wavefront-error-induced DELTA and never the
deterministic coronagraphic floor ``|E_nom|^2``, so that a consumer adding
the floor from its own throughput map cannot double count it. Displaying the
boil normally wants the total, hence ``include_floor=True`` by default -- but
the floor is precisely what the protocol does not carry, so it is recovered
from the concrete field's ``e_nom``/``normalization`` when those exist, or
supplied by the caller through ``floor=``. It is never guessed.
"""

import matplotlib.pyplot as plt
import numpy as np
from hwoutils.radial import radial_distance

from physicaloptix.viz import _require

# Image panel over trace panel: the trace is a companion, not a co-equal.
_TRACE_HEIGHT_RATIOS = (3.0, 1.0)
_SECONDS = (("d", 86400.0), ("h", 3600.0), ("min", 60.0), ("s", 1.0))


def _scalar(value, name):
    """A python float from a 0-d array, or a raise naming the channel axis.

    A chromatic field carries a leading channel axis on ``normalization``
    (and on ``e_nom``), and picking a channel is the library's job, not the
    viz layer's guess -- so a chromatic input is rejected with the remedy
    named rather than being silently reduced.

    Raises:
        ValueError: ``value`` is not scalar.
    """
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        msg = (
            f"boiling: {name} has shape {array.shape}, not a scalar -- this "
            "looks like a chromatic field. Select a channel first, or pass "
            "precomputed frames."
        )
        raise ValueError(msg)
    return float(array)


def _floor_map(field, floor):
    """The deterministic floor in ``realize``'s own flux-fraction units.

    ``floor=`` wins when given. Otherwise the floor is reconstructed as
    ``|E_nom|^2 / normalization`` from the concrete field, which both
    shipping implementations carry even though the protocol does not
    require it.

    Raises:
        ValueError: Neither ``floor=`` nor the ``e_nom``/``normalization``
            pair is available, or the field is chromatic.
    """
    if floor is not None:
        return np.asarray(floor, dtype=float)

    e_nom = getattr(field, "e_nom", None)
    normalization = getattr(field, "normalization", None)
    if e_nom is None or normalization is None:
        msg = (
            "boiling: include_floor=True needs the deterministic floor, which "
            "realize() deliberately excludes and the speckle-field protocol "
            "does not carry. Pass floor= (the |E_nom|^2 / normalization map, "
            "in realize's flux-fraction units), or include_floor=False to "
            "show the drift delta alone."
        )
        raise ValueError(msg)

    e = np.asarray(e_nom)
    if e.ndim != 2:
        msg = (
            f"boiling: e_nom has shape {e.shape}, not (y, x) -- this looks "
            "like a chromatic field. Pass floor= for the channel you are "
            "showing."
        )
        raise ValueError(msg)
    return np.abs(e) ** 2 / _scalar(normalization, "normalization")


def _contrast_scale(field, telescope_peak):
    """The ``normalization / telescope_peak`` factor onto peak-referenced units.

    Raises:
        ValueError: The field carries no ``normalization``.
    """
    normalization = getattr(field, "normalization", None)
    if normalization is None:
        msg = (
            "boiling: telescope_peak= needs the field's normalization to "
            "convert flux fractions into peak-referenced contrast, and the "
            "speckle-field protocol does not carry one. Pass precomputed "
            "frames already in contrast units instead."
        )
        raise ValueError(msg)
    return _scalar(normalization, "normalization") / float(telescope_peak)


def _protocol_cube(
    field, times, *, wavelength_nm, telescope_peak, include_floor, floor
):
    """Evaluate the field at every epoch, in the requested display units.

    With ``telescope_peak=`` and no floor, this routes through the field's
    own ``peak_contrast`` when it has one -- the exact conversion, done by
    the library. Otherwise the same conversion is applied as a factor, which
    is what lets the floor be added first: ``peak_contrast`` rescales the
    delta alone and cannot carry a floor with it. A test pins the two routes
    against each other.

    Raises:
        ValueError: ``wavelength_nm`` is None.
    """
    if wavelength_nm is None:
        msg = (
            "boiling: wavelength_nm= is required for a speckle field "
            "(realize() takes it); precomputed frames do not need it"
        )
        raise ValueError(msg)

    if telescope_peak is not None and not include_floor:
        peak_contrast = getattr(field, "peak_contrast", None)
        if peak_contrast is not None:
            return np.stack(
                [
                    np.asarray(
                        peak_contrast(
                            telescope_peak=float(telescope_peak),
                            wavelength_nm=wavelength_nm,
                            time_s=float(t),
                        )
                    )
                    for t in times
                ]
            )

    cube = np.stack(
        [
            np.asarray(field.realize(wavelength_nm=wavelength_nm, time_s=float(t)))
            for t in times
        ]
    )
    if include_floor:
        cube = cube + _floor_map(field, floor)
    if telescope_peak is not None:
        cube = cube * _contrast_scale(field, telescope_peak)
    return cube


def _frames(
    field,
    times_s,
    *,
    wavelength_nm,
    telescope_peak,
    include_floor,
    floor,
    pixscale_lod=None,
):
    """The ``(n_t, y, x)`` display cube and the pixel scale, if one is known.

    Two branches. A protocol object is evaluated per epoch. A precomputed
    ``(n_t, y, x)`` array is taken AS GIVEN: whoever built it chose its
    units and already included whatever floor it has, so ``include_floor``
    does not apply to it (``floor=`` is still added when passed, and
    ``telescope_peak=`` still divides, both being unambiguous on a raw
    cube). A raw array carries no pixel scale of its own, so ``pixscale_lod``
    is the only way its panels get lambda/D axes rather than pixel indices --
    a cropped cube is the common case, and its scale survives the crop even
    though its extent does not.

    Returns:
        A ``(cube, pixel_scale_lod)`` pair; the scale is None for a raw
        array with no ``pixscale_lod``, or a field without one.

    Raises:
        ValueError: A precomputed cube is not 3D, or its length does not
            match ``times_s``.
    """
    times = np.asarray(times_s, dtype=float).ravel()

    if not hasattr(field, "realize"):
        cube = np.asarray(field, dtype=float)
        if cube.ndim != 3:
            msg = f"boiling: precomputed frames must be (n_t, y, x), got {cube.shape}"
            raise ValueError(msg)
        if len(cube) != len(times):
            msg = f"boiling: {len(cube)} precomputed frames but {len(times)} times_s"
            raise ValueError(msg)
        if floor is not None:
            cube = cube + np.asarray(floor, dtype=float)
        if telescope_peak is not None:
            cube = cube / float(telescope_peak)
        return cube, pixscale_lod

    cube = _protocol_cube(
        field,
        times,
        wavelength_nm=wavelength_nm,
        telescope_peak=telescope_peak,
        include_floor=include_floor,
        floor=floor,
    )
    if pixscale_lod is not None:
        return cube, pixscale_lod
    return cube, getattr(field, "pixel_scale_lod", None)


def _time_unit(times):
    """The largest time unit that keeps the numbers readable, and its divisor."""
    span = float(np.max(np.abs(times))) if len(times) else 0.0
    return next(((u, d) for u, d in _SECONDS if span >= 2.0 * d), _SECONDS[-1])


def _time_titles(times):
    """Epoch labels in the unit :func:`_time_unit` picks."""
    unit, divisor = _time_unit(times)
    return [f"t = {t / divisor:.3g} {unit}" for t in times]


def _extent_for(cube, pixel_scale_lod, ep):
    """The lambda/D extent from the protocol's pixel scale, or None."""
    if pixel_scale_lod is None:
        return None
    return ep.extent_lod_from_pixels(cube.shape[-1], float(pixel_scale_lod))


_LOW_PERCENTILE = 1.0
_HIGH_PERCENTILE = 99.9


def _log_bounds(cube, vmin, vmax):
    """The shared log norm's bounds: percentiles, not the raw extremes.

    A dark hole's total intensity has a long thin tail at both ends -- a few
    pixels within numerical noise of zero, a few pinned speckles far above
    the bulk. Taking the raw min and max hands those outliers the whole
    color range and flattens everything else into the top decade: on a real
    EAC-1 cube the raw minimum sits four decades below the first percentile,
    so four of the seven rendered decades show nothing at all.

    The default therefore spans the 1st to 99.9th percentile of the WHOLE
    cube (still one norm for every frame, which is the invariant that
    matters), the same choice the hand-rolled EAC-1 boiling figure reached
    for its own upper bound. Pass ``vmin``/``vmax`` to pin an absolute
    window instead; an explicit bound is always used as given.

    Returns:
        A ``(vmin, vmax)`` pair.
    """
    positive = cube[cube > 0.0]
    if not positive.size:
        return vmin, vmax
    low = float(np.percentile(positive, _LOW_PERCENTILE)) if vmin is None else vmin
    high = float(np.percentile(positive, _HIGH_PERCENTILE)) if vmax is None else vmax
    return low, high


_PANEL_WIDTH_IN = 3.2
_PANEL_HEIGHT_IN = 3.6


def _row_axes(axes, n):
    """Caller axes, or a row whose FIGURE is sized for ``n`` square panels.

    An owned figure is built here rather than left to the drawing primitive
    because a row of images knows something the primitive does not: the
    panels are square and there are ``n`` of them, so the figure has to grow
    with the panel count. Left at matplotlib's default figure size, a
    four-epoch strip renders as four small squares stranded in a tall
    figure beside a full-height colorbar -- the panels shrink to fit a
    height nothing else needed.
    """
    if axes is not None:
        return axes
    _, panel_axes = plt.subplots(
        1,
        n,
        figsize=(_PANEL_WIDTH_IN * n, _PANEL_HEIGHT_IN),
        squeeze=False,
        layout="constrained",
    )
    return np.asarray(panel_axes[0], dtype=object)


def _label_row(axes, ep):
    """Label the row's coordinates once, not once per panel.

    Every panel shares one extent and one norm, so repeating the y label and
    its ticks on all of them spends space on redundancy -- and on a strip of
    six epochs the repetition is what squeezes the images.
    """
    for i, panel_ax in enumerate(axes):
        ep.label_lod(panel_ax)
        if i:
            panel_ax.set_ylabel("")
            panel_ax.set_yticklabels([])


def _is_signed(cube):
    """Whether the frames contain negative values.

    This is the stated rule that picks the figure's anatomy. A coherent
    field's floor-free delta really does go negative -- the pinning cross
    term ``2 Re(E_nom* G eps)`` subtracts wherever the drift lands out of
    phase with the nominal field -- and a log norm cannot show a negative
    number, so those frames are drawn on a symmetric diverging norm
    instead. All-positive frames (any total, any incoherent halo) take the
    log norm, which is what a dark hole spanning decades needs.
    """
    return bool(np.nanmin(cube) < 0.0)


def boiling_strip(
    field,
    times_s,
    *,
    wavelength_nm=None,
    telescope_peak=None,
    include_floor=True,
    floor=None,
    pixscale_lod=None,
    titles=None,
    axes=None,
    vmin=None,
    vmax=None,
    imshow_kw=None,
):
    """Draw one dark-hole map per epoch, all on one shared norm.

    The still counterpart of :func:`animate_speckles`, and the figure that
    belongs in a paper: a strip of epochs reads as boiling only because
    every panel shares one norm, built from the whole time series. Per-panel
    norms would make a stable dark hole and a degrading one look the same.

    The panels' anatomy follows the data's sign, by the rule in
    ``_is_signed``: all-positive frames take a shared log norm with one
    shared colorbar (``eyepiece.compare_row``); frames containing negatives
    take a shared symmetric diverging norm (``eyepiece.imshow_diverging``),
    because a floor-free coherent delta genuinely goes negative and a log
    norm cannot render it.

    Args:
        field: A speckle field implementing ``realize(wavelength_nm=,
            time_s=)`` and, for lambda/D axes, ``pixel_scale_lod``; or a
            precomputed ``(n_t, y, x)`` cube, taken as given.
        times_s: Epoch times in seconds since the field's own epoch, one
            per panel.
        wavelength_nm: Wavelength passed to ``realize``. Required for a
            speckle field, unused for a precomputed cube.
        telescope_peak: Peak intensity of the unocculted telescope PSF on
            this grid. Given, the maps are peak-referenced contrast rather
            than flux fractions -- through the field's own ``peak_contrast``
            where that applies. None leaves flux fractions.
        include_floor: Whether to add the deterministic floor ``|E_nom|^2 /
            normalization`` that ``realize`` excludes by contract. Default
            True (the total intensity, which is what boiling looks like).
            Does not apply to a precomputed cube.
        floor: The floor map in ``realize``'s flux-fraction units,
            overriding the reconstruction from ``e_nom``/``normalization``.
            The way a field with no ``e_nom`` gets a floor.
        pixscale_lod: Pixel scale in lambda/D per pixel, for a precomputed
            cube (which carries none) or to override a field's own. Without
            it a bare cube's panels fall back to pixel indices. A cropped
            cube keeps its pixel scale even though it loses its extent, so
            this is how a zoomed dark hole keeps lambda/D axes.
        titles: Per-panel titles. None labels each panel with its epoch, in
            the largest time unit that keeps the numbers readable.
        axes: Axes to draw into, shape ``(n,)`` for ``n = len(times_s)``.
            None creates the figure.
        vmin: Lower bound of the shared norm. None derives it from the
            cube. Pin both ends to read a strip against an absolute,
            document-wide contrast window rather than against itself --
            which is the only way two strips of different runs compare.
        vmax: Upper bound of the shared norm; on the diverging anatomy it
            is the symmetric bound (the norm spans ``(-vmax, vmax)``).
        imshow_kw: Extra kwargs passed to every panel's ``imshow``.

    Returns:
        A ``MosaicResult`` with ``axes`` shape ``(n,)``. ``artists["image"]``
        is the list of ``AxesImage``; ``artists["cbar"]`` the single shared
        colorbar.

    Raises:
        ValueError: ``axes`` of the wrong shape (naming both); a missing
            ``wavelength_nm``; an unavailable floor under
            ``include_floor=True``; a malformed precomputed cube; or a
            ``vmin`` on signed frames, whose norm is symmetric by
            construction and takes ``vmax`` alone.
    """
    ep = _require.eyepiece()
    cube, pixel_scale_lod = _frames(
        field,
        times_s,
        wavelength_nm=wavelength_nm,
        telescope_peak=telescope_peak,
        include_floor=include_floor,
        floor=floor,
        pixscale_lod=pixscale_lod,
    )
    n = len(cube)

    if axes is not None:
        axes = np.atleast_1d(np.asarray(axes, dtype=object))
        if axes.shape != (n,):
            msg = f"boiling_strip: expected axes shape ({n},), got {axes.shape}"
            raise ValueError(msg)

    times = np.asarray(times_s, dtype=float).ravel()
    panel_titles = _time_titles(times) if titles is None else list(titles)
    extent = _extent_for(cube, pixel_scale_lod, ep)
    label = "contrast" if telescope_peak is not None else "flux fraction"

    if not _is_signed(cube):
        low, high = _log_bounds(cube, vmin, vmax)
        result = ep.compare_row(
            list(cube),
            panel_titles,
            axes=_row_axes(axes, n),
            norm="log",
            extent=extent,
            vmin=low,
            vmax=high,
            cbar_label=label,
            imshow_kw=imshow_kw,
        )
        if pixel_scale_lod is not None:
            _label_row(result.axes, ep)
        return result

    # Signed frames: one symmetric norm across the strip, and a single
    # colorbar on the last panel rather than one per panel.
    if vmin is not None:
        msg = (
            "boiling_strip: these frames contain negatives, so the shared norm "
            "is symmetric about zero and takes vmax alone; vmin is -vmax by "
            "construction"
        )
        raise ValueError(msg)
    axes = _row_axes(axes, n)

    vlim = float(np.nanmax(np.abs(cube))) if vmax is None else float(vmax)
    images = []
    cbar = None
    for i in range(n):
        panel = ep.imshow_diverging(
            cube[i],
            ax=axes[i],
            extent=extent,
            vlim=vlim,
            colorbar=(i == n - 1),
            cbar_label=label,
            imshow_kw=imshow_kw,
        )
        axes[i].set_title(panel_titles[i])
        images.append(panel.artists["image"])
        cbar = panel.artists.get("cbar", cbar)

    if pixel_scale_lod is not None:
        _label_row(axes, ep)

    artists = {"image": images}
    if cbar is not None:
        artists["cbar"] = cbar
    return ep.MosaicResult(axes=axes, artists=artists)


def _trace_series(trace, cube, pixel_scale_lod):
    """The companion trace: a supplied series, or a dark-zone mean per frame.

    A TUPLE is read as ``(iwa_lod, owa_lod)`` and the series is computed as
    the mean over that annulus, frame by frame. Anything else is read as an
    already-computed ``(n_t,)`` series, which is how a maintenance loop's
    ``contrast_history`` drops straight in. The tuple-versus-sequence split
    is what keeps a two-frame animation unambiguous.

    Raises:
        ValueError: A tuple form with no pixel scale, or a series whose
            length does not match the frame count.
    """
    if isinstance(trace, tuple):
        if pixel_scale_lod is None:
            msg = (
                "boiling: trace=(iwa_lod, owa_lod) needs a pixel scale to "
                "place the annulus; pass a precomputed (n_t,) series instead"
            )
            raise ValueError(msg)
        iwa_lod, owa_lod = trace
        radius = np.asarray(radial_distance(cube.shape[-2:])) * float(pixel_scale_lod)
        mask = (radius >= float(iwa_lod)) & (radius <= float(owa_lod))
        return cube[:, mask].mean(axis=1)

    series = np.asarray(trace, dtype=float).ravel()
    if len(series) != len(cube):
        msg = f"boiling: trace has {len(series)} points but {len(cube)} frames"
        raise ValueError(msg)
    return series


def animate_speckles(
    field,
    times_s,
    *,
    wavelength_nm=None,
    telescope_peak=None,
    include_floor=True,
    floor=None,
    pixscale_lod=None,
    trace=None,
    fps=10,
    fig=None,
):
    """Animate a dark hole boiling, optionally over a contrast trace.

    Returns an ``eyepiece.Animation``, which is LAZY: the frames are
    computed and the figure is built here, but nothing is rendered until the
    caller asks for a sink (``.save("run.mp4", "run.gif")``, ``.jshtml()``,
    ``.video()``). That keeps this function stateless and file-free, lets one
    animation go to several destinations, and keeps ffmpeg off the critical
    path for anyone who only wants the notebook player.

    Every frame shares one norm, built once from the whole cube, for the
    reason spelled out in :func:`boiling_strip`: a per-frame norm turns any
    time series into a stable-looking one. Frames are materialized up front,
    so each animation frame is a ``set_data`` rather than a re-evaluation of
    the field.

    Args:
        field: A speckle field, or a precomputed ``(n_t, y, x)`` cube; see
            :func:`boiling_strip`.
        times_s: Frame times in seconds since the field's own epoch.
        wavelength_nm: Wavelength passed to ``realize``.
        telescope_peak: Peak-reference the maps; see :func:`boiling_strip`.
        include_floor: Add the deterministic floor; see
            :func:`boiling_strip`.
        floor: Explicit floor map in flux-fraction units.
        pixscale_lod: Pixel scale in lambda/D per pixel; see
            :func:`boiling_strip`.
        trace: Optional companion panel under the image. A TUPLE
            ``(iwa_lod, owa_lod)`` computes the dark-zone mean per frame; a
            sequence is used as an already-computed ``(n_t,)`` series (a
            maintenance loop's ``contrast_history``). A marker tracks the
            current frame along it.
        fps: Playback rate baked into the returned animation.
        fig: Figure to build into. None creates one. The function owns its
            internal gridspec, so there is no axes-shape contract here.

    Returns:
        An ``eyepiece.Animation``. Its ``.fig`` is the figure drawn into and
        ``.n_frames`` the frame count.

    Raises:
        ValueError: The same conditions as :func:`boiling_strip`, plus a
            malformed ``trace``.
    """
    ep = _require.eyepiece()
    cube, pixel_scale_lod = _frames(
        field,
        times_s,
        wavelength_nm=wavelength_nm,
        telescope_peak=telescope_peak,
        include_floor=include_floor,
        floor=floor,
        pixscale_lod=pixscale_lod,
    )
    times = np.asarray(times_s, dtype=float).ravel()
    titles = _time_titles(times)
    extent = _extent_for(cube, pixel_scale_lod, ep)
    label = "contrast" if telescope_peak is not None else "flux fraction"
    series = None if trace is None else _trace_series(trace, cube, pixel_scale_lod)

    if fig is None:
        fig = plt.figure(figsize=(4.8, 6.0 if series is not None else 4.8))
    if series is not None:
        image_ax, trace_ax = fig.subplots(2, 1, height_ratios=_TRACE_HEIGHT_RATIOS)
    else:
        image_ax, trace_ax = fig.subplots(), None

    # One norm for the whole run: built from the full cube, then held.
    if _is_signed(cube):
        panel = ep.imshow_diverging(
            cube[0],
            ax=image_ax,
            extent=extent,
            vlim=float(np.nanmax(np.abs(cube))),
            colorbar=False,
        )
    else:
        low, high = _log_bounds(cube, None, None)
        panel = ep.imshow_log(
            cube[0],
            ax=image_ax,
            extent=extent,
            vmin=low,
            vmax=high,
            colorbar=False,
        )
    # The colorbar is attached here, not left to the image primitive: that one
    # hangs it in an inset just outside the axes, and an inset is invisible to
    # constrained layout, so its label lands off the canvas and the recorded
    # frames carry the clipped version. A figure-level colorbar is placed by
    # the layout engine, which reserves the room before the layout freezes.
    fig.colorbar(panel.artists["image"], ax=image_ax, label=label)
    if pixel_scale_lod is not None:
        ep.label_lod(image_ax)

    clock = image_ax.text(
        0.03, 0.97, titles[0], transform=image_ax.transAxes, va="top", fontsize=9
    )

    marker = None
    if series is not None:
        # The trace shares the clock's unit: a companion panel reading in
        # seconds while the label overhead reads in days is two clocks.
        unit, divisor = _time_unit(times)
        trace_ax.plot(times / divisor, series)
        (marker,) = trace_ax.plot(
            [times[0] / divisor], [series[0]], marker="o", ls="none"
        )
        trace_ax.set_xlabel(f"time [{unit}]")
        trace_ax.set_ylabel(label)

    def draw(_fig, index):
        panel.update(cube[index])
        clock.set_text(titles[index])
        if marker is not None:
            marker.set_data([times[index] / _time_unit(times)[1]], [series[index]])

    return ep.animate(fig, draw, len(cube), fps=fps)
