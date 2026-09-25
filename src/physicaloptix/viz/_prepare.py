"""Speckle boiling, prepared once: the science behind every boiling view.

``prepare_speckles`` does all of the scientific work a boiling dark hole
needs -- evaluating a speckle field at each epoch, adding the deterministic
floor, peak referencing, choosing the shared display bounds, and computing
the dark-zone trace -- exactly once, and returns an ``eyepiece`` prepared
``Sequence``. A still (``sequence.frame(i)``), a strip of epochs
(``sequence.strip([...])``), a Matplotlib movie (``eyepiece.mpl.animate``),
and a Manim clip (``eyepiece.manim.animate``) are then all views of that one
prepared storage; none of them evaluates the field again.

This module imports NumPy, hwoutils, and ``eyepiece.prepared`` only. It
never imports Matplotlib or a renderer.

**What ``realize`` does and does not include.** The speckle-field protocol
contract is that ``realize`` returns the wavefront-error-induced DELTA and
never the deterministic coronagraphic floor ``|E_nom|^2``, so that a
consumer adding the floor from its own throughput map cannot double count
it. Displaying the boil normally wants the total, hence
``include_floor=True`` by default -- but the floor is precisely what the
protocol does not carry, so it is recovered from the concrete field's
``e_nom``/``normalization`` when those exist, or supplied by the caller
through ``floor=``. It is never guessed.

**One scale across time, always.** The display bounds are resolved once
from the whole sequence (or taken from the caller) and held for every
frame, every strip panel, and every movie frame. Renormalizing per epoch
rescales each frame to fill the same color range, so a dark hole that is
quietly boiling and one that is degrading by an order of magnitude would
render identically. The shared scale is the measurement.
"""

import numpy as np
from eyepiece.prepared import (
    ArrayChannel,
    AxisSpec,
    Clock,
    CurveView,
    ImageView,
    Label,
    PanelGroup,
    Path,
    Points,
    Region,
    Scale,
    Sequence,
    resolve_bounds,
)
from hwoutils.radial import radial_distance

# Stable element IDs of a prepared speckle view. The image, its acquisition
# clock, and the dark-zone annulus always sit on one image panel; the trace
# panel (present only when a trace is requested) carries the full series and
# the one-point active datum.
GROUP_ID = "speckles"
IMAGE_ID = "image"
CLOCK_ID = "clock"
ANNULUS_ID = "annulus"
TRACE_ID = "trace"
SERIES_ID = "series"
ACTIVE_ID = "active"

_QUANTITIES = ("total", "delta")
_INSTANTANEOUS = "instantaneous"
_SECONDS = (("d", 86400.0), ("h", 3600.0), ("min", 60.0), ("s", 1.0))
_LOW_PERCENTILE = 1.0
_HIGH_PERCENTILE = 99.9
_CLOCK_XY = (0.03, 0.97)
_TRACE_MARGIN = 0.05
_LOD_LABELS = ("x [\u03bb/D]", "y [\u03bb/D]")


# --- declared semantics -------------------------------------------------------


def _check_sample_kind(sample_kind, *, is_field):
    """Accept instantaneous samples only; a bare array must say so.

    Raises:
        ValueError: ``sample_kind`` is anything but ``"instantaneous"``, or a
            bare array leaves it undeclared.
    """
    if sample_kind is None:
        if is_field:
            return
        msg = (
            "prepare_speckles: a bare array must declare its samples, "
            "sample_kind='instantaneous' (a speckle field declares this itself)"
        )
        raise ValueError(msg)
    if sample_kind != _INSTANTANEOUS:
        msg = (
            f"prepare_speckles: sample_kind={sample_kind!r} is unsupported; "
            "prepared speckle sequences hold instantaneous samples only "
            "(exposure-integrated frames need a later contract extension)"
        )
        raise ValueError(msg)


def _output_quantity(is_field, *, include_floor, floor, quantity):
    """Whether the prepared frames are a total intensity or a signed delta.

    A field's frames are the ``realize`` delta, made total by the floor, so
    the quantity follows ``include_floor`` and a conflicting declaration is
    refused. A bare array states what it holds; adding ``floor=`` to it makes
    the result a total.

    Raises:
        ValueError: A bare array without a valid ``quantity``, or a field
            whose declared ``quantity`` contradicts ``include_floor``.
    """
    if is_field:
        derived = "total" if include_floor else "delta"
        if quantity is not None and quantity != derived:
            msg = (
                f"prepare_speckles: quantity={quantity!r} contradicts "
                f"include_floor={include_floor}, which makes a speckle field's "
                f"frames a {derived}"
            )
            raise ValueError(msg)
        return derived
    if quantity not in _QUANTITIES:
        msg = (
            "prepare_speckles: a bare array must declare quantity='total' or "
            f"quantity='delta', got {quantity!r}"
        )
        raise ValueError(msg)
    return "total" if floor is not None else quantity


# --- the protocol branch: evaluate each epoch once ---------------------------


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
    field, times_s, *, wavelength_nm, telescope_peak, include_floor, floor
):
    """Evaluate the field once per epoch into one preallocated cube.

    With ``telescope_peak=`` and no floor, this routes through the field's
    own ``peak_contrast`` when it has one -- the exact conversion, done by
    the library. Otherwise the same conversion is applied as a factor, which
    is what lets the floor be added first: ``peak_contrast`` rescales the
    delta alone and cannot carry a floor with it. A test pins the two routes
    against each other.

    Every precondition (wavelength, floor, normalization) is checked before
    the first evaluation, and the output is allocated once from the first
    frame's shape and dtype, then filled frame by frame.

    Raises:
        ValueError: ``wavelength_nm`` is None, the floor or normalization is
            unavailable, or an evaluated frame is not 2D (a chromatic field).
    """
    if wavelength_nm is None:
        msg = (
            "boiling: wavelength_nm= is required for a speckle field "
            "(realize() takes it); precomputed frames do not need it"
        )
        raise ValueError(msg)

    peak_contrast = None
    if telescope_peak is not None and not include_floor:
        peak_contrast = getattr(field, "peak_contrast", None)
    floor_map = None
    factor = None
    if peak_contrast is None:
        floor_map = _floor_map(field, floor) if include_floor else None
        if telescope_peak is not None:
            factor = _contrast_scale(field, telescope_peak)

    def evaluate(t):
        if peak_contrast is not None:
            return np.asarray(
                peak_contrast(
                    telescope_peak=float(telescope_peak),
                    wavelength_nm=wavelength_nm,
                    time_s=float(t),
                )
            )
        frame = np.asarray(field.realize(wavelength_nm=wavelength_nm, time_s=float(t)))
        if floor_map is not None:
            frame = frame + floor_map
        if factor is not None:
            frame = frame * factor
        return frame

    first = evaluate(times_s[0])
    if first.ndim != 2:
        msg = (
            f"boiling: the field evaluates to shape {first.shape}, not (y, x) "
            "-- this looks like a chromatic field. Select a channel first, or "
            "pass precomputed frames."
        )
        raise ValueError(msg)
    cube = np.empty((len(times_s), *first.shape), dtype=first.dtype)
    cube[0] = first
    for index in range(1, len(times_s)):
        cube[index] = evaluate(times_s[index])
    return cube


# --- the bare-array branch: borrow unless a transformation is asked for -----


def _split_mask(values):
    """Plain data and a validity mask (None when the input carries no mask)."""
    mask = np.ma.getmask(values)
    data = np.asarray(np.ma.getdata(values))
    valid = None if mask is np.ma.nomask else ~np.ma.getmaskarray(values)
    return data, valid


def _bare_cube(data, n_times, *, floor, telescope_peak):
    """A precomputed cube, borrowed as given or transformed into one buffer.

    A precomputed cube is taken AS GIVEN: whoever built it chose its units
    and whatever floor it has, so ``include_floor`` does not apply. An
    explicit ``floor=`` is still added, and ``telescope_peak=`` still divides,
    both being unambiguous on a raw cube. Without either, the caller's
    storage is borrowed (no copy, no dtype promotion); with either, one
    output buffer is allocated and filled frame by frame.

    Raises:
        ValueError: The cube is not a numeric ``(n_t, y, x)`` array, or its
            length does not match ``times_s``.
    """
    if not np.issubdtype(data.dtype, np.number):
        msg = f"boiling: precomputed frames must be numeric, got {data.dtype}"
        raise ValueError(msg)
    if data.ndim != 3:
        msg = f"boiling: precomputed frames must be (n_t, y, x), got {data.shape}"
        raise ValueError(msg)
    if len(data) != n_times:
        msg = f"boiling: {len(data)} precomputed frames but {n_times} times_s"
        raise ValueError(msg)
    if floor is None and telescope_peak is None:
        return data

    floor_map = None if floor is None else np.asarray(floor)
    dtypes = [data.dtype] if floor_map is None else [data.dtype, floor_map.dtype]
    out = np.empty(data.shape, dtype=np.result_type(*dtypes, 1.0))
    for index in range(len(data)):
        frame = out[index]
        frame[...] = data[index]
        if floor_map is not None:
            frame += floor_map
        if telescope_peak is not None:
            frame /= float(telescope_peak)
    return out


# --- display scale -------------------------------------------------------------


def _valid_frames(cube, valid):
    """(data, valid) pairs per frame, for eyepiece's streaming bound resolver."""
    for index in range(len(cube)):
        yield cube[index], None if valid is None else valid[index]


def _log_bounds(cube, valid):
    """The shared log scale's bounds: whole-sequence percentiles.

    A dark hole's total intensity has a long thin tail at both ends -- a few
    pixels within numerical noise of zero, a few pinned speckles far above
    the bulk. Taking the raw min and max hands those outliers the whole
    color range and flattens everything else into the top decade: on a real
    EAC-1 cube the raw minimum sits four decades below the first percentile.
    So the default spans the 1st to 99.9th percentile of every positive,
    finite, valid sample of the WHOLE sequence (still one scale for every
    frame, which is the invariant that matters).

    Raises:
        ValueError: No positive finite valid sample exists, or the
            percentiles coincide (constant input needs explicit bounds).
    """
    keep = np.isfinite(cube) & (cube > 0)
    if valid is not None:
        keep &= valid
    positive = cube[keep]
    if not positive.size:
        msg = (
            "prepare_speckles: cannot resolve display bounds, no positive "
            "finite valid sample in any frame"
        )
        raise ValueError(msg)
    low = float(np.percentile(positive, _LOW_PERCENTILE))
    high = float(np.percentile(positive, _HIGH_PERCENTILE))
    if not low < high:
        msg = (
            f"prepare_speckles: display bounds are degenerate ({low}, {high}) "
            "(constant input); pass bounds="
        )
        raise ValueError(msg)
    return low, high


def _scale(cube, valid, quantity, bounds):
    """One Scale for the whole sequence: log for a total, symmetric for a delta.

    Supplied bounds are validated and used as given, and bypass every
    statistic. Otherwise a total spans whole-sequence percentiles and a
    delta spans the largest valid finite magnitude, symmetric about zero.
    """
    if quantity == "delta":
        low, high = resolve_bounds(
            _valid_frames(cube, valid), kind="symmetric", bounds=bounds
        )
        return Scale("symmetric", low, high, cmap_role="residual")
    if bounds is not None:
        low, high = resolve_bounds((), kind="log", bounds=bounds)
    else:
        low, high = _log_bounds(cube, valid)
    return Scale("log", low, high, cmap_role="intensity", floor=low)


# --- time, coordinates, and the trace ---------------------------------------


def _time_unit(times_s):
    """The largest time unit that keeps the numbers readable, and its divisor."""
    span = float(np.max(np.abs(times_s))) if len(times_s) else 0.0
    return next(((u, d) for u, d in _SECONDS if span >= 2.0 * d), _SECONDS[-1])


def _image_axes(shape, pixel_scale_lod):
    """Pixel-edge extent: lambda/D centered on zero, or bare pixel indices."""
    ny, nx = shape
    if pixel_scale_lod is None:
        return AxisSpec("", "", (-0.5, nx - 0.5), (-0.5, ny - 0.5), show_ticks=False)
    scale = float(pixel_scale_lod)
    half_x = nx * scale / 2.0
    half_y = ny * scale / 2.0
    return AxisSpec(*_LOD_LABELS, (-half_x, half_x), (-half_y, half_y))


def _annulus_series(cube, valid, annulus):
    """The dark-zone mean per frame, over that frame's valid finite pixels.

    A frame with no valid finite pixel in the annulus has no mean: its datum
    is NaN (a missing sample drawn as a gap), never zero.
    """
    series = np.empty(len(cube), dtype=float)
    for index, frame in enumerate(cube):
        keep = annulus & np.isfinite(frame)
        if valid is not None:
            keep &= valid[index]
        series[index] = frame[keep].mean() if np.any(keep) else np.nan
    return series


def _trace_series(trace, cube, valid, pixel_scale_lod):
    """The companion trace: a supplied series, or a dark-zone mean per frame.

    A TUPLE is read as ``(iwa_lod, owa_lod)`` and the series is computed as
    the mean over that annulus, frame by frame. Anything else is read as an
    already-computed ``(n_t,)`` series, which is how a maintenance loop's
    ``contrast_history`` drops straight in. The tuple-versus-sequence split
    is what keeps a two-frame sequence unambiguous.

    Returns:
        A ``(series, radii)`` pair; ``radii`` is ``(iwa_lod, owa_lod)`` for
        the annulus form and None for a supplied series.

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
        iwa_lod, owa_lod = (float(v) for v in trace)
        radius = np.asarray(radial_distance(cube.shape[-2:])) * float(pixel_scale_lod)
        annulus = (radius >= iwa_lod) & (radius <= owa_lod)
        return _annulus_series(cube, valid, annulus), (iwa_lod, owa_lod)

    series = np.asarray(trace, dtype=float).ravel()
    if len(series) != len(cube):
        msg = f"boiling: trace has {len(series)} points but {len(cube)} frames"
        raise ValueError(msg)
    return series, None


def _padded_limits(values):
    """Finite, strictly increasing limits around the finite ``values``."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return (0.0, 1.0)
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    if span > 0:
        pad = _TRACE_MARGIN * span
    else:
        pad = _TRACE_MARGIN * abs(high) if high else 0.5
    return (low - pad, high + pad)


# --- the prepared sequence -----------------------------------------------------


def prepare_speckles(
    field_or_cube,
    *,
    times_s,
    wavelength_nm=None,
    telescope_peak=None,
    include_floor=True,
    floor=None,
    pixscale_lod=None,
    trace=None,
    bounds=None,
    sample_kind=None,
    quantity=None,
    clock_fmt=None,
):
    """Prepare a boiling dark hole once, as a replayable eyepiece ``Sequence``.

    Args:
        field_or_cube: A speckle field implementing ``realize(wavelength_nm=,
            time_s=)`` (and, for lambda/D axes, ``pixel_scale_lod``); or a
            precomputed ``(n_t, y, x)`` array (a ``numpy.ma.MaskedArray``'s
            mask is captured as validity). A field is evaluated once per
            epoch into one allocated cube; a bare array is borrowed as given
            unless ``floor=`` or ``telescope_peak=`` transforms it.
        times_s: Acquisition times in seconds, finite and strictly
            increasing, one per frame. The sequence holds them in the
            largest unit that keeps the numbers readable (``"d"``, ``"h"``,
            ``"min"``, or ``"s"``, named by ``sequence.time_unit``), which
            the clock label and the trace axis share.
        wavelength_nm: Wavelength passed to ``realize``. Required for a
            field, unused for a bare array.
        telescope_peak: Peak intensity of the unocculted telescope PSF on
            this grid. Given, frames are peak-referenced contrast (through
            the field's own ``peak_contrast`` where that applies; a bare
            array is divided by it). None leaves flux fractions.
        include_floor: For a field, add the deterministic floor ``|E_nom|^2 /
            normalization`` that ``realize`` excludes by contract (True, the
            default, prepares a total intensity; False the signed delta).
            Does not apply to a bare array.
        floor: The floor map in ``realize``'s flux-fraction units,
            overriding the reconstruction from ``e_nom``/``normalization``;
            on a bare array it is added as given and makes the frames a
            total.
        pixscale_lod: Pixel scale in lambda/D per pixel, for a bare array
            or to override a field's own. With a pixel scale the image has
            lambda/D pixel-edge extents centered on zero; without one it
            keeps pixel-index edges with no tick labels.
        trace: Optional companion trace under the image. A TUPLE
            ``(iwa_lod, owa_lod)`` computes the dark-zone mean over that
            annulus per frame, from valid finite pixels only (a frame with
            none is a missing datum, never zero); any other value is an
            already-computed ``(n_t,)`` series used as given.
        bounds: Display ``(vmin, vmax)`` in the prepared quantity's units.
            A total needs ``0 < vmin < vmax``; a delta needs symmetric
            bounds, ``vmin == -vmax``. None resolves them from the whole
            sequence: the 1st to 99.9th percentile of the positive valid
            finite samples for a total, the largest valid finite magnitude
            for a delta.
        sample_kind: What each frame samples. Only ``"instantaneous"`` is
            supported. A field declares it itself; a bare array must.
        quantity: ``"total"`` or ``"delta"``. A bare array must declare it;
            a field derives it from ``include_floor`` (a conflicting value
            is refused).
        clock_fmt: ``str.format`` template for the acquisition-time label,
            using only ``{value}`` (the time, a float, in ``time_unit``)
            and ``{unit}``, e.g. ``"t = {value:.2f} {unit}"``. None uses
            eyepiece's compact default. The frame labels and a strip's
            per-panel time labels share it.

    Returns:
        A ``Sequence``. Without a trace its frames are one ``ImageView``
        (id ``"image"``) carrying a ``"clock"`` label, whose text is the
        acquisition time. With a trace they are a column ``PanelGroup``
        (id ``"speckles"``) of that image, with an ``"annulus"`` ``Region``
        for the annulus form, over a ``CurveView`` (id ``"trace"``) holding
        the full ``"series"`` path and the one-point ``"active"`` datum. A
        total uses a log scale (``"intensity"`` colormap role, display floor
        at ``vmin``); a delta a symmetric one (``"residual"`` role), with a
        quantity label that names it a delta.

    Raises:
        ValueError: Undeclared or unsupported sample semantics, a missing or
            contradictory quantity, a missing ``wavelength_nm``, an
            unavailable floor under ``include_floor=True``, a chromatic
            field, a malformed cube or trace, invalid bounds, or a sequence
            with no valid sample to scale.
    """
    is_field = hasattr(field_or_cube, "realize")
    _check_sample_kind(sample_kind, is_field=is_field)
    quantity = _output_quantity(
        is_field, include_floor=include_floor, floor=floor, quantity=quantity
    )
    # Validated before any evaluation, like every other argument.
    clock = Clock(CLOCK_ID) if clock_fmt is None else Clock(CLOCK_ID, fmt=clock_fmt)
    times_s = np.asarray(times_s, dtype=float).ravel()

    if is_field:
        cube = _protocol_cube(
            field_or_cube,
            times_s,
            wavelength_nm=wavelength_nm,
            telescope_peak=telescope_peak,
            include_floor=include_floor,
            floor=floor,
        )
        valid = None
        if pixscale_lod is None:
            pixscale_lod = getattr(field_or_cube, "pixel_scale_lod", None)
    else:
        data, valid = _split_mask(field_or_cube)
        cube = _bare_cube(
            data, len(times_s), floor=floor, telescope_peak=telescope_peak
        )

    scale = _scale(cube, valid, quantity, bounds)
    base_label = "contrast" if telescope_peak is not None else "flux fraction"
    label = base_label if quantity == "total" else f"{base_label} delta"
    unit, divisor = _time_unit(times_s)
    times = times_s / divisor
    clock_text = clock.fmt.format(value=float(times[0]), unit=unit)

    series = radii = None
    if trace is not None:
        series, radii = _trace_series(trace, cube, valid, pixscale_lod)

    marks = [Label(CLOCK_ID, clock_text, _CLOCK_XY, space="panel")]
    if radii is not None:
        marks.append(
            Region(
                ANNULUS_ID,
                np.zeros(2),
                outer_radius=radii[1],
                inner_radius=radii[0],
                label="trace annulus",
            )
        )
    image = ImageView(
        IMAGE_ID,
        cube[0],
        _image_axes(cube.shape[-2:], pixscale_lod),
        scale,
        label,
        valid=None if valid is None else valid[0],
        marks=tuple(marks),
    )
    channels = [ArrayChannel(IMAGE_ID, "data", cube), clock]
    if valid is not None:
        channels.append(ArrayChannel(IMAGE_ID, "valid", valid))

    if series is None:
        return Sequence(image, times, unit, tuple(channels))

    trace_xy = np.stack([times, series], axis=1)
    trace_view = CurveView(
        TRACE_ID,
        AxisSpec(
            f"time [{unit}]",
            label,
            _padded_limits(times),
            _padded_limits(series),
            aspect="auto",
        ),
        marks=(
            Path(SERIES_ID, trace_xy, label="dark-zone trace"),
            Points(ACTIVE_ID, trace_xy[:1]),
        ),
    )
    channels.append(ArrayChannel(ACTIVE_ID, "xy", trace_xy[:, None, :]))
    template = PanelGroup(GROUP_ID, (image, trace_view), direction="column")
    return Sequence(template, times, unit, tuple(channels))
