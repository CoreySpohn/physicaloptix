"""Speckle boiling in time: the epoch strip and the animation.

Two views of one thing -- a dark hole whose speckles drift -- typed on the
SPECKLE-FIELD PROTOCOL rather than on any concrete class. Anything with
``realize(wavelength_nm=..., time_s=...) -> (y, x)`` and a
``pixel_scale_lod`` drives both functions: the analytic generator, a
replayed testbed trajectory, a fitted reduced-order model, a maintained
residual, or a ten-line fake in a test. A precomputed ``(n_t, y, x)`` cube
enters through the same door, declaring ``sample_kind="instantaneous"`` and
``quantity="total"`` or ``"delta"``.

Both functions are conveniences over one pipeline: :func:`prepare_speckles`
computes the frames, the floor, the shared display scale, and the trace
once, and ``eyepiece.mpl`` draws the prepared sequence. Each call prepares
its own input, so code that wants a strip AND a movie of one run should
call ``prepare_speckles`` once and render ``sequence.strip(...)`` and
``eyepiece.mpl.animate(sequence, ...)`` from it.

**One scale across time, always.** The display scale is resolved once from
the whole time series and held for every panel and every frame; see
:mod:`physicaloptix.viz._prepare` for the floor and the bounds.

**The scale follows the declared quantity, not the data's sign.** A total
intensity takes a log scale; a delta (a field with ``include_floor=False``,
or a bare cube declared ``quantity="delta"``) takes a symmetric diverging
scale about zero, whether or not its values happen to go negative.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicaloptix.viz import _require
from physicaloptix.viz._prepare import (
    CLOCK_ID,
    IMAGE_ID,
    prepare_speckles,
)

# Image panel over trace panel: the trace is a companion, not a co-equal.
_TRACE_HEIGHT_RATIOS = (3.0, 1.0)
_PANEL_WIDTH_IN = 3.2
_PANEL_HEIGHT_IN = 3.2


def _pair_bounds(bounds, vmin, vmax):
    """``bounds=``, or the older ``vmin=``/``vmax=`` pair spelled out.

    Raises:
        ValueError: Both spellings are given, or only one end of the pair.
    """
    if vmin is None and vmax is None:
        return bounds
    if bounds is not None:
        msg = "boiling_strip: pass bounds= or vmin=/vmax=, not both"
        raise ValueError(msg)
    if vmin is None or vmax is None:
        msg = (
            "boiling_strip: pass both vmin= and vmax= (a delta's symmetric "
            "bounds are (-vmax, vmax)), or bounds=(vmin, vmax)"
        )
        raise ValueError(msg)
    return (vmin, vmax)


def boiling_strip(
    field,
    times_s,
    *,
    wavelength_nm=None,
    telescope_peak=None,
    include_floor=True,
    floor=None,
    pixscale_lod=None,
    bounds=None,
    vmin=None,
    vmax=None,
    sample_kind=None,
    quantity=None,
    axes=None,
    cast=None,
    profile=None,
):
    """Draw one dark-hole map per epoch, all on one shared scale.

    The still counterpart of :func:`animate_speckles`, and the figure that
    belongs in a paper: a strip of epochs reads as boiling only because
    every panel shares one scale, built from the whole time series.
    Per-panel scales would make a stable dark hole and a degrading one look
    the same. The frames are prepared by :func:`prepare_speckles` and drawn
    as ``sequence.strip`` of every epoch through ``eyepiece.mpl.render``.
    Each panel is labelled with its acquisition time. Because every panel
    shares one extent and one scale, only the last panel shows a colorbar
    and only the first labels its y axis.

    Args:
        field: A speckle field, or a precomputed ``(n_t, y, x)`` cube; see
            :func:`prepare_speckles`.
        times_s: Epoch times in seconds, one per panel.
        wavelength_nm: Wavelength passed to ``realize``.
        telescope_peak: Peak-reference the maps; see
            :func:`prepare_speckles`.
        include_floor: Add the deterministic floor to a field's frames.
        floor: Explicit floor map in flux-fraction units.
        pixscale_lod: Pixel scale in lambda/D per pixel.
        bounds: Display ``(vmin, vmax)``; a delta needs symmetric bounds.
            Pin both ends to read strips of different runs against one
            absolute window.
        vmin: Older spelling of ``bounds[0]``; give it with ``vmax``.
        vmax: Older spelling of ``bounds[1]``; give it with ``vmin``.
        sample_kind: Required ``"instantaneous"`` for a bare cube.
        quantity: Required ``"total"`` or ``"delta"`` for a bare cube.
        axes: Axes to draw into, shape ``(n,)`` for ``n = len(times_s)``.
            None creates a figure sized for ``n`` square panels.
        cast: ``eyepiece.style.SourceCast`` passed to the renderer.
        profile: ``eyepiece.style.RenderProfile`` passed to the renderer;
            None snapshots the current style once.

    Returns:
        An ``eyepiece.mpl.MplResult`` of the strip. Panel ``k``'s image is
        ``parts["k/image"]`` and its time label ``parts["k/time"]``.

    Raises:
        ValueError: ``axes`` of the wrong shape (naming both), or any
            condition :func:`prepare_speckles` refuses.
    """
    _require.eyepiece()
    import eyepiece.mpl as mpl

    sequence = prepare_speckles(
        field,
        times_s=times_s,
        wavelength_nm=wavelength_nm,
        telescope_peak=telescope_peak,
        include_floor=include_floor,
        floor=floor,
        pixscale_lod=pixscale_lod,
        bounds=_pair_bounds(bounds, vmin, vmax),
        sample_kind=sample_kind,
        quantity=quantity,
    )
    n = len(sequence.times)
    if axes is None:
        _, axes = plt.subplots(
            1,
            n,
            figsize=(_PANEL_WIDTH_IN * n, _PANEL_HEIGHT_IN),
            squeeze=False,
            layout="constrained",
        )
    axes = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    if axes.shape != (n,):
        msg = f"boiling_strip: expected axes shape ({n},), got {axes.shape}"
        raise ValueError(msg)

    result = mpl.render(sequence.strip(range(n)), axes=axes, cast=cast, profile=profile)
    # The strip adds its own per-panel time label, so each panel's clock
    # would say the same thing twice; one shared colorbar and one y label
    # serve every panel of a shared extent and scale.
    for slot, panel_ax in enumerate(axes):
        result.parts[f"{slot}/{CLOCK_ID}"].set_visible(False)
        if slot < n - 1:
            result.parts[f"{slot}/{IMAGE_ID}/colorbar"].ax.set_visible(False)
        if slot:
            panel_ax.set_ylabel("")
            panel_ax.tick_params(labelleft=False)
    return result


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
    bounds=None,
    sample_kind=None,
    quantity=None,
    fps=10,
    run_time=None,
    fig=None,
    cast=None,
    profile=None,
):
    """Animate a dark hole boiling, optionally over a contrast trace.

    Returns an ``eyepiece.Animation``, which is LAZY: the frames are
    prepared and the figure is built here, but nothing is encoded until the
    caller asks for a sink (``.save("run.mp4", "run.gif")``, ``.jshtml()``,
    ``.video()``). Every frame shares one scale, built once from the whole
    sequence by :func:`prepare_speckles`, so each animation frame only
    re-maps a prepared array and never re-evaluates the field.

    **Playback follows physical time.** The movie runs ``run_time`` seconds
    at ``fps``, over ``ceil(run_time * fps)`` output frames spaced evenly in
    PHYSICAL time from the first epoch to the last; each shows the most
    recent epoch acquired at or before its time (sample-and-hold). Omitting
    ``run_time`` gives ``len(times_s) / fps``, so uniformly spaced epochs
    still play one output frame per epoch, as before. Nonuniformly spaced
    epochs no longer do: a long gap between two epochs now holds the
    earlier one on screen for its share of the physical span, instead of
    every epoch getting one frame regardless of spacing.

    Args:
        field: A speckle field, or a precomputed ``(n_t, y, x)`` cube; see
            :func:`prepare_speckles`.
        times_s: Frame times in seconds.
        wavelength_nm: Wavelength passed to ``realize``.
        telescope_peak: Peak-reference the maps.
        include_floor: Add the deterministic floor to a field's frames.
        floor: Explicit floor map in flux-fraction units.
        pixscale_lod: Pixel scale in lambda/D per pixel.
        trace: Optional companion panel under the image: a TUPLE
            ``(iwa_lod, owa_lod)`` for the dark-zone mean per frame, or an
            already-computed ``(n_t,)`` series. A marker tracks the held
            epoch along it.
        bounds: Display ``(vmin, vmax)``; see :func:`prepare_speckles`.
        sample_kind: Required ``"instantaneous"`` for a bare cube.
        quantity: Required ``"total"`` or ``"delta"`` for a bare cube.
        fps: Output frame rate.
        run_time: Presentation duration in seconds; None gives
            ``len(times_s) / fps``.
        fig: Figure to build into. None creates one with a constrained
            layout, which keeps the colorbar and its label on the canvas.
        cast: ``eyepiece.style.SourceCast`` passed to the renderer.
        profile: ``eyepiece.style.RenderProfile`` passed to the renderer.

    Returns:
        An ``eyepiece.Animation``. Its ``.fig`` is the figure drawn into and
        ``.n_frames`` the output frame count.

    Raises:
        ValueError: Any condition :func:`prepare_speckles` refuses.
    """
    _require.eyepiece()
    import eyepiece.mpl as mpl

    sequence = prepare_speckles(
        field,
        times_s=times_s,
        wavelength_nm=wavelength_nm,
        telescope_peak=telescope_peak,
        include_floor=include_floor,
        floor=floor,
        pixscale_lod=pixscale_lod,
        trace=trace,
        bounds=bounds,
        sample_kind=sample_kind,
        quantity=quantity,
    )
    if run_time is None:
        run_time = len(sequence.times) / fps
    if fig is None:
        height = 6.0 if trace is not None else 4.8
        fig = plt.figure(figsize=(4.8, height), layout="constrained")
    if trace is not None:
        axes = fig.subplots(2, 1, height_ratios=_TRACE_HEIGHT_RATIOS)
        return mpl.animate(
            sequence, run_time=run_time, fps=fps, axes=axes, cast=cast, profile=profile
        )
    return mpl.animate(
        sequence,
        run_time=run_time,
        fps=fps,
        ax=fig.subplots(),
        cast=cast,
        profile=profile,
    )
