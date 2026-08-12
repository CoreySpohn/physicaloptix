"""plot_path: the typed optical-train rail + per-stage field panels.

The replacement for the frozen ``_legacy.render_path``: the rail is drawn by
``eyepiece.rail`` (a fixed, reviewed glyph vocabulary) instead of a hand-rolled
patch set, and every panel is chromatic-safe -- it goes through
``fields._resolve`` / ``fields._draw_panel``, the same machinery
``plot_field`` uses, so a chromatic ``Field`` (the bug ``render_path`` is
frozen with) just works.

``native_dpi`` is the named replacement for a dpi hack (render an image
panel's pixels 1:1 at a given inch height) that used to be copy-pasted in an
analysis script.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure

from physicaloptix.elements import MultiScaleVortex, SampledOptic
from physicaloptix.system import OpticalSystem
from physicaloptix.transforms import Fraunhofer
from physicaloptix.viz import _require
from physicaloptix.viz.fields import (
    _draw_panel,
    _intensity_from,
    _label_plane,
    _resolve,
)

# Legacy kind (physicaloptix's own vocabulary, from _infer_kind) -> eyepiece's
# rail glyph vocabulary (eyepiece.schematic.GLYPHS).
_GLYPH_FOR_KIND = {
    "source": "source",
    "pupil_mask": "mask",
    "lyot_stop": "lyot",
    "apodizer": "apodizer",
    "fpm": "fpm",
    "detector": "detector",
}

_LOG_FLOOR_RATIO = 1e-8
_PHASE_FLOOR = 1e-20  # unused by the phase branch of _draw_panel; kept explicit
_FIG_WIDTH_PER_PANEL_IN = 2.2  # matches the legacy render_path default (2.2 * n)
_RAIL_HEIGHT_RATIO = 0.6


def _infer_kind(op, name):
    """A display glyph kind for a stage, from its type and name.

    Ported verbatim from ``_legacy.render_path._infer_kind`` (frozen file,
    not importable without dragging in the deprecated module's matplotlib
    import guard).
    """
    if isinstance(op, MultiScaleVortex):
        return "fpm"
    if isinstance(op, Fraunhofer):
        return "detector"
    if isinstance(op, SampledOptic):
        lowered = name.lower()
        if "lyot" in lowered:
            return "lyot_stop"
        if "apod" in lowered:
            return "apodizer"
        return "pupil_mask"
    return "pupil_mask"


def native_dpi(n_pix, panel_height_in=1.6):
    """The dpi that renders ``n_pix`` pixels 1:1 at a panel this tall.

    Args:
        n_pix: Number of pixels along the panel's side.
        panel_height_in: Panel height in inches.

    Returns:
        The smallest integer dpi at which ``n_pix`` pixels span
        ``panel_height_in`` inches without downsampling.
    """
    return int(np.ceil(n_pix / panel_height_in))


def _validate_fig(fig):
    """Raise if ``fig`` is neither None nor a Figure/SubFigure plot_path owns.

    Raises:
        TypeError: ``fig`` is some other object (e.g. a bare Axes or an
            Axes array) -- named explicitly since a caller who wants to
            embed into an existing layout needs a SubFigure, not an Axes.
    """
    if fig is not None and not isinstance(fig, (Figure, SubFigure)):
        msg = (
            f"plot_path: fig= must be a Figure or SubFigure, got "
            f"{type(fig).__name__}; pass a Figure for plot_path to own, a "
            "SubFigure to embed into (e.g. from fig.subfigures(...)), or "
            "leave fig=None"
        )
        raise TypeError(msg)


def _row_layout(rail, show_phase, panel_height_in):
    """``(rows, height_ratios)`` for the gridspec of one rail+panel block."""
    heights = [panel_height_in]
    if show_phase:
        heights.append(panel_height_in)
    if rail:
        heights = [_RAIL_HEIGHT_RATIO, *heights]
    return len(heights), heights


def _owned_figsize(n_cols, height_ratios):
    width = _FIG_WIDTH_PER_PANEL_IN * max(n_cols, 1)
    height = sum(height_ratios) + 0.8
    return width, height


def _rail_positions(n, rail_kw):
    """X positions for ``n`` rail planes; caller-supplied ``positions`` wins."""
    if rail_kw and "positions" in rail_kw:
        return list(rail_kw["positions"])
    if n == 1:
        return [0.5]
    return list(np.linspace(0.10, 0.90, n))


def _draw_rail(ep, target, gs, stages, rail_kw, highlight):
    """Draw the rail spanning the top gridspec row, with plane sublabels.

    ``eyepiece.rail`` owns the glyph + label; the ``PlaneKind.value`` text
    under each plane is physicaloptix's own decoration, added post-hoc so it
    lines up with the exact x position each plane was drawn at.
    """
    rail_ax = target.add_subplot(gs[0, :])
    positions = _rail_positions(len(stages), rail_kw)
    rail_kwargs = dict(rail_kw or {})
    rail_kwargs["positions"] = positions
    if highlight is not None:
        rail_kwargs.setdefault("highlight", highlight)
    planes = [(name, _GLYPH_FOR_KIND[kind]) for name, kind, _ in stages]
    rail_result = ep.rail(planes, ax=rail_ax, **rail_kwargs)
    for x, (_, _, item) in zip(positions, stages, strict=True):
        rail_ax.text(
            x, -0.06, item.plane.value, ha="center", va="top", fontsize=7, color="0.5"
        )
    return rail_ax, rail_result.artists


def _draw_stage_panel(ep, ax, item, kind_of_panel, override, imshow_kw):
    """Draw one panel (intensity or phase) for a tapped Field, via fields.py.

    Args:
        ep: The eyepiece module.
        ax: Axes to draw into.
        item: The tapped ``Field``.
        kind_of_panel: "intensity" or "phase".
        override: An optional ``(vmin, vmax)`` pair overriding the default
            per-panel log floor (intensity only; ignored for phase).
        imshow_kw: Extra kwargs forwarded to the underlying ``imshow``.

    Returns:
        The drawn image artist.
    """
    data, extent, plane = _resolve(item, None, kind_of_panel)
    vmin = vmax = None
    floor = _PHASE_FLOOR
    if kind_of_panel == "intensity":
        peak = float(_intensity_from(data).max())
        if override is not None:
            vmin, vmax = override
            floor = vmin
        else:
            floor = peak * _LOG_FLOOR_RATIO
    result, _, _ = _draw_panel(
        ep,
        ax,
        data,
        kind_of_panel,
        extent,
        None,
        vmin,
        vmax,
        floor,
        False,
        None,
        imshow_kw,
        None,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    _label_plane(ep, result.ax, plane)
    return result.artists["image"]


def _draw_block(
    ep,
    target,
    stages,
    *,
    rail,
    show_phase,
    panel_norms,
    rail_kw,
    imshow_kw,
    panel_height_in,
    highlight=None,
):
    """Draw one rail+panels block (a whole ``OpticalPath``, or one branch row).

    Returns:
        An ``ep.MosaicResult`` with flat ``axes`` ``[rail?, panel_0, ...,
        panel_n, phase_0?, ..., phase_n?]`` and ``artists`` with ``"image"``
        (flat list, intensity panels then phase panels) and, when ``rail``
        is True, ``"rail"`` (the rail's own artists dict).
    """
    n = len(stages)
    rows, height_ratios = _row_layout(rail, show_phase, panel_height_in)
    gs = target.add_gridspec(rows, n, height_ratios=height_ratios)

    axes = []
    artists = {}
    row = 0
    if rail:
        rail_ax, rail_artists = _draw_rail(ep, target, gs, stages, rail_kw, highlight)
        axes.append(rail_ax)
        artists["rail"] = rail_artists
        row = 1

    panel_norms = panel_norms or ()
    images = []
    panel_axes = []
    for col, (_, _, item) in enumerate(stages):
        ax = target.add_subplot(gs[row, col])
        override = panel_norms[col] if col < len(panel_norms) else None
        images.append(_draw_stage_panel(ep, ax, item, "intensity", override, imshow_kw))
        panel_axes.append(ax)
    axes.extend(panel_axes)

    if show_phase:
        phase_axes = []
        for col, (_, _, item) in enumerate(stages):
            ax = target.add_subplot(gs[row + 1, col])
            images.append(_draw_stage_panel(ep, ax, item, "phase", None, imshow_kw))
            phase_axes.append(ax)
        axes.extend(phase_axes)

    artists["image"] = images
    return ep.MosaicResult(axes=np.array(axes, dtype=object), artists=artists)


def _collect_path_stages(path, field, taps, kinds):
    """``(name, kind, Field)`` triples: the input first, then every tap.

    ``taps=None`` means every stage; ``taps=()`` means the input panel only.
    """
    names = tuple(stage.name for stage in path.stages)
    if taps is None:
        taps = names
    _, tapped = path.propagate(field, taps=taps)
    overrides = kinds or {}
    stages = [("input", "source", field)]
    for stage in path.stages:
        if stage.name in taps:
            kind = overrides.get(stage.name, _infer_kind(stage.op, stage.name))
            stages.append((stage.name, kind, tapped[stage.name]))
    return stages


def _collect_system_stages(system, field, taps, kinds):
    """Per-branch ``(name, kind, Field)`` stage lists, trunk taps shared.

    Each branch's list is ``[input, *trunk taps, *that branch's own taps]``
    -- a self-contained "you are here" train, so ``_draw_block`` can render
    each branch's row without knowing about the others.

    Returns:
        A dict mapping branch name to its stage list, in ``system.branches``
        order.
    """
    trunk_names = tuple(stage.name for stage in system.trunk.stages)
    branch_stage_names = {
        branch.name: tuple(s.name for s in branch.path.stages)
        for branch in system.branches
    }
    if taps is None:
        taps = tuple(f"trunk/{n}" for n in trunk_names) + tuple(
            f"{b}/{n}" for b, names in branch_stage_names.items() for n in names
        )
    _, tapped = system.propagate(field, taps=taps)
    overrides = kinds or {}

    trunk_ops = {stage.name: stage.op for stage in system.trunk.stages}
    branch_ops = {
        branch.name: {stage.name: stage.op for stage in branch.path.stages}
        for branch in system.branches
    }

    trunk_stages = [("input", "source", field)]
    for name in trunk_names:
        tap = f"trunk/{name}"
        if tap in taps:
            kind = overrides.get(tap, _infer_kind(trunk_ops[name], name))
            trunk_stages.append((name, kind, tapped[tap]))

    per_branch = {}
    for branch in system.branches:
        stages = list(trunk_stages)
        for name in branch_stage_names[branch.name]:
            tap = f"{branch.name}/{name}"
            if tap in taps:
                kind = overrides.get(
                    tap, _infer_kind(branch_ops[branch.name][name], name)
                )
                stages.append((name, kind, tapped[tap]))
        per_branch[branch.name] = stages
    return per_branch, (trunk_names[-1] if trunk_names else None)


def _plot_system(
    ep,
    system,
    field,
    fig,
    *,
    taps,
    rail,
    show_phase,
    panel_norms,
    panel_height_in,
    kinds,
    rail_kw,
    imshow_kw,
):
    """One rail+panel block per branch, stacked as rows of subfigures."""
    per_branch, split_highlight = _collect_system_stages(system, field, taps, kinds)
    branch_names = [branch.name for branch in system.branches]
    n_cols = max(len(per_branch[name]) for name in branch_names)
    _, height_ratios = _row_layout(rail, show_phase, panel_height_in)

    if fig is None:
        width, block_height = _owned_figsize(n_cols, height_ratios)
        fig = plt.figure(
            layout="constrained", figsize=(width, block_height * len(branch_names))
        )

    outer_gs = fig.add_gridspec(len(branch_names), 1)
    axes = []
    images = []
    rails = {}
    for row_idx, name in enumerate(branch_names):
        sub = fig.add_subfigure(outer_gs[row_idx, 0])
        sub.suptitle(name, fontsize=9)
        block = _draw_block(
            ep,
            sub,
            per_branch[name],
            rail=rail,
            show_phase=show_phase,
            panel_norms=panel_norms,
            rail_kw=rail_kw,
            imshow_kw=imshow_kw,
            panel_height_in=panel_height_in,
            highlight=split_highlight,
        )
        axes.extend(block.axes.tolist())
        images.extend(block.artists["image"])
        if rail:
            rails[name] = block.artists["rail"]

    artists = {"image": images}
    if rail:
        artists["rail"] = rails
    return ep.MosaicResult(axes=np.array(axes, dtype=object), artists=artists)


def plot_path(
    path_or_system,
    field,
    *,
    taps=None,
    fig=None,
    rail=True,
    show_phase=False,
    panel_norms=None,
    panel_height_in=1.6,
    kinds=None,
    rail_kw=None,
    imshow_kw=None,
):
    """Propagate with the requested stages tapped and draw the rail + panels.

    Consumes an ``OpticalPath`` (a single rail+panel block) or an
    ``OpticalSystem`` (one rail+panel block per branch, stacked as
    subfigure rows -- taps are namespaced ``"trunk/<stage>"`` /
    ``"<branch>/<stage>"``). Every panel goes through ``fields._resolve``,
    so a chromatic ``Field`` renders its weight-summed intensity instead of
    crashing (the bug ``_legacy.render_path`` is frozen with).

    The returned axes are flat, in this fixed order: the rail (if
    ``rail=True``), then one intensity panel per stage (input first, then
    every tapped stage in path order), then, if ``show_phase=True``, one
    phase panel per stage in the same order. For an ``OpticalSystem``, this
    whole sequence repeats once per branch, branch by branch.

    Args:
        path_or_system: An ``OpticalPath`` or an ``OpticalSystem``.
        field: The input field (must match the first stage's plane).
        taps: Static tuple of stage names (namespaced for an
            ``OpticalSystem``) whose output fields get their own panel.
            ``None`` (default) taps every stage; ``()`` taps none, so only
            the input panel is drawn.
        fig: A ``Figure`` for ``plot_path`` to own, or a ``SubFigure`` to
            embed into (e.g. one cell of ``fig.subfigures(...)``). ``None``
            creates and owns a new ``Figure``. Anything else (a bare Axes,
            an Axes array) raises ``TypeError`` naming ``SubFigure`` as the
            fix.
        rail: Whether to draw the schematic rail spanning the top row.
        show_phase: Append a row of phase panels, NaN-masked below
            ``1e-10`` of each panel's peak intensity (the phase is
            undefined where there is no light).
        panel_norms: Optional sequence of ``(vmin, vmax)`` pairs, one per
            intensity panel in the same order as the panels (``None``
            entries, or a shorter sequence, fall back to the default
            per-panel log floor at ``peak * 1e-8``).
        panel_height_in: Height, in inches, of one panel row (also the
            gridspec height ratio unit; the rail row is a fixed ``0.6``).
        kinds: Optional ``{stage_name: glyph_kind}`` overrides (namespaced
            for an ``OpticalSystem``); kinds are otherwise inferred from
            the stage's element type (source, pupil_mask, apodizer, fpm,
            lyot_stop, detector) via the ported legacy ``_infer_kind``.
        rail_kw: Extra kwargs forwarded to ``eyepiece.rail`` (e.g.
            ``positions``, ``accent``, ``cap``).
        imshow_kw: Extra kwargs forwarded to every panel's underlying
            ``imshow`` call.

    Returns:
        An ``eyepiece.MosaicResult`` with the flat axes order described
        above. ``artists["image"]`` is the flat list of image artists in
        the same order as the (non-rail) axes. ``artists["rail"]`` is the
        rail's own artists dict (``OpticalPath``), or a ``{branch_name:
        artists}`` dict (``OpticalSystem``); absent when ``rail=False``.

    Raises:
        TypeError: ``fig`` is neither ``None`` nor a ``Figure``/``SubFigure``.
    """
    ep = _require.eyepiece()
    _validate_fig(fig)

    if isinstance(path_or_system, OpticalSystem):
        return _plot_system(
            ep,
            path_or_system,
            field,
            fig,
            taps=taps,
            rail=rail,
            show_phase=show_phase,
            panel_norms=panel_norms,
            panel_height_in=panel_height_in,
            kinds=kinds,
            rail_kw=rail_kw,
            imshow_kw=imshow_kw,
        )

    stages = _collect_path_stages(path_or_system, field, taps, kinds)
    _, height_ratios = _row_layout(rail, show_phase, panel_height_in)
    if fig is None:
        fig = plt.figure(
            layout="constrained", figsize=_owned_figsize(len(stages), height_ratios)
        )
    return _draw_block(
        ep,
        fig,
        stages,
        rail=rail,
        show_phase=show_phase,
        panel_norms=panel_norms,
        rail_kw=rail_kw,
        imshow_kw=imshow_kw,
        panel_height_in=panel_height_in,
    )
