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

import hwostyle
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
# Passed as the floor= positional arg on the phase path too, but _draw_panel's
# phase branch (-> fields._plot_phase) never reads it -- only the intensity
# branch does. Kept as an explicit placeholder rather than None so the call
# signature stays uniform across both kinds.
_PHASE_FLOOR = 1e-20
_FIG_WIDTH_PER_PANEL_IN = 2.2  # matches the legacy render_path default (2.2 * n)
_RAIL_HEIGHT_RATIO = 0.6
_MINIMAP_CHIP_ALPHA = 0.55  # translucent watermark chip behind the rail
_MINIMAP_MUTE_ALPHA = 0.35  # extra fade on every plane but the active one


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


def _planes_for_stages(name_kind_pairs):
    """``(name, glyph)`` rail entries from ``(name, kind)`` pairs.

    The one place the legacy-kind -> eyepiece-glyph mapping
    (``_GLYPH_FOR_KIND``) is applied, shared by ``plot_path``'s
    ``_draw_rail`` and ``minimap`` so the two never build a second,
    divergent path from stages to rail entries. A ``kind`` not in
    ``_GLYPH_FOR_KIND`` (e.g. a ``kinds=`` override already spelled in
    eyepiece's own glyph vocabulary, such as ``"focal"`` or ``"pupil"``)
    passes through unchanged rather than raising a bare ``KeyError`` here;
    ``eyepiece.rail`` itself validates the final glyph name and raises a
    ``ValueError`` naming the accepted set for anything still unknown.
    """
    return [(name, _GLYPH_FOR_KIND.get(kind, kind)) for name, kind in name_kind_pairs]


def _stage_kinds(path, kinds=None):
    """``(name, kind)`` pairs for every stage of ``path``, input first.

    The same traversal ``_collect_path_stages`` uses to build the
    ``(name, kind, Field)`` rows ``plot_path`` taps into, minus the field
    propagation -- ``minimap`` only needs labels and glyphs for the rail,
    never a field to draw a panel from.
    """
    overrides = kinds or {}
    pairs = [("input", "source")]
    for stage in path.stages:
        kind = overrides.get(stage.name, _infer_kind(stage.op, stage.name))
        pairs.append((stage.name, kind))
    return pairs


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
    """X positions for ``n`` rail planes; caller-supplied ``positions`` wins.

    The rail axes spans the SAME gridspec width as the ``n`` panel columns
    below it (``gs[0, :]`` over the same ``n``-column row), so its [0, 1]
    axes-fraction coordinate maps linearly onto that combined width. The
    default is therefore each column's own center, ``(i + 0.5) / n`` -- the
    legacy ``render_path`` recipe (``_legacy.py``'s ``xs = (np.arange(n) +
    0.5) / n``) -- so every glyph sits directly above its panel; anything
    else (e.g. an even ``linspace`` padded away from the edges) drifts off
    by a growing fraction of a panel width as ``n`` shrinks.
    """
    if rail_kw and "positions" in rail_kw:
        return list(rail_kw["positions"])
    return list((np.arange(n) + 0.5) / n)


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
    planes = _planes_for_stages([(name, kind) for name, kind, _ in stages])
    rail_result = ep.rail(planes, ax=rail_ax, **rail_kwargs)
    for x, (_, _, item) in zip(positions, stages, strict=True):
        rail_ax.text(
            x, -0.06, item.plane.value, ha="center", va="top", fontsize=7, color="0.5"
        )
    return rail_ax, rail_result.artists


def _phase_channel(item, channel):
    """The chromatic channel index a phase panel draws, or None for mono.

    ``channel=None`` (the default) auto-selects the middle band -- a
    chromatic Field has no single phase, so *some* deterministic choice is
    needed for the phase row, and the intensity row already gets the
    physically meaningful chromatic answer (the weight-summed intensity)
    for free. A caller who wants a specific band passes ``channel=``.

    Raises:
        ValueError: ``channel`` is out of range for this Field's band
            count -- named and range-listing, rather than the bare
            ``IndexError`` a raw ``data[channel]`` would raise deeper
            inside ``fields._resolve``.
    """
    data = np.asarray(item.data)
    if data.ndim != 3:
        return None
    nlam = data.shape[0]
    if channel is None:
        return nlam // 2
    if not (0 <= channel < nlam):
        msg = (
            f"plot_path: channel={channel} out of range for a "
            f"{nlam}-band chromatic Field; valid range is 0..{nlam - 1}"
        )
        raise ValueError(msg)
    return channel


def _draw_stage_panel(ep, ax, item, kind_of_panel, override, imshow_kw, channel=None):
    """Draw one panel (intensity or phase) for a tapped Field, via fields.py.

    Args:
        ep: The eyepiece module.
        ax: Axes to draw into.
        item: The tapped ``Field``.
        kind_of_panel: "intensity" or "phase".
        override: An optional ``(vmin, vmax)`` pair overriding the default
            per-panel log floor (intensity only; ignored for phase). Either
            entry may be ``None`` to fall back individually: the floor for
            ``vmin``, the panel's own peak for ``vmax``.
        imshow_kw: Extra kwargs forwarded to the underlying ``imshow``.
        channel: Wavelength index for a chromatic Field's phase row
            (ignored for intensity, which always shows the weight-summed
            answer, and for a mono Field). ``None`` auto-selects the middle
            band; see ``_phase_channel``.

    Returns:
        The drawn image artist.
    """
    is_phase = kind_of_panel == "phase"
    resolve_channel = _phase_channel(item, channel) if is_phase else None
    data, extent, plane = _resolve(item, resolve_channel, kind_of_panel)
    vmin = vmax = None
    floor = _PHASE_FLOOR
    if kind_of_panel == "intensity":
        peak = float(_intensity_from(data).max())
        default_floor = peak * _LOG_FLOOR_RATIO
        if override is not None:
            ov_vmin, ov_vmax = override
            vmin = default_floor if ov_vmin is None else ov_vmin
            # ov_vmax=None passes through: imshow_log then derives it itself.
            vmax = ov_vmax
            floor = vmin
        else:
            floor = default_floor
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
    channel=None,
    gs=None,
    label=None,
):
    """Draw one rail+panels block (a whole ``OpticalPath``, or one branch row).

    Every axes this draws is added directly to ``target`` (via
    ``target.add_subplot``), so ``axes.figure is target`` for every one of
    them regardless of whether ``target`` is the top-level owned ``Figure``
    or a caller-supplied ``SubFigure`` -- see ``gs``.

    Args:
        ep: The eyepiece module.
        target: The Figure or SubFigure every axes is added to.
        stages: ``(name, kind, Field)`` triples, input first, in draw order.
        rail: Whether to draw the schematic rail spanning the top row.
        show_phase: Whether to append a row of phase panels.
        panel_norms: Optional sequence of ``(vmin, vmax)`` overrides, one
            per intensity panel; see ``plot_path``.
        rail_kw: Extra kwargs forwarded to ``eyepiece.rail``.
        imshow_kw: Extra kwargs forwarded to every panel's ``imshow``.
        panel_height_in: Height, in inches, of one panel row.
        highlight: Plane label to draw in the accent color on the rail, or
            None. Must be one of ``stages``' own names -- ``eyepiece.rail``
            raises if it is not.
        channel: Wavelength index for a chromatic Field's phase row; see
            ``plot_path``.
        gs: A pre-built gridspec (or ``SubplotSpec`` gridspec, from
            ``outer_gs[row, 0].subgridspec(...)``) sized ``(rows, len(stages))``
            to draw this block's cells into. ``None`` (the ``OpticalPath``
            case) builds one fresh via ``target.add_gridspec(...)``; an
            ``OpticalSystem`` branch row instead passes its own nested
            subgridspec so every branch's axes still resolve to the ONE
            owned/caller-supplied Figure or SubFigure, never a
            per-branch ``SubFigure`` (whose axes would (a) make
            ``MosaicResult.fig`` unreachable as the actual owned figure
            and (b) have no ``savefig`` of their own).
        label: Optional text identifying this block (e.g. a branch name),
            so a multi-block figure can tell its blocks apart without a
            per-block ``SubFigure``. Always applied -- as the rail axes'
            title when ``rail=True``, or the first panel's title when
            ``rail=False`` (there is then no rail axes to carry it), so a
            ``rail=False`` ``OpticalSystem`` render does not lose branch
            labels entirely.

    Returns:
        An ``ep.MosaicResult`` with flat ``axes`` ``[rail?, panel_0, ...,
        panel_n, phase_0?, ..., phase_n?]`` and ``artists`` with ``"image"``
        (flat list, intensity panels then phase panels) and, when ``rail``
        is True, ``"rail"`` (the rail's own artists dict).
    """
    n = len(stages)
    rows, height_ratios = _row_layout(rail, show_phase, panel_height_in)
    if gs is None:
        gs = target.add_gridspec(rows, n, height_ratios=height_ratios)

    axes = []
    artists = {}
    row = 0
    label_ax = None
    if rail:
        rail_ax, rail_artists = _draw_rail(ep, target, gs, stages, rail_kw, highlight)
        axes.append(rail_ax)
        artists["rail"] = rail_artists
        row = 1
        label_ax = rail_ax

    panel_norms = panel_norms or ()
    images = []
    panel_axes = []
    for col, (_, _, item) in enumerate(stages):
        ax = target.add_subplot(gs[row, col])
        override = panel_norms[col] if col < len(panel_norms) else None
        images.append(_draw_stage_panel(ep, ax, item, "intensity", override, imshow_kw))
        panel_axes.append(ax)
    axes.extend(panel_axes)

    if label is not None:
        # No rail axes to carry the label when rail=False -- fall back to
        # the block's first panel so an OpticalSystem render never loses
        # branch identity just because rail was turned off.
        (label_ax if label_ax is not None else panel_axes[0]).set_title(
            label, fontsize=9, loc="left"
        )

    if show_phase:
        phase_axes = []
        for col, (_, _, item) in enumerate(stages):
            ax = target.add_subplot(gs[row + 1, col])
            images.append(
                _draw_stage_panel(ep, ax, item, "phase", None, imshow_kw, channel)
            )
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
    channel,
):
    """One rail+panel block per branch, stacked as nested-gridspec rows.

    Every branch's axes are added directly to ``fig`` (via nested
    ``SubplotSpec.subgridspec`` cells, never ``fig.add_subfigure(...)``),
    so ``MosaicResult.fig`` (``axes.flat[0].figure``) resolves to the real
    owned/caller-supplied ``fig`` -- a ``SubFigure`` has no ``savefig`` and
    is not accepted by ``plt.close``, so a per-branch ``SubFigure`` would
    make the returned figure unusable for either.
    """
    per_branch, split_name = _collect_system_stages(system, field, taps, kinds)
    branch_names = [branch.name for branch in system.branches]
    n_cols = max(len(per_branch[name]) for name in branch_names)
    rows, height_ratios = _row_layout(rail, show_phase, panel_height_in)

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
        stages = per_branch[name]
        # split_name is a candidate (the last REQUESTED trunk stage overall)
        # that this particular branch's own stage list may not contain --
        # e.g. taps= omits it, or omits this branch's downstream taps
        # entirely. eyepiece.rail validates highlight against the planes it
        # is actually given, so passing a name this block never drew raises.
        present = {stage_name for stage_name, _, _ in stages}
        highlight = split_name if split_name in present else None
        inner_gs = outer_gs[row_idx, 0].subgridspec(
            rows, max(len(stages), 1), height_ratios=height_ratios
        )
        block = _draw_block(
            ep,
            fig,
            stages,
            rail=rail,
            show_phase=show_phase,
            panel_norms=panel_norms,
            rail_kw=rail_kw,
            imshow_kw=imshow_kw,
            panel_height_in=panel_height_in,
            highlight=highlight,
            channel=channel,
            gs=inner_gs,
            label=name,
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
    channel=None,
    rail_kw=None,
    imshow_kw=None,
):
    """Propagate with the requested stages tapped and draw the rail + panels.

    Consumes an ``OpticalPath`` (a single rail+panel block) or an
    ``OpticalSystem`` (one rail+panel block per branch, stacked as nested
    gridspec rows -- taps are namespaced ``"trunk/<stage>"`` /
    ``"<branch>/<stage>"``). Every intensity panel goes through
    ``fields._resolve``, so a chromatic ``Field`` renders its weight-summed
    intensity instead of crashing (the bug ``_legacy.render_path`` is
    frozen with).

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
            intensity panel in the same order as the panels. ``None`` (or a
            shorter sequence) falls back to the default per-panel log floor
            at ``peak * 1e-8``; within a given pair, either entry may
            individually be ``None`` to fall back just that bound (the
            floor for ``vmin``, the panel's own peak for ``vmax``).
        panel_height_in: Height, in inches, of one panel row (also the
            gridspec height ratio unit; the rail row is a fixed ``0.6``).
        kinds: Optional ``{stage_name: glyph_kind}`` overrides (namespaced
            for an ``OpticalSystem``); kinds are otherwise inferred from
            the stage's element type (source, pupil_mask, apodizer, fpm,
            lyot_stop, detector) via the ported legacy ``_infer_kind``. A
            value already spelled in eyepiece's own glyph vocabulary (e.g.
            ``"focal"``) passes through unchanged.
        channel: Wavelength index selecting which band's phase to draw for
            a chromatic Field's phase row (``show_phase=True`` only; a mono
            Field, or the intensity row of any Field, ignores it -- the
            intensity row always shows the physically meaningful
            weight-summed answer). ``None`` (the default) auto-selects the
            middle band, so ``show_phase=True`` never raises on a chromatic
            Field.
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
        ``.fig`` is always the real owned or caller-supplied ``Figure``/
        ``SubFigure`` -- never a per-branch wrapper -- for both input kinds.

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
            channel=channel,
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
        channel=channel,
    )


def _minimap_accent():
    """The color ``minimap`` highlights ``active`` in, resolved fresh.

    Mirrors ``fields._ring_color``: ``hwostyle.palette`` carries only
    per-family named hues -- no registered family (cyberpunk, spectral,
    biosignature, tol, barbie) defines a universal "primary" swatch every
    other family also has. A fixed attribute name is therefore not safe:
    ``hwostyle.use("paper")`` (the ordinary way this repo makes
    publication figures) defaults to the "tol" family, and "tol" has no
    "cyan" key, so a bare ``hwostyle.palette.cyan`` raises ``AttributeError``
    there. Falling back to the family's own first color
    (``Palette.as_list``, guaranteed non-empty by every registry entry)
    keeps this resolved fresh on every call -- never bound at import --
    instead of crashing under a non-default family.
    """
    return getattr(hwostyle.palette, "cyan", hwostyle.palette.as_list[0])


def minimap(path, *, ax=None, active=None):
    """A greyed "you are here" rail over ``path``'s planes.

    Draws the same ``(name, glyph)`` planes ``plot_path``'s rail draws --
    the input first, then every stage of ``path`` in order -- with
    ``active`` picked out in the accent color by ``eyepiece.rail`` and
    every other plane's marker and label further muted, plus a translucent
    chip behind the whole rail. The result reads as a small "you are here"
    orientation strip, not a standalone figure.

    This function never carves an inset out of a host axes -- the caller
    reserves the slot. Two ways to do that: give the block its own small
    gridspec cell sized for a rail (``fig.add_gridspec(...)[row, -1]``,
    ``height_ratios``/``width_ratios`` pinning it thin), or carve the inset
    yourself over a quiet margin of an existing axes (e.g.
    ``host_ax.inset_axes([...])``) and pass the result in as ``ax=``.

    Args:
        path: An ``OpticalPath``; its stages become rail planes in order.
        ax: Axes to draw into. None creates and owns a new Figure.
        active: The "you are here" stage name (or ``"input"``), matched
            case-insensitively against the plane labels. None leaves every
            plane in the muted color.

    Returns:
        A ``PlotResult`` with the same ``artists`` keys as
        ``eyepiece.rail`` ("fill", "lines", "text").

    Raises:
        ValueError: ``active`` is not one of the plane names.
    """
    ep = _require.eyepiece()
    planes = _planes_for_stages(_stage_kinds(path))

    result = ep.rail(planes, ax=ax, highlight=active, accent=_minimap_accent())

    active_key = active.lower() if active is not None else None
    for (name, _), line, text in zip(
        planes, result.artists["lines"], result.artists["text"], strict=True
    ):
        if active_key is not None and name.lower() == active_key:
            continue
        line.set_alpha(_MINIMAP_MUTE_ALPHA)
        text.set_alpha(_MINIMAP_MUTE_ALPHA)

    result.ax.patch.set_alpha(_MINIMAP_CHIP_ALPHA)
    result.ax.axis("off")

    return result
