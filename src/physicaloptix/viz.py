"""render_path: the optical-path cartoon plus per-stage field panels.

Because taps carry name, plane, and grid, the renderer consumes a tapped
propagation directly: a schematic rail of element glyphs on top and, aligned
under each element, the intensity (and optionally phase) of the field just
after it, in plane-native extents. Requires matplotlib (a plotting extra,
not a core dependency).
"""

import numpy as np

from physicaloptix.elements import MultiScaleVortex, SampledOptic
from physicaloptix.transforms import Fraunhofer

GLYPH_COLOR = "0.25"
ACCENT = "#b23a6f"


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import FancyArrow, Polygon, Rectangle
    except ImportError as err:
        raise ImportError(
            "render_path requires matplotlib (pip install matplotlib)"
        ) from err
    return plt, LogNorm, FancyArrow, Polygon, Rectangle


def _glyph(ax, kind, x, patches):
    """Draw one element glyph centered at axis coordinate x in [0, 1]."""
    _, _, _, polygon, rectangle = patches
    if kind == "source":
        ax.plot([x], [0.5], marker="*", ms=16, color=ACCENT, zorder=3)
    elif kind in ("pupil_mask", "lyot_stop"):
        for y0, y1 in ((0.05, 0.32), (0.68, 0.95)):
            ax.add_patch(
                rectangle(
                    (x - 0.006, y0),
                    0.012,
                    y1 - y0,
                    facecolor=GLYPH_COLOR,
                    edgecolor="none",
                    zorder=3,
                )
            )
    elif kind == "apodizer":
        ax.add_patch(
            rectangle(
                (x - 0.006, 0.15),
                0.012,
                0.70,
                facecolor=GLYPH_COLOR,
                alpha=0.45,
                edgecolor="none",
                zorder=3,
            )
        )
    elif kind == "fpm":
        ax.add_patch(
            polygon(
                [(x, 0.30), (x + 0.018, 0.5), (x, 0.70), (x - 0.018, 0.5)],
                facecolor=ACCENT,
                edgecolor="none",
                zorder=3,
            )
        )
    elif kind == "detector":
        ax.add_patch(
            rectangle(
                (x - 0.012, 0.30),
                0.024,
                0.40,
                facecolor="none",
                edgecolor=GLYPH_COLOR,
                lw=1.6,
                hatch="///",
                zorder=3,
            )
        )


def _infer_kind(op, name):
    """A display glyph for a stage, from its type and name."""
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


def _panel(field):
    """The (name-free) render dict for one tapped field."""
    extent = field.grid.extent
    return {
        "field": np.asarray(field.data),
        "plane": field.plane.value,
        "extent": [-extent, extent, -extent, extent],
    }


def render_path(
    path,
    field,
    *,
    title=None,
    show_phase=False,
    panel_norm=None,
    figwidth=None,
    kinds=None,
):
    """Propagate with every stage tapped and render the rail + panels.

    Args:
        path: The ``OpticalPath`` to visualize.
        field: The input field.
        title: Optional figure title.
        show_phase: Add a row of phase panels under the intensity row.
        panel_norm: Default matplotlib norm for the intensity panels.
        figwidth: Figure width in inches (default 2.2 per stage).
        kinds: Optional ``{stage_name: glyph_kind}`` overrides; kinds are
            otherwise inferred from the stage types (source, pupil_mask,
            apodizer, fpm, lyot_stop, detector).

    Returns:
        The matplotlib figure.
    """
    patches = _matplotlib()
    plt, log_norm, fancy_arrow = patches[0], patches[1], patches[2]

    names = tuple(stage.name for stage in path.stages)
    _, tapped = path.propagate(field, taps=names)

    stages = [{"name": "input", "kind": "source", **_panel(field)}]
    overrides = kinds or {}
    for stage in path.stages:
        stages.append(
            {
                "name": stage.name,
                "kind": overrides.get(stage.name, _infer_kind(stage.op, stage.name)),
                **_panel(tapped[stage.name]),
            }
        )

    n = len(stages)
    width = figwidth or 2.2 * n
    rows = 3 if show_phase else 2
    heights = [0.6, 1.6, 1.6][:rows]
    fig, axes = plt.subplots(
        rows,
        n,
        figsize=(width, sum(heights) + 0.8),
        gridspec_kw={"height_ratios": heights},
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    # The rail: one shared schematic across the top row.
    rail = fig.add_subplot(axes[0, 0].get_gridspec()[0, :])
    for ax in axes[0]:
        ax.set_visible(False)
    xs = (np.arange(n) + 0.5) / n
    rail.add_patch(
        fancy_arrow(
            0.01,
            0.5,
            0.97,
            0.0,
            width=0.004,
            head_width=0.06,
            head_length=0.012,
            length_includes_head=True,
            color="0.6",
        )
    )
    for x, st in zip(xs, stages, strict=True):
        _glyph(rail, st["kind"], x, patches)
        rail.text(x, 1.02, st["name"], ha="center", va="bottom", fontsize=9)
        rail.text(x, -0.06, st["plane"], ha="center", va="top", fontsize=7, color="0.5")
    rail.set_xlim(0, 1)
    rail.set_ylim(0, 1)
    rail.axis("off")
    if title:
        rail.set_title(title, fontsize=11, pad=26)

    # Field panels.
    for col, st in enumerate(stages):
        data = st["field"]
        inten = np.abs(data) ** 2
        ax = axes[1, col]
        norm = (
            st.get("norm")
            or panel_norm
            or log_norm(max(inten.max() * 1e-8, inten[inten > 0].min()), inten.max())
        )
        ax.imshow(inten, norm=norm, extent=st["extent"], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(
            {"pupil": "pupil plane", "focal": "focal (lambda/D)"}.get(st["plane"], ""),
            fontsize=7,
        )
        if show_phase:
            axp = axes[2, col]
            phase = np.where(inten > inten.max() * 1e-10, np.angle(data), np.nan)
            axp.imshow(
                phase,
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
                extent=st["extent"],
                interpolation="nearest",
            )
            axp.set_xticks([])
            axp.set_yticks([])
            axp.set_xlabel("phase", fontsize=7)
    return fig
