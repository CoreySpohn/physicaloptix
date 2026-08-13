"""Mode gallery: an OPD mode beside the focal-plane speckle it produces.

The causal-pair figure of the whole speckle story -- a ripple on the pupil in
the left column, the speckle pair it puts in the dark hole in the right one,
row by row. The pairing is the content, so the two columns are drawn on
deliberately different terms: the OPD is SIGNED, so it takes a white-centered
diverging map on a symmetric scale, while the response is a positive
intensity spanning decades, so it takes eyepiece's log-contrast image.

Arrays-in is the primary path, not a fallback: every sampled consumer holds
npz arrays rather than live objects, so ``(modes, responses)`` as plain
``(k, y, x)`` stacks is a first-class call. ``(ModeBasis, Linearization)``
is accepted too and reduces to exactly the same arrays.

The cycles-per-pupil to lambda/D equivalence travels in MATCHED TITLES, not
in cross-panel arrows: a ripple of k cycles across the pupil puts its speckle
at k lambda/D, and saying so in the two titles of one row survives being
embedded in a caller's figure, which an arrow between axes does not.
"""

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from physicaloptix.viz import _require

_MODE_TITLE = r"${k}$ cycles/pupil"
_RESPONSE_TITLE = r"speckle at ${k}\,\lambda/D$"


def _mode_stack(modes):
    """The ``(k, y, x)`` OPD stack, from a ``ModeBasis`` or a bare array."""
    stack = getattr(modes, "B", modes)
    array = np.asarray(stack, dtype=float)
    if array.ndim != 3:
        msg = f"plot_mode_gallery: modes must be a (k, y, x) stack, got {array.shape}"
        raise ValueError(msg)
    return array


def _response_stack(responses):
    """The ``(k, y, x)`` intensity-response stack.

    A ``Linearization`` carries the complex sensitivity ``G``; the visible
    response is its squared modulus, the intensity a unit mode coefficient
    puts in the focal plane.

    Raises:
        ValueError: A chromatic ``Linearization`` (its ``G`` is
            ``(w, m, y, x)``) -- select a channel's columns first; or a bare
            array that is not a ``(k, y, x)`` stack.
    """
    if responses is None:
        return None
    g = getattr(responses, "G", None)
    if g is not None:
        g = np.asarray(g)
        if g.ndim == 4:
            msg = (
                "plot_mode_gallery: chromatic Linearization has G of shape "
                f"{g.shape}; select one channel's (m, y, x) columns first"
            )
            raise ValueError(msg)
        return np.abs(g) ** 2
    array = np.asarray(responses, dtype=float)
    if array.ndim != 3:
        msg = (
            f"plot_mode_gallery: responses must be a (k, y, x) stack, got {array.shape}"
        )
        raise ValueError(msg)
    return array


def _masked_opd(stack):
    """NaN outside the basis's common support.

    The support is inferred from the stack itself: a pixel that is EXACTLY
    zero in every mode is outside the aperture, since no mode of a basis
    built on a pupil reaches there. A pixel that is zero in one mode but not
    another is a genuine node of that mode and stays drawn. Callers who
    already carry a NaN-masked stack keep their own mask -- NaN propagates
    through the ``!= 0`` test as True, i.e. "in support", and is then
    rendered by the colormap's bad color either way.
    """
    support = np.any(stack != 0.0, axis=0)
    return np.where(support[None, :, :], stack, np.nan)


def _opd_cmap(ax):
    """The ``opd`` diverging colormap, masked pixels painted the axes ground.

    Resolved at CALL time, never bound at import: ``hwostyle.cmaps.opd`` is a
    property of the active mode. It is a distinct semantic key from
    ``residual`` (ratio semantics) and ``phase`` (the cyclic map), so a
    later divergence in what those keys resolve to lands here for free.
    """
    value = hwostyle.cmaps.opd
    resolved = matplotlib.colormaps[value] if isinstance(value, str) else value
    return resolved.with_extremes(bad=ax.get_facecolor())


def _titles_for(titles, index):
    """The matched ``(mode_title, response_title)`` pair for one row.

    A numeric entry is read as cycles per pupil and becomes the matched pair
    that states the equivalence; a string entry is used verbatim on both
    panels, so a caller with its own vocabulary still gets a matched row.
    """
    if titles is None:
        return None, None
    entry = titles[index]
    if isinstance(entry, str):
        return entry, entry
    k = f"{entry:g}"
    return _MODE_TITLE.format(k=k), _RESPONSE_TITLE.format(k=k)


def plot_mode_gallery(
    modes,
    responses=None,
    *,
    indices=None,
    axes=None,
    response_scale=None,
    titles=None,
    imshow_kw=None,
):
    """Draw a gallery of OPD modes beside the speckles they produce.

    One row per selected mode: the signed OPD map on the left, its
    focal-plane intensity response on the right.

    The OPD column is drawn on a SYMMETRIC scale about zero with the
    ``opd`` diverging colormap, shared across every row so the rows are
    comparable. Symmetry is not cosmetic: a diverging map's white center
    means "zero", so letting the limits follow a one-sided mode's own
    min/max would put white somewhere other than zero and the picture would
    assert a sign change that the data does not contain. Pixels outside the
    basis's common support render as the axes' own ground rather than as a
    misleading zero (see ``_masked_opd``).

    The response column is eyepiece's log-contrast image on a shared norm.
    ``response_scale`` multiplies every response before drawing and is where
    the ``(rms)^2 / telescope_peak`` weighting goes: ``G`` is a sensitivity
    per unit mode coefficient, so ``|G|^2`` alone is a response per unit^2,
    not a contrast. Pass a scalar for a common rms, or a ``(k,)`` array for
    per-mode rms values, and the column reads as the contrast each mode
    actually contributes.

    Args:
        modes: A ``ModeBasis``, or a bare ``(k, y, x)`` OPD stack.
        responses: A ``Linearization`` (its ``|G|^2`` is used), or a bare
            ``(k, y, x)`` intensity stack. None draws the OPD column alone.
        indices: Sequence of mode indices to show. None shows every mode.
        axes: Axes to draw into, shape ``(k, 2)`` with ``responses`` and
            ``(k, 1)`` without, for ``k = len(indices)`` (or the number of
            modes). None creates the figure.
        response_scale: Scalar or ``(k,)`` multiplier applied to the
            responses; see above. None leaves them as given.
        titles: Per-row titles, in ``indices`` order. A NUMBER is read as
            cycles per pupil and becomes the matched pair "k cycles/pupil"
            and "speckle at k lambda/D"; a STRING is used verbatim on both
            panels. None omits titles.
        imshow_kw: Extra kwargs passed to every panel's ``imshow``.

    Returns:
        A ``MosaicResult`` with ``axes`` of the shape above.
        ``artists["opd"]`` is the list of OPD ``AxesImage``;
        ``artists["response"]`` the list of response ``AxesImage`` (absent
        without ``responses``).

    Raises:
        ValueError: ``modes``/``responses`` are not ``(k, y, x)`` stacks; a
            chromatic ``Linearization``; ``responses`` has a different mode
            count than ``modes``; or ``axes`` of the wrong shape, naming
            both the expected and the received shape.
    """
    ep = _require.eyepiece()
    mode_stack = _mode_stack(modes)
    response_stack = _response_stack(responses)
    if response_stack is not None and len(response_stack) != len(mode_stack):
        msg = (
            f"plot_mode_gallery: {len(mode_stack)} modes but "
            f"{len(response_stack)} responses"
        )
        raise ValueError(msg)

    selection = np.arange(len(mode_stack)) if indices is None else np.asarray(indices)
    n_rows = len(selection)
    n_cols = 1 if response_stack is None else 2

    if axes is not None:
        axes = np.asarray(axes, dtype=object)
        if axes.ndim == 1:
            axes = axes[:, None]
        if axes.shape != (n_rows, n_cols):
            msg = (
                f"plot_mode_gallery: expected axes shape ({n_rows}, {n_cols}), "
                f"got {np.asarray(axes, dtype=object).shape}"
            )
            raise ValueError(msg)
    else:
        _, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.4 * n_cols, 3.2 * n_rows),
            squeeze=False,
            layout="constrained",
        )
        axes = np.asarray(axes, dtype=object)

    selected_modes = _masked_opd(mode_stack)[selection]
    # One symmetric limit shared by every row: rows are meant to be compared.
    finite = selected_modes[np.isfinite(selected_modes)]
    span = float(np.max(np.abs(finite))) if finite.size else 1.0
    span = span if span > 0.0 else 1.0

    if response_stack is not None:
        selected_responses = response_stack[selection]
        if response_scale is not None:
            scale = np.asarray(response_scale, dtype=float)
            if scale.ndim == 1:
                scale = scale[selection][:, None, None]
            selected_responses = selected_responses * scale
        positive = selected_responses[selected_responses > 0.0]
        vmin = float(positive.min()) if positive.size else None
        vmax = float(selected_responses.max()) if positive.size else None

    kw = {"interpolation": "nearest", "origin": "lower", **(imshow_kw or {})}
    opd_images = []
    response_images = []
    for row in range(n_rows):
        opd_ax = axes[row, 0]
        image = opd_ax.imshow(
            selected_modes[row],
            cmap=_opd_cmap(opd_ax),
            vmin=-span,
            vmax=span,
            **kw,
        )
        opd_images.append(image)
        mode_title, response_title = _titles_for(titles, row)
        if mode_title is not None:
            opd_ax.set_title(mode_title)

        if response_stack is None:
            continue
        panel = ep.imshow_log(
            selected_responses[row],
            ax=axes[row, 1],
            vmin=vmin,
            vmax=vmax,
            imshow_kw=imshow_kw,
        )
        response_images.append(panel.artists["image"])
        if response_title is not None:
            axes[row, 1].set_title(response_title)

    artists = {"opd": opd_images}
    if response_images:
        artists["response"] = response_images
    return ep.MosaicResult(axes=axes, artists=artists)
