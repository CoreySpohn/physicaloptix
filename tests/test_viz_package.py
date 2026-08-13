"""viz package import mechanics: lazy eyepiece, zero-extra legacy path."""

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")


def test_base_install_survives_without_eyepiece(tmp_path):
    code = (
        "import sys; sys.modules['eyepiece'] = None\n"  # makes 'import eyepiece' raise
        "import physicaloptix\n"
        "assert callable(physicaloptix.render_path)\n"
        "import physicaloptix.viz\n"  # bare viz import stays cheap too
        "print('survived')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "survived"


def test_lazy_name_gives_actionable_error_without_eyepiece():
    code = (
        "import sys; sys.modules['eyepiece'] = None\n"
        "import physicaloptix.viz as v\n"
        "try:\n"
        "    v.plot_field\n"
        "except ImportError as e:\n"
        "    print('physicaloptix[viz]' in str(e))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "True"


def test_dir_lists_lazy_names():
    import physicaloptix.viz as v

    listing = dir(v)
    for name in ("plot_field", "contrast_row", "plot_path", "minimap", "render_path"):
        assert name in listing


def test_viz_extra_hwostyle_interface_is_present():
    """The viz extra's hwostyle must carry the INTERFACE viz imports, not a version.

    ``viz/train.py`` and ``viz/modes.py`` import hwostyle directly, so the
    extra pins it directly. Asserting on ``hwostyle.__version__`` would pass
    against a yanked or mis-tagged build and fail against a valid dev
    checkout; the thing that actually breaks ``plot_mode_gallery`` is a
    missing ``cmaps.opd``, so that is what is asserted -- the same
    interface-over-presence rewrite the R2 gate needed.
    """
    import hwostyle

    # The active mode is process-wide state, so it is restored: R2 shipped a
    # test that leaked style state and left the suite passing only by filename
    # sort order. This module sorts before the profiles/speckle/train viz
    # tests, so a leak here would hand them whatever mode ran last.
    prev_mode = hwostyle.current_mode()
    try:
        for mode in ("dark", "light", "paper", "barbie"):
            hwostyle.use(mode)
            assert isinstance(hwostyle.cmaps.opd, str)
            for role in ("star", "planet", "measured", "model"):
                assert isinstance(getattr(hwostyle.roles, role), str)
    finally:
        hwostyle.use(prev_mode or "dark")


def test_render_path_unchanged():
    # The pre-existing test_viz.py suite is the real check; this pins the import path.
    from physicaloptix.viz import render_path
    from physicaloptix.viz._legacy import render_path as legacy

    assert render_path is legacy


def test_render_path_warns_deprecation(small_path, mono_field):
    import warnings

    from physicaloptix.viz import render_path

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render_path(small_path, mono_field)
    assert any(
        issubclass(w.category, DeprecationWarning) and "plot_path" in str(w.message)
        for w in caught
    )
