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


# A meta-path finder whose find_spec refuses the named modules. This is the
# blocker that works on Python 3.12+: the legacy find_module hook is never
# consulted there, so a find_module-based blocker silently blocks nothing.
_BLOCKER = (
    "import sys\n"
    "class _Block:\n"
    "    def __init__(self, names):\n"
    "        self.names = names\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if any(name == n or name.startswith(n + '.') for n in self.names):\n"
    "            raise ModuleNotFoundError(f'blocked: {name}', name=name)\n"
    "        return None\n"
)


def _run_blocked(names, body):
    code = _BLOCKER + f"sys.meta_path.insert(0, _Block({names!r}))\n" + body
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_base_import_is_lazy_and_eyepiece_free_under_a_find_spec_blocker():
    body = (
        "import physicaloptix\n"
        "import physicaloptix.viz\n"
        "assert 'eyepiece' not in sys.modules\n"
        "assert 'matplotlib' not in sys.modules\n"
        "assert 'physicaloptix.viz._legacy' not in sys.modules\n"
        "assert callable(physicaloptix.render_path)\n"  # resolves on first use
        "print('survived')\n"
    )
    assert _run_blocked(["eyepiece"], body) == "survived"


def test_viz_is_an_attribute_after_a_bare_import_without_loading_eyepiece():
    body = (
        "import physicaloptix\n"
        "viz = physicaloptix.viz\n"
        "assert viz is sys.modules['physicaloptix.viz']\n"
        "assert 'prepare_speckles' in dir(viz)\n"
        "assert 'eyepiece' not in sys.modules\n"
        "print('survived')\n"
    )
    assert _run_blocked(["eyepiece"], body) == "survived"


def test_preparation_runs_without_either_renderer_interface():
    body = (
        "import numpy as np\n"
        "from physicaloptix.viz import prepare_speckles\n"
        "cube = np.arange(1, 1 + 3 * 5 * 5, dtype=float).reshape(3, 5, 5)\n"
        "seq = prepare_speckles(cube, times_s=np.arange(3.0), pixscale_lod=1.0,\n"
        "                       trace=(0.5, 1.2), sample_kind='instantaneous',\n"
        "                       quantity='total')\n"
        "seq.frame(2); seq.strip([0, 2]); seq.at(1.5); seq.schedule(1.0, 10)\n"
        "for name in ('matplotlib', 'manim', 'eyepiece.mpl', 'eyepiece.manim',\n"
        "             'physicaloptix.viz._legacy', 'physicaloptix.viz.boiling'):\n"
        "    assert name not in sys.modules, name\n"
        "print('prepared')\n"
    )
    blocked = ["matplotlib", "manim", "eyepiece.mpl", "eyepiece.manim"]
    assert _run_blocked(blocked, body) == "prepared"


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
    names = (
        "plot_field",
        "contrast_row",
        "field_columns",
        "plot_path",
        "minimap",
        "render_path",
    )
    for name in (*names, "prepare_speckles"):
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
