"""Stateless renderers for physicaloptix objects, built on eyepiece.

Lazy by design: the eyepiece-consuming submodules import on first attribute
access, so the base install never needs the viz extra. The deprecated
render_path stays eagerly importable with matplotlib alone.
"""

from physicaloptix.viz._legacy import render_path

_LAZY = {
    "plot_field": "fields",
    "contrast_row": "fields",
    "draw_dark_zone": "fields",
    "plot_path": "train",
    "minimap": "train",
    "native_dpi": "train",
    "animate_speckles": "boiling",
    "boiling_strip": "boiling",
    "plot_contrast_profile": "profiles",
    "plot_mode_gallery": "modes",
    "plot_field_ellipse": "speckle",
    "plot_process": "speckle",
    "plot_speckle_ensemble": "speckle",
}

__all__ = ["render_path", *sorted(_LAZY)]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        from physicaloptix.viz import _require

        _require.eyepiece()
        module = importlib.import_module(f"physicaloptix.viz.{_LAZY[name]}")
        return getattr(module, name)
    msg = f"module 'physicaloptix.viz' has no attribute '{name}'"
    raise AttributeError(msg)


def __dir__():
    return sorted(set(globals()) | set(__all__))
