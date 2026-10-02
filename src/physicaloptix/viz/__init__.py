"""Stateless renderers for physicaloptix objects, built on eyepiece.

Lazy by design: every public name imports on first attribute access, so the
base install never needs the viz extra, and loading one function never loads
another's module. The eyepiece-consuming names raise an actionable
ImportError without the viz extra; the deprecated render_path needs only
matplotlib, and only once it is called.
"""

_LAZY = {
    "plot_field": "fields",
    "contrast_row": "fields",
    "draw_dark_zone": "fields",
    "field_columns": "columns",
    "plot_path": "train",
    "minimap": "train",
    "native_dpi": "train",
    "prepare_speckles": "_prepare",
    "animate_speckles": "boiling",
    "boiling_strip": "boiling",
    "plot_contrast_profile": "profiles",
    "plot_propagation": "propagation",
    "plot_mode_gallery": "modes",
    "plot_zernike_pyramid": "modes",
    "plot_field_ellipse": "speckle",
    "plot_process": "speckle",
    "plot_speckle_ensemble": "speckle",
}

# Names that do not go through eyepiece at all.
_LAZY_PLAIN = {"render_path": "_legacy"}

__all__ = sorted([*_LAZY, *_LAZY_PLAIN])


def __getattr__(name):
    if name in _LAZY or name in _LAZY_PLAIN:
        import importlib

        if name in _LAZY:
            from physicaloptix.viz import _require

            _require.eyepiece()
            module_name = _LAZY[name]
        else:
            module_name = _LAZY_PLAIN[name]
        module = importlib.import_module(f"physicaloptix.viz.{module_name}")
        return getattr(module, name)
    msg = f"module 'physicaloptix.viz' has no attribute '{name}'"
    raise AttributeError(msg)


def __dir__():
    return sorted(set(globals()) | set(__all__))
