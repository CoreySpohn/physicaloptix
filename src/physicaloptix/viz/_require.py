"""Import guard: eyepiece is the viz extra's backbone."""


def eyepiece():
    """Return the eyepiece module or raise an actionable ImportError."""
    try:
        import eyepiece as _eyepiece
    except ImportError as err:
        msg = "physicaloptix.viz requires eyepiece: pip install 'physicaloptix[viz]'"
        raise ImportError(msg) from err
    return _eyepiece
