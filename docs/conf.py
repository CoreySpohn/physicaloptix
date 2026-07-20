"""Sphinx configuration file."""

from importlib.metadata import version as get_version

project = "physicaloptix"
copyright = "2026, Corey Spohn"
author = "Corey Spohn"
release = get_version("physicaloptix")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
    "IPython.sphinxext.ipython_console_highlighting",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "optixstuff": ("https://optixstuff.readthedocs.io/en/latest/", None),
    "coronagraphoto": ("https://coronagraphoto.readthedocs.io/en/latest/", None),
    "yippy": ("https://yippy.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

autoapi_dirs = ["../src"]
autoapi_ignore = ["**/*version.py"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # document top-level re-exports so ``{class}`~physicaloptix.X``` roles
    # resolve; the only cost is one benign duplicate-object warning where the
    # ``linearize`` function shares its name with the ``linearize`` module.
    "imported-members",
]
autodoc_typehints = "description"

# Render Google-style ``Attributes:`` sections as inline ``:ivar:`` fields, so
# they do not collide with the ``py:attribute`` directives autoapi generates
# from the same Equinox module fields (avoids duplicate-object warnings).
napoleon_use_ivar = True

# Silence the harmless generated _version import note.
suppress_warnings = ["autoapi.python_import_resolution"]

myst_enable_extensions = ["amsmath", "dollarmath"]
myst_fence_as_directive = ["mermaid"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
master_doc = "index"
html_title = "physicaloptix"

html_theme_options = {
    "repository_url": "https://www.github.com/CoreySpohn/physicaloptix",
    "repository_branch": "main",
    "use_repository_button": True,
    "show_toc_level": 2,
}
html_context = {
    "default_mode": "dark",
}
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}
nb_execution_mode = "auto"
nb_execution_timeout = 300
nb_execution_raise_on_error = True
# Drop benign import-time stderr (e.g. tqdm's IProgress warning) from the
# rendered output; genuine execution errors still raise via the flag above.
nb_output_stderr = "remove"
