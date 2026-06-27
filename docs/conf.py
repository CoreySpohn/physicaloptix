"""Sphinx configuration file."""

import physicaloptix

project = "physicaloptix"
copyright = "2026, Corey Spohn"
author = "Corey Spohn"
version = physicaloptix.__version__
release = physicaloptix.__version__

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.autodoc",
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

language = "Python"

autoapi_dirs = ["../src"]
autoapi_ignore = ["**/*version.py"]
autodoc_typehints = "description"

myst_enable_extensions = ["amsmath", "dollarmath"]
myst_fence_as_directive = ["mermaid"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
master_doc = "index"
html_title = "physicaloptix"
html_sidebars = {"posts/*": ["sbt-sidebar-nav.html"]}

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
nb_execution_mode = "off"
nb_execution_timeout = 300
