# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'spinpynamics'
copyright = '2022, Edmund Little'
author = 'Edmund Little'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
]

# Turn on sphinx.ext.autosummary
autosummary_generate = True
# Include by default include all members (inc. imported) of a module
autodoc_default_options = {
    'members': True,
    'inherited-members': True,
}
# Add __init__ doc (ie. params) to class summaries
autoclass_content = "both"
# Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = False
# If no docstring, inherit from base class
autodoc_inherit_docstrings = True
# Enable 'expensive' imports for sphinx_autodoc_typehints
set_type_checking_flag = True
# Remove namespaces from class/method signatures
add_module_names = False
# Tabulate class attributes/methods
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# # -- Options for HTML output -----------------------------------------------

# # The theme to use for HTML and HTML Help pages.  See the documentation for
# # a list of builtin themes.
# #
# html_theme = 'alabaster'

# Readthedocs theme
# on_rtd is whether on readthedocs.org, this line of code grabbed from
# docs.readthedocs.org...
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# Override some CSS settings
html_css_files = ["readthedocs-custom.css"]
