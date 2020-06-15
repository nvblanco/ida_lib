import sphinx_bootstrap_theme
import guzzle_sphinx_theme
import sphinx_rtd_theme
import os
import sys

#html_theme_path = guzzle_sphinx_theme.html_theme_path()
#html_theme = 'guzzle_sphinx_theme'
#html_theme = 'bootstrap'
html_theme = "sphinx_rtd_theme"
html_logo = 'icon.png'
html_theme_options = {

    # Set the path to a special layout to include for the homepage
    "index_template": "special_index.html",


    'logo_only': True,
    'display_version': True,
    "html_logo": "icon.png",

    # Allow the project link to be overriden to a custom URL.
    "projectlink": "https://github.com/raquelvilas18/ida_lib",


    # If False, expand all TOC entries
    "globaltoc_collapse": False,

    # If True, show hidden TOC entries
    "globaltoc_includehidden": False,
}
# Register the theme as an extension to generate a sitemap.xml


# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.join(os.path.dirname(__name__), '..'))


# -- Project information -----------------------------------------------------

project = 'IdaLib'
copyright = '2020, Raquel Vilas'
author = 'Raquel Vilas'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    "sphinx_rtd_theme",
]
extensions.append("guzzle_sphinx_theme")

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

numpydoc_show_class_members = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#html_theme = 'bootstrap'
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------