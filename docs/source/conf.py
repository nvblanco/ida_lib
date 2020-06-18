import sphinx_rtd_theme
import os
import sys

html_theme = "sphinx_rtd_theme"
html_logo = 'icon.png'
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

autodoc_mock_imports = ["functools", 'typing', 'numpy', 'torch', 'string', 'cv2', 'kornia', 'ida_lib', 'random', 'abc', 'os', 'bokeh']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

numpydoc_show_class_members = False

# -- Options for HTML output -------------------------------------------------

html_static_path = ['_static']

master_doc = 'index'

# -- Extension configuration -------------------------------------------------