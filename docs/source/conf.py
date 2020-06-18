import sphinx_rtd_theme
import os
import sys

html_theme = "sphinx_rtd_theme"
html_logo = 'icon.png'
sys.path.insert(0, os.path.abspath('.'))
sys.path.append(os.path.join(os.path.dirname(__name__), '..'))
_pysrc = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))


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

from unittest.mock import MagicMock
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
            return MagicMock()

MOCK_MODULES = ["functools",
                #'typing',
                'numpy',
                'torch',
                'string',
                'cv2',
                'kornia',
                'random',
                #'abc',
                'os',
                'bokeh']
#MOCK_MODULES = []
autodoc_mock_imports = MOCK_MODULES

# sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

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