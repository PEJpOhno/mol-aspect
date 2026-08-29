# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / 'src'))

# molaspect/__init__.py を経由せず _version.py だけ import する
_version_path = _REPO_ROOT / 'src' / 'molaspect' / '_version.py'
_spec = importlib.util.spec_from_file_location('molaspect._version', _version_path)
_version_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_version_mod)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mol-aspect'
copyright = '2026, M. Ohno'
author = 'M. Ohno'
version = _version_mod.__version__
release = '2026/08/29'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'myst_parser',
    'sphinx_markdown_builder',
    'sphinx.ext.todo',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = ['rdkit', 'numpy', 'py3Dmol']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
