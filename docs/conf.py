# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

# Add the repository root to the Python path
sys.path.insert(0, os.path.abspath('..'))

project = 'jumufraktiv'
copyright = '2026, Si-Yang Li, Josh Speagle'
author = 'Si-Yang Li, Josh Speagle'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',          # for NumPy/Google docstrings
    'sphinx.ext.viewcode',          # optional: links to source
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# `html_theme` is set above, and was assigned a second time here. The second
# assignment won, so the declared theme was inert: the `docs` extra installs
# sphinx-rtd-theme and every build shipped alabaster.
html_static_path = ['_static']
