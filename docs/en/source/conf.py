import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath('../../../'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MoGen'
copyright = '2024, S-Lab'
author = 'Mingyuan Zhang'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode',
    'sphinx_markdown_tables', 'sphinx_copybutton', 'myst_parser',
    'sphinx.ext.autosummary'
]


# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_logo = '_static/logo-removebg-preview.png'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

language = 'en'

# html_static_path = ['_static']
# html_css_files = ['css/readthedocs.css']

# Enable ::: for my_st
myst_enable_extensions = ['colon_fence']

master_doc = 'index'

