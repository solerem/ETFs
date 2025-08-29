# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
root_doc = "index"        # NOT "docs/index", NOT an absolute path

import os, sys
sys.path.insert(0, os.path.abspath("../.."))        # repo root
sys.path.insert(0, os.path.abspath("../../etf_code"))   # your source folder (adjust if different)



project = 'ETF Portfolio Optimization'
copyright = '2025, Maxime Solère'
author = 'Maxime Solère'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google/NumPy-style docstrings
    "sphinx.ext.viewcode"
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
