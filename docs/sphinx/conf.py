"""Sphinx configuration for the code documentation of transport_ml and transport_backend.

Build (from the repository root, inside the project virtual environment):

    .venv/bin/python -m pip install "sphinx>=8,<9"
    .venv/bin/python -m sphinx -b html docs/sphinx docs/sphinx/_build/html
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))
sys.path.insert(0, str(ROOT / "backend" / "src"))

project = "Предиктор изменений в графике транспорта"
author = "Команда хакатона"
release = "0.1.0"
language = "ru"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "member-order": "bysource",
    "show-inheritance": True,
}
viewcode_follow_imported_members = False
autodoc_typehints = "description"
napoleon_google_docstring = True
# Release documentation must build without network after dependencies are installed.
intersphinx_mapping = {}

templates_path = []
exclude_patterns = ["_build"]
html_theme = "alabaster"
html_title = project
