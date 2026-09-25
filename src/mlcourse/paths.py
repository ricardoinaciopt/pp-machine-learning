"""Repository paths shared by notebooks ands utilities."""

from pathlib import Path
from .setup import find_root


def repo_root(start=None):
    return find_root(start)


def data_path(*parts):
    return repo_root() / "data" / Path(*parts)
