"""The package imports, and its version is the same in pyproject.toml, the package
and CITATION.cff."""
import re
from pathlib import Path

import gradinv

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_consistent():
    pyproject = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    citation = re.search(r'^version: "([^"]+)"', (ROOT / "CITATION.cff").read_text(), re.M).group(1)
    assert gradinv.__version__ == pyproject == citation
