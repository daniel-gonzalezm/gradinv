"""Every input number of the article has a traceable provenance.

analysis/data/provenance.csv lists, row by row, where each input comes from
(source, table or equation, page), its kind, units and conversion, how it was
checked, the errata that affect it and its rule of use. The values themselves live
only in the data files and scripts the rows point to (single source of truth).

The tests fail if a numeric entry of the data files is not covered by exactly one
row, if a row points to nothing, or if a row's declared count does not match.
The digit-by-digit check against the PDFs is analysis/check_transcriptions.py
(the PDFs are not in the repository).
"""
import csv
import importlib.util
import json
import numbers
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "data"
CSV = DATA / "provenance.csv"
JSON_FILES = ("cubic_gradient_constants.json", "published_reference_values.json")
KINDS = {"published", "measured", "derived", "catalogue", "assumed", "synthetic"}
REQUIRED = ("id", "location", "quantity", "n_values", "kind", "source", "locator", "units",
            "conversion", "check", "use_rule", "used_in")


def rows():
    with CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def numeric_leaves(obj, prefix=""):
    """(path, value) of the numeric leaves of a JSON object; keys starting with '_' are metadata."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not str(k).startswith("_"):
                out += numeric_leaves(v, f"{prefix}/{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple, np.ndarray)):
        for i, v in enumerate(obj):
            out += numeric_leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, numbers.Number) and not isinstance(obj, bool):
        out.append((prefix, obj))
    return out


def count_numbers(obj):
    """Numeric leaves of a Python value (dict values only, not keys; strings ignored)."""
    if isinstance(obj, dict):
        return sum(count_numbers(v) for v in obj.values())
    if isinstance(obj, (list, tuple, np.ndarray)):
        return sum(count_numbers(v) for v in obj)
    return int(isinstance(obj, (numbers.Number, np.number)) and not isinstance(obj, bool))


_MODULES = {}


def module(relpath):
    if relpath not in _MODULES:
        spec = importlib.util.spec_from_file_location(relpath.replace("/", "_")[:-3], ROOT / relpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[relpath] = mod
    return _MODULES[relpath]


def resolve(loc):
    """Numeric count and, for JSON locations, the covered leaf paths."""
    if "::" in loc:
        path, sym = loc.split("::")
        mod = module(path)
        assert hasattr(mod, sym), f"{loc}: symbol not found"
        return count_numbers(getattr(mod, sym)), []
    fname, sub = loc.split(":", 1)
    assert fname in JSON_FILES, f"{loc}: unknown data file"
    obj = json.loads((DATA / fname).read_text())
    for k in sub.split("/"):
        assert k in obj, f"{loc}: key {k!r} not found"
        obj = obj[k]
    leaves = [(f"{fname}:{sub}/{p}" if p else f"{fname}:{sub}", v) for p, v in numeric_leaves(obj)]
    leaves = [(p.replace("/[", "["), v) for p, v in leaves]
    return len(leaves), [p for p, _ in leaves]


def test_columns_and_kinds():
    rs = rows()
    assert rs, "empty provenance table"
    ids = [r["id"] for r in rs]
    assert len(ids) == len(set(ids)), "repeated id"
    for r in rs:
        for c in REQUIRED:
            assert r[c].strip(), f"{r['id']}: empty column {c}"
        assert r["kind"] in KINDS, f"{r['id']}: kind {r['kind']!r}"
        assert int(r["n_values"]) > 0


@pytest.mark.parametrize("row", rows(), ids=lambda r: r["id"])
def test_row_resolves_with_declared_count(row):
    n = sum(resolve(loc)[0] for loc in row["location"].split("|"))
    assert n == int(row["n_values"]), f"{row['id']}: declared {row['n_values']}, found {n}"


@pytest.mark.core
def test_every_data_entry_covered_exactly_once():
    covered = []
    for r in rows():
        for loc in r["location"].split("|"):
            if "::" not in loc:
                covered += resolve(loc)[1]
    assert len(covered) == len(set(covered)), "an entry is covered by more than one row"
    for fname in JSON_FILES:
        leaves = numeric_leaves(json.loads((DATA / fname).read_text()))
        missing = [f"{fname}:{p}" for p, _ in leaves if f"{fname}:{p}" not in set(covered)]
        assert not missing, f"entries without provenance: {missing[:5]}"


def test_published_inputs_have_a_page_or_table():
    for r in rows():
        if r["kind"] == "published":
            assert any(w in r["locator"] for w in ("Table", "Section", "Fig", "eq")), r["id"]
            assert "p." in r["locator"], f"{r['id']}: page missing"


def test_reference_modules_read_the_data_file():
    """Single source: the published comparison values used in tests are those of the data file."""
    from tests.references import averages_sources as av_ref
    from tests.references import solyaev_sources as so_ref
    pub = json.loads((DATA / "published_reference_values.json").read_text())
    assert av_ref.TABLE6 == pub["lazar2022_table6"]["abar"]
    assert so_ref.FE_EXAMPLE == {k: pub["solyaev2023_fe"][k] for k in ("C11", "C12", "C44", "k", "nu")}
