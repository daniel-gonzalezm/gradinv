"""The audit map of section S2 and the `core` marker.

The map (analysis/audit_map.py) names the key tests of every claim and result. These
checks keep it honest: every test it names exists and is marked `core`, every `core`
test is in the map exactly once, each row has one to four tests, and the LaTeX table
of the supplement is the generator's output.
"""
import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
spec = importlib.util.spec_from_file_location("audit_map", ROOT / "analysis" / "audit_map.py")
am = importlib.util.module_from_spec(spec)
spec.loader.exec_module(am)


def _marked_core():
    """(file, function) of every test function decorated with pytest.mark.core."""
    out = []
    for f in sorted(TESTS.glob("test_*.py")):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                if any(ast.unparse(d) == "pytest.mark.core" for d in node.decorator_list):
                    out.append((f.name, node.name))
    return out


def _defined(fname, func):
    tree = ast.parse((TESTS / fname).read_text())
    return any(isinstance(n, ast.FunctionDef) and n.name == func for n in ast.walk(tree))


def test_every_mapped_test_exists():
    for fname, func in am.core_tests():
        assert (TESTS / fname).exists(), fname
        assert _defined(fname, func), f"{fname}::{func}"


def test_map_and_core_marker_agree():
    mapped = am.core_tests()
    assert len(mapped) == len(set(mapped)), "a test is named twice in the map"
    assert sorted(mapped) == sorted(_marked_core())


def test_core_size_and_rows():
    assert len(am.core_tests()) == 42
    assert all(1 <= len(tests) <= 4 for *_, tests in am.MAP)
    assert all(level and set(level.replace(" ", "").split(",")) <= {"V", "CC", "C"}
               for _g, _r, _a, level, _t in am.MAP)
    groups = {g for g, *_ in am.MAP}
    assert groups == set(range(len(am.GROUPS)))


def test_table_matches_the_generator():
    assert am.OUT.read_text() == am.render()


@pytest.mark.parametrize("fname, func", am.core_tests())
def test_core_tests_have_no_skip_marker(fname, func):
    """A core test may be exact or slow, but never skipped or expected to fail."""
    tree = ast.parse((TESTS / fname).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    marks = [ast.unparse(d) for d in node.decorator_list]
    assert not any(m.startswith(("pytest.mark.skip", "pytest.mark.xfail")) for m in marks)


def test_readme_assigns_every_test_file_to_one_family():
    import re
    text = (TESTS / "README.md").read_text()
    families = re.split(r"^\*\*\d\. ", text, flags=re.M)[1:]
    assert len(families) == 4
    named = [f for fam in families for f in re.findall(r"`(test_\w+\.py)`", fam)]
    assert sorted(named) == sorted(p.name for p in TESTS.glob("test_*.py"))
