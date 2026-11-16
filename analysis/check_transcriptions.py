"""Independent check of the transcribed input data against the source PDFs.

The PDFs are not stored in the repository (copyright). Pass the folder that holds
them; the file names are matched by substring (e.g. 'lazar2022', 'po2019',
'lazar2024', 'solyaev2023'). The script extracts the text of each table with
`pdftotext -layout` (poppler), parses the numbers row by row, and compares them
with analysis/data/cubic_gradient_constants.json and with the values hard-coded in
the analysis scripts. Every printed digit must match exactly (string comparison of
the rounded values), not just to a tolerance.

Usage:
    python3 analysis/check_transcriptions.py /path/to/pdf/folder
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "analysis" / "data" / "cubic_gradient_constants.json").read_text())
DS = DATA["datasets"]

NUM = r"[−\-–]?\s?\d+\.\d+(?:·10[−\-–]?\d+)?"


def to_float(tok):
    t = tok.replace("−", "-").replace("–", "-").replace(" ", "")
    if "·10" in t:
        mant, exp = t.split("·10")
        return float(mant) * 10.0 ** int(exp)
    return float(t)


def ndec(tok):
    """Printed decimal places of a token, as a float written plainly."""
    t = tok.replace("−", "-").replace("–", "-").replace(" ", "")
    if "·10" in t:
        mant, exp = t.split("·10")
        return len(mant.split(".")[1]) - int(exp)
    return len(t.split(".")[1])


def pdf_text(folder, key, layout=True):
    hits = [p for p in Path(folder).glob("*.pdf") if key in p.name.lower()]
    if len(hits) != 1:
        raise SystemExit(f"expected one PDF matching {key!r} in {folder}, found {hits}")
    return subprocess.run(["pdftotext"] + (["-layout"] if layout else []) + [str(hits[0]), "-"], capture_output=True,
                          text=True, check=True).stdout


def table_rows(text, start, labels, ncols):
    """Rows of the table whose caption contains `start`, keyed by label."""
    i = text.index(start)
    rows = {}
    for lab in labels:
        m = re.search(r"^\s*" + re.escape(lab) + r"\s*\[[^\]]*\]\s*((?:" + NUM + r"\s*){%d})" % ncols,
                      text[i:], re.M)
        if m is None:
            raise SystemExit(f"row {lab!r} not found after {start!r}")
        rows[lab] = re.findall(NUM, m.group(1))
    return rows


RESULTS = []


def check(name, printed_tokens, stored):
    """Exact match: the stored value rounded to the printed decimals equals the print."""
    assert len(printed_tokens) == len(stored), name
    for k, (tok, val) in enumerate(zip(printed_tokens, stored)):
        p, d = to_float(tok), ndec(tok)
        ok = round(val, d) == round(p, d) and abs(val - p) <= 0.5 * 10.0 ** (-d) + 1e-12
        RESULTS.append((ok, f"{name}[{k}]: printed {tok.strip()} stored {val}"))


def main(folder):
    voigt = ["D1,1", "D1,2", "D1,3", "D2,2", "D2,3", "D2,4", "D2,5", "D3,3", "D3,5", "D16,16", "D16,17"]
    cols = ["Al", "Cu", "Fe", "W"]

    # Lazar, Agiasofitou and Boehlke (2022), Tables 1, 2 and 6
    t = pdf_text(folder, "lazar2022")
    r = table_rows(t, "Table 1 Elastic and gradient-elastic", ["C11", "C12", "C44"] + voigt, 4)
    for j, mat in enumerate(cols):
        check(f"lazar2022 Table 1 C {mat}", [r[k][j] for k in ("C11", "C12", "C44")], DS["lazar2022"]["C"][mat])
        check(f"lazar2022 Table 1 D {mat}", [r[k][j] for k in voigt], DS["lazar2022"]["D_voigt"][mat])
    r = table_rows(t, "Table 2 Gradient-elastic constants", [f"a{i}" for i in range(1, 12)], 4)
    for j, mat in enumerate(cols):
        check(f"lazar2022 Table 2 a {mat}", [r[f"a{i}"][j] for i in range(1, 12)], DS["lazar2022"]["a_table2"][mat])
    i6 = t.index("Table 6 Voigt-type averaged")
    seg = t[i6:i6 + 1500]
    rows6 = [re.findall(NUM, line) for line in seg.splitlines() if re.match(r"\s*(?:ā|a\u0304)\d+\s*\[eV", line)]
    sys.path.insert(0, str(ROOT / "analysis"))
    import cubic_results as cr  # noqa: E402
    for j, mat in enumerate(("Al", "W")):
        check(f"lazar2022 Table 6 abar {mat}", [rw[j] for rw in rows6], cr.TABLE6[mat])

    # Po, Admal and Lazar (2019), Table 1 (columns: Cu EAM, Cu MEAM, Al MEAM)
    t = pdf_text(folder, "po2019")
    r = table_rows(t, "Table 1 Elastic and gradient-elastic", ["C1,1", "C1,2", "C4,4"] + voigt, 3)
    for j, mat in enumerate(("Cu_EAM", "Cu_MEAM", "Al_MEAM")):
        check(f"po2019 Table 1 C {mat}", [r[k][j] for k in ("C1,1", "C1,2", "C4,4")], DS["po2019"]["C"][mat])
        check(f"po2019 Table 1 D {mat}", [r[k][j] for k in voigt], DS["po2019"]["D_voigt"][mat])

    # Lazar and Po (2024), Tables 1 and 2
    t = pdf_text(folder, "lazar2024")
    r = table_rows(t, "Table 1 Elastic and gradient-elastic constants computed", ["C11", "C12", "C44"]
                   + [f"a{i}" for i in range(1, 12)], 4)
    for j, mat in enumerate(cols):
        check(f"lazar2024 Table 1 C {mat}", [r[k][j] for k in ("C11", "C12", "C44")], DS["lazarpo2024"]["C"][mat])
        check(f"lazar2024 Table 1 a {mat}", [r[f"a{i}"][j] for i in range(1, 12)], DS["lazarpo2024"]["a_table1"][mat])
    i2 = t.index("Table 2 Characteristic length scale")
    seg = t[i2:i2 + 800]
    get = lambda lab: re.findall(NUM, re.search(r"^\s*" + lab + r".*$", seg, re.M).group(0))[-4:]
    for lab, key in ((r"ℓ \[Å\]", "ell_table2"), (r"a \[Å\]", "lattice_a_table2"), (r"ℓ/a", "ell_over_a_table2")):
        check(f"lazar2024 Table 2 {key}", get(lab), [DS["lazarpo2024"][key][m] for m in cols])

    # Solyaev (2023), section 5: Fe grain constants (GPa) and classical effective values
    t = pdf_text(folder, "solyaev2023", layout=False)  # two-column page: raw reading order
    flat = re.sub(r"\s+", " ", t)
    cij = [re.search(lab + r"\s*=\s*(\d+(?:\.\d+)?)\s*GPa", flat).group(1)
           for lab in ("𝐶11", "𝐶12", "𝐶44")]
    fe = json.loads((ROOT / "analysis" / "data" / "published_reference_values.json").read_text())["solyaev2023_fe"]
    stored = [fe["C11"], fe["C12"], fe["C44"]]
    for k, (tok, val) in enumerate(zip(cij, stored)):
        RESULTS.append((float(tok) == val, f"solyaev2023 Fe C[{k}]: printed {tok} stored {val}"))

    bad = [m for ok, m in RESULTS if not ok]
    for ok, m in RESULTS:
        if not ok:
            print("FAIL", m)
    print(f"{len(RESULTS) - len(bad)} of {len(RESULTS)} values match the printed tables exactly")
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
