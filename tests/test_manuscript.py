"""Static checks of the manuscript, without the journal class (paper/build.py
compiles with it):
- every section file input by main.tex or supplement.tex exists and no section file
  is orphaned;
- every table and figure the text inputs is a generated output of analysis/, and the
  inventory of the supplement (analysis/supplement.py) points to existing generators,
  outputs and macros;
- every citation key is in references.bib, every entry records how it was checked,
  and no commented-out entry starts with '@' (BibTeX would read it);
- labels are unique and every \\ref points to a label;
- the title, the abstract, the keywords and the appendix numbering follow the guide for
  authors of the European Journal of Mechanics A/Solids;
- the flattened sources of the submission bundle (paper/build.py --bundle) input no
  repository file, keep every figure and take the bibliography from the .bbl text.
"""
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BACK_MATTER = PAPER / "sections" / "declarations.tex"


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def section_files(master):
    """Section files input by a master file, in order (back matter excluded)."""
    names = re.findall(r"\\input\{((?:sections|supplement)/[^}]+)\}", strip_comments((PAPER / master).read_text()))
    files = [PAPER / (n if n.endswith(".tex") else n + ".tex") for n in names]
    return [f for f in files if f != BACK_MATTER]


def _sources():
    files = [PAPER / "main.tex", PAPER / "supplement.tex", PAPER / "macros.tex"]
    for master in ("main.tex", "supplement.tex"):
        files += section_files(master)
    files.append(BACK_MATTER)
    return files


def _text(path):
    return strip_comments(path.read_text())


def _load(name):
    sys.path.insert(0, str(ROOT / "analysis"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_section_files_exist_and_none_is_orphaned():
    used = set()
    for master in ("main.tex", "supplement.tex"):
        for f in section_files(master):
            assert f.exists(), f
            used.add(f)
    on_disk = {f for d in ("sections", "supplement") for f in (PAPER / d).glob("*.tex")}
    on_disk.discard(BACK_MATTER)
    assert on_disk == used, f"orphaned or missing: {sorted(p.name for p in on_disk ^ used)}"


def test_tables_and_figures_are_generated_outputs():
    tables = {p.stem for p in (ROOT / "analysis" / "tables").glob("*.tex")}
    figures = {p.stem for p in (ROOT / "analysis" / "figures").glob("*.pdf")}
    used_t, used_f = set(), set()
    for f in _sources():
        t = _text(f)
        for name in re.findall(r"\\input\{([^}/]+)\}", t):
            if name not in ("macros", "paper_numbers"):
                assert name in tables, f"{f.name}: \\input{{{name}}} is not in analysis/tables"
                used_t.add(name)
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t):
            assert name in figures, f"{f.name}: figure {name} is not in analysis/figures"
            used_f.add(name)
    assert used_t == tables, f"generated tables not placed: {sorted(tables - used_t)}"
    assert used_f == figures, f"generated figures not placed: {sorted(figures - used_f)}"


def _bib():
    return (PAPER / "references.bib").read_text()


def test_bibliography_entries_record_their_check():
    bib = _bib()
    entries = re.findall(r"^@(\w+)\{([^,]+),(.*?)^\}", bib, flags=re.M | re.S)
    keys = [k for _, k, _ in entries]
    assert len(keys) == len(set(keys)), "duplicate keys"
    for kind, key, body in entries:
        assert re.search(r"\bchecked\s*=", body), f"{key}: no checked field"
        assert re.search(r"\btitle\s*=", body) and re.search(r"\byear\s*=", body), key
        if kind == "article":
            assert re.search(r"\bdoi\s*=", body), f"{key}: article without DOI"
    assert not re.search(r"^%\s*@", bib, flags=re.M), "commented entries must not start with '@'"


def test_citations_are_in_the_bibliography():
    keys = set(re.findall(r"^@\w+\{([^,]+),", _bib(), flags=re.M))
    for f in _sources():
        for group in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]+)\}", _text(f)):
            for k in group.split(","):
                assert k.strip() in keys, f"{f.name}: \\cite{{{k.strip()}}} not in references.bib"


def test_labels_unique_and_refs_resolve():
    labels, refs = [], set()
    for f in _sources():
        t = _text(f)
        labels += re.findall(r"\\label\{([^}]+)\}", t)
        refs |= set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", t))
    assert len(labels) == len(set(labels)), "duplicate labels"
    assert refs <= set(labels), f"undefined refs: {sorted(refs - set(labels))}"


def test_title_is_concise_and_has_no_formulae():
    """The guide asks for a concise title without abbreviations or formulae."""
    m = re.search(r"\\title\[mode=title\]\{(.*?)\}\s*\n\s*\n", _text(PAPER / "main.tex"), flags=re.S)
    title = " ".join(m.group(1).split())
    assert 20 < len(title) <= 150 and "$" not in title


def test_keywords_fit_the_journal_rules():
    """1 to 7 keywords, none built with 'and' or 'of'."""
    body = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", _text(PAPER / "main.tex"),
                     flags=re.S).group(1)
    words = [" ".join(k.split()) for k in body.split("\\sep")]
    assert 1 <= len(words) <= 7 and all(words)
    assert not [k for k in words if re.search(r"\b(and|of)\b", k)], words


def test_appendix_equations_are_numbered_by_their_own_letter():
    """Elsevier numbers appendix equations (A.1), (B.1): after \\appendix the equation
    counter restarts in each section and carries its letter."""
    src = (ROOT / "paper" / "main.tex").read_text()
    head = src[src.index("\\appendix"):].split("\\input")[0]
    assert "\\counterwithin{equation}{section}" in head
    assert "\\renewcommand{\\theequation}{\\thesection.\\arabic{equation}}" in head


def test_cited_entries_with_doi_were_confirmed_on_a_portal():
    """Every cited entry with a DOI was read (pdf) and confirmed on a portal; the
    entries without a DOI were read."""
    entries = {k: b for _, k, b in re.findall(r"^@(\w+)\{([^,]+),(.*?)^\}", _bib(),
                                              flags=re.M | re.S)}
    cited = set()
    for f in _sources():
        for group in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]+)\}", _text(f)):
            cited |= {k.strip() for k in group.split(",")}
    for key in sorted(cited):
        body = entries[key]
        checked = re.search(r"\bchecked\s*=\s*\{(.*)\}", body).group(1)
        if re.search(r"\bdoi\s*=", body):
            assert checked.startswith("pdf+portal"), f"{key}: DOI not confirmed ({checked})"
        else:
            assert checked.startswith("pdf"), f"{key}: not read ({checked})"


def test_abstract_fits_the_journal_rules():
    """EJM A/Solids abstract: at most 250 words, without references; inline math
    counts as one word, and so does each part of a hyphenated compound (as a submission
    system may count them)."""
    main = _text(PAPER / "main.tex")
    body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, flags=re.S).group(1)
    body = re.sub(r"(?m)%.*$", "", body)
    assert "\\pending" not in body and "\\cite" not in body
    words = re.split(r"[\s-]+", re.sub(r"\$[^$]*\$", "X", body).replace("~", " ").strip())
    assert 100 < len(words) <= 250, len(words)


def _paper_subsections():
    """{section number: number of subsections} of the paper, from its sources."""
    out = {}
    for i, f in enumerate([f for f in section_files("main.tex") if f.name[0].isdigit()], 1):
        out[i] = len(re.findall(r"\\subsection\{", _text(f)))
    return out


def test_supplement_points_to_the_paper_in_its_numbering():
    """The supplement names sections of the paper as N or N.M (CAS class), never with the
    lettered subsections of the earlier template (N(a), N\\,(a)); every N.M exists."""
    subs = _paper_subsections()
    texts = [_text(f) for f in section_files("supplement.tex")]
    texts.append((ROOT / "analysis" / "tables" / "audit_map.tex").read_text())
    for t in texts:
        assert not re.search(r"(?:section|sections)~\d+(?:\\,)?\([a-z]\)", t)
        for sec, sub in re.findall(r"section~(\d+)\.(\d+)", t):
            assert int(sub) <= subs[int(sec)], (sec, sub)


def test_highlights_fit_the_journal_rules():
    """EJM A/Solids: 3 to 5 highlights of at most 85 characters (spaces included), in a
    separate editable file with "highlights" in its name; plain text, no TeX."""
    lines = [ln.strip() for ln in (PAPER / "highlights.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    assert 3 <= len(lines) <= 5
    for ln in lines:
        assert len(ln) <= 85, (len(ln), ln)
        assert not re.search(r"[\\$~{}]", ln), ln


def test_supplement_inventory_holds():
    assert _load("supplement").check() == []


def test_supplement_tables_and_figure_are_listed():
    sup = _load("supplement")
    outs = {o for *_x, outs, _p in sup.INVENTORY for o in outs}
    for name in _load("make_tables").SUPPLEMENT_TABLES:
        assert f"analysis/tables/{name}" in outs
    assert "analysis/figures/selfconsistent_fcc.pdf" in outs


def _build():
    spec = importlib.util.spec_from_file_location("paper_build", PAPER / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flattened_sources_stand_alone():
    build = _build()
    for master in ("main", "supplement"):
        flat = build.flatten(PAPER / f"{master}.tex", bbl="\\begin{thebibliography}{1}\\end{thebibliography}\n")
        code = strip_comments(flat)
        left = [n for n in re.findall(r"\\input\{([^}]+)\}", code) if build.own_source(n)]
        assert not left, (master, left)
        assert "\\bibliography{" not in code and "\\bibliographystyle{" not in code
        assert code.count("\\begin{thebibliography}") == 1
        assert code.count("\\begin{document}") == 1 and code.count("\\end{document}") <= 1
        # every line of every section is in the flattened text, and so are the tables
        lines = set(flat.splitlines())
        for f in section_files(f"{master}.tex"):
            missing = [ln for ln in f.read_text().splitlines()
                       if ln not in lines and not build.own_source((re.findall(r"\\input\{([^}]+)\}", ln) or [""])[0] or "-")]
            assert not missing, (f, missing[:3])
            for name in re.findall(r"\\input\{([^}]+)\}", strip_comments(f.read_text())):
                assert set(build.own_source(name).read_text().splitlines()) <= lines, name
        figures = build.graphics(flat)
        assert figures and all(f.is_file() for f in figures), (master, figures)
    # a file read by \input ends with \relax\space (LaTeX's file hooks and the end of line)
    assert flat.count("\\relax\\space% end of ") >= len(section_files("supplement.tex"))
