"""Build the manuscript and the supplement with Elsevier's CAS single-column class.

The class is a publisher file and is not stored in the repository. Pass the folder
of Elsevier's CAS templates (els-cas-templates: cas-sc.cls and cas-common.sty); the
bibliography style (unsrtnat, from natbib) comes with TeX Live:

    python paper/build.py --cls <template folder> [--final] [--check] [--bundle]

Output in paper/build/ (ignored by git): main.pdf, supplement.pdf and a report of
pages, LaTeX errors, undefined references and citations, overfull boxes, words
set past the right margin of the text block, and bitmap (Type 3) fonts.

- Tables and figures are read where the generators write them (analysis/tables,
  analysis/figures) through TEXINPUTS; nothing is copied by hand.
- --final hides the \\pending boxes (\\draftnotesfalse).
- --check exits with code 1 on a LaTeX error, an undefined reference or citation,
  an overfull box wider than 1 pt, a word past the right margin, or a bitmap font.
- Fonts: the class loads STIX (stix.sty) and sets tables, captions and the front-matter
  labels in Computer Modern sans with the T1 encoding; without stix.sty, or without the
  Type 1 EC fonts of cm-super, TeX falls back to bitmap (Type 3) fonts, which this
  build reports (pdffonts).
- --bundle (implies --final) writes the submission sources to paper/build/bundle/:
  main.tex and supplement.tex with every \\input and the bibliography (.bbl) inlined,
  the figures and the class files. The folder is compiled on its own,
  without the repository, and the text of its PDFs must equal that of the normal
  build. Two archives are written next to it: gradinv-arxiv.tar.gz (manuscript
  sources, with the supplement as the ancillary file anc/supplement.pdf) and
  gradinv-source.zip (manuscript and supplement sources).
- The margin check reads the PDF (pdftotext -bbox) because TeX does not report a
  display that is too wide inside gathered/pmatrix. Text block: \\textwidth (written to
  the log by macros.tex) from the most frequent left edge; the title page (class
  layout) and rotated pages (pdflscape) are skipped.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
BUILD = PAPER / "build"
MASTERS = ("main", "supplement")
CLASS_FILES = ("cas-sc.cls", "cas-common.sty")
CLASS_DIRS = ("thumbnails",)       # icons the title page sets beside the e-mail address
SOURCE_SUFFIXES = (".tex", ".cls", ".sty", ".pdf", ".jpeg", ".jpg", ".png")


def texinputs(cls_dir):
    dirs = [PAPER, PAPER / "sections", PAPER / "supplement", ROOT / "analysis" / "tables",
            ROOT / "analysis" / "figures", cls_dir]
    return ":".join(str(d) for d in dirs) + ":"


def run(cmd, env):
    return subprocess.run(cmd, cwd=BUILD, env=env, capture_output=True, text=True, errors="replace")


def build(master, env, final):
    job = ["-jobname", master] if final else []
    src = (r"\def\gradinvfinal{1}\input{" + master + "}") if final else master + ".tex"
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error"] + job
    first = run(latex + [src], env)
    aux = BUILD / f"{master}.aux"
    bib = None
    if aux.exists() and r"\bibdata" in aux.read_text(errors="replace"):
        bib = run(["bibtex", master], env)
    for _ in range(2):
        last = run(latex + [src], env)
    return first, bib, last


def class_lines(master, source=None):
    """Source lines where the class itself builds the title page (\\maketitle); the
    overfull boxes reported there come from the class (the publisher's own template
    shows the same ones) and are not counted."""
    lines = (source or PAPER / f"{master}.tex").read_text().splitlines()
    at = [i + 1 for i, ln in enumerate(lines) if ln.strip().startswith(r"\maketitle")]
    return {n + k for n in at for k in (0, 1, 2)}  # the paragraph ends on the next line(s)


def problems(master, folder=None, source=None):
    folder = folder or BUILD
    log = (folder / f"{master}.log").read_text(errors="replace")
    errors = [ln for ln in log.splitlines() if re.match(r"^(!|.*:\d+: )", ln)]
    undefined = re.findall(r"(?:Reference|Citation) `([^']+)' .*undefined", log)
    skip = class_lines(master, source)
    over = [float(w) for w, a, b in re.findall(
        r"Overfull \\hbox \(([\d.]+)pt too wide\) in paragraph at lines (\d+)--(\d+)", log)
        if not (int(a) == int(b) and int(a) in skip)]
    blg = folder / f"{master}.blg"
    bib_warn = [ln for ln in blg.read_text(errors="replace").splitlines()
                if ln.startswith(("Warning--", "I couldn't", "I found no"))] if blg.exists() else []
    return errors, sorted(set(undefined)), [o for o in over if o > 1.0], bib_warn


def text_width_pt(log):
    """\\textwidth in big points (PDF units), as macros.tex writes it to the log."""
    m = re.search(r"GRADINV-TEXTWIDTH=([\d.]+)pt", log.read_text(errors="replace"))
    if not m:
        raise SystemExit(f"no GRADINV-TEXTWIDTH line in {log}")
    return float(m.group(1)) * 72 / 72.27


def rotated_pages(pdf):
    """Pages whose /Rotate is not 0 (landscape pages of pdflscape)."""
    n = pages(pdf)
    out = subprocess.run(["pdfinfo", "-f", "1", "-l", str(n), str(pdf)],
                         capture_output=True, text=True).stdout
    return {int(a) for a, r in re.findall(r"^Page\s+(\d+) rot:\s+(\d+)", out, re.M) if r != "0"}
_WORD = re.compile(r'xMin="([\d.]+)" yMin="[\d.]+" xMax="([\d.]+)" yMax="[\d.]+">([^<]*)<')


def margin_overflow(pdf, width, tol=1.0):
    """Words that start inside the text block and end past its right edge, or start
    left of its left edge, as (page, text). Skips page 1 (title page) and landscape pages."""
    html = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"], capture_output=True, text=True).stdout
    pages_ = [_WORD.findall(p) for p in html.split("<page ")[1:]]
    lefts = {}
    for words in pages_:
        for a, _c, _w in words:
            lefts[round(float(a))] = lefts.get(round(float(a)), 0) + 1
    x0 = max(lefts, key=lefts.get) if lefts else 0
    right = x0 + width
    skip = {1} | rotated_pages(pdf)
    out = []
    for i, words in enumerate(pages_, 1):
        if i in skip:
            continue
        out += [(i, w) for a, c, w in words
                if (float(c) > right + tol and float(a) < right + 15) or float(a) < x0 - tol]
    return out


def bitmap_fonts(pdf):
    """Names of the Type 3 (bitmap) fonts of a PDF (pdffonts)."""
    out = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    return sorted({ln.split()[0] for ln in out.splitlines()[2:] if " Type 3 " in ln})


def pages(pdf):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------- submission bundle
BUNDLE = BUILD / "bundle"
_INPUT = re.compile(r"\\input\{([^}]+)\}")
_GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_PENDING = re.compile(r"\\pending(?:cite)?\{")


def _code(line):
    """The part of a source line before its comment."""
    m = re.search(r"(?<!\\)%", line)
    return line[:m.start()] if m else line


def own_source(name):
    """The repository file that \\input{name} reads, or None for anything else."""
    name = name if name.endswith(".tex") else name + ".tex"
    for folder in (PAPER, PAPER / "sections", PAPER / "supplement", ROOT / "analysis" / "tables"):
        if (folder / name).is_file():
            return folder / name
    return None


def flatten(path, bbl=None):
    """The source of `path` with every \\input of a repository file replaced by the
    file, recursively, and (if `bbl` is given) the bibliography commands replaced by
    the text of the .bbl file. Comments are kept."""
    out = []
    for line in path.read_text().splitlines():
        code = _code(line)
        if bbl is not None and r"\bibliographystyle{" in code:
            continue
        if bbl is not None and r"\bibliography{" in code:
            out.append(bbl.rstrip("\n"))
            continue
        m = _INPUT.search(code)
        src = own_source(m.group(1)) if m else None
        if src is None:
            out.append(line)
            continue
        before, after = code[:m.start()].rstrip(), _code(line[m.end():]).strip()
        out += ([before] if before else []) + [flatten(src, bbl).rstrip("\n")]
        # For TeX, a file read by \input ends with LaTeX's file hooks (which stop an
        # \ignorespaces pending from the last environment of the file), and the end of
        # the \input line is a space. \relax and \space keep both, so that the
        # flattened source typesets exactly as the original.
        glued = after or line[m.end():].startswith("%")   # no space after the \input
        out.append((r"\relax" if glued else r"\relax\space") + f"% end of {src.name}")
        out += [after] if after else []
    return "\n".join(out) + "\n"


def graphics(text):
    """Figure files (PDF) named by the \\includegraphics commands of a flattened source."""
    names = [m.group(1) for line in text.splitlines() for m in _GRAPHIC.finditer(_code(line))]
    return [ROOT / "analysis" / "figures" / (n if n.endswith(".pdf") else n + ".pdf") for n in names]


def pending_boxes(text):
    """Uses of \\pending or \\pendingcite left in a flattened source (not their definitions)."""
    return [ln.strip() for ln in text.splitlines()
            if _PENDING.search(_code(ln)) and r"\newcommand" not in ln]


def _commit():
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return out or "unknown commit"


def _pdf_text(pdf):
    return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout


def make_bundle(cls_dir):
    """Write paper/build/bundle/ from the final build in paper/build/, compile it on its
    own and write the two archives. Returns the list of problems found."""
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir()
    bad, used = [], {}
    for name in CLASS_FILES:
        shutil.copy(cls_dir / name, BUNDLE)
    for name in CLASS_DIRS:
        if (cls_dir / name).is_dir():
            shutil.copytree(cls_dir / name, BUNDLE / name)
    head = (f"% Written by paper/build.py --bundle from {_commit()} of\n"
            "% https://github.com/daniel-gonzalezm/gradinv; every \\input and the bibliography\n"
            "% are inlined. Edit the sources in the repository, not this file.\n"
            "\\pdfoutput=1 % pdfLaTeX (arXiv reads this in the first lines)\n"
            "\\def\\gradinvfinal{1}% submission version: no drafting boxes\n")
    for master in MASTERS:
        text = flatten(PAPER / f"{master}.tex", (BUILD / f"{master}.bbl").read_text())
        (BUNDLE / f"{master}.tex").write_text(head + text)
        for fig in graphics(text):
            if fig.is_file():
                shutil.copy(fig, BUNDLE)
            else:
                bad.append(f"{master}: figure {fig.name} not found")
        for box in pending_boxes(text):
            print(f"    note, {master}: drafting box left (hidden in the PDF): {box[-70:]}")
    env = dict(os.environ, TEXINPUTS="", BIBINPUTS="", BSTINPUTS="")
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-recorder"]
    for master in MASTERS:
        for _ in range(3):
            subprocess.run(latex + [f"{master}.tex"], cwd=BUNDLE, env=env, capture_output=True,
                           text=True, errors="replace")
        pdf = BUNDLE / f"{master}.pdf"
        if not pdf.exists():
            bad.append(f"{master}: the bundle does not compile on its own")
            continue
        errors, undefined, over, _ = problems(master, BUNDLE, BUNDLE / f"{master}.tex")
        if errors or undefined or over:
            bad.append(f"{master}: {len(errors)} errors, {len(undefined)} undefined, {len(over)} overfull")
        if _pdf_text(pdf) != _pdf_text(BUILD / f"{master}.pdf"):
            bad.append(f"{master}: the PDF of the bundle differs from the PDF of the build")
        fls = (BUNDLE / f"{master}.fls").read_text(errors="replace").splitlines()
        read = set()
        for ln in fls:
            if not ln.startswith("INPUT "):
                continue
            f = Path(ln[6:])
            f = f if f.is_absolute() else (BUNDLE / f).resolve()
            if f.is_relative_to(BUNDLE.resolve()):
                read.add(f.relative_to(BUNDLE.resolve()).as_posix())
        used[master] = sorted(n for n in read if (BUNDLE / n).is_file()
                              and Path(n).suffix in SOURCE_SUFFIXES and n != f"{master}.pdf")
        print(f"bundle, {master}: {pages(pdf)} pages, compiled without the repository; "
              f"{len(used[master])} files: {', '.join(used[master])}")
    pdfs = BUILD / "bundle-pdf"                 # compiled copies, kept apart from the sources
    if pdfs.exists():
        shutil.rmtree(pdfs)
    pdfs.mkdir()
    keep = {n for names in used.values() for n in names}
    for f in sorted(BUNDLE.rglob("*"), reverse=True):     # files before their folders
        rel = f.relative_to(BUNDLE).as_posix()
        if f.is_dir():
            if not any(f.iterdir()):
                f.rmdir()
        elif f.suffix == ".pdf" and f.parent == BUNDLE and f.stem in MASTERS:
            shutil.move(f, pdfs / f.name)
        elif rel not in keep:
            f.unlink()
    if bad:
        return bad
    with tarfile.open(BUILD / "gradinv-arxiv.tar.gz", "w:gz") as tar:
        for n in used["main"]:
            tar.add(BUNDLE / n, arcname=n)
        tar.add(pdfs / "supplement.pdf", arcname="anc/supplement.pdf")
    with zipfile.ZipFile(BUILD / "gradinv-source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for n in sorted(keep):
            z.write(BUNDLE / n, n)
    print("bundle: paper/build/gradinv-arxiv.tar.gz, paper/build/gradinv-source.zip")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", type=Path, required=True, help="folder with cas-sc.cls and cas-common.sty")
    ap.add_argument("--final", action="store_true", help="hide the pending boxes")
    ap.add_argument("--check", action="store_true", help="exit 1 on any problem")
    ap.add_argument("--bundle", action="store_true",
                    help="write the submission sources and archives (implies --final)")
    args = ap.parse_args()
    args.final = args.final or args.bundle
    cls_dir = args.cls.resolve()
    missing = [n for n in CLASS_FILES if not (cls_dir / n).exists()]
    if missing:
        sys.exit(f"{', '.join(missing)} not found in {cls_dir}")
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir()
    env = dict(os.environ, TEXINPUTS=texinputs(cls_dir), BIBINPUTS=f"{PAPER}:")
    bad = False
    for master in MASTERS:
        build(master, env, args.final)
        errors, undefined, over, bib_warn = problems(master)
        pdf = BUILD / f"{master}.pdf"
        wide = margin_overflow(pdf, text_width_pt(BUILD / f"{master}.log")) if pdf.exists() else []
        type3 = bitmap_fonts(pdf) if pdf.exists() else []
        print(f"{master}: {pages(pdf) if pdf.exists() else 'no PDF'} pages; {len(errors)} errors; "
              f"{len(undefined)} undefined refs/cites; {len(over)} overfull boxes > 1 pt; "
              f"{len(wide)} words past the margin; {len(bib_warn)} bibtex warnings; "
              f"{len(type3)} bitmap fonts")
        for line in (errors[:8] + [f"  undefined: {u}" for u in undefined[:8]] + bib_warn[:8]
                     + [f"  past the margin, page {pg}: {w}" for pg, w in wide[:8]]):
            print("   ", line)
        if type3:
            print("    bitmap (Type 3) fonts: install stix and cm-super (TeX Live)")
        bad |= bool(errors or undefined or over or wide or type3 or not pdf.exists())
    if args.bundle and not bad:
        found = make_bundle(cls_dir)
        for line in found:
            print("    bundle:", line)
        bad |= bool(found)
    return 1 if (args.check and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
