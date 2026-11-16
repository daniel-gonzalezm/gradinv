"""Inventory of the supplementary material.

Each generated piece of the supplement with its generator(s) in this repository,
its outputs and the prefixes of its macros in paper/paper_numbers.tex.
tests/test_manuscript.py checks that every generator and output exists and that
every macro prefix has macros.

Run: python analysis/supplement.py   (prints the inventory)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (piece, generator(s), outputs, macro prefixes)
INVENTORY = [
    ("Table: gradient compliance of Al MEAM, Cu MEAM and Cu EAM",
     ["analysis/cubic_results.py", "analysis/make_tables.py"],
     ["analysis/tables/compliance.tex"], []),
    ("Table: provenance of every input number",
     ["analysis/make_tables.py"],
     ["analysis/tables/provenance.tex", "analysis/data/provenance.csv"], []),
    ("Figure: self-consistent scheme with full cubic grains, l(R), nu(R), residual(R)",
     ["analysis/selfconsistent_fcc.py", "analysis/make_figures.py"],
     ["analysis/figures/selfconsistent_fcc.pdf", "analysis/figures/selfconsistent_fcc.png",
      "analysis/results/selfconsistent_fcc.json"], []),
    ("Solyaev's Fe example: coherent reading with three averaged Eshelby tensors, "
     "anisotropic grain, mixed readings",
     ["analysis/fe_case.py", "analysis/solyaev_variants.py", "analysis/make_numbers.py"],
     ["analysis/results/fe_case.json"], ["NumFe"]),
    ("The three routes to the averaged Eshelby tensor of the sphere (Fourier, real space, "
     "Ma et al. eq. 3.12 averaged) and where eqs. 3.18-3.19 of Ma et al. fail",
     ["src/gradinv/eshelby.py", "tests/references/ma2018_real_space.py"],
     [], ["NumFeBone", "NumSixth"]),
    ("Index forms of the units of the isotropic, cubic and hexagonal algebras",
     ["src/gradinv/index_forms.py", "analysis/make_index_forms.py"],
     ["analysis/tables/index_forms.tex"], []),
]


def check():
    """List of problems (missing generator, output or macro); empty when the inventory holds."""
    numbers = (ROOT / "paper" / "paper_numbers.tex").read_text()
    problems = []
    for piece, gens, outs, prefixes in INVENTORY:
        for p in gens + outs:
            if not (ROOT / p).exists():
                problems.append(f"{piece}: missing {p}")
        for pre in prefixes:
            if f"\\{pre}" not in numbers:
                problems.append(f"{piece}: no macro {pre}...")
    return problems


if __name__ == "__main__":
    for piece, gens, outs, prefixes in INVENTORY:
        print(f"- {piece}\n    generators: {', '.join(gens)}\n    outputs: {', '.join(outs) or '-'}"
              f"\n    macros: {', '.join(p + '...' for p in prefixes) or '-'}")
    print("problems:", check() or "none")
