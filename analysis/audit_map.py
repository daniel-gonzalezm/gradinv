"""Audit map of the verification (supplement, section S2).

Single source of the map: for each claim and result of the paper, the key tests that
check it, what they compare against and the level of the comparison. Every test named
here is marked `core` in tests/ and every `core` test is named here exactly once
(tests/test_audit_map.py). `pytest -m core` runs them all, exact and slow included.

Levels:
    V   verification: against exact mathematics or an independent route of our own;
    CC  cross-comparison: against a published result, reconstructed, with its
        misprints corrected and declared;
    C   consistency: limit cases, invariances, inequalities checked on samples.
No comparison is with measurements: the inputs are published atomistic data.

Run: python analysis/audit_map.py   (writes analysis/tables/audit_map.tex)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "tables" / "audit_map.tex"
sys.path.insert(0, str(ROOT / "analysis"))

import citations  # noqa: E402

# (group, result, checked against, level, [(test file stem, test function), ...])
GROUPS = ("Claim 1: algebras", "Claim 2: compliance", "Claim 3: averages",
          "Illustrative application", "Data and manuscript")

MAP = [
    # ---------------------------------------------------------------- algebras (sections 2-4)
    (0, "Embeddings orthonormal; multiplication tables (15 and 32 elements)",
     "Gram matrix of the embeddings in exact arithmetic; all 225 and 1024 products "
     "evaluated in components", "V",
     [("cubic", "embeddings_gram_exact"), ("cubic", "multiplication_table_all_225_products"),
      ("hexagonal", "embeddings_gram_exact"),
      ("hexagonal", "multiplication_table_all_1024_products")]),
    (0, "Units invariant and spanning the commutant; dimensions 6/15/32 and 5/11/22",
     "exact invariance under the groups $O$ and $D_6$; commutant of the groups "
     "computed by linear algebra without the embeddings", "V",
     [("cubic", "oh_invariance_exact"), ("hexagonal", "d6_invariance_exact"),
      ("tables", "structure_table_matches_brute_force_commutant")]),
    (0, "Construction free of coordinates",
     "hexagonal units built from rotated axes equal the rotated units (exact, rational "
     "rotation)", "V",
     [("hexagonal", "coordinate_free_construction_exact")]),
    (0, "Extra hexagonal modulus: $m=\\pm3$ splitting of $\\Hrm3$; $\\Delta=4\\eta$",
     "the one direction of the hexagonal commutant outside the transversely isotropic "
     "one, computed without the basis; matrix of Auffray et al.\\ (2013)", "V, CC",
     [("hexagonal_extra_modulus", "extra_direction_is_the_m3_splitting_of_H3"),
      ("hexagonal", "auffray2013_exact")]),
    (0, "Agreement with published bases and constants",
     "the 31 transversely isotropic tensors of Monchiet and Bonnet (2013); the 11 "
     "constants of Lazar et al.\\ (2022) and their inverse map", "CC",
     [("hexagonal", "monchiet_bonnet_2013_exact"), ("cubic", "lazar2022_change_of_constants_exact")]),
    # ---------------------------------------------------------------- compliance (section 5)
    (1, "Block inverse of a general element",
     "product with the element in exact arithmetic (15 and 32 symbols); non-symmetric "
     "elements against the $18\\times18$ inverse", "V",
     [("cubic", "inverse_of_general_element_exact"),
      ("hexagonal", "inverse_of_general_element_exact"),
      ("odd_inverse", "odd_block_inverse_equals_18x18_inverse")]),
    (1, "Conditions by block; inverse and definiteness of the data",
     "spectrum of the $18\\times18$ restriction (400 random tensors and the five data "
     "sets)", "V",
     [("paper_formulas", "section5_seven_conditions_equal_positive_definiteness"),
      ("cubic", "data_blocks_inverse_and_positive_definiteness")]),
    (1, "Printed changes of basis (Lazar et al.\\ 2022; Auffray et al.\\ 2013)",
     "the formulas parsed from the manuscript, in exact arithmetic, against the package "
     "and the published matrix", "V, CC",
     [("paper_formulas", "section5_inverse_of_the_lazar_map_as_printed"),
      ("paper_formulas", "section4_change_of_basis_to_auffray2013_as_printed")]),
    (1, "Published compliances recovered",
     "isotropic inverse of Monchiet and Bonnet (2011a, eq.~2.13, corrected); separable "
     "compliance of Lazar and Po (2024, eqs.~97--102)", "CC",
     [("isotropic_sources", "monchiet_bonnet_2011a_inverse_exact"),
      ("source_data", "lazarpo2024_compliance_eqs_97_102")]),
    # ---------------------------------------------------------------- averages (section 6)
    (2, "Rule of the weighted means; Voigt estimate",
     "orthogonal projection on the isotropic algebra (exact); eq.~(158) of Lazar et al.\\ "
     "(2022)", "V, CC",
     [("averages", "weighted_rule_is_the_frobenius_projection_exact"),
      ("averages", "voigt_reproduces_lazar2022_eq158_exact")]),
    (2, "Reuss estimate in closed form",
     "Schur complement in exact arithmetic; average of compliances over random "
     "rotations (Monte Carlo)", "V",
     [("averages", "reuss_cubic_closed_schur_complement_exact"),
      ("averages", "monte_carlo_average_of_rotated_compliances")]),
    (2, "$\\tD^{\\mathrm R}\\le\\tD^{\\mathrm V}$; rank-one difference in "
     "$\\irr{T_1}$ (cubic)",
     "identity of the proof of section~6.2 in exact arithmetic; random tensors; "
     "generalised eigenvalues of the two blocks", "V, C",
     [("averages", "reuss_below_voigt_proof_identity_exact"),
      ("averages", "reuss_below_voigt_on_random_tensors"),
      ("main_results", "cubic_voigt_minus_reuss_on_H1_is_rank_one")]),
    (2, "Hexagonal: Voigt independent of $\\Delta$, Reuss even in $\\Delta$",
     "symbolic blocks", "V",
     [("averages", "hexagonal_dependence_on_the_extra_modulus_exact")]),
    (2, "Granular hexagonal tensor; $\\Delta=\\tfrac{1440}{13}(k_n-k_s)a_{66}$",
     "eq.~(74) of Pirmoradi et al.\\ (2024) for uniform contacts; transverse isotropy "
     "without $a_{66}$; the other blocks unchanged by $a_{66}$", "CC, C",
     [("main_results", "granular_hexagonal_tensor")]),
    # ---------------------------------------------------------------- application
    (3, "Concentration tensor by blocks",
     "$18\\times18$ inversion, cubic and hexagonal grains", "V",
     [("concentrator", "block_concentrator_equals_18x18_inversion")]),
    (3, "Averaged Eshelby tensor of the sphere; error of Ma et al.\\ (2018)",
     "real-space route against the Fourier route of the package; their eq.~(3.12) "
     "averaged holds, their eq.~(3.18) does not", "V, CC",
     [("eshelby_real_space", "real_space_average_equals_the_package"),
      ("eshelby_real_space", "ma_eq_312_is_right_and_eq_318_is_not")]),
    (3, "Self-consistent scheme",
     "isotropic grain as exact solution; closed large-grain limit against the full "
     "minimisation at $R=1000$\\,\\AA; Kr\\\"oner's estimate against Blaschke (2017, "
     "table~VI)", "C, V, CC",
     [("selfconsistent", "isotropic_grain_is_an_exact_solution"),
      ("application", "closed_limit_against_the_full_minimization_at_1000"),
      ("kroner_sources", "blaschke_table_vi")]),
    # ---------------------------------------------------------------- data and manuscript
    (4, "Input data transcribed and traced",
     "table~2 of Lazar et al.\\ (2022) recomputed from their table~1 by their eq.~(60); "
     "every input covered once by the provenance table", "CC, C",
     [("source_data", "table2_is_eq60_of_table1"),
      ("provenance", "every_data_entry_covered_exactly_once")]),
    (4, "Numbers, tables and figures of the paper",
     "regenerated and compared with the files; numbers recomputed by a separate route; "
     "no number typed by hand", "V, C",
     [("paper_numbers", "file_matches_the_generator"),
      ("paper_numbers", "manuscript_uses_only_defined_macros"),
      ("claim_figures", "values_by_a_separate_route"), ("figures", "redraw_gives_committed_files")]),
]


def core_tests():
    """(file name, function name) of every test named in the map, in order."""
    return [(f"test_{stem}.py", f"test_{func}") for *_, tests in MAP for stem, func in tests]


def _tt(stem, func):
    return "\\texttt{" + f"{stem}:{func}".replace(":", ":\\allowbreak{}").replace(
        "_", "\\_\\allowbreak{}") + "}"


def render():
    """Longtable for section S2. Its caption (with the label) is the macro
    \\AuditMapCaption, defined in paper/supplement/S2_verification.tex."""
    col = ">{\\raggedright\\arraybackslash}p"
    lines = ["% Generated by analysis/audit_map.py -- do not edit by hand.", "",
             "% needs the packages array, booktabs and longtable, and \\AuditMapCaption",
             f"\\begin{{longtable}}{{@{{}}{col}{{0.22\\linewidth}}{col}{{0.30\\linewidth}}"
             f"{col}{{0.06\\linewidth}}>{{\\raggedright\\arraybackslash\\scriptsize}}p{{0.32\\linewidth}}@{{}}}}",
             "\\caption{\\AuditMapCaption}\\\\",
             "\\toprule", "Result & Checked against & Level & \\footnotesize Core tests \\\\", "\\midrule",
             "\\endfirsthead",
             "\\toprule", "Result & Checked against & Level & \\footnotesize Core tests \\\\", "\\midrule",
             "\\endhead", "\\bottomrule", "\\endlastfoot"]
    for g, name in enumerate(GROUPS):
        if g:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\*")
        for grp, result, against, level, tests in MAP:
            if grp != g:
                continue
            names = "\\newline ".join(_tt(s, f) for s, f in tests)
            lines.append(f"{citations.cite(result)} & {citations.cite(against)} & {level} & {names} \\\\")
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def main():
    OUT.write_text(render())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
