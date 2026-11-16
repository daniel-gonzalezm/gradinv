"""Level (b) of explicitness: the units written in index form.

Each unit of the isotropic, cubic and hexagonal algebras is expanded, exactly
(sympy), in a fixed set of tensors built from delta and the structure tensors:

* isotropic: T1..T6 of Monchiet and Bonnet (2011a, eq. 2.3)
      (T1)_ijkpqr = d_ij d_pq d_kr          (T2) = I_ijpq d_kr
      (T3)        = I_ijkr d_pq             (T4) = I_pqkr d_ij
      (T5)        = (I_ijpr d_kq + I_ijqr d_kp)/2
      (T6)        = (I_pqir d_jk + I_pqjr d_ik)/2,   I_ijkl = (d_ik d_jl + d_il d_jk)/2
* cubic: T1..T6 and nine cubic tensors from the groups of Lazar et al. (2022, eq. 40),
  split into a tensor and its transpose where a group is not major-symmetric
  (DD_{...} = 1 iff all indices are equal):
      K6   = d_ik DD_jlmn + d_il DD_jkmn + d_jk DD_ilmn + d_jl DD_ikmn
      K7   = d_km DD_ijln + d_lm DD_ijkn       K7T = transpose
      K8   = d_mn DD_ijkl
      K9   = d_kl DD_ijmn                      K9T = transpose
      K10  = d_im DD_jkln + d_jm DD_ikln       K10T = transpose
      K11  = DD_ijklmn
* hexagonal: T_1..T_31 of Monchiet and Bonnet (2013, eqs. 32-33; explicit in n and
  pi = d - n n; transcribed from pp. 168-170) and the sixth-order structure tensor
  H6 = Re (a + i n x a)^{x6}; U_B1 = T_1/2 + H6/8, U_B2 = T_1/2 - H6/8.

`expansions(cls)` returns {unit key: {tensor name: coefficient}}; tests verify that
every expansion reconstructs its unit exactly.

LaTeX writer (supplementary section S1): `latex_index_forms()` writes the
isotropic and cubic expansions of the units with a <= b (the others are transposes)
and, for the hexagonal class, the compact form of the same expansions: in each block
p of transverse isotropy the tensors T_n of Monchiet and Bonnet (2013) are matrix units
in a non-orthonormal basis of copies, T^p_cd = P_p E_cd P_p^-1 in our blocks, so that
    U^p_ab = sum_cd (P_p^-1)_ca (P_p)_bd T_{g_p(c, d)}
with g_p the numbering of MB2013_GROUPS; `mb2013_similarity()` returns the P_p.
Run: python analysis/make_index_forms.py  (writes analysis/tables/index_forms.tex).
"""
from functools import lru_cache

import sympy as sp

from ._tensor import N, R, kd, frobenius, tensor6

__all__ = ["monchiet_bonnet_2011a_tensors", "cubic_tensors", "monchiet_bonnet_2013_tensors",
           "MB2013_GROUPS", "spanning_set", "expansions", "reconstruct",
           "mb2013_similarity", "hexagonal_from_similarity", "latex_index_forms"]


def monchiet_bonnet_2011a_tensors():
    """T1..T6 of Monchiet and Bonnet (2011a, eq. 2.3); rows (i,j,k), columns (p,q,r)."""
    def I4(a, b, c, d):
        return R(1, 2) * (kd(a, c) * kd(b, d) + kd(a, d) * kd(b, c))
    fs = [lambda i, j, k, p, q, r: kd(i, j) * kd(p, q) * kd(k, r),
          lambda i, j, k, p, q, r: I4(i, j, p, q) * kd(k, r),
          lambda i, j, k, p, q, r: I4(i, j, k, r) * kd(p, q),
          lambda i, j, k, p, q, r: I4(p, q, k, r) * kd(i, j),
          lambda i, j, k, p, q, r: R(1, 2) * (I4(i, j, p, r) * kd(k, q) + I4(i, j, q, r) * kd(k, p)),
          lambda i, j, k, p, q, r: R(1, 2) * (I4(p, q, i, r) * kd(j, k) + I4(p, q, j, r) * kd(i, k))]
    return [tensor6(f, exact=True) for f in fs]


def _DD(*idx):
    return 1 if len(set(idx)) == 1 else 0


def cubic_tensors():
    """The nine cubic tensors K6, K7, K7T, K8, K9, K9T, K10, K10T, K11 (dict)."""
    K = {
        "K6": tensor6(lambda i, j, m, k, l, n: kd(i, k) * _DD(j, l, m, n) + kd(i, l) * _DD(j, k, m, n)
                      + kd(j, k) * _DD(i, l, m, n) + kd(j, l) * _DD(i, k, m, n)),
        "K7": tensor6(lambda i, j, m, k, l, n: kd(k, m) * _DD(i, j, l, n) + kd(l, m) * _DD(i, j, k, n)),
        "K8": tensor6(lambda i, j, m, k, l, n: kd(m, n) * _DD(i, j, k, l)),
        "K9": tensor6(lambda i, j, m, k, l, n: kd(k, l) * _DD(i, j, m, n)),
        "K10": tensor6(lambda i, j, m, k, l, n: kd(i, m) * _DD(j, k, l, n) + kd(j, m) * _DD(i, k, l, n)),
        "K11": tensor6(lambda i, j, m, k, l, n: _DD(i, j, k, l, m, n)),
    }
    K["K7T"], K["K9T"], K["K10T"] = K["K7"].T, K["K9"].T, K["K10"].T
    return {k: K[k] for k in ("K6", "K7", "K7T", "K8", "K9", "K9T", "K10", "K10T", "K11")}


def monchiet_bonnet_2013_tensors(n=(0, 0, 1)):
    """T_1..T_31 of Monchiet and Bonnet (2013, eqs. 32-33), axis n (dict m -> 27 x 27)."""
    n = [sp.sympify(x) for x in n]

    def p(a, b):
        return kd(a, b) - n[a] * n[b]

    T = {}
    T[1] = lambda i, j, k, p_, q, r: (
        p(k, r) * (p(i, p_) * p(j, q) + p(j, p_) * p(i, q)) / 2 - 3 * p(i, j) * p(p_, q) * p(k, r) / 4
        + p(i, j) * (p(p_, r) * p(k, q) + p(q, r) * p(k, p_)) / 4
        + p(p_, q) * (p(i, r) * p(j, k) + p(j, r) * p(i, k)) / 4
        - p(i, k) * (p(j, p_) * p(q, r) + p(j, q) * p(p_, r)) / 4
        - p(j, k) * (p(i, p_) * p(q, r) + p(i, q) * p(p_, r)) / 4)
    T[2] = lambda i, j, k, p_, q, r: (
        (n[i] * n[p_] * p(j, q) + n[i] * n[q] * p(j, p_) + n[j] * n[p_] * p(i, q) + n[j] * n[q] * p(i, p_)) * p(k, r) / 4
        - (p(i, r) * n[j] + p(j, r) * n[i]) * n[q] * p(k, p_) / 4
        - (p(i, r) * n[j] + p(j, r) * n[i]) * n[p_] * p(k, q) / 4)
    T[3] = lambda i, j, k, p_, q, r: (p(i, p_) * p(j, q) + p(i, q) * p(j, p_) - p(i, j) * p(p_, q)) * n[k] * n[r] / 2
    T[4] = lambda i, j, k, p_, q, r: (
        -(n[p_] * p(q, r) + n[q] * p(p_, r)) * n[k] * p(i, j) / 2
        + (p(i, q) * p(j, r) + p(j, q) * p(i, r)) * n[p_] * n[k] / 2
        + (p(i, p_) * p(j, r) + p(j, p_) * p(i, r)) * n[q] * n[k] / 2)
    T[5] = lambda i, j, k, p_, q, r: (
        -(n[i] * p(j, k) + n[j] * p(i, k)) * p(p_, q) * n[r] / 4
        + (p(i, p_) * p(k, q) + p(i, q) * p(k, p_)) * n[j] * n[r] / 4
        + (p(j, p_) * p(k, q) + p(j, q) * p(k, p_)) * n[i] * n[r] / 4)
    T[6] = lambda i, j, k, p_, q, r: (
        (n[i] * n[p_] * p(j, q) + n[i] * n[q] * p(j, p_) + n[j] * n[p_] * p(i, q) + n[j] * n[q] * p(i, p_)) * p(k, r) / 4
        + (p(i, r) * n[j] * n[q] + p(j, r) * n[i] * n[q]) * p(k, p_) / 4
        + (p(i, r) * n[j] * n[p_] + p(j, r) * n[i] * n[p_]) * p(k, q) / 4
        - (n[i] * n[p_] * p(q, r) + n[i] * n[q] * p(p_, r)) * p(j, k) / 4
        - (n[j] * n[p_] * p(q, r) + n[j] * n[q] * p(p_, r)) * p(i, k) / 4)
    T[7] = lambda i, j, k, p_, q, r: p(i, j) * p(p_, q) * n[k] * n[r] / 2
    T[8] = lambda i, j, k, p_, q, r: (n[p_] * p(q, r) + n[q] * p(p_, r)) * n[k] * p(i, j) / 2
    T[9] = lambda i, j, k, p_, q, r: p(i, j) * n[p_] * n[q] * n[k] * n[r]
    T[10] = lambda i, j, k, p_, q, r: (n[i] * p(j, k) + n[j] * p(i, k)) * n[r] * p(p_, q) / 4
    T[11] = lambda i, j, k, p_, q, r: (n[p_] * p(q, r) + n[q] * p(p_, r)) * (n[i] * p(j, k) + n[j] * p(i, k)) / 4
    T[12] = lambda i, j, k, p_, q, r: (p(i, k) * n[j] + p(j, k) * n[i]) * n[r] * n[p_] * n[q] / 2
    T[13] = lambda i, j, k, p_, q, r: n[i] * n[j] * p(p_, q) * n[k] * n[r] / 2
    T[14] = lambda i, j, k, p_, q, r: (p(p_, r) * n[q] + p(q, r) * n[p_]) * n[i] * n[j] * n[k] / 2
    T[15] = lambda i, j, k, p_, q, r: n[i] * n[j] * n[k] * n[p_] * n[q] * n[r]
    T[16] = lambda i, j, k, p_, q, r: (
        p(j, k) * (p(i, p_) * p(q, r) + p(i, q) * p(p_, r) - p(p_, q) * p(i, r)) / 4
        + p(i, k) * (p(j, p_) * p(q, r) + p(j, q) * p(p_, r) - p(p_, q) * p(j, r)) / 4)
    T[17] = lambda i, j, k, p_, q, r: (
        p(j, k) * (3 * p(p_, q) * p(i, r) - p(i, p_) * p(q, r) - p(i, q) * p(p_, r)) / 8
        + p(i, k) * (3 * p(p_, q) * p(j, r) - p(j, p_) * p(q, r) - p(j, q) * p(p_, r)) / 8)
    T[18] = lambda i, j, k, p_, q, r: (p(i, r) * p(j, k) + p(j, r) * p(i, k)) * n[p_] * n[q] / 2
    T[19] = lambda i, j, k, p_, q, r: ((p(i, p_) * n[q] + p(i, q) * n[p_]) * p(j, k) * n[r] / 4
                                       + (p(j, p_) * n[q] + p(j, q) * n[p_]) * p(i, k) * n[r] / 4)
    T[20] = lambda i, j, k, p_, q, r: p(i, j) * (p(p_, r) * p(k, q) + p(q, r) * p(k, p_) - p(p_, q) * p(k, r)) / 2
    T[21] = lambda i, j, k, p_, q, r: p(i, j) * (3 * p(p_, q) * p(k, r) - p(p_, r) * p(k, q) - p(q, r) * p(k, p_)) / 4
    T[22] = lambda i, j, k, p_, q, r: p(i, j) * n[p_] * n[q] * p(k, r)
    T[23] = lambda i, j, k, p_, q, r: (n[p_] * p(q, k) + n[q] * p(p_, k)) * n[r] * p(i, j) / 2
    T[24] = lambda i, j, k, p_, q, r: (p(p_, r) * p(q, k) + p(q, r) * p(p_, k) - p(p_, q) * p(k, r)) * n[i] * n[j] / 2
    T[25] = lambda i, j, k, p_, q, r: (3 * p(p_, q) * p(k, r) - p(p_, r) * p(q, k) - p(q, r) * p(p_, k)) * n[i] * n[j] / 4
    T[26] = lambda i, j, k, p_, q, r: n[i] * n[j] * n[p_] * n[q] * p(k, r)
    T[27] = lambda i, j, k, p_, q, r: (p(p_, k) * n[q] + p(q, k) * n[p_]) * n[i] * n[j] * n[r] / 2
    T[28] = lambda i, j, k, p_, q, r: (
        (p(i, p_) * p(q, r) + p(i, q) * p(p_, r)) * n[j] * n[k] / 2
        + (p(j, p_) * p(q, r) + p(j, q) * p(p_, r)) * n[i] * n[k] / 2
        - (n[i] * p(j, r) + n[j] * p(i, r)) * n[k] * p(p_, q) / 2)
    T[29] = lambda i, j, k, p_, q, r: (
        3 * (n[i] * p(j, r) + n[j] * p(i, r)) * n[k] * p(p_, q) / 4
        - (p(i, p_) * p(q, r) + p(i, q) * p(p_, r)) * n[j] * n[k] / 4
        - (p(j, p_) * p(q, r) + p(j, q) * p(p_, r)) * n[i] * n[k] / 4)
    T[30] = lambda i, j, k, p_, q, r: (p(i, r) * n[j] + p(j, r) * n[i]) * n[k] * n[p_] * n[q]
    T[31] = lambda i, j, k, p_, q, r: (p(i, p_) * n[j] * n[q] + p(i, q) * n[j] * n[p_]
                                       + p(j, p_) * n[i] * n[q] + p(j, q) * n[i] * n[p_]) * n[k] * n[r] / 2
    return {m: tensor6(f, exact=True) for m, f in T.items()}


GROUPS = {'T1': [[1]], 'T2': [[2]], 'E2': [[3, 4], [5, 6]],
          'A2': [[7, 8, 9], [10, 11, 12], [13, 14, 15]],
          'E1': [[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]}


MB2013_GROUPS = GROUPS


@lru_cache(maxsize=None)
def spanning_set(cls):
    """(names, tensors) of the fixed tensors used for class 'isotropic', 'cubic' or 'hexagonal'."""
    T = monchiet_bonnet_2011a_tensors()
    if cls == "isotropic":
        return tuple(f"T{n + 1}" for n in range(6)), tuple(T)
    if cls == "cubic":
        K = cubic_tensors()
        return (tuple(f"T{n + 1}" for n in range(6)) + tuple(K),
                tuple(T) + tuple(K.values()))
    if cls == "hexagonal":
        from .hexagonal import structure_tensors
        T13 = monchiet_bonnet_2013_tensors()
        H6 = structure_tensors(exact=True)[2]
        return (tuple(f"T{m}" for m in range(1, 32)) + ("H6",),
                tuple(T13[m] for m in range(1, 32)) + (H6,))
    raise ValueError(cls)


@lru_cache(maxsize=None)
def expansions(cls):
    """{unit key: {tensor name: exact coefficient}} for the units of class cls."""
    from . import cubic, hexagonal, isotropic
    alg = {"isotropic": isotropic, "cubic": cubic, "hexagonal": hexagonal}[cls].algebra(exact=True)
    names, tensors = spanning_set(cls)
    n = len(tensors)
    gram = sp.Matrix(n, n, lambda a, b: frobenius(tensors[a], tensors[b]))
    ginv = gram.inv()
    out = {}
    for key, U in alg.units().items():
        rhs = sp.Matrix([frobenius(T, U) for T in tensors])
        coef = (ginv * rhs).applyfunc(lambda x: sp.nsimplify(sp.simplify(x)))
        out[key] = {names[i]: coef[i] for i in range(n) if coef[i] != 0}
    return out


def reconstruct(cls, terms):
    """Sum of coefficient x tensor for an expansion {name: coefficient}."""
    names, tensors = spanning_set(cls)
    lookup = dict(zip(names, tensors))
    out = sp.zeros(N, N)
    for name, c in terms.items():
        out += c * lookup[name]
    return out


# ---------------------------------------------------------------- hexagonal: similarity form
def _integer_scale(M):
    """Smallest positive integer s such that s*M has entries k*sqrt(m) with k integer."""
    den = 1
    for e in M:
        if e == 0:
            continue
        r, _rest = sp.nsimplify(e).as_coeff_Mul()
        den = sp.ilcm(den, sp.fraction(sp.Rational(r))[1])
    return den


@lru_cache(maxsize=None)
def mb2013_similarity():
    """{p: P_p} for p in E2, A2, E1: T^p_cd (Monchiet and Bonnet 2013) = P_p E_cd P_p^-1.

    P_p is fixed up to a scalar; it is scaled to the smallest integer multiple of the
    first column of T_{g_p(c, 0)} applied to the image of T_{g_p(0, 0)}.
    """
    from . import hexagonal
    alg = hexagonal.algebra(exact=True)
    T = monchiet_bonnet_2013_tensors()
    out = {}
    for p in ("E2", "A2", "E1"):
        grid = GROUPS[p]
        n = len(grid)
        mats = {(c, d): alg.moduli(T[grid[c][d]])[p] for c in range(n) for d in range(n)}
        v = mats[(0, 0)].columnspace()[0]
        P = sp.Matrix.hstack(*[sp.simplify(mats[(c, 0)] * v) for c in range(n)])
        P = (P * _integer_scale(P)).applyfunc(sp.radsimp)
        out[p] = sp.ImmutableMatrix(P)
    return out


def hexagonal_from_similarity():
    """{unit key: {tensor name: coefficient}} of the hexagonal units from the P_p."""
    from . import hexagonal
    out = {("A1", "2", "2"): {"T2": sp.Integer(1)},
           ("B1", "3", "3"): {"T1": R(1, 2), "H6": R(1, 8)},
           ("B2", "3", "3"): {"T1": R(1, 2), "H6": -R(1, 8)}}
    copies = {p: c for p, c, _d in hexagonal.IRREPS}
    for p, P in mb2013_similarity().items():
        Pi = P.inv()
        grid = GROUPS[p]
        n = len(grid)
        for a in range(n):
            for b in range(n):
                terms = {}
                for c in range(n):
                    for d in range(n):
                        coef = sp.radsimp(Pi[c, a] * P[b, d])
                        if coef != 0:
                            terms[f"T{grid[c][d]}"] = coef
                out[(p, copies[p][a], copies[p][b])] = terms
    return out


# ---------------------------------------------------------------- LaTeX writer
_ISO_TEX = {f"T{n}": rf"\mathbb T^{{\mathrm{{iso}}}}_{{{n}}}" for n in range(1, 7)}
_CUB_TEX = dict(_ISO_TEX)
for _k in ("K6", "K7", "K8", "K9", "K10", "K11"):
    _CUB_TEX[_k] = rf"\mathbb G_{{{_k[1:]}}}"
    _CUB_TEX[_k + "T"] = rf"\mathbb G_{{{_k[1:]}}}^{{\mathsf T}}"
_IRREP_TEX = {"H3": "3", "H2": "2", "H1": "1", "T1": "T_1", "T2": "T_2", "E": "E", "A2": "A_2",
              "A1": "A_1", "B1": "B_1", "B2": "B_2", "E1": "E_1", "E2": "E_2"}


def _unit_tex(key):
    p, a, b = key
    single = {("H3", "3"), ("H2", "2"), ("E", "e"), ("A2", "a")}
    if (p, a) in single and a == b:
        return rf"\tP{{{_IRREP_TEX[p]}}}"
    return rf"\tU{{{_IRREP_TEX[p]}}}{{{a}}}{{{b}}}"


def _coef_tex(c, first):
    c = sp.nsimplify(c)
    neg = c.could_extract_minus_sign()
    mag = -c if neg else c
    r, rest = mag.as_coeff_Mul()
    r = sp.Rational(r)
    num, den = r.p, r.q
    root = "" if rest == 1 else sp.latex(rest)
    if den == 1:
        body = (str(num) if num != 1 or not root else "") + root
    else:
        top = (str(num) if num != 1 or not root else "") + root
        body = rf"\tfrac{{{top}}}{{{den}}}"
    sign = ("-" if neg else "") if first else ("-" if neg else "+")
    return sign + body


def _expansion_lines(terms, names_tex, order, per_line=5):
    items = [(k, terms[k]) for k in order if k in terms]
    parts = []
    for i, (k, c) in enumerate(items):
        parts.append(_coef_tex(c, i == 0) + names_tex[k])
    lines = [parts[i:i + per_line] for i in range(0, len(parts), per_line)]
    return [" ".join(l) for l in lines]


def _align(rows):
    out = [r"\begin{align*}"]
    for i, (lhs, lines) in enumerate(rows):
        for j, l in enumerate(lines):
            lead = f"{lhs}&=" if j == 0 else r"&\quad"
            end = r"\\" if (i < len(rows) - 1 or j < len(lines) - 1) else ""
            out.append(f"{lead}{l}{end}")
    out.append(r"\end{align*}")
    return out


def _upper_keys(cls, exp):
    from . import cubic, isotropic
    irreps = {"isotropic": isotropic, "cubic": cubic}[cls].algebra(exact=True).irreps
    keys = []
    for p, copies, _d in irreps:
        for i, a in enumerate(copies):
            for b in copies[i:]:
                keys.append((p, a, b))
    assert set(keys) <= set(exp)
    return keys


def _matrix_tex(P):
    rows = [" & ".join(sp.latex(sp.radsimp(e)) for e in P.row(i)) for i in range(P.rows)]
    return r"\begin{pmatrix}" + r"\\ ".join(rows) + r"\end{pmatrix}"


def latex_index_forms():
    """LaTeX of the index forms of the units (supplementary material, section S1)."""
    out = ["% Generated by analysis/make_index_forms.py from gradinv.index_forms -- do not edit by hand.", ""]
    iso_order = [f"T{n}" for n in range(1, 7)]
    cub_order = iso_order + ["K6", "K7", "K7T", "K8", "K9", "K9T", "K10", "K10T", "K11"]
    for cls, names_tex, order, title in (("isotropic", _ISO_TEX, iso_order, "Isotropic units"),
                                         ("cubic", _CUB_TEX, cub_order, "Cubic units")):
        exp = expansions(cls)
        per_line = 6 if cls == "isotropic" else 5
        rows = [(_unit_tex(k), _expansion_lines(exp[k], names_tex, order, per_line))
                for k in _upper_keys(cls, exp)]
        out += [rf"\paragraph{{{title}.}}", *_align(rows), ""]
    sim = mb2013_similarity()
    out += [r"\paragraph{Hexagonal units.}",
            r"\begin{gather*}",
            r"\tP{A_1}=\mathbb T^{\mathrm{ti}}_{2},\qquad",
            r"\tP{B_1}=\tfrac12\mathbb T^{\mathrm{ti}}_{1}+\tfrac18\tHsix,\qquad",
            r"\tP{B_2}=\tfrac12\mathbb T^{\mathrm{ti}}_{1}-\tfrac18\tHsix,\\",
            r"\tU{p}{a}{b}=\sum_{c,d=1}^{n_p}(P_p^{-1})_{ca}(P_p)_{bd}\,"
            r"\mathbb T^{\mathrm{ti}}_{g_p(c,d)}\quad(p=A_2,E_1,E_2),",
            r"\end{gather*}"]
    names = {"E2": "E_2", "A2": "A_2", "E1": "E_1"}
    grids = []
    for p in ("A2", "E1", "E2"):
        g = GROUPS[p]
        n = len(g)
        off = g[0][0] - n - 1
        grids.append(rf"g_{{{names[p]}}}(c,d)=" + (f"{off}+" if off else "") + f"{n}c+d")
    out += [r"\begin{gather*}",
            r",\qquad ".join(grids) + r",\\",
            rf"P_{{A_2}}={_matrix_tex(sim['A2'])},\qquad P_{{E_2}}={_matrix_tex(sim['E2'])},\\",
            rf"P_{{E_1}}={_matrix_tex(sim['E1'])}.",
            r"\end{gather*}", ""]
    return "\n".join(out)
