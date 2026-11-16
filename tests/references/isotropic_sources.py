"""Lazar (2016) pieces (3), (4) and moduli b1..b5; Monchiet and Bonnet (2011a)
projectors J1..J4, K1, K2 (eq. 2.6), table and inverse (eq. 2.13);
Auffray (2015) Tables 2 and 3. Checked on the rendered pages; the misprints are
kept as printed and corrected by the tests."""
import sympy as sp

from gradinv._tensor import N, R, TRIPLES, index, kd
from gradinv.skeleton import vector_basis


def lazar2016_embeddings():
    """Lazar (2016), eqs. (53)-(54), normalised: (E_(3), E_(4)) as 27 x 3 matrices."""
    def e4(c):
        out = sp.zeros(N, 1)
        for (i, j, m) in TRIPLES:
            out[index(i, j, m)] = kd(i, j) * c[m] / sp.sqrt(3)
        return out

    def e3(u):
        out = sp.zeros(N, 1)
        for (i, j, m) in TRIPLES:
            out[index(i, j, m)] = sp.sqrt(R(3, 20)) * (kd(i, m) * u[j] + kd(j, m) * u[i]
                                                        - R(2, 3) * kd(i, j) * u[m])
        return out
    E4 = sp.Matrix.hstack(*[e4(v) for v in vector_basis()])
    E3L = sp.Matrix.hstack(*[e3(v) for v in vector_basis()])
    return E3L, E4


def lazar2016_b(a):
    """b1..b5 of Lazar (2016), eqs. (75)-(79), in the a1..a5 of Lazar et al. (2022)."""
    return [2 * a[3] + 2 * a[4], 2 * a[3] - a[4], (10 * a[2] + 6 * a[3] + a[4]) / 3,
            2 * (3 * a[0] + 9 * a[1] + a[2] + 3 * a[3] + a[4]) / 3, 3 * a[0] + 2 * a[2] + 2 * a[4]]


def monchiet_bonnet_JK(T):
    """J1..J4, K1, K2 of Monchiet and Bonnet (2011a), eq. (2.6), from T1..T6."""
    T1, T2, T3, T4, T5, T6 = T
    J1 = (2 * T1 - T4) / 5
    J2 = (3 * T4 - T1) / 5
    J3 = (2 * T3 - T6) / 5
    J4 = (3 * T6 - T3) / 5
    K1 = (T2 + 2 * T5) / 3 - (J1 + J2) / 3 - 2 * (J3 + J4) / 3
    K2 = 2 * (T2 - T5) / 3 - (2 * J1 + J4) / 3 + (J2 + 2 * J3) / 3
    return J1, J2, J3, J4, K1, K2


# Monchiet and Bonnet (2011a), Table 2 (non-zero products among J1..J4)
MB2011A_TABLE2 = {("J1", "J1"): "J1", ("J1", "J2"): "J2", ("J2", "J3"): "J1", ("J2", "J4"): "J2",
                  ("J3", "J1"): "J3", ("J3", "J2"): "J4", ("J4", "J3"): "J3", ("J4", "J4"): "J4"}

# Auffray (2015), Table 2, with P^{H1,str} = Pss, Q^{str/rot} = Qsr, Q^{rot/str} = Qrs,
# P^{H1,rot} = Prr: the rows involved in the misprint of Table 2, as printed and corrected.
AUFFRAY2015_TABLE2_PRINTED = {("Qsr", "Qrs"): "Pss", ("Qsr", "Prr"): "Qsr", ("Qrs", "Pss"): "Qrs",
                              ("Qrs", "Qsr"): "Prr", ("Prr", "Qrs"): "Prr", ("Prr", "Prr"): "Qrs"}
AUFFRAY2015_TABLE2_CORRECTED = {("Qsr", "Qrs"): "Pss", ("Qsr", "Prr"): "Qsr", ("Qrs", "Pss"): "Qrs",
                                ("Qrs", "Qsr"): "Prr", ("Prr", "Qrs"): "Qrs", ("Prr", "Prr"): "Prr"}


def auffray2015_table3(D):
    """Auffray (2015), Table 3: (ms3, ms1, mr2, mr1, mc1) from components A_abcdef of D."""
    def A_(s):
        d = [int(ch) - 1 for ch in s]
        return D[index(d[0], d[1], d[2]), index(d[3], d[4], d[5])]
    ms3 = A_('111111') - A_('111221') - 4 * A_('122133') - 2 * A_('122331')
    ms1 = A_('111111') + R(2, 3) * (A_('111221') + 4 * A_('122133') + 2 * A_('122331'))
    mr2 = -A_('111111') / 2 - A_('111221') + 2 * A_('122133') + 4 * A_('122331') + 3 * A_('221221') / 2
    mr1 = R(1, 6) * (-3 * A_('111111') + 2 * A_('111221') + 20 * A_('122133') - 8 * A_('122331')
                     + 9 * A_('221221'))
    mc1 = sp.sqrt(5) / 3 * (-2 * A_('111221') + 4 * A_('122133') + 2 * A_('122331'))
    return ms3, ms1, mr2, mr1, mc1


# Auffray (2015), p. 581: first columns of [P^{H1,rot}] and [P^{H1,str}] in his
# normalised coordinates, order 111, 221, 122, 331, 133.
AUFFRAY2015_P581_ORDER = [(0, 0, 0), (1, 1, 0), (0, 1, 1), (2, 2, 0), (0, 2, 2)]
AUFFRAY2015_P581_ROT = [0, -1 / sp.sqrt(3), 1 / sp.sqrt(6), -1 / sp.sqrt(3), 1 / sp.sqrt(6)]
AUFFRAY2015_P581_STR = [sp.sqrt(R(3, 5)), 1 / sp.sqrt(15), sp.sqrt(R(2, 15)), 1 / sp.sqrt(15),
                        sp.sqrt(R(2, 15))]
