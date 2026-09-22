"""TE_n0 mode matching for the thick symmetric inductive iris in WR-90.

A thick symmetric inductive iris is EXACTLY a cascade of two symmetric
H-plane width-step junctions joined by a length-t section of guide of width
d.  This module solves that cascade by TE_{n0} mode matching (odd modes only,
relative-convergence ratio N_B/N_A ~ d/a for the edge condition) and returns
the TE10 -> TE10 (S11, S21) of the whole obstacle.  Pure numpy: it takes the
physical dimensions and knows nothing about any lattice, which is what makes
it usable as a reference for an FDTD run.

:func:`validate_oracle` runs the self-witnesses that must hold before any
number from here is compared with anything: unitarity at machine precision on
a lossless obstacle, convergence in the mode count, the thin-iris limit
against the Marcuvitz closed form B/Y0 = (lambda_g/a) cot^2(pi d / 2a) (a
leading-order anchor, so ~10 % agreement is the expectation), the d -> a
identity (no obstacle, no reflection) and the deep-constriction limit
(|S11| -> 1).

Moved here verbatim from the cross-validation script that used to carry the
WR-90 inductive iris, when the case became an analytic test; only this
docstring is new.  What stands behind it is not a second copy of the same
algebra but its limits: energy conservation on a lossless obstacle, reciprocity,
convergence in the mode count, the Marcuvitz thin-iris closed form, the
no-obstacle identity and the deep-constriction limit (``validate_oracle`` and
``tests/oracle/test_wr90_inductive_iris_mode_matching.py``).
"""
from __future__ import annotations

import numpy as np

C0 = 299792458.0
MU0 = 4e-7 * np.pi
A_WR90 = 22.86e-3


def _gamma(n, width, k):
    kc = n * np.pi / width
    return np.sqrt(complex(kc * kc - k * k))


def _overlap(a, d, n, m):
    x0 = (a - d) / 2.0
    al = n * np.pi / a
    be = m * np.pi / d

    def I_ss(p, q, L):
        if abs(p - q) < 1e-30:
            return L / 2 - np.sin(2 * p * L) / (4 * p)
        return (np.sin((p - q) * L) / (p - q) - np.sin((p + q) * L) / (p + q)) / 2

    def I_cs(p, q, L):
        if abs(p - q) < 1e-30:
            return (1 - np.cos(2 * q * L)) / (4 * q) if q > 0 else 0.0
        return ((1 - np.cos((q + p) * L)) / (q + p)
                + (1 - np.cos((q - p) * L)) / (q - p)) / 2

    val = np.cos(al * x0) * I_ss(al, be, d) + np.sin(al * x0) * I_cs(al, be, d)
    return np.sqrt(2 / a) * np.sqrt(2 / d) * val


def _step_junction(a, d, k, n_a, n_b):
    Na = np.arange(1, 2 * n_a, 2)
    Nb = np.arange(1, 2 * n_b, 2)
    gA = np.array([_gamma(n, a, k) for n in Na])
    gB = np.array([_gamma(m, d, k) for m in Nb])
    w = k * C0
    YA = gA / (1j * w * MU0)
    YB = gB / (1j * w * MU0)
    C = np.array([[_overlap(a, d, n, m) for m in Nb] for n in Na])
    YAd = np.diag(YA)
    Minv = np.linalg.inv(np.diag(YB) + C.T @ YAd @ C)
    T_ba = 2 * Minv @ C.T @ YAd
    R_aa = C @ T_ba - np.eye(n_a)
    R_bb = Minv @ (np.diag(YB) - C.T @ YAd @ C)
    T_ab = C @ (np.eye(n_b) + R_bb)
    sYA, sYB = np.sqrt(YA), np.sqrt(YB)
    S11 = (sYA[:, None] * R_aa) / sYA[None, :]
    S21 = (sYB[:, None] * T_ba) / sYA[None, :]
    S12 = (sYA[:, None] * T_ab) / sYB[None, :]
    S22 = (sYB[:, None] * R_bb) / sYB[None, :]
    return S11, S12, S21, S22


def _redheffer(sa, sb):
    A11, A12, A21, A22 = sa
    B11, B12, B21, B22 = sb
    n = A22.shape[0]
    inv1 = np.linalg.inv(np.eye(n) - A22 @ B11)
    inv2 = np.linalg.inv(np.eye(n) - B11 @ A22)
    return (A11 + A12 @ B11 @ inv1 @ A21,
            A12 @ inv2 @ B12,
            B21 @ inv1 @ A21,
            B22 + B21 @ A22 @ inv2 @ B12)


def iris_smatrix(a, d, t, freq, n_a=40):
    """TE10->TE10 (S11, S21) of the thick symmetric inductive iris."""
    k = 2 * np.pi * freq / C0
    n_b = max(4, int(round(n_a * d / a)))
    s_step = _step_junction(a, d, k, n_a, n_b)
    s_rev = (s_step[3], s_step[2], s_step[1], s_step[0])
    Nb = np.arange(1, 2 * n_b, 2)
    gB = np.array([_gamma(m, d, k) for m in Nb])
    P = np.diag(np.exp(-gB * t))
    z = np.zeros((n_b, n_b), dtype=complex)
    s_tot = _redheffer(_redheffer(s_step, (z, P, P, z)), s_rev)
    return s_tot[0][0, 0], s_tot[2][0, 0]


def marcuvitz_thin_b(a, d, freq):
    k = 2 * np.pi * freq / C0
    beta = np.sqrt(k * k - (np.pi / a) ** 2)
    return (2 * np.pi / beta / a) * (1.0 / np.tan(np.pi * d / (2 * a))) ** 2


def validate_oracle() -> dict:
    """Self-witnesses; raises on failure (refuse to gate on a broken oracle)."""
    w = {}
    f = 10e9
    s11, s21 = iris_smatrix(A_WR90, 12e-3, 2e-3, f)
    w["unitarity_dev"] = abs(abs(s11) ** 2 + abs(s21) ** 2 - 1.0)
    assert w["unitarity_dev"] < 1e-9
    v40 = abs(iris_smatrix(A_WR90, 12e-3, 2e-3, f, n_a=40)[0])
    v80 = abs(iris_smatrix(A_WR90, 12e-3, 2e-3, f, n_a=80)[0])
    w["mode_convergence"] = abs(v40 - v80)
    assert w["mode_convergence"] < 1e-3
    rels = []
    for d in (16e-3, 12e-3, 8e-3):
        s11t, _ = iris_smatrix(A_WR90, d, 1e-9, f)
        b_mm = np.real(-2 * s11t / (1j * (1 + s11t)))
        assert b_mm < 0, "inductive iris must have negative shunt susceptance"
        rels.append(abs(abs(b_mm) / marcuvitz_thin_b(A_WR90, d, f) - 1))
    w["marcuvitz_max_rel"] = max(rels)
    assert w["marcuvitz_max_rel"] < 0.15   # leading-order anchor, ~10% expected
    s11o, s21o = iris_smatrix(A_WR90, A_WR90 - 1e-9, 2e-3, f)
    w["open_limit_s11"] = abs(s11o)
    assert w["open_limit_s11"] < 1e-6
    s11d, _ = iris_smatrix(A_WR90, 4e-3, 6e-3, f)
    w["deep_limit_s11"] = abs(s11d)
    assert w["deep_limit_s11"] > 0.999
    return w
