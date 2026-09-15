#!/usr/bin/env python3
"""Exact quasi-normal modes of cv02's annulus — a continuum anchor for #907.

cv02 compares rfx's harminv modes against Meep's on a 2-D dielectric ring
(``n = 3.4``, inner radius ``1``, width ``1``, in units of ``a``). Both legs
are finite-grid solvers, so their mutual agreement is a *consistency*
statement: nothing in the case says how far the two discretizations are
allowed to be from the continuum, or from each other. That missing quantity
is ``ring_mode_judge.Q_GATE_INGREDIENTS``'s third ingredient, and it is still
**absent** (#907).

This script supplies the one piece of that campaign that needs no solver at
all: the **exact** quasi-normal modes of the same annulus, from a 4x4
Bessel/Hankel matching determinant. It is a diagnostic, not a crossval — it
compares nothing and gates nothing (``docs/agent-memory``: self-tests and
diagnostics do not belong in ``validation/crossval/``). What it gives a
reader is a continuum number to put beside both solvers, and the two
*leverage* channels that any future discretization budget would have to pick
between.

The physics
-----------
2-D TM (``Ez``) modes, azimuthal order ``m``, time convention
``e^{-i w t}`` so a decaying mode has ``Im f < 0`` and
``Q = Re f / (-2 Im f)``. With ``k = 2 pi f`` (``c = a = 1``):

* ``r < R_in``   : ``Ez = A J_m(k r)``              (regular at the origin)
* ``R_in<r<R_out``: ``Ez = B J_m(n k r) + C Y_m(n k r)``
* ``r > R_out``  : ``Ez = D H^(1)_m(k r)``          (outgoing)

``Ez`` and ``dEz/dr`` are continuous at both interfaces (TM, ``mu = 1``),
which is four homogeneous equations in ``(A, B, C, D)``. A mode is a complex
``f`` at which that 4x4 is singular.

Why the convergence criterion is NOT ``|det| < eps``
----------------------------------------------------
``det`` of this matrix has no implementation-independent scale. Multiplying
one matching equation by a constant (writing ``dEz/dr`` continuity in
different units), or rescaling one basis function (``H^(1)`` vs ``J + iY``,
``Y`` vs ``-Y``), multiplies the determinant by that constant while changing
no physics and moving no root. Three independent implementations of exactly
this matching condition report ``|det|`` at the SAME roots as ``1.4e-10``,
``~1e-17`` and ``~3e-17`` — seven orders of magnitude apart, all correct.

So the criterion here is :func:`normalized_residual`: the smallest singular
value of the 4x4 after equilibrating it to unit row norms AND unit column
norms. Row scaling is equation scaling and column scaling is basis scaling,
so the equilibrated matrix — and hence this residual — is invariant under
exactly the class of choices that made ``|det|`` useless. It lives in
``[0, 1]`` and is ``~1e-16`` at every root below, i.e. at machine precision.
:data:`RESIDUAL_MAX` is the pinned ceiling, four orders of margin above that.

Usage::

    python scripts/diagnostics/cv02_exact_annulus_qnm.py
    python scripts/diagnostics/cv02_exact_annulus_qnm.py --emit \
        tests/fixtures/cv02_ring_judge/exact_annulus_qnm.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import hankel1, jv, yv

# --- the geometry cv02 builds (validation/crossval/02_ring_resonator.py) -----

#: Ring index and the two radii, in units of ``a``. These are cv02's
#: ``n_wg``, ``ring_inner_radius_over_a`` and
#: ``ring_inner_radius_over_a + ring_width_over_a``, and they are also what
#: the retained record's ``rig`` block reports.
N_RING = 3.4
R_IN = 1.0
R_OUT = 2.0

#: cv02's harminv search band, widened slightly so a root just outside it is
#: still found and reported rather than silently missed.
F_SCAN_MIN, F_SCAN_MAX = 0.09, 0.21

#: Azimuthal orders that land in that band for this annulus.
M_ORDERS = (3, 4, 5)

#: Ceiling on :func:`normalized_residual` at an accepted root. The roots below
#: sit at ~1e-16 (machine precision on a unit-norm 4x4); this is four orders
#: looser, so a different LAPACK cannot red it, while a wrong root — which
#: lands at O(0.1-1) — cannot pass it.
RESIDUAL_MAX = 1e-10

#: Sinkhorn-style equilibration sweeps inside :func:`normalized_residual`.
#: The residual is invariant under row/column rescaling only once the
#: equilibration has converged: at 16 sweeps a 1e4-to-1e5 rescaling still
#: moves it by ~1e-3, at 64 by ~1e-8, at 200 by ~1e-14. Pinned by
#: ``test_the_annulus_residual_is_invariant_under_row_and_column_rescaling``.
EQUILIBRATION_SWEEPS = 200

#: Log-derivative step for the leverage channels. Central differences at 1e-5
#: agree with 1e-4 and 1e-6 to the digits printed.
LEVERAGE_H = 1e-5


def _cyl_deriv(fun, nu: int, z: complex) -> complex:
    """``C_nu'(z)`` for any cylinder function, by the standard recurrence
    ``C_nu'(z) = C_{nu-1}(z) - (nu/z) C_nu(z)``."""
    return fun(nu - 1, z) - nu / z * fun(nu, z)


def matching_matrix(f: complex, m: int, *, n: float = N_RING,
                    r_in: float = R_IN, r_out: float = R_OUT) -> np.ndarray:
    """The 4x4 whose singularity is a quasi-normal mode.

    Rows are (Ez at r_in, dEz/dr at r_in, Ez at r_out, dEz/dr at r_out);
    columns are the amplitudes ``(A, B, C, D)`` of the four basis solutions.
    """
    k = 2.0 * math.pi * f
    kn = n * k
    M = np.zeros((4, 4), dtype=complex)
    M[0, 0] = jv(m, k * r_in)
    M[0, 1] = -jv(m, kn * r_in)
    M[0, 2] = -yv(m, kn * r_in)
    M[1, 0] = k * _cyl_deriv(jv, m, k * r_in)
    M[1, 1] = -kn * _cyl_deriv(jv, m, kn * r_in)
    M[1, 2] = -kn * _cyl_deriv(yv, m, kn * r_in)
    M[2, 1] = jv(m, kn * r_out)
    M[2, 2] = yv(m, kn * r_out)
    M[2, 3] = -hankel1(m, k * r_out)
    M[3, 1] = kn * _cyl_deriv(jv, m, kn * r_out)
    M[3, 2] = kn * _cyl_deriv(yv, m, kn * r_out)
    M[3, 3] = -k * _cyl_deriv(hankel1, m, k * r_out)
    return M


def normalized_residual(f: complex, m: int, **geom) -> float:
    """Implementation-independent singularity measure of
    :func:`matching_matrix`, in ``[0, 1]``.

    Equilibrates the matrix to unit row AND column norms, then returns its
    smallest singular value. Because row scaling is equation scaling and
    column scaling is basis scaling, this number does not move when another
    implementation writes the same physics with different constants — which a
    bare ``|det|`` does, by many orders of magnitude (see the module
    docstring). Verified in
    ``test_the_annulus_residual_is_invariant_under_row_and_column_rescaling``.
    """
    M = matching_matrix(f, m, **geom)
    if not np.all(np.isfinite(M)):
        return 1.0
    for _ in range(EQUILIBRATION_SWEEPS):
        rn = np.linalg.norm(M, axis=1)
        if not np.all(rn > 0):
            return 1.0
        M = M / rn[:, None]
        cn = np.linalg.norm(M, axis=0)
        if not np.all(cn > 0):
            return 1.0
        M = M / cn[None, :]
    return float(np.linalg.svd(M, compute_uv=False)[-1])


def _scaled_det(f: complex, m: int, **geom) -> complex:
    """``det`` divided by the product of the row norms — same roots, but a
    scale that does not overflow or underflow across the scan."""
    M = matching_matrix(f, m, **geom)
    rn = np.linalg.norm(M, axis=1)
    if not np.all(np.isfinite(rn)) or not np.all(rn > 0):
        return complex("nan")
    return complex(np.linalg.det(M / rn[:, None]))


def _newton(f0: complex, m: int, *, iters: int = 80, **geom) -> complex:
    """Complex secant iteration on :func:`_scaled_det`."""
    a = complex(f0)
    b = a * (1.0 + 1e-7) + 1e-12j
    fa, fb = _scaled_det(a, m, **geom), _scaled_det(b, m, **geom)
    for _ in range(iters):
        if not (np.isfinite(fa) and np.isfinite(fb)) or fb == fa:
            break
        step = fb * (b - a) / (fb - fa)
        a, fa = b, fb
        b = b - step
        fb = _scaled_det(b, m, **geom)
        if abs(step) <= 1e-15 * max(abs(b), 1e-12):
            break
    return b


def _scan_roots(m: int, **geom) -> list[complex]:
    """Every accepted root of order ``m`` inside the declared search box.

    The box is ``Re f in [F_SCAN_MIN, F_SCAN_MAX]`` and ``Q in [10, 1e5]`` —
    written down before the answer, not around it. Every local minimum of
    ``|_scaled_det|`` on the grid is polished; a polished point is ACCEPTED
    only if its :func:`normalized_residual` is under :data:`RESIDUAL_MAX` and
    it is still in the box with ``Im f < 0``. Duplicates are merged.

    Reporting the accepted set rather than "the best grid point" is what
    makes this a search instead of a seeded evaluation: on cv02's annulus it
    returns exactly ONE root for each of m = 3, 4, 5, and
    :func:`find_mode` refuses any other count.
    """
    f_grid = np.linspace(F_SCAN_MIN, F_SCAN_MAX, 241)
    q_grid = np.logspace(1.0, 5.0, 41)
    grid = np.empty((f_grid.size, q_grid.size))
    for i, fr in enumerate(f_grid):
        for j, q in enumerate(q_grid):
            val = abs(_scaled_det(complex(fr, -fr / (2.0 * q)), m, **geom))
            grid[i, j] = val if np.isfinite(val) else np.inf
    candidates = []
    for i in range(f_grid.size):
        for j in range(q_grid.size):
            block = grid[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            if grid[i, j] <= block.min():
                candidates.append(
                    (grid[i, j],
                     complex(f_grid[i], -f_grid[i] / (2.0 * q_grid[j]))))
    candidates.sort(key=lambda item: item[0])
    roots: list[complex] = []
    for _val, start in candidates[:25]:
        root = _newton(start, m, **geom)
        if root.imag > 0:      # the conjugate root is the same physical mode
            root = root.conjugate()
        if not (np.isfinite(root.real) and np.isfinite(root.imag)):
            continue
        if not (F_SCAN_MIN <= root.real <= F_SCAN_MAX and root.imag < 0):
            continue
        if normalized_residual(root, m, **geom) > RESIDUAL_MAX:
            continue
        if not any(abs(root - seen) < 1e-9 for seen in roots):
            roots.append(root)
    return roots


def find_mode(m: int, *, guess: complex | None = None, **geom) -> complex:
    """The quasi-normal mode of azimuthal order ``m``.

    With no ``guess`` this searches the declared box (:func:`_scan_roots`)
    and requires the answer to be unique there; nothing seeds it. A ``guess``
    is used only by :func:`leverages`, which re-solves the SAME mode under a
    small parameter change and must not re-scan.
    """
    if guess is None:
        roots = _scan_roots(m, **geom)
        if len(roots) != 1:
            raise RuntimeError(
                f"m={m}: expected exactly one quasi-normal mode in "
                f"Re f in [{F_SCAN_MIN}, {F_SCAN_MAX}], Q in [1e1, 1e5]; "
                f"found {len(roots)}: {roots!r}")
        return roots[0]
    root = _newton(guess, m, **geom)
    if root.imag > 0:          # the conjugate root is the same physical mode
        root = root.conjugate()
    return root


def quality_factor(f: complex) -> float:
    """``Q = Re f / (-2 Im f)`` for the ``e^{-i w t}`` convention."""
    return float(f.real / (-2.0 * f.imag))


def leverages(m: int, root: complex, *, h: float = LEVERAGE_H) -> dict:
    """``dln f`` and ``dln Q`` per unit ``dln n`` and per unit ``dln R``.

    TWO channels, and they do not agree — which is the whole reason this
    function exists. A discretization budget expressed as "a permitted
    frequency error, transported into Q" has to declare WHICH channel it
    assumes the error lives in:

    * **index**: ``|dlnQ/dlnf| = 5..9``; a permitted frequency error buys a
      5-9x larger permitted Q error.
    * **uniform radius**: ``dlnf/dlnR = -1`` exactly and ``dlnQ/dlnR = 0``
      exactly (Maxwell is scale invariant, so scaling the whole annulus moves
      every frequency and no ``Q``); the same permitted frequency error buys
      **nothing**.

    So ``|dlnQ/dlnf|`` for this annulus ranges from ``0`` to ``8.85``
    depending on the assumed error channel. That is an assumption about how a
    given solver misrepresents the geometry, not an analytic property of the
    annulus, and #907's remaining ingredient cannot be closed by picking the
    end of the range that keeps a board green.
    """
    q0 = quality_factor(root)

    up = find_mode(m, guess=root, n=N_RING * (1 + h))
    dn = find_mode(m, guess=root, n=N_RING * (1 - h))
    dlnf_dlnn = (math.log(up.real) - math.log(dn.real)) / (2 * h)
    dlnq_dlnn = (math.log(quality_factor(up))
                 - math.log(quality_factor(dn))) / (2 * h)

    up = find_mode(m, guess=root / (1 + h),
                   r_in=R_IN * (1 + h), r_out=R_OUT * (1 + h))
    dn = find_mode(m, guess=root / (1 - h),
                   r_in=R_IN * (1 - h), r_out=R_OUT * (1 - h))
    dlnf_dlnr = (math.log(up.real) - math.log(dn.real)) / (2 * h)
    dlnq_dlnr = (math.log(quality_factor(up))
                 - math.log(quality_factor(dn))) / (2 * h)

    return {
        "Q_exact": q0,
        "dlnf_dlnn": dlnf_dlnn,
        "dlnQ_dlnn": dlnq_dlnn,
        "leverage_index_channel": abs(dlnq_dlnn / dlnf_dlnn),
        "dlnf_dlnR": dlnf_dlnr,
        "dlnQ_dlnR": dlnq_dlnr,
        "leverage_radius_channel": abs(dlnq_dlnr / dlnf_dlnr),
    }


# --- the two solvers, as the retained record reports them --------------------

#: ``validation/crossval/_02_ring_resonator_results/crossval.json``, the
#: committed 2026-09-06 run (commit ``296cabad``). Quoted, not measured here:
#: this script imports neither rfx nor Meep.
COMMITTED_RECORD = {
    3: {"meep_f": 0.11800975089381457, "meep_Q": 77.22925249140003,
        "rfx_f": 0.11807053109824135, "rfx_Q": 81.85693736750622},
    4: {"meep_f": 0.1471678979858206, "meep_Q": 341.4140208557494,
        "rfx_f": 0.14721298788921897, "rfx_Q": 358.1786218292347},
    5: {"meep_f": 0.17524600685125438, "meep_Q": 1747.8797310929497,
        "rfx_f": 0.1753089803297846, "rfx_Q": 1619.8638887641994},
}


def build() -> dict:
    """Solve every order and assemble the fixture payload."""
    modes = {}
    for m in M_ORDERS:
        root = find_mode(m)
        lev = leverages(m, root)
        rec = COMMITTED_RECORD[m]
        f_ex, q_ex = root.real, lev["Q_exact"]
        lv = lev["leverage_index_channel"]
        dlnf_rfx = math.log(rec["rfx_f"] / f_ex)
        dlnf_meep = math.log(rec["meep_f"] / f_ex)
        dlnf_gap = math.log(rec["rfx_f"] / rec["meep_f"])
        modes[str(m)] = {
            "m": m,
            "f_exact": f_ex,
            "imag_f_exact": root.imag,
            "Q_exact": q_ex,
            "normalized_residual": normalized_residual(root, m),
            "abs_det_unnormalized_DO_NOT_THRESHOLD": abs(
                complex(np.linalg.det(matching_matrix(root, m)))),
            **{k: v for k, v in lev.items() if k != "Q_exact"},
            "rfx_dlnf_vs_exact": dlnf_rfx,
            "rfx_dlnQ_vs_exact": math.log(rec["rfx_Q"] / q_ex),
            "meep_dlnf_vs_exact": dlnf_meep,
            "meep_dlnQ_vs_exact": math.log(rec["meep_Q"] / q_ex),
            "rfx_minus_meep_dlnf": dlnf_gap,
            "rfx_minus_meep_dlnQ": math.log(rec["rfx_Q"] / rec["meep_Q"]),
            # Transported frequency errors. NOT a budget -- see "note" below.
            "transported_rfx_minus_meep": abs(dlnf_gap) * lv,
            "transported_each_solver_vs_exact": (
                (abs(dlnf_rfx) + abs(dlnf_meep)) * lv),
        }
    return {
        "schema": "cv02_exact_annulus_qnm/1",
        "produced_by": "scripts/diagnostics/cv02_exact_annulus_qnm.py",
        "geometry": {"n_ring": N_RING, "r_in": R_IN, "r_out": R_OUT,
                     "polarization": "TM (Ez)", "dimension": 2,
                     "time_convention": "exp(-i w t), Im f < 0"},
        "residual_max": RESIDUAL_MAX,
        "solver_record": (
            "validation/crossval/_02_ring_resonator_results/crossval.json"
            " (commit 296cabad, 2026-09-06T17:07:57Z)"),
        "note": (
            "DIAGNOSTIC, NOT A GATE. The transported_* fields are the "
            "measured frequency disagreements multiplied by the INDEX-channel "
            "leverage; they are reported because they bound how much of the "
            "observed Q gap a frequency-error story can carry, and they do "
            "NOT bound it -- every one of them is well below the observed "
            "rfx_minus_meep_dlnQ. They are not a declared discretization "
            "budget; Q_GATE_INGREDIENTS[2] is still 'absent' (#907). The "
            "radius channel gives leverage 0 for the same frequency error, "
            "so the transport is an assumption about the error channel."),
        "modes": modes,
    }


def format_report(doc: dict) -> str:
    lines = ["Exact annulus quasi-normal modes (cv02 geometry, continuum)",
             f"  n = {N_RING}, {R_IN} < r < {R_OUT}, TM (Ez), 2-D, "
             f"e^(-iwt) so Im f < 0",
             "",
             f"  {'m':>2s} {'f_exact':>12s} {'Q_exact':>11s} "
             f"{'norm.resid':>11s} {'|det| (unusable)':>17s}"]
    for key in sorted(doc["modes"], key=int):
        d = doc["modes"][key]
        lines.append(
            f"  {d['m']:>2d} {d['f_exact']:>12.8f} {d['Q_exact']:>11.4f} "
            f"{d['normalized_residual']:>11.3e} "
            f"{d['abs_det_unnormalized_DO_NOT_THRESHOLD']:>17.3e}")
    lines += ["",
              "  leverage channels -- |dlnQ/dlnf| is NOT a single number:",
              f"  {'m':>2s} {'dlnf/dlnn':>10s} {'dlnQ/dlnn':>10s} "
              f"{'L_index':>9s} {'dlnf/dlnR':>10s} {'dlnQ/dlnR':>10s} "
              f"{'L_radius':>9s}"]
    for key in sorted(doc["modes"], key=int):
        d = doc["modes"][key]
        lines.append(
            f"  {d['m']:>2d} {d['dlnf_dlnn']:>10.6f} {d['dlnQ_dlnn']:>10.6f} "
            f"{d['leverage_index_channel']:>9.4f} {d['dlnf_dlnR']:>10.6f} "
            f"{d['dlnQ_dlnR']:>10.6f} {d['leverage_radius_channel']:>9.4f}")
    lines += ["",
              "  the two solvers against the continuum (committed record):",
              f"  {'m':>2s} {'rfx dlnf':>11s} {'rfx dlnQ':>10s} "
              f"{'meep dlnf':>11s} {'meep dlnQ':>10s} {'gap dlnQ':>10s} "
              f"{'transported':>12s}"]
    for key in sorted(doc["modes"], key=int):
        d = doc["modes"][key]
        lines.append(
            f"  {d['m']:>2d} {d['rfx_dlnf_vs_exact']:>11.3e} "
            f"{d['rfx_dlnQ_vs_exact']:>10.6f} "
            f"{d['meep_dlnf_vs_exact']:>11.3e} "
            f"{d['meep_dlnQ_vs_exact']:>10.6f} "
            f"{d['rfx_minus_meep_dlnQ']:>10.6f} "
            f"{d['transported_rfx_minus_meep']:>12.6f}")
    lines += ["", "  " + doc["note"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emit", type=Path, default=None,
                    help="write the fixture JSON to this path")
    args = ap.parse_args(argv)
    doc = build()
    print(format_report(doc))
    worst = max(d["normalized_residual"] for d in doc["modes"].values())
    if worst > RESIDUAL_MAX:
        print(f"\nFAIL: worst normalized residual {worst:.3e} > "
              f"{RESIDUAL_MAX:.1e}")
        return 1
    if args.emit is not None:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        print(f"\nwrote {args.emit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
