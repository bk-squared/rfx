#!/usr/bin/env python3
"""Issue #517 step 1: single-ratio residue vs multi-drive solve, one measurement.

PRE-DECLARATION (written before this script was run — R2-tight, ONE attempt;
the solve-composition redesign, i.e. resolving the Kurokawa sqrt(Z) mixing
across families "properly", is explicitly OUT of scope this session per the
task brief — this script MEASURES, it does not redesign or ship anything).

## What is being compared

`compute_mixed_s_matrix()` (`rfx/api/_sparams.py`) assembles its S-matrix via
`_assemble_mixed_power_wave_s`, which forms `S[i, j] = b_i / a_j` per drive
(the SAME single-ratio rule issue #507 replaced with a `S = B*A^-1` solve for
the pure-MSL lane, `msl_solve_s_from_waves`) — issue #517 asks whether doing
the analogous solve for the MIXED (lumped/wire + MSL) lane collapses most of
the lane's known ~9% reciprocity residual (the number quoted in `#498` and in
`compute_mixed_s_matrix`'s own `reciprocity_tol=0.06` docstring, measured on
this lane's own reference fixture).

Both paths are built from the IDENTICAL recorded phasors of ONE FDTD run
(monkeypatch-captured from the exact call `_assemble_mixed_power_wave_s`
receives inside `compute_mixed_s_matrix` — no re-derivation, no second run):

  * SHIPPED (single-ratio): `S[i, j] = b_i / a_j`, captured byte-for-byte as
    `_assemble_mixed_power_wave_s` computes it in production.
  * SOLVE (candidate generalization): `S = B * A^-1` via the ALREADY-VALIDATED
    `msl_solve_s_from_waves` (issue #507's pure-MSL generalization, reused
    verbatim, not reimplemented), fed `wave_a[d][j]` / `wave_b[d][j]` — the
    incident/reflected wave at EVERY port j on EVERY drive d, not just the
    driven port.

## Construction caveat (read before trusting the SOLVE number)

For MSL ports the (a, b) pair used by the shipped path is ALREADY
non-degenerate and already evaluated uniformly at every port on every drive
(`a = (V0 + Z_hj*I)/2sqrt(Z_hj)`, `b = (V0 - Z_hj*I)/2sqrt(Z_hj)` — this is
exactly what `_b_msl` already computes, driven or passive, in the shipped
code) — no invention needed there.

For lumped/wire ports the shipped path's own "a" formula (drive wave,
`a = (-V + Z0c*I)/(2sqrt(Z0c))`) and its "passive b" formula (`_b_lw_passive`,
`b = (V - Z0c*I)/(2sqrt(Z0c))`) are ALGEBRAIC NEGATIVES of each other for ANY
(V, I) — `b_passive(V,I) == -a(V,I)` identically, by construction of the two
formulas, not as a physics fact. Applying both to the SAME port's own (V, I)
would give a degenerate (trivial) A/B pair, so this script does NOT do that.
Instead it pairs "a" with the shipped DIAGONAL formula (`b_diag =
(-V - Z0*I)/(2sqrt(Z0))` for lumped, or the shipped Zin-reflection formula
`S_jj = (Zin-Z0)/(Zin+Z0)` with `b := S_jj * a` for wire), applied to EVERY
lumped/wire port's own (V, I) on EVERY drive — not just the driven port's own
run. This is algebraically non-degenerate (verified: `b_diag(V,I) ==
-a_std(V,I)` and `a(V,I) == -b_std(V,I)` for the standard Kurokawa pair
`a_std/b_std`, so `(a, b_diag)` IS the standard pair up to an overall sign and
swap) — but it is a NAIVE, first-pass extension: it reuses the shipped
DRIVEN-port diagonal formula at PASSIVE ports too, which is not independently
validated (this is exactly the "composition" question #517 defers). Read the
SOLVE numbers below as a diagnostic signal, not a validated replacement.

## Fixture (pre-declared, the #488 lane's own committed lumped/wire<->MSL
   fixture, `tests/test_mixed_port_sparam.py::_base_sim/_add_feed/_add_msl`,
   reused verbatim — imitating the lane's own test invocation)

  RO4350B-like substrate eps_r=3.66, h_sub=254um, trace width=600um,
  dx=80um (the lane's own committed mesh; NOT the h_sub/3 alignment the
  #511/#507 entry recommends for NEW fixtures — this is the EXISTING
  committed fixture, not a new one), domain 8mm x 3mm x 754um, PEC z_lo /
  CPML elsewhere, cpml_layers=8. Vertical wire feed (`add_port`,
  component="ez", impedance=50, extent=h_sub) at x=2mm; MSL port
  (`add_msl_port`, direction="-x", facing the feed per the #488 D1 fix) at
  x=5.5mm, n_probe_offset=10, n_probe_spacing=4 (the smoke test's own probe-
  ladder override — the default ladder walks past x=0 on this domain size,
  see `test_guard_rejects_cpml_adjacent_probe_ladder`). wire_mode=True (feed
  has `extent` set).

  num_periods=60 (bumped from the fast-suite smoke test's deliberately
  truncated num_periods=4 — that test asserts bookkeeping only and WARNS on
  purpose; this measurement needs real settling). freqs = 5 points,
  1-4 GHz (same band as the smoke test). skip_preflight=False — preflight
  output is quoted verbatim below, per repo rule.

## Pre-declared expectations (falsifiable; not adjusted after the run)

  (a) If the single-ratio rule is the dominant source of the lane's ~9%
      reciprocity residual, the SOLVE's reciprocity deviation should be
      substantially smaller than the SHIPPED path's — "collapse toward the
      low single digits" per the task brief.
  (b) cond(A) (the drive-matrix condition number the solve inverts) should
      stay O(1) — no near-singular drive system.
  (c) The diagonals may MOVE between the two paths (both are unverified per
      #498) — recorded for the record, NO gate on them.

## R1 memory citations
  - `rfx-known-issues.md` "#511 + #507 MSL extractor" entry: `S = B.A^-1`
    (same algebra as the coax lane's #489 tool) is the validated pattern this
    script reuses via `msl_solve_s_from_waves`, not reimplements.
  - `project_lumped_twoport_vi_lane.md`: port-cell waves are a
    do-not-anchor quantity (#313) — consistent: this measurement compares
    two ASSEMBLY rules on the SAME recorded port-cell phasors, it does not
    change or re-derive the phasors themselves.
  - `feedback_gate_can_bind_artifact.md`: this script asserts NOTHING as a
    pass/fail gate — it is a diagnostic measurement with a verdict printed
    against the pre-declared expectations above, not a committed test.

## R3 pre-commit self-audit (filled in after running, left in place)
  1. Contradicted by known memory? See R1 above — none found after grep.
  2. R2 trip? No — this is attempt 1 of 1 for this measurement; the prior
     R2-HARD-STOP history (`20260728_i488_falsifier_ledger.md`) was for the
     ARCHITECTURE-B (extractor honesty) track, a different mechanism
     hypothesis than "does a linear solve remove the single-ratio residue".
  3. Falsifier: expectation (a) above — the solve's reciprocity deviation
     vs the shipped path's, on the SAME recorded data.

## RESULT (2026-08-03, one run, settling -122.6 / -119.9 dB — clean, not
   truncated; preflight and warnings quoted verbatim in the JSON artifact)

  Both pre-declared expectations FAIL on this construction:

  (a) FAIL. Wave-channel (pre-flux-override) reciprocity deviation:
      SHIPPED (single-ratio) = 64.4%, SOLVE (multi-drive) = 92.3% — the
      solve makes reciprocity WORSE, not better, over 1-4 GHz (5 points).
      (For scale: the shipped DEFAULT result's own runtime witness — on
      the flux-magnitude channel, a different, later step — independently
      warned "reciprocity deviation max 10.5%" on this SAME run, close to
      the historically-quoted ~9% figure; that number is NOT what (a)
      above measures, see "Construction caveat" and the note below.)
  (b) FAIL by the task's <10 bar: cond(A) = 27.1 / 12.2 / 3.6 / 2.5 / 24.9
      across the band (U-shaped, worst at the edges). Not catastrophic by
      the pure-MSL lane's own precedent threshold (<1e3,
      `test_msl_modal_voltage_and_wave_solve.py`), but far from O(1).
  (c) Diagonals MOVED a lot, asymmetrically: |S_lw0,lw0| shipped 0.38-0.40
      -> solve ~0.382 (small change); |S_msl0,msl0| shipped 0.018-0.034 ->
      solve 0.72-0.72 (~20-40x). No gate, recorded per the pre-declaration.

  DIAGNOSIS (from the algebra, not a new run): of the four (a, b) matrix
  entries the solve needs, three reuse ALREADY-VALIDATED shipped quantities
  byte-for-byte (the lw port's own driven-run a/diagonal-b; the msl port's
  a/b at every run, identical to what `_b_msl` already computes uniformly
  today). Only ONE entry is new: the lw port's own (a, b) evaluated from
  ITS OWN (V, I) during the run where it is PASSIVE (matched-terminated) —
  the "naive first-pass" quantity flagged in the Construction caveat above.
  The msl diagonal blowing up (0.03 -> 0.72) despite the msl port's own
  formulas being untouched shows the 2x2 matrix inverse propagates that one
  suspect entry into EVERY output. This is consistent with (not proof of) a
  specific physical reading: a matched-Z0-terminated port's local V/I sits
  close to Z0 by construction of the termination, so the local-Zin-based
  "b" this script computes there is dominated by measurement noise, not a
  real reflected-wave signal — exactly the composition gap #517 names and
  defers ("how it composes ... is the actual work, not the linear algebra").

  VERDICT vs the task's framing: the measurement does NOT support "the
  single-ratio rule is the dominant source of the ~9% flux-channel
  residual" via this naive solve construction — it shows the opposite
  under the only mechanically-available generalization. It does NOT rule
  out a properly-derived composition (this script's LW-passive-port entry
  is explicitly not validated); it shows that skipping the composition
  question and reusing the shipped diagonal formula at passive ports is
  not a viable shortcut.

  RECOMMENDATION: do not extend the solve to the mixed lane on this
  construction. If #517 is pursued further, the prerequisite is a
  validated formula for the lumped/wire "a" (and independently, "b") at a
  PASSIVE, matched-terminated port from its own (V, I) — which does not
  currently exist anywhere in the codebase (grepped: `decompose_lumped_s_
  matrix` / `decompose_wire_s_matrix` in `rfx/probes/probes.py` also only
  ever evaluate "a" at a port's OWN drive, never at a passive port) — that
  is a new falsifier, not a rerun of this one (R2).
"""
from __future__ import annotations

import contextlib
import io
import json
import warnings
from pathlib import Path

import numpy as np

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse
import rfx.api._sparams as _sparams_mod
from rfx.api._sparams import _mixed_reciprocity_deviation, msl_solve_s_from_waves

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "scripts" / "diagnostics" / "i517_mixed_solve_vs_ratio"

# ---------------------------------------------------------------------------
# Fixture — verbatim from tests/test_mixed_port_sparam.py (_base_sim,
# _add_feed, _add_msl), the #488 lane's own committed lumped/wire<->MSL
# fixture. Not re-derived; imitates the lane's own test invocation.
# ---------------------------------------------------------------------------
_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
_DX = 80e-6


def _base_sim(**kw):
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=_DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        **kw,
    )
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB + _DX)), material="pec")
    return sim, y_c


def _add_msl(sim, y_c, x=5.5e-3, direction="-x", **kw):
    sim.add_msl_port(position=(x, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction=direction, impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5), **kw)


def _add_feed(sim, y_c, x=2e-3):
    # Vertical probe feed: ez wire port ground-plane -> trace bottom.
    sim.add_port(position=(x, y_c, 0.0), component="ez",
                 impedance=50.0, extent=_H_SUB)


# ---------------------------------------------------------------------------
# Multi-drive solve: generalize the shipped a/b formulas to EVERY port on
# EVERY drive (see "Construction caveat" above), then reuse the already-
# validated `msl_solve_s_from_waves` (issue #507) verbatim.
# ---------------------------------------------------------------------------

def _uniform_wave_amplitudes(
    v_lw, i_lw, v0_msl, i_msl, z0_lw, n_live_lw, z0_hj_msl,
    wire_mode, drive_plan,
):
    """Per-(run, port) (a, b) using each port's OWN recorded phasors.

    MSL: the shipped `_b_msl` / drive-wave formulas, unchanged, already
    valid at every port on every run.

    Lumped/wire: pairs the shipped drive-wave "a" with the shipped DIAGONAL
    "b" (byte-identical formulas to `_assemble_mixed_power_wave_s`), applied
    to every port's own (V, I) regardless of whether it was driven this run
    — NOT paired with `_b_lw_passive`, which is an algebraic negative of "a"
    (see module docstring "Construction caveat").
    """
    v_lw = np.asarray(v_lw)
    i_lw = np.asarray(i_lw)
    v0_msl = np.asarray(v0_msl)
    i_msl = np.asarray(i_msl)
    n_runs, n_lw, _ = v_lw.shape
    n_msl = v0_msl.shape[1]
    z0_lw = np.asarray(z0_lw, dtype=np.float64)
    z0c_lw = z0_lw / np.maximum(np.asarray(n_live_lw, dtype=np.int64), 1)
    sq_lw = np.sqrt(z0c_lw)
    sqfull_lw = np.sqrt(z0_lw)
    z0_hj_msl = np.asarray(z0_hj_msl, dtype=np.float64)
    sq_msl = np.sqrt(z0_hj_msl)

    wave_a: list = []
    wave_b: list = []
    for run in range(n_runs):
        a_row, b_row = [], []
        for j in range(n_lw):
            V, I = v_lw[run, j], i_lw[run, j]
            a = (-V + z0c_lw[j] * I) / (2.0 * sq_lw[j])
            if wire_mode:
                safe_i = np.where(np.abs(I) > 0, I, 1e-30)
                z_in = -V / safe_i
                s_local = (z_in - z0_lw[j]) / (z_in + z0_lw[j])
                b = s_local * a
            else:
                b = (-V - z0_lw[j] * I) / (2.0 * sqfull_lw[j])
            a_row.append(a)
            b_row.append(b)
        for p in range(n_msl):
            V0, I = v0_msl[run, p], i_msl[run, p]
            a = (V0 + z0_hj_msl[p] * I) / (2.0 * sq_msl[p])
            b = (V0 - z0_hj_msl[p] * I) / (2.0 * sq_msl[p])
            a_row.append(a)
            b_row.append(b)
        wave_a.append(a_row)
        wave_b.append(b_row)
    return wave_a, wave_b


def main() -> int:
    sim, y_c = _base_sim()
    _add_feed(sim, y_c, x=2e-3)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)
    freqs = np.linspace(1e9, 4e9, 5)
    num_periods = 60.0

    captured: dict = {}
    orig_assemble = _sparams_mod._assemble_mixed_power_wave_s

    def _capturing_assemble(v_lw, i_lw, v0_msl, i_msl, z0_lw, n_live_lw,
                            z0_hj_msl, wire_mode, drive_plan):
        S, s21_power = orig_assemble(
            v_lw, i_lw, v0_msl, i_msl, z0_lw, n_live_lw, z0_hj_msl,
            wire_mode, drive_plan,
        )
        captured.update(
            v_lw=np.asarray(v_lw), i_lw=np.asarray(i_lw),
            v0_msl=np.asarray(v0_msl), i_msl=np.asarray(i_msl),
            z0_lw=np.asarray(z0_lw), n_live_lw=np.asarray(n_live_lw),
            z0_hj_msl=np.asarray(z0_hj_msl), wire_mode=bool(wire_mode),
            drive_plan=list(drive_plan), S_shipped=np.asarray(S),
        )
        return S, s21_power

    _sparams_mod._assemble_mixed_power_wave_s = _capturing_assemble
    stdout_buf = io.StringIO()
    caught_warnings: list = []
    try:
        with contextlib.redirect_stdout(stdout_buf), \
             warnings.catch_warnings(record=True) as wr:
            warnings.simplefilter("always")
            result = sim.compute_mixed_s_matrix(
                freqs=freqs, num_periods=num_periods, skip_preflight=False,
            )
            caught_warnings = [str(w.message) for w in wr]
    finally:
        _sparams_mod._assemble_mixed_power_wave_s = orig_assemble

    preflight_text = stdout_buf.getvalue()
    print("=" * 78)
    print("PREFLIGHT (verbatim)")
    print("=" * 78)
    print(preflight_text)
    print("=" * 78)
    print("WARNINGS (verbatim)")
    print("=" * 78)
    for w in caught_warnings:
        print(f"  - {w}")
    print()

    settling_db = np.asarray(result.settling_db)
    print(f"settling_db per run: {settling_db.tolist()}")
    print(f"drive_plan: {captured['drive_plan']}")
    print(f"wire_mode: {captured['wire_mode']}")

    S_shipped = captured["S_shipped"]
    wave_a, wave_b = _uniform_wave_amplitudes(
        captured["v_lw"], captured["i_lw"], captured["v0_msl"],
        captured["i_msl"], captured["z0_lw"], captured["n_live_lw"],
        captured["z0_hj_msl"], captured["wire_mode"], captured["drive_plan"],
    )
    S_solve, cond_a = msl_solve_s_from_waves(wave_a, wave_b)
    S_solve = np.asarray(S_solve)
    cond_a = np.asarray(cond_a)

    dev_shipped = _mixed_reciprocity_deviation(S_shipped)
    dev_solve = _mixed_reciprocity_deviation(S_solve)

    n_tot = S_shipped.shape[0]
    port_names = list(result.port_names)
    rows = []
    for k in range(len(freqs)):
        row = {
            "freq_hz": float(freqs[k]),
            "S_shipped": [[complex(S_shipped[i, j, k]).real,
                          complex(S_shipped[i, j, k]).imag]
                         for i in range(n_tot) for j in range(n_tot)],
            "S_solve": [[complex(S_solve[i, j, k]).real,
                        complex(S_solve[i, j, k]).imag]
                       for i in range(n_tot) for j in range(n_tot)],
            "abs_S_shipped": {
                f"{i}{j}": float(np.abs(S_shipped[i, j, k]))
                for i in range(n_tot) for j in range(n_tot)
            },
            "abs_S_solve": {
                f"{i}{j}": float(np.abs(S_solve[i, j, k]))
                for i in range(n_tot) for j in range(n_tot)
            },
            "cond_a_solve": float(cond_a[k]) if cond_a is not None else None,
        }
        rows.append(row)

    print()
    print("=" * 78)
    print("BOTH-WAYS TABLE (per frequency)")
    print("=" * 78)
    header = f"{'f(GHz)':>7s}"
    for i in range(n_tot):
        for j in range(n_tot):
            header += f"  |S{i}{j}|ship  |S{i}{j}|solve"
    header += "   cond(A)"
    print(header)
    for k in range(len(freqs)):
        line = f"{freqs[k] / 1e9:7.3f}"
        for i in range(n_tot):
            for j in range(n_tot):
                line += (f"  {np.abs(S_shipped[i, j, k]):9.4f}"
                        f"  {np.abs(S_solve[i, j, k]):9.4f}")
        line += f"  {cond_a[k]:8.3f}" if cond_a is not None else "     n/a"
        print(line)

    print()
    print("=" * 78)
    print("RECIPROCITY DEVIATION (max relative |S_ij| vs |S_ji| over the band)")
    print("=" * 78)
    print(f"  SHIPPED (single-ratio): pair={dev_shipped[0] if dev_shipped else None}"
         f" deviation={dev_shipped[1] if dev_shipped else None:.4f}"
         if dev_shipped else "  SHIPPED: n/a (need >=2 ports)")
    print(f"  SOLVE  (multi-drive):   pair={dev_solve[0] if dev_solve else None}"
         f" deviation={dev_solve[1] if dev_solve else None:.4f}"
         if dev_solve else "  SOLVE: n/a (need >=2 ports)")

    diag_shipped = {
        port_names[i]: [complex(S_shipped[i, i, k]) for k in range(len(freqs))]
        for i in range(n_tot)
    }
    diag_solve = {
        port_names[i]: [complex(S_solve[i, i, k]) for k in range(len(freqs))]
        for i in range(n_tot)
    }
    print()
    print("DIAGONALS (both suspect per #498 — recorded, no gate)")
    for name in port_names:
        print(f"  shipped |S_{name}{name}| = "
             f"{[round(abs(v), 4) for v in diag_shipped[name]]}")
        print(f"  solve   |S_{name}{name}| = "
             f"{[round(abs(v), 4) for v in diag_solve[name]]}")

    dev_shipped_val = dev_shipped[1] if dev_shipped else None
    dev_solve_val = dev_solve[1] if dev_solve else None
    cond_max = float(np.max(cond_a)) if cond_a is not None else None

    verdict = {
        "a_solve_collapses_residual": (
            dev_solve_val is not None and dev_shipped_val is not None
            and dev_solve_val < 0.5 * dev_shipped_val
        ),
        "a_shipped_deviation": dev_shipped_val,
        "a_solve_deviation": dev_solve_val,
        "b_cond_a_stays_o1": (cond_max is not None and cond_max < 10.0),
        "b_cond_a_max": cond_max,
        "c_diagonals_shipped": {
            k: [[c.real, c.imag] for c in v] for k, v in diag_shipped.items()
        },
        "c_diagonals_solve": {
            k: [[c.real, c.imag] for c in v] for k, v in diag_solve.items()
        },
    }

    print()
    print("=" * 78)
    print("EXPECTATION VERDICTS (pre-declared)")
    print("=" * 78)
    print(f"  (a) solve collapses reciprocity residual toward low single "
         f"digits: shipped={dev_shipped_val}, solve={dev_solve_val} -> "
         f"{'PASS' if verdict['a_solve_collapses_residual'] else 'FAIL'}")
    print(f"  (b) cond(A) stays O(1) (< 10): max={cond_max} -> "
         f"{'PASS' if verdict['b_cond_a_stays_o1'] else 'FAIL'}")
    print("  (c) diagonals recorded, no gate (see table above)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "i517_mixed_solve_vs_ratio.json"
    out_path.write_text(json.dumps({
        "fixture": {
            "eps_r": _EPS_R, "h_sub_m": _H_SUB, "w_trace_m": _W_TRACE,
            "dx_m": _DX, "feed_x_m": 2e-3, "msl_x_m": 5.5e-3,
            "num_periods": num_periods, "freqs_hz": freqs.tolist(),
        },
        "settling_db": settling_db.tolist(),
        "port_names": port_names,
        "port_families": list(result.port_families),
        "drive_plan": [list(d) for d in captured["drive_plan"]],
        "wire_mode": captured["wire_mode"],
        "preflight_text": preflight_text,
        "warnings": caught_warnings,
        "rows": rows,
        "verdict": verdict,
    }, indent=2) + "\n")
    print(f"\nwrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
