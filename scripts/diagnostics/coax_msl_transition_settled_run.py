#!/usr/bin/env python3
"""Settled run for the coax<->MSL transition lane (issue #489 leg 4, attempt 2).

WHY THIS SCRIPT EXISTS
-----------------------
``tests/test_coax_msl_transition.py::SETTLED_RUN_RECORD`` predeclares the
next, not-yet-run step for ``compute_coax_msl_transition``'s attempt-2
fixture (PR #585, merge da443f5): the committed 45000-step run only reached
-19.7/-17.9 dB ring-down settling, well short of this repo's -40 dB rule
(``CLAUDE.md``'s ring-down settling witness), so reciprocity/`|S22|`/max`|S|`
are UNMEASURED AT THIS SETTLING rather than pass/fail. The record's own
``target_n_steps_derivation`` extrapolates from the two measured local
checkpoints (20000 -> -12.3/-10.7 dB, 45000 -> -19.69/-17.94 dB) to
``target_n_steps = 135000`` as a starting estimate, not a guarantee.

This script is a STANDALONE driver (mirrors
``scripts/diagnostics/msl_ad_band_mean_s21_owner_measurement.py``'s own
"import the test module, reuse its fixture, run it outside pytest" pattern)
that builds the SAME ``_build_coax_msl_transition_sim_attempt2`` fixture and
calls ``compute_coax_msl_transition`` at ``--n-steps`` (default: the
record's own ``target_n_steps``), then prints and persists the same
witnesses ``test_coax_msl_transition_attempt2_instrument_verification``
computes: settling_db, cond_a_equilibrated, the gamma-vs-beta discriminant,
reciprocity, |S22|, max|S|, and the MSL-driven column power. It also
exercises the shared passivity guard (``_warn_if_nonpassive_smatrix``,
``strict=True``) directly on the result, the same faithful stand-in the
committed test uses instead of a second FDTD run.

Report-only: this script does not modify ``SETTLED_RUN_RECORD``. The
printed "SUMMARY (for filling SETTLED_RUN_RECORD)" block at the end is
meant to be hand-copied into that record by the PR that consumes this run's
output, per the fill-contract pattern
(``test_settled_run_record_is_committed_unrun_and_self_consistent``: UNRUN
<=> no numbers, no log path -- filling it is a separate, later edit, not
this script's job).

Runtime: unknown until measured -- the record's own derivation is a linear
extrapolation of a process that may not actually be linear (ring-down decay
is closer to exponential in time). At 45000 steps this fixture measured
~20 min wall-clock (two FDTD drives on a (142, 51, 56)-cell grid); 135000
steps is ~3x that fixture's own step count, so budget accordingly plus
margin -- the VESSL YAML wrapping this script sets a generous timeout, not
a promise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - provenance-only
        return f"<unavailable: {exc}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n-steps", type=int, default=None,
        help="default: SETTLED_RUN_RECORD['target_n_steps'] (135000)",
    )
    ap.add_argument(
        "--output", type=str, default=None,
        help="path to write the JSON result (default: alongside this script "
             "under .omx/, not committed)",
    )
    args = ap.parse_args()

    import rfx
    print(f"rfx        : {rfx.__file__}")
    print(f"git SHA    : {_git_sha()}")
    if not str(Path(rfx.__file__).resolve()).startswith(str(REPO)):
        print(f"FATAL: imported rfx is not this checkout ({REPO})")
        return 2

    import jax
    print(f"jax        : {jax.__version__}   devices: {jax.devices()}")

    import test_coax_msl_transition as t  # noqa: E402
    from rfx.api._sparams import _mixed_reciprocity_deviation, _warn_if_nonpassive_smatrix
    from rfx.core.yee import EPS_0, MU_0
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

    n_steps = args.n_steps if args.n_steps is not None else t.SETTLED_RUN_RECORD["target_n_steps"]
    print(f"fixture    : attempt2 (same junction as attempt 1, longer MSL ladder, "
          f"wider x-CPML clearance), n_steps={n_steps} "
          f"(record target: {t.SETTLED_RUN_RECORD['target_n_steps']})")

    sim = t._build_coax_msl_transition_sim_attempt2()
    t0 = time.perf_counter()
    result = sim.compute_coax_msl_transition(
        junction_x=t.JUNCTION_X, eps_r_sub=t.EPS_SUB, n_steps=n_steps,
        freqs=t.FREQS_2,
        probe_count=6, probe_start_cells=4, probe_spacing_cells=2,
        msl_probe_count=t.PROBE_COUNT_2, msl_probe_start_cells=t.PROBE_START_2,
        msl_probe_spacing_cells=t.PROBE_SPACING_2,
        skip_preflight=True, strict_passivity=False,
    )
    elapsed_s = time.perf_counter() - t0
    print(f"\nFDTD run complete in {elapsed_s:.1f}s ({elapsed_s / 60.0:.1f} min)")

    assert result.s_params.shape == (2, 2, len(t.FREQS_2))
    assert np.all(np.isfinite(result.s_params))
    assert result.port_names == ("coax", "msl")

    settling_db = np.asarray(result.settling_db).tolist()
    cond_a_equilibrated = np.asarray(result.cond_a_equilibrated).tolist()

    c0 = 1.0 / np.sqrt(MU_0 * EPS_0)
    _, eps_eff_hj = hammerstad_jensen_z0_eps_eff(t.W_TRACE, t.H_SUB, t.EPS_SUB)
    beta_analytic = 2.0 * np.pi * np.asarray(result.freqs) * np.sqrt(eps_eff_hj) / c0
    im_gamma_coax_driven = np.abs(result.gamma[1, 0, :].imag)
    gamma_ratio = (im_gamma_coax_driven / beta_analytic).tolist()

    S = result.s_params
    col_msl_driven_power = (np.abs(S[0, 1, :]) ** 2 + np.abs(S[1, 1, :]) ** 2).tolist()

    pair, worst_dev = _mixed_reciprocity_deviation(result.s_params)
    s22_abs = np.abs(result.s_params[1, 1, :]).tolist()
    max_abs_s = float(np.max(np.abs(result.s_params)))

    passivity_guard_raised = None
    passivity_guard_message = None
    try:
        _warn_if_nonpassive_smatrix(
            result, extractor="compute_coax_msl_transition", strict=True,
        )
        passivity_guard_raised = False
    except ValueError as exc:
        passivity_guard_raised = True
        passivity_guard_message = str(exc)

    settling_cleared_40db = bool(np.all(np.asarray(settling_db) < -40.0))

    print("\n=== measured witnesses ===")
    print(f"  settling_db (per drive)          : {settling_db}  "
          f"(-40 dB rule cleared: {settling_cleared_40db})")
    print(f"  cond_a_equilibrated               : {cond_a_equilibrated}")
    print(f"  gamma_ratio (coax-driven, vs HJ beta): {gamma_ratio}")
    print(f"  reciprocity worst deviation (pair={pair}): {worst_dev}")
    print(f"  |S22|                             : {s22_abs}")
    print(f"  max|S|                            : {max_abs_s}")
    print(f"  MSL-driven column power (|S01|^2+|S11|^2): {col_msl_driven_power}")
    print(f"  passivity guard (strict=True) raised: {passivity_guard_raised}")
    if passivity_guard_message:
        print(f"    message: {passivity_guard_message}")

    out = {
        "leg": t.SETTLED_RUN_RECORD["leg"],
        "n_steps": n_steps,
        "git_sha": _git_sha(),
        "elapsed_s": elapsed_s,
        "freqs_hz": np.asarray(result.freqs).tolist(),
        "settling_db": settling_db,
        "settling_cleared_40db": settling_cleared_40db,
        "cond_a_equilibrated": cond_a_equilibrated,
        "gamma_ratio_coax_driven": gamma_ratio,
        "reciprocity_worst_deviation": {"pair": list(pair) if pair is not None else None,
                                         "value": float(worst_dev)},
        "s22_abs": s22_abs,
        "max_abs_s": max_abs_s,
        "col_msl_driven_power": col_msl_driven_power,
        "passivity_guard_raised": passivity_guard_raised,
        "passivity_guard_message": passivity_guard_message,
    }

    out_path = Path(args.output) if args.output else (
        REPO / ".omx" / "coax-msl-transition-settled-run" /
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{_git_sha()[:12]}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nresult JSON written to: {out_path}")

    print("\n=== SUMMARY (for filling SETTLED_RUN_RECORD -- hand-copy, do not "
          "auto-apply) ===")
    print("  status              : RUN")
    print(f"  target_n_steps       : {t.SETTLED_RUN_RECORD['target_n_steps']}")
    print(f"  n_steps (this run)   : {n_steps}")
    print(f"  settling_db          : {settling_db}  "
          f"(-40 dB rule cleared: {settling_cleared_40db})")
    if not settling_cleared_40db:
        print("  NOTE: settling did NOT clear -40 dB at this n_steps -- "
              "reciprocity/|S22|/max|S| below are still not reportable as "
              "pass/fail per this repo's ring-down settling witness rule; "
              "treat this run as another calibration checkpoint, not the "
              "final answer, and re-derive target_n_steps if a further run "
              "is needed.")
    print(f"  reciprocity worst dev: {worst_dev}")
    print(f"  |S22|                : {s22_abs}")
    print(f"  max|S|               : {max_abs_s}")
    print(f"  col_msl_driven_power : {col_msl_driven_power}")
    print(f"  gamma_ratio          : {gamma_ratio}")
    print(f"  log_path             : {out_path}")
    print("  vessl_run_id         : <fill from the VESSL job that ran this>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
