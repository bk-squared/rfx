#!/usr/bin/env python3
"""Settled run for the coax<->MSL transition lane (issue #489 leg 4, attempt 2).

Tracked by issue #589 (#489 itself is closed and does not track this lane's
continuing work -- see SETTLED_RUN_RECORD["tracking_issue"]).

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
a promise. MEASURED since: the settled run (VESSL 369367252283) took
121.4 s on remilab-c0 gpu-rtx4090 at 135000 steps.

ISSUE #589 EXTENSION (2026-08-30) -- ADDITIVE, REPORT-ONLY
-----------------------------------------------------------
Every pre-existing JSON key and the printed "SUMMARY (for filling
SETTLED_RUN_RECORD ...)" block are kept byte-compatible with the tracked
``_coax_msl_transition_settled_run_logs/settled_run_369367252283_result.json``
so the old record stays comparable. The pre-existing keys ALWAYS refer to
the three COMMITTED bins (``FREQS_2`` = 6/8/10 GHz) even when ``--freqs``
adds a dense band; the new material lives under the ``ext_589`` key and in
new printed sections placed BEFORE the legacy SUMMARY block and in a
separate "#589 EXTENSION SUMMARY" block AFTER it.

What the settled record (369367252283) did NOT carry and this extension
dumps, per the #589 design + adversarial review (six REQUIRED changes; the
ones touching this driver are applied here):

* Per-bin COMPLEX S (re/im) for all four entries; |S00|, |S10|, |S01|, |S11|
  in dB. Port order is ALWAYS coax = 0, msl = 1 (``result.port_names``);
  ``S[j, i]`` = response at port j while driving port i, so S10 is
  coax-driven / received at MSL and S01 is MSL-driven / received at coax.
* Coax-driven column power |S00|^2 + |S10|^2 next to the existing
  col_msl_driven_power = |S01|^2 + |S11|^2 (A1; expected ~0.987 from the
  record's max|S| = 0.9933, which can only be |S00|).
* A2 passivity in PSD form: lambda_min of Q(f) = I - S^H S per bin.
  FALSIFIER: lambda_min < -0.02 at any bin => the extractor is non-passive
  => "EXTRACTOR" verdict regardless of column power (``check_passivity`` is
  one-sided, max column power only, and cannot see this). lambda_min >=
  -0.02 everywhere => passivity holds and trace(Q) is the total missing
  power. Report-only: the shared guard is NOT changed.
* Raw ``cond_a`` per bin NEXT TO ``cond_a_equilibrated`` (review item 6).
  The settled run log's third warning reads, verbatim:
  "solve_two_port_from_wave_amplitudes: the two drives are nearly linearly
  dependent at 3/3 frequencies (cond > 1000; worst 2.91e+07) -- usually
  both ports seeing essentially the same field, e.g. a symmetric structure
  driven identically, or one drive that failed to excite. S at those bins
  is degenerate. NOTE this threshold flags DEGENERACY only: cond below it
  is NOT a reliability certificate, because cond(A) also multiplies
  whatever noise is on the measured amplitudes (~1.3e-2 relative error in
  S from 1e-4 noise at cond=199, which never trips this warning)." That is
  raw cond_a ~2.9e7 vs cond_a_equilibrated 1.002-1.005 (JSON): a per-drive
  SCALE disparity that column equilibration removes (S = B @ inv(A) is
  exactly invariant to it). Dumping both side by side is so the
  reciprocity read cannot be re-attributed to it -- that attribution (the
  "drive-amplitude gap") is already RETRACTED on this lane.
* A3 single-mode validity of the extractor itself: ``fit_residual`` and
  ``recurrence_residual`` per (port array, drive, bin) for BOTH ladders --
  the MSL ladder (array 1) side by side with the coax ladder (array 0) of
  the same run. Threshold provenance (review item 7): 0.02 is the coax
  lane's own BORROWED convenience number (tests/test_coax_two_port_fdtd.py
  fit_residual gate, measured max 0.0127 there); a passing A3 on the MSL
  ladder means "not worse than the validated coax lane", NOT "single mode
  proven". The load-bearing criterion is the RELATIVE one: MSL-driven
  residual at 6 or 8 GHz > 10x its own 10 GHz value => the two-exponential
  quasi-TEM model does not describe the ladder field at those bins
  ("EXTRACTOR", (b)-on-ladder supported). Both criteria are printed per
  bin; the verdict string names which fired. The FALSIFIED branch is the
  plan's own: "(b)-on-ladder FALSIFIED" is printed ONLY when the MSL-
  ladder fit_residual AND recurrence_residual are <= 0.02 (and <= 10x own
  10 GHz) at ALL committed bins under BOTH drives; MSL-drive residuals
  clean while the coax-drive MSL-ladder residuals are not => NON-CLOSING
  ((b)-on-ladder NOT falsified), no attribution.
* A5 same-run floor witness for the sub -55 dB transmitted signals:
  |a_inc[0, 1, f] / b_out[0, 1, f]| (coax array, MSL drive -- the coax
  termination's own echo) and |a_inc[1, 0, f] / b_out[1, 0, f]| (MSL
  array, coax drive -- the MSL feed's own echo). PREDECLARED EXPECTATIONS,
  resolved from code (review item 3): the coax annular termination is sized
  to the ANALYTIC z_tem = 45.46 ohm, not the registered 50 ohm --
  rfx/api/_sparams.py, compute_coax_msl_transition: ``r_feed =
  float(feed_impedance) if feed_impedance is not None else float(z_tem)``
  and this driver passes no feed_impedance -- so the coax-side ratio is
  expected ~0 (+ discretization). The MSL feed is sized to the REGISTERED
  50 ohm (``_MSLPortLL(... impedance=msl_pe.impedance ...)``) against the
  Hammerstad-Jensen 53.11 ohm the power waves are normalized to, so the
  MSL-side ratio is expected |Gamma_feed| = 3.11/103.11 = 0.030.
  FALSIFIER: both ratios <= 0.15 at all bins and within +/-50% bin-to-bin
  => the -55..-76 dB transmitted signals are RESOLVED (genuine TEM pair
  with the expected termination echo) and the 91.4% reciprocity deviation
  stays a claims-bearing number; ratio > 0.3 or erratic => the non-driven
  port's signal is extractor FLOOR and the reciprocity item closes as
  "transmission <= X dB both ways (upper bounds); deviation not resolvable
  above the measured floor". Between => NON-CLOSING (no prose attribution).
  WARNING-TEXT DEFECT recorded, not fixed here: the method's registered-
  impedance advisory (rfx/api/_sparams.py, "diverges ... from the analytic
  ... Z0 ... this method actually uses for the power-wave normalization
  (z0_ref) and for sizing the feed resistor / termination") says the
  registered impedance sizes the coax feed resistor; for the coax side of
  THIS method it does not (see r_feed above). Its verbatim copy is in the
  tracked run log.
* PRECISION label (review item 2): ``jax.config.jax_enable_x64`` at solve
  time, the ``JAX_ENABLE_X64`` environment variable, and the ACTUAL dtype
  of ``result.s_params`` / ``a_inc``. The 369367252283 record is a
  default-float32 GPU run (no x64 pin in driver or test); A0 reproduction
  is only meaningful at the record's precision, and an f64 replicate
  (``JAX_ENABLE_X64=1``, same fixture/steps) is the cheapest extractor-
  floor discriminator: |S10| or |S01| moving by more than 2x between f32
  and f64 at any bin => FLOOR (precision/extractor), while |S22|, |S00|
  and col_power must reproduce within the A0 budget.
* settling_db per drive (already on record) and the ``_warn_if_ringdown_
  truncated`` (#662) warning text verbatim if it fires.
* Every UserWarning raised during the solve is captured and dumped verbatim
  (``solve_warnings``) in addition to being printed.

A0 REPRODUCTION (review item 4) -- ``--baseline <json>``:
  The GPU run-to-run reproduction budget on this lane is UNMEASURED (no
  GPU repeat exists on any coax lane); the first Step-A run IS the
  measurement of it. Two-tier PREDECLARATION on max |S - S_baseline| over
  the committed bins (or, against the tracked 369367252283 JSON, which
  carries only derived scalars, on max |delta| over |S22|, max|S| and
  col_msl_driven_power -- the complex S, |S00|, |S10|, |S01|, residuals,
  a/b ratios, raw cond_a and lambda_min CANNOT be compared against that
  record and are said so):
    <= 1e-3  "reproduced";
    >  1e-2  "STOP" (the fixture or extractor is not what the record says);
    between  "GPU reproduction spread" -- reported, never widened silently,
             and NO new record field may be pinned tighter than the
             measured spread.
  A comparison across different n_steps, a different fixture OR a
  different precision (this run's jax_enable_x64 vs the baseline's
  ext_589.precision.jax_enable_x64; the tracked record has no precision
  key and is documented f32) is printed but labeled NOT COMPARABLE: the
  reproduction TIER is not applied and the deltas are informational.
  Across a precision mismatch the driver instead COMPUTES the predeclared
  f64-replicate rule (review item 2): |S10| or |S01| moving > 2x between
  f32 and f64 at any committed bin => FLOOR (precision/extractor); |S22|,
  |S00| and col_power must reproduce within the A0 budget (two-tier). The
  2x part needs a baseline carrying ext_589.s_abs (a driver JSON, e.g. the
  Step-A f32 remeasure); against the tracked record only |S22| and
  col_msl_driven_power enter the budget check and the 2x part is declared
  not computable. The predeclared submission order (attempt2/X64=0 first,
  then attempt2/X64=1 against the Step-A f32 JSON) follows from this.

``--preflight`` (review item 1): ``compute_coax_msl_transition`` ACCEPTS
``skip_preflight`` (rfx/api/_sparams.py:6331, ``skip_preflight: bool =
False,`` in the signature) but NEVER USES IT -- the name does not appear
in the method body; compare ``compute_mixed_s_matrix``, which does ``if
not skip_preflight: self.preflight()``. So the settled run had no
preflight output because the METHOD has no preflight path, not because the
driver passed ``skip_preflight=True``. Production is NOT changed here; the
driver instead calls ``sim.preflight()`` and ``sim.fidelity_report()``
itself BEFORE the solve and prints both VERBATIM into the log and into the
JSON (lists of strings). The flag is still passed for kwargs parity with
the test module.

``--fixture attempt2_wide`` (Step B, falsifier variant, NOT attempt 3): the
wide fixture ``_build_coax_msl_transition_sim_attempt2_wide`` (LY 6.8 mm,
junction 3.0 mm from the -x CPML inner edge, LX 14.5 mm, feed 13.0 mm;
junction cells asserted byte-identical to attempt 2 by
``test_attempt2_wide_junction_cells_are_byte_identical_to_attempt2``) tests
the named candidate (c) absorber proximity. Step A is the settled run
re-taken with this fuller dump; B asks whether A's numbers MOVE
(col_power / |S22| shift > 0.10 at 6 or 8 GHz => absorber-proximity-
limited, a FIXTURE finding; invariant within 0.05 (|S00| 0.02) => (c)
falsified). If B's settling_db > -40 dB at the same n_steps its numbers
are truncated and UNPINNABLE -- report, name a 2x rerun, do not pin.

``--fixture attempt3b`` (issue #823): attempt 3's fixture UNCHANGED -- the
builder ``_build_coax_msl_transition_sim_attempt3b`` returns attempt 3's own
Simulation, and eps_r/sigma/mu_r/pec_mask are asserted ``np.array_equal``
over the whole domain by ``test_attempt3b_geometry_is_attempt3s_and_only_
the_msl_probe_kwargs_differ`` -- with the COMPLIANT MSL probe ladder
``_attempt3b_kwargs`` (msl_probe_count 9 -> 8, msl_probe_start_cells 4 -> 15,
msl_probe_spacing_cells 10 unchanged; realized probes at x = 2.5 ... 9.5 mm).
Every probe then stands at least ``msl_source_near_field_standoff_cells``
= 15 cells = 1.5 mm = 5*h_sub from BOTH the port feed plane (x = 11.0 mm) and
the reference plane the ladder is referred to (the junction, x = 1.0 mm),
where attempt 3's own nearest probe stands 0.4 mm = 1.33*h_sub off the feed.
3b vs 3 is therefore a PURE LADDER-RECIPE comparison on bit-identical
geometry; its predictions and falsifiers are
``tests/test_coax_msl_transition.py::PREDECLARATION_ATTEMPT3B`` (status
UNRUN), a NEW predeclaration that does not touch, relax or reinterpret
PREDECLARATION_ATTEMPT2/3 or SETTLED_RUN_RECORD.

``--freqs`` adds an OPTIONAL dense band (comma list or start:stop:n, GHz);
the three committed bins are ALWAYS retained and reported separately (the
legacy keys), and ``ext_589`` carries the full band.

``--dump-ladders`` / ``--flux`` (#589 witness half; BOTH DEFAULT OFF, so
every pre-existing key and the A0 comparison are byte-unchanged when they
are not passed):

* ``--dump-ladders`` asks the method for the RAW per-probe modal voltages
  of both ladders (``return_ladder_voltages=True``). That keyword is the
  production half of #589 and may not be merged yet, so the driver detects
  it with ``inspect.signature`` and, when it is absent, says so and runs
  the solve UNCHANGED (W1-W3 are then reported as SKIPPED; W4 and the
  label-swap counterfactual still work). The arrays are written to
  ``<output>.ladders.npz`` next to the JSON, never into the legacy keys.
* ``--flux`` passes ``extra_flux_monitors=tests/test_coax_msl_transition.py
  ::_attempt3_scratch_flux_entries()`` -- the six faces of one lossless
  control volume around the junction (plus the full-plane +x comparator).
  attempt3 / attempt3b only: the plane coordinates are that grid's, and
  attempt3b IS attempt 3's fixture (only the msl_probe_* kwargs differ).
  Non-perturbation of S is witnessed by
  ``test_extra_flux_monitors_do_not_perturb_s``.

Both feed ``scripts/diagnostics/coax_msl_ladder_witnesses.py`` (pure NumPy,
no FDTD), whose W1-W4 tables are printed here and whose full dict is written
to ``<output>.witnesses.json``. Everything they produce is REPORT-ONLY and
LABEL-INDEPENDENT by construction; the printed 'label-swap counterfactual'
is a PREDICTION of H1 (algebraically inv(S_code)), not a measurement, and is
never written into a legacy key.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling witness module

import numpy as np  # noqa: E402

# Predeclared thresholds (issue #589 design + review; report-only, no gate).
A0_REPRODUCED_MAX = 1.0e-3
A0_STOP_MIN = 1.0e-2
A2_LAMBDA_MIN_FALSIFIER = -0.02
A3_BORROWED_ABS = 0.02          # coax lane's borrowed convenience number
A3_RELATIVE_FACTOR = 10.0       # load-bearing: MSL residual at 6/8 GHz vs own 10 GHz
A5_RESOLVED_MAX = 0.15
A5_FLOOR_MIN = 0.30
A5_SMOOTH_REL = 0.50
A5_EXPECT_COAX_SIDE = 0.0       # termination sized to analytic z_tem (45.46 ohm)
A5_EXPECT_MSL_SIDE = 3.11 / 103.11  # registered 50 ohm vs HJ 53.11 ohm
SKIP_PREFLIGHT_INERT_NOTE = (
    "compute_coax_msl_transition accepts skip_preflight (rfx/api/_sparams.py:6331, "
    "'skip_preflight: bool = False,') but never references it in the method body; "
    "the method has NO preflight path. --preflight therefore runs sim.preflight() and "
    "sim.fidelity_report() from this driver, before the solve. Production unchanged."
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - provenance-only
        return f"<unavailable: {exc}>"


def _parse_freqs_ghz(spec: str) -> np.ndarray:
    """'6,7,8' or '5:11:13' (start:stop:n, inclusive) in GHz -> Hz array."""
    spec = spec.strip()
    if ":" in spec:
        start, stop, n = spec.split(":")
        vals = np.linspace(float(start), float(stop), int(n))
    else:
        vals = np.array([float(v) for v in spec.split(",") if v.strip()])
    return np.asarray(vals, dtype=float) * 1e9


def _db(x):
    return (20.0 * np.log10(np.maximum(np.abs(np.asarray(x)), np.finfo(float).tiny))).tolist()


def _classify_a2(lam_min):
    lam = np.asarray(lam_min)
    if np.any(lam < A2_LAMBDA_MIN_FALSIFIER):
        return "EXTRACTOR (lambda_min(I - S^H S) < -0.02 at >=1 bin: non-passive S)"
    return "PASSIVE (lambda_min >= -0.02 at all bins; trace(Q) is the missing power)"


def _a3_fires(r, f, committed, label):
    """Per-bin A3 criteria for one residual vector r[f] (MSL ladder, one drive)."""
    r = np.asarray(r)
    idx10 = [i for i in committed if np.isclose(f[i], 10.0e9)]
    fired = []
    for i in committed:
        if r[i] > A3_BORROWED_ABS:
            fired.append(f"{f[i] / 1e9:.3g} GHz: {label} {r[i]:.4g} > 0.02 (borrowed)")
        if idx10 and not np.isclose(f[i], 10.0e9) and r[i] > A3_RELATIVE_FACTOR * r[idx10[0]]:
            fired.append(
                f"{f[i] / 1e9:.3g} GHz: {label} {r[i]:.4g} > 10x own 10 GHz "
                f"value {r[idx10[0]]:.4g} (load-bearing)"
            )
    return fired


def _classify_a3(fit, rec, freqs, committed):
    """A3 verdict from the MSL-ladder (array 1) residuals, committed bins only.

    ``fit``/``rec`` are [port array, drive, f]. Predeclared (#589 design A3):

    * MSL-DRIVEN MSL-ladder residual (fit OR recurrence) > 0.02 or > 10x its
      own 10 GHz value at a committed bin => EXTRACTOR ((b)-on-ladder
      supported).
    * (b)-on-ladder is FALSIFIED only when fit AND recurrence residuals of
      the MSL ladder are <= 0.02 (and <= 10x own 10 GHz) at ALL committed
      bins under BOTH drives ("residuals <= 0.02 at all bins/both drives").
    * Otherwise (MSL drive clean, coax drive not) => NON-CLOSING: (b)-on-
      ladder NOT falsified, no attribution.

    The coax-ladder residuals (array 0) are the comparator and never enter
    the verdict.
    """
    f = np.asarray(freqs)
    fit = np.asarray(fit)
    rec = np.asarray(rec)
    msl_drive = (_a3_fires(fit[1, 1, :], f, committed, "fit_residual")
                 + _a3_fires(rec[1, 1, :], f, committed, "recurrence_residual"))
    coax_drive = (_a3_fires(fit[1, 0, :], f, committed, "fit_residual")
                  + _a3_fires(rec[1, 0, :], f, committed, "recurrence_residual"))
    if msl_drive:
        return ("EXTRACTOR ((b)-on-ladder supported): MSL ladder, MSL drive: "
                + "; ".join(msl_drive))
    if coax_drive:
        return ("NON-CLOSING ((b)-on-ladder NOT falsified): MSL ladder under the MSL drive "
                "is <= 0.02 and <= 10x own 10 GHz at all committed bins, but under the coax "
                "drive: " + "; ".join(coax_drive)
                + " -- the predeclared falsifier requires residuals <= 0.02 at all bins/BOTH "
                  "drives; no attribution")
    return ("(b)-ON-LADDER FALSIFIED: MSL-ladder fit_residual AND recurrence_residual "
            "<= 0.02 and <= 10x own 10 GHz value at all committed bins under BOTH drives "
            "-- not worse than the validated coax lane; single-mode NOT thereby proven")


def _classify_a5(ratio, committed):
    r = np.asarray(ratio)[list(committed)]
    if not np.all(np.isfinite(r)):
        return "NON-CLOSING (non-finite ratio)"
    erratic = False
    for a, b in zip(r[:-1], r[1:]):
        denom = max(abs(a), abs(b), 1e-300)
        if abs(a - b) / denom > A5_SMOOTH_REL:
            erratic = True
    if np.max(r) > A5_FLOOR_MIN or erratic:
        return (f"FLOOR (max ratio {np.max(r):.4g} > 0.3 or erratic bin-to-bin, committed "
                "bins): the non-driven port's signal is extractor floor")
    if np.max(r) <= A5_RESOLVED_MAX and not erratic:
        return (f"RESOLVED (max ratio {np.max(r):.4g} <= 0.15, smooth, committed bins): "
                "transmitted signals are a genuine TEM pair with the expected termination echo")
    return f"NON-CLOSING (max ratio {np.max(r):.4g} between 0.15 and 0.3, committed bins)"


def _baseline_precision(base):
    """(jax_enable_x64, source) of a baseline JSON. A record without the
    ext_589.precision key is the tracked 369367252283-class record, which is
    documented as a default-float32 GPU run (no x64 pin in driver or test)."""
    prec = (base.get("ext_589") or {}).get("precision") or {}
    if "jax_enable_x64" in prec:
        return bool(prec["jax_enable_x64"]), "ext_589.precision.jax_enable_x64"
    return False, ("no precision key: tracked 369367252283-class record, documented as a "
                   "default-float32 GPU run (no x64 pin)")


def _match_bins(f_this, f_axis):
    """Index of each f_this[k] in f_axis (None when absent)."""
    f_axis = np.asarray(f_axis, dtype=float)
    out = []
    for f in f_this:
        j = np.where(np.isclose(f_axis, f))[0]
        out.append(int(j[0]) if len(j) else None)
    return out


def _f64_replicate_rule(*, base, ext, S, freqs_all, committed, ii, jj_leg, jj_ext,
                        x64_this, x64_base, same_fixture_and_steps):
    """Review required change 2, computed: between an f32 and an f64 run of
    the same fixture/steps, |S10| or |S01| moving by more than 2x at any
    committed bin => FLOOR (precision/extractor); |S22|, |S00| and col_power
    must reproduce within the A0 budget (two-tier). Ratios are max/min, so
    the rule is symmetric in which side is f64."""
    out = {
        "rule": "|S10| or |S01| moving > 2x between f32 and f64 at any committed bin => "
                "FLOOR (precision/extractor); |S22|, |S00|, col_power must reproduce within "
                "the A0 budget (<= 1e-3 reproduced; <= 1e-2 GPU spread; > 1e-2 exceeds)",
        "applicable": True,
        "f64_side": "this run" if x64_this else "baseline",
        "f32_side": "baseline" if x64_this else "this run",
        "same_fixture_and_steps": bool(same_fixture_and_steps),
        "freqs_hz": [float(freqs_all[i]) for i in ii],
        "abs_S10_this": None, "abs_S10_baseline": None, "ratio_S10": None,
        "abs_S01_this": None, "abs_S01_baseline": None, "ratio_S01": None,
        "s10_s01_max_ratio": None,
        "magnitude_deltas": {},
        "magnitude_max_delta": None,
        "not_computable": [],
        "verdict": None,
    }
    tiny = np.finfo(float).tiny
    S_this = np.asarray(S)
    ratio_parts = []
    s_abs = ext.get("s_abs") if isinstance(ext, dict) else None
    if s_abs and jj_ext is not None and all(k in s_abs for k in ("S10", "S01", "S00")):
        for name, (j, i) in (("S10", (1, 0)), ("S01", (0, 1))):
            a = np.abs(S_this[j, i, ii])
            b = np.asarray(s_abs[name], dtype=float)[jj_ext]
            ratio = np.maximum(a, b) / np.maximum(np.minimum(a, b), tiny)
            out[f"abs_{name}_this"] = a.tolist()
            out[f"abs_{name}_baseline"] = b.tolist()
            out[f"ratio_{name}"] = ratio.tolist()
            for k, r in enumerate(ratio):
                if not np.isfinite(r) or r > 2.0:
                    ratio_parts.append(f"{freqs_all[ii[k]] / 1e9:.3g} GHz |{name}| ratio {r:.3g}")
        out["s10_s01_max_ratio"] = float(np.max(np.concatenate(
            [out["ratio_S10"], out["ratio_S01"]])))
    else:
        out["not_computable"].append(
            "|S10|/|S01| 2x rule: baseline carries no ext_589.s_abs (tracked record); "
            "compare against the Step-A f32 remeasure JSON instead")
    # Magnitudes that must reproduce within the A0 budget.
    mags = {}
    if "s22_abs" in base:
        mags["S22"] = np.abs(np.abs(S_this[1, 1, ii])
                             - np.asarray(base["s22_abs"], dtype=float)[jj_leg])
    else:
        out["not_computable"].append("|S22| (baseline has no s22_abs)")
    if s_abs and jj_ext is not None and "S00" in s_abs:
        mags["S00"] = np.abs(np.abs(S_this[0, 0, ii])
                             - np.asarray(s_abs["S00"], dtype=float)[jj_ext])
    else:
        out["not_computable"].append("|S00| (baseline has no ext_589.s_abs.S00)")
    if "col_msl_driven_power" in base:
        col_this = np.abs(S_this[0, 1, ii]) ** 2 + np.abs(S_this[1, 1, ii]) ** 2
        mags["col_msl_driven_power"] = np.abs(
            col_this - np.asarray(base["col_msl_driven_power"], dtype=float)[jj_leg])
    else:
        out["not_computable"].append("col_msl_driven_power (absent from baseline)")
    if isinstance(ext, dict) and "col_coax_driven_power" in ext and jj_ext is not None:
        col_this = np.abs(S_this[0, 0, ii]) ** 2 + np.abs(S_this[1, 0, ii]) ** 2
        mags["col_coax_driven_power"] = np.abs(
            col_this - np.asarray(ext["col_coax_driven_power"], dtype=float)[jj_ext])
    else:
        out["not_computable"].append("col_coax_driven_power (baseline has no ext_589)")
    out["magnitude_deltas"] = {k: v.tolist() for k, v in mags.items()}
    mag_max = float(max(float(np.max(v)) for v in mags.values())) if mags else None
    out["magnitude_max_delta"] = mag_max

    if not same_fixture_and_steps:
        out["verdict"] = ("NOT COMPARABLE for the f64 rule (predeclared for the SAME fixture "
                          "and n_steps); ratios/deltas above are informational only")
        return out
    if out["ratio_S10"] is None:
        part1 = out["not_computable"][0]
        part1 = part1.replace("|S10|/|S01| 2x rule:", "|S10|/|S01| 2x rule NOT computable:")
    elif ratio_parts:
        part1 = ("FLOOR (precision/extractor): |S10| or |S01| moved > 2x between f32 and f64 "
                 "at " + ", ".join(ratio_parts)
                 + f" (max ratio {out['s10_s01_max_ratio']:.3g})")
    else:
        part1 = (f"|S10|/|S01| within 2x between f32 and f64 at all committed bins (max ratio "
                 f"{out['s10_s01_max_ratio']:.3g}): NOT floor by this rule")
    names = "/".join(f"|{k}|" if k.startswith("S") else k for k in mags) or "<none>"
    if mag_max is None:
        part2 = "no magnitude (|S22|/|S00|/col_power) is comparable against this baseline"
    elif mag_max <= A0_REPRODUCED_MAX:
        part2 = f"{names} reproduce within the A0 reproduced tier (max delta {mag_max:.3g} <= 1e-3)"
    elif mag_max <= A0_STOP_MIN:
        part2 = (f"{names} reproduce within the A0 GPU-spread tier (max delta {mag_max:.3g}, "
                 "1e-3 < delta <= 1e-2)")
    else:
        part2 = (f"{names} do NOT reproduce within the A0 budget (max delta {mag_max:.3g} > "
                 "1e-2): the predeclared rule requires them to; report, do not pin, no "
                 "attribution")
    out["verdict"] = part1 + " ; " + part2
    return out


def _a0_compare(baseline_path, *, freqs_all, committed, S, n_steps, fixture,
                legacy, x64_enabled):
    """Two-tier reproduction comparison against a baseline JSON.

    Compares what the baseline carries: full complex S when it has
    ext_589.s_complex, otherwise the derived scalars of the tracked
    369367252283 record. The reproduction TIER is applied only when this run
    and the baseline share n_steps, fixture AND precision (jax_enable_x64);
    otherwise the comparison is labeled NOT COMPARABLE and the deltas are
    informational. Across a precision mismatch the predeclared f64-replicate
    rule (review required change 2) is computed instead. Returns a JSON-able
    dict (never raises on a missing/odd baseline -- the comparison is
    reported, not enforced).
    """
    out = {"baseline_path": str(baseline_path), "status": None, "tier": None,
           "compared": {}, "not_comparable": [], "notes": [], "precision": None,
           "f64_replicate": None}
    try:
        base = json.loads(Path(baseline_path).read_text())
    except Exception as exc:
        out["status"] = f"baseline unreadable: {exc}"
        return out
    f_this = np.asarray(freqs_all)[list(committed)]
    f_base = np.asarray(base.get("freqs_hz", []), dtype=float)
    if len(f_base) == 0:
        out["status"] = "baseline has no freqs_hz"
        return out
    ext = base.get("ext_589") or {}
    # Legacy keys are committed-only (freqs_hz); ext_589 arrays are all-bins
    # (freqs_hz_all). Match each by its own axis.
    match_leg = _match_bins(f_this, f_base)
    f_ext = ext.get("freqs_hz_all") if isinstance(ext, dict) else None
    match_ext = _match_bins(f_this, f_ext) if f_ext else list(match_leg)
    unmatched = [float(f) for f, j in zip(f_this, match_leg) if j is None]
    if unmatched:
        out["notes"].append(f"committed bins absent from baseline: {unmatched}")
    pairs = [(committed[k], jl, match_ext[k]) for k, jl in enumerate(match_leg)
             if jl is not None and match_ext[k] is not None]
    if not pairs:
        out["status"] = "no shared bins"
        return out
    ii = [i for i, _, _ in pairs]
    jj = [j for _, j, _ in pairs]
    jj_ext = [j for _, _, j in pairs]

    x64_base, x64_source = _baseline_precision(base)
    out["precision"] = {
        "this_run_jax_enable_x64": bool(x64_enabled),
        "baseline_jax_enable_x64": bool(x64_base),
        "baseline_source": x64_source,
    }
    reasons = []
    if base.get("n_steps") != n_steps:
        reasons.append(f"n_steps this run {n_steps} vs baseline {base.get('n_steps')}")
    if base.get("fixture", "attempt2") != fixture:
        reasons.append(f"fixture this run {fixture} vs baseline {base.get('fixture', 'attempt2')}")
    if bool(x64_enabled) != bool(x64_base):
        reasons.append(f"precision this run jax_enable_x64={bool(x64_enabled)} vs baseline "
                       f"jax_enable_x64={bool(x64_base)} ({x64_source}); the A0 tier is only "
                       "meaningful at the record's own precision")
    for r in reasons:
        out["notes"].append("NOT COMPARABLE as a reproduction test: " + r)
    comparable = not reasons

    deltas = {}
    if "s_complex" in ext:
        sb = np.asarray(ext["s_complex"]["re"]) + 1j * np.asarray(ext["s_complex"]["im"])
        d = np.abs(S[:, :, ii] - sb[:, :, jj_ext])
        deltas["max_abs_delta_S_complex"] = float(np.max(d))
        out["compared"]["complex_S_per_bin_max_abs_delta"] = np.max(d, axis=(0, 1)).tolist()
        tier_value = float(np.max(d))
    else:
        out["not_comparable"] += [
            "complex S (re/im)", "|S00|", "|S10|", "|S01| (only derivable as "
            "col_power - |S22|^2)", "fit/recurrence residuals", "a_inc/b_out ratios",
            "raw cond_a", "lambda_min(I - S^H S)", "preflight output",
        ]
        tier_terms = []
        for key in ("s22_abs", "col_msl_driven_power"):
            if key in base:
                bv = np.asarray(base[key], dtype=float)[jj]
                tv = np.asarray(legacy[key], dtype=float)
                d = np.abs(tv - bv)
                deltas[key] = d.tolist()
                tier_terms.append(float(np.max(d)))
        if "max_abs_s" in base:
            d = abs(float(legacy["max_abs_s"]) - float(base["max_abs_s"]))
            deltas["max_abs_s"] = d
            tier_terms.append(d)
        tier_value = max(tier_terms) if tier_terms else None
    # Informational deltas (not part of the tier).
    for key in ("gamma_ratio_coax_driven", "cond_a_equilibrated"):
        if key in base:
            bv = np.asarray(base[key], dtype=float)[jj]
            tv = np.asarray(legacy[key], dtype=float)
            deltas[key + "_info"] = np.abs(tv - bv).tolist()
    if "settling_db" in base:
        deltas["settling_db_info"] = (
            np.abs(np.asarray(legacy["settling_db"]) - np.asarray(base["settling_db"])).tolist()
        )
    if "reciprocity_worst_deviation" in base:
        deltas["reciprocity_worst_deviation_info"] = abs(
            float(legacy["reciprocity_worst_deviation"]["value"])
            - float(base["reciprocity_worst_deviation"]["value"])
        )
    out["deltas"] = deltas
    out["tier_value"] = tier_value
    if not comparable:
        out["tier"] = ("NOT COMPARABLE (" + "; ".join(r.split(";")[0] for r in reasons)
                       + "): reproduction tier NOT applied; tier_value/deltas are informational")
    elif tier_value is None:
        out["tier"] = "UNDETERMINED (baseline carries none of |S22|/col_power/max|S|/complex S)"
    elif tier_value <= A0_REPRODUCED_MAX:
        out["tier"] = "reproduced (<= 1e-3)"
    elif tier_value > A0_STOP_MIN:
        out["tier"] = "STOP (> 1e-2: fixture or extractor is not what the record says)"
    else:
        out["tier"] = ("GPU reproduction spread (1e-3 < delta <= 1e-2): report; do not pin "
                       "any new record field tighter than this spread")
    out["budget_status"] = "UNMEASURED before this run (no GPU repeat exists on any coax lane)"
    out["status"] = "compared" if comparable else "compared but NOT a reproduction test"

    if bool(x64_enabled) != bool(x64_base):
        same = (base.get("n_steps") == n_steps
                and base.get("fixture", "attempt2") == fixture)
        out["f64_replicate"] = _f64_replicate_rule(
            base=base, ext=ext, S=S, freqs_all=freqs_all, committed=committed, ii=ii,
            jj_leg=jj, jj_ext=jj_ext if ("s_abs" in ext or "col_coax_driven_power" in ext) else None,
            x64_this=bool(x64_enabled), x64_base=bool(x64_base), same_fixture_and_steps=same,
        )
    return out


def _witness_dump(*, result, ext, out_path, freqs_all, beta_coax_analytic,
                  beta_msl_analytic, ladders_requested, ladders_available,
                  flux_requested):
    """Write ``<output>.ladders.npz`` + ``<output>.witnesses.json`` and print
    the W1-W4 tables (issue #589 witness half; REPORT-ONLY).

    Mutates ``ext`` by adding ``ladders_npz``, ``witnesses_json`` and
    ``w4_flux`` -- and ONLY when this function runs, i.e. only when at least
    one of ``--dump-ladders`` / ``--flux`` was passed. No legacy key and no
    pre-existing ``ext_589`` key is touched.

    CALL ORDER IS LOAD-BEARING: the caller writes the result JSON BEFORE
    calling this, and calls it inside try/except. Everything here is
    report-only, and it runs at the end of a multi-hour GPU solve whose result
    JSON is the irreplaceable artifact -- an exception raised in here (a
    failed ``np.savez`` on NFS, an all-zero subset raising inside the pencil,
    ``np.linalg.lstsq`` not converging) must never be able to destroy that
    measurement. Do not move this call above the write.
    """
    import coax_msl_ladder_witnesses as W

    a_inc = np.asarray(result.a_inc)
    b_out = np.asarray(result.b_out)
    payload = {
        "freqs": np.asarray(freqs_all, dtype=float),
        "a_inc": a_inc.astype(np.complex128),
        "b_out": b_out.astype(np.complex128),
        "gamma": np.asarray(result.gamma).astype(np.complex128),
        "beta_coax_analytic": np.asarray(beta_coax_analytic, dtype=float),
        "beta_msl_analytic": np.asarray(beta_msl_analytic, dtype=float),
        # Predeclared precondition input for the W1/W4 H1 verdicts: a truncated
        # run is not in the steady state the witnesses assume, so the witness
        # module refuses to emit a hypothesis verdict without it (and treats an
        # absent settling_db as a FAILED precondition, not as a pass).
        "settling_db": np.asarray(result.settling_db, dtype=float),
    }
    d = {k: v for k, v in payload.items()}
    d["coax_ladder_v"] = None
    d["msl_ladder_v"] = None

    lv = getattr(result, "ladder_voltages", None)
    ladder_keys_missing = []
    if lv:
        wanted = ("coax_ladder_v", "coax_ladder_z_m", "coax_ladder_k", "msl_ladder_v",
                  "msl_ladder_x_m", "msl_ladder_i", "ref_coax_m", "ref_msl_m",
                  "z0_ref", "drive_order")
        for key in wanted:
            if key not in lv:
                ladder_keys_missing.append(key)
                continue
            val = lv[key]
            arr = np.asarray(val)
            if arr.dtype.kind in "USO":
                arr = np.asarray([str(x) for x in np.atleast_1d(val)])
            payload[key] = arr
            d[key] = arr if arr.ndim else arr.item()
        if ladder_keys_missing:
            print(f"  NOTE: result.ladder_voltages is missing keys {ladder_keys_missing} "
                  f"(present: {sorted(lv)}) -- dumped what is there")

    flux = getattr(result, "flux_monitors", None) or {}
    flux_by_drive = {}
    for drive_key, spectra in flux.items():
        for name, arr in spectra.items():
            a = np.asarray(arr, dtype=float)
            payload[f"flux__{drive_key}__{name}"] = a
            flux_by_drive.setdefault(drive_key, {})[name] = a
    d["flux_by_drive"] = flux_by_drive or None

    npz_path = out_path.with_suffix(".ladders.npz")
    np.savez(npz_path, **payload)
    print("\n=== #589 witness dump ===")
    print(f"  ladders requested      : {ladders_requested}  "
          f"(method keyword available: {ladders_available})")
    print(f"  ladder arrays in dump  : "
          f"{sorted(k for k in payload if k.startswith(('coax_ladder', 'msl_ladder')))}")
    print(f"  flux requested         : {flux_requested}  "
          f"drives: {sorted(flux_by_drive)}  faces: "
          f"{sorted(next(iter(flux_by_drive.values()))) if flux_by_drive else []}")
    print(f"  npz written to         : {npz_path}")

    witnesses = W.compute_witnesses(d)
    for line in W.format_tables(witnesses):
        print(line)

    wit_path = out_path.with_suffix(".witnesses.json")
    wit_path.write_text(json.dumps(W._jsonable(witnesses), indent=2))
    print(f"  witness JSON written to: {wit_path}")

    ext["ladders_npz"] = str(npz_path)
    ext["witnesses_json"] = str(wit_path)
    ext["w4_flux"] = W._jsonable(witnesses.get("W4_flux"))
    ext["w4_flux_rules"] = W._jsonable(witnesses.get("W4_rules"))
    ext["witness_note"] = (
        "W1-W4 + the label-swap counterfactual are REPORT-ONLY and computed by "
        "scripts/diagnostics/coax_msl_ladder_witnesses.py from the dumped raw "
        "ladders/flux; the counterfactual is a PREDICTION of H1 (= inv(S_code)), "
        "not a measurement, and no legacy key is derived from any of it."
    )
    return witnesses


# ---------------------------------------------------------------------------
# Fixture dispatch (issue #823 attempt 3b).
#
# This used to be an if/elif chain inside main() whose ``else`` branch was
# attempt2_wide, so a new ``--fixture`` choice added to the argparse list
# without touching the chain would silently have run attempt2_wide's builder
# and kwargs under the new label. That is exactly the class of defect this
# lane exists to fix (#822: a mapping copied from a sibling whose geometry
# made it correct there), so the dispatch is now a TOTAL function over
# FIXTURE_CHOICES that raises on anything else, and
# tests/test_coax_msl_transition.py::
# test_settled_run_driver_fixture_selection_is_explicit_for_every_label pins
# each label to its own builder and kwargs by identity.
# ---------------------------------------------------------------------------
FIXTURE_CHOICES = ("attempt2", "attempt2_wide", "attempt3", "attempt3b")

FluxFixtureSelection = collections.namedtuple(
    "FluxFixtureSelection", ("build", "kwargs", "fixture_geom", "banner"))


def _select_fixture(t, fixture: str, n_steps: int) -> "FluxFixtureSelection":
    """(build, kwargs, fixture_geom, banner) for one ``--fixture`` label.

    ``t`` is the imported ``tests/test_coax_msl_transition.py`` module: the
    fixtures, their kwargs and their geometry constants live there and are
    reused verbatim, never re-declared here.
    """
    target = t.SETTLED_RUN_RECORD["target_n_steps"]
    geom_2 = {"LX_m": t.LX_2, "LY_m": t.LY, "LZ_m": t.LZ_2,
              "junction_x_m": t.JUNCTION_X, "feed_x_m": t.FEED_X_2, "y_c_m": t.Y_C}
    if fixture == "attempt2":
        return FluxFixtureSelection(
            t._build_coax_msl_transition_sim_attempt2,
            t._attempt2_kwargs(n_steps),
            dict(geom_2),
            f"attempt2 (same junction as attempt 1, longer MSL ladder, wider "
            f"x-CPML clearance), n_steps={n_steps} (record target: {target})",
        )
    if fixture == "attempt3":
        return FluxFixtureSelection(
            t._build_coax_msl_transition_sim_attempt3,
            t._attempt2_kwargs(n_steps),
            dict(geom_2),
            f"attempt3 (#589 fix: attempt 2 with the 0.4 mm ground clearance "
            f"hole REALIZED as {t.N_GROUND_BOXES_3} half-cell PEC boxes; "
            f"pec_mask differs from attempt2 in {t.HOLE_XOR_CELLS_3} cells at "
            f"the junction node only; kwargs = _attempt2_kwargs), "
            f"n_steps={n_steps} (record target: {target})",
        )
    if fixture == "attempt3b":
        return FluxFixtureSelection(
            t._build_coax_msl_transition_sim_attempt3b,
            t._attempt3b_kwargs(n_steps),
            dict(geom_2),
            f"attempt3b (#823: attempt 3's GEOMETRY byte-for-byte -- the "
            f"builder IS attempt 3's -- with the COMPLIANT MSL ladder "
            f"msl_probe_count={t.PROBE_COUNT_3B} / start_cells="
            f"{t.PROBE_START_3B} / spacing_cells={t.PROBE_SPACING_3B}, i.e. "
            f"probes at x = 2.5 ... 9.5 mm, every one at least "
            f"{t._msl_near_field_standoff_cells(t.DX, t.H_SUB)} cells = "
            f"5*h_sub from BOTH the port feed plane and the junction; "
            f"attempt 3's ladder violates that at 2 of its 9 probes), "
            f"n_steps={n_steps} (record target: {target})",
        )
    if fixture == "attempt2_wide":
        return FluxFixtureSelection(
            t._build_coax_msl_transition_sim_attempt2_wide,
            t._attempt2_wide_kwargs(n_steps),
            {"LX_m": t.LX_2W, "LY_m": t.LY_2W, "LZ_m": t.LZ_2,
             "junction_x_m": t.JUNCTION_X_2W, "feed_x_m": t.FEED_X_2W,
             "y_c_m": t.Y_C_2W},
            f"attempt2_wide (Step B falsifier for candidate (c): junction "
            f"cells byte-identical to attempt2, LY {t.LY_2W * 1e3:.1f} mm, "
            f"junction {t.JUNCTION_X_2W * 1e3:.1f} mm from the -x CPML inner "
            f"edge, LX {t.LX_2W * 1e3:.1f} mm, feed {t.FEED_X_2W * 1e3:.1f} "
            f"mm), n_steps={n_steps}",
        )
    raise ValueError(
        f"unknown --fixture {fixture!r}; expected one of {FIXTURE_CHOICES}")


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
    ap.add_argument(
        "--fixture", choices=FIXTURE_CHOICES, default="attempt2",
        help="attempt2 = the settled fixture (Step A); attempt2_wide = Step B "
             "domain-clearance falsifier for candidate (c) absorber proximity; "
             "attempt3 = attempt 2 with the ground-plane clearance hole REALIZED "
             "(#589 fix; NOT COMPARABLE to the attempt-2 baseline by design); "
             "attempt3b = attempt 3's GEOMETRY UNCHANGED with the #823 compliant "
             "MSL probe ladder (count 8 / start 15 / spacing 10, probes at "
             "x = 2.5..9.5 mm, every probe >= 5*h_sub from both the port feed "
             "plane and the junction) -- 3b vs 3 is a pure ladder-recipe "
             "comparison on bit-identical geometry",
    )
    ap.add_argument(
        "--preflight", action="store_true",
        help="run sim.preflight() and sim.fidelity_report() BEFORE the solve and "
             "dump both verbatim (the method's own skip_preflight flag is inert)",
    )
    ap.add_argument(
        "--baseline", type=str, default=None,
        help="baseline result JSON for the A0 two-tier reproduction comparison "
             "(e.g. the tracked settled_run_369367252283_result.json)",
    )
    ap.add_argument(
        "--freqs", type=str, default=None,
        help="optional dense band in GHz: comma list '5,5.5,6' or 'start:stop:n'; "
             "the committed 6/8/10 GHz bins are always retained",
    )
    ap.add_argument(
        "--dump-ladders", action="store_true",
        help="ask the method for the raw per-probe ladder voltages "
             "(return_ladder_voltages=True) and write them to <output>.ladders.npz; "
             "the keyword is detected with inspect.signature, and when it is not on "
             "this checkout the solve runs UNCHANGED and W1-W3 report SKIPPED",
    )
    ap.add_argument(
        "--flux", action="store_true",
        help="pass the attempt-3 W4 flux box (six faces of one lossless control "
             "volume + the full-plane +x comparator) as extra_flux_monitors; "
             "attempt3 / attempt3b only (they share one grid; the plane "
             "coordinates are that grid's)",
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
    x64_enabled = bool(jax.config.jax_enable_x64)
    solve_complex_dtype = "complex128" if x64_enabled else "complex64"
    print(f"precision  : jax_enable_x64={x64_enabled}  "
          f"JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64', '<unset>')}  "
          f"solve complex dtype (method rule)={solve_complex_dtype}  "
          f"(369367252283 record: default float32 GPU run, no x64 pin)")

    import test_coax_msl_transition as t  # noqa: E402
    from rfx.api._sparams import _mixed_reciprocity_deviation, _warn_if_nonpassive_smatrix
    from rfx.core.yee import EPS_0, MU_0
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

    n_steps = args.n_steps if args.n_steps is not None else t.SETTLED_RUN_RECORD["target_n_steps"]
    selection = _select_fixture(t, args.fixture, n_steps)
    build, kwargs, fixture_geom = selection.build, selection.kwargs, selection.fixture_geom
    print(f"fixture    : {selection.banner}")

    committed_freqs = np.asarray(t.FREQS_2, dtype=float)
    if args.freqs:
        merged = list(committed_freqs)
        for f in _parse_freqs_ghz(args.freqs):
            if not np.any(np.isclose(merged, f)):
                merged.append(float(f))
        freqs_all = np.array(sorted(merged), dtype=float)
    else:
        freqs_all = committed_freqs.copy()
    committed = []
    for f in committed_freqs:
        j = np.where(np.isclose(freqs_all, f))[0]
        assert len(j) == 1, f"committed bin {f} lost from the frequency axis"
        committed.append(int(j[0]))
    kwargs["freqs"] = freqs_all
    print(f"freqs      : {len(freqs_all)} bins ({freqs_all.min() / 1e9:.3g}-"
          f"{freqs_all.max() / 1e9:.3g} GHz); committed bins at indices {committed} "
          f"= {(freqs_all[committed] / 1e9).tolist()} GHz")
    print(f"skip_preflight: {SKIP_PREFLIGHT_INERT_NOTE}")

    sim = build()

    # -- #589 witness half: both opt-ins default OFF (legacy keys unchanged) --
    ladders_available = None
    if args.dump_ladders:
        import inspect
        try:
            params = inspect.signature(type(sim).compute_coax_msl_transition).parameters
            ladders_available = "return_ladder_voltages" in params
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            ladders_available = False
            print(f"--dump-ladders: inspect.signature failed ({exc}); assuming absent")
        if ladders_available:
            kwargs["return_ladder_voltages"] = True
            print("--dump-ladders: compute_coax_msl_transition accepts "
                  "return_ladder_voltages -- raw ladders will be dumped")
        else:
            print("--dump-ladders: this checkout's compute_coax_msl_transition has NO "
                  "return_ladder_voltages keyword (the #589 production half is not "
                  "merged here) -- the solve runs UNCHANGED and W1-W3 will report "
                  "SKIPPED; W4 and the label-swap counterfactual are unaffected")
    if args.flux:
        # attempt3b IS attempt 3's fixture (same builder, same grid); only the
        # msl_probe_* kwargs differ, and the W4 plane coordinates do not
        # depend on them. Any other fixture has a different grid.
        if args.fixture not in ("attempt3", "attempt3b"):
            print(f"FATAL: --flux is attempt3/attempt3b only (the plane coordinates "
                  f"are that grid's); got --fixture {args.fixture}")
            return 2
        kwargs["extra_flux_monitors"] = t._attempt3_scratch_flux_entries()
        print(f"--flux: {len(kwargs['extra_flux_monitors'])} extra flux monitors "
              f"({', '.join(m.name for m in kwargs['extra_flux_monitors'])}) -- "
              "non-perturbation witnessed by test_extra_flux_monitors_do_not_perturb_s")

    preflight_lines = None
    fidelity_lines = None
    fidelity_rows = None
    if args.preflight:
        print("\n=== preflight (driver-side; the method's own flag is inert) ===")
        try:
            report = sim.preflight()
            preflight_lines = report.format().splitlines()
        except Exception as exc:  # report verbatim, do not swallow
            preflight_lines = [f"sim.preflight() raised {type(exc).__name__}: {exc}"]
        for line in preflight_lines:
            print(line)
        if hasattr(sim, "fidelity_report"):
            print("\n=== fidelity_report (declared vs realized, pre-solve) ===")
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    fidelity_rows = sim.fidelity_report(print_report=True)
                fidelity_lines = buf.getvalue().splitlines()
                fidelity_rows = json.loads(json.dumps(fidelity_rows, default=str))
            except Exception as exc:
                fidelity_lines = buf.getvalue().splitlines() + [
                    f"sim.fidelity_report() raised {type(exc).__name__}: {exc}"
                ]
            for line in fidelity_lines:
                print(line)
        else:
            fidelity_lines = ["Simulation has no fidelity_report() on this checkout"]
            print(fidelity_lines[0])

    solve_warnings: list[str] = []
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.compute_coax_msl_transition(**kwargs)
    elapsed_s = time.perf_counter() - t0
    for w in caught:
        text = f"{w.filename}:{w.lineno}: {w.category.__name__}: {w.message}"
        solve_warnings.append(text)
        print(text, file=sys.stderr)
    print(f"\nFDTD run complete in {elapsed_s:.1f}s ({elapsed_s / 60.0:.1f} min)")

    assert result.s_params.shape == (2, 2, len(freqs_all))
    assert np.all(np.isfinite(result.s_params))
    assert result.port_names == ("coax", "msl")

    S_all = np.asarray(result.s_params)
    S = S_all[:, :, committed]  # legacy keys: committed bins only
    freqs_committed = np.asarray(result.freqs)[committed]

    settling_db = np.asarray(result.settling_db).tolist()
    cond_a_equilibrated = np.asarray(result.cond_a_equilibrated)[committed].tolist()

    c0 = 1.0 / np.sqrt(MU_0 * EPS_0)
    _, eps_eff_hj = hammerstad_jensen_z0_eps_eff(t.W_TRACE, t.H_SUB, t.EPS_SUB)
    beta_analytic_all = 2.0 * np.pi * np.asarray(result.freqs) * np.sqrt(eps_eff_hj) / c0
    im_gamma_coax_driven_all = np.abs(np.asarray(result.gamma)[1, 0, :].imag)
    gamma_ratio_all = im_gamma_coax_driven_all / beta_analytic_all
    gamma_ratio = gamma_ratio_all[committed].tolist()

    col_msl_driven_power = (np.abs(S[0, 1, :]) ** 2 + np.abs(S[1, 1, :]) ** 2).tolist()

    pair, worst_dev = _mixed_reciprocity_deviation(S)
    s22_abs = np.abs(S[1, 1, :]).tolist()
    max_abs_s = float(np.max(np.abs(S)))

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
        "freqs_hz": freqs_committed.tolist(),
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

    # ------------------------------------------------------------------
    # Issue #589 extension: everything below is ADDITIVE (new keys only).
    # ------------------------------------------------------------------
    nf = len(freqs_all)
    q_lambda_min = np.empty(nf)
    q_trace = np.empty(nf)
    for k in range(nf):
        Sk = S_all[:, :, k]
        Q = np.eye(2) - Sk.conj().T @ Sk
        q_lambda_min[k] = float(np.min(np.linalg.eigvalsh(Q)))
        q_trace[k] = float(np.real(np.trace(Q)))
    col_coax_all = np.abs(S_all[0, 0, :]) ** 2 + np.abs(S_all[1, 0, :]) ** 2
    col_msl_all = np.abs(S_all[0, 1, :]) ** 2 + np.abs(S_all[1, 1, :]) ** 2
    a_inc = np.asarray(result.a_inc)
    b_out = np.asarray(result.b_out)
    with np.errstate(divide="ignore", invalid="ignore"):
        a5_coax_side = np.abs(a_inc[0, 1, :] / b_out[0, 1, :])   # coax array, MSL drive
        a5_msl_side = np.abs(a_inc[1, 0, :] / b_out[1, 0, :])    # MSL array, coax drive
    fit = np.asarray(result.fit_residual)
    rec = np.asarray(result.recurrence_residual)
    gam = np.asarray(result.gamma)
    pair_all, worst_dev_all = _mixed_reciprocity_deviation(S_all)

    a2_verdict = _classify_a2(q_lambda_min[committed])
    a3_verdict = _classify_a3(fit, rec, freqs_all, committed)
    a5_coax_verdict = _classify_a5(a5_coax_side, committed)
    a5_msl_verdict = _classify_a5(a5_msl_side, committed)

    def _pd(arr):  # per (array, drive) nested lists
        return {
            "coax_array": {"coax_drive": arr[0, 0, :].tolist(), "msl_drive": arr[0, 1, :].tolist()},
            "msl_array": {"coax_drive": arr[1, 0, :].tolist(), "msl_drive": arr[1, 1, :].tolist()},
        }

    ext = {
        "fixture": args.fixture,
        "fixture_geometry": fixture_geom,
        "grid_shape_padded": None,
        "port_order": list(result.port_names),
        "index_convention": "S[j, i, f] = response at port j driving port i; "
                            "a_inc/b_out/fit_residual/recurrence_residual/gamma are "
                            "[port array, drive, f]",
        "z0_ref_ohm": np.asarray(result.z0_ref).tolist(),
        "reference_planes_m": np.asarray(result.reference_planes).tolist(),
        "freqs_hz_all": np.asarray(result.freqs).tolist(),
        "committed_bin_indices": committed,
        "committed_freqs_hz": freqs_committed.tolist(),
        "precision": {
            "jax_enable_x64": x64_enabled,
            "env_JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
            "solve_complex_dtype_per_method_rule": solve_complex_dtype,
            "solve_dtype_note": "compute_coax_msl_transition sets _complex_dtype = complex128 if "
                                "jax.config.x64_enabled else complex64 for the DFT-plane/modal-"
                                "voltage path; the FDTD fields follow the same x64 switch. "
                                "s_params/a_inc/gamma below are the NumPy ASSEMBLY dtype "
                                "(always complex128), not the solve precision.",
            "s_params_dtype": str(S_all.dtype),
            "a_inc_dtype": str(a_inc.dtype),
            "gamma_dtype": str(gam.dtype),
            "record_369367252283": "default float32 GPU run (no x64 pin in driver or test)",
            "f64_replicate_falsifier": "|S10| or |S01| moving > 2x between f32 and f64 at any "
                                       "bin => FLOOR; |S22|, |S00|, col_power must reproduce "
                                       "within the A0 budget",
        },
        "s_complex": {"re": S_all.real.tolist(), "im": S_all.imag.tolist()},
        "s_abs_db": {
            "S00_coax_refl": _db(S_all[0, 0, :]),
            "S10_coax_drive_to_msl": _db(S_all[1, 0, :]),
            "S01_msl_drive_to_coax": _db(S_all[0, 1, :]),
            "S11_msl_refl": _db(S_all[1, 1, :]),
        },
        "s_abs": {
            "S00": np.abs(S_all[0, 0, :]).tolist(), "S10": np.abs(S_all[1, 0, :]).tolist(),
            "S01": np.abs(S_all[0, 1, :]).tolist(), "S11": np.abs(S_all[1, 1, :]).tolist(),
        },
        "col_coax_driven_power": col_coax_all.tolist(),
        "col_msl_driven_power_all_bins": col_msl_all.tolist(),
        "max_abs_s_all_bins": float(np.max(np.abs(S_all))),
        "reciprocity_worst_deviation_all_bins": {
            "pair": list(pair_all) if pair_all is not None else None,
            "value": float(worst_dev_all),
        },
        "a2_passivity_psd": {
            "lambda_min_I_minus_SHS": q_lambda_min.tolist(),
            "trace_I_minus_SHS": q_trace.tolist(),
            "falsifier": "lambda_min < -0.02 at any committed bin => EXTRACTOR",
            "verdict": a2_verdict,
        },
        "cond_a_raw": np.asarray(result.cond_a).tolist(),
        "cond_a_equilibrated_all_bins": np.asarray(result.cond_a_equilibrated).tolist(),
        "cond_a_note": "raw cond_a is a per-drive SCALE disparity (run-log warning: 'nearly "
                       "linearly dependent ... worst 2.91e+07'); equilibration removes it and "
                       "S is exactly invariant to it -- not a reciprocity attribution",
        "a3_residuals": {
            "fit_residual": _pd(fit),
            "recurrence_residual": _pd(rec),
            "threshold_provenance": "0.02 = coax lane's borrowed convenience number "
                                    "(tests/test_coax_two_port_fdtd.py, measured 0.0127); "
                                    "load-bearing criterion = MSL-driven residual at 6/8 GHz "
                                    "> 10x own 10 GHz value",
            "falsifier": "MSL-DRIVEN MSL-ladder fit or recurrence residual > 0.02 or > 10x "
                         "own 10 GHz at a committed bin => EXTRACTOR ((b)-on-ladder supported); "
                         "(b)-on-ladder FALSIFIED only when fit AND recurrence residuals of the "
                         "MSL ladder are <= 0.02 (and <= 10x own 10 GHz) at ALL committed bins "
                         "under BOTH drives; else NON-CLOSING (not falsified)",
            "verdict_msl_ladder_both_drives": a3_verdict,
        },
        "a_inc": {"abs": _pd(np.abs(a_inc)), "re": _pd(a_inc.real), "im": _pd(a_inc.imag)},
        "b_out": {"abs": _pd(np.abs(b_out)), "re": _pd(b_out.real), "im": _pd(b_out.imag)},
        "a5_echo_ratios": {
            "coax_array_msl_drive_abs_a_over_b": a5_coax_side.tolist(),
            "msl_array_coax_drive_abs_a_over_b": a5_msl_side.tolist(),
            "expected_coax_side": A5_EXPECT_COAX_SIDE,
            "expected_coax_side_basis": "coax annular termination sized to analytic z_tem "
                                        "45.46 ohm (r_feed = z_tem when feed_impedance is None)",
            "expected_msl_side": A5_EXPECT_MSL_SIDE,
            "expected_msl_side_basis": "MSL feed sized to registered 50 ohm vs HJ 53.11 ohm "
                                       "=> |Gamma_feed| = 3.11/103.11",
            "verdict_coax_side": a5_coax_verdict,
            "verdict_msl_side": a5_msl_verdict,
        },
        "gamma": {"re": _pd(gam.real), "im": _pd(gam.imag)},
        "gamma_ratio_coax_driven_all_bins": gamma_ratio_all.tolist(),
        "settling_db": settling_db,
        "settling_cleared_40db": settling_cleared_40db,
        "solve_warnings": solve_warnings,
        "preflight": {
            "requested": bool(args.preflight),
            "skip_preflight_inert_note": SKIP_PREFLIGHT_INERT_NOTE,
            "report_lines": preflight_lines,
            "fidelity_report_lines": fidelity_lines,
            "fidelity_report_rows": fidelity_rows,
        },
        "a0_reproduction": None,
    }
    try:
        grid = sim._build_grid()
        ext["grid_shape_padded"] = [int(grid.nx), int(grid.ny), int(grid.nz)]
    except Exception as exc:  # provenance-only
        ext["grid_shape_padded"] = f"<unavailable: {exc}>"

    # ---- #823: the ladder RECIPE next to its REALIZED coordinates --------
    # The whole point of #823 is that the two were never compared. Report-only:
    # nothing here gates or refuses, and the standoff numbers are computed with
    # the test module's own helpers so the driver and the ladder-contract test
    # cannot drift apart.
    try:
        xs_realized = t._msl_ladder_x_coords(
            sim, count=kwargs["msl_probe_count"],
            start_cells=kwargs["msl_probe_start_cells"],
            spacing_cells=kwargs["msl_probe_spacing_cells"])
        n_port, n_ref, d_port, d_ref, required_m = t._msl_standoff_violations(
            xs_realized, feed_x=fixture_geom["feed_x_m"],
            ref_x=fixture_geom["junction_x_m"], dx=t.DX, h_sub=t.H_SUB)
        ext["msl_ladder_standoff"] = {
            "recipe": {"msl_probe_count": int(kwargs["msl_probe_count"]),
                       "msl_probe_start_cells": int(kwargs["msl_probe_start_cells"]),
                       "msl_probe_spacing_cells": int(kwargs["msl_probe_spacing_cells"])},
            "realized_x_m": xs_realized.tolist(),
            "feed_plane_x_m": float(fixture_geom["feed_x_m"]),
            "reference_plane_x_m": float(fixture_geom["junction_x_m"]),
            "required_standoff_m": required_m,
            "required_standoff_cells": t._msl_near_field_standoff_cells(t.DX, t.H_SUB),
            "required_standoff_over_h_sub": required_m / t.H_SUB,
            "min_d_port_m": d_port, "min_d_reference_m": d_ref,
            "min_d_port_over_h_sub": d_port / t.H_SUB,
            "min_d_reference_over_h_sub": d_ref / t.H_SUB,
            "n_violating_port_end": n_port,
            "n_violating_reference_end": n_ref,
            "rule": ("max(3, round(5*h_sub/dx)) cells from BOTH the MSL port "
                     "feed plane and the reference plane the ladder is referred "
                     "to; the same issue-#80 Fix B constant add_msl_port already "
                     "floors its AUTO probe offset at. REPORT-ONLY here."),
        }
        print(f"\n=== #823 MSL ladder standoff (report-only) ===")
        print(f"  recipe (count/start/spacing) : "
              f"{kwargs['msl_probe_count']}/{kwargs['msl_probe_start_cells']}/"
              f"{kwargs['msl_probe_spacing_cells']}")
        print(f"  realized x (mm)              : "
              f"{[round(x * 1e3, 4) for x in xs_realized.tolist()]}")
        print(f"  required standoff            : {required_m * 1e3:.3f} mm = "
              f"{required_m / t.H_SUB:.3f} h_sub "
              f"({t._msl_near_field_standoff_cells(t.DX, t.H_SUB)} cells)")
        print(f"  min distance port / ref (mm) : {d_port * 1e3:.3f} / {d_ref * 1e3:.3f}"
              f"  ({d_port / t.H_SUB:.3f} / {d_ref / t.H_SUB:.3f} h_sub)")
        print(f"  VIOLATING probes port / ref  : {n_port} / {n_ref}")
    except Exception as exc:  # noqa: BLE001 -- report-only
        ext["msl_ladder_standoff"] = f"<unavailable: {type(exc).__name__}: {exc}>"
        print(f"\n  WARNING: #823 ladder standoff report unavailable: {exc}")

    # ---- #823: the extractor's disjoint-half self-consistency witness ----
    # Report-only, and DETECTED rather than assumed: on a checkout where the
    # production half of #823 is not merged the result carries no such fields
    # and the driver says so instead of failing (the same pattern
    # --dump-ladders uses for return_ladder_voltages).
    split_fields = ("ladder_split_gamma_dev", "ladder_split_reflection_decades")
    if all(hasattr(result, f) for f in split_fields):
        ext["ladder_split_witness"] = {f: _pd(np.asarray(getattr(result, f)))
                                       for f in split_fields}
        print("\n=== #823 ladder self-consistency witness (disjoint halves; report-only) ===")
        for f in split_fields:
            arr = np.asarray(getattr(result, f))
            print(f"  {f}:")
            for name, jj in (("coax_array", 0), ("msl_array", 1)):
                print(f"    {name}: coax_drive {np.array2string(arr[jj, 0, :], precision=5)}"
                      f"  msl_drive {np.array2string(arr[jj, 1, :], precision=5)}")
    else:
        ext["ladder_split_witness"] = (
            "NOT ON THIS CHECKOUT: CoaxMSLTransitionResult carries no "
            f"{'/'.join(split_fields)} field (the #823 production half is not "
            "merged here); the solve ran UNCHANGED and no legacy key is affected"
        )
        print(f"\n  #823 ladder self-consistency witness: {ext['ladder_split_witness']}")

    if args.baseline:
        ext["a0_reproduction"] = _a0_compare(
            args.baseline, freqs_all=freqs_all, committed=committed, S=S_all,
            n_steps=n_steps, fixture=args.fixture, legacy=out, x64_enabled=x64_enabled,
        )
    out["fixture"] = args.fixture
    out["ext_589"] = ext

    fg = freqs_all / 1e9
    print("\n=== #589 extended witnesses (all bins; committed bins marked *) ===")
    print(f"  precision: jax_enable_x64={x64_enabled}, solve complex dtype={solve_complex_dtype} "
          f"(method rule); s_params dtype={S_all.dtype}, a_inc dtype={a_inc.dtype} (assembly)")
    print("  bin  f[GHz]   |S00|dB   |S10|dB   |S01|dB   |S11|dB   colP_coax  colP_msl   "
          "lam_min(I-S^HS)  cond_a_raw   cond_a_eq")
    for k in range(nf):
        mark = "*" if k in committed else " "
        print(f"  {mark}{k:2d}  {fg[k]:6.3f}  {ext['s_abs_db']['S00_coax_refl'][k]:8.2f}  "
              f"{ext['s_abs_db']['S10_coax_drive_to_msl'][k]:8.2f}  "
              f"{ext['s_abs_db']['S01_msl_drive_to_coax'][k]:8.2f}  "
              f"{ext['s_abs_db']['S11_msl_refl'][k]:8.2f}  {col_coax_all[k]:9.5f}  "
              f"{col_msl_all[k]:9.5f}  {q_lambda_min[k]:+.5e}  "
              f"{ext['cond_a_raw'][k]:.3e}  {ext['cond_a_equilibrated_all_bins'][k]:.6f}")
    print("  complex S per bin (re, im):")
    for k in range(nf):
        mark = "*" if k in committed else " "
        cells = "  ".join(
            f"S{j}{i}=({S_all[j, i, k].real:+.6e},{S_all[j, i, k].imag:+.6e})"
            for j in range(2) for i in range(2)
        )
        print(f"  {mark}{k:2d}  {fg[k]:6.3f}  {cells}")
    print(f"  A2 verdict: {a2_verdict}")
    print("  A3 fit_residual [array/drive] per bin (recurrence in parentheses):")
    for k in range(nf):
        mark = "*" if k in committed else " "
        print(f"  {mark}{k:2d}  {fg[k]:6.3f}  coax/coax {fit[0, 0, k]:.4g} ({rec[0, 0, k]:.3g})  "
              f"coax/msl {fit[0, 1, k]:.4g} ({rec[0, 1, k]:.3g})  "
              f"msl/coax {fit[1, 0, k]:.4g} ({rec[1, 0, k]:.3g})  "
              f"msl/msl {fit[1, 1, k]:.4g} ({rec[1, 1, k]:.3g})")
    print(f"  A3 verdict (MSL ladder, BOTH drives, fit+recurrence, committed bins): {a3_verdict}")
    print("  |a_inc| / |b_out| per bin [coax array: coax drive, msl drive | msl array: coax drive, msl drive]:")
    for k in range(nf):
        mark = "*" if k in committed else " "
        print(f"  {mark}{k:2d}  {fg[k]:6.3f}  a: {np.abs(a_inc[0, 0, k]):.4e}, {np.abs(a_inc[0, 1, k]):.4e} | "
              f"{np.abs(a_inc[1, 0, k]):.4e}, {np.abs(a_inc[1, 1, k]):.4e}   "
              f"b: {np.abs(b_out[0, 0, k]):.4e}, {np.abs(b_out[0, 1, k]):.4e} | "
              f"{np.abs(b_out[1, 0, k]):.4e}, {np.abs(b_out[1, 1, k]):.4e}")
    print("  A5 |a_inc/b_out| per bin: coax array under MSL drive (expect ~0) | "
          "MSL array under coax drive (expect ~0.030)")
    for k in range(nf):
        mark = "*" if k in committed else " "
        print(f"  {mark}{k:2d}  {fg[k]:6.3f}  {a5_coax_side[k]:.5g} | {a5_msl_side[k]:.5g}")
    print(f"  A5 verdict coax side: {a5_coax_verdict}")
    print(f"  A5 verdict MSL side : {a5_msl_verdict}")
    print(f"  settling_db per drive: {settling_db}  (cleared -40 dB: {settling_cleared_40db})")
    print(f"  solve warnings captured: {len(solve_warnings)} (dumped verbatim in JSON)")

    if ext["a0_reproduction"] is not None:
        a0 = ext["a0_reproduction"]
        print("\n=== A0 reproduction vs baseline (two-tier predeclaration; budget UNMEASURED) ===")
        print(f"  baseline : {a0['baseline_path']}")
        print(f"  status   : {a0['status']}")
        for note in a0.get("notes", []):
            print(f"  NOTE     : {note}")
        for key, val in (a0.get("deltas") or {}).items():
            print(f"  delta {key}: {val}")
        if a0.get("compared"):
            print(f"  compared : {a0['compared']}")
        if a0.get("not_comparable"):
            print(f"  cannot compare against this baseline: {a0['not_comparable']}")
        if a0.get("precision"):
            pr = a0["precision"]
            print(f"  precision: this run jax_enable_x64={pr['this_run_jax_enable_x64']}  "
                  f"baseline jax_enable_x64={pr['baseline_jax_enable_x64']}  "
                  f"({pr['baseline_source']})")
        print(f"  tier value (max |delta| over comparable |S|-type quantities): {a0.get('tier_value')}")
        print(f"  TIER     : {a0.get('tier')}")
        if a0.get("f64_replicate"):
            f64 = a0["f64_replicate"]
            print("  --- f64 replicate rule (review required change 2; predeclared, computed) ---")
            print(f"  f64 side : {f64['f64_side']}   f32 side: {f64['f32_side']}   "
                  f"same fixture/steps: {f64['same_fixture_and_steps']}")
            print(f"  |S10| f32/f64 ratio per committed bin: {f64['ratio_S10']}")
            print(f"  |S01| f32/f64 ratio per committed bin: {f64['ratio_S01']}")
            print(f"  magnitude deltas (must be within A0 budget): {f64['magnitude_deltas']}")
            if f64["not_computable"]:
                print(f"  not computable against this baseline: {f64['not_computable']}")
            print(f"  F64 RULE : {f64['verdict']}")

    out_path = Path(args.output) if args.output else (
        REPO / ".omx" / "coax-msl-transition-settled-run" /
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{_git_sha()[:12]}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # The MEASUREMENT is persisted FIRST and unconditionally. The witness dump
    # below is report-only instrumentation on an irreplaceable ~2-4 h GPU run,
    # so it is never allowed to stand between the solve and the result JSON:
    # any exception inside it (np.savez onto a full/odd NFS path, an all-zero
    # subset raising in the pencil, np.linalg.lstsq) would otherwise destroy
    # the whole measurement. It runs after this write, inside try/except, and
    # the additive ext_589 keys are folded in by an ATOMIC re-write that
    # cannot truncate the file already on disk.
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nresult JSON written to: {out_path}")

    if args.dump_ladders or args.flux:
        beta_coax_analytic = (2.0 * np.pi * np.asarray(result.freqs)
                              * np.sqrt(t.EPS_COAX) / c0)
        try:
            _witness_dump(
                result=result, ext=ext, out_path=out_path, freqs_all=np.asarray(result.freqs),
                beta_coax_analytic=beta_coax_analytic, beta_msl_analytic=beta_analytic_all,
                ladders_requested=bool(args.dump_ladders),
                ladders_available=ladders_available, flux_requested=bool(args.flux),
            )
        except Exception:  # noqa: BLE001 -- report-only witness must not kill the run
            import traceback as _tb
            witness_tb = _tb.format_exc()
            ext["witness_error"] = witness_tb
            print("\n  WARNING: the REPORT-ONLY witness dump FAILED. The measurement "
                  "JSON above is already on disk and is unaffected; the traceback is "
                  "recorded in ext_589['witness_error'].")
            print(witness_tb)
        # Fold the additive ext_589 keys (or witness_error) into the JSON
        # without ever truncating the file that is already there.
        try:
            tmp_path = out_path.with_name(out_path.name + ".tmp")
            tmp_path.write_text(json.dumps(out, indent=2))
            os.replace(tmp_path, out_path)
            print(f"result JSON re-written with the witness keys: {out_path}")
        except Exception:  # noqa: BLE001
            import traceback as _tb
            print("  WARNING: could not re-write the result JSON with the witness "
                  "keys; the measurement JSON from the first write STANDS as it is "
                  "(witness artifacts are on disk next to it).")
            print(_tb.format_exc())

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

    print("\n=== #589 EXTENSION SUMMARY (committed bins; report-only) ===")
    print(f"  fixture              : {args.fixture}  grid(padded)={ext['grid_shape_padded']}")
    print(f"  precision            : jax_enable_x64={x64_enabled}  solve={solve_complex_dtype}  "
          f"s_params(assembly)={S_all.dtype}")
    print(f"  |S00|                : {[ext['s_abs']['S00'][k] for k in committed]}")
    print(f"  |S10| dB / |S01| dB  : {[ext['s_abs_db']['S10_coax_drive_to_msl'][k] for k in committed]} / "
          f"{[ext['s_abs_db']['S01_msl_drive_to_coax'][k] for k in committed]}")
    print(f"  col_coax_driven_power: {[float(col_coax_all[k]) for k in committed]}")
    print(f"  lambda_min(I-S^HS)   : {[float(q_lambda_min[k]) for k in committed]}  -> {a2_verdict}")
    print(f"  cond_a raw / eq      : {[ext['cond_a_raw'][k] for k in committed]} / {cond_a_equilibrated}")
    print(f"  A3 (msl/msl fit_res) : {[float(fit[1, 1, k]) for k in committed]}  "
          f"(msl/coax fit_res {[float(fit[1, 0, k]) for k in committed]})  -> {a3_verdict}")
    print(f"  A5 coax|MSL ratios   : {[float(a5_coax_side[k]) for k in committed]} | "
          f"{[float(a5_msl_side[k]) for k in committed]}")
    print(f"  A5 verdicts          : coax: {a5_coax_verdict} ; msl: {a5_msl_verdict}")
    if ext["a0_reproduction"] is not None:
        print(f"  A0 tier              : {ext['a0_reproduction'].get('tier')}  "
              f"[{ext['a0_reproduction'].get('status')}]")
    print(f"  preflight lines      : "
          f"{None if preflight_lines is None else len(preflight_lines)}  "
          f"(method skip_preflight flag is inert; see header)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
