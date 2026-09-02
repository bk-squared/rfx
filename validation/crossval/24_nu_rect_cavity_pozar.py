#!/usr/bin/env python3
"""Cross-validation 24: rectangular PEC cavity eigenfrequencies on GRADED-z
Yee meshes vs the EXACT Pozar spectrum (E2), with the uniform mesh as the
control and the exact discrete lattice as the a-priori witness.

cv14's cavity (a, b, d) = (50, 30, 40) mm, PEC, air, same sources, same
probe point, same harminv -- solved four ways:

    (a) uniform       cv14's 1 mm mesh (the control)
    (b) single_band   one 0.5 mm band inside the #785 envelope (chain 0.5-0.7-0.8-1.0)
    (c) multi_band    small-large-small-large inside the envelope
    (d) uniform_fine  0.5 mm everywhere (the #810 cost control)

Gated per arm (numbers in ONE place: validation/crossval/comparators/nu_cavity_gates.py;
pre-declared in docs/design_notes/20260902_cv24_nu_cavity_predeclaration.md):
  G1 cv14's committed tolerance (TE101 < 1 %, >= 1 higher mode < 2 %)
  G2 allowance (graded arms): |dev_sp(g)| <= |dev_sp(uniform)| + A(mode) + W_est
     on the equal-dt SPATIAL deviation (the leapfrog relation inverted exactly)
  G3 lattice: |f_meas / f_lattice - 1| <= W_est = 4 ppm
  G4 mode count: 7 clusters in [4, 8.5] GHz, each declared (m, n, l) owning one
  G5 energy: #785's F-S1 envelope on the arm's own grid (source-free Remis energy)
  G6 stationarity: two 2/3 sub-windows agree per mode to W_est
  envelope: the arm's profile is inside the declared #785 envelope

Exit: 0 every gate on every arm run; 1 otherwise (a --falsifier arm MUST exit 1);
2 unreachable (no external reference).

    python validation/crossval/24_nu_rect_cavity_pozar.py                 # four arms
    python validation/crossval/24_nu_rect_cavity_pozar.py --arm single_band
    python validation/crossval/24_nu_rect_cavity_pozar.py --falsifier metric_defect
    python validation/crossval/24_nu_rect_cavity_pozar.py --smoke          # <= 20 s, no gates
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "comparators"))
import nu_cavity_gates as G  # noqa: E402

from rfx.api import Simulation  # noqa: E402
from rfx.harminv import harminv  # noqa: E402

RESULTS_DIR = os.path.join(SCRIPT_DIR, G.RESULTS_DIRNAME)
SMOKE_RECORD_DIVISOR = 8
HARMINV_DECIMATE = False      # declared before the run (note section 12)
ENERGY_SAMPLE_EVERY = 500
_CHAN = {"ex": 0, "ey": 1, "ez": 2}


def _staged_commit() -> str:
    """Provenance: the orchestrator writes the source commit to .staged_commit
    at staging time (cv22 review finding 6); git only as a fallback."""
    p = Path(_REPO_ROOT) / ".staged_commit"
    if p.is_file() and p.read_text().strip():
        return p.read_text().strip()
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# =============================================================================
# Builder (audited: returns the Simulation, never solves)
# =============================================================================

def build_cavity(lane: str, dxy: float, dz_profile) -> Simulation:
    """cv14's closed PEC cavity with its three soft sources and three scalar
    probes at its probe point. ``lane`` "uniform" builds the uniform Grid
    (cv14's recipe); "nonuniform" passes ``dz_profile`` to the NU lane."""
    if lane == "uniform":
        sim = Simulation(freq_max=G.FREQ_MAX, domain=(G.A_X, G.B_Y, G.D_Z), boundary="pec", dx=dxy)
    elif lane == "nonuniform":
        sim = Simulation(freq_max=G.FREQ_MAX, domain=(G.A_X, G.B_Y, G.D_Z), boundary="pec", dx=dxy,
                         dz_profile=np.asarray(dz_profile, dtype=np.float64))
    else:
        raise ValueError(lane)
    for pos, comp in G.SOURCES:
        sim.add_source(pos, component=comp)
    for comp in G.CHANNELS:
        sim.add_probe(G.PROBE, comp)
    return sim


def _grid_facts(sim: Simulation, lane: str) -> dict:
    grid = sim._build_nonuniform_grid() if lane == "nonuniform" else sim._build_grid()
    if lane == "nonuniform":
        dz = np.asarray(grid.dz_f64)[: grid.nz - 1]
        realized = float(np.sum(dz))
    else:
        realized = float((grid.nz - 1) * grid.dx)
    return {"nodes": [int(s) for s in grid.shape], "dt": float(grid.dt), "realized_d_m": realized}


def _preflight_lines(sim: Simulation) -> list[str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        report = sim.preflight()
    lines = buf.getvalue().splitlines()
    for item in (getattr(report, "issues", report) or []):
        lines.append(str(getattr(item, "code", item)))
    return lines or ["(preflight returned no issues)"]


# =============================================================================
# Extraction: harminv lines per channel per window, identification by index
# =============================================================================

def _lines(ts: np.ndarray, dt: float, n0: int, n1: int, band_hz) -> list[dict]:
    out = []
    for ch in G.CHANNELS:
        sig = ts[n0:n1, _CHAN[ch]].astype(np.float64)
        sig = sig - np.mean(sig)
        if len(sig) > G.HARMINV_MAX_SAMPLES:
            step = int(np.ceil(len(sig) / G.HARMINV_MAX_SAMPLES))
            sig_h, dt_h = sig[::step], dt * step
        else:
            sig_h, dt_h = sig, dt
        if len(sig_h) < 20:
            continue
        # decimate=False (note section 12): harminv's auto-decimation inflates
        # its rank threshold by sqrt(factor) and drops weakly excited modes of
        # this 12-mode signal, whose residual the retained poles absorb --
        # 10-20 ppm on the rig check; <= 0.5 ppm with the raw samples.
        for m in harminv(sig_h, dt_h, band_hz[0], band_hz[1], decimate=HARMINV_DECIMATE):
            out.append({"f_hz": float(m.freq), "amp": float(m.amplitude), "error": float(m.error),
                        "Q_artefact": float(m.Q), "channel": ch})
    return out


def extract(ts: np.ndarray, dt: float, rec: dict, band_hz, modes) -> dict:
    n_start, n_steps = rec["n_start"], ts.shape[0]
    n_post = n_steps - n_start
    n_third = n_post // 3
    windows = {"full": (n_start, n_steps),
               "A": (n_start, n_start + 2 * n_third),
               "B": (n_start + n_third, n_steps)}
    ident = {}
    for key, (a, b) in windows.items():
        lines = _lines(ts, dt, a, b, band_hz)
        ident[key] = G.identify_modes(lines, modes, band_hz)
        ident[key]["window_steps"] = [int(a), int(b)]
    full = ident["full"]
    stat = {}
    for name, rec_full in full["per_mode"].items():
        ra, rb = ident["A"]["per_mode"].get(name), ident["B"]["per_mode"].get(name)
        if rec_full is None or ra is None or rb is None:
            stat[name] = None
        else:
            stat[name] = abs(ra["f_hz"] - rb["f_hz"]) / rec_full["f_hz"]
    return {"per_mode": full["per_mode"], "n_clusters_in_band": full["n_clusters_in_band"],
            "orphans": full["orphans"], "ambiguous": full["ambiguous"], "stationarity": stat,
            "windows": {k: {"steps": v["window_steps"], "n_clusters_in_band": v["n_clusters_in_band"],
                            "per_mode": v["per_mode"]} for k, v in ident.items()}}


# =============================================================================
# Energy witness: #785's F-S1 on the arm's own grid (source-free, NU kernel)
# =============================================================================

def energy_audit(dz_profile, dxy: float, n_steps: int, expect_nodes, expect_dt: float) -> dict:
    import jax.numpy as jnp
    from validation.research.multiband_nu.harness import build_pec_fixture, random_blob_3d, run_energy_audit
    from validation.research.multiband_nu.w1_energy_drift import evaluate_fs1
    grid, mats = build_pec_fixture(np.asarray(dz_profile, np.float64), (G.A_X, G.B_Y), dxy)
    nodes = [int(grid.nx), int(grid.ny), int(grid.nz)]
    if nodes != list(expect_nodes) or abs(float(grid.dt) - expect_dt) > 1e-18:
        raise RuntimeError(f"energy-audit grid {nodes}/{float(grid.dt)} != run grid {expect_nodes}/{expect_dt}")
    init = random_blob_3d(grid, seed=1)
    init = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in init.items()}
    n = (int(n_steps) // ENERGY_SAMPLE_EVERY) * ENERGY_SAMPLE_EVERY
    steps, energies = run_energy_audit(grid, mats, init, n, sample_every=ENERGY_SAMPLE_EVERY)
    out = evaluate_fs1(np.asarray(steps), np.asarray(energies))
    out.update({"sample_every": ENERGY_SAMPLE_EVERY, "n_steps": int(n), "nodes": nodes,
                "energies": [float(e) for e in energies], "steps": [int(s) for s in steps],
                "kernel": "rfx.nonuniform._build_nu_scan (production NU step_fn) on the arm's grid"})
    return out


# =============================================================================
# One arm
# =============================================================================

def run_arm(name: str, lane: str, dxy: float, dz_profile, *, smoke: bool, search_band_hz,
            metric_swap: bool = False, with_energy: bool = True) -> dict:
    modes = G.declared_modes()
    dz = np.asarray(dz_profile, dtype=np.float64)
    pred = G.predict_arm(dz, dxy, modes)
    rec = dict(pred["record"])
    if smoke:
        rec["n_steps"] = rec["n_start"] + max(600, (rec["n_steps"] - rec["n_start"]) // SMOKE_RECORD_DIVISOR)
    sim = build_cavity(lane, dxy, dz)
    facts = _grid_facts(sim, lane)
    if abs(facts["dt"] - pred["dt"]) > 1e-18 or facts["nodes"] != pred["nodes"]:
        raise RuntimeError(f"[{name}] realized grid {facts} != predicted {pred['nodes']}/{pred['dt']}")
    pf = _preflight_lines(sim)

    import rfx.nonuniform as _nu
    saved = _nu._profile_to_inv_arrays
    if metric_swap:
        import jax.numpy as jnp

        def _swapped(profile_full):
            e_bad, h_bad = G.swapped_inv_arrays(np.asarray(profile_full, np.float64)[:-1])
            return jnp.asarray(e_bad, dtype=jnp.float32), jnp.asarray(h_bad, dtype=jnp.float32)
        _nu._profile_to_inv_arrays = _swapped
    t0 = time.time()
    energy = None
    wall_energy = 0.0
    try:
        res = sim.run(n_steps=int(rec["n_steps"]), compute_s_params=False, skip_preflight=True)
        wall_run = time.time() - t0
        # The energy audit runs INSIDE the metric-swap scope (review item 5, after
        # round 1): the round-1 artifact predates this and its metric_defect
        # energy witness saw the healthy metric (note section 14).
        if with_energy and not smoke:
            t2 = time.time()
            energy = energy_audit(dz, dxy, rec["n_steps"], facts["nodes"], float(res.dt))
            wall_energy = time.time() - t2
    finally:
        _nu._profile_to_inv_arrays = saved
    ts = np.asarray(res.time_series)
    dt = float(res.dt)
    if abs(dt - facts["dt"]) > 1e-18:
        raise RuntimeError(f"[{name}] result dt {dt} != grid dt {facts['dt']}")
    if not np.all(np.isfinite(ts)):
        raise RuntimeError(f"[{name}] non-finite time series")

    t1 = time.time()
    meas = extract(ts, dt, rec, search_band_hz, modes)
    wall_harminv = time.time() - t1
    meas["energy"] = energy or {}
    if energy is not None:
        energy["metric_swap_in_scope"] = bool(metric_swap)

    return {"name": name, "lane": lane, "dx": dxy, "profile_mm": (dz * 1e3).tolist(),
            "nodes": facts["nodes"], "cells": pred["cells"], "n_cells": pred["n_cells_total"],
            "realized_d_m": facts["realized_d_m"], "dt": dt, "n_steps": int(ts.shape[0]),
            "record": rec, "preflight": pf, "metric_swap": bool(metric_swap),
            "search_band_hz": list(search_band_hz), "measured": meas,
            "cost": {"n_cells": pred["n_cells_total"], "n_steps": int(ts.shape[0]),
                     "cell_steps": int(pred["n_cells_total"] * ts.shape[0]),
                     "wall_run_s": wall_run, "wall_harminv_s": wall_harminv, "wall_energy_s": wall_energy}}


def _print_arm(name: str, ev: dict, run: dict) -> None:
    print(f"\n[{name}] lane={run['lane']} dx={run['dx']*1e3:g} mm nodes={run['nodes']} dt={run['dt']:.4e} "
          f"n_steps={run['n_steps']} (record n_start {run['record']['n_start']}, t_post "
          f"{run['record']['t_post_s']*1e9:.2f} ns) wall run/harminv/energy "
          f"{run['cost']['wall_run_s']:.1f}/{run['cost']['wall_harminv_s']:.1f}/{run['cost']['wall_energy_s']:.1f} s")
    print(f"  profile (mm): {run['profile_mm']}")
    if run.get("declared_profile_mm") != run["profile_mm"]:
        print(f"  DECLARED profile (mm): {run['declared_profile_mm']}  (realized d {run['realized_d_m']*1e3:.4f} mm)")
    print(f"  preflight: {run['preflight']}")
    print(f"  envelope: {ev['envelope']['violations'] or 'ok'}")
    print(f"  {'mode':7s} {'Pozar GHz':>10s} {'meas GHz':>11s} {'dev ppm':>9s} {'pred ppm':>9s} "
          f"{'spatial':>9s} {'ctrl+A+W':>9s} {'lat resid':>10s} {'station.':>9s} chan")
    for mname, r in ev["rows"].items():
        if not r["found"]:
            print(f"  {mname:7s} {r['f_pozar_hz']/1e9:10.5f} {'NOT FOUND':>11s}")
            continue
        bound = r.get("allowance_bound")
        print(f"  {mname:7s} {r['f_pozar_hz']/1e9:10.5f} {r['f_meas_hz']/1e9:11.6f} {r['dev_raw']*1e6:9.1f} "
              f"{r['pred_dev_lattice']*1e6:9.1f} {r['dev_spatial']*1e6:9.1f} "
              f"{(bound*1e6 if bound is not None else float('nan')):9.1f} {r['resid_lattice']*1e6:10.2f} "
              f"{(r['stationarity']*1e6 if r['stationarity'] is not None else float('nan')):9.2f} "
              f"{','.join(r.get('channels') or [])}")
    m = run["measured"]
    print(f"  clusters in band: {m['n_clusters_in_band']} / {ev['n_declared']} declared; orphans "
          f"{[round(o['f_hz']/1e9, 4) for o in m['orphans']]}; ambiguous {len(m['ambiguous'])}")
    en = m.get("energy") or {}
    if en:
        print(f"  energy: E0 {en['E0']:.6e} max drift {en['max_drift']:.3e} end {en['drift_at_end']:.3e} "
              f"envelope@end {en['envelope_at_end']:.3e} fired={en['fs1_fired']}")
    print(f"  gates: {ev['gates']} -> {'PASS' if ev['ok'] else 'FAIL'}")


# =============================================================================
# main
# =============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default=None, help="comma-separated subset of " + ",".join(G.ARM_ORDER))
    ap.add_argument("--falsifier", choices=sorted(G.FALSIFIERS), default=None,
                    help="apply a pre-declared defect (runs the uniform control + the defective arm); MUST exit 1")
    ap.add_argument("--smoke", action="store_true",
                    help="uniform + single_band at 1/8 of the record, no energy audit, no gates; exit 0 if finite")
    ap.add_argument("--out-dir", default=None, help=f"artifact directory (default: {G.RESULTS_DIRNAME})")
    ap.add_argument("--tag", default=None, help="write rfx__<tag>.json (diagnostic; never the baseline)")
    ap.add_argument("--no-energy", action="store_true",
                    help="skip the energy witness (diagnostic only; the energy gate then FAILS; requires --tag)")
    a = ap.parse_args(argv)
    if a.no_energy and not (a.tag or a.smoke):
        ap.error("--no-energy is a diagnostic and requires --tag")

    out_dir = a.out_dir or (tempfile.mkdtemp(prefix="cv24_smoke_") if a.smoke else RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    modes = G.declared_modes()
    w_est = G.estimator_floor()

    arms = list(G.ARM_ORDER)
    if a.arm:
        arms = [s.strip() for s in a.arm.split(",") if s.strip()]
        bad = [s for s in arms if s not in G.ARMS]
        if bad:
            ap.error(f"unknown arm(s) {bad}")
    if a.smoke and not a.arm:
        arms = ["uniform", "single_band"]
    fals = G.FALSIFIERS.get(a.falsifier) if a.falsifier else None
    if fals is not None:
        arms = ["uniform", a.falsifier]

    print("=" * 96)
    print(f"CROSSVAL 24 -- graded-z PEC cavity vs EXACT Pozar; arms {arms}; falsifier={a.falsifier}; smoke={a.smoke}")
    print("=" * 96)
    print(f"cavity (a, b, d) = ({G.A_X*1e3:g}, {G.B_Y*1e3:g}, {G.D_Z*1e3:g}) mm; band {G.BAND_HZ[0]/1e9:g}-{G.BAND_HZ[1]/1e9:g} GHz; "
          f"declared modes {[m['name'] for m in modes]}; closest pair {G.closest_pair_hz(modes)[1:]} "
          f"df {G.closest_pair_hz(modes)[0]/1e6:.1f} MHz; W_est {w_est:.1e}")

    doc = {"schema": G.SCHEMA, "case_id": G.CASE_ID, "commit": _staged_commit(),
           "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "falsifier": a.falsifier, "smoke": bool(a.smoke), "tag": a.tag,
           "declared": {"a_m": G.A_X, "b_m": G.B_Y, "d_m": G.D_Z, "band_hz": list(G.BAND_HZ),
                        "modes": [{"name": m["name"], "mnl": list(m["mnl"]), "f_hz": m["f_hz"]} for m in modes],
                        "w_est": w_est, "ratio_cap": G.RATIO_CAP, "r_wall_cells": G.R_WALL_CELLS,
                        "fs1_k": G.FS1_K, "cv14_tol": [G.CV14_TOL_TE101, G.CV14_TOL_HIGHER]},
           "arms": {}, "evaluations": {}}

    control_eval = None
    any_fail = False
    for arm in arms:
        if arm in G.ARMS:
            spec = G.ARMS[arm]
            profile, dxy, lane = G.PROFILES[spec["profile"]], spec["dx"], spec["lane"]
            search = G.BAND_HZ
            swap = False
            declared = profile
        else:
            spec = G.FALSIFIERS[arm]
            profile = (G.FALSIFIER_PROFILES if spec["kind"] == "profile" else G.PROFILES)[spec["profile"]]
            declared = (G.PROFILES[spec["declared_profile"]] if spec.get("declared_profile") else profile)
            dxy, lane = G.DX_COARSE, "nonuniform"
            search = tuple(spec.get("search_band_hz", G.BAND_HZ))
            swap = spec["kind"] == "metric_swap"
            print(f"\n[{arm}] FALSIFIER: {spec['note']}; expected failing gates {spec['expect']}")
        run = run_arm(arm, lane, dxy, profile, smoke=a.smoke, search_band_hz=search, metric_swap=swap,
                      with_energy=not a.no_energy)
        run["declared_profile_mm"] = (np.asarray(declared) * 1e3).tolist()
        ctrl = None if arm == "uniform" else control_eval
        # judged against the DECLARED profile (oracle, lattice, allowance, envelope) and the realized extent
        ev = G.evaluate_arm(run["measured"], declared, dxy, run["dt"], ctrl, modes, search, w_est,
                            realized_d_m=run["realized_d_m"])
        if arm == "uniform":
            control_eval = ev
        if arm in G.FALSIFIERS:
            ev["falsifier"] = dict(G.falsifier_expectation(arm, ev), name=arm)
        doc["arms"][arm] = run
        doc["evaluations"][arm] = ev
        _print_arm(arm, ev, run)
        if arm in G.FALSIFIERS:
            print(f"  falsifier: {ev['falsifier']}")
        if not a.smoke:
            any_fail |= not ev["ok"]

    if a.smoke:
        rc, summary = 0, "SMOKE OK (rig executed at 1/8 record, artifact written, gates NOT evaluated for verdict)"
    elif any_fail:
        rc, summary = 1, "rfx accuracy: FAIL -- at least one gate failed on at least one arm (exit 1)"
    else:
        rc, summary = 0, "ALL CHECKS PASSED -- every gate on every arm (exit 0)"
    if a.falsifier and not a.smoke:
        ok_decl = doc["evaluations"][a.falsifier]["falsifier"]["as_declared"]
        summary += f"  [falsifier {a.falsifier}: {'as pre-declared' if (rc == 1 and ok_decl) else 'NOT as declared'}]"
    doc["verdict"] = {"exit_code": rc, "summary": summary, "arms": arms}
    out_path = os.path.join(out_dir, f"rfx__{a.tag}.json" if a.tag else G.rfx_json_name(a.falsifier, a.arm and a.arm.replace(",", "_")))
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\n  artifact: {out_path}")
    print(f"\n{summary}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
