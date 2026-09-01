"""W1 — Remis dual-cell energy drift, F-S1 evaluation.

1D (P-A) arms: >=1e6 steps on CPU. 3D (P-B): CPU sanity arm (1e4 steps,
NO F-S1 verdict claimed) — the full 1e6-step 3D arm runs on GPU via
vessl_w1_3d.yaml with --arm pb-full.

Usage:
    PYTHONPATH=. python -m validation.research.multiband_nu.w1_energy_drift --arm pa
    PYTHONPATH=. python -m validation.research.multiband_nu.w1_energy_drift --arm pb-sanity
    PYTHONPATH=. python -m validation.research.multiband_nu.w1_energy_drift --arm pb-full  # GPU
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import jax.numpy as jnp

from . import fixtures as fx
from .harness import build_pec_fixture, te10_blob_ex, random_blob_3d, run_energy_audit
from .remis_energy import adjointness_residual

U32 = 2.0 ** -24
FS1_K = 20.0
FS1_SLOPE_MAX = 0.75
FS1_TREND_FLOOR = 50.0 * U32


def evaluate_fs1(steps: np.ndarray, energies: np.ndarray) -> dict:
    """Apply the pre-declared F-S1 criteria. E_0 = first post-projection
    sample (index 1); drift evaluated for n >= 1e4."""
    e0 = energies[1]
    n = steps[1:].astype(float)
    d = np.abs(energies[1:] - e0) / e0
    mask = n >= 1e4
    nm, dm = n[mask], d[mask]
    envelope = FS1_K * U32 * np.sqrt(nm)
    breach = dm > envelope
    out = {
        "E0": float(e0),
        "max_drift": float(d.max()),
        "drift_at_end": float(d[-1]),
        "n_end": int(n[-1]),
        "envelope_at_end": float(FS1_K * U32 * np.sqrt(n[-1])),
        "envelope_breach": bool(breach.any()),
        "n_breach_first": int(nm[breach][0]) if breach.any() else None,
    }
    # trend: RMS drift in 8 log-spaced bins over [1e4, n_end]
    if d.max() > FS1_TREND_FLOOR and nm[-1] / nm[0] > 3:
        edges = np.logspace(np.log10(nm[0]), np.log10(nm[-1]), 9)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (nm >= lo) & (nm <= hi)
            if m.sum() >= 2:
                xs.append(np.sqrt(lo * hi))
                ys.append(np.sqrt(np.mean(dm[m] ** 2)))
        if len(xs) >= 4:
            slope = np.polyfit(np.log10(xs), np.log10(ys), 1)[0]
        else:
            slope = None
        out["trend_slope"] = float(slope) if slope is not None else None
        out["trend_evaluated"] = slope is not None
        out["trend_breach"] = bool(slope is not None and slope > FS1_SLOPE_MAX)
    else:
        out["trend_slope"] = None
        out["trend_evaluated"] = False
        out["trend_breach"] = False
    out["fs1_fired"] = bool(out["envelope_breach"] or out["trend_breach"])
    return out


def run_pa_arm(r: float, variant: str, n_steps: int, sample_every: int) -> dict:
    prof = (np.full(fx.N_FINE * 3 + fx.N_COARSE * 2, fx.DZ_FINE)
            if r == 1.0 else fx.pa_profile(r, variant))
    grid, mats = build_pec_fixture(prof, (fx.A_X, fx.B_Y), fx.DXY)
    ex0 = te10_blob_ex(grid)
    t0 = time.time()
    steps, E = run_energy_audit(grid, mats, {"ex": ex0}, n_steps,
                                sample_every=sample_every)
    wall = time.time() - t0
    res = evaluate_fs1(steps, E)
    res.update({"r": r, "variant": variant, "n_steps": n_steps,
                "grid": list(grid.shape), "dt": float(grid.dt),
                "wallclock_s": wall,
                "drift_series_n": steps[1::4].tolist(),
                "drift_series": ((E[1::4] - E[1]) / E[1]).tolist()})
    return res


def run_pb_arm(r: float, n_steps: int, sample_every: int, claim: bool) -> dict:
    prof = fx.pa_profile(r, "abrupt")
    grid, mats = build_pec_fixture(prof, fx.pb_domain_xy(), fx.DXY)
    adjr = max(adjointness_residual(grid, s) for s in range(2))
    fields = random_blob_3d(grid)
    t0 = time.time()
    steps, E = run_energy_audit(grid, mats, fields, n_steps,
                                sample_every=sample_every,
                                progress_every=max(1, 100000 // sample_every))
    wall = time.time() - t0
    res = evaluate_fs1(steps, E)
    if not claim:
        res["fs1_fired"] = None
        res["note"] = ("CPU SANITY ARM — 1e4 steps, no F-S1 verdict claimed; "
                       "full 1e6-step arm runs on GPU (vessl_w1_3d.yaml)")
    res.update({"r": r, "variant": "abrupt-3d", "n_steps": n_steps,
                "grid": list(grid.shape), "dt": float(grid.dt),
                "adjointness_residual": float(adjr),
                "wallclock_s": wall,
                "drift_series_n": steps[1::4].tolist(),
                "drift_series": ((E[1::4] - E[1]) / E[1]).tolist()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["pa", "pb-sanity", "pb-full"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = []
    if args.arm == "pa":
        arms = ([(1.0, "abrupt"), (1.1, "abrupt"), (1.2, "abrupt"),
                 (1.4, "abrupt"), (1.5, "abrupt"), (2.0, "abrupt"),
                 (1.4, "smooth"), (2.0, "smooth")])
        for r, variant in arms:
            res = run_pa_arm(r, variant, 1_000_000, 1000)
            print(f"P-A r={r} {variant}: drift_end={res['drift_at_end']:+.3e} "
                  f"max={res['max_drift']:.3e} env_end={res['envelope_at_end']:.3e} "
                  f"slope={res['trend_slope']} FIRED={res['fs1_fired']} "
                  f"({res['wallclock_s']:.0f}s)", flush=True)
            results.append(res)
        out = args.out or "validation/research/multiband_nu/results/w1_pa_1d.json"
    elif args.arm == "pb-sanity":
        for r in (1.4, 2.0):
            res = run_pb_arm(r, 10_000, 500, claim=False)
            print(f"P-B sanity r={r}: drift_end={res['drift_at_end']:+.3e} "
                  f"max={res['max_drift']:.3e} adj={res['adjointness_residual']:.2e} "
                  f"({res['wallclock_s']:.0f}s)", flush=True)
            results.append(res)
        out = args.out or "validation/research/multiband_nu/results/w1_pb_sanity.json"
    else:  # pb-full (GPU)
        for r in (1.4, 2.0):
            res = run_pb_arm(r, 1_000_000, 1000, claim=True)
            print(f"P-B FULL r={r}: drift_end={res['drift_at_end']:+.3e} "
                  f"max={res['max_drift']:.3e} slope={res['trend_slope']} "
                  f"FIRED={res['fs1_fired']} ({res['wallclock_s']:.0f}s)",
                  flush=True)
            results.append(res)
        out = args.out or "validation/research/multiband_nu/results/w1_pb_full_gpu.json"

    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
