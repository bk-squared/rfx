"""W4 — supraconvergence of the P-C microstrip-type resonator (F-S4)
plus cost actuals (documentation, no claims).

Arms: multiband P-C at s in {1,1.5,2,3}; uniform-fine control at the same
scales; reference = uniform-fine at s=2/3. Observable: lowest resonance in
3-9 GHz (harminv), matched to the reference's lowest mode. Criteria are
pre-declared in the note §4.

Usage:
    PYTHONPATH=. python -m validation.research.multiband_nu.w4_supraconvergence
"""

from __future__ import annotations

import json
import resource
import time

import numpy as np

from rfx import Simulation, Box, GaussianPulse

from . import fixtures as fx

FREQ_RANGE = (3e9, 9e9)
T_TOTAL = 20e-9           # equal physical ring-down window for every arm
# (C4: 4.5 ns gave ~0.2 GHz line resolution — too coarse for the dense
# mode spectrum; single-line fits wandered 0.6 GHz between arms)
F_MAX = 12e9
C0 = 299792458.0


def n_steps_for(scale: float, dz_min: float) -> int:
    dx = fx.PC_DX0 * scale
    dt = 0.99 / (C0 * np.sqrt(2.0 / dx ** 2 + 1.0 / dz_min ** 2))
    return int(round(T_TOTAL / dt))


def build_sim(scale: float, dz_profile: np.ndarray) -> Simulation:
    total_h = fx.PC_H_SUB + fx.PC_H_TRACE_BAND + fx.PC_AIR1 + fx.PC_H_UPPER + fx.PC_AIR2
    assert abs(dz_profile.sum() - total_h) < 1e-9
    dx = fx.PC_DX0 * scale
    assert abs(round(fx.PC_A / dx) * dx - fx.PC_A) < 1e-9
    sim = Simulation(
        freq_max=F_MAX, domain=(fx.PC_A, fx.PC_B, total_h),
        dx=dx, boundary="pec",
        dz_profile=dz_profile,
    )
    sim.add_material("sub", eps_r=fx.PC_EPS_SUB, sigma=0.0)
    sim.add_material("upper", eps_r=fx.PC_EPS_UPPER, sigma=0.0)
    sim.add(Box((0, 0, 0), (fx.PC_A, fx.PC_B, fx.PC_H_SUB)), material="sub")
    z_up0 = fx.PC_H_SUB + fx.PC_H_TRACE_BAND + fx.PC_AIR1
    sim.add(Box((0, 0, z_up0), (fx.PC_A, fx.PC_B, z_up0 + fx.PC_H_UPPER)),
            material="upper")
    # PEC trace: fills the 1.5 mm trace band exactly (C4: a one-fine-cell
    # thickness scaled WITH s, changing the physical resonator per arm;
    # 1.5 mm is an exact multiple of every dz_fine incl. the reference).
    # Edges on the 2.25 mm common grid (13.5 x 4.5 mm).
    sim.add(Box((6.75e-3, 9.0e-3, fx.PC_H_SUB),
                (20.25e-3, 13.5e-3, fx.PC_H_SUB + fx.PC_H_TRACE_BAND)),
            material="pec")
    # source under the trace edge / probe off-centre (positions on the
    # common 2.25 mm grid; z snapping tolerated — frequency observable)
    sim.add_source((6.75e-3, 11.25e-3, 0.75e-3), "ez",
                   waveform=GaussianPulse(f0=6e9, bandwidth=0.9))
    sim.add_probe((18.0e-3, 11.25e-3, 0.75e-3), "ez")
    return sim


def measure(scale: float, multiband: bool) -> dict:
    prof = (fx.pc_dz_profile_sym(scale) if multiband
            else fx.pc_uniform_profile(scale))
    t0 = time.time()
    sim = build_sim(scale, prof)
    sim.preflight()
    n_steps = n_steps_for(scale, float(prof.min()))
    result = sim.run(n_steps=n_steps)
    wall = time.time() - t0
    modes = result.find_resonances(freq_range=FREQ_RANGE)
    modes = sorted([m for m in modes if abs(m.Q) > 30], key=lambda m: m.freq)
    grid_cells = int(round(fx.PC_A / (fx.PC_DX0 * scale))) \
        * int(round(fx.PC_B / (fx.PC_DX0 * scale))) * len(prof)
    return {
        "scale": scale, "multiband": multiband,
        "nz": len(prof), "cells": grid_cells, "n_steps": n_steps,
        "modes": [(float(m.freq), float(m.Q), float(abs(m.amplitude))) for m in modes[:6]],
        "wallclock_s": wall,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }


def lowest_matched(entry: dict, f_target: float) -> float:
    freqs = [m[0] for m in entry["modes"]]
    if not freqs:
        return float("nan")
    f = min(freqs, key=lambda f: abs(f - f_target))
    if abs(f - f_target) > 0.05 * f_target:   # C4 match guard
        return float("nan")
    return f


def main():
    out = {"arms": []}
    # Correction C3 (2026-08-29): reference scale 2/3 -> 3/4. At s=2/3
    # dx=0.5mm does not divide the 2.25mm alignment grid (6.75/0.5=13.5:
    # trace edge lands mid-cell — preflight #703-class advisory fired,
    # violating the fixture's declared alignment invariant). s=3/4 gives
    # dx=0.5625 (2.25/0.5625=4) and dz_fine=0.1875 (1.5/0.1875=8), both
    # exact. The Richardson divisor follows the same declared principle
    # with the corrected ratio 4/3: (4/3)^2 - 1 = 7/9.
    ref = measure(fx.PC_REF_SCALE, multiband=False)  # C5
    out["reference"] = ref
    f_ref_low = min(m[0] for m in ref["modes"]) if ref["modes"] else float("nan")
    print(f"reference s={fx.PC_REF_SCALE}: modes {[(f'{m[0]/1e9:.4f}GHz', f'{m[1]:.0f}') for m in ref['modes']]}",
          flush=True)

    scales = list(fx.PC_SCALES)
    rows = {}
    for mb in (False, True):
        for s in scales:
            e = measure(s, mb)
            f = lowest_matched(e, f_ref_low)
            e["f_matched"] = f
            e["err_hz"] = abs(f - f_ref_low)
            rows[(mb, s)] = e
            out["arms"].append(e)
            print(f"{'MB' if mb else 'UC'} s={s}: f={f/1e9:.6f} GHz "
                  f"err={e['err_hz']/1e6:.3f} MHz cells={e['cells']} "
                  f"wall={e['wallclock_s']:.0f}s", flush=True)

    # pre-declared fit: u_ref from Richardson (p=2, ratio 1.5 between
    # s=1 uniform control and the s=2/3 reference)
    s_min = min(scales)
    ratio = s_min / fx.PC_REF_SCALE          # 4/3 with the C5 ladder
    u_ref = rows[(False, s_min)]["err_hz"] / (ratio ** 2 - 1.0)
    out["u_ref_hz"] = u_ref

    def fit_order(mb: bool):
        pts = [(fx.PC_DZF0 * s, rows[(mb, s)]["err_hz"]) for s in scales
               if rows[(mb, s)]["err_hz"] >= 3 * u_ref]
        if len(pts) < 3:
            return None, pts
        h = np.log10([p[0] for p in pts])
        e = np.log10([p[1] for p in pts])
        return float(np.polyfit(h, e, 1)[0]), pts

    p_uc, pts_uc = fit_order(False)
    p_mb, pts_mb = fit_order(True)
    out["p_uc"] = p_uc
    out["p_mb"] = p_mb
    out["n_fit_points"] = {"uc": len(pts_uc), "mb": len(pts_mb)}

    if p_uc is None or p_mb is None:
        verdict = "INCONCLUSIVE (fewer than 3 fit points above 3*u_ref)"
        fired = None
    elif p_uc < 1.7:
        verdict = ("FIXTURE-INVALID (p_uc %.2f < 1.7: reference/singularity-"
                   "limited for BOTH arms; not a multiband fault)" % p_uc)
        fired = None
    else:
        fired = bool(p_mb < 1.5 or p_mb > 2.6 or p_mb < p_uc - 0.4)
        verdict = f"p_uc={p_uc:.2f} p_mb={p_mb:.2f} fired={fired}"
    out["fs4_fired"] = fired
    out["verdict"] = verdict
    print("F-S4:", verdict, flush=True)

    with open("validation/research/multiband_nu/results/w4_supraconvergence.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote results/w4_supraconvergence.json")


if __name__ == "__main__":
    main()
