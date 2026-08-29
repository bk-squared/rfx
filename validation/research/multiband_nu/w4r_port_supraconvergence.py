"""W4 REDESIGN (W4R) — supraconvergence of the P-C resonator observed
through a mode-selective weakly coupled port pair (F-S4, second fixture).

Phase-1 W4 was INCONCLUSIVE (twice). This redesign changes TWO things,
both diagnosed at bring-up and recorded in the note's W4R section:

1. Mode-selective port (the phase-1 review recommendation): the phase-1
   reading was "near-degenerate lines whose dominance flips". The
   symmetry-selective port below isolates a single dominant line
   (dominance = inf at every bring-up scale).
2. Knife-edge-free drawing (the actual root cause, found once the port
   isolated the line and it STILL wandered non-monotonically): phase-1
   drew box corners exactly ON node planes -- the documented worst case
   of the f32 half-open Box rasterization (see build_sim) -- so the
   realized trace span flipped between 13.5-dx and 13.5-2dx across
   scales, an O(dx) electrical-length lottery common to both arms.
   Additionally, without subpixel smoothing the dielectric staircase
   makes the frequency observable ~1st order (bring-up: 0.63 GHz drift
   over the ladder), which would invalidate any 2nd-order fit; the
   ladder therefore runs with subpixel_smoothing=True (NU-validated,
   tests/test_subpixel_nonuniform.py).

Port design:

* The P-C geometry (physically unchanged) has exact discrete mirror
  symmetry about
  x = 13.5 mm and y = 11.25 mm (all edges on the 2.25 mm common grid).
* Port = ANTI-symmetric Ez pair under the two trace ends,
  (6.75, 11.25, 0.75) mm at +1 A and (20.25, 11.25, 0.75) mm at -1 A
  (amplitude_kind='current'), both on the y-mirror plane. This excites
  only the x-ODD / y-EVEN symmetry class -- the class of the trace's
  half-wave resonance (voltage antinodes of opposite sign at the two
  open ends). Weak coupling = soft source ring-down (no resistive load).
* Probe: Ez at (18.0, 11.25, 0.75) mm (off the x-center node line of the
  odd mode, on the alignment lattice).

Observable and pre-declared rules: see the design-note W4R section
(committed BEFORE any ladder measurement).

Usage:
    PYTHONPATH=. python -m validation.research.multiband_nu.w4r_port_supraconvergence [--diagnose]

--diagnose runs the bring-up spectrum scans only (instrument-validity
data, recorded in the note; no falsifier data produced).
"""

from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np

from rfx import Simulation, Box, GaussianPulse

# Absolute (not relative) import: this module is one of the two in
# this package that construct a `Simulation`, so the #737 example-
# fidelity gate classifies it 'audited' and LOADS it by file path
# (`tests/_example_fidelity_lib.load_module`), where a relative
# import has no parent package. Absolute import keeps both entry
# points working: `python -m validation.research.multiband_nu.<mod>`
# from the repo root, and the gate's by-path load.
from validation.research.multiband_nu import fixtures as fx

# --- W4R instrument (frozen in the note's W4R section) -----------------
BAND = (4.0e9, 6.5e9)     # analysis band containing the target line
F_MAX = 12e9
T_TOTAL = 20e-9           # equal physical ring-down window for every arm
C0 = 299792458.0
Q_MIN = 30.0
DOMINANCE_MIN = 10.0      # target amplitude / any other in-band line
MATCH_GUARD = 0.05        # 5 % frequency-proximity guard vs reference
SRC_P = (6.75e-3, 11.25e-3, 0.75e-3)
SRC_M = (20.25e-3, 11.25e-3, 0.75e-3)
PRB = (18.0e-3, 11.25e-3, 0.75e-3)
WAVEFORM = dict(f0=6e9, bandwidth=0.9)

# --- W4R ladder (frozen in the note's W4R section) ---------------------
SCALES = (0.5, 0.6, 0.75, 1.0, 1.5, 3.0)  # lattice-valid; see note C7
REF_SCALE = 0.25          # ratio 2 below s_min -> Richardson divisor 3
E_FLOOR_HZ = 18e6         # 3x the measured bring-up wobble class (6 MHz)


def n_steps_for(scale: float, dz_min: float) -> int:
    dx = fx.PC_DX0 * scale
    dt = 0.99 / (C0 * np.sqrt(2.0 / dx ** 2 + 1.0 / dz_min ** 2))
    return int(round(T_TOTAL / dt))


def build_sim(scale: float, dz_profile: np.ndarray,
              antisym: bool = True) -> Simulation:
    """P-C geometry as phase-1 W4 but with KNIFE-EDGE-FREE drawing.

    Bring-up finding (recorded in the note's W4R section): the phase-1
    fixture drew every box corner exactly ON node planes — the
    documented worst case of the Box rasterization convention
    (`rfx/geometry/csg.py`: half-open [lo, hi) over nodes, f32
    comparisons). The realized trace span measured per scale flipped
    erratically between 13.5-dx and 13.5-2dx (x) and 4.5-dx / 4.5 (y):
    an O(dx) electrical-length lottery, common to both arms — the true
    root cause of the phase-1 non-monotonic line wander.

    Repair: the PEC trace is drawn with HALF-CELL margins so the zeroed
    node set spans exactly [6.75, 20.25] x [9.0, 13.5] x [1.5, 3.0] mm
    at every scale (nodes strictly inside the box, no knife edges).
    Dielectric boxes keep their physical corners; the ladder runs with
    subpixel smoothing so their voxel assignment is volume-weighted
    (continuous in coordinate rounding, no whole-cell jumps)."""
    total_h = (fx.PC_H_SUB + fx.PC_H_TRACE_BAND + fx.PC_AIR1
               + fx.PC_H_UPPER + fx.PC_AIR2)
    assert abs(dz_profile.sum() - total_h) < 1e-9
    dx = fx.PC_DX0 * scale
    dzf = fx.PC_DZF0 * scale
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
    sim.add(Box((6.75e-3 - dx / 2, 9.0e-3 - dx / 2, fx.PC_H_SUB - dzf / 2),
                (20.25e-3 + dx / 2, 13.5e-3 + dx / 2,
                 fx.PC_H_SUB + fx.PC_H_TRACE_BAND + dzf / 2)),
            material="pec")
    sim.add_source(SRC_P, "ez",
                   waveform=GaussianPulse(amplitude=+1.0, **WAVEFORM),
                   amplitude_kind="current")
    sim.add_source(SRC_M, "ez",
                   waveform=GaussianPulse(
                       amplitude=(-1.0 if antisym else +1.0), **WAVEFORM),
                   amplitude_kind="current")
    sim.add_probe(PRB, "ez")
    return sim


def modes_of(result):
    modes = result.find_resonances(freq_range=(3e9, 9e9))
    return sorted([m for m in modes if abs(m.Q) > Q_MIN],
                  key=lambda m: m.freq)


def target_line(modes) -> tuple[float, float, dict]:
    """Largest-|amplitude| line in BAND + isolation diagnostics.

    Returns (f_target, dominance, info). dominance = |amp_target| /
    max other in-band |amp| (inf when alone in band)."""
    in_band = [m for m in modes if BAND[0] <= m.freq <= BAND[1]]
    if not in_band:
        return float("nan"), 0.0, {"in_band": []}
    top = max(in_band, key=lambda m: abs(m.amplitude))
    others = [abs(m.amplitude) for m in in_band if m is not top]
    dom = (abs(top.amplitude) / max(others)) if others else float("inf")
    return float(top.freq), float(dom), {
        "in_band": [(float(m.freq), float(m.Q), float(abs(m.amplitude)))
                    for m in in_band]}


SUBPIXEL = True            # frozen in the note's W4R section (bring-up:
                           # without it the dielectric staircase makes the
                           # observable ~1st order; with it the uniform arm
                           # converges monotonically at the 2nd-order class)


def measure(scale: float, multiband: bool,
            subpixel: bool = None) -> dict:
    prof = (fx.pc_dz_profile_sym(scale) if multiband
            else fx.pc_uniform_profile(scale))
    t0 = time.time()
    sim = build_sim(scale, prof)
    sim.preflight()
    n_steps = n_steps_for(scale, float(prof.min()))
    result = sim.run(n_steps=n_steps,
                     subpixel_smoothing=(SUBPIXEL if subpixel is None
                                         else subpixel))
    wall = time.time() - t0
    modes = modes_of(result)
    f_t, dom, info = target_line(modes)
    grid_cells = (int(round(fx.PC_A / (fx.PC_DX0 * scale)))
                  * int(round(fx.PC_B / (fx.PC_DX0 * scale))) * len(prof))
    return {
        "scale": scale, "multiband": multiband,
        "nz": len(prof), "cells": grid_cells, "n_steps": n_steps,
        "modes": [(float(m.freq), float(m.Q), float(abs(m.amplitude)))
                  for m in modes[:8]],
        "f_target": f_t, "dominance": dom, "in_band": info["in_band"],
        "wallclock_s": wall,
        "peak_rss_mb": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }


def diagnose():
    """Bring-up: spectrum of the anti-symmetric port at two cheap scales
    (both arms) + the symmetric-pair control (x-even class) — shows the
    class separation that motivates the instrument. Discarded data:
    informs the instrument only, judges nothing."""
    out = {"diagnostic": True, "arms": []}
    for antisym in (True, False):
        for mb in (False, True):
            for s in (0.75, 1.0):
                prof = (fx.pc_dz_profile_sym(s) if mb
                        else fx.pc_uniform_profile(s))
                sim = build_sim(s, prof, antisym=antisym)
                sim.preflight()
                res = sim.run(n_steps=n_steps_for(s, float(prof.min())))
                modes = modes_of(res)
                f_t, dom, info = target_line(modes)
                row = {"antisym": antisym, "multiband": mb, "scale": s,
                       "modes": [(float(m.freq), float(m.Q),
                                  float(abs(m.amplitude)))
                                 for m in modes[:8]],
                       "f_target": f_t, "dominance": dom}
                out["arms"].append(row)
                print(f"antisym={antisym} mb={mb} s={s}: "
                      + " | ".join(f"{m[0]/1e9:.4f}GHz Q={m[1]:.0f} "
                                   f"A={m[2]:.3g}" for m in row["modes"])
                      + f"  -> target {f_t/1e9:.4f} dom={dom:.1f}",
                      flush=True)
    path = ("validation/research/multiband_nu/results/"
            "w4r_diagnostic_bringup.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)


def main():
    out = {"arms": []}
    ref = measure(REF_SCALE, multiband=False)
    out["reference"] = ref
    f_ref = ref["f_target"]
    print(f"reference s={REF_SCALE}: f_target={f_ref/1e9:.6f} GHz "
          f"dom={ref['dominance']:.1f} wall={ref['wallclock_s']:.0f}s",
          flush=True)

    rows = {}
    for mb in (False, True):
        for s in SCALES:
            e = measure(s, mb)
            f = e["f_target"]
            valid = (np.isfinite(f)
                     and abs(f - f_ref) <= MATCH_GUARD * f_ref
                     and e["dominance"] >= DOMINANCE_MIN)
            e["valid"] = bool(valid)
            e["err_hz"] = abs(f - f_ref) if valid else float("nan")
            rows[(mb, s)] = e
            out["arms"].append(e)
            print(f"{'MB' if mb else 'UC'} s={s}: f={f/1e9:.6f} GHz "
                  f"err={e['err_hz']/1e6:.3f} MHz dom={e['dominance']:.1f} "
                  f"valid={valid} cells={e['cells']} "
                  f"wall={e['wallclock_s']:.0f}s", flush=True)

    # pre-declared fit (note W4R section): u_ref by Richardson p=2 from
    # the finest uniform arm, ladder/reference ratio 2 -> divisor 3;
    # points below max(3*u_ref, E_FLOOR_HZ) are excluded from the fit.
    s_min = min(SCALES)
    ratio = s_min / REF_SCALE
    u_ref = rows[(False, s_min)]["err_hz"] / (ratio ** 2 - 1.0)
    out["u_ref_hz"] = u_ref
    cut = max(3.0 * u_ref, E_FLOOR_HZ) if np.isfinite(u_ref) else E_FLOOR_HZ
    out["fit_cut_hz"] = cut

    def fit_order(mb: bool):
        pts = [(fx.PC_DZF0 * s, rows[(mb, s)]["err_hz"]) for s in SCALES
               if rows[(mb, s)]["valid"] and rows[(mb, s)]["err_hz"] >= cut]
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

    # Pre-declared W4R verdict structure (note W4R section, derived
    # fresh from theory + the instrument-noise model):
    #   fixture valid   iff 1.7 <= p_uc <= 2.6
    #   F-S4 fires      iff fixture valid AND
    #                       (p_mb < 1.5 OR p_mb < p_uc - 0.4)  [order LOSS]
    #   anomaly A4      iff fixture valid AND p_mb > p_uc + 0.4
    #                       (blocks promotion, filed for investigation --
    #                        grading cannot RAISE the shared fixture order)
    anomaly = None
    if p_uc is None or p_mb is None:
        verdict = "INCONCLUSIVE (fewer than 3 valid fit points >= cut)"
        fired = None
    elif p_uc < 1.7 or p_uc > 2.6:
        verdict = ("FIXTURE-INVALID (p_uc %.2f outside [1.7, 2.6]: "
                   "singularity/reference-limited or pre-asymptotic for "
                   "BOTH arms; not a multiband fault)" % p_uc)
        fired = None
    else:
        fired = bool(p_mb < 1.5 or p_mb < p_uc - 0.4)
        anomaly = bool(p_mb > p_uc + 0.4)
        verdict = (f"p_uc={p_uc:.2f} p_mb={p_mb:.2f} fired={fired}"
                   + (" ANOMALY(p_mb>p_uc+0.4)" if anomaly else ""))
    out["fs4_fired"] = fired
    out["anomaly_a4"] = anomaly
    out["verdict"] = verdict
    print("F-S4 (W4R):", verdict, flush=True)

    path = ("validation/research/multiband_nu/results/"
            "w4r_supraconvergence.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    if "--diagnose" in sys.argv:
        diagnose()
    else:
        main()
