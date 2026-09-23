"""M6: the resonance moves with the patch length, and AD knows how fast.

A 6 x 5 mm patch on a 1.5 mm eps_r 3.38 substrate over a ground plane, fed by
a 50 ohm wire port through the substrate 1 mm off centre.  The patch's
resonant length is made longer by STRETCHING the x cells between its two node
lines: the 8 cells inside the +x edge each grow by ``delta/8`` and the 8 just
outside shrink by the same total, so the domain keeps its length, the metal
never moves off the lattice, and there is no metal inside a cell anywhere.

What is measured
----------------
``d f_r / d delta`` at ``delta = dx/2``, where ``f_r`` is the S11 minimum
read as the vertex of a parabola through three FIXED frequency bins (fixed so
the map ``delta -> f_r`` is smooth and can be differentiated).  Three routes
to the same number:

* **AD** — ``jax.jvp`` straight through ``delta -> dx_profile -> the 50 ohm
  port -> S11 -> the parabola vertex``.  One tangent: forward mode, because
  there is one design variable and reverse mode would tape every step.
* **central differences** on the same function, laddered over h so the
  difference's own truncation is visible;
* the **published slope** of the same fixture measured by central differences
  alone, ``-1.2088e12 Hz/m``, converged over h = dx/8 ... dx/32 and unchanged
  at twice the record length.

The record-length witness repeats the AD number at 2x the steps: a resonance
read off a record that has not settled moves with the record, and a
derivative of it would move too.

What this board is, and what it is not for
------------------------------------------
A COARSE board, kept coarse on purpose so the number is comparable with the
published ladder measured on the same mesh.  0.5 mm cells put 25 cells per
substrate wavelength at the top of the band (13 GHz, eps_r 3.38), 33 at the
bottom and 28 at the resonance, but only 10.9 at the declared freq_max of
30 GHz -- which is the cells-per-wavelength number preflight reports -- and
the substrate itself is 3 cells thick.  Nothing here is a statement about
how accurate f_r is: the absolute resonance carries the coarse mesh's own
staircase error, and no mesh-refinement arm is run.  What IS measured is the
DERIVATIVE at that mesh, against a central difference through the same
function and against the published slope of the same board.

The step
--------
``dt`` is pinned to one concrete value, ``0.9 x`` the Courant step of the
smallest cell the sweep reaches (0.375 mm), through ``Simulation(dt=...)``.
Unpinned, the grid derives its step from the smallest cell, so every delta
would run at its own step and the step change would ride into ``f_r``
alongside the geometry change.  ``dt_min_cell=`` is the declared floor the
pin is checked against on the traced axis, which carries no host cell size.

Both conductors are declared by NODE INDEX (``add_pinned_sheet``), so the
same call realizes the same node lines at every deformation.  The realized
nodes and the realized patch length are printed and asserted before any f_r
is read off the mesh.

No verdict sentences are written here; the script prints counts and numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# this checkout first: a venv may carry an installed rfx from another tree
sys.path.insert(0, str(HERE.parents[2]))

C0 = 299792458.0

FIXTURE = dict(
    dx=0.5e-3, t_sub=1.5e-3, patch_x=6.0e-3, patch_y=5.0e-3,
    gp=10.0e-3, domain_xy=18.0e-3, air=4.0e-3, eps_r=3.38,
    tan_delta=1.0e-3, f0=1.2e10, feed_offset_x=-1.0e-3,
    band=(10.0e9, 13.0e9, 20e6), cpml=6,
)

DX = FIXTURE["dx"]
CTR = FIXTURE["domain_xy"] / 2.0
N_X = int(round(FIXTURE["domain_xy"] / DX))              # 36
N_Y = N_X
#: z is 8 air + 3 substrate + 8 air cells of dx.
N_Z_AIR = int(round(FIXTURE["air"] / DX))                # 8
N_Z_SUB = int(round(FIXTURE["t_sub"] / DX))              # 3
N_Z = 2 * N_Z_AIR + N_Z_SUB                              # 19
K_GND, K_PATCH = N_Z_AIR, N_Z_AIR + N_Z_SUB              # 8, 11
LZ = N_Z * DX

#: the patch's two x node lines, and its two y node lines, as INDICES
I_LO = int(round((CTR - FIXTURE["patch_x"] / 2) / DX))   # 12
I_HI = int(round((CTR + FIXTURE["patch_x"] / 2) / DX))   # 24
J_LO = int(round((CTR - FIXTURE["patch_y"] / 2) / DX))   # 13
J_HI = int(round((CTR + FIXTURE["patch_y"] / 2) / DX))   # 23
#: the ground plane's node lines
G_LO = int(round((CTR - FIXTURE["gp"] / 2) / DX))        # 8
G_HI = int(round((CTR + FIXTURE["gp"] / 2) / DX))        # 28
#: the feed node. Cells 0..15 never move in this window, so this node sits at
#: the same physical x on the nominal and on every deformed mesh.
I_PORT = int(round((CTR + FIXTURE["feed_offset_x"]) / DX))   # 16

#: the probe node, outside every deformed cell (the stretch and the shrink
#: both lie below it, so it sits at the same physical x on every mesh)
PROBE_NODE = 32

N_IN = 8                    # cells inside the +x edge that stretch
OUT_CELLS = list(range(I_HI, I_HI + N_IN))   # cells outside that shrink

#: the smallest cell the published sweep reaches (the narrow 4/4 window at
#: delta = dx), kept so the step is the one the reference ladder used
D_MIN_PIN = DX * (1.0 - 1.0 / 4.0)
DT_PIN = 0.9 / (C0 * np.sqrt(1.0 / D_MIN_PIN ** 2 + 2.0 / DX ** 2))
DT_UNIFORM = 0.99 * DX / (C0 * np.sqrt(3.0))
N_REF_STEPS = 6000
N_STEPS = int(round(N_REF_STEPS * DT_UNIFORM / DT_PIN))      # 7406

DELTA_0 = DX / 2.0
FD_H_OVER_DX = [1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256]

#: the published central-difference slope of this fixture at delta = dx/2
REFERENCE_SLOPE_HZ_PER_M = -1.2088e12


def _is_traced(v):
    from rfx.core.jax_utils import is_tracer
    return is_tracer(v)


def deformation_profile(delta):
    """The x cell sizes for a patch lengthened by ``delta``."""
    if _is_traced(delta):
        import jax.numpy as jnp
        d = jnp.full((N_X,), DX)
        d = d.at[I_HI - N_IN:I_HI].add(delta / N_IN)
        d = d.at[np.asarray(OUT_CELLS)].add(-delta / len(OUT_CELLS))
        return d.astype(jnp.float32)
    d = np.full((N_X,), DX, dtype=np.float64)
    d[I_HI - N_IN:I_HI] += float(delta) / N_IN
    d[np.asarray(OUT_CELLS, dtype=int)] -= float(delta) / len(OUT_CELLS)
    return d


def node_positions(profile):
    return np.concatenate([[0.0], np.cumsum(np.asarray(profile,
                                                       dtype=np.float64))])


def describe_profile(delta):
    """Realized values, asserted before any f_r is read off this mesh."""
    d = deformation_profile(delta)
    p = node_positions(d)
    ratios = d[1:] / d[:-1]
    row = {
        "delta_m": float(delta),
        "declared_L_m": float(FIXTURE["patch_x"] + delta),
        "realized_L_m": float(p[I_HI] - p[I_LO]),
        "patch_node_indices": [I_LO, I_HI],
        "patch_y_node_indices": [J_LO, J_HI],
        "ground_node_indices": [G_LO, G_HI],
        "port_node_index": I_PORT,
        "port_node_x_m": float(p[I_PORT]),
        "nominal_port_x_m": float(I_PORT * DX),
        "max_ratio": float(np.max(np.maximum(ratios, 1.0 / ratios))),
        "min_cell_m": float(d.min()), "max_cell_m": float(d.max()),
        "total_m": float(d.sum()),
        "declared_total_m": float(FIXTURE["domain_xy"]),
    }
    err = []
    if abs(row["realized_L_m"] - row["declared_L_m"]) > 1e-12:
        err.append(f"L realizes {row['realized_L_m']:.9g} against a declared "
                   f"{row['declared_L_m']:.9g}")
    if abs(row["total_m"] - row["declared_total_m"]) > 1e-12:
        err.append("the shrink does not return what the stretch took")
    if abs(row["port_node_x_m"] - row["nominal_port_x_m"]) > 1e-15:
        err.append(f"the feed node moved to {row['port_node_x_m']:.9g} from "
                   f"{row['nominal_port_x_m']:.9g} — the traced route resolves"
                   " it on the nominal mesh and the two would name different "
                   "nodes")
    if abs(float(d[0]) - DX) > 1e-15 or abs(float(d[-1]) - DX) > 1e-15:
        err.append("the first/last profile cell must stay at the boundary dx")
    if err:
        raise ValueError(f"delta={float(delta):.6g}: " + "; ".join(err))
    return row


# ---------------------------------------------------------------------------
# the board
# ---------------------------------------------------------------------------

def build(delta, *, pin_dt=True):
    from rfx import Simulation
    from rfx.geometry import Box
    from rfx.sources import GaussianPulse

    c = FIXTURE
    kw = dict(dt=float(DT_PIN), dt_min_cell=float(D_MIN_PIN)) if pin_dt else {}
    sim = Simulation(freq_max=3.0e10,
                     domain=(c["domain_xy"], c["domain_xy"], LZ), dx=DX,
                     boundary="cpml", cpml_layers=c["cpml"],
                     dx_profile=deformation_profile(delta), **kw)
    z_g, z_p = K_GND * DX, K_PATCH * DX
    sigma = (2 * np.pi * c["f0"] * 8.8541878128e-12 * c["eps_r"]
             * c["tan_delta"])
    sim.add_material("sub", eps_r=c["eps_r"], sigma=float(sigma))
    g = c["gp"] / 2
    sim.add_pinned_sheet(plane_index=K_GND, i_range=(G_LO, G_HI),
                         j_range=(G_LO, G_HI), name="ground")
    sim.add(Box((CTR - g, CTR - g, z_g), (CTR + g, CTR + g, z_p)),
            material="sub")
    sim.add_pinned_sheet(plane_index=K_PATCH, i_range=(I_LO, I_HI),
                         j_range=(J_LO, J_HI), name="patch")
    sim.add_port(position=(I_PORT * DX, CTR, z_g), component="ez",
                 impedance=50.0, extent=z_p - z_g,
                 waveform=GaussianPulse(f0=c["f0"], bandwidth=0.9))
    # Node 32 sits at 16 mm on every mesh in this window (the stretch and the
    # shrink both lie below it), so the settling witness reads the same point
    # of the board at every delta.
    sim.add_probe(position=(PROBE_NODE * DX, CTR, 0.5 * (z_g + z_p)),
                  component="ez")
    return sim


def band_freqs():
    lo, hi, df = FIXTURE["band"]
    return np.arange(lo, hi + 0.5 * df, df)


def s11(delta, freqs, n_steps, *, pin_dt=True):
    import jax.numpy as jnp
    from rfx.runners.nonuniform import run_nonuniform_path
    r = run_nonuniform_path(build(delta, pin_dt=pin_dt), n_steps=int(n_steps),
                            compute_s_params=True, s_param_freqs=freqs)
    return jnp.asarray(r.s_params).reshape(-1), r


def f_min_parabolic(freqs, s):
    """The S11 minimum by a parabola in dB through the three bins around it.
    Returns (f_r, bin, used_parabola)."""
    mag = np.abs(np.asarray(s))
    i = int(np.argmin(mag))
    if i == 0 or i == mag.size - 1:
        return float(freqs[i]), i, False
    y0, y1, y2 = 20 * np.log10(np.maximum(mag[i - 1:i + 2], 1e-300))
    den = y0 - 2 * y1 + y2
    d = 0.0 if den == 0 else float(np.clip(0.5 * (y0 - y2) / den, -1.0, 1.0))
    return float(freqs[i] + d * float(freqs[1] - freqs[0])), i, True


def f_r_fixed_bin(s, freqs, i):
    """The same parabola with the bin index HELD, so it is differentiable."""
    import jax.numpy as jnp
    y = 20.0 * jnp.log10(jnp.abs(s[i - 1:i + 2]) + 1e-300)
    den = y[0] - 2 * y[1] + y[2]
    d = 0.5 * (y[0] - y[2]) / den
    return freqs[i] + d * (freqs[1] - freqs[0])


def settling_db(ts):
    e = np.abs(np.asarray(ts).ravel())
    if e.size == 0 or e.max() <= 0:
        return None
    return float(20.0 * np.log10(max(e[int(e.size * 0.875):].max(), 1e-300)
                                 / e.max()))


# ---------------------------------------------------------------------------

def run(out_dir, *, smoke=False):
    import jax
    import jax.numpy as jnp

    freqs = band_freqs()
    n_steps = 400 if smoke else N_STEPS
    rec = {"kind": "metal_edge_M6_wire_port_ad", "fixture": FIXTURE,
           "band_hz": list(FIXTURE["band"]),
           "window": {"n_in": N_IN, "out_cells": OUT_CELLS},
           "patch_node_indices": [I_LO, I_HI],
           "patch_y_node_indices": [J_LO, J_HI],
           "ground_node_indices": [G_LO, G_HI],
           "port_node_index": I_PORT,
           "dt_pinned_s": float(DT_PIN), "dt_min_cell_m": float(D_MIN_PIN),
           "n_steps": int(n_steps), "smoke": bool(smoke),
           "delta_0_m": float(DELTA_0),
           "reference_slope_hz_per_m": REFERENCE_SLOPE_HZ_PER_M,
           "profiles": [], "base": None, "ad": None, "fd": [],
           "record_witness": None, "status": "started"}
    t_all = time.time()

    def dump():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "m6_wire_port_ad.json").write_text(
            json.dumps(rec, indent=2, default=str))

    # ---- 0. what the mesh realizes, before any solve ----------------------
    print("=== the deformation window ===", flush=True)
    for d in (0.0, DELTA_0, DX):
        info = describe_profile(d)
        rec["profiles"].append(info)
        print(f"  delta {d * 1e3:.4f} mm  L {info['realized_L_m'] * 1e3:.4f} mm"
              f"  cells {info['min_cell_m'] * 1e3:.4f}.."
              f"{info['max_cell_m'] * 1e3:.4f} mm  max ratio "
              f"{info['max_ratio']:.4f}  feed node {info['port_node_index']} "
              f"at {info['port_node_x_m'] * 1e3:.4f} mm", flush=True)
    print(f"  patch nodes x {I_LO}..{I_HI}, y {J_LO}..{J_HI}, plane {K_PATCH}"
          f"; ground nodes {G_LO}..{G_HI}, plane {K_GND}", flush=True)
    print(f"  dt pinned to {DT_PIN * 1e12:.5f} ps, {n_steps} steps "
          f"({n_steps * DT_PIN * 1e9:.4f} ns)", flush=True)
    dump()

    # ---- 1. the base point, and which bin the dip sits in ------------------
    t0 = time.time()
    s0, r0 = s11(DELTA_0, freqs, n_steps)
    s0_np = np.asarray(s0)
    fr0, bin0, parab = f_min_parabolic(freqs, s0_np)
    ts = np.asarray(r0.time_series).ravel() if r0.time_series is not None \
        else np.zeros(1)
    rec["base"] = {
        "delta_m": float(DELTA_0), "f_r_hz": fr0, "f_r_bin": int(bin0),
        "parabolic": bool(parab), "dt_s": float(r0.dt),
        "s11_min_db": float(20 * np.log10(max(np.abs(s0_np).min(), 1e-300))),
        "s11_max_abs": float(np.abs(s0_np).max()),
        "settling_db": settling_db(ts),
        "all_finite": bool(np.all(np.isfinite(s0_np))),
        "wall_s": time.time() - t0,
    }
    if abs(float(r0.dt) - float(DT_PIN)) > 1e-18:
        raise ValueError(f"the run used dt = {float(r0.dt):.9e} against a "
                         f"pinned {DT_PIN:.9e} — the pin is not in force")
    b = rec["base"]
    print(f"\n=== base point, delta = dx/2 ===\n  f_r {b['f_r_hz'] / 1e9:.5f} "
          f"GHz at bin {b['f_r_bin']}  min|S11| {b['s11_min_db']:.2f} dB  "
          f"max|S11| {b['s11_max_abs']:.4f}  settling "
          f"{b['settling_db']} dB  ({b['wall_s']:.0f} s)", flush=True)
    print("  per-bin |S11| (dB): " + " ".join(
        f"{f / 1e9:.2f}:{20 * np.log10(max(abs(v), 1e-300)):.2f}"
        for f, v in zip(freqs, s0_np)), flush=True)
    dump()

    # ---- 2. the observable, and AD through it -----------------------------
    def observable(delta, n=n_steps):
        s, _ = s11(delta, freqs, n)
        return f_r_fixed_bin(s, jnp.asarray(freqs), int(bin0))

    t0 = time.time()
    val, tang = jax.jvp(observable, (jnp.float32(DELTA_0),),
                        (jnp.float32(1.0),))
    # the concrete route's own value of the SAME fixed-bin parabola, taken off
    # the base run above rather than re-solved
    concrete = float(f_r_fixed_bin(jnp.asarray(s0_np), jnp.asarray(freqs),
                                   int(bin0)))
    rec["ad"] = {
        "traced_f_r_hz": float(val), "concrete_f_r_hz": concrete,
        "primal_rel": abs(float(val) - concrete) / abs(concrete),
        "slope_hz_per_m": float(tang),
        "rel_to_reference": (float(tang) - REFERENCE_SLOPE_HZ_PER_M)
        / abs(REFERENCE_SLOPE_HZ_PER_M),
        "bin_held": int(bin0), "wall_s": time.time() - t0,
    }
    a = rec["ad"]
    print(f"\n=== AD ===\n  traced f_r {a['traced_f_r_hz'] / 1e9:.6f} vs "
          f"concrete {a['concrete_f_r_hz'] / 1e9:.6f} GHz "
          f"(rel {a['primal_rel']:.3e})\n  d f_r/d delta = "
          f"{a['slope_hz_per_m']:.6e} Hz/m  vs published "
          f"{REFERENCE_SLOPE_HZ_PER_M:.6e} -> "
          f"{100 * a['rel_to_reference']:+.3f} %  ({a['wall_s']:.0f} s)",
          flush=True)
    dump()

    # ---- 3. the same function, by central differences ---------------------
    # Two slopes come off each pair of runs. The FIXED-bin one is the
    # derivative of the function AD differentiates, and it is the one that has
    # to agree with AD. The MOVING-bin one re-finds the S11 minimum at every
    # delta — the published ladder's observable — so it is the physical
    # resonance rather than a quadratic extrapolation from the base point's
    # three bins, and it converges in h much sooner.
    print("\n=== central differences ===\n  fixed bin = the function AD "
          "differentiates; moving bin = the published ladder's observable",
          flush=True)
    for hf in FD_H_OVER_DX:
        h = hf * DX
        t0 = time.time()
        s_up, _ = s11(DELTA_0 + h, freqs, n_steps)
        s_dn, _ = s11(DELTA_0 - h, freqs, n_steps)
        up = float(f_r_fixed_bin(s_up, jnp.asarray(freqs), int(bin0)))
        dn = float(f_r_fixed_bin(s_dn, jnp.asarray(freqs), int(bin0)))
        slope = (up - dn) / (2 * h)
        mv_up, b_up, _ = f_min_parabolic(freqs, np.asarray(s_up))
        mv_dn, b_dn, _ = f_min_parabolic(freqs, np.asarray(s_dn))
        mv_slope = (mv_up - mv_dn) / (2 * h)
        row = {"h_over_dx": hf, "h_m": h,
               "f_up_hz": up, "f_dn_hz": dn, "slope_hz_per_m": slope,
               "rel_to_ad": (slope - rec["ad"]["slope_hz_per_m"])
               / abs(slope),
               "moving_bin_f_up_hz": mv_up, "moving_bin_f_dn_hz": mv_dn,
               "moving_bin_bins": [int(b_up), int(b_dn)],
               "moving_bin_slope_hz_per_m": mv_slope,
               "moving_bin_rel_to_reference":
                   (mv_slope - REFERENCE_SLOPE_HZ_PER_M)
                   / abs(REFERENCE_SLOPE_HZ_PER_M),
               "wall_s": time.time() - t0}
        rec["fd"].append(row)
        print(f"  h = dx/{1 / hf:<6.0f} fixed {slope:.6e} Hz/m (vs AD "
              f"{100 * row['rel_to_ad']:+.4f} %)   moving {mv_slope:.6e} "
              f"(vs published "
              f"{100 * row['moving_bin_rel_to_reference']:+.3f} %, bins "
              f"{b_up}/{b_dn})  ({row['wall_s']:.0f} s)", flush=True)
        dump()

    # ---- 4. the record-length witness -------------------------------------
    if not smoke:
        t0 = time.time()
        s2, r2 = s11(DELTA_0, freqs, 2 * n_steps)
        s2_np = np.asarray(s2)
        fr2, bin2, _ = f_min_parabolic(freqs, s2_np)
        _, tang2 = jax.jvp(lambda d: observable(d, 2 * n_steps),
                           (jnp.float32(DELTA_0),), (jnp.float32(1.0),))
        ts2 = np.asarray(r2.time_series).ravel() \
            if r2.time_series is not None else np.zeros(1)
        rec["record_witness"] = {
            "n_steps": int(2 * n_steps), "f_r_hz": fr2, "f_r_bin": int(bin2),
            "settling_db": settling_db(ts2),
            "slope_hz_per_m": float(tang2),
            "rel_to_1x": (float(tang2) - rec["ad"]["slope_hz_per_m"])
            / abs(rec["ad"]["slope_hz_per_m"]),
            "wall_s": time.time() - t0,
        }
        w = rec["record_witness"]
        print(f"\n=== 2x record ({w['n_steps']} steps) ===\n  f_r "
              f"{w['f_r_hz'] / 1e9:.6f} GHz at bin {w['f_r_bin']}  settling "
              f"{w['settling_db']} dB\n  AD slope {w['slope_hz_per_m']:.6e}"
              f" Hz/m, {100 * w['rel_to_1x']:+.4f} % against the 1x record  "
              f"({w['wall_s']:.0f} s)", flush=True)

    rec["status"] = "ok"
    rec["wall_s"] = time.time() - t_all
    dump()
    print(f"\nwrote {out_dir / 'm6_wire_port_ad.json'}  "
          f"({rec['wall_s']:.0f} s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "M6"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(Path(a.out), smoke=a.smoke)


if __name__ == "__main__":
    main()
