#!/usr/bin/env python
"""RT5880 patch antenna -- does the realized patch and ground WIDTH move R(f0)?

A DIAGNOSTIC on branch ``archive/rt5880-patch-width-probe``; not part of the
cross-validation case and not run by CI.

On the GPU ladder of VESSL 369367264572 (commit 0a92af49) the case's rfx board
read R(f0) = 71.093 / 72.997 / 71.920 ohm at h/4, h/8, h/12 (+2.678 %,
-1.476 %) while f0 converged.  On those rungs the lattice realizes the patch's
y node span as 49.2125 / 49.2125 / 49.7417 mm and the ground's as 65.0875 /
65.8812 / 65.6167 mm; the x spans are the same on every rung.  Hypothesis H1
(the lane leader's, not verified): the R reversal comes from the width spans
changing from rung to rung.

Two arms, on the case's own code (``tests/crossval/rt5880_patch/
test_rt5880_patch.py``: ``build``, ``realized_geometry``, ``check_realized``,
``run_rung`` with its ring-down and passivity witnesses, ``resonance`` and
``zin_from_s11``) and the shared estimators (``tests/crossval/_v2_judging.py``:
``refined_remax``, ``mesh_statement``):

* arm A, the control: the case's board as it is, at h/4.  It must reproduce the
  logged R and f0; both are printed beside the new numbers.
* arm B: the same board with the patch width W = 49.2125 mm (62 h/4) and the
  ground width GP_Y = 65.0875 mm (82 h/4) -- node-aligned, so every rung
  realizes the same node spans -- everything else unchanged (L 40 mm, GP_X
  56 mm, the probe at -8.73125 mm, the box, the CPML, NUM_PERIODS, the band), at
  h/4, h/8 and h/12.  Before each rung is solved its realized node spans are
  held to the declared widths and to one another within 1 nm, and the probe to
  its node (``check_realized``); a rung that differs is refused.

The case's board constants are module globals read when the builders run (the
box and the patch centre depend on h only), so arm B sets ``W_PATCH`` and
``GP_Y`` on the module and the realized spans printed per rung show what was
built.

Per rung: realized node spans and solved sizes (node span + 2 EDGE_OFFSET dx),
f0 / R / X at the Re(Zin) peak, the |S11| minimum and depth, the settling
energy, cells, steps, wall time; then arm B's R ladder through
``mesh_statement`` at 0.2 % / 2 % (the case's R bars) and its f0 ladder at the
default 0.1 % / 1 %.  Per-rung Zin arrays go to ``<out>/<arm>_<rung>.npz``, one
figure (Re and Im Zin, both arms) to ``<out>/rt5880_patch_width_probe.png`` and
the numbers to ``<out>/summary.json``.

``--smoke`` (a local build check): h/4 only, both arms, a record of
``--smoke-periods`` periods solved directly (``run_rung``'s ring-down and
passivity witnesses cannot hold on such a record, so it is bypassed); the
numbers are not meaningful, the realized spans are.

    python scripts/diagnostics/rt5880_patch_width_probe.py --out DIR
    python scripts/diagnostics/rt5880_patch_width_probe.py --out DIR --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.crossval._v2_judging import (  # noqa: E402
    RESISTANCE_BAR,
    mesh_statement,
    refined_remax,
)
from tests.crossval.rt5880_patch import test_rt5880_patch as T  # noqa: E402

# The GPU ladder this diagnostic starts from: VESSL 369367264572 at 0a92af49
# (lab share research/rfx/.omx/rt5880-patch-ladder/20260924T142517Z-0a92af49/
# ladder.log), as the lane leader read it.
LOGGED_H4 = {"f0_hz": 2.300078e9, "r_ohm": 71.093}
LOGGED_R_OHM = (71.093, 72.997, 71.920)

ARM_B_W_PATCH = 62 * T.H_SUB / 4      # 49.2125 mm, node-aligned on h/4, h/8, h/12
ARM_B_GP_Y = 82 * T.H_SUB / 4         # 65.0875 mm
SPAN_TOL_M = 1e-9
SPAN_KEYS = ("patch_x_node_m", "patch_y_node_m", "ground_x_node_m", "ground_y_node_m")


def _set_board(arm: str) -> None:
    """Arm A: the case's constants as imported.  Arm B: W and GP_Y moved."""
    if arm == "A":
        T.W_PATCH, T.GP_Y = _ORIGINAL["W_PATCH"], _ORIGINAL["GP_Y"]
    else:
        T.W_PATCH, T.GP_Y = ARM_B_W_PATCH, ARM_B_GP_Y


_ORIGINAL = {"W_PATCH": T.W_PATCH, "GP_Y": T.GP_Y}


def _spans(g: dict, dx: float) -> dict:
    from rfx.mesh_edges import EDGE_OFFSET
    p, gr = g["patch"], g["ground"]
    return {
        "patch_x_node_m": p["x_node_span_m"], "patch_y_node_m": p["y_node_span_m"],
        "ground_x_node_m": gr["x_node_span_m"], "ground_y_node_m": gr["y_node_span_m"],
        "patch_x_solved_m": p["x_node_span_m"] + 2 * EDGE_OFFSET * dx,
        "patch_y_solved_m": p["y_node_span_m"] + 2 * EDGE_OFFSET * dx,
        "ground_x_solved_m": gr["x_node_span_m"] + 2 * EDGE_OFFSET * dx,
        "ground_y_solved_m": gr["y_node_span_m"] + 2 * EDGE_OFFSET * dx,
        "probe_offset_m": list(g["probe_offset_from_patch_centre_m"]),
        "edge_offset_cells": float(EDGE_OFFSET),
    }


def _substrate_extent(dx: float) -> dict:
    """The laminate as the assembled material realizes it: along x and y
    through the patch centre, halfway up the substrate, the number of cells
    carrying the laminate's permittivity, and that count times dx.  The
    builder draws the substrate over the ground's footprint, so it follows
    GP_X and GP_Y."""
    from tests._realized_geometry import _node_line
    sim = T.build(dx)
    grid = sim._build_grid()
    eps = np.asarray(sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0].eps_r,
                     dtype=float)
    xs, ys, zs = (_node_line(grid, a) for a in (0, 1, 2))
    i = int(np.argmin(np.abs(xs - T.FRAME.cx)))
    j = int(np.argmin(np.abs(ys - T.FRAME.cy)))
    k = int(np.argmin(np.abs(zs - (T.FRAME.z_ground + T.H_SUB / 2))))
    on = lambda v: np.abs(v - T.EPS_R) < 1e-5  # noqa: E731
    nx, ny = int(on(eps[:, j, k]).sum()), int(on(eps[i, :, k]).sum())
    other = sorted({round(float(v), 6) for v in np.concatenate([eps[:, j, k], eps[i, :, k]])}
                   - {1.0, round(T.EPS_R, 6)})
    return {"substrate_x_cells": nx, "substrate_y_cells": ny,
            "substrate_x_m": nx * dx, "substrate_y_m": ny * dx,
            "substrate_other_eps": other}


def _print_spans(label: str, s: dict) -> None:
    print(f"  [{label}] realized node spans: patch {s['patch_x_node_m']*1e3:.6f} x "
          f"{s['patch_y_node_m']*1e3:.6f} mm, ground {s['ground_x_node_m']*1e3:.6f} x "
          f"{s['ground_y_node_m']*1e3:.6f} mm; solved (node span + 2 x "
          f"{s['edge_offset_cells']:.2f} dx): patch {s['patch_x_solved_m']*1e3:.4f} x "
          f"{s['patch_y_solved_m']*1e3:.4f} mm, ground {s['ground_x_solved_m']*1e3:.4f} x "
          f"{s['ground_y_solved_m']*1e3:.4f} mm; probe offset "
          f"({s['probe_offset_m'][0]*1e3:+.6f}, {s['probe_offset_m'][1]*1e3:+.6f}) mm")
    if "substrate_y_cells" in s:
        print(f"  [{label}] realized substrate: {s['substrate_x_cells']} x "
              f"{s['substrate_y_cells']} cells = {s['substrate_x_m']*1e3:.4f} x "
              f"{s['substrate_y_m']*1e3:.4f} mm (drawn {T.GP_X*1e3:.4f} x "
              f"{T.GP_Y*1e3:.4f} mm, the ground's footprint); other permittivities on "
              f"those lines: {s['substrate_other_eps']}")


def _check_arm_b(s: dict, first: dict | None, label: str) -> None:
    """Arm B's rungs must build one board: the declared widths as node spans,
    the same four spans on every rung (the probe is held to its node by
    check_realized)."""
    bad = []
    for k, v in (("patch_y_node_m", ARM_B_W_PATCH), ("ground_y_node_m", ARM_B_GP_Y)):
        if abs(s[k] - v) > SPAN_TOL_M:
            bad.append(f"{k} {s[k]*1e3:.6f} mm, declared {v*1e3:.6f} mm")
    if first is not None:
        for k in SPAN_KEYS:
            if abs(s[k] - first[k]) > SPAN_TOL_M:
                bad.append(f"{k} {s[k]*1e3:.6f} mm vs {first[k]*1e3:.6f} mm on the first rung")
    if bad:
        raise AssertionError(f"[{label}] arm B does not realize one board: " + "; ".join(bad))


def _estimates(freqs, s11) -> dict:
    zin = T.zin_from_s11(s11)
    rm = refined_remax(freqs, zin, *T.RESONANCE_BAND_HZ)
    dip = T.resonance(freqs, np.abs(s11), zin)
    return {"zin": zin, "remax": rm, "dip": dip}


def _solve(dx: float, smoke: bool, smoke_periods: float) -> dict:
    """The case's own solve.  Smoke: the same build and realization check, and
    a short record solved directly, without run_rung's witnesses."""
    if not smoke:
        return T.run_rung(dx)
    import jax.numpy as jnp
    sim = T.build(dx)
    g = T.assert_realized(sim, dx)
    grid = sim._build_grid()
    n_steps = grid.num_timesteps(num_periods=smoke_periods)
    t0 = time.perf_counter()
    res = sim.run(num_periods=smoke_periods, compute_s_params=True,
                  s_param_freqs=jnp.asarray(T.FREQS_HZ), skip_preflight=True)
    wall = time.perf_counter() - t0
    return dict(dx_m=dx, freqs_hz=np.asarray(T.FREQS_HZ, float),
                s11=np.asarray(np.asarray(res.s_params)[0, 0, :], dtype=complex),
                settling_db=res.settling_db, wall_s=wall, n_steps=int(n_steps),
                n_cells=g["n_cells"], realized=g)


def run_one(arm: str, dx: float, out: Path, smoke: bool, smoke_periods: float,
            first_b: dict | None) -> dict:
    n = int(round(T.H_SUB / dx))
    label = f"arm {arm}, h/{n}"
    _set_board(arm)
    print(f"\n=== {label}: W {T.W_PATCH*1e3:.4f} mm, GP_Y {T.GP_Y*1e3:.4f} mm, "
          f"L {T.L_PATCH*1e3:.3f} mm, GP_X {T.GP_X*1e3:.3f} mm, probe "
          f"{T.FEED_OFFSET_X*1e3:+.5f} mm, dx {dx*1e6:.3f} um ===")
    g0 = T.check_realized(T.realized_geometry(T.build(dx)), dx)
    s = _spans(g0, dx)
    s.update(_substrate_extent(dx))
    _print_spans(label + ", before the solve", s)
    if arm == "B":
        _check_arm_b(s, first_b, label)
    r = _solve(dx, smoke, smoke_periods)
    s_run = _spans(r["realized"], dx)
    if any(abs(s_run[k] - s[k]) > SPAN_TOL_M for k in SPAN_KEYS):
        raise AssertionError(f"[{label}] the solved board is not the checked one")
    e = _estimates(r["freqs_hz"], r["s11"])
    rm, dip = e["remax"], e["dip"]
    row = {
        "arm": arm, "rung": f"h/{n}", "dx_m": dx, "spans": s,
        "w_patch_m": T.W_PATCH, "gp_y_m": T.GP_Y,
        "f0_hz": rm["f0_hz"], "r_ohm": rm["r_ohm"], "x_ohm": rm["x_ohm"],
        "remax_flags": rm["flags"], "sub_bin_shift": rm["sub_bin_shift"],
        "dip_hz": dip["f"], "dip_depth_db": dip["depth_db"],
        "settling_db": None if r["settling_db"] is None else float(r["settling_db"]),
        "n_cells": int(r["n_cells"]), "n_steps": int(r["n_steps"]),
        "wall_s": float(r["wall_s"]), "smoke": smoke,
    }
    print(f"  [{label}] f0 (Re(Zin) peak) {rm['f0_hz']/1e9:.6f} GHz, R {rm['r_ohm']:.3f} ohm, "
          f"X {rm['x_ohm']:+.3f} ohm, flags {rm['flags']}; |S11| minimum "
          f"{dip['f']/1e9:.6f} GHz at {dip['depth_db']:.3f} dB; settling "
          f"{row['settling_db']} dB; {row['n_cells']} cells, {row['n_steps']} steps, "
          f"{row['wall_s']:.1f} s" + ("  (SMOKE: a short record, numbers not meaningful)"
                                     if smoke else ""))
    if arm == "A" and n == 4:
        print(f"  [{label}] logged at 0a92af49 (VESSL 369367264572): f0 "
              f"{LOGGED_H4['f0_hz']/1e9:.6f} GHz, R {LOGGED_H4['r_ohm']:.3f} ohm; this run: "
              f"f0 {rm['f0_hz']/1e9:.6f} GHz ({100*(rm['f0_hz']-LOGGED_H4['f0_hz'])/LOGGED_H4['f0_hz']:+.4f} %), "
              f"R {rm['r_ohm']:.3f} ohm ({100*(rm['r_ohm']-LOGGED_H4['r_ohm'])/LOGGED_H4['r_ohm']:+.4f} %)")
    np.savez(out / f"arm{arm}_h{n}.npz", freqs_hz=r["freqs_hz"], s11=r["s11"],
             zin=e["zin"], dx_m=dx, w_patch_m=T.W_PATCH, gp_y_m=T.GP_Y)
    row["_zin"] = e["zin"]
    row["_freqs"] = r["freqs_hz"]
    return row


def _figure(rows: list, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for row in rows:
        f = row["_freqs"] / 1e9
        ls = "-" if row["arm"] == "B" else "--"
        lab = (f"arm {row['arm']} {row['rung']} (W {row['w_patch_m']*1e3:.4f}, "
               f"GP_Y {row['gp_y_m']*1e3:.4f} mm)")
        a.plot(f, row["_zin"].real, ls, lw=1.2, label=lab)
        b.plot(f, row["_zin"].imag, ls, lw=1.2, label=lab)
    for ax, name in ((a, "Re Zin (ohm)"), (b, "Im Zin (ohm)")):
        ax.set_xlim(2.1, 2.55)
        ax.set_xlabel("frequency (GHz)")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    a.legend(fontsize=7)
    fig.tight_layout()
    path = out / "rt5880_patch_width_probe.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--smoke", action="store_true",
                   help="h/4 only, both arms, a short record without the witnesses")
    p.add_argument("--smoke-periods", type=float, default=2.0)
    p.add_argument("--rungs", default="4,8,12",
                   help="arm B's rungs as n of h/n (arm A runs h/4 only)")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rungs_n = [4] if args.smoke else [int(v) for v in args.rungs.split(",") if v.strip()]
    for n in rungs_n:
        if n not in (4, 8, 12):
            raise SystemExit(f"h/{n} is not a rung of the case's ladder")
    print(f"RT5880 patch width probe -- arm A (the case's board) at h/4; arm B (W "
          f"{ARM_B_W_PATCH*1e3:.4f} mm, GP_Y {ARM_B_GP_Y*1e3:.4f} mm) at "
          f"{['h/%d' % n for n in rungs_n]}; NUM_PERIODS "
          f"{args.smoke_periods if args.smoke else T.NUM_PERIODS}"
          + ("  [SMOKE]" if args.smoke else ""))
    rows = [run_one("A", T.H_SUB / 4, args.out, args.smoke, args.smoke_periods, None)]
    first_b = None
    for n in rungs_n:
        row = run_one("B", T.H_SUB / n, args.out, args.smoke, args.smoke_periods, first_b)
        if first_b is None:
            first_b = row["spans"]
        rows.append(row)
    _set_board("A")

    b_rows = [r for r in rows if r["arm"] == "B"]
    print("\n  arm B along the ladder:")
    for r in b_rows:
        print(f"    {r['rung']:5s} f0 {r['f0_hz']/1e9:.6f} GHz  R {r['r_ohm']:.3f} ohm  "
              f"X {r['x_ohm']:+.3f} ohm  |S11| min {r['dip_hz']/1e9:.6f} GHz "
              f"{r['dip_depth_db']:.3f} dB")
    print(f"  the case's board, logged at 0a92af49: R {' / '.join(f'{v:.3f}' for v in LOGGED_R_OHM)} ohm")
    statements = {}
    if len(b_rows) >= 2:
        labels = [r["rung"] for r in b_rows]
        rs = mesh_statement([r["r_ohm"] for r in b_rows], labels,
                            flat_step=RESISTANCE_BAR / 10.0, agreement=RESISTANCE_BAR,
                            unit=("ohm", 1.0, 3))
        fs = mesh_statement([r["f0_hz"] for r in b_rows], labels)
        print(f"  R(f0), flat within {RESISTANCE_BAR/10*100:.1f} %, last two within "
              f"{RESISTANCE_BAR*100:.0f} %: {rs['verdict']} -- steps "
              + ", ".join(f"{v:+.3f} %" for v in rs["steps_pct"]) + f" -- {rs['reason']}")
        print(f"  f0, flat within 0.1 %, last two within 1 %: {fs['verdict']} -- steps "
              + ", ".join(f"{v:+.3f} %" for v in fs["steps_pct"]) + f" -- {fs['reason']}")
        statements = {"r": {k: rs[k] for k in ("verdict", "steps_pct", "flat", "last_two_pct", "reason")},
                      "f0": {k: fs[k] for k in ("verdict", "steps_pct", "flat", "last_two_pct", "reason")}}
    fig = _figure(rows, args.out)
    summary = {"smoke": args.smoke,
               "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
               "mesh_statements_arm_b": statements,
               "logged_case_board_r_ohm": LOGGED_R_OHM, "logged_h4": LOGGED_H4}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\n  figure: {fig}\n  summary: {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
