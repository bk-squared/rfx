"""Sensitivity of a probe-fed patch's |S11|^2 to the permittivity of every substrate cell.

The board is the RT/Duroid 5880 probe-fed patch of the cross-validation case
``tests/crossval/rt5880_patch/test_rt5880_patch.py``, built by that module's own
``build(dx)`` at dx = h/4 (imported by path, never edited).  One reverse-mode
pass through the 3-D FDTD solve gives dJ/d(eps_r) for every cell of a design box
in the substrate, J = |S11(f_t)|^2 at one frequency f_t just above the rfx
resonance.  The rules, steps and thresholds are pre-declared in
``docs/design_notes/20260927_showcase_predeclaration.md``; this script only
carries them out.

Stages (each writes its arrays before the next starts)::

    python scripts/showcase/patch_sensitivity.py --stage main --out DIR
    python scripts/showcase/patch_sensitivity.py --stage x64fd --out DIR      # only if DIR/x64_needed.json exists
    python scripts/showcase/patch_sensitivity.py --stage timing --box design --mode grad --out DIR
    python scripts/showcase/patch_sensitivity.py --stage finalize --out DIR

``main`` runs (a) the baseline, (b) f_t, (c) the gradient map with its
short-record and long-record companions, (d) the block finite differences,
(e) |E_z(f_t)|^2 in the substrate.  ``timing`` is one process per design box
and mode, so a device's peak memory belongs to that one program.  ``finalize``
assembles ``result.json`` from the files the other stages left.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _record  # noqa: E402

CASE_ID = "patch-sensitivity"
QUESTION = ("Which substrate cells of a probe-fed RT/Duroid 5880 patch move "
            "|S11(f_t)|^2 at f_t = 1.01 f_r, from one reverse-mode pass, checked "
            "against central finite differences on pre-declared cell blocks; and "
            "what the gradient costs in time and device memory.")
PATCH_MODULE = REPO / "tests" / "crossval" / "rt5880_patch" / "test_rt5880_patch.py"

# ------------------------------------------------------------ pre-declared
# docs/design_notes/20260927_showcase_predeclaration.md, section "Patch".
N_PER_H = 4                    # dx = H_SUB / 4
FT_FACTOR = 1.01               # f_t = FT_FACTOR * f_r
F_R_ESTIMATOR = "s11_minimum"  # "s11_minimum": the module's resonance(); "remax": refined_remax
MARGIN_H = 2                   # design box margin, in substrate thicknesses
SHORT_RECORD_STEPS = 600       # A1 check: a record far shorter than the ring-down
WITNESS_PERIOD_FACTOR = 1.5    # the record-length witness: 1.5 x the module's NUM_PERIODS
WITNESS_TOL = 0.05             # Amendment 1 (lane leader): judged, per block, |S_1.5 - S_1.0| / |S_1.5|
FLANK_DF_HZ = 1.0e6            # Amendment 1: d|S11|^2/df at f_t by central difference over f_t +- 1 MHz
FD_STEPS = (0.2, 0.1, 0.05)    # Delta eps_r of a whole block
FD_JUDGED_STEP = 0.1
FD_REL_BAR = 0.05              # |AD - FD| / |FD| at the judged step
FD_JUDGE_FRACTION = 0.1        # judged where |FD| >= this x the largest |FD| of the six
# Blocks: (name, x cells, y cells) as inclusive cell-index offsets from the
# patch centre NODE (I, J) at h/4; every block spans the full substrate
# thickness.  The patch covers cells I-25 .. I+24 along x (the resonant length)
# and J-31 .. J+30 along y.  Cell i spans nodes i .. i+1.
BLOCKS = (
    ("edge_minus_x", (-25, -23), (-1, 1)),   # inside the patch, at the -x radiating edge
    ("edge_plus_x", (22, 24), (-1, 1)),      # inside the patch, at the +x radiating edge
    ("centre_plus_y", (-1, 1), (14, 16)),    # on the centre line x = 0, +y side
    ("centre_minus_y", (-1, 1), (-17, -15)),  # on the centre line x = 0, -y side
    ("margin_plus_x", (25, 27), (-1, 1)),    # outside the patch, beyond the +x radiating edge
    ("margin_plus_y", (9, 11), (31, 33)),    # outside the patch, beyond the +y edge
)
TIMING_BOXES = ("patch", "design", "layer")
TIMING_REPEATS = 3


# ------------------------------------------------------------------ helpers
SMOKE = False   # --smoke: a plumbing check with a record of a few dozen steps


def load_patch_module(name: str = "_showcase_rt5880_patch"):
    """The cross-validation module, imported by path, read-only."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PATCH_MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if SMOKE:
        # a record far too short for physics: only the code path is exercised
        mod.NUM_PERIODS = 0.25
        mod.SETTLING_DB = float("inf")
        mod.PASSIVITY_EXCESS_BAR = float("inf")
    return mod


def _save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_record._jsonable) + "\n")
    tmp.replace(path)


def _load_json(path: Path):
    return json.loads(path.read_text())


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Board:
    """The h/4 board, its grid, and the cell-index bookkeeping of every box."""

    def __init__(self, mod, dx: float, *, precision_patch=None):
        self.mod = mod
        self.dx = dx
        self.sim = mod.build(dx)
        self.g = mod.assert_realized(self.sim, dx)
        self.grid = self.sim._build_grid()
        self.pad = (self.grid.pad_x_lo, self.grid.pad_y_lo, self.grid.pad_z_lo)
        p = self.g["patch"]
        (i0, i1), (j0, j1) = p["cols"], p["rows"]
        assert (i0 + i1) % 2 == 0 and (j0 + j1) % 2 == 0, (p["cols"], p["rows"])
        self.I, self.J = (i0 + i1) // 2, (j0 + j1) // 2       # patch centre node
        self.patch_cells_x = (i0, i1 - 1)                      # cells under the patch
        self.patch_cells_y = (j0, j1 - 1)
        self.k_ground, self.k_patch = self.g["ground_plane_k"], self.g["patch_plane_k"]
        self.sub_cells_z = (self.k_ground, self.k_patch - 1)
        n_sub = self.k_patch - self.k_ground
        assert n_sub == N_PER_H, n_sub
        # the laminate's cells, from the assembled material (the realized board)
        mats = self.sim._assemble_materials(self.grid, pec_sheets=[], pec_wires=[])[0]
        self.eps = np.asarray(mats.eps_r)
        k_mid = self.k_ground + n_sub // 2
        on = np.abs(self.eps[:, :, k_mid] - mod.EPS_R) < 1e-6
        xi, yj = np.flatnonzero(on.any(axis=1)), np.flatnonzero(on.any(axis=0))
        self.lam_cells_x = (int(xi.min()), int(xi.max()))
        self.lam_cells_y = (int(yj.min()), int(yj.max()))
        m = MARGIN_H * N_PER_H
        n_int = [int(round(L / dx)) for L in mod.FRAME.domain[:2]]
        self.boxes = {
            "patch": (self.patch_cells_x, self.patch_cells_y, self.sub_cells_z),
            "design": ((max(self.patch_cells_x[0] - m, self.lam_cells_x[0]),
                        min(self.patch_cells_x[1] + m, self.lam_cells_x[1])),
                       (max(self.patch_cells_y[0] - m, self.lam_cells_y[0]),
                        min(self.patch_cells_y[1] + m, self.lam_cells_y[1])),
                       self.sub_cells_z),
            "laminate": (self.lam_cells_x, self.lam_cells_y, self.sub_cells_z),
            # Amendment 1: the substrate z-layer across the whole physical
            # interior, vacuum beside the laminate included, out of the CPML
            "layer": ((self.pad[0], self.pad[0] + n_int[0] - 1),
                      (self.pad[1], self.pad[1] + n_int[1] - 1), self.sub_cells_z),
        }
        self.xs = self.mod._node_line(self.grid, 0)
        self.ys = self.mod._node_line(self.grid, 1)
        self.zs = self.mod._node_line(self.grid, 2)

    def corners(self, name: str):
        """Metre corners a quarter cell inside each end cell's lower node, so
        each corner resolves to its cell (never to a tie)."""
        (ia, ib), (ja, jb), (ka, kb) = self.boxes[name]
        lo = tuple((c - p + 0.25) * self.dx for c, p in zip((ia, ja, ka), self.pad))
        hi = tuple((c - p + 0.25) * self.dx for c, p in zip((ib, jb, kb), self.pad))
        want = (ia, ib + 1, ja, jb + 1, ka, kb + 1)
        got = tuple(self.sim._design_box_bounds_from_corners(self.grid, (lo, hi)))
        assert got == want, (name, got, want)
        return lo, hi

    def box_slice(self, name: str):
        (ia, ib), (ja, jb), (ka, kb) = self.boxes[name]
        return (slice(ia, ib + 1), slice(ja, jb + 1), slice(ka, kb + 1))

    def background(self, name: str):
        return np.asarray(self.eps[self.box_slice(name)], dtype=np.float32)

    def block_cells(self, name: str, block):
        """Block cells relative to the design box ``name`` (index slices)."""
        _, (dxa, dxb), (dya, dyb) = block
        (ia, _), (ja, _), _ = self.boxes[name]
        xa, xb = self.I + dxa - ia, self.I + dxb - ia
        ya, yb = self.J + dya - ja, self.J + dyb - ja
        return (slice(xa, xb + 1), slice(ya, yb + 1), slice(None))

    def block_realized(self, block) -> dict:
        _, (dxa, dxb), (dya, dyb) = block
        ia, ib, ja, jb = self.I + dxa, self.I + dxb, self.J + dya, self.J + dyb
        return {
            "cells_x": [ia, ib], "cells_y": [ja, jb],
            "cells_z": list(self.sub_cells_z),
            "n_cells": (ib - ia + 1) * (jb - ja + 1) * N_PER_H,
            "x_m_from_patch_centre": [float(self.xs[ia] - self.xs[self.I]),
                                      float(self.xs[ib + 1] - self.xs[self.I])],
            "y_m_from_patch_centre": [float(self.ys[ja] - self.ys[self.J]),
                                      float(self.ys[jb + 1] - self.ys[self.J])],
            "footprint_m": [float(self.xs[ib + 1] - self.xs[ia]),
                            float(self.ys[jb + 1] - self.ys[ja])],
            "eps_r_background": sorted({float(v) for v in np.round(
                self.eps[ia:ib + 1, ja:jb + 1, self.sub_cells_z[0]:self.sub_cells_z[1] + 1], 6).ravel()}),
        }

    def describe_box(self, name: str) -> dict:
        (ia, ib), (ja, jb), (ka, kb) = self.boxes[name]
        lam = int(np.sum(np.abs(self.eps[self.box_slice(name)] - self.mod.EPS_R) < 1e-6))
        n = (ib - ia + 1) * (jb - ja + 1) * (kb - ka + 1)
        return {"cells_x": [ia, ib], "cells_y": [ja, jb], "cells_z": [ka, kb],
                "n_laminate_cells": lam, "n_other_cells": n - lam,
                "shape": [ib - ia + 1, jb - ja + 1, kb - ka + 1],
                "n_cells": (ib - ia + 1) * (jb - ja + 1) * (kb - ka + 1),
                "x_m": [float(self.xs[ia]), float(self.xs[ib + 1])],
                "y_m": [float(self.ys[ja]), float(self.ys[jb + 1])],
                "z_m": [float(self.zs[ka]), float(self.zs[kb + 1])]}


def make_objective(board: Board, box: str, f_t: float, *, n_steps: int,
                   checkpoint_segments=None):
    """J(eps_box) = |S11(f_t)|^2 through ``forward`` with a design box."""
    import jax.numpy as jnp

    corners = board.corners(box)
    freqs = jnp.asarray([f_t])

    def J(eps_box):
        res = board.sim.forward(
            design_box=corners, design_eps_override=eps_box,
            design_box_holds_ports=True, n_steps=n_steps, checkpoint=False,
            checkpoint_segments=checkpoint_segments,
            port_s11_freqs=freqs, skip_preflight=True)
        s = res.s_params
        s = s if s.ndim == 1 else s[0]
        return jnp.real(s[0]) ** 2 + jnp.imag(s[0]) ** 2

    return J


def _write_s1p(path: Path, freqs, s11, comments) -> None:
    from rfx.io import write_touchstone
    s = np.asarray(s11, dtype=complex).reshape(1, 1, -1)
    write_touchstone(str(path), s, np.asarray(freqs, dtype=float), z0=50.0,
                     freq_unit="Hz", fmt="RI", comments=comments)


def _pearson(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def _device_memory():
    import jax
    ms = jax.devices()[0].memory_stats() or {}
    return {"peak_bytes_in_use": int(ms.get("peak_bytes_in_use", -1)),
            "bytes_limit": int(ms.get("bytes_limit", -1))}


# ------------------------------------------------------------------ stage main
def stage_main(out: Path) -> None:
    import jax
    import jax.numpy as jnp

    mod = load_patch_module()
    dx = mod.H_SUB / N_PER_H
    board = Board(mod, dx)
    num_periods = float(mod.NUM_PERIODS)
    n_steps = int(board.grid.num_timesteps(num_periods=num_periods))
    wall = {}
    _log(f"board h/{N_PER_H}: grid {board.g['grid_shape']} ({board.g['n_cells']} cells), "
         f"{n_steps} steps, patch centre node ({board.I}, {board.J}), boxes "
         + ", ".join(f"{k} {board.describe_box(k)['shape']}" for k in board.boxes))

    model = {
        "structure": "RT/Duroid 5880 probe-fed patch, tests/crossval/rt5880_patch "
                     "build(dx) imported by path",
        "patch_module": str(PATCH_MODULE.relative_to(REPO)),
        "dx_m": dx, "n_per_h_sub": N_PER_H, "h_sub_m": mod.H_SUB, "eps_r": mod.EPS_R,
        "sigma_sub_s_per_m": mod.SIGMA_SUB, "patch_m": [mod.L_PATCH, mod.W_PATCH],
        "ground_m": [mod.GP_X, mod.GP_Y], "feed_offset_x_m": mod.FEED_OFFSET_X,
        "port_impedance_ohm": mod.PORT_IMPEDANCE_OHM,
        "grid_shape": list(board.g["grid_shape"]), "n_cells": board.g["n_cells"],
        "num_periods": num_periods, "n_steps": n_steps, "dt_s": float(board.grid.dt),
        "record_s": n_steps * float(board.grid.dt),
        "patch_centre_node": [board.I, board.J],
        "patch_cells": {"x": list(board.patch_cells_x), "y": list(board.patch_cells_y)},
        "laminate_cells": {"x": list(board.lam_cells_x), "y": list(board.lam_cells_y),
                           "z": list(board.sub_cells_z)},
        "boxes": {k: board.describe_box(k) for k in board.boxes},
        "blocks": {b[0]: board.block_realized(b) for b in BLOCKS},
        "pad": list(board.pad),
    }
    _save_json(out / "model.json", model)

    # ---- (a) baseline: the module's own run_rung, then forward() on the same board
    _log("(a) baseline run_rung at h/4")
    t0 = time.perf_counter()
    base = mod.run_rung(dx)
    wall["baseline_run_rung_s"] = time.perf_counter() - t0
    freqs = np.asarray(base["freqs_hz"], dtype=float)
    s11 = np.asarray(base["s11"], dtype=complex)
    zin = mod.zin_from_s11(s11)
    res_min = mod.resonance(freqs, np.abs(s11), zin)
    res_remax = mod.refined_remax(freqs, zin, *mod.RESONANCE_BAND_HZ)
    np.savez(out / "baseline.npz", freqs_hz=freqs, s11=s11, zin_ohm=zin)
    _write_s1p(out / "baseline_h4.s1p", freqs, s11, [
        "rfx RT/Duroid 5880 probe-fed patch, tests/crossval/rt5880_patch build(h/4)",
        f"dx {dx:.6e} m, num_periods {num_periods}, run(compute_s_params=True)"])
    baseline = {
        "estimator_used_for_f_r": F_R_ESTIMATOR,
        "s11_minimum": {"f_hz": res_min["f"], "depth_db": res_min["depth_db"],
                        "bin_f_hz": res_min["bin_f"], "f_lo_10db_hz": res_min["f_lo_10db"],
                        "f_hi_10db_hz": res_min["f_hi_10db"],
                        "estimator": "module resonance(): refined_extremum on log|S11|"},
        "remax": {"f0_hz": res_remax["f0_hz"], "r_ohm": res_remax["r_ohm"],
                  "x_ohm": res_remax["x_ohm"], "flags": res_remax["flags"],
                  "estimator": "tests/crossval/_v2_judging.refined_remax"},
        "settling_db": float(base["settling_db"]),
        "settling_witness": base["settling_witness"],
        "max_abs_s11": float(np.max(np.abs(s11))),
        "n_preflight_findings": int(base["n_preflight"]),
        "wall_s": base["wall_s"], "n_steps": base["n_steps"], "dt_s": base["dt_s"],
    }
    _save_json(out / "baseline.json", baseline)

    f_r = res_min["f"] if F_R_ESTIMATOR == "s11_minimum" else res_remax["f0_hz"]
    f_t = FT_FACTOR * f_r
    _save_json(out / "f_t.json", {"f_r_hz": f_r, "f_t_hz": f_t, "factor": FT_FACTOR,
                                  "estimator": F_R_ESTIMATOR})
    _log(f"(b) f_r {f_r/1e9:.6f} GHz ({F_R_ESTIMATOR}), f_t {f_t/1e9:.6f} GHz")

    # forward() S11 on the record's bins against run()'s: the objective reads this path
    _log("(a2) forward(port_s11_freqs=...) on the same board, no design box")
    t0 = time.perf_counter()
    fres = board.sim.forward(num_periods=num_periods, port_s11_freqs=jnp.asarray(freqs),
                             skip_preflight=True, checkpoint=False)
    s11_fwd = np.asarray(fres.s_params, dtype=complex).reshape(-1)
    wall["forward_all_bins_s"] = time.perf_counter() - t0
    np.savez(out / "forward_s11.npz", freqs_hz=freqs, s11=s11_fwd)
    fwd = {"max_abs_diff_vs_run": float(np.max(np.abs(s11_fwd - s11))),
           "settling_db": fres.settling_db, "n_steps": int(fres.time_series.shape[0])}
    _save_json(out / "forward_vs_run.json", fwd)
    _log(f"    max |S11_forward - S11_run| = {fwd['max_abs_diff_vs_run']:.3e}")

    # ---- (c) gradient map: short record (A1), full record, 1.5x record
    eps_bg = jnp.asarray(board.background("design"))
    assert np.allclose(np.asarray(eps_bg), mod.EPS_R, atol=1e-6), "design box is not all laminate"
    grads = {}

    def _grad_info(g, v, steps, memory, how):
        return {"n_steps": steps, "J": float(v), "finite": bool(np.all(np.isfinite(g))),
                "n_nonzero": int(np.count_nonzero(g)), "sum": float(np.sum(g)),
                "max_abs": float(np.max(np.abs(g))), "memory": memory, "how": how}

    def _save_grad(tag, g, v, steps):
        np.savez(out / f"gradient_{tag}.npz", grad=g, J=float(v), n_steps=steps, f_t_hz=f_t,
                 box_cells=np.asarray(board.boxes["design"]), dx_m=dx,
                 x_m=board.xs[board.boxes["design"][0][0]:board.boxes["design"][0][1] + 2],
                 y_m=board.ys[board.boxes["design"][1][0]:board.boxes["design"][1][1] + 2],
                 z_m=board.zs[board.boxes["design"][2][0]:board.boxes["design"][2][1] + 2])

    steps_short = SHORT_RECORD_STEPS if not SMOKE else 10
    _log(f"(c) value_and_grad, design box, {steps_short} steps (short, A1)")
    vg = jax.jit(jax.value_and_grad(make_objective(board, "design", f_t, n_steps=steps_short)))
    t0 = time.perf_counter()
    v, g = vg(eps_bg)
    g = np.asarray(g)
    wall["grad_short_first_call_s"] = time.perf_counter() - t0
    grads["short"] = _grad_info(g, v, steps_short, _device_memory(), "jit(value_and_grad)")
    _save_grad("short", g, v, steps_short)
    _save_json(out / "gradient_summary.json", grads)
    _log(f"    J {float(v):.6e}, finite {grads['short']['finite']}, nonzero "
         f"{grads['short']['n_nonzero']}/{g.size}")
    if not grads["short"]["finite"] or grads["short"]["n_nonzero"] == 0:
        raise SystemExit("A1 failed on the short record: the gradient is "
                         + ("not finite" if not grads["short"]["finite"] else "all zero")
                         + "; stopping before the full record.")

    # the base record and 1.5 x it, through rfx's own record-length witness
    from rfx import gradient_record_length_witness
    _log(f"(c) gradient_record_length_witness, design box, {n_steps} steps and "
         f"{WITNESS_PERIOD_FACTOR} x (tol {WITNESS_TOL}, the helper's norm verdict)")

    def witness_objective(p, n):
        return make_objective(board, "design", f_t, n_steps=n)(p)

    t0 = time.perf_counter()
    w = gradient_record_length_witness(witness_objective, eps_bg, n_steps, tol=WITNESS_TOL,
                                       factor=WITNESS_PERIOD_FACTOR)
    wall["witness_both_arms_s"] = time.perf_counter() - t0
    mem = _device_memory()
    g_full = np.asarray(next(iter(w.grad.values())))[0]
    g_long = np.asarray(next(iter(w.grad_long.values())))[0]
    for tag, g, v, steps in (("full", g_full, w.value[0], w.n_steps),
                             ("long", g_long, w.value_long[0], w.n_steps_long)):
        grads[tag] = _grad_info(g, np.real(v), steps, mem,
                                "gradient_record_length_witness (jax.vjp); memory is the "
                                "process peak through both arms")
        _save_grad(tag, g, np.real(v), steps)
    _save_json(out / "gradient_summary.json", grads)
    wit = {"helper": "rfx.gradient_record_length_witness", "tol": w.tol, "factor": w.factor,
           "n_steps": w.n_steps, "n_steps_long": w.n_steps_long,
           "norm_rel_change": float(w.worst), "cosine": float(w.cosine_by_bin[0]),
           "helper_passed": bool(w.passed), "value": float(np.real(w.value[0])),
           "value_long": float(np.real(w.value_long[0])),
           "worst_value_rel_change": float(w.worst_value_rel_change),
           "worst_elementwise_floored": float(w.worst_elementwise)}
    _save_json(out / "record_length_witness.json", wit)
    _log(f"    helper: ||g_1.5 - g_1.0|| / ||g_1.5|| = {wit['norm_rel_change']:.3e}, cosine "
         f"{wit['cosine']:.8f}, J {wit['value']:.6e} -> {wit['value_long']:.6e}, "
         f"peak {mem['peak_bytes_in_use']/1e9:.2f} GB")
    J_full = make_objective(board, "design", f_t, n_steps=n_steps)

    # ---- (d) block finite differences, float32
    _log("(d) block central differences (float32)")
    Jf = jax.jit(J_full)
    j0 = float(Jf(eps_bg))
    fd = {"precision": "float32", "J0": j0, "blocks": {}}
    for block in BLOCKS:
        name = block[0]
        sl = board.block_cells("design", block)
        entry = {"realized": board.block_realized(block),
                 "ad_sum_full": float(np.sum(g_full[sl])),
                 "ad_sum_long": float(np.sum(g_long[sl])), "fd": {}}
        for h in FD_STEPS:
            jp = float(Jf(eps_bg.at[sl].add(h)))
            jm = float(Jf(eps_bg.at[sl].add(-h)))
            entry["fd"][str(h)] = {"J_plus": jp, "J_minus": jm, "fd": (jp - jm) / (2 * h)}
        fd["blocks"][name] = entry
        _save_json(out / "fd_float32.json", fd)
        _log(f"    {name}: AD {entry['ad_sum_full']:+.5e}  FD "
             + "  ".join(f"h={h}: {entry['fd'][str(h)]['fd']:+.5e}" for h in FD_STEPS))
    judged = _judge_blocks(fd)
    fd["judgement"] = judged
    _save_json(out / "fd_float32.json", fd)
    wj = _judge_witness(fd, judged)
    wit["blocks_float32_cutoff"] = wj
    _save_json(out / "record_length_witness.json", wit)
    _log("    record-length witness per block: " + ", ".join(
        f"{n} {r['rel_change'] if r['rel_change'] is None else format(r['rel_change'], '.2e')}"
        f"{' (judged)' if r['judged'] else ''}" for n, r in wj.items()))
    failed = [n for n, r in wj.items() if r["judged"] and not r["passed"]]
    if failed:
        _save_json(out / "witness_failed.json", {"blocks": failed, "rows": wj})
        _save_json(out / "wall_main.json", wall)
        _log(f"record-length witness FAILED at {failed}: the case stops here (Amendment 1)")
        raise SystemExit(3)
    if judged["roundoff_blocks"]:
        _save_json(out / "x64_needed.json", {
            "reason": "a judged block's ladder moves more between 0.1 and 0.05 than "
                      "between 0.2 and 0.1", "blocks": judged["roundoff_blocks"]})
        _log(f"    round-off rule met at {judged['roundoff_blocks']}: x64 stage required")

    # ---- (e) |E_z(f_t)|^2 in the substrate, same model, DFT plane probes
    _log("(e) |E_z(f_t)|^2 from DFT plane probes either side of the substrate mid-plane")
    sim_e = mod.build(dx)
    z_ground = float(board.zs[board.k_ground])
    # metres from the domain origin, a quarter cell above a node so the index
    # does not tie: the planes are the Ez edges at ground + 1.5 dx and
    # ground + 2.5 dx (0.375 h and 0.625 h at h/4)
    z_rel = [z_ground + (k + 0.25) * dx for k in (N_PER_H // 2 - 1, N_PER_H // 2)]
    for n, z in enumerate(z_rel):
        sim_e.add_dft_plane_probe(axis="z", coordinate=z, component="ez",
                                  freqs=jnp.asarray([f_t, f_r]), name=f"ez_mid{n}")
    t0 = time.perf_counter()
    flank_f = [f_t - FLANK_DF_HZ, f_t, f_t + FLANK_DF_HZ]
    res_e = sim_e.forward(num_periods=num_periods, skip_preflight=True, checkpoint=False,
                          port_s11_freqs=jnp.asarray(flank_f))
    wall["dft_plane_forward_s"] = time.perf_counter() - t0
    planes = {k: np.asarray(v.accumulator) for k, v in res_e.dft_planes.items()}
    idx = {k: int(v.index) for k, v in res_e.dft_planes.items()}
    (ia, ib), (ja, jb), _ = board.boxes["design"]
    ez2 = 0.5 * sum(np.abs(p[0, ia:ib + 1, ja:jb + 1]) ** 2 for p in planes.values())
    ez2_fr = 0.5 * sum(np.abs(p[1, ia:ib + 1, ja:jb + 1]) ** 2 for p in planes.values())
    gmap = g_full.sum(axis=2)
    corr = {"pearson_grad_vs_ez2_ft": _pearson(gmap, ez2),
            "pearson_grad_vs_ez2_fr": _pearson(gmap, ez2_fr),
            "pearson_grad_full_vs_long": _pearson(g_full, g_long),
            "pearson_gmap_full_vs_long": _pearson(gmap, g_long.sum(axis=2)),
            "plane_indices": idx, "plane_z_m": [float(board.zs[k]) + 0.5 * dx for k in idx.values()],
            "method": "forward() with two add_dft_plane_probe(axis='z', component='ez') "
                      "planes at the Ez edges 1.5 and 2.5 cells above the ground; "
                      "|Ez|^2 is their mean; Ez index (i, j) is compared with design "
                      "cell (i, j) of the z-summed gradient over the design box footprint",
            "s11_at_f_t_from_this_forward": complex(np.asarray(res_e.s_params).reshape(-1)[1])}
    s_fl = np.asarray(res_e.s_params, dtype=complex).reshape(-1)
    j_fl = np.abs(s_fl) ** 2
    j_run = np.abs(s11) ** 2
    dj_run = np.gradient(j_run, freqs)
    flank = {"freqs_hz": flank_f, "s11": s_fl, "J": j_fl,
             "abs_s11_ft_db": float(20 * np.log10(np.abs(s_fl[1]))),
             "dJ_df_forward_per_hz": float((j_fl[2] - j_fl[0]) / (2 * FLANK_DF_HZ)),
             "dJ_df_forward_formula": "(|S11(f_t + 1 MHz)|^2 - |S11(f_t - 1 MHz)|^2) / 2 MHz, "
                                      "forward(port_s11_freqs=...) on the board without a box",
             "dJ_df_run_bins_per_hz": float(np.interp(f_t, freqs, dj_run)),
             "dJ_df_run_bins_formula": "numpy.gradient of |S11|^2 over the 901 run() bins, "
                                       "linearly interpolated at f_t",
             "s11_min_hz": res_min["f"], "remax_f0_hz": res_remax["f0_hz"], "f_t_hz": f_t}
    _save_json(out / "flank.json", flank)
    _log(f"    |S11(f_t)| {flank['abs_s11_ft_db']:.3f} dB, d|S11|^2/df "
         f"{flank['dJ_df_forward_per_hz']:.4e} /Hz (forward), {flank['dJ_df_run_bins_per_hz']:.4e} /Hz (run bins)")
    np.savez(out / "ez2_mid.npz", ez2_ft=ez2, ez2_fr=ez2_fr, gmap=gmap,
             planes_ft=np.stack([p[0] for p in planes.values()]),
             x_nodes_m=board.xs[ia:ib + 1], y_nodes_m=board.ys[ja:jb + 1])
    _save_json(out / "correlation.json", corr)
    _log(f"    Pearson(z-summed grad, |Ez(f_t)|^2) = {corr['pearson_grad_vs_ez2_ft']:+.4f}")
    _save_json(out / "wall_main.json", wall)
    _log("main stage done")


def _judge_witness(fd: dict, judgement: dict) -> dict:
    """Amendment 1: |S_1.5 - S_1.0| / |S_1.5| per block, judged on the blocks
    the FD check judges (the same 10 %-of-max cutoff)."""
    rows = {}
    for n, b in fd["blocks"].items():
        s0, s1 = b["ad_sum_full"], b["ad_sum_long"]
        rel = abs(s1 - s0) / abs(s1) if s1 else None
        is_judged = n in judgement["judged"]
        rows[n] = {"sum_1.0x": s0, "sum_1.5x": s1, "rel_change": rel, "judged": is_judged,
                   "passed": (rel is not None and rel <= WITNESS_TOL) if is_judged else None}
    return rows


def _judge_blocks(fd: dict) -> dict:
    fds = {n: b["fd"][str(FD_JUDGED_STEP)]["fd"] for n, b in fd["blocks"].items()}
    fmax = max(abs(v) for v in fds.values())
    out = {"fd_max_abs": fmax, "judged": [], "reported": [], "roundoff_blocks": [], "rows": {}}
    for n, b in fd["blocks"].items():
        f = {h: b["fd"][str(h)]["fd"] for h in FD_STEPS}
        ad = b.get("ad_sum", b.get("ad_sum_full"))
        is_judged = fmax > 0 and abs(fds[n]) >= FD_JUDGE_FRACTION * fmax
        rel = abs(ad - fds[n]) / abs(fds[n]) if fds[n] != 0 else None
        coarse, mid, fine = (f[h] for h in FD_STEPS)
        roundoff = abs(fine - mid) > abs(mid - coarse)
        out["rows"][n] = {"ad": ad, "fd_judged_step": fds[n], "rel": rel,
                          "rel_by_step": {str(h): (abs(ad - f[h]) / abs(f[h]) if f[h] else None)
                                          for h in FD_STEPS},
                          "judged": is_judged,
                          "passed": (rel <= FD_REL_BAR) if is_judged else None,
                          "ladder_roundoff": roundoff}
        (out["judged"] if is_judged else out["reported"]).append(n)
        if is_judged and roundoff:
            out["roundoff_blocks"].append(n)
    return out


# ------------------------------------------------------------------ stage x64fd
def stage_x64fd(out: Path) -> None:
    """The same AD block sums and FD ladder with float64 fields (JAX_ENABLE_X64=1)."""
    import jax
    import jax.numpy as jnp
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("the x64 stage needs JAX_ENABLE_X64=1 in its environment")
    from rfx import Simulation
    mod = load_patch_module()
    # float64 fields: the module's builder calls its module-global Simulation;
    # this process hands it one that defaults to precision="float64".
    mod.Simulation = functools.partial(Simulation, precision="float64")
    dx = mod.H_SUB / N_PER_H
    board = Board(mod, dx)
    assert board.sim._precision == "float64"
    f_t = _load_json(out / "f_t.json")["f_t_hz"]
    n_steps = int(board.grid.num_timesteps(num_periods=float(mod.NUM_PERIODS)))
    eps_bg = jnp.asarray(board.background("design"), dtype=jnp.float64)
    J = make_objective(board, "design", f_t, n_steps=n_steps)
    t0 = time.perf_counter()
    v, g = jax.jit(jax.value_and_grad(J))(eps_bg)
    g = np.asarray(g)
    grad_s = time.perf_counter() - t0
    np.savez(out / "gradient_full_x64.npz", grad=g, J=float(v), n_steps=n_steps, f_t_hz=f_t)
    Jf = jax.jit(J)
    fd = {"precision": "float64", "J0": float(Jf(eps_bg)), "grad_wall_s": grad_s,
          "memory": _device_memory(), "blocks": {}}
    for block in BLOCKS:
        name = block[0]
        sl = board.block_cells("design", block)
        entry = {"realized": board.block_realized(block), "ad_sum": float(np.sum(g[sl])), "fd": {}}
        for h in FD_STEPS:
            jp, jm = float(Jf(eps_bg.at[sl].add(h))), float(Jf(eps_bg.at[sl].add(-h)))
            entry["fd"][str(h)] = {"J_plus": jp, "J_minus": jm, "fd": (jp - jm) / (2 * h)}
        fd["blocks"][name] = entry
        _save_json(out / "fd_float64.json", fd)
        _log(f"    x64 {name}: AD {entry['ad_sum']:+.6e}  FD "
             + "  ".join(f"h={h}: {entry['fd'][str(h)]['fd']:+.6e}" for h in FD_STEPS))
    fd["judgement"] = _judge_blocks(fd)
    _save_json(out / "fd_float64.json", fd)


# ------------------------------------------------------------------ stage timing
def stage_timing(out: Path, box: str, mode: str) -> None:
    import jax
    import jax.numpy as jnp
    mod = load_patch_module()
    dx = mod.H_SUB / N_PER_H
    board = Board(mod, dx)
    f_t = _load_json(out / "f_t.json")["f_t_hz"]
    n_steps = int(board.grid.num_timesteps(num_periods=float(mod.NUM_PERIODS)))
    eps_bg = jnp.asarray(board.background(box))
    J = make_objective(board, box, f_t, n_steps=n_steps)
    fn = jax.jit(J if mode == "forward" else jax.value_and_grad(J))
    rec = {"box": box, "mode": mode, "box_desc": board.describe_box(box), "n_steps": n_steps,
           "grid_cells": board.g["n_cells"], "device_kind": jax.devices()[0].device_kind,
           "preset": os.environ.get("RFX_SHOWCASE_PRESET")}
    try:
        t0 = time.perf_counter()
        r = fn(eps_bg)
        jax.block_until_ready(r)
        rec["first_call_s"] = time.perf_counter() - t0
        ts = []
        for _ in range(TIMING_REPEATS):
            t0 = time.perf_counter()
            r = fn(eps_bg)
            jax.block_until_ready(r)
            ts.append(time.perf_counter() - t0)
        rec["timed_s"] = ts
        rec["wall_s"] = float(np.median(ts))
        rec["compile_s"] = rec["first_call_s"] - rec["wall_s"]
        rec["J"] = float(r if mode == "forward" else r[0])
        rec.update(_device_memory())
        rec["status"] = "ok"
    except Exception as exc:  # an out-of-memory is a measurement, recorded
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:800]}"
    _save_json(out / f"timing_{box}_{mode}.json", rec)
    _log(json.dumps({k: v for k, v in rec.items() if k not in ("box_desc", "error")}))
    if rec["status"] != "ok":
        raise SystemExit(rec["error"])


# ------------------------------------------------------------------ stage finalize
def stage_finalize(out: Path, repo_dir: Path) -> None:
    model = _load_json(out / "model.json")
    base = _load_json(out / "baseline.json")
    ft = _load_json(out / "f_t.json")
    grads = _load_json(out / "gradient_summary.json")
    corr = _load_json(out / "correlation.json")
    fwdrun = _load_json(out / "forward_vs_run.json")
    fd32 = _load_json(out / "fd_float32.json")
    x64 = (out / "fd_float64.json").is_file()
    if (out / "x64_needed.json").is_file():
        # the pre-declared round-off rule fired: the judged FD is the float64
        # one, and a record without it (or from a failed x64 stage) is refused
        rc_path = out / "x64fd.rc"
        rc = rc_path.read_text().strip() if rc_path.is_file() else None
        if rc is not None and rc != "0":
            raise SystemExit(f"x64_needed.json is present and the x64 stage exited {rc}; "
                             "the float64 judgement is not a record, finalize refuses")
        if not x64 or "judgement" not in _load_json(out / "fd_float64.json"):
            raise SystemExit("x64_needed.json is present but fd_float64.json holds no "
                             "judgement; finalize refuses to judge in float32")
    fd_judged = _load_json(out / "fd_float64.json") if x64 else fd32
    precision_judged = "float64" if x64 else "float32"
    claims = [
        _record.claim("rfx |S11| minimum frequency at h/4", base["s11_minimum"]["f_hz"], "Hz",
                      "baseline.json"),
        _record.claim("rfx |S11| minimum depth at h/4", base["s11_minimum"]["depth_db"], "dB",
                      "baseline.json"),
        _record.claim("rfx Re(Zin) peak frequency f0 at h/4", base["remax"]["f0_hz"], "Hz",
                      "baseline.json"),
        _record.claim("f_r used for f_t (" + ft["estimator"] + ")", ft["f_r_hz"], "Hz", "f_t.json"),
        _record.claim("f_t = 1.01 f_r", ft["f_t_hz"], "Hz", "f_t.json"),
        _record.claim("ring-down settling of the baseline record", base["settling_db"], "dB",
                      "baseline.json"),
        _record.claim("max |S11_forward - S11_run| over the 901 bins",
                      fwdrun["max_abs_diff_vs_run"], "1", "forward_vs_run.json"),
        _record.claim("J = |S11(f_t)|^2 at the baseline", grads["full"]["J"], "1",
                      "gradient_summary.json"),
        _record.claim("A1: short-record gradient finite and nonzero cells",
                      grads["short"]["n_nonzero"], "cells", "gradient_summary.json"),
        _record.claim("peak device memory of the main process through the witness's two arms "
                      "(1.0x and 1.5x record, design box)",
                      grads["full"]["memory"]["peak_bytes_in_use"], "B", "gradient_summary.json"),
        _record.claim("Pearson r, z-summed dJ/deps_r map vs |Ez(f_t)|^2 over the design footprint",
                      corr["pearson_grad_vs_ez2_ft"], "1", "correlation.json"),
        _record.claim("Pearson r, dJ/deps_r at 1.0x vs 1.5x record (all design cells)",
                      corr["pearson_grad_full_vs_long"], "1", "correlation.json"),
    ]
    rows = fd_judged["judgement"]["rows"]
    wit = "fd_float64.json" if x64 else "fd_float32.json"
    for name, r in rows.items():
        q = f"|AD - FD| / |FD| at d eps_r = {FD_JUDGED_STEP}, block {name} ({precision_judged})"
        if r["judged"]:
            claims.append(_record.claim(q, r["rel"], "1", wit, threshold=FD_REL_BAR,
                                        rule="pre-declared: judged blocks, "
                                             "|FD| >= 0.1 x the largest |FD| of the six"))
        else:
            claims.append(_record.claim(q, r["rel"], "1", wit,
                                        note="|FD| below 0.1 x the largest |FD|: reported"))
    if x64:
        for name, r in fd32["judgement"]["rows"].items():
            claims.append(_record.claim(
                f"|AD - FD| / |FD| at d eps_r = {FD_JUDGED_STEP}, block {name} (float32)",
                r["rel"], "1", "fd_float32.json"))
    # Amendment 1: the record-length witness, judged per block on the blocks
    # the FD check judges (cutoff from the precision the FD check was judged in)
    rlw = _load_json(out / "record_length_witness.json")
    wrows = _judge_witness(fd32, fd_judged["judgement"])
    rlw["blocks_judged_cutoff"] = wrows
    rlw["cutoff_precision"] = precision_judged
    _save_json(out / "record_length_witness.json", rlw)
    for name, r in wrows.items():
        q = (f"record-length witness, block {name}: |S_1.5x - S_1.0x| / |S_1.5x| of the AD "
             f"block sum ({rlw['n_steps']} vs {rlw['n_steps_long']} steps)")
        if r["judged"]:
            claims.append(_record.claim(q, r["rel_change"], "1", "record_length_witness.json",
                                        threshold=WITNESS_TOL,
                                        rule="Amendment 1 (lane leader): judged on the blocks "
                                             "the FD check judges"))
        else:
            claims.append(_record.claim(q, r["rel_change"], "1", "record_length_witness.json",
                                        note="block below the FD cutoff: reported"))
    claims += [
        _record.claim("record-length witness, helper norm ||g_1.5x - g_1.0x|| / ||g_1.5x|| "
                      "over all design cells", rlw["norm_rel_change"], "1",
                      "record_length_witness.json",
                      note=f"gradient_record_length_witness(tol=0.05) passed: {rlw['helper_passed']}"),
        _record.claim("record-length witness, helper cosine between the two gradients",
                      rlw["cosine"], "1", "record_length_witness.json"),
        _record.claim("Pearson r, z-summed dJ/deps_r maps at 1.0x vs 1.5x record",
                      corr["pearson_gmap_full_vs_long"], "1", "correlation.json"),
    ]
    flank = _load_json(out / "flank.json")
    claims += [
        _record.claim("|S11(f_t)| from forward() on the board without a box",
                      flank["abs_s11_ft_db"], "dB", "flank.json"),
        _record.claim("d|S11|^2/df at f_t, central difference over f_t +- 1 MHz (forward)",
                      flank["dJ_df_forward_per_hz"], "1/Hz", "flank.json"),
        _record.claim("d|S11|^2/df at f_t, gradient over the 901 run() bins",
                      flank["dJ_df_run_bins_per_hz"], "1/Hz", "flank.json"),
    ]
    derived = []
    timing = {}
    for box in TIMING_BOXES:
        for mode in ("forward", "grad"):
            p = out / f"timing_{box}_{mode}.json"
            if p.is_file():
                timing[(box, mode)] = _load_json(p)
    for box in TIMING_BOXES:
        tf, tg = timing.get((box, "forward")), timing.get((box, "grad"))
        if not tf or not tg or tf["status"] != "ok" or tg["status"] != "ok":
            continue
        n = tf["box_desc"]["n_cells"]
        w = f"timing_{box}_forward.json"
        claims += [
            _record.claim(f"forward wall time, box {box} ({n} cells)", tf["wall_s"], "s", w),
            _record.claim(f"forward compile time, box {box}", tf["compile_s"], "s", w),
            _record.claim(f"forward peak device memory, box {box}", tf["peak_bytes_in_use"], "B", w),
            _record.claim(f"value_and_grad wall time, box {box}", tg["wall_s"], "s",
                          f"timing_{box}_grad.json"),
            _record.claim(f"value_and_grad compile time, box {box}", tg["compile_s"], "s",
                          f"timing_{box}_grad.json"),
            _record.claim(f"value_and_grad peak device memory, box {box}",
                          tg["peak_bytes_in_use"], "B", f"timing_{box}_grad.json"),
        ]
        derived.append(_record.derived(
            f"central finite-difference cost for every cell of box {box}",
            2 * n * tf["wall_s"], "s", "2 x N_cells x forward wall time",
            {"N_cells": n, "forward_wall_s": tf["wall_s"], "witness": w}))
        derived.append(_record.derived(
            f"value_and_grad / forward wall-time ratio, box {box}",
            tg["wall_s"] / tf["wall_s"], "1", "value_and_grad wall / forward wall",
            {"grad_wall_s": tg["wall_s"], "forward_wall_s": tf["wall_s"]}))
    derived.append(_record.derived(
        "f_t", ft["f_t_hz"], "Hz", "1.01 x f_r", {"f_r_hz": ft["f_r_hz"]}))
    files = _record.data_files(out)
    rec = {
        "schema": _record.SCHEMA, "id": CASE_ID, "question": QUESTION,
        "source": _record.source_block(repo_dir, precision=precision_judged + " (judged FD); "
                                       "float32 (map, timing)"),
        "run": {"platform": "VESSL", "preset": os.environ.get("RFX_SHOWCASE_PRESET"),
                "run_id": None, "wall_s": _load_json(out / "wall_main.json"),
                "timing_devices": sorted({t["device_kind"] for t in timing.values()})},
        "model": model,
        "claims": claims, "derived": derived,
        "notes": ["timing box 'layer' is the substrate z-layer across the whole physical "
                  "interior; it includes cells that are not laminate (model.boxes.layer."
                  "n_laminate_cells / n_other_cells)"],
        "out_of_scope": [
            "the openEMS comparison of this board (tests/crossval/rt5880_patch judges it)",
            "mesh convergence of the gradient map: one rung, h/4",
            "any interpretation of the map",
        ],
    }
    path = _record.write_result(out, rec, files)
    _log(f"wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["main", "x64fd", "timing", "finalize"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--box", choices=TIMING_BOXES)
    ap.add_argument("--mode", choices=["forward", "grad"])
    ap.add_argument("--repo-dir", type=Path, default=REPO)
    ap.add_argument("--smoke", action="store_true", help="plumbing check only")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    global SMOKE
    SMOKE = a.smoke
    if a.stage == "main":
        stage_main(a.out)
    elif a.stage == "x64fd":
        stage_x64fd(a.out)
    elif a.stage == "timing":
        if not (a.box and a.mode):
            ap.error("--stage timing needs --box and --mode")
        stage_timing(a.out, a.box, a.mode)
    else:
        stage_finalize(a.out, a.repo_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
