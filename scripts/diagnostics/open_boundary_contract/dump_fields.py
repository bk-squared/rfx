"""#801 field dumps: full final field state plus three truncated-record states,
for one (tree, arm) pair, on the committed ring-down oracle's own builder.

Usage:
    python dump_fields.py <label> [--n-steps N] [--out-label NAME] [--dumps-root PATH]

<label> is one of the five keys in LABELS below. Each run:
  1. builds the arm with tests/oracle/test_lossless_open_domain_ringdown_does_not_grow._build
     (the same builder run_arms.py and the committed gate use), from the exported tree
     named in LABELS;
  2. runs it to the FULL record (num_periods=oracle.NUM_PERIODS, i.e. the same 150-period
     record run_arms.py scored) and keeps that run's final FDTDState;
  3. re-runs the SAME deterministic builder three more times with n_steps set to 85%, 90%
     and 95% of the full record's step count, keeping each run's final FDTDState. This is
     the brief's A2 fallback, not SnapshotSpec: the FDTD update is a deterministic scan from
     a zero initial state with no randomness, so a fresh run truncated at step S produces the
     exact field the full run had at step S; three separate short runs are cheaper in peak
     memory than holding 20 full 3-D six-component snapshots in device memory at once to land
     near 85/90/95% by SnapshotSpec's fixed interval.
  4. writes final_state.npz, snap_85.npz, snap_90.npz, snap_95.npz (six float32 arrays each,
     plus the step count actually reached) and meta.json under dumps/<out-label>/.

--n-steps overrides step 2 with a fixed short record (bypassing num_periods) for the CPU
smoke test; the 85/90/95% split is then taken of that short record, exercising the same
code path. --out-label overrides the output directory name (the smoke test uses
"_cpu_smoke" so it never collides with a real label's directory).

Environment: CEIL_SIZING is not read here (unlike run_arms.py) because the label
"n2_pad0_main_ceil" already carries that choice explicitly, so a fixed label always
means a fixed build regardless of what is set in the calling shell.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

BASE = "/root/workspace/bk-workspace/.801-measure"

# label -> (exported tree dir name under BASE, n cells/h, lateral pad in h, cpml layers,
#           whether to put the pre-#1136 ceil grid sizing back)
LABELS = {
    "n2_pad10_main":     dict(tree="src-main", n=2, pad_h=10, cpml=4, ceil=False),
    "n2_pad0_main":      dict(tree="src-main", n=2, pad_h=0, cpml=4, ceil=False),
    "n2_pad0_main_ceil": dict(tree="src-main", n=2, pad_h=0, cpml=4, ceil=True),
    "n3_pad10_main":     dict(tree="src-main", n=3, pad_h=10, cpml=6, ceil=False),
    "n3_pad10_main1012": dict(tree="src-main-plus-1012", n=3, pad_h=10, cpml=6, ceil=False),
}


def _import_tree(tree_name: str, ceil_sizing: bool):
    tree_root = os.path.join(BASE, tree_name)
    sys.path.insert(0, tree_root)
    import jax  # noqa: E402
    import rfx  # noqa: E402
    from tests.oracle import test_lossless_open_domain_ringdown_does_not_grow as oracle  # noqa: E402

    if ceil_sizing:
        import rfx.grid as grid_mod
        if not hasattr(grid_mod, "cells_spanning"):
            raise SystemExit("ceil sizing requested but this tree has no rfx.grid.cells_spanning")
        grid_mod.cells_spanning = lambda length, dx, **_kw: int(math.ceil(length / dx))

    print(f"rfx from {rfx.__file__} | devices {jax.devices()} | x64 {jax.config.jax_enable_x64} "
          f"| ceil_sizing {ceil_sizing}", flush=True)
    return rfx, oracle, tree_root


def _run(oracle, n, pad_h, cpml, *, n_steps=None, num_periods=None):
    sim = oracle._build(n=n, pad_h=pad_h, cpml=cpml)
    kwargs = dict(skip_preflight=True)
    if n_steps is not None:
        kwargs["n_steps"] = int(n_steps)
    else:
        kwargs["num_periods"] = num_periods
    result = sim.run(**kwargs)
    return sim, result


def _save_state(path, state):
    np.savez(
        path,
        ex=np.asarray(state.ex, dtype=np.float32),
        ey=np.asarray(state.ey, dtype=np.float32),
        ez=np.asarray(state.ez, dtype=np.float32),
        hx=np.asarray(state.hx, dtype=np.float32),
        hy=np.asarray(state.hy, dtype=np.float32),
        hz=np.asarray(state.hz, dtype=np.float32),
        step=np.asarray(state.step),
    )


def _pec_blocks(pec_mask_np: np.ndarray):
    """Connected components of the True cells, each as an inclusive index box."""
    from scipy import ndimage
    labeled, n_blocks = ndimage.label(pec_mask_np)
    blocks = []
    for lbl in range(1, n_blocks + 1):
        idx = np.argwhere(labeled == lbl)
        blocks.append(dict(
            id=lbl, n_cells=int(idx.shape[0]),
            x=[int(idx[:, 0].min()), int(idx[:, 0].max())],
            y=[int(idx[:, 1].min()), int(idx[:, 1].max())],
            z=[int(idx[:, 2].min()), int(idx[:, 2].max())],
        ))
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", choices=sorted(LABELS))
    ap.add_argument("--n-steps", type=int, default=None,
                     help="Override the full record length (bypasses num_periods). CPU smoke test only.")
    ap.add_argument("--out-label", default=None,
                     help="Output subdirectory name under dumps/; defaults to <label>.")
    ap.add_argument("--dumps-root", default=os.path.join(BASE, "dumps"))
    args = ap.parse_args()

    cfg = LABELS[args.label]
    out_label = args.out_label or args.label
    out_dir = os.path.join(args.dumps_root, out_label)
    if os.path.exists(out_dir):
        raise SystemExit(f"refusing to overwrite existing directory: {out_dir}")

    rfx, oracle, tree_root = _import_tree(cfg["tree"], cfg["ceil"])

    n, pad_h, cpml = cfg["n"], cfg["pad_h"], cfg["cpml"]

    print(f"label={args.label} tree={cfg['tree']} n={n} pad_h={pad_h} cpml={cpml} "
          f"ceil={cfg['ceil']} n_steps_override={args.n_steps}", flush=True)

    if args.n_steps is not None:
        sim_full, result_full = _run(oracle, n, pad_h, cpml, n_steps=args.n_steps)
    else:
        sim_full, result_full = _run(oracle, n, pad_h, cpml, num_periods=oracle.NUM_PERIODS)
    total_steps = int(np.asarray(result_full.time_series).shape[0])
    print(f"full run: {total_steps} steps", flush=True)

    snap_steps = {pct: int(round(pct / 100.0 * total_steps)) for pct in (85, 90, 95)}
    snap_results = {}
    for pct, steps in snap_steps.items():
        _, res = _run(oracle, n, pad_h, cpml, n_steps=steps)
        actual = int(np.asarray(res.time_series).shape[0])
        if actual != steps:
            print(f"WARNING: snap_{pct} asked for {steps} steps, run returned {actual}", flush=True)
        snap_results[pct] = (steps, actual, res)
        print(f"snap_{pct}: requested {steps} steps, got {actual}", flush=True)

    os.makedirs(out_dir)

    _save_state(os.path.join(out_dir, "final_state.npz"), result_full.state)
    for pct, (_, _, res) in snap_results.items():
        _save_state(os.path.join(out_dir, f"snap_{pct}.npz"), res.state)

    grid = sim_full._build_grid()
    mats_out = sim_full._assemble_materials(grid)
    mats, pec_mask = mats_out[0], mats_out[3]
    eps_r_np = np.asarray(mats.eps_r, dtype=float)
    eps_r_unique = sorted(float(v) for v in np.unique(np.round(eps_r_np, 6)))

    pec_mask_note = None
    if pec_mask is None:
        pec_blocks = []
        pec_mask_note = "sim._assemble_materials returned pec_mask=None; A3 did not hold as assumed."
    else:
        pec_mask_np = np.asarray(pec_mask, dtype=bool)
        pec_blocks = _pec_blocks(pec_mask_np)

    settling_db = oracle._settling_db(result_full.time_series)
    rates_per_step = oracle._late_time_log_rate_per_step(result_full.time_series)

    meta = dict(
        label=args.label,
        out_label=out_label,
        tree=cfg["tree"],
        tree_root=tree_root,
        arm=dict(n=n, pad_h=pad_h, cpml_layers=cpml, ceil_sizing=cfg["ceil"]),
        grid_shape=[int(grid.nx), int(grid.ny), int(grid.nz)],
        dx=float(grid.dx),
        dt=float(grid.dt),
        steps=total_steps,
        n_steps_override=args.n_steps,
        snapshot_steps={str(pct): dict(requested=req, actual=act)
                         for pct, (req, act, _res) in snap_results.items()},
        face_pads=dict(x_lo=int(grid.pad_x_lo), x_hi=int(grid.pad_x_hi),
                        y_lo=int(grid.pad_y_lo), y_hi=int(grid.pad_y_hi),
                        z_lo=int(grid.pad_z_lo), z_hi=int(grid.pad_z_hi)),
        settling_db=float(settling_db),
        rates_per_step=[float(r) for r in rates_per_step],
        worst_rate_per_step=float(max(rates_per_step)),
        pec_mask_blocks=pec_blocks,
        pec_mask_note=pec_mask_note,
        eps_r_unique=eps_r_unique,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    print(f"wrote {out_dir}: final_state.npz, snap_85.npz, snap_90.npz, snap_95.npz, meta.json",
          flush=True)
    for name in ("final_state.npz", "snap_85.npz", "snap_90.npz", "snap_95.npz", "meta.json"):
        p = os.path.join(out_dir, name)
        print(f"  {name}: {os.path.getsize(p)} bytes", flush=True)


if __name__ == "__main__":
    main()
