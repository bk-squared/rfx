"""#801 field-dump reductions. CPU only, reads dumps/<label>/*.npz + meta.json, no jax.

For each label under dumps/ (see LABELS in dump_fields.py) this writes:
  dumps/<label>/summary.json   -- the five pre-declared numeric reductions
  dumps/<label>/slice_z.png, slice_x.png, slice_y.png, profiles.png
  dumps/TABLE.md                -- one table across all labels, numbers only

Energy density: w = eps0*eps_r*(ex^2+ey^2+ez^2) + mu0*mu_r*(hx^2+hy^2+hz^2) per cell, each
component read from its OWN array at the SAME (i, j, k) index -- E and H (and the three E
components, and the three H components) live at different Yee half-cell offsets in this
codebase's storage convention, and this script does not interpolate them onto a common point.
So `w` here is a per-index combination, not a physically registered energy density; every
number derived from it inherits that.

eps_r / mu_r / the PEC mask are not in the npz dumps (only field arrays are); this script
rebuilds them by calling the same tree's oracle._build(...) + Simulation._build_grid() +
Simulation._assemble_materials(grid) again -- no time-stepping, so this is cheap on CPU and
byte-for-byte the same materials the GPU run used, since both come from the same deterministic
builder. Rebuilding per label in one process requires purging the previously imported `rfx`/
`tests` modules first (each label may come from a different exported tree); see
_fresh_materials.

Usage: python reduce_dumps.py [--dumps-root PATH] [--labels a,b,c]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

BASE = "/root/workspace/bk-workspace/.801-measure"

# SI CODATA constants -- same two floats as rfx/core/yee.py:EPS_0/MU_0 in both trees (verified
# by grep; hardcoded here rather than re-importing per label since that module is unaffected by
# the two trees' rfx/boundaries/cpml.py difference and this avoids one more purge/reimport for
# two numbers).
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6

sys.path.insert(0, BASE)
from dump_fields import LABELS  # noqa: E402


def _fresh_materials(tree_name: str, ceil_sizing: bool, n: int, pad_h: int, cpml: int):
    """Rebuild eps_r, mu_r, pec_mask for one label, purging any previously imported tree."""
    for mod in [m for m in list(sys.modules)
                if m == "rfx" or m.startswith("rfx.") or m == "tests" or m.startswith("tests.")]:
        del sys.modules[mod]
    sys.path[:] = [p for p in sys.path if not p.startswith(os.path.join(BASE, "src-"))]
    tree_root = os.path.join(BASE, tree_name)
    sys.path.insert(0, tree_root)

    import rfx  # noqa: F401
    from tests.oracle import test_lossless_open_domain_ringdown_does_not_grow as oracle

    if ceil_sizing:
        import rfx.grid as grid_mod
        grid_mod.cells_spanning = lambda length, dx, **_kw: int(math.ceil(length / dx))

    sim = oracle._build(n=n, pad_h=pad_h, cpml=cpml)
    grid = sim._build_grid()
    mats_out = sim._assemble_materials(grid)
    mats, pec_mask = mats_out[0], mats_out[3]
    eps_r = np.asarray(mats.eps_r, dtype=float)
    mu_r = np.asarray(mats.mu_r, dtype=float)
    pec_mask_np = (np.asarray(pec_mask, dtype=bool) if pec_mask is not None
                   else np.zeros(eps_r.shape, dtype=bool))
    return eps_r, mu_r, pec_mask_np


def _w(state, eps_r, mu_r):
    e2 = state["ex"].astype(np.float64) ** 2 + state["ey"].astype(np.float64) ** 2 \
        + state["ez"].astype(np.float64) ** 2
    h2 = state["hx"].astype(np.float64) ** 2 + state["hy"].astype(np.float64) ** 2 \
        + state["hz"].astype(np.float64) ** 2
    return EPS0 * eps_r * e2 + MU0 * mu_r * h2


def _region_masks(shape, face_pads):
    nx, ny, nz = shape
    ix = np.arange(nx)[:, None, None]
    iy = np.arange(ny)[None, :, None]
    iz = np.arange(nz)[None, None, :]
    in_x = (ix < face_pads["x_lo"]) | (ix >= nx - face_pads["x_hi"])
    in_y = (iy < face_pads["y_lo"]) | (iy >= ny - face_pads["y_hi"])
    in_z = (iz < face_pads["z_lo"]) | (iz >= nz - face_pads["z_hi"])
    count = in_x.astype(int) + in_y.astype(int) + in_z.astype(int)
    interior = count == 0
    faces = {
        "x_lo": (count == 1) & in_x & (ix < face_pads["x_lo"]),
        "x_hi": (count == 1) & in_x & (ix >= nx - face_pads["x_hi"]),
        "y_lo": (count == 1) & in_y & (iy < face_pads["y_lo"]),
        "y_hi": (count == 1) & in_y & (iy >= ny - face_pads["y_hi"]),
        "z_lo": (count == 1) & in_z & (iz < face_pads["z_lo"]),
        "z_hi": (count == 1) & in_z & (iz >= nz - face_pads["z_hi"]),
    }
    edge = count == 2
    corner = count == 3
    return interior, faces, edge, corner


def _region_name(i, j, k, face_pads, nx, ny, nz):
    in_x = i < face_pads["x_lo"] or i >= nx - face_pads["x_hi"]
    in_y = j < face_pads["y_lo"] or j >= ny - face_pads["y_hi"]
    in_z = k < face_pads["z_lo"] or k >= nz - face_pads["z_hi"]
    count = int(in_x) + int(in_y) + int(in_z)
    if count == 0:
        return "interior"
    if count == 3:
        return "corner_overlap"
    if count == 2:
        return "edge_overlap"
    if in_x:
        return "x_lo" if i < face_pads["x_lo"] else "x_hi"
    if in_y:
        return "y_lo" if j < face_pads["y_lo"] else "y_hi"
    return "z_lo" if k < face_pads["z_lo"] else "z_hi"


def _dist_to_domain_face(i, j, k, nx, ny, nz):
    return int(min(i, nx - 1 - i, j, ny - 1 - j, k, nz - 1 - k))


def _dist_to_absorber_inner_boundary(i, j, k, face_pads, nx, ny, nz):
    candidates = []
    if face_pads["x_lo"] > 0:
        candidates.append(abs(i - face_pads["x_lo"]))
    if face_pads["x_hi"] > 0:
        candidates.append(abs(i - (nx - 1 - face_pads["x_hi"])))
    if face_pads["y_lo"] > 0:
        candidates.append(abs(j - face_pads["y_lo"]))
    if face_pads["y_hi"] > 0:
        candidates.append(abs(j - (ny - 1 - face_pads["y_hi"])))
    if face_pads["z_lo"] > 0:
        candidates.append(abs(k - face_pads["z_lo"]))
    if face_pads["z_hi"] > 0:
        candidates.append(abs(k - (nz - 1 - face_pads["z_hi"])))
    return int(min(candidates)) if candidates else None


def _dist_to_pec(i, j, k, pec_idx):
    if pec_idx.shape[0] == 0:
        return None
    d = pec_idx - np.array([i, j, k])
    return float(np.sqrt((d ** 2).sum(axis=1)).min())


def _sign_alt_fraction(line):
    s = np.sign(line)
    prod = s[:-1] * s[1:]
    finite = prod != 0
    if finite.sum() == 0:
        return None
    return float((prod[finite] < 0).mean())


def _pearson(a, b):
    a = a.ravel()
    b = b.ravel()
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def process_label(label: str, dumps_root: str):
    out_dir = os.path.join(dumps_root, label)
    with open(os.path.join(out_dir, "meta.json")) as fh:
        meta = json.load(fh)
    cfg = LABELS[meta["label"]] if meta["label"] in LABELS else LABELS[label]
    nx, ny, nz = meta["grid_shape"]
    face_pads = meta["face_pads"]

    eps_r, mu_r, pec_mask = _fresh_materials(cfg["tree"], cfg["ceil"], cfg["n"], cfg["pad_h"], cfg["cpml"])
    if eps_r.shape != (nx, ny, nz):
        raise SystemExit(f"{label}: rebuilt materials shape {eps_r.shape} != dumped grid {(nx, ny, nz)}")

    states = {}
    for tag in ("final", "snap_85", "snap_90", "snap_95"):
        fname = "final_state.npz" if tag == "final" else f"{tag}.npz"
        states[tag] = dict(np.load(os.path.join(out_dir, fname)))

    w = {tag: _w(st, eps_r, mu_r) for tag, st in states.items()}
    total_w = {tag: float(arr.sum()) for tag, arr in w.items()}

    # (1) region fractions, on the FINAL field.
    interior, faces, edge, corner = _region_masks((nx, ny, nz), face_pads)
    w_final = w["final"]
    tw = total_w["final"]
    region_fracs = dict(interior=float(w_final[interior].sum() / tw) if tw else None)
    for name, mask in faces.items():
        region_fracs[name] = float(w_final[mask].sum() / tw) if tw else None
    region_fracs["edge_overlap"] = float(w_final[edge].sum() / tw) if tw else None
    region_fracs["corner_overlap"] = float(w_final[corner].sum() / tw) if tw else None
    region_fracs["_sum_check"] = float(sum(v for k, v in region_fracs.items() if not k.startswith("_")))

    # (2) cell of max |E|, on the FINAL field.
    e_mag = np.sqrt(states["final"]["ex"].astype(np.float64) ** 2
                     + states["final"]["ey"].astype(np.float64) ** 2
                     + states["final"]["ez"].astype(np.float64) ** 2)
    flat_idx = int(np.argmax(e_mag))
    i0, j0, k0 = np.unravel_index(flat_idx, e_mag.shape)
    pec_idx = np.argwhere(pec_mask)
    max_e_cell = dict(
        index=[int(i0), int(j0), int(k0)],
        value=float(e_mag[i0, j0, k0]),
        region=_region_name(i0, j0, k0, face_pads, nx, ny, nz),
        dist_to_domain_face_cells=_dist_to_domain_face(i0, j0, k0, nx, ny, nz),
        dist_to_absorber_inner_boundary_cells=_dist_to_absorber_inner_boundary(
            i0, j0, k0, face_pads, nx, ny, nz),
        dist_to_pec_cells=_dist_to_pec(i0, j0, k0, pec_idx),
    )

    # (3) 1-D profiles of w, on the FINAL field.
    profiles = dict(
        x=w_final.sum(axis=(1, 2)).tolist(),
        y=w_final.sum(axis=(0, 2)).tolist(),
        z=w_final.sum(axis=(0, 1)).tolist(),
    )

    # (4) snapshot-pair correlation / ratio / implied growth rate.
    steps_at = dict(
        snap_85=meta["snapshot_steps"]["85"]["actual"],
        snap_90=meta["snapshot_steps"]["90"]["actual"],
        snap_95=meta["snapshot_steps"]["95"]["actual"],
        final=meta["steps"],
    )
    pairs = [("snap_85", "snap_90"), ("snap_90", "snap_95"), ("snap_95", "final")]
    pair_stats = {}
    for a, b in pairs:
        wa, wb = w[a], w[b]
        na = wa / wa.sum() if wa.sum() else wa
        nb = wb / wb.sum() if wb.sum() else wb
        ratio = (total_w[b] / total_w[a]) if total_w[a] else None
        dstep = steps_at[b] - steps_at[a]
        rate = (math.log(ratio) / dstep) if (ratio and ratio > 0 and dstep) else None
        pair_stats[f"{a}_to_{b}"] = dict(
            correlation_normalized_w=_pearson(na, nb),
            total_w_ratio=ratio,
            steps_between=dstep,
            implied_growth_rate_per_step=rate,
        )

    # (5) adjacent-cell sign alternation through the max-|E| cell, dominant component there.
    comps = {"ex": states["final"]["ex"], "ey": states["final"]["ey"], "ez": states["final"]["ez"]}
    dom_name = max(comps, key=lambda c: abs(comps[c][i0, j0, k0]))
    dom = comps[dom_name]
    sign_alt = dict(
        dominant_component=dom_name,
        x=_sign_alt_fraction(dom[:, j0, k0]),
        y=_sign_alt_fraction(dom[i0, :, k0]),
        z=_sign_alt_fraction(dom[i0, j0, :]),
    )

    summary = dict(
        label=label,
        tree=meta["tree"],
        arm=meta["arm"],
        grid_shape=meta["grid_shape"],
        settling_db=meta["settling_db"],
        worst_rate_per_step=meta["worst_rate_per_step"],
        note="w = eps0*eps_r*|E|^2 + mu0*mu_r*|H|^2 per cell, components co-located by index, "
             "not interpolated. Region fractions, profiles and the max-|E| cell are all on the "
             "FINAL field.",
        region_fractions=region_fracs,
        max_e_cell=max_e_cell,
        profiles_length=dict(x=nx, y=ny, z=nz),
        snapshot_pairs=pair_stats,
        sign_alternation=sign_alt,
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(out_dir, "profiles.json"), "w") as fh:
        json.dump(profiles, fh, indent=1)

    _make_plots(label, out_dir, w_final, pec_mask, face_pads, max_e_cell, profiles, meta)
    print(f"{label}: summary.json + profiles.json + 4 PNGs written to {out_dir}", flush=True)
    return summary


def _make_plots(label, out_dir, w_final, pec_mask, face_pads, max_e_cell, profiles, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nx, ny, nz = w_final.shape
    i0, j0, k0 = max_e_cell["index"]
    log_w = np.log10(np.maximum(w_final, 1e-300))

    # z-plane through the substrate middle: midway between the two PEC blocks' z indices.
    pec_z = sorted(set(int(z) for z in np.argwhere(pec_mask)[:, 2])) if pec_mask.any() else []
    z_mid = int(round(sum(pec_z) / len(pec_z))) if pec_z else nz // 2

    def _panel(ax, plane2d, pec2d, axis_labels, hline_idx, vline_idx, title):
        im = ax.imshow(plane2d.T, origin="lower", aspect="auto", cmap="viridis")
        ax.contour(pec2d.T, levels=[0.5], colors="red", linewidths=0.8)
        for h in hline_idx:
            ax.axhline(h, color="white", linewidth=0.5, linestyle="--")
        for v in vline_idx:
            ax.axvline(v, color="white", linewidth=0.5, linestyle="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig, ax = plt.subplots(figsize=(6, 5))
    _panel(ax, log_w[:, :, z_mid], pec_mask[:, :, z_mid], ("x index", "y index"),
           hline_idx=[face_pads["y_lo"], ny - 1 - face_pads["y_hi"]],
           vline_idx=[face_pads["x_lo"], nx - 1 - face_pads["x_hi"]],
           title=f"{label}: log10(w), z={z_mid} (substrate middle)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "slice_z.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    _panel(ax, log_w[i0, :, :], pec_mask[i0, :, :], ("y index", "z index"),
           hline_idx=[face_pads["z_lo"], nz - 1 - face_pads["z_hi"]],
           vline_idx=[face_pads["y_lo"], ny - 1 - face_pads["y_hi"]],
           title=f"{label}: log10(w), x={i0} (through max|E|)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "slice_x.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    _panel(ax, log_w[:, j0, :], pec_mask[:, j0, :], ("x index", "z index"),
           hline_idx=[face_pads["z_lo"], nz - 1 - face_pads["z_hi"]],
           vline_idx=[face_pads["x_lo"], nx - 1 - face_pads["x_hi"]],
           title=f"{label}: log10(w), y={j0} (through max|E|)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "slice_y.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, axis_name in zip(axes, ("x", "y", "z")):
        ax.plot(profiles[axis_name])
        ax.set_title(f"{label}: w profile along {axis_name}", fontsize=9)
        ax.set_xlabel(f"{axis_name} index")
        ax.set_ylabel("sum w over other two axes")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "profiles.png"), dpi=130)
    plt.close(fig)


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}g}"


def write_table(summaries: dict, dumps_root: str):
    cols = [
        "label", "settling_db", "worst_rate",
        "interior", "x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi", "edge", "corner",
        "maxE_region", "d_face", "d_absorber", "d_pec",
        "corr_85_90", "corr_90_95", "corr_95_final",
        "rate_85_90", "rate_90_95", "rate_95_final",
        "signalt_x", "signalt_y", "signalt_z",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for label, s in summaries.items():
        rf = s["region_fractions"]
        me = s["max_e_cell"]
        sp = s["snapshot_pairs"]
        sa = s["sign_alternation"]
        row = [
            label, _fmt(s["settling_db"]), _fmt(s["worst_rate_per_step"]),
            _fmt(rf["interior"]), _fmt(rf["x_lo"]), _fmt(rf["x_hi"]),
            _fmt(rf["y_lo"]), _fmt(rf["y_hi"]), _fmt(rf["z_lo"]), _fmt(rf["z_hi"]),
            _fmt(rf["edge_overlap"]), _fmt(rf["corner_overlap"]),
            me["region"], _fmt(me["dist_to_domain_face_cells"]),
            _fmt(me["dist_to_absorber_inner_boundary_cells"]), _fmt(me["dist_to_pec_cells"]),
            _fmt(sp["snap_85_to_snap_90"]["correlation_normalized_w"]),
            _fmt(sp["snap_90_to_snap_95"]["correlation_normalized_w"]),
            _fmt(sp["snap_95_to_final"]["correlation_normalized_w"]),
            _fmt(sp["snap_85_to_snap_90"]["implied_growth_rate_per_step"]),
            _fmt(sp["snap_90_to_snap_95"]["implied_growth_rate_per_step"]),
            _fmt(sp["snap_95_to_final"]["implied_growth_rate_per_step"]),
            _fmt(sa["x"]), _fmt(sa["y"]), _fmt(sa["z"]),
        ]
        lines.append("| " + " | ".join(row) + " |")
    path = os.path.join(dumps_root, "TABLE.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps-root", default=os.path.join(BASE, "dumps"))
    ap.add_argument("--labels", default=None, help="comma-separated; default = all of LABELS")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else sorted(LABELS)
    summaries = {}
    for label in labels:
        out_dir = os.path.join(args.dumps_root, label)
        if not os.path.isdir(out_dir):
            print(f"SKIP {label}: no directory {out_dir}", flush=True)
            continue
        summaries[label] = process_label(label, args.dumps_root)

    if summaries:
        table_path = write_table(summaries, args.dumps_root)
        print(f"wrote {table_path}", flush=True)
    else:
        print("no labels processed; TABLE.md not written", flush=True)


if __name__ == "__main__":
    main()
