#!/usr/bin/env python3
"""Where is the electrical edge of a PEC sheet on the Yee grid?

A 2-D PEC box (20 x 10 mm) with a thin PEC fin rising from the bottom wall at
x = 10 mm. The fin's tip is a sheet edge, in both polarizations:
``2d_tez`` (E perpendicular to the edge, the singular case) and ``2d_tmz``
(E parallel to it). The lowest resonance depends smoothly on the fin length,
so it reads the fin's effective length.

* reference: the fin tip ON a node at dx = 0.25 and 0.125 mm, extrapolated
  at the order the tip actually converges at. ``--reference-order`` measures
  it on four grids: FIRST order (0.94 ... 1.01, both polarizations) -- the
  empty box is second order, the fin tip is not. The empty box is checked
  against the analytic value first (comparator before anything else);
* today's uniform grid at dx = 1.0 and 0.5 mm, the drawn length swept through
  the cells in 0.125 mm steps;
* the same sweep on :func:`rfx.mesh_edges.edge_aware_profile`. The
  non-uniform lane ignores ``mode=`` and solves a 3-D box, so TMz runs as a
  one-cell-thick PEC box (Ez survives) and TEz as a two-cell box with
  magnetic walls in z (Ex, Ey, Hz survive); on the uniform grid that 3-D TEz
  arrangement reproduces the 2-D lane's number.

Writes ``pec_sheet_edge_offset/pec_sheet_edge_offset.json``. Runs on CPU,
about 20 minutes.

    python scripts/diagnostics/pec_sheet_edge_offset.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
warnings.filterwarnings("ignore")

from rfx import Box  # noqa: E402
from rfx.api import Simulation  # noqa: E402
from rfx.grid import C0  # noqa: E402
from rfx.harminv import harminv  # noqa: E402
from rfx.mesh_edges import SheetEdge, edge_aware_profile  # noqa: E402

A, B = 0.020, 0.010
#: Convergence order of the on-node reference, measured by
#: ``--reference-order`` (pec_sheet_edge_offset/reference_order.json).
REFERENCE_ORDER = 1.0


def _extrapolate(d):
    """Zero-cell value from the 0.25 and 0.125 mm runs at REFERENCE_ORDER."""
    return d["0.000125"] + (d["0.000125"] - d["0.000250"]) / (2.0**REFERENCE_ORDER - 1.0)


def derive_offset(out):
    """The edge offset, in cells, that each uniform-grid row implies: the
    length at which the reference curve has the row's frequency, minus the
    realized tip node, over the cell. Rows whose frequency falls outside the
    reference curve (its ends are 4 and 6 mm) are left out and counted."""
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq
    derived = {"reference_order": REFERENCE_ORDER, "per_arm": {}}
    for mode, ref in out["reference"].items():
        lengths = np.array(sorted(float(k) for k in ref))
        curve = CubicSpline(
            lengths, [ref[f"{v:.6f}"]["extrapolated_hz"] for v in lengths])
        for key, rows in out["uniform"].items():
            if not key.startswith(mode):
                continue
            dx, vals, skipped = float(key.split("|")[1]), [], 0
            for r in rows:
                g = lambda x, f=r["f_hz"]: float(curve(x)) - f   # noqa: E731
                if g(lengths[0]) * g(lengths[-1]) > 0:
                    skipped += 1
                    continue
                vals.append((brentq(g, lengths[0], lengths[-1])
                             - r["tip_node_m"]) / dx)
            derived["per_arm"][key] = {
                "rows": len(vals), "outside_reference": skipped,
                "offset_cells_mean": float(np.mean(vals)),
                "offset_cells_min": float(np.min(vals)),
                "offset_cells_max": float(np.max(vals))}
    out["derived"] = derived
    return derived
BANDS = {"2d_tmz": (12e9, 30e9), "2d_tez": (2e9, 12e9)}
LENGTHS = np.round(np.arange(4.0e-3, 6.0e-3 + 1e-9, 0.125e-3), 9)
OUT = Path(__file__).with_suffix("") / "pec_sheet_edge_offset.json"


def lowest_resonance(mode, dx, fin_len, *, dy_profile=None, periods=60,
                     pmc_z=False, thickness=0.0, dx_sim=None,
                     eps_left=None):
    """Lowest resonance (Hz) and the realized fin-tip node (m).

    ``eps_left`` fills x < A/2 with that permittivity, so the fin lies IN a
    dielectric interface (a printed conductor's situation). ``dx_sim`` is
    the simulation's base cell (across the fin) when it differs from ``dx``,
    the cell along the fin that ``dy_profile`` carries."""
    lo, hi = BANDS[mode]
    kw = {} if dy_profile is None else {"dy_profile": dy_profile}
    cell = float(dx if dx_sim is None else dx_sim)
    lz = 2 * cell if pmc_z else cell
    if pmc_z:
        from rfx.boundaries.spec import Boundary, BoundarySpec
        sim = Simulation(freq_max=hi, domain=(A, B, lz), dx=cell,
                         boundary=BoundarySpec(x="pec", y="pec",
                                               z=Boundary(lo="pmc", hi="pmc")),
                         **kw)
    else:
        sim = Simulation(freq_max=hi, domain=(A, B, lz), boundary="pec",
                         dx=cell, mode=mode, **kw)
    tip = None
    if eps_left is not None:
        sim.add_material("fill", eps_r=float(eps_left))
        sim.add(Box((0.0, 0.0, 0.0), (A / 2, B, lz)), material="fill")
    if fin_len is not None:
        sim.add(Box((A / 2, 0.0, 0.0),
                    (A / 2 + float(thickness), float(fin_len), lz)),
                material="pec")
        for row in sim.fidelity_report(print_report=False):
            if "pec" in str(row.get("entity")):
                tip = [a["realized_um"][1] * 1e-6 for a in row["axes"]
                       if a["axis"] == "y"][0]
    comp = "ez" if mode == "2d_tmz" else "ey"
    z_obs = cell if pmc_z else 0.0
    sim.add_source((0.0063, 0.0031, z_obs), component=comp)
    sim.add_probe((0.0137, 0.0069, z_obs), component=comp)
    res = sim.run(num_periods=periods, compute_s_params=False)
    ts = np.asarray(res.time_series)[:, 0]
    modes = [m for m in harminv(ts[int(0.15 * len(ts)):], float(res.dt), lo, hi)
             if m.amplitude > 0]
    top = max(m.amplitude for m in modes)
    modes = sorted((m for m in modes if m.amplitude > 1e-3 * top),
                   key=lambda m: m.freq)
    return float(modes[0].freq), tip


def edge_aware_sweep(out, modes=("2d_tmz", "2d_tez"), cells=(1e-3, 0.5e-3)):
    for mode in modes:
        for dx in cells:
            rows = []
            for length in LENGTHS:
                prof = edge_aware_profile(
                    0.0, B, dx, boundary_cell=dx,
                    sheet_edges=[SheetEdge(float(length), -1)])
                f, tip = lowest_resonance(mode, dx, length,
                                          dy_profile=prof.cells,
                                          pmc_z=(mode == "2d_tez"))
                rows.append({"declared_m": float(length), "tip_node_m": tip,
                             "f_hz": f,
                             "edge_cell_m": float(prof.edge_cells[0]),
                             "cell_min_m": float(prof.cells.min()),
                             "cell_max_m": float(prof.cells.max())})
            out["edge_aware"][f"{mode}|{dx:.6f}"] = rows
    # the 3-D TEz arrangement against the 2-D lane, uniform grid, one length
    f3, _ = lowest_resonance("2d_tez", 1e-3, 4.75e-3, pmc_z=True)
    out["tez_3d_pmc_uniform_check_hz"] = f3


def thick_fin(mode):
    """The same sweep for a fin that is a PEC VOLUME 1 mm thick: its tip is a
    face between two corners, not a sheet edge. Reference at dx = 0.25 and
    0.125 mm (the fin is 4 and 8 cells thick there), uniform grid at 1.0 and
    0.5 mm. Written to its own record."""
    t = 1.0e-3
    out = {"mode": mode, "thickness_m": t, "reference": {}, "uniform": {}}
    for dx in (0.25e-3, 0.125e-3):
        for length in LENGTHS[::2]:
            f, _ = lowest_resonance(mode, dx, length, thickness=t)
            out["reference"].setdefault(f"{length:.6f}", {})[f"{dx:.6f}"] = f
    for d in out["reference"].values():
        d["extrapolated_hz"] = _extrapolate(d)
    for dx in (1e-3, 0.5e-3):
        rows = []
        for length in LENGTHS:
            f, tip = lowest_resonance(mode, dx, length, thickness=t)
            rows.append({"declared_m": float(length), "tip_node_m": tip, "f_hz": f})
        out["uniform"][f"{dx:.6f}"] = rows
    path = OUT.with_name(f"thick_fin_{mode}.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path.relative_to(_REPO)}")
    return 0


def generality(variant, mode):
    """Does the offset hold away from the case it was measured on?

    ``interface``: the fin lies in the face of a dielectric half (eps_r 2.2
    on one side, air on the other), with its own on-node reference.
    ``aspect``: cells across the fin half / twice the cells along it, against
    the square-cell reference (the continuum limit does not know the cells).
    Uniform-grid sweep at a 1 mm cell along the fin; the offset is derived
    exactly as in :func:`derive_offset`. One record per (variant, mode)."""
    out = {"variant": variant, "mode": mode, "reference": {mode: {}},
           "uniform": {}}
    pmc = mode == "2d_tez"
    if variant == "interface":
        for dx in (0.25e-3, 0.125e-3):
            for length in LENGTHS[::2]:
                f, _ = lowest_resonance(mode, dx, length, eps_left=2.2)
                out["reference"][mode].setdefault(f"{length:.6f}", {})[f"{dx:.6f}"] = f
        arms = {"1.000": {"eps_left": 2.2}}
    else:
        out["reference"][mode] = json.loads(OUT.read_text())["reference"][mode]
        # The lane wants each profile's end cells equal to the base cell, so
        # the base cell is the one ACROSS the fin and the profile along it
        # carries end cells of that size; the fin tip stays among 1 mm cells.
        # Ratio 2 only for TEz: a 2 mm cell is 7 per wavelength at the TMz
        # resonance and its dispersion would read as an offset.
        arms = {}
        for ratio in ((0.5, 2.0) if pmc else (0.5,)):
            c = ratio * 1e-3
            ends = [c] * int(round(1e-3 / c)) if c < 1e-3 else [c]
            mid = int(round((B - 2 * sum(ends)) / 1e-3))
            arms[f"{ratio:.3f}"] = {
                "dx_sim": c, "pmc_z": pmc,
                "dy_profile": np.array(ends + [1e-3] * mid + ends)}
    for d in out["reference"][mode].values():
        d["extrapolated_hz"] = _extrapolate(d)
    for name, kw in arms.items():
        rows = []
        for length in LENGTHS:
            f, tip = lowest_resonance(mode, 1e-3, length, **kw)
            rows.append({"declared_m": float(length), "tip_node_m": tip, "f_hz": f})
        out["uniform"][f"{mode}|0.001000|cross_cell_ratio={name}"] = rows
    print(json.dumps(derive_offset(out), indent=1), flush=True)
    path = OUT.with_name(f"generality_{variant}_{mode}.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path.relative_to(_REPO)}")
    return 0


def reference_order():
    """Observed convergence order of the on-node reference at the fin tip.

    Four grids (0.5 ... 0.0625 mm), three lengths that sit on a node at all
    of them. ``p = log2((f_h - f_h/2) / (f_h/2 - f_h/4))`` from each
    consecutive triple; the extrapolated value uses the measured p of the
    finest triple. Written to its own record."""
    grids = (0.5e-3, 0.25e-3, 0.125e-3, 0.0625e-3)
    out = {"grids_m": list(grids), "rows": []}
    for mode in BANDS:
        for length in (4.0e-3, 5.0e-3, 6.0e-3):
            f = [lowest_resonance(mode, dx, length)[0] for dx in grids]
            row = {"mode": mode, "length_m": length, "f_hz": f, "order": []}
            for i in range(len(grids) - 2):
                d1, d2 = f[i] - f[i + 1], f[i + 1] - f[i + 2]
                row["order"].append(float(np.log2(d1 / d2))
                                    if d1 * d2 > 0 else None)
            p = row["order"][-1]
            if p:
                row["extrapolated_hz"] = f[-1] + (f[-1] - f[-2]) / (2.0**p - 1.0)
            out["rows"].append(row)
            print(row, flush=True)
    path = OUT.with_name("reference_order.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path.relative_to(_REPO)}")
    return 0


def main() -> int:
    if "--generality" in sys.argv:
        i = sys.argv.index("--generality")
        return generality(sys.argv[i + 1], sys.argv[i + 2])
    if "--reference-order" in sys.argv:
        return reference_order()
    if "--thick-fin" in sys.argv:
        return thick_fin(sys.argv[sys.argv.index("--thick-fin") + 1])
    if "--derive" in sys.argv:
        # re-extrapolate the stored reference runs and re-derive; no solve
        out = json.loads(OUT.read_text())
        for ref in out["reference"].values():
            for d in ref.values():
                d.pop("richardson_hz", None)
                d["extrapolated_hz"] = _extrapolate(d)
        print(json.dumps(derive_offset(out), indent=1))
        OUT.write_text(json.dumps(out, indent=1) + "\n")
        return 0
    if "--edge-aware-arm" in sys.argv:
        # one (mode, cell) arm into its own part file, for running the four
        # arms as separate jobs; --merge-arms folds the parts into the record
        i = sys.argv.index("--edge-aware-arm")
        mode, dx = sys.argv[i + 1], float(sys.argv[i + 2])
        part = {"edge_aware": {}}
        edge_aware_sweep(part, modes=(mode,), cells=(dx,))
        part.pop("tez_3d_pmc_uniform_check_hz", None)
        path = OUT.with_name(f"edge_aware_part_{mode}_{dx:.6f}.json")
        path.write_text(json.dumps(part, indent=1) + "\n")
        return 0
    if "--merge-arms" in sys.argv:
        out = json.loads(OUT.read_text())
        for path in sorted(OUT.parent.glob("edge_aware_part_*.json")):
            out["edge_aware"].update(json.loads(path.read_text())["edge_aware"])
            path.unlink()
        OUT.write_text(json.dumps(out, indent=1) + "\n")
        return 0
    if "--edge-aware-only" in sys.argv:
        out = json.loads(OUT.read_text())
        edge_aware_sweep(out)
        OUT.write_text(json.dumps(out, indent=1) + "\n")
        print(f"updated {OUT.relative_to(_REPO)}")
        return 0
    import rfx
    out = {"rfx_file": str(Path(rfx.__file__).resolve().relative_to(_REPO)),
           "box_m": [A, B], "lengths_m": LENGTHS.tolist(),
           "empty_box": [], "reference": {}, "uniform": {}, "edge_aware": {}}
    for mode, f_an in (("2d_tmz", C0 / 2 * np.sqrt(1 / A**2 + 1 / B**2)),
                       ("2d_tez", C0 / (2 * A))):
        for dx in (1e-3, 0.5e-3):
            f, _ = lowest_resonance(mode, dx, None)
            out["empty_box"].append({"mode": mode, "dx_m": dx, "f_hz": f,
                                     "analytic_hz": float(f_an),
                                     "rel_err": f / f_an - 1.0})
    for mode in BANDS:
        ref = {}
        for dx in (0.25e-3, 0.125e-3):
            for length in LENGTHS[::2]:            # on a node at both dx
                f, tip = lowest_resonance(mode, dx, length)
                ref.setdefault(f"{length:.6f}", {})[f"{dx:.6f}"] = f
        for d in ref.values():
            d["extrapolated_hz"] = _extrapolate(d)
        out["reference"][mode] = ref
        for dx in (1e-3, 0.5e-3):
            rows = []
            for length in LENGTHS:
                f, tip = lowest_resonance(mode, dx, length)
                rows.append({"declared_m": float(length), "tip_node_m": tip, "f_hz": f})
            out["uniform"][f"{mode}|{dx:.6f}"] = rows
    edge_aware_sweep(out)
    derive_offset(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
