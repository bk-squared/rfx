#!/usr/bin/env python3
"""Numerical quasi-static eps_eff of the MSL phase-referee cross-section.

No FDTD, no time stepping.  For the board the phase-referee fixture
builds, this runs the repo's own 2-D electrostatic twin solve
(``rfx.sources.msl_port.compute_msl_mode_profile``: one solve with the
substrate, one with the substrate replaced by air, eps_eff = C_sub/C_air)
on the RASTERIZED cross-section at three cell sizes, and puts the answer
next to the Hammerstad-Jensen closed form.

Why.  The referee's signed-beta finding says the solved beta sits about
+1.3 % above the closed form for the board rfx actually builds.  The
closed form is a fit to a zero-thickness strip on a semi-infinite-width
board; the twin solve is a second model point on the SAME geometry with
none of that fit's assumptions, so it says how much of the residual the
closed form's own model error can carry.  It is NOT an independent
referee for the solver -- it models a zero-thickness strip too, and it is
quasi-static, so it cannot see dispersion or the finite conductor.

What it does NOT do.  It does not re-solve the FDTD fixture, and it does
not move any gate.

Realized-board assertion first.  Every dx run prints its
``sim.fidelity_report(print_report=False)`` rows for the substrate and
the conductor, plus the conductor's realized WALL PLANES read from the
shared owner (``rfx.boundaries.pec``).  The two disagree by one cell for
a PEC volume, for the documented reason in the fixture builder's
``_realized_trace_geometry`` docstring, so both are reported rather than
one being picked silently.

Box-size witness.  The 2-D box is Neumann-truncated; its default is 5 W
laterally and 4 H above the substrate.  The run repeats the solve with
the box widened and heightened to show whether the fringing field is
being cut off.  A second witness varies the solve's internal refinement.

Two discretization terms the run exposes, both measured here and neither
fixed here (they change ``rfx/`` and belong to their own evidence):

  1. The strip is placed on the FIRST AIR ROW above the substrate, so it
     floats half a fine cell (``dz_fine / 2``) above the dielectric: 6.25
     um at the production default of dx = 50 um and refine = 4, against a
     250 um board.  That air gap sits in series with the dielectric and
     pulls eps_eff down; it is why the answer climbs steadily as either
     dx or ``refine`` shrinks, and why the run is a convergence ladder
     rather than three independent boards.
  2. The conductor is ``n_y_trace = w_hi - w_lo + 1`` cells wide, but
     ``w_lo``/``w_hi`` are inclusive NODE indices, so the modelled strip
     is W + dx rather than W -- 650 / 625 / 612.5 um at dx = 50 / 25 /
     12.5 against a declared 600 um.  The one-cell node-vs-cell error the
     fixture builder's ``_realized_trace_geometry`` docstring warns about,
     in the mode solver.  It is why two settings with the SAME strip air
     gap (dx = 25, refine = 4 and dx = 50, refine = 8) still differ by
     0.43 %.

Neither reaches the beta anchor: that comes from the Hammerstad-Jensen
closed form in ``rfx/sparams/msl.py``, not from this solve.  They do
reach ``z0_static`` and the injected source profile.

Outputs:
    scripts/diagnostics/msl_quasistatic_eps_eff_vs_dx/*.json
    docs/crossval/figures/msl_quasistatic_eps_eff_vs_dx.png
    docs/crossval/figures/msl_quasistatic_eps_eff_convergence.png

The figures go to ``docs/crossval/figures/`` because ``**/*.png`` is
gitignored everywhere else; that directory is one of the negations, and a
claims-bearing curve nobody can open from a fresh clone is not evidence.
``--figures-only`` redraws them from the committed JSONs without solving.

Usage::

    PYTHONPATH=. python scripts/diagnostics/msl_quasistatic_eps_eff_vs_dx.py
    PYTHONPATH=. python scripts/diagnostics/msl_quasistatic_eps_eff_vs_dx.py \\
        --dx-um 50 25          # skip the finest rung
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import rfx
# The fixture producer is the geometry owner: its _build_sim() draws the
# board this whole case is about, so the geometry is imitated from it
# rather than redrawn here.  NOTE it flips jax x64 at import (it is a
# script, not a test module).
import importlib.util as _ilu

from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
from rfx.sources.msl_port import (
    compute_msl_mode_profile, msl_cross_section_span, msl_port_from_entry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER = Path(__file__).resolve().parent / "build_msl_thru_phase_dx50um_reference.py"
OUT_DIR = Path(__file__).resolve().parent / "msl_quasistatic_eps_eff_vs_dx"
FIG_DIR = REPO_ROOT / "docs" / "crossval" / "figures"


def rfx_provenance() -> str:
    """Which rfx this ran against, as a repo-relative path.

    The check that matters is that the import resolved inside THIS checkout
    and not to an installed copy; recording the absolute path would put the
    author's machine layout in a committed artifact, so it is relative and
    says so loudly when it is not.
    """
    path = Path(rfx.__file__).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return f"NOT THIS CHECKOUT: an installed rfx at {path.name}"


def load_builder():
    spec = _ilu.spec_from_file_location("_msl_thru_builder", BUILDER)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_at_dx(builder, dx_m: float):
    """The builder's own ``_build_sim()`` at a different cell size.

    ``_build_sim`` reads module-level DX and the derived domain LY (the
    lateral clearance is written as ``8 * DX`` per side), so the cell size
    is changed by setting those two and calling the builder's own
    function.  LX and LZ do not depend on DX.  Nothing else about the
    geometry -- materials, boxes, both ports -- is retyped here.
    """
    builder.DX = float(dx_m)
    builder.LY = builder.W_TRACE + 2 * (2 * builder.H_SUB + 8 * builder.DX)
    return builder._build_sim()


def realized_rows(sim) -> dict:
    """Substrate and conductor rows of the live fidelity report, verbatim."""
    report = sim.fidelity_report(print_report=False)
    out = {}
    for key, entity in (("substrate", "geometry[0] 'ro4350b'"),
                        ("conductor", "geometry[1] 'pec'")):
        item = next((it for it in report if it["entity"] == entity), None)
        if item is None:
            out[key] = None
            continue
        out[key] = {
            "entity": item["entity"],
            "axes": {
                ax["axis"]: {
                    "declared_um": [float(v) for v in ax["declared_um"]],
                    "realized_um": [float(v) for v in ax["realized_um"]],
                    "declared_extent_um": float(ax["declared_extent_um"]),
                    "realized_extent_um": float(ax["realized_extent_um"]),
                }
                for ax in item.get("axes", [])
            },
        }
    return out


def solve_one(sim, builder, eps_r: float, **profile_kwargs) -> dict:
    grid = sim._build_grid()
    port = msl_port_from_entry(sim._msl_ports[0])
    span = msl_cross_section_span(grid, port)
    n_z_sub = int(span["n_hi"] - span["n_lo"])
    t0 = time.time()
    prof = compute_msl_mode_profile(grid, port, eps_r, **profile_kwargs)
    elapsed = time.time() - t0
    dz = float(prof["dz"])
    return {
        "eps_eff": float(prof["eps_eff"]),
        "z0_static_ohm": float(prof["z0_static"]),
        "n_z_sub": n_z_sub,
        "dz_m": dz,
        # The dielectric height the 2-D solve actually sees: the ground row
        # is the substrate's bottom cell and the strip row is the first AIR
        # row above the substrate, so their separation is n_z_sub cells.
        "h_solver_m": n_z_sub * dz,
        "elapsed_s": round(elapsed, 2),
    }


def box_dims(sim, h_declared_m: float, w_trace_m: float, refine: int,
             pad_y_cells=None, pad_z_cells=None) -> dict:
    """The Laplace box the solve builds, in physical units and in h / W.

    Mirrors ``compute_msl_mode_profile``'s own sizing arithmetic so the
    truncation distance can be quoted without instrumenting the function.
    """
    grid = sim._build_grid()
    port = msl_port_from_entry(sim._msl_ports[0])
    span = msl_cross_section_span(grid, port)
    dy = float(grid.dx)
    dz = float(grid.dx)
    h_sub = float(port.z_hi - port.z_lo)
    w = float(port.y_hi - port.y_lo)
    if pad_y_cells is None:
        pad_y_cells = max(1, int(round(1.0 * h_sub / dy)))
    if pad_z_cells is None:
        pad_z_cells = max(4, int(round(4.0 * h_sub / dz)))
    dy_fine, dz_fine = dy / refine, dz / refine
    pad_y_fine = max(pad_y_cells * refine, int(round(5.0 * w / dy_fine)))
    pad_z_fine = max(pad_z_cells * refine, int(round(4.0 * h_sub / dz_fine)))
    n_z_sub = int(span["n_hi"] - span["n_lo"])
    lateral_m = pad_y_fine * dy_fine
    above_m = pad_z_fine * dz_fine
    return {
        "lateral_clearance_beside_strip_m": lateral_m,
        "lateral_clearance_in_W": lateral_m / w,
        "lateral_clearance_in_h_solver": lateral_m / (n_z_sub * dz),
        "clearance_above_substrate_m": above_m,
        "clearance_above_in_h_declared": above_m / h_sub,
        "clearance_above_in_h_solver": above_m / (n_z_sub * dz),
        "box_cells": [int(span["w_hi"] - span["w_lo"] + 1) * refine
                      + 2 * pad_y_fine, n_z_sub * refine + pad_z_fine],
        "pad_y_cells": int(pad_y_cells), "pad_z_cells": int(pad_z_cells),
        "refine": int(refine),
    }


# ---------------------------------------------------------------------------
# Extrapolated limits -- arithmetic on the committed rungs, no solving
# ---------------------------------------------------------------------------
#
# None of the three ladders here has converged, so any statement about the
# limit is a choice of recipe, not a measurement.  Rather than quote one
# number, this tabulates the limit for every combination of:
#
#   * both step-halving ladders -- the dx ladder (dx = 50 / 25 / 12.5 um at
#     the production refine = 4) and the internal-refinement ladder at
#     dx = 50 um (refine = 4 / 8 / 16).  Both halve the Laplace box's fine
#     cell dz_fine = dx / refine at each rung, so both are h-ratio-2 ladders
#     in the same quantity;
#   * both extrapolation recipes -- first order at the nominal h-ratio of 2,
#     and geometric at the ratio the differences actually show;
#   * every box correction MEASURED in these runs, each tagged with the rung
#     it came from, since the box axis has not converged either.
#
# The spread of the resulting table is the answer's uncertainty.

def _richardson(values: list[float], recipe: str) -> dict:
    """Limit of a step-halving sequence, coarse -> fine.

    ``first_order_h_ratio_2``: assume error ~ C*h and the nominal ratio of 2,
    so the limit is ``e_fine + (e_fine - e_mid) / (2 - 1)``.
    ``geometric_measured_ratio``: take the ratio the last two differences
    actually show and sum the remaining geometric tail.
    """
    e_coarse, e_mid, e_fine = values[-3], values[-2], values[-1]
    d_prev, d_last = e_mid - e_coarse, e_fine - e_mid
    if recipe == "first_order_h_ratio_2":
        return {"limit": e_fine + d_last / (2.0 - 1.0), "ratio": None}
    if recipe == "geometric_measured_ratio":
        r = d_last / d_prev
        return {"limit": e_fine + d_last * r / (1.0 - r), "ratio": float(r)}
    raise ValueError(recipe)


def _measured_box_corrections(reports: dict) -> list[dict]:
    """Every box-widening measurement in the committed JSONs, with its rung."""
    out = []
    for name, rep in reports.items():
        for row in rep.get("box_witness", []):
            out.append({
                "source_json": name,
                "measured_on_rung": f"dx={row['dx_um']:g} um, refine=4",
                "box": f"{row['box']['lateral_clearance_in_W']:.0f}W/"
                       f"{row['box']['clearance_above_in_h_solver']:.1f}h",
                "correction_frac": float(row["rel_to_default_box_frac"]),
            })
        for row in rep.get("joint_witness", []):
            # rel_to_default_frac on a joint row is against the DEFAULT rung
            # (refine = 4, 5W box), so it carries the refinement change as
            # well as the box change and is not a box correction. Divide out
            # the refinement by referencing the same-refine 5W value.
            same_refine = next(
                (float(w["eps_eff"]) for w in rep.get("refine_witness", [])
                 if int(w["refine"]) == int(row["refine"])), None)
            if same_refine is None:
                continue
            out.append({
                "source_json": name,
                "measured_on_rung": f"dx={row['dx_um']:g} um, refine={row['refine']}",
                "box": f"{row['box']['lateral_clearance_in_W']:.0f}W/"
                       f"{row['box']['clearance_above_in_h_solver']:.1f}h",
                "correction_frac": float(row["eps_eff"]) / same_refine - 1.0,
            })
    return sorted(out, key=lambda r: (r["box"], r["measured_on_rung"]))


def extrapolation_table(out_dir: Path, ladder_name: str) -> dict:
    """Build the limit table from the committed JSONs. Pure arithmetic."""
    reports = {p.stem: json.loads(p.read_text())
               for p in sorted(out_dir.glob("*.json"))
               if p.stem != "msl_quasistatic_eps_eff_extrapolation"}
    ladder = reports[ladder_name]
    e_closed = float(ladder["closed_form"]["h_250um"]["eps_eff"])

    rungs = sorted(ladder["rungs"], key=lambda r: -r["dx_um"])
    dx_ladder = {
        "name": "dx ladder (production refine = 4)",
        "steps_um": [r["dx_um"] / 4.0 for r in rungs],
        "labels": [f"dx={r['dx_um']:g} um" for r in rungs],
        "values": [float(r["solve"]["eps_eff"]) for r in rungs],
    }

    conv = reports["msl_quasistatic_eps_eff_convergence"]
    by_refine = {4: float(conv["rungs"][0]["solve"]["eps_eff"])}
    for row in conv["refine_witness"]:
        by_refine[int(row["refine"])] = float(row["eps_eff"])
    refines = [r for r in (4, 8, 16) if r in by_refine]
    refine_ladder = {
        "name": "internal-refinement ladder at dx = 50 um",
        "steps_um": [50.0 / r for r in refines],
        "labels": [f"refine={r}" for r in refines],
        "values": [by_refine[r] for r in refines],
    }

    corrections = [{"source_json": None, "measured_on_rung": "none",
                    "box": "default 5W/4h", "correction_frac": 0.0}]
    corrections += _measured_box_corrections(reports)

    rows = []
    for ladder_spec in (dx_ladder, refine_ladder):
        for recipe in ("first_order_h_ratio_2", "geometric_measured_ratio"):
            base = _richardson(ladder_spec["values"], recipe)
            for corr in corrections:
                limit = base["limit"] * (1.0 + corr["correction_frac"])
                eps_dev = limit / e_closed - 1.0
                rows.append({
                    "ladder": ladder_spec["name"],
                    "ladder_rungs": ladder_spec["labels"],
                    "ladder_values": ladder_spec["values"],
                    "ladder_steps_dz_fine_um": ladder_spec["steps_um"],
                    "recipe": recipe,
                    "measured_ratio": base["ratio"],
                    "uncorrected_limit": base["limit"],
                    "box_correction_frac": corr["correction_frac"],
                    "box_correction_box": corr["box"],
                    "box_correction_measured_on_rung": corr["measured_on_rung"],
                    "limit": limit,
                    "eps_eff_dev_vs_closed_form_frac": eps_dev,
                    "beta_dev_vs_closed_form_frac": float(
                        np.sqrt(limit / e_closed) - 1.0),
                })

    def _span(sel) -> dict:
        chosen = [r for r in rows if sel(r)]
        eps = [r["eps_eff_dev_vs_closed_form_frac"] for r in chosen]
        bet = [r["beta_dev_vs_closed_form_frac"] for r in chosen]
        return {"n_rows": len(chosen),
                "eps_eff_dev_frac": [min(eps), max(eps)],
                "beta_dev_frac": [min(bet), max(bet)]}

    # The two box corrections behind the two independently recomputed sets: the
    # widest box measured (20W, on dx = 50 um / refine = 4), used by the session
    # leader's recomputation, and the 10W step measured on the finest mesh
    # (dx = 12.5 um), used by the round-1 reviewer's.
    verified = {("20W/16.2h", "dx=50 um, refine=4"),
                ("10W/8.1h", "dx=12.5 um, refine=4")}
    return {
        "rfx_file": rfx_provenance(),
        "closed_form_eps_eff_h250um": e_closed,
        "support_threshold_eps_eff": e_closed * 1.015,
        "ladders": [dx_ladder, refine_ladder],
        "box_corrections": corrections,
        "rows": rows,
        "span_uncorrected": _span(lambda r: r["box_correction_frac"] == 0.0),
        "span_box_corrections_on_dx50_refine4": _span(
            lambda r: r["box_correction_frac"] != 0.0
            and r["box_correction_measured_on_rung"] == "dx=50 um, refine=4"),
        "span_box_corrections_of_the_recomputed_sets": _span(
            lambda r: (r["box_correction_box"],
                       r["box_correction_measured_on_rung"]) in verified),
        "span_all_box_corrections": _span(
            lambda r: r["box_correction_frac"] != 0.0),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dx-um", nargs="*", type=float, default=[50.0, 25.0, 12.5])
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--out-name", default="msl_quasistatic_eps_eff_vs_dx")
    p.add_argument("--fig-dir", default=str(FIG_DIR))
    p.add_argument(
        "--witness-dx-um", nargs="*", type=float, default=[50.0],
        help="cell sizes that also get the box-size and refinement witnesses")
    p.add_argument("--box-scales", nargs="*", type=float, default=[2.0, 4.0])
    p.add_argument("--refine-list", nargs="*", type=int, default=[2, 8])
    p.add_argument(
        "--joint", nargs="*", type=str, default=[],
        help="joint refine:box-scale points, e.g. 8:2 -- tests whether the "
             "two convergence directions are separable")
    p.add_argument("--skip-box-witness", action="store_true")
    p.add_argument(
        "--figures-only", action="store_true",
        help="redraw the figures from the JSONs already in --out-dir, "
             "without solving anything")
    p.add_argument(
        "--extrapolate-only", action="store_true",
        help="tabulate the extrapolated limits from the JSONs already in "
             "--out-dir, without solving anything")
    args = p.parse_args(argv)
    if args.figures_only:
        return figures_only(Path(args.out_dir), args.out_name,
                            Path(args.fig_dir))
    if args.extrapolate_only:
        return extrapolate_only(Path(args.out_dir), args.out_name)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)

    print(f"rfx.__file__ = {rfx.__file__}")
    builder = load_builder()
    eps_r = float(builder.EPS_R)
    w_trace = float(builder.W_TRACE)
    h_declared = float(builder.H_SUB)

    # Closed-form references, from the same function production anchors on.
    z0_250, eps_250 = hammerstad_jensen_z0_eps_eff(w_trace, 250e-6, eps_r)
    z0_254, eps_254 = hammerstad_jensen_z0_eps_eff(w_trace, h_declared, eps_r)
    print(f"closed form eps_eff: h=250um -> {eps_250:.6f} (Z0 {z0_250:.4f} ohm)"
          f" | h=254um -> {eps_254:.6f} (Z0 {z0_254:.4f} ohm)")

    report = {
        "rfx_file": rfx_provenance(),
        "builder": str(BUILDER.relative_to(REPO_ROOT)),
        "eps_r": eps_r, "w_trace_m": w_trace, "h_declared_m": h_declared,
        "closed_form": {
            "h_250um": {"eps_eff": float(eps_250), "z0_ohm": float(z0_250)},
            "h_254um": {"eps_eff": float(eps_254), "z0_ohm": float(z0_254)},
        },
        "rungs": [],
        "box_witness": [],
        "refine_witness": [],
    }

    for dx_um in args.dx_um:
        dx = dx_um * 1e-6
        print(f"\n===== dx = {dx_um} um =====")
        sim = build_at_dx(builder, dx)
        grid = sim._build_grid()
        print(f"grid shape {tuple(int(n) for n in grid.shape)}")
        rows = realized_rows(sim)
        for key in ("substrate", "conductor"):
            item = rows[key]
            if item is None:
                print(f"  {key}: NOT FOUND in fidelity report")
                continue
            print(f"  fidelity {item['entity']}:")
            for axis in ("x", "y", "z"):
                ax = item["axes"].get(axis)
                if ax is None:
                    continue
                print(f"    {axis}: declared [{ax['declared_um'][0]:.3f}, "
                      f"{ax['declared_um'][1]:.3f}] um -> realized "
                      f"[{ax['realized_um'][0]:.3f}, {ax['realized_um'][1]:.3f}]"
                      f" um | extent {ax['declared_extent_um']:.3f} -> "
                      f"{ax['realized_extent_um']:.3f} um")
        trace_geom = builder._realized_trace_geometry(sim)
        print(f"  realized conductor (shared PEC owner, not fidelity): "
              f"wall planes z = "
              f"{[round(v * 1e6, 3) for v in trace_geom['trace_wall_planes_realized_z_m']]} um"
              f", kind={trace_geom['trace_realization_kind']}"
              f", t_metal={trace_geom['t_metal_realized_m'] * 1e6:.3f} um"
              f", W={trace_geom['w_trace_realized_m'] * 1e6:.3f} um")

        solved = solve_one(sim, builder, eps_r)
        box = box_dims(sim, h_declared, w_trace, refine=4)
        print(f"  2-D twin solve: eps_eff = {solved['eps_eff']:.6f}"
              f"  z0_static = {solved['z0_static_ohm']:.4f} ohm"
              f"  ({solved['elapsed_s']} s, box {box['box_cells']} cells)")
        print(f"    dielectric height the solve sees: n_z_sub={solved['n_z_sub']}"
              f" x dz={solved['dz_m'] * 1e6:.3f} um = "
              f"{solved['h_solver_m'] * 1e6:.3f} um")
        print(f"    vs closed form h=250um: "
              f"{100 * (solved['eps_eff'] / eps_250 - 1):+.4f} %"
              f" | vs h=254um: {100 * (solved['eps_eff'] / eps_254 - 1):+.4f} %")
        print(f"    box: {box['lateral_clearance_in_W']:.2f} W beside the strip"
              f" ({box['lateral_clearance_in_h_solver']:.2f} h),"
              f" {box['clearance_above_in_h_solver']:.2f} h above the substrate")

        report["rungs"].append({
            "dx_m": dx, "dx_um": dx_um,
            "grid_shape": [int(n) for n in grid.shape],
            "fidelity_rows": rows,
            "realized_trace": {
                k: (v if not isinstance(v, (list, tuple)) else list(v))
                for k, v in trace_geom.items()
            },
            "solve": solved,
            "box": box,
            "dev_vs_closed_form_h250_frac": float(solved["eps_eff"] / eps_250 - 1),
            "dev_vs_closed_form_h254_frac": float(solved["eps_eff"] / eps_254 - 1),
            "beta_dev_vs_closed_form_h250_frac": float(
                np.sqrt(solved["eps_eff"] / eps_250) - 1),
        })

        if not args.skip_box_witness and dx_um in args.witness_dx_um:
            for scale in args.box_scales:
                pad_y = int(round(scale * 5.0 * w_trace / dx))
                pad_z = int(round(scale * 4.0 * h_declared / dx))
                w_solved = solve_one(sim, builder, eps_r,
                                     pad_y_cells=pad_y, pad_z_cells=pad_z)
                w_box = box_dims(sim, h_declared, w_trace, refine=4,
                                 pad_y_cells=pad_y, pad_z_cells=pad_z)
                print(f"  box x{scale:g}: eps_eff = {w_solved['eps_eff']:.6f}"
                      f" ({100 * (w_solved['eps_eff'] / solved['eps_eff'] - 1):+.4f} %"
                      f" vs default box; {w_box['lateral_clearance_in_W']:.2f} W"
                      f" lateral, {w_box['clearance_above_in_h_solver']:.2f} h above)"
                      f"  [{w_solved['elapsed_s']} s]")
                report["box_witness"].append({
                    "dx_um": dx_um, "scale": scale,
                    "eps_eff": w_solved["eps_eff"],
                    "z0_static_ohm": w_solved["z0_static_ohm"],
                    "rel_to_default_box_frac": float(
                        w_solved["eps_eff"] / solved["eps_eff"] - 1),
                    "box": w_box,
                })
            for refine in args.refine_list:
                r_solved = solve_one(sim, builder, eps_r, refine=refine)
                r_box = box_dims(sim, h_declared, w_trace, refine=refine)
                print(f"  refine={refine}: eps_eff = {r_solved['eps_eff']:.6f}"
                      f" ({100 * (r_solved['eps_eff'] / solved['eps_eff'] - 1):+.4f} %"
                      f" vs refine=4; strip floats dz_fine/2 = "
                      f"{0.5e6 * dx / refine:.4f} um above the dielectric)"
                      f"  [{r_solved['elapsed_s']} s]")
                report["refine_witness"].append({
                    "dx_um": dx_um, "refine": refine,
                    "eps_eff": r_solved["eps_eff"],
                    "z0_static_ohm": r_solved["z0_static_ohm"],
                    "strip_air_gap_um": 0.5e6 * dx / refine,
                    "box": r_box,
                    "rel_to_refine4_frac": float(
                        r_solved["eps_eff"] / solved["eps_eff"] - 1),
                })
            for spec in args.joint:
                refine, scale = spec.split(":")
                refine, scale = int(refine), float(scale)
                pad_y = int(round(scale * 5.0 * w_trace / dx))
                pad_z = int(round(scale * 4.0 * h_declared / dx))
                j_solved = solve_one(sim, builder, eps_r, refine=refine,
                                     pad_y_cells=pad_y, pad_z_cells=pad_z)
                j_box = box_dims(sim, h_declared, w_trace, refine=refine,
                                 pad_y_cells=pad_y, pad_z_cells=pad_z)
                print(f"  joint refine={refine} box x{scale:g}: eps_eff = "
                      f"{j_solved['eps_eff']:.6f} "
                      f"({100 * (j_solved['eps_eff'] / solved['eps_eff'] - 1):+.4f} %"
                      f" vs default)  [{j_solved['elapsed_s']} s]")
                report.setdefault("joint_witness", []).append({
                    "dx_um": dx_um, "refine": refine, "box_scale": scale,
                    "eps_eff": j_solved["eps_eff"],
                    "z0_static_ohm": j_solved["z0_static_ohm"],
                    "box": j_box,
                    "rel_to_default_frac": float(
                        j_solved["eps_eff"] / solved["eps_eff"] - 1),
                })

    # Spread across the dx ladder -- the stability half of the pre-declared
    # reading.
    eps_list = [r["solve"]["eps_eff"] for r in report["rungs"]]
    if len(eps_list) > 1:
        spread = (max(eps_list) - min(eps_list)) / np.mean(eps_list)
        report["eps_eff_spread_across_dx_frac"] = float(spread)
        print(f"\neps_eff spread across dx ladder: {100 * spread:.4f} % "
              f"({min(eps_list):.6f} .. {max(eps_list):.6f})")

    json_path = out_dir / f"{args.out_name}.json"
    json_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"\nwritten {json_path}")
    if len(report["rungs"]) > 1:
        fig_dir.mkdir(parents=True, exist_ok=True)
        png_path = fig_dir / f"{args.out_name}.png"
        plot(report, png_path)
        print(f"written {png_path}")
    return 0


def extrapolate_only(out_dir: Path, ladder_name: str) -> int:
    """Write and print the limit table. Arithmetic on committed JSONs only."""
    table = extrapolation_table(out_dir, ladder_name)
    path = out_dir / "msl_quasistatic_eps_eff_extrapolation.json"
    path.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")

    print(f"rfx.__file__ (repo-relative) = {table['rfx_file']}")
    print(f"closed form eps_eff (h=250 um) = "
          f"{table['closed_form_eps_eff_h250um']:.9f}; support threshold "
          f"(+1.5 %) = {table['support_threshold_eps_eff']:.6f}")
    for lad in table["ladders"]:
        print(f"\n{lad['name']}:")
        print("  " + "  ".join(
            f"{lab} (dz_fine={s:.4f} um) = {v:.6f}"
            for lab, s, v in zip(lad["labels"], lad["steps_um"], lad["values"])))
    print("\nmeasured box corrections:")
    for c in table["box_corrections"]:
        if c["correction_frac"] == 0.0:
            continue
        print(f"  {c['box']:>12s}  {100 * c['correction_frac']:+.4f} %"
              f"   measured on {c['measured_on_rung']}")

    print(f"\n{'ladder':<40s} {'recipe':<26s} {'ratio':>7s} {'box':>12s} "
          f"{'limit':>9s} {'eps dev':>9s} {'beta dev':>9s}")
    for r in table["rows"]:
        ratio = "-" if r["measured_ratio"] is None else f"{r['measured_ratio']:.4f}"
        print(f"{r['ladder']:<40s} {r['recipe']:<26s} {ratio:>7s} "
              f"{r['box_correction_box']:>12s} {r['limit']:>9.6f} "
              f"{100 * r['eps_eff_dev_vs_closed_form_frac']:>+8.3f} % "
              f"{100 * r['beta_dev_vs_closed_form_frac']:>+8.3f} %")

    print()
    for key in ("span_uncorrected", "span_box_corrections_on_dx50_refine4",
                "span_box_corrections_of_the_recomputed_sets",
                "span_all_box_corrections"):
        s = table[key]
        print(f"{key} ({s['n_rows']} rows): eps_eff "
              f"{100 * s['eps_eff_dev_frac'][0]:+.3f} % .. "
              f"{100 * s['eps_eff_dev_frac'][1]:+.3f} %   beta "
              f"{100 * s['beta_dev_frac'][0]:+.3f} % .. "
              f"{100 * s['beta_dev_frac'][1]:+.3f} %")
    print(f"\nwritten {path}")
    return 0


def figures_only(out_dir: Path, ladder_name: str, fig_dir: Path) -> int:
    """Redraw from the committed JSONs -- no solving.

    The ladder and the convergence study are separate runs (the second is
    far more expensive), so the summary figure is assembled here from
    whatever JSONs the directory holds.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    ladder = json.loads((out_dir / f"{ladder_name}.json").read_text())
    plot(ladder, fig_dir / f"{ladder_name}.png")
    print(f"written {fig_dir / f'{ladder_name}.png'}")
    extra = {}
    for path in sorted(out_dir.glob("*.json")):
        if path.stem == ladder_name:
            continue
        extra[path.stem] = json.loads(path.read_text())
    if extra:
        png = fig_dir / "msl_quasistatic_eps_eff_convergence.png"
        plot_convergence(ladder, extra, png)
        print(f"written {png}")
    return 0


def plot_convergence(ladder, extra, png_path):
    """Is the twin solve converged at the settings production uses?

    Left: eps_eff against the strip's air gap (half a fine cell), the
    quantity both the refinement knob and the cell size move.  Right: the
    same values against the Neumann box's lateral clearance.  Both axes
    have to be flat before a number off this solve can be compared with a
    closed form to better than the distance being argued about.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e250 = ladder["closed_form"]["h_250um"]["eps_eff"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))

    # --- mesh axis: the strip's air gap above the dielectric ---
    ax = axes[0]
    pts = []
    for r in ladder["rungs"]:
        pts.append((0.5 * r["dx_um"] / 4.0, r["solve"]["eps_eff"],
                    f"dx={r['dx_um']:g}"))
    for rep in extra.values():
        for r in rep.get("refine_witness", []):
            pts.append((r["strip_air_gap_um"], r["eps_eff"],
                        f"refine={r['refine']}"))
        for r in rep.get("rungs", []):
            pts.append((0.5 * r["dx_um"] / 4.0, r["solve"]["eps_eff"], ""))
    pts = sorted(set((round(a, 6), round(b, 9), c) for a, b, c in pts))
    gaps = np.array([p[0] for p in pts])
    eps = np.array([p[1] for p in pts])
    ax.plot(gaps, eps, "o", color="tab:blue")
    order = np.argsort(-gaps)
    ax.plot(gaps[order], eps[order], "-", lw=1.0, color="tab:blue")
    ax.axhline(e250, color="tab:green", ls="--", lw=1.2,
               label=f"closed form h=250 um ({e250:.4f})")
    ax.axhline(e250 * 1.015, color="tab:orange", lw=1.0,
               label="+1.5 % (H-c support threshold)")
    ax.set_xlabel("strip air gap above the dielectric = dz_fine/2  [um]")
    ax.set_ylabel("eps_eff")
    ax.invert_xaxis()
    ax.legend(fontsize=7.2, loc="lower left")
    ax.set_title("mesh axis (default 5W x 4H box)", fontsize=9)

    # --- box axis ---
    ax = axes[1]
    series = {}
    for rep in extra.values():
        for r in rep.get("rungs", []):
            key = f"dx={r['dx_um']:g}, refine=4"
            series.setdefault(key, []).append(
                (r["box"]["lateral_clearance_in_W"], r["solve"]["eps_eff"]))
        for r in rep.get("box_witness", []):
            key = f"dx={r['dx_um']:g}, refine=4"
            series.setdefault(key, []).append(
                (r["box"]["lateral_clearance_in_W"], r["eps_eff"]))
        for r in rep.get("joint_witness", []):
            key = f"dx={r['dx_um']:g}, refine={r['refine']}"
            series.setdefault(key, []).append(
                (r["box"]["lateral_clearance_in_W"], r["eps_eff"]))
        for r in rep.get("refine_witness", []):
            key = f"dx={r['dx_um']:g}, refine={r['refine']}"
            series.setdefault(key, []).append(
                (r["box"]["lateral_clearance_in_W"], r["eps_eff"]))
    for r in ladder["rungs"]:
        key = f"dx={r['dx_um']:g}, refine=4"
        series.setdefault(key, []).append(
            (r["box"]["lateral_clearance_in_W"], r["solve"]["eps_eff"]))
    for key in sorted(series):
        vals = sorted(set(series[key]))
        if len(vals) < 2:
            continue
        ax.plot([v[0] for v in vals], [v[1] for v in vals], "o-", lw=1.1,
                label=key, ms=4)
    ax.axhline(e250, color="tab:green", ls="--", lw=1.2)
    ax.axhline(e250 * 1.015, color="tab:orange", lw=1.0)
    ax.set_xlabel("Neumann box lateral clearance beside the strip  [W]")
    ax.set_ylabel("eps_eff")
    ax.legend(fontsize=7.2, loc="best")
    ax.set_title("box-truncation axis", fontsize=9)

    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def plot(report, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dx = np.array([r["dx_um"] for r in report["rungs"]], dtype=float)
    eps = np.array([r["solve"]["eps_eff"] for r in report["rungs"]], dtype=float)
    e250 = report["closed_form"]["h_250um"]["eps_eff"]
    e254 = report["closed_form"]["h_254um"]["eps_eff"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax = axes[0]
    ax.plot(dx, eps, "o-", lw=1.4, label="2-D twin solve (rasterized)")
    ax.axhline(e250, color="tab:green", ls="--", lw=1.2,
               label=f"closed form, h=250 um ({e250:.4f})")
    ax.axhline(e254, color="tab:red", ls=":", lw=1.2,
               label=f"closed form, h=254 um ({e254:.4f})")
    ax.axhspan(e250 * 1.015, e250 * 1.05, color="tab:orange", alpha=0.18)
    ax.set_xlabel("cell size dx [um]")
    ax.set_ylabel("eps_eff")
    ax.invert_xaxis()
    ax.legend(fontsize=7.2, loc="best")
    ax.text(0.02, 0.03, "orange: where H-c would be supported\n(>= +1.5 % over h=250 um)",
            transform=ax.transAxes, fontsize=7, va="bottom")

    ax = axes[1]
    dev = 100 * (eps / e250 - 1.0)
    ax.plot(dx, dev, "o-", lw=1.4, color="tab:blue")
    ax.axhline(0.0, color="tab:green", ls="--", lw=1.2)
    ax.axhline(+0.5, color="0.6", lw=0.8)
    ax.axhline(-0.5, color="0.6", lw=0.8)
    ax.axhline(+1.5, color="tab:orange", lw=1.0)
    ax.set_xlabel("cell size dx [um]")
    ax.set_ylabel("eps_eff / closed form (h=250 um) - 1  [%]")
    ax.invert_xaxis()
    ax.text(0.03, 0.9, "grey: +-0.5 % (H-c killed inside)\norange: +1.5 % (H-c supported above)",
            transform=ax.transAxes, fontsize=7, va="top")

    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
