#!/usr/bin/env python3
"""Is the coax line's outer conductor a closed screen? — a pre-declared diagnostic.

The chain battery measured, on the thru control at 4 annulus cells, a fitted
phase constant 18 % above ``omega sqrt(eps_fill) / c`` and 15 % of the column
power missing, both shrinking with the mesh. A homogeneously filled PEC-bounded
line carries TEM at ``beta = omega sqrt(eps) / c`` whatever the staircase does to
the cross-section — the staircase moves ``Z_TEM``, not ``beta`` — so that pair is
not explained by "the annulus is under-resolved".

The hypothesis under test (the leader's, pre-declared before this ran): the
one-cell staircased ring is not CLOSED — cells touching only at corners — so the
inner mode couples to the air between the shell's outside and the box walls. Two
coupled lines with ``beta_in > beta_out`` repel, pushing the inner-dominated
supermode above ``beta_PTFE``, and power that reaches the outer region leaves
through the z absorber without crossing either port plane.

Arms, all on the thru DUT (no ``eps_scale``, so the lane keeps its numpy path and
its ring-down witness):

  0  as shipped — the control, which must reproduce the battery's own rung-4 thru
  1  shell 3 cells thick, inner radius as shipped, extending OUTWARD
  2  every cell outside the shell's inner radius is conductor — sealed by
     construction
  3  shell inner radius at the DECLARED outer radius b (the dielectric fills
     a..b as drawn), extending outward 3 cells

Nothing under ``rfx/`` changes. The arms are installed by rebinding
``rfx.sources.coaxial_port.stamp_coaxial_line``, which
``compute_coaxial_two_port`` imports from that module at call time, and
``rfx.simulation.run`` is wrapped — not replaced — so this script can read the
materials the lane actually solved and the DFT planes it actually filled.

Usage::

    PYTHONPATH=. python scripts/diagnostics/coax_shell_seal_diagnostic.py \
        --arm 0 --rung 4 --out <run-dir> --run-id <id>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax.numpy as jnp  # noqa: E402

import rfx.simulation as _rfx_simulation  # noqa: E402
import rfx.sources.coaxial_port as _cp  # noqa: E402
from rfx.geometry.csg import Cylinder  # noqa: E402
from rfx.sources.coaxial_port import PEC_SIGMA, PTFE_EPS_R  # noqa: E402

# The battery's own driver: the control arm has to be the battery's own
# configuration, so the board, the band, the record length and the probe ladder
# are imported rather than restated.
_spec = importlib.util.spec_from_file_location(
    "coax_battery", REPO / "scripts" / "diagnostics" / "coax_chain_battery_measure.py")
cb = importlib.util.module_from_spec(_spec)
sys.modules["coax_battery"] = cb
_spec.loader.exec_module(cb)

SCHEMA = "rfx.coax_shell_seal_diagnostic"
SCHEMA_VERSION = 1
DRIVER = "scripts/diagnostics/coax_shell_seal_diagnostic.py"
BATTERY = "scripts/diagnostics/coax_chain_battery_measure.py"
PREDECLARATION = "docs/design_notes/coax_chain_battery_predeclaration.md"

ARMS = (0, 1, 2, 3)
THICK_CELLS = 3
# Arm 2 fills the cross-section with conductor, but PEC must not reach the CPML:
# running it into the absorber is documented unstable (stamp_coaxial_line's own
# docstring, verified NaN). The fill therefore stops this many cells short of the
# pad on each lateral face, and the margin is recorded with the measurement.
ARM2_PAD_MARGIN_CELLS = 2

_ORIGINAL_STAMP = _cp.stamp_coaxial_line
_ORIGINAL_RUN = _rfx_simulation.run

# What the wrapped run() saw, one entry per drive.
_CAPTURED: list[dict] = []


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------

def _cylinder_mask(grid, center, radius, height):
    return Cylinder(center=center, radius=radius, height=height, axis="z").mask(grid)


def _radial_grid(grid, center_xy):
    """Cell-centre radius, on the same convention the stamps use."""
    dx = float(grid.dx)
    i = np.arange(int(grid.shape[0]))
    j = np.arange(int(grid.shape[1]))
    x = (i - int(grid.pad_x_lo)) * dx - float(center_xy[0])
    y = (j - int(grid.pad_y_lo)) * dx - float(center_xy[1])
    return np.hypot(x[:, None], y[None, :])


def make_stamp(arm: int):
    """The arm's replacement for ``stamp_coaxial_line``.

    Same signature and same return contract ``(materials, shell_inner_radius)``,
    so the caller's feed stamps land on the same annulus.
    """

    def stamp(grid, materials, *, center_xy, z_lo_index, z_hi_index,
              pin_radius=_cp.SMA_PIN_RADIUS, outer_radius=_cp.SMA_OUTER_RADIUS,
              eps_r=PTFE_EPS_R):
        if arm == 0:
            return _ORIGINAL_STAMP(
                grid, materials, center_xy=center_xy, z_lo_index=z_lo_index,
                z_hi_index=z_hi_index, pin_radius=pin_radius,
                outer_radius=outer_radius, eps_r=eps_r)

        dz = float(grid.dx)
        z_lo = (int(z_lo_index) - grid.pad_z_lo) * dz
        z_hi = (int(z_hi_index) - grid.pad_z_lo) * dz
        zc = 0.5 * (z_lo + z_hi)
        height = (z_hi - z_lo) + 2.0 * dz
        center = (float(center_xy[0]), float(center_xy[1]), zc)
        a, b = float(pin_radius), float(outer_radius)

        shipped_thickness = min(dz, 0.5 * (b - a))
        if arm in (1, 2):
            shell_inner_radius = b - shipped_thickness
        else:                                   # arm 3: the dielectric fills a..b
            shell_inner_radius = b

        pin_mask = _cylinder_mask(grid, center, a, height)
        shell_inner_mask = _cylinder_mask(grid, center, shell_inner_radius, height)
        axial = _cylinder_mask(grid, center, max(b, shell_inner_radius) * 10.0, height)

        if arm == 2:
            # Every cell outside the shell's inner radius, out to a margin short
            # of the CPML pad.
            keep = np.zeros(tuple(int(s) for s in grid.shape), dtype=bool)
            i0 = int(grid.pad_x_lo) + ARM2_PAD_MARGIN_CELLS
            i1 = int(grid.shape[0]) - int(grid.pad_x_hi) - ARM2_PAD_MARGIN_CELLS
            j0 = int(grid.pad_y_lo) + ARM2_PAD_MARGIN_CELLS
            j1 = int(grid.shape[1]) - int(grid.pad_y_hi) - ARM2_PAD_MARGIN_CELLS
            keep[i0:i1, j0:j1, :] = True
            shell = axial & keep & ~shell_inner_mask
        else:
            outer_extent = shell_inner_radius + THICK_CELLS * dz
            shell = _cylinder_mask(grid, center, outer_extent, height) & ~shell_inner_mask

        eps = np.array(materials.eps_r)
        sig = np.array(materials.sigma)
        eps = np.where(shell, 1.0, eps)
        sig = np.where(shell, PEC_SIGMA, sig)
        fill = shell_inner_mask & ~pin_mask
        eps = np.where(fill, float(eps_r), eps)
        sig = np.where(fill, 0.0, sig)
        eps = np.where(pin_mask, 1.0, eps)
        sig = np.where(pin_mask, PEC_SIGMA, sig)
        return (materials._replace(eps_r=jnp.asarray(eps), sigma=jnp.asarray(sig)),
                shell_inner_radius)

    return stamp


def wrapped_run(grid, materials, n_steps, **kw):
    """The lane's own ``run``, with the inputs and the DFT planes kept.

    A wrapper, not a replacement: the solve is the shipped one, and nothing it
    returns is modified.
    """
    result = _ORIGINAL_RUN(grid, materials, n_steps, **kw)
    _CAPTURED.append({"grid": grid, "materials": materials,
                      "dft_planes": result.dft_planes,
                      "n_steps": int(n_steps)})
    return result


# ---------------------------------------------------------------------------
# what each arm has to report
# ---------------------------------------------------------------------------

def _label(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    """Flood-fill labelling of ``mask``, 4- or 8-connected. Written here rather
    than imported, so the count is this script's own arithmetic."""
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if connectivity == 8:
        nbrs = nbrs + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    labels = np.zeros(mask.shape, dtype=np.int32)
    nxx, nyy = mask.shape
    current = 0
    for si in range(nxx):
        for sj in range(nyy):
            if not mask[si, sj] or labels[si, sj]:
                continue
            current += 1
            stack = [(si, sj)]
            labels[si, sj] = current
            while stack:
                ci, cj = stack.pop()
                for di, dj in nbrs:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < nxx and 0 <= nj < nyy and mask[ni, nj] \
                            and not labels[ni, nj]:
                        labels[ni, nj] = current
                        stack.append((ni, nj))
    return labels, current


def connected_components_of_the_non_conductor(sigma_slice: np.ndarray) -> dict:
    """Is the outer conductor a closed screen on this z slice?

    The leader asked for the 4-connected components of the NON-conductor region.
    That alone cannot answer the hypothesis, and the reason is the hypothesis
    itself: a ring whose cells touch only at CORNERS is 8-connected but not
    4-connected, and it leaves the free regions 4-DISCONNECTED while a diagonal
    path still runs between them. So all four counts are taken:

      free 4-connected   — one component spanning both sides means a face-width gap
      free 8-connected   — one component means a path exists, diagonal included
      ring 4-connected   — the ring is face-closed
      ring 8-connected   — the ring is closed at worst corner-to-corner

    In the Yee lattice a corner contact does not kill the tangential E edge that
    runs through it, so "free 4-disconnected but 8-connected" is the geometry the
    hypothesis names.
    """
    solid = np.asarray(sigma_slice) > 1.0
    free = ~solid
    free4, n_free4 = _label(free, 4)
    free8, n_free8 = _label(free, 8)
    solid4, n_solid4 = _label(solid, 4)
    solid8, n_solid8 = _label(solid, 8)
    return {
        "n_components_free_4connected": n_free4,
        "n_components_free_8connected": n_free8,
        "n_components_conductor_4connected": n_solid4,
        "n_components_conductor_8connected": n_solid8,
        "n_conductor_cells": int(solid.sum()),
        "labels_free_4": free4, "labels_free_8": free8,
    }


def exterior_energy_ratio(grid, materials, dft_planes, center_xy, b: float,
                          probe_z: list[int]) -> dict:
    """Field energy outside the declared outer radius, relative to inside.

    The lane fills one ``ex`` and one ``ey`` DFT plane per probe plane, in the
    order ``probes_bot + probes_top``, so ``eps |E_t|^2`` over a cross-section is
    available without a snapshot. The accumulator is ``(n_freqs, nx, ny)`` —
    checked against ``init_dft_plane_probe`` rather than assumed — and each
    plane's own z index comes from the lane's probe order, so the permittivity
    weight is the one at that plane.

    It is a TRANSVERSE-energy proxy, not the total: ``E_z`` and the magnetic part
    are not in these planes. Said here rather than implied.
    """
    if not dft_planes:
        return {"available": False, "why": "the wrapped run returned no DFT planes"}
    r = _radial_grid(grid, center_xy)
    eps = np.asarray(materials.eps_r)
    inside = r <= b
    outside = ~inside
    n_pairs = min(len(probe_z), len(dft_planes) // 2)
    ins, out = [], []
    for pi in range(n_pairs):
        ex = np.asarray(dft_planes[pi * 2 + 0].accumulator)
        ey = np.asarray(dft_planes[pi * 2 + 1].accumulator)
        if ex.ndim != 3 or ex.shape[1:] != r.shape:
            return {"available": False,
                    "why": f"plane {pi} has shape {ex.shape}, expected (n_freqs,"
                           f" {r.shape[0]}, {r.shape[1]})"}
        w = eps[:, :, int(probe_z[pi])]                       # (nx, ny)
        e2 = (np.abs(ex) ** 2 + np.abs(ey) ** 2) * w[None, :, :]
        ins.append(e2[:, inside].sum(axis=1))
        out.append(e2[:, outside].sum(axis=1))
    if not ins:
        return {"available": False, "why": "no plane had the expected shape"}
    ins = np.asarray(ins, dtype=float)                        # (n_planes, n_freqs)
    out = np.asarray(out, dtype=float)
    ratio = out / np.maximum(ins, 1e-300)
    return {
        "available": True,
        "what": ("sum of eps (|Ex|^2 + |Ey|^2) over each probe plane, split at the "
                 "declared outer radius; a TRANSVERSE-field proxy — Ez and the "
                 "magnetic energy are not in these planes"),
        "probe_z_indices": [int(z) for z in probe_z[:n_pairs]],
        "per_plane_per_freq_ratio": ratio.astype(float).tolist(),
        "max_ratio": float(ratio.max()), "median_ratio": float(np.median(ratio)),
        "ratio_at_lowest_bin": ratio[:, 0].astype(float).tolist(),
        "ratio_at_highest_bin": ratio[:, -1].astype(float).tolist(),
        "inside_sum": ins.astype(float).tolist(),
        "outside_sum": out.astype(float).tolist(),
    }


def beta_from_s21_phase(freqs: np.ndarray, s21: np.ndarray, l12: float,
                        eps_fill: float) -> dict:
    """A phase constant that does not pass through the matrix-pencil fit.

    On a matched thru ``S21 ~ exp(-gamma L12)``, so ``angle(S21) = -beta L12``
    modulo 2 pi. The absolute branch is unknown, but its SLOPE is not: with the
    bins 100 MHz apart the phase turns about 0.18 rad per bin, far inside the
    unwrapping limit, so ``d(angle)/d(omega) = -L12 / v_p`` is recoverable
    directly. ``beta = omega / v_p`` follows without ever choosing a branch.
    """
    freqs = np.asarray(freqs, dtype=float)
    omega = 2.0 * np.pi * freqs
    ph = np.unwrap(np.angle(np.asarray(s21)))
    step = np.abs(np.diff(ph))
    slope, intercept = np.polyfit(omega, ph, 1)
    v_p = -l12 / slope if slope != 0.0 else float("nan")
    beta_phase = omega / v_p if np.isfinite(v_p) else np.full_like(omega, np.nan)
    beta_analytic = omega * math.sqrt(eps_fill) / cb.C0
    # The branch that the slope implies, reported per bin as a cross-check on
    # the linear fit rather than as a second method.
    k = np.round((beta_phase * l12 + ph) / (2.0 * np.pi))
    beta_branch = (-(ph - 2.0 * np.pi * k)) / l12
    return {
        "l12_m": l12,
        "unwrapped_phase_rad": ph.astype(float).tolist(),
        "max_bin_to_bin_phase_step_rad": float(step.max()) if step.size else 0.0,
        "unwrap_safe": bool(step.max() < math.pi) if step.size else True,
        "slope_rad_per_rad_per_s": float(slope),
        "intercept_rad": float(intercept),
        "phase_velocity_m_per_s": float(v_p),
        "eps_eff_from_phase_slope": float((cb.C0 / v_p) ** 2) if np.isfinite(v_p) else None,
        "beta_phase_rad_per_m": beta_phase.astype(float).tolist(),
        "beta_analytic_rad_per_m": beta_analytic.astype(float).tolist(),
        "beta_ratio_phase_over_analytic": (beta_phase / beta_analytic).astype(float).tolist(),
        "max_beta_ratio_phase": float(np.max(beta_phase / beta_analytic)),
        "beta_per_bin_branch_resolved_rad_per_m": beta_branch.astype(float).tolist(),
    }


def probe_plane_coordinates(grid, layout: dict) -> dict:
    """The z the extractor assigns to each probe plane, against the grid's own
    node coordinate for the same index, and the realized axial cell size.

    A position-scale error would raise a fitted beta without touching Z0, which
    is the alternative to the leak hypothesis.
    """
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    gc = coords_from_uniform_grid(grid)
    nodes_z = np.asarray(gc.z, dtype=float)
    dz = float(grid.dx)
    rows = []
    for tag in ("probes_bot", "probes_top"):
        for z in layout[tag]:
            extractor = (int(z) - int(grid.pad_z_lo)) * dz
            node = float(nodes_z[int(z)]) if int(z) < nodes_z.size else float("nan")
            rows.append({"tag": tag, "index": int(z),
                         "extractor_z_m": extractor, "grid_node_z_m": node,
                         "difference_m": extractor - node})
    diffs = np.array([r["difference_m"] for r in rows], dtype=float)
    spacing_extractor = np.diff([r["extractor_z_m"] for r in rows
                                 if r["tag"] == "probes_bot"])
    spacing_node = np.diff([r["grid_node_z_m"] for r in rows
                            if r["tag"] == "probes_bot"])
    return {
        "realized_dx_m": dz,
        "grid_node_spacing_m": float(np.median(np.diff(nodes_z))) if nodes_z.size > 1 else None,
        "planes": rows,
        "max_abs_difference_m": float(np.max(np.abs(diffs))),
        "extractor_probe_spacing_m": spacing_extractor.astype(float).tolist(),
        "grid_node_probe_spacing_m": spacing_node.astype(float).tolist(),
        "spacing_agrees": bool(np.allclose(spacing_extractor, spacing_node,
                                           rtol=0, atol=1e-15)),
        "pad_z_lo": int(grid.pad_z_lo), "pad_z_hi": int(grid.pad_z_hi),
        "grid_shape": [int(s) for s in grid.shape],
    }


def realized_cross_section(grid, materials, center_xy, a: float, b: float,
                           z_index: int) -> dict:
    """What the arm actually built, measured from the solved material arrays."""
    r = _radial_grid(grid, center_xy)
    eps = np.asarray(materials.eps_r)[:, :, z_index]
    sig = np.asarray(materials.sigma)[:, :, z_index]
    dz = float(grid.dx)
    conductor = sig > 1.0
    fill = (eps > 1.0 + 1e-6) & ~conductor
    pin = conductor & (r < b)
    out = {
        "z_index": int(z_index),
        "n_conductor_cells": int(conductor.sum()),
        "n_fill_cells": int(fill.sum()),
        "fill_eps_r_median": float(np.median(eps[fill])) if fill.any() else None,
        "realized_fill_radius_min_m": float(r[fill].min()) if fill.any() else None,
        "realized_fill_radius_max_m": float(r[fill].max()) if fill.any() else None,
        "realized_conductor_radius_min_m": float(r[conductor].min()) if conductor.any() else None,
        "realized_conductor_radius_max_m": float(r[conductor].max()) if conductor.any() else None,
        "realized_pin_radius_max_m": float(r[pin & (r < 0.5 * (a + b))].max())
        if (pin & (r < 0.5 * (a + b))).any() else None,
        "declared_pin_radius_m": a, "declared_outer_radius_m": b,
    }
    r_in = out["realized_pin_radius_max_m"]
    r_out = out["realized_fill_radius_max_m"]
    if r_in and r_out and r_out > r_in:
        out["b_over_a_realized"] = r_out / r_in
        out["z_tem_on_realized_radii_ohm"] = _cp.coaxial_tem_characteristic_impedance(
            r_in, r_out, float(PTFE_EPS_R))
    out["b_over_a_declared"] = b / a
    out["z_tem_on_declared_radii_ohm"] = _cp.coaxial_tem_characteristic_impedance(
        a, b, float(PTFE_EPS_R))
    out["annulus_cells_declared"] = (b - a) / dz
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", type=int, choices=ARMS, required=True)
    ap.add_argument("--rung", type=int, choices=cb.RUNGS, default=4)
    ap.add_argument("--record-units", type=float, default=cb.DEFAULT_RECORD_UNITS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shell_seal_arm{args.arm}_rung{args.rung}.json"

    rec = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "driver": DRIVER, "battery_driver": BATTERY, "predeclaration": PREDECLARATION,
        "arm": args.arm, "rung_annulus_cells": args.rung, "dut": "thru",
        "arm_description": {
            0: "as shipped (control)",
            1: f"shell {THICK_CELLS} cells thick, inner radius as shipped, extending outward",
            2: ("every cell outside the shell's inner radius is conductor, stopping "
                f"{ARM2_PAD_MARGIN_CELLS} cells short of the CPML pad laterally"),
            3: (f"shell inner radius at the DECLARED outer radius b, extending outward "
                f"{THICK_CELLS} cells"),
        }[args.arm],
        "hypothesis": (
            "the shipped one-cell staircased ring is not closed, so the inner mode "
            "couples to the exterior; two coupled lines with beta_in > beta_out repel "
            "and the inner-dominated supermode sits above beta_PTFE, while power that "
            "reaches the exterior leaves through the z absorber without crossing a "
            "port plane"),
        "provenance": cb.provenance(args),
        "bar": cb.BAR,
    }
    cb._write(out_path, rec)

    # Install the arm. Both are rebindings of module attributes the lane looks up
    # at call time; nothing under rfx/ is edited.
    _cp.stamp_coaxial_line = make_stamp(args.arm)
    _rfx_simulation.run = wrapped_run
    _CAPTURED.clear()

    sim = cb.build_sim(args.rung, "thru")
    grid = sim._build_grid()
    layout = cb.axial_layout(grid, "two_port")
    port = sim._coaxial_ports[0]
    a, b = float(port.pin_radius), float(port.outer_radius)
    center_xy = (float(port.position[0]), float(port.position[1]))
    n_steps = cb.record_steps(grid, args.record_units)
    rec["n_steps"] = n_steps
    rec["record_units"] = float(args.record_units)
    rec["declared"] = cb.declared(args.rung, "thru")
    rec["layout"] = layout
    rec["probe_coordinates"] = probe_plane_coordinates(grid, layout)
    rec["cost"] = cb.cost_estimate(grid, n_steps, cb.N_FREQS,
                                   2 * 2 * cb.PROBE_COUNT, 2)
    cb._write(out_path, rec)

    cb._log(f"arm {args.arm} rung {args.rung}: {rec['arm_description']}")
    with cb._Captured() as cap:
        res = cb.solve_two_port(sim, n_steps=n_steps)
    cb._log_two_port(f"shell arm {args.arm} rung {args.rung}", res)

    rec["warnings"] = cap.warnings
    rec["wall_s"] = cap.wall
    rec["peak_memory"] = cb.peak_memory()
    rec["result"] = cb._two_port_record(res)
    cb._write(out_path, rec)

    # --- what the arm actually built, and where the energy went --------------
    if not _CAPTURED:
        raise RuntimeError("the wrapped run captured nothing — the lane did not "
                           "call rfx.simulation.run")
    first = _CAPTURED[0]
    g, mats = first["grid"], first["materials"]
    z_probe = int(layout["probes_bot"][-1])
    rec["realized_cross_section"] = realized_cross_section(
        g, mats, center_xy, a, b, z_probe)

    comp = connected_components_of_the_non_conductor(
        np.asarray(mats.sigma)[:, :, z_probe])
    free4 = comp.pop("labels_free_4")
    free8 = comp.pop("labels_free_8")
    r = _radial_grid(g, center_xy)
    ann = (r > a) & (r < b)
    ext = r > b + 4.0 * float(g.dx)
    for tag, labels in (("4connected", free4), ("8connected", free8)):
        al = sorted({int(v) for v in labels[ann & (labels > 0)]})
        el = sorted({int(v) for v in labels[ext & (labels > 0)]})
        comp[f"annulus_labels_{tag}"] = al
        comp[f"exterior_labels_{tag}"] = el
        comp[f"annulus_and_exterior_are_one_component_{tag}"] = bool(set(al) & set(el))
    comp["ring_is_face_closed"] = not comp[
        "annulus_and_exterior_are_one_component_8connected"]
    comp["ring_touches_only_at_corners"] = bool(
        not comp["annulus_and_exterior_are_one_component_4connected"]
        and comp["annulus_and_exterior_are_one_component_8connected"])
    comp["what"] = (
        "free 4-connected across = a face-width gap; free 8-connected across but "
        "not 4-connected = a ring whose cells touch only at corners, which is the "
        "geometry the hypothesis names")
    rec["connected_components"] = comp
    cb._write(out_path, rec)          # persist BEFORE the optional stage below
    np.save(out_dir / f"shell_seal_arm{args.arm}_rung{args.rung}_sigma_slice.npy",
            np.asarray(mats.sigma)[:, :, z_probe])
    np.save(out_dir / f"shell_seal_arm{args.arm}_rung{args.rung}_labels4.npy", free4)
    np.save(out_dir / f"shell_seal_arm{args.arm}_rung{args.rung}_labels8.npy", free8)

    try:
        rec["exterior_energy"] = exterior_energy_ratio(
            g, mats, first["dft_planes"], center_xy, b,
            list(layout["probes_bot"]) + list(layout["probes_top"]))
    except Exception as exc:                      # noqa: BLE001 — recorded, never fatal
        rec["exterior_energy"] = {"available": False, "why": f"{type(exc).__name__}: {exc}"}
    cb._write(out_path, rec)

    # --- the two phase-constant estimates ------------------------------------
    S = np.asarray(res.s_params)
    freqs = np.asarray(res.freqs, dtype=float)
    planes = np.asarray(res.reference_planes, dtype=float)
    l12 = float(abs(planes[0] - planes[1]))
    rec["beta_from_matrix_pencil"] = cb.measured_beta(
        rec["result"]["gamma"], freqs, float(PTFE_EPS_R))
    rec["beta_from_s21_phase"] = beta_from_s21_phase(
        freqs, S[1, 0, :], l12, float(PTFE_EPS_R))

    col = np.sum(np.abs(S) ** 2, axis=0)
    rec["summary"] = {
        "reference_planes_m": planes.tolist(), "l12_m": l12,
        "abs_s21": np.abs(S[1, 0, :]).astype(float).tolist(),
        "abs_s11": np.abs(S[0, 0, :]).astype(float).tolist(),
        "s11_db": (20.0 * np.log10(np.maximum(np.abs(S[0, 0, :]), 1e-300))
                   ).astype(float).tolist(),
        "max_column_power": float(col.max()),
        "min_column_power": float(col.min()),
        "reciprocity_metric": float(np.abs(S[1, 0, :] - S[0, 1, :]).max()
                                    / max(float(np.abs(S).max()), 1e-300)),
        "max_beta_ratio_matrix_pencil": rec["beta_from_matrix_pencil"]["max_beta_ratio"],
        "max_beta_ratio_s21_phase": rec["beta_from_s21_phase"]["max_beta_ratio_phase"],
        "eps_eff_matrix_pencil": rec["beta_from_matrix_pencil"]["mean_eps_eff_fitted"],
        "eps_eff_s21_phase": rec["beta_from_s21_phase"]["eps_eff_from_phase_slope"],
        "exterior_energy_max_ratio": rec["exterior_energy"].get("max_ratio"),
        "non_conductor_components_4connected": comp["n_components_free_4connected"],
        "non_conductor_components_8connected": comp["n_components_free_8connected"],
        "shell_has_a_face_width_gap": comp[
            "annulus_and_exterior_are_one_component_4connected"],
        "shell_leaks_diagonally": comp[
            "annulus_and_exterior_are_one_component_8connected"],
        "ring_touches_only_at_corners": comp["ring_touches_only_at_corners"],
        "settling": rec["result"]["settling"],
    }
    cb._write(out_path, rec)

    s = rec["summary"]
    cb._log(f"arm {args.arm} rung {args.rung}: beta ratio (pencil) "
            f"{s['max_beta_ratio_matrix_pencil']:.4f}, (S21 phase) "
            f"{s['max_beta_ratio_s21_phase']:.4f} | max column power "
            f"{s['max_column_power']:.5f} | exterior/interior energy "
            f"{s['exterior_energy_max_ratio']} | free components 4/8 "
            f"{s['non_conductor_components_4connected']}/"
            f"{s['non_conductor_components_8connected']} | face-gap "
            f"{s['shell_has_a_face_width_gap']} | diagonal leak "
            f"{s['shell_leaks_diagonally']} | corners-only "
            f"{s['ring_touches_only_at_corners']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
