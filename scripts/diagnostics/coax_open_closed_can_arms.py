#!/usr/bin/env python3
"""The coaxial open end inside a closed metal can: the arms of a pre-declared cause test.

An open-circuited lossless coaxial line reflects what it receives and no more,
so its reflection can reach |Gamma| = 1 but not exceed it. The one-port lane
(``compute_coaxial_line_reflection``) reads the PTFE-filled SMA line's open
above that at 9 annulus cells, and the excess grows with the record, while the
short on the same line reads 1.0012 (issue 1218). The lane absorbs on z only:
its x and y pads are vacuum and the grid's outer faces are PEC, so the line
sits inside a closed metal can. Both of its line ends are unshielded: at the
open the pin and the shell stop in one plane, and behind the matched feed
resistor they stop one cell past it. The two-port lane has such an end behind
each of its two feeds.

The arms, as ``docs/design_notes/20260924_coax_open_closed_can_predeclaration.md``
declares them (9 annulus cells, the battery's 8 x 8 mm board):

* O0 / S0 / T0 -- the open, the short and the two-port thru as shipped;
* O1 / S1 -- the same open and short with the lateral pads absorbing (CPML on
  x, y and z); conductors, walls and layout unchanged;
* O2 -- a shielded open: the short's own PEC disk at the DUT plane closes the
  shell, and the pin ends one annulus width (9 cells) above it;
* T2 -- the thru with both line ends shielded: behind each feed resistor the
  pin ends, the shell runs on one annulus width past it and a PEC cap closes
  it, 3 whole cells short of the z absorber;
* W1 (report only) -- the TE cutoffs of the region between the shell and the
  can, from a 2-D eigen-solve on the realized PEC edges of the battery's can
  and of the 21 mm can the stored 16 mm board makes.

The lanes refuse extra geometry and any ``cpml_axes`` but ``"z"``, so every arm
runs through a copy of the lane's own steps kept here (``_solve_one_port``,
``_solve_two_port``: the concrete, ``eps_scale=None`` path of
``rfx/sparams/coax.py``, line for line). The shipped lane code is untouched.
O0, S0 and T0 are solved twice in their job, by the lane itself (the arm's
result) and by the copy, and the record says whether the two S agree bit for
bit: that is what lets O1, O2, S1 and T2 be read as the lane with one thing
changed.

Every arm asserts its realized geometry in cells before the first step (cap
planes, pin ends, the gap between them, absorbing axes, probe planes against
the pin ends, where T2's moved sources realize their TFSF planes) and stores
it. Point probes record E inside the line and in the region outside the shell
through the whole record, as a time-domain witness beside the per-bin S.

Two layout choices the declaration leaves open, recorded in every record:

* O2's probe array starts as many cells above its pin end (node 30) as the
  shipped open's starts above its conductor end (node 19): 9. The reference
  plane stays at the lane's DUT plane, which is where the cap is. The derived
  ``|B/A|`` at the probe centroid is the reflection with no extrapolation.
* T2's caps sit 3 cells from the z pad faces, not the declared minimum of 2.
  The lane places a TFSF plane at ``round(position / dx)`` of a half-cell
  position, so near the line ends only every other plane is reachable; with 2
  cells the inward shifts are 11 and 13 cells and T2's sources would land one
  cell off their feeds' relation in T0. With 3 the shifts are 12 and 14 and
  every plane moves by exactly the shift (asserted).

This script writes numbers and no verdict: the assembled artifact carries
each measurement beside the declared threshold.

Usage (from a clean checkout; the ``rfx`` import must resolve to this tree)::

    PYTHONPATH=. python scripts/diagnostics/coax_open_closed_can_arms.py \\
        --arm O2 --out <run-dir> --run-id <id>
    PYTHONPATH=. python scripts/diagnostics/coax_open_closed_can_arms.py \\
        --w1 --out <run-dir> --run-id <id>
    PYTHONPATH=. python scripts/diagnostics/coax_open_closed_can_arms.py \\
        --layout-only [--arm T2]
    PYTHONPATH=. python scripts/diagnostics/coax_open_closed_can_arms.py \\
        --assemble --out <dir with every record> --run-index <json> \\
        --artifact-out tests/fixtures/coax_chain_battery/open_closed_can_arms.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import coax_chain_battery_measure as battery  # noqa: E402  (puts the repo on sys.path)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rfx.api._spec import CoaxialLineReflectionResult, CoaxialTwoPortResult  # noqa: E402
from rfx.probes.probes import init_dft_plane_probe  # noqa: E402
from rfx.simulation import ProbeSpec, run as _run  # noqa: E402
from rfx.sources.coaxial_port import (  # noqa: E402
    PTFE_EPS_R,
    CoaxialPort,
    build_coaxial_tem_plane_source_specs,
    coaxial_line_plane_voltage,
    coaxial_line_reflection_from_plane_voltages,
    coaxial_tem_characteristic_impedance,
    stamp_coaxial_annular_resistor,
    stamp_coaxial_line,
    stamp_coaxial_short_plane,
)
from rfx.sparams._common import (  # noqa: E402
    _assemble_coaxial_two_port_from_voltages,
    _finalize_sparam_result,
    _warn_if_ringdown_truncated,
)
from rfx.sparams.coax import _coax_pec_edge_masks  # noqa: E402

SCHEMA = "rfx.coax_open_closed_can_arms"
SCHEMA_VERSION = 1
DRIVER = "scripts/diagnostics/coax_open_closed_can_arms.py"
ARTIFACT = "tests/fixtures/coax_chain_battery/open_closed_can_arms.json"
PREDECLARATION = "docs/design_notes/20260924_coax_open_closed_can_predeclaration.md"
PREDECLARATION_COMMIT = "043556f2c8d4f9f5676ab8b59f84303d588f7dc1"
OPEN_ABSORBER_ARTIFACT = "tests/fixtures/coax_chain_battery/open_absorber_diagnostic.json"

RUNG = battery.CLAIMS_RUNG            # 9 annulus cells: dx = (b - a) / 9
GAP_CELLS = RUNG                      # one annulus width between a cap and the pin end
CAP_CLEARANCE_CELLS = 3               # T2: whole cells from a cap's outer face to its z pad face
WIDE_LATERAL_M = 0.016                # the stored 16 mm board: a 21.05 mm can at 9 cells
SMOKE_UNITS = (1.0, 2.0)               # smoke: a short and a doubled record
W1_N_MODES = 8
WITNESS_OFFSET_CELLS = 3              # outer-region probes: this far outside the shell / the ends
FIELD_SCALE = 1.0e4                   # the lanes' default
COND_WARN = 1.0e3                     # the two-port lane's default

ARMS = {
    "O0": dict(lane="one_port", dut="open", cpml_axes="z", shield=False, shipped=True,
               units=(12.0, 24.0),
               what="open, as shipped: absorbers on z only, pin and shell end in one plane"),
    "O1": dict(lane="one_port", dut="open", cpml_axes="xyz", shield=False, shipped=False,
               units=(12.0, 24.0),
               what=("open with CPML on x, y and z: the lateral pads absorb; conductors, "
                     "walls and layout as shipped")),
    "O2": dict(lane="one_port", dut="open", cpml_axes="z", shield=True, shipped=False,
               units=(12.0, 24.0),
               what=("shielded open: the short's PEC disk at the DUT plane closes the shell; "
                     "the pin ends one annulus width above it; lateral boundary as shipped")),
    "S0": dict(lane="one_port", dut="short", cpml_axes="z", shield=False, shipped=True,
               units=(12.0,), what="short, as shipped"),
    "S1": dict(lane="one_port", dut="short", cpml_axes="xyz", shield=False, shipped=False,
               units=(12.0,), what="short with CPML on x, y and z; otherwise as shipped"),
    "T0": dict(lane="two_port", dut="thru", cpml_axes="z", shield=False, shipped=True,
               units=(12.0,), what="two-port thru, as shipped"),
    "T2": dict(lane="two_port", dut="thru", cpml_axes="z", shield=True, shipped=False,
               units=(12.0,),
               what=("two-port thru with both line ends shielded: behind each feed the pin "
                     "ends, the shell runs one annulus width past it and a PEC cap closes "
                     "it; feeds, sources and probes moved inward by the same whole cells")),
}

# The thresholds the pre-declaration names.
MAX_ABS_GAMMA = 1.02
DOUBLING_SHIFT_MAX = (10 ** (battery.BAR["magnitude_db"] / 20.0) - 1.0) / 10.0   # 0.025893
MIN_ABS_GAMMA_SHIELDED = 0.98
P3_MAX_ABS_DELTA = 0.005
COLUMN_POWER_T2_FLOOR = 0.995
COLUMN_POWER_SAME_WITHIN = 0.005


def units_for(arm: str, smoke: bool = False) -> tuple[float, ...]:
    """The record lengths an arm runs at; a smoke run keeps the arm's count of
    records (one or two) at SMOKE_UNITS."""
    n = len(ARMS[arm]["units"])
    return SMOKE_UNITS[:n] if smoke else ARMS[arm]["units"]


def record_name(arm: str, units: float, smoke: bool = False) -> str:
    return f"{'smoke_' if smoke else ''}closed_can_{arm}_u{units:g}.json"


def w1_name(smoke: bool = False) -> str:
    return f"{'smoke_' if smoke else ''}closed_can_W1.json"


# ---------------------------------------------------------------------------
# realized geometry helpers
# ---------------------------------------------------------------------------

def _node_radius(grid, center_xy) -> np.ndarray:
    """Distance of every (i, j) node from the line's axis, on the stamps'
    convention ``x = (i - pad_x_lo) * dx``."""
    dx = float(grid.dx)
    x = (np.arange(int(grid.shape[0])) - int(grid.pad_x_lo)) * dx - float(center_xy[0])
    y = (np.arange(int(grid.shape[1])) - int(grid.pad_y_lo)) * dx - float(center_xy[1])
    return np.hypot(x[:, None], y[None, :])


def _axis_node(grid, center_xy) -> tuple[int, int]:
    dx = float(grid.dx)
    return (int(round(float(center_xy[0]) / dx)) + int(grid.pad_x_lo),
            int(round(float(center_xy[1]) / dx)) + int(grid.pad_y_lo))


def _runs(idx) -> list[list[int]]:
    """Contiguous runs of a sorted integer sequence, as ``[first, last]``."""
    idx = [int(v) for v in idx]
    out: list[list[int]] = []
    for v in idx:
        if out and v == out[-1][1] + 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return out


def _tangential_pec_node_runs(edges, i: int, j: int) -> list[list[int]]:
    """The node planes on which a tangential E edge incident to node (i, j)
    is shorted, as runs. An E_x edge sits at node i and i + 1, an E_y edge at
    node j and j + 1, so node (i, j) is touched by E_x[i], E_x[i-1], E_y[j],
    E_y[j-1]. A conductor cell k shorts its tangential edges on node planes k
    and k + 1, so a body of cells k0..k1 shows as the node run k0..k1+1."""
    ex = np.asarray(edges[0], dtype=bool)
    ey = np.asarray(edges[1], dtype=bool)
    hit = ex[i, j, :] | ey[i, j, :]
    if i > 0:
        hit = hit | ex[i - 1, j, :]
    if j > 0:
        hit = hit | ey[i, j - 1, :]
    return _runs(np.nonzero(hit)[0])


def _cell_runs(mask_z) -> list[list[int]]:
    return _runs(np.nonzero(np.asarray(mask_z, dtype=bool))[0])


def _cross_section(grid, center_xy, a: float, b: float) -> dict:
    """The battery driver's own radial masks (pin, fill, shell) in one plane."""
    return battery.cross_section_masks(grid, center_xy, a, b)


def _assert_cross_section(pec, eps, k: int, masks: dict, where: str) -> dict:
    """The conductor cells at plane ``k`` are exactly the replicated pin and
    shell, and the fill is PTFE, as the battery asserts for the shipped line."""
    pec_k = np.asarray(pec[:, :, k], dtype=bool)
    want = masks["pin"] | masks["shell"]
    mismatch = int(np.count_nonzero(pec_k ^ want))
    fill_eps = np.asarray(eps[:, :, k])[masks["fill"]]
    fill_ok = bool(np.allclose(fill_eps, float(PTFE_EPS_R), rtol=1e-6, atol=0.0))
    if mismatch or not fill_ok:
        raise RuntimeError(
            f"{where}: the cross-section at plane {k} is not the line: {mismatch} conductor "
            f"cells differ from the replicated pin and shell; fill PTFE everywhere: {fill_ok}")
    return {"plane": int(k), "conductor_mismatch_cells": mismatch, "fill_is_ptfe": fill_ok,
            "pin_cells": int(masks["pin"].sum()), "shell_cells": int(masks["shell"].sum()),
            "fill_cells": int(masks["fill"].sum())}


def _cap_closes(pec, k: int, disk) -> dict:
    """A cap plane closes the line when every cell of the line's full disk
    (pin, fill and shell, out to the shell's outer face) is a conductor cell."""
    covered = np.asarray(pec[:, :, k], dtype=bool) & disk
    missing = int(np.count_nonzero(disk & ~covered))
    return {"cell_plane": int(k), "disk_cells": int(disk.sum()), "missing_cells": missing,
            "closes": missing == 0}


def _conductor_span_cells(pec_col_z) -> tuple[int, int]:
    runs = _cell_runs(pec_col_z)
    if not runs:
        raise RuntimeError("no conductor cell in the column")
    return runs[0][0], runs[-1][1]


def _source_planes(spec) -> dict:
    """Where a TEM plane source actually injects: the k of its E and H cells."""
    ek = sorted({int(s.k) for s in spec.electric_sources})
    hk = sorted({int(s.k) for s in spec.magnetic_sources})
    if len(ek) != 1 or len(hk) != 1:
        raise RuntimeError(f"a TEM plane source spans more than one plane: E {ek}, H {hk}")
    return {"plane_axial_index": int(spec.plane_axial_index), "e_plane": ek[0], "h_plane": hk[0],
            "source_cells": int(spec.source_cell_count)}


def _lateral_clearance(edges, grid) -> dict:
    """Whole cells between the outermost shorted edge and each lateral pad face."""
    nx, ny = int(grid.shape[0]), int(grid.shape[1])
    any_x = np.zeros(nx, dtype=bool)
    any_y = np.zeros(ny, dtype=bool)
    for c, m in enumerate(edges):
        m = np.asarray(m, dtype=bool)
        hx = m.any(axis=(1, 2))
        hy = m.any(axis=(0, 2))
        any_x |= hx
        any_y |= hy
        if c == 0:                       # an E_x edge also touches node i + 1
            any_x[1:] |= hx[:-1]
        if c == 1:
            any_y[1:] |= hy[:-1]
    ix, iy = np.nonzero(any_x)[0], np.nonzero(any_y)[0]
    return {
        "x_lo": int(ix.min()) - int(grid.pad_x_lo),
        "x_hi": (nx - 1 - int(grid.pad_x_hi)) - int(ix.max()),
        "y_lo": int(iy.min()) - int(grid.pad_y_lo),
        "y_hi": (ny - 1 - int(grid.pad_y_hi)) - int(iy.max()),
    }


def _witness_probes(grid, center_xy, a: float, b: float, shell_outer: float,
                    z_planes: dict) -> list[tuple[str, ProbeSpec]]:
    """Point probes outside the shell: WITNESS_OFFSET_CELLS beyond its outer
    face, on the +x ray and the +45 degree diagonal, E_x and E_y, at the given
    node planes. The same positions serve every arm of a lane."""
    dx = float(grid.dx)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r_w = shell_outer + WITNESS_OFFSET_CELLS * dx
    pts = {"xray": (cx + r_w, cy),
           "diag": (cx + r_w / math.sqrt(2.0), cy + r_w / math.sqrt(2.0))}
    out = []
    for zname, k in z_planes.items():
        for pname, (x, y) in pts.items():
            i = int(round(x / dx)) + int(grid.pad_x_lo)
            j = int(round(y / dx)) + int(grid.pad_y_lo)
            for comp in ("ex", "ey"):
                out.append((f"outer_{zname}_{pname}_{comp}",
                            ProbeSpec(i=i, j=j, k=int(k), component=comp)))
    return out


def _inline_probe(grid, center_xy, a: float, b: float, k: int) -> ProbeSpec:
    """Mid-annulus on the +x ray: the two-port lane's own settling witness."""
    dz = float(grid.dx)
    x_mid = float(center_xy[0]) + 0.5 * (a + b)
    return ProbeSpec(i=int(round(x_mid / dz)) + int(grid.pad_x_lo),
                     j=int(grid.pad_y_lo) + int(round(float(center_xy[1]) / dz)),
                     k=int(k), component="ex")


def _assert_probe_positions(probes, edges, grid, center_xy, shell_outer, where) -> list[dict]:
    comp_idx = {"ex": 0, "ey": 1, "ez": 2}
    r = _node_radius(grid, center_xy)
    nx, ny = int(grid.shape[0]), int(grid.shape[1])
    rows = []
    for name, p in probes:
        on_pec = bool(np.asarray(edges[comp_idx[p.component]])[p.i, p.j, p.k])
        in_pad = not (int(grid.pad_x_lo) <= p.i <= nx - 1 - int(grid.pad_x_hi)
                      and int(grid.pad_y_lo) <= p.j <= ny - 1 - int(grid.pad_y_hi))
        row = {"name": name, "i": int(p.i), "j": int(p.j), "k": int(p.k),
               "component": p.component, "node_radius_m": float(r[p.i, p.j]),
               "on_pec_edge": on_pec, "in_lateral_pad": in_pad}
        if name.startswith("outer_") and (on_pec or in_pad or r[p.i, p.j] <= shell_outer):
            raise RuntimeError(f"{where}: witness probe {row} is not in the region outside "
                               f"the shell (shell outer radius {shell_outer:.6g} m)")
        if on_pec:
            raise RuntimeError(f"{where}: probe {row} sits on a shorted edge")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# the one-port arms
# ---------------------------------------------------------------------------

def build_one_port(sim, arm: str, n_steps: int) -> dict:
    """The one-port lane's line, stamps and layout for one arm, with its
    realized geometry measured and asserted. Stamp order and arguments are the
    lane's (``rfx/sparams/coax.py``: line, feed resistor, then the
    termination); O2 adds the short's own disk as the cap and retracts the pin
    above it, after the lane's stamps."""
    spec = ARMS[arm]
    dut = spec["dut"]
    grid = sim._build_grid()
    port = sim._coaxial_ports[0]
    a, b = float(port.pin_radius), float(port.outer_radius)
    center_xy = (float(port.position[0]), float(port.position[1]))
    dz = float(grid.dx)
    shipped = battery.axial_layout(grid, "one_port")
    z_dut, z_hi_coax = shipped["z_dut"], shipped["z_hi_coax"]
    z_feed, z_src = shipped["z_feed"], shipped["z_src"]
    R_feed = float(coaxial_tem_characteristic_impedance(a, b))

    materials, _, _ = sim._build_materials(grid)
    materials, shell_inner, pec_cells = stamp_coaxial_line(
        grid, materials, center_xy=center_xy, z_lo_index=z_dut,
        z_hi_index=z_hi_coax, pin_radius=a, outer_radius=b,
    )
    materials = stamp_coaxial_annular_resistor(
        grid, materials, center_xy=center_xy, z_index=z_feed, pin_radius=a,
        outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
        pec_cell_mask=pec_cells,
    )
    pec_shipped_open = np.asarray(pec_cells, dtype=bool).copy()
    if dut == "short":
        materials, short_cells = stamp_coaxial_short_plane(
            grid, materials, center_xy=center_xy, z_index=z_dut, outer_radius=b,
        )
        pec_cells = pec_cells | short_cells

    masks = _cross_section(grid, center_xy, a, b)
    k_mid = shipped["probes"][len(shipped["probes"]) // 2]
    removed = 0
    if spec["shield"]:
        materials, cap_cells = stamp_coaxial_short_plane(
            grid, materials, center_xy=center_xy, z_index=z_dut, outer_radius=b,
        )
        pec_cells = np.asarray(pec_cells | cap_cells, dtype=bool).copy()
        pin_col = pec_shipped_open[:, :, k_mid] & masks["pin"]
        eps = np.array(materials.eps_r)
        for k in range(z_dut + 1, z_dut + 1 + GAP_CELLS):
            removed += int(np.count_nonzero(pec_cells[:, :, k] & pin_col))
            pec_cells[:, :, k] &= ~pin_col
            eps[:, :, k][pin_col] = float(PTFE_EPS_R)
        materials = materials._replace(eps_r=jnp.asarray(eps))
    pec_cells = np.asarray(pec_cells, dtype=bool)

    # --- realized geometry, measured from the masks the run will use -------
    edges = _coax_pec_edge_masks(pec_cells)
    edges_open = _coax_pec_edge_masks(pec_shipped_open)
    i0, j0 = _axis_node(grid, center_xy)
    i_s = int(round((center_xy[0] + b + 0.5 * (masks["shell_outer_radius"] - b)) / dz)) \
        + int(grid.pad_x_lo)
    pin_runs = _tangential_pec_node_runs(edges, i0, j0)
    shell_runs = _tangential_pec_node_runs(edges, i_s, j0)
    shipped_pin_runs = _tangential_pec_node_runs(edges_open, i0, j0)
    if len(shipped_pin_runs) != 1:
        raise RuntimeError(f"{arm}: the shipped line's pin is not one run: {shipped_pin_runs}")
    shipped_lo_node, shipped_hi_node = shipped_pin_runs[0]
    if len(shell_runs) != 1 or shell_runs[0] != [shipped_lo_node, shipped_hi_node]:
        raise RuntimeError(f"{arm}: the shell does not run {shipped_lo_node}..{shipped_hi_node} "
                           f"as shipped: {shell_runs}")

    disk = masks["pin"] | masks["fill"] | masks["shell"]
    caps = [c for c in (_cap_closes(pec_cells, k, disk) for k in range(int(grid.shape[2])))
            if c["closes"]]
    eps_now = np.asarray(materials.eps_r)
    cross = _assert_cross_section(pec_cells, eps_now, k_mid, masks, arm)

    probe_start = battery.PROBE_START_CELLS
    if spec["shield"]:
        # expected: the cap at the DUT plane (cell z_dut, faces z_dut, z_dut+1),
        # then GAP_CELLS empty node planes, then the pin from node z_dut+1+GAP.
        if [c["cell_plane"] for c in caps] != [z_dut]:
            raise RuntimeError(f"{arm}: expected one closing cap at cell {z_dut}, got {caps}")
        cap_inner_node = z_dut + 1
        if len(pin_runs) != 2:
            raise RuntimeError(f"{arm}: the pin column shows {pin_runs}, not cap and pin")
        pin_lo_node = pin_runs[1][0]
        gap = pin_lo_node - cap_inner_node
        if pin_runs[0][1] != cap_inner_node or gap != GAP_CELLS:
            raise RuntimeError(f"{arm}: cap inner face {cap_inner_node}, pin runs {pin_runs}: "
                               f"gap {gap} cells, declared {GAP_CELLS}")
        if pin_runs[1][1] != shipped_hi_node:
            raise RuntimeError(f"{arm}: the pin's feed end moved: {pin_runs}")
        cup_pin_cells = pin_col[:, :, None] & pec_cells[:, :, cap_inner_node:pin_lo_node]
        if cup_pin_cells.any():
            raise RuntimeError(f"{arm}: a pin cell is left in the gap")
        cup_eps = eps_now[:, :, cap_inner_node:pin_lo_node][pin_col]
        if not np.allclose(cup_eps, float(PTFE_EPS_R), rtol=1e-6, atol=0.0):
            raise RuntimeError(f"{arm}: the gap under the pin end is not PTFE")
        # the probe array keeps the shipped distance from the pin end
        probe_start = pin_lo_node + (shipped["probes"][0] - shipped_lo_node) - z_dut
    else:
        if pin_runs != [[shipped_lo_node, shipped_hi_node]]:
            raise RuntimeError(f"{arm}: the pin is not the shipped run: {pin_runs}")
        want_caps = [z_dut] if dut == "short" else []
        if [c["cell_plane"] for c in caps] != want_caps:
            raise RuntimeError(f"{arm}: closing planes {caps}, expected {want_caps}")
        pin_lo_node = shipped_lo_node
        cap_inner_node = None
        gap = None

    probes_z = [z_dut + probe_start + battery.PROBE_SPACING_CELLS * k
                for k in range(battery.PROBE_COUNT)]
    probes_z = [z for z in probes_z if z < z_src - 4]
    if len(probes_z) != battery.PROBE_COUNT:
        raise RuntimeError(f"{arm}: only {len(probes_z)} probe planes fit before the source")
    if min(probes_z) <= pin_lo_node:
        raise RuntimeError(f"{arm}: probe planes {probes_z} reach below the pin end {pin_lo_node}")

    cpml_axes = spec["cpml_axes"]
    grid_axes = str(getattr(grid, "cpml_axes", ""))
    if any(ax not in grid_axes for ax in cpml_axes):
        raise RuntimeError(f"{arm}: absorbing axes {cpml_axes!r} but the grid pads only "
                           f"{grid_axes!r}")

    shell_outer = float(masks["shell_outer_radius"])
    lo_w = shipped_lo_node + WITNESS_OFFSET_CELLS
    hi_w = shipped_hi_node - WITNESS_OFFSET_CELLS
    outer = _witness_probes(grid, center_xy, a, b, shell_outer,
                            {"dut_end": lo_w, "mid": (shipped_lo_node + shipped_hi_node) // 2,
                             "feed_end": hi_w})
    probes = [("inline_mid", _inline_probe(grid, center_xy, a, b, k_mid))] + outer
    probe_rows = _assert_probe_positions(probes, edges, grid, center_xy, shell_outer, arm)

    src_port = CoaxialPort(
        position=(center_xy[0], center_xy[1], (z_src - grid.pad_z_lo) * dz),
        face="top", pin_length=dz, pin_radius=a, outer_radius=b,
        impedance=port.impedance, excitation=port.excitation,
    )
    src_spec = build_coaxial_tem_plane_source_specs(
        grid=grid, port=src_port, n_steps=int(n_steps), field_scale=FIELD_SCALE,
        magnetic_ratio=1.0, shell_inner_radius=shell_inner,
    )
    nz = int(grid.shape[2])
    realized = {
        "arm": arm, "lane": "one_port", "dut": dut,
        "grid_shape": [int(s) for s in grid.shape], "dx_m": dz, "dt_s": float(grid.dt),
        "pads_cells": {f"{ax}_{s}": int(getattr(grid, f"pad_{ax}_{s}"))
                       for ax in "xyz" for s in ("lo", "hi")},
        "absorbing_axes": cpml_axes,
        "absorbing_axes_source": ("the cpml_axes this arm hands the runner; the grid pads "
                                  "every axis, and a padded axis not named here is vacuum "
                                  "backed by the PEC outer face"),
        "grid_cpml_axes": grid_axes,
        "layout": {"z_dut": z_dut, "z_hi_coax": z_hi_coax, "z_feed": z_feed, "z_src": z_src,
                   "probe_start_cells": int(probe_start),
                   "probe_spacing_cells": battery.PROBE_SPACING_CELLS,
                   "probes": [int(z) for z in probes_z],
                   "reference_plane_node": z_dut,
                   "reference_plane_m": (z_dut - grid.pad_z_lo) * dz,
                   "probe_centroid_node": float(np.mean(probes_z))},
        "source_planes": _source_planes(src_spec),
        "shipped_conductor_nodes": [shipped_lo_node, shipped_hi_node],
        "pin_node_runs_on_axis": pin_runs,
        "shell_node_runs": shell_runs,
        "pin_end_node": int(pin_lo_node),
        "closing_cap_planes": caps,
        "cap_inner_face_node": cap_inner_node,
        "gap_cells": gap,
        "pin_cells_removed": removed,
        "first_probe_above_pin_end_cells": int(min(probes_z) - pin_lo_node),
        "conductor_top_node_minus_first_plus_z_slab_node": int(shipped_hi_node
                                                              - (nz - int(grid.pad_z_hi))),
        "conductor_bottom_node_above_minus_z_pad_face_cells": int(shipped_lo_node
                                                                 - int(grid.pad_z_lo)),
        "lateral_clearance_cells": _lateral_clearance(edges, grid),
        "cross_section": cross,
        "shell_inner_radius_m": float(shell_inner),
        "shell_outer_radius_m": shell_outer,
        "probes": probe_rows,
    }
    return {"grid": grid, "materials": materials, "pec_cells": pec_cells, "edges": edges,
            "center_xy": center_xy, "a": a, "b": b, "shell_inner": shell_inner,
            "port": port, "layout": realized["layout"], "src_spec": src_spec,
            "probes": probes, "realized": realized, "termination": dut}


def _solve_one_port(geom: dict, n_steps: int, *, cpml_axes: str, with_probes: bool = True):
    """``compute_coaxial_line_reflection``'s concrete path (``eps_scale=None``),
    line for line, on a geometry ``build_one_port`` built."""
    grid, materials = geom["grid"], geom["materials"]
    lay = geom["layout"]
    a, b = geom["a"], geom["b"]
    center_xy = geom["center_xy"]
    dz = float(grid.dx)
    probes_z = lay["probes"]
    z_dut = lay["z_dut"]
    freqs = jnp.asarray(jnp.asarray(battery.FREQS), dtype=jnp.float32)
    spec = geom["src_spec"]
    planes = []
    for z in probes_z:
        for comp in ("ex", "ey"):
            planes.append(
                init_dft_plane_probe(
                    axis=2, index=int(z), component=comp, freqs=freqs,
                    grid_shape=grid.shape, dft_total_steps=int(n_steps),
                )
            )
    kwargs = {}
    if with_probes:
        kwargs["probes"] = [p for _, p in geom["probes"]]
    result = _run(
        grid, materials, int(n_steps), boundary="cpml", cpml_axes=cpml_axes,
        sources=list(spec.electric_sources), mag_sources=list(spec.magnetic_sources),
        dft_planes=planes, pec_edge_masks=_coax_pec_edge_masks(geom["pec_cells"]),
        return_state=False, **kwargs,
    )
    n_f = int(freqs.shape[0])
    z_planes_m = np.array([(z - grid.pad_z_lo) * dz for z in probes_z], dtype=np.float64)
    ref_m = (z_dut - grid.pad_z_lo) * dz
    annulus_cells = float((b - a) / dz)
    v_by_plane = []
    for pi in range(len(probes_z)):
        ex = result.dft_planes[pi * 2 + 0].accumulator
        ey = result.dft_planes[pi * 2 + 1].accumulator
        v_by_plane.append(
            coaxial_line_plane_voltage(
                grid, ex, ey, center_xy=center_xy, pin_radius=a, outer_radius=b,
            )
        )
    V = np.stack(v_by_plane, axis=0)
    s11 = np.zeros(n_f, dtype=np.complex128)
    gamma = np.zeros(n_f, dtype=np.complex128)
    rec_resid = np.zeros(n_f, dtype=np.float64)
    fit_resid = np.zeros(n_f, dtype=np.float64)
    z0_num = np.full(n_f, np.nan + 1j * np.nan, dtype=np.complex128)
    for fi in range(n_f):
        if not np.all(np.isfinite(V[:, fi])):
            # Not in the lane: a diverged run leaves nothing to fit, and the
            # lane's least squares would raise on it. Recorded as NaN instead.
            s11[fi] = gamma[fi] = complex(np.nan, np.nan)
            rec_resid[fi] = fit_resid[fi] = np.nan
            continue
        out = coaxial_line_reflection_from_plane_voltages(
            z_planes_m, V[:, fi], reference_plane_m=ref_m,
        )
        s11[fi] = out.reflection
        gamma[fi] = out.gamma
        rec_resid[fi] = out.recurrence_residual
        fit_resid[fi] = out.fit_residual
    if not np.all(np.isfinite(V)):
        status = "non_finite"
    elif annulus_cells < 3.5:
        status = "under_resolved"
    elif float(np.max(rec_resid)) > 0.1:
        status = "contaminated"
    else:
        status = "passed"
    res = CoaxialLineReflectionResult(
        s11=s11, freqs=np.asarray(freqs, dtype=float), gamma=gamma,
        recurrence_residual=rec_resid, fit_residual=fit_resid,
        annulus_cells=annulus_cells, z0_numerical_ohm=z0_num,
        termination=geom["termination"], status=status,
    )
    ts = np.asarray(result.time_series, dtype=float) if with_probes else None
    return res, ts


# ---------------------------------------------------------------------------
# the two-port arms
# ---------------------------------------------------------------------------

def _two_port_sources(grid, port, center_xy, a, b, shell_inner, z_src_top, z_src_bot,
                      n_steps):
    dz = float(grid.dx)
    top = CoaxialPort(
        position=(center_xy[0], center_xy[1], (z_src_top - grid.pad_z_lo) * dz),
        face="top", pin_length=dz, pin_radius=a, outer_radius=b,
        impedance=port.impedance, excitation=port.excitation,
    )
    bot = CoaxialPort(
        position=(center_xy[0], center_xy[1], (z_src_bot - grid.pad_z_lo) * dz),
        face="bottom", pin_length=dz, pin_radius=a, outer_radius=b,
        impedance=port.impedance, excitation=port.excitation,
    )
    spec_top = build_coaxial_tem_plane_source_specs(
        grid=grid, port=top, n_steps=int(n_steps), field_scale=FIELD_SCALE,
        magnetic_ratio=1.0, shell_inner_radius=shell_inner,
    )
    spec_bot = build_coaxial_tem_plane_source_specs(
        grid=grid, port=bot, n_steps=int(n_steps), field_scale=FIELD_SCALE,
        magnetic_ratio=1.0, shell_inner_radius=shell_inner,
    )
    return spec_top, spec_bot


def build_two_port(sim, arm: str, n_steps: int) -> dict:
    """The two-port lane's line, feeds and layout for one arm, measured and
    asserted. T0 is the lane's own stamps in the lane's order. T2 draws the
    shell from cap to cap, retracts the pin GAP_CELLS inside each cap, closes
    each end with the short's own disk, and moves feed, source and probes
    inward by the whole cells that keep each one's distance from its pin end
    as in T0."""
    spec = ARMS[arm]
    grid = sim._build_grid()
    port = sim._coaxial_ports[0]
    a, b = float(port.pin_radius), float(port.outer_radius)
    center_xy = (float(port.position[0]), float(port.position[1]))
    dz = float(grid.dx)
    nz = int(grid.shape[2])
    pad_lo, pad_hi = int(grid.pad_z_lo), int(grid.pad_z_hi)
    shipped = battery.axial_layout(grid, "two_port")
    R_feed = float(coaxial_tem_characteristic_impedance(a, b))
    masks = _cross_section(grid, center_xy, a, b)
    i0, j0 = _axis_node(grid, center_xy)
    i_s = int(round((center_xy[0] + b + 0.5 * (masks["shell_outer_radius"] - b)) / dz)) \
        + int(grid.pad_x_lo)

    # the shipped line's realized ends, from its own stamp
    m0, _, _ = sim._build_materials(grid)
    _, _, pec0 = stamp_coaxial_line(
        grid, m0, center_xy=center_xy, z_lo_index=shipped["z_lo_coax"],
        z_hi_index=shipped["z_hi_coax"], pin_radius=a, outer_radius=b,
    )
    pec0 = np.asarray(pec0, dtype=bool)
    shipped_runs = _tangential_pec_node_runs(_coax_pec_edge_masks(pec0), i0, j0)
    if len(shipped_runs) != 1:
        raise RuntimeError(f"{arm}: the shipped pin is not one run: {shipped_runs}")
    s_lo, s_hi = shipped_runs[0]
    spec_top0, spec_bot0 = _two_port_sources(
        grid, port, center_xy, a, b, b, shipped["z_src_top"], shipped["z_src_bot"], 4)
    planes0 = {"top": _source_planes(spec_top0), "bot": _source_planes(spec_bot0)}

    if spec["shield"]:
        lo_face, hi_face = pad_lo, nz - 1 - pad_hi
        kc_bot = lo_face + CAP_CLEARANCE_CELLS            # cap cell: faces kc, kc + 1
        kc_top = hi_face - CAP_CLEARANCE_CELLS - 1
        pin_lo_node = kc_bot + 1 + GAP_CELLS
        pin_hi_node = kc_top - GAP_CELLS
        d_bot = pin_lo_node - s_lo
        d_top = s_hi - pin_hi_node
        lay = {
            "z_shell_lo_cell": kc_bot, "z_shell_hi_cell": kc_top,
            "z_feed_bot": shipped["z_feed_bot"] + d_bot,
            "z_feed_top": shipped["z_feed_top"] - d_top,
            "z_src_bot": shipped["z_src_bot"] + d_bot,
            "z_src_top": shipped["z_src_top"] - d_top,
            "probes_bot": [z + d_bot for z in shipped["probes_bot"]],
            "probes_top": [z - d_top for z in shipped["probes_top"]],
            "shift_bot_cells": d_bot, "shift_top_cells": d_top,
        }
        materials, _, _ = sim._build_materials(grid)
        materials, shell_inner, pec_cells = stamp_coaxial_line(
            grid, materials, center_xy=center_xy, z_lo_index=kc_bot + 1,
            z_hi_index=kc_top - 1, pin_radius=a, outer_radius=b,
        )
        pec_cells = np.asarray(pec_cells, dtype=bool).copy()
        k_mid = (pin_lo_node + pin_hi_node) // 2
        pin_col = pec_cells[:, :, k_mid] & masks["pin"]
        eps = np.array(materials.eps_r)
        removed = 0
        for k in list(range(kc_bot + 1, pin_lo_node)) + list(range(pin_hi_node, kc_top)):
            removed += int(np.count_nonzero(pec_cells[:, :, k] & pin_col))
            pec_cells[:, :, k] &= ~pin_col
            eps[:, :, k][pin_col] = float(PTFE_EPS_R)
        materials = materials._replace(eps_r=jnp.asarray(eps))
        for kc in (kc_bot, kc_top):
            materials, cap_cells = stamp_coaxial_short_plane(
                grid, materials, center_xy=center_xy, z_index=kc, outer_radius=b,
            )
            pec_cells = pec_cells | np.asarray(cap_cells, dtype=bool)
        materials = stamp_coaxial_annular_resistor(
            grid, materials, center_xy=center_xy, z_index=lay["z_feed_top"], pin_radius=a,
            outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
            pec_cell_mask=pec_cells,
        )
        materials = stamp_coaxial_annular_resistor(
            grid, materials, center_xy=center_xy, z_index=lay["z_feed_bot"], pin_radius=a,
            outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
            pec_cell_mask=pec_cells,
        )
    else:
        lay = {k: shipped[k] for k in ("z_feed_bot", "z_feed_top", "z_src_bot", "z_src_top",
                                       "probes_bot", "probes_top")}
        lay.update({"z_lo_coax": shipped["z_lo_coax"], "z_hi_coax": shipped["z_hi_coax"],
                    "shift_bot_cells": 0, "shift_top_cells": 0})
        materials, _, _ = sim._build_materials(grid)
        materials, shell_inner, pec_cells = stamp_coaxial_line(
            grid, materials, center_xy=center_xy, z_lo_index=shipped["z_lo_coax"],
            z_hi_index=shipped["z_hi_coax"], pin_radius=a, outer_radius=b,
        )
        materials = stamp_coaxial_annular_resistor(
            grid, materials, center_xy=center_xy, z_index=shipped["z_feed_top"], pin_radius=a,
            outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
            pec_cell_mask=pec_cells,
        )
        materials = stamp_coaxial_annular_resistor(
            grid, materials, center_xy=center_xy, z_index=shipped["z_feed_bot"], pin_radius=a,
            outer_radius=b, target_impedance=R_feed, shell_inner_radius=shell_inner,
            pec_cell_mask=pec_cells,
        )
        removed = 0
    pec_cells = np.asarray(pec_cells, dtype=bool)

    # --- realized geometry ---------------------------------------------------
    edges = _coax_pec_edge_masks(pec_cells)
    pin_runs = _tangential_pec_node_runs(edges, i0, j0)
    shell_runs = _tangential_pec_node_runs(edges, i_s, j0)
    disk = masks["pin"] | masks["fill"] | masks["shell"]
    caps = [c for c in (_cap_closes(pec_cells, k, disk) for k in range(nz)) if c["closes"]]
    probes_all = list(lay["probes_bot"]) + list(lay["probes_top"])
    k_cross = lay["probes_bot"][-1]
    cross = _assert_cross_section(pec_cells, np.asarray(materials.eps_r), k_cross, masks, arm)
    spec_top, spec_bot = _two_port_sources(
        grid, port, center_xy, a, b, shell_inner, lay["z_src_top"], lay["z_src_bot"],
        n_steps)
    planes = {"top": _source_planes(spec_top), "bot": _source_planes(spec_bot)}

    problems = []
    if spec["shield"]:
        if [c["cell_plane"] for c in caps] != [kc_bot, kc_top]:
            problems.append(f"closing caps {caps}, expected cells {kc_bot} and {kc_top}")
        want_pin = [[kc_bot, kc_bot + 1], [pin_lo_node, pin_hi_node], [kc_top, kc_top + 1]]
        if pin_runs != want_pin:
            problems.append(f"pin node runs {pin_runs}, expected {want_pin}")
        if shell_runs != [[kc_bot, kc_top + 1]]:
            problems.append(f"shell node runs {shell_runs}, expected [[{kc_bot}, {kc_top + 1}]]")
        gap_bot = pin_lo_node - (kc_bot + 1)
        gap_top = kc_top - pin_hi_node
        if gap_bot != GAP_CELLS or gap_top != GAP_CELLS:
            problems.append(f"gaps {gap_bot} / {gap_top}, declared {GAP_CELLS}")
        cl_bot = kc_bot - pad_lo
        cl_top = (nz - 1 - pad_hi) - (kc_top + 1)
        if min(cl_bot, cl_top) < 2:
            problems.append(f"cap clearance {cl_bot} / {cl_top} cells, declared at least 2")
        cup = np.concatenate([np.arange(kc_bot + 1, pin_lo_node),
                              np.arange(pin_hi_node, kc_top)])
        if (pin_col[:, :, None] & pec_cells[:, :, cup]).any():
            problems.append("a pin cell is left in a gap")
        if not np.allclose(np.asarray(materials.eps_r)[:, :, cup][pin_col],
                           float(PTFE_EPS_R), rtol=1e-6, atol=0.0):
            problems.append("a gap is not PTFE-filled")
        # every plane keeps its T0 distance from its own pin end
        for end, d in (("bot", d_bot), ("top", -d_top)):
            for key in ("plane_axial_index", "e_plane", "h_plane"):
                if planes[end][key] != planes0[end][key] + d:
                    problems.append(f"{end} source {key} realizes {planes[end][key]}, T0's "
                                    f"{planes0[end][key]} shifted by {d} is "
                                    f"{planes0[end][key] + d}")
        gaps = {"bottom": gap_bot, "top": gap_top}
        clearance = {"bottom": cl_bot, "top": cl_top}
    else:
        if pin_runs != [[s_lo, s_hi]] or shell_runs != [[s_lo, s_hi]]:
            problems.append(f"pin {pin_runs} / shell {shell_runs}, shipped [[{s_lo}, {s_hi}]]")
        if caps:
            problems.append(f"the thru has closing planes {caps}")
        if planes != planes0:
            problems.append(f"source planes {planes} differ from the shipped {planes0}")
        pin_lo_node, pin_hi_node = s_lo, s_hi
        gaps = None
        clearance = {"bottom": s_lo - pad_lo, "top": (nz - 1 - pad_hi) - s_hi}
    if min(lay["probes_bot"]) <= pin_lo_node or max(lay["probes_top"]) >= pin_hi_node:
        problems.append(f"probe planes leave the line between the pin ends "
                        f"{pin_lo_node}..{pin_hi_node}")
    if lay["probes_bot"][-1] >= lay["probes_top"][0]:
        problems.append("the two probe arrays overlap")
    cpml_axes = spec["cpml_axes"]
    grid_axes = str(getattr(grid, "cpml_axes", ""))
    if any(ax not in grid_axes for ax in cpml_axes):
        problems.append(f"absorbing axes {cpml_axes!r}, grid pads {grid_axes!r}")
    if problems:
        raise RuntimeError(f"{arm}: the realized geometry is not the declared one: "
                           + "; ".join(problems))

    shell_outer = float(masks["shell_outer_radius"])
    n_bot, n_top = len(lay["probes_bot"]), len(lay["probes_top"])
    lane_witness = [
        ("lane_settling_bot", _inline_probe(grid, center_xy, a, b,
                                            lay["probes_bot"][n_bot // 2])),
        ("lane_settling_top", _inline_probe(grid, center_xy, a, b,
                                            lay["probes_top"][n_top // 2])),
    ]
    outer = _witness_probes(grid, center_xy, a, b, shell_outer,
                            {"bot_end": s_lo + WITNESS_OFFSET_CELLS, "mid": (s_lo + s_hi) // 2,
                             "top_end": s_hi - WITNESS_OFFSET_CELLS})
    probes = lane_witness + outer
    probe_rows = _assert_probe_positions(probes, edges, grid, center_xy, shell_outer, arm)

    layout = dict(lay)
    layout.update({
        "reference_plane_top_m": (lay["z_feed_top"] - pad_lo) * dz,
        "reference_plane_bot_m": (lay["z_feed_bot"] - pad_lo) * dz,
        "probe_gap_cells": lay["probes_top"][0] - lay["probes_bot"][-1],
        "feed_to_feed_cells": lay["z_feed_top"] - lay["z_feed_bot"],
    })
    realized = {
        "arm": arm, "lane": "two_port", "dut": spec["dut"],
        "grid_shape": [int(s) for s in grid.shape], "dx_m": dz, "dt_s": float(grid.dt),
        "pads_cells": {f"{ax}_{s}": int(getattr(grid, f"pad_{ax}_{s}"))
                       for ax in "xyz" for s in ("lo", "hi")},
        "absorbing_axes": cpml_axes, "grid_cpml_axes": grid_axes,
        "layout": layout,
        "shipped_conductor_nodes": [s_lo, s_hi],
        "source_planes": planes, "shipped_source_planes": planes0,
        "pin_node_runs_on_axis": pin_runs, "shell_node_runs": shell_runs,
        "pin_end_nodes": [int(pin_lo_node), int(pin_hi_node)],
        "closing_cap_planes": caps,
        "gap_cells": gaps,
        "cap_clearance_cells_declared_min": 2, "cap_clearance_cells_used": (
            CAP_CLEARANCE_CELLS if spec["shield"] else None),
        "conductor_end_to_z_pad_face_cells": clearance,
        "pin_cells_removed": removed,
        "feed_minus_pin_end_cells": {"bottom": lay["z_feed_bot"] - pin_lo_node,
                                     "top": pin_hi_node - lay["z_feed_top"]},
        "first_probe_from_pin_end_cells": {"bottom": min(probes_all) - pin_lo_node,
                                           "top": pin_hi_node - max(probes_all)},
        "lateral_clearance_cells": _lateral_clearance(edges, grid),
        "cross_section": cross,
        "shell_inner_radius_m": float(shell_inner),
        "shell_outer_radius_m": shell_outer,
        "probes": probe_rows,
    }
    return {"grid": grid, "materials": materials, "pec_cells": pec_cells, "edges": edges,
            "center_xy": center_xy, "a": a, "b": b, "shell_inner": shell_inner,
            "port": port, "layout": layout, "spec_top": spec_top, "spec_bot": spec_bot,
            "probes": probes, "realized": realized}


def _solve_two_port(geom: dict, n_steps: int, *, cpml_axes: str, with_probes: bool = True):
    """``compute_coaxial_two_port``'s concrete path (``eps_scale=None``, no flux
    monitors), line for line, on a geometry ``build_two_port`` built. The
    lane's two settling probes are the first two of the probe list, so its
    ``settling_db`` is computed on exactly the lane's columns."""
    grid, materials = geom["grid"], geom["materials"]
    lay = geom["layout"]
    a, b = geom["a"], geom["b"]
    center_xy = geom["center_xy"]
    dz = float(grid.dx)
    probes_bot, probes_top = lay["probes_bot"], lay["probes_top"]
    freqs = jnp.asarray(jnp.asarray(battery.FREQS), dtype=jnp.float32)
    n_f = int(freqs.shape[0])
    n_bot, n_top = len(probes_bot), len(probes_top)
    all_probes_z = list(probes_bot) + list(probes_top)
    probe_list = [p for _, p in geom["probes"]]
    witness_probes = probe_list[:2]              # the lane's own two, first
    run_probes = probe_list if with_probes else witness_probes
    z_planes_bot_m = np.array([(z - grid.pad_z_lo) * dz for z in probes_bot], dtype=np.float64)
    z_planes_top_m = np.array([(z - grid.pad_z_lo) * dz for z in probes_top], dtype=np.float64)
    ref_top_m = (lay["z_feed_top"] - grid.pad_z_lo) * dz
    ref_bot_m = (lay["z_feed_bot"] - grid.pad_z_lo) * dz
    annulus_cells = float((b - a) / dz)
    _coax_edges = _coax_pec_edge_masks(geom["pec_cells"])
    v_bot_by_drive = np.zeros((2, n_bot, n_f), dtype=np.complex128)
    v_top_by_drive = np.zeros((2, n_top, n_f), dtype=np.complex128)
    settling_db = np.full(2, np.nan, dtype=np.float64)
    series = []
    for drive_idx, spec in enumerate((geom["spec_top"], geom["spec_bot"])):
        planes = []
        for z in all_probes_z:
            for comp in ("ex", "ey"):
                planes.append(
                    init_dft_plane_probe(
                        axis=2, index=int(z), component=comp, freqs=freqs,
                        grid_shape=grid.shape, dft_total_steps=int(n_steps),
                    )
                )
        result = _run(
            grid, materials, int(n_steps), boundary="cpml", cpml_axes=cpml_axes,
            sources=list(spec.electric_sources), mag_sources=list(spec.magnetic_sources),
            probes=run_probes, dft_planes=planes,
            pec_edge_masks=_coax_edges, return_state=False,
        )
        top_off = n_bot * 2
        v_bot_by_drive[drive_idx] = np.stack(
            [
                coaxial_line_plane_voltage(
                    grid, result.dft_planes[pi * 2 + 0].accumulator,
                    result.dft_planes[pi * 2 + 1].accumulator,
                    center_xy=center_xy, pin_radius=a, outer_radius=b,
                )
                for pi in range(n_bot)
            ],
            axis=0,
        )
        v_top_by_drive[drive_idx] = np.stack(
            [
                coaxial_line_plane_voltage(
                    grid, result.dft_planes[top_off + pi * 2 + 0].accumulator,
                    result.dft_planes[top_off + pi * 2 + 1].accumulator,
                    center_xy=center_xy, pin_radius=a, outer_radius=b,
                )
                for pi in range(n_top)
            ],
            axis=0,
        )
        ts_all = np.asarray(result.time_series, dtype=float)
        ts = ts_all[:, :len(witness_probes)]
        if ts.ndim == 2 and ts.shape[0] >= 10 and ts.shape[1] == len(witness_probes):
            power = ts ** 2
            tail = max(1, power.shape[0] // 10)
            end = power[-tail:, :].mean(axis=0)
            peak = power.max(axis=0)
            tiny = np.finfo(float).tiny
            ratio_db = 10.0 * np.log10((end + tiny) / (peak + tiny))
            settling_db[drive_idx] = float(np.max(ratio_db))
        series.append(ts_all)
    finite = bool(np.all(np.isfinite(v_bot_by_drive)) and np.all(np.isfinite(v_top_by_drive)))
    if finite:
        s_params, cond_a, rec_resid, fit_resid, gamma = _assemble_coaxial_two_port_from_voltages(
            z_planes_bot_m=z_planes_bot_m, z_planes_top_m=z_planes_top_m,
            ref_bot_m=ref_bot_m, ref_top_m=ref_top_m,
            v_bot_by_drive=v_bot_by_drive, v_top_by_drive=v_top_by_drive,
            cond_warn=COND_WARN, _prefer_jnp=False,
        )
    else:
        # Not in the lane: a diverged run is recorded as NaN rather than fitted.
        nan_c = np.full((2, 2, n_f), np.nan + 1j * np.nan, dtype=np.complex128)
        s_params, gamma = nan_c, nan_c.copy()
        cond_a = np.full(n_f, np.nan)
        rec_resid = fit_resid = np.full((2, 2, n_f), np.nan)
    if not finite:
        status = "non_finite"
    elif annulus_cells < 3.5:
        status = "under_resolved"
    elif float(np.max(rec_resid)) > 0.1:
        status = "contaminated"
    else:
        status = "passed"
    res = CoaxialTwoPortResult(
        s_params=s_params, freqs=np.asarray(freqs, dtype=float),
        port_names=("port1", "port2"),
        reference_planes=np.asarray([ref_top_m, ref_bot_m], dtype=float),
        cond_a=cond_a, recurrence_residual=rec_resid, fit_residual=fit_resid,
        gamma=gamma, annulus_cells=annulus_cells, settling_db=settling_db,
        status=status, flux_monitors=None,
    )
    _warn_if_ringdown_truncated(settling_db, ("port1", "port2"), n_steps=int(n_steps))
    res = _finalize_sparam_result(res, extractor="compute_coaxial_two_port", strict=False)
    return res, (series if with_probes else None)


# ---------------------------------------------------------------------------
# witnesses and summaries
# ---------------------------------------------------------------------------

def witness_record(ts, names: list[str], n_steps: int) -> dict:
    """Per point probe: the peak, when it came, the lane's settling measure
    (mean E^2 over the last tenth of the record against its peak, in dB) and
    a block-maximum envelope of |E| over the record (100 blocks, 4 significant
    digits: a shape, not a number to compare). A probe whose field is exactly
    zero through the whole record (a region no field reaches) has no settling
    measure; it is marked ``identically_zero`` instead of reading 0 dB."""
    if ts is None:
        return None
    ts = np.asarray(ts, dtype=float)
    block = max(1, -(-int(n_steps) // 100))
    tiny = np.finfo(float).tiny
    out = {"block_steps": block, "n_steps": int(n_steps),
           "all_finite": bool(np.all(np.isfinite(ts))), "probes": {}}
    for c, name in enumerate(names):
        e = ts[:, c]
        p = e ** 2
        tail = max(1, p.size // 10)
        k = int(np.argmax(p))
        zero = bool(p[k] == 0.0)
        env = [float(f"{np.max(np.abs(e[s:s + block])):.4g}") for s in range(0, e.size, block)]
        out["probes"][name] = {
            "identically_zero": zero,
            "peak_abs": float(math.sqrt(p[k])), "peak_step": k,
            "end_tenth_rms": float(math.sqrt(p[-tail:].mean())),
            "settling_db": None if zero else float(
                10.0 * np.log10((p[-tail:].mean() + tiny) / (p[k] + tiny))),
            "envelope_max_abs": env,
        }
    return out


def identity_record(a, b) -> dict:
    a = np.asarray(a)
    b = np.asarray(b)
    d = np.abs(a - b)
    return {"bit_identical": bool(np.array_equal(a, b)),
            "max_abs_complex_diff": float(np.nanmax(d)) if d.size else 0.0,
            "max_abs_magnitude_diff": float(np.nanmax(np.abs(np.abs(a) - np.abs(b)))),
            "all_finite_both": bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))}


def one_port_summary(result: dict, layout: dict, dx: float) -> dict:
    g = battery._S(result["S11"])
    f = np.asarray(result["freqs_hz"], dtype=float)
    mag = np.abs(g)
    if not np.all(np.isfinite(mag)):
        return {"all_finite": False, "n_finite_bins": int(np.sum(np.isfinite(mag))),
                "abs_s11": mag.astype(float).tolist(), "max_abs_s11": None,
                "argmax_hz": None, "min_abs_s11": None, "argmin_hz": None,
                "n_bins_abs_s11_above_1": None, "max_alpha_np_per_m": None,
                "derived_abs_b_over_a_at_probe_centroid": {"max": None},
                "status": result["status"]}
    k_max, k_min = int(np.nanargmax(mag)), int(np.nanargmin(mag))
    gam = battery._S(result["gamma"])
    alpha = np.real(gam)
    d = (float(np.mean(layout["probes"])) - float(layout["z_dut"])) * dx
    gain = np.exp(2.0 * alpha * d)
    b_over_a = mag / gain
    mb = battery.measured_beta(result["gamma"], f, float(PTFE_EPS_R))
    return {
        "all_finite": bool(np.all(np.isfinite(g))),
        "abs_s11": mag.astype(float).tolist(),
        "max_abs_s11": float(mag[k_max]), "argmax_bin": k_max, "argmax_hz": float(f[k_max]),
        "max_abs_s11_sq": float(mag[k_max] ** 2),
        "min_abs_s11": float(mag[k_min]), "argmin_bin": k_min, "argmin_hz": float(f[k_min]),
        "n_bins_abs_s11_above_1": int(np.sum(mag > 1.0)),
        "phase_deg_at_reference_plane": np.degrees(np.angle(g)).astype(float).tolist(),
        "max_recurrence_residual": float(np.max(result["recurrence_residual"])),
        "max_fit_residual": float(np.max(result["fit_residual"])),
        "derived_abs_b_over_a_at_probe_centroid": {
            "derived": True,
            "formula": ("|S11| / exp(2 alpha d), d = (mean(probe nodes) - z_dut) dx: the "
                        "ratio of the two fitted waves at the probe centroid, with no "
                        "extrapolation to the reference plane"),
            "centroid_to_reference_plane_m": d,
            "values": b_over_a.astype(float).tolist(),
            "max": float(np.max(b_over_a)), "argmax_hz": float(f[int(np.argmax(b_over_a))]),
            "min": float(np.min(b_over_a)), "argmin_hz": float(f[int(np.argmin(b_over_a))]),
        },
        "alpha_np_per_m": alpha.astype(float).tolist(),
        "max_alpha_np_per_m": float(np.max(alpha)),
        "eps_eff_fitted_mean": mb["mean_eps_eff_fitted"],
        "beta_ratio_min": mb["min_beta_ratio"], "beta_ratio_max": mb["max_beta_ratio"],
        "status": result["status"],
    }


def two_port_summary(result: dict) -> dict:
    S = battery._S(result["S"])
    f = np.asarray(result["freqs_hz"], dtype=float)
    if not np.all(np.isfinite(S)):
        return {"all_finite": False, "column_power": None, "column_power_min": None,
                "column_power_argmin_hz": None, "column_power_argmin_column": None,
                "column_power_max": None, "column_power_per_column": None,
                "max_abs_s11": None, "max_abs_s22": None, "min_abs_s21": None,
                "settling_db": result["settling"]["settling_db"], "status": result["status"]}
    pm = battery.power_metrics(S)
    col = np.asarray(pm["column_power"])          # (2, n_freqs): column j = port j driven
    j_min, k_min = np.unravel_index(int(np.nanargmin(col)), col.shape)
    per_col = {}
    for j, name in enumerate(("port1", "port2")):
        kk = int(np.nanargmin(col[j]))
        kx = int(np.nanargmax(col[j]))
        per_col[name] = {"min": float(col[j, kk]), "argmin_hz": float(f[kk]),
                         "max": float(col[j, kx]), "argmax_hz": float(f[kx])}
    s11, s22 = np.abs(S[0, 0]), np.abs(S[1, 1])
    s21 = np.abs(S[1, 0])
    return {
        "all_finite": bool(np.all(np.isfinite(S))),
        "column_power": col.astype(float).tolist(),
        "column_power_min": float(col[j_min, k_min]),
        "column_power_argmin_column": ("port1", "port2")[int(j_min)],
        "column_power_argmin_hz": float(f[int(k_min)]),
        "column_power_max": float(np.nanmax(col)),
        "column_power_per_column": per_col,
        "max_abs_s11": float(s11.max()), "argmax_abs_s11_hz": float(f[int(np.argmax(s11))]),
        "max_abs_s22": float(s22.max()), "argmax_abs_s22_hz": float(f[int(np.argmax(s22))]),
        "min_abs_s21": float(s21.min()), "argmin_abs_s21_hz": float(f[int(np.argmin(s21))]),
        "reciprocity_metric": pm["reciprocity_metric"],
        "settling_db": result["settling"]["settling_db"],
        "max_cond_a": float(np.max(result["cond_a"])),
        "max_recurrence_residual": float(np.max(result["recurrence_residual"])),
        "status": result["status"],
    }


# ---------------------------------------------------------------------------
# one arm, one record length
# ---------------------------------------------------------------------------

def build_geometry(sim, arm: str, n_steps: int) -> dict:
    if ARMS[arm]["lane"] == "one_port":
        return build_one_port(sim, arm, n_steps)
    return build_two_port(sim, arm, n_steps)


def stage_record(args, arm: str, units: float, out: Path) -> None:
    spec = ARMS[arm]
    dut, lane = spec["dut"], spec["lane"]
    sim = battery.build_sim(RUNG, dut)
    shipped_geo = battery.assert_realized(sim, RUNG, dut)
    grid = sim._build_grid()
    n_steps = battery.record_steps(grid, units)
    geom = build_geometry(sim, arm, n_steps)
    pf = battery.preflight_record(sim)
    two = lane == "two_port"
    n_planes = (2 * 2 * battery.PROBE_COUNT) if two else (2 * battery.PROBE_COUNT)
    est = battery.cost_estimate(grid, n_steps, battery.N_FREQS, n_planes, 2 if two else 1)

    rec = battery._base(args, "closed_can_arm", dut, RUNG)
    rec.update({
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "driver": DRIVER,
        "reuses": battery.DRIVER, "predeclaration": PREDECLARATION,
        "predeclaration_commit": PREDECLARATION_COMMIT,
        "arm": arm, "arm_definition": spec, "lane": lane, "smoke": bool(args.smoke),
        "record_units": float(units), "n_steps": int(n_steps),
        "device_kind": [d.device_kind for d in jax.devices()],
        "declared": battery.declared(RUNG, dut), "shipped_realized": shipped_geo,
        "realized": geom["realized"], "preflight": pf, "cost": est,
        "result": None, "result_source": None,
    })
    # persist the realized geometry before the first step
    battery._write(out, rec)

    names = [n for n, _ in geom["probes"]]
    battery._log(f"closed-can arm={arm} units={units:g} n_steps={n_steps} "
                 f"cpml_axes={spec['cpml_axes']} shield={spec['shield']}")
    solve = _solve_two_port if two else _solve_one_port
    record_of = battery._two_port_record if two else battery._one_port_record
    log_of = battery._log_two_port if two else battery._log_one_port
    s_key = "s_params" if two else "s11"

    with battery._Captured() as cap_copy:
        res_copy, ts = solve(geom, n_steps, cpml_axes=spec["cpml_axes"])
    log_of(f"copy {arm} u{units:g}", res_copy)
    copy_record = record_of(res_copy)
    if two:
        wit = [witness_record(t, names, n_steps) for t in ts]
        rec["witness"] = {"drive_port1": wit[0], "drive_port2": wit[1]}
    else:
        rec["witness"] = witness_record(ts, names, n_steps)
    rec["copy_warnings"] = cap_copy.warnings
    rec["copy_wall_s"] = cap_copy.wall

    if spec["shipped"]:
        rec["copy_result"] = copy_record
        battery._write(out, rec)
        with battery._Captured() as cap_lane:
            if two:
                res_lane = battery.solve_two_port(sim, n_steps=n_steps)
            else:
                res_lane = battery.solve_one_port(sim, dut, n_steps=n_steps)
        log_of(f"lane {arm} u{units:g}", res_lane)
        rec["result"] = record_of(res_lane)
        rec["result_source"] = ("the shipped lane itself (" + (
            "compute_coaxial_two_port" if two else "compute_coaxial_line_reflection") + ")")
        rec["cross_check"] = battery.cross_check_result_against_layout(
            shipped_geo, res_lane, lane)
        rec["warnings"] = cap_lane.warnings
        rec["wall_s"] = cap_lane.wall
        ident = {"copy_with_witness_probes": identity_record(
            getattr(res_lane, s_key), getattr(res_copy, s_key))}
        if not ident["copy_with_witness_probes"]["bit_identical"]:
            with battery._Captured():
                res_np, _ = solve(geom, n_steps, cpml_axes=spec["cpml_axes"],
                                  with_probes=False)
            ident["copy_without_witness_probes"] = identity_record(
                getattr(res_lane, s_key), getattr(res_np, s_key))
        rec["identity_lane_vs_copy"] = ident
    else:
        rec["result"] = copy_record
        rec["result_source"] = "the copy of the lane's steps in this driver"
        rec["warnings"] = cap_copy.warnings
        rec["wall_s"] = cap_copy.wall
    if two:
        rec["summary"] = two_port_summary(rec["result"])
    else:
        rec["summary"] = one_port_summary(rec["result"], geom["layout"], float(grid.dx))
    rec["peak_memory"] = battery.peak_memory()
    battery._write(out, rec)


def stage_arm(args, out_dir: Path) -> None:
    arms = tuple(ARMS) if args.arm == "ALL" else (args.arm,)
    for arm in arms:
        for units in units_for(arm, args.smoke):
            stage_record(args, arm, units, out_dir / record_name(arm, units, args.smoke))


# ---------------------------------------------------------------------------
# W1: TE cutoffs of the region between the shell and the can
# ---------------------------------------------------------------------------

def outer_region_te_cutoffs(grid, pec_cells, k: int, n_modes: int = W1_N_MODES) -> dict:
    """The lowest TE (H_z) cutoffs of the cross-section outside the shell, on
    the Yee lattice the solver steps.

    At cutoff (k_z = 0) a TE field is H_z on the cell centres with E_x, E_y on
    the edges between them; a shorted edge holds its E at zero, so no H_z
    difference is carried across it. The operator is the graph Laplacian over
    the H_z cells, one link per live edge, with weight 1/dx^2: its nonzero
    eigenvalues are (k_c dx)^2. The domain is the H_z cells between the PEC
    outer faces (nodes 0 and n-1 on x and y); the shell's shorted edges cut
    the region inside the shell off, and the component that holds the corner
    cell is the region between the shell and the can. ``f_c = c k_c / 2 pi``;
    ``f_c_time_discrete`` also carries the leapfrog's own dispersion,
    ``sin(pi f dt) = c dt k_c / 2``.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh

    edges = _coax_pec_edge_masks(pec_cells)
    ex = np.asarray(edges[0], dtype=bool)[:, :, k]
    ey = np.asarray(edges[1], dtype=bool)[:, :, k]
    nx, ny = ex.shape
    mx, my = nx - 1, ny - 1
    idx = np.arange(mx * my).reshape(mx, my)
    live_x = ~ey[1:mx, :my]          # H_z(i,j) -- H_z(i+1,j) across E_y[i+1, j]
    live_y = ~ex[:mx, 1:my]          # H_z(i,j) -- H_z(i,j+1) across E_x[i, j+1]
    ai = np.concatenate([idx[:-1, :][live_x], idx[:, :-1][live_y]])
    aj = np.concatenate([idx[1:, :][live_x], idx[:, 1:][live_y]])
    n = mx * my
    A = coo_matrix((np.ones(ai.size), (ai, aj)), shape=(n, n)).tocsr()
    A = (A + A.T).tocsr()
    ncomp, labels = connected_components(A, directed=False)
    outer_label = labels[idx[0, 0]]
    sel = np.nonzero(labels == outer_label)[0]
    As = A[sel][:, sel]
    L = (diags(np.asarray(As.sum(axis=1)).ravel()) - As).tocsc()
    vals = eigsh(L, k=n_modes + 1, sigma=-1.0e-6, which="LM", return_eigenvectors=False)
    vals = np.sort(np.asarray(vals, dtype=float))
    dx, dt = float(grid.dx), float(grid.dt)
    kc = np.sqrt(np.clip(vals[1:], 0.0, None)) / dx
    f = battery.C0 * kc / (2.0 * math.pi)
    f_td = np.arcsin(np.clip(battery.C0 * dt * kc / 2.0, -1.0, 1.0)) / (math.pi * dt)
    comp_sizes = np.bincount(labels)
    return {
        "cell_plane": int(k),
        "hz_cells": int(n), "components": int(ncomp),
        "outer_region_hz_cells": int(sel.size),
        "largest_other_component_cells": int(np.sort(comp_sizes)[-2]) if ncomp > 1 else 0,
        "lowest_eigenvalue_kdx_sq": float(vals[0]),
        "eigenvalues_kdx_sq": vals[1:].astype(float).tolist(),
        "cutoff_hz": f.astype(float).tolist(),
        "cutoff_hz_time_discrete": f_td.astype(float).tolist(),
    }


def can_record(grid, center_xy, masks) -> dict:
    """The can as realized: PEC outer faces at nodes 0 and n-1 on x and y."""
    dx = float(grid.dx)
    nx, ny = int(grid.shape[0]), int(grid.shape[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    lo_x, hi_x = -int(grid.pad_x_lo) * dx, (nx - 1 - int(grid.pad_x_lo)) * dx
    lo_y, hi_y = -int(grid.pad_y_lo) * dx, (ny - 1 - int(grid.pad_y_lo)) * dx
    return {"grid_xy": [nx, ny], "dx_m": dx,
            "wall_nodes": {"x": [0, nx - 1], "y": [0, ny - 1]},
            "can_width_m": {"x": hi_x - lo_x, "y": hi_y - lo_y},
            "axis_to_wall_m": {"x_lo": cx - lo_x, "x_hi": hi_x - cx,
                               "y_lo": cy - lo_y, "y_hi": hi_y - cy},
            "shell_outer_radius_m": float(masks["shell_outer_radius"]),
            "lateral_pad_cells": [int(grid.pad_x_lo), int(grid.pad_x_hi),
                                  int(grid.pad_y_lo), int(grid.pad_y_hi)],
            "lateral_pads": "vacuum, part of the can (the lane absorbs on z only)"}


def stage_w1(args, out: Path) -> None:
    rec = battery._base(args, "closed_can_W1", "open", RUNG)
    rec.update({"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "driver": DRIVER,
                "reuses": battery.DRIVER, "predeclaration": PREDECLARATION,
                "predeclaration_commit": PREDECLARATION_COMMIT, "smoke": bool(args.smoke),
                "what": outer_region_te_cutoffs.__doc__, "cans": {}})
    for name, lateral in (("battery_can_8mm_board", battery.DOMAIN_ONEPORT[0]),
                          ("wide_can_16mm_board", WIDE_LATERAL_M)):
        domain = (lateral, lateral, battery.DOMAIN_ONEPORT[2])
        sim = battery.build_sim(RUNG, "open", domain=domain)
        grid = sim._build_grid()
        port = sim._coaxial_ports[0]
        a, b = float(port.pin_radius), float(port.outer_radius)
        center_xy = (float(port.position[0]), float(port.position[1]))
        lay = battery.axial_layout(grid, "one_port")
        materials, _, _ = sim._build_materials(grid)
        materials, _, pec = stamp_coaxial_line(
            grid, materials, center_xy=center_xy, z_lo_index=lay["z_dut"],
            z_hi_index=lay["z_hi_coax"], pin_radius=a, outer_radius=b)
        pec = np.asarray(pec, dtype=bool)
        masks = _cross_section(grid, center_xy, a, b)
        k = lay["probes"][len(lay["probes"]) // 2]
        cross = _assert_cross_section(pec, np.asarray(materials.eps_r), k, masks, name)
        rec["cans"][name] = {"domain_m": list(domain), "can": can_record(grid, center_xy, masks),
                             "cross_section": cross,
                             "te_cutoffs": outer_region_te_cutoffs(grid, pec, k)}
        c = rec["cans"][name]
        battery._log(f"W1 {name}: can {c['can']['can_width_m']['x']*1e3:.3f} mm, lowest TE "
                     f"cutoffs {[round(v / 1e9, 3) for v in c['te_cutoffs']['cutoff_hz']]} GHz")
    battery._write(out, rec)


# ---------------------------------------------------------------------------
# layout only — build and assert every arm, solve nothing
# ---------------------------------------------------------------------------

def stage_layout_only(arm_sel: str | None) -> int:
    arms = tuple(ARMS) if arm_sel in (None, "ALL") else (arm_sel,)
    for arm in arms:
        spec = ARMS[arm]
        sim = battery.build_sim(RUNG, spec["dut"])
        geom = build_geometry(sim, arm, 4)
        r = geom["realized"]
        print(f"--- {arm} ({spec['lane']}, {spec['dut']}): grid {r['grid_shape']}, absorbing "
              f"'{r['absorbing_axes']}', pin runs {r['pin_node_runs_on_axis']}, shell runs "
              f"{r['shell_node_runs']}, caps {[c['cell_plane'] for c in r['closing_cap_planes']]}"
              f", gap {r['gap_cells']}, lateral clearance {r['lateral_clearance_cells']}")
        print(f"    layout {r['layout']}")
        print(f"    source planes {r['source_planes']}")
        if spec["lane"] == "one_port":
            print(f"    pin end node {r['pin_end_node']}, first probe +"
                  f"{r['first_probe_above_pin_end_cells']} cells")
        else:
            print(f"    pin ends {r['pin_end_nodes']}, feed - pin end "
                  f"{r['feed_minus_pin_end_cells']}, first probe from pin end "
                  f"{r['first_probe_from_pin_end_cells']}, end clearance "
                  f"{r['conductor_end_to_z_pad_face_cells']}")
        print(f"    probes {[(p['name'], p['i'], p['j'], p['k'], p['component']) for p in r['probes']][:3]} "
              f"... ({len(r['probes'])} total)")
    return 0


# ---------------------------------------------------------------------------
# assemble — arithmetic only
# ---------------------------------------------------------------------------

def _gamma(rec: dict) -> np.ndarray:
    return battery._S(rec["result"]["S11"])


def _per_bin_shift(r12: dict, r24: dict) -> float:
    """The battery's record-length metric: max over bins of ||S_24| - |S_12||."""
    return float(np.max(np.abs(np.abs(_gamma(r24)) - np.abs(_gamma(r12)))))


def _all(values):
    values = list(values)
    if any(v is False for v in values):
        return False
    return None if any(v is None for v in values) else True


def stage_assemble(args, out: Path, artifact_out: Path) -> None:
    smoke = bool(args.smoke)
    records: dict[str, dict] = {}
    missing = []
    for arm in ARMS:
        for u in units_for(arm, smoke):
            p = out / record_name(arm, u, smoke)
            if not p.exists():
                missing.append(p.name)
                continue
            rec = json.loads(p.read_text())
            if rec.get("result") is None:
                missing.append(p.name + " (geometry only: the solve did not finish)")
                continue
            records[f"{arm}_u{u:g}"] = rec
    w1_path = out / w1_name(smoke)
    w1 = json.loads(w1_path.read_text()) if w1_path.exists() else None
    if w1 is None:
        missing.append(w1_path.name)
    commits = sorted({(r.get("provenance") or {}).get("commit") for r in records.values()}
                     | ({(w1.get("provenance") or {}).get("commit")} if w1 else set()))
    if len(commits) != 1:
        raise RuntimeError(f"the records name {len(commits)} commits: {commits}; one campaign "
                           "is one commit")
    index = json.loads(Path(args.run_index).read_text()) if args.run_index else None

    art = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "what": ("The one-port lane's open and short and the two-port lane's thru at 9 "
                 "annulus cells on the battery's 8 x 8 mm board, as shipped and with one "
                 "thing changed per arm: lateral absorbers (O1, S1), a shielded open (O2), "
                 "shielded line ends behind the feeds (T2). Complex S per bin, the realized "
                 "geometry and provenance per arm, the declared predictions evaluated as "
                 "arithmetic, and W1's TE cutoffs. No verdict."),
        "driver": DRIVER, "reuses": battery.DRIVER, "artifact": ARTIFACT,
        "predeclaration": PREDECLARATION, "predeclaration_commit": PREDECLARATION_COMMIT,
        "assembled_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "assembler_commit": battery.git_sha(),
        "measured_at_commit": commits[0],
        "smoke": smoke,
        "resource_choice": args.resource_choice,
        "missing_records": missing,
        "arms": ARMS,
        "rung_annulus_cells": RUNG,
        "gap_cells": GAP_CELLS,
        "cap_clearance_cells": {"declared_min": 2, "used": CAP_CLEARANCE_CELLS,
                                "why": ("with 2 the thru's ends move inward by 11 and 13 cells; "
                                        "the lane places a TFSF plane at round(position / dx) "
                                        "of a half-cell position, so near the line ends only "
                                        "every other plane is reachable and T2's sources would "
                                        "sit one cell off their T0 distance from the feeds. With "
                                        "3 the shifts are 12 and 14 and every plane moves by "
                                        "exactly the shift (asserted per run)")},
        "predeclared": {
            "max_abs_gamma": MAX_ABS_GAMMA,
            "doubling_shift_max": DOUBLING_SHIFT_MAX,
            "doubling_shift_definition": ("max over bins of | |Gamma| at 24 units - |Gamma| at "
                                          "12 units |, the battery's record-length metric "
                                          "(also given: the change of max |Gamma| itself)"),
            "min_abs_gamma_shielded_open": MIN_ABS_GAMMA_SHIELDED,
            "p3_max_abs_delta_gamma": P3_MAX_ABS_DELTA,
            "column_power_readings": {
                "t2_floor": COLUMN_POWER_T2_FLOOR, "same_within": COLUMN_POWER_SAME_WITHIN,
                "text": ("If T2's minimum is at least 0.995 where T0's is about 0.98, the "
                         "missing power is radiation from the unshielded line ends behind the "
                         "feeds. If T2 stays within 0.005 of T0, it is something else.")},
            "P1": ("O1: max |Gamma| <= 1.02 at 12 and at 24 units, and the 12 -> 24 shift of "
                   "max |Gamma| < 0.0259"),
            "P2": ("O2: max |Gamma| <= 1.02 at 12 and at 24 units, the 12 -> 24 shift < "
                   "0.0259, and min |Gamma| >= 0.98 over the band"),
            "P3": "S1 against S0: |Delta Gamma| <= 0.005 at every bin",
            "falsified_if": "P1 or P2 fails; P3 failing means O1 is not a clean test",
        },
        "freqs_hz": battery.FREQS.astype(float).tolist(),
        "records": {},
        "table": [],
    }

    for key, rec in records.items():
        fname = record_name(rec["arm"], rec["record_units"], smoke)
        art["records"][key] = {
            "arm": rec["arm"], "lane": rec["lane"], "record_units": rec["record_units"],
            "n_steps": rec["n_steps"], "device_kind": rec.get("device_kind"),
            "provenance": battery.fixture_provenance(rec["provenance"], index, fname),
            "declared": rec["declared"], "realized": rec["realized"],
            "preflight_text": rec["preflight"]["text"],
            "result_source": rec["result_source"], "result": rec["result"],
            "summary": rec["summary"], "witness": rec.get("witness"),
            "identity_lane_vs_copy": rec.get("identity_lane_vs_copy"),
            "copy_result": rec.get("copy_result"),
            "cross_check": rec.get("cross_check"),
            "warnings": rec.get("warnings"), "copy_warnings": rec.get("copy_warnings"),
            "wall_s": rec.get("wall_s"), "copy_wall_s": rec.get("copy_wall_s"),
            "peak_memory": rec.get("peak_memory"),
        }
        s = rec["summary"]
        row = {"arm": rec["arm"], "record_units": rec["record_units"], "n_steps": rec["n_steps"],
               "absorbing_axes": rec["realized"]["absorbing_axes"],
               "device_kind": (rec.get("device_kind") or [None])[0],
               "status": s["status"], "all_finite": s["all_finite"]}
        if rec["lane"] == "one_port":
            row.update({k: s[k] for k in ("max_abs_s11", "argmax_hz", "min_abs_s11", "argmin_hz",
                                          "n_bins_abs_s11_above_1", "max_alpha_np_per_m")})
            row["derived_max_abs_b_over_a"] = s["derived_abs_b_over_a_at_probe_centroid"]["max"]
        else:
            row.update({k: s[k] for k in ("column_power_min", "column_power_argmin_hz",
                                          "column_power_argmin_column", "column_power_max",
                                          "max_abs_s11", "max_abs_s22", "settling_db")})
        art["table"].append(row)

    def rec_of(arm, which=0):
        """The arm's short (0) or doubled (1) record: 12 / 24 units, or the
        smoke run's pair."""
        units = units_for(arm, smoke)
        return None if which >= len(units) else records.get(f"{arm}_u{units[which]:g}")

    def one_port_arm(arm):
        r12, r24 = rec_of(arm, 0), rec_of(arm, 1)
        out_ = {}
        for tag, r in (("u12", r12), ("u24", r24)):
            if r is None:
                out_[tag] = None
                continue
            s = r["summary"]
            out_[tag] = {"max_abs_gamma": s["max_abs_s11"], "argmax_hz": s["argmax_hz"],
                         "min_abs_gamma": s["min_abs_s11"], "argmin_hz": s["argmin_hz"],
                         "all_finite": s["all_finite"],
                         "max_le_1_02": (None if s["max_abs_s11"] is None
                                         else bool(s["max_abs_s11"] <= MAX_ABS_GAMMA))}
        finite = all(r is not None and r["summary"]["all_finite"] for r in (r12, r24))
        if finite:
            out_["shift_12_24_per_bin"] = _per_bin_shift(r12, r24)
            out_["shift_12_24_of_max"] = (r24["summary"]["max_abs_s11"]
                                          - r12["summary"]["max_abs_s11"])
            out_["shift_below_0_0259"] = bool(out_["shift_12_24_per_bin"] < DOUBLING_SHIFT_MAX)
            out_["shift_of_max_below_0_0259"] = bool(abs(out_["shift_12_24_of_max"])
                                                     < DOUBLING_SHIFT_MAX)
        else:
            out_["shift_12_24_per_bin"] = out_["shift_12_24_of_max"] = None
            out_["shift_below_0_0259"] = out_["shift_of_max_below_0_0259"] = None
        return out_

    o0, o1, o2 = one_port_arm("O0"), one_port_arm("O1"), one_port_arm("O2")
    art["o0_reference"] = o0

    def p_max(block):
        return _all(None if block[t] is None else block[t]["max_le_1_02"] for t in ("u12", "u24"))

    art["P1"] = {"arm": "O1", **o1,
                 "holds": _all([p_max(o1), o1["shift_below_0_0259"]]),
                 "holds_with_shift_of_max": _all([p_max(o1), o1["shift_of_max_below_0_0259"]])}
    p2_min = _all(None if (o2[t] is None or o2[t]["min_abs_gamma"] is None)
                  else bool(o2[t]["min_abs_gamma"] >= MIN_ABS_GAMMA_SHIELDED)
                  for t in ("u12", "u24"))
    art["P2"] = {"arm": "O2", **o2, "min_ge_0_98": p2_min,
                 "holds": _all([p_max(o2), o2["shift_below_0_0259"], p2_min]),
                 "holds_with_shift_of_max": _all([p_max(o2), o2["shift_of_max_below_0_0259"],
                                                  p2_min])}
    s0, s1 = rec_of("S0"), rec_of("S1")
    if (s0 is not None and s1 is not None and s0["summary"]["all_finite"]
            and s1["summary"]["all_finite"]):
        d = np.abs(_gamma(s1) - _gamma(s0))
        dm = np.abs(np.abs(_gamma(s1)) - np.abs(_gamma(s0)))
        f = np.asarray(art["freqs_hz"])
        art["P3"] = {"compared": "S1 (copy, CPML xyz) against S0 (the shipped lane)",
                     "abs_delta_gamma": d.astype(float).tolist(),
                     "max_abs_delta_gamma": float(d.max()),
                     "argmax_hz": float(f[int(np.argmax(d))]),
                     "max_abs_delta_magnitude": float(dm.max()),
                     "s0_max_abs_gamma": s0["summary"]["max_abs_s11"],
                     "s1_max_abs_gamma": s1["summary"]["max_abs_s11"],
                     "holds": bool(d.max() <= P3_MAX_ABS_DELTA)}
    else:
        art["P3"] = {"holds": None,
                     "missing_or_non_finite": [k for k, r in (("S0", s0), ("S1", s1))
                                               if r is None or not r["summary"]["all_finite"]]}

    t0, t2 = rec_of("T0"), rec_of("T2")
    cp = {}
    for tag, r in (("T0", t0), ("T2", t2)):
        if r is None:
            cp[tag] = None
            continue
        s = r["summary"]
        cp[tag] = {k: s[k] for k in ("column_power_min", "column_power_argmin_hz",
                                      "column_power_argmin_column", "column_power_max",
                                      "column_power_per_column", "max_abs_s11", "max_abs_s22",
                                      "min_abs_s21", "settling_db")}
    if (t0 is not None and t2 is not None and t0["summary"]["all_finite"]
            and t2["summary"]["all_finite"]):
        cp["t2_min_minus_t0_min"] = cp["T2"]["column_power_min"] - cp["T0"]["column_power_min"]
        cp["t2_min_ge_0_995"] = bool(cp["T2"]["column_power_min"] >= COLUMN_POWER_T2_FLOOR)
        cp["t2_within_0_005_of_t0"] = bool(abs(cp["t2_min_minus_t0_min"])
                                          <= COLUMN_POWER_SAME_WITHIN)
        col0 = np.asarray(t0["summary"]["column_power"])
        col2 = np.asarray(t2["summary"]["column_power"])
        cp["per_bin_t2_minus_t0"] = (col2 - col0).astype(float).tolist()
        cp["max_abs_per_bin_t2_minus_t0"] = float(np.max(np.abs(col2 - col0)))
    art["column_power"] = cp

    # W1 beside the growth frequencies of O0 and of the stored 16 mm board
    growth = {"this_campaign_O0": {t: (None if o0[t] is None else o0[t]["argmax_hz"])
                                   for t in ("u12", "u24")}}
    stored = json.loads((battery.REPO / OPEN_ABSORBER_ARTIFACT).read_text())
    growth["stored_open_absorber_records"] = {
        f"{row['arm']}_rung9_u{row['record_units']:g}": {
            "board_m": row["board_m"], "max_abs_s11": row["max_abs_s11"],
            "argmax_hz": row["argmax_hz"]}
        for row in stored["table"] if row["rung_annulus_cells"] == 9}
    growth["stored_open_absorber_measured_at_commit"] = stored["measured_at_commit"]
    art["W1"] = None if w1 is None else {
        "provenance": battery.fixture_provenance(w1["provenance"], index, w1_name(smoke)),
        "what": w1["what"], "cans": w1["cans"], "growth_frequencies_beside": growth}

    art["identity"] = {key: r.get("identity_lane_vs_copy") for key, r in records.items()
                       if r.get("identity_lane_vs_copy") is not None}

    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = artifact_out.with_suffix(artifact_out.suffix + ".tmp")
    tmp.write_text(json.dumps(art, indent=1, sort_keys=False) + "\n")
    tmp.replace(artifact_out)
    battery._log(f"wrote {artifact_out} ({len(records)} records, missing {missing})")
    for row in art["table"]:
        print(json.dumps(row))
    for p in ("P1", "P2", "P3"):
        print(p, json.dumps({k: v for k, v in art[p].items() if not isinstance(v, list)}))
    print("column_power", json.dumps({k: v for k, v in art["column_power"].items()
                                      if not isinstance(v, list)}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=tuple(ARMS) + ("ALL",))
    ap.add_argument("--w1", action="store_true", help="the TE cutoffs of the outer region")
    ap.add_argument("--smoke", action="store_true",
                    help=f"every stage at {SMOKE_UNITS} record units, written as "
                         "smoke_*.json; for exercising the code paths")
    ap.add_argument("--out", help="directory for the record JSONs")
    ap.add_argument("--run-id", default=None, help="the compute run id, recorded as-is")
    ap.add_argument("--layout-only", action="store_true",
                    help="build and assert the arms' geometry, solve nothing")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--run-index", default=None,
                    help="assemble only: a JSON mapping each record file to its compute "
                         "run id and run directory name")
    ap.add_argument("--resource-choice", default=None,
                    help="assemble only: the resource decision record, stored as-is")
    ap.add_argument("--artifact-out", default=str(battery.REPO / ARTIFACT))
    args = ap.parse_args()

    if args.layout_only:
        return stage_layout_only(args.arm)
    if not args.out:
        ap.error("--out is required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.assemble:
        stage_assemble(args, out, Path(args.artifact_out))
        return 0
    if args.w1:
        stage_w1(args, out / w1_name(args.smoke))
        return 0
    if args.arm is None:
        ap.error("one of --arm, --w1, --assemble or --layout-only is required")
    if args.arm == "ALL" and not args.smoke:
        ap.error("--arm ALL is for --smoke; the campaign runs one job per arm")
    stage_arm(args, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
