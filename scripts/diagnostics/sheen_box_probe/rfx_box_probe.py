#!/usr/bin/env python
"""Does the domain wall reach the Sheen filter's stopband?  The rfx side.

Two 2.413 mm wide 50 ohm microstrip feeds on 0.794 mm of lossless RT/Duroid
(eps_r 2.2) are joined by one wide 20.320 x 2.540 mm low-impedance section.
The section is a shunt capacitance, so |S21| is flat to about 5.5 GHz and then
falls into a stopband whose transmission zeros near 7 and 8 GHz come from the
transverse resonance of that wide section -- they are set by the field at its
two transverse (y) edges, and the case's domain wall sits 3.0 mm, 3.8 substrate
thicknesses, beyond each of those edges.  What was seen: rfx's ladder puts the
deepest 5-10 GHz minimum at 8.29750 GHz (h/7) and 8.29781 GHz (h/12), 3.2 %
above the openEMS record's 8.0379 GHz and Palace's 8.0482 GHz, with the same
curve shape stretched up in frequency across the whole band.

ASSUMPTION this probe measures, stated as an assumption and not as a finding:
that the transverse wall at 3.8 h from the section's edges enters the stopband
features at all.  Nothing here decides whether it does, which side it moves, or
what the 3.2 % is; this script produces the numbers and the leader reads them.

WHAT IT VARIES
--------------
One mesh rung (default dx = h/7 = 113.429 um, ``--dx``), five boxes:

    id   Y_CLEAR   Z_AIR   CPML_LAYERS   why
    A0    3 mm      3 mm    8            control = the ladder's rung
    A1    6 mm      3 mm    8            side wall moved out
    A2   12 mm      3 mm    8            side wall moved far out
    A3    3 mm      3 mm   16            absorber twice as thick, same clearance
    A4    3 mm      6 mm    8            top wall moved out

Both clearances are snapped to a WHOLE number of cells away from the control's
3.0 mm, per rung: at h/7 the 6 and 12 mm above are built as 5.9491 and
11.9609 mm.  Why, measured: the clearance sets where both feed centres sit, and
6.0 mm at h/3 is 11.335 cells from the control, which lands the two nominally
identical 2.413 mm feeds at different sub-cell offsets and realizes them 9 and
10 node rows wide -- the case refuses that board, with that message, before it
solves anything.  A whole number of cells moves every y coordinate by the same
integer number of rows, so each config is the control's board translated and
the clearance is the only thing that changed.  ``--no-cell-snap`` hands the case
the table's own value and reproduces the refusal.

Read per config: the realized domain and grid, the realized distance from the
section's y edges to the absorber's inner face, the deepest |S21| minimum in
5-10 GHz (bin, sub-bin refined, depth), the -3 dB corner, the passband mean,
the ring-down settling and the passivity excess ``run_rung`` already measures,
the wall time and the cell count.  Then the residual of each config's null
against A0's, and max |S21| difference in dB against A0 and against the frozen
openEMS ``stage_b_fine`` record over 2-12 GHz where both curves are above
-20 dB.

WHAT THIS IS NOT
----------------
A probe.  It runs no Stage A reproduce gate, it writes nothing under
``tests/crossval/sheen_lpf/reference/``, it commits no record and it asserts no
threshold of its own.  The only assertions it makes are the ones the case's own
``run_rung`` already makes on every rung: the realized board, the -40 dB
ring-down and the passivity bound.  A0 reproducing the ladder's number is the
check that the patching below did not change the control.

MECHANICS, AND WHAT WAS CHECKED RATHER THAN ASSUMED
---------------------------------------------------
``tests/crossval/sheen_lpf/test_sheen_lpf.py`` is imported by path and its
y-derived and z-derived module globals are re-derived per config exactly as the
module derives them at its lines 121-161:

    Y_CLEAR, Y_SHIFT, IN_FEED_YC, OUT_FEED_YC, PATCH_Y_LO, PATCH_Y_HI, LY,
    Z_AIR, LZ, CPML_LAYERS

``LX`` and the whole x layout do not depend on any of them.  Checked, not
assumed, that this is enough: ``_sim``, ``_add_ports``, ``build``,
``assert_realized``, ``realized_geometry``, ``_print_realized`` and ``run_rung``
all read these names from the module at CALL time and none of them carries one
in a default argument -- ``_audit_definition_time_capture`` re-checks both
properties at run time, per invocation, and prints which of the names each
function reads.  ``_rungs()`` reads only the environment and ``LADDER_M`` holds
cell sizes, so neither is affected.  The control's globals are restored after
every config.

The CPML is not inside the declared domain.  ``rfx/grid.py`` builds
``nx = cells_spanning(Lx, dx) + 1 + pad_x_lo + pad_x_hi`` with
``pad = cpml_layers`` on each absorbing face, so the absorber is padding ADDED
OUTSIDE the declared box and its inner face is the declared wall itself.
Measured at h/3: A0 and A3 realize the same section-edge-to-inner-face distance
(3.176 mm) on a grid that grows from 121x117x24 to 137x133x32 cells.  A3 is
therefore a thicker absorber at an UNCHANGED clearance, which is what the table
above says it is.  (The z_lo face is PEC, so it takes no pad.)

CLI: ``--dx``, ``--configs A0,A2``, ``--out DIR``, ``--dry-run`` (no solver, no
GPU, no jax needed), ``--no-cell-snap``, ``--mutate-skip-patch ID`` (dry run
only; see its help).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------- the configs
# Every length in metres.
CONFIGS = (
    {"id": "A0", "y_clear": 3.0e-3, "z_air": 3.0e-3, "cpml": 8,
     "why": "control = the ladder's rung"},
    {"id": "A1", "y_clear": 6.0e-3, "z_air": 3.0e-3, "cpml": 8,
     "why": "side wall moved out"},
    {"id": "A2", "y_clear": 12.0e-3, "z_air": 3.0e-3, "cpml": 8,
     "why": "side wall moved far out"},
    {"id": "A3", "y_clear": 3.0e-3, "z_air": 3.0e-3, "cpml": 16,
     "why": "absorber twice as thick at the same clearance"},
    {"id": "A4", "y_clear": 3.0e-3, "z_air": 6.0e-3, "cpml": 8,
     "why": "top wall moved out"},
)
CONTROL_ID = "A0"

# The module globals this probe re-derives per config. Everything else in the
# test module is left alone.
PATCHED = ("Y_CLEAR", "Y_SHIFT", "IN_FEED_YC", "OUT_FEED_YC", "PATCH_Y_LO",
           "PATCH_Y_HI", "LY", "Z_AIR", "LZ", "CPML_LAYERS")

# The functions whose globals-at-call-time behaviour the patch depends on.
AUDITED = ("_sim", "_add_ports", "build", "assert_realized",
           "realized_geometry", "_print_realized", "run_rung")

H_SUB = 0.794e-3
DEFAULT_DX = H_SUB / 7.0
OPENEMS_STAGE = "stage_b_fine"


# ------------------------------------------------------------- the test module
def repo_root() -> Path:
    env = os.environ.get("RFX_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def load_test_module():
    """``tests/crossval/sheen_lpf/test_sheen_lpf.py`` as a module.

    Loaded by path, with the repository root on ``sys.path`` first so its own
    ``from rfx import ...`` and ``from tests._realized_geometry import ...``
    resolve.  The module computes its own ``_REPO_ROOT`` from ``__file__``, so
    the estimator and the reference files it reads are the repository's.
    """
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    path = root / "tests" / "crossval" / "sheen_lpf" / "test_sheen_lpf.py"
    if not path.is_file():
        raise SystemExit(f"the Sheen case is not at {path}")
    spec = importlib.util.spec_from_file_location("_sheen_box_probe_case", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_sheen_box_probe_case"] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------- the patch
def snap_to_cells(target: float, control: float, dx: float) -> float:
    """``target`` moved to a WHOLE number of cells away from ``control``.

    Measured, and the reason this exists: the clearance sets where both feed
    centres sit, and the case refuses a board whose two 2.413 mm feeds realize
    different node-row counts -- they are one declared object at two sub-cell
    offsets, and a row apart makes a symmetric board asymmetric.  Y_CLEAR
    6.0 mm at h/3 is 11.335 cells away from the control's 3.0 mm, which lands
    the two feeds at different offsets and realizes them 9 and 10 rows wide;
    ``run_rung`` stops there with exactly that message.  Snapping the DELTA to
    a whole number of cells moves every y-derived coordinate by the same
    integer number of rows, so the board the lattice builds is the control's
    board translated, and the clearance is the only thing that changed.

    The snapped value is reported everywhere beside the one the table asks
    for; at h/7 the five boxes ask for 6 and 12 mm and get 5.9491 and
    11.9609 mm.  ``--no-cell-snap`` turns this off and hands the case the
    table's value, which is how the refusal above is reproduced.
    """
    return control + round((target - control) / dx) * dx


def config_globals(m, cfg: dict, dx: float, *, snap: bool = True) -> dict:
    """The ten globals this config implies, derived as the module derives them.

    ``Y_SHIFT = Y_CLEAR - PATCH_XS_LO`` maps the Sheen transverse frame onto
    this domain's y, so every y-derived quantity moves with the clearance; LZ
    is the substrate plus the air above it.  Both clearances are snapped to a
    whole number of cells away from the control's -- see ``snap_to_cells``.
    """
    control = CONFIGS[0]
    y_clear = float(cfg["y_clear"])
    z_air = float(cfg["z_air"])
    if snap:
        y_clear = snap_to_cells(y_clear, float(control["y_clear"]), dx)
        z_air = snap_to_cells(z_air, float(control["z_air"]), dx)
    y_shift = y_clear - m.PATCH_XS_LO
    patch_y_lo = m.PATCH_XS_LO + y_shift
    patch_y_hi = m.PATCH_XS_HI + y_shift
    return {
        "Y_CLEAR": y_clear,
        "Y_SHIFT": y_shift,
        "IN_FEED_YC": m.IN_FEED_XS_C + y_shift,
        "OUT_FEED_YC": m.OUT_FEED_XS_C + y_shift,
        "PATCH_Y_LO": patch_y_lo,
        "PATCH_Y_HI": patch_y_hi,
        "LY": patch_y_hi + y_clear,
        "Z_AIR": z_air,
        "LZ": m.H_SUB + z_air,
        "CPML_LAYERS": int(cfg["cpml"]),
    }


def snapshot(m) -> dict:
    return {name: getattr(m, name) for name in PATCHED}


def apply_globals(m, values: dict) -> None:
    for name, value in values.items():
        setattr(m, name, value)


def _audit_definition_time_capture(m) -> dict:
    """Did anything bind one of the patched constants at definition time?

    Two properties, both re-checked here rather than trusted: no audited
    function carries a patched value as a DEFAULT argument (a default is
    evaluated once, at def time, and would freeze the control's box into every
    config), and each function reads the patched names out of the module at
    call time -- which is what ``co_names`` membership shows, since a global
    read compiles to LOAD_GLOBAL by name.
    """
    control = {name: getattr(m, name) for name in PATCHED}
    report: dict = {"defaults_carrying_a_patched_value": [], "reads": {}}
    for fname in AUDITED:
        func = getattr(m, fname)
        defaults = list(func.__defaults__ or ()) + list(
            (func.__kwdefaults__ or {}).values())
        for d in defaults:
            for name, value in control.items():
                if isinstance(d, (int, float)) and not isinstance(d, bool) and d == value:
                    report["defaults_carrying_a_patched_value"].append(
                        f"{fname}: default {d!r} equals {name}")
        report["reads"][fname] = [n for n in PATCHED if n in func.__code__.co_names]
    return report


# ------------------------------------------------------- the box, from numbers
def _cells_spanning(length: float, dx: float) -> int:
    """``rfx.grid.cells_spanning`` when rfx imports, a copy of its rule when not.

    The dry run has to state a cell count on a box with no jax and no GPU, and
    the real run must not carry a second spelling of the rule that could drift
    from the product's.
    """
    try:
        from rfx.grid import cells_spanning
    except Exception:
        ratio = length / dx
        nearest = round(ratio)
        if abs(ratio - nearest) <= 8 * float(np.finfo(float).eps) * max(1.0, abs(ratio)):
            cells = int(nearest)
        else:
            cells = int(math.ceil(ratio))
        return 1 if (ratio > 0.0 and cells < 1) else cells
    return int(cells_spanning(length, dx))


def expected_grid(m, values: dict, dx: float) -> dict:
    """The grid shape the constants imply, before anything is built.

    ``Grid.__init__``: ``n = cells_spanning(L, dx) + 1 + pad_lo + pad_hi`` with
    ``pad = cpml_layers`` on every absorbing face and 0 on a PEC one.  This
    board's z_lo is PEC (``BoundarySpec(z=Boundary(lo="pec", hi="cpml"))``), so
    only z_lo takes no pad.
    """
    n = int(values["CPML_LAYERS"])
    nx = _cells_spanning(m.LX, dx) + 1 + n + n
    ny = _cells_spanning(values["LY"], dx) + 1 + n + n
    nz = _cells_spanning(values["LZ"], dx) + 1 + 0 + n
    return {"grid_shape": (nx, ny, nz), "n_cells": nx * ny * nz,
            "pads": {"x_lo": n, "x_hi": n, "y_lo": n, "y_hi": n,
                     "z_lo": 0, "z_hi": n}}


def wall_geometry(m, g: dict, dx: float) -> dict:
    """Where the grid put the section's y edges and the absorber's inner face.

    Read from the node line the simulation builds, not from the constants: the
    declared clearance is 3.0 mm but the section's realized lo edge lands on
    whichever node row the rasterizer picked.  ``_node_line`` returns the
    DECLARED-domain coordinate, so the absorber's inner faces are the nodes at
    index ``CPML_LAYERS`` and ``-1-CPML_LAYERS``.
    """
    from tests._realized_geometry import _node_line

    sim = m.build(dx)
    grid = sim._build_grid()
    ys = np.asarray(_node_line(grid, 1), dtype=float)
    n = int(m.CPML_LAYERS)
    j0, j1 = g["patch"]["rows"]
    inner_lo, inner_hi = float(ys[n]), float(ys[-1 - n])
    return {
        "section_y_edges_mm": (float(ys[j0]) * 1e3, float(ys[j1]) * 1e3),
        "absorber_inner_face_mm": (inner_lo * 1e3, inner_hi * 1e3),
        "absorber_outer_face_mm": (float(ys[0]) * 1e3, float(ys[-1]) * 1e3),
        "section_edge_to_inner_face_mm": (
            (float(ys[j0]) - inner_lo) * 1e3, (inner_hi - float(ys[j1])) * 1e3),
        "absorber_thickness_mm": (
            (inner_lo - float(ys[0])) * 1e3, (float(ys[-1]) - inner_hi) * 1e3),
        "n_y_nodes": int(ys.size),
    }


# ------------------------------------------------------------------ the solve
def run_config(m, cfg: dict, dx: float, *, snap: bool = True) -> dict:
    """One box: patch, solve through the case's own ``run_rung``, read it."""
    values = config_globals(m, cfg, dx, snap=snap)
    apply_globals(m, values)
    print(f"\n{'=' * 78}\n=== config {cfg['id']}: Y_CLEAR {cfg['y_clear'] * 1e3:g} mm, "
          f"Z_AIR {cfg['z_air'] * 1e3:g} mm, CPML_LAYERS {cfg['cpml']} "
          f"-- {cfg['why']}\n{'=' * 78}")
    print(f"  asked for Y_CLEAR {cfg['y_clear'] * 1e3:.4f} / Z_AIR "
          f"{cfg['z_air'] * 1e3:.4f} mm; "
          + (f"snapped to {values['Y_CLEAR'] * 1e3:.4f} / "
             f"{values['Z_AIR'] * 1e3:.4f} mm = "
             f"{round((values['Y_CLEAR'] - CONFIGS[0]['y_clear']) / dx):+d} / "
             f"{round((values['Z_AIR'] - CONFIGS[0]['z_air']) / dx):+d} cells "
             f"from the control" if snap else "cell snap OFF"))
    print(f"  declared box {m.LX * 1e3:.3f} x {m.LY * 1e3:.3f} x {m.LZ * 1e3:.3f} mm; "
          f"feeds at y {m.IN_FEED_YC * 1e3:.4f} / {m.OUT_FEED_YC * 1e3:.4f} mm; "
          f"section y {m.PATCH_Y_LO * 1e3:.4f}-{m.PATCH_Y_HI * 1e3:.4f} mm")

    r = m.run_rung(dx)
    g = r["realized"]
    null = m.stopband_null(r["freqs_hz"], np.abs(r["s21"]))
    cut = m.passband_cutoff_3db(r["freqs_hz"], np.abs(r["s21"]))
    walls = wall_geometry(m, g, dx)

    print(f"  stopband null {null['f'] / 1e9:.5f} GHz (bin {null['bin_f'] / 1e9:.5f}, "
          f"sub-bin shift {null['sub_bin_shift']:+.3f}), depth {null['depth_db']:.2f} dB")
    print(f"  passband mean {cut['mean_db']:.3f} dB; -3 dB corner "
          + ("none in the sweep" if cut["f_3db"] is None
             else f"{cut['f_3db'] / 1e9:.5f} GHz"))
    print(f"  the section's y edges sit at {walls['section_y_edges_mm'][0]:.4f} and "
          f"{walls['section_y_edges_mm'][1]:.4f} mm; the absorber's inner faces at "
          f"{walls['absorber_inner_face_mm'][0]:.4f} and "
          f"{walls['absorber_inner_face_mm'][1]:.4f} mm")
    print(f"  section edge to absorber inner face: "
          f"{walls['section_edge_to_inner_face_mm'][0]:.4f} / "
          f"{walls['section_edge_to_inner_face_mm'][1]:.4f} mm; absorber thickness "
          f"{walls['absorber_thickness_mm'][0]:.4f} / "
          f"{walls['absorber_thickness_mm'][1]:.4f} mm "
          f"({int(m.CPML_LAYERS)} cells at dx {dx * 1e6:.3f} um)")

    return {
        "id": cfg["id"], "why": cfg["why"],
        "y_clear_asked_mm": cfg["y_clear"] * 1e3,
        "z_air_asked_mm": cfg["z_air"] * 1e3,
        "y_clear_mm": values["Y_CLEAR"] * 1e3, "z_air_mm": values["Z_AIR"] * 1e3,
        "cell_snap": bool(snap),
        "cpml_layers": int(cfg["cpml"]),
        "dx_m": dx,
        "declared_box_mm": (m.LX * 1e3, m.LY * 1e3, m.LZ * 1e3),
        "grid_shape": list(g["grid_shape"]), "n_cells": int(g["n_cells"]),
        "walls": walls,
        "null": null, "cutoff": cut,
        "settling_db": (None if r["settling_db"] is None
                        else np.asarray(r["settling_db"], dtype=float).tolist()),
        "max_excess_in_band": r["max_excess_in_band"],
        "max_excess_full_sweep": r["max_excess_full_sweep"],
        "energy_sum_band": list(r["energy_sum_band"]),
        "wall_s": float(r["wall_s"]),
        "n_preflight": int(r["n_preflight"]),
        "n_probe_clearance_findings": int(r["n_probe_clearance_findings"]),
        "_rung": r,
    }


# ------------------------------------------------------------------ the reads
def residual(f: float, f_ref: float) -> float:
    return abs(f - f_ref) / f_ref


def compare_against(m, label: str, rung: dict, ref_f_ghz, ref_s21_mag) -> dict:
    """The case's own comparator, so this probe cannot drift from it.

    ``_compare`` takes any reference arrays: handed A0's curve it reads config
    against control, handed the frozen record's it reads config against
    openEMS.  Both apply the case's two-sided -20 dB deep-null exclusion over
    2-12 GHz.
    """
    c = m._compare(label, rung, ref_f_ghz, ref_s21_mag)
    return {
        "label": label,
        "max_abs_delta_db": float(c["max_abs_delta_db"]),
        "worst_f_ghz": (float(c["ref_f_ghz"][c["worst_index"]])
                        if c["worst_index"] >= 0 else None),
        "n_compared": int(c["n_compared"]), "n_in_band": int(c["n_in_band"]),
        "our_null_ghz": float(c["our_null_ghz"]),
        "ref_null_ghz": float(c["ref_null_ghz"]),
        "null_pct": float(c["null_pct"]),
        "our_cutoff_ghz": (None if c["our_cutoff"]["f_3db"] is None
                           else c["our_cutoff"]["f_3db"] / 1e9),
        "ref_cutoff_ghz": (None if c["ref_cutoff"]["f_3db"] is None
                           else c["ref_cutoff"]["f_3db"] / 1e9),
        "our_passband_db": float(c["our_cutoff"]["mean_db"]),
        "ref_passband_db": float(c["ref_cutoff"]["mean_db"]),
    }


def write_figure(m, results, openems, directory: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sheen_box_probe_rfx.png"
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for res in results:
        r = res["_rung"]
        ax.plot(np.asarray(r["freqs_hz"]) / 1e9, m._db(np.abs(r["s21"])),
                lw=1.4,
                label=f"{res['id']}  y_clear {res['y_clear_mm']:.3f} mm, "
                      f"z_air {res['z_air_mm']:.3f} mm, cpml {res['cpml_layers']}")
    if openems is not None:
        st = openems[OPENEMS_STAGE]
        ax.plot(st["freqs_ghz"], m._db(st["s21_mag"]), "k-", lw=1.0,
                label=f"openEMS {OPENEMS_STAGE} (frozen record)")
    ax.set_xlim(*(f / 1e9 for f in m.REFERENCE_BAND_HZ))
    ax.set_ylim(-70, 5)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("|S21| (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- the dry run
def dry_run(m, configs, dx: float, skip_patch_for: str | None,
            snap: bool = True) -> int:
    print("=" * 78)
    print("The Sheen low-pass filter -- rfx box probe, DRY RUN (nothing is solved)")
    print("=" * 78)
    print(f"  rung dx = {dx * 1e6:.4f} um = h_sub/{m.H_SUB / dx:.4g}, "
          f"h_sub = {m.H_SUB * 1e6:.1f} um")
    print(f"  x layout is untouched by every config: LX = {m.LX * 1e3:.3f} mm, "
          f"section x {m.PATCH_X0 * 1e3:.3f}-{m.PATCH_X1 * 1e3:.3f} mm")
    print(f"  the wide section is {m.PATCH_TRV_LEN * 1e3:.3f} mm across y, so its "
          f"two transverse edges are what Y_CLEAR moves the wall away from")
    audit = _audit_definition_time_capture(m)
    print("\n  definition-time capture audit "
          "(does anything freeze a patched constant at def time?):")
    print(f"    defaults carrying a patched value: "
          f"{audit['defaults_carrying_a_patched_value'] or 'none'}")
    for fname, reads in audit["reads"].items():
        print(f"    {fname:20s} reads {reads if reads else '(none directly)'}")
    if skip_patch_for:
        print(f"\n  MUTATION: the globals patch is SKIPPED for config "
              f"{skip_patch_for}. Its box below should then be the control's, "
              f"which is what shows the printed box is produced by the patch and "
              f"not by the config table.")
    print("\n  rfx's CPML is padding ADDED OUTSIDE the declared box "
          "(rfx/grid.py: n = cells_spanning(L, dx) + 1 + pad_lo + pad_hi), so the "
          "absorber's inner face is the declared wall and the clearance below "
          "does not move with CPML_LAYERS. z_lo is PEC and takes no pad.")
    if snap:
        print("  Y_CLEAR and Z_AIR are snapped to a WHOLE number of cells away "
              "from the control's 3.0 mm, so every config builds the control's "
              "board translated by an integer number of rows. Without it the two "
              "feeds land at different sub-cell offsets and the case refuses the "
              "board (--no-cell-snap reproduces that).")
    else:
        print("  CELL SNAP IS OFF: Y_CLEAR and Z_AIR are the table's own values.")
    print()
    control = snapshot(m)
    rows = []
    for cfg in configs:
        if skip_patch_for is not None and cfg["id"] == skip_patch_for:
            values = dict(control)
            note = "  <-- PATCH SKIPPED (mutation)"
        else:
            values = config_globals(m, cfg, dx, snap=snap)
            note = ""
        est = expected_grid(m, values, dx)
        clear_cells = values["Y_CLEAR"] / dx
        rows.append((cfg, values, est))
        print(f"  {cfg['id']}  {cfg['why']}{note}")
        print(f"      asked Y_CLEAR {cfg['y_clear'] * 1e3:6.3f} mm, Z_AIR "
              f"{cfg['z_air'] * 1e3:6.3f} mm -> "
              f"{round((values['Y_CLEAR'] - CONFIGS[0]['y_clear']) / dx):+d} / "
              f"{round((values['Z_AIR'] - CONFIGS[0]['z_air']) / dx):+d} cells "
              f"from the control")
        print(f"      Y_CLEAR {values['Y_CLEAR'] * 1e3:6.3f} mm   "
              f"Z_AIR {values['Z_AIR'] * 1e3:6.3f} mm   "
              f"CPML_LAYERS {values['CPML_LAYERS']:2d}")
        print(f"      declared box  {m.LX * 1e3:.3f} x {values['LY'] * 1e3:.3f} x "
              f"{values['LZ'] * 1e3:.3f} mm")
        print(f"      feeds y {values['IN_FEED_YC'] * 1e3:.4f} / "
              f"{values['OUT_FEED_YC'] * 1e3:.4f} mm; section y "
              f"{values['PATCH_Y_LO'] * 1e3:.4f}-{values['PATCH_Y_HI'] * 1e3:.4f} mm")
        print(f"      grid estimate {est['grid_shape']}  cells {est['n_cells']}  "
              f"pads {est['pads']}")
        print(f"      declared clearance {values['Y_CLEAR'] * 1e3:.3f} mm = "
              f"{clear_cells:.2f} cells; absorber thickness "
              f"{values['CPML_LAYERS'] * dx * 1e3:.3f} mm "
              f"({values['CPML_LAYERS']} cells), outside the box")
        print()
    base = rows[0]
    print("  boxes side by side (the y extent is what the mutation check reads):")
    for cfg, values, est in rows:
        same = ("LY equals the first row's"
                if abs(values["LY"] - base[1]["LY"]) < 1e-15 else "")
        print(f"    {cfg['id']}  LY {values['LY'] * 1e3:8.3f} mm  "
              f"LZ {values['LZ'] * 1e3:6.3f} mm  cells {est['n_cells']:9d}  {same}")
    print("\n  WHAT THIS DRY RUN CANNOT TELL YOU: every number above comes from the "
          "constants and from rfx's own cell-count rule. What the lattice realizes "
          "-- the section's rasterized y edges, the realized strips, the port rows "
          "-- is printed by the real run, per config, from the grid.")
    return 0


# ------------------------------------------------------------------- the main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dx", type=float, default=DEFAULT_DX,
                   help="cell size in metres; default h_sub/7 = 1.1342857142857143e-4. "
                        "Give the exact literal: a rounded dx puts the substrate top "
                        "off the node plane and the case refuses the board.")
    p.add_argument("--configs", default=",".join(c["id"] for c in CONFIGS),
                   help="comma separated subset of "
                        f"{','.join(c['id'] for c in CONFIGS)}")
    p.add_argument("--out", default=None,
                   help="directory for the JSON and the figure; "
                        "RFX_CROSSVAL_FIG_DIR is used for the figure when set")
    p.add_argument("--dry-run", action="store_true",
                   help="print each config's box, grid estimate and cell count "
                        "from the constants and solve nothing")
    p.add_argument("--no-cell-snap", action="store_true",
                   help="hand the case the table's own Y_CLEAR / Z_AIR instead "
                        "of the nearest whole number of cells away from the "
                        "control's. Y_CLEAR 6.0 mm at h/3 is then 11.335 cells "
                        "from the control, the two feeds land at different "
                        "sub-cell offsets and the case refuses the board. Kept "
                        "so that refusal can be reproduced.")
    p.add_argument("--mutate-skip-patch", default=None, metavar="ID",
                   help="DRY RUN ONLY. Skip the globals patch for this config, so "
                        "its printed box falls back to the control's. The proof "
                        "that the per-config box in the real run comes from the "
                        "patch and not from the table this script prints.")
    args = p.parse_args(argv)

    wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
    known = {c["id"]: c for c in CONFIGS}
    bad = [c for c in wanted if c not in known]
    if bad:
        print(f"ERROR: unknown config(s) {bad}; known {list(known)}", file=sys.stderr)
        return 3
    configs = [known[c] for c in wanted]

    if args.mutate_skip_patch and not args.dry_run:
        print("ERROR: --mutate-skip-patch is a dry-run check", file=sys.stderr)
        return 3

    m = load_test_module()

    if args.dry_run:
        return dry_run(m, configs, args.dx, args.mutate_skip_patch,
                       snap=not args.no_cell_snap)

    out_dir = Path(args.out) if args.out else Path(os.environ.get(
        "RFX_CROSSVAL_FIG_DIR", ".")) / "sheen_box_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(os.environ.get("RFX_CROSSVAL_FIG_DIR", out_dir))

    print("=" * 78)
    print("The Sheen low-pass filter -- rfx box probe")
    print("=" * 78)
    print(f"  rung dx = {args.dx * 1e6:.4f} um; configs {wanted}; out {out_dir}")
    audit = _audit_definition_time_capture(m)
    print(f"  definition-time capture audit: defaults carrying a patched value = "
          f"{audit['defaults_carrying_a_patched_value'] or 'none'}")
    for fname, reads in audit["reads"].items():
        print(f"    {fname:20s} reads {reads if reads else '(none directly)'}")
    if audit["defaults_carrying_a_patched_value"]:
        print("ERROR: a patched constant is frozen in a default argument; the "
              "per-config box would not be what this script says it is.",
              file=sys.stderr)
        return 3

    control = snapshot(m)
    results = []
    t_all = time.perf_counter()
    for cfg in configs:
        try:
            results.append(run_config(m, cfg, args.dx,
                                      snap=not args.no_cell_snap))
        finally:
            apply_globals(m, control)
    print(f"\n  total wall time {time.perf_counter() - t_all:.1f} s")

    openems = None
    ref_path = (repo_root() / "tests" / "crossval" / "sheen_lpf" / "reference"
                / "openems_sheen.json")
    if ref_path.is_file():
        with ref_path.open() as fh:
            openems = json.load(fh)

    by_id = {r["id"]: r for r in results}
    base = by_id.get(CONTROL_ID)

    print("\n" + "=" * 78)
    print("READING 1 -- the box, the grid and the realized wall distance")
    print("=" * 78)
    print("  id  y_clear  z_air  cpml   declared box (mm)        grid            "
          "cells      edge->absorber (mm)")
    print("      (y_clear and z_air as realized after the cell snap; what each "
          "config asked for is in the JSON)")
    for r in results:
        b = r["declared_box_mm"]
        w = r["walls"]["section_edge_to_inner_face_mm"]
        print(f"  {r['id']}  {r['y_clear_mm']:6.2f}  {r['z_air_mm']:5.2f}  "
              f"{r['cpml_layers']:4d}   {b[0]:.3f} x {b[1]:7.3f} x {b[2]:6.3f}   "
              f"{str(tuple(r['grid_shape'])):16s} {r['n_cells']:9d}  "
              f"{w[0]:7.4f} / {w[1]:7.4f}")

    print("\n" + "=" * 78)
    print("READING 2 -- the features, the witnesses and the cost")
    print("=" * 78)
    print("  id   null (GHz)   depth (dB)  -3 dB (GHz)  passband (dB)  "
          "settling (dB)     excess     wall (s)")
    for r in results:
        cut = r["cutoff"]
        corner = "none" if cut["f_3db"] is None else f"{cut['f_3db'] / 1e9:.5f}"
        settle = ("n/a" if r["settling_db"] is None
                  else "/".join(f"{v:.2f}" for v in r["settling_db"]))
        print(f"  {r['id']}  {r['null']['f'] / 1e9:10.5f}  {r['null']['depth_db']:9.2f}  "
              f"{corner:>11s}  {cut['mean_db']:12.3f}  {settle:>16s}  "
              f"{r['max_excess_in_band']:9.5f}  {r['wall_s']:9.1f}")

    print("\n" + "=" * 78)
    print(f"READING 3 -- residual of the stopband null against {CONTROL_ID}")
    print("=" * 78)
    residuals = {}
    if base is None:
        print(f"  {CONTROL_ID} was not run in this invocation; no residual table.")
    else:
        f0 = base["null"]["f"]
        print(f"  {CONTROL_ID} null = {f0 / 1e9:.5f} GHz")
        print("  id   null (GHz)    delta (MHz)    residual (%)")
        for r in results:
            res = residual(r["null"]["f"], f0)
            residuals[r["id"]] = res
            print(f"  {r['id']}  {r['null']['f'] / 1e9:10.5f}  "
                  f"{(r['null']['f'] - f0) / 1e6:+12.3f}  {100.0 * res:13.4f}")

    print("\n" + "=" * 78)
    print("READING 4 -- |S21| in dB against the control and against openEMS "
          f"{OPENEMS_STAGE}")
    print("=" * 78)
    comparisons = {}
    for r in results:
        entry = {}
        if base is not None:
            b = base["_rung"]
            entry["vs_control"] = compare_against(
                m, f"{r['id']} vs {CONTROL_ID}", r["_rung"],
                np.asarray(b["freqs_hz"], dtype=float) / 1e9, np.abs(b["s21"]))
        if openems is not None:
            st = openems[OPENEMS_STAGE]
            entry["vs_openems"] = compare_against(
                m, f"{r['id']} vs openEMS {OPENEMS_STAGE}", r["_rung"],
                st["freqs_ghz"], st["s21_mag"])
        comparisons[r["id"]] = entry
        for key, c in entry.items():
            ours = ("none" if c["our_cutoff_ghz"] is None
                    else f"{c['our_cutoff_ghz']:.5f}")
            theirs = ("none" if c["ref_cutoff_ghz"] is None
                      else f"{c['ref_cutoff_ghz']:.5f}")
            print(f"  {r['id']} {key}: max |dS21| {c['max_abs_delta_db']:.3f} dB at "
                  f"{c['worst_f_ghz']} GHz over {c['n_compared']} of "
                  f"{c['n_in_band']} bins; null {c['our_null_ghz']:.5f} vs "
                  f"{c['ref_null_ghz']:.5f} GHz ({c['null_pct']:.3f} %); "
                  f"-3 dB {ours} vs {theirs} GHz; "
                  f"passband {c['our_passband_db']:.3f} vs "
                  f"{c['ref_passband_db']:.3f} dB")

    fig_path = None
    try:
        fig_path = write_figure(m, results, openems, fig_dir)
        print(f"\n  figure: {fig_path}")
    except Exception as exc:  # a missing matplotlib must not lose the numbers
        print(f"\n  figure NOT written: {exc!r}")

    payload = {
        "what": "rfx box probe for the Sheen low-pass filter -- a diagnostic, "
                "not a test and not a reference record",
        "dx_m": args.dx,
        "control": CONTROL_ID,
        "openems_stage_compared": OPENEMS_STAGE if openems is not None else None,
        "definition_time_capture_audit": audit,
        "configs": [{k: v for k, v in r.items() if k != "_rung"} for r in results],
        "residual_vs_control": residuals,
        "comparisons": comparisons,
        "figure": None if fig_path is None else str(fig_path),
        "curves": {r["id"]: {
            "freqs_ghz": (np.asarray(r["_rung"]["freqs_hz"], dtype=float) / 1e9).tolist(),
            "s21_db": m._db(np.abs(r["_rung"]["s21"])).tolist(),
            "s11_db": m._db(np.abs(r["_rung"]["s11"])).tolist(),
        } for r in results},
    }
    json_path = out_dir / "sheen_box_probe_rfx.json"
    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"  json:   {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
