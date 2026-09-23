#!/usr/bin/env python
"""The Sheen low-pass filter, solved by openEMS on rfx's own lattice.

Two 2.413 mm wide 50 ohm microstrip feeds on 0.794 mm of lossless RT/Duroid
(eps_r 2.2) are joined by one wide 20.320 x 2.540 mm low-impedance section, and
the section's transverse resonance puts a double transmission zero near 7 and
8 GHz into |S21|.  Two FDTD codes that solve the same discrete structure --
the same cells, the same metal edges, the same substrate cells, the absorbers
in the same places -- should put those zeros at the same frequencies.  What was
seen: rfx's uniform ladder reads the upper zero at 8.38113 / 8.36112 / 8.29750
/ 8.29781 GHz (h/3, h/5, h/7, h/12), openEMS on its own smoothed mesh reads
7.9939 / 8.0204 / 8.0379 GHz and Palace 8.0482 GHz, and moving or changing the
domain wall moved the zero by 0.18 % in rfx and 0.24-0.33 % in openEMS
(``rfx_box_probe.py``, ``openems_box_probe.py``; VESSL 369367263565).

WHAT IT VARIES
--------------
The rung: dx = h_sub/n, n in {3, 5, 7, 12} (``--rungs``).  At each rung openEMS
is handed rfx's lattice for that rung and nothing else: every rfx node line
(the 8-cell absorber pad included) as an openEMS mesh line, no
``SmoothMeshLines``, no thirds rule, no added line; PML_8 on x_lo, x_hi, y_lo,
y_hi and z_hi, so each absorber occupies exactly rfx's 8 pad cells; PEC on
z_lo; the substrate in cells k = 0 .. n_sub-1 over the whole lateral extent,
pad included; and the metal on the nodes rfx's rasterizer put it on, not on
the drawn coordinates.  The rfx rungs are not one board -- the section's
length along the line realizes 2.117 / 2.382 / 2.495 / 2.448 mm against a
drawn 2.540 mm -- so each rung here reproduces THAT rung's board.

Read per rung: both zeros (the |S21| minimum in 6.5-7.6 GHz and in
7.6-9.5 GHz, the repository's ``refined_extremum`` in log), the -3 dB corner
and the passband mean through the maker's ``_stage_b_features``, the box
energy and step count openEMS ended at, the passive port's incident wave;
then a table against rfx's ladder (the numbers hard-coded below, from its log)
and against openEMS's own frozen record, and one figure.

WHAT THIS IS NOT
----------------
A diagnostic.  STAGE A IS NOT RUN (the reproduce gate passed in VESSL
369367263406/407/408, which made the frozen record).  It writes nothing under
``tests/crossval/sheen_lpf/reference/``, commits no record and gates nothing of
its own.  Two controls stop it: (1) ``--self-check``, run before any solve,
must rebuild rfx's realized board node for node from the integers the ladder
log printed -- grid shape with the pad, substrate cells, sheet plane, every
sheet's node columns and rows, both joints -- and must show the metal edges
openEMS would set equal to rfx's realized PEC edges; (2) every openEMS pass
must end on its own energy criterion (1e-4, -40 dB): truncation is NOT
accepted, so a pass that reaches its step cap fires the shared runner's gate.
The shared runner's other gates stay on: unused primitives, off-mesh ports,
excitation energy, finite |S| <= 2, passivity over 2-12 GHz.

WHAT WAS READ IN rfx, AND WHAT IS MIRRORED (file:line on this branch)
----------------------------------------------------------------------
* Node lines: ``(i - pad_lo) * dx`` (rfx/geometry/rasterize_grid.py:70), with
  ``n = cells_spanning(L, dx) + 1 + pad_lo + pad_hi`` per axis
  (rfx/grid.py:212-221) and pad = cpml_layers = 8 on every absorbing face,
  0 on the PEC z_lo face (rfx/grid.py:161-178).
* A zero-thickness PEC Box is a SHEET: plane = the node nearest its z, footprint
  sampled CLOSED ``lo - 1e-9 dx <= x_i <= hi + 1e-9 dx`` on the node lines
  (rfx/geometry/rasterize_grid.py:369-378, called from
  ``sheet_spec_from_shape``, :441).  Footprints on one plane are UNIONED, then an
  in-plane E edge is PEC iff both its end nodes are in the union
  (rfx/boundaries/pec.py:293-330).  That union is what joins each feed to the
  section: their footprints sit in ADJACENT node columns, so the joining edge
  belongs to neither sheet alone.
* The substrate Box ``[0, H_SUB)`` is sampled half-open on NODES
  (rfx/geometry/csg.py:335), so ``eps_r`` is 2.2 at nodes k < n_sub and 1.0 at
  k >= n_sub; the CPML pad copies the edge slice outward
  (rfx/api/_compile.py:395, rfx/geometry/rasterize_grid.py:1057); PEC never
  enters the pad (rfx/api/_compile.py:404-405, and measured: the input feed's
  first PEC edge is at node column 8 on every rung).
* The PEC geometry openEMS gets is therefore: one box per realized sheet, from
  its first to its last realized node, plus one JOINT box per feed-section
  pair spanning the one cell between their adjacent columns -- openEMS marks an
  edge PEC when the edge's midpoint lies in a metal box
  (openEMS FDTD/operator.cpp:2029-2061, ``CalcPEC_Range`` with ``GetYeeCoords``;
  CSXCAD ``CoordInRange`` is a closed interval with no tolerance,
  src/CSPrimitives.cpp:54-76), i.e. per box, so two abutting boxes would leave
  the joining edge live.  ``--self-check`` computes both edge sets and
  requires them equal.  The openEMS and CSXCAD file:line references are to
  the sources the job's image builds: openEMS-Project 5b423bdf (the
  Dockerfile's OPENEMS_COMMIT), whose submodules are openEMS 2000574e and
  CSXCAD e5581710 -- read, not run.

ASSUMPTIONS, stated and not settled here
----------------------------------------
* The feeds are the MSLPort metal (openEMS python/openEMS/ports.py, MSLPort:
  the port adds ``metal_prop.AddBox(start, stop with stop[exc] = start[exc])``).
  The two ports are soft E sources with no Feed_R (the maker passes none, so
  ``feed_R`` is inf), and the shared runner forms S21 = b2/a1 from ONE drive,
  which is S21 only if the wave leaving port 2 is not sent back (a2 = 0).
  rfx solves two drives (S = B A^-1) and does not need that.  rfx's input strip
  starts at the absorber face and its output strip stops one cell short of it,
  so with rfx's metal exactly, both openEMS lines end OPEN at the absorber.
  ``--line-ends pml`` (the default) continues each strip through the absorber
  cells, as the maker's own board and openEMS's own tutorial run them; that
  adds metal in the 8 pad cells and, at the output end, the one in-domain cell
  between rfx's last output-feed node and the absorber face -- beyond port 2's
  start, outside both measurement planes.  ``--line-ends rfx`` keeps rfx's
  metal exactly.  The passive port's incident wave |a2/a1| over 2-12 GHz is
  recorded either way.
* openEMS's per-edge permittivity on the sheet plane is the quarter-cell
  average (FDTD/operator.cpp:104 default QuarterCell, 1437-1508), which on the
  plane z = H_SUB averages two substrate and two air quarters.  rfx's update
  uses the one node value there.  This probe mirrors the MATERIAL BOXES; it
  does not and cannot make the two codes average the interface alike.

CLI: ``--rungs 3,5,7``, ``--out DIR``, ``--threads``, ``--nrts``,
``--line-ends {pml,rfx}``, ``--dry-run`` (numpy only: the lattice, control 1,
the boundary list, the metal node extents, the cost estimate; exits non-zero if
control 1 fails), ``--self-check`` (control 1 only), ``--against-rfx`` (needs
rfx and jax: compares this lattice's node lines and edge set with rfx's own
realized arrays), and the three mutation checks ``--mutate-shift-section``,
``--mutate-no-y-pad`` and ``--mutate-no-joints``.
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

# ---------------------------------------------------------------- constants
RUNGS = (3, 5, 7, 12)
DEFAULT_RUNGS = "3,5,7"
PAD = 8                                      # rfx cpml_layers = openEMS PML_8
BOUNDARY = ["PML_8", "PML_8", "PML_8", "PML_8", "PEC", "PML_8"]
END_CRITERIA = 1.0e-4                        # -40 dB of the peak box energy
DEFAULT_NRTS = 400000                        # a ceiling, not a record length
ZERO_BANDS_GHZ = {"zero_7": (6.5, 7.6), "zero_8": (7.6, 9.5)}
C0 = 2.99792458e8
TOL_CELLS = 1e-9                             # rfx _REL_TOL, on-lattice tolerance

# rfx's realized board per rung, copied from the ladder log's "realized at dx"
# blocks: VESSL 369367263494 at commit e2ba011e, lab share
# research/rfx/.omx/sheen-lpf-ladder/20260922T160810Z-e2ba011e/ladder.log
# (Mac backup ~/Documents/vessl-run-logs/369367263494_sheen-ladder-4rung.log).
# cols/rows are FULL-grid node indices, pad included.  Joints: (row, (col_lo,
# col_hi), edges on, edges total) -- the log's "feed joint" lines.
LOGGED = {
    3: {"grid": (121, 117, 24), "n_sub": 3, "plane_k": 3,
        "in_feed": ((8, 55), (41, 49)), "section": ((56, 64), (20, 96)),
        "out_feed": ((65, 111), (66, 74)),
        "joints": {"in": (45, (8, 64), 56, 56), "out": (70, (56, 111), 55, 55)}},
    5: {"grid": (190, 183, 33), "n_sub": 5, "plane_k": 5,
        "in_feed": ((8, 86), (63, 77)), "section": ((87, 102), (27, 154)),
        "out_feed": ((103, 180), (105, 119)),
        "joints": {"in": (70, (8, 102), 94, 94), "out": (112, (87, 180), 93, 93)}},
    7: {"grid": (260, 250, 43), "n_sub": 7, "plane_k": 7,
        "in_feed": ((8, 117), (85, 105)), "section": ((118, 140), (35, 213)),
        "out_feed": ((141, 250), (143, 163)),
        "joints": {"in": (95, (8, 140), 132, 132), "out": (153, (118, 250), 132, 132)}},
    12: {"grid": (433, 415, 67), "n_sub": 12, "plane_k": 12,
         "in_feed": ((8, 196), (139, 175)), "section": ((197, 234), (54, 360)),
         "out_feed": ((235, 423), (239, 275)),
         "joints": {"in": (157, (8, 234), 226, 226), "out": (257, (197, 423), 226, 226)}},
}
# rfx's own numbers from the same log (the 8 GHz zero is the deepest 5-10 GHz
# minimum it printed; its 7 GHz zero is not in the log as a number).
RFX_LADDER = {3: {"zero_8": 8.38113, "corner": 5.78586},
              5: {"zero_8": 8.36112, "corner": 5.74375},
              7: {"zero_8": 8.29750, "corner": 5.71532},
              12: {"zero_8": 8.29781, "corner": 5.70016}}
FROZEN_STAGES = ("stage_b_coarse", "stage_b_mid", "stage_b_fine")

# The cost model's measured inputs.  Throughput, cells x steps / wall on 8
# threads: the frozen record's coarse and fine rungs (meta.stages) and the box
# probe's B2 box.  Decay: B2 (PML on x and y) reached -43.5 dB at 8251 steps x
# 1.6774e-13 s = 1.384 ns (VESSL 369367263565).
THROUGHPUT_CELL_STEPS_PER_S = {"coarse record (417000 cells)": 1.08e8,
                               "box probe B2 (417000 cells)": 2.02e8,
                               "fine record (3248430 cells)": 3.62e8}
DECAY_NS = {"B2 measured": 1.384, "3x B2": 4.2, "10 ns": 10.0}


# ------------------------------------------------------------------ modules
def repo_root() -> Path:
    env = os.environ.get("RFX_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


def load_maker():
    """The reference maker, by path.  Pure numpy until a solve is asked for."""
    root = repo_root()
    path = (root / "tests" / "crossval" / "sheen_lpf" / "reference"
            / "make_openems_reference.py")
    if not path.is_file():
        raise SystemExit(f"the reference maker is not at {path}")
    spec = importlib.util.spec_from_file_location("_sheen_identical_grid_maker", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_sheen_identical_grid_maker"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ lattice
def cells_spanning(length: float, dx: float) -> int:
    """rfx/grid.py:32 ``cells_spanning``, copied so this runs without jax.
    ``--self-check`` holds the result against the grid shapes rfx logged."""
    ratio = length / dx
    nearest = round(ratio)
    if abs(ratio - nearest) <= 8 * float(np.finfo(float).eps) * max(1.0, abs(ratio)):
        cells = int(nearest)
    else:
        cells = int(math.ceil(ratio))
    return 1 if (ratio > 0.0 and cells < 1) else cells


def _closed(nodes, lo, hi, dx):
    tol = TOL_CELLS * dx
    return np.flatnonzero((nodes >= lo - tol) & (nodes <= hi + tol))


def build_lattice(maker, n: int, *, shift_section: int = 0,
                  y_pad: bool = True) -> dict:
    """rfx's uniform lattice for dx = h_sub/n and the board it realizes on it.

    ``shift_section`` and ``y_pad`` exist only for the mutation checks.
    """
    dx = maker.H_SUB / n
    p_y = PAD if y_pad else 0
    pads = {"x_lo": PAD, "x_hi": PAD, "y_lo": p_y, "y_hi": p_y,
            "z_lo": 0, "z_hi": PAD}
    nx = cells_spanning(maker.LX, dx) + 1 + pads["x_lo"] + pads["x_hi"]
    ny = cells_spanning(maker.LY, dx) + 1 + pads["y_lo"] + pads["y_hi"]
    nz = cells_spanning(maker.LZ, dx) + 1 + pads["z_lo"] + pads["z_hi"]
    xs = (np.arange(nx, dtype=np.float64) - pads["x_lo"]) * dx
    ys = (np.arange(ny, dtype=np.float64) - pads["y_lo"]) * dx
    zs = (np.arange(nz, dtype=np.float64) - pads["z_lo"]) * dx
    plane_k = int(np.argmin(np.abs(zs - maker.H_SUB)))
    n_sub = int(np.count_nonzero((zs >= 0.0) & (zs < maker.H_SUB)))
    hw = maker.W_FEED / 2.0
    declared = {
        "in_feed": ((0.0, maker.PATCH_X0),
                    (maker.IN_FEED_YC - hw, maker.IN_FEED_YC + hw)),
        "section": ((maker.PATCH_X0, maker.PATCH_X1),
                    (maker.PATCH_Y_LO, maker.PATCH_Y_HI)),
        "out_feed": ((maker.PATCH_X1, maker.LX),
                     (maker.OUT_FEED_YC - hw, maker.OUT_FEED_YC + hw)),
    }
    sheets = {}
    for name, ((x0, x1), (y0, y1)) in declared.items():
        ci, rj = _closed(xs, x0, x1, dx), _closed(ys, y0, y1, dx)
        cols = (int(ci.min()), int(ci.max()))
        rows = (int(rj.min()), int(rj.max()))
        if name == "section" and shift_section:
            cols = (cols[0] + shift_section, cols[1] + shift_section)
        sheets[name] = {"cols": cols, "rows": rows}
    return {"n": n, "dx": dx, "pads": pads, "shape": (nx, ny, nz),
            "xs": xs, "ys": ys, "zs": zs, "plane_k": plane_k, "n_sub": n_sub,
            "sheets": sheets, "declared": declared}


def rfx_union_edges(lat) -> tuple:
    """rfx/boundaries/pec.py:293 ``_sheet_edge_masks`` on the sheet plane:
    union the footprints, then an edge is PEC iff both end nodes are in it."""
    nx, ny, _ = lat["shape"]
    U = np.zeros((nx, ny), dtype=bool)
    for s in lat["sheets"].values():
        U[s["cols"][0]:s["cols"][1] + 1, s["rows"][0]:s["rows"][1] + 1] = True
    ex = np.zeros((nx, ny), dtype=bool)
    ey = np.zeros((nx, ny), dtype=bool)
    ex[:-1, :] = U[:-1, :] & U[1:, :]
    ey[:, :-1] = U[:, :-1] & U[:, 1:]
    return ex, ey


def openems_edges(boxes, lat) -> tuple:
    """openEMS's rule on the sheet plane: an edge is PEC iff its midpoint lies
    in a metal box, closed interval (FDTD/operator.cpp CalcPEC_Range,
    CSXCAD CoordInRange).  ``boxes`` are (x0, x1, y0, y1) in metres."""
    xs, ys = lat["xs"], lat["ys"]
    tol = TOL_CELLS * lat["dx"]
    xm, ym = 0.5 * (xs[:-1] + xs[1:]), 0.5 * (ys[:-1] + ys[1:])
    ex = np.zeros((xs.size, ys.size), dtype=bool)
    ey = np.zeros((xs.size, ys.size), dtype=bool)
    for x0, x1, y0, y1 in boxes:
        xlo, xhi = min(x0, x1), max(x0, x1)
        ylo, yhi = min(y0, y1), max(y0, y1)
        in_xm = (xm >= xlo - tol) & (xm <= xhi + tol)
        in_yn = (ys >= ylo - tol) & (ys <= yhi + tol)
        in_xn = (xs >= xlo - tol) & (xs <= xhi + tol)
        in_ym = (ym >= ylo - tol) & (ym <= yhi + tol)
        ex[:-1, :] |= in_xm[:, None] & in_yn[None, :]
        ey[:, :-1] |= in_xn[:, None] & in_ym[None, :]
    return ex, ey


def metal_plan(lat, line_ends: str) -> list:
    """The PEC primitives openEMS gets, as node-index ranges.

    ``lat["drop_joints"]`` exists only for the mutation check: it hands openEMS
    the three sheets as three abutting boxes, the drawing whose joining edges
    openEMS's per-box rule leaves live."""
    s = lat["sheets"]
    nx = lat["shape"][0]
    plan = [
        {"name": "input feed (port 1 metal)", "role": "port1",
         "cols": s["in_feed"]["cols"], "rows": s["in_feed"]["rows"]},
        {"name": "wide section", "role": "box",
         "cols": s["section"]["cols"], "rows": s["section"]["rows"]},
        {"name": "output feed (port 2 metal)", "role": "port2",
         "cols": s["out_feed"]["cols"], "rows": s["out_feed"]["rows"]},
    ]
    sec = s["section"]
    for label, feed, lo, hi in (
            ("input joint", s["in_feed"], s["in_feed"]["cols"][1], sec["cols"][0]),
            ("output joint", s["out_feed"], sec["cols"][1], s["out_feed"]["cols"][0])):
        if hi == lo + 1 and not lat.get("drop_joints"):
            r0 = max(feed["rows"][0], sec["rows"][0])
            r1 = min(feed["rows"][1], sec["rows"][1])
            if r0 <= r1:
                plan.append({"name": label, "role": "joint",
                             "cols": (lo, hi), "rows": (r0, r1)})
    if line_ends == "pml":
        f = s["in_feed"]
        if f["cols"][0] > 0:
            plan.append({"name": "input strip continued through the x_lo absorber",
                         "role": "extension", "cols": (0, f["cols"][0]),
                         "rows": f["rows"]})
        f = s["out_feed"]
        if f["cols"][1] < nx - 1:
            plan.append({"name": "output strip continued through the x_hi absorber",
                         "role": "extension", "cols": (f["cols"][1], nx - 1),
                         "rows": f["rows"]})
    return plan


def _box_xy(lat, p):
    xs, ys = lat["xs"], lat["ys"]
    return (xs[p["cols"][0]], xs[p["cols"][1]], ys[p["rows"][0]], ys[p["rows"][1]])


# ------------------------------------------------------------- the builder
def make_builder(maker, lat, line_ends: str):
    """The openEMS model of this rung, in the shared runner's builder shape."""
    xs, ys, zs = lat["xs"], lat["ys"], lat["zs"]
    k = lat["plane_k"]
    z_sheet = float(zs[k])
    plan = metal_plan(lat, line_ends)
    by_role = {p["role"]: p for p in plan if p["role"] in ("port1", "port2")}
    out_len = maker.LX - maker.PATCH_X1

    def build(ContinuousStructure, openEMS, MSLPort, *, nrts, end_criteria):
        kw = {}
        if nrts is not None:
            kw["NrTS"] = nrts
        if end_criteria is not None:
            kw["EndCriteria"] = end_criteria
        fdtd = openEMS(**kw)
        fdtd.SetGaussExcite(maker.F_MAX / 2, maker.F_MAX / 2)
        csx = ContinuousStructure()
        fdtd.SetCSX(csx)
        fdtd.SetBoundaryCond(list(BOUNDARY))
        mesh = csx.GetGrid()
        mesh.SetDeltaUnit(1.0)
        mesh.AddLine("x", xs)
        mesh.AddLine("y", ys)
        mesh.AddLine("z", zs)
        for ax, want in zip("xyz", lat["shape"]):
            got = len(np.asarray(mesh.GetLines(ax)))
            if got != want:
                raise RuntimeError(
                    f"the {ax} mesh holds {got} lines, rfx's lattice {want}: "
                    "CSXCAD merged or added lines, so this is not rfx's lattice")
        sub = csx.AddMaterial("duroid", epsilon=maker.EPS_R)
        sub.AddBox([xs[0], ys[0], 0.0], [xs[-1], ys[-1], float(zs[lat["n_sub"]])])
        pec = csx.AddMetal("PEC")
        p1, p2 = by_role["port1"], by_role["port2"]
        port0 = fdtd.AddMSLPort(
            1, pec, [xs[p1["cols"][0]], ys[p1["rows"][0]], z_sheet],
            [xs[p1["cols"][1]], ys[p1["rows"][1]], 0.0], "x", "z", excite=-1,
            FeedShift=maker.PORT_MARGIN, MeasPlaneShift=0.45 * maker.PATCH_X0,
            priority=10)
        port1 = fdtd.AddMSLPort(
            2, pec, [xs[p2["cols"][1]], ys[p2["rows"][0]], z_sheet],
            [xs[p2["cols"][0]], ys[p2["rows"][1]], 0.0], "x", "z",
            MeasPlaneShift=0.45 * out_len, priority=10)
        for p in plan:
            if p["role"] in ("port1", "port2"):
                continue
            x0, x1, y0, y1 = _box_xy(lat, p)
            pec.AddBox([x0, y0, z_sheet], [x1, y1, z_sheet], priority=10)
        return fdtd, port0, port1

    build.plan = plan
    return build


# ------------------------------------------------------------------- stubs
class _Rec:
    def __init__(self):
        self.boundary = []
        self.lines = {"x": [], "y": [], "z": []}
        self.material_boxes = []
        self.metal_boxes = []
        self.ports = []
        self.kwargs = {}


def _stubs(rec: _Rec):
    """Recording stand-ins for openEMS, ContinuousStructure and the mesh.

    ``AddMSLPort`` records the port AND the metal box MSLPort adds (its own
    rule: start -> stop with stop[exc] = start[exc], ports.py MSLPort)."""

    class _Prim:
        def __init__(self, sink, name):
            self._sink, self._name = sink, name

        def AddBox(self, start, stop, **kw):
            self._sink.append({"prop": self._name, "start": [float(v) for v in start],
                               "stop": [float(v) for v in stop]})

    class _Mesh:
        def SetDeltaUnit(self, unit):
            rec.kwargs["delta_unit"] = unit

        def AddLine(self, ax, vals):
            rec.lines[ax].extend(float(v) for v in np.atleast_1d(vals))

        def GetLines(self, ax):
            return np.unique(np.asarray(rec.lines[ax], dtype=float))

    class _CSX:
        def __init__(self):
            self._mesh = _Mesh()

        def GetGrid(self):
            return self._mesh

        def AddMaterial(self, name, **kw):
            rec.kwargs["material"] = {"name": name, **kw}
            return _Prim(rec.material_boxes, name)

        def AddMetal(self, name):
            return _Prim(rec.metal_boxes, name)

    class _FDTD:
        def __init__(self, **kw):
            rec.kwargs["fdtd"] = kw

        def SetGaussExcite(self, f0, fc):
            rec.kwargs["gauss"] = (f0, fc)

        def SetCSX(self, csx):
            self._csx = csx

        def GetCSX(self):
            return self._csx

        def SetBoundaryCond(self, bc):
            rec.boundary.append(list(bc))

        def AddMSLPort(self, number, metal, start, stop, prop, exc, **kw):
            rec.ports.append({"number": number, "start": [float(v) for v in start],
                              "stop": [float(v) for v in stop], "kw": dict(kw)})
            s = [float(v) for v in start]
            e = [float(v) for v in stop]
            e[2] = s[2]
            metal.AddBox(s, e)
            return object()

    return _CSX, _FDTD


# ----------------------------------------------------------------- control 1
def control_one(maker, n: int, lat, line_ends: str, *, verbose: bool) -> tuple:
    """Rebuild rfx's logged board node for node, then compare metal edge sets.

    Returns (ok, lines, stub_record)."""
    log = LOGGED[n]
    lines, ok = [], True

    def check(cond, what):
        nonlocal ok
        lines.append(f"    [{'ok ' if cond else 'FAIL'}] {what}")
        ok = ok and bool(cond)

    check(tuple(lat["shape"]) == tuple(log["grid"]),
          f"grid shape {tuple(lat['shape'])} vs logged {tuple(log['grid'])} "
          f"(pads {lat['pads']})")
    check(lat["n_sub"] == log["n_sub"],
          f"substrate cells {lat['n_sub']} vs logged {log['n_sub']}")
    check(lat["plane_k"] == log["plane_k"],
          f"sheet plane k={lat['plane_k']} (z={lat['zs'][lat['plane_k']] * 1e6:.4f} um, "
          f"h_sub {maker.H_SUB * 1e6:.4f} um) vs logged k={log['plane_k']}")
    check(float(lat["zs"][lat["plane_k"]]) == float(maker.H_SUB),
          "the sheet plane's node coordinate equals h_sub bit for bit "
          "(openEMS's box test is a closed interval with no tolerance)")
    for name in ("in_feed", "section", "out_feed"):
        got = lat["sheets"][name]
        want_cols, want_rows = log[name]
        check(tuple(got["cols"]) == tuple(want_cols) and tuple(got["rows"]) == tuple(want_rows),
              f"{name:9s} cols {got['cols']} rows {got['rows']} vs logged "
              f"cols {want_cols} rows {want_rows}")

    ex_r, ey_r = rfx_union_edges(lat)
    # Build the model against the recording stubs, so the edge set below is
    # what the builder actually hands openEMS, not a second plan of it.
    rec = _Rec()
    csx_cls, fdtd_cls = _stubs(rec)
    builder = make_builder(maker, lat, line_ends)
    builder(csx_cls, fdtd_cls, None, nrts=None, end_criteria=END_CRITERIA)
    boxes = []
    k_z = float(lat["zs"][lat["plane_k"]])
    flat = True
    for b in rec.metal_boxes:
        flat = flat and (b["start"][2] == k_z == b["stop"][2])
        boxes.append((b["start"][0], b["stop"][0], b["start"][1], b["stop"][1]))
    check(flat, f"all {len(boxes)} metal boxes are zero-thickness at z = "
                f"{k_z * 1e6:.4f} um")
    ex_o, ey_o = openems_edges(boxes, lat)
    ext = [p for p in builder.plan if p["role"] == "extension"]
    ex_x, ey_x = openems_edges([_box_xy(lat, p) for p in ext], lat)
    want_x, want_y = ex_r | ex_x, ey_r | ey_x
    n_diff = int((ex_o ^ want_x).sum() + (ey_o ^ want_y).sum())
    check(n_diff == 0,
          f"openEMS PEC edges == rfx's union-rule edges"
          f"{' + the absorber-continuation edges' if ext else ''}: "
          f"{int(ex_o.sum())} Ex / {int(ey_o.sum())} Ey vs "
          f"{int(ex_r.sum())} / {int(ey_r.sum())} rfx; {n_diff} edge(s) differ")
    # The continuation edges, split by where they sit.
    nx, ny, _ = lat["shape"]
    pd = lat["pads"]
    in_dom_x = np.zeros((nx, ny), dtype=bool)
    in_dom_x[pd["x_lo"]:nx - 1 - pd["x_hi"], pd["y_lo"]:ny - pd["y_hi"]] = True
    in_dom_y = np.zeros((nx, ny), dtype=bool)
    in_dom_y[pd["x_lo"]:nx - pd["x_hi"], pd["y_lo"]:ny - 1 - pd["y_hi"]] = True
    extra_x, extra_y = ex_o & ~ex_r, ey_o & ~ey_r
    n_extra_dom = int((extra_x & in_dom_x).sum() + (extra_y & in_dom_y).sum())
    n_extra_pad = int((extra_x & ~in_dom_x).sum() + (extra_y & ~in_dom_y).sum())
    lines.append(f"    [info] metal edges openEMS has that rfx does not: "
                 f"{n_extra_dom} inside the declared domain, {n_extra_pad} in the "
                 f"absorber pad (line ends: {line_ends})")
    for side in ("in", "out"):
        row, (c0, c1), on, total = log["joints"][side]
        got_on = int(ex_o[c0:c1, row].sum())
        check(got_on == on and (c1 - c0) == total,
              f"{side:3s} joint row {row}, columns ({c0}, {c1}): openEMS "
              f"{got_on}/{c1 - c0} PEC edges vs logged {on}/{total}")
    check(rec.boundary == [BOUNDARY],
          f"SetBoundaryCond calls {rec.boundary}")
    mb = rec.material_boxes[0] if rec.material_boxes else None
    sub_ok = (mb is not None and mb["start"][0] == lat["xs"][0]
              and mb["stop"][0] == lat["xs"][-1] and mb["start"][1] == lat["ys"][0]
              and mb["stop"][1] == lat["ys"][-1] and mb["start"][2] == 0.0
              and mb["stop"][2] == float(lat["zs"][lat["n_sub"]]))
    check(sub_ok, f"substrate box spans every lateral line (pad included) and "
                  f"z 0 -> {lat['zs'][lat['n_sub']] * 1e6:.4f} um = cells k 0..{lat['n_sub'] - 1}")
    got_lines = tuple(len(np.unique(rec.lines[a])) for a in "xyz")
    check(got_lines == tuple(lat["shape"]),
          f"mesh lines handed to openEMS {got_lines} == lattice {tuple(lat['shape'])}")
    return ok, lines, rec, builder.plan


# -------------------------------------------------------------------- cost
def dt_estimate(dx: float) -> float:
    """openEMS's default timestep (Rennings_2, FDTD/operator.cpp:1942) on a
    uniform cubic vacuum cell reduces to the Courant limit dx / (c sqrt 3)."""
    return dx / (C0 * math.sqrt(3.0))


def cost_table(lat) -> list:
    nx, ny, nz = lat["shape"]
    cells = (nx - 1) * (ny - 1) * (nz - 1)
    dt = dt_estimate(lat["dx"])
    rows = []
    for dname, dns in DECAY_NS.items():
        steps = int(math.ceil(dns * 1e-9 / dt))
        for tname, rate in THROUGHPUT_CELL_STEPS_PER_S.items():
            rows.append((dname, dns, steps, tname, rate, cells * steps / rate))
    return cells, dt, rows


# ---------------------------------------------------------------- printing
def print_rung(maker, n, lat, line_ends, ok, lines, rec, plan, *, full: bool):
    nx, ny, nz = lat["shape"]
    print(f"\n=== rung h/{n}: dx = {lat['dx'] * 1e6:.4f} um ===")
    if full:
        xs, ys, zs = lat["xs"], lat["ys"], lat["zs"]
        pd = lat["pads"]
        print(f"  lattice   {nx} x {ny} x {nz} node lines, pads {pd}")
        print(f"  x lines   {xs[0] * 1e3:.4f} .. {xs[-1] * 1e3:.4f} mm; declared "
              f"domain face x = {xs[pd['x_lo']] * 1e3:.4f} / "
              f"{xs[nx - 1 - pd['x_hi']] * 1e3:.4f} mm (LX {maker.LX * 1e3:.3f})")
        print(f"  y lines   {ys[0] * 1e3:.4f} .. {ys[-1] * 1e3:.4f} mm; declared "
              f"domain face y = {ys[pd['y_lo']] * 1e3:.4f} / "
              f"{ys[ny - 1 - pd['y_hi']] * 1e3:.4f} mm (LY {maker.LY * 1e3:.3f})")
        print(f"  z lines   {zs[0] * 1e3:.4f} .. {zs[-1] * 1e3:.4f} mm; top domain "
              f"face z = {zs[nz - 1 - pd['z_hi']] * 1e3:.4f} mm (LZ {maker.LZ * 1e3:.3f})")
        print(f"  boundary  {BOUNDARY}: each PML_8 covers mesh lines 0..8 / "
              f"N-9..N-1 (openEMS FDTD/extensions/operator_ext_upml.cpp "
              f"Create_UPML), i.e. rfx's 8 pad cells; its inner face is the "
              f"declared domain face above")
        print(f"  substrate eps_r {maker.EPS_R} in cells k = 0..{lat['n_sub'] - 1} "
              f"over the full lateral extent, pad included (rfx copies the edge "
              f"slice into its pad); air above")
        print("  metal on z = h_sub, node indices (cols = x, rows = y):")
        for p in plan:
            x0, x1, y0, y1 = _box_xy(lat, p)
            print(f"    {p['name']:50s} cols {p['cols']} rows {p['rows']}  "
                  f"x {x0 * 1e3:.4f}..{x1 * 1e3:.4f} mm, y {y0 * 1e3:.4f}..{y1 * 1e3:.4f} mm")
        for prt in rec.ports:
            st, sp = prt["start"], prt["stop"]
            direction = 1.0 if sp[0] > st[0] else -1.0
            feed = prt["kw"].get("FeedShift")
            meas = prt["kw"].get("MeasPlaneShift")
            mline = xs[int(np.argmin(np.abs(xs - (st[0] + direction * meas))))]
            txt = (f"    port {prt['number']}: start x {st[0] * 1e3:.4f} mm, "
                   f"{'+x' if direction > 0 else '-x'}; MeasPlaneShift "
                   f"{meas * 1e3:.4f} mm -> nearest line {mline * 1e3:.4f} mm")
            if feed is not None:
                fline = xs[int(np.argmin(np.abs(xs - (st[0] + direction * feed))))]
                txt += (f"; FeedShift {feed * 1e3:.4f} mm -> nearest line "
                        f"{fline * 1e3:.4f} mm; excite {prt['kw'].get('excite')}")
            print(txt)
    print(f"  control 1 -- rfx's realized board, node for node "
          f"({'PASS' if ok else 'FAIL'}):")
    for line in lines:
        print(line)
    if full:
        cells, dt, rows = cost_table(lat)
        print(f"  cost      {cells} openEMS cells, dt ~ {dt:.4e} s (dx / c sqrt3)")
        for dname, dns, steps, tname, rate, secs in rows:
            print(f"    decay {dname:12s} {dns:5.2f} ns = {steps:7d} steps at "
                  f"{tname:30s} {rate:.2e} cell-steps/s -> {secs:8.0f} s "
                  f"({secs / 3600:.2f} h)")


# ------------------------------------------------------------- the solve
def zeros_of(sf, freqs_ghz, s21_mag) -> dict:
    out = {}
    f = np.asarray(freqs_ghz, dtype=float)
    s = np.asarray(s21_mag, dtype=float)
    for key, (lo, hi) in ZERO_BANDS_GHZ.items():
        try:
            e = sf.refined_extremum(f, s, lo, hi, transform="log")
            idx = int(e["index"])
            band = np.flatnonzero((f >= lo) & (f <= hi))
            out[key] = {"f_ghz": float(e["refined_f"]), "bin_f_ghz": float(e["bin_f"]),
                        "depth_db": float(e["depth_db"]),
                        "at_window_edge": bool(band.size and idx in (int(band[0]), int(band[-1])))}
        except Exception as exc:
            out[key] = {"error": repr(exc)}
    return out


def run_rung(maker, sf, n, lat, *, line_ends, sim_root, threads, nrts):
    gate = maker._gate
    builder = make_builder(maker, lat, line_ends)
    label = f"identical_h{n}"
    xs_um = lat["xs"] * 1e6

    def mesh_realized_fn(lines):
        return gate._mesh_realized(gate.lines_in_um(lines, 1.0),
                                   substrate_thickness_um=maker.H_SUB * 1e6)

    def meta_extra_fn(*, lines, port0, port1) -> dict:
        out = {"rung_n": n, "dx_um": lat["dx"] * 1e6,
               "lattice_shape": list(lat["shape"]), "pads": lat["pads"],
               "boundary": BOUNDARY, "line_ends": line_ends,
               "metal_node_indices": [{k: v for k, v in p.items()} for p in builder.plan],
               "rfx_logged_board": LOGGED[n]}
        if lines is not None:
            out["lines_equal_lattice"] = {
                ax: bool(np.asarray(lines[ax]).size == arr.size
                         and np.array_equal(np.asarray(lines[ax], dtype=float), arr))
                for ax, arr in zip("xyz", (lat["xs"], lat["ys"], lat["zs"]))}
        p1 = builder.plan[0]
        p2 = builder.plan[2]
        out["port0"] = gate._port_declared_and_snap(
            xs_um, start_x_um=lat["xs"][p1["cols"][0]] * 1e6, direction=+1.0,
            feed_shift_um=maker.PORT_MARGIN * 1e6,
            measplane_shift_um=0.45 * maker.PATCH_X0 * 1e6, port_obj=port0,
            csx_unit_m=1.0)
        out["port1"] = gate._port_declared_and_snap(
            xs_um, start_x_um=lat["xs"][p2["cols"][1]] * 1e6, direction=-1.0,
            feed_shift_um=0.0,
            measplane_shift_um=0.45 * (maker.LX - maker.PATCH_X1) * 1e6,
            port_obj=port1, csx_unit_m=1.0)
        try:
            f = np.linspace(maker.F_LO, maker.F_MAX, maker.B_N_FREQS) / 1e9
            a1 = np.abs(np.asarray(port0.uf_inc, dtype=np.complex128))
            a2 = np.abs(np.asarray(port1.uf_inc, dtype=np.complex128))
            ratio = a2 / np.maximum(a1, 1e-300)
            band = (f >= 2.0) & (f <= 12.0)
            out["passive_port_incident_ratio"] = {
                "what": "|a2/a1|: the wave arriving at port 2 from its own side, "
                        "over the driven port's incident wave, 2-12 GHz",
                "max": float(ratio[band].max()), "median": float(np.median(ratio[band])),
                "at_max_ghz": float(f[band][int(np.argmax(ratio[band]))])}
        except Exception as exc:
            out["passive_port_incident_ratio"] = {"error": repr(exc)}
        return out

    t0 = time.time()
    record, meta = gate.run_stage(
        label=label, sim_root=sim_root, threads=threads, build=builder,
        freqs_hz=np.linspace(maker.F_LO, maker.F_MAX, maker.B_N_FREQS),
        witness_band_hz=maker.WITNESS_BAND_HZ, passivity_tol=maker.PASSIVITY_TOL,
        real_nrts=nrts, real_end_criteria=END_CRITERIA,
        mesh_realized_fn=mesh_realized_fn, meta_extra_fn=meta_extra_fn,
        features_fn=maker._stage_b_features(sf),
        calcport_ref_impedance=maker.B_CALCPORT_REF_IMPEDANCE,
        record_deficit=True, accept_truncation=False)
    wall = time.time() - t0
    record["zeros"] = zeros_of(sf, record["freqs_ghz"], record["s21_mag"])
    return record, meta, wall


def write_figure(results, frozen, directory: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  figure NOT written: {exc!r}")
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sheen_identical_grid_openems.png"
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for n, rec in results:
        ax.plot(rec["freqs_ghz"], 20 * np.log10(np.maximum(rec["s21_mag"], 1e-300)),
                lw=1.3, label=f"openEMS on rfx's h/{n} lattice")
    if frozen is not None and "stage_b_fine" in frozen:
        st = frozen["stage_b_fine"]
        ax.plot(st["freqs_ghz"], 20 * np.log10(np.maximum(st["s21_mag"], 1e-300)),
                "k-", lw=1.0, label="openEMS frozen record stage_b_fine (99.25 um)")
    ax.set_xlim(2, 12)
    ax.set_ylim(-70, 5)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("|S21| (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------------------ against rfx
def against_rfx(maker, rungs) -> int:
    """Hold this lattice against rfx's own arrays (needs rfx + jax)."""
    root = repo_root()
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "_sheen_identical_grid_case",
        root / "tests" / "crossval" / "sheen_lpf" / "test_sheen_lpf.py")
    case = importlib.util.module_from_spec(spec)
    sys.modules["_sheen_identical_grid_case"] = case
    spec.loader.exec_module(case)
    import warnings
    from tests._realized_geometry import _node_line, realized
    rc = 0
    for n in rungs:
        lat = build_lattice(maker, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = case.build(lat["dx"])
            rz = realized(sim)
            mats = sim._assemble_materials(rz.grid, pec_sheets=[], pec_wires=[])[0]
        k = lat["plane_k"]
        mx, my, mz = (np.asarray(e) for e in rz.edge_masks)
        ex_r, ey_r = rfx_union_edges(lat)
        lines_ok = all(np.array_equal(_node_line(rz.grid, a), arr)
                       for a, arr in enumerate((lat["xs"], lat["ys"], lat["zs"])))
        edges_ok = (np.array_equal(mx[:, :, k], ex_r) and np.array_equal(my[:, :, k], ey_r)
                    and int(mx.sum()) == int(mx[:, :, k].sum())
                    and int(my.sum()) == int(my[:, :, k].sum()) and int(mz.sum()) == 0)
        eps = np.asarray(mats.eps_r)
        eps_ok = (np.allclose(eps[:, :, :lat["n_sub"]], maker.EPS_R)
                  and np.allclose(eps[:, :, lat["n_sub"]:], 1.0))
        print(f"  h/{n}: node lines equal rfx's {lines_ok}; PEC edges equal rfx's "
              f"realized masks {edges_ok} ({int(mx.sum())} Ex / {int(my.sum())} Ey, "
              f"Ez {int(mz.sum())}); rfx eps_r = {maker.EPS_R} at every node k < "
              f"{lat['n_sub']} and 1.0 at every node k >= {lat['n_sub']}, pads "
              f"included: {eps_ok}")
        rc = rc or (0 if (lines_ok and edges_ok and eps_ok) else 1)
    return rc


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rungs", default=DEFAULT_RUNGS,
                   help=f"comma separated n of dx = h_sub/n, from {RUNGS}")
    p.add_argument("--out", default=None, help="directory for the JSON and figure")
    p.add_argument("--sim-root", default="/tmp/sheen_identical_grid_openems")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--nrts", type=int, default=DEFAULT_NRTS,
                   help="openEMS NrTS: a ceiling; every pass must end on its "
                        "1e-4 energy criterion before it")
    p.add_argument("--line-ends", choices=("pml", "rfx"), default="pml",
                   help="pml: continue both strips through the absorber cells "
                        "(default); rfx: rfx's metal exactly, both lines end "
                        "open at the absorber")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--against-rfx", action="store_true")
    p.add_argument("--mutate-shift-section", action="store_true",
                   help="MUTATION: the wide section one node along +x in the "
                        "lattice builder; control 1 must refuse it")
    p.add_argument("--mutate-no-joints", action="store_true",
                   help="MUTATION: no joint boxes, so each feed abuts the "
                        "section as a separate box; the edge-set check must "
                        "refuse it")
    p.add_argument("--mutate-no-y-pad", action="store_true",
                   help="MUTATION: no absorber pad on the y faces; the "
                        "grid-shape check must refuse it")
    args = p.parse_args(argv)

    try:
        rungs = [int(v) for v in args.rungs.split(",") if v.strip()]
    except ValueError:
        print(f"ERROR: --rungs {args.rungs!r} is not a list of integers", file=sys.stderr)
        return 3
    bad = [n for n in rungs if n not in RUNGS]
    if bad:
        print(f"ERROR: rung(s) {bad} not in {RUNGS} (no logged board to hold them "
              "against)", file=sys.stderr)
        return 3
    mutating = (args.mutate_shift_section or args.mutate_no_y_pad
                or args.mutate_no_joints)
    if mutating and not (args.dry_run or args.self_check):
        print("ERROR: the mutation flags are dry-run / self-check checks", file=sys.stderr)
        return 3

    maker = load_maker()
    if args.against_rfx:
        return against_rfx(maker, rungs)

    def lattice(n):
        lat = build_lattice(maker, n, shift_section=1 if args.mutate_shift_section else 0,
                            y_pad=not args.mutate_no_y_pad)
        lat["drop_joints"] = bool(args.mutate_no_joints)
        return lat

    if args.dry_run or args.self_check:
        full = args.dry_run
        print("=" * 78)
        print("The Sheen low-pass filter -- openEMS on rfx's lattice, "
              + ("DRY RUN (numpy only, no openEMS)" if full else "SELF-CHECK (control 1)"))
        print("=" * 78)
        if args.mutate_shift_section:
            print("MUTATION: the wide section is shifted one node along +x.")
        if args.mutate_no_y_pad:
            print("MUTATION: the y faces get no absorber pad.")
        if args.mutate_no_joints:
            print("MUTATION: no joint boxes -- each feed abuts the section.")
        if full:
            print(f"  line ends {args.line_ends}; end criterion {END_CRITERIA:g} "
                  f"(-40 dB), NrTS ceiling {args.nrts}, truncation NOT accepted; "
                  f"frequency grid linspace({maker.F_LO:g}, {maker.F_MAX:g}, "
                  f"{maker.B_N_FREQS}); zeros in {ZERO_BANDS_GHZ} GHz")
            print("  STAGE A is not run: the reproduce gate passed in VESSL "
                  "369367263406/407/408, which made the frozen record.")
        all_ok = True
        for n in rungs:
            lat = lattice(n)
            try:
                ok, lines, rec, plan = control_one(maker, n, lat, args.line_ends,
                                                   verbose=full)
            except Exception as exc:  # a mutated lattice can break the builder
                ok, lines, rec, plan = False, [f"    [FAIL] {exc!r}"], _Rec(), []
            print_rung(maker, n, lat, args.line_ends, ok, lines, rec, plan,
                       full=full and ok)
            all_ok = all_ok and ok
        print(f"\ncontrol 1 over rungs {rungs}: {'PASS' if all_ok else 'FAIL'}")
        return 0 if all_ok else 1

    # --------------------------------------------------------- the solve
    try:
        sf = maker._gate.load_spectral_features()
    except Exception as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 3
    try:
        maker._gate._import_openems()
    except Exception as exc:
        print(f"openEMS IS NOT IMPORTABLE: {exc!r}", file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else Path("sheen_identical_grid")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("The Sheen low-pass filter -- openEMS on rfx's lattice")
    print("=" * 78)
    print(f"  openEMS build {os.environ.get('RFX_OPENEMS_COMMIT', '(unstamped)')}; "
          f"image {os.environ.get('RFX_OPENEMS_IMAGE', '(unset)')}")
    print(f"  rungs {rungs}; line ends {args.line_ends}; end criterion "
          f"{END_CRITERIA:g}; NrTS ceiling {args.nrts}; STAGE A NOT RUN")

    frozen = None
    ref_path = (repo_root() / "tests" / "crossval" / "sheen_lpf" / "reference"
                / "openems_sheen.json")
    if ref_path.is_file():
        with ref_path.open() as fh:
            frozen = json.load(fh)

    results, metas, walls, failures = [], {}, {}, {}
    rc = 0
    for n in rungs:
        lat = lattice(n)
        ok, lines, rec, plan = control_one(maker, n, lat, args.line_ends, verbose=False)
        print_rung(maker, n, lat, args.line_ends, ok, lines, rec, plan, full=False)
        if not ok:
            print(f"  CONTROL 1 FAILED at h/{n}: not solved, and the rung is not read.")
            failures[n] = "control 1"
            rc = 1
            continue
        try:
            record, meta, wall = run_rung(maker, sf, n, lat, line_ends=args.line_ends,
                                          sim_root=args.sim_root,
                                          threads=args.threads, nrts=args.nrts)
        except maker._gate.StageFailure as exc:
            print(f"  GATE FIRED at h/{n}: {exc}", flush=True)
            failures[n] = str(exc)
            metas[n] = {"failed": str(exc), "partial_meta": exc.meta}
            rc = 1
            continue
        results.append((n, record))
        metas[n] = meta
        walls[n] = wall
        z = record["zeros"]
        print(f"  h/{n}: box energy {meta.get('final_energy_db')} dB at step "
              f"{meta.get('final_timestep')} (end criterion reached: "
              f"{meta.get('end_criteria_reached')}); dt {meta.get('dt_s')} s; "
              f"zeros {z.get('zero_7', {}).get('f_ghz')} / "
              f"{z.get('zero_8', {}).get('f_ghz')} GHz; corner "
              f"{(record.get('cutoff_3db') or {}).get('f_ghz')} GHz; "
              f"|a2/a1| max {(meta.get('passive_port_incident_ratio') or {}).get('max')}; "
              f"wall {wall:.1f} s", flush=True)

    frozen_feat = {}
    if frozen is not None:
        for st in FROZEN_STAGES:
            if st in frozen:
                frozen_feat[st] = {
                    "zeros": zeros_of(sf, frozen[st]["freqs_ghz"], frozen[st]["s21_mag"]),
                    "corner": frozen[st]["cutoff_3db"]["f_ghz"],
                    "resolution_um": frozen["meta"]["stages"][st]["resolution_um"]}

    print("\n" + "=" * 78)
    print("READING 1 -- openEMS on rfx's lattice against rfx's ladder (same rung)")
    print("=" * 78)
    print("  rung   oE zero7   oE zero8   rfx zero8   d8 (%)   oE corner   rfx corner"
          "   dcorner (%)  energy (dB)  steps")
    table = {}
    for n, rec in results:
        z7 = rec["zeros"].get("zero_7", {}).get("f_ghz")
        z8 = rec["zeros"].get("zero_8", {}).get("f_ghz")
        corner = (rec.get("cutoff_3db") or {}).get("f_ghz")
        r = RFX_LADDER[n]
        d8 = None if z8 is None else 100.0 * (z8 - r["zero_8"]) / r["zero_8"]
        dc = None if corner is None else 100.0 * (corner - r["corner"]) / r["corner"]
        table[n] = {"openems_zero_7_ghz": z7, "openems_zero_8_ghz": z8,
                    "rfx_zero_7_ghz": None, "rfx_zero_8_ghz": r["zero_8"],
                    "zero_8_openems_minus_rfx_pct": d8,
                    "openems_corner_ghz": corner, "rfx_corner_ghz": r["corner"],
                    "corner_openems_minus_rfx_pct": dc,
                    "final_energy_db": metas[n].get("final_energy_db"),
                    "final_timestep": metas[n].get("final_timestep")}
        print(f"  h/{n:<3d} {z7 if z7 is None else f'{z7:9.5f}'}  "
              f"{z8 if z8 is None else f'{z8:9.5f}'}  {r['zero_8']:9.5f}  "
              f"{'' if d8 is None else f'{d8:+7.3f}'}   "
              f"{corner if corner is None else f'{corner:9.5f}'}   {r['corner']:9.5f}"
              f"   {'' if dc is None else f'{dc:+8.3f}'}     "
              f"{metas[n].get('final_energy_db')}  {metas[n].get('final_timestep')}")

    print("\n" + "=" * 78)
    print("READING 2 -- against openEMS's own frozen record (its smoothed meshes)")
    print("=" * 78)
    def _g(v, fmt):
        return "n/a" if v is None else format(v, fmt)

    for st, ff in frozen_feat.items():
        print(f"  {st:15s} ({ff['resolution_um']:.2f} um): zero7 "
              f"{_g(ff['zeros'].get('zero_7', {}).get('f_ghz'), '.5f')}  zero8 "
              f"{_g(ff['zeros'].get('zero_8', {}).get('f_ghz'), '.5f')}  corner "
              f"{_g(ff['corner'], '.5f')} GHz")
    vs_frozen = {}
    for n, rec in results:
        vs_frozen[n] = {}
        for st, ff in frozen_feat.items():
            row = {}
            for key in ("zero_7", "zero_8"):
                a = rec["zeros"].get(key, {}).get("f_ghz")
                b = ff["zeros"].get(key, {}).get("f_ghz")
                row[key + "_pct"] = (None if a is None or b is None
                                     else 100.0 * (a - b) / b)
            a = (rec.get("cutoff_3db") or {}).get("f_ghz")
            row["corner_pct"] = None if a is None else 100.0 * (a - ff["corner"]) / ff["corner"]
            vs_frozen[n][st] = row
            print(f"  h/{n} vs {st:15s}: zero7 {_g(row['zero_7_pct'], '+.3f')} %  "
                  f"zero8 {_g(row['zero_8_pct'], '+.3f')} %  corner "
                  f"{_g(row['corner_pct'], '+.3f')} %")

    fig = write_figure(results, frozen, out_dir)
    if fig is not None:
        print(f"\n  figure: {fig}")
    payload = {
        "what": "openEMS on rfx's own lattice for the Sheen low-pass filter -- "
                "a diagnostic, not a reference record; Stage A not run",
        "line_ends": args.line_ends, "end_criteria": END_CRITERIA,
        "nrts_ceiling": args.nrts, "boundary": BOUNDARY,
        "openems_commit": os.environ.get("RFX_OPENEMS_COMMIT"),
        "openems_image": os.environ.get("RFX_OPENEMS_IMAGE"),
        "rfx_ladder_source": "VESSL 369367263494, research/rfx/.omx/sheen-lpf-ladder/"
                             "20260922T160810Z-e2ba011e/ladder.log",
        "rfx_ladder": RFX_LADDER, "rfx_logged_boards": LOGGED,
        "table_vs_rfx": table, "vs_frozen": vs_frozen, "frozen_features": frozen_feat,
        "failures": failures, "wall_s": walls,
        "rungs": {str(n): {"record": rec, "meta": metas[n]} for n, rec in results},
        "figure": None if fig is None else str(fig),
    }
    path = out_dir / "sheen_identical_grid_openems.json"
    with path.open("w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print(f"  json:   {path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
