"""Phase-2 two-sided fixture — one geometry pathway for every arm.

WHY THIS FILE EXISTS (PI decisions, 2026-08-27)
-----------------------------------------------
1. **The design box is TWO-SIDED.** Metal may be placed on BOTH sides of the
   through-line, not just the +y side the Stage-0 script used. This applies
   identically to every arm — classical two-stub baseline and topology
   optimization alike. It is a RELAXATION: we are no longer trying to make
   the classical baseline fail, we are measuring how much margin each
   approach has.
2. **Tolerance robustness is part of the spec.** Ordinary PCB etching
   tolerance is about +-50 um. The coarse cell here is 127 um and the
   refined cell is 63.5 um, so the nearest representable perturbation is
   +-1 fine cell = +-63.5 um, which is slightly CONSERVATIVE (it over-states
   the etch error by 27 %). ``etch_fields`` below emits the
   eroded / nominal / dilated triplet, and it is a DISPATCHER, not an
   implementation: the morphology lives in ``robust_eval`` and there is
   exactly one copy of it in the codebase. This file's job is to tell
   ``robust_eval`` which array edge of which SIDE abuts the through-line
   (``BoxSide.etch_outside``), because that is fixture knowledge and getting
   it wrong detaches every stub on one side from its feed.
3. **The phase is a MARGIN COMPARISON, not a feasibility gate.** The old
   DEAD / CONTESTED / LIVE verdict is gone. Every arm reports M and Omega
   from ``score_dualband`` nominally and under the etch bracket, and the
   deliverable is the margin table.

WHAT IS MIRRORED, AND WHAT IS NOT
---------------------------------
``build_sim`` reproduces ``msl_stub_notch_tuning.build_sim`` exactly in
every respect that touches the validated physics — substrate (RO4350B,
eps_r 3.66, h 254 um), dx = 127 um, W_TRACE = 600 um, L_LINE = 30 mm,
PORT_MARGIN, F_MAX = 9 GHz, the boundary spec (x/y CPML, z lo PEC / hi
CPML, 8 CPML layers), both MSL ports at 50 ohm, and both plane-probe sets
registered through ``register_msl_plane_probes``.

The ONE deliberate change is the trace's y position and hence ``LY``:

    original      LY = W + msl_clearance + L_STUB_MAX + 2*(2 h + 8 dx)
                     = 20.696 mm, trace at y = 1.824 mm
                  -> 14 mm of design depth on +y ONLY, 1*CLEAR to the -y wall

    this file     LY = EDGE_MARGIN + DEPTH + W + DEPTH + EDGE_MARGIN
                     = 27.559 mm, trace at y = 13.762 mm
                  -> DESIGN_DEPTH = 9 mm on BOTH sides, symmetric

``EDGE_MARGIN`` is set to ``3*CLEAR`` where ``CLEAR = 2*h_sub + 8*dx`` is the
quantity the original's own LY formula is built from. Two facts fix that
choice rather than taste:

  * preflight's lateral-clearance rule needs ``trace_edge - 8*dx >= 2*h_sub``,
    i.e. exactly ``1*CLEAR`` from trace edge to wall. The original sits AT
    that bound on its -y side (trace_y_lo = 1.524 mm = 1*CLEAR).
  * the original leaves ``3*CLEAR = 4.572 mm`` between the FURTHEST design
    metal it admits (a stub of L_STUB_MAX = 14 mm) and the +y wall. Design
    metal, unlike a bare trace, can be placed by the optimizer right at the
    box edge, so the margin that must be preserved on BOTH sides is the
    design-metal margin, not the bare-trace one.

So the new fixture is strictly MORE conservative than the original on the
side that used to be tight, and identical on the side that was already
generous. LY grows 20.696 -> 27.559 mm (+30 % cells: the grid goes
(279, 180, 19) = 954 180 -> (279, 234, 19) = 1 240 434). dt is unchanged at
0.242 ps -- the CFL step depends on dx, not on LY -- so cost scales with the
cell count alone, and the Stage-0 measured windows (45-period descent,
90-period verification) carry over with a ~1.3x wall multiplier.

CELL BOOKKEEPING
----------------
``DESIGN_DEPTH = 9 mm`` is 70.87 coarse cells, and ``BOX_X = 12 mm`` is 94.49.
Both are FLOORED, never rounded, so no design cell can ever stick out of the
bounded box:

    NX_BOX  = 94 cells = 11.938 mm  (<= 12 mm)
    NY_SIDE = 70 cells =  8.890 mm  (<=  9 mm)  per side
    -> 2 * 94 * 70 = 13 160 binary variables

(The PLAN's "94 x 71" was written for the one-sided box and rounded up; the
floor is used here because "all metal inside a bounded design box" is the
headline claim of the benchmark and 71 cells would overrun it by 17 um.)

The whole y stack is laid out in WHOLE CELLS, which is what keeps the trace on
the same lattice the original puts it on:

    36  |  70  |  5  |  70  |  36   = 217 cells = LY = 27.559 mm
   edge   -y box  trace  +y box   edge

``TRACE_Y_LO = 106*dx`` is exactly on-lattice, as the original's
``trace_y_lo = 12*dx`` is, and ``TRACE_Y_HI = 106*dx + 600 um`` has the SAME
fractional lattice offset (0.724 cells) the original's ``16.724*dx`` has.
That is not cosmetic: rfx's ``off_lattice_design_edges`` check measures
face-to-nearest-node residuals, and a first cut of this file that derived the
layout from continuous coordinates moved the trace's worst residual from
35 um to 52 um -- a different realized trace, quietly, under the same name.
The 5-cell rasterized trace width is identical to the original's.

The two design sides are anchored on the RASTERIZED trace cell block, not on
the continuous trace edges, so both sides get exactly ``NY_SIDE`` cells and the
design region is contiguous with the trace on both sides.

MESH REFINEMENT (``build_sim(freqs, dx=...)``, ``Mesh``, ``mesh()``)
--------------------------------------------------------------------
``robust_eval`` is explicit that the COARSE mesh cannot express the +-50 um
etch tolerance -- its only representable offset is +-127 um = 2.54x the spec --
so the quotable robustness number has to be computed at dx/2 = 63.5 um, where
+-1 cell is 1.27x. ``build_sim`` therefore takes ``dx``, and every cell count
above is re-derived by ``mesh(dx)`` rather than hardcoded.

What makes a dx/2 number COMPARABLE to a dx number is that both meshes describe
ONE physical structure. The layout is therefore frozen as PHYSICAL PLANES
(``X_BOX_LO_M`` ... ``Y_BOX_HI_M``, ``T_METAL``), every mesh derives its indices
from those, and ``assert_same_physical_bounds`` refuses two meshes whose design
boxes differ by so much as a nanometre. Three consequences, all deliberate:

  * the design box is anchored on the frozen planes, NOT on a re-rounded
    half-width and not on the rasterized trace span. At the coarse mesh the
    frozen planes ARE the rasterized trace block, so nothing changes; deriving
    the x block from ``round(x_mid/dx - nx_box/2)`` at dx/2 instead lands it one
    fine cell low, i.e. a different rectangle under the same name.
  * for ``dx < DX`` the trace Box's upper face is snapped to the coarse REALIZED
    plane (14.097 mm, the top of cell 111) instead of the nominal 600 um edge.
    Without the snap, the same 600 um Box rasterizes to 9 fine cells and stops
    at 14.033 mm, leaving a 63.5 um seam of bare dielectric between the trace
    and the design box -- every design DETACHED FROM ITS FEED at the fine mesh.
    With it, both meshes carry the identical 635 um conductor.
  * the conductor is ``T_METAL = DX`` thick at every mesh (``BoxSide.nz`` cells),
    because a one-cell sheet would otherwise halve in thickness under
    refinement -- a different structure, not a better-resolved one.

``dx == DX`` is byte-identical to this file before ``dx`` existed; the smoke's
preflight gate against the validated one-sided fixture still passes unchanged.
``phase2_calibrate.refine_mask`` maps a coarse design onto the fine mesh by 2x2
cell replication, which is exact because every count above doubles.

ONE GEOMETRY PATHWAY
--------------------
``mask_from_stubs`` -> mask -> ``boxes_from_mask`` -> ``Box(material="pec")``
is the ONLY route geometry reaches the solver, for the classical arm and the
TO arm alike. The classical baseline is therefore rasterized by the same code
that rasterizes an optimized design, which is the failure the Phase-1
retraction came from (an un-calibrated baseline on a different pathway).
``boxes_from_mask`` uses the same run-length merge and the same cell-edge
convention as ``xval1_imperative.mask_to_boxes``.

Run the smoke (CPU, no solve):
    JAX_PLATFORMS=cpu PYTHONPATH=<repo root> python research/metal_to/phase2_fixture.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "validation" / "tmtt_paper"))
sys.path.insert(0, str(HERE))          # robust_eval + score_dualband

import msl_stub_notch_tuning as notch  # noqa: E402
import robust_eval  # noqa: E402  -- the ONLY etch morphology in the codebase
from rfx import Simulation, Box  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.probes.msl_wave_decomp import register_msl_plane_probes  # noqa: E402

C0 = 2.998e8

# ---------------------------------------------------------------------------
# Constants inherited VERBATIM from the validated fixture. Imported rather
# than re-typed so a change upstream cannot silently desynchronize this file.
# ---------------------------------------------------------------------------
EPS_R = notch.EPS_R                 # 3.66
H_SUB = notch.H_SUB                 # 254 um
W_TRACE = notch.W_TRACE             # 600 um
DX = notch.DX                       # 127 um = h_sub / 2
L_LINE = notch.L_LINE               # 30 mm
PORT_MARGIN = notch.PORT_MARGIN     # 1.6 mm
F_MAX = notch.F_MAX                 # 9 GHz
EPS_EFF = notch.EPS_EFF             # 2.869 (Hammerstad)
CPML_LAYERS = 8

# The clearance quantum the original's LY formula is built from.
CLEAR = 2 * H_SUB + 8 * DX          # 1.524 mm

# ---------------------------------------------------------------------------
# Two-sided design box
# ---------------------------------------------------------------------------
BOX_X = 12.0e-3                     # along the line, centred on LX/2
DESIGN_DEPTH = 9.0e-3               # transverse, PER SIDE

NX_BOX = int(np.floor(BOX_X / DX + 1e-9))          # 94 cells, 11.938 mm
NY_SIDE = int(np.floor(DESIGN_DEPTH / DX + 1e-9))  # 70 cells,  8.890 mm

SIDES = ("lo", "hi")                # -y side, +y side. Never a bare default.

# ---------------------------------------------------------------------------
# Domain. LX and LZ are the original's; only LY moves, and it moves in whole
# cells so the trace lands on the same lattice the original puts it on.
# ---------------------------------------------------------------------------
EDGE_MARGIN_CELLS = int(round(3.0 * CLEAR / DX))                # 36 cells
EDGE_MARGIN = EDGE_MARGIN_CELLS * DX                            # 4.572 mm
# Cells a W_TRACE-wide Box rasterizes to under cell-centre containment when
# its lower face is on-lattice: floor(W/dx - 0.5) + 1 = 5 for 600 um / 127 um.
N_TRACE_CELLS = int(np.floor(W_TRACE / DX - 0.5 + 1e-9)) + 1    # 5 cells

LX = L_LINE + 2 * PORT_MARGIN                                   # 33.2 mm
LZ = H_SUB + 1.0e-3
P_BOX_LO = EDGE_MARGIN_CELLS                                    # cell 36
P_TRACE_LO = P_BOX_LO + NY_SIDE                                 # cell 106
P_BOX_HI = P_TRACE_LO + N_TRACE_CELLS                           # cell 111
NY_CELLS = P_BOX_HI + NY_SIDE + EDGE_MARGIN_CELLS               # 217 cells

TRACE_Y_LO = P_TRACE_LO * DX                                    # 13.462 mm
TRACE_Y_HI = TRACE_Y_LO + W_TRACE                               # 14.062 mm
Y_TRACE = 0.5 * (TRACE_Y_LO + TRACE_Y_HI)                       # 13.762 mm
LY = NY_CELLS * DX                                              # 27.559 mm

# Etch bracket. +-1 coarse cell = +-127 um (pessimistic); the fine grid's
# +-1 cell = +-63.5 um is the nearest representable to the +-50 um PCB spec
# and is what the dx/2 re-evaluation must use.
ETCH_SPEC_M = 50e-6
ETCH_COARSE_M = DX                  # 127 um
ETCH_FINE_M = DX / 2.0              # 63.5 um

# ---------------------------------------------------------------------------
# 0b. The layout as PHYSICAL planes, so a refined mesh can reproduce it
#     (gap G-B)
# ---------------------------------------------------------------------------
# Everything above states the layout in WHOLE COARSE CELLS. Those counts are a
# statement about PHYSICAL planes; the planes, not the counts, are what a
# refined mesh has to reproduce. A dx/2 robustness number is only comparable to
# a dx number if the design box occupies the SAME physical rectangle at both
# meshes, so the planes are frozen here once, at the coarse mesh, and every
# other mesh derives its cell indices from them (``Mesh`` below).
#
# Two consequences, both deliberate and both measured by the smoke:
#
#  * THE REFINED MESH RESOLVES THE COARSE *REALIZED* TRACE, NOT THE NOMINAL
#    DRAWING. The 600 um trace Box rasterizes to 5 coarse cells = 635 um,
#    spanning [13.462, 14.097] mm. At dx/2 the same 600 um Box would rasterize
#    to 9 cells = 571.5 um and stop at 14.033 mm, leaving a 63.5 um seam of
#    bare dielectric between the trace and the design box -- every mask design
#    would be DETACHED FROM THE FEED at the fine mesh. So for dx < DX the trace
#    Box's upper face is snapped to the coarse realized plane Y_TRACE_HI_M, and
#    both meshes then realize the identical 635 um conductor. At dx == DX
#    nothing changes: the snap is a no-op on the rasterization and the code
#    path is byte-identical to the validated fixture.
#  * THE CONDUCTOR IS T_METAL THICK AT EVERY MESH (2 cells at dx/2, 1 at dx),
#    for the same reason: a one-cell-thick sheet would halve in thickness under
#    refinement, which is a different structure, not a better-resolved one.
P_BOX_X_LO = int(round(LX / 2.0 / DX - NX_BOX / 2.0))           # cell 84
P_BOX_X_HI = P_BOX_X_LO + NX_BOX                                # cell 178
P_BOX_HI_OUTER = P_BOX_HI + NY_SIDE                             # cell 181

X_BOX_LO_M = P_BOX_X_LO * DX                    # 10.668 mm
X_BOX_HI_M = P_BOX_X_HI * DX                    # 22.606 mm
Y_BOX_LO_M = P_BOX_LO * DX                      #  4.572 mm
Y_TRACE_LO_M = P_TRACE_LO * DX                  # 13.462 mm  (== TRACE_Y_LO)
Y_TRACE_HI_M = P_BOX_HI * DX                    # 14.097 mm  REALIZED trace top
Y_BOX_HI_M = P_BOX_HI_OUTER * DX                # 22.987 mm
T_METAL = DX                                    # conductor thickness, fixed


def _cells(length_m: float, dx: float, what: str) -> int:
    """Whole cells in ``length_m`` at ``dx``, refusing a non-integer count."""
    n = length_m / dx
    k = int(round(n))
    if abs(n - k) > 1e-6:
        raise ValueError(
            f"{what} = {length_m*1e6:.3f} um is {n:.6f} cells at "
            f"dx = {dx*1e6:.3f} um -- not a whole number. A mesh that cannot "
            f"land this plane exactly cannot reproduce the design box.")
    return k


@dataclass(frozen=True)
class Mesh:
    """Cell bookkeeping for one mesh. Every count is DERIVED, none hardcoded.

    ``mesh(DX)`` reproduces the coarse constants above exactly; ``mesh(DX/2)``
    doubles every one of them, which is what makes the 2x2 mask refinement in
    ``phase2_calibrate.refine_mask`` an identity on physical space.
    """

    dx: float
    refine: int                 # DX / dx, 1 at the coarse mesh
    p_box_x_lo: int
    p_box_x_hi: int
    p_box_lo: int
    p_trace_lo: int
    p_box_hi: int
    p_box_hi_outer: int
    ny_cells: int
    n_metal_cells: int
    trace_box_y_hi: float       # the Box face handed to rfx (snapped if fine)

    @property
    def nx_box(self) -> int:
        return self.p_box_x_hi - self.p_box_x_lo

    @property
    def ny_side(self) -> int:
        return self.p_trace_lo - self.p_box_lo

    @property
    def n_trace_cells(self) -> int:
        return self.p_box_hi - self.p_trace_lo

    @property
    def edge_margin_cells(self) -> int:
        return self.p_box_lo

    @property
    def n_vars(self) -> int:
        return 2 * self.nx_box * self.ny_side

    def physical_bounds(self) -> dict:
        """The design box's realized planes, in metres, FROM THE CELL INDICES.

        Computed from the integers the mesh actually uses, not from the
        constants they were derived from, so comparing two meshes with this
        compares what the solver will really see.
        """
        d = self.dx
        return {
            "x_lo": self.p_box_x_lo * d, "x_hi": self.p_box_x_hi * d,
            "lo_y_lo": self.p_box_lo * d, "lo_y_hi": self.p_trace_lo * d,
            "hi_y_lo": self.p_box_hi * d, "hi_y_hi": self.p_box_hi_outer * d,
            "trace_y_lo": self.p_trace_lo * d, "trace_y_hi": self.p_box_hi * d,
            "ly": self.ny_cells * d,
            "metal_t": self.n_metal_cells * d,
        }

    def describe(self) -> str:
        return (f"dx = {self.dx*1e6:.2f} um (refine x{self.refine})  "
                f"y stack {self.edge_margin_cells} | {self.ny_side} | "
                f"{self.n_trace_cells} | {self.ny_side} | "
                f"{self.edge_margin_cells} = {self.ny_cells} cells   "
                f"box {self.nx_box} x {self.ny_side} x 2 = {self.n_vars} vars  "
                f"metal {self.n_metal_cells} cell(s) = "
                f"{self.n_metal_cells*self.dx*1e6:.1f} um")


def mesh(dx: float = DX) -> Mesh:
    """Derive the full cell bookkeeping at ``dx``.

    ``dx`` must divide ``DX`` an integer number of times, which is what keeps
    every reference plane on a cell edge at both meshes and what makes a
    coarse mask refinable by whole-cell replication.
    """
    r = DX / dx
    refine = int(round(r))
    if refine < 1 or abs(r - refine) > 1e-9:
        raise ValueError(
            f"dx = {dx*1e6:.4f} um must divide DX = {DX*1e6:.1f} um an integer "
            f"number of times (got DX/dx = {r:.6f}); otherwise the design-box "
            f"planes do not land on cell edges and the two meshes are not "
            f"comparable.")
    m = Mesh(
        dx=float(dx), refine=refine,
        p_box_x_lo=_cells(X_BOX_LO_M, dx, "design box x_lo"),
        p_box_x_hi=_cells(X_BOX_HI_M, dx, "design box x_hi"),
        p_box_lo=_cells(Y_BOX_LO_M, dx, "design box -y outer edge"),
        p_trace_lo=_cells(Y_TRACE_LO_M, dx, "trace y_lo"),
        p_box_hi=_cells(Y_TRACE_HI_M, dx, "realized trace y_hi"),
        p_box_hi_outer=_cells(Y_BOX_HI_M, dx, "design box +y outer edge"),
        ny_cells=_cells(LY, dx, "LY"),
        n_metal_cells=_cells(T_METAL, dx, "conductor thickness"),
        trace_box_y_hi=(TRACE_Y_HI if refine == 1 else Y_TRACE_HI_M),
    )
    if m.ny_side != m.p_box_hi_outer - m.p_box_hi:
        raise RuntimeError("the two design sides came out different depths")
    if m.refine == 1 and (m.nx_box, m.ny_side, m.n_trace_cells,
                          m.edge_margin_cells, m.ny_cells) != (
            NX_BOX, NY_SIDE, N_TRACE_CELLS, EDGE_MARGIN_CELLS, NY_CELLS):
        raise RuntimeError("mesh(DX) does not reproduce the coarse constants")
    return m


COARSE = mesh(DX)
FINE = mesh(DX / 2.0)


def assert_same_physical_bounds(a: Mesh, b: Mesh, tol_m: float = 1e-12) -> dict:
    """Every design-box plane must land at the same METRE value on both meshes.

    Returns ``{plane: (a_m, b_m, delta_m)}`` so the caller can print it. Raises
    on the first plane that disagrees: a dx/2 robustness number computed inside
    a different rectangle is not a robustness number for the dx design.

    ``metal_t`` is compared too and is expected to MATCH, because ``T_METAL``
    is fixed rather than tied to dx.
    """
    pa, pb = a.physical_bounds(), b.physical_bounds()
    out = {}
    bad = []
    for k in pa:
        d = pb[k] - pa[k]
        out[k] = (pa[k], pb[k], d)
        if abs(d) > tol_m:
            bad.append(f"{k}: {pa[k]*1e6:.3f} um at dx={a.dx*1e6:.2f} vs "
                       f"{pb[k]*1e6:.3f} um at dx={b.dx*1e6:.2f} "
                       f"(delta {d*1e9:.1f} nm)")
    if bad:
        raise AssertionError(
            "design box does not occupy the same physical rectangle at the "
            "two meshes:\n  " + "\n  ".join(bad))
    return out

# Preflight codes the ORIGINAL (empty, one-sided) fixture already emits, read
# off the live report rather than guessed:
#   msl_port_geometry         x2  -- 2 substrate cells in z at dx = h_sub/2
#   lossless_q                    -- lossless dielectric in an open domain
#   pec_faces_finite_pec          -- pec_faces={z_lo} plus finite PEC objects
#   off_lattice_design_edges      -- the 600 um trace on a 127 um lattice
# The smoke diffs the FULL message text against the original, not just these
# codes; the set is kept as the second, coarser tripwire.
BASELINE_PREFLIGHT_CODES = frozenset({
    "msl_port_geometry", "lossless_q", "pec_faces_finite_pec",
    "off_lattice_design_edges",
})

# Codes that appear only once per-cell design metal is added, on the ORIGINAL
# fixture too. They are a property of the shared per-column box rasterization
# (``xval1_imperative.mask_to_boxes`` emits 1-cell-wide boxes with 1-3 cell
# gaps, and so does ``boxes_from_mask``), NOT of the two-sided geometry. The
# smoke does not take this on faith: it rasterizes equivalent column geometry
# on the ORIGINAL fixture and requires the two code sets to match.
PATHWAY_PREFLIGHT_CODES = frozenset({"mesh_resolution"})


def quarter_wave(f_hz: float) -> float:
    """Analytic lambda_g/4 on this substrate (Hammerstad eps_eff)."""
    return C0 / (f_hz * np.sqrt(EPS_EFF)) / 4.0


# ---------------------------------------------------------------------------
# 1. Fixture
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fixture:
    """What ``build_sim`` hands back.

    A dataclass rather than the original's 5-tuple because this fixture has
    TWO trace edges that callers need (``trace_y_lo`` and ``trace_y_hi``), and
    a positional tuple is exactly how a caller ends up putting the -y box on
    the +y side.
    """

    sim: Simulation
    y_trace: float
    trace_y_lo: float
    trace_y_hi: float
    d_set: object          # MSLPlaneProbeSet, port 0
    p_set: object          # MSLPlaneProbeSet, port 1
    mesh: Mesh = COARSE    # the cell bookkeeping this fixture was built on


def build_sim(freqs, dx: float = DX) -> Fixture:
    """Two-sided through-line fixture. Mirrors ``notch.build_sim`` except LY.

    Parameters
    ----------
    freqs : array of Hz -- the DFT bins the plane probes accumulate. Passed
        straight to ``register_msl_plane_probes`` exactly as the original does.
    dx : cell size. ``DX`` (the default) is the validated production mesh and
        that path is byte-identical to the fixture before ``dx`` existed.
        ``DX/2`` is the mesh on which an etch-tolerance number is quotable
        (``robust_eval``: +-1 fine cell = 63.5 um = 1.27x the +-50 um PCB spec,
        against 2.54x at the coarse mesh). ``dx`` must divide ``DX``.

    Notes
    -----
    The domain, the port definitions and every design-box plane are held at the
    SAME physical values for every mesh -- see the ``Mesh`` block above and
    ``assert_same_physical_bounds``. The trace Box's upper face is snapped to
    the coarse REALIZED plane when ``dx < DX`` so that both meshes carry the
    identical 635 um conductor and the design region stays contiguous with it.

    Known asymmetry, stated rather than hidden: the MSL ports keep their
    validated definition (centred on the NOMINAL trace centre, ``W_TRACE``
    wide), so at ``dx/2`` the port's integration window covers 9 of the 10
    realized trace cell rows rather than all 10. Check that against the port
    Z0 readback when the real calibration runs.

    Port 1's excitation is disabled here, as in the original, so the
    differentiable ``forward()`` path drives port 0 only. ``solve()`` re-enables
    it for the imperative two-port extraction.
    """
    import jax.numpy as jnp

    m = mesh(dx)
    freqs_j = jnp.asarray(np.asarray(freqs, dtype=np.float64), dtype=jnp.float32)

    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=m.dx, cpml_layers=CPML_LAYERS,
        boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"),
        ),
    )
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    sim.add(Box((0, TRACE_Y_LO, H_SUB), (LX, m.trace_box_y_hi, H_SUB + T_METAL)),
            material="pec")

    sim.add_msl_port(position=(PORT_MARGIN, Y_TRACE, 0.0),
                     width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0)
    sim.add_msl_port(position=(PORT_MARGIN + L_LINE, Y_TRACE, 0.0),
                     width=W_TRACE, height=H_SUB,
                     direction="-x", impedance=50.0)

    d_set = register_msl_plane_probes(sim, port_index=0, freqs=freqs_j,
                                      name_prefix="d")
    p_set = register_msl_plane_probes(sim, port_index=1, freqs=freqs_j,
                                      name_prefix="p")

    object.__setattr__(sim._msl_ports[1], "excite", False)
    return Fixture(sim=sim, y_trace=Y_TRACE, trace_y_lo=TRACE_Y_LO,
                   trace_y_hi=m.trace_box_y_hi, d_set=d_set, p_set=p_set,
                   mesh=m)


def build_original_sim(freqs):
    """The one-sided validated fixture, for preflight and physics diffing."""
    import jax.numpy as jnp
    freqs_j = jnp.asarray(np.asarray(freqs, dtype=np.float64), dtype=jnp.float32)
    return notch.build_sim(freqs_j)


# ---------------------------------------------------------------------------
# 2. The two-sided design box
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoxSide:
    """One side of the two-sided box, in ARRAY (padded) cell indices.

    Ranges are half-open: cells ``ix_lo <= i < ix_hi``, ``iy_lo <= j < iy_hi``,
    at the single trace layer ``iz``. A mask for this side has shape
    ``(nx, ny)`` and ``mask[i, j]`` is grid cell ``(ix_lo + i, iy_lo + j, iz)``.

    ``mask[:, j]`` is indexed in ASCENDING GLOBAL y on BOTH sides. On the
    ``lo`` side that means j = ny-1 is the row touching the trace and j = 0 is
    the outermost row. ``outward_index`` converts a distance-from-trace index
    into the global-order index; ``mask_from_stubs`` uses it, and nothing else
    should need to.
    """

    name: str
    ix_lo: int
    ix_hi: int
    iy_lo: int
    iy_hi: int
    iz: int
    pads: tuple
    dx: float
    nz: int = 1            # metal-layer thickness IN CELLS (2 at dx/2), so the
                           # realized conductor is T_METAL thick at every mesh

    @property
    def nx(self) -> int:
        return self.ix_hi - self.ix_lo

    @property
    def ny(self) -> int:
        return self.iy_hi - self.iy_lo

    @property
    def shape(self) -> tuple:
        return (self.nx, self.ny)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def _edge(self, axis: int, idx: int) -> float:
        return (idx - self.pads[axis]) * self.dx

    @property
    def extent_m(self) -> tuple:
        """((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)) at CELL EDGES, metres."""
        return (
            (self._edge(0, self.ix_lo), self._edge(0, self.ix_hi)),
            (self._edge(1, self.iy_lo), self._edge(1, self.iy_hi)),
            (self._edge(2, self.iz), self._edge(2, self.iz + self.nz)),
        )

    def outward_index(self, k: int) -> int:
        """Global-order column index of the k-th row OUT from the trace."""
        if not 0 <= k < self.ny:
            raise IndexError(f"{self.name}: outward row {k} outside 0..{self.ny-1}")
        return (self.ny - 1 - k) if self.name == "lo" else k

    @property
    def trace_row(self) -> int:
        """Array index of the mask row that abuts the through-line.

        ``0`` on the ``hi`` side, ``ny-1`` on the ``lo`` side, because
        ``mask[:, j]`` ascends in GLOBAL y on both sides. Same fact as
        ``outward_index(0)``, named for the reader who is thinking about the
        etch rather than about stub placement.
        """
        return self.outward_index(0)

    @property
    def etch_outside(self) -> tuple:
        """EROSION boundary convention for THIS side: (x_lo, x_hi, y_lo, y_hi).

        The one place the per-side answer is decided, and the value
        ``robust_eval.erode`` / ``robust_eval.three_fields`` REQUIRE (they have
        no default, deliberately). ``1`` means "assume metal beyond that array
        edge", ``0`` means "assume bare dielectric".

        Only the trace-adjacent edge is metal: the etch does not open a seam
        between a stub and the line it is rooted on. The outer transverse edge
        and both along-line edges face bare dielectric, and an over-etch DOES
        attack metal there -- which matters because a bounded-box objective
        actively pushes design metal against those walls.

        The side decides WHICH y edge is the trace edge:

            'hi' (+y): trace at j = 0      -> (0, 0, 1, 0)
            'lo' (-y): trace at j = ny-1   -> (0, 0, 0, 1)

        Using the ``hi`` tuple on the ``lo`` side is not a cosmetic error: a
        5-cell lo-side stub root erodes to ZERO cells under it (measured on
        this box), i.e. the whole design silently detaches from the feed line.

        Dilation takes no convention at all -- see ``robust_eval.dilate``.
        """
        return (robust_eval.OUTSIDE_TRACE_AT_Y_HI if self.name == "lo"
                else robust_eval.OUTSIDE_TRACE_AT_Y_LO)

    def empty_mask(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.uint8)


@dataclass(frozen=True)
class DesignBox:
    """The two-sided box. Both sides are named; there is no bare default.

    ``for name, side in box.items()`` is the intended iteration. There is
    deliberately no ``box.side`` / ``box[0]`` accessor: a caller that wants one
    side has to name it, which is the whole point of the PI's decision 1.
    """

    lo: BoxSide
    hi: BoxSide
    iz: int
    grid_shape: tuple
    dx: float
    mesh: Mesh = COARSE

    def items(self):
        return (("lo", self.lo), ("hi", self.hi))

    def side(self, name: str) -> BoxSide:
        if name not in SIDES:
            raise KeyError(f"side must be one of {SIDES}, got {name!r}")
        return self.lo if name == "lo" else self.hi

    @property
    def n_vars(self) -> int:
        return self.lo.n_cells + self.hi.n_cells

    def empty_mask(self) -> dict:
        return {n: s.empty_mask() for n, s in self.items()}


def _mesh_for_grid(grid, m: Mesh | None = None) -> Mesh:
    """The ``Mesh`` a grid was built on, checked against the grid's own dx."""
    got = float(grid.dx)
    if m is None:
        return mesh(got)
    if abs(m.dx - got) > 1e-15:
        raise ValueError(
            f"mesh dx = {m.dx*1e6:.4f} um but the grid was built at "
            f"{got*1e6:.4f} um")
    return m


def _trace_cell_span(grid, m: Mesh | None = None) -> tuple:
    """Array-index range of the RASTERIZED trace, by cell-centre containment."""
    m = _mesh_for_grid(grid, m)
    ny = grid.shape[1]
    pad_y = grid.axis_pads[1]
    yc = (np.arange(ny) - pad_y + 0.5) * m.dx
    j = np.where((yc >= TRACE_Y_LO) & (yc <= m.trace_box_y_hi))[0]
    if j.size == 0:
        raise RuntimeError("trace rasterized to zero cells -- geometry is wrong")
    return int(j[0]), int(j[-1] + 1)      # half-open


def design_box(grid, m: Mesh | None = None,
               nx_box: int | None = None,
               ny_side: int | None = None) -> DesignBox:
    """Cell index ranges of the TWO-SIDED design box.

    12 mm along the line centred on LX/2 (floored to 94 COARSE cells), 9 mm on
    each side of the trace (floored to 70 coarse cells), at the trace layer z.
    At a refined mesh every count is scaled by ``m.refine`` and the box occupies
    the IDENTICAL physical rectangle -- the indices come from the frozen planes
    ``X_BOX_LO_M`` ... ``Y_BOX_HI_M``, never from a re-rounded half-width, which
    is what used to move the x block by one cell at dx/2.

    Both sides are returned explicitly, so a caller cannot accidentally use
    one and believe it has the whole box.
    """
    m = _mesh_for_grid(grid, m)
    nx, ny, nz = grid.shape
    pad_x, pad_y, pad_z = grid.axis_pads
    nx_box = m.nx_box if nx_box is None else int(nx_box)
    ny_side = m.ny_side if ny_side is None else int(ny_side)

    # --- x: the frozen physical block, in this mesh's cells
    ix_lo, ix_hi = m.p_box_x_lo + pad_x, m.p_box_x_lo + nx_box + pad_x
    if ix_lo < 0 or ix_hi > nx:
        raise RuntimeError(f"design box x range [{ix_lo},{ix_hi}) outside grid nx={nx}")

    # --- z: the bottom cell of the T_METAL-thick trace layer
    zc = (np.arange(nz) - pad_z + 0.5) * m.dx
    iz = int(np.argmin(np.abs(zc - (H_SUB + 0.5 * m.dx))))

    # --- y: the frozen planes. At the coarse mesh these are exactly the
    #     rasterized trace block; the smoke asserts the contiguity at every mesh.
    lo = BoxSide("lo", ix_lo, ix_hi, m.p_trace_lo - ny_side + pad_y,
                 m.p_trace_lo + pad_y, iz, grid.axis_pads, m.dx, m.n_metal_cells)
    hi = BoxSide("hi", ix_lo, ix_hi, m.p_box_hi + pad_y,
                 m.p_box_hi + ny_side + pad_y, iz, grid.axis_pads, m.dx,
                 m.n_metal_cells)
    for s in (lo, hi):
        if s.iy_lo < 0 or s.iy_hi > ny:
            raise RuntimeError(
                f"design box side '{s.name}' y range [{s.iy_lo},{s.iy_hi}) "
                f"outside grid ny={ny} -- LY is too small")
    return DesignBox(lo=lo, hi=hi, iz=iz, grid_shape=tuple(grid.shape),
                     dx=m.dx, mesh=m)


# ---------------------------------------------------------------------------
# 3. mask <-> Box geometry. ONE pathway, shared by every arm.
# ---------------------------------------------------------------------------
def _as_sides(mask) -> dict:
    """Normalize a two-sided mask into {'lo': (nx,ny), 'hi': (nx,ny)}."""
    if isinstance(mask, Mapping):
        missing = set(SIDES) - set(mask)
        if missing:
            raise KeyError(f"two-sided mask is missing side(s) {sorted(missing)}")
        extra = set(mask) - set(SIDES)
        if extra:
            raise KeyError(f"two-sided mask has unknown side(s) {sorted(extra)}")
        return {n: np.asarray(mask[n]) for n in SIDES}
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] == 2:
        return {"lo": arr[0], "hi": arr[1]}
    raise TypeError(
        "mask must be a mapping with keys ('lo','hi') or an array of shape "
        f"(2, nx, ny); got {type(mask).__name__} with shape "
        f"{getattr(arr, 'shape', None)}")


def boxes_from_mask(mask, box: DesignBox, threshold: float = 0.5) -> list:
    """Run-length merge a two-sided per-cell metal mask into PEC boxes.

    Returns ``[((x_lo, y_lo, z_lo), (x_hi, y_hi, z_hi)), ...]`` in METRES at
    CELL EDGES -- byte-for-byte the convention
    ``xval1_imperative.mask_to_boxes`` uses, so a design solved here and a
    design solved there rasterize identically. Merging is along y within each
    x column, which turns a few-thousand-cell design into a few dozen boxes.

    Both sides are emitted into one flat list; the ``lo`` side comes first.
    """
    out: list = []
    sides = _as_sides(mask)
    for name, side in box.items():
        m = sides[name]
        if m.shape != side.shape:
            raise ValueError(
                f"side '{name}' mask shape {m.shape} != box shape {side.shape}")
        hard = np.asarray(m, dtype=float) >= threshold
        pad_x, pad_y, pad_z = side.pads
        z_lo = (side.iz - pad_z) * side.dx
        z_hi = (side.iz + side.nz - pad_z) * side.dx
        for i in range(side.nx):
            j = 0
            while j < side.ny:
                if not hard[i, j]:
                    j += 1
                    continue
                j0 = j
                while j < side.ny and hard[i, j]:
                    j += 1
                gi = side.ix_lo + i
                gj0, gj1 = side.iy_lo + j0, side.iy_lo + j
                out.append((
                    ((gi - pad_x) * side.dx, (gj0 - pad_y) * side.dx, z_lo),
                    ((gi + 1 - pad_x) * side.dx, (gj1 - pad_y) * side.dx, z_hi),
                ))
    return out


def mask_from_boxes(boxes: Iterable, box: DesignBox) -> dict:
    """Inverse of :func:`boxes_from_mask` -- rasterize boxes back to a mask.

    Cell-centre containment, the same rule rfx uses for a ``Box``. Exists so
    the round trip can be ASSERTED rather than assumed; a box that lands
    outside the design box raises.
    """
    out = box.empty_mask()
    for lo, hi in boxes:
        placed = False
        for name, side in box.items():
            pad_x, pad_y, pad_z = side.pads
            xc = (np.arange(side.ix_lo, side.ix_hi) - pad_x + 0.5) * side.dx
            yc = (np.arange(side.iy_lo, side.iy_hi) - pad_y + 0.5) * side.dx
            zc = (np.arange(side.iz, side.iz + side.nz) - pad_z + 0.5) * side.dx
            if not (lo[2] <= zc.min() and zc.max() <= hi[2]):
                continue
            mi = (xc >= lo[0]) & (xc <= hi[0])
            mj = (yc >= lo[1]) & (yc <= hi[1])
            if not (mi.any() and mj.any()):
                continue
            out[name][np.ix_(mi, mj)] = 1
            placed = True
        if not placed:
            raise ValueError(f"box {lo}->{hi} lands outside the design box")
    return out


def add_pec_boxes(sim: Simulation, boxes: Iterable) -> int:
    """Add every ``(lo, hi)`` tuple to ``sim`` as hard PEC. Returns the count."""
    n = 0
    for lo, hi in boxes:
        sim.add(Box(lo, hi), material="pec")
        n += 1
    return n


# ---------------------------------------------------------------------------
# 3b. Stubs -> mask. The classical arm enters through the SAME pathway.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stub:
    """A straight open stub rooted on the trace.

    side     : 'lo' or 'hi' -- which side of the trace it grows from
    x_centre : metres, absolute domain coordinate along the line
    width    : metres, extent along the line
    length   : metres, extent away from the trace
    """

    side: str
    x_centre: float
    width: float
    length: float


def _coerce_stub(s) -> Stub:
    if isinstance(s, Stub):
        stub = s
    else:
        side, x_centre, width, length = s
        stub = Stub(side, float(x_centre), float(width), float(length))
    if stub.side not in SIDES:
        raise ValueError(f"stub side must be one of {SIDES}, got {stub.side!r}")
    return stub


def mask_from_stubs(stubs: Sequence, box: DesignBox,
                    clip: bool = False) -> dict:
    """Place straight stubs of given ``(side, x_centre, width, length)``.

    This is the classical arm's ONLY way into the solver, and it is the same
    way the TO arm gets there: both hand a per-cell mask to
    :func:`boxes_from_mask`. A stub that does not fit inside the bounded box
    raises unless ``clip=True`` (which is for a deliberate sweep past the
    edge, and says so in the exception it no longer raises).

    Rasterization: cells whose CENTRE falls inside the requested rectangle,
    with a one-cell floor on each dimension so a sub-cell width or length
    still produces metal rather than silently nothing.
    """
    out = box.empty_mask()
    for raw in stubs:
        stub = _coerce_stub(raw)
        side = box.side(stub.side)
        pad_x, pad_y, _ = side.pads
        xc = (np.arange(side.ix_lo, side.ix_hi) - pad_x + 0.5) * side.dx

        x_lo = stub.x_centre - stub.width / 2.0
        x_hi = stub.x_centre + stub.width / 2.0
        mi = (xc >= x_lo) & (xc <= x_hi)
        if not mi.any():
            i_near = int(np.argmin(np.abs(xc - stub.x_centre)))
            if not clip and not (side.extent_m[0][0] <= stub.x_centre
                                 <= side.extent_m[0][1]):
                raise ValueError(
                    f"stub at x={stub.x_centre*1e3:.3f} mm is outside the "
                    f"design box x range "
                    f"[{side.extent_m[0][0]*1e3:.3f}, "
                    f"{side.extent_m[0][1]*1e3:.3f}] mm")
            mi = np.zeros_like(xc, dtype=bool)
            mi[i_near] = True

        n_len = max(1, int(np.floor(stub.length / side.dx + 1e-9)))
        if n_len > side.ny:
            if not clip:
                raise ValueError(
                    f"stub length {stub.length*1e3:.3f} mm = {n_len} cells "
                    f"exceeds the {side.ny}-cell "
                    f"({side.ny*side.dx*1e3:.3f} mm) design depth on side "
                    f"'{stub.side}'")
            n_len = side.ny
        cols = [side.outward_index(k) for k in range(n_len)]
        out[stub.side][np.ix_(mi, np.array(cols, dtype=int))] = 1
    return out


# ---------------------------------------------------------------------------
# 3c. Etch tolerance (PI decision 2) -- DISPATCH ONLY, no morphology here
# ---------------------------------------------------------------------------
# This file used to carry its own ``_dilate`` plus a
# ``shrink = ~dilate(~m)`` erosion. Both are DELETED. Two reasons, and the
# second is the one that mattered:
#
#  * a second implementation of the same operator is a second thing to keep
#    correct, and ``robust_eval`` is where the operator is specified, exact,
#    and self-tested;
#  * ``shrink = ~dilate(~m)`` with a clipped dilation is an erosion that
#    assumes METAL beyond ALL FOUR box edges. Only ONE edge -- the
#    trace-adjacent one -- is metal. The other three face bare dielectric, so
#    that erosion never attacked metal pressed against a box wall, which is
#    exactly where a bounded-box objective puts it. Measured on this box: a
#    3x3 pad in the corner went 9 cells -> 4 instead of 9 -> 1, and a
#    full-depth 70-cell stub kept its tip entirely (70 -> 70 instead of
#    70 -> 69). Both overstate worst-case margin.
#
# What is left here is the two-sided DISPATCH: which array edge abuts the
# through-line is a property of the fixture, not of the morphology, and
# ``BoxSide.etch_outside`` is where it is decided.
def etch_fields(mask, box: DesignBox, cells: int = 1,
                connectivity: int = 4) -> dict:
    """The etch bracket as ``{'eroded', 'nominal', 'dilated'}`` two-sided masks.

    Field names are ``robust_eval.FIELD_ORDER`` verbatim, so the result feeds
    ``robust_eval.robust_score`` (keyed by the same three names) without a
    translation step. The old ``{'shrink','nominal','grow'}`` names did not
    compose with it and are gone.

    Each side is routed through ``robust_eval.three_fields`` with THAT SIDE's
    ``etch_outside`` convention. There is no morphology in this function.

    ``cells=1`` on the coarse grid is +-127 um, which BRACKETS the +-50 um PCB
    etching tolerance pessimistically (2.54x) and is a stress case, not a
    quotable tolerance result; on the dx/2 grid the same call is +-63.5 um,
    the nearest representable to the spec and 1.27x conservative. That is the
    only mesh on which a robustness number is quotable --
    ``robust_eval.calibrate_etch(...).quotable`` is the gate, and
    ``robust_score`` annotates the report when it is False.
    """
    sides = _as_sides(mask)
    out = {k: {} for k in robust_eval.FIELD_ORDER}
    for name, side in box.items():
        m = np.asarray(sides[name])
        if m.shape != side.shape:
            raise ValueError(
                f"side '{name}' mask shape {m.shape} != box shape {side.shape}")
        tf = robust_eval.three_fields(m, cells, outside=side.etch_outside,
                                      connectivity=connectivity)
        for k in robust_eval.FIELD_ORDER:
            out[k][name] = tf[k].astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# 4. Solve -- imperative extractor, full validity block
# ---------------------------------------------------------------------------
def solve(sim: Simulation, freqs, num_periods: float) -> dict:
    """Imperative two-port solve. Returns S21/S11 in dB + the validity block.

    Mirrors ``xval1_imperative``: hard-PEC ``Box`` geometry, the imperative
    ``compute_msl_s_matrix``, absolute (not empty-line-normalized) S-params,
    and the solver's OWN reliability verdict carried into the record so a
    high-Q notch read from a truncated record cannot be quoted as physics.

    The returned keys ``settling_db`` / ``reliable`` / ``passivity_correction``
    feed ``score_dualband.check_validity`` unchanged, and
    ``settling_worst_db`` / ``reliable_bins`` / ``passivity_worst`` are the
    same summary fields ``xval1_imperative`` writes into its JSON.

    ONE DELIBERATE DEPARTURE from xval1_imperative, and it is a bug fix.
    ``compute_msl_s_matrix`` sets ``passivity_correction = None`` when NO bin
    needed projecting, i.e. exactly when the extraction was PERFECTLY PASSIVE.
    ``np.asarray(None, dtype=float)`` is ``array(nan)``, size 1, so the
    xval1 idiom ``float(np.max(pcorr)) if pcorr.size else 0.0`` records
    ``passivity_worst = nan`` for the cleanest possible run. That is not
    cosmetic: ``score_dualband.check_validity`` computes
    ``pworst <= PASSIVITY_MAX``, ``nan <= 0.05`` is False, and the run is
    declared NOT QUOTABLE for having been too good. Measured here on a
    2-period smoke solve. None is mapped to an all-zero correction vector
    instead. (Every xval1 record on disk happens to carry a non-zero
    correction, so the latch never fired there -- it is latent, not
    historical.)
    """
    import jax.numpy as jnp

    f_np = np.asarray(freqs, dtype=np.float64)
    freqs_j = jnp.asarray(f_np, dtype=jnp.float32)

    # compute_msl_s_matrix drives BOTH ports; build_sim disabled port 1.
    object.__setattr__(sim._msl_ports[1], "excite", True)

    grid = sim._build_grid()
    dt = float(grid.dt)
    n_steps = int(round(num_periods * (1.0 / F_MAX) / dt))

    t0 = time.time()
    res = sim.compute_msl_s_matrix(freqs=freqs_j, num_periods=float(num_periods))
    wall = time.time() - t0

    n_f = int(np.asarray(res.freqs).size)
    settling = np.asarray(getattr(res, "settling_db", None)
                          if getattr(res, "settling_db", None) is not None
                          else [], dtype=float)
    _rel = getattr(res, "reliable", None)
    reliable = np.asarray(_rel if _rel is not None else [], dtype=bool)
    _pc = getattr(res, "passivity_correction", None)
    # None == nothing was projected == already passive == correction is 0.
    pcorr = (np.asarray(_pc, dtype=float) if _pc is not None
             else np.zeros(n_f, dtype=float))
    projected = _pc is not None
    s_worst = float(np.max(settling)) if settling.size else float("nan")
    p_worst = float(np.max(pcorr)) if pcorr.size else 0.0

    f = np.asarray(res.freqs)
    s21 = np.asarray(res.S[1, 0, :])
    s11 = np.asarray(res.S[0, 0, :])
    db21 = 20 * np.log10(np.abs(s21) + 1e-30)
    db11 = 20 * np.log10(np.abs(s11) + 1e-30)

    return dict(
        num_periods=float(num_periods),
        n_freqs=int(f.size),
        n_steps=n_steps,
        record_ns=n_steps * dt * 1e9,
        dft_res_GHz=(1.0 / (n_steps * dt) / 1e9) if n_steps else float("nan"),
        wall_s=round(wall, 1),
        freqs_GHz=[float(x) / 1e9 for x in f],
        freqs_MHz=[int(round(float(x) / 1e6)) for x in f],
        s21_db=[float(x) for x in db21],
        s11_db=[float(x) for x in db11],
        f_min_GHz=float(f[int(np.argmin(db21))] / 1e9),
        depth_min_db=float(np.min(db21)),
        # ---- validity block, as xval1_imperative records it ----
        settling_worst_db=(s_worst if settling.size else None),
        settled=bool(settling.size and s_worst <= -40.0),
        reliable_bins=[int(reliable.sum()), int(reliable.size)],
        passivity_worst=p_worst,
        passivity_projected=bool(projected),
        settling_db=[float(x) for x in settling.ravel()],
        reliable=reliable.tolist(),
        passivity_correction=[float(x) for x in pcorr.ravel()],
    )




# ---------------------------------------------------------------------------
# 5. Smoke -- CPU, no solve
# ---------------------------------------------------------------------------
def _preflight_records(sim) -> list:
    """(code, severity, full message) triples. Full text: a message that keeps
    its code but changes its NUMBERS is a different fixture, and the first cut
    of this file was caught exactly that way."""
    return [(getattr(m, "code", "uncoded"),
             getattr(m, "severity", "warning"), str(m))
            for m in sim.preflight()]


def _codes(records) -> list:
    return sorted({c for c, _, _ in records})


def _stub_column_boxes(grid, y_root, x_centre, width, length, outward=+1):
    """1-cell-wide-in-x PEC column boxes for a straight stub, on ANY grid.

    Same cell-edge convention and same per-column decomposition
    ``boxes_from_mask`` uses. Exists so the ORIGINAL one-sided fixture can be
    given geometry of the same SHAPE CLASS as a pixelated design, which is what
    turns "these mesh_resolution warnings come from the pathway, not from the
    two-sided box" from an assertion into a measurement.
    """
    nx, ny, nz = grid.shape
    pad_x, pad_y, pad_z = grid.axis_pads
    xc = (np.arange(nx) - pad_x + 0.5) * DX
    yc = (np.arange(ny) - pad_y + 0.5) * DX
    zc = (np.arange(nz) - pad_z + 0.5) * DX
    iz = int(np.argmin(np.abs(zc - (H_SUB + 0.5 * DX))))
    mi = np.where((xc >= x_centre - width / 2) & (xc <= x_centre + width / 2))[0]
    if outward > 0:
        mj = np.where((yc >= y_root) & (yc <= y_root + length))[0]
    else:
        mj = np.where((yc <= y_root) & (yc >= y_root - length))[0]
    if mi.size == 0 or mj.size == 0:
        raise RuntimeError("control stub rasterized to nothing")
    j0, j1 = int(mj[0]), int(mj[-1] + 1)
    z_lo, z_hi = (iz - pad_z) * DX, (iz + 1 - pad_z) * DX
    return [(((int(i) - pad_x) * DX, (j0 - pad_y) * DX, z_lo),
             ((int(i) + 1 - pad_x) * DX, (j1 - pad_y) * DX, z_hi))
            for i in mi]


def _smoke() -> int:
    import jax.numpy as jnp  # noqa: F401  (pay the import cost once, up front)

    ok = True
    freqs = np.linspace(4.0e9, 8.0e9, 9)
    x_mid = LX / 2.0
    l_lo, l_hi = quarter_wave(5.25e9), quarter_wave(5.775e9)
    x_off = 4.0e-3

    print("=" * 78)
    print("PHASE-2 TWO-SIDED FIXTURE — smoke (CPU, no solve)")
    print("=" * 78)
    print(f"  substrate  eps_r={EPS_R}  h_sub={H_SUB*1e6:.0f}um  "
          f"W={W_TRACE*1e6:.0f}um  dx={DX*1e6:.1f}um  eps_eff={EPS_EFF:.3f}")
    print(f"  line       L_LINE={L_LINE*1e3:.1f}mm  "
          f"PORT_MARGIN={PORT_MARGIN*1e3:.2f}mm  F_MAX={F_MAX/1e9:.1f}GHz  "
          f"cpml_layers={CPML_LAYERS}")
    print(f"  CLEAR = 2*h_sub + 8*dx = {CLEAR*1e3:.3f} mm ({CLEAR/DX:.0f} cells)"
          f"   EDGE_MARGIN = 3*CLEAR = {EDGE_MARGIN*1e3:.3f} mm "
          f"({EDGE_MARGIN_CELLS} cells)")
    print(f"  y stack (cells): {EDGE_MARGIN_CELLS} | {NY_SIDE} | "
          f"{N_TRACE_CELLS} | {NY_SIDE} | {EDGE_MARGIN_CELLS} = {NY_CELLS}")

    # ---- original, for the LY / preflight diff -----------------------------
    print("\n-- original one-sided fixture (validated reference) --")
    sim0, y0, ty_hi0, _, _ = build_original_sim(freqs)
    grid0 = sim0._build_grid()
    pre0 = _preflight_records(sim0)
    print(f"  LY = {notch.LY*1e3:.3f} mm   y_trace = {y0*1e3:.3f} mm   "
          f"grid = {tuple(grid0.shape)}  ({np.prod(grid0.shape):,d} cells)  "
          f"dt = {float(grid0.dt)*1e12:.3f} ps")
    print(f"  design depth: +y only, {notch.L_STUB_MAX*1e3:.1f} mm max stub; "
          f"trace-to--y-wall {(y0 - W_TRACE/2)*1e3:.3f} mm (= 1*CLEAR); "
          f"metal-to-+y-wall "
          f"{(notch.LY - ty_hi0 - notch.L_STUB_MAX)*1e3:.3f} mm (= 3*CLEAR)")

    # ---- new two-sided fixture --------------------------------------------
    print("\n-- new two-sided fixture --")
    fx = build_sim(freqs)
    grid = fx.sim._build_grid()
    pre1 = _preflight_records(fx.sim)
    print(f"  LY = {LY*1e3:.3f} mm   y_trace = {fx.y_trace*1e3:.3f} mm   "
          f"trace y = [{fx.trace_y_lo*1e3:.3f}, {fx.trace_y_hi*1e3:.3f}] mm "
          f"= [{fx.trace_y_lo/DX:.3f}, {fx.trace_y_hi/DX:.3f}] cells")
    print(f"  GRID SHAPE = {tuple(grid.shape)}  "
          f"({np.prod(grid.shape):,d} cells)  dt = {float(grid.dt)*1e12:.3f} ps"
          f"   ({100*np.prod(grid.shape)/np.prod(grid0.shape)-100:+.1f} % cells "
          f"vs original)")
    print(f"  LX = {LX*1e3:.3f} mm (unchanged)   LZ = {LZ*1e3:.3f} mm (unchanged)")
    if float(grid.dt) != float(grid0.dt):
        print("  !! dt changed — the CFL step must not depend on LY")
        ok = False
    if tuple(grid.shape)[0] != tuple(grid0.shape)[0] or \
            tuple(grid.shape)[2] != tuple(grid0.shape)[2]:
        print("  !! nx or nz changed — only LY was supposed to move")
        ok = False

    # lattice offset of the two trace faces, vs the original's
    for tag, v, v0 in (("trace_y_lo", fx.trace_y_lo, y0 - W_TRACE / 2),
                       ("trace_y_hi", fx.trace_y_hi, ty_hi0)):
        fr, fr0 = (v / DX) % 1.0, (v0 / DX) % 1.0
        same = abs(fr - fr0) < 1e-9
        print(f"  lattice offset {tag}: {fr:.4f} cell "
              f"(original {fr0:.4f}) -> {'SAME' if same else 'DIFFERENT'}")
        if not same:
            print(f"  !! {tag} sits on a different sub-cell offset than the "
                  f"original — the realized trace is not the validated one")
            ok = False

    # preflight lateral-clearance arithmetic, spelled out so it is auditable
    cpml_t = CPML_LAYERS * DX
    c_lo = fx.trace_y_lo - cpml_t
    c_hi = (LY - cpml_t) - fx.trace_y_hi
    print(f"  preflight lateral clearance: -y {c_lo*1e6:.0f} um, "
          f"+y {c_hi*1e6:.0f} um  (rule: >= 2*h_sub = {2*H_SUB*1e6:.0f} um) -> "
          f"{'OK' if min(c_lo, c_hi) >= 2*H_SUB else 'VIOLATED'}")
    if min(c_lo, c_hi) < 2 * H_SUB:
        ok = False

    # ---- design box --------------------------------------------------------
    print("\n-- two-sided design box --")
    box = design_box(grid)
    j_tr = _trace_cell_span(grid)
    print(f"  rasterized trace cells (y, array idx): [{j_tr[0]}, {j_tr[1]}) "
          f"= {j_tr[1]-j_tr[0]} cells   (predicted N_TRACE_CELLS="
          f"{N_TRACE_CELLS})")
    if (j_tr[1] - j_tr[0]) != N_TRACE_CELLS:
        print("  !! rasterized trace width differs from the layout assumption")
        ok = False
    j_tr0 = np.where(
        ((np.arange(grid0.shape[1]) - grid0.axis_pads[1] + 0.5) * DX
         >= y0 - W_TRACE / 2)
        & ((np.arange(grid0.shape[1]) - grid0.axis_pads[1] + 0.5) * DX
           <= ty_hi0))[0]
    print(f"  original rasterized trace width: {j_tr0.size} cells -> "
          f"{'MATCH' if j_tr0.size == N_TRACE_CELLS else 'MISMATCH'}")
    if j_tr0.size != N_TRACE_CELLS:
        ok = False
    print(f"  z layer iz = {box.iz}")
    for name, s in box.items():
        (xl, xh), (yl, yh), (zl, zh) = s.extent_m
        print(f"  side '{name}': cells x[{s.ix_lo},{s.ix_hi}) "
              f"y[{s.iy_lo},{s.iy_hi}) -> {s.nx} x {s.ny} = {s.n_cells} cells")
        print(f"             x = [{xl*1e3:8.3f}, {xh*1e3:8.3f}] mm "
              f"({(xh-xl)*1e3:.3f} mm of {BOX_X*1e3:.1f} budget)   "
              f"y = [{yl*1e3:8.3f}, {yh*1e3:8.3f}] mm "
              f"({(yh-yl)*1e3:.3f} mm of {DESIGN_DEPTH*1e3:.1f} budget)")
        print(f"             z = [{zl*1e6:.1f}, {zh*1e6:.1f}] um "
              f"(trace layer; h_sub = {H_SUB*1e6:.0f} um)")
        if (xh - xl) > BOX_X + 1e-12 or (yh - yl) > DESIGN_DEPTH + 1e-12:
            print(f"  !! side '{name}' OVERRUNS the bounded box")
            ok = False
    print(f"  TOTAL binary variables = {box.n_vars}  "
          f"({NX_BOX} x {NY_SIDE} per side x 2 sides)")
    if box.lo.shape != box.hi.shape:
        print("  !! the two sides have DIFFERENT shapes")
        ok = False
    if box.lo.iy_hi != j_tr[0] or box.hi.iy_lo != j_tr[1]:
        print("  !! design box is not contiguous with the rasterized trace")
        ok = False
    m_lo = box.lo.extent_m[1][0]
    m_hi = LY - box.hi.extent_m[1][1]
    print(f"  design-metal-to-wall margin: -y {m_lo*1e3:.3f} mm, "
          f"+y {m_hi*1e3:.3f} mm   (original leaves "
          f"{(notch.LY - ty_hi0 - notch.L_STUB_MAX)*1e3:.3f} mm; "
          f"minimum for preflight is 1*CLEAR = {CLEAR*1e3:.3f} mm)")
    if min(m_lo, m_hi) < CLEAR - 1e-12:
        print("  !! design metal can reach closer to a wall than 1*CLEAR")
        ok = False
    # box centring on the line
    (bxl, bxh), _, _ = box.lo.extent_m
    print(f"  box x-centre = {(bxl+bxh)/2*1e3:.3f} mm vs LX/2 = "
          f"{x_mid*1e3:.3f} mm  (offset {abs((bxl+bxh)/2 - x_mid)*1e6:.0f} um "
          f"= {abs((bxl+bxh)/2 - x_mid)/DX:.2f} cell)")

    # ---- preflight diff: EMPTY fixtures. HARD GATE. -------------------------
    print("\n-- preflight, empty fixtures (HARD GATE) --")
    print(f"  original : {len(pre0)} message(s), codes {_codes(pre0)}")
    for c, sev, m in pre0:
        print(f"    [{sev}:{c}] {m[:130]}")
    print(f"  two-sided: {len(pre1)} message(s), codes {_codes(pre1)}")
    for c, sev, m in pre1:
        print(f"    [{sev}:{c}] {m[:130]}")
    base_full = {(c, sev, m) for c, sev, m in pre0}
    new_empty = [r for r in pre1 if r not in base_full]
    off_codes = sorted({c for c, _, _ in pre1} - BASELINE_PREFLIGHT_CODES)
    if new_empty or off_codes:
        print("\n" + "!" * 78)
        print("!! NEW PREFLIGHT MESSAGES — introduced by the two-sided fixture")
        for c, sev, m in new_empty:
            print(f"!!   [{sev}:{c}] {m}")
        for c in off_codes:
            print(f"!!   code {c!r} is outside the recorded baseline set "
                  f"{sorted(BASELINE_PREFLIGHT_CODES)}")
        print("!" * 78 + "\n")
        ok = False
    else:
        print(f"  -> NO NEW PREFLIGHT MESSAGES. Byte-identical to the "
              f"original's {len(pre0)}: {_codes(pre0)}")

    # ---- mask -> boxes -> mask round trip ----------------------------------
    print("\n-- mask <-> boxes round trip --")
    print(f"  lambda_g/4: {l_lo*1e3:.3f} mm @ 5.250 GHz, "
          f"{l_hi*1e3:.3f} mm @ 5.775 GHz   "
          f"(design depth {box.lo.ny*DX*1e3:.3f} mm — both fit)")

    cases = {}
    # (a) the two-sided relaxation the classical arm is now entitled to
    cases["classical pair, two-sided"] = mask_from_stubs(
        [("lo", x_mid - x_off, W_TRACE, l_lo),
         ("hi", x_mid + x_off, W_TRACE, l_hi)], box)
    # (b) the Stage-0 geometry as-was, both stubs on +y, in the same box
    cases["classical pair, one-sided"] = mask_from_stubs(
        [("hi", x_mid - x_off, W_TRACE, l_lo),
         ("hi", x_mid + x_off, W_TRACE, l_hi)], box)
    # (c) free-form -- what the TO arm hands over
    rng = np.random.default_rng(20260827)
    cases["random free-form p=0.25"] = {
        n: (rng.random(s.shape) < 0.25).astype(np.uint8)
        for n, s in box.items()}
    # (d) the degenerate ends
    cases["full box"] = {n: np.ones(s.shape, np.uint8) for n, s in box.items()}
    cases["empty box"] = box.empty_mask()
    # (e) an array-form (2, nx, ny) mask, to exercise the other input form
    cases["stacked-array form"] = np.stack(
        [cases["random free-form p=0.25"]["lo"],
         cases["random free-form p=0.25"]["hi"]])

    x_span = box.lo.extent_m[0]
    y_spans = {n: s.extent_m[1] for n, s in box.items()}
    for label, mk in cases.items():
        bx = boxes_from_mask(mk, box)
        back = mask_from_boxes(bx, box)
        src = _as_sides(mk)
        same = all(np.array_equal(np.asarray(src[n]), back[n]) for n in SIDES)
        fill = sum(int(np.asarray(src[n]).sum()) for n in SIDES)
        inside = True
        for lo, hi in bx:
            in_x = x_span[0] - 1e-12 <= lo[0] and hi[0] <= x_span[1] + 1e-12
            in_y = any(ys[0] - 1e-12 <= lo[1] and hi[1] <= ys[1] + 1e-12
                       for ys in y_spans.values())
            inside = inside and in_x and in_y
        print(f"  {label:28s} fill={fill:5d}/{box.n_vars}  "
              f"boxes={len(bx):5d}  round-trip={'OK' if same else 'MISMATCH'}  "
              f"inside-box={'OK' if inside else 'VIOLATED'}")
        if not same or not inside:
            ok = False

    # stub cell arithmetic, both sides
    mk = cases["classical pair, two-sided"]
    _xc = (np.arange(box.lo.ix_lo, box.lo.ix_hi) - box.lo.pads[0] + 0.5) * DX
    n_w = int(((_xc >= x_mid - x_off - W_TRACE / 2)
               & (_xc <= x_mid - x_off + W_TRACE / 2)).sum())
    for name, L in (("lo", l_lo), ("hi", l_hi)):
        n_cells = int(np.asarray(mk[name]).sum())
        n_len = int(np.floor(L / DX + 1e-9))
        print(f"  stub '{name}': {n_cells} cells = {n_w} wide x {n_len} long "
              f"({n_w*DX*1e6:.0f} um x {n_len*DX*1e3:.3f} mm; asked "
              f"{W_TRACE*1e6:.0f} um x {L*1e3:.3f} mm) -> "
              f"{'OK' if n_cells == n_w*n_len else 'MISMATCH'}")
        if n_cells != n_w * n_len:
            ok = False
    touch = {"lo": bool(np.asarray(mk["lo"])[:, -1].any()),
             "hi": bool(np.asarray(mk["hi"])[:, 0].any())}
    print(f"  stub roots touch the trace: lo={touch['lo']}  hi={touch['hi']}  "
          f"(lo grows inward from column ny-1 by the global-y convention)")
    if not all(touch.values()):
        print("  !! a stub is not rooted on the trace")
        ok = False
    # a stub longer than the box must be refused, not silently truncated
    try:
        mask_from_stubs([("hi", x_mid, W_TRACE, DESIGN_DEPTH + 1e-3)], box)
    except ValueError as e:
        print(f"  over-long stub refused as it should be: {str(e)[:80]}")
    else:
        print("  !! an over-long stub was silently accepted")
        ok = False

    # ---- etch bracket ------------------------------------------------------
    print("\n-- etch tolerance bracket (PI decision 2) --")
    print(f"  PCB spec +-{ETCH_SPEC_M*1e6:.0f} um; coarse cell "
          f"{ETCH_COARSE_M*1e6:.1f} um, fine cell {ETCH_FINE_M*1e6:.1f} um. "
          f"+-1 fine cell = +-{ETCH_FINE_M*1e6:.1f} um is the nearest "
          f"representable and is {100*(ETCH_FINE_M/ETCH_SPEC_M - 1):.0f} % "
          f"conservative.")
    print(f"  morphology: robust_eval (the ONE implementation); this file only "
          f"dispatches per side. Field names = robust_eval.FIELD_ORDER "
          f"{robust_eval.FIELD_ORDER}, so the bracket feeds robust_score "
          f"directly.")
    br = etch_fields(cases["classical pair, two-sided"], box, cells=1)

    if set(br) != set(robust_eval.FIELD_ORDER):
        print(f"  !! etch field names {sorted(br)} do not compose with "
              f"robust_score, which needs {robust_eval.FIELD_ORDER}")
        ok = False
    fills = {}
    for k in robust_eval.FIELD_ORDER:
        fills[k] = sum(int(br[k][n].sum()) for n in SIDES)
        print(f"  {k:8s} fill={fills[k]:5d}  "
              f"boxes={len(boxes_from_mask(br[k], box))}")
    if not fills["eroded"] < fills["nominal"] < fills["dilated"]:
        print("  !! etch bracket is not monotone")
        ok = False
    for k in ("eroded", "dilated"):
        if any(np.asarray(br[k][n]).shape != box.side(n).shape for n in SIDES):
            print(f"  !! {k} mask left the design box")
            ok = False

    # -- boundary-convention regression gates, on the REAL box ---------------
    # Four failures a joint review measured on the deleted implementation.
    # Each is asserted here so it cannot come back silently; the numbers in the
    # comments are what the BROKEN code produced.
    print("\n  boundary convention (regression gates on the real box):")
    print(f"    per-side erosion convention: "
          + ", ".join(f"{n}={box.side(n).etch_outside} "
                      f"(trace row j={box.side(n).trace_row})" for n in SIDES))

    # (1) D2 -- a lo-side stub root must SURVIVE the over-etch.
    #     Broken: the lo side used the hi convention -> root 5 cells -> 0,
    #     i.e. the design detached from the feed line.
    root_lo = int(np.asarray(br["eroded"]["lo"])[:, -1].sum())
    root_hi = int(np.asarray(br["eroded"]["hi"])[:, 0].sum())
    wrong_lo = int(robust_eval.erode(
        np.asarray(cases["classical pair, two-sided"]["lo"]), 1,
        outside=robust_eval.OUTSIDE_TRACE_AT_Y_LO)[:, -1].sum())
    print(f"    (1) stub ROOT after over-etch: lo {n_w} -> {root_lo} cells, "
          f"hi {n_w} -> {root_hi} cells   "
          f"(lo under the WRONG (hi) convention: {n_w} -> {wrong_lo} "
          f"= DETACHED)")
    if not (root_lo == root_hi == n_w - 2) or wrong_lo != 0:
        print("  !! a stub root does not survive the over-etch")
        ok = False

    # (2) D3 -- dilation must NOT fill the trace-adjacent row.
    #     Broken: _step ORed the metal boundary in, filling the whole row --
    #     94 cells = 11.9 mm, shorting every stub root together.
    row_lo = int(np.asarray(br["dilated"]["lo"])[:, -1].sum())
    row_hi = int(np.asarray(br["dilated"]["hi"])[:, 0].sum())
    print(f"    (2) trace-adjacent row after under-etch: lo {row_lo}, "
          f"hi {row_hi} of {box.lo.nx} cells   (ORing the metal boundary in "
          f"gave the full {box.lo.nx}-cell "
          f"{box.lo.nx*box.lo.dx*1e3:.1f} mm row)")
    if not (row_lo == row_hi == n_w + 2):
        print("  !! the under-etch field fills (or fails to widen) the "
              "trace-adjacent row")
        ok = False

    # (3) D1 -- a pad hard against the OUTER walls must be eroded there.
    #     Broken: shrink = ~dilate(~m) assumed metal on all four edges, so the
    #     corner pad went 9 cells -> 4 instead of 9 -> 1.
    pad_mask = box.empty_mask()
    pad_mask["lo"][0:3, 0:3] = 1              # x_lo AND the outer y edge
    pad_br = etch_fields(pad_mask, box, cells=1)
    n_pad = int(pad_br["eroded"]["lo"].sum())
    n_pad_broken = int(robust_eval.erode(
        np.asarray(pad_mask["lo"]), 1, outside=(1, 1, 1, 1)).sum())
    print(f"    (3) 3x3 pad against x_lo and the outer y wall: 9 -> {n_pad} "
          f"cells   (all-metal-outside erosion gave 9 -> {n_pad_broken})")
    if n_pad != 1 or bool(pad_br["eroded"]["lo"][0, :].any()):
        print("  !! a pad against the outer box wall is not eroded there")
        ok = False

    # (4) D1 -- a full-depth stub must lose its TIP at the outer wall.
    #     Broken: it kept the tip entirely (70 -> 70 cells).
    deep = mask_from_stubs([("lo", x_mid, W_TRACE, box.lo.ny * box.lo.dx),
                            ("hi", x_mid, W_TRACE, box.hi.ny * box.hi.dx)], box)
    deep_br = etch_fields(deep, box, cells=1)
    # the CENTRE column of the stub -- the outer ones are eaten by the erosion
    _cols = np.flatnonzero(np.asarray(deep["lo"]).sum(axis=1))
    i_mid = int(_cols[_cols.size // 2])
    len_lo = int(np.asarray(deep_br["eroded"]["lo"])[i_mid, :].sum())
    tip_lo = bool(np.asarray(deep_br["eroded"]["lo"])[:, 0].any())
    tip_hi = bool(np.asarray(deep_br["eroded"]["hi"])[:, -1].any())
    print(f"    (4) full-depth stub: {box.lo.ny} -> {len_lo} cells; tip row at "
          f"the outer wall still metal? lo={tip_lo} hi={tip_hi} "
          f"(both must be False; the broken erosion kept "
          f"{box.lo.ny} -> {box.lo.ny})")
    if len_lo != box.lo.ny - 1 or tip_lo or tip_hi:
        print("  !! the outer-wall tip of a full-depth stub is not eroded")
        ok = False

    # ---- preflight with design metal, against a matched control ------------
    print("\n-- preflight with design metal (pathway control) --")
    fx2 = build_sim(freqs)
    n_added = add_pec_boxes(
        fx2.sim, boxes_from_mask(cases["classical pair, two-sided"], box))
    pre2 = _preflight_records(fx2.sim)
    new2 = sorted({c for c, _, _ in pre2} - {c for c, _, _ in pre1})

    # Control: the SAME shape class of geometry (per-column 1-cell-wide PEC
    # boxes) on the ORIGINAL one-sided fixture.
    sim0c, y0c, ty_hi0c, _, _ = build_original_sim(freqs)
    grid0c = sim0c._build_grid()
    ctrl_boxes = (_stub_column_boxes(grid0c, ty_hi0c, x_mid - x_off,
                                     W_TRACE, l_lo, outward=+1)
                  + _stub_column_boxes(grid0c, ty_hi0c, x_mid + x_off,
                                       W_TRACE, l_hi, outward=+1))
    add_pec_boxes(sim0c, ctrl_boxes)
    pre0c = _preflight_records(sim0c)
    new0c = sorted({c for c, _, _ in pre0c} - {c for c, _, _ in pre0})

    print(f"  two-sided + {n_added} column boxes : {len(pre2)} message(s); "
          f"codes added vs empty = {new2}")
    print(f"  ORIGINAL  + {len(ctrl_boxes)} column boxes : {len(pre0c)} "
          f"message(s); codes added vs empty = {new0c}")
    if new2 == new0c and set(new2) <= PATHWAY_PREFLIGHT_CODES:
        print(f"  -> the added codes {new2} are PATHWAY-INTRINSIC: the "
              f"validated one-sided fixture emits exactly the same set when "
              f"given per-column boxes. 1-cell-wide columns and 1-3 cell gaps "
              f"are what a pixel design IS; xval1_imperative.mask_to_boxes "
              f"produces them too. Not a two-sided regression.")
    else:
        print("\n" + "!" * 78)
        print("!! DESIGN-METAL PREFLIGHT DIVERGES FROM THE CONTROL")
        print(f"!!   two-sided added {new2}")
        print(f"!!   original  added {new0c}")
        for c, sev, m in pre2:
            if c in set(new2) - set(new0c):
                print(f"!!   [{sev}:{c}] {m}")
        print("!" * 78 + "\n")
        ok = False

    print("\n" + "=" * 78)
    print(f"SMOKE: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_smoke())
