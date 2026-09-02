"""Cross-solver validation 15: probe-fed rectangular microstrip patch antenna
on RT/Duroid 5880 (thick low-eps laminate, uniform mesh).

rfx vs openEMS (local /usr/bin/openEMS), CPU-only. Validates the RESONANT
frequency (S11 dip) against openEMS and an in-script analytic anchor, and
reports a best-effort broadside directivity (NTFF).

COMPARATOR-FIRST (mandatory)
----------------------------
`mode=tutorial` reproduces openEMS's OWN canonical "Simple Patch Antenna"
tutorial (Thorsten Liebig, shipped with openEMS; 32x40 mm patch on eps_r=3.38 /
1.524 mm, probe-fed -6 mm off centre, finite 60x60 mm ground plane). The
known-good result for that geometry (openEMS with its thirds-rule edge meshing,
recorded in rfx's own examples/tutorials/patch_antenna_demo.py) is
  f_res(S11 dip) ~ 2.42 GHz,  broadside directivity ~ 6.8 dBi.
The tutorial mode prints openEMS's re-computed f_res + Dmax so the SETUP is
confirmed before any cross-comparison.

CHOSEN COMPARISON GEOMETRY (thick low-eps, CPU-affordable, #325-safe)
--------------------------------------------------------------------
The canonical thin (1.524 mm) substrate forces a uniform dx <= 0.38 mm (>=4
substrate cells) -> ~2M cells and a high radiation-Q ring-down: not CPU-tractable
here. Per the task's explicit option ("~2.4 GHz patch on a low-eps substrate"),
the comparison patch is on a THICK LOW-EPS laminate so a uniform dx = h/4 stays
coarse enough for CPU while keeping >=4 substrate cells:

  substrate : Rogers RT/Duroid 5880  eps_r = 2.2, tan_delta = 0.001, h = 3.175 mm
              (1/8 inch, a standard stock thickness)
  patch     : L = 40.0 mm (resonant, x) x W = 50.0 mm (radiating edge, y)
  ground    : 56 x 66 mm finite PEC plane (patch + ~8 mm skirt)
  feed      : probe/lumped 50 ohm, ez, 9 mm off centre along L (inset feed)

  Analytic (Balanis TL, re-derived in-script): see f_res_analytic() -> ~2.42 GHz.

The SAME geometry + materials are built in BOTH solvers (identical comparator).
"Line-fed" in the task title is realised as the canonical Simple_Patch
probe/lumped inset feed so the comparator geometry is byte-for-byte identical in
both tools (openEMS AddLumpedPort <-> rfx add_port, ez).

#325 AVOIDANCE (mandatory)
--------------------------
UNIFORM grid only (no dz_profile). dx = dy = dz = h/4 = 793.75 um. The rfx build
ASSERTS the substrate rasterizes to exactly N_SUB=4 z-cells (by index, from the
same position_to_index mapping the solver uses) BEFORE any f0 is quoted -- this
is the uniform-mesh form of the plan's mandatory graded-mesh rasterization
check. It covers the substrate's z EXTENT only, NOT which node plane the
bounding PEC walls realize -- see #740 GEOMETRY FIDELITY below, the check that
covers the wall plane.

#740 GEOMETRY FIDELITY (mandatory) -- realized wall PLANES, not Box extents
----------------------------------------------------------------------------
Issue #740: the #325 check above kept passing while the ground wall sat one
full cell BELOW the declared substrate floor (the #693 "vacuum ground cell"
trap, closed on the canonical patch lane by PRs #716/#718) -- a one-cell
one-plane PEC Box (the default, #677-validated) presents an electric wall on
its LOWER node plane only, so the ground Box here realized its wall at
z_sub_lo - DX, not z_sub_lo, leaving a live VACUUM cell in the cavity
(measured: +55.0% electrical thickness). Fixed with two_plane=True on the
GROUND Box ONLY (issue #706, the #693-precedent remedy: 0.0% electrical-
thickness error measured against the mask-derived realized planes) -- NOT on
the patch, whose default one-plane wall already sits on its LOWER face at
z_sub_hi (two_plane there would add an unreferenced wall the openEMS
zero-thickness-patch reference has no counterpart for).
``assert_realized_stack()`` (called from ``run_rfx()``) asserts the
REALIZED electric-wall planes -- ground at z_sub_lo, patch at z_sub_hi --
across the whole patch-footprint, not the declared Box coordinates, and
refuses to quote f0 if they disagree. ``compare()`` re-verifies the measured
planes against this module's own constants (not a recorded label) before
gating the f0 comparison.

rfx f0 = ring-down Harminv (NOT the |S11| dip)
----------------------------------------------
The rfx SINGLE-CELL lumped port has a known parasitic cell reactance that gives
a shallow, poorly-defined |S11| dip (~-3 dB here; documented in
validation/crossval/05_patch_antenna.py and examples/tutorials/patch_antenna_demo.py).
So the PRIMARY rfx resonance is the ring-down Harminv frequency (clean,
port-calibration independent); the |S11| dip is reported as a secondary /
passivity check and its local minimum is located near the ring-down frequency.
openEMS's lumped-port |S11| dip is deep (~-20 dB) and used directly.

HONEST SCOPE (PI penalises overclaiming)
----------------------------------------
  - PRIMARY gate: resonant f0 rfx(ring-down) vs openEMS(S11 dip) within a stated
    coarse-mesh envelope; also vs the analytic anchor. The coarse-mesh bias
    DIRECTION is geometry-dependent and is NOT pre-assumed: on the PRE-#740-FIX
    one-plane-ground realization (5-cell electrical gap, preserved as
    ``_15_patch_results/rfx_one_plane_ground_<commit>.json`` -- see the #740
    CHANGELOG entry for its numbers), rfx read +6.09% vs openEMS's declared
    stack; staircase PEC edges + a coarse 4-cell substrate under-resolve the
    patch-edge fringing capacitance in the SAME direction, while openEMS
    (thirds-rule edge meshing) reads lower and drifts down with refinement; the
    thin-substrate demo shows the opposite low bias. Post-#740-fix (two_plane
    ground, 4-cell electrical gap matching the declared stack), MEASURED and
    committed as ``_15_patch_results/rfx.json`` (CPU, 206.5 s): rfx
    f_primary 2.3139 GHz vs openEMS 2.330 GHz -- 0.69% and now LOW, not
    HIGH; rfx -4.21% vs analytic, openEMS -3.54%, i.e. both solvers sit on
    the same side of the closed form by a similar margin. The vacuum ground
    cell, not edge fringing, was the dominant term in the +6.09%. All of
    this is discretisation, reported not hidden, and always against the
    STRUCTURE actually solved, not the declared one.
  - Open (CPML) domain -> the -40 dB ring-down settling witness IS required and
    is printed BEFORE any frequency (unlike a closed cavity).
  - Broadside gain: order-of-magnitude / few-dB envelope only. The CPU-tight
    domain places the NTFF box well inside lambda/2 of the radiator; preflight's
    NTFF advisories are quoted verbatim and the number is labelled accordingly
    (#303: run() skips the NTFF check family -> we call preflight(check_ntff=True)
    explicitly).
  - dB is always recomputed from raw |S|. Deltas are rfx-centric. |S11|>1 is
    flagged as an extraction/passivity artifact, not physics.

RELATIONSHIP TO CASE 05 (patch ACCURACY is NOT claimed here)
------------------------------------------------------------
`validation/crossval/05_patch_antenna.py` remains the registered patch case and
authoritative patch-accuracy evidence stays DELEGATED to the committed tests it
names (`tests/crossval/test_patch_canonical_farfield_e4.py` and friends). This case 15 is
a SEPARATE, differently-meshed integration study on a different laminate: it
does not supersede id 05, does not reopen the #325/#378 demotion decision, and
adds no patch-accuracy claim of its own.

Exit contract (crossval registry, validation/crossval/manifest.json)
-------------------------------------------------------------------
    0 -> every configured gate passed (f0 envelope, settling, passivity,
         broadside-D envelope)
    1 -> a gate failed, or the rfx result leg is missing (execution gap)
    2 -> the openEMS reference is unavailable (missing result file, or
         CSXCAD/openEMS not importable for a solver mode): inconclusive

Usage:
    python validation/crossval/15_patch_antenna_rt5880.py            # default: compare (gates)
    python validation/crossval/15_patch_antenna_rt5880.py tutorial   # comparator-first: openEMS Simple_Patch
    python validation/crossval/15_patch_antenna_rt5880.py openems    # chosen patch in openEMS -> _15_patch_results/openems.json
    python validation/crossval/15_patch_antenna_rt5880.py rfx        # chosen patch in rfx     -> _15_patch_results/rfx.json
    python validation/crossval/15_patch_antenna_rt5880.py compare    # load both, gate the comparison
"""
import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Resolve `import rfx` from THIS checkout: a bare run would otherwise pick up
# whatever rfx is installed or first on sys.path, which in a multi-worktree
# setup is a DIFFERENT copy of the solver than the one under test.
REPO_ROOT = str(Path(__file__).resolve().parents[2])
RES_DIR = os.path.join(SCRIPT_DIR, "_15_patch_results")
os.makedirs(RES_DIR, exist_ok=True)
C0 = 2.99792458e8
EPS0 = 8.8541878128e-12

# ---------------------------------------------------------------------------
# Chosen comparison geometry (metres). Thick low-eps => CPU-affordable uniform mesh.
# ---------------------------------------------------------------------------
EPS_R = 2.2                 # RT/Duroid 5880
TAN_DELTA = 1.0e-3
H_SUB = 3.175e-3            # 1/8 inch
SIGMA_SUB = 2 * math.pi * 2.4e9 * EPS0 * EPS_R * TAN_DELTA   # ~ conductivity for tanδ
L_PATCH = 40.0e-3          # resonant length (x)
W_PATCH = 50.0e-3          # radiating width (y)
GP_X = 56.0e-3
GP_Y = 66.0e-3
FEED_OFFSET_X = -9.0e-3    # inset feed, 9 mm off centre along L

N_SUB = 4
DX = H_SUB / N_SUB         # 793.75 um uniform (H_SUB == 4*DX exactly)
N_CPML = 8                 # CPML pads OUTSIDE `domain` (rfx Grid convention)
AIR_LAT = 8.0e-3
AIR_BELOW = 10 * DX        # integer cells -> clean substrate rasterization
AIR_ABOVE = 15.0e-3

# `domain` is the physical INTERIOR only; rfx adds cpml_layers as padding outside.
DOM_X = GP_X + 2 * AIR_LAT
DOM_Y = GP_Y + 2 * AIR_LAT
DOM_Z = AIR_BELOW + H_SUB + AIR_ABOVE

F_DESIGN = 2.4e9
F_LO, F_HI = 1.6e9, 3.4e9   # S11 search / sweep band


def f_res_analytic():
    """Balanis transmission-line model (Antenna Theory, Ch.14), re-derived.

    eps_eff = (er+1)/2 + (er-1)/2 (1 + 12 h/W)^-1/2
    dL      = 0.412 h (eps_eff+0.3)(W/h+0.264) / [(eps_eff-0.258)(W/h+0.8)]
    f_r     = c / (2 (L + 2 dL) sqrt(eps_eff))
    Resonant dimension is L; W is the radiating-edge width.
    """
    er, h, L, W = EPS_R, H_SUB, L_PATCH, W_PATCH
    eps_eff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / W) ** -0.5
    dL = 0.412 * h * ((eps_eff + 0.3) * (W / h + 0.264)) / \
        ((eps_eff - 0.258) * (W / h + 0.8))
    L_eff = L + 2 * dL
    fr = C0 / (2 * L_eff * math.sqrt(eps_eff))
    return fr, eps_eff, dL


def _geom_banner():
    fr, eps_eff, dL = f_res_analytic()
    print(f"Patch (RT/Duroid 5880): eps_r={EPS_R}, tan_delta={TAN_DELTA}, "
          f"h={H_SUB*1e3:.3f} mm")
    print(f"  L(resonant)={L_PATCH*1e3:.1f} mm  W(radiating)={W_PATCH*1e3:.1f} mm  "
          f"GP {GP_X*1e3:.0f}x{GP_Y*1e3:.0f} mm  feed offset {FEED_OFFSET_X*1e3:+.1f} mm")
    print(f"  analytic (Balanis TL): eps_eff={eps_eff:.3f}, dL={dL*1e3:.3f} mm, "
          f"f_res={fr/1e9:.4f} GHz")
    print(f"  domain (x,y,z) = {DOM_X*1e3:.1f} x {DOM_Y*1e3:.1f} x {DOM_Z*1e3:.1f} mm, "
          f"uniform dx={DX*1e6:.2f} um (N_SUB={N_SUB})")


# ===========================================================================
# rfx side
# ===========================================================================
def assert_realized_stack(sim, grid, patch_shape):
    """MANDATORY geometry-fidelity self-check (issue #740): assert the
    REALIZED electric-wall PLANES the solver actually built match the
    declared stack, not the declared Box extents.

    This is the check the module docstring's "#325 AVOIDANCE (mandatory)"
    section does NOT cover: that section (see the ``[#325 CHECK]`` print in
    ``run_rfx`` below) asserts the substrate's z EXTENT rasterizes to
    ``N_SUB`` cells, and it kept passing throughout issue #740 while the
    ground wall sat one full cell BELOW the declared substrate floor -- a
    check named "geometry fidelity" that covers one axis and not the wall
    PLANE is worse than none, because a passing check gets read as
    evidence. This function is the wall-plane check.

    A one-cell one-plane PEC ``Box`` (``rfx.api.__init__.add()``'s default,
    #677-validated) presents an electric wall on its LOWER node plane only
    (rfx.boundaries.pec.tangential_edge_masks: the tangential Ex/Ey edges
    zeroed at the cell's own z-index), so the ground Box here -- occupying
    the cell BELOW ``z_sub_lo`` -- realizes its wall one cell below the
    declared substrate floor unless flagged ``two_plane=True`` (issue #706,
    the #693-precedent remedy applied in ``run_rfx``), which also
    zeroes the NEXT node plane and lands the wall exactly at ``z_sub_lo``.

    Evaluated over the WHOLE patch-cap substrate footprint (every (i, j)
    cell ``patch_shape`` rasterizes to -- the patch lies entirely inside
    the substrate's footprint, so patch-footprint == patch-intersect-
    substrate-footprint here), not one feed column, so an in-plane
    rasterization edge case cannot hide.

    Reads the PRE-port-clearing ``pec_mask`` (``sim._assemble_materials``,
    called before ``run()``'s per-port live-cell clearing runs). The feed
    port sits at z in [``z_sub_lo`` + DX, ``z_sub_lo`` + 3*DX] here (see
    ``run_rfx``'s ``port_z0``/``port_extent``), off BOTH the ground
    wall plane (``z_sub_lo``) and the patch wall plane (``z_sub_hi``)
    checked below, so port clearing cannot move this check's verdict.

    Raises RuntimeError (refusing to quote f0) if either wall plane is
    missing across the footprint. Returns a dict of the MEASURED planes for
    the caller to embed verbatim in the result leg, so ``compare()`` can
    re-verify them against the module's OWN constants independently rather
    than trust a recorded label (issue #740 review, required change 1).
    """
    from rfx.boundaries.pec import tangential_edge_masks, two_plane_extension_masks

    mats, _, _, pec_mask, *_ = sim._assemble_materials(grid)
    if pec_mask is None:
        raise RuntimeError(
            "assert_realized_stack: no PEC cells rasterized -- refuse to "
            "quote f0 for a patch antenna with no conductor")
    two_plane_mask = sim._two_plane_cell_mask(grid)
    ex, ey, _ = tangential_edge_masks(pec_mask)
    if two_plane_mask is not None:
        ex2, ey2, _ = two_plane_extension_masks(pec_mask, two_plane_mask)
        ex = ex | ex2
        ey = ey | ey2
    ex = np.asarray(ex)
    ey = np.asarray(ey)
    eps = np.asarray(mats.eps_r)

    # Footprint from the shape's OWN rasterization (not a recomputed index
    # range): avoids an off-by-one at the box edge from re-deriving (i, j)
    # bounds via ``position_to_index`` on the corner coordinates separately.
    footprint = np.asarray(patch_shape.mask(grid)).any(axis=2)

    def _full_wall(k):
        return bool(np.all(ex[:, :, k][footprint])) and \
            bool(np.all(ey[:, :, k][footprint]))

    z_sub_lo = AIR_BELOW
    z_sub_hi = AIR_BELOW + H_SUB
    k_ground = grid.position_to_index((DOM_X / 2, DOM_Y / 2, z_sub_lo))[2]
    k_patch = grid.position_to_index((DOM_X / 2, DOM_Y / 2, z_sub_hi))[2]
    ground_ok = _full_wall(k_ground)
    patch_ok = _full_wall(k_patch)

    if not (ground_ok and patch_ok):
        realized_wall_k = [k for k in range(ex.shape[2]) if _full_wall(k)]
        raise RuntimeError(
            "assert_realized_stack: realized electric-wall plane(s) do not "
            f"match the declared stack -- no electric wall at "
            f"z_sub_lo={z_sub_lo*1e3:.4f} mm (k={k_ground}, ok={ground_ok}) "
            f"and/or z_sub_hi={z_sub_hi*1e3:.4f} mm (k={k_patch}, "
            f"ok={patch_ok}); realized full-footprint wall planes at "
            f"k={realized_wall_k} "
            f"(z~{[round(k*DX*1e3, 4) for k in realized_wall_k]} mm, pad "
            "included). Refuse to quote f0 for a cavity taller than the "
            "declared substrate (issue #740).")

    n_sub_cells = k_patch - k_ground
    eps_between = [float(np.mean(eps[:, :, k][footprint]))
                   for k in range(k_ground, k_patch)]
    print(f"\n[STACK CHECK #740] realized walls: ground z={z_sub_lo*1e3:.4f} "
          f"mm (k={k_ground}), patch z={z_sub_hi*1e3:.4f} mm (k={k_patch}); "
          f"n_sub_cells={n_sub_cells} (intended {N_SUB}); eps_between="
          f"{['%.3f' % e for e in eps_between]} (intended {EPS_R})")

    return dict(
        ground_wall_z=z_sub_lo, patch_wall_z=z_sub_hi,
        n_sub_cells=n_sub_cells, eps_between=eps_between,
        # Recorded PROVENANCE only -- the gate above (and compare()'s
        # re-check) is on the MEASURED planes, not this label, so a later
        # realization landing the same walls by a different mechanism does
        # not need this string to match to pass.
        ground_realization="two_plane" if two_plane_mask is not None else "one_plane",
    )


def build_rfx_sim(*, do_gain: bool = False, two_plane: bool = True):
    """Build cv15's rfx Simulation WITHOUT solving it -- the production
    geometry, feed, probe and (optionally) NTFF box exactly as ``run_rfx``
    uses them. Returns ``(sim, patch_shape, geom)`` with ``geom`` the
    derived z-planes and feed location the caller needs.

    Separated from ``run_rfx`` for the #740 review: the first version of
    this fix kept build and solve fused, so the wall-plane tests had to
    MIRROR the geometry in a test-local copy -- and deleting
    ``two_plane=True`` from the production script left every test green.
    With the builder separable, the tests exercise THIS function with the
    production toggle, and the example-fidelity contract gate audits cv15
    (it was classified builder_fused_with_solve before; now ``audited``).

    ``two_plane`` is the #740 fix itself and defaults to the fixed value;
    the negative-control test passes ``two_plane=False`` to reproduce the
    pre-fix one-plane ground and confirm ``assert_realized_stack`` rejects
    it through the production path.
    """
    from rfx import Simulation, Box, GaussianPulse
    from rfx.boundaries.spec import BoundarySpec

    cx, cy = DOM_X / 2, DOM_Y / 2
    z_sub_lo = AIR_BELOW                          # top face of ground plane
    # Lattice-arithmetic spelling (#802): z_sub_hi is INTENDED on-lattice
    # (AIR_BELOW = 10 cells, H_SUB == N_SUB*DX exactly), but the sum
    # AIR_BELOW + H_SUB lands one f64 ulp ABOVE the node value 14*DX, so
    # under exact node coordinates the patch's lo corner would miss its
    # node (the Box docstring's knife-edge class). Same real number,
    # bit-exact spelling.
    z_sub_hi = (10 + N_SUB) * DX                  # == AIR_BELOW + H_SUB
    z_patch_lo, z_patch_hi = z_sub_hi, z_sub_hi + DX
    feed_x = cx + FEED_OFFSET_X

    sim = Simulation(
        freq_max=4e9, domain=(DOM_X, DOM_Y, DOM_Z), dx=DX,
        boundary=BoundarySpec.uniform("cpml"), cpml_layers=N_CPML,
    )
    sim.add_material("sub", eps_r=EPS_R, sigma=SIGMA_SUB)
    # finite ground plane (1 cell), substrate (4 cells), patch (1 cell).
    #
    # #740 (the #693 "vacuum ground cell" trap, closed on the canonical
    # patch lane by PRs #716/#718): a one-cell one-plane PEC Box's electric
    # wall sits on its LOWER node plane only (rfx/api/__init__.py add()
    # docstring), so this ground Box's realized wall would land at
    # z_sub_lo - DX -- one cell BELOW the declared substrate floor, leaving
    # a live VACUUM cell between the wall and z_sub_lo (measured: +55.0% of
    # the cavity's electrical thickness). two_plane=True (issue #706, the
    # #693-precedent remedy) also zeroes the ground body's UPPER node
    # plane, landing the wall exactly at z_sub_lo -- 0.0% versus declared.
    # GROUND ONLY: two_plane on the patch would add an unreferenced wall
    # one cell ABOVE z_sub_hi (measured 11.9062 mm) that the openEMS
    # reference's zero-thickness patch has no counterpart for -- the
    # patch's default one-plane wall already sits on its LOWER face, which
    # IS z_sub_hi, so the patch Box is left at the one-plane default.
    sim.add(Box((cx - GP_X / 2, cy - GP_Y / 2, z_sub_lo - DX),
                (cx + GP_X / 2, cy + GP_Y / 2, z_sub_lo)), material="pec",
            two_plane=two_plane)
    sim.add(Box((cx - GP_X / 2, cy - GP_Y / 2, z_sub_lo),
                (cx + GP_X / 2, cy + GP_Y / 2, z_sub_hi)), material="sub")
    patch_shape = Box((cx - L_PATCH / 2, cy - W_PATCH / 2, z_patch_lo),
                      (cx + L_PATCH / 2, cy + W_PATCH / 2, z_patch_hi))
    sim.add(patch_shape, material="pec")

    # probe/lumped feed: sit ~1.5 cells above the substrate floor so the port
    # cell does NOT land in the ground-plane PEC (verified via preflight).
    port_z0 = z_sub_lo + 1.0 * DX
    port_extent = 2.0 * DX                       # cells strictly between GP & patch
    sim.add_port(position=(feed_x, cy, port_z0), component="ez",
                 impedance=50.0, extent=port_extent,
                 waveform=GaussianPulse(f0=F_DESIGN, bandwidth=1.0))
    # ring-down witness probe (offset from feed, inside substrate)
    sim.add_probe(position=(feed_x + 5e-3, cy + 5e-3, z_sub_lo + 1.5 * DX),
                  component="ez")

    if do_gain:
        # Box = the interior region exactly (corner at cpml_layers*dx clears the
        # CPML-overlap preflight; the lambda/4 face advisories still fire -- the
        # CPU-tight domain cannot clear them, so rfx D is order-of-magnitude only).
        pad = N_CPML * DX
        sim.add_ntff_box(corner_lo=(pad, pad, pad),
                         corner_hi=(DOM_X - pad, DOM_Y - pad, DOM_Z - pad),
                         freqs=np.array([2.2e9, 2.3e9, 2.4e9, 2.5e9]))
    geom = dict(z_sub_lo=z_sub_lo, z_sub_hi=z_sub_hi, feed_x=feed_x, cy=cy)
    return sim, patch_shape, geom


def run_rfx(num_periods, n_freqs, do_gain, *, two_plane: bool = True):
    sys.path.insert(0, REPO_ROOT)
    import io
    import contextlib
    import jax.numpy as jnp

    print("=" * 72)
    print("rfx side (uniform mesh, #325-safe)")
    print("=" * 72)
    _geom_banner()

    sim, patch_shape, _geom = build_rfx_sim(do_gain=do_gain, two_plane=two_plane)
    z_sub_lo, z_sub_hi = _geom["z_sub_lo"], _geom["z_sub_hi"]

    # ---- Build the actual grid: exact dt + FAITHFUL substrate rasterization ----
    grid = sim._build_grid()
    # Count z Yee cells the substrate Box actually rasterizes to, from the SAME
    # index mapping the solver uses (uniform-mesh form of the #325 check).
    # NOTE (#740): this is the docstring's "#325 AVOIDANCE (mandatory)"
    # self-check -- it covers the z EXTENT of the rasterization only, and
    # kept passing throughout #740 while the wall PLANE was displaced a
    # full cell. assert_realized_stack() below is the check that covers the
    # wall plane; this one stays because the z-extent question (does the
    # substrate rasterize to exactly N_SUB cells at all) is still real and
    # distinct from where its bounding walls land.
    k_lo = int(round(z_sub_lo / DX))
    k_hi = int(round(z_sub_hi / DX))
    n_sub_raster = k_hi - k_lo
    print(f"\n[#325 CHECK] substrate rasterizes to {n_sub_raster} z-cells "
          f"(intended {N_SUB}); z_sub=[{z_sub_lo*1e3:.3f},{z_sub_hi*1e3:.3f}] mm; "
          f"grid shape={grid.shape}, dt={float(grid.dt)*1e12:.3f} ps")
    if n_sub_raster != N_SUB:
        raise RuntimeError(
            f"substrate landed on {n_sub_raster} cells, intended {N_SUB} -- "
            "uniform-mesh rasterization mismatch, refuse to quote f0")

    stack_check = assert_realized_stack(sim, grid, patch_shape)
    dt_grid = float(grid.dt)

    # preflight verbatim (explicit NTFF check family, #303)
    print("\n--- rfx preflight (verbatim) ---")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.preflight(strict=False, check_ntff=True)
    preflight_txt = buf.getvalue()
    print(preflight_txt if preflight_txt.strip() else "(no preflight output)")
    print("--- end preflight ---\n")

    # n_steps = num_periods AT THE DESIGN FREQ (run(num_periods=...) would count
    # periods at freq_max=4 GHz and under-run the 2.4 GHz ring-down). Pass n_steps
    # explicitly so the main field run AND the S-param run share the same length,
    # i.e. the settling witness reflects exactly the record that produced S11.
    n_steps = int(num_periods * (1.0 / F_DESIGN) / dt_grid)
    freqs = jnp.linspace(F_LO, F_HI, n_freqs)
    print(f"running rfx ({num_periods} periods @ {F_DESIGN/1e9:.1f} GHz "
          f"-> {n_steps} steps, n_freqs={n_freqs}) ...")
    t0 = time.time()
    res = sim.run(n_steps=n_steps, compute_s_params=True,
                  s_param_freqs=freqs, s_param_n_steps=n_steps)
    dt_run = time.time() - t0
    print(f"  done in {dt_run:.1f} s")

    S = np.asarray(res.s_params)
    s11 = S[0, 0, :]
    f = np.asarray(freqs)

    # settling witness (open CPML domain, -40 dB bar) -- BEFORE any frequency
    ts = np.asarray(res.time_series).ravel()
    dt_ts = float(res.dt)
    env = np.abs(ts)
    peak = float(env[int(len(env) * 0.15):].max()) if len(env) else 0.0
    tail = float(env[int(len(env) * 0.95):].max()) if len(env) else 0.0
    settle_db = 20 * math.log10(tail / peak) if (peak > 0 and tail > 0) else float("-inf")
    settled = settle_db < -40.0
    print(f"SETTLING WITNESS (open CPML, bar -40 dB): end/peak = {settle_db:.1f} dB "
          f"-> {'SETTLED' if settled else 'UNDER-SETTLED (f0 carries truncation error)'}")

    # PRIMARY rfx f0 = ring-down Harminv (clean, port-calibration independent).
    # The rfx single-cell lumped-port |S11| dip is SHALLOW (parasitic cell
    # reactance -> monotonic background, documented in validation/crossval/05 and
    # examples/tutorials/patch_antenna_demo.py), so it is reported as SECONDARY
    # and its local dip is located NEAR the ring-down frequency, not by a global
    # argmin (which lands on the band-edge background).
    fr_an = f_res_analytic()[0]
    f_harminv, q_harminv = _harminv_f0(ts, dt_ts)
    f_center = f_harminv if f_harminv else fr_an
    s11_db = 20 * np.log10(np.maximum(np.abs(s11), 1e-9))
    lo = int(np.searchsorted(f, f_center * 0.94))
    hi = int(np.searchsorted(f, f_center * 1.06))
    idx = lo + int(np.argmin(s11_db[max(lo, 0):max(hi, lo + 1)]))
    f_dip = float(f[idx])
    s11_dip_db = float(s11_db[idx])
    max_abs = float(np.max(np.abs(s11)))
    f_primary = f_harminv if f_harminv else f_dip

    d_dbi = None
    if do_gain:
        d_dbi = _rfx_gain(res)

    out = dict(
        solver="rfx", dx_um=DX * 1e6, n_sub_cells=stack_check["n_sub_cells"],
        num_periods=num_periods, n_steps=n_steps, runtime_s=dt_run,
        settle_db=settle_db, settled=bool(settled),
        freqs_hz=f.tolist(), s11_mag=np.abs(s11).tolist(),
        s11_re=np.real(s11).tolist(), s11_im=np.imag(s11).tolist(),
        f_primary_hz=f_primary,
        f_dip_hz=f_dip, s11_dip_db=s11_dip_db, max_abs_s11=max_abs,
        f_harminv_hz=f_harminv, q_harminv=q_harminv,
        f_analytic_hz=fr_an,
        gain_dbi=d_dbi, preflight=preflight_txt,
        stack_check=stack_check,
    )
    with open(os.path.join(RES_DIR, "rfx.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\n[rfx] PRIMARY f0 (ring-down Harminv) = "
          f"{f_harminv/1e9 if f_harminv else float('nan'):.4f} GHz "
          f"(Q={q_harminv if q_harminv else float('nan'):.1f}) | "
          f"S11 local dip {f_dip/1e9:.4f} GHz @ {s11_dip_db:.2f} dB (shallow, "
          f"secondary) | analytic {fr_an/1e9:.4f} GHz | max|S11|={max_abs:.3f}"
          + (f" | D={d_dbi:.2f} dBi (order-of-mag)" if d_dbi is not None else ""))
    print(f"saved {os.path.join(RES_DIR, 'rfx.json')}")


def _harminv_f0(ts, dt_ts):
    from rfx.harminv import harminv
    fr_an = f_res_analytic()[0]
    sig = ts[int(len(ts) * 0.30):]
    try:
        modes = harminv(sig, dt_ts, F_LO, F_HI)
    except Exception as e:
        print(f"  (harminv failed: {e})")
        return None, None
    good = [m for m in modes if m.Q > 2 and m.amplitude > 1e-9]
    if not good:
        return None, None
    good.sort(key=lambda m: abs(m.freq - fr_an))
    return float(good[0].freq), float(good[0].Q)


def _rfx_gain(res):
    """Best-effort broadside directivity. Labelled order-of-magnitude: the
    CPU-tight domain places the NTFF faces inside lambda/2 (preflight flags it)."""
    try:
        from rfx import compute_far_field_jax, directivity
        theta = np.linspace(0, np.pi, 91)
        phi = np.linspace(0, 2 * np.pi, 25)
        ff = compute_far_field_jax(res.ntff_data, res.ntff_box, res.grid, theta, phi)
        d = np.asarray(directivity(ff))
        # broadside = theta ~ 0; report the 2.4 GHz bin (index 2 of the 4 bins)
        return float(d[2])
    except Exception as e:
        print(f"  (rfx gain skipped: {e})")
        return None


# ===========================================================================
# openEMS side
# ===========================================================================
def _require_openems():
    """Exit 2 (visible SKIP) if the openEMS reference solver is unavailable.

    CSXCAD + openEMS are NOT rfx dependencies, so a machine without them cannot
    produce the reference. That is an inconclusive crossval, never a pass.
    """
    try:
        import CSXCAD  # noqa: F401
        import openEMS  # noqa: F401
    except ImportError as exc:
        print("\n" + "!" * 72)
        print("!! CROSSVAL 15 SKIPPED — CSXCAD / openEMS not installed")
        print(f"!!   reason: {exc}")
        print("!!   the openEMS reference cannot be produced, so this run is NOT")
        print("!!   a crossval. Exiting 2 (inconclusive) so CI does not read it")
        print("!!   as PASS.")
        print("!" * 72)
        raise SystemExit(2)


def _oems_numpy_shim():
    _require_openems()
    for _n in ("float", "int", "complex"):
        if not hasattr(np, _n):
            setattr(np, _n, {"float": float, "int": int, "complex": complex}[_n])


def _run_tutorial_once(mesh_div):
    """Build+run openEMS's verbatim canonical Simple_Patch at lambda/mesh_div and
    return (n_cells, f_dip_hz, s11_dip_db, Dmax_dbi, Rin_at_dip)."""
    _oems_numpy_shim()
    import tempfile
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from openEMS.physical_constants import EPS0 as _EPS0, C0 as _C0

    patch_w, patch_l = 32.0, 40.0          # mm (32 = resonant length, x)
    sub_epsR = 3.38
    sub_kappa = 1e-3 * 2 * np.pi * 2.45e9 * _EPS0 * sub_epsR
    sub_w = sub_l = 60.0
    sub_th, sub_cells = 1.524, 4
    feed_pos, feed_R = -6.0, 50.0
    SimBox = np.array([200, 200, 150])
    f0, fc = 2e9, 1e9

    FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['MUR'] * 6)
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(1e-3)
    mesh_res = _C0 / (f0 + fc) / 1e-3 / mesh_div
    mesh.AddLine('x', [-SimBox[0] / 2, SimBox[0] / 2])
    mesh.AddLine('y', [-SimBox[1] / 2, SimBox[1] / 2])
    mesh.AddLine('z', [-SimBox[2] / 3, SimBox[2] * 2 / 3])
    patch = CSX.AddMetal('patch')
    patch.AddBox(priority=10, start=[-patch_w / 2, -patch_l / 2, sub_th],
                 stop=[patch_w / 2, patch_l / 2, sub_th])
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res / 2)
    substrate = CSX.AddMaterial('substrate', epsilon=sub_epsR, kappa=sub_kappa)
    substrate.AddBox(priority=0, start=[-sub_w / 2, -sub_l / 2, 0],
                     stop=[sub_w / 2, sub_l / 2, sub_th])
    mesh.AddLine('z', np.linspace(0, sub_th, sub_cells + 1))
    gnd = CSX.AddMetal('gnd')
    gnd.AddBox([-sub_w / 2, -sub_l / 2, 0], [sub_w / 2, sub_l / 2, 0], priority=10)
    FDTD.AddEdges2Grid(dirs='xy', properties=gnd)
    port = FDTD.AddLumpedPort(1, feed_R, [feed_pos, 0, 0], [feed_pos, 0, sub_th],
                              'z', 1.0, priority=5, edges2grid='xy')
    mesh.SmoothMeshLines('all', mesh_res, 1.4)
    nf2ff = FDTD.CreateNF2FFBox()
    nx = len(mesh.GetLines('x')); ny = len(mesh.GetLines('y')); nz = len(mesh.GetLines('z'))
    sim_path = os.path.join(tempfile.gettempdir(), f'patch05_tut_{mesh_div}')
    FDTD.Run(sim_path, cleanup=True, numThreads=8)
    f = np.linspace(max(1e9, f0 - fc), f0 + fc, 401)
    port.CalcPort(sim_path, f)
    s11 = port.uf_ref / port.uf_inc
    s11_db = 20 * np.log10(np.abs(s11))
    Zin = port.uf_tot / port.if_tot
    i = int(np.argmin(s11_db))
    f_res = float(f[i])
    Dmax = float("nan")
    if s11_db[i] < -10:
        theta = np.arange(-180.0, 180.0, 2.0)
        nf = nf2ff.CalcNF2FF(sim_path, f_res, theta, [0., 90.], center=[0, 0, 1e-3])
        Dmax = float(10 * np.log10(nf.Dmax[0]))
    return nx * ny * nz, f_res, float(s11_db[i]), Dmax, float(Zin[i].real)


def run_openems_tutorial():
    """COMPARATOR-FIRST: openEMS's OWN verbatim canonical Simple_Patch tutorial,
    at two mesh densities to show the resonance CONVERGES toward the documented
    ~2.42 GHz (rfx examples/tutorials/patch_antenna_demo.py records openEMS-fine
    f_res=2.4221 GHz, D=6.79 dBi). The coarse-mesh downshift is a shared FDTD
    trait -- documenting it here is the whole point of running the tutorial
    first: it confirms the setup AND calibrates the coarse-mesh envelope."""
    print("=" * 72)
    print("COMPARATOR-FIRST: openEMS canonical Simple_Patch_Antenna tutorial")
    print("  (32x40 mm patch, eps_r=3.38, h=1.524 mm, feed -6 mm, GP 60x60 mm)")
    print("=" * 72)
    rows = []
    for div in (20, 40):
        t0 = time.time()
        n_cells, f_res, s11db, Dmax, Rin = _run_tutorial_once(div)
        dt = time.time() - t0
        rows.append((div, n_cells, f_res, s11db, Dmax, Rin, dt))
        print(f"  lambda/{div:<3d} ({n_cells:>7d} cells, {dt:4.0f}s): "
              f"S11 dip {f_res/1e9:.4f} GHz @ {s11db:6.2f} dB | R_in~{Rin:5.1f} ohm"
              + (f" | Dmax {Dmax:.2f} dBi" if not math.isnan(Dmax) else ""))
    f_coarse, f_fine = rows[0][2], rows[1][2]
    conv_up = f_fine > f_coarse
    near_doc = abs(f_fine / 1e9 - 2.42) < 0.15
    matched = rows[0][3] < -10 and 30 < rows[0][5] < 80    # deep dip + ~50 ohm R
    print(f"\n  convergence: {f_coarse/1e9:.3f} -> {f_fine/1e9:.3f} GHz as mesh "
          f"halves ({'UP toward ~2.42' if conv_up else 'NOT converging up'})")
    ok = matched and conv_up and (near_doc or f_fine / 1e9 > 2.25)
    print("  SETUP CONFIRMED: clean matched patch resonance converging toward the "
          "documented ~2.42 GHz / ~6.8 dBi" if ok else
          "  !! tutorial did NOT reproduce a converging matched resonance -- "
          "check openEMS install")


def run_openems(n_freqs, do_gain):
    """The CHOSEN comparison patch, built identically to the rfx side."""
    _oems_numpy_shim()
    import tempfile
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from CSXCAD.SmoothMeshLines import SmoothMeshLines

    print("=" * 72)
    print("openEMS side (chosen comparison patch, identical geometry)")
    print("=" * 72)
    _geom_banner()

    unit = 1e-3
    Lp, Wp = L_PATCH * 1e3, W_PATCH * 1e3
    h = H_SUB * 1e3
    gpx, gpy = GP_X * 1e3, GP_Y * 1e3
    feed = FEED_OFFSET_X * 1e3
    f0, fc = 2.4e9, 1.2e9

    FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['MUR'] * 6)
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)
    mesh_res = C0 / (f0 + fc) / unit / 30        # ~lambda/30 in air (well-resolved)
    patch_res = 1.2                              # dense lines across patch (converged ref)

    # air box: generous (openEMS is cheap) -> trustworthy far-field reference
    air = 60.0
    mesh.AddLine('x', [-gpx / 2 - air, gpx / 2 + air])
    mesh.AddLine('y', [-gpy / 2 - air, gpy / 2 + air])
    mesh.AddLine('z', [-40.0, 90.0])
    # explicit fine lines across the patch (+2 mm skirt) so openEMS is at least
    # as converged laterally as rfx's uniform 0.79 mm -> a fair reference.
    mesh.AddLine('x', np.arange(-Lp / 2 - 2, Lp / 2 + 2 + patch_res, patch_res))
    mesh.AddLine('y', np.arange(-Wp / 2 - 2, Wp / 2 + 2 + patch_res, patch_res))

    patch = CSX.AddMetal('patch')
    patch.AddBox(priority=10, start=[-Lp / 2, -Wp / 2, h], stop=[Lp / 2, Wp / 2, h])
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=patch_res / 2)
    sub_kappa = 2 * np.pi * 2.4e9 * 8.8541878128e-12 * EPS_R * TAN_DELTA
    substrate = CSX.AddMaterial('sub', epsilon=EPS_R, kappa=sub_kappa)
    substrate.AddBox(priority=0, start=[-gpx / 2, -gpy / 2, 0], stop=[gpx / 2, gpy / 2, h])
    mesh.AddLine('z', np.linspace(0, h, N_SUB + 1))
    gnd = CSX.AddMetal('gnd')
    gnd.AddBox([-gpx / 2, -gpy / 2, 0], [gpx / 2, gpy / 2, 0], priority=10)
    FDTD.AddEdges2Grid(dirs='xy', properties=gnd)
    port = FDTD.AddLumpedPort(1, 50.0, [feed, 0, 0], [feed, 0, h], 'z', 1.0,
                              priority=5, edges2grid='xy')
    mesh.SmoothMeshLines('all', mesh_res, 1.4)
    nf2ff = FDTD.CreateNF2FFBox() if do_gain else None

    nx = len(mesh.GetLines('x')); ny = len(mesh.GetLines('y')); nz = len(mesh.GetLines('z'))
    print(f"  openEMS mesh: {nx} x {ny} x {nz} = {nx*ny*nz/1e6:.2f} M cells "
          f"(mesh_res~{mesh_res:.2f} mm, {N_SUB} substrate cells)")

    sim_path = os.path.join(tempfile.gettempdir(), 'patch05_openems')
    t0 = time.time()
    FDTD.Run(sim_path, cleanup=True, numThreads=8)
    dt_run = time.time() - t0
    print(f"  openEMS run done in {dt_run:.1f} s")

    f = np.linspace(F_LO, F_HI, n_freqs)
    port.CalcPort(sim_path, f)
    s11 = port.uf_ref / port.uf_inc
    s11_db = 20 * np.log10(np.maximum(np.abs(s11), 1e-9))
    fr_an = f_res_analytic()[0]
    lo = int(np.searchsorted(f, fr_an * 0.80)); hi = int(np.searchsorted(f, fr_an * 1.20))
    idx = lo + int(np.argmin(s11_db[lo:hi]))
    f_dip = float(f[idx]); s11_dip_db = float(s11_db[idx])

    d_dbi = None
    if do_gain and s11_dip_db < -6:
        theta = np.arange(-180.0, 180.0, 2.0)
        nf = nf2ff.CalcNF2FF(sim_path, f_dip, theta, [0., 90.], center=[0, 0, 1e-3])
        d_dbi = float(10 * np.log10(nf.Dmax[0]))

    out = dict(
        solver="openems", mesh_res_mm=mesh_res, n_cells=nx * ny * nz, runtime_s=dt_run,
        freqs_hz=f.tolist(), s11_mag=np.abs(s11).tolist(),
        s11_re=np.real(s11).tolist(), s11_im=np.imag(s11).tolist(),
        f_dip_hz=f_dip, s11_dip_db=s11_dip_db, max_abs_s11=float(np.max(np.abs(s11))),
        f_analytic_hz=fr_an, gain_dbi=d_dbi,
    )
    with open(os.path.join(RES_DIR, "openems.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"\n[openems] S11 dip {f_dip/1e9:.4f} GHz @ {s11_dip_db:.2f} dB | "
          f"analytic {fr_an/1e9:.4f} GHz"
          + (f" | D={d_dbi:.2f} dBi" if d_dbi is not None else ""))
    print(f"saved {os.path.join(RES_DIR, 'openems.json')}")


# ===========================================================================
# compare
# ===========================================================================
def _load_legs():
    """Load the two committed result legs, or exit with the right status.

    Missing openEMS reference -> 2 (inconclusive, the reference is required for
    a PASS). Missing rfx leg   -> 1 (the rfx side never ran: execution gap).
    """
    rfx_path = os.path.join(RES_DIR, "rfx.json")
    oems_path = os.path.join(RES_DIR, "openems.json")
    if not os.path.isfile(rfx_path):
        print(f"!! CROSSVAL 15 FAILED — rfx result leg missing: {rfx_path}")
        print(f"!!   run `python {os.path.basename(__file__)} rfx` first. "
              "Exiting 1 (execution gap).")
        raise SystemExit(1)
    if not os.path.isfile(oems_path):
        print("!" * 72)
        print("!! CROSSVAL 15 SKIPPED — openEMS reference leg missing: "
              f"{oems_path}")
        print("!!   the reference is required for a PASS. Exiting 2 "
              "(inconclusive).")
        print("!" * 72)
        raise SystemExit(2)
    with open(rfx_path) as fp:
        R = json.load(fp)
    with open(oems_path) as fp:
        O = json.load(fp)
    return R, O


def _stack_check_ok(sc, tol=1e-9, eps_tol=1e-4):
    """Re-verify a leg's ``stack_check`` dict against THIS MODULE's OWN
    constants -- issue #740 review, required change 1: gate on the
    MEASURED planes recomputed against ``AIR_BELOW``/``H_SUB``/``N_SUB``/
    ``EPS_R``, not a recorded label. A missing ``stack_check`` (a leg from
    before this check existed) is a FAIL, not a skip: the #740 defect is
    exactly a leg that looks fine without it. ``ground_realization`` is
    NOT part of this test -- it is provenance only, so a later realization
    that lands the same walls by a different mechanism still passes.

    Pure function (synthetic dicts in, bool+detail out) so
    ``tests/crossval/test_crossval_cv15_wall_planes.py`` can pin the gate MATH
    without a solve, following ``tests/crossval/test_crossval_gate_logic.py``'s
    precedent for this crossval directory.
    """
    if not sc:
        return False, "missing stack_check (leg predates the #740 wall-plane self-check)"
    z_sub_lo = AIR_BELOW
    z_sub_hi = AIR_BELOW + H_SUB
    eps_between = sc.get("eps_between") or []
    ok = (
        abs(sc.get("ground_wall_z", float("nan")) - z_sub_lo) < tol
        and abs(sc.get("patch_wall_z", float("nan")) - z_sub_hi) < tol
        and sc.get("n_sub_cells") == N_SUB
        and len(eps_between) == N_SUB
        and all(abs(e - EPS_R) < eps_tol for e in eps_between)
    )
    detail = (
        f"ground_wall_z={sc.get('ground_wall_z')} (want {z_sub_lo:.6g}), "
        f"patch_wall_z={sc.get('patch_wall_z')} (want {z_sub_hi:.6g}), "
        f"n_sub_cells={sc.get('n_sub_cells')} (want {N_SUB}), "
        f"eps_between={eps_between} (want {N_SUB}x{EPS_R}); "
        f"ground_realization(provenance)={sc.get('ground_realization')}"
    )
    return ok, detail


def compare(f0_env_pct):
    R, O = _load_legs()
    fr_an = R["f_analytic_hz"]
    # rfx PRIMARY f0 = ring-down Harminv (clean); openEMS f0 = its deep S11 dip.
    f_rfx = R.get("f_primary_hz") or R["f_dip_hz"]
    f_oe = O["f_dip_hz"]
    f_rfx_dip = R["f_dip_hz"]

    print("=" * 72)
    print("Patch antenna  --  rfx vs openEMS cross-validation")
    print("=" * 72)
    print(f"rfx : uniform dx={R['dx_um']:.1f}um ({R['n_sub_cells']} sub-cells), "
          f"{R['num_periods']} periods, {R['runtime_s']:.0f}s, "
          f"settle={R['settle_db']:.1f}dB ({'SETTLED' if R['settled'] else 'UNDER-SETTLED'})")
    print(f"oems: mesh_res~{O['mesh_res_mm']:.2f}mm ({O['n_cells']/1e6:.2f}M cells), "
          f"{O['runtime_s']:.0f}s")
    nbad_r = int(np.sum(np.array(R["s11_mag"]) > 1.0))
    nbad_o = int(np.sum(np.array(O["s11_mag"]) > 1.0))
    print(f"max|S11|: rfx={R['max_abs_s11']:.3f} oems={O['max_abs_s11']:.3f}")
    if nbad_r or nbad_o:
        print(f"!! |S11|>1 bins (EXTRACTION/PASSIVITY artifact, not physics): "
              f"rfx={nbad_r} oems={nbad_o}")
    print()
    print("f0 method: rfx = ring-down Harminv (the rfx single-cell lumped-port S11")
    print("dip is shallow -> secondary); openEMS = its deep matched S11 dip.")
    print()
    print(f"{'quantity':<36}{'rfx':>11}{'openEMS':>11}{'analytic':>11}")
    print("-" * 69)
    print(f"{'resonant f0 [GHz]  (PRIMARY gate)':<36}{f_rfx/1e9:>11.4f}"
          f"{f_oe/1e9:>11.4f}{fr_an/1e9:>11.4f}")
    print(f"{'  rfx ring-down Q':<36}"
          f"{(R.get('q_harminv') or float('nan')):>11.1f}{'':>11}{'':>11}")
    print(f"{'  rfx S11 local dip [GHz] (2ndary)':<36}{f_rfx_dip/1e9:>11.4f}"
          f"{'':>11}{'':>11}")
    print(f"{'  S11 dip depth [dB]':<36}{R['s11_dip_db']:>11.1f}"
          f"{O['s11_dip_db']:>11.1f}{'':>11}")
    if R.get("gain_dbi") is not None or O.get("gain_dbi") is not None:
        gr = R.get("gain_dbi"); go = O.get("gain_dbi")
        print(f"{'broadside D [dBi] (envelope)':<36}"
              f"{(f'{gr:.2f}' if gr is not None else 'n/a'):>11}"
              f"{(f'{go:.2f}' if go is not None else 'n/a'):>11}{'':>11}")
    print("-" * 69)

    d_rfx_oe = abs(f_rfx - f_oe) / f_oe * 100
    d_rfx_an = abs(f_rfx - fr_an) / fr_an * 100
    d_oe_an = abs(f_oe - fr_an) / fr_an * 100
    print(f"\nrfx-vs-openEMS f0 method distance = {d_rfx_oe:.2f} %  "
          f"(rfx {'LOW' if f_rfx < f_oe else 'HIGH'} vs openEMS)")
    print(f"rfx-vs-analytic f0 distance       = {d_rfx_an:.2f} %  "
          f"(rfx {'LOW' if f_rfx < fr_an else 'HIGH'})")
    print(f"openEMS-vs-analytic f0 distance   = {d_oe_an:.2f} %  "
          f"(openEMS {'LOW' if f_oe < fr_an else 'HIGH'})")

    # --------------------------- GATES (fail-closed) ---------------------------
    all_ok = True

    def gate(name, passed, detail):
        nonlocal all_ok
        all_ok = all_ok and bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    print(f"\nGATES (coarse-mesh integration envelope: f0 rfx-vs-openEMS "
          f"<= {f0_env_pct:.0f}%)")
    print("  NOTE: this is a same-geometry INTEGRATION envelope, not a patch")
    print("  accuracy claim — accuracy stays delegated to case 05's committed")
    print("  tests (see the module banner).")
    gate("f0 agreement (rfx ring-down vs openEMS)", d_rfx_oe <= f0_env_pct,
         f"{d_rfx_oe:.2f}% <= {f0_env_pct:.0f}%")
    sc_ok, sc_detail = _stack_check_ok(R.get("stack_check"))
    gate("stack geometry fidelity (realized wall planes, #740)", sc_ok, sc_detail)
    gate("settling witness (open CPML, -40 dB bar)", bool(R["settled"]),
         f"{R['settle_db']:.1f} dB")
    # Passivity is GATED on both legs: |S11| > ~1.05 on a passive radiator is an
    # extraction defect, never physics (tests/test_sparam_passivity_guard.py).
    gate("rfx passivity (max|S11| <= 1.05)", R["max_abs_s11"] <= 1.05,
         f"max|S11| = {R['max_abs_s11']:.3f}")
    gate("openEMS passivity (max|S11| <= 1.05)", O["max_abs_s11"] <= 1.05,
         f"max|S11| = {O['max_abs_s11']:.3f}")
    if R.get("gain_dbi") is not None and O.get("gain_dbi") is not None:
        dg = abs(R["gain_dbi"] - O["gain_dbi"])
        gate("broadside D envelope (<= 3 dB, rfx near-field box)", dg <= 3.0,
             f"rfx {R['gain_dbi']:.2f} vs oems {O['gain_dbi']:.2f} dBi, "
             f"dD = {dg:.2f} dB")
    else:
        print("  [skip] broadside D envelope: one or both legs have no NTFF "
              "directivity (run with --gain)")

    # overlay figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fr = np.array(R["freqs_hz"]); s11r = np.array(R["s11_mag"])
        fo = np.array(O["freqs_hz"]); s11o = np.array(O["s11_mag"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fr / 1e9, 20 * np.log10(s11r + 1e-9), "C0-", label="|S11| rfx (uniform)")
        ax.plot(fo / 1e9, 20 * np.log10(s11o + 1e-9), "C3--", label="|S11| openEMS")
        ax.axvline(fr_an / 1e9, color="k", ls=":", alpha=0.6,
                   label=f"analytic {fr_an/1e9:.3f} GHz")
        ax.axvline(f_rfx / 1e9, color="C0", ls="-", lw=1.0, alpha=0.7,
                   label=f"rfx ring-down f0 {f_rfx/1e9:.3f} GHz")
        ax.axvline(f_oe / 1e9, color="C3", ls="--", lw=1.0, alpha=0.7,
                   label=f"openEMS S11 dip {f_oe/1e9:.3f} GHz")
        ax.set_xlabel("Frequency [GHz]"); ax.set_ylabel("|S11| [dB]")
        ax.set_ylim(-30, 2); ax.set_xlim(F_LO / 1e9, F_HI / 1e9)
        ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=8)
        ax.set_title(f"Patch S11: rfx vs openEMS  "
                     f"(f0 {f_rfx/1e9:.3f}/{f_oe/1e9:.3f} GHz, dip depth not gated)")
        p = os.path.join(RES_DIR, "patch_compare.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        print(f"\nsaved figure {p}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    print("\n" + ("ALL GATES PASSED" if all_ok else "SOME CHECKS FAILED"))
    print("Scope: same-geometry rfx-vs-openEMS integration study on RT/Duroid "
          "5880.\nPatch ACCURACY remains delegated to case 05's committed tests.")
    return all_ok


def main():
    ap = argparse.ArgumentParser(
        description="Probe-fed RT/Duroid 5880 patch, rfx vs openEMS "
                    "(diagnostic-reporter).")
    # `compare` is the default so a bare `python 15_patch_antenna_rt5880.py`
    # (how the crossval runner invokes every case) gates the committed
    # comparison instead of dying in argparse.
    ap.add_argument("mode", nargs="?", default="compare",
                    choices=["tutorial", "openems", "rfx", "compare"])
    ap.add_argument("--num-periods", type=float, default=70.0)
    ap.add_argument("--n-freqs", type=int, default=181)
    ap.add_argument("--gain", action="store_true", help="compute NTFF broadside directivity")
    ap.add_argument("--f0-env-pct", type=float, default=8.0)
    a = ap.parse_args()
    # Solver-producing modes write a result leg and are not themselves gated;
    # only `compare` evaluates the configured gates and can return 1.
    if a.mode == "tutorial":
        run_openems_tutorial()
        return 0
    if a.mode == "openems":
        run_openems(a.n_freqs, a.gain)
        return 0
    if a.mode == "rfx":
        run_rfx(a.num_periods, a.n_freqs, a.gain)
        return 0
    return 0 if compare(a.f0_env_pct) else 1


if __name__ == "__main__":
    raise SystemExit(main())
