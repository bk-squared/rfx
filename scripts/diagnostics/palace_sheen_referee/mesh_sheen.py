#!/usr/bin/env python3
"""Gmsh mesh generator for the Sheen 1990 microstrip LPF — Palace FEM referee.

Frequency-domain FEM (Palace) model of the classic Sheen low-pass filter, built
to REFEREE the two FDTD references (rfx + openEMS) whose cv07 cross-check
(``validation/crossval/_07_sheen_results/{rfx,openems}.json``) locks a ~1.4% first-null
split: rfx 7.874 GHz, openEMS 7.983 GHz (raw argmin bins, post-regeneration —
PR #468/#516; the earlier num_periods=20/default-offset leg read rfx 7.218 GHz,
a ~9.6% split). Both refs are staircased FDTD, so
neither resolves the wide-patch open-end fringing exactly. Palace on a conformal
tetrahedral mesh (no staircase) captures the fringing exactly, so it is the right
independent arbiter of whether rfx's null is the expected staircase-fringing
under-resolution.

Geometry (mm) — locked to the EXACT domain frame of validation/crossval/07_sheen_lpf.py
(propagation x, transverse y, stack z; Sheen board mapped rfx_x = Sheen_y):

    substrate  eps_r = 2.2   h = 0.794 mm   (LOSSLESS — matches both FDTD refs)
    50-ohm feed  W = 2.413 mm
    wide patch  20.320 mm (transverse) x 2.540 mm (propagation), low-Z section
    input feed  x [0, 12.466]  centred y = 9.8565   (runs into -x absorber wall)
    output feed x [15.006, 27.472] centred y = 16.4635 (runs into +x absorber wall)
    ports       two 50-ohm lumped sheets (ground->strip) at x = 2.5 and x = 24.972
    far box     first-order absorbing on all non-ground outer faces
    domain      27.472 x 26.320 x 3.794 mm  (air 3.0 mm above the trace)

Palace convention (mirrors the sibling palace_notch_referee/mesh_notch.py):
  * metal = zero-thickness PEC surfaces at the substrate/air interface (z=H_SUB)
    plus the substrate bottom face (z=0 ground plane);
  * the drive is a LUMPED PORT: a small vertical rectangle from ground (z=0) up to
    the strip (z=H_SUB), width = feed width, Direction +Z;
  * the outer box faces except the ground plane are first-order absorbing = the
    radiation truncation.

PORT ORIENTATION (same PATTERN as the notch referee): the two feeds run the full
x-extent into the ±x walls (both FDTD frames run the line through the absorber),
so the two ports are INTERIOR vertical sheets at x = 2.5 (input feed) and
x = 24.972 (output feed), each a ground->strip rectangle with Direction +Z. The
port location only sets the (de-embedded) reference plane phase; the FIRST-NULL
FREQUENCY of |S21| is a property of the filter transfer, invariant to it.

Physical groups (tag -> Palace attribute):
    1 substrate_vol   3 gnd (PEC)     5 port1 (LumpedPort 1)   7 farfield (Absorbing)
    2 air_vol         4 metal (PEC)   6 port2 (LumpedPort 2)

Run:  python3 mesh_sheen.py [--out palace_sheen.msh] [--lc-min ..] [--lc-sub ..] [--lc-max ..]
Coarse (default) and the sqrt2 "mid" refinement are produced by the SAME script
with different --lc-* (no source edit): the committed lanes use
    coarse: --lc-min 0.25 --lc-sub 0.30 --lc-max 1.60
    mid:    --lc-min 0.18 --lc-sub 0.21 --lc-max 1.20   (~1/sqrt2 refine)
Writes the mesh (msh 2.2 for MFEM), prints node/element counts, the physical-group
table, and geometric verification (PASS/FAIL). Does NOT run Palace.
"""

import argparse
import os

import gmsh

# --- geometry (mm) — locked to validation/crossval/07_sheen_lpf.py domain frame ---
X_LO, X_HI = 0.0, 27.472
Y_LO, Y_HI = 0.0, 26.320
H_SUB = 0.794                 # substrate thickness (z of the metal interface)
Z_TOP = 3.794
AIR_H = Z_TOP - H_SUB         # 3.0

EPS_SUB = 2.2                 # substrate permittivity (lossless)
W_FEED = 2.413

# metal footprints (z = H_SUB), domain frame:
PATCH_X0, PATCH_X1 = 12.466, 15.006
# input 50-ohm feed: full x [0, PATCH_X0], centred y = 9.8565
IN_FEED_YC = 9.8565
IN_Y_LO, IN_Y_HI = IN_FEED_YC - W_FEED / 2, IN_FEED_YC + W_FEED / 2   # 8.650, 11.063
# wide low-Z patch: x [PATCH_X0, PATCH_X1], y [PATCH_Y_LO, PATCH_Y_HI]
PATCH_Y_LO, PATCH_Y_HI = 3.0, 23.320
# output 50-ohm feed: full x [PATCH_X1, X_HI], centred y = 16.4635
OUT_FEED_YC = 16.4635
OUT_Y_LO, OUT_Y_HI = OUT_FEED_YC - W_FEED / 2, OUT_FEED_YC + W_FEED / 2  # 15.257, 17.670

# lumped-port planes (interior vertical sheets, ground->strip)
PORT1_X = 2.5                 # in the input feed
PORT2_X = 24.972             # in the output feed (X_HI - 2.5)

# mesh sizing (mm) — DEFAULTS = coarse lane; override on the CLI for the mid mesh
LC_MIN = 0.25          # on metal edges / ports (H_SUB/LC_MIN = 3.2 layers)
LC_MAX = 1.60          # air / far box
DIST_MIN = 0.40
DIST_MAX = 3.0
LC_SUB = 0.30          # thin substrate slab under the metal footprint

TAG = {
    "substrate_vol": 1,
    "air_vol": 2,
    "gnd": 3,
    "metal": 4,
    "port1": 5,
    "port2": 6,
    "farfield": 7,
}

EPS = 1e-3             # bbox query padding (mm)


def _bbox(x0, y0, z0, x1, y1, z1, dim=2, eps=EPS):
    """Tags of entities of `dim` whose bbox is contained in the padded box."""
    ents = gmsh.model.getEntitiesInBoundingBox(
        x0 - eps, y0 - eps, z0 - eps, x1 + eps, y1 + eps, z1 + eps, dim)
    return [t for (d, t) in ents]


def _vport(x, y_lo, y_hi):
    """Build a vertical ground->strip port rectangle in the x=const plane."""
    occ = gmsh.model.occ
    p = [occ.addPoint(x, y_lo, 0.0),
         occ.addPoint(x, y_hi, 0.0),
         occ.addPoint(x, y_hi, H_SUB),
         occ.addPoint(x, y_lo, H_SUB)]
    ls = [occ.addLine(p[0], p[1]), occ.addLine(p[1], p[2]),
          occ.addLine(p[2], p[3]), occ.addLine(p[3], p[0])]
    return occ.addPlaneSurface([occ.addCurveLoop(ls)])


def build(out_path, lc_min, lc_sub, lc_max):
    gmsh.initialize()
    gmsh.model.add("palace_sheen")
    occ = gmsh.model.occ

    # --- volumes: substrate slab + air box stacked on top ---
    sub = occ.addBox(X_LO, Y_LO, 0.0, X_HI - X_LO, Y_HI - Y_LO, H_SUB)
    air = occ.addBox(X_LO, Y_LO, H_SUB, X_HI - X_LO, Y_HI - Y_LO, AIR_H)

    # --- metal (zero-thickness surfaces at z = H_SUB): 3 footprints ---
    in_feed = occ.addRectangle(X_LO, IN_Y_LO, H_SUB, PATCH_X0 - X_LO, IN_Y_HI - IN_Y_LO)
    patch = occ.addRectangle(PATCH_X0, PATCH_Y_LO, H_SUB,
                             PATCH_X1 - PATCH_X0, PATCH_Y_HI - PATCH_Y_LO)
    out_feed = occ.addRectangle(PATCH_X1, OUT_Y_LO, H_SUB,
                                X_HI - PATCH_X1, OUT_Y_HI - OUT_Y_LO)

    # --- lumped-port sheets (interior vertical rectangles, ground->strip) ---
    port1 = _vport(PORT1_X, IN_Y_LO, IN_Y_HI)
    port2 = _vport(PORT2_X, OUT_Y_LO, OUT_Y_HI)

    # --- fragment so the mesh is conformal + metal / port sheets are embedded ---
    occ.fragment([(3, sub), (3, air)],
                 [(2, in_feed), (2, patch), (2, out_feed), (2, port1), (2, port2)])
    occ.synchronize()

    # --- re-identify entities by location (fragment renumbers tags) ---
    vols = gmsh.model.getEntities(3)
    assert len(vols) == 2, f"expected 2 volumes, got {vols}"
    sub_vol = _bbox(X_LO, Y_LO, 0.0, X_HI, Y_HI, H_SUB, dim=3, eps=0.02)
    air_vol = _bbox(X_LO, Y_LO, H_SUB, X_HI, Y_HI, Z_TOP, dim=3, eps=0.02)
    assert len(sub_vol) == 1 and len(air_vol) == 1, (sub_vol, air_vol)

    # metal faces at z = H_SUB: union over the 3 footprints (fragment splits the
    # metal at the port-touch chords and the feed/patch junctions)
    footprints = {
        "in_feed": (X_LO, IN_Y_LO, PATCH_X0, IN_Y_HI),
        "patch":   (PATCH_X0, PATCH_Y_LO, PATCH_X1, PATCH_Y_HI),
        "out_feed": (PATCH_X1, OUT_Y_LO, X_HI, OUT_Y_HI),
    }
    metal_faces = set()
    for name, (x0, y0, x1, y1) in footprints.items():
        f = _bbox(x0, y0, H_SUB, x1, y1, H_SUB, dim=2, eps=1e-2)
        assert f, f"no metal face found for {name}"
        metal_faces.update(f)

    # ground = bottom face z = 0
    gnd = _bbox(X_LO, Y_LO, 0.0, X_HI, Y_HI, 0.0, dim=2, eps=1e-2)
    assert gnd, "ground face not found"

    # port faces: interior vertical sheets at x = PORT{1,2}_X, z in [0, H_SUB]
    port1_face = _bbox(PORT1_X, IN_Y_LO, 0.0, PORT1_X, IN_Y_HI, H_SUB, dim=2, eps=1e-2)
    port2_face = _bbox(PORT2_X, OUT_Y_LO, 0.0, PORT2_X, OUT_Y_HI, H_SUB, dim=2, eps=1e-2)
    assert len(port1_face) == 1, f"port1 face query -> {port1_face}"
    assert len(port2_face) == 1, f"port2 face query -> {port2_face}"

    # farfield = the five non-ground outer planes (full z), minus metal/ports
    ff = set()
    ff.update(_bbox(X_LO, Y_LO, Z_TOP, X_HI, Y_HI, Z_TOP, dim=2, eps=1e-2))  # top
    ff.update(_bbox(X_HI, Y_LO, 0.0, X_HI, Y_HI, Z_TOP, dim=2, eps=1e-2))    # +x
    ff.update(_bbox(X_LO, Y_LO, 0.0, X_LO, Y_HI, Z_TOP, dim=2, eps=1e-2))    # -x
    ff.update(_bbox(X_LO, Y_HI, 0.0, X_HI, Y_HI, Z_TOP, dim=2, eps=1e-2))    # +y
    ff.update(_bbox(X_LO, Y_LO, 0.0, X_HI, Y_LO, Z_TOP, dim=2, eps=1e-2))    # -y
    ff -= metal_faces                        # metal is interior (defensive)
    ff.discard(port1_face[0])                # ports are interior (defensive)
    ff.discard(port2_face[0])
    ff -= set(gnd)                           # ground is its own group
    assert ff, "no farfield faces found"

    # --- physical groups (tag == Palace attribute) ---
    def pg(dim, tags, name):
        gmsh.model.addPhysicalGroup(dim, list(tags), TAG[name])
        gmsh.model.setPhysicalName(dim, TAG[name], name)

    pg(3, sub_vol, "substrate_vol")
    pg(3, air_vol, "air_vol")
    pg(2, gnd, "gnd")
    pg(2, sorted(metal_faces), "metal")
    pg(2, port1_face, "port1")
    pg(2, port2_face, "port2")
    pg(2, sorted(ff), "farfield")

    # --- mesh size fields: fine near metal + port edges ---
    edges = set()
    port_faces = [port1_face[0], port2_face[0]]
    for s in list(metal_faces) + port_faces:
        for (_, c) in gmsh.model.getBoundary([(2, s)], oriented=False):
            edges.add(abs(c))
    fd = gmsh.model.mesh.field
    fd.add("Distance", 1)
    fd.setNumbers(1, "CurvesList", sorted(edges))
    fd.setNumber(1, "Sampling", 600)
    fd.add("Threshold", 2)
    fd.setNumber(2, "InField", 1)
    fd.setNumber(2, "SizeMin", lc_min)
    fd.setNumber(2, "SizeMax", lc_max)
    fd.setNumber(2, "DistMin", DIST_MIN)
    fd.setNumber(2, "DistMax", DIST_MAX)
    # keep the thin substrate slab resolved (>=2 layers through H_SUB) under the
    # whole metal footprint
    fd.add("Box", 3)
    fd.setNumber(3, "VIn", lc_sub)
    fd.setNumber(3, "VOut", lc_max)
    fd.setNumber(3, "XMin", X_LO - 0.1)
    fd.setNumber(3, "XMax", X_HI + 0.1)
    fd.setNumber(3, "YMin", PATCH_Y_LO - 0.4)
    fd.setNumber(3, "YMax", PATCH_Y_HI + 0.4)
    fd.setNumber(3, "ZMin", -0.05)
    fd.setNumber(3, "ZMax", H_SUB + 0.05)
    fd.setNumber(3, "Thickness", 0.6)
    fd.add("Min", 4)
    fd.setNumbers(4, "FieldsList", [2, 3])
    fd.setAsBackgroundMesh(4)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_max)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # Delaunay
    gmsh.option.setNumber("Mesh.SaveAll", 0)       # only physical groups
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 0)

    gmsh.model.mesh.generate(3)

    # MFEM's Gmsh reader: dedup nodes + msh 2.2 (4.1 multi-entity node blocks trip
    # 'vertices indices are not unique')
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(out_path)

    # --- report ---
    ntag, _, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(ntag)
    _, etags, _ = gmsh.model.mesh.getElements(3)
    n_tets = sum(len(t) for t in etags)
    _, stags, _ = gmsh.model.mesh.getElements(2)
    n_tris = sum(len(t) for t in stags)

    print(f"\nwrote {out_path}")
    print(f"  lc_min={lc_min} lc_sub={lc_sub} lc_max={lc_max}")
    print(f"  nodes        : {n_nodes}")
    print(f"  tets (3D)    : {n_tets}")
    print(f"  tris (2D bnd): {n_tris}")
    print(f"  ~order-2 DOF : {int(round(n_tets * 6.6))}")
    print("\n  physical groups (dim tag name -> #entities, #tris|#tets):")
    for (dim, tag) in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        ents = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        n_el = 0
        for e in ents:
            _, et, _ = gmsh.model.mesh.getElements(dim, e)
            n_el += sum(len(t) for t in et)
        kind = "tris=" if dim == 2 else "tets="
        print(f"    {dim}  {tag:<2} {name:<14} entities={len(ents):<3} {kind}{n_el}")

    # --- geometric verification (areas / volumes via OCC mass) ---
    def area(tags):
        return sum(occ.getMass(2, t) for t in tags)

    def volume(tags):
        return sum(occ.getMass(3, t) for t in tags)

    sub_v = volume(sub_vol)
    air_v = volume(air_vol)
    gnd_a = area(gnd)
    metal_a = area(sorted(metal_faces))
    p1_a = area(port1_face)
    p2_a = area(port2_face)

    exp_sub_v = (X_HI - X_LO) * (Y_HI - Y_LO) * H_SUB
    exp_air_v = (X_HI - X_LO) * (Y_HI - Y_LO) * AIR_H
    exp_gnd_a = (X_HI - X_LO) * (Y_HI - Y_LO)
    exp_metal_a = (PATCH_X0 - X_LO) * (IN_Y_HI - IN_Y_LO) \
        + (PATCH_X1 - PATCH_X0) * (PATCH_Y_HI - PATCH_Y_LO) \
        + (X_HI - PATCH_X1) * (OUT_Y_HI - OUT_Y_LO)
    exp_port_a = (IN_Y_HI - IN_Y_LO) * H_SUB

    print("\n  geometric verification (value vs expected, tol 1%):")
    checks = []

    def chk(name, got, exp, tol=0.01):
        ok = abs(got - exp) <= tol * abs(exp)
        checks.append(ok)
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:<24} "
              f"got={got:.4f}  exp={exp:.4f}  rel={abs(got-exp)/abs(exp):.2e}")

    chk("substrate_vol volume", sub_v, exp_sub_v)
    chk("air_vol volume", air_v, exp_air_v)
    chk("gnd area", gnd_a, exp_gnd_a)
    chk("metal area (3 footprints)", metal_a, exp_metal_a)
    chk("port1 area", p1_a, exp_port_a)
    chk("port2 area", p2_a, exp_port_a)

    # --- closed-boundary check: every exterior face is in exactly one of
    #     {gnd, farfield}; ports/metal are interior; nothing ungrouped ---
    exterior = set()
    exterior.update(_bbox(X_LO, Y_LO, 0.0, X_HI, Y_HI, 0.0, dim=2, eps=1e-2))   # z=0
    exterior.update(_bbox(X_LO, Y_LO, Z_TOP, X_HI, Y_HI, Z_TOP, dim=2, eps=1e-2))  # z=top
    exterior.update(_bbox(X_HI, Y_LO, 0.0, X_HI, Y_HI, Z_TOP, dim=2, eps=1e-2))  # +x
    exterior.update(_bbox(X_LO, Y_LO, 0.0, X_LO, Y_HI, Z_TOP, dim=2, eps=1e-2))  # -x
    exterior.update(_bbox(X_LO, Y_HI, 0.0, X_HI, Y_HI, Z_TOP, dim=2, eps=1e-2))  # +y
    exterior.update(_bbox(X_LO, Y_LO, 0.0, X_HI, Y_LO, Z_TOP, dim=2, eps=1e-2))  # -y
    covered = set(gnd) | ff
    ungrouped = exterior - covered
    overlap = set(gnd) & ff
    ports_on_ext = ({port1_face[0], port2_face[0]} | metal_faces) & exterior
    ok_closed = (not ungrouped) and (not overlap) and (not ports_on_ext)
    checks.append(ok_closed)
    print("\n  closed-boundary check:")
    print(f"    exterior faces          : {len(exterior)}")
    print(f"    covered by gnd|farfield : {len(covered & exterior)}")
    print(f"    ungrouped exterior      : {sorted(ungrouped)}")
    print(f"    gnd&farfield overlap    : {sorted(overlap)}")
    print(f"    metal/port on exterior  : {sorted(ports_on_ext)}")
    print(f"    [{'PASS' if ok_closed else 'FAIL'}] every exterior face in "
          f"exactly one of {{gnd, farfield}}")

    n_layers = H_SUB / lc_min
    print(f"\n  through-substrate resolution near strip: "
          f"H_SUB/lc_min = {H_SUB}/{lc_min} = {n_layers:.2f} layers "
          f"({'>=2 OK' if n_layers >= 2 else 'FAIL <2'})")

    all_pass = all(checks) and n_layers >= 2
    print(f"\n  === {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'} ===")

    gmsh.finalize()
    return 0 if all_pass else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "palace_sheen.msh"))
    ap.add_argument("--lc-min", type=float, default=LC_MIN)
    ap.add_argument("--lc-sub", type=float, default=LC_SUB)
    ap.add_argument("--lc-max", type=float, default=LC_MAX)
    args = ap.parse_args()
    return build(args.out, args.lc_min, args.lc_sub, args.lc_max)


if __name__ == "__main__":
    raise SystemExit(main())
