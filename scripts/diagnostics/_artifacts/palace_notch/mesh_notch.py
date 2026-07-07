"""Gmsh mesh generator for the microstrip open-stub notch filter — Palace FEM.

Frequency-domain FEM (Palace) model of a quarter-wave open-stub notch filter,
built to compare against the two FDTD references (rfx + a sibling FDTD) whose
cross-solver fixtures lock this exact geometry (mm units):

    substrate  eps_r = 3.66  h = 0.254 mm  (LOSSLESS — matches both FDTD refs)
    trace      50-ohm-ish microstrip, w = 0.6 mm, runs the FULL x extent
    open stub  w = 0.6 mm, L = 12.0 mm, branches +y from the trace edge

Palace convention (mirrors the sibling patch-antenna REF_mesh_patch.py):
  * metal = zero-thickness PEC surfaces embedded at the substrate/air interface
    (z = H_SUB) and the substrate bottom face (z = 0 ground plane);
  * the drive is a LUMPED PORT: a small vertical rectangle from ground (z=0) up
    to the strip (z=H_SUB), width = strip width, Direction +Z (strip above gnd);
  * the outer box faces except the ground plane are an absorbing (first-order)
    boundary = the far-field / radiation truncation.

PORT ORIENTATION DEVIATION FROM REF (documented):
  REF placed its single lumped port ON the -x domain wall (x = X_LINE_LO), the
  far end of the feed line. Here the trace runs the FULL x extent [0,7] and
  touches BOTH x walls (both FDTD frames run the line through the absorber), so
  the two ports are INTERIOR vertical sheets at x = 1.0 and x = 6.0, each still
  a ground->strip rectangle with Direction +Z — the identical REF port PATTERN,
  just imprinted at an interior x instead of on the boundary wall.

Coordinate frame (mm): origin at the box corner, z=0 = ground / substrate bottom.

Physical groups (tag -> Palace attribute):
    1 substrate_vol   3 gnd (PEC)     5 port1 (LumpedPort 1)   7 farfield (Absorbing)
    2 air_vol         4 metal (PEC)   6 port2 (LumpedPort 2)

Run:  python3 mesh_notch.py [--out palace_notch.msh]
Writes the mesh (msh 2.2 for MFEM), prints node/element counts, the
physical-group table, and geometric verification checks (PASS/FAIL).
Does NOT run Palace.
"""

import argparse
import os

import gmsh

# --- geometry (mm) — locked to the committed cross-solver fixtures -----------
X_LO, X_HI = 0.0, 7.0
Y_LO, Y_HI = 0.0, 16.232
H_SUB = 0.254                 # substrate thickness (z of the metal interface)
Z_TOP = 1.754
AIR_H = Z_TOP - H_SUB         # 1.5

EPS_SUB = 3.66               # substrate permittivity (lossless)

# trace: full x extent, width 0.6 centred at y=1.208
TR_Y_LO, TR_Y_HI = 0.908, 1.508
# open stub: width 0.6 centred at x=3.5, length 12 from the trace edge (+y)
ST_X_LO, ST_X_HI = 3.2, 3.8
ST_Y_LO, ST_Y_HI = 1.508, 13.508
# lumped-port planes (interior), each ground->trace, width = trace width
PORT1_X = 1.0
PORT2_X = 6.0
P_Y_LO, P_Y_HI = TR_Y_LO, TR_Y_HI   # port spans the trace footprint in y

# mesh sizing (mm)
LC_MIN = 0.12          # on metal edges / ports (>=2 layers through H_SUB=0.254)
LC_MAX = 0.9           # air / far box
DIST_MIN = 0.30
DIST_MAX = 3.0
LC_SUB = 0.12          # thin substrate slab under the metal footprint

# physical-group tags = Palace attributes
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


def _vport(x):
    """Build a vertical ground->strip port rectangle in the x=const plane."""
    occ = gmsh.model.occ
    p = [occ.addPoint(x, P_Y_LO, 0.0),
         occ.addPoint(x, P_Y_HI, 0.0),
         occ.addPoint(x, P_Y_HI, H_SUB),
         occ.addPoint(x, P_Y_LO, H_SUB)]
    ls = [occ.addLine(p[0], p[1]), occ.addLine(p[1], p[2]),
          occ.addLine(p[2], p[3]), occ.addLine(p[3], p[0])]
    return occ.addPlaneSurface([occ.addCurveLoop(ls)])


def build(out_path):
    gmsh.initialize()
    gmsh.model.add("palace_notch")
    occ = gmsh.model.occ

    # --- volumes: substrate slab + air box stacked on top ---
    sub = occ.addBox(X_LO, Y_LO, 0.0, X_HI - X_LO, Y_HI - Y_LO, H_SUB)
    air = occ.addBox(X_LO, Y_LO, H_SUB, X_HI - X_LO, Y_HI - Y_LO, AIR_H)

    # --- metal (zero-thickness surfaces at z = H_SUB): trace + open stub ---
    trace = occ.addRectangle(X_LO, TR_Y_LO, H_SUB, X_HI - X_LO, TR_Y_HI - TR_Y_LO)
    stub = occ.addRectangle(ST_X_LO, ST_Y_LO, H_SUB, ST_X_HI - ST_X_LO,
                            ST_Y_HI - ST_Y_LO)

    # --- lumped-port sheets (interior vertical rectangles, ground->strip) ---
    port1 = _vport(PORT1_X)
    port2 = _vport(PORT2_X)

    # --- fragment everything so the mesh is conformal and the metal / port
    #     surfaces are embedded (imprinted) into the volume faces ---
    occ.fragment([(3, sub), (3, air)],
                 [(2, trace), (2, stub), (2, port1), (2, port2)])
    occ.synchronize()

    # --- re-identify entities by location (fragment renumbers tags) ---
    vols = gmsh.model.getEntities(3)
    assert len(vols) == 2, f"expected 2 volumes, got {vols}"
    sub_vol = _bbox(X_LO, Y_LO, 0.0, X_HI, Y_HI, H_SUB, dim=3, eps=0.02)
    air_vol = _bbox(X_LO, Y_LO, H_SUB, X_HI, Y_HI, Z_TOP, dim=3, eps=0.02)
    assert len(sub_vol) == 1 and len(air_vol) == 1, (sub_vol, air_vol)

    # metal faces at z = H_SUB: union over the trace + stub footprints
    # (fragment splits the trace at the port-touch chords x=1,6 and elsewhere)
    footprints = {
        "trace": (X_LO, TR_Y_LO, X_HI, TR_Y_HI),
        "stub":  (ST_X_LO, ST_Y_LO, ST_X_HI, ST_Y_HI),
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
    port1_face = _bbox(PORT1_X, P_Y_LO, 0.0, PORT1_X, P_Y_HI, H_SUB, dim=2, eps=1e-2)
    port2_face = _bbox(PORT2_X, P_Y_LO, 0.0, PORT2_X, P_Y_HI, H_SUB, dim=2, eps=1e-2)
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
    fd.setNumber(1, "Sampling", 400)
    fd.add("Threshold", 2)
    fd.setNumber(2, "InField", 1)
    fd.setNumber(2, "SizeMin", LC_MIN)
    fd.setNumber(2, "SizeMax", LC_MAX)
    fd.setNumber(2, "DistMin", DIST_MIN)
    fd.setNumber(2, "DistMax", DIST_MAX)
    # keep the thin substrate slab resolved (>=2 layers through H_SUB) under
    # the whole metal footprint
    fd.add("Box", 3)
    fd.setNumber(3, "VIn", LC_SUB)
    fd.setNumber(3, "VOut", LC_MAX)
    fd.setNumber(3, "XMin", X_LO - 0.1)
    fd.setNumber(3, "XMax", X_HI + 0.1)
    fd.setNumber(3, "YMin", TR_Y_LO - 0.4)
    fd.setNumber(3, "YMax", ST_Y_HI + 0.4)
    fd.setNumber(3, "ZMin", -0.05)
    fd.setNumber(3, "ZMax", H_SUB + 0.05)
    fd.setNumber(3, "Thickness", 0.6)
    fd.add("Min", 4)
    fd.setNumbers(4, "FieldsList", [2, 3])
    fd.setAsBackgroundMesh(4)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", LC_MIN)
    gmsh.option.setNumber("Mesh.MeshSizeMax", LC_MAX)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # Delaunay
    gmsh.option.setNumber("Mesh.SaveAll", 0)       # only physical groups
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 0)

    gmsh.model.mesh.generate(3)

    # MFEM's Gmsh reader: dedup nodes + msh 2.2 (4.1 multi-entity node blocks
    # trip 'vertices indices are not unique')
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
    print(f"  nodes        : {n_nodes}")
    print(f"  tets (3D)    : {n_tets}")
    print(f"  tris (2D bnd): {n_tris}")
    print("\n  physical groups (dim tag name -> #entities, #tris|#tets):")
    grp_tris = {}
    for (dim, tag) in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        ents = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        n_el = 0
        for e in ents:
            _, et, _ = gmsh.model.mesh.getElements(dim, e)
            n_el += sum(len(t) for t in et)
        if dim == 2:
            grp_tris[name] = n_el
        kind = "tris=" if dim == 2 else "tets="
        print(f"    {dim}  {tag:<2} {name:<14} entities={len(ents):<3} "
              f"{kind}{n_el}")

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

    exp_sub_v = (X_HI - X_LO) * (Y_HI - Y_LO) * H_SUB          # 28.86
    exp_air_v = (X_HI - X_LO) * (Y_HI - Y_LO) * AIR_H          # 170.4
    exp_gnd_a = (X_HI - X_LO) * (Y_HI - Y_LO)                  # 113.6
    # trace 7*0.6=4.2 + stub 12*0.6=7.2, abut at y=1.508 (no area overlap) => 11.4
    exp_metal_a = 4.2 + 7.2                                    # 11.4
    exp_port_a = (P_Y_HI - P_Y_LO) * H_SUB                     # 0.1524

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
    chk("metal area (trace+stub)", metal_a, exp_metal_a)
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
    print(f"\n  closed-boundary check:")
    print(f"    exterior faces          : {len(exterior)}")
    print(f"    covered by gnd|farfield : {len(covered & exterior)}")
    print(f"    ungrouped exterior      : {sorted(ungrouped)}")
    print(f"    gnd&farfield overlap    : {sorted(overlap)}")
    print(f"    metal/port on exterior  : {sorted(ports_on_ext)}")
    print(f"    [{'PASS' if ok_closed else 'FAIL'}] every exterior face in "
          f"exactly one of {{gnd, farfield}}")

    n_layers = H_SUB / LC_MIN
    print(f"\n  through-substrate resolution near strip: "
          f"H_SUB/LC_MIN = {H_SUB}/{LC_MIN} = {n_layers:.2f} layers "
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
                             "palace_notch.msh"))
    args = ap.parse_args()
    return build(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
