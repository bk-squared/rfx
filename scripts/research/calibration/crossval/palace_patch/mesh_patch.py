"""Gmsh mesh generator for the X-band inset-fed patch — Palace FEM arbiter.

Third solver (Palace, FEM in the frequency domain) for the S11 dip that rfx
(FDTD) and openEMS (FDTD) disagree on:  rfx ~10.1 GHz vs openEMS ~9.26 GHz,
both ~ -12 dB.  Geometry is the exact design from
scripts/research/calibration/crossval/rfx_patch_inset_xband.py (issue80 frame, inset feed):

    RO4003C  eps_r=3.38  h=0.787 mm
    patch    L=8.595 mm (x, resonant)  x  W=10.129 mm (y)
    feed     50-ohm microstrip w=1.8 mm, L_line=12 mm before the patch edge
    inset    depth d=2.4 mm through two etched notch slots, gap g=0.9 mm/side

Palace convention modelled here (see Palace examples/antenna, examples/cpw):
  * metal = zero-thickness PEC surfaces embedded at the substrate/air interface
    (z = h) and on the substrate bottom face (z = 0 ground plane);
  * the drive is a LUMPED PORT: a small vertical rectangle at the far end of
    the feed line (x = -L_line), from ground (z=0) up to the strip (z=h),
    width = strip width; direction +Z (strip is above ground);
  * the radiation box outer faces (all except the ground plane) are an
    absorbing (first-order) boundary = the far-field truncation.

Coordinate frame (mm): origin at the patch leading (-x, radiating) edge,
y centred on 0, z=0 at the ground plane / substrate bottom.

Physical groups (tag -> Palace attribute):
    1 substrate_vol   3 gnd (PEC)     5 port (LumpedPort)
    2 air_vol         4 metal (PEC)   6 farfield (Absorbing)

Run:  pip install gmsh; python mesh_patch.py [--out palace_patch.msh]
Writes the mesh and prints node/element counts and the physical-group table.
Does NOT run Palace.
"""

import argparse
import os

import gmsh

# --- design (mm) -- matches rfx_patch_inset_xband.py -------------------------
H_SUB = 0.787          # substrate thickness
PATCH_L = 8.595        # x, resonant
PATCH_W = 10.129       # y
W_MSL = 1.8            # 50-ohm feed strip width
NOTCH_GAP = 0.9        # etched slot width beside the feed strip
L_LINE = 12.0          # feed line length before the patch edge
INSET_D = 2.4          # inset penetration depth (rfx match point)

# radiation-box air margins (mm)
MARGIN_XY = 10.0       # lateral air beyond the metal footprint
AIR_ABOVE = 10.0       # air above the substrate

# derived x/y extents of the metal footprint
X_EDGE = 0.0                       # patch leading edge
X_CONN = X_EDGE + INSET_D          # inset connection plane
X_PATCH_HI = X_EDGE + PATCH_L
X_LINE_LO = X_EDGE - L_LINE        # feed far end (= port plane)
YF_LO, YF_HI = -W_MSL / 2, W_MSL / 2
YN_LO, YN_HI = YF_LO - NOTCH_GAP, YF_HI + NOTCH_GAP   # notch outer edges
YP_LO, YP_HI = -PATCH_W / 2, PATCH_W / 2

# domain box
X_LO = X_LINE_LO                   # -x wall carries the port (no extra margin)
X_HI = X_PATCH_HI + MARGIN_XY
Y_LO = YP_LO - MARGIN_XY
Y_HI = YP_HI + MARGIN_XY
Z_TOP = H_SUB + AIR_ABOVE

# mesh sizing (mm)
LC_MIN = 0.25          # on metal edges / port
LC_MAX = 2.5           # at the outer box
DIST_MIN = 0.4
DIST_MAX = 6.0
LC_SUB = 0.5           # inside the thin substrate slab

# physical-group tags = Palace attributes
TAG = {
    "substrate_vol": 1,
    "air_vol": 2,
    "gnd": 3,
    "metal": 4,
    "port": 5,
    "farfield": 6,
}

EPS = 1e-3             # bounding-box query padding (mm)


def _bbox(x0, y0, z0, x1, y1, z1, dim=2, eps=EPS):
    """Tags of entities of `dim` whose bbox is contained in the padded box."""
    ents = gmsh.model.getEntitiesInBoundingBox(
        x0 - eps, y0 - eps, z0 - eps, x1 + eps, y1 + eps, z1 + eps, dim)
    return [t for (d, t) in ents]


def build(out_path):
    gmsh.initialize()
    gmsh.model.add("palace_patch")
    occ = gmsh.model.occ

    # --- volumes: substrate slab + air box stacked on top ---
    sub = occ.addBox(X_LO, Y_LO, 0.0, X_HI - X_LO, Y_HI - Y_LO, H_SUB)
    air = occ.addBox(X_LO, Y_LO, H_SUB, X_HI - X_LO, Y_HI - Y_LO, AIR_ABOVE)

    # --- metal (zero-thickness surfaces at z = H_SUB) ---
    # feed strip + inset tongue (one strip, far end -> inset connection plane)
    feed = occ.addRectangle(X_LINE_LO, YF_LO, H_SUB, X_CONN - X_LINE_LO, W_MSL)
    # patch body beyond the inset
    body = occ.addRectangle(X_CONN, YP_LO, H_SUB, X_PATCH_HI - X_CONN, PATCH_W)
    # two flanks beside the notch slots (x in [edge, conn])
    lflank = occ.addRectangle(X_EDGE, YP_LO, H_SUB, INSET_D, YN_LO - YP_LO)
    rflank = occ.addRectangle(X_EDGE, YN_HI, H_SUB, INSET_D, YP_HI - YN_HI)
    metal_in = [feed, body, lflank, rflank]

    # --- lumped-port surface: vertical rectangle at x = X_LINE_LO,
    #     ground (z=0) -> strip (z=H_SUB), width = strip width ---
    p = [occ.addPoint(X_LINE_LO, YF_LO, 0.0),
         occ.addPoint(X_LINE_LO, YF_HI, 0.0),
         occ.addPoint(X_LINE_LO, YF_HI, H_SUB),
         occ.addPoint(X_LINE_LO, YF_LO, H_SUB)]
    ls = [occ.addLine(p[0], p[1]), occ.addLine(p[1], p[2]),
          occ.addLine(p[2], p[3]), occ.addLine(p[3], p[0])]
    port = occ.addPlaneSurface([occ.addCurveLoop(ls)])

    # --- fragment everything so the mesh is conformal and the metal / port
    #     surfaces are embedded (imprinted) into the volume faces ---
    occ.fragment([(3, sub), (3, air)],
                 [(2, s) for s in metal_in + [port]])
    occ.synchronize()

    # --- re-identify entities by location (fragment renumbers tags) ---
    vols = gmsh.model.getEntities(3)
    assert len(vols) == 2, f"expected 2 volumes, got {vols}"
    sub_vol = _bbox(X_LO, Y_LO, 0.0, X_HI, Y_HI, H_SUB, dim=3, eps=0.05)
    air_vol = _bbox(X_LO, Y_LO, H_SUB, X_HI, Y_HI, Z_TOP, dim=3, eps=0.05)
    assert len(sub_vol) == 1 and len(air_vol) == 1, (sub_vol, air_vol)

    # metal faces at z = H_SUB, one bbox query per footprint rectangle
    footprints = {
        "feed":   (X_LINE_LO, YF_LO, X_CONN, YF_HI),
        "body":   (X_CONN, YP_LO, X_PATCH_HI, YP_HI),
        "lflank": (X_EDGE, YP_LO, X_CONN, YN_LO),
        "rflank": (X_EDGE, YN_HI, X_CONN, YP_HI),
    }
    metal_faces = set()
    for name, (x0, y0, x1, y1) in footprints.items():
        f = _bbox(x0, y0, H_SUB, x1, y1, H_SUB, dim=2, eps=1e-2)
        assert f, f"no metal face found for {name}"
        metal_faces.update(f)

    # ground = bottom face z = 0
    gnd = _bbox(X_LO, Y_LO, 0.0, X_HI, Y_HI, 0.0, dim=2, eps=1e-2)
    assert gnd, "ground face not found"

    # port face at x = X_LINE_LO, z in [0, H_SUB]
    port_face = _bbox(X_LINE_LO, YF_LO, 0.0, X_LINE_LO, YF_HI, H_SUB,
                      dim=2, eps=1e-2)
    assert len(port_face) == 1, f"port face query -> {port_face}"

    # farfield = the five non-ground outer planes (minus the port face)
    ff = set()
    ff.update(_bbox(X_LO, Y_LO, Z_TOP, X_HI, Y_HI, Z_TOP, dim=2, eps=1e-2))  # top
    ff.update(_bbox(X_HI, Y_LO, 0.0, X_HI, Y_HI, Z_TOP, dim=2, eps=1e-2))    # +x
    ff.update(_bbox(X_LO, Y_LO, 0.0, X_LO, Y_HI, Z_TOP, dim=2, eps=1e-2))    # -x
    ff.update(_bbox(X_LO, Y_HI, 0.0, X_HI, Y_HI, Z_TOP, dim=2, eps=1e-2))    # +y
    ff.update(_bbox(X_LO, Y_LO, 0.0, X_HI, Y_LO, Z_TOP, dim=2, eps=1e-2))    # -y
    ff.discard(port_face[0])                 # port lives on the -x plane
    ff -= metal_faces                        # (metal is interior, defensive)
    assert ff, "no farfield faces found"

    # --- physical groups (tag == Palace attribute) ---
    def pg(dim, tags, name):
        gmsh.model.addPhysicalGroup(dim, list(tags), TAG[name])
        gmsh.model.setPhysicalName(dim, TAG[name], name)

    pg(3, sub_vol, "substrate_vol")
    pg(3, air_vol, "air_vol")
    pg(2, gnd, "gnd")
    pg(2, sorted(metal_faces), "metal")
    pg(2, port_face, "port")
    pg(2, sorted(ff), "farfield")

    # --- mesh size fields ---
    edges = set()
    for (_, s) in [(2, t) for t in metal_faces] + [(2, port_face[0])]:
        for (_, c) in gmsh.model.getBoundary([(2, s)], oriented=False):
            edges.add(abs(c))
    fd = gmsh.model.mesh.field
    fd.add("Distance", 1)
    fd.setNumbers(1, "CurvesList", sorted(edges))
    fd.setNumber(1, "Sampling", 200)
    fd.add("Threshold", 2)
    fd.setNumber(2, "InField", 1)
    fd.setNumber(2, "SizeMin", LC_MIN)
    fd.setNumber(2, "SizeMax", LC_MAX)
    fd.setNumber(2, "DistMin", DIST_MIN)
    fd.setNumber(2, "DistMax", DIST_MAX)
    # keep the thin substrate slab resolved
    fd.add("Box", 3)
    fd.setNumber(3, "VIn", LC_SUB)
    fd.setNumber(3, "VOut", LC_MAX)
    fd.setNumber(3, "XMin", X_LINE_LO - 1)
    fd.setNumber(3, "XMax", X_PATCH_HI + 1)
    fd.setNumber(3, "YMin", YP_LO - 1)
    fd.setNumber(3, "YMax", YP_HI + 1)
    fd.setNumber(3, "ZMin", -0.1)
    fd.setNumber(3, "ZMax", H_SUB + 0.1)
    fd.setNumber(3, "Thickness", 2.0)
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

    # MFEM's Gmsh reader: dedup nodes + msh 2.2 (4.1 multi-entity node

    # blocks trip 'vertices indices are not unique')

    gmsh.model.mesh.removeDuplicateNodes()

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.write(out_path)

    # --- report ---
    ntag, ncoord, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(ntag)
    etypes, etags, _ = gmsh.model.mesh.getElements(3)
    n_tets = sum(len(t) for t in etags)
    stypes, stags, _ = gmsh.model.mesh.getElements(2)
    n_tris = sum(len(t) for t in stags)

    print(f"\nwrote {out_path}")
    print(f"  nodes        : {n_nodes}")
    print(f"  tets (3D)    : {n_tets}")
    print(f"  tris (2D bnd): {n_tris}")
    print("\n  physical groups (dim tag name -> #entities, #boundary tris):")
    for (dim, tag) in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        ents = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        n_el = 0
        if dim == 2:
            for e in ents:
                _, et, _ = gmsh.model.mesh.getElements(2, e)
                n_el += sum(len(t) for t in et)
        print(f"    {dim}  {tag:<2} {name:<14} entities={len(ents):<3} "
              f"{'tris=' + str(n_el) if dim == 2 else ''}")
    gmsh.finalize()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "palace_patch.msh"))
    args = ap.parse_args()
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
