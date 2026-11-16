"""CAD mesh import tutorial (issue #358) — an STL part straight into the solver.

The full workflow: export a part to STL, import it with ``MeshShape`` and an
explicit mm->m ``scale``, assign it PEC, resolve its 1.5 mm thickness with
uniform 0.75 mm z cells, read the preflight report and step it.

Requires the optional CAD extra:  pip install 'rfx-fdtd[cad]'  (trimesh + rtree).
Without it the script prints guidance and exits 0.

``build_simulation()`` returns the model (steps 1-3) without solving it, so the
imported part can be inspected and preflighted before any run; ``main()``
builds it, prints the preflight report and steps it.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

CAD_EXTRA_MESSAGE = "This demo needs the optional CAD extra:  pip install 'rfx-fdtd[cad]'"


def build_simulation():
    """Import the STL plate as PEC in an open (CPML) domain; no solve.

    Raises ``ModuleNotFoundError`` (``name="trimesh"``) carrying
    ``CAD_EXTRA_MESSAGE`` when the optional CAD extra is not installed.
    """
    try:
        import trimesh
    except ModuleNotFoundError as exc:
        # Only a trimesh that is not installed is "the CAD extra is missing". A
        # trimesh that is installed but fails to import (a numpy ABI mismatch, a
        # missing dependency of its own) is a real error and propagates.
        if exc.name != "trimesh":
            raise
        raise ModuleNotFoundError(CAD_EXTRA_MESSAGE, name="trimesh") from exc

    from rfx.api import Simulation
    from rfx.geometry import MeshShape

    # 1. A "CAD" part — a 30x20x1.5 mm plate, drawn in MILLIMETRES like a real CAD
    #    export. (Swap this for trimesh.load('your_part.stl').) The 1.5 mm thickness
    #    is deliberate: it needs nodes on both faces to preserve its drawn size.
    part_mm = trimesh.creation.box(extents=(30.0, 20.0, 1.5))
    with tempfile.TemporaryDirectory() as td:
        stl = Path(td) / "patch.stl"
        part_mm.export(stl)

        # 2. Import: STL is unitless, so scale mm -> m explicitly; place it in the domain.
        #    (STEP works the same way — MeshShape.from_file('part.step', scale=1.0), since the
        #    cascadio backend already converts STEP units to metres on load.)
        #    from_file loads the triangles into memory, so the temporary STL can go
        #    when this block ends.
        patch = MeshShape.from_file(str(stl), scale=1e-3, translate=(0.03, 0.02, 0.015))
    print(f"imported mesh: bbox = {patch.bounding_box()}  "
          f"min feature = {patch.min_feature_size() * 1e3:.2f} mm")

    # 3. Compose it like any CSG shape and assign PEC.  An imported solid
    #    is a VOLUME — that is the whole point of a CAD body — so it goes
    #    through sim.add(..., material="pec"): the primal cells whose
    #    CENTRES lie inside the mesh, with every E edge incident to one of
    #    them zeroed (lattice ownership contract, #931 §1.2).  Foil is the
    #    other declaration, add_thin_conductor on a zero-thickness shape,
    #    and it is what a CAD sheet body would want.
    #
    #    At uniform dx = 1 mm the plate is solved 2.0 mm thick, and the
    #    default strict preflight refuses it. Uniform 0.75 mm z cells put
    #    nodes on both faces, preserving 1.5 mm thickness, and keep the mesh
    #    uniform up to the absorbers.
    domain = (0.06, 0.04, 0.03)
    dz_profile = np.full(40, 0.75e-3)
    sim = Simulation(freq_max=10e9, domain=domain, dx=0.001,
                     dz_profile=dz_profile,
                     boundary="cpml", cpml_layers=8, mode="3d")
    sim.add(patch, material="pec")
    # source in free space ABOVE the plate (plate top is z≈15.75 mm), not inside the PEC
    sim.add_source((0.03, 0.02, 0.022), component="ez",
                   amplitude_kind="current")
    return sim


def main() -> int:
    try:
        sim = build_simulation()
    except ModuleNotFoundError as exc:
        if exc.name != "trimesh":
            raise
        print(exc)
        return 0

    # 4. Preflight — check the imported part on the uniform z mesh.
    #    Preflight output is part of the result: quote it before trusting a number.
    report = sim.preflight()
    print("\n--- preflight ---")
    for msg in report:
        print(" ", msg)

    # 5. Run a few steps to confirm the imported geometry rasterises and steps cleanly.
    result = sim.run(num_periods=2)
    print(f"\nran OK — probe finite: {bool(np.all(np.isfinite(result.time_series)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
