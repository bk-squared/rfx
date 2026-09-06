"""CAD mesh import tutorial (issue #358) — an STL part straight into the solver.

The full workflow: export a part to STL, import it with ``MeshShape`` and an
explicit mm->m ``scale``, assign it PEC, read the preflight resolution advisory
(``mesh_import_underresolved``) that fires on the plate's 1.5 mm thickness — 1.5
cells at dx = 1 mm, under the 2-cell floor — and step it.

Requires the optional CAD extra:  pip install 'rfx-fdtd[cad]'  (trimesh + rtree).
Without it the script prints guidance and exits 0.
"""
import sys
import tempfile
from pathlib import Path


def main() -> int:
    try:
        import trimesh
    except ImportError:
        print("This demo needs the optional CAD extra:  pip install 'rfx-fdtd[cad]'")
        return 0

    from rfx.api import Simulation
    from rfx.geometry import MeshShape

    # 1. A "CAD" part — a 30x20x1.5 mm plate, drawn in MILLIMETRES like a real CAD
    #    export. (Swap this for trimesh.load('your_part.stl').) The 1.5 mm thickness
    #    is deliberate: under-resolved at dx = 1 mm, so the advisory in step 4 fires.
    part_mm = trimesh.creation.box(extents=(30.0, 20.0, 1.5))
    with tempfile.TemporaryDirectory() as td:
        stl = Path(td) / "patch.stl"
        part_mm.export(stl)

        # 2. Import: STL is unitless, so scale mm -> m explicitly; place it in the domain.
        #    (STEP works the same way — MeshShape.from_file('part.step', scale=1.0), since the
        #    cascadio backend already converts STEP units to metres on load.)
        patch = MeshShape.from_file(str(stl), scale=1e-3, translate=(0.03, 0.02, 0.015))
        print(f"imported mesh: bbox = {patch.bounding_box()}  "
              f"min feature = {patch.min_feature_size() * 1e3:.2f} mm")

        # 3. Compose it like any CSG shape and assign PEC.
        sim = Simulation(freq_max=10e9, domain=(0.06, 0.04, 0.03), dx=0.001,
                         boundary="cpml", cpml_layers=8, mode="3d")
        sim.add(patch, material="pec")
        # source in free space ABOVE the plate (plate top is z≈15.75 mm), not inside the PEC
        sim.add_source((0.03, 0.02, 0.022), component="ez",
                       amplitude_kind="current")

        # 4. Preflight — the advisory fires here. The check (rfx/api/_preflight.py,
        #    code "mesh_import_underresolved") compares the mesh's thinnest bbox extent
        #    against 2 cells: 1.5 mm at dx = 1 mm is 1.5 cells, so it warns. A 2.0 mm
        #    plate sits exactly AT the 2-cell floor and stays silent, so this is the
        #    last warning before a feature is staircased away, not a wide margin.
        #    Preflight output is part of the result: quote it before trusting a number.
        report = sim.preflight()
        print("\n--- preflight ---")
        for msg in report:
            print(" ", msg)

        # 5. Run a few steps to confirm the imported geometry rasterises and steps cleanly.
        result = sim.run(num_periods=2)
        import numpy as np
        print(f"\nran OK — probe finite: {bool(np.all(np.isfinite(result.time_series)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
