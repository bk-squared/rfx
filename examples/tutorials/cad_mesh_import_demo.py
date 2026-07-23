"""CAD mesh import tutorial (issue #358) — bring an STL part straight into the solver.

Shows the full workflow: export a simple part to STL, import it with ``MeshShape`` (an
explicit mm→m ``scale``), assign it PEC, watch the preflight resolution advisory fire on
an under-resolved feature, and run a few FDTD steps.

Requires the optional CAD extra:  pip install 'rfx-fdtd[cad]'  (trimesh + rtree).
Degrades gracefully (prints guidance, exits 0) if the extra is absent.
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

    # 1. A "CAD" part — a 30x20x2 mm rectangular patch, drawn in MILLIMETRES like a real
    #    CAD export. (Swap this for trimesh.load('your_part.stl').)
    part_mm = trimesh.creation.box(extents=(30.0, 20.0, 2.0))
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
        # source in free space ABOVE the patch (patch top is z≈16 mm), not inside the PEC
        sim.add_source((0.03, 0.02, 0.022), component="ez")

        # 4. Preflight — the 2 mm patch thickness is 2 cells at dx=1 mm; a finer feature
        #    would trip the mesh_import_underresolved advisory. Preflight output is part of
        #    the result: quote it before trusting any number.
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
