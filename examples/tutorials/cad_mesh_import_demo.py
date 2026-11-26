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
        if exc.name != "trimesh":
            raise
        raise ModuleNotFoundError(CAD_EXTRA_MESSAGE, name="trimesh") from exc

    from rfx.api import Simulation
    from rfx.geometry import MeshShape

    part_mm = trimesh.creation.box(extents=(30.0, 20.0, 1.5))
    with tempfile.TemporaryDirectory() as td:
        stl = Path(td) / "patch.stl"
        part_mm.export(stl)

        patch = MeshShape.from_file(str(stl), scale=1e-3, translate=(0.03, 0.02, 0.015))
    print(f"imported mesh: bbox = {patch.bounding_box()}  "
          f"min feature = {patch.min_feature_size() * 1e3:.2f} mm")

    domain = (0.06, 0.04, 0.03)
    dz_profile = np.full(40, 0.75e-3)
    sim = Simulation(freq_max=10e9, domain=domain, dx=0.001,
                     dz_profile=dz_profile,
                     boundary="cpml", cpml_layers=8, mode="3d")
    sim.add(patch, material="pec")
    sim.add_source((0.03, 0.02, 0.022), component="ez",
                   amplitude_kind="current")
    sim.add_probe((0.04, 0.02, 0.022), component="ez")
    return sim


def main() -> int:
    try:
        sim = build_simulation()
    except ModuleNotFoundError as exc:
        if exc.name != "trimesh":
            raise
        print(exc)
        return 0

    report = sim.preflight()
    print("\n--- preflight ---")
    for msg in report:
        print(" ", msg)

    result = sim.run(num_periods=2)
    print(f"\nran OK — probe finite: {bool(np.all(np.isfinite(result.time_series)))}; "
          f"peak |Ez| > 0: {bool(np.max(np.abs(result.time_series)) > 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
