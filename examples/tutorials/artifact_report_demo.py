"""Create a shareable rfx artifact bundle (scene + mesh + report) without running FDTD.

Use this when you want to hand a design to a colleague or archive a setup:
it exports the native scene, mesh/preflight metadata, a Markdown review
report, and a manifest. Run:

    python examples/tutorials/artifact_report_demo.py

The demo exports the native rfx scene, mesh/preflight metadata, legacy
``geometry.json``, a Markdown review report, and a manifest.  It intentionally
does not import CAD, open a GUI, use body-fitted meshing, or claim deterministic
replay; those boundaries are recorded in the report limitations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

# Allow `python examples/artifact_report_demo.py` from a source checkout before
# the package is installed in editable mode.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rfx.api import Simulation
from rfx.geometry.csg import Box


def build_demo_simulation() -> Simulation:
    sim = Simulation(
        freq_max=8.0e9,
        domain=(12e-3, 10e-3, 4e-3),
        boundary="pec",
        dx=1.0e-3,
    )
    sim.add_material("substrate", eps_r=3.55, sigma=0.0027)
    sim.add(Box((2e-3, 2e-3, 0.5e-3), (8e-3, 8e-3, 1.5e-3)), material="substrate")
    sim.add_port((2e-3, 5e-3, 1e-3), "ez", impedance=50.0)
    sim.add_probe((10e-3, 5e-3, 1e-3), "ez")
    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Bundle directory to write. Defaults to a temporary directory.",
    )
    args = parser.parse_args()

    output = args.output or Path(tempfile.mkdtemp(prefix="rfx-artifact-demo-"))
    bundle = build_demo_simulation().export_artifact_bundle(output)
    print(bundle.root)
    for file_path in bundle.files:
        print(file_path.relative_to(bundle.root))


if __name__ == "__main__":
    main()
