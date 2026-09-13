"""Transfer cv05's realized board into the external solver's coordinates.

The air margin is a solver-domain choice. All board objects and the realized
wire terminals undergo the same translation; none is reconstructed from the
off-lattice design dimensions. Coordinates in this record are millimetres.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import tempfile


def external_patch_board(stack: dict, feed: dict, *, margin_mm: float,
                         ground_z_mm: float = 0.) -> dict:
    ground = stack["ground"]
    origin = np.array([ground["realized_x_lo_mm"], ground["realized_y_lo_mm"],
                       ground["realized_z_mm"]], dtype=float)
    shift = np.array([margin_mm, margin_mm, ground_z_mm]) - origin
    board = {"translation_mm": shift.tolist()}
    for name in ("ground", "patch", "substrate"):
        part = stack[name]
        lo = np.array([part["realized_x_lo_mm"], part["realized_y_lo_mm"],
                       part.get("realized_z_lo_mm", part.get("realized_z_mm"))])
        hi = np.array([part["realized_x_hi_mm"], part["realized_y_hi_mm"],
                       part.get("realized_z_hi_mm", part.get("realized_z_mm"))])
        if not (np.all(np.isfinite([lo, hi])) and np.all(hi[:2] > lo[:2])
                and hi[2] >= lo[2]):
            raise ValueError(f"cv05 invalid realized {name} bounds: {lo}, {hi}")
        board[name] = {"lo": (lo + shift).tolist(), "hi": (hi + shift).tolist()}
    board["feed"] = {
        "lo": (1000 * np.asarray(feed["realized_start_m"]) + shift).tolist(),
        "hi": (1000 * np.asarray(feed["realized_end_m"]) + shift).tolist(),
    }
    assert_external_patch_contacts(board)
    return board


def assert_external_patch_contacts(board: dict) -> None:
    """Reject a translated feed that misses either finite conductor sheet."""
    for terminal, conductor in (("lo", "ground"), ("hi", "patch")):
        point = np.asarray(board["feed"][terminal])
        lo, hi = (np.asarray(board[conductor][key]) for key in ("lo", "hi"))
        if not (np.all(np.isfinite(point)) and np.all(point[:2] >= lo[:2])
                and np.all(point[:2] <= hi[:2])
                and np.isclose(point[2], lo[2], rtol=0, atol=1e-10)):
            raise ValueError(f"cv05 external feed misses {conductor}: {point}")
    for name in ("ground", "patch"):
        lo, hi = (np.asarray(board[name][key]) for key in ("lo", "hi"))
        if not np.isclose(lo[2], hi[2], rtol=0, atol=1e-10):
            raise ValueError(f"cv05 external {name} must remain a sheet")


def add_openems_patch_board(csx, fdtd, board: dict, *, eps_r: float):
    """Install exactly the inspected bounds; callable without solver stepping."""
    assert_external_patch_contacts(board)
    substrate = csx.AddMaterial("FR4")
    substrate.SetMaterialProperty(epsilon=eps_r)
    substrate.AddBox(board["substrate"]["lo"], board["substrate"]["hi"], priority=1)
    for name in ("ground", "patch"):
        metal = csx.AddMetal(name)
        metal.AddBox(board[name]["lo"], board[name]["hi"], priority=10)
    return fdtd.AddLumpedPort(
        port_nr=1, R=50.0, start=board["feed"]["lo"], stop=board["feed"]["hi"],
        p_dir="z", excite=1.0,
    )


def external_patch_mesh(board: dict, *, margin_mm: float, air_above_mm: float,
                        substrate_cells: int = 4) -> tuple[list[float], dict]:
    """Preserve the actual material, sheet and terminal coordinates in the mesh.

    A translated stack can include an intentional air gap in the sheet-plane
    falsifier. Reconstructing z from the nominal laminate thickness would
    silently drop those conductor planes again.
    """
    names = ("ground", "patch", "substrate", "feed")
    upper = [max(board[name]["hi"][axis] for name in names)
             + (air_above_mm if axis == 2 else margin_mm) for axis in range(3)]
    lines = {}
    for axis, label in enumerate("xyz"):
        points = [0., upper[axis]]
        points.extend(board[name][corner][axis]
                      for name in names for corner in ("lo", "hi"))
        if axis == 2:
            points.extend(np.linspace(board["substrate"]["lo"][2],
                                      board["substrate"]["hi"][2], substrate_cells+1))
        lines[label] = np.unique(points)
    return upper, lines


def assert_external_patch_mesh(board: dict, lines: dict) -> None:
    """A smoother must retain the prescribed faces and terminals exactly."""
    for axis, label in enumerate("xyz"):
        values = np.asarray(lines[label])
        if not (np.all(np.isfinite(values)) and np.all(np.diff(values) > 0)):
            raise ValueError(f"cv05 external {label} mesh is not strictly increasing")
        for name in ("ground", "patch", "substrate", "feed"):
            for corner in ("lo", "hi"):
                if not np.any(values == board[name][corner][axis]):
                    raise ValueError(f"cv05 external mesh lost {name}.{corner}.{label}")


def assert_external_absorber_clearance(board: dict, lines: dict, *, pml_layers: int) -> None:
    """The finite antenna and feed must lie outside the actual PML cells."""
    for axis, label in enumerate("xyz"):
        values = np.asarray(lines[label])
        if values.size < 2*pml_layers + 3:
            raise ValueError(f"cv05 external {label} mesh has no PML-free interior")
        lower, upper = values[pml_layers], values[-pml_layers-1]
        for name in ("ground", "patch", "substrate", "feed"):
            if not (board[name]["lo"][axis] > lower and board[name]["hi"][axis] < upper):
                raise ValueError(f"cv05 external {name} intersects {label} PML cells")


def run_openems_reference(fdtd, directory_root) -> str:
    """Never adopt a previous board's files as a new reference measurement."""
    root = Path(directory_root)
    root.mkdir(parents=True, exist_ok=True)
    path = tempfile.mkdtemp(prefix="run-", dir=root)
    print(f"  OpenEMS fresh output: {path}")
    fdtd.Run(path, verbose=0, cleanup=True)
    return path
