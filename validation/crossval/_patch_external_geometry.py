"""Transfer cv05's realized board into the external solver's coordinates.

The air margin is a solver-domain choice. All board objects and the realized
wire terminals undergo the same translation; none is reconstructed from the
off-lattice design dimensions. Coordinates in this record are millimetres.
"""
from __future__ import annotations

import numpy as np


def external_patch_board(stack: dict, feed: dict, *, margin_mm: float) -> dict:
    ground = stack["ground"]
    origin = np.array([ground["realized_x_lo_mm"], ground["realized_y_lo_mm"],
                       ground["realized_z_mm"]], dtype=float)
    shift = np.array([margin_mm, margin_mm, 0.0]) - origin
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
