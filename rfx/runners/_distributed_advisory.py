"""Advisory records for multi-device runs, built from realized grid counts."""

from rfx.diagnostic_records import Diagnostic

__all__ = ["split_axis_advisory"]


def split_axis_advisory(grid, n_devices, lane):
    """Describe the x-slab communication cost using realized grid counts."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if not (n_devices > 1 and nx < max(ny, nz)):
        return ()
    axis, n_axis = ("y", ny) if ny >= nz else ("z", nz)
    return (Diagnostic(
        code="multi_device_split_axis", severity="advisory",
        subject="device split axis",
        path=lane,
        values={"nx": nx, "ny": ny, "nz": nz, "n_devices": n_devices,
                "longest_axis": axis, "measured_slowdown_3_devices": "1.51-1.69"},
        message=(
            f"This run splits the grid along x across {n_devices} devices, but x "
            f"has {nx} cells while {axis} has {n_axis}. Each device exchanges a full "
            "y-z face every step, so a box whose longest axis is not x ran "
            "1.51-1.69x slower per step than the same box with its long axis along x "
            "on 3 RTX A6000 GPUs (vacuum box, CPML and PEC boundaries), and with "
            "CPML on 2 GPUs it ran slower than on one. If the structure allows, "
            "put its longest axis along x. The run is not changed."
        ),
    ),)
