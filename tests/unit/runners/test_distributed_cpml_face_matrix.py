"""A point pulse illuminates one PEC/PMC wall with CPML on the other faces.
The wall must reflect and the multi-device probe records must match one device.
Probe distances are 3 and 12 mm on a 1 mm boundary mesh, away from box symmetry.
"""

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

pytestmark = pytest.mark.distributed

FACES = tuple(f"{axis}_{side}" for axis in "xyz" for side in ("lo", "hi"))
CASES = [pytest.param("cpml", "y_hi", 2, 25, id="all_cpml")]
for _wall in ("pec", "pmc"):
    for _face in FACES:
        CASES.append(pytest.param(
            _wall, _face, 2, 25, id=f"{_wall}_{_face}",
            marks=() if _face in ("y_lo", "y_hi") else pytest.mark.slow_physics,
        ))
CASES.append(pytest.param("unequal", "y_hi", 2, 25, id="unequal", marks=pytest.mark.slow_physics))
# One x wall gives 35 / 55 nodes, leaving alignment cells on 2 / 3 devices.
# The longer three-device box also satisfies the NU lane's slab-width check.
for _devices, _x_cells in ((2, 26), (3, 46)):
    for _wall, _face in (("pec", "x_hi"), ("pmc", "x_lo")):
        CASES.append(pytest.param(
            _wall, _face, _devices, _x_cells, id=f"{_wall}_{_face}_pad_{_devices}",
            marks=() if (_wall, _devices) == ("pec", 2) else pytest.mark.slow_physics,
        ))


def _build(lane, wall, face, x_cells=25):
    dx = 1e-3
    cells = np.array([x_cells, 24, 24])
    axis, side = face.split("_")
    axis_index = "xyz".index(axis)
    boundaries = dict(x="cpml", y="cpml", z="cpml")
    if wall == "unequal":
        boundaries[axis] = Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=8)
    elif wall != "cpml":
        boundaries[axis] = Boundary(
            lo=wall if side == "lo" else "cpml",
            hi=wall if side == "hi" else "cpml",
        )
    profiles = {}
    if lane == "nu":
        profile = np.full(cells[0], dx)
        # Keep the 3/12-cell probe positions on integer-mm edges from either
        # wall; the unequal cell widths cancel before the far-side probes.
        profile[14:16] = 0.95 * dx
        profile[16] = 1.1 * dx
        profiles["dx_profile"] = profile
    sim = Simulation(
        freq_max=15e9, domain=tuple(cells * dx), dx=dx,
        cpml_layers=8, boundary=BoundarySpec(**boundaries), **profiles,
    )
    # Use tangential E for each wall normal, and asymmetric transverse positions.
    component = "ey" if axis == "z" else "ez"
    position = np.array([7, 9, 10], dtype=float)
    position[axis_index] = 6 if side == "lo" else cells[axis_index] - 6
    sim.add_source(tuple(position * dx), component)
    for distance in (3, 12):
        probe = position.copy()
        probe[axis_index] = distance if side == "lo" else cells[axis_index] - distance
        sim.add_probe(tuple(probe * dx), component)
    return sim


@pytest.mark.parametrize("lane", ("uniform", "nu"))
@pytest.mark.parametrize("wall,face,n_devices,x_cells", CASES)
def test_probe_records_match_single_device(lane, wall, face, n_devices, x_cells, record_property):
    devices = jax.devices("cpu")[:n_devices]
    if len(devices) != n_devices:
        pytest.skip(f"requires XLA_FLAGS=--xla_force_host_platform_device_count={n_devices}")
    steps = 240
    with jax.default_device(devices[0]):
        single_sim = _build(lane, wall, face, x_cells)
        multi_sim = _build(lane, wall, face, x_cells)
        if lane == "uniform":
            single_result = single_sim.run(n_steps=steps)
            multi_result = multi_sim.run(n_steps=steps, devices=devices)
        else:
            single_result = single_sim.forward(n_steps=steps)
            multi_result = multi_sim.forward(n_steps=steps, distributed=True, devices=devices)
    if x_cells != 25:
        # Read the runner's realized grid, not a separately rebuilt grid (#623).
        nx = int(multi_result.grid.nx)
        pad_x = (-nx) % n_devices
        record_property("realized_nx", nx)
        record_property("realized_pad_x", pad_x)
        assert nx % 2 == 1 and pad_x > 0, (
            f"requires odd nx and pad_x > 0; realized nx={nx}, devices={n_devices}, pad_x={pad_x}"
        )
    single = np.asarray(single_result.time_series)
    multi = np.asarray(multi_result.time_series)
    assert single.shape == multi.shape == (steps, 2)
    assert np.isfinite(single).all() and np.isfinite(multi).all()
    peak = np.max(np.abs(single), axis=0)
    assert np.all(peak > 0), "both probes must see the pulse"
    relative_error = np.max(np.abs(multi - single), axis=0) / peak
    record_property("single_peak", peak.tolist())
    record_property("relative_error_3_12_cells", relative_error.tolist())
    assert np.all(relative_error <= 1e-4), (
        f"{lane} {wall} {face}: max|{n_devices}-device - one-device| / one-device peak "
        f"at 3/12 cells = {relative_error.tolist()} (limit 1e-4)"
    )
    if wall == "unequal":
        grid = (single_sim._build_grid() if lane == "uniform"
                else single_sim._build_nonuniform_grid())
        depths = (grid.pad_y_lo, grid.pad_y_hi)
        record_property("realized_y_lo_hi_layers", depths)
        assert depths == (4, 8), f"requested 4/8 layers; realized {depths}"
