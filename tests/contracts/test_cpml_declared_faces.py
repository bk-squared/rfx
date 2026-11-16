"""Declared reflecting faces preserve the even Ez image, including on slabs."""

import sys
from time import perf_counter

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse
from tests._multi_device_scene import host_devices


H = 1e-3
CELLS = (24, 20, 16)
STEPS = 60
IMAGE_BAR = 1e-5
LANE_ULP_BAR = 9
# Design A18/A19: uniform vs constant-profile graded uses two kernels;
# tests/unit/nonuniform/test_nonuniform_face_depths.py:94 sets this bar.
TWO_KERNEL_BAR = STEPS * 10 * np.finfo(np.float32).eps
# PMC run/forward/nonuniform/distributed images already have rows in
# test_realized_boundary.py::test_declared_face. Only distributed_nu is new.
ROWS = [("pec-zlo", entry) for entry in
        ("run", "nonuniform", "distributed", "distributed_nu")]
ROWS += [("pmc-cpml", "distributed_nu")]


def _model(case, entry, layers, *, image=False, opposite=False):
    axis = 2 if case == "pec-zlo" else 0
    token = "pec" if axis == 2 else "pmc"
    if opposite:
        token = "pmc" if token == "pec" else "pec"
    cells = list(CELLS)
    source = [9, 7, 3] if axis == 2 else [9, 7, 6]
    probes = [[11, 8, 5]] if axis == 2 else [[5, 8, 5], [19, 8, 5]]
    faces = dict.fromkeys("xyz", "cpml")
    if image:
        cells[axis] *= 2
        # Ez is half-cell displaced along z, node-centred along x.
        # Reflection of its lattice index is 2*N-i-1 on z, 2*N-i on x.
        source[axis] += CELLS[axis]
        mirror = source.copy()
        mirror[axis] = 2 * CELLS[axis] - source[axis] - (axis == 2)
        for probe in probes:
            probe[axis] = CELLS[axis] - probe[axis] - (axis == 2)
        if axis == 0:
            # Mirroring across x=0 retains the far PMC wall and its mirror
            # at the doubled box's two outer x faces; y/z stay CPML.
            faces["x"] = "pmc"
        sources = (source, mirror)
    else:
        faces["xyz"[axis]] = Boundary(token, "cpml") if axis == 2 else token
        sources = (source,)
    profiles = {"dz_profile": np.full(cells[2], H)} if entry in ("nonuniform", "distributed_nu") else {}
    sim = Simulation(freq_max=20e9, domain=tuple(n * H for n in cells), dx=H,
                     cpml_layers=layers, boundary=BoundarySpec(**faces), **profiles)
    for position in sources:
        sim.add_source(tuple(i * H for i in position), "ez",
                       waveform=GaussianPulse(f0=15e9, bandwidth=0.8), amplitude_kind="field")
    for probe in probes:
        sim.add_probe(tuple(i * H for i in probe), "ez")
    return sim, tuple(cells)


def _record(case, entry, layers, *, image=False, opposite=False):
    sim, cells = _model(case, entry, layers, image=image, opposite=opposite)
    devices = host_devices(2) if entry.startswith("distributed") else host_devices(1)
    with jax.default_device(devices[0]):
        result = sim.run(n_steps=STEPS, compute_s_params=False, devices=devices)
    trace = np.asarray(result.time_series)
    assert trace.shape == (STEPS, 1 if case == "pec-zlo" else 2)
    assert np.isfinite(trace).all()
    assert np.max(np.abs(trace)) > 0, f"{case}/{entry}: empty pulse record"
    grid = result.grid
    pads = tuple(getattr(grid, f"pad_{a}_{side}") for a in "xyz" for side in ("lo", "hi"))
    realized = tuple(grid.shape[a] - pads[2*a] - pads[2*a+1]
                     - ("xyz"[a] not in grid.periodic_axes) for a in range(3))
    assert realized == cells, f"declared cells={cells}, realized cells={realized}"
    print(f"mesh {case}/{entry} image={image} opposite={opposite} layers={layers} "
          f"cells={realized} shape={grid.shape} pads={pads} dt={result.dt:.12g} "
          f"steps={STEPS} devices={len(devices)}", file=sys.stderr)
    return trace


def _difference(row, quantity, actual, expected, *, ulp=False, separated=False, two_kernel=False, distributed_graded=False):
    peak = float(np.max(np.abs(expected)))
    scale = float(np.spacing(np.float32(peak))) if ulp else peak
    measured = float(np.max(np.abs(actual.astype(np.float64) - expected))) / scale
    bar = 1e-4 if distributed_graded else TWO_KERNEL_BAR if two_kernel else LANE_ULP_BAR if ulp else IMAGE_BAR
    print(f"{row} {quantity}: measured={measured:.12g} bar={bar} "
          f"expected_peak={peak:.12g} max_abs_error={measured*scale:.12g}", file=sys.stderr)
    assert (measured > bar if separated else measured <= bar), (
        f"{row} {quantity}: measured={measured:.12g} {'<=' if separated else '>'} "
        f"bar={bar}; expected_peak={peak:.12g}; steps={STEPS}; cells={CELLS}")


@pytest.mark.parametrize("case,entry", ROWS, ids=[f"{case}-{entry}" for case, entry in ROWS])
def test_declared_reflector_matches_even_image(case, entry):
    """Image agreement, lane agreement, and a different wall's live reflection."""
    started = perf_counter()
    records = []
    for layers in (8, 16):
        row = f"{case}/{entry}/cpml{layers}"
        half = _record(case, entry, layers)
        image = _record(case, entry, layers, image=True)
        _difference(row, "even-image/peak", half, image)
        if entry != "run":
            reference = "nonuniform" if entry == "distributed_nu" else "run"
            single = _record(case, reference, layers)
            two_kernel = entry == "nonuniform"
            # Design A21: two-vs-one-device graded probe bar from
            # tests/unit/runners/test_distributed_graded_run.py:93-99.
            distributed_graded = entry == "distributed_nu"
            quantity = f"lane-vs-{reference}/" + ("peak" if two_kernel or distributed_graded else "peak-ULP")
            _difference(row, quantity, half, single, ulp=not (two_kernel or distributed_graded),
                        two_kernel=two_kernel, distributed_graded=distributed_graded)
        other = _record(case, entry, layers, opposite=True)
        _difference(row, "opposite-wall/peak", other, half, separated=True)
        records.append(half)
    movement = float(np.max(np.abs(records[0] - records[1]))) / float(np.max(np.abs(records[1])))
    print(f"{case}/{entry} absorber-doubling/peak: measured={movement:.12g}; "
          "image bar checked independently at both thicknesses", file=sys.stderr)
    print(f"wall {case}/{entry}: {perf_counter()-started:.6f}s", file=sys.stderr)
