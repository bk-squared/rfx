"""Lossy Lorentz material crossing y/z CPML: public one-device parity."""
import itertools
import sys

import numpy as np
import pytest

from rfx import Box
from rfx._internal._grid_metric import field_index
from tests._distributed_dispersive_cpml import check, measure, scene
from tests._multi_device_scene import host_devices

pytestmark = pytest.mark.distributed


@pytest.mark.parametrize("name", ["lorentz_lossy_yz"])
def test_lossy_lorentz_law_c_and_bar_s(name):
    devices = host_devices(2)
    sim = scene("lorentz", cells=False, sigma=.5)
    sim.add(Box((.009, -1., -1.), (.015, 1., 1.)), material="m")
    grid = sim.freeze_mesh()
    # As in Scene T, observe every distinct physical E cell. Indexing only
    # chooses observation sites; the reference is a public one-device solve.
    axes = [np.arange(0., length + 1e-12, 5e-5) for length in (.024, .012, .012)]
    for component in ("ex", "ey", "ez"):
        sites = []
        for axis, positions in enumerate(axes):
            distinct = {}
            for position in positions:
                xyz = [.012, .006, .006]
                xyz[axis] = position
                distinct[field_index(grid, xyz, component)[axis]] = position
            sites.append(tuple(distinct.values()))
        for position in itertools.product(*sites):
            sim.add_probe(tuple(float(v) for v in position), component)

    one = sim.forward(n_steps=400, checkpoint=False, skip_preflight=True)
    multi = sim.forward(n_steps=400, checkpoint=False, skip_preflight=True,
                        distributed=True, devices=devices)
    expected, actual = np.asarray(one.time_series), np.asarray(multi.time_series)
    assert expected.shape[0] == actual.shape[0] == 400
    assert np.any(expected[:, 4:]), "Scene T has no electric signal"
    assert "distributed.one_device_fallback" not in [d.code for d in multi.diagnostics]
    # Existing pair/bar: test_distributed_dispersive_cpml_lorentz_lossy.py:12,
    # tests/_distributed_dispersive_cpml.py:60-87: accumulated law C and bar S.
    row = dict(kind=name, devices=2, steps=400, mesh=tuple(one.grid.shape),
               law_bar=1., spectral_bar=1e-6,
               peak=float(np.abs(expected[:, 4:]).max()),
               **measure(expected, actual, float(one.dt)))
    step = row["worst_step"] - 1
    column = int(np.abs(actual[step, 4:] - expected[step, 4:]).argmax()) + 4
    row.update(expected=float(expected[step, column]), actual=float(actual[step, column]),
               difference=float(abs(float(actual[step, column]) - float(expected[step, column]))),
               allowance=float(1.2e-7 * np.abs(expected[:step + 1, 4:]).max(axis=1).astype(np.float64).sum()))
    print(f"B3.5 {row}", file=sys.stderr, flush=True)
    check(row)
