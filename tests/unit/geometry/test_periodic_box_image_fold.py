"""A Box wider than the period: the image fold is per axis, and equals the nested fold.

A PEC sheet or PEC volume declared across many periods (the natural way to say
"infinite" on a periodic axis) used to sample the whole grid once per PAIR of
images: 2 m of sheet on two periodic axes of 0.75 mm took 1602 s to assemble.
"""
import time
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx._periodic import periodic_mask

DX = 1e-3


def _sim(kind, lo, hi, nx=12, ny=3, nz=2):
    sim = Simulation(freq_max=15e9, domain=(nx * DX, ny * DX, nz * DX), dx=DX,
                     boundary=BoundarySpec(x='pec', y='periodic', z='periodic'))
    x0 = 5 * DX
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if kind == 'sheet':
            sim.add_thin_conductor(Box((x0, lo[0], lo[1]), (x0, hi[0], hi[1])),
                                   sigma_bulk=5.8e7, thickness=35e-6)
        else:
            sim.add(Box((x0, lo[0], lo[1]), (x0 + 3 * DX, hi[0], hi[1])), material='pec')
    return sim


def _assembled(sim):
    grid = sim._build_grid()
    sheets, wires = [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = sim._assemble_materials(grid, pec_sheets=sheets, pec_wires=wires)
    if sheets:
        return np.asarray(sheets[0].unwrapped_footprint)
    return np.asarray(out[3])


# Bounds chosen off the lattice and off multiples of the period, partly covering
# a period, covering none of an axis's base cell centres, and spanning many.
CASES = [((-0.0004, 0.0003), (0.0011, 0.0009)),
         ((-0.0052, -0.0031), (0.0047, 0.0029)),
         ((0.0007, -0.0100), (0.0008, 0.0100)),
         ((-0.0300, -0.0200), (0.0300, 0.0200)),
         # entirely outside the base period: only shifted images contribute
         ((0.0061, 0.0041), (0.0079, 0.0052)),
         ((-0.0052, 0.0021), (-0.0034, 0.0032)),
         # on the lattice and on the periodic seam (closed-footprint rims)
         ((0.0, 0.0), (0.003, 0.002)),
         ((-0.003, 0.001), (0.0, 0.004)),
         ((0.001, -0.002), (0.015, 0.0))]


@pytest.mark.parametrize('kind', ['sheet', 'volume'])
@pytest.mark.parametrize('lo,hi', CASES)
def test_per_axis_fold_equals_the_nested_fold(kind, lo, hi, monkeypatch):
    import rfx._periodic as periodic
    try:
        fast = _assembled(_sim(kind, lo, hi))
    except ValueError as exc:          # an empty footprint is refused either way
        fast = str(exc)
    original = periodic.periodic_mask

    def nested(*args, **kwargs):       # the old route: outer-product sampler only
        kwargs.pop('axis_samplers', None)
        return original(*args, **kwargs)
    # Both callers import the function inside their bodies, from this module.
    monkeypatch.setattr(periodic, 'periodic_mask', nested)
    try:
        slow = _assembled(_sim(kind, lo, hi))
    except ValueError as exc:
        slow = str(exc)
    if isinstance(fast, str) or isinstance(slow, str):
        assert fast == slow
    else:
        assert fast.dtype == slow.dtype and np.array_equal(fast, slow)
        assert fast.any()


@pytest.mark.parametrize('kind', ['sheet', 'volume'])
def test_assembly_time_grows_with_the_sum_of_image_counts(kind):
    def seconds(half):
        sim = _sim(kind, (-half, -half), (half, half), nx=40, ny=2, nz=2)
        start = time.perf_counter()
        _assembled(sim)
        return time.perf_counter() - start
    seconds(0.004)                      # warm-up
    many = seconds(1.024)               # 1024 periods per axis
    # The nested fold took 18 s (sheet) and 45 s (volume) here; the per-axis
    # fold takes about 0.02 s. A bound a product law cannot meet and a loaded
    # machine can.
    assert many < 2.0, many


def test_a_periodic_pec_volume_mask_is_a_jax_array_as_before():
    import jax
    sim = _sim('volume', (-0.0052, -0.0031), (0.0047, 0.0029))
    grid = sim._build_grid()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[3]
    assert isinstance(mask, jax.Array) and mask.dtype == bool
