"""An Ez pulse six cells from a PEC wall must reflect while the other faces absorb.
Splitting the NU mesh across two CPU devices must preserve the single-device probe fields (#1235).
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec

pytestmark = pytest.mark.distributed


def test_y_hi_pec_probe_records_match_single_device():
    """An Ez pulse six cells from a PEC wall must reflect without loss at that wall.
    Two-device NU probe fields must match one device at 3 and 12 cells from it (#1235).
    """
    devices = jax.devices("cpu")[:2]
    if len(devices) < 2:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=2")

    def build():
        nx, ny, nz, dx = 40, 24, 16, 1e-3
        sim = Simulation(
            freq_max=15e9, domain=(nx * dx, ny * dx, nz * dx), dx=dx,
            dx_profile=np.full(nx, dx), cpml_layers=8,
            boundary=BoundarySpec(x="cpml", y=Boundary(lo="cpml", hi="pec"), z="cpml"),
        )
        sim.add_source((nx // 4 * dx, (ny - 6) * dx, nz // 2 * dx), "ez")
        for distance in (3, 12):
            sim.add_probe((nx // 4 * dx, (ny - distance) * dx, nz // 2 * dx), "ez")
        return sim

    with jax.default_device(devices[0]):
        single = np.asarray(build().forward(n_steps=600).time_series)
        distributed = np.asarray(build().forward(
            n_steps=600, distributed=True, devices=devices).time_series)
    assert single.shape == distributed.shape == (600, 2)
    assert np.isfinite(single).all() and np.isfinite(distributed).all()
    peak = np.max(np.abs(single), axis=0)
    assert np.all(peak > 0), "both probes must see the pulse"
    relative_error = np.max(np.abs(distributed - single), axis=0) / peak
    print(f"\n[NU PEC y-hi] max|difference| / single-device peak: {relative_error.tolist()}")
    assert np.all(relative_error <= 1e-4), (
        f"PEC y-hi probe fields at 3/12 cells differ by {relative_error.tolist()} "
        "of the single-device peaks (limit 1e-4)"
    )
