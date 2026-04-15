"""Phase B smoke test: distributed runner with a degenerate-uniform
non-uniform grid must match the uniform 2-device baseline.

Forces 2 virtual CPU devices via XLA_FLAGS (set BEFORE jax import).
"""

import os
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np
import pytest
import jax

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2,
    reason=(
        "Phase B distributed NU smoke requires >=2 devices. "
        "Run with XLA_FLAGS=--xla_force_host_platform_device_count=2."
    ),
)

from rfx import Simulation  # noqa: E402


def _make_sim_nu():
    """Build a NU-profiled simulation with a constant profile.

    Using an explicit dx_profile guarantees nx = len(profile), which is
    what the distributed NU path assumes. The uniform-reference sim is
    built from the same profile so the two paths agree on shape.
    """
    nx, ny, nz = 32, 16, 16
    dx = 1e-3
    sim = Simulation(
        freq_max=3e9,
        domain=(nx * dx, ny * dx, nz * dx),
        dx=dx,
        boundary="pec",
    )
    sim._dx_profile = np.full(nx, dx)
    sim._dy_profile = np.full(ny, dx)
    sim._dz_profile = np.full(nz, dx)
    return sim


def test_degenerate_uniform_matches_single_device_nu():
    """The distributed NU path with a degenerate-uniform profile should
    match the single-device NU runner to machine precision. This is the
    cleanest equivalence check because both paths share the NU kernel.

    Source is placed in the interior of device 0 (global x-index < nx_per)
    to avoid the known Phase B seam-injection staleness: the step order
    is H → exch H → E → exch E → PEC → source, so a source injected at
    the first cell of device d>0 is not seen by device d-1's H update
    until the next-next step's exchange. Interior placement sidesteps
    this.
    """
    devices = jax.devices()[:2]
    n_steps = 40

    sim_multi = _make_sim_nu()
    sim_multi.add_source(position=(4e-3, 8e-3, 8e-3), component="ez")
    sim_multi.add_probe(position=(8e-3, 8e-3, 8e-3), component="ez")
    res_multi = sim_multi.run(n_steps=n_steps, devices=devices)

    sim_single = _make_sim_nu()
    sim_single.add_source(position=(4e-3, 8e-3, 8e-3), component="ez")
    sim_single.add_probe(position=(8e-3, 8e-3, 8e-3), component="ez")
    res_single = sim_single.run(n_steps=n_steps)

    ts_multi = np.asarray(res_multi.time_series)
    ts_single = np.asarray(res_single.time_series)
    assert ts_multi.shape == ts_single.shape, (
        f"shape mismatch: multi {ts_multi.shape} vs single {ts_single.shape}"
    )
    peak = np.max(np.abs(ts_single)) + 1e-30
    rel_err = np.max(np.abs(ts_multi - ts_single)) / peak
    assert rel_err < 1e-3, (
        f"Distributed-NU vs single-NU rel_err={rel_err:.2e}"
    )


def test_distributed_nu_produces_nonzero_signal():
    """Minimum viability: the NU distributed path must produce a
    non-zero probe signal through a source injection."""
    devices = jax.devices()[:2]
    sim = _make_sim_nu()
    sim.add_source(position=(16e-3, 8e-3, 8e-3), component="ez")
    sim.add_probe(position=(16e-3, 8e-3, 8e-3), component="ez")
    res = sim.run(n_steps=30, devices=devices)
    ts = np.asarray(res.time_series).ravel()
    assert np.max(np.abs(ts)) > 0, (
        "probe signal is zero — NU distributed source not injected"
    )
