"""Tracker 1598's current-source scene, shared by CPU and GPU contracts."""
import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation


def scene(devices, *, n=16, steps=6, uniform=False, debye=False, sharded=False, **options):
    dy = np.where(np.arange(n) % 2 == 0, .9e-3, 1.1e-3)
    dy[[0, -1]] = 1e-3
    sim = Simulation(freq_max=15e9, domain=(n * 1e-3,) * 3, dx=1e-3,
                     boundary="pec", **({} if uniform else {"dy_profile": dy}))
    sim.add_source(tuple(n * 1e-3 * p for p in (.25, .5, .5)), "ez",
                   amplitude_kind="current")
    for pos in ((.6, .5, .5), (.75, .55, .45)):
        sim.add_probe(tuple(n * 1e-3 * p for p in pos), "ez")
    if debye:  # pole staging makes its own device-position array
        from rfx import Box, DebyePole
        sim.add_material("pole", eps_r=2.5, debye_poles=[DebyePole(.5, 1e-11)])
        sim.add(Box((2e-3,) * 3, (5e-3,) * 3), material="pole")
    shape = (n + 1,) * 3 if uniform else sim._build_nonuniform_grid().shape
    host = np.ones(shape, np.float32)
    host[tuple(slice(s // 3, 2 * s // 3) for s in shape)] = 1.1
    # An already-sharded design reaches the halo program's device positions.
    eps = sim.shard_distributed_override(host, devices) if sharded else jnp.asarray(host)

    def forward(e):
        return sim.forward(eps_override=e, distributed=True, devices=devices,
                           n_steps=steps, checkpoint=False, skip_preflight=True,
                           **options).time_series

    def objective(e):
        return jnp.sum(forward(e) ** 2)

    def run():
        return sim.run(devices=devices, n_steps=steps, skip_preflight=True).time_series

    return eps, forward, objective, run


def finite(value):
    leaves = jax.tree.leaves(jax.block_until_ready(value))
    assert leaves
    assert all(np.isfinite(np.asarray(a)).all() for a in leaves)
