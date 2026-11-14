"""Pin the traced cell-occupancy pole cap and its actionable refusal message.

The contract also pins equal-valued poles, eager gradients, and concrete meshes as routes through admission.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation, Sphere
from rfx.materials.lorentz import drude_pole

DX = 1e-3
NX = 16
BASE = np.full(NX, DX)
DT = 0.5 * DX / (3e8 * np.sqrt(3.0))


def _profile(d):
    bump = np.zeros(NX)
    bump[7], bump[8] = 1.0, -1.0
    return jnp.asarray(BASE) + d * jnp.asarray(bump)


def _sim(d, n_media, *, equal, traced_mesh, corner=None):
    kw = (dict(dx_profile=_profile(d), dt=DT, dt_min_cell=0.5e-3, nominal_profiles={"x": BASE})
          if traced_mesh else {})
    sim = Simulation(freq_max=20e9, domain=(NX * DX, 8 * DX, 8 * DX), dx=DX, boundary="pec", **kw)
    for k in range(n_media):
        fp = 5e9 if equal else (3.0 + 0.5 * k) * 1e9
        sim.add_material(f"m{k}", eps_r=1.0, lorentz_poles=[drude_pole(2 * np.pi * fp, 2 * np.pi * 1e9)])
        x0 = (3 + k) * DX
        lo = (x0 if (corner is None or k != 0) else corner, 2 * DX, 2 * DX)
        sim.add(Box(lo, (x0 + DX, 6 * DX, 6 * DX)), material=f"m{k}")
    sim.add_source((2 * DX, 4 * DX, 4 * DX), "ez", waveform=GaussianPulse(f0=8e9, bandwidth=0.8))
    sim.add_probe((14 * DX, 4 * DX, 4 * DX), "ez")
    return sim


def _radius_objective(r):
    sim = Simulation(freq_max=20e9, domain=(NX * DX, 8 * DX, 8 * DX), dx=DX, boundary="pec")
    for k in range(9):
        sim.add_material(f"m{k}", eps_r=2.0,
                         lorentz_poles=[drude_pole(2 * np.pi * (3.0 + 0.5 * k) * 1e9, 2 * np.pi * 1e9)])
        if k == 0:
            sim.add(Sphere((3.5 * DX, 4.0 * DX, 4.0 * DX), r), material="m0")
        else:
            x0 = (4 + k) * DX
            sim.add(Box((x0, 2 * DX, 2 * DX), (x0 + DX, 6 * DX, 6 * DX)), material=f"m{k}")
    sim.add_source((1.0 * DX, 4.0 * DX, 4.0 * DX), "ez", waveform=GaussianPulse(f0=8e9, bandwidth=0.8))
    sim.add_probe((15.0 * DX, 4.0 * DX, 4.0 * DX), "ez")
    return jnp.sum(sim.forward(n_steps=4, skip_preflight=True).time_series ** 2)


def _mesh_objective(d, *, equal=False, traced_mesh=True):
    sim = _sim(d, 9, equal=equal, traced_mesh=traced_mesh)
    return jnp.sum(sim.forward(n_steps=4, skip_preflight=True).time_series ** 2)


@pytest.mark.parametrize("case", ["C1", "C2", "C3"])
def test_traced_pole_cap_refusal(case):
    calls = {
        "C1": lambda: jax.jit(_mesh_objective)(0.0),
        "C2": lambda: jax.vmap(_mesh_objective)(jnp.zeros(2)),
        "C3": lambda: jax.jit(_radius_objective)(1.6e-3),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError) as exc_info:
            jax.block_until_ready(calls[case]())
    message = str(exc_info.value)
    for substring in (
        "9 distinct poles",
        f"all {2**9 - 1} subsets would be judged",
        "at most 8 poles",
        "equal values",
        "keep that axis or shape concrete",
        "jax.grad or jax.jvp without jax.jit",
    ):
        assert substring in message


@pytest.mark.parametrize("case", ["C4", "C5", "C6"])
def test_traced_pole_cap_routes(case):
    calls = {
        "C4": lambda: jax.jit(lambda d: _mesh_objective(d, equal=True))(0.0),
        "C5": lambda: jax.grad(_mesh_objective)(0.0),
        "C6": lambda: jax.jit(lambda: _mesh_objective(0.0, traced_mesh=False))(),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = jax.block_until_ready(calls[case]())
    assert np.isfinite(np.asarray(result)).all()
