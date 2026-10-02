"""Every distributed executable passes mesh position as data (#1441).

CPU compilation also runs the SPMD partitioner: inspect compiled HLO, not
just StableHLO, where a device-identity operation can still be hidden.
The root conftest requests two virtual host devices before importing JAX.
"""
from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, LorentzPole, Simulation


# Graded run+CPML and extended wire ports refuse before compilation.
# Distributed forward admits uniform spacing through an explicit NU profile.
CASES = [
    (lane, mode, boundary)
    for lane in ("uniform", "graded")
    for mode in ("run", "forward", "gradient")
    for boundary in ("pec", "cpml")
    if (lane, mode, boundary) != ("graded", "run", "cpml")
] + [("uniform", mode, "cpml") for mode in ("ntff", "debye", "lumped")] + [
    ("uniform", "lorentz", "cpml"),
    ("graded", "forward-debye", "cpml"),
    ("graded", "gradient-debye", "cpml"),
    ("graded", "gradient-lorentz", "cpml"),
    ("graded", "gradient-sharded", "cpml"),
]


def model(lane, mode, boundary):
    material_kind = mode.split("-")[-1]
    mode = mode.split("-")[0]
    profile = np.full(15, 1e-3)
    if lane == "graded":
        profile[3:5] *= [0.9, 1.1]
    sim = Simulation(freq_max=15e9, domain=(15e-3, 7e-3, 7e-3), dx=1e-3,
                     boundary=boundary, cpml_layers=2, precision="float32",
                     **({"dx_profile": profile} if lane == "graded" or mode in ("forward", "gradient") else {}))
    pos = (6e-3, 3e-3, 3e-3)
    if mode == "lumped":
        sim.add_port(pos, "ez", impedance=50)
    else:
        amplitude = 1e-9 if material_kind == "sharded" else 1.
        sim.add_source(pos, "ez", amplitude_kind=("current" if material_kind == "sharded" else "field"),
                       waveform=lambda t: amplitude * jnp.cos(t * 2e10))
    sim.add_probe(pos, "ez")
    if mode == "ntff":
        sim.add_ntff_box((3e-3, 2e-3, 2e-3), (12e-3, 5e-3, 5e-3), freqs=[5e9])
    if material_kind in ("debye", "lorentz"):
        sim.add_material("debye", eps_r=4.,
                         **({"debye_poles": [DebyePole(delta_eps=1., tau=1e-11)]}
                            if material_kind == "debye" else
                            {"lorentz_poles": [LorentzPole(omega_0=2*np.pi*8e9,
                                delta=2*np.pi*1e9, kappa=(2*np.pi*8e9)**2)]}))
        sim.add(Box((5e-3, 2e-3, 2e-3), (9e-3, 5e-3, 5e-3)), material="debye")
    if material_kind == "sharded":
        sim.add(Box((10e-3, 4e-3, 4e-3), (11e-3, 5e-3, 5e-3)), material="pec")
    return sim


def assert_no_device_identity(hlo):
    # Both spellings cover HLO and StableHLO diagnostics.
    for forbidden in ("partition-id", "replica-id", "partition_id", "replica_id"):
        assert forbidden not in hlo, f"distributed executable contains {forbidden}"


@pytest.mark.parametrize("lane,mode,boundary", CASES)
def test_distributed_program_has_no_device_identity(monkeypatch, lane, mode, boundary):
    devices = jax.devices("cpu")[:2]
    if len(devices) != 2:
        pytest.skip("requires two virtual CPU devices (root conftest)")
    import rfx.runners.distributed_v2 as uniform
    import rfx.runners.distributed_nu as graded

    checked = []
    differentiated = []

    class JaxProxy:
        def __getattr__(self, name):
            return getattr(jax, name)

        def jit(self, fun, *args, **kwargs):
            compiled = jax.jit(fun, *args, **kwargs)

            @wraps(fun)
            def invoke(*values, **kw):
                if not any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves((values, kw))):
                    inspected = compiled
                    if mode.startswith("gradient") and fun.__name__ == "run_fn":
                        # Differentiate the actual staged scan, with all its inputs
                        # (including rank) dynamic, exactly as forward's AD does.
                        def loss(*a, **k):
                            return jnp.sum(fun(*a, **k)[1] ** 2)
                        differentiated.append(fun.__name__)
                        inspected = jax.jit(jax.value_and_grad(
                            loss, argnums=1, allow_int=True))
                    hlo = inspected.lower(*values, **kw).compile().as_text()
                    assert_no_device_identity(hlo)
                    checked.append(hlo)
                return compiled(*values, **kw)
            return invoke

    monkeypatch.setattr(uniform, "jax", JaxProxy())
    monkeypatch.setattr(graded, "jax", JaxProxy())
    sim = model(lane, mode, boundary)
    if mode.startswith(("forward", "gradient")):
        grid = sim._build_nonuniform_grid()
        eps = jnp.ones(grid.shape, jnp.float32)

        overrides = {"eps_override": eps}
        if mode.endswith("sharded"):
            overrides = {name: sim.shard_distributed_override(value) for name, value in
                         (("eps_override", eps), ("sigma_override", eps * .01),
                          ("pec_occupancy_override", eps * .1))}
        sim.forward(n_steps=4, distributed=True, devices=devices,
                    **overrides, skip_preflight=True)
    else:
        kwargs = dict(n_steps=4, devices=devices, skip_preflight=True)
        if mode == "lumped":
            kwargs.update(compute_s_params=True, s_param_freqs=[5e9], s_param_n_steps=4)
        sim.run(**kwargs)
    if mode.startswith("gradient"):
        assert differentiated == ["run_fn"], "the backward scan must be inspected"
    assert checked, "the contract must inspect an executable"
    assert any("while(" in hlo or "while (" in hlo for hlo in checked), "time loop was not lowered"
