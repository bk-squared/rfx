"""Every distributed executable passes mesh position as data (#1441).

CPU compilation also runs the SPMD partitioner: inspect compiled HLO, not
just StableHLO, where a device-identity operation can still be hidden.
One fresh subprocess uses the same two virtual host devices as root conftest,
with XLA dumping every optimized module, including eager and setup JITs.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

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


# Each forward case executes the public value and value-and-gradient paths.
# Keep the independent setup, scan, AD, and eager-result compilations visible.
PROGRAMS = [case for case in CASES if not case[1].startswith("forward")] + [
    ("uniform", "gradient-debye", "cpml"),
    ("graded", "gradient-checkpoint", "cpml"),
]


def exercise_programs():
    devices = jax.devices("cpu")
    assert len(devices) == 2, devices
    for lane, mode, boundary in PROGRAMS:
        sim = model(lane, mode, boundary)
        kwargs = dict(n_steps=4, devices=devices, skip_preflight=True)
        if mode.startswith("gradient"):
            grid = sim._build_nonuniform_grid()
            eps = jnp.full(grid.shape, 1.5, jnp.float32)
            overrides = {}
            if mode.endswith("sharded"):
                eps = sim.shard_distributed_override(eps)
                overrides = {name: sim.shard_distributed_override(
                    np.full(grid.shape, value, np.float32)) for name, value in
                    (("sigma_override", .01), ("pec_occupancy_override", .1))}
            if mode.endswith("checkpoint"):
                kwargs["checkpoint_every"] = 2

            def loss(design):
                trace = sim.forward(distributed=True, eps_override=design,
                                    **overrides, **kwargs).time_series
                return jnp.sum(trace ** 2)

            value = loss(eps)
            differentiated = jax.value_and_grad(loss)(eps)
            jax.block_until_ready((value, differentiated))
            assert all(np.isfinite(np.asarray(a)).all() for a in
                       jax.tree.leaves((value, differentiated)))
        else:
            if mode == "lumped":
                kwargs.update(compute_s_params=True, s_param_freqs=[5e9],
                              s_param_n_steps=4)
            result = sim.run(**kwargs)
            jax.block_until_ready((result.time_series, result.state, result.ntff_data))
        print(f"completed {lane}-{mode}-{boundary}", flush=True)


def assert_clean_dumps(directory):
    modules = sorted(directory.rglob("*after_optimizations*.txt"))
    assert modules, "no optimized XLA modules were dumped"
    violations = []
    for module in modules:
        text = module.read_text()
        for op in ("partition-id", "replica-id", "partition_id", "replica_id"):
            if op in text:
                violations.append(f"{module.name}: {op}")
    assert not violations, "device identity in optimized modules:\n" + "\n".join(violations)
    assert any(" while(" in m.read_text() for m in modules), "no time loop was compiled"
    return len(modules)


def test_distributed_programs_have_no_device_identity(tmp_path, record_property):
    root = Path(__file__).resolve().parents[2]
    dump = tmp_path / "hlo"
    dump.mkdir()
    env = dict(os.environ, PYTHONPATH=str(root), JAX_PLATFORMS="cpu",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               XLA_FLAGS=f"--xla_force_host_platform_device_count=2 --xla_dump_to={dump}",
               # A persistent compilation cache would hide optimized modules.
               JAX_ENABLE_COMPILATION_CACHE="false")
    started = time.monotonic()
    run = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                         cwd=root, env=env, capture_output=True, text=True, timeout=240)
    elapsed = time.monotonic() - started
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.count("completed ") == len(PROGRAMS), run.stdout
    count = assert_clean_dumps(dump)
    record_property("wall_seconds", elapsed)
    record_property("optimized_dump_files", count)
    print(f"{len(PROGRAMS)} programs; {count} optimized dump files; {elapsed:.2f} s")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--worker"], sys.argv
    exercise_programs()
