"""The distributed time loop owns slabs, never whole-domain setup buffers.

Measure in a fresh process so arrays retained by unrelated tests cannot satisfy
(or violate) the live-array invariant. No distributed runtime is started: the
root conftest supplies two virtual CPU devices.
"""

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, SingleDeviceSharding
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation
from rfx.materials.debye import DebyeCoeffs
from rfx.materials.lorentz import LorentzCoeffs, lorentz_pole
from rfx.runners import distributed_v2
from rfx.runners._distributed_common import shard_stacked, shard_x_slabs, split_array_x

pytestmark = pytest.mark.distributed


def _build_box(case):
    sim = Simulation(
        freq_max=15e9,
        domain=((14 if case == "pad" else 15) * 1e-3, 7e-3, 7e-3), dx=1e-3,
        boundary="cpml" if case == "cpml" else "pec",
        cpml_layers=2 if case == "cpml" else 0,
        dz_profile=np.array([0.8, 1.2, 0.9, 1.1, 1.0, 0.8, 1.2]) * 1e-3
        if case == "nu" else None,
    )
    if case == "volume":
        sim.add(Box((5e-3, 2e-3, 2e-3), (7e-3, 3e-3, 5e-3)), material="pec")
    elif case in ("debye", "lorentz"):
        poles = (
            {"debye_poles": [DebyePole(delta_eps=1.0, tau=1e-11)]}
            if case == "debye" else
            {"lorentz_poles": [lorentz_pole(
                delta_eps=1.0, omega_0=2 * np.pi * 3e9, delta=1e9)]}
        )
        sim.add_material("dispersive", eps_r=4.0, **poles)
        sim.add(Box((5e-3, 2e-3, 2e-3), (10e-3, 6e-3, 6e-3)), material="dispersive")
    sim.add_source((8e-3, 4e-3, 4e-3), "ez", amplitude_kind="field")
    sim.add_probe((6e-3, 4e-3, 4e-3), "ez")
    return sim


def _measure(case, explicit_args):
    sim = _build_box(case)
    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2
    grid = sim._build_nonuniform_grid() if case == "nu" else sim._build_grid()
    cells = int(np.prod(grid.shape))
    records = []
    scan = distributed_v2.lax.scan

    def traced_scan(body, carry, *args, **kwargs):
        per_device = {str(d): 0 for d in devices}
        whole = []
        for arr in jax.live_arrays():
            for shard in arr.addressable_shards:
                per_device[str(shard.device)] += shard.data.nbytes
                # Per device, whatever the sharding: a whole-domain array kept
                # on one device and one replicated onto every device (P())
                # cost a device the same memory.
                if shard.data.size >= cells:
                    whole.append((str(shard.device), arr.shape, str(arr.dtype),
                                  type(arr.sharding).__name__))
        # Both scan entry variants capture the coefficient tuple directly;
        # in the explicit-argument variant its leaves are JIT tracers.
        coeffs = {
            type(value).__name__: value
            for value in inspect.getclosurevars(body).nonlocals.values()
            if isinstance(value, (DebyeCoeffs, LorentzCoeffs))
        }
        assert set(coeffs) == {"DebyeCoeffs", "LorentzCoeffs"}
        placeholder_shapes = {}
        for material, coeff_type in (("debye", "DebyeCoeffs"), ("lorentz", "LorentzCoeffs")):
            if case == material:
                continue
            for slot, value in (("state", carry[material]), ("coeffs", coeffs[coeff_type])):
                for name, array in zip(value._fields, value):
                    placeholder_shapes[f"{material}.{slot}.{name}"] = array.shape
        records.append({"whole": whole, "bytes": per_device, "placeholders": placeholder_shapes})
        return scan(body, carry, *args, **kwargs)

    distributed_v2.lax = SimpleNamespace(**{**vars(distributed_v2.lax), "scan": traced_scan})
    if explicit_args:
        # Exercise the multi-process argument path without starting jax.distributed.
        distributed_v2.jax = SimpleNamespace(**{**vars(jax), "process_index": lambda: 1})
    result = sim.run(n_steps=3, devices=devices)
    jax.block_until_ready(result.time_series)
    assert records, "the memory gate must observe the time-stepping scan"
    print("MEMORY_STAGING " + json.dumps(records))


@pytest.mark.parametrize("case,explicit_args", [
    ("pec", False), ("cpml", False), ("pad", False), ("volume", False),
    ("nu", False), ("debye", False), ("lorentz", False),
    ("pec", True), ("cpml", True),
])
def test_time_loop_holds_only_local_slabs(case, explicit_args):
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    run = subprocess.run(
        [sys.executable, "-W", "ignore", str(Path(__file__).resolve()), case, str(int(explicit_args))],
        env=env, text=True, capture_output=True, timeout=45,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    records = [json.loads(line.removeprefix("MEMORY_STAGING "))
               for line in run.stdout.splitlines() if line.startswith("MEMORY_STAGING ")]
    assert len(records) == 1, run.stdout
    for record in records[0]:
        assert not record["whole"], f"whole-domain single-device arrays: {record['whole']}"
        sizes = list(record["bytes"].values())
        assert len(sizes) == 2 and min(sizes) > 0
        assert max(sizes) <= 1.25 * min(sizes), f"unbalanced device bytes: {record['bytes']}"
        shapes = record["placeholders"]
        expected_slots = 12 if case == "debye" else 8 if case == "lorentz" else 20
        assert len(shapes) == expected_slots
        for name, shape in shapes.items():
            assert np.prod(shape) <= 2, f"per-cell placeholder {name}: {shape}"


@pytest.mark.parametrize("nx", [8, 7])
@pytest.mark.parametrize("kind,pad_value", [("eps_r", 1.0), ("sigma", 0.0), ("mu_r", 1.0), ("pec", False)])
def test_direct_slabs_are_bit_identical(nx, kind, pad_value):
    devices = jax.devices("cpu")[:2]
    if len(devices) < 2:
        pytest.skip("requires two virtual CPU devices")
    sharding = NamedSharding(Mesh(np.array(devices), ("x",)), P("x"))
    rng = np.random.default_rng(1053)
    values = rng.standard_normal((nx, 3, 4)).astype(np.float32)
    if kind == "pec":
        values = values > 0
    arr = jnp.asarray(values)
    pad_x = nx % 2
    if pad_x:
        arr = jnp.pad(arr, ((0, pad_x), (0, 0), (0, 0)),
                      constant_values=True if kind == "pec" else pad_value)
    expected = shard_stacked(split_array_x(arr, 2, 1, pad_value), sharding)
    actual = shard_x_slabs(arr, 2, (nx + pad_x) // 2, 1, pad_value, sharding)
    assert actual.dtype == expected.dtype
    for got, want in zip(actual.addressable_shards, expected.addressable_shards):
        assert got.device == want.device and got.index == want.index
        assert np.array_equal(np.asarray(got.data), np.asarray(want.data))


def test_direct_slabs_callback_only_builds_addressable_shards(monkeypatch):
    """Emulate rank 1's callback index; fail if rank 0 is staged as well."""
    values = jnp.arange(8 * 3 * 4, dtype=jnp.float32).reshape(8, 3, 4)
    seen = []
    reads = []

    class ReadRecorder:
        shape = values.shape
        dtype = values.dtype

        def __getitem__(self, index):
            reads.append(index)
            return values[index]

    # Same signature as jax 0.4.33's make_array_from_callback (the VESSL image):
    # no ``dtype`` parameter, which arrived only after jax 0.5.0. Passing one
    # from shard_x_slabs would fail there, and fails here.
    def local_callback(shape, sharding, callback):
        assert shape == (12, 3, 4)
        index = (slice(6, 12), slice(None), slice(None))
        result = callback(index)
        seen.append(np.asarray(result))
        return result

    monkeypatch.setattr(jax, "make_array_from_callback", local_callback)
    shard_x_slabs(ReadRecorder(), 2, 4, 1, 0.0, None)
    assert len(seen) == 1
    assert reads == [slice(3, 8)]
    expected = np.pad(np.asarray(values)[3:8], ((0, 1), (0, 0), (0, 0)))
    assert np.array_equal(seen[0], expected)


if __name__ == "__main__":
    _measure(sys.argv[1], bool(int(sys.argv[2])))
