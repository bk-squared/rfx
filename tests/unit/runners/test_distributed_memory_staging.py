"""The distributed time loop owns slabs, never whole-domain setup buffers.

Measure in a fresh process so arrays retained by unrelated tests cannot satisfy
(or violate) the live-array invariant. No distributed runtime is started: the
root conftest supplies two virtual CPU devices.
"""

from functools import partial
import inspect
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from types import FunctionType, SimpleNamespace

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, SingleDeviceSharding
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation
from rfx.core.yee import EPS_0, MaterialArrays
from rfx.materials.debye import DebyeCoeffs
from rfx.materials.lorentz import LorentzCoeffs, lorentz_pole
from rfx.runners import distributed_v2
from rfx.runners import _distributed_common as common
from rfx.runners._distributed_common import shard_stacked, shard_x_slabs, split_array_x

pytestmark = pytest.mark.distributed

_DISPERSION_CASES = ("debye", "debye2", "lorentz", "mixed", "pad_debye", "cpml_lorentz")


def _dispersion_kinds(case):
    return (("debye", "lorentz") if case == "mixed" else
            ("debye",) if case in ("debye", "debye2", "pad_debye") else
            ("lorentz",) if case in ("lorentz", "cpml_lorentz") else ())


def _build_box(case):
    sim = Simulation(
        freq_max=15e9,
        domain=((16 if case == "pad_debye" else 14 if case == "pad" else 15) * 1e-3,
                7e-3, 7e-3), dx=1e-3,
        boundary="cpml" if case in ("cpml", "cpml_lorentz") else "pec",
        cpml_layers=2 if case in ("cpml", "cpml_lorentz") else 0,
        dz_profile=np.array([0.8, 1.2, 0.9, 1.1, 1.0, 0.8, 1.2]) * 1e-3
        if case == "nu" else None,
    )
    if case == "volume":
        sim.add(Box((5e-3, 2e-3, 2e-3), (7e-3, 3e-3, 5e-3)), material="pec")
    elif case == "lossy":
        # eps_r and sigma vary in space, so a compiler cannot reduce them to
        # scalars; a captured copy would show up as a per-cell HLO literal.
        sim.add_material("lossy", eps_r=4.4, sigma=0.02)
        sim.add(Box((5e-3, 2e-3, 2e-3), (10e-3, 6e-3, 6e-3)), material="lossy")
    elif case in _DISPERSION_CASES:
        poles = {}
        if "debye" in _dispersion_kinds(case):
            poles["debye_poles"] = [DebyePole(delta_eps=1.0, tau=1e-11)]
            if case == "debye2":
                poles["debye_poles"].append(DebyePole(delta_eps=0.5, tau=3e-11))
        if "lorentz" in _dispersion_kinds(case):
            poles["lorentz_poles"] = [lorentz_pole(
                delta_eps=1.0, omega_0=2 * np.pi * 3e9, delta=1e9)]
        sim.add_material("dispersive", eps_r=4.0, **poles)
        sim.add(Box((5e-3, 2e-3, 2e-3), (10e-3, 6e-3, 6e-3)), material="dispersive")
        if case == "mixed":
            sim.add(Box((5e-3, 2e-3, 2e-3), (7e-3, 3e-3, 5e-3)), material="pec")
            sim.add_material("lossy", eps_r=4.4, sigma=0.02)
            sim.add(Box((10e-3, 2e-3, 2e-3), (13e-3, 6e-3, 6e-3)), material="lossy")
    sim.add_source((8e-3, 4e-3, 4e-3), "ez", amplitude_kind="field")
    sim.add_probe((6e-3, 4e-3, 4e-3), "ez")
    return sim


def _measure(case, multi_process):
    sim = _build_box(case)
    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2
    grid = sim._build_nonuniform_grid() if case == "nu" else sim._build_grid()
    cells = int(np.prod(grid.shape))
    records = []
    scan = distributed_v2.lax.scan
    boundary_ghosts_true = None
    hlo_constants = []
    hlo_constants_parsed = []
    slab_cells = cells // len(devices)
    nx_local = (grid.shape[0] + (-grid.shape[0]) % len(devices)) // len(devices) + 2
    init_extents = {kind: [] for kind in _dispersion_kinds(case)}
    ghost_checks = []

    # Record every call of the real init functions by their code objects, so
    # any call path (module attribute, direct import, a helper elsewhere) is
    # seen. Active during setup only: traced_scan switches it off.
    from rfx.materials.debye import init_debye as _init_debye
    from rfx.materials.lorentz import init_lorentz as _init_lorentz
    init_codes = {_init_debye.__code__: "debye", _init_lorentz.__code__: "lorentz",
                  common.debye_pole_coeffs.__code__: "debye",
                  common.lorentz_pole_coeffs.__code__: "lorentz"}

    def profile_init(frame, event, arg):
        if event == "call" and frame.f_code in init_codes:
            if "shape" in frame.f_locals:
                init_extents[init_codes[frame.f_code]].append(int(frame.f_locals["shape"][0]))
            materials = frame.f_locals.get("materials")
            if materials is not None:
                init_extents.setdefault(init_codes[frame.f_code], []).append(
                    int(materials.eps_r.shape[0]))

    sys.setprofile(profile_init)

    def traced_jit(f, *args, **kwargs):
        entry = jax.jit(f, *args, **kwargs)

        def run(*entry_args, **entry_kwargs):
            nonlocal boundary_ghosts_true
            # #931: inspect the actual PEC mask argument before it becomes a
            # tracer. Physical-boundary ghost rows must remain False.
            bound = inspect.signature(f).bind(*entry_args, **entry_kwargs).arguments
            mask = bound.get("pec_mask_arg")
            if mask is not None:
                slabs = np.asarray(mask).reshape(len(devices), -1, *mask.shape[1:])
                boundary_ghosts_true = int(slabs[0, 0].sum() + slabs[-1, -1].sum())
            # The compiled program itself: a per-cell array that reaches the
            # scan by any route (closure, dict, host numpy copy, global) is
            # compiled in as a literal of at least one slab's worth of elements.
            hlo = entry.lower(*entry_args, **entry_kwargs).compile().as_text()
            matches = list(re.finditer(
                r"(\w+)\[([\d,]*)\](?:\{[^}]*\})?\s+constant\(", hlo))
            # Every program has scalar constants; none parsed means the HLO
            # print format changed and this check would pass vacuously.
            hlo_constants_parsed.append(len(matches))
            for match in matches:
                dims = [int(d) for d in match.group(2).split(",") if d]
                if dims and int(np.prod(dims)) >= slab_cells:
                    hlo_constants.append((match.group(1), dims))
            result = entry(*entry_args, **entry_kwargs)
            # Host inspection caches single-device shard views in JAX. Do it
            # after the live-array scan sample so those inspection-only views
            # cannot look like retained full-domain multi-pole setup copies.
            for kind in _dispersion_kinds(case):
                partial_coeffs = bound[kind + "_coeffs_arg"]
                assert partial_coeffs.ca is partial_coeffs.cb is partial_coeffs.cc is None

                @partial(shard_map, mesh=Mesh(np.array(devices), ("x",)),
                         in_specs=(P("x"), P("x")), out_specs=P("x"), check_rep=False)
                def finish(c, m):
                    means = common.slab_e_component_materials(m, nx_local - 2, grid.shape[0])
                    return common.slab_dispersion_coeffs(c, means, grid.dt, nx_local - 2, grid.shape[0])

                coeffs = jax.jit(finish)(partial_coeffs, bound["materials_arg"])
                # #1302: rows outside the domain hold a vacuum cell's
                # coefficients (ca 1, cb dt/eps_0, cc 1/eps_0, pole terms 0)
                # (float32 arithmetic, as the coefficient builders run it)
                eps0 = np.float32(EPS_0)
                vacuum = {"ca": np.float32(1.0), "cb": np.float32(grid.dt) / eps0,
                          "cc": np.float32(1.0) / eps0}
                for name, field in zip(coeffs._fields, coeffs):
                    pad = np.float32(vacuum.get(name, 0.0))
                    # a per-E-component field (#1260) is an (x, y, z) tuple
                    for arr in jax.tree_util.tree_leaves(field):
                        slabs = np.asarray(arr).reshape(len(devices), -1, nx_local, *grid.shape[1:])
                        for row in (slabs[0, :, 0], slabs[-1, :, -1]):
                            expected = np.full_like(row, pad)
                            assert np.array_equal(row, expected), (kind, name, "physical ghost pad")
                            assert row.tobytes() == expected.tobytes(), (kind, name, "ghost bits")
                    ghost_checks.append(kind + "." + name)
            return result

        return run

    def traced_scan(body, carry, *args, **kwargs):
        sys.setprofile(None)  # setup is over; keep the scan trace fast
        captured = []
        seen = set()

        def walk(value, path):
            if isinstance(value, jax.core.Tracer) or id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, jax.Array):
                if value.ndim >= 3:
                    captured.append((path, value.shape, value.nbytes))
                return
            if isinstance(value, FunctionType):
                for name, cell in zip(value.__code__.co_freevars, value.__closure__ or ()):
                    try:
                        child = cell.cell_contents
                    except ValueError:
                        continue
                    walk(child, f"{path}.{name}")
            if isinstance(value, (tuple, list)):
                for name, child in zip(getattr(value, "_fields", range(len(value))), value):
                    walk(child, f"{path}[{name}]")
            if isinstance(value, partial):
                walk(value.func, f"{path}.func")
                walk(value.args, f"{path}.args")
                for name, child in (value.keywords or {}).items():
                    walk(child, f"{path}.keywords[{name}]")
            if hasattr(value, "__wrapped__"):
                walk(value.__wrapped__, f"{path}.__wrapped__")

        # Per-cell captures become whole-domain compiled constants on every
        # device. NU's one-dimensional spacing arrays are intentionally allowed.
        walk(body, "scan")
        per_device = {str(d): 0 for d in devices}
        whole = []
        for arr in jax.live_arrays():
            for shard in arr.addressable_shards:
                per_device[str(shard.device)] += shard.data.nbytes
                # Per device, whatever the sharding: a whole-domain array kept
                # on one device and one replicated onto every device (P())
                # cost a device the same memory.
                # An x-sharded array is judged by the spatial extent of its
                # shard (the last three axes): a correct multi-pole model's
                # polarization shard (n_poles, nx_local, ny, nz) can hold more
                # elements than the domain has cells without covering it. A
                # single-device or replicated array is judged by its element
                # count, so a device-stacked copy (n_devices, nx_local, ny, nz)
                # kept on one device is still whole-domain.
                split = (not arr.sharding.is_fully_replicated
                         and len(arr.sharding.device_set) > 1)
                spatial = (int(np.prod(shard.data.shape[-3:])) if shard.data.ndim >= 3
                           else shard.data.size)
                if (spatial if split else shard.data.size) >= cells:
                    whole.append((str(shard.device), arr.shape, str(arr.dtype),
                                  type(arr.sharding).__name__))
        # Both topologies capture coefficient tuples with JIT-tracer leaves.
        coeffs = {
            type(value).__name__: value
            for value in inspect.getclosurevars(body).nonlocals.values()
            if isinstance(value, (DebyeCoeffs, LorentzCoeffs))
        }
        assert set(coeffs) == {"DebyeCoeffs", "LorentzCoeffs"}
        placeholder_shapes = {}
        for material, coeff_type in (("debye", "DebyeCoeffs"), ("lorentz", "LorentzCoeffs")):
            if material in _dispersion_kinds(case):  # real arrays, not placeholders
                continue
            for slot, value in (("state", carry[material]), ("coeffs", coeffs[coeff_type])):
                for name, array in zip(value._fields, value):
                    placeholder_shapes[f"{material}.{slot}.{name}"] = array.shape
        records.append({"whole": whole, "bytes": per_device, "placeholders": placeholder_shapes,
                        "captured": captured, "hlo_constants": hlo_constants,
                        "hlo_constants_parsed": hlo_constants_parsed,
                        "init_extents": init_extents, "nx_local": nx_local,
                        "ghost_checks": ghost_checks,
                        "pec_mask_boundary_ghosts_true": boundary_ghosts_true})
        return scan(body, carry, *args, **kwargs)

    distributed_v2.lax = SimpleNamespace(**{**vars(distributed_v2.lax), "scan": traced_scan})
    runner_jax = SimpleNamespace(**{**vars(jax), "jit": traced_jit})
    if multi_process:
        # Exercise the multi-process topology without starting jax.distributed.
        runner_jax.process_index = lambda: 1
    distributed_v2.jax = runner_jax
    result = sim.run(n_steps=3, devices=devices)
    jax.block_until_ready(result.time_series)
    assert records, "the memory gate must observe the time-stepping scan"
    print("MEMORY_STAGING " + json.dumps(records))


@pytest.mark.parametrize("case,multi_process", [
    ("pec", False), ("cpml", False), ("pad", False), ("volume", False),
    ("nu", False), ("debye", False), ("debye2", False), ("lorentz", False),
    ("lossy", False), ("pec", True), ("cpml", True), ("volume", True), ("lossy", True),
    ("mixed", False), ("pad_debye", False), ("cpml_lorentz", False),
    *((case, True) for case in _DISPERSION_CASES),
])
def test_time_loop_holds_only_local_slabs(case, multi_process):
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
           "PYTHONPATH": str(Path(__file__).resolve().parents[3])}
    run = subprocess.run(
        [sys.executable, "-W", "ignore", str(Path(__file__).resolve()), case, str(int(multi_process))],
        env=env, text=True, capture_output=True, timeout=45,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    records = [json.loads(line.removeprefix("MEMORY_STAGING "))
               for line in run.stdout.splitlines() if line.startswith("MEMORY_STAGING ")]
    assert len(records) == 1, run.stdout
    for record in records[0]:
        assert not record["captured"], f"per-cell scan captures: {record['captured']}"
        assert record["hlo_constants_parsed"] and min(record["hlo_constants_parsed"]) > 0, (
            "no HLO constant parsed: the literal check would pass vacuously")
        assert not record["hlo_constants"], (
            f"per-cell literals compiled into the scan: {record['hlo_constants']}")
        assert not record["whole"], f"whole-domain single-device arrays: {record['whole']}"
        sizes = list(record["bytes"].values())
        assert len(sizes) == 2 and min(sizes) > 0
        assert max(sizes) <= 1.25 * min(sizes), f"unbalanced device bytes: {record['bytes']}"
        shapes = record["placeholders"]
        kinds = _dispersion_kinds(case)
        expected_slots = 20 - (8 if "debye" in kinds else 0) - (12 if "lorentz" in kinds else 0)
        assert len(shapes) == expected_slots
        for name, shape in shapes.items():
            assert np.prod(shape) <= 2, f"per-cell placeholder {name}: {shape}"
        for kind in kinds:
            extents = record["init_extents"][kind]
            # The shared slab convention needs only its existing halo.
            assert extents and max(extents) <= record["nx_local"], (
                f"whole-domain {kind} initialization: {extents}; nx_local={record['nx_local']}")
        assert len(record["ghost_checks"]) == (5 if "debye" in kinds else 0) + (6 if "lorentz" in kinds else 0)
        if case in ("volume", "mixed"):
            assert record["pec_mask_boundary_ghosts_true"] == 0, (
                "PEC mask physical-boundary ghost rows must be False (#931): "
                f"{record['pec_mask_boundary_ghosts_true']} True cells")


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


_MANY_DEVICE_SLABS = """
import itertools, numpy as np, jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from rfx.runners._distributed_common import shard_stacked, shard_x_slabs, split_array_x
devices = jax.devices("cpu")
assert len(devices) == 4, devices
rng = np.random.default_rng(931)
checked = 0
for n, nx, (kind, pad_value) in itertools.product(
        (3, 4), (8, 9, 10, 11, 13),
        (("eps_r", 1.0), ("sigma", 0.0), ("pec", False))):
    mesh_devices = devices[:n][::-1] if nx % 2 else devices[:n]  # also a permuted mesh order
    sharding = NamedSharding(Mesh(np.array(mesh_devices), ("x",)), P("x"))
    values = rng.standard_normal((nx, 3, 4)).astype(np.float32)
    if kind == "pec":
        values = values > 0
    pad_x = (-nx) % n
    arr = jnp.asarray(values)
    if pad_x:
        arr = jnp.pad(arr, ((0, pad_x), (0, 0), (0, 0)),
                      constant_values=True if kind == "pec" else pad_value)
    nx_per = (nx + pad_x) // n
    expected = shard_stacked(split_array_x(arr, n, 1, pad_value), sharding)
    actual = shard_x_slabs(arr, n, nx_per, 1, pad_value, sharding)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    for got, want in zip(actual.addressable_shards, expected.addressable_shards):
        assert got.device == want.device and got.index == want.index
        assert np.array_equal(np.asarray(got.data), np.asarray(want.data)), (n, nx, kind)
    checked += 1
print("MANY_DEVICE_SLABS_OK", checked)
"""


def test_direct_slabs_bit_identical_on_three_and_four_devices():
    """Interior ranks (a slab with neighbours on both sides) exist only with 3+ devices."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
           "PYTHONPATH": str(Path(__file__).resolve().parents[3])}
    run = subprocess.run([sys.executable, "-W", "ignore", "-c", _MANY_DEVICE_SLABS],
                         env=env, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "MANY_DEVICE_SLABS_OK 30" in run.stdout, run.stdout

def test_direct_slabs_callback_only_builds_addressable_shards(monkeypatch):
    """Emulate rank 1's callback index; fail if rank 0 is staged as well."""
    values = jnp.arange(8 * 3 * 4, dtype=jnp.float32).reshape(8, 3, 4)
    seen = []
    reads = []

    class ReadRecorder:
        shape = values.shape
        ndim = values.ndim
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


def _direct_dispersion_slabs():
    """Owned coefficients, fixed pole terms and physical pads on uneven meshes."""
    import itertools
    from rfx.materials.debye import init_debye
    from rfx.materials.lorentz import init_lorentz
    from rfx.runners import distributed_nu as nu
    from tests.unit.runners.test_distributed_nu_forward_staging import _completed_slabs, _check_slabs

    devices = jax.devices("cpu")
    assert len(devices) == 4
    rng = np.random.default_rng(1208)
    checked = 0
    for n, pad, poles in itertools.product((2, 3, 4), (0, 1), (1, 2)):
        nx = n * 4 - pad
        shape = (nx, 3, 4)
        mesh_devices = devices[:n][::-1] if pad else devices[:n]
        mesh = Mesh(np.array(mesh_devices), ("x",))
        shd = NamedSharding(mesh, P("x"))
        dt = np.float64(1e-12)
        sg = SimpleNamespace(n_devices=n, ghost_width=1, pad_x=pad, nx=nx,
                             nx_padded=n * 4, nx_per_rank=4, nx_local=6, ny=3, nz=4,
                             dt=dt)
        materials = MaterialArrays(
            eps_r=jnp.asarray(rng.uniform(1, 6, shape).astype(np.float32)),
            sigma=jnp.asarray(rng.uniform(0, 0.1, shape).astype(np.float32)),
            mu_r=jnp.ones(shape, dtype=jnp.float32))
        masks = [jnp.asarray(rng.random(shape) > 0.5) for _ in range(poles)]
        padded, padded_masks = materials, masks
        if pad:
            widths = ((0, pad), (0, 0), (0, 0))
            padded = MaterialArrays(*(jnp.pad(a, widths, constant_values=v)
                                      for a, v in zip(materials[:3], (1., 0., 1.))))
            padded_masks = [jnp.pad(m, widths, constant_values=False) for m in masks]
        debye = ([DebyePole(1.0, 1e-11), DebyePole(0.5, 3e-11)][:poles], padded_masks)
        lorentz = ([lorentz_pole(1.0, 2 * np.pi * 3e9, 1e9),
                    lorentz_pole(0.5, 2 * np.pi * 5e9, 2e9)][:poles], padded_masks)
        actual = common.stage_dispersion_slabs(padded, dt, debye, lorentz, n, 4, 1, shd, nx=nx)
        placed = MaterialArrays(*(shard_x_slabs(a, n, 4, 1, v, shd)
                                  for a, v in zip(padded[:3], (1., 0., 1.))))
        for spec, init, split_c, split_s, got in zip(
                (debye, lorentz), (init_debye, init_lorentz),
                (nu.shard_debye_coeffs_x_slab, nu.shard_lorentz_coeffs_x_slab),
                (nu.shard_debye_state_x_slab, nu.shard_lorentz_state_x_slab), actual):
            coeffs, state = init(spec[0], materials, dt, mask=masks)
            expected = split_c(coeffs, sg, mesh), split_s(state, sg, mesh)
            got = _completed_slabs(got, placed, sg, dt, mesh)
            checked += _check_slabs(got, expected, sg)
    print("DISPERSION_SLABS_OK", checked)


def _addressable_dispersion_slabs():
    """A remote rank stages only its own pole-mask halo before shard_map."""
    devices = jax.devices("cpu")[:2]
    mesh = Mesh(np.array(devices), ("x",))
    local_only = SimpleNamespace(mesh=mesh)
    values = jnp.arange(8 * 3 * 4, dtype=jnp.float32).reshape(8, 3, 4) + 1
    materials = MaterialArrays(values, values * 0.001, jnp.ones_like(values))
    mask = values > 10
    reads, constructions = [], []

    class MaskRecorder:
        shape = mask.shape
        ndim = mask.ndim
        dtype = mask.dtype

        def __getitem__(self, index):
            reads.append(index)
            return mask[index]

    def local_callback(shape, sharding, callback):
        assert shape == (12, 3, 4) and sharding is local_only
        return jax.device_put(callback((slice(6, 12), slice(None), slice(None))), devices[1])

    def pole_setup(poles, masks, dt, kind, got_mesh, nx_per, nx, shape):
        assert got_mesh is mesh and (nx_per, nx, shape) == (4, 8, (12, 3, 4))
        assert masks[0].devices() == {devices[1]}
        expected = np.pad(np.asarray(mask[3:8]), ((0, 1), (0, 0), (0, 0)))
        assert np.array_equal(np.asarray(masks[0]), expected)
        constructions.append(kind)
        return None

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax, "make_array_from_callback", local_callback)
        patch.setattr(common, "stage_slab_pole_coeffs", pole_setup)
        common.stage_dispersion_slabs(
            materials, np.float64(1e-12), ([DebyePole(1., 1e-11)], [MaskRecorder()]),
            ([lorentz_pole(1., 2 * np.pi * 3e9, 1e9)], [MaskRecorder()]),
            2, 4, 1, local_only, nx=8)
    assert reads == [slice(3, 8), slice(3, 8)]
    assert constructions == ["debye", "lorentz"]
    print("ADDRESSABLE_DISPERSION_OK")


@pytest.mark.parametrize("mode,marker", [
    # 36 coefficient/state arrays x (2 + 3 + 4) devices x 2 pads x 2 pole
    # counts; 20 arrays (720) before the per-component tuples of #1260.
    ("direct_dispersion", "DISPERSION_SLABS_OK 1296"),
    ("addressable_dispersion", "ADDRESSABLE_DISPERSION_OK"),
])
def test_dispersion_slabs_in_subprocess(mode, marker):
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
           "PYTHONPATH": str(Path(__file__).resolve().parents[3])}
    run = subprocess.run([sys.executable, "-W", "ignore", str(Path(__file__).resolve()), mode],
                         env=env, text=True, capture_output=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    assert marker in run.stdout, run.stdout


if __name__ == "__main__":
    if sys.argv[1] == "direct_dispersion":
        _direct_dispersion_slabs()
    elif sys.argv[1] == "addressable_dispersion":
        _addressable_dispersion_slabs()
    else:
        _measure(sys.argv[1], bool(int(sys.argv[2])))
