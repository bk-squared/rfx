"""Live Yee halos and a single post-scan probe reduction, checked against the full exchange."""

from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest
import matplotlib

from rfx import Box, DebyePole, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import FDTDState
from rfx.materials.lorentz import lorentz_pole
from rfx.runners import distributed_v2 as runner
from rfx.runners import _distributed_common as common

pytestmark = pytest.mark.distributed
_FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")
_STEPS = 8


def _full_exchange_component(field, mesh, n_devices):
    """Frozen origin/main 56bbb077 exchange; do not call production helpers."""
    @partial(shard_map, mesh=mesh, in_specs=P("x"), out_specs=P("x"),
             check_rep=False)
    def exchange(f):
        right_boundary = f[-2:-1, :, :]
        left_boundary = f[1:2, :, :]
        perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
        left_recv = lax.ppermute(right_boundary, "x", perm=perm_right)
        perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
        right_recv = lax.ppermute(left_boundary, "x", perm=perm_left)
        device_idx = lax.axis_index("x")
        left_val = jnp.where(device_idx > 0, left_recv, f[0:1, :, :])
        right_val = jnp.where(device_idx < n_devices - 1, right_recv, f[-1:, :, :])
        f = f.at[0:1, :, :].set(left_val)
        return f.at[-1:, :, :].set(right_val)
    return exchange(field)


def _full_h(st, mesh, n_devices):
    return st._replace(**{c: _full_exchange_component(getattr(st, c), mesh, n_devices)
                          for c in ("hx", "hy", "hz")})


def _full_e(st, mesh, n_devices):
    return st._replace(**{c: _full_exchange_component(getattr(st, c), mesh, n_devices)
                          for c in ("ex", "ey", "ez")})


def _full_sample(st, mesh, n_prb, specs, owners, **_kwargs):
    """Frozen owner-mask + per-step psum, with the original replicated shape."""
    if n_prb == 0:
        return jnp.zeros(0, dtype=jnp.float32)

    @partial(shard_map, mesh=mesh, in_specs=(P("x"),) * 6,
             out_specs=P(), check_rep=False)
    def sample(ex, ey, ez, hx, hy, hz):
        device_idx = lax.axis_index("x")
        samples = []
        for idx in range(n_prb):
            i, j, k, component = specs[idx]
            fields = dict(zip(_FIELDS, (ex, ey, ez, hx, hy, hz)))
            raw = fields[component][i, j, k]
            samples.append(jnp.where(device_idx == owners[idx], raw, 0.0))
        return lax.psum(jnp.stack(samples), "x")
    return sample(st.ex, st.ey, st.ez, st.hx, st.hy, st.hz)


@contextmanager
def _reference(patch, *, exchange=True, probes=True):
    with patch.context() as p:
        if exchange:
            p.setattr(runner, "_exchange_h_ghosts_shmap", _full_h)
            p.setattr(runner, "_exchange_e_ghosts_shmap", _full_e)
        if probes:
            p.setattr(runner, "sample_probes_shmap", _full_sample)
            scan = runner.lax.scan

            def old_scan(*args, **kwargs):
                carry, trace = scan(*args, **kwargs)
                # Adapter OUTSIDE the frozen scan: the new post-scan sum
                # sees a singleton axis and leaves the old trace unchanged.
                return carry, trace[:, None, :]

            p.setattr(runner, "lax", SimpleNamespace(**{**vars(lax), "scan": old_scan}))
        yield


def _build_model(boundary, n_devices, model="composed"):
    layers = 1 if boundary == "cpml" else 0
    spec = boundary if model == "plain" else BoundarySpec(
        x=boundary, y=Boundary(lo="pmc", hi=boundary), z=boundary)
    sim = Simulation(freq_max=15e9, domain=((16 - 2 * layers) * 1e-3, 4e-3, 4e-3),
                     dx=1e-3, boundary=spec, cpml_layers=layers)
    grid = sim._build_grid()
    nx = grid.shape[0]
    assert nx == 17 and nx % n_devices != 0, grid.shape
    width = (nx + n_devices - 1) // n_devices
    # Two devices have no interior rank: witness both sides of their seam.
    # With 3/4, rank 1 is interior and both its first and last real cells
    # have a source and all six probes, as do their adjacent seam cells.
    indices = sorted({width - 1, width, 2 * width - 1, 2 * width}
                     if n_devices > 2 else {width - 1, width})
    positions = [((i - grid.pad_x_lo) * grid.dx, 2e-3, 2e-3) for i in indices]
    for pos, i in zip(positions, indices):
        assert grid.position_to_index(pos)[0] == i
        for component in ("ey", "ez"):
            sim.add_source(pos, component, amplitude_kind="field",
                           waveform=lambda t: jnp.cos(t * 2e10))
        for component in _FIELDS:
            sim.add_probe(pos, component)
    if model != "plain":
        x = (width - grid.pad_x_lo) * grid.dx
        sim.add_material("lossy", eps_r=2.5, sigma=0.03)
        sim.add(Box((x - 1e-3, 1e-3, 1e-3), (x + 1e-3, 2e-3, 3e-3)), material="lossy")
        sim.add(Box((x, 3e-3, 1e-3), (x + 1e-3, 4e-3, 3e-3)), material="pec")
        sim.add_material("debye", eps_r=2.0,
                         debye_poles=[DebyePole(delta_eps=0.5, tau=1e-11)])
        sim.add_material("lorentz", eps_r=2.0, lorentz_poles=[lorentz_pole(
            delta_eps=0.5, omega_0=2 * np.pi * 3e9, delta=1e9)])
        sim.add(Box((x - 1e-3, 2e-3, 1e-3), (x, 3e-3, 3e-3)), material="debye")
        sim.add(Box((x, 2e-3, 1e-3), (x + 1e-3, 3e-3, 3e-3)), material="lorentz")
        sim.add_port(positions[-1], "ez", impedance=50.0,
                     waveform=lambda t: 0.01 * jnp.cos(t * 2e10))
    return sim


def _run(boundary, n_devices, model="composed"):
    devices = jax.devices("cpu")[:n_devices]
    assert len(devices) == n_devices
    sim = _build_model(boundary, n_devices, model)
    result = sim.run(n_steps=_STEPS, devices=devices)
    assert result.time_series.shape == (_STEPS, len(sim._probes))
    assert result.time_series.dtype == jnp.float32
    assert result.time_series.sharding.is_fully_replicated
    return {"trace": np.asarray(result.time_series),
            **{c: np.asarray(getattr(result.state, c)) for c in _FIELDS}}


# Wrong halo data moves a seam-adjacent value by a sizeable fraction of the
# field within a step or two; float32 rounding of the same arithmetic moves it
# by a few ULPs. The minimal exchange is bit-identical to the full one on JAX
# 0.10.2 with default flags, and on 0.4.33 with fusion disabled; with fusion on,
# 0.4.33 compiles the shorter program differently and a model with every
# feature moves by 1-2 ULP. The gate allows rounding and nothing larger.
_ROUNDING_ULPS = 16


def _assert_same_fields(actual, expected):
    for name in actual:
        a, b = actual[name], expected[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert np.isfinite(a).all() and np.isfinite(b).all(), name
        tol = _ROUNDING_ULPS * float(np.finfo(np.float32).eps) * float(np.max(np.abs(b)))
        err = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        assert err <= tol, (name, err, tol)
    assert np.any(expected["trace"]), "zero traces cannot witness seam coupling"
    for name in _FIELDS:
        assert np.any(expected[name]), f"zero {name} cannot witness coupling"


def _poison(st, mesh, n_devices, *, live=False):
    @partial(shard_map, mesh=mesh, in_specs=(P("x"),) * 6,
             out_specs=(P("x"),) * 6, check_rep=False)
    def poison(*fields):
        rank = lax.axis_index("x")
        result = []
        for name, field in zip(_FIELDS, fields):
            if name in ("ex", "hx", "ey", "ez") or (live and name == "hz"):
                field = field.at[0].set(jnp.where(rank > 0, jnp.nan, field[0]))
            if name in ("ex", "hx", "hy", "hz"):
                field = field.at[-1].set(jnp.where(rank < n_devices - 1, jnp.nan, field[-1]))
            result.append(field)
        return tuple(result)
    fields = poison(*(getattr(st, c) for c in _FIELDS))
    return st._replace(**dict(zip(_FIELDS, fields)))


def _install_poison(patch, *, live=False):
    for name in ("_exchange_h_ghosts_shmap", "_exchange_e_ghosts_shmap"):
        original = getattr(runner, name)

        def poisoned(st, mesh, n_devices, original=original):
            return _poison(original(st, mesh, n_devices), mesh, n_devices, live=live)

        patch.setattr(runner, name, poisoned)


def _bit_case(patch, boundary, n_devices, model):
    actual = _run(boundary, n_devices, model)
    with _reference(patch):
        expected = _run(boundary, n_devices, model)
    _assert_same_fields(actual, expected)


def _poison_case(patch, boundary, n_devices):
    expected = _run(boundary, n_devices)
    with patch.context() as p:
        _install_poison(p)
        actual = _run(boundary, n_devices)
    _assert_same_fields(actual, expected)


def _subprocess_case(kind, boundary, n_devices, model="composed"):
    env = dict(os.environ, XLA_FLAGS=f"--xla_force_host_platform_device_count={n_devices}",
               JAX_PLATFORMS="cpu")
    # Reuse the parent's font cache instead of rebuilding it in every child
    # when the user's default matplotlib directory is read-only.
    env.setdefault("MPLCONFIGDIR", matplotlib.get_configdir())
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), kind,
                           boundary, str(n_devices), model], env=env,
                          capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
@pytest.mark.parametrize("n_devices", [2, 3, 4])
@pytest.mark.parametrize("model", ["plain", "composed"])
def test_matches_frozen_full_exchange(monkeypatch, boundary, n_devices, model):
    if n_devices > 2:
        _subprocess_case("bits", boundary, n_devices, model)
    else:
        _bit_case(monkeypatch, boundary, n_devices, model)


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
@pytest.mark.parametrize("n_devices", [2, 3, 4])
def test_dead_ghosts(monkeypatch, boundary, n_devices):
    if n_devices > 2:
        _subprocess_case("poison", boundary, n_devices)
    else:
        _poison_case(monkeypatch, boundary, n_devices)


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_live_ghost_poison_is_detected(monkeypatch, boundary):
    expected = _run(boundary, 2)
    _install_poison(monkeypatch, live=True)
    with pytest.raises(AssertionError):
        _assert_same_fields(_run(boundary, 2), expected)


def _loop_collectives(hlo):
    """Parse actual while bodies and their callees, not metadata substrings."""
    computations = {}
    current = None
    for line in hlo.splitlines():
        match = re.match(r"^(?:ENTRY )?%([\w.-]+).*\{$", line)
        if match:
            current = match[1]
            computations[current] = []
        elif line == "}":
            current = None
        elif current is not None:
            computations[current].append(line)
    bodies = re.findall(r"\bwhile\([^\n]*\bbody=%([\w.-]+)", hlo)
    assert bodies, "no while body parsed; collective gate would be vacuous"
    visited = set()

    def walk(name):
        assert name in computations, f"missing computation {name}"
        if name in visited:
            return []
        visited.add(name)
        lines = computations[name]
        nested = {n for line in lines for n in re.findall(r"%([\w.-]+)", line)
                  if n in computations}
        return lines + [line for n in nested for line in walk(n)]

    lines = [line for body in bodies for line in walk(body)]
    assert any("ROOT " in line for line in lines), "incomplete while computation"
    result = {}
    for op in ("collective-permute", "all-reduce"):
        matches = [line.strip() for line in lines
                   if re.search(rf"\s{op}(?:-start|-done)?\(", line)]
        # Async start/done is one logical collective; require balanced pairs.
        starts = sum(f" {op}-start(" in line for line in matches)
        dones = sum(f" {op}-done(" in line for line in matches)
        assert starts == dones, (op, matches)
        result[op] = (len(matches) - dones, matches)
    return result


def _compiled_hlo(patch):
    captured = []

    def traced_jit(f, *args, **kwargs):
        entry = jax.jit(f, *args, **kwargs)

        def run(*a, **k):
            compiled = entry.lower(*a, **k).compile()
            captured.append(compiled.as_text())
            result = compiled(*a, **k)
            assert result[1].sharding.is_fully_replicated
            return result
        return run

    with patch.context() as p:
        p.setattr(runner, "jax", SimpleNamespace(**{**vars(jax), "jit": traced_jit}))
        _run("cpml", 2, "plain")
    assert len(captured) == 1
    return captured[0]


def test_time_loop_collectives(monkeypatch):
    counts = _loop_collectives(_compiled_hlo(monkeypatch))
    _assert_loop_counts(counts)


def _assert_loop_counts(counts):
    assert counts["collective-permute"][0] == 2, counts
    assert counts["all-reduce"][0] == 0, counts


@pytest.mark.parametrize("mutation", ["full_exchange", "per_step_psum"])
def test_collective_gate_rejects_mutations(monkeypatch, mutation):
    with _reference(monkeypatch, exchange=mutation == "full_exchange",
                    probes=mutation == "per_step_psum"):
        counts = _loop_collectives(_compiled_hlo(monkeypatch))
    expected = (12, 0) if mutation == "full_exchange" else (2, 1)
    assert (counts["collective-permute"][0], counts["all-reduce"][0]) == expected
    with pytest.raises(AssertionError):
        _assert_loop_counts(counts)


def test_bit_gate_rejects_missing_ez(monkeypatch):
    original = runner._exchange_e_ghosts_shmap

    def missing_ez(st, mesh, n_devices):
        return original(st, mesh, n_devices)._replace(ez=st.ez)

    monkeypatch.setattr(runner, "_exchange_e_ghosts_shmap", missing_ez)
    with pytest.raises(AssertionError, match="trace"):
        _bit_case(monkeypatch, "pec", 2, "composed")


@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_empty_probes_remain_replicated(boundary):
    sim = _build_model(boundary, 2, "plain")
    sim._probes.clear()
    result = sim.run(n_steps=2, devices=jax.devices("cpu")[:2])
    assert result.time_series.shape == (2, 0)
    assert result.time_series.dtype == jnp.float32
    assert result.time_series.sharding.is_fully_replicated


@pytest.mark.parametrize("n_devices", [1, 2])
def test_exchange_preserves_physical_ghosts_and_packs_both_fields(n_devices):
    devices = jax.devices("cpu")[:n_devices]
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    original = {c: (np.arange(n_devices * 5 * 3 * 2, dtype=np.float32).reshape(
        n_devices, 5, 3, 2) + i * 1000) for i, c in enumerate(_FIELDS)}
    st = FDTDState(**{c: jax.device_put(a.reshape(-1, 3, 2), sharding)
                      for c, a in original.items()}, step=jnp.int32(0))
    for exchange, components, dst, src, ranks in (
        (common.exchange_h_yee_shmap, ("hy", "hz"), 0, -2, range(1, n_devices)),
        (common.exchange_e_yee_shmap, ("ey", "ez"), -1, 1, range(n_devices - 1)),
    ):
        result = jax.jit(lambda s: exchange(s, mesh, n_devices))(st)
        for c in _FIELDS:
            expected = original[c].copy()
            if c in components:
                for rank in ranks:
                    owner = rank - 1 if dst == 0 else rank + 1
                    expected[rank, dst] = original[c][owner, src]
            assert np.asarray(getattr(result, c)).tobytes() == expected.tobytes()


if __name__ == "__main__":
    kind, boundary, count, model = sys.argv[1:]
    with pytest.MonkeyPatch.context() as patch:
        if kind == "bits":
            _bit_case(patch, boundary, int(count), model)
        else:
            assert kind == "poison"
            _poison_case(patch, boundary, int(count))
    print(f"{kind} {boundary} {count} {model}: matches")
