"""Uniform run NTFF parity, ownership and seam falsifiers (#1402)."""
import os
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.antenna import antenna_gain
from rfx.farfield import compute_far_field
from rfx.geometry import Box

# Per-array peak-relative bar. PEC compensation is compared to its own peak.
# CPML fields already differ from native on main (trace 9.71e-7); residual
# bits therefore differ. There we bound compensation error by two float32
# epsilons of its sum, while reporting its own-peak relative error too.
BAR = 5e-5


def _model(case, n_devices=2):
    length_x = 12e-3 if case == "x_hi_padded" else 11e-3
    sim = Simulation(freq_max=15e9, domain=(length_x, 9e-3, 7e-3),
                     dx=1e-3, boundary="cpml" if case == "cpml" else "pec",
                     cpml_layers=2)
    sim.add_source((5e-3, 5e-3, 4e-3), "ez")
    sim.add_source((7e-3, 4e-3, 3e-3), "ey")
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        sim.add_probe((5e-3, 4e-3, 3e-3), component)
    if case == "dielectric":
        sim.add_material("block", eps_r=3)
        sim.add(Box((4e-3, 4e-3, 3e-3), (7e-3, 6e-3, 5e-3)), material="block")
    lo = (3e-3, 3e-3, 2e-3)
    if case == "x_seam":
        grid = sim._build_grid()
        lo = (((grid.shape[0] + n_devices - 1) // n_devices - grid.pad_x_lo) * grid.dx,
              3e-3, 2e-3)
    hi = (9e-3, 7e-3, 6e-3)
    if case == "x_hi_padded":
        grid = sim._build_grid()
        assert grid.shape[0] % n_devices != 0
        seam = (grid.shape[0] + n_devices - 1) // n_devices
        hi = ((seam - grid.pad_x_lo) * grid.dx, 7e-3, 6e-3)
    sim.add_ntff_box(lo, hi, freqs=[5e9, 10e9, 15e9])
    return sim


def _error(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    delta = np.max(np.abs(a.astype(np.complex128) - b.astype(np.complex128)), initial=0)
    peak = np.max(np.abs(b), initial=0)
    return float(delta / peak if peak else delta)


def _compare(actual, reference, case):
    errors = {}
    for name in reference.ntff_data._fields:
        errors[name] = _error(getattr(actual.ntff_data, name), getattr(reference.ntff_data, name))
    assert any(np.any(a) for a in reference.ntff_data[6:]), "compensation witness"
    for a, b in zip(actual.ntff_box, reference.ntff_box):
        np.testing.assert_array_equal(a, b)
    angles = (np.linspace(0, np.pi, 19), np.linspace(0, 2 * np.pi, 37))
    ff = [compute_far_field(r.ntff_data, r.ntff_box, r.grid, *angles)
          for r in (actual, reference)]
    for name in ("E_theta", "E_phi"):
        errors[name] = _error(getattr(ff[0], name), getattr(ff[1], name))
    errors["gain"] = _error(antenna_gain(ff[0]), antenna_gain(ff[1]))
    errors["trace"] = _error(actual.time_series, reference.time_series)
    print(f"case={case} errors={errors} bar={BAR}")
    judged = dict(errors)
    if case in ("cpml", "patch"):
        scaled = {}
        for name in reference.ntff_data._fields[6:]:
            delta = np.max(np.abs(np.asarray(getattr(actual.ntff_data, name))
                                 - np.asarray(getattr(reference.ntff_data, name))))
            peak = np.max(np.abs(np.asarray(getattr(reference.ntff_data, name[2:]))))
            scaled[name] = float(delta / peak if peak else delta)
            assert scaled[name] <= 2 * np.finfo(np.float32).eps, f"NTFF compensation {scaled}"
            del judged[name]
        print(f"case={case} compensation_error_over_sum={scaled}")
    assert max(judged.values()) <= BAR, f"NTFF parity {errors}"
    return errors


def _parity(case, n_devices=2):
    sim = _model(case, n_devices)
    kwargs = dict(n_steps=160, skip_preflight=True)
    reference = sim.run(**kwargs)
    actual = sim.run(devices=jax.devices("cpu")[:n_devices], **kwargs)
    if case == "x_hi_padded":
        assert actual.ntff_box.i_hi == (actual.grid.shape[0] + n_devices - 1) // n_devices
    print(f"devices={n_devices} case={case}")
    errors = _compare(actual, reference, case)
    if case == "x_hi_padded":
        assert all(error == 0 for error in errors.values())
    return actual


@pytest.mark.parametrize("case", ["spanning", "x_seam", "x_hi_padded", "dielectric", "cpml"])
def test_parity(case):
    _parity(case)


def _ghost_hx_invariant(monkeypatch):
    from rfx.runners import _distributed_ntff as owner
    sim = _model("cpml")
    grid = sim._build_grid()
    seam = ((grid.shape[0] + 1) // 2 - grid.pad_x_lo) * grid.dx
    sim.add_material("magnetic", mu_r=3, sigma=0.3)
    sim.add(Box((seam - 2e-3, 3e-3, 2e-3),
                (seam + 2e-3, 7e-3, 6e-3)), material="magnetic")
    sim.add_source((seam, 5e-3, 4e-3), "ez")
    samples = []
    original = owner.SlabNTFF.update

    def observe(self, buffer, state, dt, step, *, ranks=None):
        width = state.hx.shape[0] // self.n_devices
        # All internal right ghosts and their owners, at the real sampler.
        jax.debug.callback(lambda ghost, real: samples.append(
            (np.array(ghost), np.array(real))),
            state.hx[width - 1:-1:width], state.hx[width + 1::width])
        return original(self, buffer, state, dt, step, ranks=ranks)

    monkeypatch.setattr(owner.SlabNTFF, "update", observe)
    result = sim.run(n_steps=160, devices=jax.devices("cpu")[:2], skip_preflight=True)
    jax.block_until_ready(result.ntff_data)
    jax.effects_barrier()
    assert len(samples) == 160
    peak = max(float(np.max(np.abs(real))) for _, real in samples)
    assert peak > 1e-6, "seam Hx must carry energy"
    for ghost, real in samples:
        np.testing.assert_array_equal(ghost, real, err_msg="right-ghost Hx invariant")
    print(f"right-ghost Hx invariant: 160 steps, exact equality, owner peak={peak}")


def test_right_ghost_hx_invariant(monkeypatch):
    _ghost_hx_invariant(monkeypatch)


def test_mutation_h_update_skips_ghost_rows(monkeypatch):
    from rfx.runners import distributed_v2 as runner
    original = runner._update_h_local

    def skip_ghosts(state, *args):
        updated = original(state, *args)
        return updated._replace(**{
            name: getattr(updated, name).at[0].set(getattr(state, name)[0])
                  .at[-1].set(getattr(state, name)[-1])
            for name in ("hx", "hy", "hz")})

    monkeypatch.setattr(runner, "_update_h_local", skip_ghosts)
    with pytest.raises(AssertionError, match="right-ghost Hx invariant"):
        _ghost_hx_invariant(monkeypatch)
    print("mutation=H update skips ghost rows: RED")


def _patch_parity(n_devices=2):
    sim = Simulation(freq_max=10e9, domain=(16e-3, 12e-3, 8e-3),
                     dx=1e-3, boundary="cpml", cpml_layers=2)
    sim.add_material("substrate", eps_r=2.2)
    sim.add(Box((4e-3, 3e-3, 2e-3), (12e-3, 9e-3, 4e-3)), material="substrate")
    sim.add(Box((4e-3, 3e-3, 2e-3), (12e-3, 9e-3, 3e-3)), material="pec")
    sim.add(Box((6e-3, 4e-3, 4e-3), (10e-3, 8e-3, 5e-3)), material="pec")
    sim.add_port((8e-3, 6e-3, 3e-3), "ez", impedance=50)
    sim.add_probe((8e-3, 6e-3, 3e-3), "ez")
    sim.add_ntff_box((2e-3, 2e-3, 1e-3), (14e-3, 10e-3, 7e-3), freqs=[3e9, 5e9, 7e9])
    kwargs = dict(n_steps=160, skip_preflight=True, compute_s_params=True,
                  s_param_freqs=np.array([3e9, 5e9, 7e9]), s_param_n_steps=320)
    reference = sim.run(**kwargs)
    actual = sim.run(devices=jax.devices("cpu")[:n_devices], **kwargs)
    print(f"devices={n_devices} case=patch")
    _compare(actual, reference, "patch")
    error = _error(actual.s_params, reference.s_params)
    print(f"patch S relative error={error}")
    assert error <= BAR


def test_lumped_patch_s_and_ntff():
    _patch_parity()


@pytest.mark.parametrize("mutation", ["skip", "duplicate", "compensation"])
def test_mutations(monkeypatch, mutation):
    from rfx.runners import _distributed_ntff as owner
    if mutation == "skip":
        monkeypatch.setattr(owner.SlabNTFF, "update", lambda self, buffer, *args, **kwargs: buffer)
    elif mutation == "duplicate":
        # The wrong rank writes the x face; y/z ownership uses clipped ranges.
        import inspect
        import textwrap
        source = textwrap.dedent(inspect.getsource(owner.SlabNTFF.__init__))
        source = source.replace("index // nx_per == rank", "True")
        namespace = dict(vars(owner))
        exec(compile(source, "<drop-owner-filter>", "exec"), namespace)
        monkeypatch.setattr(owner.SlabNTFF, "__init__", namespace["__init__"])
    else:
        original = owner.SlabNTFF.assemble
        def drop(self, buffer):
            data = original(self, buffer)
            return data._replace(**{name: np.zeros_like(getattr(data, name))
                                    for name in data._fields[6:]})
        monkeypatch.setattr(owner.SlabNTFF, "assemble", drop)
    with pytest.raises(AssertionError, match="NTFF parity"):
        _parity("spanning")
    print(f"mutation={mutation}: RED")


@pytest.mark.skipif(os.environ.get("RFX_LOCAL_DISTRIBUTED") != "1",
                    reason="3/4 CPU devices: opt-in subprocess")
@pytest.mark.parametrize("n_devices", [3, 4])
def test_more_devices(n_devices):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]),
               XLA_FLAGS="--xla_force_host_platform_device_count=4", JAX_PLATFORMS="cpu")
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(n_devices)],
                          env=env, capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    print(proc.stdout)


@pytest.mark.parametrize("kind", ["forward", "graded_forward", "graded_run", "graded_direct"])
def test_out_of_scope_refused(kind):
    sim = _model("spanning")
    if kind == "forward":
        with pytest.raises(NotImplementedError, match="only for non-uniform meshes"):
            sim.forward(n_steps=8, distributed=True, devices=jax.devices("cpu")[:2],
                        skip_preflight=True)
    else:
        sim._dz_profile = np.array([1e-3] * 3 + [0.5e-3] * 8)
        with pytest.raises(NotImplementedError, match="NTFF|ntff|graded"):
            if kind == "graded_forward":
                sim.forward(n_steps=8, distributed=True, devices=jax.devices("cpu")[:2],
                            skip_preflight=True)
            elif kind == "graded_run":
                sim.run(n_steps=8, devices=jax.devices("cpu")[:2], skip_preflight=True)
            else:
                from rfx.runners.distributed_v2 import run_distributed
                run_distributed(sim, n_steps=8, devices=jax.devices("cpu")[:2])


def test_no_box_does_not_allocate_ntff(monkeypatch):
    from rfx.runners import _distributed_ntff as owner
    def unexpected(*args):
        pytest.fail("no-box run allocated NTFF")
    monkeypatch.setattr(owner.SlabNTFF, "initial", unexpected)
    sim = _model("spanning")
    sim._ntff = None
    result = sim.run(n_steps=8, devices=jax.devices("cpu")[:2], skip_preflight=True)
    assert result.ntff_data is None and result.ntff_box is None


def test_compact_storage_and_unique_placement():
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from rfx.farfield import NTFFBox, init_ntff_data
    from rfx.runners._distributed_ntff import SlabNTFF
    box = NTFFBox(5, 75, 5, 75, 5, 25, jnp.array([1e9, 2e9, 3e9]),
                  face_centre=True)
    mesh = Mesh(np.array(jax.devices("cpu")[:2]), ("x",))
    layout = SlabNTFF(box, (80, 80, 30), 40, mesh, jnp.float32)
    buffer = layout.initial()
    actual_bytes = [shard.data.nbytes for shard in buffer.addressable_shards]
    full = init_ntff_data(box)
    assert actual_bytes == [1478400, 1478400]
    assert sum(a.nbytes for a in full) == 2956800
    # A different nonzero marker per cell makes transposition, duplication,
    # omission and compensation loss visible without time-stepping.
    expected = type(full)(*(jnp.arange(a.size, dtype=jnp.float32).reshape(a.shape)
                            + (index + 1) * 100000 for index, a in enumerate(full)))
    packed = []
    for lo, hi, faces, _, _, _ in layout.parts:
        arrays = []
        for index, a in enumerate(expected):
            face = index % 6
            arrays.append((a if faces[face] else a[:, :0]) if face < 2
                          else a[:, lo - box.i_lo:hi - box.i_lo])
        packed.append(jnp.concatenate([a.ravel() for a in arrays]))
    assembled = layout.assemble(jnp.stack(packed))
    for a, b in zip(assembled, expected):
        np.testing.assert_array_equal(a, b)
    print(f"NTFF storage per_device_bytes={actual_bytes} full_bytes=2956800")


if __name__ == "__main__":
    for case in ("spanning", "x_seam", "dielectric", "cpml"):
        _parity(case, int(sys.argv[1]))
    _patch_parity(int(sys.argv[1]))
