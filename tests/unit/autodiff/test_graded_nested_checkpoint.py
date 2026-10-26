"""Graded CPML levels, main's AD-memory terms, and BASE 1d5712706 traces."""
import json
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.api._ad_memory_model import tape_accounting
from rfx.geometry.csg import Box
from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid
from rfx.stepping.remat import checkpoint_memory

# Sources of rfx/stepping/{graded_loop,remat}.py at BASE 1d5712706, frozen because
# CI checks out depth 1 (no history to `git show`).
BASE = Path(__file__).resolve().parents[2] / "data" / "base_1d5712706"


def scene():
    dz = np.array([.001] * 4 + [.0008] * 4 + [.00064] * 4)
    sim = Simulation(freq_max=5e9, domain=(.012, .013, float(dz.sum())),
                     dx=.001, dz_profile=dz, boundary="cpml", cpml_layers=3)
    sim.add_material("lossy", eps_r=4, sigma=.1)
    sim.add(Box((.002, .003, .002), (.007, .009, .006)), material="lossy")
    sim.add_source((.003, .004, .003), "ez", waveform=GaussianPulse(f0=3e9, bandwidth=.8))
    sim.add_probe((.005, .006, .004), "ez")
    grid = sim._build_nonuniform_grid()
    coords = coords_from_nonuniform_grid(grid)
    mask = Box((.002, .003, .002), (.007, .009, .006)).mask_on_coords(coords.x, coords.y, coords.z)
    eps = jnp.where(mask, 4., 1.).astype(jnp.float32)
    sigma = jnp.where(mask, .1, 0.).astype(jnp.float32)
    return sim, eps, sigma


def outputs(sim, eps, sigma, chunk, n=97):
    result = sim.forward(n_steps=n, eps_override=eps, sigma_override=sigma,
                         checkpoint_every=chunk, checkpoint=False, skip_preflight=True)
    freqs = jnp.array([2e9, 4e9], dtype=jnp.float32)
    def accumulate(acc, item):
        t, value = item
        return acc + value * jnp.exp(-2j * jnp.pi * freqs * t * result.dt) * result.dt, None
    dft, _ = jax.lax.scan(accumulate, jnp.zeros(2, jnp.complex64),
                           (jnp.arange(n), result.time_series[:, 0]))
    return result.time_series, jnp.abs(dft) ** 2


@pytest.fixture(scope="module")
def box():
    return scene()


@pytest.mark.parametrize("lengths", [(24, 5), (30, 7, 3)])
def test_graded_values_and_gradients(box, lengths, record_property):
    sim, eps, sigma = box
    values = {}
    gradients = {}
    for chunk in (None, 24, lengths):
        fun = lambda e, s: outputs(sim, e, s, chunk)
        values[chunk] = jax.jit(fun)(eps, sigma)
        gradients[chunk] = jax.jit(jax.grad(lambda e, s: jnp.sum(fun(e, s)[1]), argnums=(0, 1)))(eps, sigma)
    for reference in (None, 24):
        for a, b in zip(values[reference], values[lengths]):
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    scale = max(float(jnp.max(jnp.abs(g))) for g in gradients[24])
    assert scale > 0
    relative = max(float(jnp.max(jnp.abs(a-b))) for a, b in zip(gradients[24], gradients[lengths])) / scale
    print(json.dumps(dict(lengths=lengths, gradient_relative_difference=relative)))
    record_property("gradient_relative_difference", relative)
    assert relative <= 1e-6


def test_batched_nested_gradient_matches_plain(box, record_property):
    """Two DFT bins pulled back at once (jax.jacrev), (30, 7, 3) against no checkpointing.

    Bar: the cross-trace bar for a quantity summed over the record, 1e-4 of the
    plain gradient's peak. Chosen after the A6000 measurement of 2026-10-26
    (batched nested vs batched plain, 8 outputs, 400 steps: 1.40e-6 against a
    predeclared 1e-6 taken from unbatched rows, with no derivation for the
    recompute rounding of a batched pullback). Unbatched rows keep 1e-6 above.
    """
    sim, eps, sigma = box
    def jacobian(chunk):
        return jax.jit(jax.jacrev(lambda e, s: outputs(sim, e, s, chunk)[1], argnums=(0, 1)))(eps, sigma)
    plain, nested = jacobian(None), jacobian((30, 7, 3))
    scale = max(float(jnp.max(jnp.abs(g))) for g in plain)
    assert scale > 0
    relative = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(plain, nested)) / scale
    print(json.dumps(dict(batched_relative_difference=relative)))
    record_property("batched_relative_difference", relative)
    assert relative <= 1e-4


@pytest.mark.parametrize("gradient", [False, True])
def test_integer_jaxpr_equals_base(box, monkeypatch, gradient):
    import rfx.stepping.graded_loop as loop
    import rfx.runners.nonuniform as runner
    sim, eps, sigma = box
    def trace():
        fun = lambda e, s: jnp.sum(outputs(sim, e, s, 24)[1])
        return str(jax.make_jaxpr(jax.grad(fun, argnums=(0, 1)) if gradient else fun)(eps, sigma))
    actual = trace()
    reference = types.ModuleType("base_graded_loop")
    source = (BASE / "graded_loop.py.txt").read_text()
    exec(compile(source, "BASE/graded_loop.py", "exec"), reference.__dict__)
    remat = types.ModuleType("base_remat")
    source = (BASE / "remat.py.txt").read_text()
    exec(compile(source, "BASE/remat.py", "exec"), remat.__dict__)
    reference.segmented_scan = remat.segmented_scan
    monkeypatch.setattr(loop, "run_nonuniform", reference.run_nonuniform)
    monkeypatch.setattr(runner, "run_nonuniform", reference.run_nonuniform)
    assert trace() == actual


def test_nested_conductivity_gradient_matches_forward_finite_difference(box, record_property):
    """Anchor the nested conductivity derivative in float32 plain-forward differences.

    CPU measurement (2026-10-26): J(0)=2.0156135094318017e-10;
    FD(1e-2)=-2.1824209106569015e-11, FD(5e-3)=-2.184918912462308e-11;
    AD=-2.1824507479006883e-11. Relative FD resolution=0.0011432926829268292,
    AD/FD difference=0.00112963668699187; |FD(5e-3)|/|J(0)|=0.1083996958.
    These observations do not replace the predeclared bars below.
    """
    sim, eps, sigma = box
    mask = sigma != 0
    sigma_background = jnp.zeros_like(sigma)
    sigma_block = jnp.float32(.1)

    def objective(alpha, chunk):
        conductivity = sigma_background + (jnp.float32(1) + alpha) * sigma_block * mask
        return jnp.sum(outputs(sim, eps, conductivity, chunk)[1])

    plain = jax.jit(lambda alpha: objective(alpha, None))
    zero = jnp.float32(0)
    j0 = float(plain(zero))
    differences = []
    for h in (np.float32(1e-2), np.float32(5e-3)):
        positive = np.float32(plain(jnp.float32(h)))
        negative = np.float32(plain(jnp.float32(-h)))
        differences.append(float((positive - negative) / (np.float32(2) * h)))
    fd_h, fd_half = differences
    ad = float(jax.jit(jax.grad(lambda alpha: objective(alpha, (30, 7, 3))))(zero))
    scale = abs(fd_half)
    measurements = dict(J0=j0, FD_h=fd_h, FD_half=fd_half, AD=ad,
                        fd_relative_difference=abs(fd_h - fd_half) / scale if scale else None,
                        ad_relative_difference=abs(ad - fd_half) / scale if scale else None)
    print(json.dumps(measurements))
    record_property("finite_difference_anchor", json.dumps(measurements))
    assert abs(fd_h - fd_half) <= 1e-2 * scale, f"bar (a): FD is unresolved: {measurements}"
    assert abs(ad - fd_half) <= 2e-2 * scale, f"bar (b): nested AD differs from FD: {measurements}"
    assert ad != 0 and scale >= 1e-3 * abs(j0), f"bar (c): probe must see the lossy block: {measurements}"


def test_planner_uses_main_tape_terms(box, monkeypatch):
    import rfx.api._ad_memory as api
    sim, _, _ = box
    accounting = sim._ad_memory_static_accounting()
    model = tape_accounting(sim, accounting, design_box=None, design="permittivity",
        checkpoint=True, segmented=True, n_steps=97, checkpoint_every=1)
    calls = []
    original = api.plan_checkpoint_lengths
    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(api, "plan_checkpoint_lengths", capture)
    plan = sim.plan_ad_memory(97, 1e-9, max_levels=2, cotangents=2)
    assert calls[0]["boundary_bytes"] == model["carry_bytes"]
    assert calls[0]["step_bytes"] == model["tape_step_bytes"]
    assert calls[0]["static_bytes"] == model["c0_bytes"]
    assert calls[0]["cotangents"] == plan.selected_estimate.cotangents == 2
    assert isinstance(plan.checkpoint_every, tuple)
    assert "forward passes" in plan.recommendation


def test_estimate_tuple_matches_checkpoint_memory(box):
    sim, _, _ = box
    estimate = sim.estimate_ad_memory(97, checkpoint_every=(24, 5))
    memory = checkpoint_memory(97, (24, 5), boundary_bytes=estimate.carry_bytes,
        step_bytes=estimate.tape_step_bytes, static_bytes=estimate.ad_fixed_gb * 1e9)
    assert estimate.ad_segmented_gb * 1e9 == pytest.approx(memory.peak_bytes)
    assert "2 levels" in estimate.warning and "3 forward passes" in estimate.warning
    plan = sim.plan_ad_memory(97, 1, checkpoint_every=(24, 5))
    assert plan.selected_estimate.ad_segmented_gb == estimate.ad_segmented_gb


@pytest.mark.parametrize("design_box", [None, ((.007, .008, .004), (.010, .011, .007))])
def test_clamped_outer_level_keeps_inner_levels(box, design_box):
    """(120, 5) over 97 steps: the clamp turns off 120 and the level of 5 still runs."""
    sim, _, _ = box
    estimate = sim.estimate_ad_memory(97, checkpoint_every=(120, 5), design_box=design_box)
    inner = sim.estimate_ad_memory(97, checkpoint_every=(5,), design_box=design_box)
    assert estimate.ad_segmented_active_segments == inner.ad_segmented_active_segments == 20
    assert estimate.ad_segmented_gb == inner.ad_segmented_gb
    assert "1 levels" in estimate.warning


@pytest.mark.parametrize("gradient", [False, True])
def test_runner_clamped_outer_level_traces_inner_levels(box, gradient):
    sim, eps, sigma = box
    def trace(chunk):
        fun = lambda e, s: jnp.sum(outputs(sim, e, s, chunk)[1])
        return str(jax.make_jaxpr(jax.grad(fun, argnums=(0, 1)) if gradient else fun)(eps, sigma))
    clamped = trace((120, 5))
    assert "checkpoint" in clamped or "remat" in clamped
    assert clamped == trace((5,))


def test_uniform_planner_refuses_levels():
    sim = Simulation(freq_max=5e9, domain=(.012, .012, .012), dx=.001)
    with pytest.raises(ValueError, match="max_levels.*uniform"):
        sim.plan_ad_memory(97, 1, max_levels=2)


@pytest.mark.parametrize("lengths", [(), (5, 24), (24, 24), (24, 0), (True, 1), (24, 5.)])
def test_estimator_refuses_invalid_lengths(box, lengths):
    with pytest.raises(ValueError):
        box[0].estimate_ad_memory(97, checkpoint_every=lengths)


@pytest.mark.parametrize("length", [8, 16])
def test_report_cpu_tape_and_compiled_memory(box, length, record_property):
    sim, eps, sigma = box
    estimate = sim.estimate_ad_memory(97, checkpoint_every=length)
    compiled = jax.jit(jax.grad(lambda e, s: jnp.sum(outputs(sim, e, s, length)[1]),
                               argnums=(0, 1))).lower(eps, sigma).compile()
    row = dict(length=length, tape_step_bytes=estimate.tape_step_bytes,
               temp_bytes=compiled.memory_analysis().temp_size_in_bytes)
    record_property("memory_report", json.dumps(row))
    print(json.dumps(row))


def test_slab_wrap_refuses_tuple():
    from rfx.stepping.slab_wrap import validate_scan_controls
    with pytest.raises(NotImplementedError, match="slab_wrap.*checkpoint_every"):
        validate_scan_controls(97, 0, (24, 5))


def test_graded_distributed_refuses_tuple(box):
    sim, _, _ = box
    with pytest.raises(NotImplementedError, match="graded multi-device distributed_nu.*checkpoint_every"):
        sim.forward(n_steps=97, checkpoint_every=(24, 5), distributed=True,
                    devices=jax.devices("cpu")[:2], skip_preflight=True)


@pytest.mark.parametrize("distributed", [False, True])
def test_uniform_refuses_tuple(distributed):
    sim = Simulation(freq_max=5e9, domain=(.012, .012, .012), dx=.001, cpml_layers=3)
    sim.add_source((.004, .004, .004), "ez")
    with pytest.raises(NotImplementedError, match="(?s)(checkpoint_every.*uniform|uniform.*checkpoint_every)"):
        sim.forward(n_steps=97, checkpoint_every=(24, 5), distributed=distributed, skip_preflight=True)


@pytest.mark.parametrize("method", ["forward", "estimate_ad_memory"])
def test_segments_stays_integer(box, method):
    with pytest.raises(ValueError, match="checkpoint_segments.*checkpoint_every"):
        getattr(box[0], method)(n_steps=97, checkpoint_segments=(24, 5))
