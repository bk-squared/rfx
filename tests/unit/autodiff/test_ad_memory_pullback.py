"""Independent one-step compiler reference and hand-counted checkpoint costs."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.stepping.remat import checkpoint_memory, plan_checkpoint_lengths


def scene(graded=True):
    profile = dict(dz_profile=np.array([.001] * 4 + [.0008] * 4 + [.00064] * 4)) if graded else {}
    sim = Simulation(freq_max=5e9, domain=(.0123, .0137, .0101), dx=.001,
                     boundary="cpml", cpml_layers=3, **profile)
    sim.add_material("lossy", eps_r=40, sigma=1.)
    sim.add(Box((.0023, .0031, .0022), (.0072, .0093, .0061)), material="lossy")
    sim.add_source((.0033, .0042, .0031), "ez", amplitude_kind="current",
                   waveform=GaussianPulse(f0=3e9, bandwidth=.8))
    sim.add_probe((.0054, .0062, .0043), "ez")
    return sim


def reference_step_memory(sim, n, **options):
    """Compile a public forward's actual innermost scan body, without AD-memory helpers.

    Coefficients, the complete carry and step operands are abstract arguments.
    This independent compiler reading is the anchor for the user-facing bytes;
    it makes no assertion about GPU full-record peak memory.
    """
    grid = sim._build_realized_grid()
    options = dict(options)
    design = options.pop("design", "permittivity")
    box = options.get("design_box")
    shape = grid.shape
    if box is not None:
        bounds = sim._design_box_bounds_from_corners(grid, box)
        shape = tuple(bounds[2*a+1] - bounds[2*a] for a in range(3))
    names = (("design_occupancy_override",) if box is not None and design == "occupancy"
             else ("pec_occupancy_override",) if design == "occupancy"
             else ("design_eps_override", "design_sigma_override") if box is not None
             else ("eps_override", "sigma_override"))
    def forward(*args):
        return sim.forward(n_steps=n, skip_preflight=True, **options,
                           **dict(zip(names, args))).time_series
    program = jax.make_jaxpr(forward)(*(jax.ShapeDtypeStruct(shape, jnp.float32) for _ in names))
    candidates = []

    def visit(jaxpr):
        jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
        for eq in jaxpr.eqns:
            if eq.primitive.name == "scan":
                start = eq.params["num_consts"]
                carry = eq.invars[start:start + eq.params["num_carry"]]
                if sum(v.aval.shape == grid.shape for v in carry) >= 6:
                    candidates.append((eq.params["jaxpr"].jaxpr, carry))
            for child in eq.params.values():
                if hasattr(child, "jaxpr") or hasattr(child, "eqns"):
                    visit(child)
    visit(program)
    body, carry = candidates[-1]
    nc = len(body.constvars)

    def one_step(*args):
        return jax.core.eval_jaxpr(body, args[:nc], *args[nc:])

    def pullback(primals, cotangents):
        _, backward = jax.vjp(one_step, *primals)
        return backward(cotangents)

    primals = tuple(jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype)
                    for v in (*body.constvars, *body.invars))
    cotangents = [jax.ShapeDtypeStruct(v.aval.shape,
        v.aval.dtype if np.issubdtype(v.aval.dtype, np.inexact) else jax.dtypes.float0)
        for v in body.outvars]
    memory = jax.jit(pullback).lower(primals, cotangents).compile().memory_analysis()
    carry_bytes = sum(math.prod(v.aval.shape) * v.aval.dtype.itemsize for v in carry)
    return memory, carry_bytes


def test_t3_hand_count_and_planner_cotangents():
    """T3 holds ceil(6007/1100)+ceil(1100/130)+ceil(130/14)=6+9+10 carries.

    Add 14 live steps: 292,900,000 + 25,521,500 * (6+9+10+14) = 1,288,238,500 B.
    The separate eight-step search has minima L=2: c+4b+2s=29; B=2 adds 11+3=14.
    """
    memory = checkpoint_memory(6007, (1100, 130, 14), boundary_bytes=25_521_500,
                               step_bytes=25_521_500, static_bytes=292_900_000, cotangents=1)
    assert memory.held_boundaries == 6 + 9 + 10
    assert memory.live_steps == 14
    assert memory.peak_bytes == 292_900_000 + 25_521_500 * (6 + 9 + 10) + 25_521_500 * 14
    plan = plan_checkpoint_lengths(8, boundary_bytes=3, step_bytes=5, static_bytes=7,
                                   working_bytes=11, cotangents=2, budget_bytes=42, max_levels=1)
    assert plan.lengths == (2,)
    assert plan.peak_bytes == 43
    assert not plan.fits


CASES = [
    (True, dict(checkpoint=False, checkpoint_every=(30, 7, 3))),
    (True, dict(checkpoint=True, checkpoint_every=24)),
    (True, dict(checkpoint=False)),
    (True, dict(checkpoint=True)),
    (False, dict(checkpoint=False, checkpoint_segments=3)),
    (False, dict(checkpoint=True)),
    (True, dict(checkpoint=False, checkpoint_every=(30, 7, 3),
                design_box=((.0081, .0092, .0063), (.0104, .0115, .0084)))),
    (False, dict(checkpoint=False, design="occupancy")),
    (False, dict(checkpoint=False, design="occupancy", checkpoint_segments=3,
                 design_box=((.0081, .0092, .0063), (.0104, .0115, .0084)))),
    (True, dict(checkpoint=False, design="occupancy", checkpoint_every=(30, 7, 3))),
    (False, dict(checkpoint=False, checkpoint_segments=3,
                 design_box=((.0081, .0092, .0063), (.0104, .0115, .0084)))),
    (True, dict(checkpoint=True,
                design_box=((.0081, .0092, .0063), (.0104, .0115, .0084)))),
]


@pytest.mark.parametrize("graded,options", CASES)
def test_compiled_step_is_independent_reference(graded, options):
    sim = scene(graded)
    # 99 permits the uniform three-segment path while retaining nested tails.
    reference, carry = reference_step_memory(sim, 99, **options)
    one = sim.estimate_ad_memory(99, cotangents=1, **options)
    two = sim.estimate_ad_memory(99, cotangents=2, **options)
    assert one.calibration == two.calibration == "compiled one-step pullback (estimate)"
    assert one.carry_bytes == carry
    assert one.ad_fixed_gb * 1e9 == pytest.approx(reference.argument_size_in_bytes + reference.temp_size_in_bytes)
    assert one.pullback_working_bytes == reference.temp_size_in_bytes
    assert two.cotangents == 2
    assert (two.ad_full_gb - one.ad_full_gb) * 1e9 == pytest.approx(reference.temp_size_in_bytes + carry)
    if one.ad_segmented_gb is not None:
        assert (two.ad_segmented_gb - one.ad_segmented_gb) * 1e9 == pytest.approx(reference.temp_size_in_bytes + carry)
    assert two.to_dict()["pullback_working_bytes"] == reference.temp_size_in_bytes
    assert two.to_dict()["cotangents"] == 2
    if graded and options == CASES[0][1]:
        plan = sim.plan_ad_memory(99, 1., checkpoint_every=(30, 7, 3), cotangents=2)
        assert plan.selected_estimate.cotangents == 2
        assert plan.selected_estimate.ad_segmented_gb == two.ad_segmented_gb
        explanation = sim.explain_ad_memory(99, cotangents=2, **options)
        component = next(c for c in explanation.components if c.name == "additional_cotangent_working_set")
        assert component.memory_gb * 1e9 == pytest.approx(reference.temp_size_in_bytes + carry)
        assert sum(c.memory_gb for c in explanation.components) == pytest.approx(two.ad_segmented_gb)


@pytest.mark.parametrize("window", [False, True])
def test_compile_failure_is_labelled_per_cell_fallback(monkeypatch, window):
    import rfx.api._ad_pullback as pullback
    def unavailable(_):
        raise RuntimeError("forced compiler failure")
    monkeypatch.setattr(pullback, "compile_pullback", unavailable)
    monkeypatch.setattr(pullback, "_COMPILED_STEP_BYTES", {})
    sim = scene()
    options = dict(checkpoint=False, checkpoint_every=(30, 7, 3))
    if window:
        options["design_box"] = ((.0081, .0092, .0063), (.0104, .0115, .0084))
    one = sim.estimate_ad_memory(99, cotangents=1, **options)
    two = sim.estimate_ad_memory(99, cotangents=2, **options)
    # Historical fallback calibration, independently counted on the realized grid.
    expected = (192 if window else 320) * math.prod(sim._build_realized_grid().shape)
    assert one.ad_fixed_gb * 1e9 == pytest.approx(expected)
    assert one.pullback_working_bytes == two.pullback_working_bytes == 0
    assert "per-cell fallback" in one.calibration
    assert "forced compiler failure" in one.warning
    assert "AD_MEMORY_C0_PER_CELL fallback" in one.warning
    assert "w=0" in one.warning
    assert (two.ad_segmented_gb - one.ad_segmented_gb) * 1e9 == pytest.approx(one.carry_bytes)
    assert (two.ad_full_gb - one.ad_full_gb) * 1e9 == pytest.approx(one.carry_bytes)


def test_equal_step_programs_compile_once(monkeypatch):
    """Lengths change the scan around the step, never the step: one compile serves them all."""
    import rfx.api._ad_pullback as pullback
    calls = []
    original = pullback.compile_pullback
    def counted(program):
        calls.append(program)
        return original(program)
    monkeypatch.setattr(pullback, "compile_pullback", counted)
    monkeypatch.setattr(pullback, "_COMPILED_STEP_BYTES", {})
    sim = scene()
    estimates = [sim.estimate_ad_memory(99, checkpoint=False, checkpoint_every=every)
                 for every in (5, 9, 12, (30, 7, 3))]
    assert len(calls) == 1
    reference, _ = reference_step_memory(sim, 99, checkpoint=False, checkpoint_every=9)
    for estimate in estimates:
        assert estimate.ad_fixed_gb * 1e9 == pytest.approx(
            reference.argument_size_in_bytes + reference.temp_size_in_bytes)
        assert estimate.pullback_working_bytes == reference.temp_size_in_bytes


@pytest.mark.parametrize("cotangents", [True, False, 0, -1, 1.5])
def test_cotangents_requires_positive_integer(cotangents):
    sim = scene()
    for call in (
        lambda: sim.estimate_ad_memory(99, cotangents=cotangents),
        lambda: sim.plan_ad_memory(99, 1., cotangents=cotangents),
        lambda: checkpoint_memory(99, (30, 7, 3), boundary_bytes=1, step_bytes=1,
                                   static_bytes=1, cotangents=cotangents),
        lambda: plan_checkpoint_lengths(99, boundary_bytes=1, step_bytes=1,
                                       static_bytes=1, budget_bytes=100, cotangents=cotangents),
    ):
        with pytest.raises(ValueError, match="cotangents must be a positive int"):
            call()
