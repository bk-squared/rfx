"""Hand-counted checkpoint costs and the cotangents= count of the AD-memory estimate."""
import math

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.stepping.remat import checkpoint_memory, plan_checkpoint_lengths


def scene(graded=True):
    profile = dict(dz_profile=np.array([.001] * 4 + [.0008] * 4 + [.00064] * 4)) if graded else {}
    z = float(np.sum(profile["dz_profile"])) if graded else .0101
    sim = Simulation(freq_max=5e9, domain=(.0123, .0137, z), dx=.001,
                     boundary="cpml", cpml_layers=3, **profile)
    sim.add_material("lossy", eps_r=40, sigma=1.)
    sim.add(Box((.0023, .0031, .0022), (.0072, .0093, .0061)), material="lossy")
    sim.add_source((.0033, .0042, .0031), "ez", amplitude_kind="current",
                   waveform=GaussianPulse(f0=3e9, bandwidth=.8))
    sim.add_probe((.0054, .0062, .0043), "ez")
    return sim


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


@pytest.mark.parametrize("graded,options", [
    (True, dict(checkpoint=False, checkpoint_every=(30, 7, 3))),
    (True, dict(checkpoint=True)),
    (False, dict(checkpoint=False, checkpoint_segments=3)),
    (True, dict(checkpoint=False, checkpoint_every=(30, 7, 3),
                design_box=((.0081, .0092, .0063), (.0104, .0115, .0084)))),
])
def test_extra_cotangent_adds_working_set_and_carry_on_the_calibrated_term(graded, options):
    """Per-cell c (320 B/cell whole grid, 192 B/cell box, GPU jax 0.10.2) is unchanged by B."""
    sim = scene(graded)
    one = sim.estimate_ad_memory(99, cotangents=1, **options)
    three = sim.estimate_ad_memory(99, cotangents=3, **options)
    cells = math.prod(sim._build_realized_grid().shape)
    assert one.ad_fixed_gb * 1e9 == pytest.approx((192 if "design_box" in options else 320) * cells)
    assert one.calibration == "calibrated (GPU, jax 0.10.2)"
    assert (one.cotangents, three.cotangents) == (1, 3)
    assert three.to_dict()["cotangents"] == 3
    assert (three.ad_full_gb - one.ad_full_gb) * 1e9 == pytest.approx(2 * 7.5 * one.carry_bytes)
    if one.ad_segmented_gb is not None:
        assert (three.ad_segmented_gb - one.ad_segmented_gb) * 1e9 == pytest.approx(2 * 7.5 * one.carry_bytes)
    assert one.working_bytes == 6.5 * one.carry_bytes
    costs = dict(boundary_bytes=one.carry_bytes, step_bytes=one.tape_step_bytes,
                 static_bytes=round(one.ad_fixed_gb * 1e9), working_bytes=one.working_bytes)
    base = checkpoint_memory(99, (30, 7, 3), cotangents=1, **costs)
    batch = checkpoint_memory(99, (30, 7, 3), cotangents=3, **costs)
    assert batch.peak_bytes - base.peak_bytes == 2 * 7.5 * one.carry_bytes
    explanation = sim.explain_ad_memory(99, cotangents=3, **options)
    component = next(c for c in explanation.components if c.name == "additional_cotangent_working_set")
    assert component.memory_gb * 1e9 == pytest.approx(2 * 7.5 * one.carry_bytes)
    assert component.bytes_per_unit_gb * 1e9 == pytest.approx(7.5 * one.carry_bytes)


@pytest.mark.parametrize("max_levels", [None, 2])
def test_plan_and_explain_carry_the_cotangent_count(monkeypatch, max_levels):
    """A budget nothing fits forces the candidate search; its pick must equal a direct B=3 estimate."""
    import rfx.api._ad_memory as api
    sim = scene()
    calls = []
    original = api.plan_checkpoint_lengths
    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(api, "plan_checkpoint_lengths", capture)
    plan = sim.plan_ad_memory(99, 1e-9, max_levels=max_levels, cotangents=3)
    selected = plan.selected_estimate
    assert selected.cotangents == 3 and not plan.segmented_fits
    direct = sim.estimate_ad_memory(99, checkpoint=True, cotangents=3,
                                    checkpoint_every=selected.checkpoint_every)
    single = sim.estimate_ad_memory(99, checkpoint=True, cotangents=1,
                                    checkpoint_every=selected.checkpoint_every)
    assert selected.ad_segmented_gb == pytest.approx(direct.ad_segmented_gb)
    assert (direct.ad_segmented_gb - single.ad_segmented_gb) * 1e9 == pytest.approx(2 * 7.5 * single.carry_bytes)
    if max_levels is not None:
        assert calls and calls[0]["cotangents"] == 3
    one = sim.explain_ad_memory(99, checkpoint_every=(30, 7, 3), cotangents=1)
    three = sim.explain_ad_memory(99, checkpoint_every=(30, 7, 3), cotangents=3)
    assert (three.selected_memory_gb - one.selected_memory_gb) * 1e9 == pytest.approx(2 * 7.5 * single.carry_bytes)


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


def test_planner_peak_includes_batched_working_arrays(monkeypatch):
    """The max_levels planner must price eight pullbacks before choosing a plan."""
    import rfx.api._ad_memory as api
    plans = []
    original = api.plan_checkpoint_lengths

    def capture(*args, **kwargs):
        plan = original(*args, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(api, "plan_checkpoint_lengths", capture)
    selected = scene().plan_ad_memory(99, 1e-9, max_levels=2, cotangents=8).selected_estimate
    expected = checkpoint_memory(
        99, plans[0].lengths, boundary_bytes=selected.carry_bytes,
        step_bytes=selected.tape_step_bytes, static_bytes=round(selected.ad_fixed_gb * 1e9),
        working_bytes=6.5 * selected.carry_bytes, cotangents=8)
    assert plans[0].peak_bytes == expected.peak_bytes
    assert selected.ad_segmented_gb * 1e9 == pytest.approx(expected.peak_bytes)


def test_scene_s_measured_per_cotangent_anchor():
    """Frozen GPU anchor: rfx/records/20261027-compiled-gradient-estimate/, T1/T4.

    Public API scene S from the M2 harness; static accounting only, no solve.
    """
    ramp = np.geomspace(.001, .00025, 8)
    remaining = .012 - 2 * ramp.sum()
    middle = np.full(int(remaining / .00025), remaining / int(remaining / .00025))
    band = np.concatenate((ramp, middle, ramp[::-1]))
    dz = np.concatenate((np.full(20, .001), band, np.full(20, .001)))
    sim = Simulation(freq_max=5e9, domain=(.07, .07, dz.sum()), dx=.001,
                     dx_profile=np.full(70, .001), dy_profile=np.full(70, .001),
                     dz_profile=dz, boundary="cpml", cpml_layers=10)
    sim.add_material("lossy", eps_r=40, sigma=1.)
    sim.add(Box((.004, .007, .019), (.034, .037, .029)), material="lossy")
    sim.add_source((.013, .017, .022), "ez", amplitude_kind="current",
                   waveform=GaussianPulse(f0=3.25e9, bandwidth=.8))
    sim.add_probe((.018, .022, .025), "ez")
    options = dict(checkpoint=False, checkpoint_every=(304, 16))
    one = sim.estimate_ad_memory(6007, cotangents=1, **options)
    assert one.carry_bytes / 1e6 == pytest.approx(25.4, rel=.01)
    eight = sim.estimate_ad_memory(6007, cotangents=8, **options)
    per_extra_mb = (eight.ad_segmented_gb - one.ad_segmented_gb) * 1e3 / 7
    assert per_extra_mb == pytest.approx(191.2, rel=.10)
