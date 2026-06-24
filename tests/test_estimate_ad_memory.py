"""Issue #39 pin: estimate_ad_memory predictions must match observed
memory on the segmented scan-of-scan path within a tolerance.

Reference observations from VESSL job 369367233490 on RTX 4090:
  geometry: 2.4 GHz FR4 patch, dx=0.5mm NU (~608k cells)
  n_steps = 10000, emit_time_series=False

    checkpoint_every | peak GB
    ---------------- | -------
    50               | 4.82
    100              | 2.45
    200              | 1.26
    500              | 0.59
    1000             | 0.33

The formula `2 × n_segments × field_bytes + forward_bytes` fits these
points within ~25% (factor-of-2 accounts for carry + cotangent stacks
during reverse-mode).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfx import ADMemoryPlan, AD_MemoryEstimate, MeshIntelligenceReport, Simulation
from rfx.api import AD_MEMORY_FIT_SAFETY_FACTOR

_FORBIDDEN_CURRENT_EVIDENCE_FIELDS = {
    "compiler_memory_gb",
    "compiler_reported_required_gb",
    "observed_peak_gb",
    "profile_peak_gb",
    "certificate_status",
    "scope_digest",
    "config_digest",
    "environment_digest",
    "is_valid_certificate",
    "is_budget_safe",
    "peak_bound_gb",
}

_FORBIDDEN_RECOMMENDATION_TERMS = (
    "guarantee",
    "guaranteed",
    "certified",
    "certificate",
    "peak_bound",
)


def _assert_no_forbidden_current_fields(artifact: dict[str, object]) -> None:
    assert _FORBIDDEN_CURRENT_EVIDENCE_FIELDS.isdisjoint(artifact)


def _patch_like_sim():
    """Mirror the VESSL 369367233490 geometry: ext=40mm cube, dx=0.5mm,
    graded dz (0.3mm × 20 + 0.6mm × 30). Grid ends up 96 × 96 × 66 ≈ 608k.
    """
    ext = 40e-3
    dx = 0.5e-3
    dz = np.concatenate([np.full(20, 0.3e-3), np.full(30, 0.6e-3)])
    sim = Simulation(freq_max=10e9, domain=(ext, ext, float(np.sum(dz))),
                     dx=dx, dz_profile=dz, boundary="cpml", cpml_layers=8)
    sim.add_source((ext / 2, ext / 2, 1e-3), "ez")
    sim.add_probe((ext / 2, ext / 2, 2e-3), "ez")
    return sim


def _uniform_sim():
    sim = Simulation(
        freq_max=10e9,
        domain=(10e-3, 10e-3, 5e-3),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=2,
    )
    sim.add_source((5e-3, 5e-3, 2e-3), "ez")
    sim.add_probe((5e-3, 5e-3, 3e-3), "ez")
    return sim


def _grid_shape(sim: Simulation) -> tuple[int, int, int]:
    dx = sim._dx or (299_792_458.0 / sim._freq_max / 20.0)

    def _axis_cells(extent: float, profile) -> int:
        if profile is not None:
            return len(profile) + 1 + 2 * sim._cpml_layers
        return int(np.ceil(extent / dx)) + 1 + 2 * sim._cpml_layers

    return (
        _axis_cells(sim._domain[0], sim._dx_profile),
        _axis_cells(sim._domain[1], sim._dy_profile),
        _axis_cells(sim._domain[2], sim._dz_profile),
    )


def _design_mask(sim: Simulation, *, fraction: float = 0.25) -> np.ndarray:
    mask = np.zeros(_grid_shape(sim), dtype=bool)
    selected = max(1, int(mask.size * fraction))
    mask.reshape(-1)[:selected] = True
    return mask


def test_public_imports_and_json_roundtrip():
    from rfx import ADMemoryPlan as TopPlan
    from rfx import AD_MemoryEstimate as TopEstimate
    from rfx import Simulation as TopSimulation
    from rfx.api import ADMemoryPlan as ApiPlan
    from rfx.api import AD_MemoryEstimate as ApiEstimate
    from rfx.api import Simulation as ApiSimulation

    assert TopEstimate is ApiEstimate is AD_MemoryEstimate
    assert TopPlan is ApiPlan is ADMemoryPlan
    assert TopSimulation is ApiSimulation is Simulation

    sim = _uniform_sim()
    estimate = sim.estimate_ad_memory(n_steps=120, checkpoint_segments=10)
    plan = sim.plan_ad_memory(n_steps=120, available_memory_gb=100.0)

    assert json.loads(estimate.to_json()) == estimate.to_dict()
    assert json.loads(plan.to_json()) == plan.to_dict()
    assert estimate.to_dict()["checkpoint_segments"] == 10
    estimate_keys = {
        "evidence_class",
        "forward_gb",
        "ad_checkpointed_gb",
        "ad_full_gb",
        "ntff_dft_gb",
        "available_gb",
        "warning",
        "ad_segmented_gb",
        "checkpoint_every",
        "checkpoint_segments",
        "ad_active_steps",
        "ad_active_design_fraction",
        "ad_segmented_active_segments",
    }
    assert estimate.to_dict()["evidence_class"] == "static_estimate"
    plan_keys = {
        "evidence_class",
        "n_steps",
        "available_memory_gb",
        "target_fraction",
        "target_memory_gb",
        "fit_safety_factor",
        "checkpoint_every",
        "checkpoint_segments",
        "checkpoint_mode",
        "selected_estimate",
        "full_ad_fits",
        "segmented_fits",
        "recommendation",
    }
    assert plan.to_dict()["evidence_class"] == "calibrated_conservative_plan"
    assert set(estimate.to_dict()) == estimate_keys
    assert set(plan.to_dict()) == plan_keys
    assert set(plan.to_dict()["selected_estimate"]) == estimate_keys
    assert plan.to_dict()["selected_estimate"]["evidence_class"] == "static_estimate"
    _assert_no_forbidden_current_fields(estimate.to_dict())
    _assert_no_forbidden_current_fields(plan.to_dict())
    _assert_no_forbidden_current_fields(plan.to_dict()["selected_estimate"])
    assert "evidence_class" not in AD_MemoryEstimate._fields
    assert "evidence_class" not in ADMemoryPlan._fields
    with pytest.raises(ValueError):
        estimate._replace(evidence_class="observed_profile")
    with pytest.raises(ValueError):
        plan._replace(evidence_class="bounded_certificate")


def test_current_memory_artifacts_keep_evidence_classes_separate():
    sim = _patch_like_sim()
    estimate = sim.estimate_ad_memory(n_steps=10_000, checkpoint_every=500)
    plan = sim.plan_ad_memory(n_steps=10_000, available_memory_gb=1.0)
    report = sim.mesh_intelligence_report(
        n_steps=10_000,
        checkpoint_every=plan.checkpoint_every,
        available_memory_gb=1.0,
    )

    estimate_artifact = estimate.to_dict()
    plan_artifact = plan.to_dict()
    report_artifact = report.to_dict()

    assert estimate_artifact["evidence_class"] == "static_estimate"
    assert plan_artifact["evidence_class"] == "calibrated_conservative_plan"
    assert plan_artifact["selected_estimate"]["evidence_class"] == "static_estimate"
    assert report_artifact["ad_memory"]["evidence_class"] == "static_estimate"

    _assert_no_forbidden_current_fields(estimate_artifact)
    _assert_no_forbidden_current_fields(plan_artifact)
    _assert_no_forbidden_current_fields(plan_artifact["selected_estimate"])
    _assert_no_forbidden_current_fields(report_artifact["ad_memory"])


def test_current_memory_recommendations_do_not_claim_guarantees():
    sim = _patch_like_sim()
    plan = sim.plan_ad_memory(n_steps=10_000, available_memory_gb=1.0)
    report = sim.mesh_intelligence_report(
        n_steps=10_000,
        checkpoint_every=plan.checkpoint_every,
        available_memory_gb=1.0,
    )

    texts = [
        plan.recommendation,
        report.recommendation,
        sim.estimate_ad_memory(n_steps=10_000, available_memory_gb=1e-6).warning,
    ]
    for text in texts:
        assert text is not None
        lowered = text.lower()
        assert not any(term in lowered for term in _FORBIDDEN_RECOMMENDATION_TERMS)


def test_memory_reduction_docs_separate_planning_from_certificate_evidence():
    doc = Path("docs/public/guide/memory-reduction.mdx").read_text()
    assert "`static_estimate`" in doc
    assert "`calibrated_conservative_plan`" in doc
    assert "`bounded_certificate`" in doc
    assert "Current `estimate_ad_memory(...)` and `plan_ad_memory(...)` artifacts are not certificates" in doc
    assert "not a universal guaranteed peak predictor" in doc


@pytest.mark.parametrize("chunk,observed_gb", [
    (50, 4.82),
    (100, 2.45),
    (200, 1.26),
    (500, 0.59),
    (1000, 0.33),
])
def test_segmented_estimate_within_tolerance(chunk, observed_gb):
    sim = _patch_like_sim()
    est = sim.estimate_ad_memory(n_steps=10000, checkpoint_every=chunk)
    assert est.ad_segmented_gb is not None
    pred = est.ad_segmented_gb
    # Predictions should be within 2x (very loose to tolerate XLA
    # allocator slack); typical is ~1.2x.
    assert 0.5 * observed_gb <= pred <= 2.0 * observed_gb, (
        f"chunk={chunk}: predicted {pred:.3f} GB vs observed {observed_gb} GB"
    )
    assert pred * AD_MEMORY_FIT_SAFETY_FACTOR >= observed_gb


def test_checkpoint_every_none_leaves_segmented_null():
    sim = _patch_like_sim()
    est = sim.estimate_ad_memory(n_steps=1000)
    assert est.ad_segmented_gb is None
    assert est.checkpoint_every is None


def test_monotone_in_chunk():
    """Bigger chunk → smaller segmented memory (fewer segment boundaries)."""
    sim = _patch_like_sim()
    gbs = [
        sim.estimate_ad_memory(n_steps=10000, checkpoint_every=c).ad_segmented_gb
        for c in [50, 100, 500, 1000]
    ]
    assert gbs == sorted(gbs, reverse=True), gbs


def test_plan_ad_memory_returns_none_when_full_ad_fits():
    sim = _patch_like_sim()

    plan = sim.plan_ad_memory(n_steps=100, available_memory_gb=100.0)

    assert isinstance(plan, ADMemoryPlan)
    assert plan.full_ad_fits is True
    assert plan.segmented_fits is False
    assert plan.checkpoint_every is None
    assert plan.selected_estimate.checkpoint_every is None
    assert plan.checkpoint_segments is None
    assert plan.checkpoint_mode is None
    assert plan.selected_estimate.ad_full_gb * plan.fit_safety_factor <= plan.target_memory_gb
    assert plan.selected_estimate.ad_segmented_gb is None


def test_plan_ad_memory_recommends_checkpoint_every_for_budget():
    sim = _patch_like_sim()

    plan = sim.plan_ad_memory(n_steps=10_000, available_memory_gb=1.0)

    assert plan.full_ad_fits is False
    assert plan.segmented_fits is True
    assert plan.checkpoint_every is not None
    assert plan.selected_estimate.checkpoint_every == plan.checkpoint_every
    assert plan.checkpoint_segments is None
    assert plan.checkpoint_mode == "checkpoint_every"
    assert plan.selected_estimate.ad_segmented_gb * plan.fit_safety_factor <= plan.target_memory_gb
    if plan.checkpoint_every > 1:
        previous = sim.estimate_ad_memory(
            n_steps=10_000,
            available_memory_gb=1.0,
            checkpoint_every=plan.checkpoint_every - 1,
        )
        assert previous.ad_segmented_gb * plan.fit_safety_factor > plan.target_memory_gb


def test_plan_ad_memory_checkpoint_every_handles_unaligned_warmup():
    sim = _patch_like_sim()

    plan = sim.plan_ad_memory(
        n_steps=120,
        available_memory_gb=0.35,
        n_warmup=49,
    )

    assert plan.segmented_fits is True
    assert plan.checkpoint_mode == "checkpoint_every"
    assert plan.checkpoint_every == 12
    previous = sim.estimate_ad_memory(
        n_steps=120,
        available_memory_gb=0.35,
        checkpoint_every=11,
        n_warmup=49,
    )
    assert previous.ad_segmented_gb * plan.fit_safety_factor > plan.target_memory_gb
    assert plan.selected_estimate.ad_segmented_gb * plan.fit_safety_factor <= plan.target_memory_gb


def test_plan_ad_memory_reports_unfit_budget():
    sim = _patch_like_sim()

    plan = sim.plan_ad_memory(n_steps=10_000, available_memory_gb=0.001)

    assert plan.full_ad_fits is False
    assert plan.segmented_fits is False
    assert plan.checkpoint_every == 10_000
    assert plan.selected_estimate.ad_segmented_gb > plan.target_memory_gb
    assert plan.checkpoint_segments is None
    assert plan.checkpoint_mode == "checkpoint_every"
    assert "reduce mesh size" in plan.recommendation

    report = sim.mesh_intelligence_report(
        n_steps=10_000,
        checkpoint_every=plan.checkpoint_every,
        available_memory_gb=0.001,
    )
    assert "exceeds 85%" in report.recommendation
    assert "use segmented AD estimate" not in report.recommendation


def test_plan_ad_memory_reports_uniform_unfit_budget():
    sim = _uniform_sim()

    plan = sim.plan_ad_memory(n_steps=120, available_memory_gb=1e-6)

    assert plan.full_ad_fits is False
    assert plan.segmented_fits is False
    assert plan.checkpoint_every is None
    assert plan.checkpoint_segments == 1
    assert plan.checkpoint_mode == "checkpoint_segments"
    assert plan.selected_estimate.ad_segmented_gb > plan.target_memory_gb
    assert "checkpoint_segments=1" in plan.recommendation
    assert "reduce mesh size" in plan.recommendation


def test_plan_ad_memory_uses_valid_custom_target_fraction():
    sim = _patch_like_sim()

    plan = sim.plan_ad_memory(
        n_steps=10_000,
        available_memory_gb=1.0,
        target_fraction=0.5,
    )

    assert plan.target_fraction == 0.5
    assert plan.target_memory_gb == 0.5
    assert plan.selected_estimate.ad_segmented_gb * plan.fit_safety_factor <= plan.target_memory_gb
    if plan.checkpoint_every > 1:
        previous = sim.estimate_ad_memory(
            n_steps=10_000,
            available_memory_gb=1.0,
            checkpoint_every=plan.checkpoint_every - 1,
        )
        assert previous.ad_segmented_gb * plan.fit_safety_factor > plan.target_memory_gb


def test_plan_ad_memory_serializes_artifact():
    sim = _patch_like_sim()

    plan = sim.plan_ad_memory(n_steps=10_000, available_memory_gb=1.0)
    artifact = plan.to_dict()
    parsed = json.loads(plan.to_json())

    assert artifact["checkpoint_every"] == plan.checkpoint_every
    assert artifact["selected_estimate"]["checkpoint_every"] == plan.checkpoint_every
    assert artifact["segmented_fits"] is True
    assert artifact["checkpoint_segments"] == plan.checkpoint_segments
    assert artifact["checkpoint_mode"] == plan.checkpoint_mode
    assert artifact["fit_safety_factor"] == plan.fit_safety_factor
    assert artifact["selected_estimate"]["ad_segmented_active_segments"] is not None
    assert parsed == artifact


def test_checkpoint_segments_accounting_matches_uniform_segment_count():
    sim = _uniform_sim()

    by_segments = sim.estimate_ad_memory(n_steps=120, checkpoint_segments=10)
    by_chunk = sim.estimate_ad_memory(n_steps=120, checkpoint_every=12)

    assert by_segments.checkpoint_segments == 10
    assert by_segments.checkpoint_every is None
    assert by_segments.ad_segmented_gb == by_chunk.ad_segmented_gb
    assert by_segments.ad_segmented_active_segments == 10
    assert by_chunk.ad_segmented_active_segments == 10
    assert by_segments.to_dict()["checkpoint_segments"] == 10


def test_checkpoint_segments_active_count_honors_warmup():
    sim = _uniform_sim()

    full = sim.estimate_ad_memory(n_steps=120, checkpoint_segments=10)
    warm = sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_segments=10,
        n_warmup=60,
    )

    assert warm.forward_gb == full.forward_gb
    assert warm.ad_active_steps == 60
    assert warm.ad_segmented_gb < full.ad_segmented_gb
    assert warm.ad_segmented_active_segments == 5
    same_active_segments = sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_every=12,
        n_warmup=60,
    )
    assert warm.ad_segmented_gb == same_active_segments.ad_segmented_gb

    partial_warm = sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_segments=10,
        n_warmup=61,
    )
    partial_same_active_segments = sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_every=12,
        n_warmup=61,
    )
    assert partial_warm.ad_active_steps == 59
    assert partial_warm.ad_segmented_gb == partial_same_active_segments.ad_segmented_gb
    assert partial_warm.ad_segmented_gb == warm.ad_segmented_gb
    assert partial_warm.ad_segmented_active_segments == 5
    assert partial_same_active_segments.ad_segmented_active_segments == 5


def test_checkpoint_every_active_count_honors_unaligned_warmup():
    sim = _uniform_sim()

    unaligned = sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_every=50,
        n_warmup=49,
    )
    three_active_segments = sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_segments=3,
    )
    boundary = sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_every=50,
        n_warmup=50,
    )
    two_active_segments = sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_segments=2,
    )

    assert unaligned.ad_active_steps == 71
    assert unaligned.ad_segmented_gb == three_active_segments.ad_segmented_gb
    assert unaligned.ad_segmented_active_segments == 3
    assert boundary.ad_active_steps == 70
    assert boundary.ad_segmented_gb == two_active_segments.ad_segmented_gb
    assert boundary.ad_segmented_active_segments == 2


def test_plan_ad_memory_recommends_checkpoint_segments_for_uniform_budget():
    sim = _uniform_sim()

    plan = sim.plan_ad_memory(n_steps=120, available_memory_gb=0.002)

    assert plan.full_ad_fits is False
    assert plan.segmented_fits is True
    assert plan.checkpoint_every is None
    assert plan.checkpoint_segments == 10
    assert plan.checkpoint_mode == "checkpoint_segments"
    assert plan.selected_estimate.checkpoint_segments == plan.checkpoint_segments
    assert plan.selected_estimate.ad_segmented_gb * plan.fit_safety_factor <= plan.target_memory_gb
    assert "0.00 GB" not in plan.recommendation
    assert "MB" in plan.recommendation


def test_estimate_rejects_ambiguous_checkpoint_modes():
    sim = _uniform_sim()

    with pytest.raises(ValueError, match="mutually exclusive"):
        sim.estimate_ad_memory(
            n_steps=120,
            checkpoint_every=12,
            checkpoint_segments=10,
        )


def test_estimate_rejects_invalid_checkpoint_segments():
    sim = _uniform_sim()

    with pytest.raises(ValueError, match="does not divide n_steps"):
        sim.estimate_ad_memory(n_steps=120, checkpoint_segments=11)
    with pytest.raises(ValueError, match="checkpoint_segments must be"):
        sim.estimate_ad_memory(n_steps=120, checkpoint_segments=0)
    with pytest.raises(ValueError, match="checkpoint_segments must be"):
        sim.estimate_ad_memory(n_steps=120, checkpoint_segments=-1)


def test_estimate_rejects_invalid_active_tape_inputs():
    sim = _uniform_sim()

    with pytest.raises(ValueError, match="n_steps must be positive"):
        sim.estimate_ad_memory(n_steps=0)
    with pytest.raises(ValueError, match="n_warmup must be >= 0"):
        sim.estimate_ad_memory(n_steps=120, n_warmup=-1)
    with pytest.raises(ValueError, match="must be < n_steps"):
        sim.estimate_ad_memory(n_steps=120, n_warmup=120)
    with pytest.raises(ValueError, match="checkpoint_every must be positive"):
        sim.estimate_ad_memory(n_steps=120, checkpoint_every=0)
    with pytest.raises(ValueError, match="checkpoint_every must be positive"):
        sim.estimate_ad_memory(n_steps=120, checkpoint_every=-1)
    with pytest.raises(ValueError, match="design_mask must be non-empty"):
        sim.estimate_ad_memory(n_steps=120, design_mask=np.array([], dtype=bool))
    with pytest.raises(ValueError, match="boolean array matching grid shape"):
        sim.estimate_ad_memory(n_steps=120, design_mask=np.array(True))
    with pytest.raises(TypeError, match="boolean dtype"):
        sim.estimate_ad_memory(
            n_steps=120,
            design_mask=np.ones(_grid_shape(sim), dtype=np.int8),
        )
    with pytest.raises(ValueError, match="select at least one cell"):
        sim.estimate_ad_memory(
            n_steps=120,
            design_mask=np.zeros(_grid_shape(sim), dtype=bool),
        )
    with pytest.raises(ValueError, match="must match simulation grid shape"):
        sim.estimate_ad_memory(
            n_steps=120,
            design_mask=np.ones((2, 2, 2), dtype=bool),
        )
    with pytest.raises(TypeError, match="n_steps must be an integer"):
        sim.estimate_ad_memory(n_steps=1.5)
    with pytest.raises(TypeError, match="n_steps must be an integer"):
        sim.estimate_ad_memory(n_steps=True)
    with pytest.raises(TypeError, match="n_warmup must be an integer"):
        sim.estimate_ad_memory(n_steps=120, n_warmup=1.5)
    with pytest.raises(TypeError, match="checkpoint_every must be an integer"):
        sim.estimate_ad_memory(n_steps=120, checkpoint_every=12.5)
    with pytest.raises(TypeError, match="checkpoint_segments must be an integer"):
        sim.estimate_ad_memory(n_steps=120, checkpoint_segments=10.5)
    with pytest.raises(ValueError, match="available_memory_gb must be positive"):
        sim.estimate_ad_memory(n_steps=120, available_memory_gb=0.0)
    with pytest.raises(ValueError, match="available_memory_gb must be positive"):
        sim.estimate_ad_memory(n_steps=120, available_memory_gb=-1.0)
    with pytest.raises(TypeError, match="finite real scalar"):
        sim.estimate_ad_memory(n_steps=120, available_memory_gb=True)
    with pytest.raises(ValueError, match="available_memory_gb must be finite"):
        sim.estimate_ad_memory(n_steps=120, available_memory_gb=float("nan"))
    with pytest.raises(ValueError, match="available_memory_gb must be finite"):
        sim.estimate_ad_memory(n_steps=120, available_memory_gb=float("inf"))
    with pytest.raises(TypeError, match="finite real scalar"):
        sim.estimate_ad_memory(n_steps=120, available_memory_gb=[1.0])
    with pytest.raises(TypeError, match="finite real scalar"):
        sim.estimate_ad_memory(n_steps=120, available_memory_gb=1 + 0j)


def test_plan_ad_memory_rejects_invalid_scalar_inputs():
    sim = _uniform_sim()

    with pytest.raises(ValueError, match="n_steps must be positive"):
        sim.plan_ad_memory(n_steps=0, available_memory_gb=1.0)
    with pytest.raises(ValueError, match="n_warmup must be >= 0"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=1.0, n_warmup=-1)
    with pytest.raises(ValueError, match="must be < n_steps"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=1.0, n_warmup=120)
    with pytest.raises(ValueError, match="available_memory_gb must be positive"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=0.0)
    with pytest.raises(ValueError, match="available_memory_gb must be positive"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=-1.0)
    with pytest.raises(TypeError, match="finite real scalar"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=True)
    with pytest.raises(ValueError, match="available_memory_gb must be finite"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=float("nan"))
    with pytest.raises(ValueError, match="available_memory_gb must be finite"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=float("inf"))
    with pytest.raises(TypeError, match="finite real scalar"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=[1.0])
    with pytest.raises(TypeError, match="finite real scalar"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=1 + 0j)
    with pytest.raises(ValueError, match="target_fraction must be positive"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=1.0, target_fraction=0.0)
    with pytest.raises(ValueError, match="target_fraction must be positive"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=1.0, target_fraction=-0.1)
    with pytest.raises(ValueError, match="interval"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=1.0, target_fraction=1.5)
    with pytest.raises(ValueError, match="target_fraction must be finite"):
        sim.plan_ad_memory(
            n_steps=120,
            available_memory_gb=1.0,
            target_fraction=float("inf"),
        )
    with pytest.raises(ValueError, match="target_fraction must be finite"):
        sim.plan_ad_memory(
            n_steps=120,
            available_memory_gb=1.0,
            target_fraction=float("nan"),
        )
    with pytest.raises(TypeError, match="finite real scalar"):
        sim.plan_ad_memory(
            n_steps=120,
            available_memory_gb=1.0,
            target_fraction=True,
        )
    with pytest.raises(TypeError, match="finite real scalar"):
        sim.plan_ad_memory(
            n_steps=120,
            available_memory_gb=1.0,
            target_fraction=[0.85],
        )
    with pytest.raises(TypeError, match="finite real scalar"):
        sim.plan_ad_memory(
            n_steps=120,
            available_memory_gb=1.0,
            target_fraction=1 + 0j,
        )
    with pytest.raises(ValueError, match="safety_factor must be positive"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=1.0, safety_factor=0.0)
    with pytest.raises(ValueError, match="safety_factor must be >= 1"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=1.0, safety_factor=0.5)
    with pytest.raises(ValueError, match="safety_factor must be finite"):
        sim.plan_ad_memory(
            n_steps=120,
            available_memory_gb=1.0,
            safety_factor=float("nan"),
        )
    with pytest.raises(TypeError, match="finite real scalar"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=1.0, safety_factor=True)


def test_plan_ad_memory_rejects_invalid_design_masks():
    sim = _uniform_sim()

    with pytest.raises(ValueError, match="design_mask must be non-empty"):
        sim.plan_ad_memory(
            n_steps=120,
            available_memory_gb=1.0,
            design_mask=np.array([], dtype=bool),
        )
    with pytest.raises(ValueError, match="boolean array matching grid shape"):
        sim.plan_ad_memory(
            n_steps=120,
            available_memory_gb=1.0,
            design_mask=np.array(True),
        )
    with pytest.raises(TypeError, match="boolean dtype"):
        sim.plan_ad_memory(
            n_steps=120,
            available_memory_gb=1.0,
            design_mask=np.ones(_grid_shape(sim), dtype=np.int8),
        )
    with pytest.raises(ValueError, match="select at least one cell"):
        sim.plan_ad_memory(
            n_steps=120,
            available_memory_gb=1.0,
            design_mask=np.zeros(_grid_shape(sim), dtype=bool),
        )
    with pytest.raises(ValueError, match="must match simulation grid shape"):
        sim.plan_ad_memory(
            n_steps=120,
            available_memory_gb=1.0,
            design_mask=np.ones((2, 2, 2), dtype=bool),
        )


def test_warning_uses_realistic_selected_estimate():
    sim = _uniform_sim()

    full = sim.estimate_ad_memory(n_steps=120, available_memory_gb=1e-6)
    assert full.warning is not None
    assert "non-checkpointed" in full.warning
    assert "Use plan_ad_memory()" in full.warning

    segmented = sim.estimate_ad_memory(
        n_steps=120,
        available_memory_gb=1e-6,
        checkpoint_segments=10,
    )
    assert segmented.warning is not None
    assert "segmented" in segmented.warning
    assert "Reduce checkpoint_segments" in segmented.warning

    chunked = sim.estimate_ad_memory(
        n_steps=120,
        available_memory_gb=1e-6,
        checkpoint_every=12,
    )
    assert chunked.warning is not None
    assert "segmented" in chunked.warning
    assert "Increase checkpoint_every" in chunked.warning

    minimum_segmented = sim.estimate_ad_memory(
        n_steps=120,
        available_memory_gb=1e-6,
        checkpoint_segments=1,
    )
    assert minimum_segmented.warning is not None
    assert "Reduce checkpoint_segments" not in minimum_segmented.warning
    assert "more aggressive memory-reduction lane" in minimum_segmented.warning

    minimum_chunked = sim.estimate_ad_memory(
        n_steps=120,
        available_memory_gb=1e-6,
        checkpoint_every=120,
    )
    assert minimum_chunked.warning is not None
    assert "Increase checkpoint_every" not in minimum_chunked.warning
    assert "more aggressive memory-reduction lane" in minimum_chunked.warning


def test_warning_has_no_false_positive_at_boundary():
    sim = _uniform_sim()

    baseline = sim.estimate_ad_memory(n_steps=120)
    exact_available = baseline.ad_full_gb / 0.85
    exact = sim.estimate_ad_memory(
        n_steps=120,
        available_memory_gb=exact_available,
    )
    below = sim.estimate_ad_memory(
        n_steps=120,
        available_memory_gb=exact_available * 0.99,
    )

    assert exact.warning is None
    assert below.warning is not None

    segmented_baseline = sim.estimate_ad_memory(n_steps=120, checkpoint_segments=10)
    segmented_available = segmented_baseline.ad_segmented_gb / 0.85
    segmented_exact = sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_segments=10,
        available_memory_gb=segmented_available,
    )
    assert segmented_exact.warning is None

    chunked_baseline = sim.estimate_ad_memory(n_steps=120, checkpoint_every=12)
    chunked_available = chunked_baseline.ad_segmented_gb / 0.85
    chunked_exact = sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_every=12,
        available_memory_gb=chunked_available,
    )
    assert chunked_exact.warning is None


def test_ntff_dft_bytes_contribute_to_ad_estimates():
    baseline = _uniform_sim().estimate_ad_memory(n_steps=120, checkpoint_segments=10)
    chunk_baseline = _uniform_sim().estimate_ad_memory(n_steps=120, checkpoint_every=12)
    sim = _uniform_sim()
    sim.add_ntff_box(
        (1e-3, 1e-3, 1e-3),
        (9e-3, 9e-3, 4e-3),
        n_freqs=3,
    )

    with_ntff = sim.estimate_ad_memory(n_steps=120, checkpoint_segments=10)
    chunk_with_ntff = sim.estimate_ad_memory(n_steps=120, checkpoint_every=12)
    report = sim.mesh_intelligence_report(n_steps=120, checkpoint_every=12)

    assert with_ntff.ntff_dft_gb > 0.0
    assert with_ntff.ad_full_gb > baseline.ad_full_gb
    assert with_ntff.ad_segmented_gb > baseline.ad_segmented_gb
    assert chunk_with_ntff.ad_segmented_gb > chunk_baseline.ad_segmented_gb
    assert report.ad_memory is not None
    assert report.ad_memory.ntff_dft_gb > 0.0

    nu_baseline_sim = _patch_like_sim()
    nu_baseline = nu_baseline_sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_every=12,
    )
    nu_with_ntff_sim = _patch_like_sim()
    nu_with_ntff_sim.add_ntff_box(
        (1e-3, 1e-3, 1e-3),
        (39e-3, 39e-3, 20e-3),
        n_freqs=3,
    )
    nu_with_ntff = nu_with_ntff_sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_every=12,
    )
    nu_report = nu_with_ntff_sim.mesh_intelligence_report(
        n_steps=120,
        checkpoint_every=12,
    )

    assert nu_with_ntff.ntff_dft_gb > 0.0
    assert nu_with_ntff.ad_segmented_gb > nu_baseline.ad_segmented_gb
    assert nu_report.ad_memory is not None
    assert nu_report.ad_memory.ntff_dft_gb > 0.0


def test_nonfinite_artifacts_refuse_json_serialization():
    bad_estimate = AD_MemoryEstimate(
        forward_gb=float("nan"),
        ad_checkpointed_gb=1.0,
        ad_full_gb=1.0,
        ntff_dft_gb=0.0,
        available_gb=None,
        warning=None,
    )
    with pytest.raises(ValueError, match="Out of range"):
        bad_estimate.to_json()
    with pytest.raises(ValueError, match="Out of range"):
        bad_estimate.to_json(allow_nan=True)

    good_estimate = _uniform_sim().estimate_ad_memory(n_steps=120)
    bad_plan = ADMemoryPlan(
        n_steps=120,
        available_memory_gb=float("inf"),
        target_fraction=0.85,
        target_memory_gb=1.0,
        checkpoint_every=None,
        selected_estimate=good_estimate,
        full_ad_fits=True,
        segmented_fits=True,
        recommendation="bad nonfinite artifact",
    )
    with pytest.raises(ValueError, match="Out of range"):
        bad_plan.to_json()
    with pytest.raises(ValueError, match="Out of range"):
        bad_plan.to_json(allow_nan=True)

    bad_report = MeshIntelligenceReport(
        grid_shape=(1, 1, 1),
        cells=1,
        uniform_fine_shape=(1, 1, 1),
        uniform_fine_cells=1,
        cell_savings_factor=float("nan"),
        min_cell_size=1.0,
        nominal_dx=1.0,
        uses_nonuniform=False,
        preflight_issues=(),
        ad_memory=None,
        recommendation="bad nonfinite report",
    )
    with pytest.raises(ValueError, match="Out of range"):
        bad_report.to_json()
    with pytest.raises(ValueError, match="Out of range"):
        bad_report.to_json(allow_nan=True)


def test_n_warmup_reduces_reverse_tape_only():
    sim = _patch_like_sim()

    full = sim.estimate_ad_memory(n_steps=1_000)
    warm = sim.estimate_ad_memory(n_steps=1_000, n_warmup=600)

    assert warm.forward_gb == full.forward_gb
    assert warm.ad_active_steps == 400
    assert warm.ad_full_gb < full.ad_full_gb


def test_design_mask_records_fraction_without_reducing_primary_memory():
    sim = _patch_like_sim()
    design_mask = _design_mask(sim)
    expected_fraction = float(np.count_nonzero(design_mask)) / float(design_mask.size)

    full = sim.estimate_ad_memory(n_steps=1_000)
    masked = sim.estimate_ad_memory(n_steps=1_000, design_mask=design_mask)

    assert masked.forward_gb == full.forward_gb
    assert masked.ad_active_design_fraction == pytest.approx(expected_fraction)
    assert masked.ad_full_gb == full.ad_full_gb


def test_plan_ad_memory_rejects_non_integral_counts():
    sim = _uniform_sim()

    with pytest.raises(TypeError, match="n_steps must be an integer"):
        sim.plan_ad_memory(n_steps=120.5, available_memory_gb=1.0)
    with pytest.raises(TypeError, match="n_steps must be an integer"):
        sim.plan_ad_memory(n_steps=True, available_memory_gb=1.0)
    with pytest.raises(TypeError, match="n_warmup must be an integer"):
        sim.plan_ad_memory(n_steps=120, available_memory_gb=1.0, n_warmup=1.5)


def test_plan_ad_memory_records_warmup_and_mask_metadata():
    sim = _patch_like_sim()
    budget_gb = 0.2
    design_mask = _design_mask(sim)
    expected_fraction = float(np.count_nonzero(design_mask)) / float(design_mask.size)
    warm_only = sim.plan_ad_memory(
        n_steps=10_000,
        available_memory_gb=budget_gb,
        n_warmup=9_000,
    )

    baseline = sim.plan_ad_memory(n_steps=10_000, available_memory_gb=budget_gb)
    masked = sim.plan_ad_memory(
        n_steps=10_000,
        available_memory_gb=budget_gb,
        n_warmup=9_000,
        design_mask=design_mask,
    )

    assert baseline.full_ad_fits is False
    assert masked.selected_estimate.ad_active_steps == 1_000
    assert masked.selected_estimate.ad_active_design_fraction == pytest.approx(expected_fraction)
    assert masked.selected_estimate.ad_full_gb == warm_only.selected_estimate.ad_full_gb
    assert masked.checkpoint_every == warm_only.checkpoint_every
    assert masked.checkpoint_every is None or masked.checkpoint_every < baseline.checkpoint_every
