"""Issue #39 / #277 pin: estimate_ad_memory predictions must match observed
memory on the segmented scan-of-scan path within a tolerance.

Model (#277): segmented reverse-mode AD peak =
``(2 × active_segments + live_tape_steps) × field_bytes + forward + ntff``.
The ``2 × active_segments`` term is boundary carry + cotangent storage
(issue #39). The ``live_tape_steps`` term is the live-segment
rematerialization tape: the backward pass replays one segment at a time,
so that segment's per-step field tape is resident on top of the boundary
storage (the runners document peak as O((K + s) · |carry|)).

Observed evidence (2.4 GHz FR4 patch, dx=0.5mm NU, RTX 4090):

* VESSL 369367233509 (inverse-design GRADIENT smoke — the regime this
  estimator serves) is the calibration reference:

      cells  | n_steps | checkpoint_every | peak GB
      ------ | ------- | ---------------- | -------
      ~603k  | 10000   | 100              | 5.84
      ~603k  | 2000    | 64               | 2.56

  The boundary-only #39 formula under-predicts these by 1.9-2.6x; the
  corrected #277 formula lands within ~1.3x below (conservative planning
  band, see AD_MEMORY_FIT_SAFETY_FACTOR).

* VESSL 369367233490 (the original #39 "segmented scan validation" sweep:
  4.82/2.45/1.26/0.59/0.33 GB at chunk 50/100/200/500/1000, n_steps=10000)
  tracked the boundary term only — its backward pass did not materialize
  field-sized per-step residuals, so it cannot calibrate the live-segment
  term. It is retained here as history, NOT as the estimator reference:
  a planner that matched ...490 would under-predict the ...509
  gradient-run peaks by up to ~2.6x (and the #277 desk ladder by ~272x at
  n_steps=6999, checkpoint_segments=3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfx import (
    ADMemoryActionHint,
    ADMemoryComponent,
    ADMemoryExplainabilityReport,
    ADMemoryPreflightReport,
    ADCompiledMemoryCertificate,
    ADMemoryPlan,
    AD_MemoryEstimate,
    MeshIntelligenceReport,
    Simulation,
)
from rfx.api import (
    AD_MEMORY_FIT_SAFETY_FACTOR,
    AD_MEMORY_PREFLIGHT_EVIDENCE_BOUNDARIES,
)

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
    "estimate_within_budget",
    "peak_bound_gb",
}

_FORBIDDEN_RECOMMENDATION_TERMS = (
    "guarantee",
    "guaranteed",
    "certified",
    "certificate",
    "peak_bound",
    "runtime peak",
    "profiler",
    "profile",
    "xla",
    "compiler memory",
    "rf validation",
)

# Pinned to the source constant so the doc-boundary check cannot silently drift
# from the boundaries rfx actually emits.
AD_MEMORY_PREFLIGHT_BOUNDARIES = AD_MEMORY_PREFLIGHT_EVIDENCE_BOUNDARIES


def _assert_no_forbidden_current_fields(artifact: dict[str, object]) -> None:
    assert _FORBIDDEN_CURRENT_EVIDENCE_FIELDS.isdisjoint(artifact)


def _assert_no_forbidden_fields_recursive(value: object) -> None:
    if isinstance(value, dict):
        assert _FORBIDDEN_CURRENT_EVIDENCE_FIELDS.isdisjoint(value)
        for item in value.values():
            _assert_no_forbidden_fields_recursive(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_fields_recursive(item)


def _assert_validate_physics_hint(report: ADMemoryPreflightReport) -> None:
    hint = next(
        hint
        for hint in report.action_hints
        if hint.code == "validate_physics_separately"
    )
    assert hint.severity == "info"
    assert hint.blocking is False
    assert hint.checkpoint_mode is None
    assert "electromagnetic correctness" in hint.message.lower()
    assert "physics claims" in hint.action.lower()


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


class _Missing:
    pass


_MISSING = _Missing()


class _FakeMemoryAnalysis:
    def __init__(
        self,
        *,
        temp_size_in_bytes=100,
        argument_size_in_bytes=200,
        output_size_in_bytes=300,
        alias_size_in_bytes=50,
    ):
        if temp_size_in_bytes is not _MISSING:
            self.temp_size_in_bytes = temp_size_in_bytes
        if argument_size_in_bytes is not _MISSING:
            self.argument_size_in_bytes = argument_size_in_bytes
        if output_size_in_bytes is not _MISSING:
            self.output_size_in_bytes = output_size_in_bytes
        if alias_size_in_bytes is not _MISSING:
            self.alias_size_in_bytes = alias_size_in_bytes

    def __repr__(self):
        return "<raw-analysis-repr-must-not-be-serialized>"


class _FakeCompiled:
    def __init__(
        self,
        analysis=_MISSING,
        *,
        raises: Exception | None = None,
        scope=None,
    ):
        self._analysis = _FakeMemoryAnalysis() if analysis is _MISSING else analysis
        self._raises = raises
        if scope is not None:
            self.rfx_memory_scope = scope

    def memory_analysis(self):
        if self._raises is not None:
            raise self._raises
        return self._analysis


class _NoMemoryAnalysis:
    pass


class _RaisingMemoryAnalysisAttribute:
    @property
    def memory_analysis(self):
        raise RuntimeError("attribute boom")


class _RaisingAnalysisField:
    @property
    def temp_size_in_bytes(self):
        raise RuntimeError("field boom")

    argument_size_in_bytes = 1
    output_size_in_bytes = 1
    alias_size_in_bytes = 0


class _RaisingIntrospectionCompiled(_FakeCompiled):
    @property
    def rfx_memory_scope(self):
        raise RuntimeError("scope boom")


class _RaisingToDictAttribute:
    @property
    def to_dict(self):
        raise RuntimeError("to_dict descriptor boom")


class _RaisingShapeAttribute:
    @property
    def shape(self):
        raise RuntimeError("shape descriptor boom")


class _ListPreflight:
    def to_dict(self):
        return ["not", "a", "mapping"]


class _FakeJaxDevice:
    def __init__(
        self,
        *,
        platform="cpu",
        device_kind="Fake CPU",
        raise_platform=False,
        raise_kind=False,
    ):
        self._platform = platform
        self._device_kind = device_kind
        self._raise_platform = raise_platform
        self._raise_kind = raise_kind

    @property
    def platform(self):
        if self._raise_platform:
            raise RuntimeError("platform boom")
        return self._platform

    @property
    def device_kind(self):
        if self._raise_kind:
            raise RuntimeError("kind boom")
        return self._device_kind


def _certificate_kwargs(**overrides):
    kwargs = {
        "n_steps": 120,
        "available_memory_gb": 1.0,
        "precision": "float32",
        "input_signature": {
            "eps_r": {
                "shape": [4, 4],
                "dtype": "float32",
                "weak_type": False,
                "role": "design",
            }
        },
        "static_signature": {"dx": 0.01, "boundary": "pml"},
        "compiled_object_id": "toy-loss-compiled",
        "runner_or_objective": "toy_loss",
    }
    kwargs.update(overrides)
    return kwargs


def test_public_imports_and_json_roundtrip():
    from rfx import ADMemoryPlan as TopPlan
    from rfx import AD_MemoryEstimate as TopEstimate
    from rfx import ADMemoryComponent as TopComponent
    from rfx import ADMemoryActionHint as TopHint
    from rfx import ADMemoryExplainabilityReport as TopExplanation
    from rfx import ADMemoryPreflightReport as TopPreflight
    from rfx import ADCompiledMemoryCertificate as TopCertificate
    from rfx import Simulation as TopSimulation
    from rfx.api import ADMemoryPlan as ApiPlan
    from rfx.api import AD_MemoryEstimate as ApiEstimate
    from rfx.api import ADMemoryComponent as ApiComponent
    from rfx.api import ADMemoryActionHint as ApiHint
    from rfx.api import ADMemoryExplainabilityReport as ApiExplanation
    from rfx.api import ADMemoryPreflightReport as ApiPreflight
    from rfx.api import ADCompiledMemoryCertificate as ApiCertificate
    from rfx.api import Simulation as ApiSimulation

    assert TopEstimate is ApiEstimate is AD_MemoryEstimate
    assert TopPlan is ApiPlan is ADMemoryPlan
    assert TopComponent is ApiComponent is ADMemoryComponent
    assert TopHint is ApiHint is ADMemoryActionHint
    assert TopExplanation is ApiExplanation is ADMemoryExplainabilityReport
    assert TopPreflight is ApiPreflight is ADMemoryPreflightReport
    assert TopCertificate is ApiCertificate is ADCompiledMemoryCertificate
    assert TopSimulation is ApiSimulation is Simulation

    sim = _uniform_sim()
    estimate = sim.estimate_ad_memory(n_steps=120, checkpoint_segments=10)
    plan = sim.plan_ad_memory(n_steps=120, available_memory_gb=100.0)
    explanation = sim.explain_ad_memory(n_steps=120, checkpoint_segments=10)
    preflight = sim.ad_memory_preflight(n_steps=120, available_memory_gb=100.0)
    certificate = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(),
    )

    assert json.loads(estimate.to_json()) == estimate.to_dict()
    assert json.loads(plan.to_json()) == plan.to_dict()
    assert json.loads(explanation.to_json()) == explanation.to_dict()
    assert json.loads(preflight.to_json()) == preflight.to_dict()
    assert json.loads(certificate.to_json()) == certificate.to_dict()
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
    explanation_keys = {
        "evidence_class",
        "n_steps",
        "strategy",
        "selected_memory_gb",
        "selected_memory_field",
        "estimate",
        "components",
        "dominant_component",
        "recommendations",
    }
    preflight_keys = {
        "evidence_class",
        "source_evidence_classes",
        "n_steps",
        "available_memory_gb",
        "target_fraction",
        "target_memory_gb",
        "fit_safety_factor",
        "status",
        "supported_checkpoint_mode",
        "checkpoint_every",
        "checkpoint_segments",
        "full_ad_fits",
        "checkpointing_fits",
        "memory_plan",
        "explainability",
        "mesh_report",
        "residual_diagnostic",
        "action_hints",
        "evidence_boundaries",
        "recommendation",
    }
    certificate_keys = {
        "evidence_class",
        "status",
        "status_reason",
        "is_valid_certificate",
        "estimate_within_budget",
        "available_memory_gb",
        "target_fraction",
        "target_memory_gb",
        "compiler_reported_required_bytes",
        "compiler_reported_required_gb",
        "temp_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_gb",
        "argument_gb",
        "output_gb",
        "alias_gb",
        "exact_scope",
        "scope_status",
        "scope_status_reason",
        "scope_digest",
        "config_digest",
        "environment_digest",
        "memory_analysis_status",
        "memory_analysis_status_reason",
        "jax_version",
        "evidence_boundaries",
        "recommendations",
        "source_preflight",
    }
    hint_keys = {
        "evidence_class",
        "code",
        "severity",
        "message",
        "action",
        "checkpoint_mode",
        "checkpoint_every",
        "checkpoint_segments",
        "blocking",
    }
    assert plan.to_dict()["evidence_class"] == "calibrated_conservative_plan"
    assert set(estimate.to_dict()) == estimate_keys
    assert set(plan.to_dict()) == plan_keys
    assert set(explanation.to_dict()) == explanation_keys
    assert set(preflight.to_dict()) == preflight_keys
    assert set(certificate.to_dict()) == certificate_keys
    assert all(set(hint) == hint_keys for hint in preflight.to_dict()["action_hints"])
    assert all(
        hint["evidence_class"] == "static_action_hint"
        for hint in preflight.to_dict()["action_hints"]
    )
    assert explanation.to_dict()["estimate"]["evidence_class"] == "static_estimate"
    assert explanation.to_dict()["evidence_class"] == "static_ad_explainability"
    assert preflight.to_dict()["evidence_class"] == "composite_ad_memory_preflight"
    assert certificate.to_dict()["evidence_class"] == "bounded_certificate"
    assert set(plan.to_dict()["selected_estimate"]) == estimate_keys
    assert plan.to_dict()["selected_estimate"]["evidence_class"] == "static_estimate"
    _assert_no_forbidden_current_fields(estimate.to_dict())
    _assert_no_forbidden_current_fields(plan.to_dict())
    _assert_no_forbidden_current_fields(plan.to_dict()["selected_estimate"])
    _assert_no_forbidden_current_fields(preflight.to_dict())
    _assert_no_forbidden_fields_recursive(preflight.to_dict())
    assert "evidence_class" not in AD_MemoryEstimate._fields
    assert "evidence_class" not in ADMemoryPlan._fields
    assert "evidence_class" not in ADMemoryComponent._fields
    assert "evidence_class" not in ADMemoryExplainabilityReport._fields
    assert "evidence_class" not in ADMemoryActionHint._fields
    assert "evidence_class" not in ADMemoryPreflightReport._fields
    assert "evidence_class" not in ADCompiledMemoryCertificate._fields
    with pytest.raises(ValueError):
        estimate._replace(evidence_class="observed_profile")
    with pytest.raises(ValueError):
        plan._replace(evidence_class="bounded_certificate")


def test_ad_memory_compiled_certificate_fit_and_budget_exceeded():
    sim = _uniform_sim()
    fit = sim.ad_memory_compiled_certificate(
        _FakeCompiled(
            _FakeMemoryAnalysis(
                temp_size_in_bytes=1_000,
                argument_size_in_bytes=2_000,
                output_size_in_bytes=3_000,
                alias_size_in_bytes=500,
            )
        ),
        **_certificate_kwargs(),
    )
    artifact = fit.to_dict()

    assert fit.evidence_class == "bounded_certificate"
    assert fit.status == "compiler_estimate_within_budget"
    assert fit.is_valid_certificate is True
    assert fit.estimate_within_budget is True
    assert artifact["compiler_reported_required_bytes"] == 5_500
    assert artifact["compiler_reported_required_gb"] == pytest.approx(5_500 / 1e9)
    assert artifact["temp_size_in_bytes"] == 1_000
    assert artifact["argument_size_in_bytes"] == 2_000
    assert artifact["output_size_in_bytes"] == 3_000
    assert artifact["alias_size_in_bytes"] == 500
    assert artifact["scope_status"] == "complete"
    assert artifact["memory_analysis_status"] == "complete"
    assert artifact["exact_scope"]["memory_analysis_fields"] == [
        "temp_size_in_bytes",
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
    ]
    assert "Digests are audit identities only" in " ".join(fit.recommendations)
    assert "raw-analysis-repr-must-not-be-serialized" not in fit.to_json()

    exceeded = sim.ad_memory_compiled_certificate(
        _FakeCompiled(
            _FakeMemoryAnalysis(
                temp_size_in_bytes=2_000_000_000,
                argument_size_in_bytes=0,
                output_size_in_bytes=0,
                alias_size_in_bytes=0,
            )
        ),
        **_certificate_kwargs(),
    )
    assert exceeded.status == "compiler_estimate_exceeds_budget"
    assert exceeded.is_valid_certificate is True
    assert exceeded.estimate_within_budget is False


def test_ad_memory_compiled_certificate_budget_boundary_and_estimate_framing():
    """The ``<=`` budget boundary is inclusive, and a fit reads as a compiler
    *estimate* — not a runtime guarantee."""
    sim = _uniform_sim()
    # target_bytes = available_memory_gb * target_fraction * 1e9 = 1.0 * 0.5 * 1e9
    target_bytes = 500_000_000
    at_budget = sim.ad_memory_compiled_certificate(
        _FakeCompiled(
            _FakeMemoryAnalysis(
                temp_size_in_bytes=target_bytes,
                argument_size_in_bytes=0,
                output_size_in_bytes=0,
                alias_size_in_bytes=0,
            )
        ),
        **_certificate_kwargs(target_fraction=0.5),
    )
    assert at_budget.compiler_reported_required_bytes == target_bytes
    assert at_budget.status == "compiler_estimate_within_budget"
    assert at_budget.estimate_within_budget is True
    # A fit must read as a compiler ESTIMATE, never a runtime guarantee.
    fit_recs = " ".join(at_budget.recommendations).lower()
    assert "compiler estimate" in fit_recs
    assert "guarantee" not in fit_recs

    over_budget = sim.ad_memory_compiled_certificate(
        _FakeCompiled(
            _FakeMemoryAnalysis(
                temp_size_in_bytes=target_bytes + 1,
                argument_size_in_bytes=0,
                output_size_in_bytes=0,
                alias_size_in_bytes=0,
            )
        ),
        **_certificate_kwargs(target_fraction=0.5),
    )
    assert over_budget.status == "compiler_estimate_exceeds_budget"
    assert over_budget.estimate_within_budget is False
    # The budget verdict is surfaced in a dedicated structured field.
    assert "exceed" in over_budget.status_reason
    assert over_budget.to_dict()["status_reason"] == over_budget.status_reason


def test_ad_memory_compiled_certificate_summarizes_array_signature_inputs():
    sim = _uniform_sim()
    report = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(
            input_signature={
                "eps_r": np.ones((2, 3), dtype=np.float32),
                "scalar": np.array(1.0, dtype=np.float64),
            },
            static_signature={
                "bias": np.arange(4, dtype=np.int32),
                "mode": "toy",
            },
        ),
    )
    scope = report.exact_scope

    assert report.status == "compiler_estimate_within_budget"
    assert scope["input_signature"]["eps_r"] == {
        "shape": [2, 3],
        "dtype": "float32",
    }
    assert scope["input_signature"]["scalar"] == {
        "shape": [],
        "dtype": "float64",
    }
    assert scope["static_signature"]["bias"] == {
        "shape": [4],
        "dtype": "int32",
    }
    assert scope["static_signature"]["mode"] == "toy"
    assert scope["input_signature"]["eps_r"] != np.ones((2, 3), dtype=np.float32).tolist()


def test_ad_memory_compiled_certificate_scope_failures_are_non_certifying():
    sim = _uniform_sim()

    missing = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(input_signature=None),
    )
    assert missing.status == "scope_incomplete"
    assert missing.is_valid_certificate is False
    assert missing.estimate_within_budget is None
    assert missing.memory_analysis_status == "complete"
    assert "input_signature" in missing.scope_status_reason

    non_json = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(scope_context={"bad": object()}),
    )
    assert non_json.status == "scope_incomplete"
    assert "unsupported value object" in non_json.scope_status_reason

    mismatch = sim.ad_memory_compiled_certificate(
        _FakeCompiled(scope={"n_steps": 999}),
        **_certificate_kwargs(),
    )
    assert mismatch.status == "scope_mismatch"
    assert mismatch.scope_status == "scope_mismatch"
    assert "n_steps" in mismatch.scope_status_reason

    non_string_key = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(input_signature={1: "not-exact-json"}),
    )
    assert non_string_key.status == "scope_incomplete"
    assert "non-string mapping key" in non_string_key.scope_status_reason

    colliding_key = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(input_signature={1: "a", "1": "b"}),
    )
    assert colliding_key.status == "scope_incomplete"
    assert "non-string mapping key" in colliding_key.scope_status_reason

    introspection_error = sim.ad_memory_compiled_certificate(
        _RaisingIntrospectionCompiled(),
        **_certificate_kwargs(),
    )
    assert introspection_error.status == "scope_mismatch"
    assert "introspection attribute" in introspection_error.scope_status_reason

    raising_to_dict = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(input_signature={"bad": _RaisingToDictAttribute()}),
    )
    assert raising_to_dict.status == "scope_incomplete"
    assert "to_dict attribute read raised RuntimeError" in raising_to_dict.scope_status_reason

    raising_shape = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(input_signature={"bad": _RaisingShapeAttribute()}),
    )
    assert raising_shape.status == "scope_incomplete"
    assert "shape attribute read raised RuntimeError" in raising_shape.scope_status_reason


def test_ad_memory_compiled_certificate_fails_closed_without_environment(monkeypatch):
    import rfx.api._compiled_memory as compiled_memory

    def fail_default_backend():
        raise RuntimeError("no backend")

    monkeypatch.setattr(compiled_memory.jax, "default_backend", fail_default_backend)
    report = _uniform_sim().ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(),
    )

    assert report.status == "scope_incomplete"
    assert report.scope_status == "scope_incomplete"
    assert "default_backend" in report.scope_status_reason
    assert report.scope_digest is None
    assert report.environment_digest is None


def test_ad_memory_compiled_certificate_fails_closed_without_devices(monkeypatch):
    import rfx.api._compiled_memory as compiled_memory

    def fail_local_devices():
        raise RuntimeError("no devices")

    monkeypatch.setattr(compiled_memory.jax, "local_devices", fail_local_devices)
    report = _uniform_sim().ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(),
    )

    assert report.status == "scope_incomplete"
    assert "local_devices" in report.scope_status_reason
    assert report.exact_scope is None

@pytest.mark.parametrize("version", ["unknown", None, "  "])
def test_ad_memory_compiled_certificate_fails_closed_on_invalid_jax_version(
    monkeypatch,
    version,
):
    import rfx.api._compiled_memory as compiled_memory

    monkeypatch.setattr(compiled_memory.jax, "__version__", version)
    report = _uniform_sim().ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(),
    )

    assert report.status == "scope_incomplete"
    assert "__version__" in report.scope_status_reason
    assert report.exact_scope is None


@pytest.mark.parametrize("backend", [None, "unknown", "  "])
def test_ad_memory_compiled_certificate_fails_closed_on_invalid_backend(
    monkeypatch,
    backend,
):
    import rfx.api._compiled_memory as compiled_memory

    monkeypatch.setattr(compiled_memory.jax, "default_backend", lambda: backend)
    report = _uniform_sim().ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(),
    )

    assert report.status == "scope_incomplete"
    assert "default_backend" in report.scope_status_reason
    assert report.exact_scope is None


@pytest.mark.parametrize(
    "device, reason",
    [
        (_FakeJaxDevice(platform=None), "JAX device platform"),
        (_FakeJaxDevice(platform="unknown"), "JAX device platform"),
        (_FakeJaxDevice(platform="  "), "JAX device platform"),
        (_FakeJaxDevice(device_kind=None), "JAX device_kind"),
        (_FakeJaxDevice(device_kind="unknown"), "JAX device_kind"),
        (_FakeJaxDevice(device_kind="  "), "JAX device_kind"),
        (_FakeJaxDevice(raise_platform=True), "metadata read failed"),
        (_FakeJaxDevice(raise_kind=True), "metadata read failed"),
    ],
)
def test_ad_memory_compiled_certificate_fails_closed_on_invalid_device_metadata(
    monkeypatch,
    device,
    reason,
):
    import rfx.api._compiled_memory as compiled_memory

    monkeypatch.setattr(compiled_memory.jax, "local_devices", lambda: (device,))
    report = _uniform_sim().ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(),
    )

    assert report.status == "scope_incomplete"
    assert report.scope_status == "scope_incomplete"
    assert reason in report.scope_status_reason
    assert report.exact_scope is None


@pytest.mark.parametrize(
    "compiled, reason",
    [
        (_NoMemoryAnalysis(), "no callable memory_analysis"),
        (_FakeCompiled(analysis=None), "returned None"),
        (_FakeCompiled(raises=RuntimeError("boom")), "raised RuntimeError"),
        (_RaisingMemoryAnalysisAttribute(), "attribute boom"),
    ],
)
def test_ad_memory_compiled_certificate_analysis_unavailable(compiled, reason):
    report = _uniform_sim().ad_memory_compiled_certificate(
        compiled,
        **_certificate_kwargs(),
    )

    assert report.status == "analysis_unavailable"
    assert report.scope_status == "complete"
    assert report.memory_analysis_status == "analysis_unavailable"
    assert reason in report.memory_analysis_status_reason
    assert report.is_valid_certificate is False
    assert report.compiler_reported_required_bytes is None
    assert report.exact_scope["memory_analysis_fields"] is None


@pytest.mark.parametrize(
    "analysis, reason",
    [
        (
            _FakeMemoryAnalysis(temp_size_in_bytes=_MISSING),
            "missing required byte field",
        ),
        (
            _FakeMemoryAnalysis(temp_size_in_bytes=-1),
            "must be non-negative",
        ),
        (
            _FakeMemoryAnalysis(temp_size_in_bytes=1.5),
            "non-negative integer byte count",
        ),
        (
            _FakeMemoryAnalysis(temp_size_in_bytes=float("nan")),
            "non-negative integer byte count",
        ),
        (
            _FakeMemoryAnalysis(temp_size_in_bytes="1"),
            "non-negative integer byte count",
        ),
        (
            _FakeMemoryAnalysis(
                temp_size_in_bytes=0,
                argument_size_in_bytes=0,
                output_size_in_bytes=0,
                alias_size_in_bytes=1,
            ),
            "computed required bytes must be non-negative",
        ),
        (
            _RaisingAnalysisField(),
            "reading temp_size_in_bytes raised RuntimeError",
        ),
    ],
)
def test_ad_memory_compiled_certificate_analysis_incomplete(analysis, reason):
    report = _uniform_sim().ad_memory_compiled_certificate(
        _FakeCompiled(analysis),
        **_certificate_kwargs(),
    )

    assert report.status == "analysis_incomplete"
    assert report.scope_status == "complete"
    assert report.memory_analysis_status == "analysis_incomplete"
    assert reason in report.memory_analysis_status_reason
    assert report.estimate_within_budget is None


def test_ad_memory_compiled_certificate_digests_are_stable_and_scope_sensitive():
    sim = _uniform_sim()

    first = sim.ad_memory_compiled_certificate(_FakeCompiled(), **_certificate_kwargs())
    second = sim.ad_memory_compiled_certificate(_FakeCompiled(), **_certificate_kwargs())
    changed = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(runner_or_objective="different_loss"),
    )

    assert first.scope_digest == second.scope_digest
    assert first.config_digest == second.config_digest
    assert first.environment_digest == second.environment_digest
    assert first.scope_digest != changed.scope_digest
    assert first.config_digest != changed.config_digest


def test_ad_memory_compiled_certificate_rejects_invalid_budget_scalars():
    sim = _uniform_sim()

    with pytest.raises(ValueError, match="available_memory_gb must be positive"):
        sim.ad_memory_compiled_certificate(
            _FakeCompiled(),
            **_certificate_kwargs(available_memory_gb=0.0),
        )
    with pytest.raises(ValueError, match="target_fraction must be positive"):
        sim.ad_memory_compiled_certificate(
            _FakeCompiled(),
            **_certificate_kwargs(target_fraction=0.0),
        )
    with pytest.raises(ValueError, match="interval"):
        sim.ad_memory_compiled_certificate(
            _FakeCompiled(),
            **_certificate_kwargs(target_fraction=1.1),
        )
    with pytest.raises(TypeError, match="finite real scalar"):
        sim.ad_memory_compiled_certificate(
            _FakeCompiled(),
            **_certificate_kwargs(available_memory_gb=[1.0]),
        )


def test_ad_memory_compiled_certificate_preflight_snapshot_and_mismatch():
    sim = _uniform_sim()
    preflight = sim.ad_memory_preflight(n_steps=120, available_memory_gb=1.0)
    report = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(preflight=preflight),
    )

    assert report.status == "compiler_estimate_within_budget"
    assert report.source_preflight["evidence_class"] == "composite_ad_memory_preflight"
    assert report.source_preflight["status"] == preflight.status
    _assert_no_forbidden_fields_recursive(preflight.to_dict())

    mismatched = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(preflight=preflight, available_memory_gb=2.0),
    )
    assert mismatched.status == "scope_mismatch"
    assert "available_memory_gb" in mismatched.scope_status_reason

    checkpoint_mismatched = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(preflight=preflight, checkpoint_segments=10),
    )
    assert checkpoint_mismatched.status == "scope_mismatch"
    assert "checkpoint mode" in checkpoint_mismatched.scope_status_reason

    malformed_preflight = sim.ad_memory_compiled_certificate(
        _FakeCompiled(),
        **_certificate_kwargs(preflight=_ListPreflight()),
    )
    assert malformed_preflight.status == "scope_incomplete"
    assert "preflight snapshot must be a JSON object mapping" in malformed_preflight.scope_status_reason


def test_current_memory_artifacts_keep_evidence_classes_separate():
    sim = _patch_like_sim()
    estimate = sim.estimate_ad_memory(n_steps=10_000, checkpoint_every=500)
    plan = sim.plan_ad_memory(n_steps=10_000, available_memory_gb=1.0)
    explanation = sim.explain_ad_memory(
        n_steps=10_000,
        checkpoint_every=500,
        available_memory_gb=1.0,
    )
    report = sim.mesh_intelligence_report(
        n_steps=10_000,
        checkpoint_every=plan.checkpoint_every,
        available_memory_gb=1.0,
    )

    estimate_artifact = estimate.to_dict()
    plan_artifact = plan.to_dict()
    report_artifact = report.to_dict()
    explanation_artifact = explanation.to_dict()

    assert estimate_artifact["evidence_class"] == "static_estimate"
    assert plan_artifact["evidence_class"] == "calibrated_conservative_plan"
    assert plan_artifact["selected_estimate"]["evidence_class"] == "static_estimate"
    assert report_artifact["ad_memory"]["evidence_class"] == "static_estimate"
    assert explanation_artifact["evidence_class"] == "static_ad_explainability"
    assert explanation_artifact["estimate"]["evidence_class"] == "static_estimate"

    _assert_no_forbidden_current_fields(estimate_artifact)
    _assert_no_forbidden_current_fields(plan_artifact)
    _assert_no_forbidden_current_fields(plan_artifact["selected_estimate"])
    _assert_no_forbidden_current_fields(report_artifact["ad_memory"])
    _assert_no_forbidden_current_fields(explanation_artifact["estimate"])


def test_explain_ad_memory_decomposes_selected_estimate():
    sim = _patch_like_sim()

    explanation = sim.explain_ad_memory(n_steps=10_000, checkpoint_every=500)
    artifact = explanation.to_dict()
    components = {component["name"]: component for component in artifact["components"]}

    assert explanation.strategy == "segmented_checkpoint_every"
    assert explanation.selected_memory_field == "ad_segmented_gb"
    assert explanation.selected_memory_gb == explanation.estimate.ad_segmented_gb
    # #277: at chunk=500 (well above the balanced ≈141) the live-chunk
    # rematerialization tape dominates the decomposition.
    assert explanation.dominant_component == "segmented_live_segment_tape"
    assert {
        "field_state",
        "material_auxiliary_state",
        "cpml_auxiliary_state",
        "segmented_boundary_field_tape",
        "segmented_live_segment_tape",
        "ntff_dft_state",
    } == set(components)
    assert components["segmented_boundary_field_tape"]["kind"] == "reverse_ad_saved_state"
    assert components["segmented_boundary_field_tape"]["count"] == (
        2 * explanation.estimate.ad_segmented_active_segments
    )
    assert components["segmented_live_segment_tape"]["kind"] == "reverse_ad_saved_state"
    assert components["segmented_live_segment_tape"]["count"] == 500
    assert components["segmented_boundary_field_tape"]["memory_gb"] > components["field_state"]["memory_gb"]
    assert sum(
        component["memory_gb"] for component in artifact["components"]
    ) == pytest.approx(explanation.selected_memory_gb)
    assert json.loads(explanation.to_json()) == artifact

def test_explain_ad_memory_covers_ntff_and_segmented_warmup_paths():
    baseline = _uniform_sim().explain_ad_memory(n_steps=120, checkpoint_segments=10)
    sim = _uniform_sim()
    sim.add_ntff_box(
        (1e-3, 1e-3, 1e-3),
        (9e-3, 9e-3, 4e-3),
        n_freqs=3,
    )

    segmented = sim.explain_ad_memory(
        n_steps=120,
        checkpoint_segments=10,
        n_warmup=61,
    )
    chunked = sim.explain_ad_memory(
        n_steps=120,
        checkpoint_every=12,
        n_warmup=61,
    )

    for explanation, strategy in (
        (segmented, "segmented_checkpoint_segments"),
        (chunked, "segmented_checkpoint_every"),
    ):
        artifact = explanation.to_dict()
        components = {component["name"]: component for component in artifact["components"]}

        assert explanation.strategy == strategy
        assert explanation.selected_memory_field == "ad_segmented_gb"
        assert explanation.estimate.ntff_dft_gb > baseline.estimate.ntff_dft_gb
        assert explanation.estimate.ad_segmented_active_segments == 5
        assert components["ntff_dft_state"]["memory_gb"] == pytest.approx(
            explanation.estimate.ntff_dft_gb
        )
        assert components["segmented_boundary_field_tape"]["count"] == (
            2 * explanation.estimate.ad_segmented_active_segments
        )
        # #277: both segmented paths carry the same 12-step live tape here
        # (segment length 120/10 on the uniform path, chunk 12 on the
        # scan-of-scan path).
        assert components["segmented_live_segment_tape"]["count"] == 12
        assert sum(
            component["memory_gb"] for component in artifact["components"]
        ) == pytest.approx(explanation.selected_memory_gb)


def test_explain_ad_memory_records_warmup_mask_and_full_tape():
    sim = _uniform_sim()
    design_mask = _design_mask(sim, fraction=0.5)

    explanation = sim.explain_ad_memory(
        n_steps=120,
        n_warmup=20,
        design_mask=design_mask,
    )
    artifact = explanation.to_dict()
    components = {component["name"]: component for component in artifact["components"]}

    assert explanation.strategy == "full_reverse_ad_static_tape"
    assert explanation.selected_memory_field == "ad_full_gb"
    assert components["full_reverse_field_tape"]["count"] == 100
    assert explanation.estimate.ad_active_design_fraction == pytest.approx(
        np.count_nonzero(design_mask) / design_mask.size
    )
    assert any("design_mask" in rec for rec in explanation.recommendations)
    assert sum(
        component["memory_gb"] for component in artifact["components"]
    ) == pytest.approx(explanation.selected_memory_gb)


def test_current_memory_recommendations_do_not_claim_guarantees():
    sim = _patch_like_sim()
    plan = sim.plan_ad_memory(n_steps=10_000, available_memory_gb=1.0)
    explanation = sim.explain_ad_memory(
        n_steps=10_000,
        checkpoint_every=plan.checkpoint_every,
        available_memory_gb=1.0,
    )
    report = sim.mesh_intelligence_report(
        n_steps=10_000,
        checkpoint_every=plan.checkpoint_every,
        available_memory_gb=1.0,
    )

    texts = [
        plan.recommendation,
        report.recommendation,
        sim.estimate_ad_memory(n_steps=10_000, available_memory_gb=1e-6).warning,
        *explanation.recommendations,
    ]
    for text in texts:
        assert text is not None
        lowered = text.lower()
        assert not any(term in lowered for term in _FORBIDDEN_RECOMMENDATION_TERMS)


def test_ad_memory_preflight_full_fit_branch():
    sim = _patch_like_sim()

    report = sim.ad_memory_preflight(
        n_steps=100,
        available_memory_gb=100.0,
        include_mesh_report=False,
    )
    artifact = report.to_dict()
    hint_codes = {hint.code for hint in report.action_hints}

    assert isinstance(report, ADMemoryPreflightReport)
    assert report.status == "full_ad_fits"
    assert report.supported_checkpoint_mode is None
    assert report.checkpoint_every is None
    assert report.checkpoint_segments is None
    assert report.full_ad_fits is True
    assert report.checkpointing_fits is False
    assert report.memory_plan.full_ad_fits is True
    assert report.explainability.selected_memory_field == "ad_full_gb"
    assert report.explainability.strategy == "full_reverse_ad_static_tape"
    assert "full_ad_fits" in hint_codes
    assert "use_checkpoint_every" not in hint_codes
    assert "use_checkpoint_segments" not in hint_codes
    _assert_validate_physics_hint(report)
    assert report.source_evidence_classes == (
        "calibrated_conservative_plan",
        "static_estimate",
        "static_ad_explainability",
    )
    assert artifact["mesh_report"] is None
    assert json.loads(report.to_json()) == artifact


def test_ad_memory_preflight_nonuniform_checkpoint_branch():
    sim = _patch_like_sim()

    # Budget raised 1.0 → 8.0 GB for #277 (see
    # test_plan_ad_memory_recommends_checkpoint_every_for_budget).
    report = sim.ad_memory_preflight(n_steps=10_000, available_memory_gb=8.0)
    hint = next(
        hint for hint in report.action_hints if hint.code == "use_checkpoint_every"
    )

    assert report.status == "checkpointing_fits"
    assert report.full_ad_fits is False
    assert report.checkpointing_fits is True
    assert report.supported_checkpoint_mode == "checkpoint_every"
    assert report.checkpoint_every == report.memory_plan.checkpoint_every
    assert report.checkpoint_segments is None
    assert report.explainability.selected_memory_field == "ad_segmented_gb"
    assert report.explainability.strategy == "segmented_checkpoint_every"
    assert hint.checkpoint_mode == "checkpoint_every"
    assert hint.checkpoint_every == report.checkpoint_every
    assert hint.checkpoint_segments is None
    assert hint.blocking is False
    assert report.mesh_report is not None
    assert report.mesh_report.ad_memory.checkpoint_every == report.checkpoint_every
    _assert_validate_physics_hint(report)


def test_ad_memory_preflight_uniform_checkpoint_branch():
    sim = _uniform_sim()

    # Budget raised 0.002 → 0.003 GB for #277 (see
    # test_plan_ad_memory_recommends_checkpoint_segments_for_uniform_budget).
    report = sim.ad_memory_preflight(
        n_steps=120,
        available_memory_gb=0.003,
        include_mesh_report=False,
    )
    hint = next(
        hint for hint in report.action_hints if hint.code == "use_checkpoint_segments"
    )

    assert report.status == "checkpointing_fits"
    assert report.checkpointing_fits is True
    assert report.supported_checkpoint_mode == "checkpoint_segments"
    assert report.checkpoint_every is None
    assert report.checkpoint_segments == report.memory_plan.checkpoint_segments == 10
    assert report.explainability.selected_memory_field == "ad_segmented_gb"
    assert report.explainability.strategy == "segmented_checkpoint_segments"
    assert hint.checkpoint_mode == "checkpoint_segments"
    assert hint.checkpoint_every is None
    assert hint.checkpoint_segments == 10
    _assert_validate_physics_hint(report)


def test_ad_memory_preflight_unfit_branch_is_diagnostic_only():
    sim = _patch_like_sim()

    report = sim.ad_memory_preflight(
        n_steps=10_000,
        available_memory_gb=0.001,
        include_mesh_report=False,
    )
    blocker = next(
        hint for hint in report.action_hints if hint.code == "memory_budget_unfit"
    )

    assert report.status == "does_not_fit"
    assert report.full_ad_fits is False
    assert report.checkpointing_fits is False
    assert report.supported_checkpoint_mode == "checkpoint_every"
    # #277: the diagnostic candidate is the least-memory chunk, not
    # checkpoint_every=n_steps (see test_plan_ad_memory_reports_unfit_budget).
    assert report.checkpoint_every == 137
    assert report.checkpoint_segments is None
    assert report.explainability.selected_memory_field == "ad_segmented_gb"
    assert blocker.blocking is True
    assert blocker.severity == "blocker"
    assert "diagnostic-only" in report.recommendation
    assert "do not launch" in report.recommendation.lower()
    assert "as a fit" in blocker.action
    assert not any(
        hint.code in {"full_ad_fits", "use_checkpoint_every", "use_checkpoint_segments"}
        for hint in report.action_hints
    )
    _assert_validate_physics_hint(report)


def test_ad_memory_preflight_boundaries_and_action_text_are_safe():
    patch = _patch_like_sim()
    uniform = _uniform_sim()
    reports = [
        patch.ad_memory_preflight(
            n_steps=100,
            available_memory_gb=100.0,
            include_mesh_report=False,
        ),
        patch.ad_memory_preflight(
            n_steps=10_000,
            available_memory_gb=1.0,
            include_mesh_report=False,
        ),
        uniform.ad_memory_preflight(
            n_steps=120,
            available_memory_gb=0.002,
            include_mesh_report=False,
        ),
        patch.ad_memory_preflight(
            n_steps=10_000,
            available_memory_gb=0.001,
            include_mesh_report=False,
        ),
    ]

    for report in reports:
        artifact = report.to_dict()
        assert report.evidence_boundaries == AD_MEMORY_PREFLIGHT_BOUNDARIES
        assert artifact["evidence_boundaries"] == list(AD_MEMORY_PREFLIGHT_BOUNDARIES)
        _assert_no_forbidden_fields_recursive(artifact)

        action_texts = [
            report.recommendation,
            *(
                f"{hint.message} {hint.action}"
                for hint in report.action_hints
            ),
        ]
        for text in action_texts:
            lowered = text.lower()
            assert not any(
                term in lowered for term in _FORBIDDEN_RECOMMENDATION_TERMS
            )

def test_memory_reduction_docs_separate_planning_from_certificate_evidence():
    doc = Path("docs/public/guide/memory-reduction.mdx").read_text()
    assert "`static_estimate`" in doc
    assert "`calibrated_conservative_plan`" in doc
    assert "`static_ad_explainability`" in doc
    assert "`composite_ad_memory_preflight`" in doc
    assert "`static_action_hint`" in doc
    assert "explain_ad_memory(...)" in doc
    assert "ad_memory_preflight(...)" in doc
    assert "residual_context" in doc
    assert "checkpoint_every" in doc
    assert "checkpoint_segments" in doc
    assert "**checkpoint_kwargs" in doc
    assert "`bounded_certificate`" in doc
    assert "Current `estimate_ad_memory(...)`, `plan_ad_memory(...)`, and `explain_ad_memory(...)` artifacts are not certificates" in doc
    assert "a runtime peak-memory guarantee" in doc
    assert "ad_memory_compiled_certificate(...)" in doc
    assert "`bounded_certificate` | yes" in doc
    assert "Compiled.memory_analysis()" in doc
    assert "temp + argument + output - alias" in doc
    assert "audit identities" in doc
    assert "version/backend dependent and may be unavailable" in doc
    assert "not full array values" in doc
    for boundary in AD_MEMORY_PREFLIGHT_BOUNDARIES:
        assert boundary in doc


# Calibration rows from VESSL 369367233509 (inverse-design gradient smoke,
# ~603k-cell FR4 patch — see module docstring). The pre-#277 boundary-only
# formula fails the 0.5x floor at the (64, 2000) row (0.39x observed);
# these rows are what discriminate the corrected model.
@pytest.mark.parametrize("chunk,n_steps,observed_gb", [
    (100, 10_000, 5.84),
    (64, 2_000, 2.56),
])
def test_segmented_estimate_within_tolerance(chunk, n_steps, observed_gb):
    sim = _patch_like_sim()
    est = sim.estimate_ad_memory(n_steps=n_steps, checkpoint_every=chunk)
    assert est.ad_segmented_gb is not None
    pred = est.ad_segmented_gb
    # Predictions should be within 2x (very loose to tolerate XLA
    # allocator slack); typical is ~1.3x below the observed gradient peak.
    assert 0.5 * observed_gb <= pred <= 2.0 * observed_gb, (
        f"chunk={chunk}: predicted {pred:.3f} GB vs observed {observed_gb} GB"
    )
    assert pred * AD_MEMORY_FIT_SAFETY_FACTOR >= observed_gb


def test_checkpoint_every_none_leaves_segmented_null():
    sim = _patch_like_sim()
    est = sim.estimate_ad_memory(n_steps=1000)
    assert est.ad_segmented_gb is None
    assert est.checkpoint_every is None


def test_u_shaped_in_chunk_with_minimum_near_balanced_size():
    """#277: the segmented estimate is U-shaped in the chunk size.

    Boundary storage (2 · n_segments) shrinks with bigger chunks while the
    live-chunk tape grows with them, so the estimate is minimized where the
    terms balance, near chunk ≈ sqrt(2 · n_steps) (~141 for n_steps=10000)
    — it is NOT monotone decreasing in the chunk size (the pre-#277
    boundary-only model was).
    """
    sim = _patch_like_sim()
    gbs = [
        sim.estimate_ad_memory(n_steps=10000, checkpoint_every=c).ad_segmented_gb
        for c in [50, 100, 141, 500, 1000]
    ]
    # Decreasing toward the balanced chunk...
    assert gbs[0] > gbs[1] > gbs[2], gbs
    # ...then increasing beyond it as the live-chunk tape dominates.
    assert gbs[2] < gbs[3] < gbs[4], gbs


def test_live_segment_term_invariance_witness_277():
    """#277 invariance witness: exact, hand-checkable component accounting.

    n_steps=100 on the uniform sim. Boundary storage is 2·K field states;
    the live segment adds n_steps//K more. At K far below sqrt(n_steps)
    the pre-#277 estimate missed almost the entire peak: the omitted live
    tape is (n_steps//K)/(2·K) times the boundary term it kept — 12.5x at
    K=2 here, and ~389x in the issue #277 λ/100 desk ladder (n_steps=6999,
    checkpoint_segments=3 → live/boundary = 2333/6, the ~272x total
    under-count after the forward-state offset). At K = sqrt(n_steps) the
    two terms are the same order (0.5x), which is why the omission was
    invisible in balanced-sqrt(N) usage.
    """
    sim = _uniform_sim()
    accounting = sim._ad_memory_static_accounting()
    field_bytes = accounting["field_bytes"]
    overhead = accounting["forward_bytes"] + accounting["ntff_bytes"]
    to_gb = 1.0 / 1e9

    est_2 = sim.estimate_ad_memory(n_steps=100, checkpoint_segments=2)
    est_10 = sim.estimate_ad_memory(n_steps=100, checkpoint_segments=10)

    # Exact corrected-formula bytes: (2·K + n_steps//K)·field + overhead.
    expected_2 = (2 * 2 + 50) * field_bytes + overhead
    expected_10 = (2 * 10 + 10) * field_bytes + overhead
    assert est_2.ad_segmented_gb == expected_2 * to_gb
    assert est_10.ad_segmented_gb == expected_10 * to_gb

    # The delta versus the pre-#277 formula (2·K·field + overhead) is
    # exactly one segment of per-step field tape.
    old_2 = 2 * 2 * field_bytes + overhead
    old_10 = 2 * 10 * field_bytes + overhead
    assert expected_2 - old_2 == 50 * field_bytes
    assert expected_10 - old_10 == 10 * field_bytes

    # Ratio demonstration: omitted-live-tape / kept-boundary-term =
    # (n_steps//K) / (2·K).
    assert (50 * field_bytes) / (2 * 2 * field_bytes) == 12.5
    assert (10 * field_bytes) / (2 * 10 * field_bytes) == 0.5

    # Fewer segments no longer implies less memory: K=2 must now exceed
    # K=10 (under the boundary-only model K=2 looked "cheaper").
    assert est_2.ad_segmented_gb > est_10.ad_segmented_gb

    # checkpoint_every mirror: a 50-step chunk carries the same live tape
    # as a 50-step segment, byte-for-byte.
    est_chunk = sim.estimate_ad_memory(n_steps=100, checkpoint_every=50)
    assert est_chunk.ad_segmented_gb == est_2.ad_segmented_gb


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

    # Budget raised 1.0 → 8.0 GB for #277: with the live-chunk tape counted,
    # the least segmented memory for this case is ≈4.3 GB (balanced chunk),
    # so 1.0 GB is genuinely unfit — the old budget only fit under the
    # boundary-only under-count.
    plan = sim.plan_ad_memory(n_steps=10_000, available_memory_gb=8.0)

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
            available_memory_gb=8.0,
            checkpoint_every=plan.checkpoint_every - 1,
        )
        assert previous.ad_segmented_gb * plan.fit_safety_factor > plan.target_memory_gb


def test_plan_ad_memory_checkpoint_every_handles_unaligned_warmup():
    sim = _patch_like_sim()

    # Budget 0.35 → 0.7 GB and expected chunk 12 → 8 for #277: the plan now
    # also carries the live-chunk tape, so the first fitting chunk shrinks
    # and the old budget no longer fits any chunk.
    plan = sim.plan_ad_memory(
        n_steps=120,
        available_memory_gb=0.7,
        n_warmup=49,
    )

    assert plan.segmented_fits is True
    assert plan.checkpoint_mode == "checkpoint_every"
    assert plan.checkpoint_every == 8
    previous = sim.estimate_ad_memory(
        n_steps=120,
        available_memory_gb=0.7,
        checkpoint_every=7,
        n_warmup=49,
    )
    assert previous.ad_segmented_gb * plan.fit_safety_factor > plan.target_memory_gb
    assert plan.selected_estimate.ad_segmented_gb * plan.fit_safety_factor <= plan.target_memory_gb


def test_plan_ad_memory_reports_unfit_budget():
    sim = _patch_like_sim()

    plan = sim.plan_ad_memory(n_steps=10_000, available_memory_gb=0.001)

    assert plan.full_ad_fits is False
    assert plan.segmented_fits is False
    # #277: the diagnostic candidate is the least-memory chunk (boundary and
    # live-chunk terms balanced near sqrt(2 · n_steps)), not
    # checkpoint_every=n_steps, which is ~full-AD-sized under the corrected
    # model.
    assert plan.checkpoint_every == 137
    assert plan.selected_estimate.ad_segmented_gb > plan.target_memory_gb
    assert plan.checkpoint_segments is None
    assert plan.checkpoint_mode == "checkpoint_every"
    assert "least-memory candidate" in plan.recommendation
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
    # #277: the diagnostic candidate is the least-memory divisor of n_steps
    # (K=8 balances 2·K boundary states against 120/K live-tape steps), not
    # checkpoint_segments=1, which is ~full-AD-sized under the corrected
    # model.
    assert plan.checkpoint_segments == 8
    assert plan.checkpoint_mode == "checkpoint_segments"
    assert plan.selected_estimate.ad_segmented_gb > plan.target_memory_gb
    assert "least-memory candidate" in plan.recommendation
    assert "checkpoint_segments=8" in plan.recommendation
    assert "reduce mesh size" in plan.recommendation


def test_plan_ad_memory_uses_valid_custom_target_fraction():
    sim = _patch_like_sim()

    # Budget raised 1.0 → 16.0 GB for #277 (least segmented memory for this
    # case is ≈4.3 GB once the live-chunk tape is counted).
    plan = sim.plan_ad_memory(
        n_steps=10_000,
        available_memory_gb=16.0,
        target_fraction=0.5,
    )

    assert plan.target_fraction == 0.5
    assert plan.target_memory_gb == 8.0
    assert plan.selected_estimate.ad_segmented_gb * plan.fit_safety_factor <= plan.target_memory_gb
    if plan.checkpoint_every > 1:
        previous = sim.estimate_ad_memory(
            n_steps=10_000,
            available_memory_gb=16.0,
            checkpoint_every=plan.checkpoint_every - 1,
        )
        assert previous.ad_segmented_gb * plan.fit_safety_factor > plan.target_memory_gb


def test_plan_ad_memory_serializes_artifact():
    sim = _patch_like_sim()

    # Budget raised 1.0 → 8.0 GB for #277 so the serialized artifact still
    # exercises the segmented-fit branch.
    plan = sim.plan_ad_memory(n_steps=10_000, available_memory_gb=8.0)
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
    assert unaligned.ad_segmented_active_segments == 3
    assert three_active_segments.ad_segmented_active_segments == 3
    assert boundary.ad_active_steps == 70
    assert boundary.ad_segmented_active_segments == 2
    assert two_active_segments.ad_segmented_active_segments == 2

    # #277: totals across the two knobs are no longer equal at matching
    # active-segment counts, because the live rematerialization tapes
    # differ — a 50-step chunk replays 50 steps, while a 120/3=40-step
    # (or 120/2=60-step) segment replays its own length. Assert the exact
    # (2·active_segments + live_tape_steps)·field + forward + ntff
    # composition for each instead.
    accounting = sim._ad_memory_static_accounting()
    field_bytes = accounting["field_bytes"]
    overhead = accounting["forward_bytes"] + accounting["ntff_bytes"]
    assert unaligned.ad_segmented_gb * 1e9 == pytest.approx(
        (2 * 3 + 50) * field_bytes + overhead
    )
    assert three_active_segments.ad_segmented_gb * 1e9 == pytest.approx(
        (2 * 3 + 40) * field_bytes + overhead
    )
    assert boundary.ad_segmented_gb * 1e9 == pytest.approx(
        (2 * 2 + 50) * field_bytes + overhead
    )
    assert two_active_segments.ad_segmented_gb * 1e9 == pytest.approx(
        (2 * 2 + 60) * field_bytes + overhead
    )


def test_plan_ad_memory_recommends_checkpoint_segments_for_uniform_budget():
    sim = _uniform_sim()

    # Budget raised 0.002 → 0.003 GB for #277: K=10 now also carries the
    # 12-step live-segment tape (≈1.87 MB total, ≈2.43 MB with safety).
    plan = sim.plan_ad_memory(n_steps=120, available_memory_gb=0.003)

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

    # #277 direction-aware advice: when the live-segment tape dominates
    # the boundary term, the fix is to move TOWARD sqrt(n_steps), not
    # away from it.
    live_dominated_segments = sim.estimate_ad_memory(
        n_steps=120,
        available_memory_gb=1e-6,
        checkpoint_segments=2,
    )
    assert live_dominated_segments.warning is not None
    assert "Increase checkpoint_segments" in live_dominated_segments.warning

    live_dominated_chunk = sim.estimate_ad_memory(
        n_steps=120,
        available_memory_gb=1e-6,
        checkpoint_every=60,
    )
    assert live_dominated_chunk.warning is not None
    assert "Reduce checkpoint_every" in live_dominated_chunk.warning

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

    # Float round-trip headroom: gb / 0.85 * 0.85 does not round-trip
    # exactly for every float value (observed gap ~0.5 ulp), and whether it
    # lands above or below is value luck. The intent is "no warning
    # at/above the boundary", so nudge the availability by 1e-12 relative —
    # orders of magnitude above the round-trip gap, physically negligible.
    segmented_baseline = sim.estimate_ad_memory(n_steps=120, checkpoint_segments=10)
    segmented_available = segmented_baseline.ad_segmented_gb / 0.85 * (1 + 1e-12)
    segmented_exact = sim.estimate_ad_memory(
        n_steps=120,
        checkpoint_segments=10,
        available_memory_gb=segmented_available,
    )
    assert segmented_exact.warning is None

    chunked_baseline = sim.estimate_ad_memory(n_steps=120, checkpoint_every=12)
    chunked_available = chunked_baseline.ad_segmented_gb / 0.85 * (1 + 1e-12)
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
