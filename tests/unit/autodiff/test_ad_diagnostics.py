import json
from pathlib import Path


import jax.ad_checkpoint as ad_checkpoint
import jax.numpy as jnp
import numpy as np
import pytest

import rfx
from rfx.ad_diagnostics import (
    ADParserHealth,
    ADResidualGroup,
    ADResidualInspection,
    ADResidualRecord,
    ADSavedResidualDiagnosticReport,
    diagnose_ad_saved_residuals,
    inspect_ad_saved_residuals,
    parse_saved_residual_line,
)


def test_parse_saved_residual_line_preserves_named_record():
    record = parse_saved_residual_line("f32[2,3] named 'sin_x' from demo.py:10 (loss)")

    assert record.aval == "f32[2,3]"
    assert record.dtype == "f32"
    assert record.shape == (2, 3)
    assert record.size == 6
    assert record.estimated_bytes == 24
    assert record.source_kind == "named"
    assert record.name == "sin_x"
    assert record.to_dict()["shape"] == [2, 3]

def test_parse_saved_residual_line_handles_scalar_and_unknown_dtype():
    scalar = parse_saved_residual_line("f32[] from the argument x")
    unknown = parse_saved_residual_line("token[2] output of custom from demo.py:1 (f)")

    assert scalar.shape == ()
    assert scalar.size == 1
    assert scalar.estimated_bytes == 4
    assert unknown.dtype == "token"
    assert unknown.shape == (2,)
    assert unknown.size == 2
    assert unknown.estimated_bytes is None


def test_parse_saved_residual_line_handles_unknown_future_format():
    record = parse_saved_residual_line("future-format without parsed aval")

    assert record.raw_line == "future-format without parsed aval"
    assert record.dtype is None
    assert record.shape is None
    assert record.estimated_bytes is None
    assert record.source_kind == "unknown"


def test_inspect_ad_saved_residuals_returns_structured_static_artifact():
    def loss(x, scale=1.0):
        named = ad_checkpoint.checkpoint_name(jnp.sin(x * scale), "sin_x")
        return jnp.sum(named * jnp.cos(x))

    inspection = inspect_ad_saved_residuals(loss, jnp.ones((2, 3)), scale=2.0)
    artifact = inspection.to_dict()

    assert inspection.evidence_class == "jax_saved_residuals_inspection"
    assert artifact["evidence_class"] == "jax_saved_residuals_inspection"
    assert inspection.records
    assert any(record.name == "sin_x" for record in inspection.records)
    assert any(record.source_kind == "argument" for record in inspection.records)
    assert all(record.raw_line for record in inspection.records)
    assert inspection.unknown_estimate_count == 0
    known_bytes = [
        record.estimated_bytes
        for record in inspection.records
        if record.estimated_bytes is not None
    ]
    assert inspection.total_estimated_bytes == sum(known_bytes)
    assert inspection.total_estimated_gb == pytest.approx(
        inspection.total_estimated_bytes / 1e9
    )
    assert json.loads(inspection.to_json()) == artifact


def test_inspect_ad_saved_residuals_public_export_identity():
    assert rfx.ADResidualInspection is ADResidualInspection
    assert rfx.ADResidualRecord is ADResidualRecord
    assert rfx.inspect_ad_saved_residuals is inspect_ad_saved_residuals
    assert rfx.parse_saved_residual_line is parse_saved_residual_line
    assert rfx.ADParserHealth is ADParserHealth
    assert rfx.ADResidualGroup is ADResidualGroup
    assert rfx.ADSavedResidualDiagnosticReport is ADSavedResidualDiagnosticReport
    assert rfx.diagnose_ad_saved_residuals is diagnose_ad_saved_residuals


def test_ad_residual_inspection_rejects_nonfinite_json():
    bad = ADResidualInspection(
        records=(),
        raw_lines=(),
        total_estimated_bytes=1,
        total_estimated_gb=float("nan"),
        unknown_estimate_count=0,
    )

    with pytest.raises(ValueError, match="Out of range"):
        bad.to_json()
    with pytest.raises(ValueError, match="Out of range"):
        bad.to_json(allow_nan=True)

def test_diagnose_ad_saved_residuals_returns_actionable_report():
    def loss(x, scale=1.0):
        named = ad_checkpoint.checkpoint_name(jnp.sin(x * scale), "sin_x")
        return jnp.sum(named * jnp.cos(x))

    report = diagnose_ad_saved_residuals(
        loss,
        jnp.ones((2, 3)),
        scale=2.0,
        top_n=2,
        workflow="toy-loss",
        context={"n_steps": 12},
        artifacts={"plan": {"evidence_class": "calibrated_conservative_plan"}},
    )
    artifact = report.to_dict()

    assert report.evidence_class == "rfx_ad_saved_residuals_diagnostic"
    assert report.source_evidence_class == "jax_saved_residuals_inspection"
    assert report.workflow == "toy-loss"
    assert report.context == {"n_steps": 12}
    assert artifact["artifact_snapshots"]["plan"]["evidence_class"] == "calibrated_conservative_plan"
    assert report.top_residuals
    assert len(report.top_residuals) <= 2
    assert report.top_residuals[0].estimated_bytes >= report.top_residuals[-1].estimated_bytes
    assert report.groups
    assert report.parser_health.record_count == len(report.inspection.records)
    assert report.parser_health.unknown_count == report.inspection.unknown_estimate_count
    assert report.known_estimated_bytes == sum(
        record.estimated_bytes or 0 for record in report.inspection.records
    )
    assert report.known_only_bytes == report.known_estimated_bytes
    assert artifact["known_only_bytes"] == report.known_estimated_bytes
    assert report.total_estimated_bytes == report.inspection.total_estimated_bytes
    assert report.jax_version
    assert any("trace-time JAX saved-residual" in rec for rec in report.recommendations)
    assert any("residual bytes do not prove budget fit" in rec for rec in report.recommendations)
    assert any("separate gradient checks" in rec for rec in report.recommendations)
    assert json.loads(report.to_json()) == artifact


def test_diagnostic_report_handles_unknown_lines_without_fake_totals():
    inspection = ADResidualInspection(
        records=(
            ADResidualRecord(
                aval="f32[2]",
                source="from the argument x",
                dtype="f32",
                shape=(2,),
                size=2,
                estimated_bytes=8,
                source_kind="argument",
                raw_line="f32[2] from the argument x",
                line_index=0,
            ),
            ADResidualRecord(
                aval="token[2]",
                source="output of custom",
                dtype="token",
                shape=(2,),
                size=2,
                estimated_bytes=None,
                source_kind="intermediate",
                raw_line="token[2] output of custom",
                line_index=1,
            ),
        ),
        raw_lines=("f32[2] from the argument x", "token[2] output of custom"),
        total_estimated_bytes=None,
        total_estimated_gb=None,
        unknown_estimate_count=1,
    )
    health = ADParserHealth(
        record_count=2,
        known_count=1,
        unknown_count=1,
        unknown_fraction=0.5,
        warnings=("partial",),
    )
    group = ADResidualGroup(
        key="argument|dtype=f32",
        source_kind="argument",
        name=None,
        dtype="f32",
        record_count=1,
        known_count=1,
        unknown_count=0,
        known_estimated_bytes=8,
        line_indices=(0,),
    )
    report = ADSavedResidualDiagnosticReport(
        inspection=inspection,
        top_residuals=(inspection.records[0],),
        groups=(group,),
        parser_health=health,
        known_estimated_bytes=8,
        known_estimated_gb=8e-9,
        total_estimated_bytes=None,
        total_estimated_gb=None,
        jax_version="test",
        workflow=None,
        context=None,
        artifact_snapshots={},
        recommendations=("Inspect raw_lines before comparing byte totals.",),
    )
    artifact = report.to_dict()

    assert artifact["known_estimated_bytes"] == 8
    assert artifact["total_estimated_bytes"] is None
    assert artifact["parser_health"]["unknown_count"] == 1
    assert artifact["top_residuals"][0]["line_index"] == 0
    assert json.loads(report.to_json()) == artifact


def test_diagnostic_report_rejects_nonfinite_json():
    report = ADSavedResidualDiagnosticReport(
        inspection=ADResidualInspection(
            records=(),
            raw_lines=(),
            total_estimated_bytes=None,
            total_estimated_gb=None,
            unknown_estimate_count=0,
        ),
        top_residuals=(),
        groups=(),
        parser_health=ADParserHealth(
            record_count=0,
            known_count=0,
            unknown_count=0,
            unknown_fraction=float("nan"),
            warnings=(),
        ),
        known_estimated_bytes=0,
        known_estimated_gb=0.0,
        total_estimated_bytes=None,
        total_estimated_gb=None,
        jax_version="test",
        workflow=None,
        context=None,
        artifact_snapshots={},
        recommendations=(),
    )

    with pytest.raises(ValueError, match="Out of range"):
        report.to_json()
    with pytest.raises(ValueError, match="Out of range"):
        report.to_json(allow_nan=True)


def _small_preflight_sim() -> rfx.Simulation:
    sim = rfx.Simulation(
        freq_max=10e9,
        domain=(10e-3, 10e-3, 5e-3),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=2,
    )
    sim.add_source((5e-3, 5e-3, 2e-3), "ez")
    sim.add_probe((5e-3, 5e-3, 3e-3), "ez")
    return sim


def _toy_residual_loss(x, *, scale=1.0):
    named = ad_checkpoint.checkpoint_name(jnp.sin(x * scale), "preflight_sin_x")
    return jnp.sum(named * jnp.cos(x))


class _ValidDictContext:
    def to_dict(self):
        return {"kind": "dict", "values": np.array([1, 2], dtype=np.int32)}


class _ValidJsonContext:
    def to_json(self):
        return json.dumps({"kind": "json", "values": [3, 4]})


class _BadDictContext:
    def to_dict(self):
        return {"bad": object()}


class _BadJsonContext:
    def to_json(self):
        return json.dumps({"bad": float("nan")})


def test_ad_memory_preflight_composes_residual_diagnostic():
    sim = _small_preflight_sim()

    report = sim.ad_memory_preflight(
        n_steps=120,
        available_memory_gb=0.002,
        residual_fun=_toy_residual_loss,
        residual_args=(jnp.ones((2, 3)),),
        residual_kwargs={"scale": 2.0},
        residual_top_n=2,
        residual_workflow="preflight-toy-loss",
        residual_context={
            "case": "toy",
            "shape": np.array([2, 3], dtype=np.int32),
            "scale": np.float32(2.0),
            "jax_array": jnp.array([1.0, 2.0]),
            "jax_scalar": jnp.array(3.0),
            "nested": [{"values": (jnp.array(4.0), np.array([5, 6]))}],
            "dict_obj": _ValidDictContext(),
            "json_obj": _ValidJsonContext(),
        },
    )
    diagnostic = report.residual_diagnostic
    artifact = report.to_dict()
    diagnostic_artifact = artifact["residual_diagnostic"]

    assert diagnostic is not None
    assert diagnostic.evidence_class == "rfx_ad_saved_residuals_diagnostic"
    assert diagnostic.source_evidence_class == "jax_saved_residuals_inspection"
    assert report.source_evidence_classes == (
        "calibrated_conservative_plan",
        "static_estimate",
        "static_ad_explainability",
        "jax_saved_residuals_inspection",
        "rfx_ad_saved_residuals_diagnostic",
    )
    assert diagnostic.workflow == "preflight-toy-loss"
    assert diagnostic.context["case"] == "toy"
    assert diagnostic.context["shape"] == [2, 3]
    assert diagnostic.context["scale"] == pytest.approx(2.0)
    assert diagnostic.context["jax_array"] == [1.0, 2.0]
    assert diagnostic.context["jax_scalar"] == pytest.approx(3.0)
    assert diagnostic.context["nested"] == [{"values": [4.0, [5, 6]]}]
    assert diagnostic.context["dict_obj"] == {"kind": "dict", "values": [1, 2]}
    assert diagnostic.context["json_obj"] == {"kind": "json", "values": [3, 4]}
    assert diagnostic.context["n_steps"] == report.n_steps
    assert diagnostic.context["available_memory_gb"] == report.available_memory_gb
    assert diagnostic.context["target_fraction"] == report.target_fraction
    assert diagnostic.context["target_memory_gb"] == report.target_memory_gb
    assert diagnostic.context["fit_safety_factor"] == report.fit_safety_factor
    assert diagnostic.context["preflight_status"] == report.status
    assert diagnostic.context["checkpoint_mode"] == report.supported_checkpoint_mode
    assert diagnostic.context["checkpoint_every"] == report.checkpoint_every
    assert diagnostic.context["checkpoint_segments"] == report.checkpoint_segments
    assert diagnostic_artifact["evidence_class"] == "rfx_ad_saved_residuals_diagnostic"
    assert diagnostic_artifact["source_evidence_class"] == "jax_saved_residuals_inspection"
    assert diagnostic_artifact["artifact_snapshots"]["memory_plan"]["evidence_class"] == "calibrated_conservative_plan"
    assert diagnostic_artifact["artifact_snapshots"]["explainability"]["evidence_class"] == "static_ad_explainability"
    assert diagnostic_artifact["artifact_snapshots"]["mesh_report"]["ad_memory"]["evidence_class"] == "static_estimate"
    assert any(hint.code == "inspect_saved_residuals" for hint in report.action_hints)
    assert any(
        hint.code == "validate_physics_separately" and not hint.blocking
        for hint in report.action_hints
    )
    assert json.loads(report.to_json()) == artifact


def test_ad_memory_preflight_propagates_residual_trace_errors():
    sim = _small_preflight_sim()

    def bad_loss(x):
        raise RuntimeError("trace exploded")

    with pytest.raises(RuntimeError, match="trace exploded"):
        sim.ad_memory_preflight(
            n_steps=120,
            available_memory_gb=0.002,
            include_mesh_report=False,
            residual_fun=bad_loss,
            residual_args=(jnp.ones((2, 3)),),
        )


def test_ad_memory_preflight_residual_context_rejects_unsupported_values():
    sim = _small_preflight_sim()

    with pytest.raises(TypeError, match="residual_context"):
        sim.ad_memory_preflight(
            n_steps=120,
            available_memory_gb=0.002,
            include_mesh_report=False,
            residual_fun=_toy_residual_loss,
            residual_args=(jnp.ones((2, 3)),),
            residual_context={"bad": object()},
        )

    with pytest.raises(TypeError, match="residual_context"):
        sim.ad_memory_preflight(
            n_steps=120,
            available_memory_gb=0.002,
            include_mesh_report=False,
            residual_context={"bad": object()},
        )

    with pytest.raises(TypeError, match="residual_context"):
        sim.ad_memory_preflight(
            n_steps=120,
            available_memory_gb=0.002,
            include_mesh_report=False,
            residual_context={"bad": _BadDictContext()},
        )


def test_ad_memory_preflight_residual_context_rejects_nonfinite_values():
    sim = _small_preflight_sim()

    with pytest.raises(ValueError, match="residual_context.*non-finite JSON|non-finite JSON.*residual_context"):
        sim.ad_memory_preflight(
            n_steps=120,
            available_memory_gb=0.002,
            include_mesh_report=False,
            residual_fun=_toy_residual_loss,
            residual_args=(jnp.ones((2, 3)),),
            residual_context={"bad": float("nan")},
        )

    with pytest.raises(ValueError, match="residual_context.*non-finite JSON|non-finite JSON.*residual_context"):
        sim.ad_memory_preflight(
            n_steps=120,
            available_memory_gb=0.002,
            include_mesh_report=False,
            residual_context={"bad": float("nan")},
        )

    with pytest.raises(ValueError, match="residual_context.*non-finite JSON|non-finite JSON.*residual_context"):
        sim.ad_memory_preflight(
            n_steps=120,
            available_memory_gb=0.002,
            include_mesh_report=False,
            residual_context={"bad": np.array([np.nan])},
        )

    with pytest.raises(ValueError, match="residual_context.*non-finite JSON|non-finite JSON.*residual_context"):
        sim.ad_memory_preflight(
            n_steps=120,
            available_memory_gb=0.002,
            include_mesh_report=False,
            residual_context={"bad": jnp.array([jnp.inf])},
        )

    with pytest.raises(ValueError, match="residual_context.*non-finite JSON|non-finite JSON.*residual_context"):
        sim.ad_memory_preflight(
            n_steps=120,
            available_memory_gb=0.002,
            include_mesh_report=False,
            residual_context={"bad": _BadJsonContext()},
        )

def test_memory_reduction_docs_include_residual_inspection_boundary():
    doc = Path("docs/public/guide/memory-reduction.mdx").read_text()

    assert "`jax_saved_residuals_inspection`" in doc
    assert "`rfx_ad_saved_residuals_diagnostic`" in doc
    assert "diagnose_ad_saved_residuals(...)" in doc
    assert "inspect_ad_saved_residuals(...)" in doc
    assert "not profiling" in doc
    assert "not an XLA memory report" in doc or "not XLA memory analysis" in doc
    assert "not a certificate" in doc
    assert "not RF validation" in doc
