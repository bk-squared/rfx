"""Manifest-backed batch provenance and resume tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.batch import (
    BATCH_MANIFEST_SCHEMA,
    ParameterSweep,
    case_id_from_params,
    run_batch_with_manifest,
)


class _FakeSimulation:
    def __init__(self, value: float, calls: list[float]):
        self.value = value
        self.calls = calls

    def run(self, **kwargs):
        self.calls.append(self.value)
        return type("R", (), {"value": self.value, "kwargs": kwargs})()


def _write_metric_artifact(case_dir: Path, params, result):
    path = case_dir / "metrics.json"
    path.write_text(json.dumps({"value": float(result.value)}))
    # Return a case-local path.  The manifest runner should normalize this to
    # an output-root-relative path so resume validation works across processes.
    return {"metrics": "metrics.json"}


def test_case_id_from_params_is_stable_across_dict_order():
    assert case_id_from_params({"b": 2.0, "a": 1.0}) == case_id_from_params({"a": 1.0, "b": 2.0})


def test_parameter_sweep_preserves_typed_values():
    sweep = ParameterSweep(material=["air", "pec"], enabled=[True, False], n=[3])

    combos = list(sweep.combinations())

    assert combos == [
        {"material": "air", "enabled": True, "n": 3},
        {"material": "air", "enabled": False, "n": 3},
        {"material": "pec", "enabled": True, "n": 3},
        {"material": "pec", "enabled": False, "n": 3},
    ]
    assert all(isinstance(c["material"], str) for c in combos)
    assert all(isinstance(c["enabled"], bool) for c in combos)
    assert all(isinstance(c["n"], int) for c in combos)


def test_run_batch_with_manifest_records_and_resumes_completed_cases(tmp_path: Path):
    calls: list[float] = []
    sweep = ParameterSweep(width=[1.0, 2.0])

    def factory(width):
        return _FakeSimulation(width, calls)

    first = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        run_kwargs={"n_steps": 3},
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
    )
    assert [r.status for r in first] == ["completed", "completed"]
    assert calls == [1.0, 2.0]

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == BATCH_MANIFEST_SCHEMA
    assert manifest["sweep"] == {"keys": ["width"], "total": 2}
    assert len(manifest["cases"]) == 2
    assert all(record["status"] == "completed" for record in manifest["cases"].values())

    def forbidden_factory(width):  # pragma: no cover - only runs on regression
        raise AssertionError("resume should skip completed cases")

    second = run_batch_with_manifest(
        forbidden_factory,
        sweep,
        tmp_path,
        run_kwargs={"n_steps": 3},
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
        resume=True,
    )
    assert [r.status for r in second] == ["skipped", "skipped"]
    assert all(r.skipped for r in second)
    assert [r.metrics["value"] for r in second] == [1.0, 2.0]
    assert all(r.artifacts["metrics"].startswith(r.case_id + "/") for r in first)
    assert all(
        "sha256" in record["artifact_metadata"]["metrics"]
        for record in manifest["cases"].values()
    )


def test_resume_does_not_skip_completed_record_with_missing_artifact(tmp_path: Path):
    calls: list[float] = []
    sweep = ParameterSweep(width=[1.0])

    def factory(width):
        return _FakeSimulation(width, calls)

    first = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
    )
    assert first[0].status == "completed"
    artifact = tmp_path / first[0].artifacts["metrics"]
    artifact.unlink()

    second = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
        resume=True,
    )
    assert second[0].status == "completed"
    assert calls == [1.0, 1.0]
    assert artifact.exists()


def test_resume_does_not_skip_when_manifest_params_are_corrupt(tmp_path: Path):
    calls: list[float] = []
    sweep = ParameterSweep(width=[1.0])

    def factory(width):
        return _FakeSimulation(width, calls)

    first = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][first[0].case_id]["params"] = {"width": 999.0}
    manifest_path.write_text(json.dumps(manifest))

    second = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
        resume=True,
    )

    assert second[0].status == "completed"
    assert calls == [1.0, 1.0]


def test_resume_does_not_skip_when_artifact_content_changes(tmp_path: Path):
    calls: list[float] = []
    sweep = ParameterSweep(width=[1.0])

    def factory(width):
        return _FakeSimulation(width, calls)

    first = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
    )
    artifact = tmp_path / first[0].artifacts["metrics"]
    artifact.write_text(json.dumps({"value": 999.0}))

    second = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
        resume=True,
    )

    assert second[0].status == "completed"
    assert calls == [1.0, 1.0]


def test_resume_does_not_skip_completed_record_with_empty_artifact_path(tmp_path: Path):
    calls: list[float] = []
    sweep = ParameterSweep(width=[1.0])

    def factory(width):
        return _FakeSimulation(width, calls)

    first = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
    )
    assert first[0].status == "completed"

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][first[0].case_id]["artifacts"] = {"bad": ""}
    manifest_path.write_text(json.dumps(manifest))

    second = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
        resume=True,
    )
    assert second[0].status == "completed"
    assert calls == [1.0, 1.0]


def test_resume_does_not_skip_completed_record_with_empty_metrics(tmp_path: Path):
    calls: list[float] = []
    sweep = ParameterSweep(width=[1.0])

    def factory(width):
        return _FakeSimulation(width, calls)

    first = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
    )
    assert first[0].status == "completed"

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][first[0].case_id]["metrics"] = {}
    manifest_path.write_text(json.dumps(manifest))

    second = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
        resume=True,
    )
    assert second[0].status == "completed"
    assert calls == [1.0, 1.0]


def test_artifact_fn_empty_path_marks_case_failed(tmp_path: Path):
    sweep = ParameterSweep(width=[1.0])

    def factory(width):
        return _FakeSimulation(width, [])

    def bad_artifact(case_dir: Path, params, result):
        return {"bad": ""}

    with pytest.raises(ValueError, match="artifact paths"):
        run_batch_with_manifest(
            factory,
            sweep,
            tmp_path,
            metric_fn=lambda r: {"value": float(r.value)},
            artifact_fn=bad_artifact,
        )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    record = next(iter(manifest["cases"].values()))
    assert record["status"] == "failed"
    assert "ValueError" in record["error"]


def test_run_kwargs_change_invalidates_resume(tmp_path: Path):
    calls: list[float] = []
    sweep = ParameterSweep(width=[1.0])

    def factory(width):
        return _FakeSimulation(width, calls)

    first = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        run_kwargs={"n_steps": 3},
        metric_fn=lambda r: {"value": float(r.value), "n_steps": r.kwargs["n_steps"]},
        artifact_fn=_write_metric_artifact,
    )
    assert first[0].status == "completed"

    second = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        run_kwargs={"n_steps": 4},
        metric_fn=lambda r: {"value": float(r.value), "n_steps": r.kwargs["n_steps"]},
        artifact_fn=_write_metric_artifact,
        resume=True,
    )
    assert second[0].status == "completed"
    assert second[0].metrics["n_steps"] == 4
    assert calls == [1.0, 1.0]


def test_user_run_fingerprint_change_invalidates_resume(tmp_path: Path):
    calls: list[float] = []
    sweep = ParameterSweep(width=[1.0])

    def factory(width):
        return _FakeSimulation(width, calls)

    first = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        run_kwargs={"n_steps": 3},
        run_fingerprint="mesh-a",
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
    )
    first_manifest = json.loads((tmp_path / "manifest.json").read_text())
    first_fp = first_manifest["cases"][first[0].case_id]["run_fingerprint"]

    second = run_batch_with_manifest(
        factory,
        sweep,
        tmp_path,
        run_kwargs={"n_steps": 3},
        run_fingerprint="mesh-b",
        metric_fn=lambda r: {"value": float(r.value)},
        artifact_fn=_write_metric_artifact,
        resume=True,
    )
    second_manifest = json.loads((tmp_path / "manifest.json").read_text())
    record = second_manifest["cases"][second[0].case_id]

    assert second[0].status == "completed"
    assert calls == [1.0, 1.0]
    assert record["run_config"]["user_fingerprint"] == "mesh-b"
    assert record["run_fingerprint"] != first_fp


def test_failed_manifest_record_is_not_resumed_as_completed(tmp_path: Path):
    calls: list[str] = []
    sweep = ParameterSweep(width=[1.0])

    class _FailingSimulation:
        def run(self, **kwargs):
            calls.append("failed")
            raise RuntimeError("intentional boom")

    def failing_factory(width):
        return _FailingSimulation()

    with pytest.raises(RuntimeError, match="intentional boom"):
        run_batch_with_manifest(
            failing_factory,
            sweep,
            tmp_path,
            run_kwargs={"n_steps": 3},
            run_fingerprint="failure-case",
        )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    record = next(iter(manifest["cases"].values()))
    assert record["status"] == "failed"
    assert "RuntimeError" in record["error"]
    assert record["params"] == {"width": 1.0}
    assert record["params_digest"]
    assert record["run_fingerprint"]
    assert record["run_config"]["run_kwargs"] == {"n_steps": 3}
    assert record["run_config"]["user_fingerprint"] == "failure-case"
    assert record["started_at"]
    assert record["completed_at"]
    assert record["duration_s"] >= 0.0

    def factory(width):
        return _FakeSimulation(width, [])

    second = run_batch_with_manifest(factory, sweep, tmp_path, resume=True)
    assert second[0].status == "completed"
    assert calls == ["failed"]


def test_manifest_without_metric_fn_still_records_resume_metric(tmp_path: Path):
    calls: list[float] = []
    sweep = ParameterSweep(width=[1.0])

    def factory(width):
        return _FakeSimulation(width, calls)

    first = run_batch_with_manifest(factory, sweep, tmp_path)
    assert first[0].metrics == {"completed": True}

    def forbidden_factory(width):  # pragma: no cover - only runs on regression
        raise AssertionError("default completion metric should permit resume")

    second = run_batch_with_manifest(forbidden_factory, sweep, tmp_path, resume=True)
    assert second[0].status == "skipped"
    assert second[0].metrics == {"completed": True}
    assert calls == [1.0]


def _physical_factory(amplitude: float):
    sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02), boundary="pec", dx=0.002)
    sim.add_port(
        (0.005, 0.01, 0.01),
        "ez",
        waveform=GaussianPulse(f0=3e9, amplitude=amplitude),
    )
    return sim


def _physical_metric(result):
    ts = np.asarray(result.time_series)
    samples = int(ts.shape[0]) if ts.ndim >= 1 else 0
    ts_peak = float(np.max(np.abs(ts))) if ts.size else 0.0
    state = result.state
    field_peak = max(
        float(np.max(np.abs(np.asarray(getattr(state, comp)))))
        for comp in ("ex", "ey", "ez", "hx", "hy", "hz")
    )
    return {
        "peak_time_series": ts_peak,
        "peak_field": field_peak,
        "samples": samples,
    }


def _physical_artifact(case_dir: Path, params, result):
    path = case_dir / "physical_metric.json"
    path.write_text(json.dumps(_physical_metric(result), sort_keys=True))
    return {"physical_metric": "physical_metric.json"}


def test_tiny_physical_sweep_records_metrics_and_resumes(tmp_path: Path):
    """Physical gate: tiny FDTD sweep metrics persist and resume skips cases."""
    sweep = ParameterSweep(amplitude=[1.0, 2.0])
    first = run_batch_with_manifest(
        _physical_factory,
        sweep,
        tmp_path,
        run_kwargs={"n_steps": 20},
        metric_fn=_physical_metric,
        artifact_fn=_physical_artifact,
    )

    assert [r.status for r in first] == ["completed", "completed"]
    assert all(r.metrics["samples"] == 20 for r in first)
    assert all(r.metrics["peak_field"] >= 0.0 for r in first)
    assert first[1].metrics["peak_field"] > first[0].metrics["peak_field"]
    assert first[1].metrics["peak_time_series"] >= first[0].metrics["peak_time_series"]
    manifest_before = json.loads((tmp_path / "manifest.json").read_text())

    def forbidden_factory(amplitude):  # pragma: no cover - only runs on regression
        raise AssertionError("resume should skip completed physical cases")

    second = run_batch_with_manifest(
        forbidden_factory,
        sweep,
        tmp_path,
        run_kwargs={"n_steps": 20},
        metric_fn=_physical_metric,
        artifact_fn=_physical_artifact,
        resume=True,
    )
    assert [r.status for r in second] == ["skipped", "skipped"]
    assert [r.metrics for r in second] == [r.metrics for r in first]
    manifest_after = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest_after["cases"] == manifest_before["cases"]
