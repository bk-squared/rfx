from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "golden_workflows"
EXPECTED_IDS = {"patch-antenna", "wr90-waveguide", "multilayer-fresnel"}
EXPECTED_JOURNEY = ["create", "edit", "preview", "validate", "run", "inspect", "export"]
COMPARATORS = {
    "lt": lambda value, limit: value < limit,
    "lte": lambda value, limit: value <= limit,
    "gt": lambda value, limit: value > limit,
    "gte": lambda value, limit: value >= limit,
}


def _fixtures() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(FIXTURE_ROOT.glob("*.json"))
    ]


def _repo_path(value: str) -> Path:
    relative = Path(value)
    assert not relative.is_absolute()
    assert ".." not in relative.parts
    resolved = (ROOT / relative).resolve()
    assert resolved.is_relative_to(ROOT)
    assert resolved.exists(), value
    return resolved


def test_exactly_three_g0_golden_workflows_are_declared():
    fixtures = _fixtures()

    assert {fixture["id"] for fixture in fixtures} == EXPECTED_IDS
    assert all(
        fixture["schema_version"] == "rfx-golden-workflow/v1" for fixture in fixtures
    )


def test_golden_inputs_and_evidence_are_repo_confined_and_present():
    for fixture in _fixtures():
        support_source = _repo_path(fixture["support_lane"]["support_source"])
        assert support_source.is_file()

        canonical = _repo_path(fixture["inputs"]["canonical_source"])
        manifest_path = _repo_path(fixture["inputs"]["evidence_manifest"])
        assert canonical.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["validation"]["passed"] is True

        cpu_smoke = fixture["inputs"]["cpu_smoke_spec"]
        if cpu_smoke is not None:
            smoke = json.loads(_repo_path(cpu_smoke).read_text(encoding="utf-8"))
            assert smoke["execution"]["backend"] == "cpu"
            assert "not-for-quantitative" in smoke["metadata"]["claims"]


def test_each_metric_has_a_unique_machine_gate_and_baseline_respects_it():
    for fixture in _fixtures():
        metrics = fixture["expected_metrics"]
        assert metrics
        assert len({metric["id"] for metric in metrics}) == len(metrics)
        for metric in metrics:
            comparator = COMPARATORS[metric["comparison"]]
            assert isinstance(metric["limit"], (int, float))
            assert metric["observable"]
            assert metric["reference"]
            baseline = metric["observed_baseline"]
            if baseline is not None:
                assert comparator(baseline, metric["limit"]), (
                    fixture["id"],
                    metric["id"],
                )


def test_each_workflow_locks_the_same_end_to_end_ui_journey():
    for fixture in _fixtures():
        assert [item["step"] for item in fixture["ui_journey"]] == EXPECTED_JOURNEY
        assert all(item["acceptance"] for item in fixture["ui_journey"])
        assert {
            "experiment-spec",
            "generated-python",
            "scene-json",
            "preflight-report",
            "run-events",
            "validation-report",
            "bundle-manifest",
        } <= set(fixture["required_artifacts"])
        assert fixture["limitations"]


def test_checked_in_json_schema_names_the_same_contract_and_required_sections():
    schema_path = (
        ROOT
        / "docs"
        / "design_notes"
        / "schemas"
        / "rfx-golden-workflow-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["properties"]["schema_version"]["const"] == "rfx-golden-workflow/v1"
    assert {
        "support_lane",
        "inputs",
        "expected_metrics",
        "ui_journey",
        "required_artifacts",
        "limitations",
    } <= set(schema["required"])
