"""Contract tests for the canonical cross-validation manifest."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Literal, TypedDict


REPO_ROOT = Path(__file__).resolve().parents[1]
CROSSVAL_DIR = REPO_ROOT / "examples" / "crossval"
MANIFEST_PATH = CROSSVAL_DIR / "manifest.json"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_crossval_cpu.py"

EvidenceLevel = Literal["E0", "E1", "E2", "E3", "E4", "E5"]
CaseRole = Literal["claims-bearing", "diagnostic-reporter", "research-only"]
PublicGroup = Literal["A", "B"]
ReferenceKind = Literal["analytic", "external-solver", "self-invariant"]
ExecutionTier = Literal[
    "cpu-runner",
    "scheduled-external",
    "gpu-manual",
    "external-manual",
    "research-manual",
]


class ReferenceEntry(TypedDict):
    """Named oracle used by one cross-validation case."""

    kind: ReferenceKind
    name: str
    required_for_script_pass: bool


class CpuRunnerEntry(TypedDict, total=False):
    """Exactly one CPU-runner disposition for a cross-validation case."""

    order: int
    excluded_reason: str


class CrossvalCase(TypedDict):
    """Machine-readable governance entry for one cross-validation script."""

    id: str
    script: str
    role: CaseRole
    public_group: PublicGroup | None
    execution_tiers: list[ExecutionTier]
    evidence_levels: list[EvidenceLevel]
    references: list[ReferenceEntry]
    claim_scope: str
    external_dependencies: list[str]
    expected_exit_codes: list[int]
    gate_paths: list[str]
    artifact_paths: list[str]
    cpu_runner: CpuRunnerEntry
    scheduled_external_order: int | None
    failure_sentinel: str | None


class CrossvalManifest(TypedDict):
    """Top-level cross-validation manifest schema."""

    version: int
    evidence_rule: str
    exit_codes: dict[str, str]
    cases: list[CrossvalCase]


def _load_manifest() -> CrossvalManifest:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_crossval_cpu_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_manifest_covers_every_crossval_script_exactly_once() -> None:
    manifest = _load_manifest()
    assert set(manifest) == {"version", "evidence_rule", "exit_codes", "cases"}
    assert manifest["version"] == 1
    assert (REPO_ROOT / manifest["evidence_rule"]).is_file()
    assert set(manifest["exit_codes"]) == {"0", "1", "2"}

    actual_scripts = {
        path.relative_to(REPO_ROOT).as_posix() for path in CROSSVAL_DIR.glob("*.py")
    }
    registered_scripts = [case["script"] for case in manifest["cases"]]
    assert len(registered_scripts) == len(set(registered_scripts))
    assert set(registered_scripts) == actual_scripts


def test_manifest_entries_are_self_consistent_and_grounded() -> None:
    manifest = _load_manifest()
    ids: list[str] = []
    scheduled_orders: list[int] = []
    cpu_orders: list[int] = []
    valid_roles = {"claims-bearing", "diagnostic-reporter", "research-only"}
    valid_groups = {"A", "B", None}
    valid_evidence = {"E0", "E1", "E2", "E3", "E4", "E5"}
    valid_reference_kinds = {"analytic", "external-solver", "self-invariant"}
    valid_execution_tiers = {
        "pr-fast",
        "cpu-runner",
        "scheduled-external",
        "vessl-external",
        "gpu-manual",
        "external-manual",
        "research-manual",
    }

    for case in manifest["cases"]:
        assert set(case) == {
            "id",
            "script",
            "role",
            "public_group",
            "execution_tiers",
            "evidence_levels",
            "references",
            "claim_scope",
            "external_dependencies",
            "expected_exit_codes",
            "gate_paths",
            "artifact_paths",
            "cpu_runner",
            "scheduled_external_order",
            "failure_sentinel",
        }
        ids.append(case["id"])
        assert Path(case["script"]).stem == case["id"]
        assert case["role"] in valid_roles
        assert case["public_group"] in valid_groups
        assert case["evidence_levels"]
        assert set(case["evidence_levels"]) <= valid_evidence
        assert len(case["evidence_levels"]) == len(set(case["evidence_levels"]))
        assert case["execution_tiers"]
        assert set(case["execution_tiers"]) <= valid_execution_tiers
        assert len(case["execution_tiers"]) == len(set(case["execution_tiers"]))
        assert case["references"]
        assert all(
            set(reference) == {"kind", "name", "required_for_script_pass"}
            and reference["kind"] in valid_reference_kinds
            and reference["name"].strip()
            and isinstance(reference["required_for_script_pass"], bool)
            for reference in case["references"]
        )
        reference_kinds = {reference["kind"] for reference in case["references"]}
        for evidence_level, required_kind in {
            "E1": "self-invariant",
            "E2": "analytic",
            "E4": "external-solver",
        }.items():
            if evidence_level in case["evidence_levels"]:
                assert required_kind in reference_kinds, case["id"]
        assert case["claim_scope"].strip()
        assert case["gate_paths"]
        assert len(case["gate_paths"]) == len(set(case["gate_paths"]))
        assert len(case["artifact_paths"]) == len(set(case["artifact_paths"]))
        assert all(dependency.strip() for dependency in case["external_dependencies"])
        assert set(case["expected_exit_codes"]) <= {0, 1, 2}
        assert case["expected_exit_codes"] == sorted(set(case["expected_exit_codes"]))
        assert {0, 1} <= set(case["expected_exit_codes"])

        required_external_reference = any(
            reference["kind"] == "external-solver"
            and reference["required_for_script_pass"]
            for reference in case["references"]
        )
        assert bool(case["external_dependencies"]) == required_external_reference
        assert (2 in case["expected_exit_codes"]) == required_external_reference

        if case["role"] == "claims-bearing":
            assert set(case["evidence_levels"]) & {"E2", "E3", "E4", "E5"}

        if case["role"] == "research-only":
            assert case["public_group"] is None
            assert "research-manual" in case["execution_tiers"]

        for relative_path in [
            case["script"],
            *case["gate_paths"],
            *case["artifact_paths"],
        ]:
            assert (REPO_ROOT / relative_path).exists(), relative_path

        cpu_entry = case["cpu_runner"]
        assert set(cpu_entry) in ({"order"}, {"excluded_reason"})
        has_order = "order" in cpu_entry
        has_exclusion = "excluded_reason" in cpu_entry
        assert has_order != has_exclusion, case["id"]
        if has_order:
            cpu_orders.append(cpu_entry["order"])
            assert "cpu-runner" in case["execution_tiers"]
        else:
            assert cpu_entry["excluded_reason"].strip()
            assert "cpu-runner" not in case["execution_tiers"]

        scheduled_order = case["scheduled_external_order"]
        if scheduled_order is not None:
            scheduled_orders.append(scheduled_order)
            assert "scheduled-external" in case["execution_tiers"]
        else:
            assert "scheduled-external" not in case["execution_tiers"]

        failure_sentinel = case["failure_sentinel"]
        assert failure_sentinel is None or failure_sentinel.strip()

    assert len(ids) == len(set(ids))
    assert sorted(cpu_orders) == list(range(1, len(cpu_orders) + 1))
    assert sorted(scheduled_orders) == list(range(1, len(scheduled_orders) + 1))


def test_runner_derives_cpu_policy_from_manifest() -> None:
    manifest = _load_manifest()
    runner = _load_runner()
    cases = manifest["cases"]

    expected_cpu = tuple(
        Path(case["script"]).name
        for case in sorted(
            (case for case in cases if "order" in case["cpu_runner"]),
            key=lambda case: case["cpu_runner"]["order"],
        )
    )
    expected_excluded = {
        Path(case["script"]).name: case["cpu_runner"]["excluded_reason"]
        for case in cases
        if "excluded_reason" in case["cpu_runner"]
    }
    expected_sentinels = {
        Path(case["script"]).name: case["failure_sentinel"]
        for case in cases
        if case["failure_sentinel"] is not None
    }
    expected_exit_codes = {
        Path(case["script"]).name: frozenset(case["expected_exit_codes"])
        for case in cases
    }

    assert tuple(runner.CPU_SUBSET) == expected_cpu
    assert runner.EXCLUDED == expected_excluded
    assert runner.EXIT0_FAIL_SENTINELS == expected_sentinels
    assert runner.EXPECTED_EXIT_CODES == expected_exit_codes


def test_runner_exit_classification_matches_manifest_contract() -> None:
    runner = _load_runner()

    assert (
        runner.classify("10_pmc_cpml_half_symmetric.py", 0, "ALL CHECKS PASSED", False)[
            0
        ]
        == "PASS"
    )
    assert (
        runner.classify(
            "10_pmc_cpml_half_symmetric.py", 1, "numeric gate failed", False
        )[0]
        == "FAIL"
    )
    assert (
        runner.classify("01_waveguide_bend.py", 2, "reference unavailable", False)[0]
        == "SELF-CHECK-ONLY"
    )
    assert (
        runner.classify(
            "11_waveguide_port_wr90.py", 2, "unexpected inconclusive", False
        )[0]
        == "FAIL"
    )
    assert (
        runner.classify("10_pmc_cpml_half_symmetric.py", 124, "", True)[0] == "TIMEOUT"
    )
    assert (
        runner.classify(
            "10_pmc_cpml_half_symmetric.py",
            3,
            "unexpected process error",
            False,
        )[0]
        == "FAIL"
    )
    assert (
        runner.classify(
            "01_waveguide_bend.py",
            0,
            "SOME CHECKS FAILED",
            False,
        )[0]
        == "FAIL"
    )
    assert (
        runner.classify(
            "02_ring_resonator.py",
            1,
            "ModuleNotFoundError: No module named 'meep'",
            False,
        )[0]
        == "ENV-SKIP"
    )


def test_runner_timeout_fails_the_aggregate_gate(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner, "CPU_SUBSET", ("case.py",))
    monkeypatch.setattr(runner, "EXCLUDED", {})
    monkeypatch.setattr(
        runner,
        "run_one",
        lambda script: runner.RunResult(
            script=script,
            returncode=124,
            elapsed=0.0,
            status=runner.TIMEOUT,
            note="test timeout",
        ),
    )

    assert runner.main() == 1


def _role_evidence_label(case: CrossvalCase) -> str:
    reference_kinds = "/".join(
        dict.fromkeys(reference["kind"] for reference in case["references"])
    )
    return f"{case['role']} · {'/'.join(case['evidence_levels'])} · {reference_kinds}"


def test_public_tables_match_manifest_classification() -> None:
    manifest = _load_manifest()
    public_cases = {
        case["id"]: (case["public_group"], _role_evidence_label(case))
        for case in manifest["cases"]
        if case["public_group"] is not None
    }

    benchmark_text = (
        REPO_ROOT / "docs" / "public" / "guide" / "benchmarks.mdx"
    ).read_text(encoding="utf-8")
    benchmark_rows = re.findall(
        r"^\|\s*`([0-9][0-9a-z]*_[^`]+)`\s*\|\s*([^|]+?)\s*\|",
        benchmark_text,
        re.MULTILINE,
    )
    benchmark_classifications = {
        case_id: classification.strip() for case_id, classification in benchmark_rows
    }
    assert len(benchmark_rows) == len(benchmark_classifications)
    assert benchmark_classifications == {
        case_id: classification for case_id, (_, classification) in public_cases.items()
    }

    readme_text = (REPO_ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    readme_rows = re.findall(
        r"^\|\s*[AB][^|]*\|\s*`crossval/([^`]+\.py)`\s*\|\s*([AB])\s*\|\s*([^|]+?)\s*\|",
        readme_text,
        re.MULTILINE,
    )
    readme_classifications = {
        Path(script).stem: (group, classification.strip())
        for script, group, classification in readme_rows
    }
    assert len(readme_rows) == len(readme_classifications)
    assert readme_classifications == public_cases

    reference_lane_text = (
        REPO_ROOT / "docs" / "guides" / "reference_lane_contract.md"
    ).read_text(encoding="utf-8")
    canonical_paths = {
        "examples/crossval/manifest.json",
        "examples/crossval/01_waveguide_bend.py",
        "scripts/run_crossval_cpu.py",
    }
    referenced_paths = set(
        re.findall(r"(?:examples|scripts)/[A-Za-z0-9_./-]+", reference_lane_text)
    )
    assert canonical_paths <= referenced_paths
    for referenced_path in referenced_paths:
        assert (REPO_ROOT / referenced_path).exists(), referenced_path


def test_scheduled_workflow_loads_manifest_instead_of_copying_case_list() -> None:
    workflow_text = (REPO_ROOT / ".github" / "workflows" / "validation.yml").read_text(
        encoding="utf-8"
    )
    assert "examples/crossval/manifest.json" in workflow_text
    assert "scheduled_external_order" in workflow_text
    assert "expected_exit_codes" in workflow_text
    assert "could not load scheduled cases from crossval manifest" in workflow_text
    assert "crossval manifest selected no scheduled cases" in workflow_text
    assert "scripts=(" not in workflow_text


def test_repo_map_defers_crossval_claims_to_manifest() -> None:
    """The public agent repo-map must not blanket-label crossval as validated.

    The directory holds mixed roles (claims-bearing / diagnostic-reporter /
    research-only); the map must send readers to the manifest instead of
    asserting a directory-wide validation status (2026-07-10 rules-review
    finding, issue #300 comment).
    """
    repo_map_text = (REPO_ROOT / "docs" / "agent" / "repo-map.mdx").read_text(
        encoding="utf-8"
    )
    assert "Validated external-reference cross-validation scripts" not in repo_map_text
    assert "validated external" not in repo_map_text.lower()
    assert "examples/crossval/manifest.json" in repo_map_text


def test_vessl_external_lane_matches_manifest_classification() -> None:
    manifest = _load_manifest()
    expected_cases = {
        case["id"]
        for case in manifest["cases"]
        if "vessl-external" in case["execution_tiers"]
    }
    vessl_text = (REPO_ROOT / "scripts" / "vessl_crossval_external.yaml").read_text(
        encoding="utf-8"
    )
    configured_cases = set(
        re.findall(r"examples/crossval/([0-9][0-9a-z]*_[A-Za-z0-9_]+)\.py", vessl_text)
    )
    assert configured_cases == expected_cases
