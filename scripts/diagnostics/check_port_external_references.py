#!/usr/bin/env python3
"""Audit external-reference coverage required for broad E5 port validation.

This checker is intentionally stricter than the slow VESSL shard discipline
check. It tracks port-family external/reference obligations directly, so a
release or E5 audit cannot mistake internal replay/oracle evidence for broad
external-solver coverage.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_port_external_shard_execution_manifest import build_execution_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
BROAD_E5_PASS_STATUS = "broad_e5_passed"
DEFAULT_SUPPORT_MATRIX = "docs/guides/sparameter_support_matrix.json"

# Per-port physics-set TARGET ceiling (formalized 2026-06-17). broad-E5 is NOT a
# universal goal: some ports top out at E4 by their physical nature (single-cell
# feeds have no transmission-line oracle to sweep an envelope against), so a port
# meeting its ceiling reads "validated to ceiling", not "blocked from E5". This
# is descriptive context emitted alongside the broad_e5 verdict — it does NOT
# change the gate logic. The vocab is ENFORCED by the contract test
# (test_physics_gate_reporting.py::test_every_family_declares_target_ceiling_and_usage_rule),
# not by this auditor — the emit path passes an unknown ceiling string through
# unchallenged (it is descriptive, never a gate input).
VALID_TARGET_CEILINGS = {
    "broad-E5",                          # full bar reachable (waveguide: achieved)
    "broad-E5-regime-restricted",        # E5 only in a sub-regime (MSL: matched-only)
    "broad-E5-needs-differentiable-api", # E5 physics ok, needs a new API (coax)
    "broad-E5-structural-partial",       # structural ceiling -> sub-case only (floquet broadside)
    "E4-natural-ceiling",                # E4 IS the physical ceiling (lumped/wire)
    "needs-implementation",              # no API yet (generalized_planar)
}
BROAD_E5_ENVELOPE_BLOCKING_TOKENS = (
    "narrow",
    "enabling",
    "blocked",
    "partial",
    "limited",
    "experimental",
    "shadow",
    "only",
)
BROAD_E4_COMPARISON_BLOCKING_TOKENS = BROAD_E5_ENVELOPE_BLOCKING_TOKENS

# Numeric definition of "broad" (T1, 2026-06-16). Prior to this the auditor
# decided "broad" purely by substring-matching the word 'broad' in a producer-
# authored prose string + a token blocklist — gameable by writing the right
# adjectives. These minima make "broad" a property of the artifact's own
# machine-readable summary, enforced for EVERY family (the floor previously lived
# only in tests/test_waveguide_broad_e5_envelope_gates.py for one family).
MIN_BROAD_E5_ENVELOPE_CASES = 4
MIN_BROAD_E5_MESH_POINTS = 2          # distinct dx values (mesh-refinement axis)
MIN_BROAD_E5_GEOMETRY_VARIANTS = 2    # distinct eps_r OR distinct geometries
MIN_BROAD_E5_FREQ_SPAN_RATIO = 1.4    # freq_hi/freq_lo; admits standard WR
                                      # single-mode bands (intrinsically ~1.45-1.5:1)
                                      # while rejecting a near-single-frequency fake
MIN_BROAD_E4_COMPARISON_GEOMETRIES = 2


def _envelope_breadth_ok(payload: dict[str, Any]) -> tuple[bool, str]:
    """Numeric breadth + all-cases-pass check on a broad-E5 envelope's summary.

    Fail-closed: an artifact with no machine-readable ``envelope_summary`` (or
    one whose spans fall below the minima) is NOT broad, regardless of prose.
    """
    s = payload.get("envelope_summary")
    if not isinstance(s, dict):
        return False, "no envelope_summary block (cannot verify breadth numerically)"
    try:
        case_count = int(s.get("case_count", 0))
        passed = int(s.get("passed_case_count", -1))
        dx = {float(x) for x in (s.get("dx_values_m") or [])}
        eps = {float(x) for x in (s.get("eps_r_values") or [])}
        geoms = set(s.get("geometries") or [])
        fr = s.get("freq_range_hz") or [0.0, 0.0]
        lo, hi = float(fr[0]), float(fr[1])
    except (TypeError, ValueError) as exc:
        return False, f"unparsable envelope_summary fields: {exc}"
    if case_count < MIN_BROAD_E5_ENVELOPE_CASES:
        return False, f"case_count {case_count} < {MIN_BROAD_E5_ENVELOPE_CASES}"
    if passed != case_count:
        return False, f"passed_case_count {passed} != case_count {case_count}"
    if len(dx) < MIN_BROAD_E5_MESH_POINTS:
        return False, f"only {len(dx)} distinct dx value(s) < {MIN_BROAD_E5_MESH_POINTS}"
    if max(len(eps), len(geoms)) < MIN_BROAD_E5_GEOMETRY_VARIANTS:
        return False, f"only {max(len(eps), len(geoms))} geometry/eps variant(s) < {MIN_BROAD_E5_GEOMETRY_VARIANTS}"
    if lo <= 0 or hi / lo < MIN_BROAD_E5_FREQ_SPAN_RATIO:
        return False, f"freq span ratio {hi}/{lo} < {MIN_BROAD_E5_FREQ_SPAN_RATIO}"
    tol = payload.get("max_mag_abs_tol")
    md = s.get("max_mag_abs_diff_across_cases")
    if tol is not None and md is not None and float(md) > float(tol):
        return False, f"max_mag_abs_diff {md} > tol {tol}"
    return True, ""


def _comparison_breadth_ok(payload: dict[str, Any]) -> tuple[bool, str]:
    """Numeric breadth + all-pairs-pass check on a broad-E4 comparison summary."""
    s = payload.get("summary")
    if not isinstance(s, dict):
        return False, "no summary block (cannot verify breadth numerically)"
    try:
        g = int(s.get("geometry_count", 0))
        failed = int(s.get("failed_pair_count", -1))
    except (TypeError, ValueError) as exc:
        return False, f"unparsable comparison summary fields: {exc}"
    if g < MIN_BROAD_E4_COMPARISON_GEOMETRIES:
        return False, f"geometry_count {g} < {MIN_BROAD_E4_COMPARISON_GEOMETRIES}"
    if failed != 0:
        return False, f"failed_pair_count {failed} != 0"
    return True, ""


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_GIT_COMMITTED_CACHE: set[str] | None = None


def _reset_git_cache() -> None:
    """Test hook: clear the committed-path cache (it is process-global)."""
    global _GIT_COMMITTED_CACHE
    _GIT_COMMITTED_CACHE = None


def _git_committed_paths() -> set[str]:
    """Repo-relative paths committed to HEAD (T2.5).

    One ``git ls-tree -r HEAD --name-only`` call, cached. Uses HEAD membership —
    genuinely COMMITTED, not merely ``git ls-files``-staged — and definitely not
    ``path.exists()``: a gitignored ``.omx`` artifact merely present on the
    author's disk (the coaxial-overclaim root cause, audit #2) is correctly seen
    as uncommitted. If git is unavailable the set is empty, which under
    require_committed BLOCKS everything (fail-closed, the safe direction).
    """
    global _GIT_COMMITTED_CACHE
    if _GIT_COMMITTED_CACHE is None:
        try:
            r = subprocess.run(
                ["git", "ls-tree", "-r", "HEAD", "--name-only"], cwd=REPO_ROOT,
                capture_output=True, text=True,
            )
            _GIT_COMMITTED_CACHE = (
                set(r.stdout.splitlines()) if r.returncode == 0 else set()
            )
        except OSError:
            _GIT_COMMITTED_CACHE = set()
    return _GIT_COMMITTED_CACHE


def _is_committed(artifact: str) -> bool:
    return _display(_repo_path(artifact)) in _git_committed_paths()


def _artifact_check(value: str) -> dict[str, Any]:
    path = _repo_path(value)
    return {"artifact": value, "path_checked": _display(path), "exists": path.exists()}


_SKIP_XFAIL_MARKERS = {"xfail", "skip", "skipif"}


def _decorator_marker_names(node: ast.AST) -> set[str]:
    """Trailing attribute names of a decorator (``pytest.mark.xfail`` -> {xfail})."""
    target = node.func if isinstance(node, ast.Call) else node
    names: set[str] = set()
    while isinstance(target, ast.Attribute):
        names.add(target.attr)
        target = target.value
    return names


def _module_marks_skip_xfail(tree: ast.Module) -> bool:
    """Detect a module-level ``pytestmark = pytest.mark.{xfail,skip,skipif}``."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            continue
        values = node.value.elts if isinstance(node.value, ast.List) else [node.value]
        for v in values:
            if _decorator_marker_names(v) & _SKIP_XFAIL_MARKERS:
                return True
    return False


def _ad_test_check(value: Any) -> dict[str, Any]:
    """Static AST check of a family's named AD-vs-FD test (T2.2).

    ``value`` is a ``path::testname`` pytest nodeid, or None when the family has
    no committed AD-vs-FD test. Uses ``ast`` (NOT a naked substring) so it (a)
    finds the real ``def`` — no comment / ``def test_foo`` ⊂ ``def test_foobar``
    false positives — and (b) BACKSTOPS the xfail/skip check: a statically
    xfail/skip/skipif-marked test (on the function or a module ``pytestmark``)
    fails here too. The AUTHORITATIVE, collection-time proof (catches conditional
    / parametrize / fixture markers the AST can't see, and dynamic skips it also
    can't — those remain a documented boundary) lives in
    ``tests/test_ad_surface_contract.py::test_ad_fd_gate_tests_are_collected_and_not_xfail_skip``.
    Both gates share this manifest, so they cannot point at different tests, and
    the static backstop means the release verdict cannot silently diverge from
    the collection check (reviewer T2.2). Together they wire the differentiability
    "moat" into the broad-E5 verdict (framework audit finding #6).
    """
    check: dict[str, Any] = {
        "ad_fd_test": value,
        "declared": bool(value),
        "exists": False,
        "found": False,
        "path_checked": "",
        "testname": "",
        "error": "",
    }
    if not value:
        check["error"] = "no ad_fd_test declared"
        return check
    nodeid = str(value)
    if "::" not in nodeid:
        check["error"] = f"ad_fd_test {nodeid!r} is not a path::testname nodeid"
        return check
    path_str, testname = nodeid.split("::", 1)
    # Strip any parametrization suffix and a class qualifier for the def lookup.
    testfunc = testname.split("[", 1)[0].split("::")[-1]
    path = _repo_path(path_str)
    check["path_checked"] = _display(path)
    check["testname"] = testname
    if not path.exists():
        check["error"] = "ad_fd_test file is missing"
        return check
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        check["error"] = f"ad_fd_test file does not parse: {exc}"
        return check
    func = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == testfunc
        ),
        None,
    )
    if func is None:
        check["error"] = f"ad_fd_test {testfunc!r} not defined in {path_str}"
        return check
    check["found"] = True
    marked = any(
        _decorator_marker_names(d) & _SKIP_XFAIL_MARKERS for d in func.decorator_list
    ) or _module_marks_skip_xfail(tree)
    if marked:
        check["error"] = (
            f"ad_fd_test {testfunc!r} is statically xfail/skip/skipif-marked — "
            "cannot prove the differentiability moat"
        )
        return check
    check["exists"] = True
    return check


def _comparison_artifact_check(value: str) -> dict[str, Any]:
    path = _repo_path(value)
    check: dict[str, Any] = {
        "artifact": value,
        "path_checked": _display(path),
        "exists": path.exists(),
        "status": "",
        "evidence_level": "",
        "scope_text": "",
        "is_passed_e4_comparison": False,
        "is_passed_broad_e4_comparison": False,
        "error": "",
    }
    if not path.exists():
        check["error"] = "missing"
        return check
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        check["error"] = f"invalid_json: {exc}"
        return check
    status = str(payload.get("status", ""))
    evidence_level = str(payload.get("evidence_level", ""))
    scope_text = " ".join(
        str(payload.get(key, ""))
        for key in ("claim_scope", "scope", "required_scope", "comparison_scope")
    ).lower()
    normalized_level = evidence_level.lower()
    has_broad_blocking_token = any(
        token in normalized_level or token in scope_text
        for token in BROAD_E4_COMPARISON_BLOCKING_TOKENS
    )
    check["status"] = status
    check["evidence_level"] = evidence_level
    check["scope_text"] = scope_text
    check["is_passed_e4_comparison"] = status == "passed" and evidence_level.startswith(
        "E4"
    )
    breadth_ok, breadth_err = _comparison_breadth_ok(payload)
    check["breadth_ok"] = breadth_ok
    check["is_passed_broad_e4_comparison"] = (
        check["is_passed_e4_comparison"]
        and "broad" in scope_text
        and not has_broad_blocking_token
        and breadth_ok
    )
    if not check["is_passed_e4_comparison"]:
        check["error"] = "comparison artifact is not a passed E4/E4-enabling report"
    elif "broad" in scope_text and not has_broad_blocking_token and not breadth_ok:
        check["error"] = f"E4 comparison claims broad but fails numeric breadth: {breadth_err}"
    return check


def _broad_e5_envelope_artifact_check(value: str) -> dict[str, Any]:
    path = _repo_path(value)
    check: dict[str, Any] = {
        "artifact": value,
        "path_checked": _display(path),
        "exists": path.exists(),
        "status": "",
        "evidence_level": "",
        "scope_text": "",
        "is_passed_broad_e5_envelope": False,
        "error": "",
    }
    if not path.exists():
        check["error"] = "missing"
        return check
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        check["error"] = f"invalid_json: {exc}"
        return check
    status = str(payload.get("status", ""))
    evidence_level = str(payload.get("evidence_level", ""))
    scope_text = " ".join(
        str(payload.get(key, ""))
        for key in ("claim_scope", "scope", "required_scope", "envelope_scope")
    ).lower()
    normalized_level = evidence_level.lower()
    has_blocking_token = any(
        token in normalized_level or token in scope_text
        for token in BROAD_E5_ENVELOPE_BLOCKING_TOKENS
    )
    breadth_ok, breadth_err = _envelope_breadth_ok(payload)
    is_broad_e5 = (
        status == "passed"
        and normalized_level.startswith("e5")
        and "broad" in scope_text
        and not has_blocking_token
        and breadth_ok
    )
    check["status"] = status
    check["evidence_level"] = evidence_level
    check["scope_text"] = scope_text
    check["breadth_ok"] = breadth_ok
    check["is_passed_broad_e5_envelope"] = is_broad_e5
    if not is_broad_e5:
        # Prefer the numeric-breadth reason when the prose/status looked broad —
        # that is the gameability the T1 numeric check closes.
        if (status == "passed" and normalized_level.startswith("e5")
                and "broad" in scope_text and not has_blocking_token and not breadth_ok):
            check["error"] = f"envelope claims broad but fails numeric breadth: {breadth_err}"
        else:
            check["error"] = "envelope artifact is not a passed broad E5 report"
    return check


def _requirement_result(
    entry: dict[str, Any], require_committed: bool = False
) -> dict[str, Any]:
    required = bool(entry.get("required_for_e5", False))
    status = str(entry.get("current_status", ""))
    scope = str(entry.get("required_scope", ""))
    artifact_checks = [_artifact_check(str(a)) for a in entry.get("existing_artifacts", [])]
    yaml_checks = [_artifact_check(str(a)) for a in entry.get("existing_vessl_yamls", [])]
    comparison_checks = [
        _comparison_artifact_check(str(a))
        for a in entry.get("external_comparison_artifacts", [])
    ]
    envelope_checks = [
        _broad_e5_envelope_artifact_check(str(a))
        for a in entry.get("broad_e5_envelope_artifacts", [])
    ]
    ad_test_check = _ad_test_check(entry.get("ad_fd_test"))
    ad_gate_ok = bool(ad_test_check["declared"] and ad_test_check["exists"])
    # T2.5: annotate git-committed status ONLY under require_committed (so the
    # default path stays a true no-op — no git subprocess). git_tracked is None
    # when the check was not run.
    for c in artifact_checks + yaml_checks + comparison_checks + envelope_checks:
        c["git_tracked"] = _is_committed(str(c["artifact"])) if require_committed else None

    missing_artifacts = [a for a in artifact_checks + yaml_checks if not a["exists"]]
    failed_comparison_artifacts = [
        a for a in comparison_checks if not a["is_passed_e4_comparison"]
    ]
    # Counts are CONTENT-only — a passing-content artifact counts as passing. The
    # committed requirement is a SEPARATE gate (uncommitted_gating_artifacts +
    # result_status), so the count-based blockers never contradict the committed
    # blocker for an artifact that genuinely passes on content (M3).
    passed_comparison_artifact_count = sum(
        1 for a in comparison_checks if a["is_passed_e4_comparison"]
    )
    passed_broad_e4_comparison_artifact_count = sum(
        1 for a in comparison_checks if a["is_passed_broad_e4_comparison"]
    )
    failed_envelope_artifacts = [
        a for a in envelope_checks if not a["is_passed_broad_e5_envelope"]
    ]
    passed_envelope_artifact_count = sum(
        1 for a in envelope_checks if a["is_passed_broad_e5_envelope"]
    )
    # Gating artifacts that pass on CONTENT but are not committed to HEAD — the
    # precise overclaim require_committed exists to catch (present-but-untracked,
    # which path.exists() missed; audit #2).
    uncommitted_gating_artifacts = []
    if require_committed:
        uncommitted_gating_artifacts = [
            a["path_checked"]
            for a in comparison_checks
            if a["is_passed_e4_comparison"] and not a["git_tracked"]
        ] + [
            a["path_checked"]
            for a in envelope_checks
            if a["is_passed_broad_e5_envelope"] and not a["git_tracked"]
        ]

    blockers = []
    if required and status != BROAD_E5_PASS_STATUS:
        blockers.append(f"current_status is {status!r}, not {BROAD_E5_PASS_STATUS!r}")
    if required and scope != "broad_e5":
        blockers.append(f"required_scope is {scope!r}, not 'broad_e5'")
    blockers.extend(str(item) for item in entry.get("missing_evidence", []))
    if required and not yaml_checks:
        blockers.append("no VESSL external-reference shard YAML is listed")
    if missing_artifacts:
        blockers.append("listed existing artifact/YAML path is missing")
    if required and status == BROAD_E5_PASS_STATUS and passed_comparison_artifact_count == 0:
        blockers.append(
            "broad_e5_passed requires at least one passed external S-parameter "
            "comparison artifact"
        )
    if (
        required
        and status == BROAD_E5_PASS_STATUS
        and passed_broad_e4_comparison_artifact_count == 0
    ):
        blockers.append(
            "broad_e5_passed requires at least one passed broad E4 external "
            "S-parameter comparison artifact; E4-enabling/narrow artifacts are "
            "not sufficient"
        )
    if required and status == BROAD_E5_PASS_STATUS and passed_envelope_artifact_count == 0:
        blockers.append(
            "broad_e5_passed requires at least one passed broad E5 envelope "
            "artifact covering mesh/frequency/geometry scope"
        )
    if required and status == BROAD_E5_PASS_STATUS and not ad_gate_ok:
        blockers.append(
            "broad_e5_passed requires a named AD-vs-FD test (ad_fd_test) that "
            f"exists and is collected/non-xfail: {ad_test_check['error'] or 'not satisfied'} "
            "— the differentiability moat must be wired into the broad-E5 claim "
            "(framework audit #6)"
        )
    if failed_comparison_artifacts:
        blockers.append("listed external comparison artifact is missing, invalid, or not passed")
    if failed_envelope_artifacts:
        blockers.append("listed broad E5 envelope artifact is missing, invalid, or not passed")
    if uncommitted_gating_artifacts:
        blockers.append(
            "broad_e5_passed gating artifact(s) pass on content but are NOT "
            "committed to HEAD (present on disk only, e.g. gitignored .omx) under "
            f"--require-committed: {uncommitted_gating_artifacts} — commit them to "
            "tests/fixtures/ (audit #2, the coaxial-overclaim hole)"
        )

    result_status = (
        "passed"
        if required
        and status == BROAD_E5_PASS_STATUS
        and not entry.get("missing_evidence")
        and not missing_artifacts
        and passed_comparison_artifact_count > 0
        and passed_broad_e4_comparison_artifact_count > 0
        and passed_envelope_artifact_count > 0
        and ad_gate_ok
        and not failed_comparison_artifacts
        and not failed_envelope_artifacts
        and not uncommitted_gating_artifacts
        else "blocked"
    )
    if not required:
        result_status = "not_required"

    return {
        "family": str(entry.get("family", "")),
        "primitive": str(entry.get("primitive", "")),
        "required_for_e5": required,
        "required_scope": scope,
        "current_status": status,
        "recommended_vessl_shard_id": str(entry.get("recommended_vessl_shard_id", "")),
        "recommended_reference": str(entry.get("recommended_reference", "")),
        "status": result_status,
        "artifact_checks": artifact_checks,
        "yaml_checks": yaml_checks,
        "comparison_artifact_checks": comparison_checks,
        "broad_e5_envelope_artifact_checks": envelope_checks,
        "comparison_artifact_count": len(comparison_checks),
        "passed_comparison_artifact_count": passed_comparison_artifact_count,
        "passed_broad_e4_comparison_artifact_count": (
            passed_broad_e4_comparison_artifact_count
        ),
        "broad_e5_envelope_artifact_count": len(envelope_checks),
        "passed_broad_e5_envelope_artifact_count": passed_envelope_artifact_count,
        "ad_fd_test_check": ad_test_check,
        "ad_gate_ok": ad_gate_ok,
        "require_committed": require_committed,
        "uncommitted_gating_artifacts": uncommitted_gating_artifacts,
        "vessl_yaml_count": len(yaml_checks),
        "missing_artifact_count": len(missing_artifacts),
        "failed_comparison_artifact_count": len(failed_comparison_artifacts),
        "failed_broad_e5_envelope_artifact_count": len(failed_envelope_artifacts),
        "blockers": blockers,
        "target_ceiling": str(entry.get("target_ceiling", "")),
        "usage_rule": str(entry.get("usage_rule", "")),
        "broad_e5_is_the_target_ceiling": (
            # Pure restatement of the declared target_ceiling LABEL — NOT an
            # achieved-vs-ceiling check (it reads no evidence/status field). It
            # exists only to contextualize the broad_e5 verdict: when this is
            # False, the family's physical ceiling is below full broad-E5 (E4,
            # structural-partial, needs-impl, regime-restricted), so a 'blocked'
            # broad_e5 verdict is by-design ("validated to ceiling"), not a
            # failure. Whether the family actually MEETS its ceiling is a
            # separate question this field does not answer.
            str(entry.get("target_ceiling", "")) == "broad-E5"
        ),
        "notes": str(entry.get("notes", "")),
    }


def _surface_families_requiring_e5(support_matrix_path: Path) -> list[dict[str, Any]]:
    support_matrix = _read_json(support_matrix_path)
    families: list[dict[str, Any]] = []
    for row in support_matrix.get("port_families", []):
        if not bool(row.get("is_port", False)):
            continue
        families.append(
            {
                "family": str(row.get("family", "")),
                "source": "port_families",
                "primitive": str(row.get("primitive", "")),
                "evidence_level": str(row.get("evidence_level", "")),
            }
        )
    unavailable_key = (
        "unavailable_port_families"
        if "unavailable_port_families" in support_matrix
        else "future_port_families"
    )
    for row in support_matrix.get(unavailable_key, []):
        families.append(
            {
                "family": str(row.get("family", "")),
                "source": unavailable_key,
                "status": str(row.get("status", "")),
                "planned_primitives": list(
                    row.get(
                        "current_primitives", row.get("planned_primitives", [])
                    )
                ),
            }
        )
    return [row for row in families if row["family"]]


def _surface_coverage(
    results: list[dict[str, Any]], support_matrix_path: Path
) -> dict[str, Any]:
    required_surface_rows = _surface_families_requiring_e5(support_matrix_path)
    required_manifest_families = {
        row["family"] for row in results if row["required_for_e5"]
    }
    missing_rows = [
        row
        for row in required_surface_rows
        if row["family"] not in required_manifest_families
    ]
    return {
        "support_matrix": _display(support_matrix_path),
        "surface_coverage_status": "passed" if not missing_rows else "failed",
        "required_surface_family_count": len(required_surface_rows),
        "missing_manifest_family_count": len(missing_rows),
        "missing_manifest_families": [row["family"] for row in missing_rows],
        "missing_manifest_rows": missing_rows,
    }


def build_external_reference_audit(
    manifest_path: Path,
    support_matrix_path: Path | None = None,
    require_committed: bool = False,
) -> dict[str, Any]:
    if support_matrix_path is None:
        support_matrix_path = _repo_path(DEFAULT_SUPPORT_MATRIX)
    manifest = _read_json(manifest_path)
    default_manifest_path = _repo_path(
        "scripts/diagnostics/port_external_reference_requirements.json"
    )
    check_yaml_contract = manifest_path.resolve() == default_manifest_path.resolve()
    yaml_contract: dict[str, Any] | None = None
    if check_yaml_contract:
        yaml_contract = build_execution_manifest(manifest_path)
    results = [
        _requirement_result(entry, require_committed=require_committed)
        for entry in manifest.get("requirements", [])
    ]
    coverage = _surface_coverage(results, support_matrix_path)
    required_results = [row for row in results if row["required_for_e5"]]
    missing_vessl_yaml = [
        row for row in required_results if row["vessl_yaml_count"] == 0
    ]
    missing_passed_comparison = [
        row for row in required_results if row["passed_comparison_artifact_count"] == 0
    ]
    incomplete = [row for row in required_results if row["status"] != "passed"]
    missing_artifact_count = sum(row["missing_artifact_count"] for row in results)
    failed_comparison_artifact_count = sum(
        row["failed_comparison_artifact_count"] for row in results
    )
    passed_comparison_artifact_count = sum(
        row["passed_comparison_artifact_count"] for row in results
    )
    passed_broad_e4_comparison_artifact_count = sum(
        row["passed_broad_e4_comparison_artifact_count"] for row in results
    )
    failed_broad_e5_envelope_artifact_count = sum(
        row["failed_broad_e5_envelope_artifact_count"] for row in results
    )
    passed_broad_e5_envelope_artifact_count = sum(
        row["passed_broad_e5_envelope_artifact_count"] for row in results
    )
    missing_broad_e5_envelope = [
        row
        for row in required_results
        if row["current_status"] == BROAD_E5_PASS_STATUS
        and row["passed_broad_e5_envelope_artifact_count"] == 0
    ]
    missing_broad_e4_comparison = [
        row
        for row in required_results
        if row["current_status"] == BROAD_E5_PASS_STATUS
        and row["passed_broad_e4_comparison_artifact_count"] == 0
    ]
    vessl_yaml_coverage_status = "passed" if not missing_vessl_yaml else "failed"
    comparison_artifact_coverage_status = (
        "passed"
        if not missing_passed_comparison and failed_comparison_artifact_count == 0
        else "blocked"
    )
    broad_e5_envelope_artifact_coverage_status = (
        "passed"
        if not missing_broad_e5_envelope
        and failed_broad_e5_envelope_artifact_count == 0
        else "blocked"
    )
    schema_status = (
        "passed"
        if results
        and missing_artifact_count == 0
        and coverage["surface_coverage_status"] == "passed"
        and vessl_yaml_coverage_status == "passed"
        and (
            yaml_contract is None
            or yaml_contract["status"] == "passed"
        )
        else "failed"
    )
    status = "passed" if schema_status == "passed" and not incomplete else "blocked"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": _display(manifest_path),
        **coverage,
        "purpose": manifest.get("purpose", ""),
        "completion_rule": manifest.get("completion_rule", ""),
        "schema_status": schema_status,
        "vessl_yaml_coverage_status": vessl_yaml_coverage_status,
        "vessl_yaml_contract_status": (
            yaml_contract["status"] if yaml_contract is not None else "not_checked"
        ),
        "vessl_yaml_contract_launchable_family_count": (
            yaml_contract["launchable_family_count"] if yaml_contract is not None else None
        ),
        "vessl_yaml_contract_diagnostic_command_family_count": (
            yaml_contract["diagnostic_command_family_count"] if yaml_contract is not None else None
        ),
        "vessl_yaml_contract_missing_diagnostic_command_families": (
            yaml_contract["missing_diagnostic_command_families"] if yaml_contract is not None else []
        ),
        "comparison_artifact_coverage_status": comparison_artifact_coverage_status,
        "broad_e5_envelope_artifact_coverage_status": (
            broad_e5_envelope_artifact_coverage_status
        ),
        "status": status,
        "required_family_count": len(required_results),
        "passed_family_count": len(required_results) - len(incomplete),
        "incomplete_count": len(incomplete),
        "missing_vessl_yaml_count": len(missing_vessl_yaml),
        "missing_vessl_yaml_families": [
            row["family"] for row in missing_vessl_yaml
        ],
        "missing_passed_comparison_artifact_count": len(missing_passed_comparison),
        "missing_passed_comparison_artifact_families": [
            row["family"] for row in missing_passed_comparison
        ],
        "missing_broad_e5_envelope_artifact_count": len(missing_broad_e5_envelope),
        "missing_broad_e5_envelope_artifact_families": [
            row["family"] for row in missing_broad_e5_envelope
        ],
        "missing_broad_e4_comparison_artifact_count": len(missing_broad_e4_comparison),
        "missing_broad_e4_comparison_artifact_families": [
            row["family"] for row in missing_broad_e4_comparison
        ],
        "passed_comparison_artifact_count": passed_comparison_artifact_count,
        "passed_broad_e4_comparison_artifact_count": (
            passed_broad_e4_comparison_artifact_count
        ),
        "failed_comparison_artifact_count": failed_comparison_artifact_count,
        "passed_broad_e5_envelope_artifact_count": (
            passed_broad_e5_envelope_artifact_count
        ),
        "failed_broad_e5_envelope_artifact_count": (
            failed_broad_e5_envelope_artifact_count
        ),
        "missing_artifact_count": missing_artifact_count,
        "requirements": results,
        "incomplete": incomplete,
        "completion_decision": (
            "Do not call update_goal: broad E5 external/reference coverage is incomplete or untracked."
            if incomplete or schema_status != "passed"
            else "External/reference manifest is complete; still run the full RF-infra E5 audit before update_goal."
        ),
    }


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "port_external_reference_audit.json"
    md_path = output_dir / "port_external_reference_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Port external-reference E5 audit",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- manifest: `{payload['manifest']}`",
        f"- support_matrix: `{payload['support_matrix']}`",
        f"- status: `{payload['status']}`",
        f"- schema_status: `{payload['schema_status']}`",
        f"- surface_coverage_status: `{payload['surface_coverage_status']}`",
        f"- vessl_yaml_coverage_status: `{payload['vessl_yaml_coverage_status']}`",
        f"- vessl_yaml_contract_status: `{payload['vessl_yaml_contract_status']}`",
        f"- vessl_yaml_contract_launchable_family_count: `{payload['vessl_yaml_contract_launchable_family_count']}`",
        f"- vessl_yaml_contract_diagnostic_command_family_count: `{payload['vessl_yaml_contract_diagnostic_command_family_count']}`",
        f"- comparison_artifact_coverage_status: `{payload['comparison_artifact_coverage_status']}`",
        f"- broad_e5_envelope_artifact_coverage_status: `{payload['broad_e5_envelope_artifact_coverage_status']}`",
        f"- required_surface_family_count: `{payload['required_surface_family_count']}`",
        f"- missing_manifest_family_count: `{payload['missing_manifest_family_count']}`",
        f"- required_family_count: `{payload['required_family_count']}`",
        f"- passed_family_count: `{payload['passed_family_count']}`",
        f"- incomplete_count: `{payload['incomplete_count']}`",
        f"- missing_vessl_yaml_count: `{payload['missing_vessl_yaml_count']}`",
        f"- missing_passed_comparison_artifact_count: `{payload['missing_passed_comparison_artifact_count']}`",
        f"- missing_broad_e4_comparison_artifact_count: `{payload['missing_broad_e4_comparison_artifact_count']}`",
        f"- missing_broad_e5_envelope_artifact_count: `{payload['missing_broad_e5_envelope_artifact_count']}`",
        f"- passed_comparison_artifact_count: `{payload['passed_comparison_artifact_count']}`",
        f"- passed_broad_e4_comparison_artifact_count: `{payload['passed_broad_e4_comparison_artifact_count']}`",
        f"- failed_comparison_artifact_count: `{payload['failed_comparison_artifact_count']}`",
        f"- passed_broad_e5_envelope_artifact_count: `{payload['passed_broad_e5_envelope_artifact_count']}`",
        f"- failed_broad_e5_envelope_artifact_count: `{payload['failed_broad_e5_envelope_artifact_count']}`",
        f"- missing_artifact_count: `{payload['missing_artifact_count']}`",
        "",
        "| Family | Status | Current external status | Recommended shard |",
        "|---|---:|---|---|",
    ]
    for row in payload["requirements"]:
        lines.append(
            f"| `{row['family']}` | `{row['status']}` | "
            f"`{row['current_status']}` | `{row['recommended_vessl_shard_id']}` |"
        )
    if payload["incomplete"]:
        lines.extend(["", "## Incomplete requirements", ""])
        for row in payload["incomplete"]:
            lines.append(f"### `{row['family']}`")
            for blocker in row["blockers"]:
                lines.append(f"- {blocker}")
            lines.append("")
    if payload["missing_manifest_rows"]:
        lines.extend(["", "## Missing support-matrix families", ""])
        for row in payload["missing_manifest_rows"]:
            lines.append(
                f"- `{row['family']}` from `{row['source']}` is not tracked "
                "as a required broad-E5 external/reference item."
            )
    if payload["missing_vessl_yaml_families"]:
        lines.extend(["", "## Missing VESSL shard YAMLs", ""])
        for family in payload["missing_vessl_yaml_families"]:
            lines.append(
                f"- `{family}` has no listed VESSL shard YAML for parallel "
                "external-reference validation."
            )
    if payload["missing_passed_comparison_artifact_families"]:
        lines.extend(["", "## Missing passed external comparison artifacts", ""])
        for family in payload["missing_passed_comparison_artifact_families"]:
            lines.append(
                f"- `{family}` has no listed passed E4/E4-enabling "
                "S-parameter comparison artifact."
            )
    if payload["missing_broad_e4_comparison_artifact_families"]:
        lines.extend(["", "## Missing broad E4 comparison artifacts", ""])
        for family in payload["missing_broad_e4_comparison_artifact_families"]:
            lines.append(
                f"- `{family}` is marked `broad_e5_passed` but has no listed "
                "passed broad E4 external S-parameter comparison artifact. "
                "E4-enabling/narrow artifacts are not sufficient for broad E5."
            )
    if payload["missing_broad_e5_envelope_artifact_families"]:
        lines.extend(["", "## Missing broad E5 envelope artifacts", ""])
        for family in payload["missing_broad_e5_envelope_artifact_families"]:
            lines.append(
                f"- `{family}` is marked `broad_e5_passed` but has no listed "
                "passed broad-E5 mesh/frequency/geometry envelope artifact."
            )
    lines.append(f"\nDecision: {payload['completion_decision']}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_display(json_path)}")
    print(f"wrote {_display(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="scripts/diagnostics/port_external_reference_requirements.json",
    )
    parser.add_argument(
        "--support-matrix",
        default=DEFAULT_SUPPORT_MATRIX,
        help=(
            "Support matrix used to verify that every current/future port family "
            "has a broad-E5 external/reference requirement."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-port-external-reference-audit",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit 2 unless every required port family has broad E5 external/reference coverage.",
    )
    parser.add_argument(
        "--require-committed",
        action="store_true",
        help=(
            "T2.5: a broad_e5_passed family's GATING artifacts (comparison + "
            "envelope) must be committed to HEAD, not merely present on disk "
            "(catches gitignored .omx — the coaxial-overclaim hole). REQUIRES git "
            "(fail-closed if absent); use in git-having CI, off for unit tests "
            "that use tmp_path artifacts."
        ),
    )
    args = parser.parse_args(argv)

    manifest_path = _repo_path(args.manifest)
    payload = build_external_reference_audit(
        manifest_path,
        _repo_path(args.support_matrix),
        require_committed=args.require_committed,
    )
    _write_report(payload, _repo_path(args.output_dir))
    print(f"status={payload['status']} incomplete_count={payload['incomplete_count']}")
    if payload["schema_status"] != "passed":
        return 1
    if args.require_complete and payload["status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
