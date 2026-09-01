#!/usr/bin/env python3
"""Build a claims-aware MSLPort envelope report from gate artifacts."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

_CV06B_FLOATS = {
    "notch_frequency_error_pct": re.compile(r"Notch frequency error\s*=\s*([0-9.]+)\s*%"),
    "notch_depth_db": re.compile(r"Notch depth \|S21\|\s*=\s*(-?[0-9.]+)\s*dB"),
    "z0_median_ohm": re.compile(r"Re\(Z0\) median\s*=\s*([0-9.]+)\s*Ω"),
    # #812 P3: emitted only by post-2026-09-01 cv06b runs (see that script's
    # "ESTIMATOR RESOLUTION" docstring section).
    "stopband_bw_ratio": re.compile(
        r"ratio to ideal r=1 stub\s*([0-9.]+)"),
    "half_grid_witness_bins": re.compile(
        r"half-grid witness spread\s*=\s*([0-9.]+)\s*bin"),
}

# The cv06b windows this reporter mirrors. They are OWNED by
# validation/crossval/06b_msl_notch_filter_uniform.py (NOTCH_FREQ_TOL_PCT,
# STOPBAND_BW_RATIO_WINDOW, HALF_GRID_WITNESS_BINS) and derived in
# docs/design_notes/estimator_resolution_regate.md; they are restated here
# because this reporter parses stdout rather than importing the case.
_CV06B_FREQ_TOL_PCT = 4.0
_CV06B_BW_RATIO_WINDOW = (0.80, 1.20)
_CV06B_WITNESS_BINS = 1.0


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact_path(raw_path: str | None, *, base_json: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, base_json.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def infer_strict_xfail_count(slow_result: dict[str, Any], *, base_json: Path) -> int:
    """Return strict xfail count, including legacy gate JSON without metadata."""
    if "strict_xfail_count" in slow_result:
        return int(slow_result.get("strict_xfail_count") or 0)

    stdout_path = _resolve_artifact_path(
        slow_result.get("stdout_path"),
        base_json=base_json,
    )
    if stdout_path is None or not stdout_path.exists():
        return 0
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    explicit = len(re.findall(r"(?m)^XFAIL\s+", text))
    if explicit:
        return explicit
    match = re.search(r"(\d+)\s+xfailed\b", text)
    return int(match.group(1)) if match else 0


def parse_cv06b_stdout(text: str, rc: int) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    for key, regex in _CV06B_FLOATS.items():
        match = regex.search(text)
        if match:
            metrics[key] = float(match.group(1))
    lo_r, hi_r = _CV06B_BW_RATIO_WINDOW
    gates = {
        "notch_freq_error_lt_4pct":
            metrics.get("notch_frequency_error_pct", 1e9) < _CV06B_FREQ_TOL_PCT,
        # RETAINED as a witness, not removed: on a 63.6364 MHz grid an ideal
        # r=1 stub's WORST sampled minimum is -31.23 dB, so this cannot fail
        # while a notch exists at all (#812). The real depth requirement is
        # the stopband-width gate below.
        "notch_depth_lt_minus_10db_WITNESS":
            metrics.get("notch_depth_db", 1e9) < -10.0,
        "z0_median_40_65_ohm": 40.0 < metrics.get("z0_median_ohm", -1e9) < 65.0,
    }
    # Fail-closed on a post-#812 stdout that is missing them; absent on a
    # pre-#812 log, where they are reported as unavailable rather than passed.
    if "stopband_bw_ratio" in metrics:
        gates["stopband_bw_ratio_in_window"] = \
            lo_r < metrics["stopband_bw_ratio"] < hi_r
    if "half_grid_witness_bins" in metrics:
        gates["half_grid_witness_sub_bin"] = \
            metrics["half_grid_witness_bins"] < _CV06B_WITNESS_BINS
    # A pre-#812 stdout carries neither of the two estimator-resolution
    # quantities, so its "passed" means "passed the gates in force when it
    # ran", NOT "passed the current gate set". Say so in the report rather
    # than letting a reader infer coverage that does not exist.
    measured = ("stopband_bw_ratio" in metrics
                and "half_grid_witness_bins" in metrics)
    return {
        "status": "passed" if rc == 0 and all(gates.values()) else "failed",
        "returncode": rc,
        "metrics": metrics,
        "gates": gates,
        "estimator_resolution_gates_measured": measured,
        "gate_set": "post-812-P3" if measured else "pre-812-P3 (partial: the "
                    "stopband-width and half-grid-witness gates are not "
                    "measured by this stdout)",
    }


def _gate_window_z0(openems_cmp: dict[str, Any]) -> dict[str, Any]:
    freqs = openems_cmp["rfx_freqs_hz"]
    z0_pairs = openems_cmp["rfx_z0"]
    lo, hi = openems_cmp["metrics"]["frequency_window_hz"]
    values = [pair[0] for f, pair in zip(freqs, z0_pairs) if lo <= f <= hi]
    if not values:
        return {"n_points": 0, "mean_re_z0": None, "gate_40_65_ohm": False}
    mean_z0 = sum(values) / len(values)
    return {
        "n_points": len(values),
        "mean_re_z0": mean_z0,
        "min_re_z0": min(values),
        "max_re_z0": max(values),
        "gate_40_65_ohm": 40.0 < mean_z0 < 65.0,
    }


def build_report(
    *,
    slow_msl_json: Path,
    openems_comparison_json: Path,
    cv06b_stdout: Path,
    cv06b_rc: int,
    msl_3probe_replay_json: Path | None = None,
) -> dict[str, Any]:
    slow = _read_json(slow_msl_json)
    slow_result = slow["results"][0]
    openems = _read_json(openems_comparison_json)
    cv06b = parse_cv06b_stdout(cv06b_stdout.read_text(encoding="utf-8"), cv06b_rc)
    z0_gate = _gate_window_z0(openems)
    strict_xfail_count = infer_strict_xfail_count(
        slow_result,
        base_json=slow_msl_json,
    )

    openems_metrics = openems["metrics"]
    openems_ok = openems.get("status") == "passed"
    slow_ok = slow_result.get("status") == "passed"
    laplace_ok = slow_ok and openems_ok and z0_gate["gate_40_65_ohm"]
    notch_ok = cv06b["status"] == "passed"
    eigenmode_blocked = bool(strict_xfail_count)

    real_vi_replay_artifact = None
    real_vi_replay_status = "blocked"
    real_vi_replay: dict[str, Any] | None = None
    if msl_3probe_replay_json is not None:
        real_vi_replay = _read_json(msl_3probe_replay_json)
        real_vi_replay_artifact = _rel(msl_3probe_replay_json)
        real_vi_replay_status = real_vi_replay.get("status", "unknown")

    real_vi_replay_ok = real_vi_replay_status == "passed"
    status = (
        "passed"
        if laplace_ok and notch_ok and real_vi_replay_ok and eigenmode_blocked
        else "failed"
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "claim_level": (
            "E5-narrow/eigenmode-blocked" if status == "passed" else "blocked"
        ),
        "claim_scope": (
            "Uniform-Yee MSLPort laplace/quasi-TEM narrow envelope: thru-line "
            "magnitudes/Z0 in the 3.0--4.5 GHz window plus a uniform-mesh "
            "stub-notch demo plus real raw 3-probe V/I replay. This is not a "
            "broad all-mode MSL envelope because eigenmode support and "
            "nonuniform MSL remain blocked."
        ),
        "slow_msl_result_json": _rel(slow_msl_json),
        "openems_comparison_json": _rel(openems_comparison_json),
        "cv06b_stdout": _rel(cv06b_stdout),
        "slow_msl": {
            "status": slow_result.get("status"),
            "coverage_status": slow_result.get("coverage_status")
            or ("passed_with_xfails" if strict_xfail_count else None),
            "strict_xfail_count": strict_xfail_count,
            "summary_line": slow_result.get("pytest_summary", {}).get("summary_line"),
            "xfail_reasons": slow_result.get("xfail_reasons", []),
        },
        "openems_reference": {
            "status": openems.get("status"),
            "evidence_level": openems.get("evidence_level"),
            "reference_artifact": openems.get("reference_artifact"),
            "metrics": openems_metrics,
            "z0_gate": z0_gate,
        },
        "cv06b_notch": cv06b,
        "msl_3probe_replay": real_vi_replay,
        "evidence_inventory": [
            {
                "level": "E2",
                "claim": "MSL laplace thru-line passive/Z0 gate",
                "artifact": _rel(slow_msl_json),
            },
            {
                "level": "E2",
                "claim": "uniform-mesh MSL stub-notch analytic demo",
                "artifact": _rel(cv06b_stdout),
            },
            {
                "level": "E4",
                "claim": "narrow MSL thru magnitude smoke check against stored openEMS reference",
                "artifact": _rel(openems_comparison_json),
            },
            {
                "level": "E3",
                "claim": "real raw V/I replay of MSL 3-probe de-embedding",
                "artifact": real_vi_replay_artifact,
                "status": real_vi_replay_status,
            },
        ],
        "blocked_claims": [
            {
                "claim": "broad all-mode MSL E5 claims-bearing envelope",
                "reason": "mode='eigenmode' and nonuniform MSL are not yet implemented/validated",
            },
            {
                "claim": "MSL mode='eigenmode' source/extractor envelope",
                "reason": "strict-xfailed until Path-B/FDFD eigenmode implementation and validation land",
            },
            {
                "claim": "nonuniform MSL S-parameters",
                "reason": "compute_msl_s_matrix currently supports the uniform Yee lane only",
            },
        ],
    }


def _write_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "msl_envelope_report.json"
    md_path = output_dir / "msl_envelope_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    om = report["openems_reference"]["metrics"]
    z0 = report["openems_reference"]["z0_gate"]
    cv = report["cv06b_notch"]
    lines = [
        "# MSLPort claims envelope report",
        "",
        f"- status: `{report['status']}`",
        f"- claim_level: `{report['claim_level']}`",
        f"- slow_msl_result_json: `{report['slow_msl_result_json']}`",
        f"- openems_comparison_json: `{report['openems_comparison_json']}`",
        f"- cv06b_stdout: `{report['cv06b_stdout']}`",
        "",
        report["claim_scope"],
        "",
        "## Thru-line / openEMS metrics",
        "",
        "| Metric | Value | Gate |",
        "|---|---:|---|",
        f"| S11 mean abs diff vs openEMS | {om['s11_mean_abs_diff']:.5f} | <= 0.15 |",
        f"| S21 mean abs diff vs openEMS | {om['s21_mean_abs_diff']:.5f} | <= 0.15 |",
        f"| mean `\\|S11\\|` rfx | {om['s11_mean_rfx']:.5f} | report |",
        f"| mean `\\|S21\\|` rfx | {om['s21_mean_rfx']:.5f} | 0.85..1.10 |",
        f"| mean Re(Z0) rfx | {z0['mean_re_z0']:.2f} | 40..65 Ω |",
        "",
        "## Stub/notch demo metrics",
        "",
        "| Metric | Value | Gate |",
        "|---|---:|---|",
    ]
    cmetrics = cv["metrics"]
    lines.extend(
        [
            f"| notch frequency error | {cmetrics.get('notch_frequency_error_pct', float('nan')):.2f}% | < 15% |",
            f"| notch depth | {cmetrics.get('notch_depth_db', float('nan')):.1f} dB | < -10 dB |",
            f"| median Re(Z0) | {cmetrics.get('z0_median_ohm', float('nan')):.1f} Ω | 40..65 Ω |",
            "",
            "## Evidence inventory",
            "",
        ]
    )
    for item in report["evidence_inventory"]:
        artifact = item.get("artifact") or "none"
        status = f" ({item['status']})" if item.get("status") else ""
        lines.append(f"- `{item['level']}` {item['claim']}: `{artifact}`{status}")
    lines.extend(["", "## Blocked claims", ""])
    for item in report["blocked_claims"]:
        lines.append(f"- {item['claim']}: {item['reason']}")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_rel(json_path)}")
    print(f"wrote {_rel(md_path)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slow-msl-json", required=True)
    parser.add_argument("--openems-comparison-json", required=True)
    parser.add_argument("--cv06b-stdout", required=True)
    parser.add_argument("--cv06b-rc", type=int, required=True)
    parser.add_argument("--msl-3probe-replay-json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = build_report(
        slow_msl_json=_repo_path(args.slow_msl_json),
        openems_comparison_json=_repo_path(args.openems_comparison_json),
        cv06b_stdout=_repo_path(args.cv06b_stdout),
        cv06b_rc=args.cv06b_rc,
        msl_3probe_replay_json=(
            None
            if args.msl_3probe_replay_json is None
            else _repo_path(args.msl_3probe_replay_json)
        ),
    )
    _write_reports(report, _repo_path(args.output_dir))
    print(f"status={report['status']} claim_level={report['claim_level']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
