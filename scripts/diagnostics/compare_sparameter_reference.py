#!/usr/bin/env python3
"""Generic S-parameter reference comparator for external-solver E4/E5 gates.

This is infrastructure for the port-family broad-E5 VESSL shards.  It compares
an rfx S-matrix artifact against an independent reference artifact using the
repo convention ``S[receiver_port, driven_port, frequency_index]``.  Passing
this comparator is E4-enabling evidence for a stated geometry; it is not broad
E5 by itself without the family-specific mesh/frequency/geometry envelope.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rfx.io import read_touchstone


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SParameterDataset:
    path: Path
    freqs_hz: np.ndarray
    s_params: np.ndarray
    source_format: str
    z0_ohm: float | None = None


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _complex_from_json_pairs(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape[-1] != 2:
        raise ValueError("JSON complex-pair array must have final dimension 2")
    return arr[..., 0].astype(float) + 1j * arr[..., 1].astype(float)


def _coerce_smatrix_shape(s_params: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    s_params = np.asarray(s_params, dtype=np.complex128)
    if s_params.ndim == 1:
        s_params = s_params[None, None, :]
    if s_params.ndim != 3:
        raise ValueError(
            "s_params must have shape (n_ports, n_ports, n_freqs) or (n_freqs,)"
        )
    if s_params.shape[0] != s_params.shape[1]:
        raise ValueError(f"s_params must be square in port axes; got {s_params.shape}")
    if s_params.shape[2] != freqs_hz.size:
        raise ValueError(
            f"s_params frequency axis {s_params.shape[2]} does not match "
            f"freq count {freqs_hz.size}"
        )
    return s_params


def load_sparameter_dataset(path: str | Path) -> SParameterDataset:
    resolved = _repo_path(path)
    suffix = resolved.suffix.lower()
    if suffix.startswith(".s") and suffix.endswith("p"):
        s_params, freqs_hz, z0_ohm = read_touchstone(resolved)
        return SParameterDataset(
            path=resolved,
            freqs_hz=np.asarray(freqs_hz, dtype=float),
            s_params=_coerce_smatrix_shape(s_params, np.asarray(freqs_hz, dtype=float)),
            source_format="touchstone",
            z0_ohm=float(z0_ohm),
        )
    if suffix == ".npz":
        data = np.load(resolved)
        if "freqs_hz" in data:
            freqs_hz = np.asarray(data["freqs_hz"], dtype=float)
        elif "freqs" in data:
            freqs_hz = np.asarray(data["freqs"], dtype=float)
        else:
            raise ValueError(f"{resolved} missing freqs_hz or freqs")
        if "s_params" not in data:
            raise ValueError(f"{resolved} missing s_params")
        s_params = _coerce_smatrix_shape(data["s_params"], freqs_hz)
        return SParameterDataset(
            path=resolved,
            freqs_hz=freqs_hz,
            s_params=s_params,
            source_format="npz",
            z0_ohm=None,
        )
    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if "freqs_hz" in payload:
            freqs_hz = np.asarray(payload["freqs_hz"], dtype=float)
        elif "freqs_ghz" in payload:
            freqs_hz = np.asarray(payload["freqs_ghz"], dtype=float) * 1e9
        else:
            raise ValueError(f"{resolved} missing freqs_hz or freqs_ghz")
        if "s_params_real" in payload and "s_params_imag" in payload:
            s_params = np.asarray(payload["s_params_real"], dtype=float) + 1j * np.asarray(
                payload["s_params_imag"], dtype=float
            )
        elif "s_params" in payload:
            s_params = _complex_from_json_pairs(payload["s_params"])
        else:
            raise ValueError(
                f"{resolved} missing s_params or s_params_real/s_params_imag"
            )
        return SParameterDataset(
            path=resolved,
            freqs_hz=freqs_hz,
            s_params=_coerce_smatrix_shape(s_params, freqs_hz),
            source_format="json",
            z0_ohm=payload.get("z0_ohm"),
        )
    raise ValueError(f"unsupported S-parameter artifact format: {resolved}")


def parse_terms(terms: str, n_ports: int) -> list[tuple[int, int]]:
    if terms.strip().lower() == "all":
        return [(i, j) for j in range(n_ports) for i in range(n_ports)]
    parsed: list[tuple[int, int]] = []
    for raw in terms.split(","):
        term = raw.strip().upper()
        match = re.fullmatch(r"S([1-9])([1-9])", term)
        if not match:
            raise ValueError(f"unsupported S-parameter term: {raw!r}")
        receiver = int(match.group(1)) - 1
        driven = int(match.group(2)) - 1
        if receiver >= n_ports or driven >= n_ports:
            raise ValueError(f"{term} exceeds n_ports={n_ports}")
        parsed.append((receiver, driven))
    return parsed


def _interp_complex(
    freqs_hz: np.ndarray,
    values: np.ndarray,
    target_freqs_hz: np.ndarray,
) -> np.ndarray:
    return np.interp(target_freqs_hz, freqs_hz, values.real) + 1j * np.interp(
        target_freqs_hz,
        freqs_hz,
        values.imag,
    )


def compare_sparameter_datasets(
    candidate: SParameterDataset,
    reference: SParameterDataset,
    *,
    terms: str = "all",
    comparison_mode: str = "complex",
    f_lo_hz: float | None = None,
    f_hi_hz: float | None = None,
    max_abs_tol: float = 5e-2,
    mean_abs_tol: float = 2e-2,
    max_mag_abs_tol: float = 5e-2,
    mean_mag_abs_tol: float = 2e-2,
    min_phase_mag: float = 1e-6,
    max_phase_abs_tol_rad: float | None = None,
) -> dict[str, Any]:
    if comparison_mode not in {"complex", "magnitude"}:
        raise ValueError(f"unsupported comparison_mode: {comparison_mode!r}")
    if candidate.s_params.shape[:2] != reference.s_params.shape[:2]:
        raise ValueError(
            "candidate/reference port counts differ: "
            f"{candidate.s_params.shape[:2]} vs {reference.s_params.shape[:2]}"
        )
    n_ports = candidate.s_params.shape[0]
    selected_terms = parse_terms(terms, n_ports)

    lower = max(float(candidate.freqs_hz.min()), float(reference.freqs_hz.min()))
    upper = min(float(candidate.freqs_hz.max()), float(reference.freqs_hz.max()))
    if f_lo_hz is not None:
        lower = max(lower, float(f_lo_hz))
    if f_hi_hz is not None:
        upper = min(upper, float(f_hi_hz))
    mask = (reference.freqs_hz >= lower) & (reference.freqs_hz <= upper)
    if not np.any(mask):
        raise ValueError("candidate/reference artifacts have no overlapping frequencies")
    target_freqs_hz = reference.freqs_hz[mask]

    metrics_by_term: list[dict[str, Any]] = []
    for receiver, driven in selected_terms:
        candidate_values = _interp_complex(
            candidate.freqs_hz,
            candidate.s_params[receiver, driven, :],
            target_freqs_hz,
        )
        reference_values = reference.s_params[receiver, driven, mask]
        diff = candidate_values - reference_values
        abs_diff = np.abs(diff)
        mag_abs_diff = np.abs(np.abs(candidate_values) - np.abs(reference_values))
        phase_mask = (
            np.minimum(np.abs(candidate_values), np.abs(reference_values))
            >= min_phase_mag
        )
        if np.any(phase_mask):
            phase_diff = np.angle(
                candidate_values[phase_mask] * np.conj(reference_values[phase_mask])
            )
            max_phase_abs_diff_rad = float(np.max(np.abs(phase_diff)))
        else:
            max_phase_abs_diff_rad = None
        complex_passed = (
            float(np.max(abs_diff)) <= max_abs_tol
            and float(np.mean(abs_diff)) <= mean_abs_tol
        )
        magnitude_passed = (
            float(np.max(mag_abs_diff)) <= max_mag_abs_tol
            and float(np.mean(mag_abs_diff)) <= mean_mag_abs_tol
        )
        # Phase gate (T2.1) — OPT-IN. The framework audit (2026-06-16, #7) found
        # phase was COMPUTED (`max_phase_abs_diff_rad`) but never GATED. When a
        # caller supplies `max_phase_abs_tol_rad`, the masked-bin phase diff must
        # also clear it. Fail-CLOSED: requesting a phase gate when NO bin clears
        # `min_phase_mag` (so phase is unverifiable) is a failure, not a pass.
        # NOTE (cv11 143° saga): cross-solver absolute phase conventions disagree
        # 100°+, so callers MUST pass a LOOSE envelope (e.g. 60°≈1.047 rad) for
        # external-solver comparisons; a tight value is valid only against an
        # analytic oracle de-embedded to the same reference plane.
        # Asymmetry by design: ungated (tol None) stays vacuously True even when no
        # bin clears the mask (legacy preserved), but REQUESTING a gate that cannot
        # be evaluated is a failure — absent evidence != pass (mirrors T1).
        phase_n_points = int(np.sum(phase_mask))
        if max_phase_abs_tol_rad is None:
            phase_passed = True
        elif max_phase_abs_diff_rad is None:
            phase_passed = False
        else:
            phase_passed = max_phase_abs_diff_rad <= float(max_phase_abs_tol_rad)
        term_status = "passed" if (magnitude_passed and phase_passed) else "failed"
        if comparison_mode == "complex":
            term_status = (
                "passed"
                if (complex_passed and magnitude_passed and phase_passed)
                else "failed"
            )
        metrics_by_term.append(
            {
                "term": f"S{receiver + 1}{driven + 1}",
                "status": term_status,
                "n_points": int(target_freqs_hz.size),
                "max_abs_diff": float(np.max(abs_diff)),
                "mean_abs_diff": float(np.mean(abs_diff)),
                "max_mag_abs_diff": float(np.max(mag_abs_diff)),
                "mean_mag_abs_diff": float(np.mean(mag_abs_diff)),
                "max_phase_abs_diff_rad": max_phase_abs_diff_rad,
                "phase_passed": phase_passed,
                "phase_n_points": phase_n_points,
            }
        )

    summary = {
        "max_abs_diff": max(row["max_abs_diff"] for row in metrics_by_term),
        "mean_abs_diff_max_over_terms": max(
            row["mean_abs_diff"] for row in metrics_by_term
        ),
        "max_mag_abs_diff": max(row["max_mag_abs_diff"] for row in metrics_by_term),
        "mean_mag_abs_diff_max_over_terms": max(
            row["mean_mag_abs_diff"] for row in metrics_by_term
        ),
    }
    status = (
        "passed"
        if all(row["status"] == "passed" for row in metrics_by_term)
        else "failed"
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "evidence_level": "E4-enabling",
        "claim": "generic S-parameter comparison against an independent reference artifact",
        "claim_scope": (
            "external-reference comparator only; broad E5 still requires the "
            "family-specific mesh/frequency/geometry envelope and manifest promotion"
        ),
        "candidate_artifact": _display(candidate.path),
        "candidate_format": candidate.source_format,
        "reference_artifact": _display(reference.path),
        "reference_format": reference.source_format,
        "n_ports": n_ports,
        "terms": [row["term"] for row in metrics_by_term],
        "comparison_mode": comparison_mode,
        "frequency_window_hz": [float(target_freqs_hz[0]), float(target_freqs_hz[-1])],
        "target_frequency_count": int(target_freqs_hz.size),
        "tolerances": {
            "max_abs_tol": max_abs_tol,
            "mean_abs_tol": mean_abs_tol,
            "max_mag_abs_tol": max_mag_abs_tol,
            "mean_mag_abs_tol": mean_mag_abs_tol,
            "min_phase_mag": min_phase_mag,
            "max_phase_abs_tol_rad": max_phase_abs_tol_rad,
        },
        "summary": summary,
        "metrics_by_term": metrics_by_term,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--terms", default="all")
    parser.add_argument(
        "--comparison-mode",
        choices=["complex", "magnitude"],
        default="complex",
        help="Use complex S-parameter tolerances or magnitude-only tolerances.",
    )
    parser.add_argument("--f-lo-hz", type=float, default=None)
    parser.add_argument("--f-hi-hz", type=float, default=None)
    parser.add_argument("--max-abs-tol", type=float, default=5e-2)
    parser.add_argument("--mean-abs-tol", type=float, default=2e-2)
    parser.add_argument("--max-mag-abs-tol", type=float, default=5e-2)
    parser.add_argument("--mean-mag-abs-tol", type=float, default=2e-2)
    parser.add_argument("--min-phase-mag", type=float, default=1e-6)
    parser.add_argument(
        "--max-phase-abs-tol-rad",
        type=float,
        default=None,
        help=(
            "OPT-IN phase gate (T2.1): masked-bin |∠candidate − ∠reference| must "
            "clear this. Omit to leave phase ungated (legacy). For external-solver "
            "comparisons use a LOOSE envelope (~1.047 rad = 60°) per the cv11 143° "
            "cross-solver phase-convention saga; tight only vs an analytic oracle."
        ),
    )
    args = parser.parse_args(argv)

    candidate = load_sparameter_dataset(args.candidate)
    reference = load_sparameter_dataset(args.reference)
    payload = compare_sparameter_datasets(
        candidate,
        reference,
        terms=args.terms,
        comparison_mode=args.comparison_mode,
        f_lo_hz=args.f_lo_hz,
        f_hi_hz=args.f_hi_hz,
        max_abs_tol=args.max_abs_tol,
        mean_abs_tol=args.mean_abs_tol,
        max_mag_abs_tol=args.max_mag_abs_tol,
        mean_mag_abs_tol=args.mean_mag_abs_tol,
        min_phase_mag=args.min_phase_mag,
        max_phase_abs_tol_rad=args.max_phase_abs_tol_rad,
    )

    output_path = _repo_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "status={status} max_abs_diff={max_abs_diff:.6g}".format(
            status=payload["status"],
            max_abs_diff=payload["summary"]["max_abs_diff"],
        )
    )
    print(f"wrote {_display(output_path)}")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
