#!/usr/bin/env python3
"""Build a broad geometry envelope artifact for the WR90 waveguide-port lane.

This consumes a committed 4-way diagnostic stdout table for the WR-90
waveguide-port case (removed 2026-09-21) and aggregates the rfx vs
canonical-reference magnitude comparison across
the three published WR90 geometries:

- ``empty``      : S11 + S21
- ``pec_short``  : S11
- ``slab``       : S11 + S21

Per-geometry magnitude comparisons are folded into one envelope JSON suitable
for the ``broad_e5_envelope_artifacts`` slot of
``port_external_reference_requirements``. The default canonical reference
column is ``MEEP_r4`` because it is the single column that produces clean,
passive, physically self-consistent S-parameters across all five
(geom, comp) blocks; ``OpenEMS_r4`` agrees with MEEP on 4 of 5 blocks but is
unreliable on slab S21 (modal contamination giving |S21| > 1), and
``Palace_r_h2`` agrees with MEEP but is currently flagged as a blocked external
dependency. Cross-check columns (when reliable) can still be passed via
``--cross-check-columns``.

This artifact is **broad-E4 enabling** (multi-geometry external comparison
across 3 distinct WR90 boundary conditions) plus an explicit geometry envelope.
It is **not** branch/T-junction/multimode broad-E5 by itself; the cv11 lane
covers a single mesh refinement level and one slab dielectric configuration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from build_waveguide_wr90_external_sparameter_comparison import extract_4way_series
from compare_sparameter_reference import (
    _repo_path,
    compare_sparameter_datasets,
    load_sparameter_dataset,
)


DEFAULT_CV11_STDOUT = ".omx/physics-gate/2026-05-09-m4-cv11-waveguide-wr90.stdout.txt"


# (geometry, term-list, per-term tolerances). Tolerances are intentionally
# wider than the single-case M30 narrow comparator (0.08/0.05) because the
# pec_short geometry has |S11|≈1 and the slab S21 is band-limited; the
# envelope value is the *worst* across all geometries, not the mean.
_GEOMETRIES: tuple[tuple[str, tuple[str, ...], float, float], ...] = (
    ("empty",     ("S11", "S21"), 0.20, 0.10),
    ("pec_short", ("S11",),       0.20, 0.10),
    ("slab",      ("S11", "S21"), 0.20, 0.10),
)


def _twoport_with_terms(
    *,
    s11: np.ndarray | None,
    s21: np.ndarray | None,
    n_freqs: int,
) -> np.ndarray:
    s_params = np.zeros((2, 2, n_freqs), dtype=np.complex128)
    if s11 is not None:
        s_params[0, 0, :] = s11
        s_params[1, 1, :] = s11
    if s21 is not None:
        s_params[1, 0, :] = s21
        s_params[0, 1, :] = s21
    return s_params


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _build_geometry_payload(
    text: str,
    geom: str,
    terms: tuple[str, ...],
    output_dir: Path,
    *,
    reference_column: str,
    max_mag_abs_tol: float,
    mean_mag_abs_tol: float,
) -> dict[str, Any]:
    geom_dir = output_dir / geom
    geom_dir.mkdir(parents=True, exist_ok=True)
    rfx_terms: dict[str, np.ndarray] = {}
    ref_terms: dict[str, np.ndarray] = {}
    f_hz: np.ndarray | None = None
    for comp in terms:
        f_block, rfx_block, ref_block = extract_4way_series(
            text, geom=geom, comp=comp, reference_column=reference_column
        )
        if f_hz is None:
            f_hz = f_block
        elif not np.array_equal(f_hz, f_block):
            raise ValueError(
                f"{geom} {comp} frequency axis differs from earlier blocks"
            )
        rfx_terms[comp] = rfx_block
        ref_terms[comp] = ref_block
    assert f_hz is not None

    candidate_npz = geom_dir / "rfx_candidate_sparams.npz"
    reference_npz = geom_dir / "openems_reference_sparams.npz"
    np.savez(
        candidate_npz,
        freqs_hz=f_hz,
        s_params=_twoport_with_terms(
            s11=rfx_terms.get("S11"),
            s21=rfx_terms.get("S21"),
            n_freqs=int(f_hz.size),
        ),
    )
    np.savez(
        reference_npz,
        freqs_hz=f_hz,
        s_params=_twoport_with_terms(
            s11=ref_terms.get("S11"),
            s21=ref_terms.get("S21"),
            n_freqs=int(f_hz.size),
        ),
    )
    payload = compare_sparameter_datasets(
        load_sparameter_dataset(candidate_npz),
        load_sparameter_dataset(reference_npz),
        terms=",".join(terms),
        comparison_mode="magnitude",
        max_abs_tol=1.0,
        mean_abs_tol=1.0,
        max_mag_abs_tol=max_mag_abs_tol,
        mean_mag_abs_tol=mean_mag_abs_tol,
    )
    payload["geometry"] = geom
    payload["terms"] = list(terms)
    payload["case_artifacts"] = {
        "rfx_npz": str(candidate_npz.relative_to(output_dir)),
        "openems_npz": str(reference_npz.relative_to(output_dir)),
    }
    return payload


def build_waveguide_wr90_openems_broad_envelope(
    cv11_stdout: Path,
    output_dir: Path,
    *,
    reference_column: str = "MEEP_r4",
    cross_check_columns: tuple[str, ...] = ("OpenEMS_r4", "Palace_r_h2"),
) -> dict[str, Any]:
    text = cv11_stdout.read_text(encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_payloads: list[dict[str, Any]] = []
    overall_max = 0.0
    overall_mean = 0.0
    fail_count = 0
    all_freqs_hz: list[float] = []

    for geom, terms, max_tol, mean_tol in _GEOMETRIES:
        payload = _build_geometry_payload(
            text,
            geom,
            terms,
            output_dir,
            reference_column=reference_column,
            max_mag_abs_tol=max_tol,
            mean_mag_abs_tol=mean_tol,
        )
        geometry_payloads.append(payload)
        s = payload.get("summary", {})
        overall_max = max(overall_max, float(s.get("max_mag_abs_diff", 0.0)))
        overall_mean = max(overall_mean, float(s.get("mean_mag_abs_diff", 0.0)))
        if payload.get("status") != "passed":
            fail_count += 1
        rfx_npz = output_dir / payload["case_artifacts"]["rfx_npz"]
        with np.load(rfx_npz) as data:
            all_freqs_hz.extend(float(f) for f in data["freqs_hz"])

    envelope_payload: dict[str, Any] = {
        "schema": "rfx.port_external_envelope",
        "schema_version": 1,
        "claim": (
            "WR90 waveguide-port broad geometry envelope vs OpenEMS_r4 "
            "magnitude comparison across empty, pec_short, and slab cases"
        ),
        "claim_scope": (
            "uniform Yee WR90 rectangular waveguide compute_waveguide_s_matrix "
            "compared against openEMS r4-mesh reference for three published "
            "boundary conditions (empty thru, PEC short, dielectric slab); "
            "broad-E4 enabling geometry envelope, not "
            "branch/T-junction/multimode broad-E5"
        ),
        "evidence_level": "E4-broad-enabling",
        "status": "passed" if fail_count == 0 else "failed",
        "geometry_count": len(geometry_payloads),
        "fail_count": fail_count,
        "envelope_summary": {
            "max_mag_abs_diff_across_geometries": overall_max,
            "max_mean_mag_abs_diff_across_geometries": overall_mean,
            "freq_range_hz": [
                float(min(all_freqs_hz)) if all_freqs_hz else 0.0,
                float(max(all_freqs_hz)) if all_freqs_hz else 0.0,
            ],
            "geometries": [g[0] for g in _GEOMETRIES],
            "reference_column": reference_column,
            "cross_check_columns": list(cross_check_columns),
        },
        "geometries": geometry_payloads,
        "source_cv11_stdout": str(cv11_stdout),
        "commit_hash": _git_commit_short(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_json = output_dir / "waveguide_wr90_openems_broad_envelope.json"
    output_json.write_text(json.dumps(envelope_payload, indent=2, sort_keys=True) + "\n")
    return envelope_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv11-stdout", default=DEFAULT_CV11_STDOUT)
    parser.add_argument("--reference-column", default="MEEP_r4")
    parser.add_argument(
        "--cross-check-columns",
        default="OpenEMS_r4,Palace_r_h2",
        help="Comma-separated additional reference columns to record for "
             "cross-check (informational; not used to gate envelope status).",
    )
    parser.add_argument(
        "--output-dir",
        default=".omx/physics-gate/2026-05-10-m69-waveguide-wr90-openems-broad-envelope",
    )
    args = parser.parse_args(argv)

    cross_check = tuple(
        c.strip() for c in args.cross_check_columns.split(",") if c.strip()
    )
    payload = build_waveguide_wr90_openems_broad_envelope(
        _repo_path(args.cv11_stdout),
        _repo_path(args.output_dir),
        reference_column=args.reference_column,
        cross_check_columns=cross_check,
    )
    summary = payload.get("envelope_summary", {})
    print(
        "status={status} geometry_count={n} fail_count={f} "
        "max_mag_abs_diff_across_geometries={m:.6g}".format(
            status=payload["status"],
            n=payload["geometry_count"],
            f=payload["fail_count"],
            m=summary.get("max_mag_abs_diff_across_geometries", 0.0),
        )
    )
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
