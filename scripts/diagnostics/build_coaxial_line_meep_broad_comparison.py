#!/usr/bin/env python3
"""Broad-E4 external comparison: rfx coaxial-LINE reflection vs MEEP power-flux FDTD.

The MEEP-side analogue of ``build_coaxial_line_openems_broad_comparison.py``: it
compares the PROMOTED rfx API ``Simulation.compute_coaxial_line_reflection``
against an independent MEEP full-wave coax LINE (power-flux two-run extraction;
see ``scripts/diagnostics/meep_coax_line_reference.py``) for the two |Gamma| = 1
full-reflection calibration terminations short / open across a broad 4-12 GHz band.

Both solvers model the SAME coax line (SMA pin / PTFE / shell, Z0=48.6 ohm)
terminated at one end; |Gamma| is reference-plane-independent for a lossless line,
so the |S11| magnitude comparison is robust to the (different) reference planes
the two solvers use. MEEP and rfx run in SEPARATE environments (MEEP needs a
conda numpy<2 stack; rfx needs jax), so MEEP is run first as a standalone producer
on the cluster (LINEAGE A conda-forge pymeep) and this consumer reads its
per-termination ``.npz`` artifacts from ``--meep-artifact-dir``.

The matched (|Gamma|=0) and resistive 25/100 ohm (|Gamma|=0.32/0.35) terminations
are NOT cross-checked against MEEP: matched is 0-by-construction in a PML-terminated
line (no independent evidence), and a 1-cell resistive sheet merely re-tests rfx's
own resistor stamp and is staircase-sensitive (an R5 misdiagnosis trap). They are
instead covered by the exact-analytic broad-E5 envelope
(coax_line_broad_e5_envelope.json), so |Gamma| spanning 0-0.35 is still validated
against exact truth — just not against MEEP.

``--stub`` substitutes the EXACT analytic |Gamma| for the MEEP side too (plumbing
only), and HARD-FORCES status='stub' + an 'enabling-stub' evidence token so a stub
run can never be mistaken for real cross-FDTD evidence (mirrors the openEMS
sibling's stub forcing).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from compare_sparameter_reference import (
    _repo_path,
    compare_sparameter_datasets,
    load_sparameter_dataset,
)
from rfx.api import Simulation
from rfx.sources.sources import GaussianPulse
from rfx.sources.coaxial_port import (
    coaxial_tem_characteristic_impedance,
    SMA_OUTER_RADIUS,
    SMA_PIN_RADIUS,
)

DEFAULT_FREQS_HZ = np.linspace(4.0e9, 12.0e9, 9)
Z0 = coaxial_tem_characteristic_impedance(SMA_PIN_RADIUS, SMA_OUTER_RADIUS)

# Citation for where the resistive loads ARE validated (exact-analytic broad-E5).
E5_ENVELOPE_CITATION = "coax_line_broad_e5_envelope.json"

# MEEP covers the two |Gamma|=1 full-reflection cross-checks (short/open) — the
# robust geometry-only terminations. matched (|Gamma|=0, 0-by-construction in a
# PML-terminated line) and the resistive 25/100 ohm loads stay analytic-E5-only.
# (name, analytic |Gamma|)
TERMINATIONS = [
    ("short", 1.0),
    ("open", 1.0),
]


def _oneport(mag_or_complex: np.ndarray) -> np.ndarray:
    arr = np.asarray(mag_or_complex, dtype=np.complex128)
    s = np.zeros((1, 1, arr.size), dtype=np.complex128)
    s[0, 0, :] = arr
    return s


def _rfx_line_abs_gamma(term: str, *, freqs: np.ndarray, freq_max: float,
                        n_steps: int):
    """rfx |S11| via the promoted coaxial-line reflection API.

    Copied from the openEMS sibling (``_rfx_line_abs_gamma``): single coax port,
    face='top', matched-feed CPML line, calibration termination at the -z end.
    Returns (|S11|(f), max recurrence residual, status).
    """
    sim = Simulation(domain=(0.008, 0.008, 0.040), freq_max=freq_max, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    res = sim.compute_coaxial_line_reflection(
        termination=term, n_steps=n_steps, freqs=np.asarray(freqs, dtype=float),
    )
    return np.abs(np.asarray(res.s11)), float(np.max(res.recurrence_residual)), res.status


def _load_meep_abs_gamma(term: str, *, meep_artifact_dir: Path,
                         freqs: np.ndarray) -> np.ndarray:
    """Read the MEEP power-flux |S11| for ``term`` and interpolate onto ``freqs``.

    Expects ``meep_coax_<term>.npz`` (produced by meep_coax_line_reference.py) with
    ``freqs_hz`` + ``s11_mag``. Fails loud if the artifact is missing or was a stub
    (a stub MEEP npz must not silently feed a non-stub comparison).
    """
    npz_path = meep_artifact_dir / f"meep_coax_{term}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"missing MEEP artifact for term={term!r}: {npz_path}. Run "
            f"scripts/diagnostics/meep_coax_line_reference.py --termination {term} "
            f"--output-dir {meep_artifact_dir} first (LINEAGE A conda pymeep)."
        )
    data = np.load(npz_path)
    if bool(data["stub"]):
        raise RuntimeError(
            f"MEEP artifact {npz_path} is a STUB (analytic), not a real meep run; "
            f"refusing to feed it to a non-stub comparison. Re-run the producer "
            f"WITHOUT --stub on the cluster, or run this consumer with --stub."
        )
    meep_freqs = np.asarray(data["freqs_hz"], dtype=float)
    meep_mag = np.asarray(data["s11_mag"], dtype=float)
    return np.interp(np.asarray(freqs, dtype=float), meep_freqs, meep_mag)


def build(output_dir: Path, *, meep_artifact_dir: Path, freqs: np.ndarray,
          freq_max: float, rfx_n_steps: int, stub: bool,
          terms: list[tuple[str, float]]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_term: list[dict[str, Any]] = []
    worst_max = 0.0
    worst_mean = 0.0
    all_passed = True
    rfx_resid_max = 0.0
    for name, gamma_an in terms:
        rfx_mag, resid, status = _rfx_line_abs_gamma(
            name, freqs=freqs, freq_max=freq_max, n_steps=rfx_n_steps)
        rfx_resid_max = max(rfx_resid_max, resid)
        if stub:
            ext_mag = np.full(freqs.shape, float(gamma_an))   # analytic stand-in
        else:
            ext_mag = _load_meep_abs_gamma(
                name, meep_artifact_dir=meep_artifact_dir, freqs=freqs)
        cand = output_dir / f"rfx_{name}.npz"
        ref = output_dir / f"meep_{name}.npz"
        np.savez(cand, freqs_hz=freqs, s_params=_oneport(rfx_mag))
        np.savez(ref, freqs_hz=freqs, s_params=_oneport(ext_mag))
        # Cross-FDTD magnitude tolerance. The waveguide cross-FDTD precedent is
        # 0.11; |Gamma|=1 short/open and |Gamma|=0 matched should be tighter, so we
        # set 0.10/0.06 (confirmable on the real cluster run).
        cmp = compare_sparameter_datasets(
            load_sparameter_dataset(cand), load_sparameter_dataset(ref),
            terms="S11", comparison_mode="magnitude",
            max_abs_tol=0.20, mean_abs_tol=0.15,
            max_mag_abs_tol=0.10, mean_mag_abs_tol=0.06,
        )
        worst_max = max(worst_max, cmp["summary"]["max_mag_abs_diff"])
        worst_mean = max(worst_mean, cmp["summary"]["mean_mag_abs_diff_max_over_terms"])
        all_passed = all_passed and cmp["status"] == "passed" and status == "passed"
        per_term.append(dict(
            termination=name,
            gamma_analytic_mag=round(gamma_an, 4),
            rfx_abs_gamma=[round(float(x), 4) for x in rfx_mag],
            meep_abs_gamma=[round(float(x), 4) for x in ext_mag],
            max_mag_abs_diff=round(cmp["summary"]["max_mag_abs_diff"], 4),
            rfx_recurrence_residual_max=round(resid, 5),
            rfx_status=status, term_status=cmp["status"]))

    payload: dict[str, Any] = {
        "schema": "rfx.coaxial_line_meep_broad_comparison",
        "schema_version": 1,
        "status": "passed" if all_passed else "failed",
        "evidence_level": "E4-broad-coaxial-line-termination-meep-fdtd-comparison",
        "claim": (
            f"rfx Simulation.compute_coaxial_line_reflection |Gamma| agrees with an "
            f"independent MEEP power-flux full-wave coax line to <= {worst_max:.3f} "
            f"(mean <= {worst_mean:.3f}) across a broad {freqs[0]/1e9:.0f}-{freqs[-1]/1e9:.0f} GHz "
            f"band for the short/open (|Gamma|=1) full-reflection calibration terminations; "
            f"rfx single-TEM-mode recurrence residual <= {rfx_resid_max:.4f}."),
        "claim_scope": (
            "broad external full-wave cross-validation of the promoted rfx coaxial_port "
            "line reflection API against an independent MEEP power-flux coax LINE over a "
            "frequency axis (4-12 GHz) and the short/open full-reflection calibration "
            "terminations (|Gamma| = 1); |Gamma| magnitude comparison (reference-plane "
            "independent for the lossless line). The matched (|Gamma|=0) and resistive "
            "25/100 ohm (|Gamma|=0.32/0.35) terminations are covered by the exact-analytic "
            f"broad-E5 envelope (see {E5_ENVELOPE_CITATION}) rather than MEEP, because "
            "matched is 0-by-construction in a PML-terminated line and a 1-cell MEEP "
            "resistive sheet merely re-tests rfx's own resistor stamp (staircase-sensitive)."),
        # Machine-readable breadth summary the auditor's _comparison_breadth_ok
        # fails-closed without (short/open = the termination geometry axis).
        "summary": {
            "geometry_count": len(per_term),
            "geometries": [p["termination"] for p in per_term],
            "pair_count": len(per_term),
            "passed_pair_count": sum(
                1 for p in per_term
                if p["term_status"] == "passed" and p["rfx_status"] == "passed"
            ),
            "failed_pair_count": sum(
                1 for p in per_term
                if not (p["term_status"] == "passed" and p["rfx_status"] == "passed")
            ),
            "max_mag_abs_diff": round(worst_max, 4),
            "mean_mag_abs_diff": round(worst_mean, 4),
        },
        "cross_solver_max_mag_abs_diff": round(worst_max, 4),
        "cross_solver_mean_mag_abs_diff": round(worst_mean, 4),
        "tolerances": {"max_mag_abs_tol": 0.10, "mean_mag_abs_tol": 0.06},
        "rfx_recurrence_residual_max": round(rfx_resid_max, 5),
        "resistive_loads_reference": E5_ENVELOPE_CITATION,
        "per_termination": per_term,
        "stub": bool(stub),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if stub:
        # Never claim a real external pass from the stub: hard-force status + an
        # 'enabling-stub' evidence token (mirrors the openEMS sibling).
        payload["status"] = "stub"
        payload["evidence_level"] = "E4-enabling-stub-local-plumbing-check"
        payload["claim_scope"] += " STUB run: MEEP replaced by analytic |Gamma| (plumbing only)."
    out_json = output_dir / "coaxial_line_meep_broad_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meep-artifact-dir",
                   default=".omx/physics-gate/coaxial-meep-reference",
                   help="dir holding meep_coax_<term>.npz from the MEEP producer")
    p.add_argument("--output-dir",
                   default=".omx/physics-gate/latest-coaxial-line-meep-broad")
    p.add_argument("--rfx-n-steps", type=int, default=5000)
    p.add_argument("--freq-max", type=float, default=40.0e9)
    p.add_argument("--stub", action="store_true",
                   help="local plumbing check: substitute analytic |Gamma| for MEEP")
    p.add_argument("--only", default="", help="comma-separated termination names to run")
    args = p.parse_args(argv)

    terms = TERMINATIONS
    if args.only:
        keep = set(args.only.split(","))
        terms = [t for t in TERMINATIONS if t[0] in keep]

    payload = build(
        _repo_path(args.output_dir),
        meep_artifact_dir=_repo_path(args.meep_artifact_dir),
        freqs=DEFAULT_FREQS_HZ, freq_max=args.freq_max,
        rfx_n_steps=args.rfx_n_steps, stub=args.stub, terms=terms)
    print(f"status={payload['status']} evidence_level={payload['evidence_level']} "
          f"cross_solver_max_mag_abs_diff={payload['cross_solver_max_mag_abs_diff']}")
    return 0 if payload["status"] in ("passed", "stub") else 1


if __name__ == "__main__":
    raise SystemExit(main())
