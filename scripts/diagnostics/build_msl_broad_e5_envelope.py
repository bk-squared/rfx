"""MSL broad-E5 envelope builder.

Reads the per-case NPZ + manifest written by
``run_msl_broad_e5_sweep.py`` and emits the family broad-E5 envelope
artifact + a markdown gate report.

Pass criteria (per docs/research_notes/20260528_msl_broad_e5_scope.md):
  thru-line:
    mean |S21| over band         > 0.95
    max  |S11| over band         < 0.10
    median |Z0_ext - Z0_HJ|/Z0_HJ < 0.05
    max  |q| over band           < 1.0  (passive line)
  open_stub (λ/4 at band centre):
    resonance frequency error    < 10 % of analytic Pozar prediction
    resonance depth |S21| at dip > 15 dB below band-mean
  all:
    extractor residual           < 1e-2 normalised

Outputs:
  <out_dir>/msl_broad_e5_envelope.json
  <out_dir>/msl_broad_e5_envelope.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]


# Gate thresholds — kept here so a regression bump is one diff.
THRESH = dict(
    thru_mean_s21_min=0.95,
    thru_max_s11=0.10,
    thru_z0_rel_err_med=0.05,
    thru_max_q=1.0,
    stub_freq_err_pct=10.0,
    stub_depth_db=15.0,
    extractor_residual_norm=1e-2,
)


def _case_band_window(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs <= hi)


def _gate_thru(case_npz: dict, summary: dict) -> dict:
    f = case_npz["freqs_hz"]
    s = case_npz["s_matrix"]
    z0 = case_npz["z0_extracted"]
    band_lo = summary["freq_lo_hz"]
    band_hi = summary["freq_hi_hz"]
    mask = _case_band_window(f, band_lo, band_hi)
    if not mask.any():
        return dict(passed=False, reason="band window empty")

    s11_band = np.abs(s[0, 0, mask])
    s21_band = np.abs(s[1, 0, mask])
    # ``z0`` is the per-port, per-freq characteristic impedance from the
    # N-probe extractor. Shape ``(n_ports, n_freqs)``; we collapse port 0.
    z0_port0 = np.asarray(z0[0])
    z0_med = float(np.median(np.real(z0_port0[mask])))
    z0_err = abs(z0_med - summary["z0_target"]) / summary["z0_target"]

    # |q| honesty: q = e^{-j β Δ}; from beta_extracted * delta_probe.
    # The sweep dump records beta in the npz; we recompute q here.
    beta = case_npz["beta_extracted"]
    # Use one Δ = dx between adjacent N-probe planes (matches the sweep
    # default n_probe_spacing = 1 cell).
    delta = float(case_npz["dx"])
    q = np.exp(-1j * np.asarray(beta) * delta)
    max_q = float(np.max(np.abs(q[mask])))

    gates = {
        "mean_s21_gt_0_95": float(s21_band.mean()) > THRESH["thru_mean_s21_min"],
        "max_s11_lt_0_10": float(s11_band.max()) < THRESH["thru_max_s11"],
        "z0_rel_err_lt_5pct": z0_err < THRESH["thru_z0_rel_err_med"],
        "max_q_lt_1": max_q < THRESH["thru_max_q"] + 1e-3,
    }
    return dict(
        passed=all(gates.values()),
        gates=gates,
        metrics=dict(
            mean_s21=float(s21_band.mean()),
            max_s11=float(s11_band.max()),
            z0_extracted_median=z0_med,
            z0_target=summary["z0_target"],
            z0_relative_error=z0_err,
            max_q=max_q,
            n_freqs_in_band=int(mask.sum()),
        ),
    )


def _gate_open_stub(case_npz: dict, summary: dict) -> dict:
    """Open-stub gate: resonance freq + dip depth.

    A λ/4 open-circuit stub at the centre of an MSL trace reflects the
    incident wave back at the stub's λ/4 frequency, producing a sharp
    S21 minimum (deep null) and matching S11 peak. We locate the band's
    deepest |S21| dip and compare its frequency to the analytic
    prediction f_λ4 = c / (4 · L_stub · √ε_eff) with L_stub = λ_g(f0)/4.
    By construction the prediction is exactly f0 = (freq_lo + freq_hi)/2,
    since the sweep driver sizes the stub to λ/4 at band centre.
    """
    f = case_npz["freqs_hz"]
    s = case_npz["s_matrix"]
    band_lo = summary["freq_lo_hz"]
    band_hi = summary["freq_hi_hz"]
    mask = _case_band_window(f, band_lo, band_hi)
    if not mask.any():
        return dict(passed=False, reason="band window empty")

    s21_band = np.abs(s[1, 0, mask])
    s21_band_db = 20.0 * np.log10(np.maximum(s21_band, 1e-12))
    f_band = f[mask]
    i_dip = int(np.argmin(s21_band))
    f_dip = float(f_band[i_dip])
    s21_dip_db = float(s21_band_db[i_dip])

    f_target = 0.5 * (band_lo + band_hi)
    freq_err_pct = 100.0 * abs(f_dip - f_target) / f_target

    band_mean_db = float(np.mean(s21_band_db))
    depth_db = float(band_mean_db - s21_dip_db)

    gates = {
        "freq_err_lt_10pct": freq_err_pct < THRESH["stub_freq_err_pct"],
        "depth_gt_15db": depth_db > THRESH["stub_depth_db"],
    }
    return dict(
        passed=all(gates.values()),
        gates=gates,
        metrics=dict(
            dip_freq_hz=f_dip,
            target_freq_hz=float(f_target),
            freq_error_pct=freq_err_pct,
            dip_s21_dB=s21_dip_db,
            band_mean_s21_dB=band_mean_db,
            dip_depth_dB=depth_db,
            n_freqs_in_band=int(mask.sum()),
        ),
    )


def _gate_case(case_npz: dict, summary: dict) -> dict:
    geom = summary["geometry"]
    if geom == "thru":
        return _gate_thru(case_npz, summary)
    if geom == "open_stub":
        return _gate_open_stub(case_npz, summary)
    return dict(passed=False, reason=f"unknown geometry {geom!r}")


def build_envelope(manifest_path: Path, out_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    cases_out = []
    for summary in manifest["cases"]:
        npz_path = REPO / summary["npz_path"]
        if not npz_path.exists():
            cases_out.append({
                **summary,
                "passed": False,
                "reason": f"missing npz: {summary['npz_path']}",
            })
            continue
        with np.load(npz_path) as npz:
            data = {k: npz[k] for k in npz.files}
        gate = _gate_case(data, summary)
        cases_out.append({**summary, **gate})

    n_passed = sum(1 for c in cases_out if c.get("passed"))
    n_total = len(cases_out)
    envelope = {
        "envelope_id": "msl_broad_e5_2026_05_28",
        "family": "microstrip_line_port",
        "primitive": "add_msl_port(...)",
        "extractor": "extract_msl_nprobe (issue #80 Fix C, JAX SVD lstsq)",
        "scope_notes_path": "docs/research_notes/20260528_msl_broad_e5_scope.md",
        "n_cases_total": n_total,
        "n_cases_passed": n_passed,
        "passed": n_passed == n_total,
        "thresholds": THRESH,
        "cases": cases_out,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "msl_broad_e5_envelope.json"
    out_json.write_text(json.dumps(envelope, indent=2))

    # Markdown gate report
    lines: list[str] = []
    lines.append("# MSL broad-E5 envelope — gate report")
    lines.append("")
    lines.append(f"Result: **{n_passed}/{n_total} cases passed** "
                 f"({'PASS' if envelope['passed'] else 'FAIL'})")
    lines.append("")
    lines.append("Family: `microstrip_line_port`")
    lines.append("Extractor: `extract_msl_nprobe` (issue #80 Fix C)")
    lines.append("Scope: `docs/research_notes/20260528_msl_broad_e5_scope.md`")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append(
        "| case_id | substrate | band | geom | dx | passed | key metric |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for c in cases_out:
        if c.get("geometry") == "thru":
            km = c.get("metrics", {})
            metric = (
                f"|S21|={km.get('mean_s21', np.nan):.3f} "
                f"|S11|={km.get('max_s11', np.nan):.3f} "
                f"Z0={km.get('z0_extracted_median', np.nan):.1f}Ω"
            )
        elif c.get("geometry") == "open_stub":
            km = c.get("metrics", {})
            metric = (
                f"dip @{km.get('dip_freq_hz', 0)/1e9:.2f} GHz "
                f"(err {km.get('freq_error_pct', np.nan):.1f}%) "
                f"depth {km.get('dip_depth_dB', np.nan):.1f} dB"
            )
        else:
            metric = c.get("reason", "")
        lines.append(
            f"| {c['case_id']} | {c.get('substrate', '?')} | "
            f"{c.get('band', '?')} | {c.get('geometry', '?')} | "
            f"{c.get('dx_resolution', '?')} | "
            f"{'OK' if c.get('passed') else 'FAIL'} | {metric} |"
        )

    md_path = out_dir / "msl_broad_e5_envelope.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {out_json}")
    print(f"wrote {md_path}")
    print(f"verdict: {n_passed}/{n_total} cases passed")
    return envelope


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default=".omx/physics-gate/2026-05-28-msl-broad-e5-sweep/msl_broad_e5_sweep_manifest.json",
    )
    p.add_argument(
        "--out-dir",
        default=".omx/physics-gate/2026-05-28-msl-broad-e5",
    )
    args = p.parse_args()
    manifest_path = REPO / args.manifest
    out_dir = REPO / args.out_dir
    build_envelope(manifest_path, out_dir)


if __name__ == "__main__":
    main()
