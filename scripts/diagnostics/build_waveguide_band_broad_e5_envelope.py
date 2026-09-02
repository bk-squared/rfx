"""Band-parameterized broad-E5 envelope builder for rectangular WR bands.

Reads the sweep manifest produced by `run_waveguide_band_broad_e5_flux_sweep.py`,
compares each (dx, eps_r) slab case against analytic Airy (independent
truth), and emits a port_external_references-compatible envelope JSON.

Truth source: analytic Airy formula for single dielectric slab in
rectangular waveguide (TE10 modal impedance, multi-reflection).
"""
from __future__ import annotations
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from _fixture_recapture import (  # noqa: E402  (sibling module: scripts/diagnostics is sys.path[0])
    RECAPTURE_NOTE_KEY,
    RECAPTURE_POINTER_KEY,
    UNIFORM_BAND_E5_JOB_YAML,
    repo_relative,
    uniform_band_magnitude_note,
)

REPO = Path(__file__).resolve().parents[2]
C0 = 299_792_458.0
ETA0 = 376.730313668

# Broad-E5 magnitude tolerance — DERIVED FROM THE MEASURED ENVELOPE, not a round
# number and NOT a (k·dx)² dispersion formula. T2.4 (2026-06-17) fit the 20
# committed cases to `C·(k·dx)² + floor` and FALSIFIED it (C<0, R²=0.19): the
# error is dominated by dielectric contrast / slab-interface staircasing, not
# grid dispersion (see docs/research_notes/20260617_t2.4_dispersion_tolerance_falsified.md).
# So MAX_TOL is the measured envelope: the worst committed case diff is 0.0414
# (an eps_r=4 slab at fine mesh); 0.05 = 0.0414 × ~1.2 safety margin. The margin
# is bounded on BOTH sides by tests/crossval/test_waveguide_broad_e5.py
# (>= worst case so it never fails a validated case; <= worst × 1.5 so it is not
# slack). Tightening below ~0.05 would fail real eps_r=4 cases — that ceiling is
# physics, not slack.
MAX_TOL = 0.05
RATIO_FLOOR = 0.005

# Irreducible empty-guide |S| floor — a re-derivable MEASUREMENT
# (scripts/diagnostics/measure_waveguide_noise_floor.py), not a bare constant.
# NOTE: this is DESCRIPTIVE metadata, NOT a gate input — the only gate is
# `max_mag_abs_diff <= MAX_TOL`, and the floor (~7e-4) sits far below MAX_TOL
# (0.05), so it never clamps anything. It documents the extractor's irreducible
# |S| floor for context, and replaces the old un-derived 0.0021 with a number a
# reader can re-derive.
_NOISE_FLOOR_FIXTURE = (
    REPO / "tests" / "fixtures" / "waveguide_broad_e5" / "noise_floor_measurement.json"
)


def _committed_noise_floor() -> float:
    """Read the committed empty-guide noise-floor measurement (T2.4).

    Raises rather than falling back to the retired 0.0021 constant — a missing or
    malformed measurement must surface, not silently resurrect the bare number
    T2.4 set out to retire.
    """
    import json
    return float(json.loads(_NOISE_FLOOR_FIXTURE.read_text())["noise_floor"])


def airy_slab(f, eps_r, L, fc_v):
    fc_d = fc_v / np.sqrt(eps_r)
    z_v = ETA0 / np.sqrt(1.0 - (fc_v / f) ** 2)
    z_d = (ETA0 / np.sqrt(eps_r)) / np.sqrt(1.0 - (fc_d / f) ** 2)
    rho = (z_d - z_v) / (z_d + z_v)
    tau = 2 * z_d / (z_d + z_v)
    tau_back = 2 * z_v / (z_d + z_v)
    beta_d = (2 * np.pi * f * np.sqrt(eps_r) / C0) * np.sqrt(1.0 - (fc_d / f) ** 2)
    delta = beta_d * L
    e2 = np.exp(-2j * delta)
    denom = 1.0 - rho * rho * e2
    s11 = rho * (1.0 - e2) / denom
    s21 = tau * tau_back * np.exp(-1j * delta) / denom
    return s11, s21


def _commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()[:7]
    except Exception:
        return "unknown"


def _validate_claim_scope(text: str, band_token: str) -> None:
    band_number_tag = band_token.split("_")[0].replace("wr", "wr-")
    required = ("broad", "mesh", "frequency", "geometry", band_number_tag, "airy", "flux")
    blocking = ("narrow", "enabling", "partial", "experimental", "shadow", "only")
    lower = text.lower()
    missing = [t for t in required if t not in lower]
    if missing:
        raise SystemExit(f"claim_scope missing tokens: {missing}")
    found_block = [t for t in blocking if t in lower]
    if found_block:
        raise SystemExit(f"claim_scope contains blocking tokens: {found_block}")


def _uniform_cpml(cases_out):
    """The one absorber depth every case used, or None if they differ / unknown.

    None is the honest answer for a --cpml-fraction run: the depth is derived per
    dx, so no single number describes the lane and the #496 auditor must fall
    through to the per-case record instead of trusting a summary.
    """
    depths = {c["cpml_layers"] for c in cases_out}
    if len(depths) != 1:
        return None
    only = depths.pop()
    return int(only) if only is not None else None


def build_envelope(manifest_path: Path, band_token: str, band_label: str):
    manifest = json.loads(manifest_path.read_text())
    fc_v = float(manifest["fc_te10_hz"])
    ref_left = float(manifest["reference_planes_x_m"][0])
    ref_right = float(manifest["reference_planes_x_m"][1])
    port_left = float(manifest["ports_x_m"][0])
    port_right = float(manifest["ports_x_m"][1])
    slab_center = 0.5 * (port_left + port_right)

    cases_out = []
    diffs_all = []
    for case in manifest["cases"]:
        npz_path = REPO / case["rfx_npz"]
        data = np.load(npz_path, allow_pickle=False)
        freqs = data["freqs_hz"]
        s11 = data["s11"]
        s21 = data["s21"]
        case_eps_r = float(data["eps_r"])
        case_slab_L = float(data["slab_length_m"])
        s11_e, s21_e = airy_slab(freqs, case_eps_r, case_slab_L, fc_v)
        beta_v = (2 * np.pi * freqs / C0) * np.sqrt(1.0 - (fc_v / freqs) ** 2)
        d_left = slab_center - 0.5 * case_slab_L - ref_left
        d_right = ref_right - (slab_center + 0.5 * case_slab_L)
        s11_ref = s11_e * np.exp(-2j * beta_v * d_left)
        # Reference-plane-corrected S21 phase (single forward pass through the
        # vacuum gap on BOTH sides of the slab -- see the Lane 1 phase
        # producer's convention note, build_waveguide_band_broad_e5_phase_envelope.py,
        # for the full derivation). This term is PROVABLY magnitude-output-
        # neutral here (|exp(i*anything)| == 1, so it never changes
        # s21_diff = |s21| - |s21_ref| below) -- it is phase-load-bearing only
        # in the phase producer, which is why the old
        # exp(+1j*beta_v*case_slab_L) placeholder (wrong sign AND wrong
        # distance) shipped for a long time without any magnitude gate
        # catching it. Fixed at the source per issue #490 Lane 1 review
        # (adversarial review M4); the phase producer and its falsifier keep
        # their own independent formula/inline-bug copies and do not depend
        # on this line.
        s21_ref = s21_e * np.exp(-1j * beta_v * (d_left + d_right))
        s11_diff = np.abs(np.abs(s11) - np.abs(s11_ref))
        s21_diff = np.abs(np.abs(s21) - np.abs(s21_ref))
        case_max = max(float(s11_diff.max()), float(s21_diff.max()))
        unit = np.abs(s11) ** 2 + np.abs(s21) ** 2
        diffs_all.append(case_max)
        cases_out.append({
            "tag": case["tag"],
            "dx_m": float(case["dx_m"]),
            "geometry": case["geometry"],
            "eps_r": case_eps_r,
            "slab_length_m": case_slab_L,
            "n_cells_total": int(case["n_cells_total"]),
            "cells_per_lambda_max_hz": float(case["cells_per_lambda_max_hz"]),
            # Per-case, because with --cpml-fraction the depth is derived per dx.
            # The #496 auditor reads THIS in preference to setup_recipe, so the
            # absorber a case actually ran with can never be misreported by a
            # manifest-level summary. None when an older manifest predates it.
            "cpml_layers": (int(case["cpml_layers"])
                            if case.get("cpml_layers") is not None else None),
            "freqs_hz_min": float(freqs.min()),
            "freqs_hz_max": float(freqs.max()),
            "n_freqs": int(len(freqs)),
            "s11_max_mag_abs_diff": float(s11_diff.max()),
            "s11_mean_mag_abs_diff": float(s11_diff.mean()),
            "s21_max_mag_abs_diff": float(s21_diff.max()),
            "s21_mean_mag_abs_diff": float(s21_diff.mean()),
            "max_mag_abs_diff": case_max,
            "unitarity_min": float(unit.min()),
            "unitarity_max": float(unit.max()),
            "rfx_npz": case["rfx_npz"],
            "status": "passed" if case_max <= MAX_TOL else "failed",
        })

    diffs_all = np.array(diffs_all)
    max_across = float(diffs_all.max())
    mean_across = float(diffs_all.mean())
    ratio_spread = float((diffs_all.max() - diffs_all.min()) / max(diffs_all.max(), 1e-12))
    failed_cases = [c for c in cases_out if c["status"] != "passed"]
    status = "passed" if not failed_cases else "failed"
    dxs = sorted({c["dx_m"] for c in cases_out})
    geoms = sorted({c["geometry"] for c in cases_out})
    eps_rs = sorted({c["eps_r"] for c in cases_out})

    claim_scope = (
        f"broad rfx {manifest['waveguide']} rectangular_waveguide_port "
        "compute_waveguide_s_matrix(normalize='flux') versus analytic Airy "
        "reference envelope spanning the uniform mesh refinement axis "
        f"(dx from {min(dxs)*1e6:.0f} to {max(dxs)*1e6:.0f} um), "
        f"the frequency axis ({manifest['band_hz'][0]/1e9:.1f}-"
        f"{manifest['band_hz'][1]/1e9:.1f} GHz {band_label} band, cutoff ratio "
        f"{manifest['cutoff_ratio_range'][0]:.2f}-{manifest['cutoff_ratio_range'][1]:.2f}), "
        f"and the geometry axis (eps_r in {eps_rs} dielectric slabs). "
        "Power-flux extraction (extract_waveguide_s_matrix_flux) eliminates "
        "the Z_TE impedance-mismatch error of normalize=False and the round-"
        "trip dispersion error of normalize=True diagonal formula. Truth "
        "source is independent analytic Airy formula (multi-reflection "
        "inside dielectric slab with TE10 modal impedance), not a "
        "same-class FDTD reference."
    )
    _validate_claim_scope(claim_scope, band_token)

    envelope = {
        "schema": f"rfx.waveguide_{band_token}_broad_e5_envelope",
        "schema_version": 1,
        "status": status,
        "evidence_level": f"E5-broad-mesh-frequency-geometry-flux-{band_token}",
        "claim": (
            f"rfx {manifest['waveguide']} compute_waveguide_s_matrix(normalize='flux') "
            f"S-parameter envelope versus analytic Airy reference across "
            f"{len(dxs)} mesh refinement points (dx={[int(d*1e6) for d in dxs]} um) "
            f"and {len(eps_rs)} geometries (eps_r={eps_rs}) over the "
            f"{manifest['band_hz'][0]/1e9:.1f}-{manifest['band_hz'][1]/1e9:.1f} GHz "
            f"{band_label} band {'passes' if status == 'passed' else 'fails'} "
            "the broad-E5 magnitude tolerance of 0.05."
        ),
        "claim_scope": claim_scope,
        "commit_hash": _commit_hash(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_mag_abs_tol": MAX_TOL,
        "max_mag_abs_tol_provenance": (
            "measured-envelope: worst committed case diff 0.0414 (eps_r=4 slab, "
            "fine mesh) x ~1.2 margin; (k.dx)^2 dispersion model falsified (T2.4)"
        ),
        "ratio_spread_floor": RATIO_FLOOR,
        "noise_floor_baseline": _committed_noise_floor(),
        "noise_floor_baseline_role": (
            "DESCRIPTIVE metadata, not a gate input — the only gate is "
            "max_mag_abs_diff <= max_mag_abs_tol; the floor sits far below it"
        ),
        "noise_floor_measurement": (
            "tests/fixtures/waveguide_broad_e5/noise_floor_measurement.json"
        ),
        "primary_reference": {
            "label": "analytic_airy",
            "truth_key": "airy_slab_closed_form",
            "path": "internal_closed_form",
            "meta": {
                "formula": "S_slab = rho(1-e^{-2j delta}) / (1 - rho^2 e^{-2j delta})",
                "eps_r_values": eps_rs,
                "slab_length_m": float(manifest["slab_length_m"]),
            },
        },
        "cross_check_references": [],
        "envelope_summary": {
            "case_count": len(cases_out),
            "passed_case_count": sum(1 for c in cases_out if c["status"] == "passed"),
            "failed_case_count": len(failed_cases),
            "freq_range_hz": list(manifest["band_hz"]),
            "cutoff_te10_hz": fc_v,
            "cutoff_ratio_range": list(manifest["cutoff_ratio_range"]),
            "geometries": geoms,
            "dx_values_m": dxs,
            "eps_r_values": eps_rs,
            "max_mag_abs_diff_across_cases": max_across,
            "mean_max_mag_abs_diff_across_cases": mean_across,
            "ratio_spread": ratio_spread,
            "primary_reference_label": "analytic_airy",
            "primary_truth_key": "airy_slab_closed_form",
            "mesh_axis_kind": "uniform_dx_refinement",
            "setup_recipe": {
                # Derived from the CASES, not the manifest header. The header is
                # written before --cpml-fraction is applied per dx, so on a
                # derived-absorber run it names the spec default that no case
                # used — which is what made the #496 auditor report the
                # 0.75-lambda_g WR-15/WR-28 probes as 0.060. One number here is
                # honest only when every case agrees; otherwise None, and
                # cases[].cpml_layers is the record.
                "cpml_layers": _uniform_cpml(cases_out),
                "cpml_layers_per_case": len(
                    {c["cpml_layers"] for c in cases_out}) > 1,
                "cpml_fraction": manifest.get("cpml_fraction"),
                "normalize": manifest["normalize"],
                "num_periods": int(manifest["num_periods"]),
                "domain_m": list(manifest["domain_m"]),
            },
            "runtime_env": {
                "jax_default_backend": manifest.get("jax_default_backend"),
                "jax_enable_x64": manifest.get("jax_enable_x64"),
                "jax_version": manifest.get("jax_version"),
                "numpy_version": manifest.get("numpy_version"),
            },
        },
        "diagnostic_note": (
            f"max_mag_abs_diff_across_cases {max_across:.4f} (tol {MAX_TOL}); "
            f"ratio_spread {ratio_spread:.4f} (floor {RATIO_FLOOR}); flux-mode "
            "recipe from docs/guides/sparameter_support_matrix.md."
        ),
        # The committed fixture must point at something tracked: the job YAML
        # is the lane's re-capture entry point, and the manifest this build
        # actually read (gitignored .omx/) is kept as provenance in the note.
        RECAPTURE_POINTER_KEY: UNIFORM_BAND_E5_JOB_YAML,
        RECAPTURE_NOTE_KEY: uniform_band_magnitude_note(
            band_token=band_token, band_label=band_label,
            manifest_relpath=repo_relative(manifest_path, REPO)),
        "cases": cases_out,
    }

    out_path = manifest_path.parent.parent / f"waveguide_{band_token}_broad_e5_envelope.json"
    out_path.write_text(json.dumps(envelope, indent=2))
    return out_path, envelope


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--band-token", required=True,
                   help="e.g. wr28_kaband, wr62_kuband")
    p.add_argument("--band-label", required=True,
                   help="e.g. Ka, Ku, V, S, W")
    args = p.parse_args()
    out_path, env = build_envelope(args.manifest, args.band_token, args.band_label)
    print(f"wrote {out_path}")
    print(f"status: {env['status']}")
    s = env["envelope_summary"]
    print(f"case_count: {s['case_count']}, passed: {s['passed_case_count']}")
    print(f"max_mag_abs_diff_across_cases: {s['max_mag_abs_diff_across_cases']:.4f}")
    print(f"mean: {s['mean_max_mag_abs_diff_across_cases']:.4f}")
    print(f"ratio_spread: {s['ratio_spread']:.4f}")
    print()
    print("per-case:")
    for c in env["cases"]:
        print(f"  {c['tag']:25s} max={c['max_mag_abs_diff']:.4f} "
              f"U=[{c['unitarity_min']:.4f},{c['unitarity_max']:.4f}] "
              f"-> {c['status']}")


if __name__ == "__main__":
    main()
