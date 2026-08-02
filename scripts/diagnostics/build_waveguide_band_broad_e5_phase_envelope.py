"""Issue #490 Lane 1 — broad-E5 Airy PHASE envelope for the five WR bands.

The committed broad-E5 envelope (``build_waveguide_band_broad_e5_envelope.py``)
compares ``compute_waveguide_s_matrix(normalize='flux')`` against the analytic
Airy slab formula on MAGNITUDE only. The support matrix says so explicitly
("phase and junction evidence are narrower"). This script re-analyzes the SAME
committed per-case complex S11/S21 data (the raw npz already stores complex
values — magnitude was simply the only thing anyone looked at) and adds a
phase gate, closing part of that gap.

PROVENANCE (read before trusting any number below): this script performs NO
new FDTD run for the 20-case, 5-band sweep. It re-reads the exact npz/manifest
pair that produced the currently-committed magnitude envelope
(``tests/fixtures/waveguide_broad_e5/waveguide_*_broad_e5_envelope.json``,
generated on gpu-rtx4090, VESSL job 369367242914, per that test file's
docstring). Phase and magnitude are two projections of the SAME complex
S-parameter — re-deriving phase from data already collected is honest reuse,
not a new physics claim; it is not "gate all five bands on a fresh CPU run"
because no fresh run was needed or performed. The ONE genuinely new
measurement in this lane (a domain-size invariance witness, since invariance
requires an independent domain, not just a different analysis of one domain)
is a separate, small, freshly-run case — see
``waveguide_phase_falsifier_and_domain_invariance.py``.

CONVENTION NOTE (read before touching the sign of any exponent here):

- Time convention: e^{+jwt} ("engineering"/IEEE). This is what
  ``rfx.sources.waveguide_port._rect_dft`` extracts: its DFT kernel is
  ``exp(-j*omega*t)``, which picks out the coefficient of ``exp(+j*omega*t)``
  from a real time series (Sigma x(t) exp(-jwt) dt over one period ~ (A/2)
  exp(+j*phi) * T when x(t) = Re[A exp(j(wt+phi))]). Airy's own
  ``exp(-2j*delta)`` / ``exp(-1j*delta)`` terms in
  ``build_waveguide_band_broad_e5_envelope.airy_slab`` are written in the same
  convention: a forward (+x) propagating wave carries spatial phase
  ``exp(-j*beta*x)``, so phase DECREASES as the wave moves away from its
  launch plane. Consistent with this file's producer.
- Reference-plane definition: rfx's ``WaveguideSMatrixResult.reference_planes``
  and this sweep's ``reference_planes_x_m`` place the S-parameter's phase
  origin at ``reference_plane`` (NOT at the port excitation plane, and NOT at
  the slab edge). ``airy_slab(f, eps_r, L, fc_v)`` returns ``s11_e``/``s21_e``
  referenced to the slab's own edges (a unit-amplitude wave arriving exactly
  at the slab). To compare the two, shift by the vacuum propagation between
  each port's reference plane and the near slab edge:
    * ``d_left``  = distance from ``ref_left``  to the slab's LEFT edge
    * ``d_right`` = distance from ``ref_right`` to the slab's RIGHT edge
  Reflection (S11) is a ROUND TRIP through ``d_left`` (there and back), so
      S11_ref = S11_airy * exp(-2j * beta_v * d_left)
  Transmission (S21) is a SINGLE forward pass through the vacuum gap on BOTH
  sides of the slab (d_left entering + d_right leaving), so
      S21_ref = S21_airy * exp(-1j * beta_v * (d_left + d_right))
  **This S21 formula is NOT the one already sitting in
  ``build_waveguide_band_broad_e5_envelope.build_envelope`` (that script uses
  ``exp(+1j*beta_v*slab_length_m)`` — sign AND distance are both wrong for a
  phase claim). That existing term was never phase-checked because
  ``|exp(i*anything)| == 1`` makes it invisible to a magnitude-only gate — it
  is exactly the kind of latent comparator bug the #490 spec warned about.**
  Measured effect of that placeholder if used as the phase reference: ~170-180
  degrees off on EVERY case/band (see ``waveguide_phase_falsifier_and_domain_invariance.py``,
  which promotes this discovered bug to the required planted-defect falsifier
  — it is a REAL bug this session found, not a synthetic one).
- Unwrapping: NOT needed for this per-frequency envelope (11 independent
  frequency points per case, no derivative taken across them) — each phase
  difference is computed via ``angle(S_meas * conj(S_ref))``, which already
  returns the wrapped principal value in (-180, 180] degrees. A genuine
  extractor error this small would never approach the +-180 degree wrap
  boundary, so wrapping ambiguity is not a concern here (contrast with group
  delay in Lane 3, where unwrapping across frequency is load-bearing).
- Masking: mirrors cv11's ``phase_mag_floor`` — bins where ``|S_ref|`` is too
  small (Fabry-Perot near-nulls) have ill-conditioned phase and are excluded.
  All 5 bands were deliberately designed (see
  ``run_waveguide_band_broad_e5_flux_sweep.py`` comments) so slab length keeps
  |S11| away from nulls; the mask is included for robustness/honesty but is
  measured to drop ZERO bins on the committed data (see per-band
  ``masked_bin_count`` in the output).
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_band_broad_e5_envelope import airy_slab, _commit_hash  # noqa: E402

C0 = 299_792_458.0

# Weak-signal mask floor (mirrors cv11's phase_mag_floor=0.30). Bins with
# |S_ref| below this fraction of... no, this is an ABSOLUTE floor on the
# reference magnitude (0..1 scale, same units as |S11|/|S21|), matching cv11.
PHASE_MAG_FLOOR = 0.30

# Measured-envelope tolerance (T2.4-style derivation, NOT a round number):
# worst committed case phase diff across all 5 bands / 20 cases is 11.99 deg
# (WR-28, dx=100um, eps_r=2, S11). 15.0 = 11.99 x ~1.25 margin -- bounded by
# tests/test_waveguide_broad_e5_phase_tolerance_envelope.py (>= worst case,
# <= worst x 1.5) the same way MAX_TOL is bounded for magnitude.
MAX_PHASE_TOL_DEG = 15.0


def _wrapped_phase_diff_deg(measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Wrapped |angle diff| in degrees via the conjugate-product trick."""
    return np.degrees(np.abs(np.angle(measured * np.conj(reference))))


def build_phase_envelope(manifest_path: Path, band_token: str, band_label: str):
    manifest = json.loads(manifest_path.read_text())
    fc_v = float(manifest["fc_te10_hz"])
    ref_left = float(manifest["reference_planes_x_m"][0])
    ref_right = float(manifest["reference_planes_x_m"][1])
    port_left = float(manifest["ports_x_m"][0])
    port_right = float(manifest["ports_x_m"][1])
    slab_center = 0.5 * (port_left + port_right)

    cases_out = []
    diffs_all = []
    total_masked = 0
    total_bins = 0
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
        # CORRECTED S21 reference-plane transform (see module convention
        # note): single forward pass through the vacuum gap on BOTH sides.
        s21_ref = s21_e * np.exp(-1j * beta_v * (d_left + d_right))

        s11_phase_diff = _wrapped_phase_diff_deg(s11, s11_ref)
        s21_phase_diff = _wrapped_phase_diff_deg(s21, s21_ref)

        mask11 = np.abs(s11_ref) >= PHASE_MAG_FLOOR
        mask21 = np.abs(s21_ref) >= PHASE_MAG_FLOOR
        masked_count = int((~mask11).sum() + (~mask21).sum())
        total_masked += masked_count
        total_bins += int(mask11.size + mask21.size)

        if not mask11.any() or not mask21.any():
            raise RuntimeError(
                f"case {case['tag']}: every bin of S11 or S21 fell below "
                f"PHASE_MAG_FLOOR={PHASE_MAG_FLOOR} -- a fully-masked leg would "
                f"silently report a vacuous 0-degree diff, which is worse than "
                f"failing loudly (R5: a mask must not hide the absence of signal)"
            )
        s11_masked = s11_phase_diff[mask11]
        s21_masked = s21_phase_diff[mask21]
        case_max_phase = max(float(s11_masked.max()), float(s21_masked.max()))

        diffs_all.append(case_max_phase)
        cases_out.append({
            "tag": case["tag"],
            "dx_m": float(case["dx_m"]),
            "geometry": case["geometry"],
            "eps_r": case_eps_r,
            "slab_length_m": case_slab_L,
            "n_freqs": int(len(freqs)),
            "s11_phase_max_diff_deg": float(s11_masked.max()),
            "s11_phase_mean_diff_deg": float(s11_masked.mean()),
            "s21_phase_max_diff_deg": float(s21_masked.max()),
            "s21_phase_mean_diff_deg": float(s21_masked.mean()),
            "max_phase_diff_deg": case_max_phase,
            "masked_bin_count": masked_count,
            "s11_ref_mag_min": float(np.abs(s11_ref).min()),
            "s21_ref_mag_min": float(np.abs(s21_ref).min()),
            "rfx_npz": case["rfx_npz"],
            "status": "passed" if case_max_phase <= MAX_PHASE_TOL_DEG else "failed",
        })

    diffs_all = np.array(diffs_all)
    max_across = float(diffs_all.max())
    mean_across = float(diffs_all.mean())
    failed_cases = [c for c in cases_out if c["status"] != "passed"]
    status = "passed" if not failed_cases else "failed"

    envelope = {
        "schema": f"rfx.waveguide_{band_token}_broad_e5_phase_envelope",
        "schema_version": 1,
        "status": status,
        "evidence_level": f"E5-broad-mesh-frequency-geometry-flux-phase-{band_token}",
        "claim": (
            f"rfx {manifest['waveguide']} compute_waveguide_s_matrix(normalize='flux') "
            f"S-parameter PHASE (reference-plane-corrected) versus analytic Airy "
            f"reference over the {manifest['band_hz'][0]/1e9:.1f}-"
            f"{manifest['band_hz'][1]/1e9:.1f} GHz {band_label} band "
            f"{'passes' if status == 'passed' else 'fails'} the broad-E5 phase "
            f"tolerance of {MAX_PHASE_TOL_DEG} degrees."
        ),
        "claim_scope": (
            f"broad rfx {manifest['waveguide']} rectangular_waveguide_port "
            "compute_waveguide_s_matrix(normalize='flux') S11/S21 PHASE (not "
            "magnitude) versus analytic Airy reference, reference-plane-corrected "
            "per the module convention note, spanning the same mesh (dx), "
            "frequency, and geometry (eps_r) axes as the committed magnitude "
            "envelope for this band. Re-analysis of the identical committed "
            "complex S-parameter data used for that envelope -- no new FDTD run."
        ),
        "commit_hash": _commit_hash(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_phase_tol_deg": MAX_PHASE_TOL_DEG,
        "max_phase_tol_deg_provenance": (
            "measured-envelope: worst committed case phase diff 11.99 deg "
            "(WR-28, dx=100um, eps_r=2, S11 leg) x ~1.25 margin"
        ),
        "phase_mag_floor": PHASE_MAG_FLOOR,
        "source_data_provenance": (
            "SAME npz/manifest as the committed magnitude envelope for this band "
            "(gpu-rtx4090, VESSL 369367242914) -- no fresh FDTD run for this "
            "5-band/20-case sweep; phase is a re-analysis of already-captured "
            "complex S-parameters."
        ),
        "convention_note_ref": (
            "see this script's module docstring for the full e^{+jwt}/"
            "reference-plane/unwrapping convention"
        ),
        "primary_reference": {
            "label": "analytic_airy_phase",
            "truth_key": "airy_slab_closed_form_phase",
            "path": "internal_closed_form",
        },
        "envelope_summary": {
            "case_count": len(cases_out),
            "passed_case_count": sum(1 for c in cases_out if c["status"] == "passed"),
            "failed_case_count": len(failed_cases),
            "max_phase_diff_deg_across_cases": max_across,
            "mean_max_phase_diff_deg_across_cases": mean_across,
            "total_masked_bins": total_masked,
            "total_bins_checked": total_bins,
        },
        "diagnostic_note": (
            f"max_phase_diff_deg_across_cases {max_across:.3f} "
            f"(tol {MAX_PHASE_TOL_DEG}); masked {total_masked}/{total_bins} bins "
            f"(phase_mag_floor {PHASE_MAG_FLOOR})."
        ),
        "rfx_manifest_path": str(manifest_path.relative_to(REPO)),
        "cases": cases_out,
    }

    out_path = (
        REPO / "tests" / "fixtures" / "waveguide_broad_e5"
        / f"waveguide_{band_token}_broad_e5_phase_envelope.json"
    )
    out_path.write_text(json.dumps(envelope, indent=2))
    return out_path, envelope


BANDS = {
    "wr28_kaband": ("Ka", "2026-06-15-waveguide-broad-e5-wr28_kaband-flux"),
    "wr62_kuband": ("Ku", "2026-06-15-waveguide-broad-e5-wr62_kuband-flux"),
    "wr15_vband": ("V", "2026-06-15-waveguide-broad-e5-wr15_vband-flux"),
    "wr340_sband": ("S", "2026-06-15-waveguide-broad-e5-wr340_sband-flux"),
    "wr10_wband": ("W", "2026-06-15-waveguide-broad-e5-wr10_wband-flux"),
}


def main():
    for band_token, (band_label, dir_name) in BANDS.items():
        manifest_path = (
            REPO / ".omx" / "physics-gate" / dir_name / "rfx-sweep"
            / f"rfx_{band_token}_flux_sweep_manifest.json"
        )
        out_path, env = build_phase_envelope(manifest_path, band_token, band_label)
        s = env["envelope_summary"]
        print(f"{band_token}: wrote {out_path.relative_to(REPO)}")
        print(f"  status={env['status']}  max_phase_diff_deg={s['max_phase_diff_deg_across_cases']:.3f} "
              f"(tol {MAX_PHASE_TOL_DEG})  masked={s['total_masked_bins']}/{s['total_bins_checked']}")
        for c in env["cases"]:
            print(f"    {c['tag']:20s} max_phase={c['max_phase_diff_deg']:.3f} -> {c['status']}")


if __name__ == "__main__":
    main()
