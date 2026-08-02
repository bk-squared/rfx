"""Issue #490 Lane 1 mandate: planted-defect falsifier + domain-size invariance.

Two separate checks required before the phase envelope
(``build_waveguide_band_broad_e5_phase_envelope.py``) counts as gated evidence:

1. FALSIFIER (no new FDTD run -- pure re-analysis of the committed npz data):
   recompute the phase-diff gate metric using the WRONG S21 reference-plane
   formula that was sitting, unexercised, in
   ``build_waveguide_band_broad_e5_envelope.build_envelope`` before this
   session (``S21_ref = S21_airy * exp(+1j*beta_v*slab_length_m)`` -- wrong
   sign AND wrong distance; discovered while deriving the phase convention
   note, see that script's docstring). This is a REAL bug this session found,
   not a synthetic one, which makes it a stronger falsifier than an invented
   perturbation: the gate must fail hard against it.
2. DOMAIN-SIZE INVARIANCE WITNESS (ONE fresh FDTD run -- the only new
   simulation this lane performs): the committed WR-340 dx=1500um eps_r=2.0
   case is re-run with the domain grown by +100 mm total (+50 mm buffer on
   each side beyond the ports; CPML layer count, dx, slab length, and port-to-
   reference-plane offsets held fixed) and the phase-gate verdict (max phase
   diff <= MAX_PHASE_TOL_DEG) must not flip relative to the committed
   (smaller-domain) case. A spurious phase artifact tied to the specific
   domain size (e.g. a CPML-reflection standing wave) would move under this
   growth; the physical Airy-vs-rfx residual should not.
"""
from __future__ import annotations
import dataclasses
import json
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_band_broad_e5_envelope import airy_slab  # noqa: E402
from build_waveguide_band_broad_e5_phase_envelope import (  # noqa: E402
    MAX_PHASE_TOL_DEG, PHASE_MAG_FLOOR, _wrapped_phase_diff_deg,
)
from run_waveguide_band_broad_e5_flux_sweep import BANDS, build_sim  # noqa: E402

C0 = 299_792_458.0

WR340_DIR = REPO / ".omx/physics-gate/2026-06-15-waveguide-broad-e5-wr340_sband-flux/rfx-sweep"


def _case_max_phase(s11, s21, s11_ref, s21_ref):
    m11 = np.abs(s11_ref) >= PHASE_MAG_FLOOR
    m21 = np.abs(s21_ref) >= PHASE_MAG_FLOOR
    d11 = _wrapped_phase_diff_deg(s11, s11_ref)[m11]
    d21 = _wrapped_phase_diff_deg(s21, s21_ref)[m21]
    d11 = d11 if d11.size else np.array([0.0])
    d21 = d21 if d21.size else np.array([0.0])
    return max(float(d11.max()), float(d21.max()))


def run_falsifier():
    """Re-derive the 20-case phase gate with the WRONG (pre-session) S21 ref."""
    worst = 0.0
    per_case = []
    for band_token, dir_name in [
        ("wr28_kaband", "2026-06-15-waveguide-broad-e5-wr28_kaband-flux"),
        ("wr62_kuband", "2026-06-15-waveguide-broad-e5-wr62_kuband-flux"),
        ("wr15_vband", "2026-06-15-waveguide-broad-e5-wr15_vband-flux"),
        ("wr340_sband", "2026-06-15-waveguide-broad-e5-wr340_sband-flux"),
        ("wr10_wband", "2026-06-15-waveguide-broad-e5-wr10_wband-flux"),
    ]:
        mdir = REPO / ".omx/physics-gate" / dir_name / "rfx-sweep"
        m = json.loads((mdir / f"rfx_{band_token}_flux_sweep_manifest.json").read_text())
        fc = float(m["fc_te10_hz"])
        rl = float(m["reference_planes_x_m"][0])
        pl = float(m["ports_x_m"][0])
        pr = float(m["ports_x_m"][1])
        sc = 0.5 * (pl + pr)
        for case in m["cases"]:
            data = np.load(REPO / case["rfx_npz"])
            freqs = data["freqs_hz"]
            s11 = data["s11"]; s21 = data["s21"]
            eps_r = float(data["eps_r"]); L = float(data["slab_length_m"])
            s11_e, s21_e = airy_slab(freqs, eps_r, L, fc)
            beta_v = (2 * np.pi * freqs / C0) * np.sqrt(1.0 - (fc / freqs) ** 2)
            d_left = sc - 0.5 * L - rl
            s11_ref = s11_e * np.exp(-2j * beta_v * d_left)
            # THE BUG: wrong sign, wrong distance (slab_length_m, not
            # d_left+d_right) -- exactly what shipped, unexercised, in the
            # magnitude-only envelope builder.
            s21_ref_wrong = s21_e * np.exp(+1j * beta_v * L)
            case_max = _case_max_phase(s11, s21, s11_ref, s21_ref_wrong)
            per_case.append({"band": band_token, "tag": case["tag"], "max_phase_diff_deg": case_max})
            worst = max(worst, case_max)

    result = {
        "falsifier": "wrong_s21_reference_plane_formula (pre-session bug: "
                     "exp(+1j*beta_v*slab_length_m) instead of "
                     "exp(-1j*beta_v*(d_left+d_right))",
        "gate_tol_deg": MAX_PHASE_TOL_DEG,
        "worst_case_phase_diff_deg": worst,
        "gate_reds": worst > MAX_PHASE_TOL_DEG,
        "per_case": per_case,
        "n_cases": len(per_case),
    }
    return result


def run_domain_invariance():
    """ONE fresh FDTD run: WR-340 dx=1500um eps_r=2.0 with the domain grown."""
    baseline_npz = WR340_DIR / "dx1500_slab_er2.npz"
    baseline = np.load(baseline_npz)
    freqs = baseline["freqs_hz"]
    fc_v = 0.0  # filled below from the manifest, kept explicit for clarity
    manifest = json.loads((WR340_DIR / "rfx_wr340_sband_flux_sweep_manifest.json").read_text())
    fc_v = float(manifest["fc_te10_hz"])
    ref_left_base = float(manifest["reference_planes_x_m"][0])
    ref_right_base = float(manifest["reference_planes_x_m"][1])
    port_left_base = float(manifest["ports_x_m"][0])
    port_right_base = float(manifest["ports_x_m"][1])
    slab_L = float(manifest["slab_length_m"])
    slab_center_base = 0.5 * (port_left_base + port_right_base)

    # baseline (committed) residual, for comparison
    s11_e, s21_e = airy_slab(freqs, 2.0, slab_L, fc_v)
    beta_v = (2 * np.pi * freqs / C0) * np.sqrt(1.0 - (fc_v / freqs) ** 2)
    d_left_base = slab_center_base - 0.5 * slab_L - ref_left_base
    d_right_base = ref_right_base - (slab_center_base + 0.5 * slab_L)
    s11_ref_base = s11_e * np.exp(-2j * beta_v * d_left_base)
    s21_ref_base = s21_e * np.exp(-1j * beta_v * (d_left_base + d_right_base))
    baseline_max = _case_max_phase(baseline["s11"], baseline["s21"], s11_ref_base, s21_ref_base)

    # Grown domain: +50 mm buffer on each side beyond the ports (400mm -> 500mm
    # total domain). Port-to-reference-plane offset (20mm) and slab length/
    # position relative to the port midpoint are held fixed, so the ONLY
    # change is more vacuum/CPML clearance -- a benign domain growth.
    spec = BANDS["WR340"]
    grown = dataclasses.replace(
        spec,
        domain_x_m=spec.domain_x_m + 100e-3,
        port_left_x_m=spec.port_left_x_m + 50e-3,
        port_right_x_m=spec.port_right_x_m + 50e-3,
        ref_left_x_m=spec.ref_left_x_m + 50e-3,
        ref_right_x_m=spec.ref_right_x_m + 50e-3,
    )
    t0 = time.time()
    sim = build_sim(grown, grown.dx_values_m[1], 2.0, freqs)
    result = sim.compute_waveguide_s_matrix(num_periods=grown.num_periods, normalize="flux")
    dt = time.time() - t0
    s = np.asarray(result.s_params)
    pi = {n: i for i, n in enumerate(result.port_names)}
    s11_grown = s[pi["left"], pi["left"], :]
    s21_grown = s[pi["right"], pi["left"], :]

    slab_center_grown = 0.5 * (grown.port_left_x_m + grown.port_right_x_m)
    d_left_grown = slab_center_grown - 0.5 * slab_L - grown.ref_left_x_m
    d_right_grown = grown.ref_right_x_m - (slab_center_grown + 0.5 * slab_L)
    assert abs(d_left_grown - d_left_base) < 1e-9 and abs(d_right_grown - d_right_base) < 1e-9, (
        "grown-domain geometry did not preserve the port/slab offsets -- "
        "invariance witness is not testing a benign change"
    )
    s11_ref_grown = s11_e * np.exp(-2j * beta_v * d_left_grown)
    s21_ref_grown = s21_e * np.exp(-1j * beta_v * (d_left_grown + d_right_grown))
    grown_max = _case_max_phase(s11_grown, s21_grown, s11_ref_grown, s21_ref_grown)

    return {
        "baseline_domain_m": spec.domain_x_m,
        "grown_domain_m": grown.domain_x_m,
        "baseline_max_phase_diff_deg": baseline_max,
        "grown_max_phase_diff_deg": grown_max,
        "gate_tol_deg": MAX_PHASE_TOL_DEG,
        "baseline_verdict": "passed" if baseline_max <= MAX_PHASE_TOL_DEG else "failed",
        "grown_verdict": "passed" if grown_max <= MAX_PHASE_TOL_DEG else "failed",
        "verdict_flipped": (baseline_max <= MAX_PHASE_TOL_DEG) != (grown_max <= MAX_PHASE_TOL_DEG),
        "wallclock_s": dt,
    }


def main():
    falsifier = run_falsifier()
    print(f"[falsifier] worst_case_phase_diff_deg={falsifier['worst_case_phase_diff_deg']:.2f} "
          f"(tol {falsifier['gate_tol_deg']}) gate_reds={falsifier['gate_reds']} "
          f"n_cases={falsifier['n_cases']}")

    invariance = run_domain_invariance()
    print(f"[invariance] baseline={invariance['baseline_max_phase_diff_deg']:.3f} deg "
          f"({invariance['baseline_verdict']})  grown={invariance['grown_max_phase_diff_deg']:.3f} deg "
          f"({invariance['grown_verdict']})  flipped={invariance['verdict_flipped']} "
          f"wallclock={invariance['wallclock_s']:.1f}s")

    out = {
        "schema": "rfx.waveguide_broad_e5_phase_falsifier_and_invariance",
        "schema_version": 1,
        "falsifier": falsifier,
        "domain_invariance": invariance,
    }
    out_path = REPO / "tests/fixtures/waveguide_broad_e5/phase_falsifier_and_domain_invariance.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
