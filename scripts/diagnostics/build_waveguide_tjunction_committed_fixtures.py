#!/usr/bin/env python3
"""Build the committed T-junction (H-plane, WR-single-mode-TE10) broad-E4/E5
evidence fixtures from the regenerated raw FDTD/MEEP artifacts.

This is an EVIDENCE RECOMMIT, mirroring the coax pattern (PRs #256/#259): the
June-2026 3-port H-plane T-junction validation numbers rode on gitignored
``.omx`` artifacts and were lost. This producer re-derives the two committed
fixtures from the regenerated raw ``.npz`` runs so the evidence survives a clean
checkout — the gate test imports the SAME metric functions defined here, so the
gate and the fixture share one code path (no re-implementation drift).

Two fixtures are written:

  * ``waveguide_tjunction_broad_e5_envelope.json`` — rfx-internal reciprocity /
    passivity / mesh-convergence envelope across the two converged meshes
    (dx=1.0mm / nc=48 and dx=0.667mm / nc=72, both at a FIXED 48mm CPML) over the
    full single-mode TE10 band 5.0-7.0 GHz. Carries the RAW ``S_coarse`` /
    ``S_fine`` magnitude arrays + ``band_hz`` so the gate re-derives every metric
    from the committed numbers (anti-tamper).

  * ``waveguide_tjunction_meep_external_comparison.json`` — rfx |S| vs an
    independent matched far-port MEEP FDTD flux reference (res=4000), across the
    same band. Carries the RAW interpolated MEEP ``M`` array + the rfx ``S_fine``
    it is compared against + ``band_hz``.

Metric definitions are copied verbatim from
``scripts/diagnostics/tj_finalize_converged.py`` (the June finalizer) so the
recommit reproduces the same numbers; the gates are NOT loosened or tightened.

Usage::

    python scripts/diagnostics/build_waveguide_tjunction_committed_fixtures.py \
        --artifacts-dir scripts/diagnostics/_artifacts \
        --out-dir tests/fixtures/waveguide_tjunction_e4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ARTIFACTS = _REPO_ROOT / "scripts" / "diagnostics" / "_artifacts"
_DEFAULT_OUT = _REPO_ROOT / "tests" / "fixtures" / "waveguide_tjunction_e4"

# Canonical converged mesh pair (fixed 48mm CPML). The fine run's file name says
# "0.7" because the producing script formats dx with %.1f; the real dx is
# 0.667mm (read from the npz ``dx_mm`` key for provenance honesty).
_COARSE_NPZ = "tj_farport_dx1.0_nc48.npz"
_FINE_NPZ = "tj_farport_dx0.7_nc72.npz"

# Gates (copied verbatim from tj_finalize_converged.py — DO NOT loosen/tighten).
RECIP_TOL = 0.05
PASSIVITY_TOL = 1.10
CONV_TOL = 0.08
XFDTD_TOL = 0.11


# --------------------------------------------------------------------------- #
# Metric functions — copied EXACTLY from tj_finalize_converged.py.            #
# The gate test imports these so the fixture and the gate share one code path. #
# --------------------------------------------------------------------------- #
def passivity(X: np.ndarray) -> float:
    """max over (drive, freq) of sum_recv |S|**2 (energy sum per drive column)."""
    return float(np.sum(np.asarray(X, dtype=float) ** 2, axis=0).max())


def reciprocity(X: np.ndarray) -> float:
    """max over pairs ((1,0),(2,0),(2,1)) of mean_freq |X[i,j] - X[j,i]|."""
    X = np.asarray(X, dtype=float)
    return float(max(np.mean(np.abs(X[i, j] - X[j, i])) for (i, j) in ((1, 0), (2, 0), (2, 1))))


def mesh_convergence(S_coarse: np.ndarray, S_fine: np.ndarray) -> float:
    """max |S_coarse - S_fine| over the whole (recv, drive, freq) block."""
    return float(np.abs(np.asarray(S_coarse, dtype=float) - np.asarray(S_fine, dtype=float)).max())


def cross_fdtd_max(S_fine: np.ndarray, M: np.ndarray) -> float:
    """max |S_fine - M| (rfx finest mesh vs the MEEP reference)."""
    return float(np.abs(np.asarray(S_fine, dtype=float) - np.asarray(M, dtype=float)).max())


def cross_fdtd_bandmean(S_fine: np.ndarray, M: np.ndarray) -> float:
    """max over (recv, drive) of |mean_freq(S_fine) - mean_freq(M)|."""
    S_fine = np.asarray(S_fine, dtype=float)
    M = np.asarray(M, dtype=float)
    return float(np.abs(np.mean(S_fine, axis=2) - np.mean(M, axis=2)).max())


# --------------------------------------------------------------------------- #
# Raw-artifact loaders.                                                        #
# --------------------------------------------------------------------------- #
def load_mesh_run(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a mesh-run npz -> (S (3,3,N) float |S|, band (N,) Hz, dx_mm)."""
    z = np.load(path)
    S = np.abs(np.asarray(z["S"], dtype=float))
    band = np.asarray(z["band"], dtype=float)
    dx_mm = float(z["dx_mm"])
    return S, band, dx_mm


def build_meep_M(artifacts_dir: Path, band: np.ndarray, res: int) -> np.ndarray:
    """Build the MEEP reference S-matrix M (3,3,N) interpolated onto ``band``.

    Each ``meep_tjunction_farport_r{res}_drive{d}.npz`` holds ``freqs_hz`` (N,)
    and ``col`` (3,N) = |S[:, drive]|. M[recv, drive] = interp(band, freqs, col).
    Mirrors tj_finalize_converged.py exactly.
    """
    N = len(band)
    M = np.zeros((3, 3, N), dtype=float)
    for d in range(3):
        z = np.load(artifacts_dir / f"meep_tjunction_farport_r{res}_drive{d}.npz")
        fm = np.asarray(z["freqs_hz"], dtype=float)
        col = np.abs(np.asarray(z["col"], dtype=float))
        for j in range(3):
            M[j, d] = np.interp(band, fm, col[j])
    return M


def _meep_drives_present(artifacts_dir: Path, res: int) -> bool:
    return all(
        (artifacts_dir / f"meep_tjunction_farport_r{res}_drive{d}.npz").exists()
        for d in range(3)
    )


# --------------------------------------------------------------------------- #
# Provenance.                                                                  #
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, capture_output=True, text=True,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def _provenance(artifacts_dir: Path, meep_res: int, dx_coarse_mm: float,
                dx_fine_mm: float, source_files: list[str]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "regeneration_note": (
            "2026-07 evidence recommit on current main — the June-2026 T-junction "
            "validation numbers were lost with the gitignored .omx artifacts; "
            "regenerated from raw runs and committed (coax #256/#259 pattern)."
        ),
        "commit_hash": _git_commit(),
        "source_artifacts": source_files,
        "setup": {
            "structure": "H-plane 3-port rectangular-waveguide T-junction, single-mode TE10",
            "domain_m": [0.30, 0.24, 0.02],
            "guide_width_W_m": 0.04,
            "band_ghz": [5.0, 7.0],
            "band_points": 11,
            "num_periods": 60,
            "cpml_mm_fixed": 48,
            "mesh_pair": {
                "coarse": {"dx_mm": dx_coarse_mm, "nc": 48},
                "fine": {"dx_mm": dx_fine_mm, "nc": 72},
            },
            "far_port_arms_mm": [90, 90, 70],
            "per_port_reference": (
                "per-port straight-guide PEC references via "
                "extract_waveguide_s_matrix_flux(ref_materials_per_port=...)"
            ),
            "meep_resolution": meep_res,
        },
    }


# --------------------------------------------------------------------------- #
# Fixture builders.                                                            #
# --------------------------------------------------------------------------- #
def build_envelope(artifacts_dir: Path, meep_res: int,
                   coarse_npz: str = _COARSE_NPZ,
                   fine_npz: str = _FINE_NPZ) -> dict[str, Any]:
    """Build the rfx-internal reciprocity/passivity/mesh-convergence envelope."""
    S_coarse, band_c, dx_coarse = load_mesh_run(artifacts_dir / coarse_npz)
    S_fine, band_f, dx_fine = load_mesh_run(artifacts_dir / fine_npz)
    if not np.allclose(band_c, band_f):
        raise ValueError("coarse/fine mesh runs are on different frequency bands")
    band = band_c
    N = len(band)

    conv = mesh_convergence(S_coarse, S_fine)
    cases = []
    for label, dx, S in (("coarse", dx_coarse, S_coarse), ("fine", dx_fine, S_fine)):
        r = reciprocity(S)
        p = passivity(S)
        cases.append({
            "case": f"dx={dx:.3f}mm,CPML=48mm",
            "mesh": label,
            "dx_mm": dx,
            "reciprocity": r,
            "passivity_max": p,
            "reciprocity_pass": bool(r <= RECIP_TOL),
            "passivity_pass": bool(p <= PASSIVITY_TOL),
        })
    env_pass = all(c["reciprocity_pass"] and c["passivity_pass"] for c in cases) and conv <= CONV_TOL

    fghz = "5.0-7.0 GHz"
    setup = ("far-port (>=5 evanescent decay-lengths), 48mm CPML (>~1 lambda_g near "
             "band-center), mesh dx 1.0->0.667mm")
    prov = _provenance(artifacts_dir, meep_res, dx_coarse, dx_fine, [coarse_npz, fine_npz])

    return {
        "schema": "rfx.waveguide_tjunction_broad_e5_envelope",
        "schema_version": 2,
        "status": "passed" if env_pass else "failed",
        "evidence_level": (
            "E5-broad-mesh-frequency-flux-perport-matched-guide-ref-tjunction-"
            "hplane-wr-single-mode-te10-converged"
        ),
        "claim": (
            f"rfx 3-port H-plane T-junction flux S-matrix is reciprocal "
            f"({reciprocity(S_fine):.3f}), passive "
            f"(<= {max(passivity(S_coarse), passivity(S_fine)):.3f}) and "
            f"mesh-convergent ({conv:.3f}) across the full single-mode TE10 band "
            f"{fghz} on a converged mesh refinement axis (dx 1.0->0.667 mm, fixed "
            f"48mm CPML). Regenerated 2026-07 on current main (evidence recommit)."
        ),
        "claim_scope": (
            f"broad rfx rectangular_waveguide_port H-plane T-junction 3-port flux "
            f"S-matrix envelope over the single-mode TE10 frequency axis ({fghz}, "
            f"cutoff ratio 1.33-1.87) and a converged mesh axis (dx 1.0 to 0.667 "
            f"mm) at fixed 48mm CPML; gates reciprocity<=0.05, passivity<=1.10, "
            f"mesh-convergence<=0.08. Setup: {setup}."
        ),
        "gates": {
            "reciprocity_tol": RECIP_TOL,
            "passivity_tol": PASSIVITY_TOL,
            "convergence_tol": CONV_TOL,
        },
        "mesh_convergence_max": conv,
        "cases": cases,
        "per_freq_mesh_conv": {
            f"{band[k] / 1e9:.1f}GHz": float(np.abs(S_coarse[:, :, k] - S_fine[:, :, k]).max())
            for k in range(N)
        },
        "band_hz": band.tolist(),
        "S_coarse": S_coarse.tolist(),
        "S_fine": S_fine.tolist(),
        "provenance": prov,
    }


def build_comparison(artifacts_dir: Path, meep_res: int,
                     coarse_npz: str = _COARSE_NPZ,
                     fine_npz: str = _FINE_NPZ) -> dict[str, Any]:
    """Build the rfx-vs-MEEP external cross-FDTD comparison fixture."""
    S_coarse, band_c, dx_coarse = load_mesh_run(artifacts_dir / coarse_npz)
    S_fine, band_f, dx_fine = load_mesh_run(artifacts_dir / fine_npz)
    if not np.allclose(band_c, band_f):
        raise ValueError("coarse/fine mesh runs are on different frequency bands")
    band = band_c
    N = len(band)

    M = build_meep_M(artifacts_dir, band, meep_res)
    xdev = cross_fdtd_max(S_fine, M)
    xdev_bm = cross_fdtd_bandmean(S_fine, M)
    meep_passiv = passivity(M)
    meep_recip = reciprocity(M)
    cmp_pass = (xdev <= XFDTD_TOL) and (meep_recip <= RECIP_TOL) and (meep_passiv <= PASSIVITY_TOL)

    # Optional MEEP self-drift witness (r2000 vs r4000), informational only.
    self_drift = None
    if meep_res == 4000 and _meep_drives_present(artifacts_dir, 2000):
        M2000 = build_meep_M(artifacts_dir, band, 2000)
        self_drift = float(np.abs(M - M2000).max())

    fghz = "5.0-7.0 GHz"
    prov = _provenance(artifacts_dir, meep_res, dx_coarse, dx_fine,
                       [coarse_npz, fine_npz]
                       + [f"meep_tjunction_farport_r{meep_res}_drive{d}.npz" for d in range(3)])

    out = {
        "schema": "rfx.waveguide_tjunction_meep_external_comparison",
        "schema_version": 2,
        "status": "passed" if cmp_pass else "failed",
        "evidence_level": (
            "E4-broad-external-meep-fdtd-tjunction-hplane-wr-single-mode-te10-converged"
        ),
        "claim": (
            f"rfx 3-port H-plane T-junction |S| agrees with an independent matched-"
            f"geometry MEEP FDTD flux reference (res={meep_res}) to <= {xdev:.3f} "
            f"(band-mean {xdev_bm:.3f}) across the full single-mode TE10 band "
            f"{fghz}; both passive (rfx<={passivity(S_fine):.3f}, "
            f"meep<={meep_passiv:.3f}) and reciprocal. Regenerated 2026-07 on "
            f"current main (evidence recommit)."
        ),
        "claim_scope": (
            f"broad external cross-FDTD comparison of rfx rectangular_waveguide_port "
            f"H-plane T-junction |S| vs an independent matched far-port MEEP flux "
            f"reference (res={meep_res}) over the single-mode TE10 band ({fghz}); "
            f"documented cross-FDTD tolerance {XFDTD_TOL} (two discretized FDTD "
            f"solvers, no closed-form junction truth)."
        ),
        "cross_fdtd_tol": XFDTD_TOL,
        "rfx_vs_meep_max_abs_dev": xdev,
        "rfx_vs_meep_bandmean_max_abs_dev": xdev_bm,
        "meep_passivity_max": meep_passiv,
        "meep_reciprocity": meep_recip,
        "rfx_passivity_max": passivity(S_fine),
        "rfx_reciprocity": reciprocity(S_fine),
        "per_freq_cross_fdtd": {
            f"{band[k] / 1e9:.1f}GHz": float(np.abs(S_fine[:, :, k] - M[:, :, k]).max())
            for k in range(N)
        },
        "rfx_S_bandmean": np.mean(S_fine, axis=2).tolist(),
        "meep_S_bandmean": np.mean(M, axis=2).tolist(),
        "band_hz": band.tolist(),
        "M": M.tolist(),
        "rfx_S_fine": S_fine.tolist(),
        "provenance": prov,
    }
    if self_drift is not None:
        out["meep_self_drift_r2000_vs_r4000"] = self_drift
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts-dir", default=str(_DEFAULT_ARTIFACTS))
    p.add_argument("--out-dir", default=str(_DEFAULT_OUT))
    p.add_argument("--meep-res", type=int, default=4000)
    p.add_argument("--coarse-npz", default=_COARSE_NPZ)
    p.add_argument("--fine-npz", default=_FINE_NPZ)
    args = p.parse_args(argv)

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = _REPO_ROOT / artifacts_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_envelope(artifacts_dir, args.meep_res, args.coarse_npz, args.fine_npz)
    cmp = build_comparison(artifacts_dir, args.meep_res, args.coarse_npz, args.fine_npz)

    (out_dir / "waveguide_tjunction_broad_e5_envelope.json").write_text(
        json.dumps(env, indent=2) + "\n"
    )
    (out_dir / "waveguide_tjunction_meep_external_comparison.json").write_text(
        json.dumps(cmp, indent=2) + "\n"
    )

    print(
        "T-junction fixtures: "
        f"envelope={env['status']} comparison={cmp['status']} | "
        f"recip={env['cases'][-1]['reciprocity']:.3f} "
        f"passivity={max(c['passivity_max'] for c in env['cases']):.3f} "
        f"mesh_conv={env['mesh_convergence_max']:.3f} "
        f"xdev={cmp['rfx_vs_meep_max_abs_dev']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
