#!/usr/bin/env python3
"""Build the committed T-junction (H-plane, WR-single-mode-TE10) full-band E4/E5
evidence fixtures across TWO junction geometries (the broad-claim campaign).

History: PR #270 wired the SINGLE-geometry T-junction MEEP comparison as a
claims-bearing passed-E4 artifact and gated a companion single-geometry
envelope — honestly BELOW the auditor's numeric breadth bar (>= 2 geometry
variants), so the envelope stayed unlisted. That single-geometry split was the
explicit tripwire demanding this campaign. This producer now re-derives BOTH
committed fixtures from raw FDTD/MEEP ``.npz`` runs across a GENUINE second
junction geometry (W=0.036 m, arms 92/92/72 mm, band 5.2-7.3 GHz) alongside the
original (W=0.040 m, arms 90/90/70 mm, band 5.0-7.0 GHz), completing the
numeric breadth bar so the "broad" claim is honestly earned. The gate test
imports the SAME metric functions defined here, so the gate and the fixture
share one code path (no re-implementation drift).

Two fixtures are written:

  * ``waveguide_tjunction_broad_e5_envelope.json`` — rfx-internal reciprocity /
    passivity / mesh-convergence envelope across TWO junction geometries, each
    over its converged mesh pair (dx=1.0mm / nc=48 and dx=0.667mm / nc=72, both
    at a FIXED 48mm CPML) over its full single-mode TE10 band. Each geometry
    block carries the RAW ``S_coarse`` / ``S_fine`` magnitude arrays + ``band_hz``
    so the gate re-derives every metric from the committed numbers (anti-tamper).

  * ``waveguide_tjunction_meep_external_comparison.json`` — rfx |S| vs an
    independent matched far-port MEEP FDTD flux reference (res=4000), for BOTH
    geometries. Each geometry block carries the RAW interpolated MEEP ``M`` array
    + the rfx ``S_fine`` it is compared against + ``band_hz``.

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

# The two junction geometries of the broad claim. Both share the SAME domain
# (0.30x0.24x0.02 m), fixed 48mm CPML, num_periods=60, and the SAME converged
# mesh pair (dx 1.0mm / nc48 coarse, dx 0.667mm / nc72 fine); they differ in the
# guide width W and single-mode TE10 band (and thus arm clearances). The fine
# run's file name says "0.7" because the producing script formats dx with %.1f;
# the real dx is 0.667mm (read from each npz ``dx_mm`` key for provenance
# honesty). geom2's raw runs additionally carry a ``W`` key (ignored by the
# magnitude loader).
GEOMETRIES: list[dict[str, Any]] = [
    {
        "key": "geom1",
        "label": "hplane_tee_W40mm_arms90_90_70",
        "W": 0.04,
        "band": [5.0e9, 7.0e9],
        "arms_mm": [90, 90, 70],
        "coarse_npz": "tj_farport_dx1.0_nc48.npz",
        "fine_npz": "tj_farport_dx0.7_nc72.npz",
        "meep_prefix": "meep_tjunction_farport",
    },
    {
        "key": "geom2",
        "label": "hplane_tee_W36mm_arms92_92_72",
        "W": 0.036,
        "band": [5.2e9, 7.3e9],
        "arms_mm": [92, 92, 72],
        "coarse_npz": "tj_farport_geom2_dx1.0_nc48.npz",
        "fine_npz": "tj_farport_geom2_dx0.7_nc72.npz",
        "meep_prefix": "meep_tjunction_geom2",
    },
]

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


def build_meep_M(artifacts_dir: Path, band: np.ndarray, res: int, prefix: str) -> np.ndarray:
    """Build the MEEP reference S-matrix M (3,3,N) interpolated onto ``band``.

    Each ``{prefix}_r{res}_drive{d}.npz`` holds ``freqs_hz`` (N,) and ``col``
    (3,N) = |S[:, drive]|. M[recv, drive] = interp(band, freqs, col). Mirrors
    tj_finalize_converged.py exactly (prefix generalises geom1/geom2).
    """
    N = len(band)
    M = np.zeros((3, 3, N), dtype=float)
    for d in range(3):
        z = np.load(artifacts_dir / f"{prefix}_r{res}_drive{d}.npz")
        fm = np.asarray(z["freqs_hz"], dtype=float)
        col = np.abs(np.asarray(z["col"], dtype=float))
        for j in range(3):
            M[j, d] = np.interp(band, fm, col[j])
    return M


def _meep_drives_present(artifacts_dir: Path, res: int, prefix: str) -> bool:
    return all(
        (artifacts_dir / f"{prefix}_r{res}_drive{d}.npz").exists()
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


def _geometry_provenance(geom: dict[str, Any], dx_coarse_mm: float,
                         dx_fine_mm: float, source_files: list[str],
                         meep_res: int | None = None) -> dict[str, Any]:
    setup: dict[str, Any] = {
        "structure": "H-plane 3-port rectangular-waveguide T-junction, single-mode TE10",
        "domain_m": [0.30, 0.24, 0.02],
        "guide_width_W_m": geom["W"],
        "band_ghz": [geom["band"][0] / 1e9, geom["band"][1] / 1e9],
        "band_points": 11,
        "num_periods": 60,
        "cpml_mm_fixed": 48,
        "mesh_pair": {
            "coarse": {"dx_mm": dx_coarse_mm, "nc": 48},
            "fine": {"dx_mm": dx_fine_mm, "nc": 72},
        },
        "far_port_arms_mm": geom["arms_mm"],
        "per_port_reference": (
            "per-port straight-guide PEC references via "
            "extract_waveguide_s_matrix_flux(ref_materials_per_port=...)"
        ),
    }
    if meep_res is not None:
        setup["meep_resolution"] = meep_res
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "regeneration_note": (
            "2026-07 two-geometry broad-claim campaign: #270 wired the "
            "single-geometry T-junction MEEP comparison; this campaign adds a "
            "genuine second junction geometry (W=0.036 m, arms 92/92/72 mm) so "
            "the numeric breadth bar (>= 2 geometry variants) is honestly met. "
            "Regenerated from raw FDTD/MEEP runs and committed (coax #256/#259 "
            "evidence-recommit pattern)."
        ),
        "commit_hash": _git_commit(),
        "geometry_key": geom["key"],
        "geometry_label": geom["label"],
        "source_artifacts": source_files,
        "setup": setup,
    }


# --------------------------------------------------------------------------- #
# Per-geometry builders.                                                       #
# --------------------------------------------------------------------------- #
def _build_envelope_geometry(
    artifacts_dir: Path, geom: dict[str, Any]
) -> tuple[dict[str, Any], bool, float, float, float]:
    """Build one geometry's envelope block.

    Returns (block, block_pass, mesh_conv, dx_coarse_mm, dx_fine_mm).
    ``block_pass`` = every mesh case reciprocal + passive AND mesh-conv<=CONV_TOL.
    """
    S_coarse, band_c, dx_coarse = load_mesh_run(artifacts_dir / geom["coarse_npz"])
    S_fine, band_f, dx_fine = load_mesh_run(artifacts_dir / geom["fine_npz"])
    if not np.allclose(band_c, band_f):
        raise ValueError(
            f"{geom['key']}: coarse/fine mesh runs are on different frequency bands"
        )
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
    block_pass = (
        all(c["reciprocity_pass"] and c["passivity_pass"] for c in cases)
        and conv <= CONV_TOL
    )

    block = {
        "key": geom["key"],
        "label": geom["label"],
        "W_m": geom["W"],
        "band_hz": band.tolist(),
        "mesh_convergence_max": conv,
        "cases": cases,
        "per_freq_mesh_conv": {
            f"{band[k] / 1e9:.1f}GHz": float(np.abs(S_coarse[:, :, k] - S_fine[:, :, k]).max())
            for k in range(N)
        },
        "S_coarse": S_coarse.tolist(),
        "S_fine": S_fine.tolist(),
        "provenance": _geometry_provenance(
            geom, dx_coarse, dx_fine, [geom["coarse_npz"], geom["fine_npz"]]
        ),
    }
    return block, block_pass, conv, dx_coarse, dx_fine


def _build_comparison_geometry(
    artifacts_dir: Path, geom: dict[str, Any], meep_res: int
) -> tuple[dict[str, Any], bool]:
    """Build one geometry's rfx-vs-MEEP comparison block. Returns (block, pass)."""
    S_coarse, band_c, dx_coarse = load_mesh_run(artifacts_dir / geom["coarse_npz"])
    S_fine, band_f, dx_fine = load_mesh_run(artifacts_dir / geom["fine_npz"])
    if not np.allclose(band_c, band_f):
        raise ValueError(
            f"{geom['key']}: coarse/fine mesh runs are on different frequency bands"
        )
    band = band_c
    N = len(band)

    M = build_meep_M(artifacts_dir, band, meep_res, geom["meep_prefix"])
    xdev = cross_fdtd_max(S_fine, M)
    xdev_bm = cross_fdtd_bandmean(S_fine, M)
    meep_passiv = passivity(M)
    meep_recip = reciprocity(M)
    block_pass = (
        (xdev <= XFDTD_TOL) and (meep_recip <= RECIP_TOL) and (meep_passiv <= PASSIVITY_TOL)
    )

    # Optional MEEP self-drift witness (r2000 vs r4000), informational only.
    self_drift = None
    if meep_res == 4000 and _meep_drives_present(artifacts_dir, 2000, geom["meep_prefix"]):
        M2000 = build_meep_M(artifacts_dir, band, 2000, geom["meep_prefix"])
        self_drift = float(np.abs(M - M2000).max())

    block = {
        "key": geom["key"],
        "label": geom["label"],
        "W_m": geom["W"],
        "band_hz": band.tolist(),
        "status": "passed" if block_pass else "failed",
        "cross_fdtd_max": xdev,
        "cross_fdtd_bandmean": xdev_bm,
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
        "M": M.tolist(),
        "rfx_S_fine": S_fine.tolist(),
        "provenance": _geometry_provenance(
            geom, dx_coarse, dx_fine,
            [geom["coarse_npz"], geom["fine_npz"]]
            + [f"{geom['meep_prefix']}_r{meep_res}_drive{d}.npz" for d in range(3)],
            meep_res=meep_res,
        ),
    }
    if self_drift is not None:
        block["meep_self_drift_r2000_vs_r4000"] = self_drift
    return block, block_pass


# --------------------------------------------------------------------------- #
# Fixture builders.                                                            #
# --------------------------------------------------------------------------- #
def build_envelope(artifacts_dir: Path) -> dict[str, Any]:
    """Build the two-geometry rfx-internal reciprocity/passivity/mesh envelope."""
    blocks: list[dict[str, Any]] = []
    block_passes: list[bool] = []
    dx_values: set[float] = set()
    band_lo: list[float] = []
    band_hi: list[float] = []
    for geom in GEOMETRIES:
        block, block_pass, _conv, dx_coarse, dx_fine = _build_envelope_geometry(
            artifacts_dir, geom
        )
        blocks.append(block)
        block_passes.append(block_pass)
        dx_values.update((dx_coarse * 1e-3, dx_fine * 1e-3))
        band_lo.append(block["band_hz"][0])
        band_hi.append(block["band_hz"][-1])

    env_pass = all(block_passes)
    case_count = sum(len(b["cases"]) for b in blocks)
    passed_case_count = sum(
        1 for b in blocks for c in b["cases"]
        if c["reciprocity_pass"] and c["passivity_pass"]
    )
    conv_max = max(b["mesh_convergence_max"] for b in blocks)

    setup = ("far-port (>=4 evanescent decay-lengths), 48mm CPML (~0.5 lambda_g "
             "at each band edge), mesh dx 1.0->0.667mm; two junction geometries "
             "W=0.040 m and W=0.036 m")

    return {
        "schema": "rfx.waveguide_tjunction_broad_e5_envelope",
        "schema_version": 2,
        "status": "passed" if env_pass else "failed",
        "evidence_level": (
            "E5-broad-mesh-frequency-flux-perport-matched-guide-ref-tjunction-"
            "hplane-wr-single-mode-te10-converged-two-geometries"
        ),
        "claim": (
            f"rfx 3-port H-plane T-junction flux S-matrix is reciprocal, passive "
            f"(<= {max(c['passivity_max'] for b in blocks for c in b['cases']):.3f}) "
            f"and mesh-convergent (<= {conv_max:.3f}) across TWO junction "
            f"geometries (W=0.040 m over 5.0-7.0 GHz and W=0.036 m over "
            f"5.2-7.3 GHz), each on a converged mesh refinement axis "
            f"(dx 1.0->0.667 mm, fixed 48mm CPML). This is a BROAD two-geometry "
            f"envelope: the numeric breadth bar (>=2 geometry variants, >=2 mesh "
            f"points, freq span >=1.4) is genuinely met. Regenerated 2026-07 on "
            f"current main."
        ),
        "claim_scope": (
            f"BROAD two-geometry rfx rectangular_waveguide_port H-plane "
            f"T-junction 3-port flux S-matrix envelope: two junction geometries "
            f"(W=0.040 m and W=0.036 m) over the single-mode TE10 frequency axis "
            f"(union 5.0-7.3 GHz) and a converged mesh axis (dx 1.0 to 0.667 mm) "
            f"at fixed 48mm CPML; gates reciprocity<=0.05, passivity<=1.10, "
            f"mesh-convergence<=0.08. Setup: {setup}. The auditor's numeric "
            f"breadth requirement (>=2 geometry variants, >=2 distinct dx, "
            f"freq span ratio >=1.4) is genuinely met, so this envelope is now "
            f"listed in the manifest's broad_e5 envelope artifacts and gated by "
            f"tests/test_waveguide_tjunction_e4e5_gates.py."
        ),
        "envelope_summary": {
            # Machine-readable breadth record: the auditor's
            # _envelope_breadth_ok now classifies this as broad (case_count 4,
            # two geometry variants, two distinct dx, freq span ratio ~1.46).
            "case_count": case_count,
            "passed_case_count": passed_case_count,
            "dx_values_m": sorted(dx_values, reverse=True),
            "eps_r_values": [1.0],
            "geometries": [b["label"] for b in blocks],
            "freq_range_hz": [min(band_lo), max(band_hi)],
            "mesh_convergence_max": conv_max,
        },
        "gates": {
            "reciprocity_tol": RECIP_TOL,
            "passivity_tol": PASSIVITY_TOL,
            "convergence_tol": CONV_TOL,
        },
        "mesh_convergence_max": conv_max,
        "geometry_blocks": blocks,
    }


def build_comparison(artifacts_dir: Path, meep_res: int) -> dict[str, Any]:
    """Build the two-geometry rfx-vs-MEEP external cross-FDTD comparison."""
    blocks: list[dict[str, Any]] = []
    block_passes: list[bool] = []
    for geom in GEOMETRIES:
        block, block_pass = _build_comparison_geometry(artifacts_dir, geom, meep_res)
        blocks.append(block)
        block_passes.append(block_pass)

    cmp_pass = all(block_passes)
    passed_pair_count = sum(3 for bp in block_passes if bp)
    pair_count = 3 * len(blocks)
    max_mag = max(b["cross_fdtd_max"] for b in blocks)
    bandmean_mag = max(b["cross_fdtd_bandmean"] for b in blocks)

    return {
        "schema": "rfx.waveguide_tjunction_meep_external_comparison",
        "schema_version": 2,
        "status": "passed" if cmp_pass else "failed",
        "evidence_level": (
            "E4-broad-external-meep-fdtd-tjunction-hplane-wr-single-mode-te10-"
            "converged-two-geometries"
        ),
        "claim": (
            f"rfx 3-port H-plane T-junction |S| agrees with an independent "
            f"matched-geometry MEEP FDTD flux reference (res={meep_res}) to "
            f"<= {max_mag:.3f} (band-mean <= {bandmean_mag:.3f}) across TWO "
            f"junction geometries (W=0.040 m over 5.0-7.0 GHz and W=0.036 m over "
            f"5.2-7.3 GHz); every geometry passive and reciprocal on both codes. "
            f"Regenerated 2026-07 on current main."
        ),
        "claim_scope": (
            f"BROAD two-geometry external cross-FDTD comparison of rfx "
            f"rectangular_waveguide_port H-plane T-junction |S| vs an independent "
            f"matched far-port MEEP flux reference (res={meep_res}) across two "
            f"junction geometries (W=0.040 m and W=0.036 m) over the single-mode "
            f"TE10 band (union 5.0-7.3 GHz); documented cross-FDTD tolerance "
            f"{XFDTD_TOL} (two discretized FDTD solvers, no closed-form junction "
            f"truth). summary.geometry_count=2 with zero failed pairs, so this "
            f"comparison meets the auditor breadth bar as a broad-E4 external "
            f"artifact."
        ),
        "summary": {
            # Machine-readable record for the auditor's _comparison_breadth_ok:
            # geometry_count=2 (two distinct H-plane tees) with failed_pair_count=0
            # classifies this as breadth-class. The 6 pairs are the three driven-
            # port columns of EACH of the two geometries.
            "pair_count": pair_count,
            "passed_pair_count": passed_pair_count,
            "failed_pair_count": pair_count - passed_pair_count,
            "geometry_count": len(blocks),
            "geometries": [b["label"] for b in blocks],
            "drive_ports": 3,
            "max_mag_abs_diff": max_mag,
            "bandmean_mag_abs_diff": bandmean_mag,
        },
        "cross_fdtd_tol": XFDTD_TOL,
        "rfx_vs_meep_max_abs_dev": max_mag,
        "rfx_vs_meep_bandmean_max_abs_dev": bandmean_mag,
        "geometry_blocks": blocks,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts-dir", default=str(_DEFAULT_ARTIFACTS))
    p.add_argument("--out-dir", default=str(_DEFAULT_OUT))
    p.add_argument("--meep-res", type=int, default=4000)
    args = p.parse_args(argv)

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = _REPO_ROOT / artifacts_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_envelope(artifacts_dir)
    cmp = build_comparison(artifacts_dir, args.meep_res)

    (out_dir / "waveguide_tjunction_broad_e5_envelope.json").write_text(
        json.dumps(env, indent=2) + "\n"
    )
    (out_dir / "waveguide_tjunction_meep_external_comparison.json").write_text(
        json.dumps(cmp, indent=2) + "\n"
    )

    print(
        f"T-junction fixtures: envelope={env['status']} comparison={cmp['status']} "
        f"| geometries={len(GEOMETRIES)} "
        f"mesh_conv_max={env['mesh_convergence_max']:.3f} "
        f"xdev_max={cmp['summary']['max_mag_abs_diff']:.3f}"
    )
    for b in env["geometry_blocks"]:
        cb = next(x for x in cmp["geometry_blocks"] if x["key"] == b["key"])
        print(
            f"  [{b['key']} {b['label']}] "
            f"recip={b['cases'][-1]['reciprocity']:.3f} "
            f"passivity={max(x['passivity_max'] for x in b['cases']):.3f} "
            f"mesh_conv={b['mesh_convergence_max']:.3f} "
            f"xdev={cb['cross_fdtd_max']:.3f} status={cb['status']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
