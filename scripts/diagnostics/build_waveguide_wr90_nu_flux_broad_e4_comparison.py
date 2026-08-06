#!/usr/bin/env python3
"""Broad-E4 external comparison for the NONUNIFORM (graded-dy) WR-90 waveguide port.

The nonuniform waveguide flux lane already has a committed broad-E5 *analytic*
envelope (``tests/fixtures/waveguide_nu_broad_e5/``, vs analytic Airy). Its one
remaining promotion rung — named in that envelope's own gate test — is a
broad-E4 EXTERNAL cross-solver check. This producer closes it: it runs
``compute_waveguide_s_matrix(normalize='flux')`` on a GRADED transverse ``dy``
mesh across the three canonical cv11 geometries (empty / PEC-short / slab) and
compares the rfx magnitudes against a committed independent full-wave reference,
mirroring the uniform-lane producer
``build_waveguide_wr90_rectangular_broad_e4_comparison.py``.

Reference = Palace high-order FEM (``r_h2``), the same physically-converged
reference the uniform broad-E4 fixture uses: Meep res-3/4 gives a non-physical
PEC-short ``|S11|=1.1985>1`` at this resolution (R5/R4), so Palace is the valid
reference for the ``|Gamma|->1`` geometry. Magnitude only (cross-solver phase
conventions differ by 100 deg+ — the cv11 60-deg lesson).

Runs on CPU in ~1 min total (dx=1mm, num_periods settled). The graded mesh
(grading ratio 2.0, adjacent-cell ratio 2:1) is what distinguishes this from the
uniform fixture: it exercises the per-cell graded-dA port-plane weighting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

# The waveguide port uses complex64 DFT accumulators; a global x64 flip causes a
# scan-carry dtype mismatch (same guard cv11 sets). Must precede the jax import.
os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np
import jax.numpy as jnp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402

C0 = 299_792_458.0
A_WG = 0.02286
B_WG = 0.01016
DX = 1.0e-3
DOMAIN_X = 0.200
PORT_LEFT_X = 0.040
PORT_RIGHT_X = 0.160
REF_LEFT_X = 0.050
REF_RIGHT_X = 0.150
PEC_SHORT_X = 0.145
SLAB_EPS = 2.0
SLAB_L = 0.010
FREQS = np.linspace(8.2e9, 12.4e9, 21)

# Absorber depth is DERIVED from the guided wavelength at the LOWEST band
# frequency, not a fixed cell count (the case-19 recipe, CPML_FRACTION 0.75).
# It was a hard-coded 20, which is 0.33*lambda_g at 8.2 GHz — below the repo's
# >=0.5*lambda_g far-port discipline (#496) — and that under-provisioning, not
# the extractor, was producing the PEC-short over-unity |S11|. Measured on this
# exact geometry (lossless PEC short, |S11| must be 1):
#
#     CPML  cells   lambda_g(low)   max|S11|    column power
#       20            0.33          1.01995       1.0403     <- was committed
#       24            0.39          1.00439       1.0088
#       31            0.51          1.00397       1.0080
#       46            0.76          1.00009       1.0002     <- derived value
#
# 46 cells puts the excess at 9e-5, i.e. float32 noise, and the remaining
# bins-above-1 are scattered rather than band-edge concentrated. rfx pads CPML
# OUTSIDE the requested domain, so raising this moves no port and no geometry —
# it only grows the array.
CPML_FRACTION = 0.75
_LAM_G_LOW = (C0 / float(FREQS[0])) / np.sqrt(
    1.0 - (C0 / (2.0 * A_WG) / float(FREQS[0])) ** 2)
CPML = int(np.ceil(CPML_FRACTION * _LAM_G_LOW / DX))
FC_TE10 = C0 / (2 * A_WG)
NUM_PERIODS = 60
GRADING_RATIO = 2.0

# Same per-pair gate as the uniform broad-E4 fixture: the DOCUMENTED cv11
# cross-solver envelope. rfx-vs-analytic-Airy is already pinned <=0.04 by the
# broad-E5 envelope, so the residual here is dominated by the reference's own
# resolution, not rfx — gating tighter than the reference accuracy is
# unprincipled.
# Tolerances DERIVED from the measured envelope via the repo-wide envelope
# multiplier (tests/_gate_policy.py, #528/#539), quantized to 1/1000 because the
# residual is now milli-scale. They were flat 0.10 / 0.07 with no derivation,
# which the absorber fix above leaves 12x / 99x loose:
#
#                          committed   post-#562   + absorber fix
#     summary max           0.07009     0.05322      0.008529
#     summary mean          0.010425    0.004007     0.000709
#     worst PER-PAIR mean   0.0359      0.0143       0.0030   (slab S11)
#
# The tolerances are applied PER PAIR, not to the summary, so the mean envelope
# is the worst per-pair mean (0.0030) and not the summary mean (0.000709). A
# first pass derived it from the summary and the gate immediately failed 1 of 5
# pairs — the tightened gate catching its own author, which is the point of
# tightening it.
#
# Safe to tighten without CI-flakiness risk: the gate test REPLAYS this frozen
# artifact rather than re-running FDTD, so these bounds constrain the next
# REGENERATION rather than a per-run measurement.
def _git_commit() -> str:
    """Short HEAD of the producing tree, or "unknown" outside a checkout."""
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=str(REPO), capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# Derived through the shared helper, not restated as literals (#576 review F5:
# "derived via the repo-wide envelope multiplier" was prose — nothing imported
# _gate_policy, so the arithmetic was right and unenforced). quantum=1000
# because the residual is milli-scale.
from tests._gate_policy import gate_from_envelope  # noqa: E402

_ENVELOPE_MAX_PER_PAIR = 0.008529
_ENVELOPE_MEAN_PER_PAIR = 0.002998
MAX_MAG_ABS_TOL = gate_from_envelope(_ENVELOPE_MAX_PER_PAIR, quantum=1000)
MEAN_MAG_ABS_TOL = gate_from_envelope(_ENVELOPE_MEAN_PER_PAIR, quantum=1000)

# cv11 emits S11 and S21 for these (geometry, components) pairs.
GEOMETRY_COMPONENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("empty", ("S11", "S21")),
    ("pec_short", ("S11",)),
    ("slab", ("S11", "S21")),
)

# External references live in the microwave-energy sibling repo (a gitignored
# lab dataset, same source cv11 loads). Values are embedded in the committed
# artifact, so the fixture stays clean-checkout-replayable; only the PRODUCER
# needs this file present at build time (like the uniform producer's cv11 stdout).
PALACE_REF = (
    REPO.parent
    / "microwave-energy/results/rfx_crossval_wr90_palace/wr90_palace_reference.json"
)


def _graded_dy(total: float, base_dx: float, ratio: float) -> np.ndarray:
    n = int(round(total / base_dx))
    x = np.linspace(-1.0, 1.0, n)
    w = 1.0 + (ratio - 1.0) * np.abs(x)
    return w / w.sum() * total


def _run_geometry(geom: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dy = _graded_dy(A_WG, DX, GRADING_RATIO)
    sim = Simulation(
        freq_max=float(FREQS[-1]) * 1.1,
        domain=(DOMAIN_X, A_WG, B_WG),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=CPML, dx=DX, dy_profile=dy,
    )
    if geom == "slab":
        c = 0.5 * (PORT_LEFT_X + PORT_RIGHT_X)
        sim.add_material("slab", eps_r=SLAB_EPS, sigma=0.0)
        sim.add(Box((c - 0.5 * SLAB_L, 0, 0), (c + 0.5 * SLAB_L, A_WG, B_WG)),
                material="slab")
    elif geom == "pec_short":
        sim.add(Box((PEC_SHORT_X, 0, 0), (PEC_SHORT_X + 2 * DX, A_WG, B_WG)),
                material="pec")
    pf = jnp.asarray(FREQS)
    f0 = float(np.mean(FREQS))
    sim.add_waveguide_port(
        PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE", freqs=pf, f0=f0,
        bandwidth=0.4, waveform="modulated_gaussian", reference_plane=REF_LEFT_X,
        name="left",
    )
    sim.add_waveguide_port(
        PORT_RIGHT_X, direction="-x", mode=(1, 0), mode_type="TE", freqs=pf, f0=f0,
        bandwidth=0.4, waveform="modulated_gaussian", reference_plane=REF_RIGHT_X,
        name="right",
    )
    r = sim.compute_waveguide_s_matrix(num_periods=NUM_PERIODS, normalize="flux")
    s = np.asarray(r.s_params)
    pi = {n: i for i, n in enumerate(r.port_names)}
    s11 = s[pi["left"], pi["left"], :]
    s21 = s[pi["right"], pi["left"], :]
    return np.asarray(r.freqs), s11, s21, float(dy.max() / dy.min())


def _load_palace(geom: str, comp: str) -> np.ndarray:
    d = json.loads(PALACE_REF.read_text())
    block = d["r_h2"][geom]
    key = comp.lower()
    return np.array([complex(a, b) for a, b in block[key]], dtype=np.complex128)


def build(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rfx_s: dict[str, dict[str, np.ndarray]] = {}
    adj_ratio = 0.0
    for geom, _ in GEOMETRY_COMPONENTS:
        fr, s11, s21, adj = _run_geometry(geom)
        adj_ratio = adj
        rfx_s[geom] = {"freqs": fr, "S11": s11, "S21": s21}

    per_pair: list[dict[str, Any]] = []
    all_max: list[float] = []
    all_mean: list[float] = []
    for geom, comps in GEOMETRY_COMPONENTS:
        for comp in comps:
            rfx = rfx_s[geom][comp]
            ref = _load_palace(geom, comp)
            mag_diff = np.abs(np.abs(rfx) - np.abs(ref))
            pmax = float(mag_diff.max())
            pmean = float(mag_diff.mean())
            all_max.append(pmax)
            all_mean.append(pmean)
            per_pair.append({
                "geometry": geom,
                "component": comp,
                "n_freqs": int(rfx_s[geom]["freqs"].size),
                "freq_lo_hz": float(rfx_s[geom]["freqs"].min()),
                "freq_hi_hz": float(rfx_s[geom]["freqs"].max()),
                "rfx_mag_range": [float(np.abs(rfx).min()), float(np.abs(rfx).max())],
                "ref_mag_range": [float(np.abs(ref).min()), float(np.abs(ref).max())],
                "max_mag_abs_diff": pmax,
                "mean_mag_abs_diff": pmean,
                "status": (
                    "passed"
                    if pmax <= MAX_MAG_ABS_TOL and pmean <= MEAN_MAG_ABS_TOL
                    else "failed"
                ),
            })

    geometries = sorted({g for g, _ in GEOMETRY_COMPONENTS})
    failed = [p for p in per_pair if p["status"] != "passed"]
    status = "passed" if not failed else "failed"
    payload: dict[str, Any] = {
        "schema": "rfx.waveguide_wr90_nu_flux_broad_e4_comparison",
        "schema_version": 1,
        "status": status,
        "evidence_level": (
            "E4-broad-external-palace-fem-nonuniform-graded-dy-wr90-multigeometry-te10"
        ),
        "claim": (
            "rfx NONUNIFORM (graded-dy) waveguide_port "
            "compute_waveguide_s_matrix(normalize='flux') magnitude comparison "
            "against Palace_r_h2 (high-order FEM) across the WR-90 empty / "
            "PEC-short / dielectric-slab geometry axis "
            f"{'passes' if status == 'passed' else 'fails'} the broad-E4 magnitude "
            f"tolerance of {MAX_MAG_ABS_TOL}, on a graded transverse mesh "
            f"(grading ratio {GRADING_RATIO}, max/min cell ratio {adj_ratio:.2f}:1)."
        ),
        "claim_scope": (
            "broad external cross-solver magnitude comparison of the rfx "
            "NONUNIFORM graded-dy waveguide_port flux S-matrix against an "
            "independent high-order FEM solver (Palace_r_h2) across the WR-90 "
            "geometry axis (empty guide, PEC short, dielectric slab) over the "
            "X-band frequency grid. Phase conventions are not compared "
            "(magnitude metric). This is the external-solver leg that lifts the "
            "nonuniform waveguide flux lane's last open promotion rung."
        ),
        "r5_reference_note": (
            "Meep res-3/4 gives a non-physical PEC-short |S11|=1.1985>1; the "
            "converged Palace FEM reference (|S11|=1.0000) is used, matching the "
            "uniform-lane broad-E4 fixture. The rfx NU flux |S11| lands "
            "physical/near-unity on PEC-short (witnessed here), so the residual "
            "is reference-side resolution, not an rfx NU defect (R4/R5)."
        ),
        "external_reference_column": "Palace_r_h2",
        "mesh": {
            "kind": "nonuniform_dy_profile_ratio",
            "grading_ratio": GRADING_RATIO,
            "max_min_cell_ratio": adj_ratio,
            "base_dx_m": DX,
        },
        "num_periods": NUM_PERIODS,
        "max_mag_abs_tol": MAX_MAG_ABS_TOL,
        "mean_mag_abs_tol": MEAN_MAG_ABS_TOL,
        # Setup provenance (#576 review F4): the sibling E5 fixture and case 19
        # both record what produced them, and this one recorded neither the
        # absorber depth nor the commit — so a reader could not tell which
        # absorber a number came from, which is exactly the confusion that made
        # the 0.33-lambda_g run look like a solver result.
        "setup": {
            "cpml_layers": int(CPML),
            "cpml_fraction_of_lambda_g_low": round(CPML * DX / _LAM_G_LOW, 4),
            "num_periods": float(NUM_PERIODS),
            "dx_m": float(DX),
            "grading_ratio": float(GRADING_RATIO),
            "commit": _git_commit(),
        },
        "summary": {
            "geometry_count": len(geometries),
            "geometries": geometries,
            "pair_count": len(per_pair),
            "passed_pair_count": sum(1 for p in per_pair if p["status"] == "passed"),
            "failed_pair_count": len(failed),
            "max_mag_abs_diff": float(np.max(all_max)),
            "mean_mag_abs_diff": float(np.mean(all_mean)),
        },
        "pairs": per_pair,
        "generated_at": date.today().isoformat(),
    }
    out_json = output_dir / "waveguide_wr90_nu_flux_broad_e4_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default=".omx/physics-gate/latest-waveguide-wr90-nu-flux-broad-e4",
    )
    args = p.parse_args(argv)
    payload = build(REPO / args.output_dir if not Path(args.output_dir).is_absolute()
                    else Path(args.output_dir))
    s = payload["summary"]
    print(f"status={payload['status']} "
          f"geometries={s['geometry_count']} pairs={s['passed_pair_count']}/{s['pair_count']} "
          f"max_mag_abs_diff={s['max_mag_abs_diff']:.6g} mean={s['mean_mag_abs_diff']:.6g}")
    for pr in payload["pairs"]:
        print(f"  {pr['geometry']:10s} {pr['component']:4s} "
              f"max={pr['max_mag_abs_diff']:.4f} mean={pr['mean_mag_abs_diff']:.4f} "
              f"rfx={pr['rfx_mag_range']} -> {pr['status']}")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
