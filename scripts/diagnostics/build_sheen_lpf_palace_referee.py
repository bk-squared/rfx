#!/usr/bin/env python3
"""Build the cv07 Sheen microstrip LPF Palace-FEM REFEREE evidence summary.

THE REFEREE QUESTION (and what the referee actually found)
----------------------------------------------------------
cv07's committed rfx-vs-openEMS cross-check
(``validation/crossval/_07_sheen_results/{rfx,openems}.json``) locks a ~10% split in the
"first S21 null" (defined there as argmin|S21| over 5-15 GHz): rfx 7.218 GHz,
openEMS 7.983 GHz. Both are staircased FDTD; the committed narrative asked whether
one reads the null wrong because it under-resolves the wide-patch open-end / step
fringing. An independent conformal method was needed to referee.

Palace (frequency-domain FEM on a conformal tetrahedral mesh, no staircase) was run
on the SAME matched geometry (the exact domain frame of 07_sheen_lpf.py) at two
mesh densities. IT REVEALS THE PREMISE WAS INCOMPLETE: the Sheen stopband is a
DOUBLE transmission-zero (~7.0 AND ~8.0 GHz), not a single null. Consequences,
re-derived by ``build_referee`` and locked in the ``referee`` block:

  * Palace resolves both zeros (coarse & mid meshes agree -> converged).
  * openEMS resolves the SAME double-zero structure in close agreement with the
    conformal referee (both zeros within <~1%).
  * rfx's coarser mesh + frequency sampling distorts the doublet (a spurious
    extra dip + a shifted/merged central feature), so it does not cleanly
    resolve the two-zero structure.
  * The committed argmin "first null" compares DIFFERENT members of the doublet
    in each solver (rfx's deepest ~7.22, openEMS's deepest ~7.98, Palace's
    deepest ~7.0), so the ~10% "split" is largely a comparator artifact of a
    double-null, NOT a physical single-null disagreement.

``sides_with`` therefore names the FDTD solver whose full stopband STRUCTURE (both
zeros) the conformal referee matches best — the structure-faithful metric — with
the fragile argmin metric reported alongside, explicitly labelled. NO analytic
reference is used: unlike the cv06b open stub, the Sheen stepped-impedance
transmission zeros have no clean fringing-free closed form, so the referee is a
strictly three-SOLVER comparison (rfx / openEMS / Palace).

    coarse  LC 0.25   ~140,039 tets / 27,614 nodes    81 pts 4-12 GHz
    mid     LC 0.18   ~373,388 tets (sqrt2 refine)     51 pts 6-9 GHz

FALSIFIER LANE (kept on the record)
-----------------------------------
The referee is only trustworthy if Palace is passive on the matched geometry.
Each mesh carries an 11-pt 2-12 GHz passivity probe; ``check_sparams.py --gate``
fails closed at max(|S11|^2+|S21|^2) > 1.02. A microstrip LPF driven from a lumped
port radiates from the wide patch (and the first-order ABC absorbs bound-mode
tails), so the energy sum sits WELL BELOW 1 — that means the absolute |S21| LEVELS
are not directly comparable to the FDTD wave-port runs, which is exactly why only
the zero FREQUENCIES are refereed (the crossval likewise gates frequency, not
depth).

HOW TO REGENERATE (provenance, not part of the committed fixture)
-----------------------------------------------------------------
  1. mesh:   python scripts/diagnostics/palace_sheen_referee/mesh_sheen.py \
                 --out .../_artifacts/palace_sheen/palace_sheen.msh
             python .../mesh_sheen.py --lc-min 0.18 --lc-sub 0.21 --lc-max 1.20 \
                 --out .../_artifacts/palace_sheen/palace_sheen_mid.msh
  2. solve:  vessl run create -f .../vessl_palace_sheen_4090.yaml   (coarse)
             vessl run create -f .../vessl_palace_sheen_mid.yaml    (mid)
  3. fixture: python scripts/diagnostics/build_sheen_lpf_palace_referee.py \
                 --from-artifacts --vessl-coarse <id> --vessl-mid <id>
  4. verdict: python scripts/diagnostics/build_sheen_lpf_palace_referee.py

The committed fixture stores the raw Palace port-S arrays (dB -> LINEAR); the
referee survives a clean checkout WITHOUT re-running Palace.

Usage::

    python scripts/diagnostics/build_sheen_lpf_palace_referee.py
    python scripts/diagnostics/build_sheen_lpf_palace_referee.py --from-artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests/fixtures/sheen_lpf_e4"
_REFEREE_FIXTURE = "sheen_lpf_palace_referee.json"
_ARTIFACTS = _REPO_ROOT / "scripts/diagnostics/_artifacts/palace_sheen/postpro"
_SHEEN_RESULTS = _REPO_ROOT / "validation/crossval/_07_sheen_results"

# argmin "first null" search band [GHz] — matches the cv07 compare() default.
_NULL_LO_GHZ, _NULL_HI_GHZ = 5.0, 15.0
# doublet windows [GHz] anchored to the observed two-zero stopband.
_LOWER_WIN = (6.3, 7.5)
_UPPER_WIN = (7.5, 8.6)

_CSV_MESHES = {
    "coarse": {"full": "sheen_full_4090", "probe": "sheen_probe_4090"},
    "mid": {"full": "sheen_full_mid", "probe": "sheen_probe_mid"},
}

_REF_LABELS = {"rfx": "rfx", "openems": "openems"}


def _energy_max(s11_mag: Any, s21_mag: Any) -> float:
    s11 = np.asarray(s11_mag, dtype=float)
    s21 = np.asarray(s21_mag, dtype=float)
    return float((s11 ** 2 + s21 ** 2).max())


def _n_local_minima(s21_mag: Any) -> int:
    s = np.asarray(s21_mag, dtype=float)
    return int(sum(1 for k in range(1, len(s) - 1) if s[k] < s[k - 1] and s[k] < s[k + 1]))


def _min_in_window(freqs: np.ndarray, s11: np.ndarray, s21: np.ndarray,
                   lo: float, hi: float) -> dict[str, Any]:
    """Deepest |S21| bin in [lo,hi] GHz + log-parabolic vertex refinement."""
    band = (freqs >= lo) & (freqs <= hi)
    idx = np.where(band)[0]
    if idx.size == 0:
        idx = np.arange(len(freqs))
    i = int(idx[int(np.argmin(s21[idx]))])
    bin_f = float(freqs[i])
    depth_db = float(20.0 * np.log10(max(float(s21[i]), 1e-300)))
    if 0 < i < len(freqs) - 1:
        y = np.log(s21)
        denom = float(y[i - 1] - 2.0 * y[i] + y[i + 1])
        df = 0.5 * float(y[i - 1] - y[i + 1]) / denom * float(freqs[i + 1] - freqs[i]) if denom != 0.0 else 0.0
        parab_f = bin_f + df
    else:
        parab_f = bin_f
    return {
        "bin_f_ghz": bin_f,
        "parabolic_f_ghz": float(parab_f),
        "depth_db": depth_db,
        "s11_at_min": float(s11[i]),
    }


def _null(freqs: Any, s11_mag: Any, s21_mag: Any) -> dict[str, Any]:
    """The crossval-matching 'first null' = argmin|S21| over the 5-15 GHz band."""
    f = np.asarray(freqs, dtype=float)
    s11 = np.asarray(s11_mag, dtype=float)
    s21 = np.asarray(s21_mag, dtype=float)
    return _min_in_window(f, s11, s21, _NULL_LO_GHZ, _NULL_HI_GHZ)


def _doublet(freqs: Any, s11_mag: Any, s21_mag: Any) -> dict[str, Any]:
    """The two stopband transmission zeros: deepest |S21| in the lower and upper
    windows, each parabolic-refined."""
    f = np.asarray(freqs, dtype=float)
    s11 = np.asarray(s11_mag, dtype=float)
    s21 = np.asarray(s21_mag, dtype=float)
    return {
        "lower": _min_in_window(f, s11, s21, *_LOWER_WIN),
        "upper": _min_in_window(f, s11, s21, *_UPPER_WIN),
    }


def _fdtd_arrays(result_json: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = json.loads(result_json.read_text())
    f = np.asarray(d["freqs_hz"], dtype=float) / 1e9
    return f, np.asarray(d["s11_mag"], dtype=float), np.asarray(d["s21_mag"], dtype=float)


def build_referee(fixtures_dir: Path,
                  sheen_results_dir: Path = _SHEEN_RESULTS) -> dict[str, Any]:
    """Re-derive every gated referee number from the committed fixture's raw
    Palace arrays plus the two sibling cv07 FDTD result JSONs."""
    fixtures_dir = Path(fixtures_dir)
    fix = json.loads((fixtures_dir / _REFEREE_FIXTURE).read_text())

    per_mesh: dict[str, Any] = {}
    for mesh in ("coarse", "mid"):
        blk = fix[mesh]
        per_mesh[mesh] = {
            "null": _null(blk["freqs_ghz"], blk["s11_mag"], blk["s21_mag"]),
            "doublet": _doublet(blk["freqs_ghz"], blk["s11_mag"], blk["s21_mag"]),
            "sweep_max_energy_sum": _energy_max(blk["s11_mag"], blk["s21_mag"]),
            "probe_max_energy_sum": _energy_max(
                blk["probe"]["s11_mag"], blk["probe"]["s21_mag"]
            ),
            "n_local_minima": _n_local_minima(blk["s21_mag"]),
        }

    # Palace reference doublet = the mid (finer) mesh, parabolic.
    p_lo = per_mesh["mid"]["doublet"]["lower"]["parabolic_f_ghz"]
    p_hi = per_mesh["mid"]["doublet"]["upper"]["parabolic_f_ghz"]
    shift_lo = p_lo - per_mesh["coarse"]["doublet"]["lower"]["parabolic_f_ghz"]
    shift_hi = p_hi - per_mesh["coarse"]["doublet"]["upper"]["parabolic_f_ghz"]

    # Each FDTD solver's doublet + argmin, re-derived from its committed arrays.
    fdtd: dict[str, Any] = {}
    for tag in ("rfx", "openems"):
        f, s11, s21 = _fdtd_arrays(sheen_results_dir / f"{tag}.json")
        d = {
            "lower_ghz": round(_min_in_window(f, s11, s21, *_LOWER_WIN)["parabolic_f_ghz"], 6),
            "upper_ghz": round(_min_in_window(f, s11, s21, *_UPPER_WIN)["parabolic_f_ghz"], 6),
            "argmin_first_null_ghz": round(_min_in_window(f, s11, s21, _NULL_LO_GHZ, _NULL_HI_GHZ)["parabolic_f_ghz"], 6),
        }
        fdtd[tag] = d

    # STRUCTURE distance: how well each FDTD solver matches BOTH Palace zeros.
    # Use the WORST (max) of the two per-zero % errors -> a solver must match the
    # whole doublet to score well.
    structure_pct: dict[str, float] = {}
    for tag in ("rfx", "openems"):
        e_lo = abs(fdtd[tag]["lower_ghz"] - p_lo) / p_lo * 100.0
        e_hi = abs(fdtd[tag]["upper_ghz"] - p_hi) / p_hi * 100.0
        structure_pct[tag] = round(max(e_lo, e_hi), 4)
    nearest_struct = min(structure_pct, key=lambda k: structure_pct[k])

    # ARGMIN metric (fragile / metric-dependent) — kept for transparency.
    palace_argmin = per_mesh["mid"]["null"]["parabolic_f_ghz"]
    argmin_dist = {
        tag: round(abs(palace_argmin - fdtd[tag]["argmin_first_null_ghz"])
                   / fdtd[tag]["argmin_first_null_ghz"] * 100.0, 4)
        for tag in ("rfx", "openems")
    }
    argmin_nearest = min(argmin_dist, key=lambda k: argmin_dist[k])

    referee = {
        "finding": "stopband is a DOUBLE transmission-zero (~7 & ~8 GHz); the "
                   "committed single 'first null' argmin compares different "
                   "doublet members per solver",
        "sides_with": _REF_LABELS[nearest_struct],
        "sides_with_metric": "stopband structure (both zeros); worst-zero % error",
        "palace_doublet_mid_ghz": {"lower": round(p_lo, 6), "upper": round(p_hi, 6)},
        "fdtd_doublet_ghz": fdtd,
        "structure_distance_pct": structure_pct,
        "convergence_shift_ghz": {"lower": round(shift_lo, 6), "upper": round(shift_hi, 6)},
        "argmin_first_null": {
            "note": "metric-dependent: argmin picks the DEEPER member of the "
                    "doublet, which differs by solver; do not read as a physical "
                    "single-null disagreement",
            "palace_mid_ghz": round(palace_argmin, 6),
            "distances_pct": argmin_dist,
            "nearest": _REF_LABELS[argmin_nearest],
        },
    }
    return {"coarse": per_mesh["coarse"], "mid": per_mesh["mid"], "referee": referee}


def _load_palace_csv(path: Path) -> tuple[list[float], list[float], list[float]]:
    """Read a Palace ``port-S.csv`` -> (freqs_ghz, |S11| linear, |S21| linear)."""
    rows = list(csv.reader(path.open()))
    hdr = [h.strip() for h in rows[0]]

    def col(sub: str) -> int:
        return next(i for i, h in enumerate(hdr) if sub in h)

    fi, s11i, s21i = col("f (GHz)"), col("|S[1][1]|"), col("|S[2][1]|")
    freqs: list[float] = []
    s11: list[float] = []
    s21: list[float] = []
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        freqs.append(float(r[fi]))
        s11.append(10.0 ** (float(r[s11i]) / 20.0))
        s21.append(10.0 ** (float(r[s21i]) / 20.0))
    return freqs, s11, s21


def _meta(vessl_runs: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "solver": "palace",
        "method": "frequency-domain FEM on a conformal tetrahedral mesh "
        "(independent method vs the two FDTD refs)",
        "order": 2,
        "device": "GPU rtx4090",
        "cluster": "remilab-c0",
        "geometry": (
            "cv07 Sheen 1990 LPF, exact domain frame of validation/crossval/"
            "07_sheen_lpf.py: eps_r=2.2 h=0.794mm substrate, 50-ohm feeds "
            "W=2.413mm, 20.320x2.540mm wide low-Z patch, domain "
            "27.472x26.320x3.794mm; two 50-ohm lumped ports (ground->strip, +Z) "
            "at x=2.5 and x=24.972mm"
        ),
        "substrate": "lossless eps_r=2.2 (matches both FDTD refs; LossTan=0)",
        "boundary": "first-order absorbing far box; PEC ground + metal",
        "port_note": "lumped ports radiate/leak (energy sum well below 1); absolute "
        "|S21| levels are NOT comparable to the FDTD wave-port runs -- only the "
        "zero FREQUENCIES are refereed",
        "mesh": {
            "coarse": {
                "lc_mm": 0.25,
                "tets": 140039,
                "nodes": 27614,
                "dof_order2": 924257,
                "n_freqs_sweep": 81,
                "band_ghz": [4.0, 12.0],
            },
            "mid": {
                "lc_mm": 0.18,
                "refinement": "sqrt2",
                "tets": 373388,
                "nodes": 70058,
                "dof_order2": 2464361,
                "n_freqs_sweep": 51,
                "band_ghz": [6.0, 9.0],
            },
        },
        "vessl_runs": vessl_runs or {},
        "probe": "11-pt 2-12 GHz passivity witness (FreqStep 1.0) on the same mesh as each sweep",
        "note": "raw arrays are LINEAR magnitude, converted from Palace dB columns "
        "(10**(dB/20)); referee re-derived by build_sheen_lpf_palace_referee.py",
        "analytic_ref": "n/a — the Sheen stepped-impedance transmission zeros have "
        "no clean fringing-free closed form; referee is a three-SOLVER comparison",
    }


def build_fixture_from_artifacts(artifacts_dir: Path, fixtures_dir: Path,
                                 vessl_runs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rebuild the committed referee fixture JSON from the four Palace CSVs."""
    artifacts_dir = Path(artifacts_dir)
    fixtures_dir = Path(fixtures_dir)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    if not artifacts_dir.exists():
        raise SystemExit(
            f"artifacts absent: {artifacts_dir}\n"
            "Regenerate: mesh gen -> vessl_palace_sheen_{4090,mid}.yaml -> this "
            "script. --from-artifacts only works where the Palace port-S.csv "
            "files are present (they are gitignored; the committed fixture is not)."
        )

    fixture: dict[str, Any] = {"meta": _meta(vessl_runs)}
    for mesh, names in _CSV_MESHES.items():
        full_csv = artifacts_dir / names["full"] / "port-S.csv"
        probe_csv = artifacts_dir / names["probe"] / "port-S.csv"
        for p in (full_csv, probe_csv):
            if not p.exists():
                raise SystemExit(f"missing Palace CSV: {p}")
        f, s11, s21 = _load_palace_csv(full_csv)
        pf, ps11, ps21 = _load_palace_csv(probe_csv)
        fixture[mesh] = {
            "freqs_ghz": f,
            "s11_mag": s11,
            "s21_mag": s21,
            "null": _null(f, s11, s21),
            "doublet": _doublet(f, s11, s21),
            "max_energy_sum": _energy_max(s11, s21),
            "probe": {
                "freqs_ghz": pf,
                "s11_mag": ps11,
                "s21_mag": ps21,
                "max_energy_sum": _energy_max(ps11, ps21),
            },
        }

    out = fixtures_dir / _REFEREE_FIXTURE
    out.write_text(json.dumps(fixture, indent=2) + "\n")
    fixture["referee"] = build_referee(fixtures_dir)["referee"]
    out.write_text(json.dumps(fixture, indent=2) + "\n")
    return fixture


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures-dir", default=str(_FIXTURES))
    p.add_argument("--artifacts-dir", default=str(_ARTIFACTS))
    p.add_argument("--from-artifacts", action="store_true",
                   help="rebuild the committed fixture JSON from the four Palace port-S.csv files")
    p.add_argument("--vessl-coarse", default=None)
    p.add_argument("--vessl-mid", default=None)
    args = p.parse_args(argv)

    fixtures_dir = Path(args.fixtures_dir)
    if not fixtures_dir.is_absolute():
        fixtures_dir = _REPO_ROOT / fixtures_dir

    if args.from_artifacts:
        artifacts_dir = Path(args.artifacts_dir)
        if not artifacts_dir.is_absolute():
            artifacts_dir = _REPO_ROOT / artifacts_dir
        vessl_runs = None
        if args.vessl_coarse or args.vessl_mid:
            vessl_runs = {"coarse": args.vessl_coarse, "mid": args.vessl_mid}
        build_fixture_from_artifacts(artifacts_dir, fixtures_dir, vessl_runs)
        print(f"rebuilt {fixtures_dir / _REFEREE_FIXTURE} from {artifacts_dir}")

    ref = build_referee(fixtures_dir)["referee"]
    pd = ref["palace_doublet_mid_ghz"]
    fd = ref["fdtd_doublet_ghz"]
    print("PALACE SHEEN REFEREE (double transmission-zero stopband)")
    print(f"  Palace doublet (mid): lower={pd['lower']:.3f} GHz  upper={pd['upper']:.3f} GHz")
    print(f"  convergence shift    : lower={ref['convergence_shift_ghz']['lower']:+.4f}  "
          f"upper={ref['convergence_shift_ghz']['upper']:+.4f} GHz")
    for tag in ("rfx", "openems"):
        print(f"  {tag:8s} doublet: lower={fd[tag]['lower_ghz']:.3f}  upper={fd[tag]['upper_ghz']:.3f}  "
              f"(argmin {fd[tag]['argmin_first_null_ghz']:.3f})  structure Δ={ref['structure_distance_pct'][tag]:.2f}%")
    print(f"  => sides_with (structure) = {ref['sides_with']}")
    am = ref["argmin_first_null"]
    print(f"  [argmin metric, fragile] palace={am['palace_mid_ghz']:.3f} "
          f"dist%: rfx={am['distances_pct']['rfx']} openems={am['distances_pct']['openems']} "
          f"-> nearest {am['nearest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
