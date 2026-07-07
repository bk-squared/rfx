#!/usr/bin/env python3
"""Build the cv06b MSL open-stub notch Palace-FEM REFEREE evidence summary.

THE REFEREE QUESTION
--------------------
cv06b's committed rfx-vs-openEMS cross-check (``msl_stub_notch_{rfx,openems}_dx50``)
locks a ~5.8% notch-frequency split: rfx lands at 3.6273 GHz, openEMS at
3.4286 GHz, and the fringing-free analytic quarter-wave sits at 3.69 GHz. The
committed narrative *guessed* that openEMS was closer to the truth ("full-wave
openEMS captures more open-end fringing → lower"). That is a plausibility, not
evidence — and rfx/openEMS are BOTH staircased FDTD, so neither resolves the
open-end fringing exactly. An independent METHOD is needed to referee.

Palace is a frequency-domain FEM solver on a conformal tetrahedral mesh — it
captures the open-end fringing exactly (no staircase), so it is the right
arbiter. It was run on the SAME matched geometry (lossless eps_r=3.66 substrate,
two 50-ohm lumped ports, first-order absorbing far box) at two mesh densities to
show the answer is converged:

    coarse  LC 0.12   143,812 tets / 28,639 nodes   101 pts 2-7 GHz
    mid     LC 0.085  376,802 tets (sqrt2 refine)     33 pts 3.2-4.0 GHz

RESULT
------
Palace lands at ~3.631 GHz (parabolic notch) at BOTH mesh densities (coarse->mid
convergence shift only -0.006 GHz / 0.16%). That is +0.1% from rfx, ~1.6% from
the analytic, and ~5.9% from the openEMS reference — the independent FEM value
is closest to rfx. Our earlier working interpretation ("openEMS captures more
open-end fringing") is revised by this evidence: the conformal FEM includes the
fringing and lands at ~3.63, indicating the fringing correction is a ~1-2%
effect.

FALSIFIER LANES (kept on the record)
------------------------------------
The referee is only trustworthy if it is passive on the matched geometry. Each
mesh carries an 11-pt 2-7 GHz passivity probe; ``check_sparams.py --gate`` fails
closed at max(|S11|^2+|S21|^2) > 1.02. Both meshes stay <= ~0.86.

OOM / SUBMISSION LESSONS
------------------------
Two VESSL lanes died and are recorded in the fixture meta:
  * 369367246150 — sh-syntax death: the VESSL ``run:`` block runs under dash,
    NOT bash; heredocs / bashisms abort it. The shipped YAMLs are dash-safe.
  * 369367246167 — a fine LC/2 mesh OOM'd the 24 GB rtx4090. The sqrt2 "mid"
    mesh is the convergence witness that actually fits.

HOW TO REGENERATE (provenance, not part of the committed fixture)
-----------------------------------------------------------------
  1. mesh:   python scripts/diagnostics/palace_notch_referee/mesh_notch.py
             (+ the LC=0.085 sqrt2 variant for the mid mesh)
  2. solve:  vessl run create -f .../vessl_palace_notch_4090.yaml   (coarse)
             vessl run create -f .../vessl_palace_notch_mid.yaml    (mid)
             -> writes postpro/notch_{full,probe}_{4090,mid}/port-S.csv
  3. fixture: python scripts/diagnostics/build_msl_notch_palace_referee.py \
                 --from-artifacts        # rebuilds the committed JSON from CSVs
  4. verdict: python scripts/diagnostics/build_msl_notch_palace_referee.py

The committed fixture stores the raw Palace port-S arrays (dB -> LINEAR), so the
referee survives a clean checkout WITHOUT re-running Palace: ``build_referee``
re-derives every gated number (notch argmin + log-parabolic refinement, energy
sums, convergence shift, three-way distances) from those arrays.

Usage::

    python scripts/diagnostics/build_msl_notch_palace_referee.py
    python scripts/diagnostics/build_msl_notch_palace_referee.py --from-artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests/fixtures/msl_notch_e4"
_REFEREE_FIXTURE = "msl_stub_notch_palace_referee.json"
_ARTIFACTS = _REPO_ROOT / "scripts/diagnostics/_artifacts/palace_notch/postpro"

# Fringing-free analytic quarter-wave notch (the third referee reference point).
_ANALYTIC_GHZ = 3.69

# Per-mesh Palace CSV provenance: the full 2-7 GHz sweep + the 11-pt passivity
# probe, one directory each under the (gitignored) _artifacts postpro tree.
_CSV_MESHES = {
    "coarse": {"full": "notch_full_4090", "probe": "notch_probe_4090"},
    "mid": {"full": "notch_full_mid", "probe": "notch_probe_mid"},
}

# Short labels for the three-way distance comparison / sides_with verdict.
_REF_LABELS = {"rfx": "rfx", "analytic_3p69": "analytic", "openems": "openems"}


def _energy_max(s11_mag: Any, s21_mag: Any) -> float:
    s11 = np.asarray(s11_mag, dtype=float)
    s21 = np.asarray(s21_mag, dtype=float)
    return float((s11 ** 2 + s21 ** 2).max())


def _n_local_minima(s21_mag: Any) -> int:
    """Count strict interior local minima of |S21| (a clean notch has exactly 1)."""
    s = np.asarray(s21_mag, dtype=float)
    return int(sum(1 for k in range(1, len(s) - 1) if s[k] < s[k - 1] and s[k] < s[k + 1]))


def _notch(freqs: Any, s11_mag: Any, s21_mag: Any) -> dict[str, Any]:
    """Notch bin (argmin |S21|) + log-parabolic vertex refinement.

    i=argmin(|S21|), y=log(|S21|),
    df = 0.5*(y[i-1]-y[i+1]) / (y[i-1]-2*y[i]+y[i+1]) * (f[i+1]-f[i]).
    """
    f = np.asarray(freqs, dtype=float)
    s11 = np.asarray(s11_mag, dtype=float)
    s21 = np.asarray(s21_mag, dtype=float)
    i = int(np.argmin(s21))
    bin_f = float(f[i])
    depth_db = float(20.0 * np.log10(max(float(s21[i]), 1e-300)))
    if 0 < i < len(f) - 1:
        y = np.log(s21)
        denom = float(y[i - 1] - 2.0 * y[i] + y[i + 1])
        df = 0.5 * float(y[i - 1] - y[i + 1]) / denom * float(f[i + 1] - f[i]) if denom != 0.0 else 0.0
        parab_f = bin_f + df
    else:
        parab_f = bin_f
    return {
        "bin_f_ghz": bin_f,
        "parabolic_f_ghz": float(parab_f),
        "depth_db": depth_db,
        "s11_at_notch": float(s11[i]),
    }


def build_referee(fixtures_dir: Path) -> dict[str, Any]:
    """Re-derive every gated referee number from the committed fixture's raw
    Palace arrays (no FDTD, no CSV) plus the two sibling FDTD fixtures.

    Returns per-mesh derived blocks (notch, sweep/probe energy sums, local-min
    count) and the top-level ``referee`` verdict (sides_with, palace mid notch,
    three-way distances, convergence shift)."""
    fixtures_dir = Path(fixtures_dir)
    fix = json.loads((fixtures_dir / _REFEREE_FIXTURE).read_text())
    rfx = json.loads((fixtures_dir / "msl_stub_notch_rfx_dx50.json").read_text())
    oe = json.loads((fixtures_dir / "msl_stub_notch_openems_dx50.json").read_text())

    per_mesh: dict[str, Any] = {}
    for mesh in ("coarse", "mid"):
        blk = fix[mesh]
        per_mesh[mesh] = {
            "notch": _notch(blk["freqs_ghz"], blk["s11_mag"], blk["s21_mag"]),
            "sweep_max_energy_sum": _energy_max(blk["s11_mag"], blk["s21_mag"]),
            "probe_max_energy_sum": _energy_max(
                blk["probe"]["s11_mag"], blk["probe"]["s21_mag"]
            ),
            "n_local_minima": _n_local_minima(blk["s21_mag"]),
        }

    palace_mid = per_mesh["mid"]["notch"]["parabolic_f_ghz"]
    palace_coarse = per_mesh["coarse"]["notch"]["parabolic_f_ghz"]
    shift = palace_mid - palace_coarse

    refs = {
        "rfx": float(rfx["notch"]["f_ghz"]),
        "analytic_3p69": _ANALYTIC_GHZ,
        "openems": float(oe["notch"]["f_ghz"]),
    }
    distances_pct = {
        k: round(abs(palace_mid - v) / v * 100.0, 4) for k, v in refs.items()
    }
    nearest = min(refs, key=lambda k: abs(palace_mid - refs[k]))

    referee = {
        "sides_with": _REF_LABELS[nearest],
        "palace_mid_parabolic_ghz": round(palace_mid, 6),
        "distances_pct": distances_pct,
        "convergence_shift_ghz": round(shift, 6),
    }
    return {"coarse": per_mesh["coarse"], "mid": per_mesh["mid"], "referee": referee}


def _load_palace_csv(path: Path) -> tuple[list[float], list[float], list[float]]:
    """Read a Palace ``port-S.csv`` -> (freqs_ghz, |S11| linear, |S21| linear).

    Palace writes magnitudes in dB; convert to LINEAR (10**(dB/20)), matching
    ``check_sparams.py``."""
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


def _meta() -> dict[str, Any]:
    return {
        "solver": "palace",
        "method": "frequency-domain FEM on a conformal tetrahedral mesh "
        "(independent method vs the FDTD refs)",
        "version": "58f2991-dirty",
        "order": 2,
        "device": "GPU rtx4090",
        "cluster": "remilab-c0",
        "geometry": (
            "matched openEMS dx=50um (l_line=5mm, margin=1mm, stub=12mm, "
            "eps_r=3.66, h_sub=254um, W=600um); Palace conformal-tet "
            "realisation: trace spans the full 7mm x-extent, two 50-ohm lumped "
            "ports (ground->strip) at x=1/6mm"
        ),
        "substrate": "lossless eps_r=3.66 (matches both FDTD refs; LossTan=0)",
        "boundary": "first-order absorbing far box; PEC ground + metal",
        "mesh": {
            "coarse": {
                "lc_mm": 0.12,
                "tets": 143812,
                "nodes": 28639,
                "dof_order2": 968238,
                "n_freqs_sweep": 101,
                "band_ghz": [2.0, 7.0],
            },
            "mid": {
                "lc_mm": 0.085,
                "refinement": "sqrt2",
                "tets": 376802,
                "nodes": None,
                "dof_order2": 2495418,
                "n_freqs_sweep": 33,
                "band_ghz": [3.2, 4.0],
            },
        },
        "vessl_runs": {
            "coarse": "369367246161",
            "mid": "369367246168",
            "failed_lanes": {
                "369367246150": "sh-syntax death (VESSL run-block is dash, not bash; no heredocs)",
                "369367246167": "fine LC/2 mesh OOM on the 24GB rtx4090",
            },
        },
        "probe": "11-pt 2-7 GHz passivity witness (FreqStep 0.5) on the same mesh as each sweep",
        "note": "raw arrays are LINEAR magnitude, converted from Palace dB columns "
        "(10**(dB/20)); referee re-derived by build_msl_notch_palace_referee.py",
    }


def build_fixture_from_artifacts(artifacts_dir: Path, fixtures_dir: Path) -> dict[str, Any]:
    """Rebuild the committed referee fixture JSON from the four Palace CSVs.

    Provenance path only — the CSVs live in the gitignored ``_artifacts`` tree.
    Raises a clear error if the artifacts are absent (clean checkout)."""
    artifacts_dir = Path(artifacts_dir)
    fixtures_dir = Path(fixtures_dir)
    if not artifacts_dir.exists():
        raise SystemExit(
            f"artifacts absent: {artifacts_dir}\n"
            "Regenerate: mesh gen -> vessl_palace_notch_{4090,mid}.yaml -> this "
            "script. --from-artifacts only works where the Palace port-S.csv "
            "files are present (they are gitignored; the committed fixture is not)."
        )

    fixture: dict[str, Any] = {"meta": _meta()}
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
            "notch": _notch(f, s11, s21),
            "max_energy_sum": _energy_max(s11, s21),
            "probe": {
                "freqs_ghz": pf,
                "s11_mag": ps11,
                "s21_mag": ps21,
                "max_energy_sum": _energy_max(ps11, ps21),
            },
        }

    out = fixtures_dir / _REFEREE_FIXTURE
    # Write once so build_referee can read the raw arrays, then inject the
    # re-derived referee section so committed == re-derived by construction.
    out.write_text(json.dumps(fixture, indent=2) + "\n")
    fixture["referee"] = build_referee(fixtures_dir)["referee"]
    out.write_text(json.dumps(fixture, indent=2) + "\n")
    return fixture


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures-dir", default=str(_FIXTURES))
    p.add_argument("--artifacts-dir", default=str(_ARTIFACTS))
    p.add_argument(
        "--from-artifacts",
        action="store_true",
        help="rebuild the committed fixture JSON from the four Palace port-S.csv files",
    )
    args = p.parse_args(argv)

    fixtures_dir = Path(args.fixtures_dir)
    if not fixtures_dir.is_absolute():
        fixtures_dir = _REPO_ROOT / fixtures_dir

    if args.from_artifacts:
        artifacts_dir = Path(args.artifacts_dir)
        if not artifacts_dir.is_absolute():
            artifacts_dir = _REPO_ROOT / artifacts_dir
        build_fixture_from_artifacts(artifacts_dir, fixtures_dir)
        print(f"rebuilt {fixtures_dir / _REFEREE_FIXTURE} from {artifacts_dir}")

    derived = build_referee(fixtures_dir)
    ref = derived["referee"]
    d = ref["distances_pct"]
    print(
        "PALACE REFEREE sides_with={} palace_mid={:.4f}GHz shift={:+.4f}GHz "
        "dist%: rfx={} analytic={} openems={}".format(
            ref["sides_with"],
            ref["palace_mid_parabolic_ghz"],
            ref["convergence_shift_ghz"],
            d["rfx"],
            d["analytic_3p69"],
            d["openems"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
