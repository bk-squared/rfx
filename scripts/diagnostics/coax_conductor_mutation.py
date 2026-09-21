#!/usr/bin/env python3
"""Put the old conductor realization back and run the oracle's own comparisons.

The (b) falsifier for ``tests/oracle/test_coax_conductor_realization.py``: a
gate that only proves the check can be switched off proves nothing, so this
revives the DEFECT — ``sigma = PEC_SIGMA`` per node instead of PEC edge masks —
with every helper the lane calls left exactly where it is, and reports what the
oracle's comparisons then read.

The mutation is two rebindings and no edit to shipped code:

* ``rfx.sources.coaxial_port.stamp_coaxial_line`` is wrapped so that, after the
  shipped stamper returns, the conductor cells it reported are written back into
  ``materials.sigma`` as ``PEC_SIGMA`` — the pre-fix material array.
* ``rfx.sparams.coax._coax_pec_edge_masks`` is replaced by one that returns
  ``None`` (and passes a merge target through unchanged), so the lane hands the
  runner no edge masks — the pre-fix call.

Everything else — the geometry, the feeds, the probes, the drive, the step
count — is the oracle's own.

Usage::

    PYTHONPATH=. python scripts/diagnostics/coax_conductor_mutation.py \
        --out <run-dir> [--run-id <id>]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import rfx  # noqa: E402
import rfx.sources.coaxial_port as _cp  # noqa: E402
import rfx.sparams.coax as _coax  # noqa: E402
from rfx.sources.coaxial_port import (  # noqa: E402
    PEC_SIGMA, PTFE_EPS_R, SMA_OUTER_RADIUS, SMA_PIN_RADIUS,
    coaxial_tem_characteristic_impedance,
)

sys.path.insert(0, str(REPO / "tests" / "oracle"))
from test_coax_conductor_realization import (  # noqa: E402
    BETA_FRAC, COLUMN_POWER_MAX, C0, FREQS, LOADS_OHM, LOAD_DOMAIN, LOAD_PROBES,
    LOAD_STEPS, THRU_DOMAIN, THRU_PROBES, THRU_STEPS, Z0_FRAC,
    _beta_from_s21_phase, _sim,
)

_ORIGINAL_STAMP = _cp.stamp_coaxial_line
_ORIGINAL_EDGES = _coax._coax_pec_edge_masks


def _stamp_as_sigma(grid, materials, **kw):
    """The shipped stamper, then the conductor written back into sigma."""
    materials, shell_inner, cells = _ORIGINAL_STAMP(grid, materials, **kw)
    sig = np.asarray(materials.sigma)
    sig = np.where(np.asarray(cells), float(PEC_SIGMA), sig)
    return materials._replace(sigma=jnp.asarray(sig)), shell_inner, cells


def _no_edges(pec_cells, periodic=(False, False, False), merge_with=None):
    """No edge masks, which is what the lane passed before the fix."""
    return merge_with


def install() -> None:
    _cp.stamp_coaxial_line = _stamp_as_sigma
    _coax.stamp_coaxial_line = _stamp_as_sigma
    _coax._coax_pec_edge_masks = _no_edges


def measure() -> dict:
    declared_z = coaxial_tem_characteristic_impedance(
        SMA_PIN_RADIUS, SMA_OUTER_RADIUS, float(PTFE_EPS_R))
    out: dict = {
        "what": ("the oracle's own comparisons, with the sigma-per-node "
                 "conductor realization put back and every helper call left "
                 "in place"),
        "bars": {"beta_frac": BETA_FRAC, "column_power_max": COLUMN_POWER_MAX,
                 "z0_frac": Z0_FRAC},
        "declared_z_tem_ohm": declared_z,
    }

    res = _sim(THRU_DOMAIN).compute_coaxial_two_port(
        n_steps=THRU_STEPS, freqs=FREQS, **THRU_PROBES)
    freqs = np.asarray(res.freqs, dtype=float)
    S = np.asarray(res.s_params)
    planes = np.asarray(res.reference_planes, dtype=float)
    l12 = float(abs(planes[0] - planes[1]))
    beta_analytic = 2.0 * np.pi * freqs * math.sqrt(float(PTFE_EPS_R)) / C0
    beta_phase, v_p = _beta_from_s21_phase(freqs, S[1, 0, :], l12)
    gamma = np.asarray(res.gamma)
    beta_fit = np.imag(gamma)
    beta_fit = beta_fit.mean(axis=tuple(range(beta_fit.ndim - 1)))
    col = np.sum(np.abs(S) ** 2, axis=0)
    out["thru"] = {
        "beta_ratio_s21_phase_worst": float(np.max(np.abs(beta_phase / beta_analytic - 1.0))),
        "beta_ratio_matrix_pencil_worst": float(np.max(np.abs(beta_fit / beta_analytic - 1.0))),
        "phase_velocity_m_per_s": float(v_p),
        "max_column_power": float(col.max()),
        "min_column_power": float(col.min()),
        "abs_s21": np.abs(S[1, 0, :]).astype(float).tolist(),
        "status": str(res.status),
        "settling_db": np.asarray(res.settling_db, dtype=float).tolist(),
    }
    out["thru"]["beta_within_bar"] = bool(
        out["thru"]["beta_ratio_s21_phase_worst"] <= BETA_FRAC
        and out["thru"]["beta_ratio_matrix_pencil_worst"] <= BETA_FRAC)
    out["thru"]["column_power_within_bar"] = bool(
        out["thru"]["max_column_power"] <= COLUMN_POWER_MAX
        and out["thru"]["min_column_power"] >= 1.0 - (COLUMN_POWER_MAX - 1.0))

    out["loads"] = {}
    for load in LOADS_OHM:
        r = _sim(LOAD_DOMAIN).compute_coaxial_line_reflection(
            termination="matched", dut_impedance=load, n_steps=LOAD_STEPS,
            freqs=FREQS, probe_count=LOAD_PROBES)
        z0 = np.asarray(r.z0_numerical_ohm)
        finite = np.isfinite(np.real(z0))
        measured = float(np.median(np.real(z0)[finite])) if finite.any() else float("nan")
        frac = abs(measured - declared_z) / declared_z
        out["loads"][f"{load:g}"] = {
            "z0_ohm": measured, "frac_vs_declared": frac,
            "within_bar": bool(frac <= Z0_FRAC),
            "abs_gamma": np.abs(np.asarray(r.s11)).astype(float).tolist(),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                  text=True).strip()
    install()
    rec = {
        "schema": "rfx.coax_conductor_mutation", "schema_version": 1,
        "commit": sha, "run_id": args.run_id,
        "rfx_file": str(Path(rfx.__file__).resolve()),
        "jax_version": jax.__version__, "numpy_version": np.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "python": sys.version.split()[0], "platform": platform.platform(),
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "mutation": ("stamp_coaxial_line writes its conductor cells back into "
                     "materials.sigma as PEC_SIGMA, and _coax_pec_edge_masks "
                     "returns no masks; every helper the lane calls is "
                     "unchanged"),
    }
    path = out_dir / "coax_conductor_mutation.json"
    path.write_text(json.dumps(rec, indent=1))
    rec["measured"] = measure()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)

    m = rec["measured"]
    print(f"[mutation] thru beta (S21 phase) {m['thru']['beta_ratio_s21_phase_worst']*100:.2f} %, "
          f"(pencil) {m['thru']['beta_ratio_matrix_pencil_worst']*100:.2f} % "
          f"-> within the {BETA_FRAC*100:.0f} % bar: {m['thru']['beta_within_bar']}")
    print(f"[mutation] thru column power [{m['thru']['min_column_power']:.5f}, "
          f"{m['thru']['max_column_power']:.5f}] -> within bar: "
          f"{m['thru']['column_power_within_bar']}")
    for load, d in m["loads"].items():
        print(f"[mutation] {load} ohm load: Z0 {d['z0_ohm']:.3f} ohm, "
              f"{d['frac_vs_declared']*100:.2f} % from the declared "
              f"{m['declared_z_tem_ohm']:.3f} -> within bar: {d['within_bar']}")
    reds = [not m["thru"]["beta_within_bar"],
            not m["thru"]["column_power_within_bar"]] + \
           [not d["within_bar"] for d in m["loads"].values()]
    print(f"[mutation] comparisons the oracle would FAIL on the revived defect: "
          f"{sum(reds)} of {len(reds)}")
    print(f"[mutation] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
