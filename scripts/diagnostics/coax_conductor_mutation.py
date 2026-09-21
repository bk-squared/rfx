#!/usr/bin/env python3
"""Put the old conductor realization back and see what goes red.

The (b) falsifier for the conductor-realization gates: a gate that only proves
the check can be switched off proves nothing, so this revives the DEFECT —
``sigma = PEC_SIGMA`` per node instead of PEC edge masks — with every helper
the lane calls left exactly where it is, and reports what the gates then read.

The mutation is two rebindings and no edit to shipped code:

* ``rfx.sources.coaxial_port.stamp_coaxial_line`` is wrapped so that, after the
  shipped stamper returns, the conductor cells it reported are written back
  into ``materials.sigma`` as ``PEC_SIGMA`` — the pre-fix material array.
* ``rfx.sparams.coax._coax_pec_edge_masks`` is replaced by one that returns
  ``None`` (passing a merge target through unchanged), so the lane hands the
  runner no edge masks — the pre-fix call.

It reports against BOTH halves of the oracle, because they fail for different
reasons and a mutation that only reds one of them would leave the other
untested:

1. **The fast structural gate** (``tests/unit/sparams/
   test_coax_conductor_geometry.py``), which needs no FDTD: are the conductors
   handed to ``rfx.simulation.run`` as ``pec_edge_masks``, and is
   ``materials.sigma`` clear of ``PEC_SIGMA``? Under the mutation the first is
   ``None`` and the second is not.
2. **The live physics test** (``tests/oracle/
   test_coax_conductor_realization.py``), on its own board and rung.

Usage::

    PYTHONPATH=. python scripts/diagnostics/coax_conductor_mutation.py \\
        --out <run-dir> [--run-id <id>] [--skip-live]
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
import rfx.simulation  # noqa: E402
import rfx.sources.coaxial_port as _cp  # noqa: E402
import rfx.sparams.coax as _coax  # noqa: E402
from rfx.api import Simulation  # noqa: E402
from rfx.sources.coaxial_port import (  # noqa: E402
    PEC_SIGMA, PTFE_EPS_R, SMA_OUTER_RADIUS, SMA_PIN_RADIUS,
)
from rfx.sources.sources import GaussianPulse  # noqa: E402

sys.path.insert(0, str(REPO / "tests" / "oracle"))
from test_coax_conductor_realization import (  # noqa: E402
    BETA_FRAC, COLUMN_POWER_MAX, C0, LIVE_DOMAIN, LIVE_FREQS, LIVE_PROBES,
    LIVE_RUNG, LIVE_STEPS, _beta_from_s21_phase,
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


def _live_sim():
    a, b = float(SMA_PIN_RADIUS), float(SMA_OUTER_RADIUS)
    sim = Simulation(freq_max=40.0e9, domain=LIVE_DOMAIN, boundary="cpml",
                     cpml_layers=16, dx=(b - a) / LIVE_RUNG)
    sim.add_coaxial_port((LIVE_DOMAIN[0] / 2.0, LIVE_DOMAIN[1] / 2.0,
                          LIVE_DOMAIN[2] / 2.0), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim


class _Captured(Exception):
    pass


def measure_structural() -> dict:
    """What the lane hands the runner, without running it."""
    seen = {}
    original_run = rfx.simulation.run

    def _spy(grid, materials, n_steps, **kw):
        seen["pec_edge_masks"] = kw.get("pec_edge_masks")
        seen["sigma_max"] = float(np.asarray(materials.sigma).max())
        raise _Captured

    rfx.simulation.run = _spy
    try:
        _live_sim().compute_coaxial_two_port(
            n_steps=8, freqs=np.array([8e9]), **LIVE_PROBES)
    except _Captured:
        pass
    finally:
        rfx.simulation.run = original_run

    masks = seen.get("pec_edge_masks")
    sigma_max = seen.get("sigma_max", float("nan"))
    return {
        "reached_runner": bool(seen),
        "pec_edge_masks_is_none": masks is None,
        "pec_edge_masks_any": bool(
            masks is not None and any(np.asarray(m).any() for m in masks)),
        "sigma_max": sigma_max,
        "pec_sigma_threshold": 0.5 * float(PEC_SIGMA),
        "sigma_is_clear_of_pec": bool(sigma_max < 0.5 * float(PEC_SIGMA)),
    }


def measure_live() -> dict:
    res = _live_sim().compute_coaxial_two_port(
        n_steps=LIVE_STEPS, freqs=LIVE_FREQS, **LIVE_PROBES)
    freqs = np.asarray(res.freqs, dtype=float)
    S = np.asarray(res.s_params)
    planes = np.asarray(res.reference_planes, dtype=float)
    beta_analytic = 2.0 * np.pi * freqs * math.sqrt(float(PTFE_EPS_R)) / C0
    try:
        beta_phase, _v_p = _beta_from_s21_phase(
            freqs, S[1, 0, :], float(abs(planes[0] - planes[1])))
        phase_worst = float(np.max(np.abs(beta_phase / beta_analytic - 1.0)))
        note = None
    except AssertionError as exc:
        phase_worst, note = float("nan"), str(exc)
    gamma = np.asarray(res.gamma)
    beta_fit = np.imag(gamma)
    beta_fit = beta_fit.mean(axis=tuple(range(beta_fit.ndim - 1)))
    pencil = float(np.max(np.abs(beta_fit / beta_analytic - 1.0)))
    col = np.sum(np.abs(S) ** 2, axis=0)
    out = {
        "board": "thru_long", "rung_annulus_cells": LIVE_RUNG,
        "annulus_cells": float(getattr(res, "annulus_cells", float("nan"))),
        "n_steps": LIVE_STEPS,
        "beta_ratio_s21_phase_worst": phase_worst,
        "beta_ratio_matrix_pencil_worst": pencil,
        "s21_phase_note": note,
        "max_column_power": float(col.max()),
        "min_column_power": float(col.min()),
        "status": str(res.status),
        "settling_db": np.asarray(res.settling_db, dtype=float).tolist(),
    }
    out["beta_within_bar"] = bool(
        np.isfinite(phase_worst) and phase_worst <= BETA_FRAC
        and pencil <= BETA_FRAC)
    out["column_power_within_bar"] = bool(
        out["max_column_power"] <= COLUMN_POWER_MAX
        and out["min_column_power"] >= 1.0 - (COLUMN_POWER_MAX - 1.0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--skip-live", action="store_true",
                    help="structural half only; no FDTD")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                  text=True).strip()
    install()
    rec = {
        "schema": "rfx.coax_conductor_mutation", "schema_version": 2,
        "commit": sha, "run_id": args.run_id,
        "rfx_file": str(Path(rfx.__file__).resolve()),
        "jax_version": jax.__version__, "numpy_version": np.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "python": sys.version.split()[0], "platform": platform.platform(),
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "bars": {"beta_frac": BETA_FRAC, "column_power_max": COLUMN_POWER_MAX},
        "mutation": ("stamp_coaxial_line writes its conductor cells back into "
                     "materials.sigma as PEC_SIGMA, and _coax_pec_edge_masks "
                     "returns no masks; every helper the lane calls is "
                     "unchanged"),
    }
    path = out_dir / "coax_conductor_mutation.json"
    path.write_text(json.dumps(rec, indent=1))      # persisted BEFORE the solve

    rec["structural"] = st = measure_structural()
    reds = []
    print(f"[mutation] structural: pec_edge_masks is None -> "
          f"{st['pec_edge_masks_is_none']} (the fast gate asserts it is NOT)")
    print(f"[mutation] structural: materials.sigma max {st['sigma_max']:.3e} "
          f"vs the PEC_SIGMA/2 threshold {st['pec_sigma_threshold']:.3e} -> "
          f"clear of PEC: {st['sigma_is_clear_of_pec']} "
          f"(the fast gate asserts it IS)")
    reds.append(("structural: conductors handed over as pec_edge_masks",
                 st["pec_edge_masks_is_none"]))
    reds.append(("structural: sigma clear of PEC_SIGMA",
                 not st["sigma_is_clear_of_pec"]))

    if not args.skip_live:
        rec["live"] = m = measure_live()
        print(f"[mutation] live thru ({m['board']} at "
              f"{m['annulus_cells']:.4f} cells): beta (S21 phase) "
              f"{m['beta_ratio_s21_phase_worst']*100:.2f} %, (pencil) "
              f"{m['beta_ratio_matrix_pencil_worst']*100:.2f} % -> within the "
              f"{BETA_FRAC*100:.0f} % bar: {m['beta_within_bar']}")
        if m["s21_phase_note"]:
            print(f"[mutation] live: S21-phase estimate unavailable: "
                  f"{m['s21_phase_note']}")
        print(f"[mutation] live thru column power [{m['min_column_power']:.5f}, "
              f"{m['max_column_power']:.5f}] -> within bar: "
              f"{m['column_power_within_bar']}")
        print(f"[mutation] live status {m['status']}, settling "
              f"{m['settling_db']}")
        reds.append(("live: beta within 1 %", not m["beta_within_bar"]))
        reds.append(("live: column power within the bar",
                     not m["column_power_within_bar"]))

    tmp = path.with_suffix(".json.tmp")
    rec["reds"] = {name: bool(v) for name, v in reds}
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, path)

    n_red = sum(1 for _, v in reds if v)
    print(f"[mutation] assertions the revived defect turns RED: "
          f"{n_red} of {len(reds)}")
    for name, v in reds:
        print(f"[mutation]   {'RED ' if v else 'green'}  {name}")
    print(f"[mutation] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
