#!/usr/bin/env python3
"""Measure what decimation actually costs a Q estimate at cv02's sampling density.

Why this exists
---------------
``ring_mode_judge.q_window`` briefly justified its ``tau/T`` envelope with a
second empirical prop: "what DOES degrade at short records is the decimated
path cv02 actually runs (3.49% there, 0.24% at the 0.25 cut)".  Those two
figures came from a comment on issue #907 and were never pinned by an
artifact, a test or a table anywhere in this repo.  A refute-oriented review
could not reproduce them, and neither can this script.

So the sentence is withdrawn, and this is the measurement that replaces it.
It re-runs the configuration the withdrawn claim names -- a synthetic single
damped exponential at cv02's own ``dt``, driven at the live mode's ``f`` and
``Q`` -- across the same ``T/tau`` ladder, and records for every rung:

* the decimation plan ``rfx.harminv`` actually chooses (factors and retained
  sample count), which is the thing the claim assumed and did not check;
* the relative Q error on the default path (``decimate='auto'``) and on
  ``decimate=False``.

Two frequency bands are measured -- the band the withdrawn claim used
(``[0.5 f, 1.5 f]``) and the band cv02 actually analyses (``rig.band_c_over_a``
from the committed record). This is a **plan-stability check, not an
independent witness**, and the artifact shows why: ``f_max`` enters
``rfx.harminv`` only through the decimation target ``int(1/dt/(4 f_max))`` and
a post-hoc pass-band filter, the two bands' targets both factor to the same
plan, and every rung then reports the same relative Q error in both bands to
every digit. The two columns are one computation; their agreement is by
construction. What the check buys is narrow and worth keeping: a plan that
differed between bands would make every other leg band-contingent.

The R5 independent witness on each rung is the other pair -- ``decimate='auto'``
against ``decimate=False``. Those run different sample counts through
different pencil sizes and are the only two legs here that can disagree.

Everything that identifies the case -- ``dt``, ``f``, ``Q``, the band -- is
READ from the committed cv02 record rather than typed here, so the ladder
cannot drift away from the board it claims to describe.

Regenerate with::

    python scripts/diagnostics/cv02_harminv_decimation_ladder.py \
        --output tests/fixtures/cv02_ring_judge/harminv_decimation_ladder.json

``--max-undecimated-samples`` (default 5000) caps the ``decimate=False``
reference leg, whose cost grows like the cube of the record length; the
default path is measured on every rung regardless. ``--quick`` drops the
longest rung outright.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rfx.harminv import _decimation_plan, harminv  # noqa: E402

CV02_RECORD = REPO_ROOT / "validation/crossval/_02_ring_resonator_results/crossval.json"
HARMINV_SOURCE = REPO_ROOT / "rfx/harminv.py"

#: The ladder from #907's 2026-09-10 comment, verbatim, so the rungs the
#: withdrawn numbers were attached to are the rungs re-measured here.
T_OVER_TAU_LADDER = (0.0822, 0.25, 0.3534, 1.2528)

#: What the withdrawn sentence asserted, kept as data so the retraction is
#: checkable rather than an absence. Values are relative Q error.
WITHDRAWN_CLAIM = {
    "text": (
        "what DOES degrade at short records is the decimated path cv02 "
        "actually runs (3.49% there, 0.24% at the 0.25 cut)"
    ),
    "source": "issue #907, comment of 2026-09-10; never pinned by any repo artifact",
    "asserted_relative_q_error": {"0.0822": 0.0349, "0.25": 0.0024},
    "withdrawn_in": "PR for #945/#907, 2026-09-13",
}

C_M_PER_S = 299792458.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rung(n_samples: int, dt: float, freq: float, q_true: float,
          f_min: float, f_max: float, *, max_undecimated: int) -> dict:
    """One (record length, band) cell: the plan, and both paths' Q error.

    ``max_undecimated`` bounds only the ``decimate=False`` REFERENCE leg. That
    leg hands the matrix pencil the whole record, so its cost grows like the
    cube of the sample count and the longest rung alone runs longer than the
    rest of the ladder put together. Where it is skipped the cell says so; the
    default path -- the one the withdrawn claim was about -- is always measured
    on every rung.
    """
    t = np.arange(n_samples) * dt
    tau = q_true / (math.pi * freq)
    signal = np.exp(-t / tau) * np.cos(2.0 * np.pi * freq * t)

    factors, kept = _decimation_plan(n_samples, dt, f_max, "auto")
    cell = {
        "n_samples": n_samples,
        "decimation_factors": list(factors),
        "decimation_retained_samples": kept,
        "decimation_fires": bool(factors),
        "dt_eff_over_dt": float(np.prod(factors)) if factors else 1.0,
    }
    for label, keyword in (("auto", "auto"), ("no_decimation", False)):
        if keyword is False and n_samples > max_undecimated:
            cell[label] = {
                "measured": False,
                "skipped": "runtime",
                "why": (f"undecimated pencil on {n_samples} samples; the "
                        f"reference leg is capped at {max_undecimated}"),
            }
            continue
        started = time.time()
        modes = harminv(signal, dt, f_min, f_max, decimate=keyword)
        elapsed = time.time() - started
        if not modes:
            cell[label] = {"measured": True, "found": False,
                           "wall_time_s": elapsed}
            continue
        best = min(modes, key=lambda m: abs(m.freq - freq))
        cell[label] = {
            "measured": True,
            "found": True,
            "Q": best.Q,
            "relative_q_error": abs(best.Q - q_true) / q_true,
            "relative_freq_error": abs(best.freq - freq) / freq,
            "harminv_error_field": best.error,
            "wall_time_s": elapsed,
        }
    return cell


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--quick", action="store_true",
                    help="drop the longest rung (T/tau = 1.2528)")
    ap.add_argument("--max-undecimated-samples", type=int, default=5000,
                    help=("cap on the decimate=False reference leg's record "
                          "length; the default path is always measured"))
    args = ap.parse_args()

    record = json.loads(CV02_RECORD.read_text(encoding="utf-8"))
    rig = record["rig"]
    dt = rig["rfx_dt_s"]
    a_m = rig["a_m"]
    band_lo, band_hi = rig["band_c_over_a"]

    # The live mode the withdrawn claim's synthetic was built from: cv02's
    # strongest rfx mode in the committed record.
    mode = max(record["measured"]["rfx_modes"], key=lambda m: m["amplitude"])
    freq = mode["freq_hz"]
    q_true = mode["Q"]
    tau = q_true / (math.pi * freq)

    bands = {
        "withdrawn_claim_band": {
            "f_min_hz": 0.5 * freq,
            "f_max_hz": 1.5 * freq,
            "why": "the band #907's comment used; the configuration under test",
        },
        "cv02_analysis_band": {
            "f_min_hz": band_lo * C_M_PER_S / a_m,
            "f_max_hz": band_hi * C_M_PER_S / a_m,
            "why": ("rig.band_c_over_a -- the band cv02 actually runs. "
                    "Plan-stability check, NOT an independent witness: both "
                    "bands resolve to the same decimation plan on every rung, "
                    "so the two columns are one computation. The independent "
                    "witness is auto vs no_decimation."),
        },
    }

    ladder = [x for x in T_OVER_TAU_LADDER if not (args.quick and x > 1.0)]
    rows = []
    for t_over_tau in ladder:
        n_samples = int(round(t_over_tau * tau / dt))
        row = {"t_over_tau": t_over_tau, "n_samples": n_samples, "bands": {}}
        for name, band in bands.items():
            row["bands"][name] = _rung(
                n_samples, dt, freq, q_true,
                band["f_min_hz"], band["f_max_hz"],
                max_undecimated=args.max_undecimated_samples)
            print(f"T/tau={t_over_tau:<8} n={n_samples:<6} {name:<22} "
                  f"plan={row['bands'][name]['decimation_factors']} "
                  f"auto={row['bands'][name]['auto'].get('relative_q_error')} "
                  f"none={row['bands'][name]['no_decimation'].get('relative_q_error')}",
                  flush=True)
        rows.append(row)

    payload = {
        "schema": "cv02-harminv-decimation-ladder/v1",
        "purpose": (
            "Refutes the 'the decimated path degrades at short records' prop "
            "that ring_mode_judge.q_window briefly cited. See withdrawn_claim."
        ),
        "generated_by": "scripts/diagnostics/cv02_harminv_decimation_ladder.py",
        "quick": bool(args.quick),
        "max_undecimated_samples": args.max_undecimated_samples,
        "source_record": {
            "path": str(CV02_RECORD.relative_to(REPO_ROOT)),
            "commit": record["commit"],
            "date_utc": record["date_utc"],
        },
        "signal": {
            "model": "single damped exponential, cos(2 pi f t) exp(-t/tau)",
            "dt_s": dt,
            "freq_hz": freq,
            "Q": q_true,
            "tau_s": tau,
            "samples_per_period": 1.0 / (freq * dt),
        },
        "bands": bands,
        "withdrawn_claim": WITHDRAWN_CLAIM,
        "rows": rows,
        "provenance": {
            "harminv_source_sha256": _sha256(HARMINV_SOURCE),
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
