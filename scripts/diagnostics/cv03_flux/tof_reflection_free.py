"""#812 round 2 — harness provenance for cv03's time-of-flight configurations.

Section 8.1 of ``issue812_cv03_dispersion_regate_predeclaration.md`` recorded
four configurations of the cv03 guide -- two domain lengths crossed with a DFT
window that either does or does not admit the round trip -- but its driver was
never committed, so those measured operands carry only note provenance.  This
script is that driver.  It emits every per-bin array the comparison needs into
one JSON so the design notes can reference keys, and re-derives the
matched-frequency deviation inside the harness instead of in prose.

The case itself is never modified: each configuration is a small set of exact
textual edits applied to a COPY, exactly as
``scripts/diagnostics/cv03_flux/regate_falsifiers.py`` does for the falsifiers.

FDTD.  Per SPEC-00 section 0.3b this belongs on VESSL, not on the shared Mac;
``scripts/vessl_cv03_tof_reflection_free.yaml`` submits it.

Run:
    PYTHONPATH=<repo> python scripts/diagnostics/cv03_flux/tof_reflection_free.py \
        --output <path>.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CASE = os.path.join(REPO, "validation", "crossval",
                    "03_straight_waveguide_flux.py")

_ANCHOR = "# Exit code (rfx crossval convention)"

_DUMP = '''
# --- appended by scripts/diagnostics/cv03_flux/tof_reflection_free.py ---
import json as _json                                          # noqa: E402
with open(os.environ["CV03_TOF_JSON"], "w") as _fh:
    _json.dump({
        "freqs_c_over_a": [float(v) for v in meep_freqs],
        "n_eff_rfx": [float(v) for v in neff_rfx],
        "n_eff_analytic": [float(v) for v in neff_analytic],
        "two_wave_rel_residual": [float(v) for v in neff_resid],
        "b_over_a": [float(v) for v in neff_bovera],
        "band_mask": [bool(v) for v in band_mask],
        "T_rfx_band_mean": float(T_rfx_band),
        "T_rfx_peak": float(T_rfx_peak),
        "n_steps": int(n_steps),
    }, _fh, indent=2)
'''

# key, why it exists, [(exact old text, exact new text), ...]
CONFIGS = [
    ("sx16_dft400_recipe_baseline",
     "the committed recipe: 16a domain, 400 a/c0 window, fit window [-4,+4]",
     []),
    ("sx16_dft150_no_round_trip_yet",
     "recipe domain, DFT stopped before its own round trip",
     [("rfx_total_t = 400.0 * a / C0", "rfx_total_t = 150.0 * a / C0")]),
    ("sx40_dft150_before_round_trip",
     "40a domain, DFT stopped before the far-end return can arrive: the "
     "reflection-free window.  Round trip is 231 a/c0 to the far end and "
     "back, 271 a/c0 to re-enter the fit window -- see "
     "docs/design_notes/issue812_cv03_dispersion_matched_frequency.json"
     "::oracle.tof_round_trip_a_over_c0.  (Round 1 said '~211 a/c0 at "
     "n_g ~ 3.3'; the true group index is oracle.n_group_at_carrier_bin, "
     "and 3.3 was a transcription collision with the SWR 3.33.  The error "
     "was conservative -- the return arrives LATER than round 1 claimed -- "
     "so the 150 a/c0 window below is still reflection-free and no measured "
     "row moves.)",
     [("sx = 16.0", "sx = 40.0"),
      ("src_x_meep   = -7.0", "src_x_meep   = -19.0"),
      ("neff_x_lo_meep = flux_in_meep + 1.0     # -4.0",
       "neff_x_lo_meep = -16.0"),
      ("neff_x_hi_meep = flux_out_meep - 1.0    # +4.0",
       "neff_x_hi_meep = -8.0"),
      ("rfx_total_t = 400.0 * a / C0", "rfx_total_t = 150.0 * a / C0")]),
    ("sx40_dft400_after_round_trip",
     "the same 40a domain with the window long enough to admit the return",
     [("sx = 16.0", "sx = 40.0"),
      ("src_x_meep   = -7.0", "src_x_meep   = -19.0"),
      ("neff_x_lo_meep = flux_in_meep + 1.0     # -4.0",
       "neff_x_lo_meep = -16.0"),
      ("neff_x_hi_meep = flux_out_meep - 1.0    # +4.0",
       "neff_x_hi_meep = -8.0")]),
]


def run_one(edits, workdir, json_path):
    src = open(CASE).read()
    for old, new in edits:
        assert src.count(old) == 1, f"pattern is not unique: {old!r}"
        src = src.replace(old, new)
    assert src.count(_ANCHOR) == 1, "exit-code anchor not found in the case"
    src = src.replace(_ANCHOR, _DUMP + "\n" + _ANCHOR)
    path = os.path.join(workdir, "case.py")
    with open(path, "w") as fh:
        fh.write(src)
    env = dict(os.environ, PYTHONPATH=REPO, MPLBACKEND="Agg",
               CV03_TOF_JSON=json_path)
    proc = subprocess.run([sys.executable, path], capture_output=True,
                          text=True, env=env, cwd=workdir)
    return proc.returncode, proc.stdout + proc.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = {"schema": "issue812-cv03-tof-reflection-free-v1", "issue": 812,
           "runs_fdtd": True, "configs": {}}
    for key, why, edits in CONFIGS:
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(os.path.join(REPO, "validation", "crossval",
                                         "comparators"),
                            os.path.join(td, "comparators"))
            jp = os.path.join(td, "dump.json")
            rc, log = run_one(edits, td, jp)
            payload = json.load(open(jp)) if os.path.exists(jp) else None
        if payload is None:
            out["configs"][key] = {"why": why, "exit_code": rc,
                                   "status": "no dump", "log_tail": log[-2000:]}
            print(f"[{key}] exit={rc} NO DUMP")
            continue
        f = np.array(payload["freqs_c_over_a"])
        nr = np.array(payload["n_eff_rfx"])
        na = np.array(payload["n_eff_analytic"])
        mask = np.array(payload["band_mask"], dtype=bool)
        carrier = int(np.argmin(np.abs(f - 0.15)))
        dev = nr / na - 1.0
        payload.update({
            "why": why,
            "exit_code": rc,
            "carrier_bin_index": carrier,
            "carrier_bin_f_c_over_a": float(f[carrier]),
            "carrier_dev_pct_matched": float(100.0 * dev[carrier]),
            "band_max_abs_dev_pct": float(100.0 * np.max(np.abs(dev[mask]))),
            "band_max_two_wave_rel_residual": float(
                np.max(np.array(payload["two_wave_rel_residual"])[mask])),
            "b_over_a_carrier_bin": float(payload["b_over_a"][carrier]),
        })
        out["configs"][key] = payload
        print(f"[{key}] exit={rc} carrier dev "
              f"{payload['carrier_dev_pct_matched']:+.4f}%  band-max "
              f"{payload['band_max_abs_dev_pct']:.4f}%  |B/A| "
              f"{payload['b_over_a_carrier_bin']:.4f}")

    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
