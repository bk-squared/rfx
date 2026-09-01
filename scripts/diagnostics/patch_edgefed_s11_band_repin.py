"""Edge-fed patch |S11| on the passivity-gate board — post-#702 band re-pin evidence.

Measures the two arms pre-declared in
``docs/design_notes/issue782_retired_resonance_predeclaration.md`` (Section 4):

  * ``main``    — today's tree physics;
  * ``retired`` — ``rfx.api._compile.resample_sheet_node_materials`` replaced by the
    identity (the same bypass ``tests/test_preflight_campaign_statics.py`` uses),
    which reproduces the pre-#702 tree digit for digit (issue #782).

Config is EXACTLY the committed gate's (``tests/test_patch_edgefed_s11_passivity.py``:
same ``_build_patch_sim()`` geometry — imported from the test module, not copied —
freqs = linspace(6, 14 GHz, 81), num_periods = 280). This board ("Board S") realizes
44 x 51 patch cells = 8.668 x 10.047 mm at dx = 197 um; it is NOT the harminv gate's
43 x 51 board, so no Board-H number (8.16131 GHz etc.) is reused here — the whole
point is to measure the band on the gate's own board.

Dumps per arm (R5, full per-bin trace, preflight verbatim, #332 witness outcome) and
writes ``docs/design_notes/patch_edgefed_s11_band_repin_results.json`` for the
falsifier-F1 replay.

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=<repo> python3 scripts/diagnostics/patch_edgefed_s11_band_repin.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import rfx  # noqa: E402
import rfx.api._compile as _compile  # noqa: E402
from test_patch_edgefed_s11_passivity import _build_patch_sim  # noqa: E402

FREQS = np.linspace(6e9, 14e9, 81)
NUM_PERIODS = 280.0
# Pre-declared candidate band for the crossing read-out (predeclaration E3); the
# committed band is pinned AFTER inspection, not by this constant.
CANDIDATE_BAND_GHZ = (7.8, 8.6)


def zero_crossings_ghz(fr_ghz, y):
    """All sign changes of y, linearly interpolated, in GHz."""
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    return [float(fr_ghz[i] - y[i] * (fr_ghz[i + 1] - fr_ghz[i]) / (y[i + 1] - y[i]))
            for i in idx]


def run_arm(tag: str, bypass_resample: bool) -> dict:
    print(f"\n================ ARM {tag} (bypass_resample={bypass_resample}) "
          f"================", flush=True)
    if bypass_resample:
        # Same bypass as tests/test_preflight_campaign_statics.py::_bypass_resample:
        # the assembly keeps node-sampled statics — the pre-#702 tree, digit for digit.
        _compile.resample_sheet_node_materials = (
            lambda geo, res, coords, eps, sig, **kw: (eps, sig))

    sim = _build_patch_sim()

    advisories = [str(a) for a in sim.preflight()]
    print(f"[{tag}] preflight advisories ({len(advisories)}) — quoted verbatim:", flush=True)
    for a in advisories:
        print(f"  ! {a}", flush=True)

    with warnings.catch_warnings(record=True) as settling:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(freqs=jnp.asarray(FREQS),
                                       num_periods=NUM_PERIODS)
    trunc = [str(w.message) for w in settling
             if "#332" in str(w.message) or "ring-down truncated" in str(w.message)]
    print(f"[{tag}] #332 settling witness: "
          f"{trunc if trunc else 'silent — drained below -40 dB of peak'}", flush=True)

    fr = np.asarray(res.freqs, dtype=float) / 1e9
    s = np.asarray(res.S)[0, 0, :]
    z0 = np.asarray(res.Z0)[0, :]
    zin = z0 * (1.0 + s) / (1.0 - s)
    s11 = np.abs(s)

    i_dip = int(np.argmin(s11))
    band = (fr >= CANDIDATE_BAND_GHZ[0]) & (fr <= CANDIDATE_BAND_GHZ[1])
    crossings = zero_crossings_ghz(fr, zin.imag)
    in_band_cross = [c for c in crossings
                     if CANDIDATE_BAND_GHZ[0] <= c <= CANDIDATE_BAND_GHZ[1]]

    print(f"\n[{tag}-TRACE]   f(GHz)   |S11|    Re(Zin)    Im(Zin)", flush=True)
    for k in range(len(fr)):
        print(f"  {fr[k]:7.3f}  {s11[k]:7.4f}  {zin.real[k]:9.2f}  {zin.imag[k]:9.2f}",
              flush=True)

    out = dict(
        tag=tag,
        bypass_resample=bypass_resample,
        num_periods=NUM_PERIODS,
        settling_advisories=trunc,
        preflight=advisories,
        freqs_ghz=[float(v) for v in fr],
        s11_re=[float(v) for v in s.real],
        s11_im=[float(v) for v in s.imag],
        z0_re=[float(v) for v in z0.real],
        z0_im=[float(v) for v in z0.imag],
        max_s11=float(np.max(s11)),
        f_dip_ghz=float(fr[i_dip]),
        s11_at_dip=float(s11[i_dip]),
        min_s11_candidate_band=float(np.min(s11[band])),
        im_zin_crossings_ghz=[round(c, 4) for c in crossings],
        crossings_in_candidate_band=[round(c, 4) for c in in_band_cross],
    )
    print(f"\n[{tag}] max|S11| = {out['max_s11']:.4f}   dip @ {out['f_dip_ghz']:.3f} GHz "
          f"(|S11| = {out['s11_at_dip']:.4f})", flush=True)
    print(f"[{tag}] min|S11| over candidate band {CANDIDATE_BAND_GHZ} = "
          f"{out['min_s11_candidate_band']:.4f}", flush=True)
    print(f"[{tag}] Im(Zin)=0 crossings = {out['im_zin_crossings_ghz']} GHz "
          f"(in candidate band: {out['crossings_in_candidate_band']})", flush=True)
    return out


def _out_path(name: str) -> str:
    return os.path.join(_REPO, "docs", "design_notes", name)


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                          capture_output=True, text=True).stdout.strip()
    print(f"[REPIN] rfx.__file__ = {rfx.__file__}", flush=True)
    print(f"[REPIN] git HEAD     = {head}", flush=True)
    print(f"[REPIN] num_periods  = {NUM_PERIODS}, freqs 6-14 GHz x {len(FREQS)}", flush=True)

    # Arm selector: `main`, `retired`, or (default) both in one process. Each arm is
    # PERSISTED THE MOMENT IT FINISHES (per-arm JSON) — the first execution of this
    # script was killed by the session harness between the arms and the finished main
    # arm's arrays died unpersisted (the `feedback_persist_before_the_optional_stage`
    # lesson, again). One process per arm also keeps each run under the kill horizon.
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    arms = {"main": False, "retired": True} if which == "both" else \
        {which: (which == "retired")}

    for tag, bypass in arms.items():
        arm = run_arm(tag, bypass_resample=bypass)
        per_arm = _out_path(f"patch_edgefed_s11_band_repin_{tag}.json")
        with open(per_arm, "w") as f:
            json.dump(dict(rfx_file=rfx.__file__, git_head=head, **{tag: arm}), f,
                      indent=1)
        print(f"\n[REPIN] wrote {per_arm}", flush=True)

    # Combine whatever per-arm files exist into the canonical results file.
    results = dict(rfx_file=rfx.__file__, git_head=head)
    for tag in ("main", "retired"):
        per_arm = _out_path(f"patch_edgefed_s11_band_repin_{tag}.json")
        if os.path.exists(per_arm):
            with open(per_arm) as f:
                results[tag] = json.load(f)[tag]
    if "main" in results and "retired" in results:
        combined = _out_path("patch_edgefed_s11_band_repin_results.json")
        with open(combined, "w") as f:
            json.dump(results, f, indent=1)
        print(f"[REPIN] wrote {combined}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
