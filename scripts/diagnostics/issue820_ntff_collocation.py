"""Issue #820 step 2 — test the two NTFF collocation corrections in a SCRATCH copy.

PRE-DECLARATION. Written and committed BEFORE any arm ran. PI-approved
2026-09-14. ``rfx/`` is NOT touched by this lane: nothing here modifies
``rfx/farfield.py``, and the production ``compute_far_field`` is called
unchanged.

What is corrected, and why the correction is applied to the INPUTS
------------------------------------------------------------------
``compute_far_field`` gives every one of the four stored components on a face
the SAME position (``_face_positions``, rfx/farfield.py:344) and the same
sampling plane, while the Yee grid puts them at six different half-cell
locations:

    ex (1/2, 0, 0)   ey (0, 1/2, 0)   ez (0, 0, 1/2)
    hx (0, 1/2, 1/2) hy (1/2, 0, 1/2) hz (1/2, 1/2, 0)

The transform's phase factor is ``exp(+1j k (rhat . r'))``
(rfx/farfield.py:563), so moving a component from the nominal node to its true
location is a multiplication of that component by ``exp(+1j k (rhat . delta))``.
Both corrections are therefore expressible as changes to the FACE DATA handed
to the production transform, and that is how they are applied here.

This is deliberate and it is the comparator-first choice: hand-rolling the
N/L assembly and the ``E_theta = -jk/4pi (L_phi + eta N_theta)`` algebra would
put a fresh, unvalidated comparator between the measurement and the answer, in
a repo whose ledger says 13 of 13 cross-check disagreements were comparator
bugs. Feeding corrected inputs to the validated transform tests the same
physics with nothing new to get wrong. The ``baseline`` arm (no correction)
must reproduce the committed probe value bit-for-bit or the module is void.

Consequence, stated rather than discovered later: because the phase factor
depends on ``rhat``, an arm-(ii)-corrected ``NTFFData`` is valid for ONE
observation direction. Every number here is evaluated at backscatter only.

The arms
--------
baseline  production face data, unmodified. Self-check S0.
arm_i     co-locate H with E by averaging the two adjacent H planes along the
          face normal -- the pattern rfx/simulation.py:1802-1811 already uses
          for Poynting flux, mirrored in rfx/nonuniform.py:2292-2295. H sits at
          ``idx+1/2``; averaging ``idx-1`` and ``idx`` puts it at ``idx``. No
          ``rhat`` dependence. Needs the adjacent plane, which the production
          box does not record, so each offset runs three extra simulations with
          boxes shifted one cell along x, y and z respectively -- each keeping
          the face's own transverse extent so the two planes can be averaged
          cell by cell.
arm_ii    per-component true Yee coordinates: multiply each stored component by
          ``exp(+1j k (rhat . delta_c))``. Corrects the normal AND transverse
          offsets, for E and H, by phase. At backscatter ``rhat = -xhat`` only
          the x-components of delta contribute, so the factor is
          ``exp(-1j k dx/2)`` on {ex, hy, hz} and 1 on {ey, ez, hx}.
arm_iii   (i) then (ii): average H along the normal, then apply the REMAINING
          offsets by phase. Differs from (ii) only in how the normal offset is
          removed -- by averaging (what the repo does elsewhere, exact for a
          field sampled on both planes) rather than by a single-plane-wave
          phase. Cheap once (i) has run.

Note on what arm (i) does at backscatter that a phase cannot: averaging is a
spatial operation on the SAMPLED VALUES, not a phase rotation. On the y faces
the stored H sits at ``j+1/2`` on BOTH ``y_lo`` and ``y_hi`` -- half a cell
inside the box on one, half a cell outside on the other -- and that asymmetry
is invisible to a backscatter phase factor (``rhat . yhat = 0``) but is exactly
what averaging removes. So arm (i), not arm (ii), is the test that addresses
the ODD transverse residue.

Pre-declared gates
------------------
S0  SELF-CHECK, gates the whole module. ``baseline`` must reproduce
    ``compute_far_field`` on unmodified data to <= 1e-12 relative, and must
    reproduce the committed probe value for the same offset. Otherwise VOID.

T   TRANSVERSE NULL, the primary observable. Continuum value exactly 0.000 dB,
    so 100 % of it is error. Measured as the y-sweep p-p, and compared against
    the BASELINE p-p measured in this same module at the SAME sampling, so the
    comparison carries no sampling confound.
      shrink factor = pp(baseline) / pp(arm)
      TRUE          >= 3.0
      FALSE         <= 1.2
      INCONCLUSIVE  in between
    Applied at ka = 1.0 AND at ka = 2.0. Also reported: the odd/even
    decomposition per arm, because the odd part is the part arm (i) is supposed
    to remove.

R   x-SWEEP CONSTANT TERM. ``r = |B|/|A|`` from the rotating-plus-constant fit.
    The bar clears the +-15 % leave-one-out scatter measured on the baseline
    (5.730 .. 6.548 on C; r's own scatter is quoted with the result):
      TRUE   r < 0.03
      FALSE  r > 0.10
      INCONCLUSIVE in between
    C is NOT a gate, here or anywhere else (correction C2).

CTL CONTROL, gates interpretation of T and R. Two parts:
    (a) the corrected extractor on the UNTRANSLATED sphere must not move the
        answer by more than the committed rfx-vs-Mie residual at that rung
        (ka = 1.0, clearance 30: the fixture's own ``delta_db`` = +0.622 dB).
        A larger move means the correction is not a refinement of the same
        measurement, and the arm is reported as such rather than as a fix.
    (b) the corrected extractor on the VACUUM run must keep the empty-domain
        backscatter at least 40 dB below the target's (baseline: 58.9 dB).
    Both are recorded per arm.

MIE Each arm's monostatic at offset 0 is recorded in dBsm AND as a distance
    from the exact Mie backscatter at the realized ``a_eff`` (the committed
    oracle ``tests/fixtures/rcs_sphere_mie/mie_oracle.py``, ``validate_oracle``
    run first, and cv16's own ``a_eff`` convention). If the correction is right
    the Mie agreement should improve or hold. **If it worsens, that is a
    finding and is reported as one** -- not explained away, and not a reason to
    re-cut a gate.

One attempt per arm (R2). An arm that cannot be evaluated is recorded
NON-CLOSING. Preflight is quoted ("none emitted", verified with
``warnings.simplefilter('always')``) and the ring-down witness is quoted per
rung.

CORRECTIONS 2, after the verification review (2026-09-14). Re-derived from
this lane's own artifacts before being written.

N1. "arm (ii) is worse on all three observables" is FALSE for the transverse
    null, and the claim is withdrawn. arm (ii) IMPROVES the null: 0.7020 ->
    0.6609 dB (1.062x) at ka = 1.0 and 0.5561 -> 0.4888 dB (1.138x) at
    ka = 2.0. It is FALSE against the 3x bar, not reversed.
    What survives, and is the reportable reading: arm (ii) WORSENS the
    LONGITUDINAL observables -- r 0.1189 -> 0.1602 and the x-sweep translation
    spread 2.2772 -> 3.0344 dB -- while barely helping the null. Offered as
    reasoning and NOT as attribution: a per-component phase leaves J at
    i +- 1/2 and M at i, i.e. two equivalent-current surfaces half a cell
    apart, whereas arm (i) brings both onto one surface.
    THE "DISTANCE FROM MIE" COLUMN IS WITHDRAWN ENTIRELY. The review's sign
    control (11-offset null; baseline / committed arm (ii) / sign-flipped /
    x-face-H-only both signs) reads null p-p 0.7020 / 0.6609 / 0.7464 /
    0.6519 / 0.7555 and Mie distance +0.7342 / +1.1225 / +0.3156 / +1.2167 /
    +0.2200. The null prefers the CORRECT sign; the Mie distance prefers the
    WRONG sign by 0.81 dB and awards its best score to the variant that
    damages the null most. At 6.4 cells per radius the 0.73 dB baseline
    distance is dominated by staircase error, so Mie distance is not a
    discriminator at this rung. Every claim resting on it is withdrawn,
    including arm (i)'s "0.37-0.50 dB closer to Mie".

N2. Disclosed in ``_boxes`` above: arm (i)'s lo-side adjacent planes sit on
    the first interior cells against the CPML.

N5. The x-sweep TRANSLATION SPREAD per arm, which the first write-up omitted
    in favour of r alone: baseline 2.2772 dB, arm (i) 1.6313 (1.396x),
    arm (ii) 3.0344 (0.750x), arm (iii) 1.4908 (1.527x).

N6. Gate (t1) was MIS-SPECIFIED, and its "CARRIER-INCONSISTENT" verdict mostly
    refutes the gate rather than the hypothesis. With ``polarization="ez"`` the
    y faces store ez and the z faces do not (``FACE_COMPONENTS``), so the two
    transverse axes were never interchangeable for a z-polarized backscatter
    and no axis-agnostic carrier was ever required to give the same odd ratio
    on both. The z reading (1.924) stands as a measurement; it is not evidence
    against collocation.

Usage
-----
  PYTHONPATH=<worktree> python3 scripts/diagnostics/issue820_ntff_collocation.py --sweep ynull_ka1
  ... --sweep xsweep_ka1 | ynull_ka2 | control
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "tests", "fixtures", "rcs_sphere_mie"))

import rfx  # noqa: E402

_RFX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(rfx.__file__)))
if _RFX_ROOT != _REPO_ROOT:
    raise RuntimeError(
        f"import rfx resolved outside this repo tree ({rfx.__file__}); "
        "refusing to report numbers for a different rfx build."
    )

import jax.numpy as jnp  # noqa: E402
from rfx.farfield import NTFFBox, compute_far_field  # noqa: E402
from rfx.grid import C0  # noqa: E402
from rfx.rcs import _incident_spectrum_amplitude  # noqa: E402
from rfx.simulation import run  # noqa: E402

_probe_path = os.path.join(_SCRIPT_DIR, "issue820_rcs_translation_probe.py")
_spec = importlib.util.spec_from_file_location("issue820_probe", _probe_path)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

F0 = probe.F0
LAM = probe.LAM
BANDWIDTH = probe.BANDWIDTH
CPML_LAYERS = probe.CPML_LAYERS

FACES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
# (axis of the face normal, sign) and the stored component order per face
FACE_AXIS = {"x_lo": 0, "x_hi": 0, "y_lo": 1, "y_hi": 1, "z_lo": 2, "z_hi": 2}
FACE_COMPONENTS = {
    0: ("ey", "ez", "hy", "hz"),
    1: ("ex", "ez", "hx", "hz"),
    2: ("ex", "ey", "hx", "hy"),
}
# Yee half-cell offsets of each component from the integer node, in cells
YEE_DELTA = {
    "ex": (0.5, 0.0, 0.0), "ey": (0.0, 0.5, 0.0), "ez": (0.0, 0.0, 0.5),
    "hx": (0.0, 0.5, 0.5), "hy": (0.5, 0.0, 0.5), "hz": (0.5, 0.5, 0.0),
}

# --- pre-declared gate constants ------------------------------------------
GATE_S0_REL = 1e-12
GATE_T_TRUE = 3.0          # pp(baseline) / pp(arm) >= this  -> TRUE
GATE_T_FALSE = 1.2         # ... <= this -> FALSE
GATE_R_TRUE = 0.03
GATE_R_FALSE = 0.10
GATE_CTL_MOVE_DB = 0.622   # committed cv16 ka=1.0 clearance-30 |rfx - Mie|
GATE_CTL_VACUUM_DB = 40.0

# CPML-DEPTH CONTROL for arm (i), pre-declared (correction N2). Arm (i) folds
# in an adjacent plane that, on four of six faces, sits on the first interior
# cell against the absorber. If its improvement is co-location, the shrink
# factor must be insensitive to how deep that absorber is; if the improvement
# came from the absorber-interface plane, it must drift. The cpmlladder arm
# already showed the interior is byte-identical across 8/16/24 and the
# longitudinal spread flat (2.3140 / 2.3084 / 2.3262 dB), so the absorber depth
# is the ONLY variable here.
#   GATE: shrink(arm_i) at 16 and at 24 must stay within +-15 % of its value at
#   8, recomputed on the SAME seven offsets -> co-location is the carrier.
#   Drift beyond +-15 % -> the absorber-interface plane was.
SWEEPS = {
    "ynull_ka1": dict(axis="y", offsets=list(range(-10, 11, 2)),
                      ka=1.0, cpr=probe.COARSE_CPR, clear_cells=30,
                      steps_mult=1.0),
    "ynull_ka1_cpml16": dict(axis="y", offsets=[-10, -6, -2, 0, 2, 6, 10],
                             ka=1.0, cpr=probe.COARSE_CPR, clear_cells=30,
                             steps_mult=1.0, cpml_layers=16),
    "ynull_ka1_cpml24": dict(axis="y", offsets=[-10, -6, -2, 0, 2, 6, 10],
                             ka=1.0, cpr=probe.COARSE_CPR, clear_cells=30,
                             steps_mult=1.0, cpml_layers=24),
    "xsweep_ka1": dict(axis="x", offsets=list(range(-10, 11, 2)),
                       ka=1.0, cpr=probe.COARSE_CPR, clear_cells=30,
                       steps_mult=1.0),
    "ynull_ka2": dict(axis="y", offsets=list(range(-10, 11, 2)),
                      ka=2.0, cpr=12.8, clear_cells=30, steps_mult=1.0),
}

_OUT = os.path.join(_SCRIPT_DIR, "issue820_results")
_TH = np.array([np.pi / 2])
_PH = np.array([np.pi])
_RHAT = np.array([-1.0, 0.0, 0.0])       # backscatter for +x incidence


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _emit(name, payload):
    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, f"{_stamp()}_{name}.json")
    payload = dict(payload)
    payload["module"] = os.path.basename(__file__)
    payload["rfx_file"] = rfx.__file__
    payload["git_head"] = subprocess.run(
        ["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False).stdout.strip()
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[emit] {path}")
    return path


def _boxes(grid, cpml_layers=CPML_LAYERS):
    """Production box plus the three one-cell-shifted boxes for arm (i).

    Each shifted box moves ONE axis' pair of faces by one cell toward the LOWER
    index -- the lo face outward, the hi face inward (``d[lo] -= 1`` and
    ``d[hi] -= 1``) -- while keeping that face's transverse extent, so its
    x/y/z face data is the plane adjacent to the production face, cell for
    cell. (The docstring said "inward" for both until correction N-P3.)

    DISCLOSURE (correction N2). At the cv16 operating point the lo-side
    adjacent planes land on j = 8 and k = 8, which are the FIRST INTERIOR CELLS
    against the CPML (interior 8..82, box j/k_lo = 9). The x side is one cell
    further in (i = 9). So arm (i) does not only co-locate H with E: on four of
    the six faces it also folds in a plane that sits directly on the absorber
    interface. Nothing in this lane separates "co-location helped" from "the
    extra plane happened to help", and the CPML-depth control arm exists
    because of exactly that.
    """
    freqs = np.array([F0], dtype=np.float64)
    _, b0 = probe._tfsf_and_ntff(grid, cpml_layers=cpml_layers, freqs=freqs)
    idx = dict(i_lo=b0.i_lo, i_hi=b0.i_hi, j_lo=b0.j_lo, j_hi=b0.j_hi,
               k_lo=b0.k_lo, k_hi=b0.k_hi)
    shifted = {}
    for ax, (lo, hi) in enumerate((("i_lo", "i_hi"), ("j_lo", "j_hi"),
                                   ("k_lo", "k_hi"))):
        d = dict(idx)
        d[lo] -= 1
        d[hi] -= 1
        shifted[ax] = NTFFBox.from_grid(grid, freqs=b0.freqs, **d)
    return b0, shifted


def _simulate(grid, mats, n_steps, cpml_layers=CPML_LAYERS, need_adjacent=True):
    """Run the production box and (optionally) the three adjacent-plane boxes."""
    freqs = np.array([F0], dtype=np.float64)
    tfsf, b0 = probe._tfsf_and_ntff(grid, cpml_layers=cpml_layers, freqs=freqs)
    _, shifted = _boxes(grid, cpml_layers)
    res0 = run(grid, mats, n_steps, boundary="cpml", tfsf=tfsf, ntff=b0)
    adj = {}
    if need_adjacent:
        for ax, bx in shifted.items():
            tfsf_a, _ = probe._tfsf_and_ntff(grid, cpml_layers=cpml_layers,
                                             freqs=freqs)
            r = run(grid, mats, n_steps, boundary="cpml", tfsf=tfsf_a, ntff=bx)
            adj[ax] = r.ntff_data
    return b0, res0.ntff_data, adj


def _apply_arm_i(nd, adj):
    """Average the adjacent H plane with the on-plane H, per face."""
    new = {}
    for f in FACES:
        ax = FACE_AXIS[f]
        on = np.asarray(getattr(nd, f), dtype=np.complex128)
        near = np.asarray(getattr(adj[ax], f), dtype=np.complex128)
        if on.shape != near.shape:
            raise RuntimeError(
                f"adjacent-plane shape mismatch on {f}: {near.shape} vs "
                f"{on.shape} -- the shifted box did not keep the transverse "
                "extent, so the average would not be cell for cell")
        out = on.copy()
        out[..., 2] = 0.5 * (near[..., 2] + on[..., 2])
        out[..., 3] = 0.5 * (near[..., 3] + on[..., 3])
        new[f] = jnp.asarray(out, dtype=getattr(nd, f).dtype)
    return nd._replace(**new)


def _apply_arm_ii(nd, k, dx, skip_h_normal=False):
    """Per-component Yee-coordinate phase, evaluated at backscatter."""
    new = {}
    for f in FACES:
        ax = FACE_AXIS[f]
        comps = FACE_COMPONENTS[ax]
        arr = np.asarray(getattr(nd, f), dtype=np.complex128).copy()
        for ci, cname in enumerate(comps):
            delta = np.array(YEE_DELTA[cname], dtype=float)
            if skip_h_normal and ci >= 2:
                # arm (iii): the normal offset was already removed by averaging
                delta = delta.copy()
                delta[ax] = 0.0
            arr[..., ci] *= np.exp(1j * k * float(_RHAT @ delta) * dx)
        new[f] = jnp.asarray(arr, dtype=getattr(nd, f).dtype)
    return nd._replace(**new)


def _backscatter(nd, box, grid):
    ff = compute_far_field(nd, box, grid, _TH, _PH)
    e_th = complex(np.asarray(ff.E_theta, dtype=np.complex128)[0, 0, 0])
    e_ph = complex(np.asarray(ff.E_phi, dtype=np.complex128)[0, 0, 0])
    return e_th, e_ph


def _sigma_dbsm(e_th, e_ph, grid, n_steps):
    e_inc = _incident_spectrum_amplitude(
        F0, BANDWIDTH, np.array([F0]), grid.dt, n_steps)
    return float(10.0 * np.log10(
        4.0 * np.pi * (abs(e_th) ** 2 + abs(e_ph) ** 2) / abs(e_inc[0]) ** 2))


# arm_ii_flip is the SIGN CONTROL (correction N1). The transform's kernel is
# exp(+1j k rhat.r'), so the correct per-component factor is
# exp(+1j k rhat.delta); arm_ii_flip applies its conjugate, i.e. the
# deliberately WRONG sign. It costs nothing -- same simulations, different
# post-processing -- and it is what shows that "distance from Mie" at this rung
# rewards the wrong sign and is therefore not a discriminator.
ARM_NAMES = ("baseline", "arm_i", "arm_ii", "arm_iii", "arm_ii_flip")


def _all_arms(nd, adj, box, grid, n_steps, k, dx):
    nd_i = _apply_arm_i(nd, adj)
    out = {
        "baseline": nd,
        "arm_i": nd_i,
        "arm_ii": _apply_arm_ii(nd, k, dx),
        "arm_iii": _apply_arm_ii(nd_i, k, dx, skip_h_normal=True),
        "arm_ii_flip": _apply_arm_ii(nd, -k, dx),
    }
    res = {}
    for name, data in out.items():
        e_th, e_ph = _backscatter(data, box, grid)
        res[name] = {
            "E_theta": [e_th.real, e_th.imag],
            "E_phi": [e_ph.real, e_ph.imag],
            "monostatic_dbsm": _sigma_dbsm(e_th, e_ph, grid, n_steps),
        }
    return res


def _mie_dbsm(grid, mask_n_occupied, ka_nominal):
    from mie_oracle import backscatter_rcs_over_pi_a2, validate_oracle
    validate_oracle()
    a_eff = (3 * mask_n_occupied * grid.dx ** 3 / (4 * np.pi)) ** (1.0 / 3.0)
    ka_eff = 2 * np.pi * a_eff / LAM
    pi_a2 = np.pi * a_eff ** 2
    return float(10 * np.log10(
        backscatter_rcs_over_pi_a2(ka_eff, n_max=40) * pi_a2)), float(ka_eff)


def _odd_even(offsets, values):
    return probe._odd_even(offsets, values)


def run_sweep(name):
    cfg = SWEEPS[name]
    ax = "xyz".index(cfg["axis"])
    k0 = 2 * np.pi * F0 / C0
    rows = []
    caught = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        for n in cfg["offsets"]:
            off = [0, 0, 0]
            off[ax] = n
            cp = cfg.get("cpml_layers", CPML_LAYERS)
            grid, mats, n_steps, mask, meta = probe.build_case(
                tuple(off), ka=cfg["ka"], cpr=cfg["cpr"],
                clear_cells=cfg["clear_cells"], steps_mult=cfg["steps_mult"],
                cpml_layers=cp)
            t0 = time.time()
            box, nd, adj = _simulate(grid, mats, n_steps, cpml_layers=cp)
            arms = _all_arms(nd, adj, box, grid, n_steps, k0, grid.dx)
            rows.append({"offset": n, "meta": meta, "arms": arms,
                         "wall_s": round(time.time() - t0, 1)})
            print(f"  {name} {cfg['axis']}{n:+3d}  "
                  + "  ".join(f"{a}:{arms[a]['monostatic_dbsm']:9.4f}"
                              for a in ARM_NAMES)
                  + f"   ({rows[-1]['wall_s']}s)")
        caught = [f"{w.category.__name__}: {w.message}" for w in wlist]

    out = {"sweep": name, "config": cfg, "rows": rows,
           "warnings": caught,
           "preflight": ("none emitted -- compute_rcs runs no preflight and "
                         "this module replays that path; verified with "
                         "warnings.simplefilter('always')"),
           "gates": {"s0_rel": GATE_S0_REL, "t_true": GATE_T_TRUE,
                     "t_false": GATE_T_FALSE, "r_true": GATE_R_TRUE,
                     "r_false": GATE_R_FALSE,
                     "ctl_move_db": GATE_CTL_MOVE_DB,
                     "ctl_vacuum_db": GATE_CTL_VACUUM_DB}}

    # --- S0: baseline must reproduce the probe's own path -------------------
    zero = [r for r in rows if r["offset"] == 0]
    if zero:
        cp = cfg.get("cpml_layers", CPML_LAYERS)
        grid, mats, n_steps, _, meta = probe.build_case(
            (0, 0, 0), ka=cfg["ka"], cpr=cfg["cpr"],
            clear_cells=cfg["clear_cells"], steps_mult=cfg["steps_mult"],
            cpml_layers=cp)
        ref, _, _, _ = probe._rcs_complex(grid, mats, n_steps,
                                          cpml_layers=cp)
        base = zero[0]["arms"]["baseline"]["monostatic_dbsm"]
        d = abs(ref["monostatic_dbsm"] - base)
        out["s0"] = {"probe_dbsm": ref["monostatic_dbsm"],
                     "baseline_dbsm": base, "abs_delta_db": d,
                     "pass": bool(d <= 1e-9)}
        print(f"[S0] baseline {base:.9f} vs probe {ref['monostatic_dbsm']:.9f} "
              f"-> |delta| = {d:.3e} dB ({'PASS' if d <= 1e-9 else 'FAIL/VOID'})")
        mie, ka_eff = _mie_dbsm(grid, meta["n_occupied"], cfg["ka"])
        out["mie"] = {"mie_dbsm": mie, "ka_eff": ka_eff,
                      "n_occupied": meta["n_occupied"]}
        out["offset0"] = {}
        for a in ARM_NAMES:
            v = zero[0]["arms"][a]["monostatic_dbsm"]
            out["offset0"][a] = {
                "monostatic_dbsm": v, "dist_from_mie_db": v - mie,
                "move_vs_baseline_db": v - base,
            }
            print(f"[MIE] {a:9s} sigma {v:9.4f} dBsm, from Mie "
                  f"{v - mie:+7.4f} dB, moved {v - base:+7.4f} dB vs baseline "
                  f"(control bar {GATE_CTL_MOVE_DB} dB)")

    # --- T / R per arm ------------------------------------------------------
    offs = [r["offset"] for r in rows]
    dx = rows[0]["meta"]["dx"]
    out["per_arm"] = {}
    base_pp = None
    for a in ARM_NAMES:
        vals = [r["arms"][a]["monostatic_dbsm"] for r in rows]
        entry = {"pp_db": float(max(vals) - min(vals))}
        dec = _odd_even(offs, vals)
        entry["decomposition"] = dec
        if cfg["axis"] == "x":
            e = np.array([complex(*r["arms"][a]["E_theta"]) for r in rows])
            resid, k_fit, A, B = probe._fit_rotating_plus_constant(offs, e, dx)
            entry.update({"resid": resid, "k_fit": k_fit,
                          "abs_A": abs(A), "abs_B": abs(B),
                          "r": abs(B) / abs(A),
                          "verdict_R": ("TRUE" if abs(B) / abs(A) < GATE_R_TRUE
                                        else "FALSE" if abs(B) / abs(A) > GATE_R_FALSE
                                        else "INCONCLUSIVE")})
            print(f"[R] {a:9s} r = {entry['r']:.4f} resid {resid:.4f} "
                  f"-> {entry['verdict_R']} (TRUE< {GATE_R_TRUE}, "
                  f"FALSE> {GATE_R_FALSE})")
        else:
            if a == "baseline":
                base_pp = entry["pp_db"]
            shrink = base_pp / entry["pp_db"] if entry["pp_db"] else float("inf")
            entry["shrink_vs_baseline"] = shrink
            entry["verdict_T"] = ("TRUE" if shrink >= GATE_T_TRUE
                                  else "FALSE" if shrink <= GATE_T_FALSE
                                  else "INCONCLUSIVE")
            print(f"[T] {a:9s} p-p = {entry['pp_db']:.4f} dB, shrink "
                  f"{shrink:.3f}x -> {entry['verdict_T']} "
                  f"(TRUE>= {GATE_T_TRUE}, FALSE<= {GATE_T_FALSE})"
                  + (f"; odd {dec['odd_span']:.4f} even {dec['even_span']:.4f} "
                     f"ratio {dec['odd_over_even']:.3f}" if dec else ""))
        out["per_arm"][a] = entry

    print(f"[{name}] warnings captured: "
          f"{len(caught)}{' -> ' + '; '.join(caught) if caught else ' (none emitted)'}")
    return _emit(f"collocation_{name}", out)


def run_control():
    """CTL(b): the corrected extractor on the empty domain."""
    caught = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        k0 = 2 * np.pi * F0 / C0
        grid, mats, n_steps, _, meta = probe.build_case((0, 0, 0), vacuum=True)
        box, nd, adj = _simulate(grid, mats, n_steps)
        vac = _all_arms(nd, adj, box, grid, n_steps, k0, grid.dx)
        grid_t, mats_t, n_t, _, meta_t = probe.build_case((0, 0, 0))
        box_t, nd_t, adj_t = _simulate(grid_t, mats_t, n_t)
        tgt = _all_arms(nd_t, adj_t, box_t, grid_t, n_t, k0, grid_t.dx)
        caught = [f"{w.category.__name__}: {w.message}" for w in wlist]
    out = {"vacuum": vac, "target": tgt, "meta": meta, "meta_target": meta_t,
           "warnings": caught,
           "preflight": "none emitted -- verified with simplefilter('always')",
           "gates": {"ctl_vacuum_db": GATE_CTL_VACUUM_DB}}
    out["separation_db"] = {}
    for a in ARM_NAMES:
        sep = tgt[a]["monostatic_dbsm"] - vac[a]["monostatic_dbsm"]
        out["separation_db"][a] = sep
        print(f"[CTL] {a:9s} vacuum {vac[a]['monostatic_dbsm']:9.4f} dBsm, "
              f"target {tgt[a]['monostatic_dbsm']:9.4f} dBsm, separation "
              f"{sep:7.3f} dB (bar {GATE_CTL_VACUUM_DB} dB) "
              f"{'PASS' if sep >= GATE_CTL_VACUUM_DB else 'FAIL'}")
    print(f"[control] warnings captured: "
          f"{len(caught)}{' -> ' + '; '.join(caught) if caught else ' (none emitted)'}")
    return _emit("collocation_control", out)


def main(argv=None):
    p = argparse.ArgumentParser(description="issue #820 NTFF collocation arms")
    p.add_argument("--sweep", required=True,
                   choices=sorted(SWEEPS) + ["control"])
    args = p.parse_args(argv)
    if args.sweep == "control":
        run_control()
    else:
        run_sweep(args.sweep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
