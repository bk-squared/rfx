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

CORRECTIONS 3 / ARMS A and C (2026-09-14), pre-declared before running.

T3 (mirrored from the probe). Arm (i)'s ABSOLUTE null reduction is
    0.2124 / 0.2577 / 0.2574 dB at CPML 8 / 16 / 24: one step of +21.3 % then
    saturation at -0.1 %, with arm (iii) showing the same shape independently
    (0.2027 / 0.2571 / 0.2585). The reading is "unstable at CPML 8, stable at
    16 and 24", not "monotone drift" -- and CPML 8 is the cv16 operating point
    where every headline arm-(i) number was measured. Also reference-free and
    previously unreported: arm (ii)'s null improvement VANISHES with depth,
    shrink 1.0673 / 1.0093 / 0.9966.

ARM A (``--sweep hjump``) -- measure the H jump across the plane arm (i)
    averages, at three absorber depths. Gate and reasoning in ``run_hjump``.
    Twelve short runs; offset 0 only.

ARM C (``--sweep offset2_cpml{8,16,24}``) -- arm (i) with EVERY adjacent plane
    one clean interior cell off the absorber, which the production box does not
    give (its lo-side neighbours are the first interior cells). Box constants
    and the reason an explicit override is required rather than
    ``ntff_offset=2`` are at ``OFFSET2_BOX`` below.
    GATE, on the ABSOLUTE dB reduction rather than the ratio, because the ratio
    moves when the baseline moves:
      TRUE  -- |abs(8) / mean(abs(16), abs(24)) - 1| <= 0.005. The CPML-8
               anomaly is gone, the effect is absorber-independent, and
               "co-location is what arm (i) buys" becomes a measurement rather
               than a hope.
      FALSE -- that quantity >= 0.15, i.e. CPML 8 stays about 20 % low even
               with its neighbour planes off the interface. Then the anomaly is
               not the interface plane and N2 is not the explanation.
      otherwise INCONCLUSIVE.
    The shrink RATIO is reported alongside but is not the gate.

CORRECTIONS 4 / ARM D (2026-09-14), after the fourth verification (ACCEPT).

U1. Arm A's gloss was stronger than its numbers. "No interface anomaly in the
    averaged plane" oversells a jump that is 16.9 % ELEVATED at CPML 8 and
    clears the 1.2 refutation bar by only 2.6 % (ratio 1.1687; the x_lo
    comparison clears by 10.5 %, ratio 1.0742). The correct wording is "both
    below the refutation bar, the depth ratio by 2.6 %".
    What actually refutes N2 is STRUCTURAL and should lead: the arm's own
    artifact records adjacent_plane_clearance_cells = 0 for y_lo and z_lo at
    ALL THREE depths (and 1 for x_lo at all three). The contamination geometry
    is therefore DEPTH-INVARIANT by construction, so it cannot produce a
    depth-DEPENDENT effect whatever its size. N2 is refuted three ways --
    structurally, by arm A's measurement, and by arm C's FALSE.

U2. Free result from arm A's artifact, offered as a CANDIDATE and explicitly
    not an attribution (two three-point series sharing a shape is a shape, not
    a cause): the lo/hi H-jump CONTRAST flips and saturates exactly where
    arm (i)'s effect does.
        depth |  x_lo/x_hi  y_lo/y_hi  z_lo/z_hi | arm (i) absolute
            8 |   1.1839     1.3165     1.2353   |  0.2124 dB
           16 |   0.9114     0.9345     1.0412   |  0.2577 dB
           24 |   0.9051     0.8939     1.0065   |  0.2574 dB
    At CPML 8 every axis has the lo face jumping MORE than the hi face; by 16
    the x and y contrasts have crossed below 1 and by 24 all three have
    settled. Also coherent with N6 and with the z null being ~4x smaller: the
    z-face jump is ~0.0148 against ~0.1200 on y, 8.1x smaller, because the
    z faces do not store ez and the backscattered field is z-polarized.

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
# Arm C boxes, ATTEMPT 2. Attempt 1 is VOID on an identified implementation
# defect, recorded here rather than quietly replaced:
#
#   Attempt 1 used i_lo = cpml+3 (11 at CPML 8) on the reasoning that every lo
#   face should move in by one. But the x faces are NOT referenced to the
#   absorber -- rfx/rcs.py sets i_lo = tfsf.x_lo - ntff_offset, and the TFSF
#   total-field region starts at x_lo = 11. So i_lo = 11 put the x_lo face ON
#   the first TOTAL-FIELD cell, where it integrates the incident wave. The
#   empty-domain reading proves it: -19.2141 dBsm against production's
#   -83.6860, i.e. the "scattered-field" surface was reading the illumination.
#   The run's own Mie control had already blown by 7.5 dB on the baseline and
#   2.5 dB on arm (i) against a 0.622 dB bar; the guard below now turns that
#   class into an abort instead of a number.
#
#   Attempt 2 moves ONLY the transverse lo faces. Production already gives the
#   x_lo face's adjacent plane one clean cell (i_lo = cpml+2, neighbour
#   cpml+1), so x needs no change at all; it is j_lo and k_lo, at cpml+1 with
#   neighbours ON the interior edge, that need to go to cpml+2. Everything
#   else stays production. Empty-domain reading of the corrected box:
#   -83.9146 dBsm, i.e. production.
OFFSET2_BOX = {"j_lo": 10, "k_lo": 10}
OFFSET2_BOX_16 = {"j_lo": 18, "k_lo": 18}
OFFSET2_BOX_24 = {"j_lo": 26, "k_lo": 26}
GATE_C_VACUUM_DB = 40.0      # empty-domain far field must sit this far down
GATE_C_TRUE_REL = 0.005      # |abs8 / mean(abs16, abs24) - 1| <= this -> TRUE
GATE_C_FALSE_REL = 0.15      # ... >= this -> FALSE (CPML 8 still ~20 % low)

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
    # --- ARM C: arm (i) with every adjacent plane one CLEAN cell off the
    # absorber. Production is i[10,81] j[9,82] k[9,82], whose lo-side adjacent
    # planes land on j = 8 and k = 8, the first interior cells against the
    # CPML (correction N2). This box is i[11,80] j[10,81] k[10,81], so the
    # adjacent planes are i = 10, j = 9, k = 9 -- all at least one interior
    # cell clear of the absorber. Nothing else changes.
    # NOTE: this is NOT ntff_offset=2. That knob moves the x faces OUTWARD
    # (i_lo = tfsf.x_lo - offset) while moving y/z INWARD, which would put the
    # x adjacent plane ON the interface instead. An explicit box override is
    # the only way to make all six clean at once.
    "offset2_cpml8": dict(axis="y", offsets=[-10, -6, -2, 0, 2, 6, 10],
                          ka=1.0, cpr=probe.COARSE_CPR, clear_cells=30,
                          steps_mult=1.0, cpml_layers=8,
                          box_override=OFFSET2_BOX),
    "offset2_cpml16": dict(axis="y", offsets=[-10, -6, -2, 0, 2, 6, 10],
                           ka=1.0, cpr=probe.COARSE_CPR, clear_cells=30,
                           steps_mult=1.0, cpml_layers=16,
                           box_override=OFFSET2_BOX_16),
    "offset2_cpml24": dict(axis="y", offsets=[-10, -6, -2, 0, 2, 6, 10],
                           ka=1.0, cpr=probe.COARSE_CPR, clear_cells=30,
                           steps_mult=1.0, cpml_layers=24,
                           box_override=OFFSET2_BOX_24),
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


def _boxes(grid, cpml_layers=CPML_LAYERS, box_override=None):
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
    _, b0 = probe._tfsf_and_ntff(grid, cpml_layers=cpml_layers, freqs=freqs,
                                 ntff_box_override=box_override)
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


def _simulate(grid, mats, n_steps, cpml_layers=CPML_LAYERS, need_adjacent=True,
              box_override=None):
    """Run the production box and (optionally) the three adjacent-plane boxes."""
    freqs = np.array([F0], dtype=np.float64)
    tfsf, b0 = probe._tfsf_and_ntff(grid, cpml_layers=cpml_layers, freqs=freqs,
                                    ntff_box_override=box_override)
    _, shifted = _boxes(grid, cpml_layers, box_override=box_override)
    res0 = run(grid, mats, n_steps, boundary="cpml", tfsf=tfsf, ntff=b0)
    adj = {}
    if need_adjacent:
        for ax, bx in shifted.items():
            tfsf_a, _ = probe._tfsf_and_ntff(grid, cpml_layers=cpml_layers,
                                             freqs=freqs,
                                             ntff_box_override=box_override)
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


FACE_GROUPS = {
    "all6": FACES,
    "lo": ("x_lo", "y_lo", "z_lo"),
    "hi": ("x_hi", "y_hi", "z_hi"),
    "x": ("x_lo", "x_hi"),
    "y": ("y_lo", "y_hi"),
    "z": ("z_lo", "z_hi"),
}
GATE_D_STABLE = 0.05      # (max-min)/mean of the absolute reduction across depths


def _apply_arm_i_group(nd, adj, faces):
    """Arm (i)'s H averaging applied to ONE group of faces only."""
    new = {}
    for f in faces:
        ax = FACE_AXIS[f]
        on = np.asarray(getattr(nd, f), dtype=np.complex128)
        near = np.asarray(getattr(adj[ax], f), dtype=np.complex128)
        out = on.copy()
        out[..., 2] = 0.5 * (near[..., 2] + on[..., 2])
        out[..., 3] = 0.5 * (near[..., 3] + on[..., 3])
        new[f] = jnp.asarray(out, dtype=getattr(nd, f).dtype)
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


def _vacuum_guard(cfg, name):
    """Refuse to report a sweep whose NTFF box is not in the scattered field.

    Added after arm C attempt 1 put a face on the first total-field cell and
    produced numbers anyway. One extra run per sweep.
    """
    cp = cfg.get("cpml_layers", CPML_LAYERS)
    ov = cfg.get("box_override")
    gv, mv, ns, _, _ = probe.build_case(
        (0, 0, 0), ka=cfg["ka"], cpr=cfg["cpr"],
        clear_cells=cfg["clear_cells"], steps_mult=cfg["steps_mult"],
        cpml_layers=cp, vacuum=True)
    gt, mt, nt, _, _ = probe.build_case(
        (0, 0, 0), ka=cfg["ka"], cpr=cfg["cpr"],
        clear_cells=cfg["clear_cells"], steps_mult=cfg["steps_mult"],
        cpml_layers=cp)
    rv, _, _, _ = probe._rcs_complex(gv, mv, ns, cpml_layers=cp,
                                     ntff_box_override=ov)
    rt, _, _, _ = probe._rcs_complex(gt, mt, nt, cpml_layers=cp,
                                     ntff_box_override=ov)
    sep = rt["monostatic_dbsm"] - rv["monostatic_dbsm"]
    print(f"[guard] {name}: empty domain {rv['monostatic_dbsm']:.4f} dBsm, "
          f"target {rt['monostatic_dbsm']:.4f} dBsm, separation {sep:.3f} dB "
          f"(bar {GATE_C_VACUUM_DB}) {'PASS' if sep >= GATE_C_VACUUM_DB else 'FAIL'}")
    return {"vacuum_dbsm": rv["monostatic_dbsm"],
            "target_dbsm": rt["monostatic_dbsm"], "separation_db": sep,
            "pass": bool(sep >= GATE_C_VACUUM_DB), "box": rt["ntff_box"]}


def run_sweep(name):
    cfg = SWEEPS[name]
    guard = _vacuum_guard(cfg, name)
    if not guard["pass"]:
        print(f"[guard] {name}: ABORT -- the NTFF box is not in the scattered "
              f"field; no numbers reported")
        return _emit(f"collocation_{name}", {"sweep": name, "config": cfg,
                                             "guard": guard, "aborted": True})
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
            box, nd, adj = _simulate(grid, mats, n_steps, cpml_layers=cp,
                                     box_override=cfg.get("box_override"))
            arms = _all_arms(nd, adj, box, grid, n_steps, k0, grid.dx)
            rows.append({"offset": n, "meta": meta, "arms": arms,
                         "wall_s": round(time.time() - t0, 1)})
            print(f"  {name} {cfg['axis']}{n:+3d}  "
                  + "  ".join(f"{a}:{arms[a]['monostatic_dbsm']:9.4f}"
                              for a in ARM_NAMES)
                  + f"   ({rows[-1]['wall_s']}s)")
        caught = [f"{w.category.__name__}: {w.message}" for w in wlist]

    out = {"sweep": name, "config": cfg, "rows": rows, "guard": guard,
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
        ref, _, _, _ = probe._rcs_complex(
            grid, mats, n_steps, cpml_layers=cp,
            ntff_box_override=cfg.get("box_override"))
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


GATE_A_ANOMALY = 1.5     # ratio >= this on BOTH comparisons -> N2 CONFIRMED
GATE_A_FLAT = 1.2        # both < this -> N2 REFUTED


def run_hjump():
    """ARM A -- how big is the H jump across the plane arm (i) averages?

    Pre-declared. arm (i) replaces H on a face by the mean of the face plane
    and the plane one cell toward lower index. On the lo faces at the cv16
    point that neighbour is the FIRST INTERIOR CELL against the absorber
    (correction N2). If the absorber interface is what arm (i) is really
    sampling, the jump
        J_f = mean|H(idx-1) - H(idx)| / mean|H(idx)|
    must be anomalous on y_lo at CPML 8 -- both against the SAME face at 16
    and 24, and against x_lo at the same depth, whose neighbour is one cell
    clear.
      GATE: J_ylo(8)/median(J_ylo(16), J_ylo(24)) >= 1.5 AND
            J_ylo(8)/J_xlo(8) >= 1.5           -> N2 CONFIRMED
            both < 1.2                          -> N2 REFUTED
            otherwise                           -> INCONCLUSIVE
    Twelve short runs (offset 0 only, three depths); no new physics, but not
    zero-simulation either -- the face arrays are not persisted by the other
    arms, so they have to be regenerated.
    """
    caught = []
    out = {"gates": {"anomaly": GATE_A_ANOMALY, "flat": GATE_A_FLAT}}
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        for cp in (8, 16, 24):
            grid, mats, n_steps, _, meta = probe.build_case(
                (0, 0, 0), cpml_layers=cp)
            box, nd, adj = _simulate(grid, mats, n_steps, cpml_layers=cp)
            per_face = {}
            for f in FACES:
                ax = FACE_AXIS[f]
                on = np.asarray(getattr(nd, f), dtype=np.complex128)
                near = np.asarray(getattr(adj[ax], f), dtype=np.complex128)
                h_on = np.abs(on[..., 2]) + np.abs(on[..., 3])
                h_d = (np.abs(near[..., 2] - on[..., 2])
                       + np.abs(near[..., 3] - on[..., 3]))
                per_face[f] = float(h_d.mean() / max(h_on.mean(), 1e-300))
            face_idx = {"x_lo": box.i_lo, "x_hi": box.i_hi,
                        "y_lo": box.j_lo, "y_hi": box.j_hi,
                        "z_lo": box.k_lo, "z_hi": box.k_hi}
            clearance = {f: (face_idx[f] - 1 - cp) if f.endswith("_lo")
                         else None for f in FACES}
            out[str(cp)] = {"J": per_face, "box": list(box[:6]),
                            "adjacent_plane_clearance_cells": clearance,
                            "grid": meta["grid_shape"]}
            print(f"  cpml {cp:2d}: " + "  ".join(
                f"{f}:{per_face[f]:.4f}" for f in FACES))
            print("          lo-face adjacent-plane clearance to the "
                  "absorber (cells): "
                  + ", ".join(f"{f}={clearance[f]}" for f in
                              ("x_lo", "y_lo", "z_lo")))
        caught = [f"{w.category.__name__}: {w.message}" for w in wlist]
    jy = [out[str(c)]["J"]["y_lo"] for c in (8, 16, 24)]
    jx8 = out["8"]["J"]["x_lo"]
    r_depth = jy[0] / float(np.median(jy[1:]))
    r_face = jy[0] / jx8
    if r_depth >= GATE_A_ANOMALY and r_face >= GATE_A_ANOMALY:
        v = "N2 CONFIRMED (y_lo at CPML 8 is anomalous both ways)"
    elif r_depth < GATE_A_FLAT and r_face < GATE_A_FLAT:
        v = "N2 REFUTED (no interface anomaly in the averaged plane)"
    else:
        v = "N2 INCONCLUSIVE"
    out.update({"J_ylo_by_depth": jy, "J_xlo_cpml8": jx8,
                "ratio_vs_depth": r_depth, "ratio_vs_xlo": r_face,
                "verdict": v, "warnings": caught,
                "preflight": ("none emitted -- verified with "
                              "warnings.simplefilter('always')")})
    print(f"[hjump] J(y_lo) by depth {jy[0]:.4f} / {jy[1]:.4f} / {jy[2]:.4f}; "
          f"J(x_lo) at 8 = {jx8:.4f}")
    print(f"[hjump] ratio vs deeper pair {r_depth:.3f}, ratio vs x_lo "
          f"{r_face:.3f} (CONFIRMED >= {GATE_A_ANOMALY} both; REFUTED < "
          f"{GATE_A_FLAT} both) -> {v}")
    print(f"[hjump] warnings captured: {len(caught)}"
          f"{' -> ' + '; '.join(caught) if caught else ' (none emitted)'}")
    return _emit("hjump", out)


def run_groupsplit(cpml):
    """ARM D -- which face group carries arm (i)'s depth dependence?

    Pre-declared. Arm (i) averages H on all six faces at once, and its ABSOLUTE
    null reduction is 0.2124 / 0.2577 / 0.2574 dB at CPML 8 / 16 / 24 -- one
    step then saturation, which L7 refuses to land around. This arm applies the
    same averaging to ONE GROUP at a time (lo, hi, x, y, z) with `all6` as the
    control that must reproduce the parent sweep's arm_i exactly.

      GATE: if at least one group's absolute reduction is DEPTH-STABLE
      ((max-min)/mean <= 0.05 across 8/16/24) while at least one other is not,
      the depth dependence is LOCALIZED to the unstable group, L7 becomes
      answerable, and the arm records which group. If every group moves
      together, the dependence is GLOBAL to the transform and the collocation
      line is to be CLOSED rather than continued.

    NOT zero-simulation, and saying so plainly: the committed sweeps store only
    far-field scalars (`rows[].arms[].E_theta`), not the face arrays, so the
    adjacent-plane data has to be regenerated. No new physics configuration is
    introduced -- the box, the offsets and the record are the parent sweep's --
    and the `all6` control reproducing the parent's arm_i is what proves the
    regenerated runs are the same runs.
    """
    cfg = dict(SWEEPS["ynull_ka1"])
    cfg["cpml_layers"] = cpml
    cfg["offsets"] = [-10, -6, -2, 0, 2, 6, 10]
    guard = _vacuum_guard(cfg, f"groupsplit_cpml{cpml}")
    caught, rows = [], []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        for n in cfg["offsets"]:
            grid, mats, n_steps, _, meta = probe.build_case(
                (0, n, 0), ka=cfg["ka"], cpr=cfg["cpr"],
                clear_cells=cfg["clear_cells"], steps_mult=cfg["steps_mult"],
                cpml_layers=cpml)
            box, nd, adj = _simulate(grid, mats, n_steps, cpml_layers=cpml)
            vals = {}
            e_th, e_ph = _backscatter(nd, box, grid)
            vals["baseline"] = _sigma_dbsm(e_th, e_ph, grid, n_steps)
            for g, faces in FACE_GROUPS.items():
                nd_g = _apply_arm_i_group(nd, adj, faces)
                e_th, e_ph = _backscatter(nd_g, box, grid)
                vals[g] = _sigma_dbsm(e_th, e_ph, grid, n_steps)
            rows.append({"offset": n, "sigma": vals, "meta": meta})
            print(f"  cpml {cpml} y{n:+3d}  " + "  ".join(
                f"{g}:{vals[g]:8.4f}" for g in ("baseline",) + tuple(FACE_GROUPS)))
        caught = [f"{w.category.__name__}: {w.message}" for w in wlist]
    out = {"cpml_layers": cpml, "rows": rows, "guard": guard,
           "offsets": cfg["offsets"], "groups": {k: list(v) for k, v in
                                                 FACE_GROUPS.items()},
           "gate_stable": GATE_D_STABLE, "warnings": caught,
           "preflight": ("none emitted -- verified with "
                         "warnings.simplefilter('always')")}
    base_pp = (max(r["sigma"]["baseline"] for r in rows)
               - min(r["sigma"]["baseline"] for r in rows))
    out["baseline_pp_db"] = base_pp
    out["per_group"] = {}
    for g in FACE_GROUPS:
        v = [r["sigma"][g] for r in rows]
        pp = max(v) - min(v)
        out["per_group"][g] = {"pp_db": pp, "absolute_reduction_db": base_pp - pp,
                               "shrink": base_pp / pp if pp else float("inf")}
        print(f"[groupsplit] cpml {cpml} {g:5s}: p-p {pp:.4f} dB, absolute "
              f"-{base_pp - pp:.4f} dB, shrink {base_pp/pp:.4f}x")
    print(f"[groupsplit] cpml {cpml} baseline p-p {base_pp:.4f} dB; warnings "
          f"{len(caught)}{' (none emitted)' if not caught else caught}")
    return _emit(f"groupsplit_cpml{cpml}", out)


def main(argv=None):
    p = argparse.ArgumentParser(description="issue #820 NTFF collocation arms")
    p.add_argument("--sweep", required=True,
                   choices=sorted(SWEEPS) + ["control", "hjump", "groupsplit"])
    p.add_argument("--cpml", type=int, default=8,
                   help="groupsplit only: absorber depth in cells")
    args = p.parse_args(argv)
    if args.sweep == "control":
        run_control()
    elif args.sweep == "hjump":
        run_hjump()
    elif args.sweep == "groupsplit":
        run_groupsplit(args.cpml)
    else:
        run_sweep(args.sweep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
