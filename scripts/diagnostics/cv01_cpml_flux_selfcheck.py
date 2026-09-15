"""Measure cv01's straight-guide flux self-check failure under CPML (issue #813).

It measures. It does not attribute: the arms below show the guided channel
conserves under both boundaries and that cv01's G2 integrates the off-guide
region too, but WHY that region carries net backward power under CPML is not
established by anything here, and the pre-declaration's addendum names the
measurement that would settle it.

Pre-declaration: ``docs/design_notes/cv01_cpml_flux_selfcheck_predeclaration.md``
(gate, falsifier, control and the prediction, all written before this ran).

What this measures
------------------
``mean_self`` exactly as ``validation/crossval/01_waveguide_bend.py`` defines
it -- the band mean of the smoothed straight-guide ``out/in`` flux ratio over
the script's own ``above`` mask -- on Run 1 (the straight-guide
self-calibration arm) only. The bend arm and the Meep leg play no part;
``mean_self`` is a self-invariant, so no external solver is needed and none is
imported.

Six arms, differing in the boundary and in the monitor window and in nothing
else:

===============  ========  ===========================================
arm              boundary  monitor window
===============  ========  ===========================================
upml_full        upml      full padded plane (cv01 as committed) -- CONTROL
cpml_full        cpml      full padded plane (cv01 as committed) -- #813
upml_interior    upml      ``size=(sy, dx)`` -> interior cells only
cpml_interior    cpml      ``size=(sy, dx)`` -> interior cells only
cpml_aperture    cpml      ``size=(2*w_wg, dx)`` about the guide centre
upml_aperture    upml      ``size=(2*w_wg, dx)`` -- control for the row above
===============  ========  ===========================================

The six arms run at ``cpml_n`` = 10 layers, which is the rig #813's number
came from. cv01's ``RFX_BOUNDARY=cpml`` path moved to 20 layers afterwards, on
the evidence of the sweep below, so these six arms are the HISTORICAL CPML rig
and not today's CPML variant. They are left at 10 deliberately: the committed
artifact's control is the committed number, and re-pointing the arms would
silently restate #813's measurement as something else. ``--mode layer-sweep``
is where 20 layers is measured.

``size=None`` is cv01's registration today and keeps the legacy full padded
plane (``rfx/runners/uniform.py:74``). A finite ``size=`` clamps to the
physical interior and excludes CPML and bounding-node slots (PR #990, issue
#910). The ``interior`` arms are therefore the falsifier for "the missing
power is absorber cells inside the integration plane": if clipping them out
does not move ``mean_self``, that attribution is refuted.

Layer sweep (``--mode layer-sweep``)
------------------------------------
The addendum's Arm 1, pre-declared in the same note and corrected there under
"Correction 2026-09-14": ``cpml_layers`` = 10 / 16 / 20 / 40 on the same
straight guide, with the source and BOTH monitor planes held at cv01's
physical coordinates (``src_x`` 1.1 um, input 4.0 um, output 14.5 um) and ONLY
``Simulation(cpml_layers=...)`` varied.

cv01 writes ``pml = cpml_n * dx`` and then places ``src_x = pml + dx`` and the
output plane at ``sx - pml - 5 * dx``. Letting ``pml`` track the swept layer
count would move the source and both monitors along with the absorber and
confound absorber thickness with plane position -- that is the naive sweep and
it is NOT this measurement. ``pml`` therefore stays at cv01's ``10 * dx`` here
while ``cpml_layers`` varies; rfx pads the grid OUTSIDE the declared 16 um
domain, so the interior stays 161 NODES per axis (160 cells) and only the
absorber grows.
That single deviation from cv01 is recorded in the artifact as
``deviation_from_cv01`` rather than left implicit.

Each layer count runs two arms, ``full`` and ``aperture``. The binding gate
reads the ``full`` arm alone; the ``aperture`` companion exists because the
reported observable -- the outside-aperture band-summed backward power fraction
at the output plane -- is the difference between the two windows, formed
exactly as the committed six-arm artifact's -26.04 % was. It also supplies the
realized-coordinate witness: preflight records a flux region only where a
``size=`` was requested, so the ``full`` arms carry ``realized_flux_regions:
null`` by construction and cannot witness their own plane.

Single binding gate: ``mean_self`` >= 0.90906 on the 40-layer ``full`` arm,
i.e. >= 2/3 of the gap from 0.748852 to the ``upml_full`` control 0.989162 is
closed. The fraction and its monotonicity are REPORTED observables with no gate
anywhere in the repo -- the note's "Correction 2026-09-14" measured that the
two criteria are not equivalent (a 0.0148-wide window passes one and fails the
other) and demoted the fraction to a diagnostic.

Residual split (``--mode residual-split``)
-----------------------------------------
One arm, pre-declared in the same note under "Next measurement, pre-declared
2026-09-14": the ``interior`` window (``size=(sy, dx)``) at 40 layers, which
the layer sweep does not run. With the sweep's committed ``full`` and
``aperture`` arms at the same depth it splits the 40-layer residual the way
the six-arm campaign split the 10-layer case -- ``full`` - ``interior`` is the
absorber-cell slots of the padded plane, ``interior`` - ``aperture`` is the
off-guide part of the physical interior.

It reads the other two arms out of the committed sweep artifact rather than
re-running them, so it asserts first that the new arm's grid, band mask and
frequency axis match what that artifact recorded; a mismatch fails the run
instead of producing a subtraction across two different rigs.

No gate. It is a diagnostic split of a residual that the sweep left
unattributed, and its outcome names the next suspect rather than closing
anything.

Rig fidelity
------------
The rig below is a hand-copy of cv01's Run 1. ``_assert_rig_matches_cv01``
greps the committed script for every copied line -- parameters, geometry,
material, source, monitors and the two lines that make and gate ``mean_self``
-- and fails loudly if any has changed, so an edit to cv01 reds this driver
instead of silently leaving it measuring a different waveguide.

Provenance
----------
``_assert_rfx_is_this_repo`` fails the run unless ``import rfx`` resolved under
this script's own repo root. Running the file by path puts the *script's*
directory on ``sys.path``, not the repo root, so without the check ``import
rfx`` can silently come from another tree while the artifact records this
tree's commit sha. The ``Usage`` line below pins it explicitly; `pip install
-e` is not the fix and is deliberately absent from this environment.

Runtime: CPU, a few minutes for all six arms; cv01 is a ``cpu-runner`` case in
``validation/crossval/manifest.json`` and needs no GPU.

Usage::

    PYTHONPATH=$(git rev-parse --show-toplevel) \
        python3 scripts/diagnostics/cv01_cpml_flux_selfcheck.py \
        --output scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck.json

    PYTHONPATH=$(git rev-parse --show-toplevel) \
        python3 scripts/diagnostics/cv01_cpml_flux_selfcheck.py \
        --mode layer-sweep \
        --output scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
import warnings
from contextlib import redirect_stdout

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy.ndimage import uniform_filter1d

REPO = pathlib.Path(__file__).resolve().parents[2]
CV01 = REPO / "validation" / "crossval" / "01_waveguide_bend.py"

C0 = 2.998e8

# --- cv01 Run 1 rig, copied verbatim from the committed script ---------------
a = 1.0e-6
eps_wg = 12.0
w_wg = 1.0 * a
dx = a / 10
cpml_n = 10
pml = cpml_n * dx

fcen = 0.15 * C0 / a
fwidth = 0.1 * C0 / a
n_freqs = 200
freqs = np.linspace(0.10 * C0 / a, 0.20 * C0 / a, n_freqs)

sx = 16.0 * a
sy = 16.0 * a
wg_y = sy / 2
src_x = pml + dx

f_cutoff = 1.0 / (2.0 * np.sqrt(eps_wg - 1.0))
n_steps = 25000

# Every one of these must still be present, verbatim, in the committed cv01.
# Parameters, geometry, material, source, monitors, and the two lines that turn
# the flux spectra into ``mean_self`` and gate it. The geometry/material/source
# block is here because a driver that copied cv01's numbers but not its ``Box``
# would pass a parameters-only check while solving a different waveguide.
RIG_LINES = (
    "a = 1.0e-6",
    "eps_wg = 12.0",
    "w_wg = 1.0 * a",
    "dx = a / 10",
    "cpml_n = 10",
    "pml = cpml_n * dx",
    "fcen = 0.15 * C0 / a",
    "fwidth = 0.1 * C0 / a",
    "n_freqs = 200",
    "freqs = np.linspace(0.10 * C0 / a, 0.20 * C0 / a, n_freqs)",
    "sx = 16.0 * a",
    "sy = 16.0 * a",
    "wg_y = sy / 2",
    "src_x = pml + dx",
    "f_cutoff = 1.0 / (2.0 * np.sqrt(eps_wg - 1.0))",
    "n_steps = 25000",
    # cv01's absorber depth, split from its plane-placement constant in the
    # same PR as the sweep below. `cpml_n = 10` above still has to be present
    # verbatim -- it is what places `pml`, `src_x` and both monitors -- and
    # this line is what sets the absorber.
    "cpml_layers = 20 if boundary == \"cpml\" else cpml_n",
    # Geometry, material and source -- the structure itself, not just its numbers.
    'sim_s = Simulation(freq_max=0.25 * C0 / a, domain=(sx, sy, dx), dx=dx,',
    "                   boundary=BoundarySpec.uniform(boundary),",
    '                   cpml_layers=cpml_layers, mode="2d_tmz")',
    'sim_s.add_material("wg", eps_r=eps_wg)',
    "sim_s.add(Box((0, wg_y - w_wg / 2, 0), (sx, wg_y + w_wg / 2, dx)),",
    '          material="wg")',
    "add_line_source(sim_s, src_x, wg_y, w_wg)",
    "        y = y_center - width / 2 + (i + 0.5) * width / 10",
    '        sim.add_source(position=(x, y, 0), component="ez",',
    "                       waveform=GaussianPulse(f0=fcen, bandwidth=fwidth / fcen,",
    "                                              amplitude=1.0 / 10))",
    'above = (f_meep > f_cutoff + 0.005) & (f_meep < 0.20)',
    'sim_s.add_flux_monitor(axis="x", coordinate=4 * a, freqs=freqs, name="input")',
    'sim_s.add_flux_monitor(axis="x", coordinate=sx - pml - 5 * dx,',
    "res_s = sim_s.run(n_steps=n_steps, subpixel_smoothing=True)",
    "T_self_smooth = uniform_filter1d(T_self, size=20)",
    "mean_self = float(np.mean(T_self_smooth[above]))",
    # G2's band and the comparison that applies it. Until 2026-09-15 this was
    # the single literal ``if 0.95 <= mean_self <= 1.05:``; PR #1040 (main
    # 9866bafd) lifted cv01's three gates into a ``_gates()`` helper so a
    # record's exit code could be the process's own, and that line no longer
    # exists. The band and the quantity are unchanged -- the two lines below
    # pin BOTH halves (the numbers, and that ``mean_self`` is what they are
    # applied to), where the old single line pinned them together. Anyone
    # re-splitting the helper has to keep both.
    "G2_SELF_T_BAND = (0.95, 1.05)",
    "G2_SELF_T_BAND[0] <= mean_self <= G2_SELF_T_BAND[1]",
)


def _assert_rfx_is_this_repo(rfx_module) -> dict:
    """Fail loudly unless ``import rfx`` resolved inside THIS checkout.

    ``python scripts/diagnostics/<name>.py`` puts the *script's* directory on
    ``sys.path``, not the repo root, so ``import rfx`` falls through to
    whatever install the interpreter can see -- an editable install pointing at
    a different tree, a site-packages wheel, another worktree. The run then
    records this branch's commit sha in ``provenance.commit`` while the solver
    that produced the numbers came from somewhere else, and nothing in the
    artifact says so. That is what happened on this driver's first run, whose
    ``provenance.rfx_file`` names the primary checkout rather than the branch
    tree, which is why the check exists.
    """
    rfx_file = pathlib.Path(rfx_module.__file__).resolve()
    try:
        rfx_file.relative_to(REPO)
    except ValueError:
        raise SystemExit(
            "rfx provenance check FAILED -- `import rfx` resolved to\n"
            f"    {rfx_file}\n"
            f"which is NOT under this script's repo root\n    {REPO}\n"
            "so the numbers this run would record are not this tree's. Re-run "
            "with the repo root pinned, e.g.\n"
            f"    PYTHONPATH={REPO} python3 scripts/diagnostics/"
            f"{pathlib.Path(__file__).name} --output ...\n"
            "(never `pip install -e`; see the module docstring.)"
        ) from None
    return {
        "rfx_file": str(rfx_file),
        "under_repo_root": True,
        "repo_root": str(REPO),
    }


def _assert_rig_matches_cv01() -> dict:
    text = CV01.read_text(encoding="utf-8")
    missing = [line for line in RIG_LINES if line not in text]
    if missing:
        raise SystemExit(
            "rig fidelity check FAILED -- these lines are no longer in "
            f"{CV01.relative_to(REPO)} verbatim, so this driver is measuring a "
            "different rig than cv01:\n  " + "\n  ".join(missing)
        )
    return {"checked_lines": len(RIG_LINES), "all_present_verbatim": True}


def add_line_source(sim, x, y_center, width):
    """cv01's own ``add_line_source``, verbatim -- ten ez point sources across
    the guide width. Grep the committed script for
    ``y = y_center - width / 2 + (i + 0.5) * width / 10``; the whole body is in
    RIG_LINES, so a change to it reds this driver."""
    from rfx import GaussianPulse
    for i in range(10):
        y = y_center - width / 2 + (i + 0.5) * width / 10
        sim.add_source(position=(x, y, 0), component="ez",
                       waveform=GaussianPulse(f0=fcen, bandwidth=fwidth / fcen,
                                              amplitude=1.0 / 10))


def run_arm(boundary: str, size, steps: int = n_steps,
            cpml_layers: int = cpml_n) -> dict:
    """cv01 Run 1 under one boundary, one monitor window, one absorber depth.

    ``cpml_layers`` defaults to cv01's ``cpml_n`` = 10, so every six-arm call
    is unchanged. Only ``--mode layer-sweep`` passes anything else, and it
    passes it HERE only -- ``pml``, ``src_x`` and both monitor coordinates stay
    at cv01's values, so the absorber grows and the planes do not move.
    """
    from rfx import Simulation, Box, flux_spectrum
    from rfx.boundaries.spec import BoundarySpec

    kw = {} if size is None else {"size": size}
    t0 = time.time()

    # Preflight output is part of the result (repo rule): capture the banner
    # AND every warnings.warn raised while the simulation is BUILT. The build
    # has to be INSIDE `catch_warnings`, not before it -- the first version of
    # this driver opened the recorder after the last `add_flux_monitor`, so it
    # recorded `preflight_warnings: []` on all six arms while the build's
    # DeprecationWarning went to stderr and survived only in `run.log`.
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = Simulation(freq_max=0.25 * C0 / a, domain=(sx, sy, dx), dx=dx,
                         boundary=BoundarySpec.uniform(boundary),
                         cpml_layers=cpml_layers, mode="2d_tmz")
        sim.add_material("wg", eps_r=eps_wg)
        sim.add(Box((0, wg_y - w_wg / 2, 0), (sx, wg_y + w_wg / 2, dx)),
                material="wg")
        add_line_source(sim, src_x, wg_y, w_wg)
        sim.add_flux_monitor(axis="x", coordinate=4 * a, freqs=freqs,
                             name="input", **kw)
        sim.add_flux_monitor(axis="x", coordinate=sx - pml - 5 * dx, freqs=freqs,
                             name="output", **kw)
        with redirect_stdout(buf):
            report = sim.preflight(strict=False)
        preflight_warnings = [str(w.message) for w in caught]
    preflight_stdout = buf.getvalue()
    preflight_report = report.to_dict() if hasattr(report, "to_dict") else None
    preflight_formatted = report.format() if hasattr(report, "format") else None

    with warnings.catch_warnings(record=True) as caught_run:
        warnings.simplefilter("always")
        res = sim.run(n_steps=steps, subpixel_smoothing=True)
        run_warnings = [str(w.message) for w in caught_run]

    flux_in = np.asarray(flux_spectrum(res.flux_monitors["input"]), dtype=float)
    flux_out = np.asarray(flux_spectrum(res.flux_monitors["output"]), dtype=float)

    # cv01's transmittance block, verbatim arithmetic -- grep the committed
    # script for `above = (f_meep > f_cutoff + 0.005) & (f_meep < 0.20)`
    # through `mean_self = float(np.mean(T_self_smooth[above]))`.
    f_meep = freqs * a / C0
    above = (f_meep > f_cutoff + 0.005) & (f_meep < 0.20)
    safe_in = np.maximum(np.abs(flux_in), np.max(np.abs(flux_in)) * 1e-6)
    T_self = flux_out / safe_in
    T_self_smooth = uniform_filter1d(T_self, size=20)
    mean_self = float(np.mean(T_self_smooth[above]))

    # Realized monitor geometry: preflight's own record (metres + cell
    # indices), which is the runner's resolver output, not a second copy.
    regions = (preflight_report or {}).get("flux_regions")

    grid = getattr(res, "grid", None)
    if grid is not None:
        grid_info = {
            "shape": [int(n) for n in grid.shape],
            "pad": {f"{L}_{e}": int(getattr(grid, f"pad_{L}_{e}", -1))
                    for L in "xyz" for e in ("lo", "hi")},
            "dx_m": float(getattr(grid, "dx", dx)),
        }
    else:
        grid_info = None

    return {
        "boundary": boundary,
        "n_steps": int(steps),
        "monitor_size_m": None if size is None else [float(s) for s in size],
        "mean_self": mean_self,
        "gate_G2_0p95_1p05": bool(0.95 <= mean_self <= 1.05),
        "n_bins_in_band": int(above.sum()),
        "band_f_over_c_per_a": [float(f_meep[above].min()), float(f_meep[above].max())],
        "T_self_band_min": float(np.min(T_self_smooth[above])),
        "T_self_band_max": float(np.max(T_self_smooth[above])),
        # R5: the per-bin trace, not only the headline.
        "f_over_c_per_a": [float(v) for v in f_meep],
        "above_mask": [bool(v) for v in above],
        "flux_in": [float(v) for v in flux_in],
        "flux_out": [float(v) for v in flux_out],
        "T_self": [float(v) for v in T_self],
        "T_self_smooth": [float(v) for v in T_self_smooth],
        "preflight_stdout": preflight_stdout,
        "preflight_formatted": preflight_formatted,
        "preflight_report": preflight_report,
        "preflight_warnings": preflight_warnings,
        "run_warnings": run_warnings,
        "realized_flux_regions": regions,
        "grid": grid_info,
        "wall_s": round(time.time() - t0, 1),
    }


ARMS = (
    ("upml_full", "upml", None),
    ("cpml_full", "cpml", None),
    ("upml_interior", "upml", (sy, dx)),
    ("cpml_interior", "cpml", (sy, dx)),
    ("cpml_aperture", "cpml", (2 * w_wg, dx)),
    # Added after the first five-arm run, and labelled as such in the
    # artifact: without it the aperture comparison has no UPML control, so
    # `cpml_aperture` alone could not say whether an aperture-sized plane
    # conserves under BOTH boundaries or only under CPML. It changes no
    # pre-declared gate and no pre-declared falsifier.
    ("upml_aperture", "upml", (2 * w_wg, dx)),
)

ARMS_ADDED_POST_HOC = ("upml_aperture",)

# The committed UPML run this driver's control must reproduce. This one is NOT
# re-pointed: it names cv01's committed crossval record
# (validation/crossval/_01_waveguide_bend_results/crossval.json), which this
# change does not re-run and does not touch. #1043 stage B moves what the
# driver MEASURES for that arm -- 0.9891610388008335 -> 0.9910752993577158,
# +0.194 % -- so the reproduce check now records a difference instead of a
# match, and the difference is the point rather than a failure. Adopting the
# new value here would be re-pointing a record this change never re-ran.
COMMITTED_UPML_MEAN_SELF = 0.9891610388008335
#: What this driver measures for that arm after the pad continuation. Reported
#: beside the committed value, never substituted for it.
MEASURED_UPML_MEAN_SELF_AFTER_PAD_CONTINUATION = 0.9910752993577158

# --- layer sweep (pre-declaration addendum, Arm 1) ---------------------------
LAYER_SWEEP = (10, 16, 20, 40)

# The single binding gate, per "Correction 2026-09-14" in the pre-declaration:
# mean_self on the 40-layer full arm has closed >= 2/3 of the gap from the
# committed cpml_full 0.748852 to the upml_full control 0.989162.
SWEEP_GATE_MEAN_SELF_AT_40 = 0.90906

# The 10-layer arm of the sweep is cv01's own rig, so it doubles as the sweep's
# control: it must reproduce both of these from the committed six-arm artifact,
# or no other layer count is readable.
#
# RE-POINTED 2026-09-15 (#1043 stage B), explicitly and not by editing a value
# in place. The two constants below describe the PRE-#1043 solver, which solved
# this guide with vacuum in its own absorber pad; they still describe
# selfcheck.json exactly, and that file is unchanged. What they no longer
# describe is what this driver measures, so the control was re-pointed at the
# post-continuation revision rather than left silently failing or silently
# re-keyed. Both revisions are recorded in every artifact this driver writes,
# so a reader can see which one a run reproduces.
SWEEP_BASELINE_REVISION = "r2"
SWEEP_BASELINE_ARTIFACT = {
    "r1": "scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck.json",
    "r2": "scripts/diagnostics/_artifacts/cv01_cpml_813/selfcheck_r2.json",
}
# r1 -- pre-#1043, kept as named history, never as the live control again.
SWEEP_BASELINE_CPML_FULL_MEAN_SELF_R1 = 0.7488520140093946
SWEEP_BASELINE_OUTSIDE_FRACTION_PCT_R1 = -26.041819705557895
# r2 -- measured on the tree that continues a boundary-touching dielectric
# through its pad (PR for #1043 stage B), same rig, same 25000 steps.
SWEEP_BASELINE_CPML_FULL_MEAN_SELF_R2 = 0.9876177418925891
SWEEP_BASELINE_OUTSIDE_FRACTION_PCT_R2 = 0.9442985029538306

SWEEP_BASELINE_CPML_FULL_MEAN_SELF = SWEEP_BASELINE_CPML_FULL_MEAN_SELF_R2
SWEEP_BASELINE_OUTSIDE_FRACTION_PCT = SWEEP_BASELINE_OUTSIDE_FRACTION_PCT_R2

#: Below this spread across the four layer counts the ladder carries no
#: depth dependence a verdict could turn on, so the sweep must NOT attribute a
#: deficit to absorber depth. 0.05 is G2's own half-width (the gate is
#: 0.95-1.05): a ladder whose entire spread is smaller than that cannot move an
#: arm across the band it is judged by. Measured: the pre-#1043 ladder spread
#: 0.198 (0.748852 to 0.947345), the post-continuation ladder 0.0018.
LADDER_SPREAD_FOR_DEPTH_ATTRIBUTION = 0.05

# rfx pads the grid OUTSIDE the declared 16 um x 16 um domain, so the interior
# must not move as cpml_layers does. 16 um / (a/10) + 1 = 161 -- the `+ 1` is
# the tell: this is a NODE count, 161 nodes spanning 160 cells. The name says
# cells and is wrong; it is kept because the committed artifacts and the design
# note's citations resolve through `interior_cells` / `interior_cells_expected`,
# and renaming a key to fix a word breaks a record that is otherwise correct.
# Comparisons against `grid.shape` are node-vs-node, so the check is right.
INTERIOR_CELLS_PER_AXIS = 161

# Physical positions held at cv01's values on every swept arm. The absorber
# grows; these do not move. (src_x and the output plane are both written off
# `pml = cpml_n * dx`, which stays at 10 * dx here.)
HELD_PLANES_M = {
    "src_x": src_x,
    "input_plane": 4 * a,
    "output_plane": sx - pml - 5 * dx,
    "pml_placing_the_planes": pml,
    "cpml_n_placing_the_planes": cpml_n,
}


# --- residual split (the sweep's 40-layer residual, one interior arm) -------
RESIDUAL_SPLIT_LAYERS = 40
LAYER_SWEEP_ARTIFACT = ("scripts/diagnostics/_artifacts/cv01_cpml_813/"
                        "layer_sweep.json")
SIX_ARM_ARTIFACT = ("scripts/diagnostics/_artifacts/cv01_cpml_813/"
                    "selfcheck.json")


def _band_split(full_arm: dict, mid_arm: dict, ap_arm: dict) -> dict:
    """Split the outside-aperture term into its two parts, at the output plane.

    ``full`` - ``interior``  = the padded plane's absorber-cell slots.
    ``interior`` - ``aperture`` = the off-guide part of the physical interior.
    Both band-summed over cv01's own ``above`` mask and normalized by the
    full-plane band sum, which is how the six-arm campaign reported them.
    """
    above = np.asarray(full_arm["above_mask"], dtype=bool)
    fo_full = np.asarray(full_arm["flux_out"], dtype=float)
    fo_mid = np.asarray(mid_arm["flux_out"], dtype=float)
    fo_ap = np.asarray(ap_arm["flux_out"], dtype=float)
    band_full = float(fo_full[above].sum())
    slots = fo_full - fo_mid
    inner = fo_mid - fo_ap
    return {
        "band_summed_full_plane": band_full,
        "absorber_cell_slots_band_summed": float(slots[above].sum()),
        "absorber_cell_slots_points": 100.0 * float(slots[above].sum()) / band_full,
        "interior_off_guide_band_summed": float(inner[above].sum()),
        "interior_off_guide_points": 100.0 * float(inner[above].sum()) / band_full,
        "total_outside_aperture_points": (
            100.0 * float((fo_full - fo_ap)[above].sum()) / band_full),
        "n_bins_in_band": int(above.sum()),
        "n_bins_slots_negative": int((slots[above] < 0).sum()),
        "n_bins_interior_off_guide_negative": int((inner[above] < 0).sum()),
        "per_bin_absorber_cell_slots": [float(v) for v in slots],
        "per_bin_interior_off_guide": [float(v) for v in inner],
    }


def _assert_comparable(new_arm: dict, stored_arm: dict, what: str) -> dict:
    """The stored arms come from another run; prove they are the same rig.

    Subtracting a fresh arm from arms read out of a committed artifact is only
    meaningful if the grid, the band mask and the frequency axis agree. They
    are checked, not assumed, and a mismatch raises.
    """
    checks = {
        "grid_matches": new_arm.get("grid") == stored_arm.get("grid"),
        "above_mask_matches": new_arm["above_mask"] == stored_arm["above_mask"],
        "freq_axis_matches": new_arm["f_over_c_per_a"] == stored_arm["f_over_c_per_a"],
        "n_steps_matches": new_arm["n_steps"] == stored_arm["n_steps"],
        "boundary_matches": new_arm["boundary"] == stored_arm["boundary"],
    }
    if not all(checks.values()):
        raise SystemExit(
            f"comparability check FAILED against the stored {what} arm: "
            f"{ {k: v for k, v in checks.items() if not v} }. The subtraction "
            "would cross two different rigs; not written."
        )
    return checks


def run_residual_split(args, rig: dict, provenance_check: dict,
                       provenance: dict) -> dict:
    """The pre-declared interior arm, and the split it makes possible."""
    layers = int(args.split_layers)
    sweep = json.loads((REPO / LAYER_SWEEP_ARTIFACT).read_text(encoding="utf-8"))
    six = json.loads((REPO / SIX_ARM_ARTIFACT).read_text(encoding="utf-8"))
    entry = sweep["layers"][str(layers)]
    stored_full, stored_ap = entry["full"], entry["aperture"]

    print(f"--- cpml_layers={layers}: interior (the new arm) ---", flush=True)
    interior = run_arm("cpml", (sy, dx), steps=args.n_steps, cpml_layers=layers)
    print(f"    mean_self = {interior['mean_self']:.6f} "
          f"({interior['wall_s']}s)", flush=True)

    comparable = {
        "vs_stored_full": _assert_comparable(interior, stored_full, "full"),
        "vs_stored_aperture": _assert_comparable(interior, stored_ap, "aperture"),
    }
    split = _band_split(stored_full, interior, stored_ap)

    # The same split at 10 layers, recomputed from the six-arm artifact rather
    # than retyped, so the two depths are compared through one code path.
    ref = _band_split(six["arms"]["cpml_full"], six["arms"]["cpml_interior"],
                      six["arms"]["cpml_aperture"])

    grid = interior.get("grid") or {}
    shape = grid.get("shape") or []
    pad = grid.get("pad", {})
    return {
        "issue": 813,
        "mode": "residual-split",
        "predeclaration": (
            "docs/design_notes/cv01_cpml_flux_selfcheck_predeclaration.md"
            " -- 'Next measurement, pre-declared 2026-09-14', written and"
            " committed before this ran"),
        "measures": ("the 40-layer residual split into absorber-cell slots"
                     " (full - interior) and interior off-guide"
                     " (interior - aperture), band-summed at the output plane"
                     " over cv01's own `above` mask"),
        "gate": "NONE. This is a diagnostic split, not a gated measurement.",
        "prediction_on_record": (
            "interior - aperture still dominates full - interior, as it does at"
            " 10 layers (-24.79 vs -1.25 points); a full - interior that has"
            " GROWN instead points at the padded-plane window, whose cell"
            " count grows with the absorber while the planes do not move."),
        "window_caveat": (
            "the `full` window is the whole padded plane, so its tangential"
            " extent grows with cpml_layers (181 -> 241 NODES, i.e. 180 -> 240"
            " cells) even though every plane POSITION is held fixed. The"
            " physical interior (161 nodes = 160 cells) and the aperture (21"
            " nodes = 20 cells) do not grow. That is what this arm is here to"
            " size."),
        "n_steps": int(args.n_steps),
        "n_steps_is_cv01_value": bool(args.n_steps == n_steps),
        "cpml_layers": layers,
        "rig_fidelity_check": rig,
        "rfx_provenance_check": provenance_check,
        "held_fixed_m": HELD_PLANES_M,
        "stored_arms_source": {
            "path": LAYER_SWEEP_ARTIFACT,
            "layer_key": str(layers),
            "full_mean_self": stored_full["mean_self"],
            "aperture_mean_self": stored_ap["mean_self"],
            "commit": sweep["provenance"]["commit"],
            "utc": sweep["provenance"]["utc"],
        },
        "reference_split_source": {
            "path": SIX_ARM_ARTIFACT,
            "cpml_layers": cpml_n,
            "commit": six["provenance"]["commit"],
        },
        "comparability_checks": comparable,
        "provenance": provenance,
        "interior_arm": interior,
        "interior_grid_shape": shape,
        # NODE counts, not cell counts -- `grid.shape` counts Yee NODES. The
        # 16 um interior at dx = 0.1 um is 160 CELLS and 161 nodes; the full
        # padded plane at 40 layers is 240 cells and 241 nodes. Read these two
        # as "cells" and you are off by one on each, the #868 class of error
        # (a node slice read as cell widths). The key names say `cells` and are
        # wrong; they are kept because the committed artifacts and the design
        # note's citations resolve through them, and renaming a key to fix a
        # word would break a record that is otherwise correct. The realized
        # CELL extents are in preflight's own `cell_slices`, which this arm
        # records verbatim: [40, 200] = 160 cells for the interior window,
        # [110, 130] = 20 cells for the aperture.
        #
        # Nothing measured depends on the convention: the absorber-slot count
        # is the DIFFERENCE of two node counts over the same interior (181 -
        # 161 = 20 at 10 layers, 241 - 161 = 80 at 40), and a difference of
        # node counts is a count of cells. The 4x ratio is exact either way.
        "full_plane_tangential_cells": (int(shape[1]) if len(shape) > 1 else None),
        "interior_window_cells": (
            int(shape[1]) - int(pad.get("y_lo", 0)) - int(pad.get("y_hi", 0))
            if len(shape) > 1 else None),
        "split_at_swept_layers": split,
        "split_at_cv01_layers_reference": ref,
        "comparison": {
            "absorber_cell_slots_points": {
                str(cpml_n): ref["absorber_cell_slots_points"],
                str(layers): split["absorber_cell_slots_points"],
            },
            "interior_off_guide_points": {
                str(cpml_n): ref["interior_off_guide_points"],
                str(layers): split["interior_off_guide_points"],
            },
            "interior_off_guide_still_dominates": bool(
                abs(split["interior_off_guide_points"])
                > abs(split["absorber_cell_slots_points"])),
            "absorber_cell_slots_grew": bool(
                abs(split["absorber_cell_slots_points"])
                > abs(ref["absorber_cell_slots_points"])),
        },
    }


def _outside_aperture(full_arm: dict, ap_arm: dict) -> dict:
    """Band-summed backward power outside the aperture, at the output plane.

    ``full`` minus ``aperture`` on the SAME output plane, band-summed over
    cv01's own ``above`` mask and normalized by the full-plane band sum -- the
    construction that gave -26.0418 % on the committed six-arm artifact. It is
    a REPORTED observable: it gates nothing (pre-declaration, "Correction
    2026-09-14").
    """
    above = np.asarray(full_arm["above_mask"], dtype=bool)
    fo_full = np.asarray(full_arm["flux_out"], dtype=float)
    fo_ap = np.asarray(ap_arm["flux_out"], dtype=float)
    outside = fo_full - fo_ap
    band_full = float(fo_full[above].sum())
    band_outside = float(outside[above].sum())
    return {
        "band_summed_full_plane": band_full,
        "band_summed_outside_aperture": band_outside,
        "fraction_percent": 100.0 * band_outside / band_full,
        "n_bins_in_band": int(above.sum()),
        "n_bins_outside_term_negative": int((outside[above] < 0).sum()),
        # R5: the per-bin sign trace, not only the band sum.
        "per_bin_outside_aperture": [float(v) for v in outside],
    }


def _sweep_witness(layers: int, full_arm: dict, ap_arm: dict) -> dict:
    """The pre-declared anti-confound witness, per layer count.

    Pre-declaration: "the interior must stay at 161 cells per axis while
    grid.shape goes 181 -> 193 -> 201 -> 241 and grid.pad_* follows the swept
    value; and preflight's realized monitor coordinate must read 1.45e-05 m on
    every arm". An arm failing either is reported unreadable, not averaged in.

    Quoted as written; "cells" there is wrong. Every count in that sentence is
    a NODE count (161 nodes = 160 cells, 241 nodes = 240 cells), which is what
    `grid.shape` holds, so the check compares node counts with node counts and
    is correct. See the comment at INTERIOR_CELLS_PER_AXIS.

    The realized coordinate comes from the APERTURE companion: preflight writes
    a flux-region record only where a ``size=`` was requested, so the ``full``
    arm records ``realized_flux_regions: null`` and cannot witness itself.
    """
    g = full_arm.get("grid") or {}
    ga = ap_arm.get("grid") or {}
    shape = g.get("shape")
    pad = g.get("pad", {})
    expected_shape = [INTERIOR_CELLS_PER_AXIS + 2 * layers,
                      INTERIOR_CELLS_PER_AXIS + 2 * layers, 1]
    pads_follow_layers = all(int(pad.get(f"{ax}_{e}", -1)) == layers
                             for ax in ("x", "y") for e in ("lo", "hi"))
    interior = {}
    if shape:
        for i, ax in enumerate(("x", "y")):
            interior[ax] = (int(shape[i]) - int(pad.get(f"{ax}_lo", 0))
                            - int(pad.get(f"{ax}_hi", 0)))
    interior_fixed = bool(interior) and all(
        v == INTERIOR_CELLS_PER_AXIS for v in interior.values())

    regions = ap_arm.get("realized_flux_regions") or []
    realized = {r["name"]: r for r in regions}
    out_rec = realized.get("output", {})
    in_rec = realized.get("input", {})
    out_m = out_rec.get("realized_coordinate_m")
    in_m = in_rec.get("realized_coordinate_m")
    planes_held = bool(
        out_m is not None and abs(out_m - HELD_PLANES_M["output_plane"]) <= 1e-12
        and in_m is not None and abs(in_m - HELD_PLANES_M["input_plane"]) <= 1e-12)

    return {
        "grid_shape": shape,
        "grid_shape_expected": expected_shape,
        "grid_shape_matches": shape == expected_shape,
        "pad": pad,
        "pads_follow_layers": pads_follow_layers,
        "interior_cells": interior,
        "interior_cells_expected": INTERIOR_CELLS_PER_AXIS,
        "interior_fixed": interior_fixed,
        "aperture_companion_grid_matches": g == ga,
        "realized_output_coordinate_m": out_m,
        "realized_input_coordinate_m": in_m,
        "realized_output_normal_index": out_rec.get("normal_index"),
        "requested_output_coordinate_m": out_rec.get("requested_coordinate_m"),
        "planes_held_at_cv01_values": planes_held,
        "full_arm_records_no_region": full_arm.get("realized_flux_regions") is None,
        "readable": bool(shape == expected_shape and pads_follow_layers
                         and interior_fixed and planes_held and g == ga),
    }


def run_layer_sweep(args, rig: dict, provenance_check: dict,
                    provenance: dict) -> dict:
    """The pre-declaration addendum's Arm 1, with its corrected gate."""
    wanted = {int(s) for s in args.layers.split(",") if s.strip()}
    out = {
        "issue": 813,
        "mode": "layer-sweep",
        "predeclaration": (
            "docs/design_notes/cv01_cpml_flux_selfcheck_predeclaration.md"
            " -- 'Addendum 2026-09-14', Arm 1, with the gate as corrected in"
            " 'Correction 2026-09-14'"),
        "measures": ("cv01 straight-guide mean_self, Run 1 only -- the line "
                     "`mean_self = float(np.mean(T_self_smooth[above]))` in "
                     "validation/crossval/01_waveguide_bend.py -- at four "
                     "CPML absorber depths"),
        "deviation_from_cv01": (
            "Simulation(cpml_layers=...) is swept over 10/16/20/40. cv01's own "
            "absorber depth is whatever its `cpml_layers` line selects for the "
            "boundary in force (10 for upml, 20 for cpml since #813); this "
            "mode ignores that line and sets the depth directly. Nothing else "
            "differs: `pml = cpml_n * dx` stays at 10 * dx, so `src_x = pml + "
            "dx` and the output plane at `sx - pml - 5 * dx` do NOT move with "
            "the absorber. cv01 itself is not modified or re-run by this "
            "mode."),
        "n_steps": int(args.n_steps),
        "n_steps_is_cv01_value": bool(args.n_steps == n_steps),
        "gate": ("SINGLE BINDING GATE: mean_self >= "
                 f"{SWEEP_GATE_MEAN_SELF_AT_40} on the 40-layer `full` arm, "
                 "i.e. >= 2/3 of the gap from the committed cpml_full "
                 "0.748852 to the upml_full control 0.989162 is closed"),
        "falsifier": ("conjunction: mean_self at 40 layers within 0.03 of "
                      "0.748852 AND the outside-aperture fraction flat within "
                      "3 points across all four layer counts. Both must hold; "
                      "mean_self is the binding half."),
        "reported_not_gating": (
            "the outside-aperture band-summed backward power fraction at the "
            "output plane and its monotonicity. It has no gate anywhere in the "
            "repo; if it and the mean_self verdict disagree, the disagreement "
            "is reported, not resolved."),
        "rig_fidelity_check": rig,
        "rfx_provenance_check": provenance_check,
        "held_fixed_m": HELD_PLANES_M,
        "control": {
            "revision": SWEEP_BASELINE_REVISION,
            "source": SWEEP_BASELINE_ARTIFACT[SWEEP_BASELINE_REVISION],
            "cpml_full_mean_self": SWEEP_BASELINE_CPML_FULL_MEAN_SELF,
            "outside_aperture_fraction_percent": SWEEP_BASELINE_OUTSIDE_FRACTION_PCT,
            "upml_full_mean_self": COMMITTED_UPML_MEAN_SELF,
            "note": ("the 10-layer arm below IS cv01's rig, so it must "
                     "reproduce the two cpml numbers or nothing else here is "
                     "readable"),
            # Both revisions ride in every artifact, so which one a run
            # reproduces is readable from the record rather than inferred from
            # the driver's source at that commit (#1043 stage B).
            "revisions": {
                "r1": {
                    "source": SWEEP_BASELINE_ARTIFACT["r1"],
                    "cpml_full_mean_self": SWEEP_BASELINE_CPML_FULL_MEAN_SELF_R1,
                    "outside_aperture_fraction_percent":
                        SWEEP_BASELINE_OUTSIDE_FRACTION_PCT_R1,
                    "describes": ("the pre-#1043 solver: this guide touches "
                                  "both x faces and was solved with vacuum in "
                                  "its own absorber pad"),
                },
                "r2": {
                    "source": SWEEP_BASELINE_ARTIFACT["r2"],
                    "cpml_full_mean_self": SWEEP_BASELINE_CPML_FULL_MEAN_SELF_R2,
                    "outside_aperture_fraction_percent":
                        SWEEP_BASELINE_OUTSIDE_FRACTION_PCT_R2,
                    "describes": ("#1043 stage B: the guide is continued "
                                  "through its pad, so the seam facet is gone"),
                },
            },
            "committed_upml_record": {
                "path": "validation/crossval/_01_waveguide_bend_results/crossval.json",
                "mean_self_smoothed_over_band": COMMITTED_UPML_MEAN_SELF,
                "measured_after_pad_continuation":
                    MEASURED_UPML_MEAN_SELF_AFTER_PAD_CONTINUATION,
                "note": ("the record is NOT re-pointed -- this change does not "
                         "re-run cv01's crossval case. The measured value is "
                         "reported beside it, never substituted for it."),
            },
        },
        "provenance": provenance,
        "layers": {},
    }

    for layers in LAYER_SWEEP:
        if wanted and layers not in wanted:
            continue
        print(f"--- cpml_layers={layers}: full ---", flush=True)
        full = run_arm("cpml", None, steps=args.n_steps, cpml_layers=layers)
        print(f"    mean_self = {full['mean_self']:.6f} ({full['wall_s']}s)",
              flush=True)
        print(f"--- cpml_layers={layers}: aperture ---", flush=True)
        ap_arm = run_arm("cpml", (2 * w_wg, dx), steps=args.n_steps,
                         cpml_layers=layers)
        print(f"    mean_self = {ap_arm['mean_self']:.6f} "
              f"({ap_arm['wall_s']}s)", flush=True)
        outside = _outside_aperture(full, ap_arm)
        witness = _sweep_witness(layers, full, ap_arm)
        print(f"    outside-aperture fraction = "
              f"{outside['fraction_percent']:+.4f} %   "
              f"grid={witness['grid_shape']} readable={witness['readable']}",
              flush=True)
        out["layers"][str(layers)] = {
            "cpml_layers": layers,
            "mean_self_full": full["mean_self"],
            "mean_self_aperture": ap_arm["mean_self"],
            "gate_G2_0p95_1p05_full": full["gate_G2_0p95_1p05"],
            "outside_aperture": outside,
            "witness": witness,
            "full": full,
            "aperture": ap_arm,
        }

    got = [L for L in LAYER_SWEEP if str(L) in out["layers"]]
    if 10 in got:
        m10 = out["layers"]["10"]["mean_self_full"]
        f10 = out["layers"]["10"]["outside_aperture"]["fraction_percent"]
        out["control_reproduces_committed_cpml_full"] = {
            "mean_self_measured": m10,
            "mean_self_committed": SWEEP_BASELINE_CPML_FULL_MEAN_SELF,
            "mean_self_rel_diff": (abs(m10 - SWEEP_BASELINE_CPML_FULL_MEAN_SELF)
                                   / abs(SWEEP_BASELINE_CPML_FULL_MEAN_SELF)),
            "fraction_percent_measured": f10,
            "fraction_percent_committed": SWEEP_BASELINE_OUTSIDE_FRACTION_PCT,
        }
    out["all_arms_readable"] = all(
        out["layers"][str(L)]["witness"]["readable"] for L in got)
    out["unreadable_layer_counts"] = [
        L for L in got if not out["layers"][str(L)]["witness"]["readable"]]

    if got == list(LAYER_SWEEP):
        keys = [str(L) for L in LAYER_SWEEP]
        means = [out["layers"][k]["mean_self_full"] for k in keys]
        fracs = [out["layers"][k]["outside_aperture"]["fraction_percent"]
                 for k in keys]
        mags = [abs(f) for f in fracs]
        m40 = means[-1]
        gate_pass = bool(m40 >= SWEEP_GATE_MEAN_SELF_AT_40)
        monotone = bool(all(mags[i + 1] <= mags[i] + 1e-12
                            for i in range(len(mags) - 1)))
        flat_3_points = bool((max(fracs) - min(fracs)) <= 3.0)
        # NOT re-pointed with the control above, deliberately. This is the
        # PRE-DECLARED falsifier -- "mean_self stays within 0.03 of 0.748852"
        # -- and 0.748852 is the number the pre-declaration froze, not "the
        # current baseline". Letting it follow the re-pointed constant would
        # have silently redefined a frozen falsifier into "within 0.03 of
        # whatever we just measured", which always holds and never fires.
        near_baseline = bool(
            abs(m40 - SWEEP_BASELINE_CPML_FULL_MEAN_SELF_R1) <= 0.03)
        falsified = bool(near_baseline and flat_3_points)
        # 2026-09-15 (#1043 stage B): the pass sentence used to end "the
        # absorber's own reflection is the source of the deficit"
        # UNCONDITIONALLY. That attribution was read off a ladder measured on
        # the pre-#1043 solver, where the 10-layer arm sat at 0.748852 and the
        # gap closed with depth. With the pad continuation in place the ladder
        # is flat (0.9876 / 0.9890 / 0.9894 / 0.9889, spread 0.0018) and there
        # is no depth-dependent deficit left to attribute to anything -- the
        # depth dependence was the seam facet. A canned mechanism sentence
        # printed on a gate pass is the surface-metric verdict R5 forbids, so
        # the attribution is now conditional on a spread the run measures.
        # This narrows what the driver CLAIMS; it does not move the gate.
        ladder_spread = max(means) - min(means)
        if gate_pass and ladder_spread >= LADDER_SPREAD_FOR_DEPTH_ATTRIBUTION:
            verdict = ("GATE PASS -- mean_self at 40 layers >= "
                       f"{SWEEP_GATE_MEAN_SELF_AT_40}; the absorber's own "
                       "reflection is the source of the deficit "
                       f"(ladder spread {ladder_spread:.4f})")
        elif gate_pass:
            verdict = ("GATE PASS -- mean_self at 40 layers >= "
                       f"{SWEEP_GATE_MEAN_SELF_AT_40}, and the ladder is FLAT "
                       f"(spread {ladder_spread:.4f} over "
                       f"{LAYER_SWEEP[0]}..{LAYER_SWEEP[-1]} layers, below "
                       f"{LADDER_SPREAD_FOR_DEPTH_ATTRIBUTION}): there is NO "
                       "depth-dependent deficit to attribute to the absorber")
        elif falsified:
            verdict = ("FALSIFIED -- mean_self within 0.03 of 0.748852 AND the "
                       "fraction flat within 3 points; candidate (a), a far-x "
                       "CPML return, is refuted")
        else:
            verdict = ("NON-CLOSING -- neither the gate nor the falsifier's "
                       "conjunction holds; R2 forbids a fifth layer count "
                       "without a new pre-declaration")
        out["verdict"] = {
            "gate_pass": gate_pass,
            "mean_self_at_40": m40,
            "gate_threshold": SWEEP_GATE_MEAN_SELF_AT_40,
            "mean_self_by_layer": dict(zip(keys, means)),
            "fraction_percent_by_layer": dict(zip(keys, fracs)),
            "fraction_monotone_non_increasing_in_magnitude": monotone,
            "fraction_shrink_ratio_40_over_10": (mags[-1] / mags[0]
                                                 if mags[0] else None),
            "falsifier_mean_self_within_0p03_of_baseline": near_baseline,
            "falsifier_fraction_flat_within_3_points": flat_3_points,
            "falsifier_holds": falsified,
            "gate_and_observable_disagree": bool(gate_pass != monotone),
            "verdict": verdict,
        }
    else:
        out["verdict"] = {
            "verdict": ("INCOMPLETE -- not all four pre-declared layer counts "
                        f"ran (got {got})"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode",
                    choices=("arms", "layer-sweep", "residual-split"),
                    default="arms",
                    help="`arms` (default) = the six-arm boundary/window "
                         "campaign, unchanged; `layer-sweep` = the "
                         "pre-declaration addendum's Arm 1; `residual-split` "
                         "= the one interior arm that splits its residual")
    ap.add_argument("--split-layers", type=int, default=RESIDUAL_SPLIT_LAYERS,
                    help="layer count for --mode residual-split; must be one "
                         "the committed layer sweep already ran")
    ap.add_argument("--arms", default="",
                    help="comma-separated subset (mode=arms)")
    ap.add_argument("--layers", default="",
                    help="comma-separated subset of 10,16,20,40 "
                         "(mode=layer-sweep); a subset cannot reach the "
                         "pre-declared verdict and is recorded INCOMPLETE")
    ap.add_argument("--n-steps", type=int, default=n_steps,
                    help="PLUMBING SMOKE ONLY; the recorded numbers are "
                         "cv01's 25000 and any other value is not a result")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.arms.split(",") if s.strip()}
    rig = _assert_rig_matches_cv01()

    import jax
    import rfx

    provenance_check = _assert_rfx_is_this_repo(rfx)
    provenance = {
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip(),
        # Round-1 review of the #1043 stage-B PR: the commit alone is not
        # provenance. An artifact stamped with a clean sha from a tree with
        # uncommitted edits names code that does not exist anywhere, and that
        # is exactly how the first re-stamp of this lane's sibling artifact
        # misled -- it read `dirty: true` and the sha was innocent. `git
        # status --porcelain` is the cheapest thing that tells them apart.
        "dirty": bool(subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout.strip()),
        "rfx_version": getattr(rfx, "__version__", "unknown"),
        "rfx_file": rfx.__file__,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    if args.mode == "residual-split":
        out = run_residual_split(args, rig, provenance_check, provenance)
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out_path}")
        c = out["comparison"]
        for part in ("absorber_cell_slots_points", "interior_off_guide_points"):
            at10, at40 = c[part][str(cpml_n)], c[part][str(args.split_layers)]
            print(f"  {part:32s} {cpml_n:>3} layers {at10:+9.4f} -> "
                  f"{args.split_layers:>3} layers {at40:+9.4f}")
        print(f"  interior off-guide still dominates: "
              f"{c['interior_off_guide_still_dominates']}")
        print(f"  absorber-cell slots grew: {c['absorber_cell_slots_grew']}")
        return 0

    if args.mode == "layer-sweep":
        sweep = run_layer_sweep(args, rig, provenance_check, provenance)
        sweep_path = pathlib.Path(args.output)
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_path.write_text(json.dumps(sweep, indent=2) + "\n",
                              encoding="utf-8")
        print(f"\nwrote {sweep_path}")
        print(f"{'layers':>6}  {'grid':>15}  {'mean_self':>10}  "
              f"{'outside %':>10}  readable")
        for lay in LAYER_SWEEP:
            entry = sweep["layers"].get(str(lay))
            if entry is None:
                continue
            print(f"{lay:6d}  {str(entry['witness']['grid_shape']):>15}  "
                  f"{entry['mean_self_full']:10.6f}  "
                  f"{entry['outside_aperture']['fraction_percent']:+10.4f}  "
                  f"{entry['witness']['readable']}")
        print(f"\ngate (mean_self >= {SWEEP_GATE_MEAN_SELF_AT_40} at 40 "
              f"layers): {sweep['verdict'].get('gate_pass')}")
        print(f"verdict: {sweep['verdict']['verdict']}")
        return 0

    out = {
        "issue": 813,
        "predeclaration": "docs/design_notes/cv01_cpml_flux_selfcheck_predeclaration.md",
        "measures": ("cv01 straight-guide mean_self, Run 1 only -- the line "
                     "`mean_self = float(np.mean(T_self_smooth[above]))` in "
                     "validation/crossval/01_waveguide_bend.py"),
        "n_steps": int(args.n_steps),
        "n_steps_is_cv01_value": bool(args.n_steps == n_steps),
        "gate": ("pass = mean_self in [0.95, 1.05] -- cv01's G2, the line "
                 "`if 0.95 <= mean_self <= 1.05:` in "
                 "validation/crossval/01_waveguide_bend.py"),
        "rig_fidelity_check": rig,
        "rfx_provenance_check": provenance_check,
        "arms_added_after_the_first_run": list(ARMS_ADDED_POST_HOC),
        "committed_upml_reference": {
            "path": "validation/crossval/_01_waveguide_bend_results/crossval.json",
            "mean_self_smoothed_over_band": COMMITTED_UPML_MEAN_SELF,
        },
        "provenance": provenance,
        "arms": {},
    }

    for name, boundary, size in ARMS:
        if wanted and name not in wanted:
            continue
        print(f"--- {name}: boundary={boundary} size={size} ---", flush=True)
        arm = run_arm(boundary, size, steps=args.n_steps)
        out["arms"][name] = arm
        print(f"    mean_self = {arm['mean_self']:.6f}  "
              f"G2={'PASS' if arm['gate_G2_0p95_1p05'] else 'FAIL'}  "
              f"({arm['wall_s']}s)", flush=True)

    # Verdicts, computed here so the recorded verdict is the one printed.
    arms = out["arms"]
    if "upml_full" in arms:
        ctrl = arms["upml_full"]["mean_self"]
        out["control_reproduces_committed_run"] = {
            "measured": ctrl,
            "committed": COMMITTED_UPML_MEAN_SELF,
            "rel_diff": abs(ctrl - COMMITTED_UPML_MEAN_SELF) / COMMITTED_UPML_MEAN_SELF,
        }
    if "cpml_full" in arms:
        out["gate_verdict_cpml_full"] = (
            "PASS" if arms["cpml_full"]["gate_G2_0p95_1p05"] else "FAIL")
    if "cpml_full" in arms and "cpml_interior" in arms:
        full = arms["cpml_full"]["mean_self"]
        inner = arms["cpml_interior"]["mean_self"]
        rel = abs(inner - full) / abs(full)
        out["monitor_region_falsifier"] = {
            "cpml_full_mean_self": full,
            "cpml_interior_mean_self": inner,
            "rel_diff": rel,
            "threshold": 0.01,
            "verdict": ("REFUTED: clipping the absorber cells out of the "
                        "integration plane does not move mean_self"
                        if rel <= 0.01 else
                        "NOT REFUTED: the integration plane moves mean_self"),
        }

    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    for name, arm in arms.items():
        print(f"  {name:16s} mean_self={arm['mean_self']:.6f} "
              f"G2={'PASS' if arm['gate_G2_0p95_1p05'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
