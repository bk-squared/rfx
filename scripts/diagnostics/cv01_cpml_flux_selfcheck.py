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

``size=None`` is cv01's registration today and keeps the legacy full padded
plane (``rfx/runners/uniform.py:74``). A finite ``size=`` clamps to the
physical interior and excludes CPML and bounding-node slots (PR #990, issue
#910). The ``interior`` arms are therefore the falsifier for "the missing
power is absorber cells inside the integration plane": if clipping them out
does not move ``mean_self``, that attribution is refuted.

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
    # Geometry, material and source -- the structure itself, not just its numbers.
    'sim_s = Simulation(freq_max=0.25 * C0 / a, domain=(sx, sy, dx), dx=dx,',
    "                   boundary=BoundarySpec.uniform(boundary),",
    '                   cpml_layers=cpml_n, mode="2d_tmz")',
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
    "if 0.95 <= mean_self <= 1.05:",
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


def run_arm(boundary: str, size, steps: int = n_steps) -> dict:
    """cv01 Run 1 under one boundary and one monitor window."""
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
                         cpml_layers=cpml_n, mode="2d_tmz")
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

# The committed UPML run this driver's control must reproduce.
COMMITTED_UPML_MEAN_SELF = 0.9891610388008335


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--arms", default="", help="comma-separated subset")
    ap.add_argument("--n-steps", type=int, default=n_steps,
                    help="PLUMBING SMOKE ONLY; the recorded numbers are "
                         "cv01's 25000 and any other value is not a result")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.arms.split(",") if s.strip()}
    rig = _assert_rig_matches_cv01()

    import jax
    import rfx

    provenance_check = _assert_rfx_is_this_repo(rfx)

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
        "provenance": {
            "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "commit": subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip(),
            "rfx_version": getattr(rfx, "__version__", "unknown"),
            "rfx_file": rfx.__file__,
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
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
