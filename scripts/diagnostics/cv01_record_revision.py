"""Re-measure cv01's committed record on the fixed solver (issue #813).

Why a revision exists at all
----------------------------
``validation/crossval/_01_waveguide_bend_results/crossval.json`` was measured
on 2026-09-06 (commit d2285cc5) with cv01's guide ending in a vacuum facet at
the interior/absorber seam: ``run(subpixel_smoothing=True)`` rebuilt the update
permittivity from the declared geometry and applied no absorber-pad extension
to the rebuilt array, so a structure touching a domain face was solved with
``eps_r = 1`` inside its own absorber (#831 measured it, #1043 named it, PR
#1057 fixed it). Nothing re-runs that record -- ``tests/crossval/
test_waveguide_bend_header.py`` reads it statically -- so on a tree where the
defect is gone the committed numbers describe a solver that no longer exists.

This driver re-measures them. It does NOT overwrite the committed record: per
the #928 evidence split, evidence is APPEND-ONLY and a new revision moves no
consumer until a consumer's own adoption record names it, in a later diff.

What it runs
------------
Three things per boundary, and the first is the one that matters:

1. **The case itself**, in a subprocess, through the path PR for #813 added to
   it::

       01_waveguide_bend.py --meep-from-record <committed record> --out-dir <dir>

   The rfx legs are solved by cv01's own code -- not by a copy of it here --
   so no rig can drift between the case and its revision. The Meep leg is
   taken from the committed record instead of being re-run, which is sound
   because nothing on the rfx side can move a number Meep produced; the case
   script asserts the donor's Meep inputs against the ones this run would hand
   Meep before it takes anything (see ``_borrow_meep_leg`` there).

2. **A settling witness**, in process. cv01 registers flux monitors and no
   point probe, so ``run()`` has no probe time series to score and every
   cv01 record carries ``settling_db = None`` by construction. The witness arms
   below are cv01's Run 1 and Run 2 with ONE ``add_probe`` added each and
   nothing else changed. A probe is a passive recorder, and that is checked
   rather than asserted: the witness arm's ``mean_self`` / ``mean_T`` are
   compared with the case run's own, and a difference is reported.

   The witness arms are a hand-copy of cv01's rig, so ``_assert_rig_matches_cv01``
   greps the committed case for every copied line -- the same guard
   ``scripts/diagnostics/cv01_cpml_flux_selfcheck.py`` carries, extended to
   cover Run 2 and the bend transmittance, which that driver does not run.

3. **The build**, which assembles the append-only revision: the UPML record at
   the top level so ``--replay`` re-judges it unchanged, the CPML variant's
   whole document beside it, both arms' preflight output verbatim, the
   settling witnesses, and the old-vs-new table.

Usage::

    PYTHONPATH=$(git rev-parse --show-toplevel) \
        python3 scripts/diagnostics/cv01_record_revision.py \
        --work-dir /tmp/813_runs \
        --output validation/crossval/_01_waveguide_bend_results/crossval_r2.json \
        --log-dir validation/crossval/_01_waveguide_bend_logs

``--work-dir`` must be outside ``validation/crossval/``; the case script
refuses an ``--out-dir`` inside the committed evidence tree (#967 / PR #977)
and this driver refuses one before it starts a solve.

Runtime: about 35 s per arm on CPU, eight arms, so under ten minutes. cv01 is
a ``cpu-runner`` case in ``validation/crossval/manifest.json`` and needs no GPU.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
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
COMMITTED_RECORD = (REPO / "validation" / "crossval"
                    / "_01_waveguide_bend_results" / "crossval.json")

C0 = 2.998e8

# --- cv01's rig, copied verbatim from the committed script -------------------
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
wg_x = sx / 2
src_x = pml + dx

f_cutoff = 1.0 / (2.0 * np.sqrt(eps_wg - 1.0))
n_steps = 25000

# cv01's own gate limits, copied so the witness arms judge by the same numbers.
G1_T_BAND = (0.3, 1.0)
G2_SELF_T_BAND = (0.95, 1.05)
G3_MAX_ABS_GAP = 0.10

# Every one of these must still be present, verbatim, in the committed cv01.
# The sibling driver (cv01_cpml_flux_selfcheck.py) pins Run 1 and the self-T
# arithmetic; this one runs the BEND too, so it pins Run 2, the bend geometry,
# the y-normal output monitor and the normalized-T arithmetic as well -- and
# the three gate limits, because the witness arms report against them.
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
    "wg_x = sx / 2",
    "src_x = pml + dx",
    "f_cutoff = 1.0 / (2.0 * np.sqrt(eps_wg - 1.0))",
    "n_steps = 25000",
    'cpml_layers = 20 if boundary == "cpml" else cpml_n',
    # The line source, body and all.
    "        y = y_center - width / 2 + (i + 0.5) * width / 10",
    '        sim.add_source(position=(x, y, 0), component="ez",',
    "                       waveform=GaussianPulse(f0=fcen, bandwidth=fwidth / fcen,",
    "                                              amplitude=1.0 / 10))",
    # Run 1 -- the straight guide.
    'sim_s = Simulation(freq_max=0.25 * C0 / a, domain=(sx, sy, dx), dx=dx,',
    "                   boundary=BoundarySpec.uniform(boundary),",
    '                   cpml_layers=cpml_layers, mode="2d_tmz")',
    'sim_s.add_material("wg", eps_r=eps_wg)',
    "sim_s.add(Box((0, wg_y - w_wg / 2, 0), (sx, wg_y + w_wg / 2, dx)),",
    '          material="wg")',
    "add_line_source(sim_s, src_x, wg_y, w_wg)",
    'sim_s.add_flux_monitor(axis="x", coordinate=4 * a, freqs=freqs, name="input")',
    'sim_s.add_flux_monitor(axis="x", coordinate=sx - pml - 5 * dx,',
    "res_s = sim_s.run(n_steps=n_steps, subpixel_smoothing=True)",
    # Run 2 -- the bend. Two boxes, and a y-normal output plane.
    'sim_b = Simulation(freq_max=0.25 * C0 / a, domain=(sx, sy, dx), dx=dx,',
    'sim_b.add_material("wg", eps_r=eps_wg)',
    "sim_b.add(Box((0, wg_y - w_wg / 2, 0),",
    '              (wg_x + w_wg / 2, wg_y + w_wg / 2, dx)), material="wg")',
    "sim_b.add(Box((wg_x - w_wg / 2, wg_y - w_wg / 2, 0),",
    '              (wg_x + w_wg / 2, sy, dx)), material="wg")',
    "add_line_source(sim_b, src_x, wg_y, w_wg)",
    'sim_b.add_flux_monitor(axis="x", coordinate=4 * a, freqs=freqs, name="input")',
    'sim_b.add_flux_monitor(axis="y", coordinate=sy - pml - 5 * dx,',
    "res_b = sim_b.run(n_steps=n_steps, subpixel_smoothing=True)",
    # The band, and both transmittances built on it.
    'above = (f_meep > f_cutoff + 0.005) & (f_meep < 0.20)',
    "safe_in_s = np.maximum(np.abs(flux_in_s), np.max(np.abs(flux_in_s)) * 1e-6)",
    "T_self = flux_out_s / safe_in_s",
    "T_self_smooth = uniform_filter1d(T_self, size=20)",
    "safe_in_b = np.maximum(np.abs(flux_in_b), np.max(np.abs(flux_in_b)) * 1e-6)",
    "T_bend_abs = flux_out_b / safe_in_b",
    "T_norm = T_bend_abs / np.maximum(np.abs(T_self), 1e-30)",
    "T_norm_smooth = uniform_filter1d(T_norm, size=20)",
    "mean_self = float(np.mean(T_self_smooth[above]))",
    "mean_T = float(np.mean(T_norm_smooth[above]))",
    # The three gate limits the witness arms report against.
    "G1_T_BAND = (0.3, 1.0)",
    "G2_SELF_T_BAND = (0.95, 1.05)",
    "G3_MAX_ABS_GAP = 0.10",
)


def _assert_rfx_is_this_repo(rfx_module) -> dict:
    """Fail loudly unless ``import rfx`` resolved inside THIS checkout.

    Running a script by path puts the SCRIPT's directory on ``sys.path``, not
    the repo root, so ``import rfx`` otherwise falls through to whatever the
    interpreter can see while the artifact records this tree's commit sha.
    Same check, same reason, as the sibling driver's.
    """
    rfx_file = pathlib.Path(rfx_module.__file__).resolve()
    try:
        rfx_file.relative_to(REPO)
    except ValueError:
        raise SystemExit(
            "rfx provenance check FAILED -- `import rfx` resolved to\n"
            f"    {rfx_file}\nwhich is NOT under this script's repo root\n"
            f"    {REPO}\nso the numbers this run would record are not this "
            f"tree's. Re-run with\n    PYTHONPATH={REPO} python3 "
            f"scripts/diagnostics/{pathlib.Path(__file__).name} ...\n"
        ) from None
    return {"rfx_file": str(rfx_file), "under_repo_root": True,
            "repo_root": str(REPO),
            "rfx_version": getattr(rfx_module, "__version__", None)}


def _assert_rig_matches_cv01() -> dict:
    text = CV01.read_text(encoding="utf-8")
    missing = [line for line in RIG_LINES if line not in text]
    if missing:
        raise SystemExit(
            "rig fidelity check FAILED -- these lines are no longer in "
            f"{CV01.relative_to(REPO)} verbatim, so the witness arms would "
            "measure a different structure than cv01:\n  " + "\n  ".join(missing))
    return {"checked_lines": len(RIG_LINES), "all_present_verbatim": True}


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def add_line_source(sim, x, y_center, width):
    """cv01's own ``add_line_source``, verbatim -- ten ez point sources."""
    from rfx import GaussianPulse
    for i in range(10):
        y = y_center - width / 2 + (i + 0.5) * width / 10
        sim.add_source(position=(x, y, 0), component="ez",
                       waveform=GaussianPulse(f0=fcen, bandwidth=fwidth / fcen,
                                              amplitude=1.0 / 10))


def _witness_arm(boundary: str, cpml_layers: int, which: str) -> dict:
    """cv01's Run 1 or Run 2 with one point probe added, and nothing else.

    The probe sits on the guide axis AT the arm's own output plane, which is
    where a late return would arrive and where the DFT that the gates read is
    accumulated.
    """
    from rfx import Simulation, Box, flux_spectrum
    from rfx.boundaries.spec import BoundarySpec

    t0 = time.time()
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = Simulation(freq_max=0.25 * C0 / a, domain=(sx, sy, dx), dx=dx,
                         boundary=BoundarySpec.uniform(boundary),
                         cpml_layers=cpml_layers, mode="2d_tmz")
        sim.add_material("wg", eps_r=eps_wg)
        if which == "straight":
            sim.add(Box((0, wg_y - w_wg / 2, 0), (sx, wg_y + w_wg / 2, dx)),
                    material="wg")
        else:
            sim.add(Box((0, wg_y - w_wg / 2, 0),
                        (wg_x + w_wg / 2, wg_y + w_wg / 2, dx)), material="wg")
            sim.add(Box((wg_x - w_wg / 2, wg_y - w_wg / 2, 0),
                        (wg_x + w_wg / 2, sy, dx)), material="wg")
        add_line_source(sim, src_x, wg_y, w_wg)
        sim.add_flux_monitor(axis="x", coordinate=4 * a, freqs=freqs,
                             name="input")
        if which == "straight":
            out_axis, out_coord = "x", sx - pml - 5 * dx
            probe_xy = (out_coord, wg_y)
        else:
            out_axis, out_coord = "y", sy - pml - 5 * dx
            probe_xy = (wg_x, out_coord)
        sim.add_flux_monitor(axis=out_axis, coordinate=out_coord, freqs=freqs,
                             name="output")
        # THE only difference from cv01's own arm.
        sim.add_probe(position=(probe_xy[0], probe_xy[1], 0), component="ez")
        with redirect_stdout(buf):
            report = sim.preflight(strict=False)
        build_warnings = [str(w.message) for w in caught]

    with warnings.catch_warnings(record=True) as caught_run:
        warnings.simplefilter("always")
        res = sim.run(n_steps=n_steps, subpixel_smoothing=True)
        run_warnings = [str(w.message) for w in caught_run]

    flux_in = np.asarray(flux_spectrum(res.flux_monitors["input"]), dtype=float)
    flux_out = np.asarray(flux_spectrum(res.flux_monitors["output"]), dtype=float)
    settling = getattr(res, "settling_db", None)
    witness = getattr(res, "settling_witness", None)
    return {
        "which": which,
        "boundary": boundary,
        "cpml_layers": int(cpml_layers),
        "n_steps": int(n_steps),
        "probe_position_m": [float(probe_xy[0]), float(probe_xy[1]), 0.0],
        "probe_component": "ez",
        "settling_db": None if settling is None else float(settling),
        "settling_witness": witness,
        "flux_in": [float(v) for v in flux_in],
        "flux_out": [float(v) for v in flux_out],
        "preflight_stdout": buf.getvalue(),
        "preflight_formatted": (report.format() if hasattr(report, "format")
                                else None),
        "build_warnings": build_warnings,
        "run_warnings": run_warnings,
        "grid_shape": [int(n) for n in res.grid.shape],
        "wall_s": round(time.time() - t0, 1),
    }


def _witness_pair(boundary: str) -> dict:
    """Both witness arms for one boundary, plus the quantities cv01 gates."""
    cpml_layers = 20 if boundary == "cpml" else cpml_n
    straight = _witness_arm(boundary, cpml_layers, "straight")
    print(f"    straight: settling_db={straight['settling_db']} "
          f"({straight['wall_s']}s)", flush=True)
    bend = _witness_arm(boundary, cpml_layers, "bend")
    print(f"    bend:     settling_db={bend['settling_db']} "
          f"({bend['wall_s']}s)", flush=True)

    # cv01's transmittance block, same arithmetic, on the witness arms' fluxes.
    flux_in_s = np.asarray(straight["flux_in"], dtype=float)
    flux_out_s = np.asarray(straight["flux_out"], dtype=float)
    flux_in_b = np.asarray(bend["flux_in"], dtype=float)
    flux_out_b = np.asarray(bend["flux_out"], dtype=float)
    f_meep = freqs * a / C0
    above = (f_meep > f_cutoff + 0.005) & (f_meep < 0.20)
    safe_in_s = np.maximum(np.abs(flux_in_s), np.max(np.abs(flux_in_s)) * 1e-6)
    T_self = flux_out_s / safe_in_s
    T_self_smooth = uniform_filter1d(T_self, size=20)
    safe_in_b = np.maximum(np.abs(flux_in_b), np.max(np.abs(flux_in_b)) * 1e-6)
    T_bend_abs = flux_out_b / safe_in_b
    T_norm = T_bend_abs / np.maximum(np.abs(T_self), 1e-30)
    T_norm_smooth = uniform_filter1d(T_norm, size=20)
    return {
        "boundary": boundary,
        "cpml_layers": int(cpml_layers),
        "arms": {"straight": straight, "bend": bend},
        "mean_self_smoothed_over_band": float(np.mean(T_self_smooth[above])),
        "mean_T_smoothed_over_band": float(np.mean(T_norm_smooth[above])),
        "n_bins_in_band": int(above.sum()),
        "note": ("cv01's Run 1 and Run 2 with one ez point probe added to each "
                 "and nothing else changed. cv01 itself registers no probe, so "
                 "its records carry settling_db = None by construction; these "
                 "arms are what scores the ring-down. The two means above are "
                 "compared with the case run's in "
                 "`settling_witness.non_perturbation_check`."),
    }


def _run_case(boundary: str, work_dir: pathlib.Path,
              log_dir: pathlib.Path) -> dict:
    """Run cv01 itself, borrowing the Meep leg, into ``work_dir/<boundary>``."""
    out_dir = work_dir / boundary
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["JAX_ENABLE_X64"] = "1"
    env["MPLBACKEND"] = "Agg"
    if boundary == "upml":
        env.pop("RFX_BOUNDARY", None)
    else:
        env["RFX_BOUNDARY"] = boundary
    argv = [sys.executable, str(CV01),
            "--meep-from-record", str(COMMITTED_RECORD),
            "--out-dir", str(out_dir)]
    t0 = time.time()
    proc = subprocess.run(argv, cwd=str(REPO), env=env, capture_output=True,
                          text=True, timeout=7200)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
    log_path = log_dir / f"{stamp}_r2_{boundary}_run.log"
    log_path.write_text(
        f"$ RFX_BOUNDARY={env.get('RFX_BOUNDARY', '(unset -> upml)')} "
        f"PYTHONPATH={REPO} JAX_ENABLE_X64=1 python3 "
        f"{CV01.relative_to(REPO)} --meep-from-record "
        f"{COMMITTED_RECORD.relative_to(REPO)} --out-dir {out_dir}\n\n"
        + proc.stdout + "\n--- stderr ---\n" + proc.stderr
        + f"\n--- returncode: {proc.returncode} ---\n", encoding="utf-8")
    record_path = out_dir / "crossval.json"
    if not record_path.is_file():
        raise SystemExit(
            f"cv01 ({boundary}) wrote no record; see {log_path}\n"
            + proc.stdout[-3000:] + proc.stderr[-3000:])
    doc = json.loads(record_path.read_text(encoding="utf-8"))
    if doc["verdict"]["exit_code"] != proc.returncode:
        raise SystemExit(
            f"cv01 ({boundary}) returned {proc.returncode} while its record "
            f"declares {doc['verdict']['exit_code']} -- the #946 finalizer "
            f"should have reconciled these; see {log_path}")
    return {
        "boundary": boundary,
        "argv": argv,
        "returncode": proc.returncode,
        "wall_s": round(time.time() - t0, 1),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "record": doc,
        "log_path": str(log_path.relative_to(REPO)),
        "log_sha256": _sha256(log_path),
    }


def _preflight_verbatim(stdout: str, stderr: str) -> dict:
    """Every ``[PREFLIGHT]`` line the run emitted, in order, deduplicated.

    Preflight output is part of the result (repo rule): a number is quoted
    with the warnings its run raised, not without them. The banner prints once
    per ``sim.preflight()`` and once more when ``run()`` re-checks, so the same
    text appears twice per arm; the order is kept and the repeat dropped.
    """
    seen, out, emitted = set(), [], 0
    for stream in (stdout, stderr):
        for raw in stream.splitlines():
            line = raw.strip()
            if not (line.startswith("[PREFLIGHT]")
                    or line.startswith("[FLUX REGION]")):
                continue
            emitted += 1
            if line in seen:
                continue
            seen.add(line)
            out.append(line)
    return {"distinct_lines": out, "lines_emitted_including_repeats": emitted}


def _advisory_codes(stdout: str, stderr: str) -> dict:
    """Did the two advisories PR #1057 / #1079 added fire on this rig?

    Named explicitly rather than left to a reader's grep: both are new since
    the committed record was written, both are about a structure meeting an
    absorber, and cv01's guide meets two of them.
    """
    blob = stdout + stderr
    return {
        "dielectric_at_absorber_seam": "absorber seam, and" in blob,
        "conductor_in_thin_absorber": "absorbing layer(s). Both at once" in blob,
        "signatures_grepped": {
            "dielectric_at_absorber_seam": "absorber seam, and",
            "conductor_in_thin_absorber": "absorbing layer(s). Both at once",
        },
        "searched_streams": ["stdout", "stderr"],
        "note": ("both checks were added after the committed record was "
                 "measured -- _validate_cfg_dielectric_at_absorber_seam with "
                 "PR #1057, _validate_cfg_conductor_in_thin_absorber with "
                 "#1079/#801. The full preflight text of each arm is quoted "
                 "verbatim beside this, so the answer is checkable rather "
                 "than taken from this flag."),
    }


UPSTREAM = (REPO / "validation" / "crossval" / "_01_waveguide_bend_upstream"
            / "bend-flux.py")


def _comparator_geometry_note(delta_mean_T: float) -> dict:
    """Where the Meep leg's guide ends, arithmetically, and where upstream's does.

    Recorded because this revision changes the rfx side of a comparison and
    leaves the Meep side exactly as it was, and the change is in the very
    quantity the two sides now differ on: whether the guide is continued into
    the absorber or terminated at its face. This is a geometry reading and a
    direction, NOT an attribution -- the Meep leg was not re-run here and
    nothing below measures what it would do if it were.
    """
    cell_x, cell_y = 16.0 + 2, 16.0 + 2          # `cell_over_a` in the record
    dpml = 1.0                                   # `mp.PML(1.0)` in the leg
    interior_x_lo, interior_y_hi = -cell_x / 2 + dpml, cell_y / 2 - dpml
    # mp.Block(size=(sx/(2a)+0.5, 1), center=(-sx/(4a)+0.25, 0))
    horiz_x_lo = (-16.0 / 4 + 0.25) - (16.0 / 2 + 0.5) / 2
    # mp.Block(size=(1, sy/(2a)+0.5), center=(0, sy/(4a)-0.25))
    vert_y_hi = (16.0 / 4 - 0.25) + (16.0 / 2 + 0.5) / 2
    upstream_text = UPSTREAM.read_text(encoding="utf-8")
    return {
        "question": ("after PR #1057 rfx continues a boundary-touching "
                     "dielectric through its own absorber pad. Does the Meep "
                     "leg this case compares against do the same?"),
        "meep_leg_interior_face_x_lo_over_a": interior_x_lo,
        "meep_leg_horizontal_block_x_lo_over_a": horiz_x_lo,
        "meep_leg_interior_face_y_hi_over_a": interior_y_hi,
        "meep_leg_vertical_block_y_hi_over_a": vert_y_hi,
        "meep_leg_guide_clearance_into_pml_over_a": {
            "x_lo": horiz_x_lo - interior_x_lo,
            "y_hi": interior_y_hi - vert_y_hi,
        },
        "meep_leg_guide_ends_at_the_pml_face": (
            horiz_x_lo == interior_x_lo and vert_y_hi == interior_y_hi),
        "upstream_tutorial_uses_an_infinite_block": (
            "size=mp.Vector3(mp.inf, w, mp.inf)" in upstream_text),
        "upstream_path": str(UPSTREAM.relative_to(REPO)),
        "measured_direction_on_the_rfx_side": {
            "delta_mean_T_when_the_rfx_facet_was_removed": delta_mean_T,
            "note": ("same rig, same window, same step count: removing rfx's "
                     "own end facet RAISED its band-mean T by this much. The "
                     "sign is what makes the reading below worth recording."),
        },
        "reading": (
            "cv01's Meep leg builds two FINITE blocks whose outer faces land "
            "exactly on the PML inner faces, so its guide is terminated at the "
            "absorber with vacuum inside it -- the construction rfx had until "
            "PR #1057. Upstream's own bend-flux.py instead uses an INFINITE "
            "block and runs the guide straight through the PML. That is a "
            "SEVENTH divergence from the tutorial, on top of the six this "
            "case's own REPRODUCE_GATE_RECORD do_not_repeat text already "
            "lists, and it is in the one place this revision moved."),
        "not_claimed": (
            "that the G3 gap is caused by it. Meep was not re-run here and "
            "this lane measured nothing on the Meep side. The cheap falsifier "
            "is named in the design note: re-run the Meep leg with the guide "
            "continued through the PML and see whether its band mean rises "
            "toward rfx's. Until that runs, this is a geometry reading."),
    }


def _gates(mean_T: float, mean_self: float, meep_mean) -> dict:
    """cv01's three gates, spelled from cv01's own limits (RIG_LINES-pinned)."""
    return {
        "G1_smoothed_T_in_0p3_1p0": bool(G1_T_BAND[0] <= mean_T <= G1_T_BAND[1]),
        "G2_straight_self_T_in_0p95_1p05": bool(
            G2_SELF_T_BAND[0] <= mean_self <= G2_SELF_T_BAND[1]),
        "G3_abs_rfx_minus_meep_lt_0p10": (
            None if meep_mean is None
            else bool(abs(mean_T - meep_mean) < G3_MAX_ABS_GAP)),
    }


def _old_vs_new(old: dict, new_upml: dict, new_cpml: dict) -> dict:
    """The table the issue asks for, computed rather than transcribed."""

    def row(doc):
        m = doc["measured"]
        meep = (m.get("meep") or {})
        return {
            "mean_T_smoothed_over_band": m["mean_T_smoothed_over_band"],
            "mean_self_smoothed_over_band": m["mean_self_smoothed_over_band"],
            "min_T_smoothed_over_band": m["min_T_smoothed_over_band"],
            "max_T_smoothed_over_band": m["max_T_smoothed_over_band"],
            "meep_mean_T_smoothed_over_band": meep.get(
                "mean_T_smoothed_over_band"),
            "abs_rfx_minus_meep": meep.get("abs_rfx_minus_meep"),
            "gates": doc["gates"],
            "exit_code": doc["verdict"]["exit_code"],
            "summary": doc["verdict"]["summary"],
            "boundary": doc["rig"]["boundary"],
            "boundary_layers": doc["rig"]["boundary_layers"],
            "commit": doc["commit"],
            "date_utc": doc["date_utc"],
        }

    old_row, new_row, cpml_row = row(old), row(new_upml), row(new_cpml)
    changed = {
        name: {"old": old_row["gates"][name], "new": new_row["gates"][name]}
        for name in old_row["gates"]
        if old_row["gates"][name] != new_row["gates"][name]
    }
    return {
        "committed_upml": old_row,
        "revision_upml": new_row,
        "revision_cpml_variant": cpml_row,
        "deltas_upml": {
            key: (new_row[key] - old_row[key])
            for key in ("mean_T_smoothed_over_band",
                        "mean_self_smoothed_over_band",
                        "min_T_smoothed_over_band",
                        "max_T_smoothed_over_band",
                        "abs_rfx_minus_meep")
            if isinstance(old_row[key], (int, float))
            and isinstance(new_row[key], (int, float))
        },
        "meep_leg_identical": (
            old_row["meep_mean_T_smoothed_over_band"]
            == new_row["meep_mean_T_smoothed_over_band"]),
        "gate_verdicts_that_changed": changed,
        "note": ("the Meep band mean is the SAME number in both rows by "
                 "construction -- the revision borrows the committed record's "
                 "Meep arrays rather than re-running Meep -- so every movement "
                 "in abs_rfx_minus_meep is the rfx leg moving."),
    }


def _provenance(rfx_check: dict, rig: dict) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=str(REPO),
                                           text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "case_sha256": _sha256(CV01),
        "driver_sha256": _sha256(pathlib.Path(__file__)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "date_utc": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "rfx_provenance_check": rfx_check,
        "rig_fidelity_check": rig,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", required=True,
                        help="scratch directory for the case runs; must be "
                             "outside validation/crossval/")
    parser.add_argument("--output", required=True,
                        help="where to write the revision record")
    parser.add_argument("--log-dir", required=True,
                        help="where the two run logs are written")
    parser.add_argument("--skip-witness", action="store_true",
                        help="skip the probe-companion settling arms (they "
                             "double the wall time); the revision then records "
                             "that the witness was not run, rather than "
                             "omitting the field")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO / "validation" / "crossval"))
    import _exit_evidence  # noqa: E402  (crossval dir on sys.path)
    work_dir = pathlib.Path(
        _exit_evidence.refuse_evidence_tree(args.work_dir, "--work-dir"))
    log_dir = pathlib.Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO / log_dir
    out_path = pathlib.Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO / out_path

    import rfx  # noqa: E402  (after the path pin above)
    rfx_check = _assert_rfx_is_this_repo(rfx)
    rig_check = _assert_rig_matches_cv01()
    print(f"rig fidelity: {rig_check['checked_lines']} lines present verbatim "
          f"in {CV01.relative_to(REPO)}", flush=True)

    runs, witnesses = {}, {}
    for boundary in ("upml", "cpml"):
        print(f"--- cv01, boundary={boundary} ---", flush=True)
        runs[boundary] = _run_case(boundary, work_dir, log_dir)
        m = runs[boundary]["record"]["measured"]
        print(f"    mean_self={m['mean_self_smoothed_over_band']:.6f} "
              f"mean_T={m['mean_T_smoothed_over_band']:.6f} "
              f"exit={runs[boundary]['returncode']} "
              f"({runs[boundary]['wall_s']}s)", flush=True)
        if not args.skip_witness:
            print(f"    settling witness arms ({boundary})", flush=True)
            witnesses[boundary] = _witness_pair(boundary)

    upml_doc = dict(runs["upml"]["record"])
    cpml_doc = runs["cpml"]["record"]
    committed = json.loads(COMMITTED_RECORD.read_text(encoding="utf-8"))

    # Re-judge from the NEW arrays, through this driver's own copy of cv01's
    # limits, and refuse to write if that disagrees with what the case decided.
    for name, doc in (("upml", upml_doc), ("cpml", cpml_doc)):
        m = doc["measured"]
        meep_mean = (m["meep"] or {}).get("mean_T_smoothed_over_band")
        rejudged = _gates(m["mean_T_smoothed_over_band"],
                          m["mean_self_smoothed_over_band"], meep_mean)
        if rejudged != doc["gates"]:
            raise SystemExit(
                f"re-judging the {name} arm from its own arrays disagrees with "
                f"the gates the case wrote: {rejudged} vs {doc['gates']}")

    witness_block = {}
    for boundary, pair in witnesses.items():
        case_m = runs[boundary]["record"]["measured"]
        witness_block[boundary] = dict(pair)
        witness_block[boundary]["non_perturbation_check"] = {
            "case_mean_self": case_m["mean_self_smoothed_over_band"],
            "witness_mean_self": pair["mean_self_smoothed_over_band"],
            "mean_self_identical": (
                case_m["mean_self_smoothed_over_band"]
                == pair["mean_self_smoothed_over_band"]),
            "case_mean_T": case_m["mean_T_smoothed_over_band"],
            "witness_mean_T": pair["mean_T_smoothed_over_band"],
            "mean_T_identical": (case_m["mean_T_smoothed_over_band"]
                                 == pair["mean_T_smoothed_over_band"]),
            "note": ("a probe is a passive recorder, so the witness arm's own "
                     "gated quantities must reproduce the case run's. If they "
                     "do not, the witness measures a different simulation and "
                     "its settling_db does not describe the record."),
        }

    upml_doc["revision"] = {
        "schema": "cv01-waveguide-bend-revision/v1",
        "revision": 2,
        "revision_of": str(COMMITTED_RECORD.relative_to(REPO)),
        "revision_of_sha256": _sha256(COMMITTED_RECORD),
        "revision_of_commit": committed.get("commit"),
        "revision_of_date_utc": committed.get("date_utc"),
        "issue": 813,
        "related": [831, 1027, 1043, 1057],
        "adopted_by": [],
        "adoption_note": (
            "NOTHING adopts this revision. It is evidence, not a calibration: "
            "per the #928 split a new revision moves no consumer until that "
            "consumer's own adoption record names it, and the revision and an "
            "adoption may not change in one diff. The committed record, its "
            "gates and every constant pointed at it -- "
            "scripts/diagnostics/cv01_cpml_flux_selfcheck.py's "
            "COMMITTED_UPML_MEAN_SELF among them -- are untouched here."),
        "what_this_case_uniquely_claims": (
            "A bent 2-D dielectric slab guide's transmittance against Meep on "
            "the same rasterized geometry, normalized by a straight-guide "
            "self-check measured on that same geometry. No other crossval case "
            "carries it: cv03 is the STRAIGHT guide's flux identity and "
            "dispersion with no bend and no Meep leg of its own; cv09/cv10 are "
            "half-symmetric waveguide mirror-plane cases; cv11/cv18/cv19 are "
            "hollow metal WR-90 structures, not dielectric guides. The bend "
            "loss -- radiation at a 90-degree corner in a high-index slab "
            "guide -- is what only this case measures."),
        "why_a_revision": (
            "The committed record was measured with cv01's guide ending in a "
            "vacuum facet at the interior/absorber seam: run(subpixel_"
            "smoothing=True) rebuilt the update permittivity from the declared "
            "geometry with no absorber-pad extension, so a structure touching "
            "a domain face was solved with eps_r = 1 inside its own absorber. "
            "PR #1057 continues the geometry through the pad. Nothing re-runs "
            "cv01's record -- its header test reads it statically -- so on the "
            "fixed tree the committed numbers describe a solver that is gone."),
        "meep_leg": (
            "NOT re-run. The Meep arrays are the committed record's, borrowed "
            "through 01_waveguide_bend.py --meep-from-record, which asserts "
            "the donor's Meep rig block and the rfx-side quantities that Meep "
            "geometry is derived from against this run's before taking "
            "anything (see meep_leg_source). Sound because no rfx change can "
            "move a number Meep produced, and Meep does not import in this "
            "pod (numpy 2.x against a numpy 1.x wheel)."),
        "runs_fdtd": True,
        "old_vs_new": _old_vs_new(committed, upml_doc, cpml_doc),
        "comparator_geometry_note": _comparator_geometry_note(
            upml_doc["measured"]["mean_T_smoothed_over_band"]
            - committed["measured"]["mean_T_smoothed_over_band"]),
        "preflight_verbatim": {
            boundary: _preflight_verbatim(runs[boundary]["stdout"],
                                          runs[boundary]["stderr"])
            for boundary in runs
        },
        "advisories_added_since_the_committed_record": {
            boundary: _advisory_codes(runs[boundary]["stdout"],
                                      runs[boundary]["stderr"])
            for boundary in runs
        },
        "settling_witness": witness_block or {
            "status": "NOT RUN (--skip-witness)",
            "note": ("cv01 registers no point probe, so its own records carry "
                     "settling_db = None by construction; without these arms "
                     "the revision has no ring-down witness."),
        },
        "run_logs": {boundary: {"path": runs[boundary]["log_path"],
                                "sha256": runs[boundary]["log_sha256"],
                                "returncode": runs[boundary]["returncode"],
                                "wall_s": runs[boundary]["wall_s"]}
                     for boundary in runs},
        "cpml_variant_note": (
            "cv01's committed claim is the UPML run; the CPML variant is the "
            "diagnostic arm RFX_BOUNDARY=cpml selects, at 20 absorber layers "
            "since #813/#1027. Its whole record is carried under "
            "`cpml_variant` so `--replay` on THIS file still re-judges the "
            "UPML arm, which is the one the case is defined by."),
        "provenance": _provenance(rfx_check, rig_check),
    }
    upml_doc["cpml_variant"] = cpml_doc

    # Written through the case's own writer, with `arm=False` because this
    # process's exit status is the DRIVER's and not the case's. Not cosmetic:
    # tests/crossval/test_crossval_exit_code_is_the_process_exit_code.py takes
    # every committed record apart and puts it back together through this same
    # helper, comparing bytes, so a revision written with a hand-rolled
    # json.dump (a trailing newline is enough) reds that contract. One writer,
    # one set of bytes.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verdict = upml_doc["verdict"]
    _exit_evidence.write_record(
        str(out_path), upml_doc, exit_code=verdict["exit_code"],
        summary=verdict.get("summary"), arm=False)
    print(f"\nrevision written: {out_path.relative_to(REPO)}")
    print(f"  upml  mean_T={upml_doc['measured']['mean_T_smoothed_over_band']!r}")
    print(f"  upml  gates={upml_doc['gates']}")
    print(f"  cpml  gates={cpml_doc['gates']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - driver entry point
    sys.exit(main())
