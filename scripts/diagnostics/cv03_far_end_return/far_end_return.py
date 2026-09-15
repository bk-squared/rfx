"""#831 — cv03's far-end return: the depth sweep, the pad-extension switch, the
assembled absorber profile and the time-of-flight control, in one artifact.

Pre-declaration (frozen BEFORE this ran, and the only place the verdicts may be
read from): ``docs/design_notes/issue831_far_end_return_predeclaration.md``.

WHAT THIS IS FOR.  #831 reports ``|B/A| ~ 0.54`` on cv03's straight lossless
guide and reports it getting WORSE as the absorber deepens (0.538 / 0.598 /
0.628 at 20 / 40 / 60 cells).  PR #1027 swept cv01's straight guide over the
same absorber depths and its number improved monotonically.  Every
absorber-side mechanism in ``rfx/boundaries/{cpml,upml}.py`` predicts the
#1027 direction -- ``sigma_max ~ 1/d`` keeps the designed optical depth
N-invariant and the per-cell grading step falls as ``1/N^2`` -- so the sign
reported in #831 has no mechanism in this code.  This driver measures, rather
than argues, which of the four pre-declared hypotheses carries it.

The section-8 rows of #831 were produced by a driver that was never committed
(see section 8.2 of ``issue812_cv03_dispersion_regate_predeclaration.md``), so
"the trend does not reproduce under the committed estimator" is a live outcome
and is gated as one (G-A, A-ABSENT).

HOW IT RUNS THE CASE.  Exact textual edits applied to a COPY of
``validation/crossval/03_straight_waveguide_flux.py``, the pattern
``scripts/diagnostics/cv03_flux/tof_reflection_free.py`` established for this
lane.  The committed case is never modified and no gate of it moves.  Each copy
also gets:

  * a point probe at the fit-window centre, so ``result.settling_db`` has a
    record to score (the committed case has only flux monitors and a DFT plane,
    and would return ``settling_db=None``);
  * a dump block appended at the case's exit-code anchor, carrying the per-bin
    ``|B/A|`` / ``n_eff`` / residual / ``T(f)`` traces, the settling witness,
    the realised fit window in Meep and rfx coordinates AND in grid indices,
    and ``|Ez(x)|`` at the carrier bin across the FULL x extent, pads included.

STAGES

  ``profile``  no FDTD.  Builds the cv03 ``Simulation`` at each (boundary,
               layers) and dumps the assembled absorber arrays on the guide
               centre row plus the rasterized ``eps_r(x)``.  This is the
               "instrument the complete assembled profile" the 2026-09-13
               closing comment on #831 asked for.
  ``tof``      Arm C0, the control: sx = 40a with the DFT window stopped before
               the return (150 a/c0) and after it (400 a/c0).
  ``sweep``    Arm A: {upml, cpml} x {20, 40, 60} on the committed recipe.
  ``noext``    Arm B: upml x {20, 40, 60} with
               ``include_cpml_pad_extension=False``.

Run (2-D, CPU, minutes per arm -- no VESSL):

    PYTHONPATH=<repo> python3 scripts/diagnostics/cv03_far_end_return/far_end_return.py \
        --stage all --output scripts/diagnostics/cv03_far_end_return/far_end_return.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
CASE = REPO / "validation" / "crossval" / "03_straight_waveguide_flux.py"
COMPARATORS = REPO / "validation" / "crossval" / "comparators"

# The case's own exit-code anchor -- the last line before the verdict, so every
# quantity the dump needs is already bound.
_ANCHOR = "# Exit code (rfx crossval convention)"

# oracle.n_eff_at_carrier_bin, docs/design_notes/issue812_cv03_dispersion_matched_frequency.json
N_EFF_CARRIER = 2.83887
ETA_0 = float(np.sqrt(4e-7 * np.pi / 8.8541878128e-12))


# --- provenance: `import rfx` must resolve inside THIS worktree --------------
# Copied in intent from scripts/diagnostics/cv01_cpml_flux_selfcheck.py:239-262.
# Running a script by path puts the SCRIPT's directory on sys.path, not the repo
# root, so `import rfx` otherwise falls through to whatever install the
# interpreter can see -- and the artifact then records this branch's sha beside
# numbers a different tree produced.
def _assert_rfx_is_this_repo() -> dict:
    import rfx
    rfx_file = pathlib.Path(rfx.__file__).resolve()
    try:
        rfx_file.relative_to(REPO)
    except ValueError:
        raise SystemExit(
            "rfx provenance check FAILED -- `import rfx` resolved to\n"
            f"    {rfx_file}\n"
            f"which is NOT under this script's repo root\n    {REPO}\n"
            "Re-run with the repo root pinned:\n"
            f"    PYTHONPATH={REPO} python3 {__file__}"
        )
    return {
        "rfx_file": str(rfx_file),
        "under_repo_root": True,
        "repo_root": str(REPO),
        "rfx_version": getattr(rfx, "__version__", None),
    }


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------
# The dump appended to every copy of the case
# ---------------------------------------------------------------------------
_DUMP = '''
# --- appended by scripts/diagnostics/cv03_far_end_return/far_end_return.py ---
import json as _json                                          # noqa: E402
import numpy as _np                                           # noqa: E402

_carrier = int(_np.argmin(_np.abs(_np.asarray(meep_freqs) - fcen)))
_grid = res_rfx.grid
_plx = int(_grid.pad_x_lo)
_phx = int(_grid.pad_x_hi)
_nx = int(_line_full.shape[1])
# |Ez(x)| at the carrier bin across the FULL x extent, pads included.
_ez_abs = _np.abs(_line_full[_carrier, :])
_win_idx = _np.nonzero(_win)[0]

# Last interior cell on the hi face, and the outermost pad cell there.
_last_interior_hi = _nx - _phx - 1
_outermost_pad_hi = _nx - 1
_a_meas_hi = (float(_ez_abs[_outermost_pad_hi] / _ez_abs[_last_interior_hi])
              if _ez_abs[_last_interior_hi] > 0 else None)

_dump = {
    "exit_anchor_reached": True,
    "freqs_c_over_a": [float(v) for v in meep_freqs],
    "carrier_bin_index": _carrier,
    "carrier_bin_f_c_over_a": float(meep_freqs[_carrier]),
    "b_over_a": [float(v) for v in neff_bovera],
    "b_over_a_carrier_bin": float(neff_bovera[_carrier]),
    "n_eff_rfx": [float(v) for v in neff_rfx],
    "n_eff_analytic": [float(v) for v in neff_analytic],
    "two_wave_rel_residual": [float(v) for v in neff_resid],
    "band_mask": [bool(v) for v in band_mask],
    "T_rfx": [float(v) for v in T_rfx],
    "T_rfx_band_mean": float(T_rfx_band),
    "T_rfx_peak": float(T_rfx_peak),
    "n_steps": int(n_steps),
    "rfx_total_t_a_over_c0": float(rfx_total_t * C0 / a),
    # --- rig realisation, so no verdict has to trust a constant ------------
    "grid_shape": [int(v) for v in _grid.shape],
    "pad_x_lo": _plx, "pad_x_hi": _phx,
    "pad_y_lo": int(_grid.pad_y_lo), "pad_y_hi": int(_grid.pad_y_hi),
    "grid_interior_x": [int(_grid.interior[0].start), int(_grid.interior[0].stop)],
    "sx_a": float(sx),
    "cpml_n": int(cpml_n),
    "boundary": str(sim_rfx._boundary),
    "cpml_layers_realized": int(_grid.cpml_layers),
    # --- fit window, in all three coordinate systems -----------------------
    "fit_window_meep_a": [float(neff_x_lo_meep), float(neff_x_hi_meep)],
    "fit_window_rfx_m": [float(_fit_lo), float(_fit_hi)],
    "fit_window_grid_index": [int(_win_idx[0]), int(_win_idx[-1])],
    "fit_window_n_samples": int(_win.sum()),
    "x_full_m": [float(v) for v in _x_full],
    # --- the trace the headline is read off (R5) ---------------------------
    "ez_abs_carrier_full_x": [float(v) for v in _ez_abs],
    "last_interior_cell_index_hi": int(_last_interior_hi),
    "ez_abs_last_interior_cell_hi": float(_ez_abs[_last_interior_hi]),
    "ez_abs_outermost_pad_cell_hi": float(_ez_abs[_outermost_pad_hi]),
    "A_meas_hi_face": _a_meas_hi,
    "ez_abs_max_in_fit_window": float(_np.max(_ez_abs[_win])),
    "ez_abs_min_in_fit_window": float(_np.min(_ez_abs[_win])),
    "swr_fit_window": float(_np.max(_ez_abs[_win]) / _np.min(_ez_abs[_win])),
    # --- witnesses ---------------------------------------------------------
    "settling_db": (None if getattr(res_rfx, "settling_db", None) is None
                    else float(res_rfx.settling_db)),
    "settling_witness": getattr(res_rfx, "settling_witness", None),
    "preflight_stdout": _PREFLIGHT_STDOUT,
    "preflight_warnings": _PREFLIGHT_WARNINGS,
    "run_warnings": _RUN_WARNINGS,
    "pad_extension_forced_off": bool(globals().get("_PAD_EXT_OFF", False)),
}
with open(os.environ["CV03_831_JSON"], "w") as _fh:
    _json.dump(_dump, _fh, indent=2, default=str)
'''

# The case calls `sim_rfx.preflight(strict=False)` bare and runs bare. Replace
# both so the banner, the build warnings and the run warnings are captured
# INSIDE the recorder (cv01's driver documents why the recorder has to open
# before the build, not after the last add_flux_monitor).
_PREFLIGHT_CAPTURE_OLD = "sim_rfx.preflight(strict=False)"
_PREFLIGHT_CAPTURE_NEW = """import io as _io, warnings as _warnings                       # noqa: E402
from contextlib import redirect_stdout as _redirect_stdout    # noqa: E402
_pf_buf = _io.StringIO()
with _warnings.catch_warnings(record=True) as _pf_caught:
    _warnings.simplefilter("always")
    with _redirect_stdout(_pf_buf):
        sim_rfx.preflight(strict=False)
    _PREFLIGHT_WARNINGS = [str(w.message) for w in _pf_caught]
_PREFLIGHT_STDOUT = _pf_buf.getvalue()
print(_PREFLIGHT_STDOUT)"""

_RUN_CAPTURE_OLD = "res_rfx = sim_rfx.run(n_steps=n_steps, subpixel_smoothing=True)"
_RUN_CAPTURE_NEW = """with _warnings.catch_warnings(record=True) as _run_caught:
    _warnings.simplefilter("always")
    res_rfx = sim_rfx.run(n_steps=n_steps, subpixel_smoothing=True)
    _RUN_WARNINGS = [str(w.message) for w in _run_caught]"""

# A point probe at the fit-window centre, so settling_db has a record to score.
# Placed on the same anchor as the DFT plane probe so it lands before preflight.
_PROBE_OLD = "sim_rfx.add_dft_plane_probe(axis=\"y\", coordinate=OFFSET_Y * a,"
_PROBE_NEW = ("sim_rfx.add_probe(position=("
              "(0.5 * (neff_x_lo_meep + neff_x_hi_meep) + OFFSET_X) * a, "
              "OFFSET_Y * a, dx / 2), component=\"ez\")\n"
              "sim_rfx.add_dft_plane_probe(axis=\"y\", coordinate=OFFSET_Y * a,")

# Arm B: force include_cpml_pad_extension=False. `run()` does not expose the
# flag, so the copy wraps the builder. This patches the COPY's process only.
_PAD_EXT_OFF_OLD = "sim_rfx.add_material(\"wg\", eps_r=eps_wg)"
_PAD_EXT_OFF_NEW = """_PAD_EXT_OFF = True
from rfx.api import Simulation as _SimCls                     # noqa: E402
_orig_assemble = _SimCls._assemble_materials
def _assemble_no_pad_extension(self, grid, **kw):
    kw["include_cpml_pad_extension"] = False
    return _orig_assemble(self, grid, **kw)
_SimCls._assemble_materials = _assemble_no_pad_extension
sim_rfx.add_material("wg", eps_r=eps_wg)"""

_LAYERS_OLD = "cpml_n = int(dpml * resolution)"


def _layers_edit(n: int) -> tuple[str, str]:
    """Swap the absorber depth only.

    `cpml_n` is the ONLY consumer of `dpml` on the rfx side (the Meep part
    keeps its own `dpml`), and every rfx plane -- source, both flux monitors,
    the DFT plane and the fit window -- is written off `OFFSET_X` / `OFFSET_Y`,
    which come from `interior_x` / `interior_y`, not from `cpml_n`. So the
    absorber grows and nothing else moves. (`interior_y = sy - 2*dpml` is
    evaluated before this line, so it is unaffected.)
    """
    return (_LAYERS_OLD, f"cpml_n = {int(n)}  # arm edit: absorber depth")


_BOUNDARY_OLD = 'boundary=BoundarySpec.uniform("upml"),'


def _boundary_edit(fam: str) -> tuple[str, str]:
    return (_BOUNDARY_OLD, f'boundary=BoundarySpec.uniform("{fam}"),')


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def _sweep_arms():
    for fam in ("upml", "cpml"):
        for n in (20, 40, 60):
            edits = [_layers_edit(n)]
            if fam != "upml":
                edits.append(_boundary_edit(fam))
            yield (f"sweep_{fam}_{n}",
                   f"Arm A: cv03 recipe, boundary={fam}, absorber {n} cells",
                   edits)


def _noext_arms():
    for n in (20, 40, 60):
        yield (f"noext_upml_{n}",
               f"Arm B: cv03 recipe, upml, absorber {n} cells, "
               "include_cpml_pad_extension=False (bare facet at the seam)",
               [_layers_edit(n), (_PAD_EXT_OFF_OLD, _PAD_EXT_OFF_NEW)])


# --- Arm B-prime, section 7 of the pre-declaration ---------------------------
# Arm B is void: `run(subpixel_smoothing=True)` rebuilds the update's eps from
# `sim._geometry` with `background_eps=1.0`
# (`rfx/runners/uniform.py:266-271`), so `include_cpml_pad_extension` never
# reaches the solved array and the guide ends at the interior/pad seam in
# vacuum whatever that flag says. These two arms continue the guide through the
# pad in the array the update actually reads, by two routes that do not share
# the suspect quantity.
_BOX_OLD = ("sim_rfx.add(Box((0, wg_y_lo, 0), (domain_x, wg_y_hi, dx)), "
            "material=\"wg\")")
_BOX_NEW = ("sim_rfx.add(Box((-8 * a, wg_y_lo, 0), "
            "(domain_x + 8 * a, wg_y_hi, dx)), material=\"wg\")"
            "  # arm edit: guide continued past the deepest pad (60 cells = 6a)")

_SUBPIXEL_OLD = "res_rfx = sim_rfx.run(n_steps=n_steps, subpixel_smoothing=True)"
_SUBPIXEL_NEW = "res_rfx = sim_rfx.run(n_steps=n_steps, subpixel_smoothing=False)"


def _bprime_arms():
    for n in (20, 40, 60):
        yield (f"guidecont_upml_{n}",
               "Arm B1: cv03 recipe, upml, absorber {} cells, guide Box widened "
               "past the pad so the SUBPIXEL path (the one the update reads) "
               "carries eps_r=12 through the absorber".format(n),
               [_layers_edit(n), (_BOX_OLD, _BOX_NEW)])
    for n in (20, 40, 60):
        yield (f"nosub_upml_{n}",
               "Arm B2 cross-check: cv03 recipe, upml, absorber {} cells, "
               "subpixel_smoothing=False so the update reads base_materials and "
               "therefore the CPML pad extension. Confounded (smoothing also "
               "sets interface convergence order) -- read for SIGN only."
               .format(n),
               [_layers_edit(n), (_SUBPIXEL_OLD, _SUBPIXEL_NEW)])


_SX40_EDITS = [
    ("sx = 16.0", "sx = 40.0"),
    ("src_x_meep   = -7.0", "src_x_meep   = -19.0"),
    ("neff_x_lo_meep = flux_in_meep + 1.0     # -4.0",
     "neff_x_lo_meep = -16.0"),
    ("neff_x_hi_meep = flux_out_meep - 1.0    # +4.0",
     "neff_x_hi_meep = -8.0"),
]


def _tof_arms():
    yield ("tof_sx40_dft150_reflection_free",
           "Arm C0 control: 40a domain, DFT stopped at 150 a/c0, before the "
           "230.93 a/c0 round trip (oracle.tof_round_trip_a_over_c0"
           ".src_to_far_end_and_back_64a)",
           _SX40_EDITS + [("rfx_total_t = 400.0 * a / C0",
                           "rfx_total_t = 150.0 * a / C0")])
    yield ("tof_sx40_dft400_return_in_window",
           "Arm C0 control: the same 40a domain with the window long enough to "
           "admit the return (270.62 a/c0 to re-enter the fit window)",
           list(_SX40_EDITS))


# ---------------------------------------------------------------------------
def run_one(key: str, edits, workdir: str, json_path: str) -> tuple[int, str]:
    src = CASE.read_text()
    # Instruments first, then the arm's edits. The two sets of patterns do not
    # overlap, so the order does not change the produced source for any arm --
    # it only lets an arm edit target a line that lives INSIDE an instrument
    # block (arm B2 flips `subpixel_smoothing` on the run call that
    # `_RUN_CAPTURE_NEW` now wraps). Every replacement still asserts uniqueness.
    for old, new in ((_PROBE_OLD, _PROBE_NEW),
                     (_PREFLIGHT_CAPTURE_OLD, _PREFLIGHT_CAPTURE_NEW),
                     (_RUN_CAPTURE_OLD, _RUN_CAPTURE_NEW)):
        assert src.count(old) == 1, f"[{key}] instrument anchor missing: {old!r}"
        src = src.replace(old, new)
    for old, new in edits:
        assert src.count(old) == 1, f"[{key}] pattern is not unique: {old!r}"
        src = src.replace(old, new)
    assert src.count(_ANCHOR) == 1, f"[{key}] exit-code anchor not found"
    src = src.replace(_ANCHOR, _DUMP + "\n" + _ANCHOR)
    path = os.path.join(workdir, "case.py")
    with open(path, "w") as fh:
        fh.write(src)
    env = dict(os.environ, PYTHONPATH=str(REPO), MPLBACKEND="Agg",
               CV03_831_JSON=json_path)
    proc = subprocess.run([sys.executable, path], capture_output=True,
                          text=True, env=env, cwd=workdir)
    return proc.returncode, proc.stdout + proc.stderr


def run_fdtd_arms(arms) -> dict:
    out = {}
    for key, why, edits in arms:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(COMPARATORS, os.path.join(td, "comparators"))
            jp = os.path.join(td, "dump.json")
            rc, log = run_one(key, edits, td, jp)
            payload = json.load(open(jp)) if os.path.exists(jp) else None
        wall = round(time.time() - t0, 1)
        if payload is None:
            out[key] = {"why": why, "exit_code": rc, "status": "no dump",
                        "wall_s": wall, "log_tail": log[-4000:]}
            print(f"[{key}] exit={rc} NO DUMP ({wall}s)", flush=True)
            continue
        payload.update({"why": why, "exit_code": rc, "wall_s": wall,
                        "edits": [[o, n] for o, n in edits]})
        out[key] = payload
        print(f"[{key}] exit={rc} |B/A|={payload['b_over_a_carrier_bin']:.4f} "
              f"SWR={payload['swr_fit_window']:.3f} "
              f"T_band={payload['T_rfx_band_mean']:.4f} "
              f"A_meas_hi={payload['A_meas_hi_face']:.3e} "
              f"settling={payload['settling_db']} ({wall}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Arm C: the assembled absorber profile (no FDTD)
# ---------------------------------------------------------------------------
def run_profile_stage() -> dict:
    import jax.numpy as jnp  # noqa: F401  (import side effects match the case)
    from rfx import Simulation, Box
    from rfx.boundaries.spec import BoundarySpec
    from rfx.boundaries.cpml import _cpml_profile
    from rfx.boundaries.upml import _axis_sigma_E_H

    # cv03's own constants, read from the committed case rather than retyped.
    # Every one of these is a bare literal assignment in the committed case, so
    # ast.literal_eval is enough and a line that is not one (the docstring's
    # "fcen = 0.15, fwidth = 0.1" recipe summary, for instance) is skipped
    # rather than guessed at.
    import ast
    ns: dict = {}
    text = CASE.read_text()
    for name in ("eps_wg", "wg_width", "pad", "dpml", "resolution", "sx",
                 "a", "fcen", "df"):
        for line in text.splitlines():
            s = line.strip()
            if not s.startswith(f"{name} = "):
                continue
            rhs = s.split("=", 1)[1].split("#")[0].strip()
            try:
                ns[name] = ast.literal_eval(rhs)
            except (ValueError, SyntaxError):
                continue
            break
        assert name in ns, f"constant {name!r} not found in the committed case"
    C0 = 2.998e8
    a = ns["a"]
    dx = a / ns["resolution"]
    sy = 2 * (ns["pad"] + ns["dpml"] + ns["wg_width"] / 2)
    interior_x = ns["sx"]
    interior_y = sy - 2 * ns["dpml"]
    domain_x, domain_y = interior_x * a, interior_y * a
    off_y = interior_y / 2.0

    out = {"case_constants": {k: float(v) for k, v in ns.items()},
           "dx_m": dx, "n_eff_carrier_used": N_EFF_CARRIER,
           "eta_0": ETA_0, "arms": {}}

    for fam in ("upml", "cpml"):
        for n in (20, 40, 60):
            sim = Simulation(freq_max=0.25 * C0 / a,
                             domain=(domain_x, domain_y, dx), dx=dx,
                             boundary=BoundarySpec.uniform(fam),
                             cpml_layers=n, mode="2d_tmz")
            sim.add_material("wg", eps_r=ns["eps_wg"])
            wg_lo = (off_y - ns["wg_width"] / 2) * a
            wg_hi = (off_y + ns["wg_width"] / 2) * a
            sim.add(Box((0, wg_lo, 0), (domain_x, wg_hi, dx)), material="wg")
            grid = sim._build_grid()
            mats, *_ = sim._assemble_materials(grid)
            eps = np.asarray(mats.eps_r)
            jy = int(round(off_y * a / dx)) + int(grid.pad_y_lo)
            jy = min(max(jy, 0), eps.shape[1] - 1)
            eps_row = eps[:, jy, 0]

            entry = {
                "boundary": fam, "cpml_layers": n,
                "grid_shape": [int(v) for v in grid.shape],
                "pad_x_lo": int(grid.pad_x_lo), "pad_x_hi": int(grid.pad_x_hi),
                "pad_y_lo": int(grid.pad_y_lo), "pad_y_hi": int(grid.pad_y_hi),
                "interior_x": [int(grid.interior[0].start),
                               int(grid.interior[0].stop)],
                "centre_row_index_y": jy,
                "eps_r_centre_row": [float(v) for v in eps_row],
                "eps_r_centre_row_min": float(eps_row.min()),
                "eps_r_centre_row_n_at_eps_wg": int(
                    np.sum(np.isclose(eps_row, ns["eps_wg"], rtol=1e-6))),
                "dt_s": float(grid.dt),
            }

            if fam == "cpml":
                p = _cpml_profile(n, grid.dt, dx)
                sig = np.asarray(p.sigma, dtype=float)
                entry.update({
                    "profile_kind": "cpml_1d_lo_face",
                    "sigma": [float(v) for v in sig],
                    "kappa": [float(v) for v in np.asarray(p.kappa)],
                    "alpha": [float(v) for v in np.asarray(p.alpha)],
                    "b": [float(v) for v in np.asarray(p.b)],
                    "c": [float(v) for v in np.asarray(p.c)],
                    "sigma_max": float(sig.max()),
                    "int_sigma_dx": float(sig.sum() * dx),
                })
            else:
                sE, sH = _axis_sigma_E_H(grid, "x")
                sE = np.asarray(sE)[:, jy, 0].astype(float)
                sH = np.asarray(sH)[:, jy, 0].astype(float)
                phx = int(grid.pad_x_hi)
                sig_hi = sE[-phx:] if phx > 0 else np.zeros(0)
                entry.update({
                    "profile_kind": "upml_assembled_centre_row",
                    "sigma_E_centre_row": [float(v) for v in sE],
                    "sigma_H_centre_row": [float(v) for v in sH],
                    "sigma_max": float(sE.max()),
                    "int_sigma_dx": float(sig_hi.sum() * dx),
                })

            isd = entry["int_sigma_dx"]
            entry["designed_one_way_amplitude_attenuation"] = float(
                math.exp(-N_EFF_CARRIER * ETA_0 * isd))
            entry["designed_round_trip_amplitude"] = float(
                math.exp(-2.0 * N_EFF_CARRIER * ETA_0 * isd))
            entry["vacuum_designed_round_trip_R"] = float(
                math.exp(-2.0 * ETA_0 * isd))
            out["arms"][f"{fam}_{n}"] = entry
            print(f"[profile {fam} {n}] nx={grid.shape[0]} "
                  f"int_sigma_dx={isd:.6g} "
                  f"one-way={entry['designed_one_way_amplitude_attenuation']:.3e} "
                  f"eps_row_min={eps_row.min():.3g}", flush=True)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("all", "profile", "tof", "sweep", "noext",
                             "bprime"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    prov_check = _assert_rfx_is_this_repo()
    out = {
        "schema": "issue831-cv03-far-end-return-v1",
        "issue": 831,
        "predeclaration":
            "docs/design_notes/issue831_far_end_return_predeclaration.md",
        "runs_fdtd": args.stage != "profile",
        "provenance": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "case_sha": _git("hash-object", str(CASE)),
            "python": sys.version.split()[0],
            "argv": sys.argv,
        },
        "rfx_provenance_check": prov_check,
        "stages": {},
    }

    if args.stage in ("all", "profile"):
        out["stages"]["profile"] = run_profile_stage()
    if args.stage in ("all", "tof"):
        out["stages"]["tof"] = run_fdtd_arms(_tof_arms())
    if args.stage in ("all", "sweep"):
        out["stages"]["sweep"] = run_fdtd_arms(_sweep_arms())
    if args.stage in ("all", "noext"):
        out["stages"]["noext"] = run_fdtd_arms(_noext_arms())
    if args.stage in ("all", "bprime"):
        out["stages"]["bprime"] = run_fdtd_arms(_bprime_arms())

    dest = pathlib.Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
        fh.write("\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
