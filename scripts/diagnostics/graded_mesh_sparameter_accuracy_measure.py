#!/usr/bin/env python3
"""Graded-mesh S-parameter accuracy — the WR-90 control measurement driver.

Runs the pre-declaration
``docs/design_notes/graded_mesh_sparameter_accuracy_predeclaration.md``.
Every window, profile, expectation and decision rule lives in that note and
was committed before this driver produced a number.

ONE declared geometry (the committed WR-90 chain battery,
``tests/_waveguide_chain_battery_fixture.py``) realized on five meshes at the
battery's ``mid`` rung, all on the ``normalize="flux"`` lane in float32:

* ``A``  uniform — the battery's own build at dx = 1.27 mm.
* ``B``  z multi-band, 3 fine bands (0.635 mm) / 2 coarse bands (0.889 mm),
         every adjacent ratio exactly 1.4 — inside the #785 / support-matrix
         envelope (z axis, ratio <= 1.4, <= 3 fine bands, PEC z faces so no
         grading touches an absorber).
* ``C``  z uniform at B's finest cell (0.635 mm) — the cost / dt control:
         same minimum cell as B, therefore the same dt.
* ``D1`` z multi-band with abrupt ratio-2.0 steps — out-of-envelope RATIO on
         the same axis.
* ``D2`` x band of 2.54 mm cells (abrupt ratio 2.0) between the probe planes
         — out-of-envelope ratio on the PROPAGATION axis; thru only.
* ``E``  x band of 1.778 mm cells (ratio 1.4) on the propagation axis —
         in-envelope RATIO on an axis the support matrix does NOT cover
         (in-plane grading is uncovered; in-plane threshold is 1.3). Reference
         arm, thru only, no promotion claim attaches to it.

The graded arms cannot run ``normalize=False``: ``compute_waveguide_s_matrix``
refuses a non-uniform mesh on that lane (#811, PR #828). The flux lane is the
only NU-capable S lane and it is the lane the v1.8 chain closure declares for
the NU sub-lane, so every arm here — uniform included — is read on it.

Usage (from the worktree root; the rfx import must resolve to this tree)::

    PYTHONPATH=. python scripts/diagnostics/graded_mesh_sparameter_accuracy_measure.py \
        --out-dir <run-dir> --stages arms
    PYTHONPATH=. python scripts/diagnostics/graded_mesh_sparameter_accuracy_measure.py \
        --out-dir <run-dir> --stages assemble \
        --artifact-out tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import rfx  # noqa: E402
from rfx import Simulation  # noqa: E402

from tests import _waveguide_chain_battery_fixture as F  # noqa: E402
from tests import _waveguide_chain_battery_gates as G  # noqa: E402

SCHEMA = "rfx.graded_mesh_sparameter_accuracy"
SCHEMA_VERSION = 1
PREDECLARATION = "docs/design_notes/graded_mesh_sparameter_accuracy_predeclaration.md"
DRIVER = "scripts/diagnostics/graded_mesh_sparameter_accuracy_measure.py"
REFERENCE_ARTIFACT = "tests/fixtures/waveguide_chain_battery/fixture_v18_close.json"
# The battery's LIVE artifact (schema 4, realized-PEC device operator). Arm A
# is compared with BOTH: the schema-3 run is the one the v1.8 chain closure
# quotes, the schema-4 run is the one measured under the operator today's main
# ships. See the results note, correction C1.
LIVE_ARTIFACT = ("tests/fixtures/waveguide_chain_battery/"
                 "fixture_931_realized_pec_forward2_run369367259427.json")

RUNG = "mid"
DX = G.RUNG_DX[RUNG]                 # 1.27 mm
DZ_FINE = 0.000635                   # B / C / D1 finest z cell
CPML_LAYERS = 34                     # the committed mid-rung absorber; pinned so
                                     # every arm carries the SAME absorber
LANE = "flux"
PRECISION = "float32"

# --- the committed F-S2 per-transition reflection budget (#780 / PR #785) ---
# validation/research/multiband_nu/results/predeclared_windows.json, read as
# numbers from the artifact, never retyped from the note's prose.
FS2_ARTIFACT = "validation/research/multiband_nu/results/predeclared_windows.json"


def _fs2() -> dict:
    d = json.loads((REPO / FS2_ARTIFACT).read_text())
    return d


# --- realized meshes (explicit vectors; #785 W1--W5 idiom) -----------------
# Every vector is checked against its declared sum and ratio cap at import.
PROFILES = {
    # (axis, cells) — the physical-domain profile, CPML pad excluded.
    "A": (None, None),
    "B": ("z", np.array([DZ_FINE] * 3 + [0.000889] * 2 + [DZ_FINE] * 3
                        + [0.000889] * 3 + [DZ_FINE] * 3)),
    "C": ("z", np.full(16, DZ_FINE)),
    "D1": ("z", np.array([DZ_FINE] * 2 + [0.00127] * 2 + [DZ_FINE] * 2
                         + [0.00127] * 3 + [DZ_FINE] * 2)),
    "D2": ("x", np.array([DX] * 40 + [2 * DX] * 10 + [DX] * 36)),
    "E": ("x", np.array([DX] * 41 + [0.001778] * 10 + [DX] * 41)),
}
ARM_DUTS = {
    "A": ("thru", "pec_short", "slab"),
    "B": ("thru", "pec_short", "slab"),
    "C": ("thru", "pec_short", "slab"),
    "D1": ("thru", "pec_short", "slab"),
    "D2": ("thru",),
    "E": ("thru",),
}
ARM_ORDER = ("A", "B", "C", "D1", "D2", "E")


def _check_profiles() -> dict:
    """Declared-vector invariants, asserted before any solve."""
    out = {}
    for arm, (axis, prof) in PROFILES.items():
        if prof is None:
            out[arm] = {"axis": None, "n_cells": None}
            continue
        total = {"x": F.DOMAIN_X_M, "y": F.A_M, "z": F.B_M}[axis]
        s = float(np.sum(prof))
        if abs(s - total) > 1e-12:
            raise AssertionError(f"arm {arm}: profile sums to {s!r}, declared {total!r}")
        ratios = prof[1:] / prof[:-1]
        max_ratio = float(np.max(np.maximum(ratios, 1.0 / ratios)))
        if axis in ("x", "y") and (abs(prof[0] - DX) > 1e-15 or abs(prof[-1] - DX) > 1e-15):
            raise AssertionError(f"arm {arm}: in-plane profile end cells must equal dx")
        out[arm] = {
            "axis": axis,
            "n_cells": int(prof.size),
            "cells_m": [float(v) for v in prof],
            "sum_m": s,
            "sum_error_m": s - total,
            "max_adjacent_ratio": max_ratio,
            "min_cell_m": float(prof.min()),
            "max_cell_m": float(prof.max()),
            "n_transitions": int(np.sum(np.abs(ratios - 1.0) > 1e-12)),
        }
    return out


PROFILE_FACTS = _check_profiles()


# --- build ------------------------------------------------------------------

def build(arm: str, dut: str) -> Simulation:
    """The battery's declared geometry, realized on ``arm``'s mesh.

    Everything but the mesh comes from the committed builder constants; the
    absorber is pinned to the committed mid-rung count so the arms differ in
    the mesh and in nothing else.
    """
    axis, prof = PROFILES[arm]
    kw = {}
    if prof is not None:
        kw[f"d{axis}_profile"] = prof
    sim = Simulation(
        freq_max=F.FREQ_MAX_HZ,
        domain=(F.DOMAIN_X_M, F.A_M, F.B_M),
        dx=DX,
        boundary=F._boundary(),
        cpml_layers=CPML_LAYERS,
        precision=PRECISION,
        **kw,
    )
    F._add_dut(sim, dut)
    freqs = jnp.asarray(F.FREQS)
    ref_off = int(round(F.D_REF_M / DX))
    prb_off = int(round(F.D_PROBE_M / DX))
    assert abs(ref_off * DX - F.D_REF_M) < 1e-15
    assert abs(prb_off * DX - F.D_PROBE_M) < 1e-15
    sim.add_waveguide_port(
        F.PORT_LEFT_X_M, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=F.F0_HZ, bandwidth=F.BANDWIDTH, ref_offset=ref_off,
        probe_offset=prb_off, name=F.PORT_NAMES[0],
        reference_plane=F.REF_LEFT_DEFAULT_M)
    sim.add_waveguide_port(
        F.PORT_RIGHT_X_M, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=F.F0_HZ, bandwidth=F.BANDWIDTH, ref_offset=ref_off,
        probe_offset=prb_off, name=F.PORT_NAMES[1],
        reference_plane=F.REF_RIGHT_DEFAULT_M)
    return sim


def is_nu(sim) -> bool:
    return (sim._dz_profile is not None or sim._dx_profile is not None
            or sim._dy_profile is not None)


def build_grid(sim):
    return sim._build_nonuniform_grid() if is_nu(sim) else sim._build_grid()


def n_steps(grid) -> int:
    """Record length in steps: ``ceil(num_periods / freq_max / dt)``.

    ``Grid.num_timesteps`` is the uniform lane's own implementation of that
    formula; ``NonUniformGrid`` has no such method, so the formula is written
    once here and CHECKED against ``num_timesteps`` whenever the grid has it.
    """
    n = int(math.ceil(F.NUM_PERIODS / F.FREQ_MAX_HZ / float(grid.dt)))
    ref = getattr(grid, "num_timesteps", None)
    if ref is not None:
        got = int(ref(F.NUM_PERIODS))
        if got != n:
            raise AssertionError(
                f"n_steps formula {n} disagrees with Grid.num_timesteps {got}")
    return n


# --- realized-mesh + rasterization record (#325 class) ----------------------

def realized_mesh(sim, grid) -> dict:
    """The cell sizes the solver actually got, per axis, physical domain only,
    plus the node coordinates of every DUT face the geometry declares."""
    def axis_cells(name, total):
        if not is_nu(sim):
            return np.full(int(round(total / DX)), float(DX))
        arr = np.asarray(
            {"x": grid.dx_arr_f64, "y": grid.dy_arr_f64, "z": grid.dz_f64}[name],
            dtype=float)
        lo = int({"x": grid.pad_x_lo, "y": grid.pad_y_lo, "z": grid.pad_z_lo}[name])
        hi = int({"x": grid.pad_x_hi, "y": grid.pad_y_hi, "z": grid.pad_z_hi}[name])
        core = arr[lo:arr.size - hi] if hi else arr[lo:]
        # The NU builder appends ONE trailing bounding node (#562). Which of
        # the two slices is the physical domain is decided by the declared
        # extent, not by an assumed layout.
        if (abs(float(core.sum()) - total) > 1e-12
                and abs(float(core[:-1].sum()) - total) <= 1e-12):
            core = core[:-1]
        if abs(float(core.sum()) - total) > 1e-12:
            raise AssertionError(
                f"realized {name} cells sum to {core.sum()!r}, declared {total!r} "
                f"— the realized extent is not the drawn extent (#325 class)")
        return core

    rec = {}
    for name, total in (("x", F.DOMAIN_X_M), ("y", F.A_M), ("z", F.B_M)):
        cells = axis_cells(name, total)
        rec[name] = {
            "n_cells": int(cells.size),
            "cells_m": [float(v) for v in cells],
            "sum_m": float(cells.sum()),
            "sum_error_m": float(cells.sum() - total),
            "min_cell_m": float(cells.min()),
            "max_cell_m": float(cells.max()),
        }
    return rec


def dut_rasterization(sim, grid, dut: str) -> dict:
    """Cell counts the DUT actually occupies, read off the assembled material
    arrays the solver uses — the #325-class check (a graded mesh must not move
    a declared stack onto the wrong cells)."""
    assembled = (sim._assemble_materials_nu(grid) if is_nu(sim)
                 else sim._assemble_materials(grid))
    mats, pec_mask = assembled[0], assembled[3]
    eps = np.asarray(mats.eps_r, dtype=float)
    sig = np.asarray(mats.sigma, dtype=float)
    out = {"dut": dut, "grid_shape": [int(v) for v in eps.shape]}
    if dut == "slab":
        m = eps > 2.0
        out["material"] = "eps_r=4 slab"
    elif dut == "pec_short":
        # sigma = 1e10 is above Simulation._PEC_SIGMA_THRESHOLD (1e6), so the
        # block is carried in pec_mask, not in the sigma array (#931 lattice
        # ownership). Reading sigma here would report an empty conductor.
        if pec_mask is None:
            raise AssertionError("pec_short assembled with no pec_mask (#369 class)")
        m = np.asarray(pec_mask) > 0
        out["material"] = "pec_mask (sigma=1e10 > _PEC_SIGMA_THRESHOLD)"
    else:
        out["material"] = None
        out["n_cells"] = 0
        out["eps_max"] = float(eps.max())
        out["sigma_max"] = float(sig.max())
        return out
    out["n_cells"] = int(m.sum())
    for ax, name in enumerate("xyz"):
        idx = np.where(m.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        out[f"{name}_run"] = [int(idx.min()), int(idx.max())] if idx.size else None
        out[f"{name}_count"] = int(idx.size)
    return out


def guide_span_check(sim, grid) -> dict:
    """The realized transverse extent of the guide, from the realized profile.

    The TE10 cutoff is set by ``a`` (the y span). A graded mesh that missed the
    declared extent would move the cutoff and every S-parameter with it.
    """
    rm = realized_mesh(sim, grid)
    return {
        "a_realized_m": rm["y"]["sum_m"],
        "a_declared_m": F.A_M,
        "a_error_m": rm["y"]["sum_m"] - F.A_M,
        "b_realized_m": rm["z"]["sum_m"],
        "b_declared_m": F.B_M,
        "b_error_m": rm["z"]["sum_m"] - F.B_M,
        "x_realized_m": rm["x"]["sum_m"],
        "x_declared_m": F.DOMAIN_X_M,
        "x_error_m": rm["x"]["sum_m"] - F.DOMAIN_X_M,
    }


# --- run --------------------------------------------------------------------

def preflight_findings(sim) -> list[dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    return [{"code": getattr(i, "code", "uncoded"),
             "severity": getattr(i, "severity", "warning"),
             "message": str(i)} for i in report]


def _dedupe_warnings(wlist) -> list[dict]:
    seen: dict[str, int] = {}
    for w in wlist:
        key = f"{w.category.__name__}: {w.message}"
        seen[key] = seen.get(key, 0) + 1
    return [{"message": k, "count": n} for k, n in seen.items()]


def run_arm(arm: str, dut: str, prov: dict) -> dict:
    t_build = time.time()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        sim = build(arm, dut)
        grid = build_grid(sim)
        mesh = realized_mesh(sim, grid)
        raster = dut_rasterization(sim, grid, dut)
        spans = guide_span_check(sim, grid)
        findings = preflight_findings(sim)
        t0 = time.time()
        res = sim.compute_waveguide_s_matrix(num_periods=F.NUM_PERIODS, normalize=LANE)
        S = np.asarray(res.s_params).astype(np.complex128)
        wall = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    rec = {
        "arm": arm, "dut": dut, "rung": RUNG, "lane": LANE,
        "dx_m": DX, "cpml_layers": CPML_LAYERS, "precision": PRECISION,
        "profile_declared": PROFILE_FACTS[arm],
        "realized_mesh": mesh,
        "dut_rasterization": raster,
        "guide_span_check": spans,
        "grid_shape": [int(v) for v in grid.shape],
        "dt_s": float(grid.dt),
        "n_steps": n_steps(grid),
        "num_periods": F.NUM_PERIODS,
        "n_cells": int(np.prod([int(v) for v in grid.shape])),
        "preflight": findings,
        "warnings": _dedupe_warnings(wl),
        "settling_db": {name: float(v) for name, v in
                        zip(F.PORT_NAMES, np.asarray(res.settling_db, dtype=float))},
        "s_params": G.s_to_json(S),
        "reference_planes_m": [float(x) for x in np.asarray(res.reference_planes)],
        "s21_phase_residual_deg_rms": (None if res.s21_phase_residual_deg_rms is None
                                       else float(res.s21_phase_residual_deg_rms)),
        "s21_phase_residual_meta": _jsonable(res.s21_phase_residual_meta),
        "wall_time_s": wall, "build_time_s": t0 - t_build,
        "peak_rss_bytes": int(rss),
        "provenance": prov,
        **G.cell_metrics(S),
    }
    rec["s21_phase_deg"] = [float(v) for v in np.degrees(np.unwrap(np.angle(S[1, 0])))]
    if dut == "pec_short":
        rec["referee_pec_short"] = G.referee_pec_short(S, F.FREQS)
    if dut == "slab":
        rec["referee_slab_airy"] = G.referee_slab_airy(S, F.FREQS)
    return rec


def _jsonable(o):
    if o is None:
        return None
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    return str(o)


# --- provenance -------------------------------------------------------------

def provenance(args) -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=str(REPO),
                                           text=True).strip()
        except Exception:  # noqa: BLE001
            return "unknown"
    return {
        "commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "rfx_file": rfx.__file__,
        "rfx_version": getattr(rfx, "__version__", "unknown"),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_default_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "precision": PRECISION,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "run_lane": args.run_lane,
        "run_id": args.run_id,
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "argv": sys.argv,
        "predeclaration": PREDECLARATION,
        "driver": DRIVER,
        "reference_artifact": REFERENCE_ARTIFACT,
        "live_artifact": LIVE_ARTIFACT,
        "fs2_artifact": FS2_ARTIFACT,
    }


# --- stages -----------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, flush=True)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False))


def arm_path(out_dir: Path, arm: str, dut: str) -> Path:
    return out_dir / f"arm__{arm}__{dut}.json"


def stage_arms(args, out_dir: Path, prov: dict) -> None:
    for arm in ARM_ORDER:
        if args.arms and arm not in args.arms:
            continue
        for dut in ARM_DUTS[arm]:
            path = arm_path(out_dir, arm, dut)
            if path.exists() and not args.overwrite:
                _log(f"skip existing {path.name}")
                continue
            _log(f"arm {arm} dut {dut} ...")
            rec = run_arm(arm, dut, prov)
            _write(path, rec)
            _log(f"  wrote {path.name}: dt={rec['dt_s']:.6e} shape={rec['grid_shape']} "
                 f"colpow={rec['column_power_max']:.7f} "
                 f"phi_res={rec['s21_phase_residual_deg_rms']} "
                 f"settling={rec['settling_db']} wall={rec['wall_time_s']:.1f}s")


# --- allowance --------------------------------------------------------------

def allowance_table() -> dict:
    """The derived allowance, per bin, from the committed F-S2 artifact and
    this fixture's own cells per guided wavelength. No FDTD, no measured input.
    """
    fs2 = _fs2()
    n_ref = float(fs2["cells_per_guided_wavelength_fine"])
    freqs = np.asarray(F.FREQS, dtype=float)
    beta_c = G.beta_continuous(freqs)
    lam_g = 2.0 * np.pi / beta_c

    def r1(ratio: str, d_fine: float) -> np.ndarray:
        r_ref = float(fs2["fs2"][ratio]["abrupt"]["R_model"])
        return r_ref * (n_ref * d_fine / lam_g) ** 2

    out = {
        "fs2_artifact": FS2_ARTIFACT,
        "n_ref_cells_per_guided_wavelength": n_ref,
        "lambda_g_m": [float(v) for v in lam_g],
        "floor_amplitude": 1.0e-4,
        "floor_provenance": (
            "tests/oracle/test_waveguide_chain_battery_v18_close.py live cell "
            "comparison gate (CPU vs the committed GPU artifact, 1e-4)"),
    }
    for arm in ("B", "D1"):
        axis, prof = PROFILES[arm]
        ratio = "1.4" if arm == "B" else "2.0"
        n_tr = PROFILE_FACTS[arm]["n_transitions"]
        a_tr = n_tr * r1(ratio, PROFILE_FACTS[arm]["min_cell_m"])
        out[arm] = {
            "axis": axis, "ratio": ratio, "n_transitions": n_tr,
            "r1_per_transition": [float(v) for v in r1(ratio, PROFILE_FACTS[arm]["min_cell_m"])],
            "a_transition": [float(v) for v in a_tr],
            "a_total_amplitude": [float(v) for v in (a_tr + out["floor_amplitude"])],
            "a_total_phase_deg_at_unit_mag": [
                float(v) for v in np.degrees(np.arcsin(np.clip(a_tr + out["floor_amplitude"], 0, 1)))],
            "a_total_phase_deg_at_mag_floor_0p30": [
                float(v) for v in np.degrees(
                    np.arcsin(np.clip((a_tr + out["floor_amplitude"]) / 0.30, 0, 1)))],
        }
    for arm in ("D2", "E"):
        axis, prof = PROFILES[arm]
        ratio = "2.0" if arm == "D2" else "1.4"
        n_tr = PROFILE_FACTS[arm]["n_transitions"]
        a_tr = n_tr * r1(ratio, PROFILE_FACTS[arm]["min_cell_m"])
        out[arm] = {
            "axis": axis, "ratio": ratio, "n_transitions": n_tr,
            "a_transition": [float(v) for v in a_tr],
            "a_total_amplitude": [float(v) for v in (a_tr + out["floor_amplitude"])],
        }
    return out


def dispersion_predictions() -> dict:
    """The Yee-dispersion phase terms, per bin, computed from the realized
    profiles — no FDTD. These are the terms the F-S2 reflection budget does
    NOT bound: a coarse cell changes the phase a wave accumulates in it.
    """
    freqs = np.asarray(F.FREQS, dtype=float)
    dt_a = 0.99 / (G.C0_LOCAL * math.sqrt(3.0 / DX ** 2))
    dt_fine = 0.99 / (G.C0_LOCAL * math.sqrt(2.0 / DX ** 2 + 1.0 / DZ_FINE ** 2))
    l_ref = F.REF_RIGHT_DEFAULT_M - F.REF_LEFT_DEFAULT_M
    b_a = G.beta_yee(freqs, dt_a, DX)
    out = {
        "dt_uniform_s": dt_a, "dt_fine_min_cell_s": dt_fine,
        "reference_plane_separation_m": l_ref,
    }
    # C (and B, D1): same x cells, smaller dt
    b_c = G.beta_yee(freqs, dt_fine, DX)
    d_c = np.degrees((b_c - b_a) * l_ref)
    out["dt_term_deg_per_bin"] = [float(v) for v in d_c]
    out["dt_term_deg_rms"] = float(np.sqrt((d_c ** 2).mean()))
    # D2 / E: a band of coarse cells on the propagation axis
    for arm, coarse, span in (("D2", 2 * DX, 10 * 2 * DX), ("E", 0.001778, 10 * 0.001778)):
        b_k = G.beta_yee(freqs, dt_a, coarse)
        d_k = np.degrees((b_k - b_a) * span)
        out[arm] = {
            "coarse_cell_m": coarse, "band_span_m": span,
            "excess_phase_deg_per_bin": [float(v) for v in d_k],
            "excess_phase_deg_rms": float(np.sqrt((d_k ** 2).mean())),
        }
    return out


# --- assemble ---------------------------------------------------------------

def stage_assemble(args, out_dir: Path, prov: dict) -> None:
    arms = {}
    for arm in ARM_ORDER:
        for dut in ARM_DUTS[arm]:
            p = arm_path(out_dir, arm, dut)
            if p.exists():
                arms[f"{arm}|{dut}"] = json.loads(p.read_text())
    if not arms:
        raise SystemExit(f"no arm records in {out_dir}")

    committed = {}
    for label, path in (("schema3", REFERENCE_ARTIFACT), ("schema4", LIVE_ARTIFACT)):
        ref = json.loads((REPO / path).read_text())
        for c in ref["cells"]:
            if c["rung"] == RUNG and c["lane"] == LANE:
                committed[(label, c["dut"])] = c

    art = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "predeclaration": PREDECLARATION,
        "driver": DRIVER,
        "reference_artifact": REFERENCE_ARTIFACT,
        "live_artifact": LIVE_ARTIFACT,
        "reference_note": (
            "analytic references of the committed WR-90 chain battery: PEC-short "
            "|S11| = 1, eps_r = 4 slab vs Airy (magnitude and phase), thru column "
            "power and reciprocity, S21 phase vs -beta*L. Every arm is judged "
            "against the SAME analytic reference; the committed artifact's mid|flux "
            "cells enter only as the provenance check on arm A."),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "provenance": prov,
        "rung": RUNG, "lane": LANE, "precision": PRECISION,
        "profiles": PROFILE_FACTS,
        "allowance": allowance_table(),
        "dispersion_predictions": dispersion_predictions(),
        "arms": arms,
    }

    # arm A vs the committed artifact — the provenance check
    prov_check = {}
    for (label, dut), c in committed.items():
        key = f"A|{dut}"
        if key not in arms:
            continue
        S_new = G.s_from_json(arms[key]["s_params"])
        S_old = G.s_from_json(c["s_params"])
        d = np.abs(S_new - S_old)
        prov_check[f"{label}|{dut}"] = {
            "artifact": REFERENCE_ARTIFACT if label == "schema3" else LIVE_ARTIFACT,
            "max_abs_diff": float(d.max()),
            "max_abs_diff_per_entry": {
                k: float(np.max(d[i, j])) for k, (i, j) in
                (("S11", (0, 0)), ("S21", (1, 0)), ("S12", (0, 1)), ("S22", (1, 1)))},
            "gate": 1.0e-4,
            "gate_provenance": "tests/oracle/test_waveguide_chain_battery_v18_close.py live cell gate",
            "pass": bool(d.max() <= 1.0e-4),
        }
    art["arm_a_vs_committed_artifact"] = prov_check

    # pairwise S deltas
    deltas = {}
    for dut in ("thru", "pec_short", "slab"):
        for a, b in (("B", "C"), ("B", "A"), ("C", "A"), ("D1", "C"), ("D2", "A"), ("E", "A")):
            ka, kb = f"{a}|{dut}", f"{b}|{dut}"
            if ka not in arms or kb not in arms:
                continue
            Sa, Sb = G.s_from_json(arms[ka]["s_params"]), G.s_from_json(arms[kb]["s_params"])
            d = np.abs(Sa - Sb)
            ph = G.wrap_deg(np.degrees(np.angle(Sa[1, 0])) - np.degrees(np.angle(Sb[1, 0])))
            deltas[f"{a}-{b}|{dut}"] = {
                "max_abs_delta_S": float(d.max()),
                "max_abs_delta_S_per_entry": {
                    k: float(np.max(d[i, j])) for k, (i, j) in
                    (("S11", (0, 0)), ("S21", (1, 0)), ("S12", (0, 1)), ("S22", (1, 1)))},
                "s21_phase_delta_deg_per_bin": [float(v) for v in ph],
                "s21_phase_delta_deg_rms": float(np.sqrt((ph ** 2).mean())),
            }
    art["pairwise_deltas"] = deltas

    out = Path(args.artifact_out) if args.artifact_out else (
        REPO / "tests" / "fixtures" / "graded_mesh_sparameter_accuracy" / "wr90_control.json")
    _write(out, art)
    _log(f"wrote {out}")


# --- verdicts (pure post-processing of the artifact; §4 of the note) --------

def _dev_per_bin(arms: dict, key: str) -> dict:
    """Deviation from the ANALYTIC reference, per gate, per bin."""
    r = arms[key]
    S = G.s_from_json(r["s_params"])
    out = {}
    dut = r["dut"]
    cp = np.asarray(r["column_power_per_bin"], dtype=float)
    out["column_power"] = np.abs(cp - 1.0).max(axis=0)
    out["reciprocity_complex"] = np.asarray(r["reciprocity_complex_per_bin"], dtype=float)
    if dut == "pec_short":
        out["pec_short_s11_mag"] = np.abs(np.abs(S[0, 0]) - 1.0)
        out["pec_short_s22_mag"] = np.abs(np.abs(S[1, 1]) - 1.0)
    if dut == "slab":
        ref = r["referee_slab_airy"]
        out["slab_airy_s11_mag"] = np.abs(np.asarray(ref["s11_mag_abs_diff_per_bin"], dtype=float))
        out["slab_airy_s21_mag"] = np.abs(np.asarray(ref["s21_mag_abs_diff_per_bin"], dtype=float))
        out["slab_airy_s11_phase_deg"] = np.abs(np.asarray(ref["s11_phase_diff_deg_per_bin"], dtype=float))
        out["slab_airy_s21_phase_deg"] = np.abs(np.asarray(ref["s21_phase_diff_deg_per_bin"], dtype=float))
    if dut == "thru":
        # wrap(angle(S21) + beta*L), the shipped witness's own formula
        # (rfx/sparams/_common.py::s21_phase_residual_deg_rms), rebuilt per
        # bin from the meta the result carries. The reconstruction is CHECKED
        # against the stored rms below, so a formula drift is not silent.
        meta = r.get("s21_phase_residual_meta") or {}
        beta = G.beta_yee_fc(F.FREQS, float(meta["f_cutoff_hz"]),
                             float(r["dt_s"]), float(r["dx_m"]))
        resid = G.wrap_deg(np.degrees(np.angle(S[1, 0]) + beta * float(meta["L_m"])))
        rms = float(np.sqrt((resid ** 2).mean()))
        stored = float(r["s21_phase_residual_deg_rms"])
        if abs(rms - stored) > 1e-6 * max(abs(stored), 1.0):
            raise AssertionError(
                f"{key}: reconstructed S21 phase residual rms {rms} does not "
                f"reproduce the shipped {stored}")
        out["thru_s21_phase_residual_deg"] = np.abs(resid)
    return out


PHASE_GATES = ("slab_airy_s11_phase_deg", "slab_airy_s21_phase_deg",
               "thru_s21_phase_residual_deg")


def stage_verdicts(args, out_dir: Path) -> None:
    path = Path(args.artifact_out) if args.artifact_out else (
        REPO / "tests" / "fixtures" / "graded_mesh_sparameter_accuracy" / "wr90_control.json")
    art = json.loads(path.read_text())
    arms = art["arms"]
    allow = art["allowance"]
    disp = art["dispersion_predictions"]
    a_b_amp = np.asarray(allow["B"]["a_total_amplitude"], dtype=float)
    a_b_ph = np.asarray(allow["B"]["a_total_phase_deg_at_unit_mag"], dtype=float)
    a_dt = np.abs(np.asarray(disp["dt_term_deg_per_bin"], dtype=float))

    verdicts = {}
    for dut in ("thru", "pec_short", "slab"):
        if f"B|{dut}" not in arms:
            continue
        dB = _dev_per_bin(arms, f"B|{dut}")
        dC = _dev_per_bin(arms, f"C|{dut}")
        dA = _dev_per_bin(arms, f"A|{dut}")
        for gate in dB:
            allowance = a_b_ph if gate in PHASE_GATES else a_b_amp
            dt_term = a_dt if gate in PHASE_GATES else np.zeros_like(a_dt)
            r1_margin = dC[gate] + allowance - dB[gate]
            r2_margin = dA[gate] + allowance + dt_term - dB[gate]
            verdicts[f"R-1|{dut}|{gate}"] = {
                "rule": "dev_B <= dev_C + A_B, per bin",
                "dev_B": [float(v) for v in dB[gate]],
                "dev_C": [float(v) for v in dC[gate]],
                "allowance": [float(v) for v in allowance],
                "margin_per_bin": [float(v) for v in r1_margin],
                "worst_margin": float(r1_margin.min()),
                "worst_bin_hz": float(np.asarray(F.FREQS)[int(np.argmin(r1_margin))]),
                "fired": bool(r1_margin.min() < 0.0),
            }
            verdicts[f"R-2|{dut}|{gate}"] = {
                "rule": "dev_B <= dev_A + A_B + A_dt, per bin",
                "dev_A": [float(v) for v in dA[gate]],
                "margin_per_bin": [float(v) for v in r2_margin],
                "worst_margin": float(r2_margin.min()),
                "worst_bin_hz": float(np.asarray(F.FREQS)[int(np.argmin(r2_margin))]),
                "fired": bool(r2_margin.min() < 0.0),
            }

    # W-BC and F-Z: |S_B - S_C| and |S_D1 - S_C|
    for tag, a, b in (("W-BC", "B", "C"), ("F-Z", "D1", "C")):
        worst = 0.0
        per_dut = {}
        for dut in ("thru", "pec_short", "slab"):
            k = f"{a}-{b}|{dut}"
            if k in art["pairwise_deltas"]:
                v = art["pairwise_deltas"][k]["max_abs_delta_S"]
                per_dut[dut] = v
                worst = max(worst, v)
        verdicts[tag] = {
            "rule": f"max |S_{a} - S_{b}| <= 1e-4",
            "per_dut": per_dut, "worst": worst, "window": 1.0e-4,
            "fired": bool(worst > 1.0e-4),
        }

    # F-A: arm D2 must EXCEED arm B's allowance on at least one gate/bin
    fa = {}
    if "D2|thru" in arms:
        dD2 = _dev_per_bin(arms, "D2|thru")
        dA = _dev_per_bin(arms, "A|thru")
        for gate in dD2:
            allowance = a_b_ph if gate in PHASE_GATES else a_b_amp
            excess = dD2[gate] - dA[gate] - allowance
            fa[gate] = {
                "dev_D2": [float(v) for v in dD2[gate]],
                "dev_A": [float(v) for v in dA[gate]],
                "allowance": [float(v) for v in allowance],
                "excess_per_bin": [float(v) for v in excess],
                "max_excess": float(excess.max()),
                "exceeds": bool(excess.max() > 0.0),
            }
        fa["amplitude_witness"] = {
            "max_abs_delta_S_vs_A": art["pairwise_deltas"]["D2-A|thru"]["max_abs_delta_S"],
            "allowance_max": float(a_b_amp.max()),
        }
    verdicts["F-A"] = {
        "rule": "arm D2 must exceed arm B's allowance on at least one gate and bin",
        "gates": fa,
        "falsifier_behaves": bool(any(v.get("exceeds") for v in fa.values()
                                      if isinstance(v, dict) and "exceeds" in v)),
    }

    # Arm E — reference only, no window (note §4). Recorded the same way as
    # F-A so a reader can see how an IN-envelope ratio on the propagation axis
    # compares with the reflection-derived allowance.
    if "E|thru" in arms:
        dE = _dev_per_bin(arms, "E|thru")
        dA = _dev_per_bin(arms, "A|thru")
        g = "thru_s21_phase_residual_deg"
        excess = dE[g] - dA[g] - a_b_ph
        verdicts["E-reference"] = {
            "rule": "no window; recorded against arm B's allowance for comparison",
            "dev_E": [float(v) for v in dE[g]],
            "dev_A": [float(v) for v in dA[g]],
            "allowance_B": [float(v) for v in a_b_ph],
            "excess_per_bin": [float(v) for v in excess],
            "max_excess": float(excess.max()),
            "n_bins_over_allowance": int((excess > 0).sum()),
        }

    # Yee predictions against measurement
    pred = {}
    for tag, key, predicted in (
            ("dt-term (B/C vs A, thru)", "B-A|thru", disp["dt_term_deg_rms"]),
            ("D2 band (vs A, thru)", "D2-A|thru", disp["D2"]["excess_phase_deg_rms"]),
            ("E band (vs A, thru)", "E-A|thru", disp["E"]["excess_phase_deg_rms"])):
        if key in art["pairwise_deltas"]:
            meas = art["pairwise_deltas"][key]["s21_phase_delta_deg_rms"]
            pred[tag] = {"predicted_deg_rms": float(predicted),
                         "measured_deg_rms": float(meas),
                         "rel_error": float(abs(meas - predicted) / max(predicted, 1e-12))}
    verdicts["yee_prediction_vs_measurement"] = pred

    art["verdicts"] = verdicts
    _write(path, art)
    _log(f"updated {path} with verdicts")
    for k, v in verdicts.items():
        if isinstance(v, dict) and "fired" in v:
            _log(f"  {k}: fired={v['fired']} worst_margin="
                 f"{v.get('worst_margin', v.get('worst'))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stages", default="arms,assemble")
    ap.add_argument("--arms", default="")
    ap.add_argument("--artifact-out", default="")
    ap.add_argument("--run-id", default="local")
    ap.add_argument("--run-lane", default="local")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    args.arms = tuple(a for a in args.arms.split(",") if a)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance(args)
    _log(f"rfx.__file__ = {rfx.__file__}")
    _log(f"commit {prov['commit']} dirty={prov['git_dirty']} backend={prov['jax_default_backend']}")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if "arms" in stages:
        stage_arms(args, out_dir, prov)
    if "assemble" in stages:
        stage_assemble(args, out_dir, prov)
    if "verdicts" in stages:
        stage_verdicts(args, out_dir)


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", ""))
    main()
