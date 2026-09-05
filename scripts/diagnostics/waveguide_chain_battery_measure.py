#!/usr/bin/env python3
"""WR-90 chain battery — the measurement driver (v1.8 WP2).

Runs the pre-declared battery of
``docs/design_notes/20260905_v18_close_predeclaration.md`` (run 3, the v1.8 closing
run; ``waveguide_chain_battery_remeasure_predeclaration.md`` governed run 2 and the
parent note ``waveguide_chain_battery_predeclaration.md`` run 1) on the fixture set
built by ``tests/_waveguide_chain_battery_fixture.py`` and writes
``tests/fixtures/waveguide_chain_battery/fixture_v18_close.json`` (schema:
``tests/fixtures/waveguide_chain_battery/README.md``). Gate arithmetic lives in
``tests/_waveguide_chain_battery_gates.py`` and is shared with the replay test
``tests/oracle/test_waveguide_chain_battery.py``; this file only builds, runs,
records and persists.

Stages (``--stages``), each persisting one JSON per case into ``--out-dir``
the moment the case finishes, before anything optional runs:

* ``cells``       one ``compute_waveguide_s_matrix`` per (dut, rung, lane) at
                  the default reference planes; preflight findings verbatim,
                  ``settling_db`` per drive (and the 80-period rerun when a
                  drive reads above −40 dB, §2.5), the per-cell physics
                  metrics of §4 / §6.
* ``ad_fd``       §5(a): reverse-mode ``jax.value_and_grad`` of each objective
                  at θ0 (float32, the fixture as the gates see it) and a
                  central FD reference under a scoped x64 context, with the
                  ULP-span validity of the FD pair recorded BEFORE the
                  accuracy gate. Also the criterion-1 forward identity.
* ``plane_shift`` §5(b): the shifted-plane S-matrix, |S| invariance, the
                  rotation of every entry against the Yee-discrete and the
                  continuous β, the wrong-sign witness, and the
                  report-first gradient leg (magnitude objectives invariant,
                  complex objectives rotation-covariant).
* ``assemble``    reads every per-case JSON and writes the fixture with the
                  ladder (§5(c)), referee (§5(d)), physics gates (§6) and the
                  ``verdicts`` block. Pure post-processing; no FDTD.

``--refute-flip-shift-sign`` runs the pre-declared cheap refute (§8): the
plane-shift stage under a local copy of ``_shift_modal_waves`` with the sign
of the shift flipped; the rotation gate must go red by more than 10°.

Usage (from a clean checkout; the rfx import must resolve to this tree)::

    PYTHONPATH=. python scripts/diagnostics/waveguide_chain_battery_measure.py \
        --out-dir <run-dir> --run-id <vessl run id> --run-lane vessl
    PYTHONPATH=. python scripts/diagnostics/waveguide_chain_battery_measure.py \
        --out-dir <run-dir> --stages assemble \
        --fixture-out tests/fixtures/waveguide_chain_battery/fixture_v18_close.json

Lanes ``normalize=False`` and ``normalize="flux"`` only (``normalize=True``
never enters). Nothing from ``rfx/probes/refplane.py`` is imported.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
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
import rfx.sources.waveguide_port as _wp  # noqa: E402
from rfx.simulation import _nearest_divisor  # noqa: E402

from tests import _waveguide_chain_battery_fixture as F  # noqa: E402
from tests import _waveguide_chain_battery_gates as G  # noqa: E402
from tests._x64_compat import enable_x64  # noqa: E402

SCHEMA = "rfx.waveguide_chain_battery"
# Identity stamp of the SECOND run's artifact (re-measurement pre-declaration §7).
# The first run's artifact stays at schema_version 1 and keeps pointing at the
# parent note; nothing here rewrites it.
SCHEMA_VERSION = 3
PREDECLARATION = "docs/design_notes/20260905_v18_close_predeclaration.md"
ARTIFACT = "tests/fixtures/waveguide_chain_battery/fixture_v18_close.json"
SUPERSEDES = "tests/fixtures/waveguide_chain_battery/fixture_guide_cell_aperture.json"
SUPERSEDES_REASON = (
    "same port, same battery: this artifact reads contract criterion 1 (forward identity) and "
    "3(a) (AD-vs-FD) under x64 on the flux lane per the v1.8 closing declaration, stores the "
    "float32 reading beside it, and carries the pre-declared zero-derivative leg as report_only")
README = "tests/fixtures/waveguide_chain_battery/README.md"
DRIVER = "scripts/diagnostics/waveguide_chain_battery_measure.py"
SETTLING_RERUN_NUM_PERIODS = 2.0 * F.NUM_PERIODS      # §2.5 record-length doubling


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def git_sha(override: str | None) -> str:
    if override:
        return override
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 — provenance only, never fatal
        return "unknown"


def provenance(args) -> dict:
    return {
        "commit": git_sha(args.git_sha),
        "run_id": args.run_id,
        "run_lane": args.run_lane,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "precision": "float32",
        "python": sys.version.split()[0],
        "hostname": platform.node(),
        "rfx_version": getattr(rfx, "__version__", "?"),
    }


def _write(path: Path, obj: dict) -> None:
    """Persist atomically, first (feedback: persist before the optional stage),
    then drop the JIT compile cache: every case compiles its own programs and
    one process holding forty of them ran LLVM out of memory on the coarse
    plumbing run ("LLVM compilation error: Cannot allocate memory")."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)
    jax.clear_caches()


def _log(msg: str) -> None:
    print(f"[chain-battery {_dt.datetime.now(_dt.timezone.utc).strftime('%H:%M:%S')}] {msg}",
          flush=True)


# ---------------------------------------------------------------------------
# instrumentation: preflight findings, warnings, per-record settling
# ---------------------------------------------------------------------------

def preflight_findings(sim) -> list[dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
    return [{"code": getattr(i, "code", "uncoded"), "severity": getattr(i, "severity", "warning"),
             "message": str(i)} for i in report]


class _SettlingSpy:
    """Per-record diagnostics of the production witness, captured on the
    records it consumes (pass-through; the witness itself is untouched).
    Needed because ``settling_db_from_port_records`` returns 0 dB for a record
    whose peak is exactly zero (``(end+tiny)/(peak+tiny)``), which is what a
    port behind a PEC short records once the field underflows float32."""

    def __init__(self):
        self.calls: list[list[dict]] = []
        self._orig = _wp.settling_db_from_port_records

    def __enter__(self):
        spy = self

        def wrapped(final_cfgs):
            recs = []
            for pi, cfg in enumerate(final_cfgs):
                for rec in ("v_probe_t", "v_ref_t", "i_probe_t", "i_ref_t"):
                    ts = getattr(cfg, rec)
                    try:
                        arr = np.abs(np.asarray(ts, dtype=np.float64))
                    except Exception:  # noqa: BLE001 — a tracer: the witness returns NaN
                        recs.append({"port_index": pi, "record": rec, "traced": True})
                        continue
                    p = arr ** 2
                    tail = max(1, p.shape[0] // 10)
                    end, peak = float(p[-tail:].mean()), float(p.max())
                    recs.append({"port_index": pi, "record": rec, "peak": peak, "end": end,
                                 "n_nonzero": int((arr > 0).sum()), "n_steps": int(arr.shape[0]),
                                 "db": (10.0 * math.log10(end / peak) if peak > 0 and end > 0
                                        else None),
                                 "peak_is_zero": bool(peak == 0.0)})
            spy.calls.append(recs)
            return spy._orig(final_cfgs)

        _wp.settling_db_from_port_records = wrapped
        return self

    def __exit__(self, *exc):
        _wp.settling_db_from_port_records = self._orig
        return False


def _dedupe_warnings(wlist) -> list[dict]:
    seen: dict[str, int] = {}
    for w in wlist:
        key = f"{w.category.__name__}: {w.message}"
        seen[key] = seen.get(key, 0) + 1
    return [{"message": k, "count": n} for k, n in seen.items()]


def run_smatrix(sim, lane, *, num_periods: float, spy: bool = True, **kw):
    """One ``compute_waveguide_s_matrix`` with warnings and per-record settling captured."""
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        t0 = time.time()
        if spy:
            with _SettlingSpy() as sp:
                res = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize=lane, **kw)
                S = np.asarray(res.s_params)
            calls = sp.calls
        else:
            res = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize=lane, **kw)
            S = np.asarray(res.s_params)
            calls = []
        wall = time.time() - t0
    return res, S.astype(np.complex128), wall, _dedupe_warnings(wl), calls


def settling_dict(res) -> dict[str, float]:
    sd = np.asarray(res.settling_db, dtype=float)
    return {name: float(sd[i]) for i, name in enumerate(F.PORT_NAMES)}


# ---------------------------------------------------------------------------
# stage: cells
# ---------------------------------------------------------------------------

def cell_path(out_dir: Path, dut: str, rung: str, lane_label: str) -> Path:
    return out_dir / f"cell__{dut}__{rung}__{lane_label}.json"


def stage_cells(args, out_dir: Path, rungs: list[str], prov: dict) -> None:
    for rung in rungs:
        dx = G.RUNG_DX[rung]
        for dut in F.DUTS:
            sim = F.build_simulation(dut, dx)
            grid = sim._build_grid()
            findings = preflight_findings(sim)
            masks = F.dut_masks(sim)
            dut_cells = None
            dut_runs = None
            if masks:
                (mat, mask), = masks.items()
                dut_cells = int(mask.sum())
                dut_runs = [int(x) for x in F.axis_run_lengths(mask)]
            for lane in F.LANES:
                lane_label = G.LANE_LABELS[lane]
                path = cell_path(out_dir, dut, rung, lane_label)
                if path.exists() and not args.overwrite:
                    _log(f"skip existing {path.name}")
                    continue
                _log(f"cell {dut} {rung} {lane_label}: grid={grid.shape} n_steps={grid.num_timesteps(F.NUM_PERIODS)}")
                res, S, wall, wl, calls = run_smatrix(sim, lane, num_periods=F.NUM_PERIODS)
                settling = settling_dict(res)
                rec = {
                    "dut": dut, "dx_m": dx, "rung": rung, "lane": lane_label,
                    "cpml_layers": int(sim._cpml_layers),
                    "fc_te10_numerical_hz": float(F.numerical_te10_cutoff_hz(sim)),
                    # the cutoff the PORT CONFIG carries (β / Z_TE of the extractor)
                    "port_f_cutoff_hz": [float(F.port_cutoff_hz(sim, 0)), float(F.port_cutoff_hz(sim, 1))],
                    "fc_discrete_guide_hz": float(G.discrete_guide_cutoff_hz(dx)),
                    "n_steps": int(grid.num_timesteps(F.NUM_PERIODS)),
                    "num_periods": F.NUM_PERIODS,
                    "dt_s": float(grid.dt),
                    "grid_shape": [int(x) for x in grid.shape],
                    "guide_cells_yz": [int(x - 1) for x in F.realized_guide_nodes(sim)],
                    "dut_cells": dut_cells, "dut_runs_xyz": dut_runs,
                    "preflight": findings,
                    "warnings": wl,
                    "settling_db": settling,
                    "settling_records": calls,
                    "settling_rerun": None,
                    "s_params": G.s_to_json(S),
                    "reference_planes_m": [float(x) for x in np.asarray(res.reference_planes)],
                    "wall_time_s": wall,
                    "provenance": prov,
                    **G.cell_metrics(S),
                }
                over = {k: v for k, v in settling.items() if v > G.SETTLING_DB_MAX}
                if over:
                    _log(f"  settling above {G.SETTLING_DB_MAX} dB on {over}: rerun at "
                         f"num_periods={SETTLING_RERUN_NUM_PERIODS} (§2.5)")
                    res2, S2, wall2, wl2, calls2 = run_smatrix(
                        sim, lane, num_periods=SETTLING_RERUN_NUM_PERIODS)
                    rec["settling_rerun"] = {
                        "num_periods": SETTLING_RERUN_NUM_PERIODS,
                        "n_steps": int(grid.num_timesteps(SETTLING_RERUN_NUM_PERIODS)),
                        "settling_db": settling_dict(res2),
                        "settling_records": calls2,
                        "s_params": G.s_to_json(S2),
                        "max_abs_s_shift_vs_40_periods": float(np.max(np.abs(np.abs(S2) - np.abs(S)))),
                        "wall_time_s": wall2,
                        "warnings": wl2,
                        **G.cell_metrics(S2),
                    }
                _write(path, rec)
                _log(f"  wrote {path.name}: settling={settling} colpow={rec['column_power_max']:.5f} "
                     f"recip_c={rec['reciprocity_complex_max']:.2e} max|S11|={rec['non_vacuity_max_s11']:.4f} "
                     f"wall={wall:.1f}s")


# ---------------------------------------------------------------------------
# stage: AD vs FD
# ---------------------------------------------------------------------------

def _override_kw(sim, dut, kind, theta) -> dict:
    ov = F.design_override(sim, dut, theta, kind=kind)
    return {"eps_override": ov} if kind == "eps" else {"sigma_override": ov}


def _checkpoint_segments(grid) -> int | None:
    n = int(grid.num_timesteps(F.NUM_PERIODS))
    k = _nearest_divisor(n, max(1, int(round(math.sqrt(n)))))
    return k if k > 1 else None


def ad_fd_path(out_dir: Path, dut: str, lane_label: str, kind: str) -> Path:
    return out_dir / f"ad_fd__{dut}__{lane_label}__{kind}.json"


def theta0_and_h(kind: str) -> tuple[float, float]:
    if kind == "eps":
        return F.THETA0_EPS, F.FD_STEP_EPS
    return F.THETA0_SIGMA_S_PER_M, F.FD_STEP_SIGMA_S_PER_M


def ad_grads(sim, dut, kind, lane, objectives, theta0: float, cseg, *, tag: str) -> dict:
    """Reverse-mode ``jax.value_and_grad`` per scalar objective at θ0 (float32)."""
    out = {}
    for name in objectives:
        def f(th, _name=name):
            S = sim.compute_waveguide_s_matrix(
                num_periods=F.NUM_PERIODS, normalize=lane, checkpoint_segments=cseg,
                **_override_kw(sim, dut, kind, th)).s_params
            return G.objective_value(S, _name), S
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (val, S_primal), g = jax.value_and_grad(f, has_aux=True)(jnp.asarray(theta0, jnp.float32))
        wall = time.time() - t0
        out[name] = {"value": float(val), "g_ad": float(g), "wall_time_s": wall,
                     "S_primal": np.asarray(S_primal).astype(np.complex128),
                     "grad_dtype": str(np.asarray(g).dtype)}
        _log(f"  AD {tag} {name}: value={float(val):.6e} g_ad={float(g):+.6e} wall={wall:.1f}s")
    return out


def stage_ad_fd(args, out_dir: Path, rung: str, prov: dict) -> None:
    dx = G.RUNG_DX[rung]
    for (dut, kind), objectives in G.AD_LEGS.items():
        theta0, h = theta0_and_h(kind)
        for lane in F.LANES:
            lane_label = G.LANE_LABELS[lane]
            path = ad_fd_path(out_dir, dut, lane_label, kind)
            if path.exists() and not args.overwrite:
                _log(f"skip existing {path.name}")
                continue
            t_stage = time.time()
            sim = F.build_simulation(dut, dx)
            grid = sim._build_grid()
            cseg = _checkpoint_segments(grid)
            _log(f"ad_fd {dut} {lane_label} {kind}: rung={rung} theta0={theta0} h={h} cseg={cseg}")
            # untraced references for the forward identity (criterion 1)
            _, S_plain, _, _, _ = run_smatrix(sim, lane, num_periods=F.NUM_PERIODS, spy=False)
            _, S_concrete, _, _, _ = run_smatrix(
                sim, lane, num_periods=F.NUM_PERIODS, spy=False,
                **_override_kw(sim, dut, kind, jnp.asarray(theta0, jnp.float32)))
            grads = ad_grads(sim, dut, kind, lane, objectives, theta0, cseg,
                             tag=f"{dut}/{lane_label}/{kind}")
            # FD reference under a SCOPED x64 context (never module-level)
            t_fd = time.time()
            with enable_x64():
                sim64 = F.build_simulation(dut, dx)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    S_plus = sim64.compute_waveguide_s_matrix(
                        num_periods=F.NUM_PERIODS, normalize=lane,
                        **_override_kw(sim64, dut, kind, jnp.asarray(theta0 + h, jnp.float64))).s_params
                    S_plus = jnp.asarray(S_plus)
                    S_minus = sim64.compute_waveguide_s_matrix(
                        num_periods=F.NUM_PERIODS, normalize=lane,
                        **_override_kw(sim64, dut, kind, jnp.asarray(theta0 - h, jnp.float64))).s_params
                    S_minus = jnp.asarray(S_minus)
                fd = {}
                for name in objectives:
                    fp = G.objective_value(S_plus, name)
                    fm = G.objective_value(S_minus, name)
                    fd[name] = {"f_plus": float(fp), "f_minus": float(fm),
                                "loss_dtype": np.dtype(np.asarray(fp).dtype),
                                "s_dtype": str(S_plus.dtype)}
                x64_flag = bool(jax.config.x64_enabled)
            fd_wall = time.time() - t_fd
            # x64 witnesses (report-only): (i) the reverse-mode primal identity
            # in float64 — if the float32 identity is outside rtol 1e-5 while the
            # float64 one is at rounding, the difference is reassociation of the
            # differently-compiled vjp forward, not a wrong op; (ii) any non-finite
            # float32 gradient re-evaluated in float64.
            x64_witness = {}
            t_x = time.time()
            # v1.8 closing declaration: on X64_DECLARED_LANES the x64 reading is the PRIMARY the
            # gate reads (criterion 1 and 3(a)), so it is measured for EVERY objective, not only
            # objectives[0]. The forward identity does not depend on the objective, so one S64
            # per group serves all legs. RFX_CHAIN_PRIMARY=float32 keeps the float32 primary
            # (the pre-declaration's section-4 falsifier: must reproduce run 2's 9 red).
            x64_primary = (lane_label in G.X64_DECLARED_LANES
                           and os.environ.get("RFX_CHAIN_PRIMARY", "declared") != "float32")
            with enable_x64():
                sim64w = F.build_simulation(dut, dx)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    S0_64 = np.asarray(sim64w.compute_waveguide_s_matrix(
                        num_periods=F.NUM_PERIODS, normalize=lane,
                        **_override_kw(sim64w, dut, kind, jnp.asarray(theta0, jnp.float64))).s_params)
                    nonfinite = [n for n in objectives if not np.isfinite(grads[n]["g_ad"])]
                    witness_names = (list(objectives) if x64_primary
                                     else [objectives[0]] + [n for n in nonfinite if n != objectives[0]])
                    for name in witness_names:
                        def f64(th, _name=name):
                            S = sim64w.compute_waveguide_s_matrix(
                                num_periods=F.NUM_PERIODS, normalize=lane, checkpoint_segments=cseg,
                                **_override_kw(sim64w, dut, kind, th)).s_params
                            return G.objective_value(S, _name), S
                        (v64, S64), g64 = jax.value_and_grad(f64, has_aux=True)(jnp.asarray(theta0, jnp.float64))
                        x64_witness[name] = {
                            "g_ad_x64": float(g64), "value_x64": float(v64),
                            "forward_identity_x64": G.forward_identity_metric(np.asarray(S64), S0_64),
                        }
                        _log(f"  x64 witness {name}: g_ad_x64={float(g64):+.6e} identity max|dS|="
                             f"{x64_witness[name]['forward_identity_x64']['max_abs_diff']:.3e}")
            x64_wall = time.time() - t_x
            legs = []
            for name in objectives:
                e = G.ad_fd_entry(g_ad=grads[name]["g_ad"], f_plus=fd[name]["f_plus"],
                                  f_minus=fd[name]["f_minus"], h=h, loss_dtype=fd[name]["loss_dtype"])
                # eps legs: θ0 = 0, so the untraced call IS the plain fixture; the
                # sigma leg (θ0 = 0.05 S/m) compares with the concrete override at θ0.
                ident_ref = S_plain if kind == "eps" else S_concrete
                ident = G.forward_identity_metric(grads[name]["S_primal"], ident_ref)
                ident_concrete = G.forward_identity_metric(S_concrete, S_plain) if kind == "eps" else None
                expected_skip = (dut, kind, name) in G.EXPECTED_ULP_SKIP
                w = x64_witness.get(name)
                primary = "x64" if (x64_primary and w is not None) else "float32"
                if primary == "x64":
                    ident_primary = w["forward_identity_x64"]
                    e_primary = G.ad_fd_entry(g_ad=w["g_ad_x64"], f_plus=fd[name]["f_plus"],
                                              f_minus=fd[name]["f_minus"], h=h, loss_dtype=fd[name]["loss_dtype"])
                    if expected_skip and e_primary["verdict"] != "skipped_under_ulp_floor":
                        # closing pre-declaration section 2: the pre-declared zero-derivative
                        # leg is REPORT-ONLY on the remeasure note's exit (c); the sign/factor-3
                        # entry is stored beside it, never read as the verdict.
                        zd = G.zero_derivative_entry(g_ad_x64=w["g_ad_x64"], g_fd=e_primary["g_fd"],
                                                     fd_ulp_span=e_primary["fd_ulp_span"])
                        e_primary = {**e_primary, "zero_derivative": zd, "verdict": "report_only",
                                     "report_only_reason": "pre-declared zero-derivative objective; "
                                     "AD and FD are O(1e-7) discretization residuals of a physically "
                                     "zero derivative (closing pre-declaration section 2)"}
                else:
                    ident_primary, e_primary = ident, e
                legs.append({
                    "primary_precision": primary,
                    "forward_identity_float32": ident, "ad_vs_fd_float32": e,
                    "dut": dut, "lane": lane_label, "dx_m": dx, "rung": rung,
                    "objective": name, "theta_kind": kind, "theta0": theta0, "h": h,
                    "x64_context": x64_flag, "s_dtype_fd": fd[name]["s_dtype"],
                    "checkpoint_segments": cseg,
                    "value_at_theta0": grads[name]["value"],
                    "grad_dtype": grads[name]["grad_dtype"],
                    "expected_ulp_floor_skip": expected_skip,
                    "forward_identity": ident_primary,
                    "forward_identity_concrete_override_vs_plain": ident_concrete,
                    "wall_time_s": {"ad": grads[name]["wall_time_s"], "fd_pair": fd_wall,
                                    "x64_witness": x64_wall},
                    "x64_witness": w,
                    **e_primary,
                })
                _log(f"  {name}: g_ad={e['g_ad']:+.6e} g_fd={e['g_fd']:+.6e} rel={e['rel']:.3e} "
                     f"span={e['fd_ulp_span']:.3g} -> {e['verdict']}; identity scaled={ident['max_scaled_diff']:.3f}")
            _write(path, {"dut": dut, "lane": lane_label, "theta_kind": kind, "dx_m": dx, "rung": rung,
                          "legs": legs, "wall_time_s": time.time() - t_stage, "provenance": prov})
            _log(f"  wrote {path.name} ({time.time() - t_stage:.0f}s)")


# ---------------------------------------------------------------------------
# stage: plane shift
# ---------------------------------------------------------------------------

def plane_shift_path(out_dir: Path, dut: str, lane_label: str) -> Path:
    return out_dir / f"plane_shift__{dut}__{lane_label}.json"


class _FlippedShift:
    """The cheap refute of §8: a local copy of ``_shift_modal_waves`` with the
    sign of the shift flipped (the 2026-04-22 ``step_sign`` bug class)."""

    def __enter__(self):
        self._orig = _wp._shift_modal_waves

        def flipped(forward, backward, beta, shift_m, step_sign=1):
            return self._orig(forward, backward, beta, shift_m, -step_sign)

        _wp._shift_modal_waves = flipped
        return self

    def __exit__(self, *exc):
        _wp._shift_modal_waves = self._orig
        return False


def stage_plane_shift(args, out_dir: Path, rung: str, prov: dict, *, refute: bool = False) -> None:
    dx = G.RUNG_DX[rung]
    shifted = (F.REF_LEFT_SHIFTED_M, F.REF_RIGHT_SHIFTED_M)
    results = {}
    for dut in ("pec_short", "slab"):
        for lane in F.LANES:
            lane_label = G.LANE_LABELS[lane]
            path = (out_dir / f"refute_flip_shift_sign__{dut}__{lane_label}.json" if refute
                    else plane_shift_path(out_dir, dut, lane_label))
            if path.exists() and not args.overwrite:
                _log(f"skip existing {path.name}")
                continue
            t_stage = time.time()
            base_path = cell_path(out_dir, dut, rung, lane_label)
            sim_base = F.build_simulation(dut, dx)
            grid = sim_base._build_grid()
            ctx = _FlippedShift() if refute else _NullCtx()
            with ctx:
                if base_path.exists() and not refute:
                    S_base = G.s_from_json(json.loads(base_path.read_text())["s_params"])
                    base_src = base_path.name
                else:
                    _, S_base, _, _, _ = run_smatrix(sim_base, lane, num_periods=F.NUM_PERIODS, spy=False)
                    base_src = "recomputed"
                sim_shift = F.build_simulation(dut, dx, reference_planes=shifted)
                res_s, S_shift, wall_s, wl, _ = run_smatrix(sim_shift, lane, num_periods=F.NUM_PERIODS)
            fc_port = float(F.port_cutoff_hz(sim_base, 0))
            rot = G.plane_shift_rotation(S_base, S_shift, F.FREQS, float(grid.dt), dx,
                                         fc_port_hz=fc_port)
            rec = {
                "dut": dut, "lane": lane_label, "dx_m": dx, "rung": rung,
                "reference_planes_base_m": [F.REF_LEFT_DEFAULT_M, F.REF_RIGHT_DEFAULT_M],
                "reference_planes_shifted_m": [float(x) for x in np.asarray(res_s.reference_planes)],
                "shift_m": [G.SHIFT_LEFT_M, G.SHIFT_RIGHT_M],
                "base_source": base_src,
                "s_params_shifted": G.s_to_json(S_shift),
                "settling_db_shifted": settling_dict(res_s),
                "warnings_shifted": wl,
                "wall_time_s": {"shifted_forward": wall_s},
                "provenance": prov,
                **rot,
            }
            if refute:
                rec["refute"] = "local copy of _shift_modal_waves with the shift sign flipped"
                rec["resid_yee_min_over_entries"] = min(
                    v["resid_yee_max"] for v in rot["rotation_deg"].values() if v["resid_yee_max"] is not None)
                rec["rotation_gate_would_pass"] = bool(rot["resid_yee_max"] <= G.ROTATION_TOL_YEE_DEG)
                _write(path, rec)
                _log(f"  REFUTE {dut} {lane_label}: resid_yee per entry "
                     f"{ {k: (None if v['resid_yee_max'] is None else round(v['resid_yee_max'], 2)) for k, v in rot['rotation_deg'].items()} } "
                     f"gate_would_pass={rec['rotation_gate_would_pass']} |S| invariant={rot['abs_s_allclose']}")
                continue
            _log(f"plane_shift {dut} {lane_label}: |S| max diff={rot['abs_s_max_diff']:.2e} "
                 f"resid_yee={rot['resid_yee_max']:.3f}° resid_cont={rot['resid_cont_max']:.3f}° "
                 f"resid_port_beta={rot['resid_port_beta_max']:.3f}° (fc_port={fc_port/1e9:.4f} GHz) "
                 f"wrong_sign_min={rot['wrong_sign_resid_min']:.1f}°")
            # gradient leg: base gradients from the ad_fd stage; shifted gradients here.
            # φ for the covariance = the rotation the extractor actually applied
            # (∠(S_shift/S_base) at the centre bin); the pre-declared 2β(c/2a)Δ is
            # written alongside.
            ginv = {}
            k = F.BAND_CENTRE_BIN
            phi_meas = {"S11": float(np.angle(S_shift[0, 0, k] * np.conj(S_base[0, 0, k]))),
                        "S21": float(np.angle(S_shift[1, 0, k] * np.conj(S_base[1, 0, k])))}
            beta_c = float(G.beta_yee(F.FREQS, float(grid.dt), dx)[k])
            phi_pre = {"S11": 2.0 * beta_c * G.SHIFT_LEFT_M,
                       "S21": beta_c * (G.SHIFT_LEFT_M + abs(G.SHIFT_RIGHT_M))}
            cseg = _checkpoint_segments(grid)
            for (d, kind), objectives in G.AD_LEGS.items():
                if d != dut:
                    continue
                theta0, _ = theta0_and_h(kind)
                base_file = ad_fd_path(out_dir, dut, lane_label, kind)
                if not base_file.exists():
                    _log(f"  gradient leg {kind}: no {base_file.name}; run the ad_fd stage first — skipped")
                    continue
                base_legs = {l["objective"]: l for l in json.loads(base_file.read_text())["legs"]}
                resolvable = [n for n in objectives
                              if base_legs[n]["verdict"] != "skipped_under_ulp_floor"]
                skipped = [n for n in objectives if n not in resolvable]
                for n in skipped:
                    ginv[f"{kind}:{n}"] = {"kind": G.OBJECTIVES[n][0], "skipped_under_ulp_floor": True,
                                          "reason": "base FD leg under the ULP floor: the gradient is "
                                                    "not resolvable, so its invariance is not testable"}
                if not resolvable:
                    continue
                grads = ad_grads(sim_shift, dut, kind, lane, resolvable, theta0, cseg,
                                 tag=f"{dut}/{lane_label}/{kind} shifted")
                # Criterion 3(b) is not under the x64 declaration: the base gradient is the
                # leg's FLOAT32 reading (``ad_vs_fd_float32`` from schema_version 3, ``g_ad``
                # itself before), the same precision as the shifted gradient computed here.
                # The closing run (VESSL 369367258638) read ``g_ad`` here while that key had
                # become the x64 primary on the flux lane; ``G.rebase_gradient_invariance_float32``
                # rebuilt its entries from the stored numbers in the pin step.
                def _g32(n):
                    return base_legs[n].get("ad_vs_fd_float32", base_legs[n])["g_ad"]
                for n in resolvable:
                    if G.OBJECTIVES[n][0] == "magnitude":
                        ginv[f"{kind}:{n}"] = G.gradient_invariance_entry(
                            "magnitude", _g32(n), 0.0, grads[n]["g_ad"], 0.0, None)
                        ginv[f"{kind}:{n}"].update(base_precision="float32", shift_precision="float32")
                complex_pairs = {("re_s21", "im_s21"): "s21_complex", ("re_s11", "im_s11"): "s11_complex"}
                for (re_n, im_n), label in complex_pairs.items():
                    if re_n in resolvable and im_n in resolvable:
                        entry = "S21" if label == "s21_complex" else "S11"
                        ginv[f"{kind}:{label}"] = G.gradient_invariance_entry(
                            "complex", _g32(re_n), _g32(im_n),
                            grads[re_n]["g_ad"], grads[im_n]["g_ad"], phi_meas[entry], phi_pre[entry])
                        ginv[f"{kind}:{label}"]["from_objectives"] = [re_n, im_n]
                        ginv[f"{kind}:{label}"].update(base_precision="float32", shift_precision="float32")
                for k, e in ginv.items():
                    if k.startswith(kind) and "rel_change" in e:
                        _log(f"  gradient leg {k}: rel_change={e['rel_change']:.3e} (bar {G.GRADIENT_REPORT_BAR})")
            rec["gradient_invariance"] = ginv
            rec["wall_time_s"]["total"] = time.time() - t_stage
            _write(path, rec)
            results[f"{dut}|{lane_label}"] = rec
            _log(f"  wrote {path.name} ({time.time() - t_stage:.0f}s)")


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# stage: assemble
# ---------------------------------------------------------------------------

def _fixture_constants() -> dict:
    return {
        "a_m": F.A_M, "b_m": F.B_M, "dx_ladder_m": list(F.DX_LADDER), "n_ladder": list(F.N_LADDER),
        "domain_x_m": F.DOMAIN_X_M,
        "port_planes_m": [F.PORT_LEFT_X_M, F.PORT_RIGHT_X_M],
        "reference_planes_default_m": [F.REF_LEFT_DEFAULT_M, F.REF_RIGHT_DEFAULT_M],
        "reference_planes_shifted_m": [F.REF_LEFT_SHIFTED_M, F.REF_RIGHT_SHIFTED_M],
        "probe_planes_m": [F.PROBE_LEFT_M, F.PROBE_RIGHT_M],
        "pec_short_x_m": list(F.PEC_SHORT_X_M), "slab_x_m": list(F.SLAB_X_M),
        "slab_eps_r": F.SLAB_EPS_R, "pec_short_window_x_m": list(F.PEC_SHORT_WINDOW_X_M),
        "freqs_hz": [float(f) for f in F.FREQS], "f0_hz": F.F0_HZ, "bandwidth": F.BANDWIDTH,
        "band_centre_bin": F.BAND_CENTRE_BIN, "num_periods": F.NUM_PERIODS,
        "lanes": [G.LANE_LABELS[l] for l in F.LANES],
        "boundary": "cpml-x, pec-y, pec-z",
        "theta0_eps": F.THETA0_EPS, "theta0_sigma_s_per_m": F.THETA0_SIGMA_S_PER_M,
        "fd_step_eps": F.FD_STEP_EPS, "fd_step_sigma_s_per_m": F.FD_STEP_SIGMA_S_PER_M,
    }


def _ladder_block(cells: list[dict]) -> dict:
    by = {(c["dut"], c["rung"], c["lane"]): c for c in cells}
    out = {}
    f = np.asarray(F.FREQS, dtype=float)
    s11_ref, s21_ref = G.airy_reference(f)
    for name, dut, entry, kind in G.LADDER_OBSERVABLES:
        for lane in (G.LANE_LABELS[l] for l in F.LANES):
            try:
                S_by = {r: G.s_from_json(by[(dut, r, lane)]["s_params"]) for r in G.RUNG_LABELS}
            except KeyError:
                continue
            oracle = None
            oracle_c = None
            if dut == "slab":
                ref = s11_ref if entry == (0, 0) else s21_ref
                o = np.abs(ref) if kind == "mag" else np.degrees(np.angle(ref))
                oracle = {"coarse-mid": o, "mid-fine": o}
            elif kind == "phase":
                # PEC-short phase against π − 2βd, Yee-discrete β of the finer rung of
                # each pair (§5(c)); the continuous-β oracle is written alongside.
                oracle = {}
                for pair, finer in (("coarse-mid", "mid"), ("mid-fine", "fine")):
                    c = by[(dut, finer, lane)]
                    oracle[pair] = G.pec_short_phase_oracle_deg(G.beta_yee(f, c["dt_s"], c["dx_m"]))
                oc = G.pec_short_phase_oracle_deg(G.beta_continuous(f))
                oracle_c = {"coarse-mid": oc, "mid-fine": oc}
            lad = G.ladder_eval(S_by, entry, kind, f, oracle, oracle_c)
            lad.update({"observable": name, "dut": dut, "lane": lane,
                        "pinned_richardson_gate": None, "pinned_monotone_fraction_min": None})
            out[f"{name}|{lane}"] = lad
    return out


def assemble(args, out_dir: Path, prov: dict) -> Path:
    cells = [json.loads(p.read_text()) for p in sorted(out_dir.glob("cell__*.json"))]
    legs = []
    for p in sorted(out_dir.glob("ad_fd__*.json")):
        legs.extend(json.loads(p.read_text())["legs"])
    cell_by = {(c["dut"], c["rung"], c["lane"]): c for c in cells}

    def _rerotate(d: dict) -> dict:
        # Recompute the rotation block from the stored S-matrices (the cell's
        # base S and the shifted S), so the fixture's rotation numbers are a
        # pure function of stored values — the same call the replay makes.
        base = cell_by.get((d["dut"], d["rung"], d["lane"]))
        if base is None:
            return d
        rot = G.plane_shift_rotation(G.s_from_json(base["s_params"]), G.s_from_json(d["s_params_shifted"]),
                                     F.FREQS, base["dt_s"], base["dx_m"], fc_port_hz=d.get("fc_port_hz"))
        d = dict(d)
        d.update(rot)
        if "resid_yee_min_over_entries" in d:
            live = [v["resid_yee_max"] for v in rot["rotation_deg"].values() if v["resid_yee_max"] is not None]
            d["resid_yee_min_over_entries"] = min(live)
            d["rotation_gate_would_pass"] = bool(rot["resid_yee_max"] <= G.ROTATION_TOL_YEE_DEG)
        return d

    planes = {}
    for p in sorted(out_dir.glob("plane_shift__*.json")):
        d = _rerotate(json.loads(p.read_text()))
        planes[f"{d['dut']}|{d['lane']}"] = d
    refutes = [_rerotate(json.loads(p.read_text())) for p in sorted(out_dir.glob("refute_flip_shift_sign__*.json"))]
    if refutes:
        planes["cheap_refute"] = {
            "refute": refutes[0]["refute"], "rung": refutes[0]["rung"],
            "resid_yee_min_over_entries": min(r["resid_yee_min_over_entries"] for r in refutes),
            "resid_yee_max_over_entries": max(r["resid_yee_max"] for r in refutes),
            "rotation_gate_would_pass": any(r["rotation_gate_would_pass"] for r in refutes),
            "abs_s_still_invariant": all(r["abs_s_allclose"] for r in refutes),
            "per_case": [{"dut": r["dut"], "lane": r["lane"],
                          "resid_yee_per_entry": {k: v["resid_yee_max"] for k, v in r["rotation_deg"].items()},
                          "entries_measurable": r["entries_measurable"],
                          "provenance": {k: r["provenance"][k] for k in ("commit", "run_id", "run_lane",
                                                                          "jax_default_backend")}}
                         for r in refutes],
        }
    if not cells:
        raise SystemExit(f"no cell__*.json under {out_dir}")
    # Report-only: the settling witness restricted to records whose peak lies in
    # the float32 NORMAL range. Behind a PEC short the far-port records are
    # subnormal (~1e-40) or exactly zero: on a flush-to-zero CPU the witness
    # reads 0 dB there ((end+tiny)/(peak+tiny) with peak = 0), on the GPU it
    # reads the ratio of two subnormals — neither is a ring-down.
    tiny32 = float(np.finfo(np.float32).tiny)
    for c in cells:
        for block in (c, c.get("settling_rerun") or {}):
            recs = block.get("settling_records") or []
            vals = [r["db"] for call in recs for r in call
                    if r.get("db") is not None and r.get("peak", 0.0) >= tiny32]
            degenerate = [f"call{ci}/port{r['port_index']}/{r['record']}" for ci, call in enumerate(recs)
                          for r in call if r.get("peak") is not None and r["peak"] < tiny32]
            block["settling_db_over_normal_records"] = max(vals) if vals else None
            block["settling_degenerate_records"] = degenerate
            block["float32_normal_min"] = tiny32

    # referee per rung and lane
    referee = {"pec_short": {}, "slab_airy": {}, "broad_e5_replay": {
        "fixtures": sorted(str(p.relative_to(REPO)) for p in
                           (REPO / "tests/fixtures/waveguide_broad_e5").glob("waveguide_*_broad_e5_envelope.json")),
        "gate_test": "tests/crossval/test_waveguide_broad_e5.py",
        "note": "criterion 3(d) support set; replayed by its own gate test, not re-run here"}}
    for c in cells:
        S = G.s_from_json(c["s_params"])
        key = f"{c['rung']}|{c['lane']}"
        if c["dut"] == "pec_short":
            referee["pec_short"][key] = G.referee_pec_short(S, F.FREQS)
        elif c["dut"] == "slab":
            referee["slab_airy"][key] = G.referee_slab_airy(S, F.FREQS)
    referee["conventions"] = {
        "time": "exp(+j omega t); forward wave exp(-j beta x)",
        "dft": "rectangular full-record DFT with kernel exp(-j omega t) (rfx.sources.waveguide_port._rect_dft)",
        "yee_half_step": "beta for plane shifts and the PEC-short oracle from _compute_beta(dt, dx) "
                         "(Yee-discrete dispersion); the Airy oracle uses the continuous vacuum beta",
        "external_phase_data": "none enters; if it ever does it is conjugated first (rfx-known-issues.md, "
                               "time-convention conjugation)",
    }

    # Port-cutoff witness (report-only): the cutoff the port config carries vs
    # the guide's own, fitted from the thru's S21 phase between the two
    # declared planes (ref_shift = 0 on both, so no de-embed β enters).
    port_cutoff = {"length_between_declared_planes_m": float(F.REF_RIGHT_DEFAULT_M - F.REF_LEFT_DEFAULT_M),
                   "per_rung": {}}
    for c in cells:
        if c["dut"] != "thru":
            continue
        S = G.s_from_json(c["s_params"])
        fit = G.fit_guide_cutoff(S[1, 0], F.FREQS, c["dt_s"], c["dx_m"],
                                 port_cutoff["length_between_declared_planes_m"])
        fit["rms_deg_at_port_cutoff"] = None
        if c.get("port_f_cutoff_hz"):
            fc_p = float(c["port_f_cutoff_hz"][0])
            fit["fc_port_hz"] = fc_p
            model = -G.beta_yee_fc(F.FREQS, fc_p, c["dt_s"], c["dx_m"]) * port_cutoff["length_between_declared_planes_m"]
            ph = np.unwrap(np.angle(S[1, 0]))
            fit["rms_deg_at_port_cutoff"] = float(np.degrees(np.sqrt(np.mean((ph - model - np.mean(ph - model)) ** 2))))
            fit["port_cutoff_effective_width_cells"] = float(299_792_458.0 / (2.0 * fc_p) / c["dx_m"])
        port_cutoff["per_rung"][f"{c['rung']}|{c['lane']}"] = fit

    physics = {}
    for c in cells:
        if c["dut"] == "thru":
            continue
        physics[f"{c['dut']}|{c['rung']}|{c['lane']}"] = {
            "column_power_max": c["column_power_max"], "column_power_gate": G.COLUMN_POWER_MAX,
            "reciprocity_mag_mean": c["reciprocity_mag_mean"], "reciprocity_mag_gate": G.RECIPROCITY_MAG_MAX,
            "reciprocity_complex_max": c["reciprocity_complex_max"],
            "reciprocity_complex_gate": G.RECIPROCITY_COMPLEX_MAX,
            "power_closure_max": c["power_closure_max"], "power_closure_gate": "report-only (WP3)",
            "gated": c["rung"] == G.CLAIMS_RUNG,
        }
    settling_ok = all(
        all(x <= G.SETTLING_DB_MAX for x in G.cell_settling_effective(c).values()) for c in cells)
    physics["settling_all_below_minus_40_db"] = settling_ok
    physics["claims_rung"] = G.CLAIMS_RUNG

    wall = (sum(c["wall_time_s"] for c in cells)
            + sum((c.get("settling_rerun") or {}).get("wall_time_s", 0.0) for c in cells)
            + sum(json.loads(p.read_text())["wall_time_s"] for p in sorted(out_dir.glob("ad_fd__*.json")))
            + sum(p_["wall_time_s"].get("total", 0.0) for k, p_ in planes.items() if k != "cheap_refute"))
    run_prov = dict(prov)
    # the measurement's own provenance comes from the per-case records
    run_prov.update({k: cells[0]["provenance"][k] for k in
                     ("commit", "run_id", "run_lane", "jax_version", "numpy_version",
                      "jax_default_backend", "jax_devices", "jax_enable_x64", "precision", "hostname")})
    if args.run_id != "local":
        run_prov["run_id"] = args.run_id
    if args.run_lane != "local":
        run_prov["run_lane"] = args.run_lane
    run_prov["wall_time_s"] = wall
    run_prov["wall_time_note"] = ("sum of the per-case solve wall times (cells incl. settling reruns, "
                                  "AD legs, FD pairs, shifted planes); JIT compile included")
    run_prov["recapture_command"] = (
        f"PYTHONPATH=. python {DRIVER} --out-dir <run-dir> --run-id <id> --run-lane <lane>; "
        f"then --stages assemble --fixture-out {ARTIFACT}")
    run_prov["recapture_entry_point"] = DRIVER
    if args.vessl_yaml:
        run_prov["recapture_vessl_yaml"] = args.vessl_yaml

    fx = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "predeclaration": PREDECLARATION, "predeclaration_sha": args.predeclaration_sha,
        "shift_pair_name": F.SHIFT_PAIR_NAME,
        "supersedes": SUPERSEDES, "supersedes_reason": SUPERSEDES_REASON,
        "readme": README,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "provenance": run_prov,
        "fixture": _fixture_constants(),
        "legs_rung": args.legs_rung,
        "cells": [{k: v for k, v in c.items() if k != "provenance"} for c in cells],
        "ladder": _ladder_block(cells),
        "plane_shift": {k: ({kk: vv for kk, vv in v.items() if kk != "provenance"} if k != "cheap_refute" else v)
                        for k, v in planes.items()},
        "ad_vs_fd": legs,
        "referee": referee,
        "physics_gates": physics,
        "port_cutoff": port_cutoff,
    }
    fx["verdicts"] = G.recompute_verdicts(fx)
    out = Path(args.fixture_out) if args.fixture_out else out_dir / "fixture.json"
    _write(out, fx)
    _log(f"assembled {out} ({len(cells)} cells, {len(legs)} AD/FD legs, "
         f"{len([k for k in planes if k != 'cheap_refute'])} plane-shift cases)")
    fails = {k: v for k, v in fx["verdicts"].items() if v == "fail"}
    ni = {k: v for k, v in fx["verdicts"].items() if v == "not_interpretable"}
    _log(f"verdicts: {len(fx['verdicts'])} total, {len(fails)} fail, {len(ni)} not_interpretable")
    for k in sorted(fails):
        _log(f"  FAIL {k}")
    for k in sorted(ni):
        _log(f"  NOT INTERPRETABLE {k}")
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stages", default="cells,ad_fd,plane_shift,assemble",
                    help="comma list of cells, ad_fd, plane_shift, assemble, pin (pin = the separate pin step)")
    ap.add_argument("--rungs", default=",".join(G.RUNG_LABELS), help="cells stage rungs")
    ap.add_argument("--legs-rung", default=G.LEGS_RUNG_DEFAULT, choices=G.RUNG_LABELS,
                    help="rung of the AD/FD and plane-shift legs (default: the claims rung)")
    ap.add_argument("--run-id", default="local")
    ap.add_argument("--run-lane", default="local")
    ap.add_argument("--git-sha", default=None)
    ap.add_argument("--predeclaration-sha", default="unknown")
    ap.add_argument("--fixture-out", default=None)
    ap.add_argument("--vessl-yaml", default=None, help="tracked YAML path recorded as the re-capture entry")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--refute-flip-shift-sign", action="store_true",
                    help="cheap refute (§8): plane-shift stage under a flipped shift sign")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance(args)
    _log(f"rfx from {rfx.__file__}; jax {jax.__version__} {jax.default_backend()} x64={jax.config.x64_enabled}")
    _log(f"commit {prov['commit']} run_id {prov['run_id']} lane {prov['run_lane']} out {out_dir}")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    t0 = time.time()
    if args.refute_flip_shift_sign:
        stage_plane_shift(args, out_dir, args.legs_rung, prov, refute=True)
        return 0
    if "cells" in stages:
        stage_cells(args, out_dir, rungs, prov)
    if "ad_fd" in stages:
        stage_ad_fd(args, out_dir, args.legs_rung, prov)
    if "plane_shift" in stages:
        stage_plane_shift(args, out_dir, args.legs_rung, prov)
    if "assemble" in stages:
        assemble(args, out_dir, prov)
    if "pin" in stages:
        # the separate pin step: read the assembled fixture, fill the pinned_*
        # fields from the measured envelopes, recompute the verdicts, write back
        target = Path(args.fixture_out) if args.fixture_out else out_dir / "fixture.json"
        fx = json.loads(target.read_text())
        fx = G.rebase_gradient_invariance_float32(fx)   # no-op once the plane stage writes float32 bases
        fx = G.pin_fixture(fx)
        _write(target, fx)
        _log(f"pinned {target}: {fx['pins']}")
    _log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
