"""THROWAWAY, never merged: what quick_convergence(sim, dx factor 1.0) returns
versus the user's own model run directly (predeclaration "The number for the
issue", sw-structure, 2026-11-16).

Each test PASSES when its measurement completes, whatever the outcome; an
exception from quick_convergence or from reading S is a recorded outcome.
Outputs: <worktree>/number_out/<scene>.json and .png.

Every geometric and numerical value below is a free choice for a synthetic
scene (no board from any campaign); the reason for each is in its comment.

Facts read from the code (origin/main 084b5032b):
- Lumped-port S from run(): ``Result.s_params`` (receive, drive, freq) with
  ``Result.freqs`` (rfx/runners/uniform.py:870-876; graded lumped default S
  via rfx/runners/_admission.py graded_lumped_default_s).
- MSL-port S is NEVER in ``Result.s_params``: run() drives MSL ports but forms
  no MSL S (rfx/api/_preflight.py:351 "add_msl_port(...) uses
  compute_msl_s_matrix(); run(...) does not include MSL ports in
  Result.s_params"). quick_convergence only calls run(), so for the MSL scene
  the observable compared is the spectrum of one Ez probe under the strip,
  and the user's MSL S11 is recorded on the direct side from
  ``sim.compute_msl_s_matrix(...).S[0, 0]`` for reference.
- convergence_study refuses fewer than two dx values before any run
  (rfx/convergence.py:306), so dx_factors=[1.0] raises ValueError. That is
  recorded; the factor-1.0 rebuild is then measured with dx_factors=[1.0, 1.0]
  and a metric that stores the first Result and raises to stop the second run.
"""
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

import numpy as np

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.convergence import quick_convergence

OUT = Path(__file__).resolve().parents[2] / "number_out"
C0 = 299792458.0


class _Captured(Exception):
    """Raised by the metric after it stored the rebuilt run's Result."""


# ---------------------------------------------------------------- helpers

def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else str(v)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return repr(value)


def _exc(err):
    return {"type": type(err).__name__, "message": str(err)[:300]}


def _db(mag):
    mag = np.asarray(mag, dtype=np.float64)
    return 20.0 * np.log10(np.maximum(mag, 1e-30))


def _grid_info(grid, dt=None):
    if grid is None:
        return None
    info = {"type": type(grid).__name__}
    shape = getattr(grid, "shape", None)
    if shape is None:
        shape = tuple(getattr(grid, a, None) for a in ("nx", "ny", "nz"))
    try:
        info["shape"] = [int(s) for s in shape]
    except Exception:
        info["shape"] = repr(shape)
    info["dt"] = float(dt if dt is not None else getattr(grid, "dt", float("nan")))
    for attr in ("cpml_layers", "dx"):
        val = getattr(grid, attr, None)
        if isinstance(val, (int, float, np.integer, np.floating)):
            info[attr] = float(val)
    return info


def _result_grid(res):
    if res is None:
        return None
    grid = getattr(res, "grid", None)
    return _grid_info(grid, getattr(res, "dt", None))


def _base_dx(sim):
    # As quick_convergence computes it (rfx/convergence.py:416-421).
    base = getattr(sim, "_dx", None)
    return {"sim._dx": base,
            "base_dx_used": base if base is not None else C0 / sim._freq_max / 20.0,
            "fallback_lambda_over_20": C0 / sim._freq_max / 20.0}


def _quick(build, n_steps, run_kwargs):
    """Run quick_convergence at factor 1.0; return (Result or None, record)."""
    record = {}
    # 1) Exactly as predeclared: dx_factors=[1.0].
    holder = {}

    def metric(result):
        holder["result"] = result
        raise _Captured()

    try:
        quick_convergence(build(), dx_factors=[1.0], metric=metric,
                          n_steps=n_steps, run_kwargs=dict(run_kwargs))
        record["single_factor"] = {"outcome": "returned"}
    except _Captured:
        record["single_factor"] = {"outcome": "ran; Result captured"}
    except Exception as err:  # recorded outcome
        record["single_factor"] = {"outcome": "raised", "exception": _exc(err),
                                   "result_produced": "result" in holder}
    if "result" in holder:
        return holder["result"], record
    # 2) Factor 1.0 through the same sim_factory, stopped after one run.
    try:
        quick_convergence(build(), dx_factors=[1.0, 1.0], metric=metric,
                          n_steps=n_steps, run_kwargs=dict(run_kwargs))
        record["factor_1_run"] = {"outcome": "returned without calling metric"}
    except _Captured:
        record["factor_1_run"] = {"outcome": "ran; Result captured"}
    except Exception as err:  # recorded outcome
        record["factor_1_run"] = {"outcome": "raised", "exception": _exc(err),
                                  "traceback_tail": traceback.format_exc()[-600:],
                                  "result_produced": "result" in holder}
    return holder.get("result"), record


def _s11_db(res):
    """|S11| dB from Result.s_params[0, 0, :] over Result.freqs, or None."""
    if res is None or getattr(res, "s_params", None) is None:
        return None, None
    s = np.asarray(res.s_params)
    return np.asarray(res.freqs, dtype=np.float64), _db(np.abs(s[0, 0, :]))


def _probe_db(res, freqs):
    """|DFT| dB of probe 0's record at freqs (time_series[:, 0], step Result.dt)."""
    if res is None or getattr(res, "time_series", None) is None:
        return None
    ts = np.asarray(res.time_series, dtype=np.float64)
    if ts.ndim != 2 or ts.shape[1] < 1:
        return None
    dt = float(getattr(res, "dt", None) or res.grid.dt)
    t = (np.arange(ts.shape[0]) + 1) * dt
    spec = np.exp(-2j * np.pi * np.outer(freqs, t)) @ ts[:, 0] * dt
    return _db(np.abs(spec))


def _max_delta(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    d = np.abs(np.asarray(a) - np.asarray(b))
    d = d[np.isfinite(d)]
    return float(d.max()) if d.size else None


def _plot(scene, freqs, curves, ylabel, note):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as err:
        note["figure"] = f"matplotlib unavailable: {err!r}"
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, (f, y, style) in curves.items():
        if f is not None and y is not None:
            ax.plot(np.asarray(f) / 1e9, y, style, label=label)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(scene)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{scene}.png", dpi=130)
    plt.close(fig)
    note["figure"] = str(OUT / f"{scene}.png")


def _write(scene, record):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{scene}.json"
    path.write_text(json.dumps(_safe(record), indent=1, allow_nan=False) + "\n")
    return path


# ------------------------------------------------- scene 1: MSL through line
# Uniform mesh, constructed like tests/oracle/test_msl_probe_offset_spread.py
# (ground = PEC z-lo face, substrate box, PEC strip box ending on the ports'
# nodes, add_msl_port feeds at z=0) because the tutorial's microstrip uses a
# graded xy mesh and the brief asks for a uniform one.
MSL = dict(
    eps_r=3.0,        # a common low-loss laminate value
    h=0.9e-3,         # substrate height; 3 cells
    cell=0.3e-3,      # h/3, the coarsest cells/h the oracle uses (3); keeps ~54k cells
    w=2.1e-3,         # ~2.3 h, near 50 ohm for eps_r 3 (Hammerstad), 7 cells
    length=21.0e-3,   # 70 cells; leaves 16.2 mm between feeds for probes
    air=1.8e-3,       # side and top air, 6 cells (small to stay near 60k cells)
    f_max=10e9,       # 30+ cells per guided wavelength at f_max
    cpml=8,           # half the oracle's 16 to stay small; advisory only
    feed_inset=2.4e-3,  # 8 cells from each line end
    record_s=4e-9,    # ~2.7 transits of the 21 mm line at eps_eff ~2.4, x ~10
)
MSL_FREQS = np.linspace(2e9, 10e9, 33)


def _build_msl():
    p = MSL
    domain = (p["length"], 2 * p["air"] + p["w"], p["h"] + p["air"])
    sim = Simulation(
        freq_max=p["f_max"], domain=domain, dx=p["cell"], cpml_layers=p["cpml"],
        snap="declared",
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("substrate", eps_r=p["eps_r"])
    sim.add(Box((0, 0, 0), (p["length"], domain[1], p["h"])), material="substrate")
    from rfx.preflight.line_port_coverage import port_node_coordinate
    yc = p["air"] + p["w"] / 2
    x1, x2 = p["feed_inset"], p["length"] - p["feed_inset"]
    ends = [port_node_coordinate(sim, (x, yc, 0)) for x in (x1, x2)]
    sim.add(Box((ends[0], p["air"], p["h"]), (ends[1], p["air"] + p["w"], p["h"])),
            material="pec")
    # Explicit probe layout: the automatic spacing (~lambda_g/4 per step)
    # would need more line than 16 mm. Offset 4 cells, spacing 3 cells.
    for name, x, direction in (("p1", x1, "+x"), ("p2", x2, "-x")):
        sim.add_msl_port(position=(x, yc, 0), width=p["w"], height=p["h"],
                         eps_r_sub=p["eps_r"], impedance=50.0,
                         direction=direction, name=name,
                         n_probes=3, n_probe_offset=4, n_probe_spacing=3)
    # One Ez probe mid-substrate under the strip, off the line's centre
    # (0.55 L), so the run() record has an observable on both sides.
    sim.add_probe((0.55 * p["length"], yc, p["h"] / 2), "ez")
    return sim


def test_msl_through():
    scene = "msl_through"
    rec = {"scene": scene, "parameters": MSL, "band_hz": MSL_FREQS,
           "observable": "|DFT| of probe 0 (Ez under strip) in dB; run() forms no MSL S"}
    try:
        src = _build_msl()
        rec["base_dx"] = _base_dx(src)
        g = src._build_realized_grid()
        rec["source_realized_grid"] = _grid_info(g)
        n_steps = int(np.ceil(MSL["record_s"] / float(g.dt)))
    except Exception as err:
        rec["build_exception"] = _exc(err)
        _write(scene, rec)
        return
    run_kwargs = {}
    rec["run_kwargs"] = {"n_steps": n_steps, **run_kwargs}

    # (a) direct
    direct = None
    try:
        direct = _build_msl().run(n_steps=n_steps, **run_kwargs)
    except Exception as err:
        rec["direct_exception"] = _exc(err)
    rec["direct_grid"] = _result_grid(direct)
    rec["direct_result_s_params_is_none"] = (
        None if direct is None else direct.s_params is None)
    d_db = _probe_db(direct, MSL_FREQS)
    rec["direct_probe_db"] = d_db

    # reference: the user's MSL S11 (direct model only; quick_convergence cannot form it)
    msl_s11 = None
    try:
        ms = _build_msl().compute_msl_s_matrix(
            n_steps=n_steps, freqs=MSL_FREQS.astype(np.float32))
        msl_s11 = _db(np.abs(np.asarray(ms.S)[0, 0, :]))
        rec["direct_compute_msl_s_matrix"] = {
            "S11_db": msl_s11,
            "settling_db": getattr(ms, "settling_db", None)}
    except Exception as err:
        rec["direct_compute_msl_s_matrix"] = {"exception": _exc(err)}

    # (b) quick
    quick, qrec = _quick(_build_msl, n_steps, run_kwargs)
    rec["quick"] = qrec
    rec["quick_grid"] = _result_grid(quick)
    rec["quick_result_s_params_is_none"] = (
        None if quick is None else quick.s_params is None)
    q_db = _probe_db(quick, MSL_FREQS)
    rec["quick_probe_db"] = q_db
    rec["quick_probe_peak_abs"] = (
        None if quick is None or quick.time_series is None
        else float(np.max(np.abs(np.asarray(quick.time_series)))))
    rec["direct_probe_peak_abs"] = (
        None if direct is None or direct.time_series is None
        else float(np.max(np.abs(np.asarray(direct.time_series)))))
    rec["max_abs_delta_db"] = _max_delta(d_db, q_db)

    _plot(scene, MSL_FREQS, {
        "direct run(): probe |DFT|": (MSL_FREQS, d_db, "-"),
        "quick_convergence x1.0: probe |DFT|": (MSL_FREQS, q_db, "--"),
        "direct compute_msl_s_matrix |S11|": (MSL_FREQS, msl_s11, ":"),
    }, "dB", rec)
    assert _write(scene, rec).exists()


# ------------------------------------- scene 2: graded slab + lumped port
# Constructed like tests/unit/runners/test_distributed_graded_run.py
# (dx=1 mm boundary cell, dx_profile = uniform ends around a 0.85-1.15 mm
# linspace, snap="declared").
GRADED_PROFILE = np.r_[np.full(3, 1e-3), np.linspace(0.85e-3, 1.15e-3, 14),
                       np.full(3, 1e-3)]  # sum 20 mm; ratio range as the fixture
GR = dict(
    dx=1e-3,            # boundary cell; y and z uniform at dx
    f_max=15e9,         # dx = lambda/20 at f_max
    yz=12e-3,           # 12 cells across y and z
    cpml=8,
    eps_r=4.0,
    slab_offset=0.37e-3,  # slab starts 0.37 mm past x node 11: off any node
    slab_thick=3.3e-3,    # 3.3 mm: not a whole number of cells
    port_node=6,          # lumped port on x node 6 (~5.6 mm), 5+ mm before the slab
    port_yz=(5.0e-3, 7.0e-3),  # off the y and z centre planes
    f0=9e9, bandwidth=0.9,     # pulse covering ~4-14 GHz
    record_s=5e-9,
)
GR_FREQS = np.linspace(4e9, 14e9, 21).astype(np.float32)


def _build_graded():
    p = GR
    nodes = np.r_[0.0, np.cumsum(GRADED_PROFILE)]
    sim = Simulation(
        freq_max=p["f_max"], domain=(float(nodes[-1]), p["yz"], p["yz"]),
        dx=p["dx"], dx_profile=GRADED_PROFILE, boundary="cpml",
        cpml_layers=p["cpml"], snap="declared",
    )
    sim.add_material("slab", eps_r=p["eps_r"])
    x0 = float(nodes[11]) + p["slab_offset"]
    sim.add(Box((x0, 0, 0), (x0 + p["slab_thick"], p["yz"], p["yz"])), material="slab")
    sim.add_port((float(nodes[p["port_node"]]), *p["port_yz"]), component="ez",
                 impedance=50.0,
                 waveform=GaussianPulse(f0=p["f0"], bandwidth=p["bandwidth"]))
    return sim


def test_graded_lumped():
    scene = "graded_lumped"
    rec = {"scene": scene, "parameters": GR, "dx_profile": GRADED_PROFILE,
           "band_hz": GR_FREQS, "observable": "|S11| dB from Result.s_params[0,0]"}
    try:
        src = _build_graded()
        rec["base_dx"] = _base_dx(src)
        rec["source_dx_profile_carried_by_factory"] = False  # read: rfx/convergence.py:448-470
        g = src._build_realized_grid()
        rec["source_realized_grid"] = _grid_info(g)
        n_steps = int(np.ceil(GR["record_s"] / float(g.dt)))
    except Exception as err:
        rec["build_exception"] = _exc(err)
        _write(scene, rec)
        return
    run_kwargs = {"s_param_freqs": GR_FREQS}
    rec["run_kwargs"] = {"n_steps": n_steps, **run_kwargs}

    direct = None
    try:
        direct = _build_graded().run(n_steps=n_steps, **run_kwargs)
    except Exception as err:
        rec["direct_exception"] = _exc(err)
    rec["direct_grid"] = _result_grid(direct)
    df, d_db = _s11_db(direct)
    rec["direct_freqs_hz"], rec["direct_s11_db"] = df, d_db

    quick, qrec = _quick(_build_graded, n_steps, run_kwargs)
    rec["quick"] = qrec
    rec["quick_grid"] = _result_grid(quick)
    try:
        qf, q_db = _s11_db(quick)
    except Exception as err:
        qf, q_db = None, None
        rec["quick_read_exception"] = _exc(err)
    rec["quick_freqs_hz"], rec["quick_s11_db"] = qf, q_db
    rec["max_abs_delta_db"] = _max_delta(d_db, q_db)
    _plot(scene, GR_FREQS, {
        "direct run() (graded x)": (df, d_db, "-"),
        "quick_convergence x1.0": (qf, q_db, "--"),
    }, "|S11| (dB)", rec)
    assert _write(scene, rec).exists()
