"""Compare the declared uniform/graded y-wall cavity lanes.

Run with Python; stdout is JSONL, library output goes to stderr. Set
LANE_CHECK_SAVE (even to an empty value) to save each full float64 record
beside this script. Comparisons are emitted after both lengths: the verdict
is joint, because agreement requires both lengths to pass the declared bars.
Wall time includes scene construction, grid building, running and extraction.
"""

import contextlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

with contextlib.redirect_stdout(sys.stderr):
    from rfx import GaussianPulse, Simulation
    from rfx.ringdown import identify


def emit(row):
    print(json.dumps(row, allow_nan=False), flush=True)


def make_scene(lane):
    kwargs = {"dy_profile": np.full(12, .001)} if lane == "G" else {}
    sim = Simulation(freq_max=10e9, domain=(.030, .012, .024), dx=.001,
                     boundary="pec", snap="declared", **kwargs)
    sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, faces=("y_lo", "y_hi"))
    sim.add_source((.0073, .0041, .0059), "ey",
                   waveform=GaussianPulse(f0=8e9, bandwidth=.25),
                   amplitude_kind="current")
    sim.add_probe((.0217, .0083, .0151), "ey")
    return sim


def measure(sim, lane, record_ns, dt, n_steps, setup_s):
    started = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        result = sim.run(n_steps=n_steps, compute_s_params=False)
        series = np.asarray(result.time_series)
        model = identify(series, dt, int(np.ceil(2e-9 / dt)), len(series),
                         freq_max=10e9)
    pole = model.s[np.argmin(np.abs(model.s.imag / (2 * np.pi) - 7.998e9))]
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    if len(series) != n_steps or x.size != n_steps or not np.all(np.isfinite(x)):
        raise ValueError(f"{lane}: expected one finite probe record of {n_steps} steps")
    row = dict(kind="lane", pair="y", lane=lane, record_ns=record_ns,
               Q=float(pole.imag / (-2 * pole.real)),
               pole_frequency_hz=float(pole.imag / (2 * np.pi)),
               rank=int(model.rank), n_invalid=int(model.n_invalid),
               n_growing_discarded=int(model.n_growing_discarded),
               dt=dt, n_steps=n_steps, record_peak=float(np.max(np.abs(x))),
               wall_s=setup_s + time.monotonic() - started,
               uses_nonuniform_mesh=sim._uses_nonuniform_mesh)
    if "LANE_CHECK_SAVE" in os.environ:
        np.save(Path(__file__).resolve().with_name(
            f"lane_check_{lane}_{record_ns}ns.npy"),
            np.asarray(series, dtype=np.float64), allow_pickle=False)
    emit(row)
    return row, x


def compare(record_ns, rows, records, dt, n_steps):
    u, g = records["U"], records["G"]
    phase = np.exp(-1j * 2 * np.pi * rows["U"]["pole_frequency_hz"]
                   * np.arange(n_steps, dtype=np.float64) * dt)
    du, dg = np.sum(u * phase), np.sum(g * phase)
    peak = rows["U"]["record_peak"]
    if peak == 0 or abs(du) == 0:
        raise ValueError("U record peak or D_U is zero; comparison is undefined")
    return dict(kind="comparison", pair="y", record_ns=record_ns,
                dt=dt, n_steps=n_steps,
                dQ=rows["G"]["Q"] / rows["U"]["Q"] - 1,
                df=rows["G"]["pole_frequency_hz"]
                / rows["U"]["pole_frequency_hz"] - 1,
                per_step_relative_max=float(np.max(np.abs(u - g)) / peak),
                D_U={"real": float(du.real), "imag": float(du.imag)},
                D_G={"real": float(dg.real), "imag": float(dg.imag)},
                r=float(abs(du - dg) / abs(du)))


def verdict(comparisons):
    if all(c["r"] <= 1e-4 and abs(c["dQ"]) <= 2e-5 for c in comparisons):
        return ("the lanes agree; the residual sits in the reference or the Q "
                "extraction. Ledger line (magnitude, mesh-independence, uniform "
                "against graded, cause unmeasured); no issue.")
    if any(c["r"] > 1e-4 or abs(c["dQ"]) >= 1e-4 for c in comparisons):
        return ("the lanes assemble the wall-loss cavity differently; "
                "numbers to rfx-core-desk.")
    return "reported to the desk as not decided by this check."


def main():
    comparisons = []
    for record_ns in (20, 160):
        sims, dts, setup_times = {}, {}, {}
        for lane in ("U", "G"):
            started = time.monotonic()
            with contextlib.redirect_stdout(sys.stderr):
                sim = make_scene(lane)
                realized = sim._uses_nonuniform_mesh
            if lane == "G" and realized is not True:
                emit({"stop": "G is not routed to the graded runner: "
                      "sim._uses_nonuniform_mesh is not True"})
                return 3
            assert realized is (lane == "G"), f"{lane}: unexpected realized lane"
            with contextlib.redirect_stdout(sys.stderr):
                grid = (sim._build_nonuniform_grid() if lane == "G"
                        else sim._build_grid())
            sims[lane], dts[lane] = sim, float(grid.dt)
            setup_times[lane] = time.monotonic() - started
        if abs(dts["U"] / dts["G"] - 1) > 1e-12:
            emit({"stop": "lane timesteps differ", "dt_U": dts["U"],
                  "dt_G": dts["G"], "relative_difference": dts["U"] / dts["G"] - 1})
            return 4
        dt = dts["U"]
        n_steps = math.ceil(record_ns * 1e-9 / dt)
        rows, records = {}, {}
        for lane in ("U", "G"):
            rows[lane], records[lane] = measure(
                sims[lane], lane, record_ns, dts[lane], n_steps, setup_times[lane])
        comparisons.append(compare(record_ns, rows, records, dt, n_steps))
    decision = verdict(comparisons)
    for row in comparisons:
        row.update(verdict=decision, verdict_scope="both record lengths")
        emit(row)
    return 0


if __name__ == "__main__":
    try:
        status = main()
    except Exception as exc:
        emit({"stop": f"{type(exc).__name__}: {exc}"})
        status = 1
    sys.exit(status)
