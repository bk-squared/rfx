"""Matched-line magnitudes should be independent of the first-probe offset.

Synthetic lossless lines, at their declared coarse meshes. This physical
invariance check follows test_msl_thru_line_passive_gate's top-level API.
Both ports are driven by compute_msl_s_matrix; default extraction and probe
spacing are retained. S11 and S21 have a 1e-2 spread bar; S22 and S12 are
reported witnesses. No expected-failure classification precedes measurement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.preflight.line_port_coverage import port_node_coordinate

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# eps_r, substrate height, strip width, length, air, f_max, record, cells/h.
LINES = {
    "L1": (2.2, .8e-3, 2.43e-3, .060, .010, 12e9, 30e-9, 4),
    "L2": (3.66, .254e-3, .6e-3, .030, .004, 40e9, 10e-9, 3),
    "L3": (10.2, 1.27e-3, 1.2e-3, .060, .010, 12e9, 30e-9, 4),
}
OFFSETS = (5, 12, 25)


def build_line(line, m):
    er, h, w, length, air, fmax, _, k = LINES[line]
    cell = h / k
    domain = (length, 2 * air + w, h + air)
    sim = Simulation(
        freq_max=fmax, domain=domain, dx=cell, cpml_layers=16,
        snap="declared", boundary=BoundarySpec(
            x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("substrate", eps_r=er)
    sim.add(Box((0, 0, 0), (length, domain[1], h)), material="substrate")
    ends = [port_node_coordinate(sim, (x, air + w / 2, 0))
            for x in (.0025, length - .0025)]
    sim.add(Box((ends[0], air, h), (ends[1], air + w, h)), material="pec")
    for name, x, direction in (("p1", .0025, "+x"),
                               ("p2", length - .0025, "-x")):
        sim.add_msl_port(
            position=(x, air + w / 2, 0), width=w, height=h,
            eps_r_sub=er, impedance=50, direction=direction, name=name,
            n_probes=5, n_probe_offset=round(m * h / cell),
        )
    return sim


def measure_offset(line, m, *, record_fraction=1.0):
    """The sizing driver may shorten the record; the test always uses 1.0."""
    sim = build_line(line, m)
    dt = float(sim._build_realized_grid().dt)
    record_s = LINES[line][6] * record_fraction
    n_steps = int(np.ceil(record_s / dt))
    freqs = np.linspace(1e9, LINES[line][5], 52, dtype=np.float32)
    result = sim.compute_msl_s_matrix(n_steps=n_steps, freqs=freqs)
    s = np.asarray(result.S)
    settling = np.asarray(result.settling_db, dtype=float)
    matrices = s.transpose(2, 0, 1)
    finite = np.isfinite(matrices).all(axis=(1, 2))
    sigma = np.full(len(freqs), np.nan)
    sigma[finite] = np.linalg.svd(
        matrices[finite].astype(np.complex128), compute_uv=False)[:, 0]
    settled = bool(np.all(~np.isnan(settling) & (settling <= -40)))
    return s, {
        "offset_h": m, "n_steps_per_drive": n_steps, "drives": 2,
        "record_s": record_s, "realized_record_s": n_steps * dt,
        "freqs_hz": freqs.tolist(), "settling_db": settling.tolist(),
        "magnitudes": {
            name: np.abs(s[i, j]).tolist()
            for name, i, j in (("S11", 0, 0), ("S21", 1, 0),
                               ("S22", 1, 1), ("S12", 0, 1))
        },
        "settling_db_worst": float(np.max(settling)),
        "sigma_max": sigma.tolist(), "sigma_max_worst": float(np.max(sigma)),
        "sigma_max_excess": np.asarray(result.sigma_max_excess).tolist(),
        "assembly": result.assembly,
        "status": "settled" if settled else "cannot judge",
    }


def _json_safe(value):
    # Preserve non-finite witness meanings without emitting non-standard JSON.
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _report(report):
    text = json.dumps(_json_safe(report), allow_nan=False, sort_keys=True)
    print(text, file=sys.stderr, flush=True)
    if destination := os.environ.get("MSL_SPREAD_OUT"):
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{report['line']}.json").write_text(text + "\n")


@pytest.mark.parametrize("line", tuple(LINES))
def test_msl_probe_offset_spread(line, monkeypatch):
    # The L1 and L2 witnesses need 5.39 GiB at stage 2.
    monkeypatch.setattr("rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES", 8 * 2**30)
    measurements = [measure_offset(line, m) for m in OFFSETS]
    report = {"line": line, "mesh_cells_per_h": LINES[line][7],
              "runs": [row for _, row in measurements], "spreads": {}}
    if any(row["status"] != "settled" for _, row in measurements):
        report["status"] = "cannot judge"
        _report(report)
        pytest.skip("cannot judge: at least one offset did not settle to -40 dB")
    magnitudes = np.abs(np.stack([s for s, _ in measurements]))
    if not np.isfinite(magnitudes).all():
        report["status"] = "non-finite S"
        _report(report)
        pytest.fail("non-finite S: spread is undefined")
    spreads = np.ptp(magnitudes, axis=0)
    freqs = measurements[0][1]["freqs_hz"]
    for name, i, j in (("S11", 0, 0), ("S21", 1, 0),
                       ("S22", 1, 1), ("S12", 0, 1)):
        values = spreads[i, j]
        index = int(np.argmax(values))
        report["spreads"][name] = {
            "largest": float(values[index]), "frequency_hz": freqs[index],
            "per_bin": values.tolist(),
        }
    failures = [name for name in ("S11", "S21")
                if report["spreads"][name]["largest"] > 1e-2]
    report["status"] = "fail" if failures else "pass"
    _report(report)
    assert not failures, f"{line}: spread exceeds 1e-2 for {', '.join(failures)}"
