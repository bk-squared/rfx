"""Acceptance gate: production-scan driver V/I dump == eager extractor dump.

Item-5 Stage 2 (2026-06-22) adds a ``return_vi_dump=True`` path to
``rfx.probes.sparam_driver.compute_lumped_wire_s_matrix_via_scan`` that builds
the SAME replay bundle the eager extractors produce
(``PortVIReplayBundle`` for all-lumped, ``WirePortVIReplayBundle`` for
all-wire), populated from the driver's per-drive V/I accumulators.

This is a PURE ADD: the eager ``return_vi_dump`` path is untouched.  The gate is
that the driver bundle is *byte-close* to the eager bundle on CPML (where the
production scan and the eager loop agree) AND that the driver bundle is
**replay-faithful** — feeding it through the public replay path
(``save_port_vi_dump_npz`` → ``load_port_vi_dump_npz`` →
``replay_smatrix_from_port_vi_dump`` for lumped; the wire replay diagnostic for
wire) reproduces the driver's own ``s_params``.  That proves Stage 3 can delete
the eager dump path.

Tolerances are deliberately tight (atol 2e-3 CPML byte-closeness).  A tolerance
bump to pass is a failure, not a fix (R2-STOP).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from rfx import (
    PortDumpMetadata,
    compare_replayed_smatrix,
    load_port_vi_dump_npz,
    replay_smatrix_from_port_vi_dump,
    save_port_vi_dump_npz,
)
from rfx.api import Simulation
from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.probes.probes import extract_s_matrix, extract_s_matrix_wire
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
from rfx.sources.sources import GaussianPulse, LumpedPort, WirePort

REPO_ROOT = Path(__file__).resolve().parents[1]

# Small shared geometry (matches the Stage-1 gate test).
_DOMAIN = (0.024, 0.012, 0.012)
_DX = 1e-3
_FREQ_MAX = 10e9
_CPML_LAYERS = 8
_FREQS = np.linspace(2e9, 9e9, 15)
_N_STEPS = 1500
_GATE_ATOL = 2e-3  # CPML byte-closeness


def _waveform():
    return GaussianPulse(f0=5.5e9, bandwidth=4e9)


def _eager_grid():
    return Grid(freq_max=_FREQ_MAX, domain=_DOMAIN, dx=_DX,
                cpml_layers=_CPML_LAYERS)


def _driver_sim():
    return Simulation(freq_max=_FREQ_MAX, domain=_DOMAIN, dx=_DX,
                      boundary="cpml", cpml_layers=_CPML_LAYERS)


def _load_replay_wire_module():
    path = REPO_ROOT / "scripts" / "diagnostics" / "replay_wire_port_vi_dump.py"
    spec = importlib.util.spec_from_file_location("replay_wire_port_vi_dump", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_wire_port_vi_dump"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Lumped 2-port (CPML) — dump parity + end-to-end replay
# ---------------------------------------------------------------------------

def test_driver_lumped_dump_matches_eager_and_replays():
    wf = _waveform()
    pos0 = (0.008, 0.006, 0.006)
    pos1 = (0.016, 0.006, 0.006)

    sim = _driver_sim()
    sim.add_port(position=pos0, component="ez", impedance=50.0, waveform=wf)
    sim.add_port(position=pos1, component="ez", impedance=50.0, waveform=wf)
    drv = compute_lumped_wire_s_matrix_via_scan(
        sim, _FREQS, n_steps=_N_STEPS, return_vi_dump=True)

    g = _eager_grid()
    mats = init_materials(g.shape)
    ports = [
        LumpedPort(position=pos0, component="ez", impedance=50.0, excitation=wf),
        LumpedPort(position=pos1, component="ez", impedance=50.0, excitation=wf),
    ]
    eager = extract_s_matrix(
        g, mats, ports, _FREQS, _N_STEPS, boundary="cpml", cpml_axes="xyz",
        return_vi_dump=True)

    # --- field names / order identical (mirror the eager NamedTuple) ---
    assert type(drv).__name__ == "PortVIReplayBundle"
    assert drv._fields == eager._fields == (
        "s_params", "freqs", "voltages", "currents",
        "port_impedances", "port_names", "driven_port_indices",
    )

    # --- shapes identical, (n_driven, n_ports, n_freqs) ---
    n = len(_FREQS)
    assert np.asarray(drv.voltages).shape == np.asarray(eager.voltages).shape == (2, 2, n)
    assert np.asarray(drv.currents).shape == np.asarray(eager.currents).shape == (2, 2, n)
    assert np.asarray(drv.s_params).shape == np.asarray(eager.s_params).shape == (2, 2, n)

    # --- metadata fields equal exactly ---
    assert drv.port_names == eager.port_names == ("port_0", "port_1")
    assert drv.driven_port_indices == eager.driven_port_indices == (0, 1)
    np.testing.assert_array_equal(
        np.asarray(drv.port_impedances), np.asarray(eager.port_impedances))

    # --- V / I / S byte-close on CPML ---
    d_v = float(np.max(np.abs(np.asarray(drv.voltages) - np.asarray(eager.voltages))))
    d_i = float(np.max(np.abs(np.asarray(drv.currents) - np.asarray(eager.currents))))
    d_s = float(np.max(np.abs(np.asarray(drv.s_params) - np.asarray(eager.s_params))))
    print(f"[lumped] driver-vs-eager max|dV|={d_v:.3e} max|dI|={d_i:.3e} "
          f"max|dS|={d_s:.3e}")
    assert d_v < _GATE_ATOL, f"lumped dump voltages diverged: {d_v:.3e}"
    assert d_i < _GATE_ATOL, f"lumped dump currents diverged: {d_i:.3e}"
    assert d_s < _GATE_ATOL, f"lumped dump s_params diverged: {d_s:.3e}"

    # --- END-TO-END REPLAY of the DRIVER bundle through the public path ---
    tmp = Path(REPO_ROOT) / ".pytest_lumped_dump_parity.npz"
    try:
        save_port_vi_dump_npz(
            tmp,
            voltages=drv.voltages,
            currents=drv.currents,
            freqs=np.asarray(drv.freqs),
            port_impedances=drv.port_impedances,
            metadata=PortDumpMetadata(
                commit_hash="stage2-driver",
                geometry={"kind": "two_port_lumped_cpml_driver"},
            ),
            port_names=drv.port_names,
            driven_port_indices=drv.driven_port_indices,
            production_smatrix=np.asarray(drv.s_params),
        )
        dump = load_port_vi_dump_npz(tmp)
        replayed = replay_smatrix_from_port_vi_dump(dump)
        comparison = compare_replayed_smatrix(
            replayed,
            type("P", (), {"s_params": dump.production_smatrix,
                           "freqs": dump.freqs})(),
        )
        print(f"[lumped] replay max_abs_diff={comparison.max_abs_diff:.3e} "
              f"allowed={comparison.max_allowed:.3e} ok={comparison.ok}")
        assert comparison.ok, comparison.summary()
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Wire 2-port (CPML) — dump parity + end-to-end replay
# ---------------------------------------------------------------------------

def test_driver_wire_dump_matches_eager_and_replays():
    wf = _waveform()
    start0 = (0.008, 0.006, 0.005)
    end0 = (0.008, 0.006, 0.008)
    start1 = (0.016, 0.006, 0.005)
    end1 = (0.016, 0.006, 0.008)
    extent = end0[2] - start0[2]

    sim = _driver_sim()
    sim.add_port(position=start0, component="ez", impedance=50.0,
                 waveform=wf, extent=extent)
    sim.add_port(position=start1, component="ez", impedance=50.0,
                 waveform=wf, extent=extent)
    drv = compute_lumped_wire_s_matrix_via_scan(
        sim, _FREQS, n_steps=_N_STEPS, return_vi_dump=True)

    g = _eager_grid()
    mats = init_materials(g.shape)
    wports = [
        WirePort(start=start0, end=end0, component="ez", impedance=50.0,
                 excitation=wf),
        WirePort(start=start1, end=end1, component="ez", impedance=50.0,
                 excitation=wf),
    ]
    eager = extract_s_matrix_wire(
        g, mats, wports, _FREQS, _N_STEPS, boundary="cpml", cpml_axes="xyz",
        return_vi_dump=True)

    # --- field names / order identical (mirror the eager NamedTuple) ---
    assert type(drv).__name__ == "WirePortVIReplayBundle"
    assert drv._fields == eager._fields == (
        "s_params", "freqs", "raw_voltages_fdt", "raw_currents",
        "port_impedances", "port_cell_counts", "port_names",
        "driven_port_indices",
    )

    # --- shapes identical, (n_driven, n_ports, n_freqs) ---
    n = len(_FREQS)
    assert np.asarray(drv.raw_voltages_fdt).shape == \
        np.asarray(eager.raw_voltages_fdt).shape == (2, 2, n)
    assert np.asarray(drv.raw_currents).shape == \
        np.asarray(eager.raw_currents).shape == (2, 2, n)
    assert np.asarray(drv.s_params).shape == np.asarray(eager.s_params).shape == (2, 2, n)

    # --- metadata fields equal exactly ---
    assert drv.port_names == eager.port_names == ("wire_0", "wire_1")
    assert drv.driven_port_indices == eager.driven_port_indices == (0, 1)
    np.testing.assert_array_equal(
        np.asarray(drv.port_impedances), np.asarray(eager.port_impedances))
    np.testing.assert_array_equal(
        np.asarray(drv.port_cell_counts), np.asarray(eager.port_cell_counts))

    # --- V / I / S byte-close on CPML ---
    d_v = float(np.max(np.abs(
        np.asarray(drv.raw_voltages_fdt) - np.asarray(eager.raw_voltages_fdt))))
    d_i = float(np.max(np.abs(
        np.asarray(drv.raw_currents) - np.asarray(eager.raw_currents))))
    d_s = float(np.max(np.abs(np.asarray(drv.s_params) - np.asarray(eager.s_params))))
    print(f"[wire] driver-vs-eager max|dV|={d_v:.3e} max|dI|={d_i:.3e} "
          f"max|dS|={d_s:.3e}")
    assert d_v < _GATE_ATOL, f"wire dump raw_voltages_fdt diverged: {d_v:.3e}"
    assert d_i < _GATE_ATOL, f"wire dump raw_currents diverged: {d_i:.3e}"
    assert d_s < _GATE_ATOL, f"wire dump s_params diverged: {d_s:.3e}"

    # --- END-TO-END REPLAY of the DRIVER bundle through the wire diagnostic ---
    replay_wire = _load_replay_wire_module()
    tmp = Path(REPO_ROOT) / ".pytest_wire_dump_parity.npz"
    try:
        np.savez(
            tmp,
            metadata_json=np.asarray(json.dumps({"schema": "rfx.wire_port_vi_dump"})),
            freqs_hz=np.asarray(drv.freqs),
            raw_voltages_fdt=np.asarray(drv.raw_voltages_fdt),
            raw_currents=np.asarray(drv.raw_currents),
            port_impedances_ohm=np.asarray(drv.port_impedances),
            port_cell_counts=np.asarray(drv.port_cell_counts),
            production_smatrix=np.asarray(drv.s_params),
            port_names=np.asarray(drv.port_names, dtype=object),
            driven_port_indices=np.asarray(drv.driven_port_indices),
        )
        payload = replay_wire.replay_wire_port_vi_dump(tmp)
        print(f"[wire] replay status={payload['status']} "
              f"max_abs_diff={payload['max_abs_diff']:.3e} "
              f"allowed={payload['max_allowed']:.3e}")
        assert payload["status"] == "passed", payload
        assert payload["max_abs_diff"] <= payload["max_allowed"]
    finally:
        tmp.unlink(missing_ok=True)
