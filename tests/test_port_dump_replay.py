"""Tests for independent V/I dump replay of port S-matrices.

These tests are algebraic E3-harness checks.  They do not run FDTD and do not
claim additional solver physics validation by themselves.
"""

from __future__ import annotations

import importlib.util
import json
import numpy as np
from pathlib import Path
import sys

from rfx import (
    PortDumpMetadata,
    compare_replayed_smatrix,
    load_port_vi_dump_npz,
    replay_smatrix_from_port_vi_dump,
    replay_smatrix_from_vi_dump,
    save_port_vi_dump_npz,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_replay_msl_module():
    path = REPO_ROOT / "scripts" / "diagnostics" / "replay_msl_3probe_dump.py"
    spec = importlib.util.spec_from_file_location("replay_msl_3probe_dump", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_msl_3probe_dump"] = module
    spec.loader.exec_module(module)
    return module


def _load_replay_wire_module():
    path = REPO_ROOT / "scripts" / "diagnostics" / "replay_wire_port_vi_dump.py"
    spec = importlib.util.spec_from_file_location("replay_wire_port_vi_dump", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_wire_port_vi_dump"] = module
    spec.loader.exec_module(module)
    return module


def _load_lumped_oracle_module():
    path = REPO_ROOT / "scripts" / "diagnostics" / "report_lumped_analytic_oracles.py"
    spec = importlib.util.spec_from_file_location("report_lumped_analytic_oracles", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["report_lumped_analytic_oracles"] = module
    spec.loader.exec_module(module)
    return module


def _load_lumped_sweep_module():
    path = REPO_ROOT / "scripts" / "diagnostics" / "report_lumped_replay_sweep.py"
    spec = importlib.util.spec_from_file_location("report_lumped_replay_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["report_lumped_replay_sweep"] = module
    spec.loader.exec_module(module)
    return module


def _load_wire_sweep_module():
    path = REPO_ROOT / "scripts" / "diagnostics" / "report_wire_replay_sweep.py"
    spec = importlib.util.spec_from_file_location("report_wire_replay_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["report_wire_replay_sweep"] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_two_port_dump():
    freqs = np.array([1.0e9, 2.0e9, 3.0e9])
    z0 = np.array([50.0, 75.0])
    s = np.zeros((2, 2, freqs.size), dtype=np.complex128)
    s[0, 0, :] = [0.10 + 0.02j, 0.08 + 0.01j, 0.05 + 0.00j]
    s[1, 0, :] = [0.85 - 0.03j, 0.82 - 0.04j, 0.78 - 0.06j]
    s[0, 1, :] = [0.83 - 0.02j, 0.80 - 0.03j, 0.76 - 0.04j]
    s[1, 1, :] = [0.07 + 0.01j, 0.06 + 0.02j, 0.04 + 0.01j]

    a = np.zeros((2, 2, freqs.size), dtype=np.complex128)
    b = np.zeros_like(a)
    for driven in range(2):
        incident = (1.0 + 0.1 * driven) * np.exp(1j * np.linspace(0.0, 0.3, freqs.size))
        a[driven, driven, :] = incident
        b[driven, :, :] = s[:, driven, :] * incident.reshape(1, -1)

    sqrt_z = np.sqrt(z0).reshape(1, 2, 1)
    voltages = sqrt_z * (a + b)
    currents = (a - b) / sqrt_z
    # Role-selected dump convention (issue #308): production dumps register
    # the arriving wave at a PASSIVE receive port with the opposite voltage
    # sign relative to the textbook split (the port-cell field sense in
    # V = -E*dx), so the replay's receive channel is b_recv = -(V + Z0*I) /
    # (2*sqrt(Z0)).  Encode that here: flip the voltage sign on off-diagonal
    # (receive) entries; currents already match ((a - b)/sqrt(Z0) with a=0).
    for driven in range(2):
        voltages[driven, 1 - driven, :] *= -1.0
    return freqs, z0, s, voltages, currents


def test_replay_smatrix_from_vi_phasors_matches_known_smatrix():
    freqs, z0, production, voltages, currents = _synthetic_two_port_dump()

    replayed = replay_smatrix_from_vi_dump(
        voltages,
        currents,
        freqs=freqs,
        port_impedances=z0,
        port_names=("in", "out"),
    )

    assert replayed.port_names == ("in", "out")
    np.testing.assert_allclose(replayed.s_params, production, rtol=1e-12, atol=1e-12)
    comparison = compare_replayed_smatrix(
        replayed,
        type("Production", (), {"s_params": production, "freqs": freqs})(),
        atol=1e-12,
        rtol=1e-12,
    )
    assert comparison.ok, comparison.summary()


def test_replay_accepts_positive_out_current_convention():
    freqs, z0, production, voltages, currents = _synthetic_two_port_dump()

    replayed = replay_smatrix_from_vi_dump(
        voltages,
        -currents,
        freqs=freqs,
        port_impedances=z0,
        current_convention="positive_out_of_dut",
    )

    np.testing.assert_allclose(replayed.s_params, production, rtol=1e-12, atol=1e-12)


def test_replay_dump_npz_roundtrip_includes_metadata_and_production(tmp_path):
    freqs, z0, production, voltages, currents = _synthetic_two_port_dump()
    path = tmp_path / "port_dump.npz"
    metadata = PortDumpMetadata(
        commit_hash="test-commit",
        geometry={"kind": "synthetic_thru"},
        grid={"dx_m": 1e-3},
        port_definitions=({"name": "in"}, {"name": "out"}),
        dt_s=1e-12,
        frequency_grid_hz=tuple(float(f) for f in freqs),
    )

    save_port_vi_dump_npz(
        path,
        voltages=voltages,
        currents=currents,
        freqs=freqs,
        port_impedances=z0,
        metadata=metadata,
        port_names=("in", "out"),
        production_smatrix=production,
    )

    dump = load_port_vi_dump_npz(path)
    assert dump.metadata["schema"] == "rfx.port_vi_dump"
    assert dump.metadata["commit_hash"] == "test-commit"
    assert dump.port_names == ("in", "out")
    replayed = replay_smatrix_from_port_vi_dump(dump)
    np.testing.assert_allclose(replayed.s_params, production, rtol=1e-12, atol=1e-12)
    assert dump.production_smatrix is not None
    assert compare_replayed_smatrix(replayed, type("P", (), {"s_params": dump.production_smatrix, "freqs": dump.freqs})()).ok


def test_lumped_extract_s_matrix_can_emit_replayable_real_vi_dump(tmp_path):
    """Production lumped-port extraction can save raw V/I replay phasors.

    This is a small real FDTD smoke test for the dump plumbing.  The short run
    is not a broad physics claim; it verifies that the saved sign convention
    independently replays the production S-matrix.
    """
    import jax.numpy as jnp

    from rfx.core.yee import init_materials
    from rfx.grid import Grid
    from rfx.probes.probes import extract_s_matrix
    from rfx.sources.sources import GaussianPulse, LumpedPort

    grid = Grid(freq_max=2.0e9, domain=(0.03, 0.02, 0.015), dx=5.0e-3, cpml_layers=0)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=1.0e9, bandwidth=0.8)
    ports = [
        LumpedPort((0.010, 0.010, 0.005), "ez", 50.0, pulse),
        LumpedPort((0.020, 0.010, 0.005), "ez", 50.0, pulse),
    ]
    freqs = jnp.asarray([0.8e9, 1.2e9], dtype=jnp.float32)

    extraction = extract_s_matrix(
        grid,
        materials,
        ports,
        freqs,
        n_steps=20,
        boundary="pec",
        return_vi_dump=True,
    )

    path = tmp_path / "lumped_real_vi_dump.npz"
    save_port_vi_dump_npz(
        path,
        voltages=extraction.voltages,
        currents=extraction.currents,
        freqs=np.asarray(extraction.freqs),
        port_impedances=extraction.port_impedances,
        metadata=PortDumpMetadata(
            commit_hash="test",
            geometry={"kind": "small_two_port_lumped_smoke"},
        ),
        port_names=extraction.port_names,
        driven_port_indices=extraction.driven_port_indices,
        production_smatrix=np.asarray(extraction.s_params),
    )

    dump = load_port_vi_dump_npz(path)
    replayed = replay_smatrix_from_port_vi_dump(dump)
    comparison = compare_replayed_smatrix(
        replayed,
        type("Production", (), {"s_params": dump.production_smatrix, "freqs": dump.freqs})(),
    )
    assert comparison.ok, comparison.summary()


def test_lumped_analytic_oracle_report_covers_open_short_matched_rlc():
    oracle = _load_lumped_oracle_module()

    payload = oracle.evaluate_lumped_analytic_oracles()

    assert payload["status"] == "passed"
    cases = {case["name"]: case for case in payload["cases"]}
    for name in (
        "matched_50ohm",
        "short_0ohm",
        "open_infinite",
        "resistor_25ohm",
        "capacitor_1pf",
        "inductor_10nh",
        "series_rlc_10ohm_10nh_1pf",
        "parallel_rlc_200ohm_10nh_1pf",
    ):
        assert cases[name]["status"] == "passed"


def test_lumped_replay_sweep_smoke(tmp_path):
    sweep = _load_lumped_sweep_module()
    case = sweep.LumpedSweepCase(
        name="tiny_smoke",
        domain_m=(0.03, 0.02, 0.015),
        dx_m=5.0e-3,
        freq_max_hz=2.0e9,
        freqs_hz=(0.8e9, 1.2e9),
        port1_pos_m=(0.010, 0.010, 0.005),
        port2_pos_m=(0.020, 0.010, 0.005),
        n_steps=20,
    )

    payload = sweep.run_lumped_replay_sweep(
        output_dir=tmp_path,
        cases=(case,),
        passivity_limit=10.0,
        reciprocity_limit=10.0,
    )

    assert payload["status"] == "passed"
    assert payload["cases"][0]["replay_max_abs_diff"] <= payload["cases"][0]["replay_max_allowed"]
    assert (tmp_path / "tiny_smoke_raw_vi_dump.npz").exists()


def test_wire_extract_s_matrix_can_emit_replayable_real_vi_dump(tmp_path):
    """Production wire-port extraction can save raw V/I replay phasors."""
    import jax.numpy as jnp

    from rfx.core.yee import init_materials
    from rfx.grid import Grid
    from rfx.probes.probes import extract_s_matrix_wire
    from rfx.sources.sources import GaussianPulse, WirePort

    replay_wire = _load_replay_wire_module()
    grid = Grid(freq_max=2.0e9, domain=(0.03, 0.026, 0.012), dx=4.0e-3, cpml_layers=0)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=1.0e9, bandwidth=0.8)
    ports = [
        WirePort((0.008, 0.012, 0.004), (0.008, 0.012, 0.008), "ez", 50.0, pulse),
        WirePort((0.020, 0.012, 0.004), (0.020, 0.012, 0.008), "ez", 50.0, pulse),
    ]
    freqs = jnp.asarray([0.8e9, 1.2e9], dtype=jnp.float32)

    extraction = extract_s_matrix_wire(
        grid,
        materials,
        ports,
        freqs,
        n_steps=80,
        boundary="pec",
        return_vi_dump=True,
    )

    path = tmp_path / "wire_real_vi_dump.npz"
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps({"schema": "rfx.wire_port_vi_dump"})),
        freqs_hz=np.asarray(extraction.freqs),
        raw_voltages_fdt=np.asarray(extraction.raw_voltages_fdt),
        raw_currents=np.asarray(extraction.raw_currents),
        port_impedances_ohm=np.asarray(extraction.port_impedances),
        port_cell_counts=np.asarray(extraction.port_cell_counts),
        production_smatrix=np.asarray(extraction.s_params),
        port_names=np.asarray(extraction.port_names, dtype=object),
        driven_port_indices=np.asarray(extraction.driven_port_indices),
    )

    payload = replay_wire.replay_wire_port_vi_dump(path)

    assert payload["status"] == "passed"
    assert payload["max_abs_diff"] <= payload["max_allowed"]


def test_wire_replay_sweep_smoke(tmp_path):
    sweep = _load_wire_sweep_module()
    case = sweep.WireSweepCase(
        name="tiny_wire_smoke",
        domain_m=(0.03, 0.026, 0.012),
        dx_m=4.0e-3,
        freq_max_hz=2.0e9,
        freqs_hz=(0.8e9, 1.2e9),
        port1_start_m=(0.008, 0.012, 0.004),
        port1_end_m=(0.008, 0.012, 0.008),
        port2_start_m=(0.020, 0.012, 0.004),
        port2_end_m=(0.020, 0.012, 0.008),
        n_steps=20,
    )

    payload = sweep.run_wire_replay_sweep(
        output_dir=tmp_path,
        cases=(case,),
        passivity_limit=10.0,
        reciprocity_limit=10.0,
    )

    assert payload["status"] == "passed"
    assert payload["cases"][0]["replay_max_abs_diff"] <= payload["cases"][0]["replay_max_allowed"]
    assert (tmp_path / "tiny_wire_smoke_raw_vi_dump.npz").exists()


def test_reference_plane_shift_is_explicit_metadata():
    freqs = np.array([1.0e9])
    z0 = 50.0
    gamma = np.array([1j * 2.0])
    raw_s11 = np.array([[[0.5 + 0.0j]]])
    a_raw = np.array([[[1.0 + 0.0j]]])
    b_raw = raw_s11 * a_raw
    sqrt_z = np.sqrt(z0)
    voltages = sqrt_z * (a_raw + b_raw)
    currents = (a_raw - b_raw) / sqrt_z

    replayed = replay_smatrix_from_vi_dump(
        voltages,
        currents,
        freqs=freqs,
        port_impedances=z0,
        reference_plane_offsets_m=np.array([0.25]),
        propagation_constants=gamma,
    )

    expected = raw_s11 * np.exp(2.0 * gamma[0] * 0.25)
    np.testing.assert_allclose(replayed.s_params, expected, rtol=1e-12, atol=1e-12)


def test_msl_3probe_dump_replay_matches_synthetic_production(tmp_path):
    replay_msl = _load_replay_msl_module()
    freqs = np.array([3.0e9, 3.5e9, 4.0e9])
    n_ports = 2
    n_freqs = freqs.size
    q = np.exp(-0.2j * np.arange(1, n_freqs + 1))
    z0 = np.array([50.0, 52.0])
    production = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    production[0, 0, :] = 0.05 + 0.01j
    production[1, 1, :] = 0.04 - 0.02j
    production[1, 0, :] = 0.92 - 0.03j
    production[0, 1, :] = 0.90 + 0.02j

    raw_v123 = np.zeros((n_ports, n_ports, 3, n_freqs), dtype=np.complex128)
    raw_i1 = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    for driven in range(n_ports):
        alpha_d = (1.0 + 0.1 * driven) * np.exp(0.05j * np.arange(n_freqs))
        for port in range(n_ports):
            alpha = alpha_d if port == driven else production[port, driven, :] * alpha_d
            gamma = production[port, driven, :] * alpha_d if port == driven else np.zeros(n_freqs)
            raw_v123[driven, port, 0, :] = alpha + gamma
            raw_v123[driven, port, 1, :] = alpha * q + gamma / q
            raw_v123[driven, port, 2, :] = alpha * q * q + gamma / (q * q)
            raw_i1[driven, port, :] = (alpha - gamma) / z0[port]

    dump = tmp_path / "msl_3probe_dump.npz"
    np.savez(
        dump,
        metadata_json=np.asarray(json.dumps({"schema": "rfx.msl_3probe_dump"})),
        freqs_hz=freqs,
        raw_v123=raw_v123,
        raw_i1=raw_i1,
        production_smatrix=production,
        port_names=np.asarray(("in", "out"), dtype=object),
        driven_port_indices=np.arange(n_ports),
    )

    payload = replay_msl.replay_msl_3probe_dump(dump)

    assert payload["status"] == "passed"
    assert payload["max_abs_diff"] < 1e-12
