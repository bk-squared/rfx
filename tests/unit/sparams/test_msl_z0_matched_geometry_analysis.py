"""Synthetic provenance refusals for #752; no RFX imports or field solves.

These arrays describe an internally consistent record bundle, not a simulated
microstrip. They test evidence bookkeeping separately from physical residual
thresholds, source purity and absolute impedance accuracy.
"""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/diagnostics/analyze_msl_z0_matched_geometry_sweep.py"
)


def _signature(value):
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


@pytest.fixture(scope="module")
def analyzer():
    spec = importlib.util.spec_from_file_location(
        "matched_geometry_analysis_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def baseline():
    dx, nh, nw, side = 254e-6 / 3, 3, 7, 24
    nx = math.ceil(0.040 / dx)
    shape = (nx + 17, nw + 2 * side + 17, nh + 8 * nh + 9)
    h, w = nh * dx, nw * dx
    dt = 0.99 * dx / (299792458.0 * math.sqrt(3))
    steps = math.ceil(20 / (5e9 * dt))
    frequencies = np.linspace(0.5e9, 5e9, 30, dtype=np.float32)
    nf = len(frequencies)
    margin, offset, spacing = (
        math.ceil(0.002 / dx),
        math.ceil(0.006 / dx),
        math.ceil(0.003 / dx),
    )
    feeds = (margin, nx - margin)
    jlo, jhi = 8 + side, 8 + side + nw
    jc = round(side + nw / 2) + 8
    pulse = dict(f0=3.75e9, bandwidth=0.8, amplitude=1.0, cutoff=3.0)
    definitions, declarations, positions, stencils = [], [], [], []
    for p, (feed, sign) in enumerate(zip(feeds, (1, -1))):
        x = (feed + sign * (offset + np.arange(5) * spacing)) * dx
        positions.append(x.tolist())
        definition = dict(
            name=f"p{p}",
            position_m=[feed * dx, (side + nw / 2) * dx, 0.0],
            direction="+x" if sign > 0 else "-x",
            mode="laplace",
            width_m=w,
            height_m=h,
            impedance_ohm=50.0,
            n_probes=5,
            n_probe_offset=offset,
            n_probe_spacing=spacing,
        )
        definitions.append(definition)
        declarations.append(
            dict(
                name=f"p{p}",
                position=definition["position_m"],
                width=w,
                height=h,
                direction=definition["direction"],
                mode="laplace",
                impedance=50.0,
                n_probes=5,
                n_probe_offset=offset,
                n_probe_spacing=spacing,
                waveform=pulse.copy(),
                eps_r_sub=3.66,
            )
        )
        index = feed + sign * offset + 8
        stencils.append(
            dict(
                axis="x",
                e_index=index,
                h_indices=[index - 1, index],
                registration_coordinates=[(index - 9) * dx, (index - 8) * dx],
                sample_coordinates=[(index - 8.5) * dx, (index - 7.5) * dx],
                voltage_coordinate=(index - 8) * dx,
                weights=[0.5, 0.5],
            )
        )
    u = w / h
    ee = (3.66 + 1) / 2 + (3.66 - 1) / (2 * np.sqrt(1 + 12 / u))
    reference = 120 * np.pi / (np.sqrt(ee) * (u + 1.393 + 0.667 * np.log(u + 1.444)))
    beta = 2 * np.pi * frequencies.astype(float) * np.sqrt(ee) / 299792458.0
    raw_v = np.empty((2, 2, 5, nf), np.complex64)
    raw_i = np.empty((2, 2, nf), np.complex64)
    raw_q = np.empty_like(raw_i)
    for d in range(2):
        for p, sign in enumerate((1, -1)):
            a, b = (1.0, 0.1) if d == p else (0.0, 0.8)
            voltage, current = (
                (a + b) * np.sqrt(reference),
                (a - b) / np.sqrt(reference),
            )
            alpha, gamma = (
                (voltage + sign * reference * current) / 2,
                (voltage - sign * reference * current) / 2,
            )
            x = np.asarray(positions[p])
            raw_v[d, p] = alpha * np.exp(
                -1j * (x - x[0])[:, None] * beta
            ) + gamma * np.exp(1j * (x - x[0])[:, None] * beta)
            raw_i[d, p] = current
            raw_q[d, p] = np.exp(-1j * beta * (x[1] - x[0]))
    z0 = np.full((2, nf), reference, np.complex64)
    smatrix = np.broadcast_to(
        np.array([[0.1, 0.8], [0.8, 0.1]], np.complex64)[:, :, None], (2, 2, nf)
    ).copy()
    raw = dict(
        freqs_hz=frequencies,
        raw_v=raw_v,
        raw_i1=raw_i,
        raw_i1_left=raw_i.copy(),
        raw_i1_same_index=raw_i.copy(),
        raw_q=raw_q,
        raw_z0=np.broadcast_to(z0, (2, 2, nf)).copy(),
        production_z0=z0.copy(),
        # The production result carries beta from the first driven
        # port only (one spectrum), matching beta_first.
        production_smatrix=smatrix.copy(),
        production_beta=beta.astype(np.float32),
        driven_port_indices=np.array([0, 1], np.int32),
    )
    ts = np.tile(
        np.exp(-np.arange(steps) / 1800.0).astype(np.float32)[:, None], (1, 10)
    )
    power = ts.astype(float) ** 2
    sd = float(
        10 * np.log10(power[-max(1, steps // 10) :].mean(axis=0) / power.max(axis=0))[0]
    )
    result = dict(
        S=smatrix,
        Z0=z0,
        beta=raw["production_beta"].copy(),
        freqs=frequencies.copy(),
        reliable=np.ones((2, nf), bool),
        beta_railed=np.zeros((2, nf), bool),
        settling_db=np.array([sd, sd]),
        reference_impedances=np.array([reference, reference]),
        cond_a=np.ones(nf),
    )
    material = {
        "eps_r": np.ones(shape, np.float32),
        "mu_r": np.ones(shape, np.float32),
        "sigma": np.zeros(shape, np.float32),
    }
    material["eps_r"][:, :, :nh] = np.float32(3.66)
    pec = [np.zeros(shape, bool) for _ in range(3)]
    pec[0][8 : 8 + nx, jlo : jhi + 1, nh] = True
    pec[1][8 : 8 + nx + 1, jlo:jhi, nh] = True
    plan = dict(
        label="h3",
        dx_m=dx,
        height_m=h,
        width_m=w,
        trace_thickness_m=0.0,
        eps_r=3.66,
        desired_height_m=254e-6,
        desired_width_m=600e-6,
        height_intervals=nh,
        width_intervals=nw,
        domain_m=[nx * dx, (nw + 2 * side) * dx, (nh + 8 * nh) * dx],
        grid_shape=list(shape),
        dt_s=dt,
        n_steps=steps,
        num_periods=20.0,
        actual_duration_s=steps * dt,
        pulse_f0_hz=3.75e9,
        pulse_bandwidth=0.8,
        frequencies_hz=frequencies.tolist(),
        feed_positions_m=[f * dx for f in feeds],
        probe_positions_m=positions,
        port_definitions=declarations,
        material_signatures={k: _signature(v) for k, v in material.items()},
        pec_edge_signatures=[_signature(p) for p in pec],
        driver_sha256="d" * 64,
        jax="synthetic",
        python="synthetic",
        backend="synthetic",
        x64=False,
    )
    metadata = dict(
        schema="rfx.msl_nprobe_dump",
        schema_version=4,
        s_wave_convention="power",
        current_spatial_alignment="linear_bracketing_H_to_E_node",
        current_convention="native_msl_loop_current",
        current_plane_stencils=stencils,
        port_definitions=definitions,
        n_probes_per_port=[5, 5],
        s_reference_impedances_ohm=[reference, reference],
        production_smatrix_assembly="multi_drive_solve",
        grid=dict(dx_m=dx, dt_s=dt, nx=shape[0], ny=shape[1], nz=shape[2]),
        simulation=dict(freq_max_hz=5e9, num_periods=20.0, n_steps=None),
    )
    probes, dft = [], []
    for p in range(2):
        indices = [round(x / dx) + 8 for x in positions[p]]
        probes.extend(dict(i=i, j=jc, k=round(nh / 2), component="ez") for i in indices)
        for i in indices:
            dft.append(
                dict(
                    component="ez",
                    axis=0,
                    index=i,
                    region=[jc, jc + 1, 0, nh],
                    freqs_hz=frequencies.tolist(),
                    total_steps=steps,
                    window="rect",
                    window_alpha=0.25,
                )
            )
        for component in ("hy", "hz"):
            for i in (indices[0] - 1, indices[0]):
                dft.append(
                    dict(
                        component=component,
                        axis=0,
                        index=i,
                        region=[jlo - 1, jhi + 1, nh - 1, nh + 1],
                        freqs_hz=frequencies.tolist(),
                        total_steps=steps,
                        window="rect",
                        window_alpha=0.25,
                    )
                )
    consumed, records, inputs = [], [], []
    for drive in range(2):
        waveform = np.exp(-(((np.arange(steps) - 100) / 30.0) ** 2)).astype(np.float32)
        source = dict(
            i=feeds[drive] + 8, j=jc, k=1, component="ez", waveform=_signature(waveform)
        )
        consumed.append(
            dict(
                drive=drive,
                n_steps=steps,
                material_signatures=copy.deepcopy(plan["material_signatures"]),
                probes=copy.deepcopy(probes),
                sources=[source],
                boundary="cpml",
                cpml_axes="xyz",
                pec_axes=None,
                periodic=None,
                dft_planes=copy.deepcopy(dft),
            )
        )
        inp = {f"material_{name}": value.copy() for name, value in material.items()}
        inp.update({f"pec_{i}": value.copy() for i, value in enumerate(pec)})
        inp["source_0"] = waveform
        inputs.append(inp)
        rec = {"time_series": ts.copy()}
        for i, plane in enumerate(dft):
            region = plane["region"]
            array = np.zeros(
                (nf, region[1] - region[0], region[3] - region[2]), np.complex64
            )
            if plane["component"] == "ez":
                p, n = divmod(i, 9)
                array[:] = (raw_v[drive, p, n] / (nh * dx))[:, None, None]
            rec[f"dft_{i}"] = array
        records.append(rec)
    diagnostics = dict(
        completed_drives=2,
        assembly="multi_drive_solve",
        warnings=[],
        probe_clearance=[],
    )
    return plan, metadata, raw, result, diagnostics, consumed, records, inputs


@pytest.fixture
def evidence(baseline):
    return copy.deepcopy(baseline)


def test_coherent_synthetic_bundle_is_accepted(analyzer, evidence):
    assert analyzer.validate_evidence(*evidence) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "production_z0",
        "diagonal_raw_z0",
        "passive_voltage_nan",
        "passive_current_inf",
        "consumed_eps_hash",
        "duplicate_drive",
        "current_plane",
        "current_convention",
        "pec_input",
        "probe_index",
        "source_waveform",
        "dft_voltage",
        "weighted_current",
    ],
)
def test_inconsistent_or_invalid_record_is_rejected_without_new_physics_thresholds(
    analyzer, evidence, mutation
):
    assert analyzer.validate_evidence(*evidence) == []
    plan, metadata, raw, result, _, consumed, records, inputs = evidence
    if mutation == "production_z0":
        raw["production_z0"] *= 1.1
    elif mutation == "diagonal_raw_z0":
        raw["raw_z0"][0, 0] *= 1.1
    elif mutation == "passive_voltage_nan":
        raw["raw_v"][0, 1, 2, 20] = np.nan
    elif mutation == "passive_current_inf":
        raw["raw_i1"][0, 1, 20] = np.inf
    elif mutation == "consumed_eps_hash":
        consumed[0]["material_signatures"]["eps_r"]["sha256"] = "0" * 64
    elif mutation == "duplicate_drive":
        consumed[1]["drive"] = 0
    elif mutation == "current_plane":
        metadata["current_plane_stencils"][0]["voltage_coordinate"] += plan["dx_m"]
    elif mutation == "current_convention":
        metadata["current_spatial_alignment"] = "same_index"
    elif mutation == "pec_input":
        inputs[0]["pec_0"].flat[0] = True
    elif mutation == "probe_index":
        consumed[0]["probes"][0]["i"] += 1
    elif mutation == "source_waveform":
        inputs[0]["source_0"][100] *= 2
    elif mutation == "dft_voltage":
        records[0]["dft_9"][20, 0, 0] += 100.0
    elif mutation == "weighted_current":
        raw["raw_i1_left"][0, 1, 20] *= 2
    failures = analyzer.validate_evidence(*evidence)
    assert failures, mutation
    assert all(isinstance(reason, str) and reason for reason in failures)


def _write_case(directory, evidence):
    plan, metadata, raw, result, diagnostics, consumed, records, inputs = evidence
    directory.mkdir()
    for name, value in (
        ("plan", plan),
        ("diagnostics", diagnostics),
        ("consumed-plans", consumed),
    ):
        (directory / f"{name}.json").write_text(json.dumps(value, allow_nan=False))
    np.savez(directory / "raw-vi.npz", metadata_json=json.dumps(metadata), **raw)
    np.savez(directory / "result.npz", **result)
    for drive in range(2):
        np.savez(directory / f"drive{drive}-records.npz", **records[drive])
        np.savez(directory / f"drive{drive}-inputs.npz", **inputs[drive])


def test_case_verdict_and_aggregate_cannot_pass_invalid_or_missing_evidence(
    analyzer, evidence, tmp_path
):
    directory = tmp_path / "valid"
    _write_case(directory, evidence)
    valid = analyzer.analyze_case("h3", directory, 101)
    assert valid["status"] == "usable_under_declared_screens"
    evidence[2]["raw_v"][0, 1, 2, 20] = np.nan
    bad_directory = tmp_path / "invalid"
    _write_case(bad_directory, evidence)
    invalid = analyzer.analyze_case("h3", bad_directory, 101)
    assert invalid["status"] != "usable_under_declared_screens"
    cases = [dict(valid, label=label) for label in analyzer.LABELS]
    cases[0] = invalid
    missing = analyzer.analyze_case("h3", tmp_path / "missing", 101)
    for rows in (cases, cases[1:], [valid] * 6, [missing]):
        summary = analyzer.aggregate(rows)
        for reference in summary.values():
            for group in reference.values():
                assert not group["all_six_within_0p4_percent"]
