"""One source end for every reader: the tables the time step applies (#2030 S1).

Each scene is built through run() (or the coaxial driver) and stopped at the
loop entry, before the first step; the tables the loop would apply are read
from the call. The expected end is computed here with numpy from those tables:
last sample above 1e-6 of its own peak, plus one, plus the free-space transit
ceil(d / (C0 dt)) from the farthest injecting point to the farthest record
(0 for the whole-domain reader of until_decay). A waveguide port's end is its
applied E table held at its last entry, never the analytic v_inc_t. A lane
that refuses the declaration is a "not reachable" row, its refusal pinned.
"""
import ast
import dataclasses
import math
import pathlib
import re
import sys
import warnings
from itertools import product

import numpy as np
import pytest

from rfx.grid import C0
from rfx.sources.sources import GaussianPulse
from tests.contracts.path_equivalence import builders

REPO = pathlib.Path(__file__).resolve().parents[2]
N = 6000
PULSE = GaussianPulse(f0=5e9, bandwidth=0.5)
ROWS = {
    "lumped": ("_ports", "lumped_port"),
    "wire": ("_ports", "wire_port"),
    "msl": ("_msl_ports", "msl_port"),
    "waveguide": ("_waveguide_ports", "waveguide_port"),
    "floquet": ("_floquet_ports", "floquet_port"),
    "tfsf": ("_tfsf", "plane_wave"),
    "tfsf_closed_box": ("_tfsf", "closed_box"),
}
NOT_REACHABLE = {
    ("tfsf_closed_box", True): "CustomWaveform and closed_box TFSF require a uniform",
    ("floquet", True): "Floquet ports do not support non-uniform z mesh",
    ("coax", True): "compute_coaxial_two_port() supports only a uniform Yee grid",
}


def _hand_tfsf_geometry(cfg, grid, inside=False):
    spans = []
    for axis, size in zip("xyz", grid.shape):
        lo, hi = getattr(cfg, axis + "_lo", 0), getattr(cfg, axis + "_hi", size - 1)
        if axis != "x" and hi == lo and size > 1:
            lo, hi = 0, size - 1
        spans.append((lo + int(inside), hi - int(inside)))
    indices = np.array(list(product(*spans)))
    corners = np.array([[grid.node_of(a, int(i)) for a, i in zip("xyz", c)] for c in indices])
    if hasattr(cfg, "m"):
        paths = (indices @ cfg.m - cfg.source_p) * grid.dx / np.linalg.norm(cfg.m)
    elif hasattr(cfg, "d_aux"):
        # Builder idx_E = (i*cos + j*sin)*dx/d_aux + src_idx.
        mapped = ((indices[:, 0]*cfg.cos_theta + indices[:, 1]*cfg.sin_theta)
                  * grid.dx / cfg.d_aux + cfg.src_idx)
        paths = (mapped - cfg.src_idx) * cfg.d_aux
    elif hasattr(cfg, "src_x"):
        sign = 1 if cfg.direction == "+x" else -1
        paths = sign * (cfg.i0_x + indices[:, 0] - cfg.x_lo - cfg.src_x) * cfg.dx_1d / cfg.cos_theta
    else:
        # Slab line: the source margin continues the end cell beside the box and
        # the mapped part is the realized x cells, so the path to a corner is the
        # node distance on the grid plus the margin cells (graded x admitted).
        cells = np.asarray(grid.cells("x"), dtype=float)
        nodes = np.array([grid.node_of("x", int(i)) for i in indices[:, 0]])
        if cfg.direction == "+x":
            paths = ((cfg.i0 - cfg.src_idx) * cells[cfg.x_lo - 1]
                     + nodes - grid.node_of("x", cfg.x_lo))
        else:
            paths = (grid.node_of("x", cfg.x_hi + 1) - nodes
                     + (cfg.src_idx - cfg.i0 - cfg.x_hi + cfg.x_lo - 1) * cells[cfg.x_hi + 1])
    return indices, corners, paths


def _hand_tfsf_samples(cfg, grid, n):
    import jax.numpy as jnp
    t = jnp.arange(n, dtype=jnp.float32) * grid.dt
    arg = (t - cfg.src_t0) / cfg.src_tau
    env = jnp.exp(-arg**2)
    kind = getattr(cfg, "src_waveform", "analytic")
    if kind == "analytic":
        samples = env * jnp.exp(-1j * 2*jnp.pi*cfg.src_fcen*(t-cfg.src_t0))
    elif kind == "modulated_gaussian":
        samples = env * jnp.cos(2*jnp.pi*cfg.src_fcen*(t-cfg.src_t0))
    else:
        samples = (-2*arg) * env
    return np.asarray(cfg.src_amp * samples)


# 40 mm graded x lines; the end cells equal dx (CPML rule). "-graded-wide" is the
# separate review's case: 1 mm cells beside the box and 2 mm across it.
GRADED_X = {"-graded": (.002, np.array([.002] * 5 + [.001] * 20 + [.002] * 5)),
            "-graded-wide": (.001, np.array([.001] * 10 + [.002] * 10 + [.001] * 10))}


def _s2b_tfsf(kind):
    from rfx import Simulation
    suffix = next((g for g in ("-graded-wide", "-graded") if kind.endswith(g)), "")
    dx, profile = GRADED_X.get(suffix, (.002, None))
    sim = Simulation(freq_max=10e9, domain=(.040, .024, .024), dx=dx, boundary="cpml",
                     cpml_layers=8, **({"dx_profile": profile} if suffix else {}))
    kind = kind.removesuffix(suffix)
    options = {"slab+x": {}, "slab-x": {"direction": "-x"},
               "closed": {"closed_box": True},
               "closed-oblique": {"closed_box": True, "theta_deg": 45., "polarization": "s"},
               "bloch": {"angle_deg": 20., "waveform": "modulated_gaussian", "bandwidth": .2},
               "methodB": {"angle_deg": 20., "method": "methodB"}}[kind]
    # Use modulated Gaussian: the default pulse leaves a near-DC field (#1839).
    sim.add_tfsf_source(f0=5e9, **({"bandwidth": .8, "margin": 3,
                                  "waveform": "modulated_gaussian"} | options))
    return sim


TF_KINDS = ("slab+x", "slab-x", "slab+x-graded", "slab-x-graded", "slab+x-graded-wide", "closed",
            "closed-oblique", "bloch", "methodB")


@pytest.mark.parametrize("kind", TF_KINDS)
def test_s2b_r1_tfsf_hand_reach(kind):
    from rfx.measurement.record_end import injection_end
    from rfx.sources.tfsf import tfsf_from_declaration
    sim = _s2b_tfsf(kind)
    grid = sim._build_realized_grid()
    cfg, _ = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, line_steps=1200)
    from rfx.measurement.tfsf_reach import tfsf_corner_paths
    for inset in (0, 1):
        _, hand_corners, hand_paths = _hand_tfsf_geometry(cfg, grid, inside=bool(inset))
        actual_corners, actual_paths = tfsf_corner_paths(cfg, grid, inset=inset)
        np.testing.assert_allclose(actual_corners, hand_corners, rtol=0, atol=0)
        np.testing.assert_allclose(actual_paths, hand_paths, rtol=1e-12, atol=0)
    _, corners, paths = _hand_tfsf_geometry(cfg, grid)
    off = _off(_hand_tfsf_samples(cfg, grid, 1200))
    probe = np.array([grid.node_of(a, 9) for a in "xyz"])
    for records in (None, [probe]):
        reach = max(paths) if records is None else max(
            path + np.linalg.norm(c - probe) for c, path in zip(corners, paths))
        expected = off + _transit(reach, grid.dt)
        assert injection_end(1200, grid.dt, records=records, grid=grid, tfsf=cfg) == expected


def _analytic_envelope(samples):
    """Unwindowed analytic signal: retain DC/Nyquist, double positive bins."""
    samples = np.asarray(samples)
    n = len(samples)
    weights = np.zeros(n)
    weights[0] = 1
    weights[1:(n + 1)//2] = 2
    if n % 2 == 0:
        weights[n//2] = 1
    return np.abs(np.fft.ifft(np.fft.fft(samples) * weights))


def _envelope(x, dt, f0, bw, pad=8):
    """|analytic signal| through a Gaussian window on the pulse band, zero-padded (no wrap)."""
    x = np.asarray(x)
    n = x.shape[0]
    f = np.fft.fftfreq(pad * n, dt)
    band = (f < 0) if np.iscomplexobj(x) else (f > 0)   # applied analytic waveform is exp(-j w t)
    g = np.where(band, np.exp(-((np.abs(f) - f0) / (f0 * bw / 2)) ** 2), 0.0)
    return np.abs(np.fft.ifft(2 * g * np.fft.fft(x, pad * n))[:n])


def _envelope_steps(envelope):
    return (int(envelope.argmax()),
            int(np.flatnonzero(envelope > 1e-3 * envelope.max())[-1]))


@pytest.mark.parametrize("check,delay", [("delay", 14.), ("delay", 37.4),
                                          ("delay", 100.6), ("phase", 0.)])
def test_s2b_r1_envelope_estimator_self_check(check, delay, record_property):
    from rfx.sources.tfsf import tfsf_from_declaration
    sim = _s2b_tfsf("closed" if check == "delay" else "bloch")
    grid = sim._build_grid()
    n, f0 = 1200, 5e9
    bandwidth = .8 if check == "delay" else .2
    cfg, _ = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, line_steps=n)
    t = np.arange(n)*grid.dt - cfg.src_t0
    waveform = cfg.src_amp * np.exp(-(t/cfg.src_tau)**2) * np.cos(2*np.pi*f0*t)
    delayed_t = t - delay*grid.dt
    phase = .3*2*np.pi if check == "phase" else 0.
    shifted = (cfg.src_amp * np.exp(-(delayed_t/cfg.src_tau)**2) *
               np.cos(2*np.pi*f0*delayed_t + phase))
    tolerances = (1., 2.) if check == "delay" else (1., 1.)
    original_steps = _envelope_steps(_envelope(waveform, grid.dt, f0, bandwidth))
    shifted_steps = _envelope_steps(_envelope(shifted, grid.dt, f0, bandwidth))
    errors = tuple(after-before-delay for before, after in zip(original_steps, shifted_steps))
    detail = {"original": original_steps, "shifted": shifted_steps,
              "expected_shift": delay, "errors": errors, "tolerances": tolerances}
    record_property("estimator", repr(detail))
    print(f"R1 estimator {check}: {detail}")
    assert all(abs(error) <= tolerance for error, tolerance in zip(errors, tolerances)), detail


@pytest.mark.parametrize("kind", TF_KINDS)
def test_s2b_r1_vacuum_corner_measurement(kind, record_property):
    from rfx.sources.tfsf import tfsf_from_declaration
    sim = _s2b_tfsf(kind)
    grid = sim._build_realized_grid()
    n = 1200
    cfg, _ = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, line_steps=n)
    from rfx.measurement.tfsf_reach import tfsf_corner_paths
    indices, _, _ = _hand_tfsf_geometry(cfg, grid, inside=True)
    corners, paths = tfsf_corner_paths(cfg, grid, inset=1)
    for point in corners:
        sim.add_probe(tuple(point), "ez", allow_in_absorber=True)
    waveform = _hand_tfsf_samples(cfg, grid, n)
    off = _off(waveform)
    bandwidth = .2 if kind == "bloch" else .8
    env_w = _envelope(waveform, grid.dt, 5e9, bandwidth)
    peak_w = int(env_w.argmax())
    last_w = int(np.flatnonzero(env_w > 1e-3 * env_w.max())[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(n_steps=n, skip_preflight=True)
    measurements = []
    for c, raw, path in zip(indices, np.asarray(result.time_series).T, paths):
        trace = np.abs(raw)
        envelope = _envelope(raw, grid.dt, 5e9, bandwidth)
        end = off + _transit(path, grid.dt) + 2
        assert trace.max() > 0 and end < n
        last = int(np.flatnonzero(trace > 1e-6 * trace.max())[-1])
        ratio = float(trace[end:].max() / trace.max())
        transit = float(path / (C0 * grid.dt))
        measurements.append({
            "corner": tuple(map(int, c)), "end": end,
            "last_above_1e-6": last, "post_bound_ratio": ratio, "margin": end-last,
            "predicted_peak": peak_w + transit, "measured_peak": int(envelope.argmax()),
            "carrier_argmax": int(trace.argmax()),
            "unfiltered_analytic_peak": int(_analytic_envelope(raw).argmax()),
            "carrier_difference": int(trace.argmax()) - (peak_w + transit),
            "predicted_envelope_end": last_w + transit,
            "measured_envelope_end": int(np.flatnonzero(envelope > 1e-3 * envelope.max())[-1]),
        })
    record_property("R1", repr(measurements))
    print(f"R1 {kind}: {measurements}", file=sys.stderr)
    for row in measurements:
        if "graded" in kind:
            # Safe side only (design amendments 8 and 9): a graded slab line arrives
            # up to 1.3 steps before its node path, and an entry-face floor up to
            # 3.7e-4 lifts the 1e-3 end; the computed end must not be early.
            assert row["measured_peak"] <= row["predicted_peak"] + 1, (kind, row)
            assert row["measured_envelope_end"] <= row["predicted_envelope_end"] + 2, (kind, row)
            continue
        assert abs(row["measured_peak"] - row["predicted_peak"]) <= 1, (kind, row)
        assert abs(row["measured_envelope_end"] - row["predicted_envelope_end"]) <= 2, (kind, row)
    if kind in ("closed", "closed-oblique"):
        assert all(row["post_bound_ratio"] <= 1e-6 for row in measurements), (kind, measurements)


def test_s2b_r2_scene_b_echo():
    from rfx import Box, Simulation
    from rfx.sources.tfsf import tfsf_from_declaration
    sim = Simulation(freq_max=20e9, domain=(.080, .016, .016), dx=.001,
                     boundary="cpml", cpml_layers=8)
    sim.add_tfsf_source(f0=10e9, bandwidth=.8, polarization="ez", margin=3, closed_box=True)
    sim.add(Box((.0655, .0055, .0045), (.0705, .0105, .0115)), material="pec")
    sim.add_probe((.0011, .0093, .0071), "ez")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(n_steps=900, skip_preflight=True)
    cfg, _ = tfsf_from_declaration(sim._tfsf, result.grid, sim._freq_max, line_steps=900)
    _, corners, paths = _hand_tfsf_geometry(cfg, result.grid)
    probe = np.array(sim._probes[0].position)
    expected = _off(_hand_tfsf_samples(cfg, result.grid, 900)) + _transit(
        max(path + np.linalg.norm(c - probe) for c, path in zip(corners, paths)), result.dt)
    assert expected == 423
    assert sim._settling_source_end(result) == expected
    trace = np.abs(np.asarray(result.time_series)[:, 0])
    assert trace[expected:].max() / trace.max() <= 1e-2


def _s2b_scene_c(accumulators):
    from rfx import Simulation
    sim = Simulation(freq_max=10e9, domain=(.120, .030, .030), dx=.002,
                     boundary="cpml", cpml_layers=8)
    sim.add_source((.0101, .0149, .0153), "ez", amplitude_kind="current")
    sim.add_probe((.0141, .0161, .0139), "ez")
    if "ntff" in accumulators:
        sim.add_ntff_box((.004,)*3, (.116, .026, .026), freqs=np.array([5e9]))
    if "dft" in accumulators:
        sim.add_dft_plane_probe(axis="x", coordinate=.116, name="far", freqs=np.array([5e9]))
    if "flux" in accumulators:
        sim.add_flux_monitor(axis="x", coordinate=.116, name="out", freqs=np.array([5e9]))
    return sim


def _hand_accumulator_ends(result, sim):
    grid = result.grid
    spans_by_name = {}
    box = result.ntff_box
    if box is not None:
        spans_by_name["NTFF box"] = [(getattr(box, a+"_lo"), getattr(box, a+"_hi")) for a in "ijk"]
    for attr, label in (("dft_planes", "DFT plane"), ("flux_monitors", "flux monitor")):
        for name, plane in (getattr(result, attr) or {}).items():
            spans = [(0, n-1) for n in grid.shape]
            spans[plane.axis] = (plane.index,)
            spans_by_name[f"{label} {name}"] = spans
    source = np.asarray(sim._ports[0].position)
    import jax
    import jax.numpy as jnp
    wave = sim._ports[0].waveform or GaussianPulse(f0=sim._freq_max/2, bandwidth=.8)
    off = _off(jax.vmap(wave)(jnp.arange(300, dtype=jnp.float32)*result.dt))
    return {name: off + _transit(max(np.linalg.norm(
        np.array([grid.node_of(a, int(i)) for a, i in zip("xyz", c)]) - source)
        for c in product(*spans)), result.dt) for name, spans in spans_by_name.items()}


@pytest.mark.parametrize("accumulators", [("ntff",), ("dft",), ("flux",), ("ntff", "dft", "flux")])
@pytest.mark.parametrize("n", [196, 300])
def test_s2b_r3_accumulator_coverage_and_forward(accumulators, n):
    sim = _s2b_scene_c(accumulators)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(n_steps=n, skip_preflight=True)
    expected = _hand_accumulator_ends(result, sim)
    rows = result.settling_witness["coverage"]
    assert {r["name"]: r["end_step"] for r in rows} == expected
    assert all(r["n_record"] == n for r in rows)
    if "ntff" in accumulators:
        assert expected["NTFF box"] == 241
    coverage_warnings = [w for w in caught if "cannot judge" in str(w.message)]
    if n == 196:
        assert result.settling_witness["status"] == "undetermined"
        reason = rows[0]["verdict"]
        assert reason == (f"cannot judge: the record ends at step {n} before the field leaves "
                          f"{rows[0]['name']} (step {rows[0]['end_step']})")
        assert result.settling_witness["reason"] == reason
        from rfx.errors import AccuracyWarning
        assert len(coverage_warnings) == 1 and coverage_warnings[0].category is AccuracyWarning
        assert reason in str(coverage_warnings[0].message)
    else:
        assert all(r["verdict"] == "covered" for r in rows)
        assert not coverage_warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forward = sim.forward(n_steps=n)
    assert forward.settling_accumulator_end_index == max(expected.values())
    assert forward.settling_witness["coverage"] == rows
    assert forward.settling_witness["status"] == result.settling_witness["status"]


def test_s2b_r3_internal_driver_run_attaches_rows_without_warning():
    """A library-internal driver run (#470) carries the coverage rows but stays silent."""
    sim = _s2b_scene_c(("ntff",))
    sim._internal_probe_indices = (0,)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(n_steps=196, skip_preflight=True)
    rows = result.settling_witness["coverage"]
    assert [r["name"] for r in rows] == ["NTFF box"] and rows[0]["end_step"] == 241
    assert result.settling_witness["status"] == "undetermined"
    assert not [w for w in caught if "cannot judge" in str(w.message)]


def test_s2b_r3_coverage_without_a_time_series():
    """A result with no time series (graded multi-device forward, emit_time_series=False)
    takes the record length from n_steps; with neither, every row cannot judge."""
    from rfx.measurement.coverage import accumulator_coverage
    sim = _s2b_scene_c(("ntff", "flux"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(n_steps=196, skip_preflight=True)
    full = accumulator_coverage(result, sim=sim).rows()
    bare = result._replace(time_series=None)
    assert accumulator_coverage(bare, sim=sim, n_steps=196).rows() == full
    rows = accumulator_coverage(bare, sim=sim).rows()
    assert [r["name"] for r in rows] == [r["name"] for r in full]
    assert all(r["verdict"].startswith("cannot judge: the record length is not available") for r in rows)


@pytest.mark.parametrize("n", [180, 350])
def test_s2b_r4_rcs_ntff_coverage(n, monkeypatch):
    from time import monotonic
    from rfx.core.yee import init_materials
    from rfx.grid import Grid
    from rfx.measurement.rcs import compute_rcs
    import rfx.measurement.rcs as rcs_module
    captured = {}
    original_run = rcs_module.run

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(rcs_module, "run", capture)
    grid = Grid(freq_max=10e9, domain=(.024,)*3, dx=.002, cpml_layers=6)
    materials = init_materials(grid.shape)
    materials = materials._replace(sigma=materials.sigma.at[11:14, 11:14, 11:14].set(1e7))
    started = monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_rcs(grid, materials, n, f0=5e9, bandwidth=.8, cpml_layers=6,
                             theta_obs=np.array([np.pi/2]), phi_obs=np.array([np.pi]))
    print(f"R4 {n} steps: {monotonic() - started:.3f} s")
    cfg, box = captured["tfsf"][0], captured["ntff"]
    _, corners, paths = _hand_tfsf_geometry(cfg, grid)
    records = [np.array([grid.node_of(a, i) for a, i in zip("xyz", c)])
               for c in product(*[(getattr(box, a+"_lo"), getattr(box, a+"_hi")) for a in "ijk"])]
    expected = _off(_hand_tfsf_samples(cfg, grid, n)) + _transit(
        max(p + max(np.linalg.norm(c-r) for r in records) for c, p in zip(corners, paths)), grid.dt)
    witness = result.settling_witness
    row, = witness["coverage"]
    assert row["end_step"] == expected
    assert witness["source_end_index"] == row["end_step"]
    assert row["n_record"] == n and row["name"] == "NTFF box"
    if n == 180:
        assert row["end_step"] >= n
        assert witness["status"] == "undetermined" and witness["reason"] == row["verdict"]
        warns = [w for w in caught if "cannot judge" in str(w.message)]
        assert len(warns) == 1 and row["verdict"] in str(warns[0].message)
    else:
        assert witness["status"] == "absent" and row["verdict"] == "covered"
        assert witness["reason"] == "no time-domain record; add an identification probe"
        assert not [w for w in caught if "cannot judge" in str(w.message)]


@pytest.mark.parametrize("fallback", [False, True, "spectral"])
def test_s2b_r5_sweep_member_witnesses(fallback):
    import copy
    from rfx import Box, Simulation
    from rfx.vmap_sweep import vmap_material_sweep
    mesh = {} if fallback is True else {"dx": .002}
    sim = Simulation(freq_max=10e9, domain=(.03,)*3, boundary="cpml", cpml_layers=6, **mesh)
    sim.add_material("slab", eps_r=2.)
    sim.add(Box((.012, .004, .004), (.018, .026, .026)), material="slab")
    if fallback:
        sim.add_port((.0071, .0149, .0121), "ez", impedance=50., extent=.006)
    else:
        sim.add_source((.0071, .0149, .0153), "ez", amplitude_kind="current")
        sim.add_dft_plane_probe(axis="x", coordinate=.023, freqs=np.array([5e9]), name="read")
    if fallback == "spectral":
        # D2's short, bin-free witnesses coincide; retain D2 and add a
        # spectral fallback record whose member witnesses are distinguishable.
        sim.add_dft_plane_probe(axis="x", coordinate=.023, freqs=np.array([5e9]), name="read")
    sim.add_probe((.0231, .0161, .0139), "ez")
    values = [2., 10.] if fallback else [2., 6.]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sweep = vmap_material_sweep(sim, "slab.eps_r", values, **(
            {"n_steps": 800} if fallback == "spectral" else
            {"num_periods": 2} if fallback else {"n_steps": 400}))
        members = []
        for value in values:
            member = copy.deepcopy(sim)
            member._materials["slab"] = dataclasses.replace(member._materials["slab"], eps_r=value)
            members.append(member.run(n_steps=sweep.time_series.shape[1], skip_preflight=True))
    assert sweep.dt.shape == sweep.settling_db.shape == (2,)
    assert len(sweep.settling_witness) == 2
    for i, result in enumerate(members):
        np.testing.assert_array_equal(sweep.time_series[i], result.time_series)
        assert sweep.dt[i] == result.dt
        np.testing.assert_equal(sweep.settling_db[i], np.asarray(result.settling_db, dtype=float))
        for key in ("status", "db", "share_per_bin", "coverage", "reason"):
            np.testing.assert_equal(sweep.settling_witness[i][key], result.settling_witness[key])
    if fallback is True:
        assert sweep.dt[0] != sweep.dt[1]
        assert sweep.time_series.shape[1] == 105
    else:
        assert not np.array_equal(sweep.settling_witness[0]["share_per_bin"],
                                  sweep.settling_witness[1]["share_per_bin"])


@pytest.mark.parametrize("kind", ["crop", "flux", "moment"])
def test_s2b_realized_accumulator_corners_and_forward_jit(kind):
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    from rfx.api._spec import ForwardResult
    from rfx.measurement.coverage import accumulator_coverage

    # Unequal node spacings expose both index/physical-coordinate confusion
    # and half-open crop endpoints; the drive is off after sample 2.
    nodes = [np.array([0., 1., 3., 6., 10.])*C0,
             np.array([0., 2., 5., 9., 14.])*C0,
             np.array([0., 3., 7., 12., 18.])*C0]
    grid = SimpleNamespace(shape=(5, 5, 5), dt=1.,
                           node_of=lambda a, i: nodes["xyz".index(a)][i])
    result = ForwardResult(time_series=np.zeros((40, 0)), grid=grid, dt=1.)
    if kind == "crop":
        result = result._replace(dft_planes={"cropped": SimpleNamespace(axis=0, index=2,
            region=(1, 3, 1, 4))})
        far = np.array([3., 5., 12.])
    elif kind == "flux":
        result = result._replace(flux_monitors={"finite": SimpleNamespace(axis=1, index=2,
            lo1=1, hi1=3, lo2=1, hi2=-1)})
        far = np.array([6., 5., 18.])
    else:
        result = result._replace(current_moment_monitor=SimpleNamespace(
            i_lo=1, i_hi=4, j_lo=1, j_hi=4, k_lo=1, k_hi=4))
        far = np.array([6., 9., 18.])
    drive = SimpleNamespace(impedance=0., position=(0., 0., 0.),
                            waveform=lambda t: jnp.where(t < 2., 1., 0.))
    sim = SimpleNamespace(_ports=[drive], _msl_ports=[], _tfsf=None)
    coverage = accumulator_coverage(result, sim=sim)
    row, = coverage.rows()
    assert row["end_step"] == 2 + math.ceil(np.linalg.norm(far))
    assert row["verdict"] == "covered"
    # Forward preserves named rows/reasons as static metadata, not string leaves.
    carried = jax.jit(lambda x: ForwardResult(time_series=x,
        settling_accumulator_end_index=coverage.max_end(), settling_coverage=coverage))(
            jnp.zeros((40, 0)))
    assert carried.settling_witness["coverage"] == coverage.rows()
    assert int(carried.settling_accumulator_end_index) == row["end_step"]


class _Loop(Exception):
    def __init__(self, args, kwargs):
        super().__init__("loop entry")
        self.call = (args, kwargs)


def _scene(kind, graded):
    if kind == "coax":
        from rfx import Simulation
        mesh = {"dz_profile": np.full(60, .001)} if graded else {}
        sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9, boundary="cpml", **mesh)
        sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                             waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
        return sim
    sim = builders.build(ROWS[kind], "run_nonuniform" if graded else "run_uniform", graded=graded)
    if kind == "tfsf" and graded:
        # A plane wave along x needs uniform x cells (#2078); y and z stay graded.
        sim._dx_profile = np.full_like(np.asarray(sim._dx_profile), builders.DX)
    # A pulse that ends inside the record (the builders' waveform never ends).
    sim._ports = [dataclasses.replace(p, waveform=PULSE) for p in sim._ports]
    sim._msl_ports = [dataclasses.replace(p, waveform=PULSE) for p in sim._msl_ports]
    return sim


def _loop_call(monkeypatch, sim, kind):
    import rfx.runners.nonuniform as graded_runner
    import rfx.simulation as simulation

    def stop(*args, **kwargs):
        raise _Loop(args, kwargs)

    monkeypatch.setattr(simulation, "run", stop)
    monkeypatch.setattr(graded_runner, "run_nonuniform", stop)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if kind == "coax":
                sim.compute_coaxial_two_port(n_steps=N, freqs=np.array([8e9]))
            else:
                sim.run(n_steps=N, skip_preflight=True)
        except _Loop as stopped:
            return stopped.call
    raise AssertionError("the loop entry was not reached")


def _off(table, edge=False):
    """Last sample above 1e-6 of its own peak, plus one; None if still on."""
    w = np.asarray(table)[:N]
    w = np.pad(w, (0, N - len(w)), mode="edge" if edge else "constant")
    w = np.abs(w.astype(np.complex128)) if np.iscomplexobj(w) else np.abs(w.astype(np.float64))
    on = np.flatnonzero(w > 1e-6 * w.max())
    off = int(on[-1]) + 1 if on.size else 0
    return None if off >= N and w.max() > 0 else off


def _transit(distance, dt):
    return math.ceil(distance / (C0 * dt))


def _table(src):
    return np.asarray(src.waveform if hasattr(src, "waveform") else src[4])


def _node(grid, entry):
    idx = (entry.i, entry.j, entry.k) if hasattr(entry, "i") else entry[:3]
    return np.array([grid.node_of(axis, int(i)) for axis, i in zip("xyz", idx)])


def _analytic_v_inc(cfg, dt):
    """The analytic incident voltage the modal record stores (rfx/measurement/modal.py)."""
    t = np.arange(1, N + 1, dtype=np.float64) * dt
    arg = (t - float(cfg.src_t0)) / float(cfg.src_tau)
    return float(cfg.src_amp) * (-2.0 * arg) * np.exp(-arg ** 2)


def _expected(kind, sim, grid, call):
    """(whole-domain end, end with transit to sim's probes), from the applied tables."""
    _, kw = call
    dt = float(grid.dt)
    probes = [np.asarray(p.position, dtype=float) for p in sim._probes]
    whole, probed = [], []
    for src in (*(kw.get("sources") or ()), *(kw.get("mag_sources") or ())):
        whole.append(_off(_table(src)))
    if kind in ("lumped", "wire", "msl"):
        for pe in (*sim._ports, *sim._msl_ports):
            starts = [np.asarray(pe.position, dtype=float)]
            if getattr(pe, "extent", None):
                end = starts[0].copy()
                end["xyz".index(pe.component[1])] += pe.extent
                starts.append(end)
            reach = max(np.linalg.norm(s - p) for s in starts for p in probes)
            probed += [None if o is None else o + _transit(reach, dt) for o in whole]
    if kind == "floquet":
        (entry,) = sim._floquet_ports
        center = np.asarray(sim._domain, dtype=float) / 2
        center["xyz".index(entry.axis)] = entry.position
        reach = max(np.linalg.norm(center - p) for p in probes)
        probed = [o + _transit(reach, dt) for o in whole]
    for cfg in kw.get("waveguide_ports") or ():
        off = _off(cfg.e_inc_table, edge=True)
        whole.append(off)
        axis = "xyz".index(cfg.normal_axis)
        reach = max(abs(p[axis] - float(cfg.source_x_m)) for p in probes)
        probed.append(off + _transit(reach, dt))
    if kw.get("tfsf") is not None:
        cfg = kw["tfsf"][0]
        indices, corners, paths = _hand_tfsf_geometry(cfg, grid)
        off = _off(_hand_tfsf_samples(cfg, grid, N))
        whole.append(off + _transit(max(paths), dt))
        probed.append(off + _transit(max(path + max(np.linalg.norm(c - p) for p in probes)
                                        for path, c in zip(paths, corners)), dt))
        old = off + _transit(np.linalg.norm(np.ptp(corners, axis=0)), dt)
        print(f"S1 TF/SF {type(grid).__name__}: diagonal={old}, whole={whole[-1]}, probe={probed[-1]}")
    assert None not in whole and None not in probed, (whole, probed)
    return max(whole), max(probed)


CASES = [(kind, graded) for kind in (*ROWS, "coax") for graded in (False, True)]


@pytest.mark.parametrize("kind,graded", CASES,
                         ids=[f"{k}-{'graded' if g else 'uniform'}" for k, g in CASES])
def test_end_is_the_latest_end_of_the_applied_tables(kind, graded, monkeypatch):
    from rfx.measurement.record_end import drive_end, injection_end

    if (kind, graded) in NOT_REACHABLE:
        error = NotImplementedError if kind == "tfsf_closed_box" else ValueError
        with pytest.raises(error, match=re.escape(NOT_REACHABLE[kind, graded])):
            _loop_call(monkeypatch, _scene(kind, graded), kind)
        return
    sim = _scene(kind, graded)
    call = _loop_call(monkeypatch, sim, kind)
    args, kw = call
    grid = args[0]
    dt = float(grid.dt)
    if kind == "coax":
        # The coaxial drivers read every built E and H table with transit to
        # their witness probes; run() and forward() refuse a coaxial port.
        sources, mags, probes = kw["sources"], kw["mag_sources"], kw["probes"]
        assert sources and mags and probes
        ends = []
        for src in (*sources, *mags):
            reach = max(np.linalg.norm(_node(grid, src) - _node(grid, p)) for p in probes)
            ends.append(_off(_table(src)) + _transit(reach, dt))
        assert injection_end(N, dt, records=probes, grid=grid, sources=sources,
                             mag_sources=mags) == max(ends)
        with pytest.raises(NotImplementedError, match="not wired into Simulation.run"):
            _loop_call(monkeypatch, _scene(kind, graded), "run")
        return
    whole, probed = _expected(kind, sim, grid, call)
    tfsf = kw.get("tfsf")
    configs = list(kw.get("waveguide_ports") or ())
    # until_decay's reader: TF/SF incident transit through its realized box.
    assert drive_end(grid, N, sources=kw.get("sources") or (), mag_sources=kw.get("mag_sources") or (),
                     tfsf=tfsf, waveguide_ports=configs) == whole
    if configs:
        # The record a run returns holds the analytic incident voltage; the
        # scene is chosen so that it ends elsewhere than the applied E table.
        configs = [cfg._replace(v_inc_t=_analytic_v_inc(cfg, dt)) for cfg in configs]
        assert _off(configs[0].v_inc_t) != _off(configs[0].e_inc_table, edge=True)
    # The run()/forward() settling reader: the declarations plus the built
    # waveguide configs, transit to the user probes.
    assert injection_end(N, dt, records=sim._probes, grid=grid, sim=sim,
                         waveguide_ports=configs) == probed


def test_traced_and_still_on_read_one_text_on_every_path():
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace

    from rfx.measurement.record_end import STILL_ON, TRACED, injection_end

    on = SimpleNamespace(src_amp=1., e_inc_table=np.ones(50), source_x_m=0., normal_axis="x")
    step, reason, _ = injection_end(40, 1., waveguide_ports=[on], return_detail=True)
    assert step is None and reason == STILL_ON.format(source="waveguide port[0]")

    @jax.jit
    def traced(x):
        cfg = SimpleNamespace(src_amp=1., e_inc_table=x, source_x_m=0., normal_axis="x")
        assert injection_end(40, 1., waveguide_ports=[cfg], return_detail=True)[:2] == (None, TRACED)
        return x

    traced(jnp.ones(50))


TRACED_DZ = [0.5e-3] * 6 + [0.3e-3] * 5
# Below the Courant limit of the 0.5 x 0.5 x 0.3 mm cell and fixed, so only the mesh is traced.
TRACED_DT = 0.9 / (C0 * math.sqrt(2 / 0.5e-3 ** 2 + 1 / 0.3e-3 ** 2))
TRACED_N = 900   # past the pulse end, so a judgeable record has a concrete end


def _traced_dz_scene(kind, dz):
    from rfx import Simulation
    mm, depth = 1e-3, float(np.sum(TRACED_DZ))
    fixed = dict(dt=TRACED_DT, dt_min_cell=0.3e-3, dz_profile=dz, dx=0.5 * mm, freq_max=20e9)
    if kind == "plane":
        sim = Simulation(domain=(6 * mm, 6 * mm, depth), boundary="pec", **fixed)
        sim.add_source((3 * mm, 3 * mm, 1.5 * mm), "ez")
        sim.add_dft_plane_probe(axis="z", coordinate=1.0 * mm, component="ez",
                                freqs=np.array([5e9, 10e9]), name="p")
    else:
        sim = Simulation(domain=(12 * mm, 6 * mm, depth), boundary="cpml", cpml_layers=4, **fixed)
        sim.add_tfsf_source(f0=10e9, bandwidth=0.5, polarization="ez", direction="+x", margin=2)
    sim.add_probe((3 * mm, 3 * mm, 1.0 * mm), "ez")
    return sim


def _traced_dz_forward(kind, dz):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _traced_dz_scene(kind, dz).forward(n_steps=TRACED_N, skip_preflight=True)


@pytest.mark.parametrize("kind", ["plane", "tfsf"])
def test_s2b_traced_mesh_judges_declared_positions_only(kind):
    """A mesh that is a design variable puts realized nodes on tracers: a DFT
    plane's corners and the TF/SF box (a transverse z profile; the diagonal
    reach before S2b converted it too) read "cannot judge: traced". A probe
    and a source at declared positions, with a fixed dt, keep the end the
    concrete mesh gives. forward() still returns a finite gradient."""
    import jax
    import jax.numpy as jnp
    from rfx.measurement.record_end import TRACED

    def loss(dz):
        r = _traced_dz_forward(kind, dz)
        return jnp.sum(r.time_series ** 2), (r.settling_coverage, r.settling_probe_source_end_index)

    (_, (coverage, probe_end)), grad = jax.value_and_grad(loss, has_aux=True)(
        jnp.asarray(TRACED_DZ))
    assert np.all(np.isfinite(np.asarray(grad)))
    if kind == "plane":
        concrete = _traced_dz_forward(kind, np.asarray(TRACED_DZ)).settling_probe_source_end_index
        assert concrete is not None and probe_end == concrete
        assert coverage.entries == (("DFT plane p", None, TRACED_N, TRACED),)
    else:
        assert probe_end is None and coverage.entries == ()


SOURCE_END_NAMES = {"source_end_step", "simulation_source_end_step", "sampled_source_end_step",
                    "drive_end", "injection_end"}
CALLERS = ("rfx/api/_settling.py", "rfx/probes/sparam_driver.py", "rfx/sparams/_msl_stages.py",
           "rfx/sparams/_mixed_stages.py", "rfx/sparams/coax.py", "rfx/runners/uniform.py",
           "rfx/runners/nonuniform.py", "rfx/measurement/rcs.py", "rfx/measurement/decay.py",
           "rfx/measurement/settling.py", "rfx/ringdown/__init__.py")
# An off step from a drive table: a threshold relative to the table's peak.
OFF_STEP = re.compile(r"nonzero\([^\n]*>[^\n]*peak")


def test_no_module_outside_record_end_computes_a_source_end():
    home = REPO / "rfx/measurement/record_end.py"
    found = []
    for path in sorted((REPO / "rfx").rglob("*.py")):
        if path == home:
            continue
        text = path.read_text()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in SOURCE_END_NAMES:
                found.append(f"{path.relative_to(REPO)}:{node.lineno} defines {node.name}")
        found += [f"{path.relative_to(REPO)}: off step '{m.group(0)}'" for m in OFF_STEP.finditer(text)]
    assert not found, found
    for caller in CALLERS:
        text = (REPO / caller).read_text()
        assert "from rfx.measurement.record_end import" in text, caller
        assert not re.search(r"probes\.settling import[^\n]*(source_end_step|sampled_source_end_step)", text), caller


def test_floquet_end_is_the_same_under_an_outer_jit():
    """The Floquet table is built from the declared pulse; under a trace it
    must still be concrete, or the collector raised instead of judging."""
    import jax
    import jax.numpy as jnp

    from rfx.measurement.record_end import injection_end

    sim = _scene("floquet", False)
    grid = sim._build_grid()
    host = injection_end(N, grid.dt, records=sim._probes, grid=grid, sim=sim)
    seen = []

    @jax.jit
    def traced(x):
        seen.append(injection_end(N, grid.dt, records=sim._probes, grid=grid, sim=sim))
        return x

    traced(jnp.ones(2))
    assert host is not None and seen == [host]


# S2a: flux starts at the same applied-drive end as energy.
S2_BOX = ((.012, .012, .012), (.028, .028, .028))


def _s2_scene(graded=False, boundary="cpml"):
    from rfx import Simulation

    mesh = {"dz_profile": np.full(25, .0016)} if graded else {}
    sim = Simulation(domain=(.04, .04, .04), freq_max=10e9, dx=.0016,
                     boundary=boundary, **mesh)
    sim.add_source((.0193, .0207, .0211), "ez", amplitude_kind="current")
    sim.add_probe((.0251, .0172, .0222), "ez")
    return sim


@pytest.mark.parametrize("kind", ["energy", "flux"])
@pytest.mark.parametrize("min_steps,n_d", [(50, 0), (300, 0), (50, 320)])
def test_s2_rule_first_eligible_check(kind, min_steps, n_d, monkeypatch):
    from rfx.measurement.decay import DecayStop, FluxStop, finish_decay

    # Realized default pulse; the expected end uses this file's independent sum-free
    # table inspection, never record_end or a decay constructor/helper.
    with monkeypatch.context() as patch:
        _, kw = _loop_call(patch, _s2_scene(), "source")
    n_end = max(_off(_table(src)) for src in kw["sources"])
    assert n_end == 183
    interval, consecutive = 50, 2
    # thr=2 makes every positive level below threshold from the first check,
    # isolating the start gate from the physical ringdown time.
    common = dict(thr=2., min_steps=min_steps, interval=interval, n_end=n_end, n_d=n_d)
    if kind == "energy":
        rule = DecayStop(**common, consecutive=consecutive)
    else:
        rule = FluxStop(**common, consecutive=consecutive)
    ceil_check = lambda n: math.ceil(n / interval) * interval
    first = ceil_check(max(min_steps, n_end, n_d))
    expected = (max(ceil_check(n_end) + (consecutive - 1) * interval,
                    ceil_check(max(min_steps, n_d))) if kind == "energy"
                else first + (consecutive - 1) * interval)
    for count in range(interval, 501, interval):
        stopped = rule.check(count, lambda: 1.)
        if count < expected:
            assert stopped is None
        else:
            assert stopped == expected
            break
    else:
        pytest.fail(f"{kind} never stopped at its first eligible check {expected}")
    diagnostic = finish_decay(rule, stopped)
    assert diagnostic.values["drive_end_step"] == n_end
    assert diagnostic.values["stop_step"] == expected


@pytest.mark.parametrize("graded,boundary", [(False, "cpml"), (True, "cpml"), (False, "pec")])
def test_s2_run_waits_for_applied_drive(graded, boundary, monkeypatch, record_property):
    import rfx.runners.nonuniform as graded_runner
    import rfx.simulation as simulation

    owner = graded_runner if graded else simulation
    name = "run_nonuniform_until_decay" if graded else "run_until_decay"
    original, ends = getattr(owner, name), []

    def observed(*args, **kwargs):
        ends.append(max(_off(_table(src)) for src in kwargs["sources"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, observed)
    flux = boundary == "cpml"
    result = _s2_scene(graded, boundary).run(
        until_decay=2., decay_min_steps=50, decay_max_steps=350,
        **({"radiated_flux_box": S2_BOX} if flux else {}), skip_preflight=True)
    (diagnostic,) = [d for d in result.diagnostics if d.code == "until_decay.stop"]
    values = diagnostic.values
    assert ends == [183]
    assert values["stop_step"] >= ends[0]
    assert values["drive_end_step"] == ends[0]
    assert values["criterion"] == "threshold"
    assert values["quantity"] == ("radiated_flux" if flux else "conserved_energy")
    assert values["stop_step"] == 250
    record_property("s2_stop", dict(values))


@pytest.mark.parametrize("graded,boundary,decay", [
    (False, "pec", 1e-3), (False, "cpml", None), (True, "cpml", None),
    (False, "pec", None), (True, "pec", None), (False, "cpml", 0.),
])
def test_s2_refuses_a_flux_box_without_a_flux_stop(graded, boundary, decay):
    with pytest.raises(ValueError, match="radiated_flux_box selects the radiated-flux until_decay stop"):
        _s2_scene(graded, boundary).run(n_steps=20, until_decay=decay,
                                      radiated_flux_box=S2_BOX, skip_preflight=True)


def test_s2_direct_loop_refuses_a_closed_box_flux_box(monkeypatch):
    """The public loop called directly, not through run(): the box was dropped there too."""
    import rfx.simulation as simulation

    original = simulation.run_until_decay

    def stop(*args, **kwargs):
        raise _Loop(args, kwargs)

    monkeypatch.setattr(simulation, "run_until_decay", stop)
    with pytest.raises(_Loop) as entry:
        _s2_scene(False, "pec").run(until_decay=1e-3, decay_max_steps=60, skip_preflight=True)
    args, kwargs = entry.value.call
    assert kwargs["boundary"] == "pec"
    with pytest.raises(ValueError, match="radiated_flux_box selects the radiated-flux until_decay stop"):
        original(*args, **{**kwargs, "radiated_flux_box": S2_BOX})


@pytest.mark.parametrize("change", [{"decay_by": 0.0}, {"cpml_axes": ""}], ids=["no_stop", "closed"])
def test_s2_direct_graded_loop_refuses_an_unused_flux_box(change, monkeypatch):
    """The public graded loop called directly: a box with no flux stop, or with no absorber, is refused."""
    import rfx.runners.nonuniform as graded_runner

    original = graded_runner.run_nonuniform_until_decay

    def stop(*args, **kwargs):
        raise _Loop(args, kwargs)

    monkeypatch.setattr(graded_runner, "run_nonuniform_until_decay", stop)
    with pytest.raises(_Loop) as entry:
        _s2_scene(True).run(until_decay=1e-3, decay_max_steps=60, radiated_flux_box=S2_BOX,
                            skip_preflight=True)
    args, kwargs = entry.value.call
    assert kwargs["radiated_flux_box"] == S2_BOX and kwargs["decay_by"] > 0
    with pytest.raises(ValueError, match="radiated_flux_box selects the radiated-flux until_decay stop"):
        original(*args, **{**kwargs, **change})


def test_s2_direct_constructor_and_still_on_cap(monkeypatch):
    from rfx.measurement.decay import make_field_decay_stop, finish_decay

    with monkeypatch.context() as patch:
        args, kw = _loop_call(patch, _s2_scene(), "source")
    grid, sources = args[0], kw["sources"]
    expected = max(_off(_table(src)) for src in sources)
    rule = make_field_decay_stop(grid, thr=2., min_steps=50, interval=50,
                                max_steps=350, sources=sources)
    assert rule.n_end == expected
    # A shorter cap cuts the pulse while it is still injecting. Peak tracking
    # remains live, but the rule cannot stop; the diagnostic names the drive.
    capped = make_field_decay_stop(grid, thr=2., min_steps=0, interval=50,
                                  max_steps=100, sources=sources)
    assert capped.n_end is None
    for count in (50, 100):
        assert capped.check(count, lambda: 1.) is None
    diagnostic = finish_decay(capped, 100)
    assert diagnostic.values["criterion"] == "cap"
    assert math.isnan(diagnostic.values["drive_end_step"])
    assert "source[0]" in diagnostic.message and "still injecting" in diagnostic.message


def test_s2_flux_peak_keeps_drive_checks():
    from rfx.measurement.decay import FluxStop

    flux = FluxStop(.1, 0, 10, 2, 40, env_checks=2)
    for count, value in [(10, 100.), (20, 1.), (30, 1.), (40, 1.)]:
        assert flux.check(count, lambda: value) is None
    assert flux.peak == 100. and flux.below == 1
    assert flux.check(45, lambda: pytest.fail("off-interval flux read")) is None
    assert flux.check(50, lambda: 1.) == 50



@pytest.mark.parametrize("status", ["pass", "absent", "undetermined", "fail"])
@pytest.mark.parametrize("db", [None, -80., float("nan"), -10.])
def test_s2b_coverage_preserves_tail_value_and_nonpass_reason(status, db):
    from rfx.measurement.coverage import AccumulatorCoverage, with_coverage
    coverage = AccumulatorCoverage((("NTFF box", 241, 196, "cannot judge: coverage"),))
    original = {"status": status, "reason": "tail reason", "db": db}
    value, witness = with_coverage(db, original, coverage)
    assert value is db and witness["db"] is db
    assert witness["coverage"] == coverage.rows()
    assert witness["status"] == ("undetermined" if status in {"pass", "absent"} else status)
    assert witness["reason"] == ("cannot judge: coverage" if status in {"pass", "absent"} else "tail reason")


_RECORD_END_FILES = (
    'sparams/_tail_witness.py', 'measurement/settling.py', 'probes/sparam_driver.py',
    'measurement/waveguide_s.py', 'sparams/waveguide.py', 'sparams/_common.py',
    'sparams/_settling_warning.py', 'sparams/_msl_stages.py',
    'sparams/_mixed_stages.py', 'sparams/_coax_oneport_record.py', 'sparams/coax.py',
)


def test_s4_record_end_comparisons_have_one_owner():
    # This source-name check misses comparisons on other names. Ring-down,
    # gradients, decay and coverage (S5-S7) are not yet covered.
    violations = []
    for name in _RECORD_END_FILES:
        source = (REPO / 'rfx' / name).read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Compare):
                continue
            operands = [node.left, *node.comparators]
            def is_bar(value):
                if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.USub, ast.UAdd)):
                    value = value.operand
                return isinstance(value, ast.Constant) and value.value in (1e-2, 1e-3, 40)
            if any(is_bar(value) for value in operands) and any(
                    re.search('share|tail|error|settl|db|witness',
                              ast.get_source_segment(source, value) or '', re.I)
                    for value in operands if not is_bar(value)):
                violations.append((name, node.lineno))
    assert not violations


def test_s4_tail_delegate_and_reader_sites():
    tree = ast.parse((REPO / 'rfx/sparams/_tail_witness.py').read_text())
    body = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'tail_share_witness')
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == 'record_end_error' for n in ast.walk(body))
    for name, function, minimum in (
            ('probes/sparam_driver.py', 'port_record_witness', 1),
            ('sparams/_record_selection.py', 'port_record_witness', 1),
            ('measurement/waveguide_s.py', 'settling_db_from_port_records', 2)):
        tree = ast.parse((REPO / 'rfx' / name).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == function
                 and any(k.arg == 'reader' for k in n.keywords)]
        assert len(calls) >= minimum, name


def test_s4_probe_only_advisory_uses_attached_witness():
    """The tutorial's clipped record warns; a witnessed pass through P10 is silent."""
    from examples.tutorials.run_control_and_fields import build_simulation
    from rfx.errors import AccuracyWarning

    rows = {}
    for periods in (.1, 1, 3, 10):
        sim = build_simulation()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = sim.run(num_periods=periods, compute_s_params=False, skip_preflight=True)
        witness = result.settling_witness
        messages = [str(w.message) for w in caught if issubclass(w.category, AccuracyWarning)]
        rows[periods] = (witness, messages)
        print(f'probe-only P{periods}: status={witness["status"]}, reason={witness["reason"]!r}, '
              f'settling_db={result.settling_db}, warnings={messages}')
        assert sim._ntff is None and not sim._dft_planes and result.s_params is None
        assert not sim._internal_probe_indices
    clipped, messages = rows[.1]
    assert clipped['status'] in {'fail', 'absent', 'undetermined'}, clipped
    assert len(messages) == 1 and 'ring-down settling witness' in messages[0], messages
    passing = [(p, w, messages) for p, (w, messages) in rows.items() if w['status'] == 'pass']
    assert passing, f'No probe witness passes at 0.1, 1, 3 or 10 periods: {rows}'
    for periods, witness, messages in passing:
        assert not messages, (periods, witness, messages)


def _s4_closed_record(n, dt, poles, residues, bins, kind='E', factor=1.):
    z = np.exp(poles * dt)
    q = z[None, :] * np.exp(-2j * np.pi * bins[:, None] * dt)
    samples = np.exp(np.arange(n)[:, None] * poles[None, :] * dt) @ residues
    stamp = factor * np.exp(-2j * np.pi * bins * dt * (1. if kind == 'E' else .5))
    inside = stamp * dt * ((1 - q**n) / (1 - q) @ residues)
    # Whole-record origin: tail = dt exp(-iwNdt) sum c z**N/(1-z exp(-iwdt)).
    tail = stamp * dt * np.exp(-2j * np.pi * bins * n * dt) * (
        z[None, :]**n / (1 - q) @ residues)
    return samples.astype(np.complex128), inside, tail


@pytest.mark.parametrize('frame', ['wire-whole', 'wire-legacy', 'lumped', 'waveguide'])
def test_s4_reader_closed_pole_tail(frame):
    from rfx.sparams._tail_witness import _IDENTIFICATION
    from types import SimpleNamespace
    from rfx.measurement.record_readers import port_s_reader, waveguide_s_reader
    from rfx.measurement.record_end import record_end_error
    from rfx.measurement.port_time import PortTimeRecord
    from rfx.sparams._settling_groups import port_column_groups
    from rfx.sparams._tail_witness import tail_share_witness
    n, dt = 1500, 1e-12
    bins = np.array([9e9, 10.5e9, 12e9])
    poles = -np.log(200)/(n*dt) + 2j*np.pi*np.array([9e9, 12e9])
    waveguide = frame == 'waveguide'
    specs = ([('port0/v_ref_t', 'E', [1., .8]), ('port0/i_ref_t', 'H', [.007, .004]),
              ('port0/v_probe_t', 'E', [2., 1.]), ('port0/i_probe_t', 'H', [.01, .02])]
             if waveguide else [('V', 'E', [2., 1.]), ('I', 'H', [.007, .004]),
                                ('V_port', 'E', [1., .8]), ('V_ref', 'E', [.9, 1.2])])
    arrays, spectra, tails = {}, {}, {}
    for name, kind, residues in specs:
        arrays[name], spectra[name], tails[name] = _s4_closed_record(
            n, dt, poles, np.array(residues), bins, kind, 2. if waveguide else 1.)
    if waveguide:
        from rfx.model.waveguide_modes import PortMedium
        cfg = SimpleNamespace(freqs=bins, f_cutoff=0., mode_type='TE', medium=PortMedium(),
                              dt=dt, dx=0., direction='+x')
        # Zero cutoff is a TEM-like modal impedance, sqrt(mu0/eps0).
        from rfx.core.yee import MU_0, EPS_0
        z = np.sqrt(MU_0 / EPS_0)
        reader = waveguide_s_reader([cfg], 0)
        records = tuple(arrays.items())
        groups, labels = {name: name.split('_')[0] for name in arrays}, None
        operands = ({'port0/v_ref_t': .5, 'port0/i_ref_t': -.5*z},
                    {'port0/v_ref_t': .5, 'port0/i_ref_t': .5*z})
    else:
        kind = 'lumped' if frame == 'lumped' else 'wire'
        record = PortTimeRecord('p', kind, tuple(arrays), np.column_stack(list(arrays.values())))
        reader = port_s_reader([record], {'p': (50., 3, True)}, whole_port=frame != 'wire-legacy')
        records = (('p', record.samples),)
        groups, labels = port_column_groups([record])
        voltage = 'V' if frame == 'lumped' else 'V_port'
        sign = 1.
        if frame == 'wire-legacy':
            voltage, sign = 'V_ref', -1.
        operands = ({voltage: sign, 'I': -50.}, {voltage: sign, 'I': 50.})
    expected = sum(sum(abs(w)*abs(tails[name]) for name, w in operand.items()) /
                   abs(sum(w*spectra[name] for name, w in operand.items())) for operand in operands)
    expected_kappa = sum(sum(abs(w)*abs(spectra[name]) for name, w in operand.items()) /
                         abs(sum(w*spectra[name] for name, w in operand.items())) for operand in operands)
    actual = record_end_error(records, dt, 0, bins, freq_max=12e9,
                              identification=_IDENTIFICATION,
                              reader=reader, scale_groups=groups, _channel_labels=labels)
    legacy = tail_share_witness(records, dt, 0, bins, freq_max=12e9,
                                scale_groups=groups, _channel_labels=labels)
    np.testing.assert_allclose(actual.record_end_error_per_bin, expected, rtol=2e-4)
    np.testing.assert_allclose(actual.kappa_per_bin, expected_kappa, rtol=2e-5)
    np.testing.assert_allclose(actual.share_per_bin, legacy.share_per_bin, rtol=1e-9)
    for name, tail in tails.items():
        label = name if waveguide else f'p[{name}]'
        np.testing.assert_allclose(actual.tails_per_channel[label], tail, rtol=2e-4, atol=1e-18)


@pytest.mark.parametrize('whole_port', [True, False], ids=['whole-port', 'legacy-per-cell'])
def test_s4_two_port_closed_pole_tail(whole_port):
    """Off-diagonal S uses the receive impedance and the drive's incident wave."""
    from rfx.measurement.record_readers import port_s_reader
    from rfx.measurement.port_time import PortTimeRecord
    from rfx.sparams._tail_witness import port_record_witness

    n, dt = 1500, 1e-12
    bins = np.array([9e9, 10.5e9, 12e9])
    poles = -np.log(200)/(n*dt) + 2j*np.pi*np.array([9e9, 12e9])
    residues = {
        'drive': {'V': [2., 1.], 'V_port': [1., .8], 'V_ref': [.9, 1.2], 'I': [.007, .004]},
        'receive': {'V': [.7, 1.3], 'V_port': [1.8, .6], 'V_ref': [.4, .5], 'I': [.003, .008]},
    }
    records, spectra, tails = [], {}, {}
    for port, channels in residues.items():
        samples = []
        for name, values in channels.items():
            signal, spectrum, tail = _s4_closed_record(
                n, dt, poles, np.array(values), bins, 'H' if name == 'I' else 'E')
            samples.append(signal)
            spectra[port, name], tails[port, name] = spectrum, tail
        records.append(PortTimeRecord(port, 'wire', tuple(channels), np.column_stack(samples)))
    # By hand: receive Z=80, four live cells; drive Z=50, three live cells.
    # V_ref has the legacy sign. Neither numerator nor denominator is built
    # from the reader's returned weights.
    operands = (({('receive', 'V_port'): 1., ('receive', 'I'): -80.},
                 {('drive', 'V_port'): 1., ('drive', 'I'): 50.}) if whole_port else
                ({('receive', 'V'): 1., ('receive', 'I'): -80./4},
                 {('drive', 'V_ref'): -1., ('drive', 'I'): 50./3}))
    expected = sum(sum(abs(w)*abs(tails[key]) for key, w in weights.items()) /
                   abs(sum(w*spectra[key] for key, w in weights.items())) for weights in operands)
    _, actual = port_record_witness(records, dt, 0, bins, freq_max=12e9,
        reader=port_s_reader(records, {'drive': (50., 3, True), 'receive': (80., 4, False)},
                             whole_port=whole_port))
    measured = actual['record_end_error_per_entry']['receive/drive']
    print(f'two-port whole_port={whole_port}: bins={bins}, e_Q={measured}, expected={expected}')
    np.testing.assert_allclose(measured, expected, rtol=2e-4)


def test_s4_dropped_channel_tail_decides_verdict():
    """A small in-record DFT does not remove its weighted absolute tail."""
    from rfx.measurement.record_end import record_end_error
    from rfx.sparams._tail_witness import _IDENTIFICATION

    n, dt = 1500, 1e-12
    bins = np.array([9e9])
    poles = -np.log(200)/(n*dt) + 2j*np.pi*np.array([9e9, 12e9])
    q = np.exp((poles - 2j*np.pi*bins[0])*dt)
    d = dt*(1-q**n)/(1-q)
    strong, x, tx = _s4_closed_record(n, dt, poles, np.array([1., 0.]), bins)
    small, y, ty = _s4_closed_record(n, dt, poles, .05*np.array([1., -.9*d[0]/d[1]]), bins)
    weight = 40.
    expected = (abs(tx) + weight*abs(ty)) / abs(x + weight*y)
    without_small = abs(tx) / abs(x + weight*y)
    assert np.all(abs(y) < 1e-2*abs(x))
    assert np.all(without_small < 1e-2) and np.all(expected > 1e-2)
    actual = record_end_error([('strong', strong), ('small', small)], dt, 0, bins,
        identification=_IDENTIFICATION, freq_max=12e9,
        scale_groups={'strong': 'E', 'small': 'E'},
        reader={'entries': {'Q': ({'strong': 1., 'small': weight},)},
                'sampling': {'strong': ('E', 1.), 'small': ('E', 1.)}})
    assert any('small' in row['channels'] for row in actual.dropped_per_bin)
    print(f'dropped channel: |y/x|={abs(y/x)}, e_Q={actual.record_end_error_per_bin}, '
          f'expected={expected}, without_small={without_small}')
    assert actual.status == 'fail', actual
    np.testing.assert_allclose(actual.record_end_error_per_bin, expected, rtol=2e-4)


def test_s4_deep_null_fails():
    from rfx.measurement.record_readers import port_s_reader
    from rfx.measurement.port_time import PortTimeRecord
    from rfx.sparams._tail_witness import port_record_witness
    n, dt, z0 = 1500, 1e-12, 50.
    bins = np.array([9e9])
    poles = -np.log(200)/(n*dt) + 2j*np.pi*np.array([9e9, 12e9])
    q = np.exp((poles - 2j*np.pi*bins[0])*dt)
    d = dt*(1-q**n)/(1-q)
    # Solve b_stamped/a_stamped=1e-3, including E at (n+1)dt and
    # H at (n+1/2)dt. With A=sum(d), B=.1*delta*d[0], the two
    # waves are ((E-H)A+(E+H)B)/2 and ((E+H)A+(E-H)B)/2.
    e, h = np.exp(-2j*np.pi*bins[0]*dt*np.array([1., .5]))
    ratio = 1e-3
    delta = sum(d)*(ratio*(e+h)-(e-h))/((e+h)-ratio*(e-h)) / (.1*d[0])
    a = np.ones(2)
    b = .1*np.array([1, -(1-delta)*d[0]/d[1]])
    v, vd, vt = _s4_closed_record(n, dt, poles, (a+b)/2, bins)
    i, idft, it = _s4_closed_record(n, dt, poles, (a-b)/(2*z0), bins, 'H')
    outgoing, incoming = vd-z0*idft, vd+z0*idft
    kappa_b = (abs(vd)+z0*abs(idft))/abs(outgoing)
    kappa_a = (abs(vd)+z0*abs(idft))/abs(incoming)
    expected = (abs(vt)+z0*abs(it))*(1/abs(outgoing)+1/abs(incoming))
    np.testing.assert_allclose(outgoing/incoming, ratio, rtol=1e-10)
    np.testing.assert_allclose(kappa_b, 1e3, rtol=1e-3)
    record = PortTimeRecord('p', 'wire', ('V_port', 'I'), np.column_stack([v, i]))
    _, result = port_record_witness([record], dt, 0, bins, freq_max=12e9,
                                    reader=port_s_reader([record], {'p': (z0, 1, True)}))
    assert result['share_per_bin'][0] < 1e-2
    assert result['status'] == 'fail', result
    assert result['record_end_error_per_bin'][0] > 1e-2
    np.testing.assert_allclose(result['kappa_per_bin'], kappa_b+kappa_a, rtol=2e-5)
    np.testing.assert_allclose(result['record_end_error_per_bin'], expected, rtol=2e-4)
    print(f'deep null: bins={bins}, delta={delta}, kappa_b={kappa_b}, '
          f'e_Q={result["record_end_error_per_bin"]}')


@pytest.mark.parametrize('seed', range(20))
def test_s4_noisy_two_poles_nonpass(seed):
    from rfx.sparams._tail_witness import tail_share_witness
    n, dt = 2400, 1e-12
    bins = np.array([9e9, 10.5e9, 12e9])
    lo, hi = 1e-10, 1e-9
    for _ in range(60):
        tau = (lo+hi)/2
        poles = -1/tau + 2j*np.pi*np.array([9e9, 12e9])
        signal, inside, tail = _s4_closed_record(n, dt, poles, np.ones(2), bins)
        if max(abs(tail)/abs(inside)) < .03:
            lo = tau
        else:
            hi = tau
    signal += np.random.default_rng(seed).normal(0, 1e-4*max(abs(signal)), n)
    result = tail_share_witness([('M3', signal)], dt, 0, bins, freq_max=12e9)
    assert result.status != 'pass', (seed, result)


@pytest.fixture(scope='module')
def s4_spg_records():
    import time
    import rfx.measurement.waveguide_s as extraction
    from tests.oracle.test_waveguide_port_validation_battery import _build_sim
    captured, results = [], {}
    original = extraction.settling_db_from_port_records
    def observe(configs, **kwargs):
        captured.append(tuple(configs))
        return original(configs, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(extraction, 'settling_db_from_port_records', observe)
        for periods in (30, 40):
            sim = _build_sim(np.linspace(4.5e9, 8.0e9, 10), solved_domain=True,
                obstacles=[((0.05, 0.0, 0.0), (0.07, 0.04, 0.02), 4.0)],
                waveform='modulated_gaussian')
            start = time.monotonic()
            result = sim.compute_waveguide_s_matrix(num_periods=periods, normalize=False)
            elapsed = time.monotonic() - start
            results[periods] = (result, tuple(captured[-2:]), elapsed)
            print(f'SP-g P{periods}: {elapsed:.6f} seconds')
    return results


def test_s4_b8_realized_transit_window(s4_spg_records):
    result, runs, _ = s4_spg_records[40]
    for configs, witness in zip(runs, result.settling_witness):
        n, dt = len(configs[0].v_ref_t), configs[0].dt
        expected = 0
        for cfg in configs:
            if not cfg.src_amp:
                continue
            table = abs(np.asarray(cfg.e_inc_table))
            off = int(np.flatnonzero(table[:n] > 1e-6*max(table[:n]))[-1]) + 1
            reach = max(abs(float(point) - float(cfg.source_x_m)) for other in configs
                        for point in (other.reference_x_m, other.probe_x_m))
            expected = max(expected, off + math.ceil(reach / (C0*dt)))
        assert expected == 2181
        assert witness['window_starts'][n][0] == max(round(n/2), expected)


def test_s4_spg_s11_lengthening_reduces_error(s4_spg_records):
    short = s4_spg_records[30][0].settling_witness[0]
    long = s4_spg_records[40][0].settling_witness[0]
    assert short['status_per_entry']['S11'] == 'fail', short
    e30 = short['record_end_error_per_entry']['S11'][-1]
    e40 = long['record_end_error_per_entry']['S11'][-1]
    assert e30 > 1e-2
    assert e40 < e30
    assert long['status_per_entry']['S11'] == 'pass', long
    assert e40 <= 1e-2
    print(f'SP-g S11 at 8 GHz: P30 e_Q={e30:.12g}, P40 e_Q={e40:.12g}')


def test_s4_spg_s22_windows_disagree(s4_spg_records):
    result = s4_spg_records[30][0]
    witness = result.settling_witness[1]
    main = np.asarray(witness['record_end_error_per_entry']['S22'])
    second = np.asarray(witness['record_end_error_second_per_entry']['S22'])
    straddles = (main > 1e-2) != (second > 1e-2)
    assert np.any(straddles), (
        f'SP-g P30 S22 scene no longer straddles 1e-2: main={main}, second={second}')
    k = int(np.flatnonzero(straddles)[0])
    print(f'SP-g P30 S22 at {float(result.freqs[k]):.12g} Hz: '
          f'main e_Q={main[k]:.12g}, second e_Q={second[k]:.12g}')
    assert witness['status_per_entry']['S22'] == 'undetermined', witness
    assert 'S22: windows disagree' in witness['reason'], witness
