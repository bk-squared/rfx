"""Live wire loads and owning-slab recordings on uniform run(devices=)."""
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources.sources import GaussianPulse

pytestmark = pytest.mark.distributed
FREQS = np.array([1, 2.5, 5, 7.5, 10]) * 1e9
# Summed quantities: 1e-4 of peak across traces, not relative error at zeros.
# These tighter bars are declared before measuring float32 owning-slab sums.
S_ATOL = 2e-5
TRACE_RTOL = 2e-5


def _model(case="ez", planes=False, n_devices=2):
    ex = case == "ex"
    def xyz(x, y, z):
        return (z, y, x) if ex else (x, y, z)
    sim = Simulation(freq_max=10e9, domain=xyz(.024, .016, .024 if ex and n_devices > 2 else .012),
                     dx=.001, boundary="cpml", cpml_layers=4)
    for lo, hi in ((.003, .004), (.008, .009)):
        sim.add(Box(xyz(0, .003, lo), xyz(.024, .013, hi)), material="pec")
    for p, x in enumerate((.005, .019)):
        extra = dict(extent=.004)
        if case == "mixed" and p == 0:
            extra = {}
        if planes:
            extra.update(reference_plane_cells=2, direction="-x" if p == 0 else "+x")
        sim.add_port(xyz(x, .008, .004), "ex" if ex else "ez", impedance=150,
                     waveform=GaussianPulse(f0=5e9, bandwidth=1.6), **extra)
    if case == "dead":
        sim.add(Box((.004, .007, .004), (.006, .009, .005)), material="pec")
    if case == "dielectric":
        sim.add_material("substrate", eps_r=4.)
        sim.add(Box((.007, .003, .004), (.017, .013, .008)), material="substrate")
    # Include all wire cells, the seam cell, and a transmitted field away from drive.
    for x in (.005, .019):
        for z in (.004, .005, .006, .007):
            sim.add_probe(xyz(x, .008, z), "ex" if ex else "ez")
    sim.add_probe(xyz(.012, .008, .006), "ex" if ex else "ez")
    for component in ("hx", "hy", "hz"):
        sim.add_probe(xyz(.005, .008, .006), component)
    return sim


def _live(sim):
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.sources.sources import _wire_port_live_cells, wire_port_from_entry
    grid = sim._build_grid()
    mask = sim._assemble_materials(grid)[3]
    edges = realized_pec_edge_masks(mask)
    cells, flags, _ = _wire_port_live_cells(grid, wire_port_from_entry(sim._ports[0]), edges)
    return grid, [c for c, live in zip(cells, flags) if live]


def _parity(case="ez", mode="explicit", n_devices=2):
    sim = _model(case, n_devices=n_devices)
    if case in ("ex", "dead"):
        grid, cells = _live(sim)
        if case == "ex":
            width = (grid.shape[0] + n_devices - 1) // n_devices
            owners = [c[0] // width for c in cells]
            print(f"Ex live={cells} owners={owners} slab_width={width}")
            assert len(set(owners)) > 1
        else:
            assert len(cells) == 3
    kwargs = dict(n_steps=512, skip_preflight=True)
    if mode == "explicit":
        kwargs.update(compute_s_params=True, s_param_freqs=FREQS, s_param_n_steps=576)
    elif mode == "true":
        kwargs.update(compute_s_params=True)
    elif mode == "freqs":
        kwargs.update(s_param_freqs=FREQS)
    elif mode == "steps":
        kwargs.update(s_param_n_steps=576)
    expected = sim.run(**kwargs)
    actual = sim.run(devices=jax.devices("cpu")[:n_devices], **kwargs)
    np.testing.assert_array_equal(actual.freqs, expected.freqs)
    delta = float(np.max(np.abs(actual.s_params - expected.s_params)))
    trace = float(np.max(np.abs(actual.time_series - expected.time_series)))
    peak = float(np.max(np.abs(expected.time_series)))
    print(f"wire case={case} mode={mode} devices={n_devices} max_delta_S={delta:.12g} "
          f"trace_relative_peak={trace / peak:.12g} bars={S_ATOL},{TRACE_RTOL}")
    assert delta <= S_ATOL, f"S parity max_delta={delta}"
    assert delta <= 1e-4 * np.max(np.abs(expected.s_params))
    assert trace <= TRACE_RTOL * peak, f"trace parity relative_peak={trace / peak}"
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        a, b = np.asarray(getattr(actual.state, component)), np.asarray(getattr(expected.state, component))
        field_delta = float(np.max(np.abs(a - b)))
        # Normalize residual final fields to the recorded pulse peak in the
        # same units (E or H), not a nearly-zero late-time component.
        group = expected.time_series[:, :9] if component[0] == "e" else expected.time_series[:, 9:]
        field_peak = float(np.max(np.abs(group)))
        assert field_delta <= TRACE_RTOL * field_peak, (component, field_delta, field_peak)
    return actual


@pytest.mark.parametrize("case", ["ez", "ex", "dead", "dielectric"])
def test_wire_parity(case):
    _parity(case)


# The one-device refusal of a mixed lumped + wire port set, verbatim as at
# beb0d3fd (rfx/probes/sparam_driver.py). Both lanes must keep raising it.
_BASE_MIXED_REFUSAL = (
    "compute_lumped_wire_s_matrix_via_scan: mixed lumped + wire port "
    "sets are not supported in Stage 1 (the off-diagonal wave-"
    "decomposition conventions differ).  Use a homogeneous all-lumped "
    "or all-wire port set."
)


@pytest.mark.parametrize("distributed", [False, True])
def test_mixed_refusal_matches_baseline(distributed):
    sim = _model("mixed")
    options = dict(n_steps=8, compute_s_params=True, s_param_freqs=FREQS,
                   skip_preflight=True)
    with pytest.raises(NotImplementedError) as actual:
        sim.run(devices=jax.devices("cpu")[:2] if distributed else None, **options)
    assert str(actual.value) == _BASE_MIXED_REFUSAL


@pytest.mark.parametrize("case", ["ez"])
def test_distributed_vi_dump_refused(case):
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    sim = _model()
    with pytest.raises(NotImplementedError, match="return_vi_dump=True.*wire ports"):
        compute_lumped_wire_s_matrix_via_scan(
            sim, FREQS, n_steps=8, devices=jax.devices("cpu")[:2], return_vi_dump=True)


@pytest.mark.parametrize("mode", ["default", "true", "freqs", "steps"])
def test_options(mode):
    _parity(mode=mode)


@pytest.mark.skipif(os.environ.get("RFX_LOCAL_DISTRIBUTED") != "1",
                    reason="3/4 CPU devices: opt-in local subprocess checks")
@pytest.mark.parametrize("n_devices", [3, 4])
def test_more_devices(n_devices):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]),
               XLA_FLAGS="--xla_force_host_platform_device_count=4", JAX_PLATFORMS="cpu")
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(n_devices)],
                          env=env, capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    print(proc.stdout)


def test_single_wire_and_opt_out():
    sim = _model()
    sim._ports.pop()
    for mode in (False, True, None):
        options = dict(n_steps=512, compute_s_params=mode, s_param_freqs=FREQS,
                       s_param_n_steps=512, skip_preflight=True)
        expected = sim.run(**options)
        actual = sim.run(devices=jax.devices("cpu")[:2], **options)
        if mode is False:
            assert actual.s_params is None
        else:
            np.testing.assert_allclose(actual.s_params, expected.s_params, atol=S_ATOL, rtol=0)
        delta = np.max(np.abs(actual.time_series - expected.time_series))
        assert delta <= TRACE_RTOL * np.max(np.abs(expected.time_series))
    for devices in (None, jax.devices("cpu")[:2]):
        with pytest.raises(NotImplementedError, match="s_param_n_steps"):
            sim.run(n_steps=8, s_param_n_steps=16, devices=devices, skip_preflight=True)


def test_single_wire_with_plain_source_is_refused_on_both_lanes():
    """A plain source fires in every port drive (#1420): one device and
    devices= refuse it alike, before the first step."""
    sim = Simulation(freq_max=10e9, domain=(.016, .006, .006), dx=.001, boundary="pec")
    wave = GaussianPulse(f0=5e9, bandwidth=1.6)
    sim.add_port((.008, .003, .002), "ez", impedance=50, extent=.002, waveform=wave)
    sim.add_source((.004, .003, .003), "ez", waveform=wave)
    options = dict(n_steps=320, s_param_freqs=FREQS, skip_preflight=True)
    for devices in (None, jax.devices("cpu")[:2]):
        with pytest.raises(NotImplementedError, match="plain sources"):
            sim.run(devices=devices, **options)


def test_single_wire_zero_wave():
    sim = _model()
    sim._ports = [replace(sim._ports[0], waveform=lambda t: 0.)]
    options = dict(n_steps=8, s_param_freqs=FREQS, skip_preflight=True)
    expected = sim.run(**options)
    actual = sim.run(devices=jax.devices("cpu")[:2], **options)
    np.testing.assert_array_equal(expected.s_params, np.zeros((1, 1, len(FREQS))))
    np.testing.assert_array_equal(actual.s_params, expected.s_params)


@pytest.mark.parametrize("kind,message", [
    ("planes", "reference_plane_cells"), ("radius", "radius"),
    ("passive", "passive port"), ("graded", "(?i)lumped / wire ports"), ("pmc", "PMC")])
@pytest.mark.parametrize("compute_s", [None, True, False])
def test_refusals(kind, message, compute_s):
    from rfx.boundaries.spec import Boundary, BoundarySpec
    sim = _model(planes=kind == "planes")
    if kind in ("radius", "passive"):
        sim._ports[0] = replace(sim._ports[0], **({"radius": .0001} if kind == "radius" else {"excite": False}))
    if kind == "graded":
        sim = Simulation(freq_max=10e9, domain=(.016, .006, .006), dx=.001,
                         boundary="pec", dz_profile=np.array([.001]*3 + [.0005]*6))
        sim.add_port((.008, .003, .003), "ez", impedance=50, extent=.002)
    if kind == "pmc":
        sim._boundary_spec = BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                                          y=Boundary(lo="pmc", hi="cpml"),
                                          z=Boundary(lo="cpml", hi="cpml"))
    with pytest.raises(NotImplementedError, match=message):
        sim.run(n_steps=8, devices=jax.devices("cpu")[:2], skip_preflight=True,
                compute_s_params=compute_s)


@pytest.mark.parametrize("mutation", ["no_stamps", "seam_stamp", "own_slab", "half_step"])
def test_mutations(monkeypatch, mutation):
    from rfx.sources import sources
    from rfx.probes import sparam_driver as driver
    from rfx.core import dft_utils
    from rfx.runners import distributed_v2 as runner
    import jax.numpy as jnp
    original_run = runner.run_distributed
    def mutated_run(*args, **kwargs):
        with monkeypatch.context() as patch:
            original_stamp = sources.stamp_lumped_sigma
            if mutation == "no_stamps":
                patch.setattr(sources, "setup_wire_port", lambda grid, wp, materials, **kw: materials)
            elif mutation == "seam_stamp":
                def stamp(materials, cell, sigma, component):
                    return materials if cell[0] == 11 else original_stamp(materials, cell, sigma, component)
                patch.setattr(sources, "stamp_lumped_sigma", stamp)
            result = original_run(*args, **kwargs)
            if mutation == "own_slab" and kwargs.get("_record_probes") is not None:
                # Ex line x=8,9,10,11 belongs to slabs 0,0,0,1. Drop
                # precisely the E contribution outside each port's own slab.
                assert len(kwargs["_record_probes"]) == 16
                samples = np.array(result.time_series)
                samples[:, [3, 11]] = 0
                result = result._replace(time_series=samples)
            return result
    monkeypatch.setattr(runner, "run_distributed", mutated_run)
    if mutation == "half_step":
        original_dft = driver._lumped_recording_dfts
        def dft(*args):
            with monkeypatch.context() as patch:
                patch.setattr(dft_utils, "half_step_current_phase",
                              lambda f, dt: jnp.ones_like(f, dtype=jnp.complex64))
                return original_dft(*args)
        monkeypatch.setattr(driver, "_lumped_recording_dfts", dft)
    with pytest.raises(AssertionError, match="S parity|trace parity"):
        _parity("ex")
    print(f"wire mutation={mutation}: parity RED")


@pytest.mark.parametrize("case,planes", [("ez", False), ("ez", True), ("lumped", False), ("lumped_distributed", False)])
def test_one_device_bytes_against_main(monkeypatch, case, planes):
    from rfx.probes import sparam_driver as driver
    from tests.unit.runners.test_distributed_lumped_port_s import _model as lumped_model
    sim = lumped_model() if case.startswith("lumped") else _model(case, planes=planes)
    kwargs = dict(n_steps=512, s_param_freqs=FREQS, skip_preflight=True)
    if case == "lumped_distributed":
        kwargs["devices"] = jax.devices("cpu")[:2]
    actual = sim.run(**kwargs)
    # Read the main revision locally, execute its driver in memory. No checkout,
    # secondary package copy, or measurement artifact is created.
    try:
        source = subprocess.check_output(
            ["git", "show", "beb0d3fd:rfx/probes/sparam_driver.py"], text=True,
            stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        pytest.skip("main baseline object beb0d3fd is absent from this checkout")
    namespace = dict(vars(driver))
    exec(compile(source, "<main-sparam-driver>", "exec"), namespace)
    monkeypatch.setattr(driver, "compute_lumped_wire_s_matrix_via_scan",
                        namespace["compute_lumped_wire_s_matrix_via_scan"])
    expected = sim.run(**kwargs)
    for name in ("s_params", "freqs", "time_series"):
        a, b = np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name))
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        print(f"main-byte-parity case={case} planes={planes} {name} entries={a.size} max_delta=0")
    for component in ("ex", "ey", "ez", "hx", "hy", "hz", "step"):
        a = np.asarray(getattr(actual.state, component))
        b = np.asarray(getattr(expected.state, component))
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        print(f"main-byte-parity case={case} planes={planes} state.{component} entries={a.size} max_delta=0")


if __name__ == "__main__":
    for case in ("ez", "ex", "dead", "dielectric"):
        _parity(case, n_devices=int(sys.argv[1]))
