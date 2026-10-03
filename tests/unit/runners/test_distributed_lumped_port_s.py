"""Owning-cell recordings feed the production lumped S-matrix decomposer."""

import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time

import jax
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

pytestmark = pytest.mark.distributed
FREQS = np.array([1, 2.5, 5, 7.5, 10]) * 1e9
# Summed quantities must agree within 1e-4 of the peak across all traces
# (cross-trace rule, not a per-bin relative tolerance near zeros). The
# tighter absolute S bar below leaves margin for float32 reduction rounding.
S_ATOL = 9e-6
TRACE_RTOL = 9e-6


def _plate_line(impedance=40.):
    """One-cell gap, w/h=10 PEC plates extending through the x-high CPML."""
    sim = Simulation(freq_max=10e9, domain=(24e-3, 16e-3, 6e-3),
                     dx=1e-3, boundary="cpml", cpml_layers=4)
    for lo, hi in ((1e-3, 2e-3), (3e-3, 4e-3)):
        sim.add(Box((3e-3, 3e-3, lo), (30e-3, 13e-3, hi)), material="pec")
    # eta0*h/w = 37.673 ohm; 40 ohm is the nearby one-device sweep choice.
    sim.add_port((4e-3, 8e-3, 2e-3), "ez", impedance=impedance,
                 waveform=GaussianPulse(f0=5e9, bandwidth=1.6))
    sim.add_probe(sim._ports[0].position, "ez")
    assert not sim._boundary_spec.pmc_faces()
    assert not sim._boundary_spec.pec_faces()
    return sim


def _model(seam=True, ports=2, component="ez", channel=False, n_devices=2, boundary="cpml", magnetic=False):
    if channel:
        sim = Simulation(freq_max=10e9, domain=(20e-3, 1e-3, 1e-3), dx=1e-3, cpml_layers=4,
                         boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                                               y=Boundary.from_string("pmc" if magnetic else "cpml"),
                                               z=Boundary(lo="pec", hi="pec")))
    else:
        sim = Simulation(freq_max=10e9, domain=(16e-3, 6e-3, 6e-3), dx=1e-3,
                         boundary=boundary, cpml_layers=2)
    grid = sim._build_grid()
    width = (grid.shape[0] + n_devices - 1) // n_devices
    node = width if seam else width - 2
    for x in ([node] if ports == 1 else [node, node - 2]):
        pos = ((x - grid.pad_x_lo) * grid.dx,
               0.0 if channel else 3e-3, 0.0 if channel else 3e-3)
        sim.add_port(pos, component, impedance=188.365156834 if channel else 50,
                     waveform=GaussianPulse(f0=5e9, bandwidth=1.6))
        assert grid.position_to_index(pos)[0] == x
    sim.add_probe(sim._ports[0].position, component)
    return sim


def _parity(*, seam=True, ports=2, mode="explicit", channel=False,
            component="ez", n_devices=2, case=None, n_steps=256):
    started = time.monotonic()
    sim = _model(seam, ports, component, channel, n_devices)
    freqs = FREQS
    if case == "plate_line":
        sim = _plate_line()
        freqs = np.linspace(1e9, 10e9, 19)
    if case in ("debye", "lossy", "eps4"):
        extra = {"debye_poles": [DebyePole(delta_eps=1., tau=1e-11)]} if case == "debye" else {"sigma": 0.05} if case == "lossy" else {}
        sim.add_material("block", eps_r=4., **extra)
        sim.add(Box((4e-3, 2e-3, 2e-3), (10e-3, 4e-3, 4e-3)), material="block")
    elif case == "pec":
        # Keep the body strictly between the ports, away from both port edges.
        sim._ports[1] = replace(sim._ports[1], position=(4e-3, 3e-3, 3e-3))
        sim.add(Box((6e-3, 2e-3, 2e-3), (7e-3, 4e-3, 4e-3)), material="pec")
    elif case == "cpml_box":
        # Unequal drive impedances in a CPML box, without an extra source.
        sim._ports[0] = replace(sim._ports[0], impedance=75.)
    kwargs = {"n_steps": n_steps, "skip_preflight": True}
    if mode == "explicit":
        kwargs.update(compute_s_params=True, s_param_freqs=freqs, s_param_n_steps=max(320, n_steps))
    elif mode == "true":
        kwargs.update(compute_s_params=True)
    elif mode == "freqs":
        kwargs.update(s_param_freqs=FREQS)
    elif mode == "steps":
        kwargs.update(s_param_n_steps=max(320, n_steps))
    # time_series comes from the one-device MAIN run; S has its own drive scans.
    reference = sim.run(**kwargs)
    actual = sim.run(devices=jax.devices("cpu")[:n_devices], **kwargs)
    assert actual.s_params is not None
    assert actual.s_params.shape == reference.s_params.shape
    np.testing.assert_array_equal(actual.freqs, reference.freqs)
    error = float(np.max(np.abs(actual.s_params.astype(np.complex128)
                               - reference.s_params.astype(np.complex128))))
    minimum = float(np.min(20 * np.log10(np.abs(reference.s_params[0, 0]))))
    print(f"devices={n_devices} seam={seam} ports={ports} mode={mode} "
          f"component={component} channel={channel} max_delta={error:.12g} "
          f"min_S11_dB={minimum:.9g} gate={S_ATOL:g}")
    assert actual.time_series.shape == (n_steps, 1)
    peak = float(np.max(np.abs(reference.time_series)))
    trace_error = float(np.max(np.abs(actual.time_series - reference.time_series))) / peak
    print(f"case={case} steps={n_steps} time_series_relative_peak_error={trace_error:.12g} gate={TRACE_RTOL:g}")
    assert trace_error <= TRACE_RTOL, f"time_series parity {trace_error}"
    assert error <= S_ATOL, f"S parity max_delta={error} gate={S_ATOL}"
    assert error <= 1e-4 * np.max(np.abs(reference.s_params))
    assert len(sim._probes) == 1 and all(pe.excite for pe in sim._ports)
    if case == "plate_line":
        k = int(np.argmin(np.abs(reference.s_params[0, 0])))
        print(f"plate_line Z0={sim._ports[0].impedance:g} ohm min_S11_dB={minimum:.9g} "
              f"bin={k} frequency_Hz={reference.freqs[k]:g} "
              f"parity_runtime_s={time.monotonic() - started:.3f}")
        assert minimum <= -10
    if channel:
        # The all-nonmagnetic replacement radiates transversely as well as
        # along x; it is not the former PMC TEM matched-line fixture.
        assert minimum <= -1
        assert np.max(np.abs(reference.s_params[1, 0])) > 0.1
    return actual


@pytest.mark.parametrize("seam", [False, True], ids=["interior", "seam"])
@pytest.mark.parametrize("ports", [1, 2])
@pytest.mark.parametrize("mode", ["default", "explicit"])
def test_all_entries_all_bins(seam, ports, mode):
    _parity(seam=seam, ports=ports, mode=mode)


@pytest.mark.parametrize("mode", ["default", "explicit", "true", "freqs", "steps"])
def test_cpml_channel(mode):
    _parity(channel=True, mode=mode)


def test_matched_plate_line():
    _parity(case="plate_line", ports=1, n_steps=512)


@pytest.mark.parametrize("component", ["ex", "ey"])
def test_other_components(component):
    _parity(component=component)


@pytest.mark.parametrize("kind, message", [("wire", "reference_plane_cells"),
                                            ("passive", "passive port"),
                                            ("graded", "Phase B distributed\\+NU")])
@pytest.mark.parametrize("explicit", [False, True])
def test_refusals(kind, message, explicit):
    kwargs = {"dz_profile": np.array([1e-3] * 3 + [0.5e-3] * 6)} if kind == "graded" else {}
    sim = Simulation(freq_max=10e9, domain=(16e-3, 6e-3, 6e-3),
                     dx=1e-3, boundary="pec", **kwargs)
    extra = {"extent": 1e-3, "reference_plane_cells": 1, "direction": "+x"} if kind == "wire" else {"excite": False} if kind == "passive" else {}
    sim.add_port((8e-3, 3e-3, 3e-3), "ez", impedance=50, **extra)
    with pytest.raises(NotImplementedError, match=message):
        sim.run(n_steps=8, devices=jax.devices("cpu")[:2],
                **({"compute_s_params": True} if explicit else {}))


@pytest.mark.skipif(os.environ.get("RFX_LOCAL_DISTRIBUTED") != "1",
                    reason="3/4 CPU devices: opt-in local subprocess checks")
@pytest.mark.parametrize("n_devices", [3, 4])
def test_more_devices(n_devices):
    root = str(Path(__file__).resolve().parents[3])
    env = dict(os.environ, PYTHONPATH=root,
               XLA_FLAGS="--xla_force_host_platform_device_count=4", JAX_PLATFORMS="cpu")
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(n_devices)],
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    print(proc.stdout)


@pytest.mark.parametrize("mutation", ["disable", "own_slab", "half_step"])
def test_parity_detects_mutations(monkeypatch, mutation):
    from rfx.probes import sparam_driver as driver
    from rfx.runners import distributed_v2 as runner
    from rfx.core import dft_utils
    import jax.numpy as jnp

    if mutation == "disable":
        original = driver.compute_lumped_wire_s_matrix_via_scan

        def disabled(sim, freqs, **kwargs):
            if kwargs.get("devices") is not None:
                return np.zeros((2, 2, len(freqs)), dtype=np.complex64), freqs
            return original(sim, freqs, **kwargs)

        monkeypatch.setattr(driver, "compute_lumped_wire_s_matrix_via_scan", disabled)
    elif mutation == "own_slab":
        original = runner.sample_probes_shmap

        def wrong_owner(st, mesh, count, specs, owners, **kwargs):
            if count == 10:
                # Keep each sample's local index, but incorrectly read all
                # four H samples from the port's slab, including seam H(i-1).
                owners = [owners[(k // 5) * 5] for k in range(count)]
            return original(st, mesh, count, specs, owners, **kwargs)

        monkeypatch.setattr(runner, "sample_probes_shmap", wrong_owner)
    else:
        original = driver._lumped_recording_dfts

        def no_half_step(*args):
            # Only mutate replay; the single-device reference keeps its phase.
            with monkeypatch.context() as patch:
                patch.setattr(dft_utils, "half_step_current_phase",
                              lambda freqs, dt: jnp.ones_like(freqs, dtype=jnp.complex64))
                return original(*args)

        monkeypatch.setattr(driver, "_lumped_recording_dfts", no_half_step)
    with pytest.raises(AssertionError, match="S parity max_delta"):
        _parity(channel=True)
    print(f"mutation={mutation}: parity RED")


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("channel", [False, True])
def test_pmc_matched_line_refused(explicit, channel):
    from tests.unit.ports.test_lumped_two_port_matched_line import _build
    sim = _build("lumped")
    if channel:
        sim = _model(channel=True, magnetic=True)
        # This row checks admission, not the old half-cell line's matching.
    with pytest.raises(NotImplementedError, match=r"PMC magnetic face.*y_lo.*distributed_v2.*magnetic image"):
        sim.run(n_steps=8, devices=jax.devices("cpu")[:2], skip_preflight=True,
                **({"compute_s_params": True} if explicit else {}))


@pytest.mark.parametrize("face", ["x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"])
@pytest.mark.parametrize("explicit", [False, True])
def test_each_magnetic_face_refused_before_scan(monkeypatch, face, explicit):
    from rfx.runners import distributed_v2 as runner
    axes = {axis: Boundary(lo="cpml", hi="cpml") for axis in "xyz"}
    axis, side = face.split("_")
    axes[axis] = replace(axes[axis], **{side: "pmc"})
    sim = Simulation(freq_max=10e9, domain=(16e-3, 6e-3, 6e-3), dx=1e-3,
                     boundary=BoundarySpec(**axes), cpml_layers=2)
    sim.add_port((8e-3, 3e-3, 3e-3), "ez", impedance=50)

    def unexpected(*args, **kwargs):
        pytest.fail("magnetic S request must refuse before any distributed scan")

    monkeypatch.setattr(runner, "run_distributed", unexpected)
    with pytest.raises(NotImplementedError, match=rf"PMC magnetic face.*{face}.*distributed_v2.*magnetic image"):
        sim.run(n_steps=8, devices=jax.devices("cpu")[:2], skip_preflight=True,
                **({"compute_s_params": True} if explicit else {}))


@pytest.mark.parametrize("case", ["debye", "lossy", "pec", "cpml_box", "eps4"])
def test_material_and_cpml_box_parity(case):
    _parity(case=case)


@pytest.mark.parametrize("case", ["soft_pec", "legacy_soft_pec"])
@pytest.mark.parametrize("distributed", [False, True])
def test_pec_plain_source_refused(case, distributed):
    sim = _model(boundary="pec")
    sim.add_source((4e-3, 3e-3, 3e-3), "ez",
                   **({} if case == "legacy_soft_pec" else {"amplitude_kind": "field"}))
    with pytest.raises(NotImplementedError, match="plain sources.*compute_s_params=False"):
        sim.run(n_steps=8, skip_preflight=True,
                devices=jax.devices("cpu")[:2] if distributed else None)


def test_long_record():
    # Small domain keeps this 2048-step record in the fast suite.
    _parity(n_steps=2048)


def test_one_scan_per_drive_and_shared_replay_bundle(monkeypatch):
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    from rfx.probes.probes import PortVIReplayBundle
    from rfx.runners import distributed_v2 as runner
    calls = []
    original = runner.run_distributed

    def recorded(sim, **kwargs):
        calls.append((kwargs.get("_source_port_indices"),
                      len(kwargs.get("_record_probes", ())), kwargs["n_steps"]))
        return original(sim, **kwargs)

    monkeypatch.setattr(runner, "run_distributed", recorded)
    sim = _model(channel=True)
    expected = compute_lumped_wire_s_matrix_via_scan(sim, FREQS, n_steps=320,
                                                   return_vi_dump=True)
    actual = compute_lumped_wire_s_matrix_via_scan(sim, FREQS, n_steps=320,
                                                  return_vi_dump=True,
                                                  devices=jax.devices("cpu")[:2])
    assert calls == [((0,), 10, 320), ((1,), 10, 320)]
    assert isinstance(actual, PortVIReplayBundle)
    assert actual.port_names == expected.port_names
    assert actual.driven_port_indices == expected.driven_port_indices
    np.testing.assert_array_equal(actual.freqs, expected.freqs)
    np.testing.assert_array_equal(actual.port_impedances, expected.port_impedances)
    np.testing.assert_allclose(actual.s_params, expected.s_params, rtol=0, atol=S_ATOL)
    for name in ("voltages", "currents"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   rtol=1e-5, atol=0)


def _assert_shared_step_order(source):
    """Lock the common numerical step, including cases where walls are inert."""
    import ast
    tree = ast.parse(source)
    steps = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
             and node.name in ("step_fn_cpml", "step_fn_pec")]
    assert len(steps) == 2
    for step in steps:
        calls = [node.value for node in step.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Call)]
        names = [call.func.id for call in calls if isinstance(call.func, ast.Name)]
        assert names.count("_inject_sources_shmap") == 1, "shared step injection order"
        assert names.index("_inject_sources_shmap") < names.index("_apply_pec_shmap"), "shared step injection order"
        assert not any(isinstance(node, ast.Name) and node.id.startswith(("_sparam", "_source_port", "_record_probe"))
                       for node in ast.walk(step)), "drive-dependent step"


def test_shared_step_order():
    import inspect
    from rfx.runners import distributed_v2 as runner
    _assert_shared_step_order(inspect.getsource(runner.run_distributed))


def test_drive_only_order_mutation(monkeypatch):
    import inspect
    from rfx.runners import distributed_v2 as runner
    source = inspect.getsource(runner.run_distributed)
    source = source.replace("        st = _inject_sources_shmap(st, src_vals, ranks=ranks)",
                            "        if _source_port_indices is None:\n            st = _inject_sources_shmap(st, src_vals, ranks=ranks)")
    source = source.replace("        # 7. Exchange E ghost cells", "        if _source_port_indices is not None:\n            st = _inject_sources_shmap(st, src_vals, ranks=ranks)\n\n        # 7. Exchange E ghost cells")
    source = source.replace("        # 6. Exchange E ghost cells", "        if _source_port_indices is not None:\n            st = _inject_sources_shmap(st, src_vals, ranks=ranks)\n\n        # 6. Exchange E ghost cells")
    namespace = dict(vars(runner))
    exec(compile(source, "<drive-only-order-mutation>", "exec"), namespace)
    monkeypatch.setattr(runner, "run_distributed", namespace["run_distributed"])
    # All these sources are away from walls/masks: numerical parity alone
    # cannot require a shared step. The structural invariant must also fail.
    for case in (None, "debye", "lossy", "pec", "cpml_box", "eps4"):
        _parity(case=case)
    with pytest.raises(AssertionError, match="shared step injection order"):
        _assert_shared_step_order(source)
    print("mutation=drive_only_order: numerical parity GREEN; shared step check RED")


def test_host_dft_warning_fix_preserves_s(monkeypatch):
    import warnings
    import jax.numpy as jnp
    from rfx.core import dft_utils
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    sim = _model()
    kwargs = dict(n_steps=320, devices=jax.devices("cpu")[:2])
    with warnings.catch_warnings(record=True) as caught:
        actual, _ = compute_lumped_wire_s_matrix_via_scan(sim, FREQS, **kwargs)
    assert not any("float64 requested" in str(w.message) for w in caught)

    def old_phase(step, freqs, dt):
        t = jnp.asarray(step, dtype=jnp.float32) * dt
        return jnp.exp(-1j * 2. * jnp.pi * jnp.asarray(freqs).astype(jnp.float64)
                       * t.astype(jnp.float64)).astype(jnp.complex64) * dt

    old_half = dft_utils.half_step_current_phase
    monkeypatch.setattr(dft_utils, "port_dft_phase", old_phase)
    monkeypatch.setattr(dft_utils, "half_step_current_phase",
                        lambda f, dt: old_half(f.astype(jnp.float64), dt))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        before, _ = compute_lumped_wire_s_matrix_via_scan(sim, FREQS, **kwargs)
    np.testing.assert_array_equal(actual, before)
    print(f"host_dft_warning_fix: {actual.size} complex S entries bit-identical; max_delta=0")


def test_explicit_opt_out_keeps_s_options_inert(monkeypatch):
    from rfx.probes import sparam_driver

    def unexpected(*args, **kwargs):
        pytest.fail("compute_s_params=False must not start S scans")

    monkeypatch.setattr(sparam_driver, "compute_lumped_wire_s_matrix_via_scan", unexpected)
    result = _model().run(n_steps=8, devices=jax.devices("cpu")[:2],
                          compute_s_params=False, s_param_freqs=FREQS,
                          s_param_n_steps=320, skip_preflight=True)
    assert result.s_params is None and result.freqs is None
    assert result.time_series.shape == (8, 1)


if __name__ == "__main__":
    _parity(n_devices=int(sys.argv[1]))
    _parity(n_devices=int(sys.argv[1]), channel=True)
    _parity(n_devices=int(sys.argv[1]), case="plate_line", ports=1, n_steps=512)
    _parity(n_devices=int(sys.argv[1]), case="cpml_box")
