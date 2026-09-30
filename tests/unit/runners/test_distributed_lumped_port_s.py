"""Owning-cell recordings feed the production lumped S-matrix decomposer."""

import os
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

pytestmark = pytest.mark.distributed
FREQS = np.array([1, 2.5, 5, 7.5, 10]) * 1e9
# Measured max 2.2043e-6 across 2/3/4 CPUs; 9e-6 is 4.08x for
# float32 scan/reduction rounding across compiler versions (rtol=0).
S_ATOL = 9e-6


def _model(seam=True, ports=2, component="ez", matched=False, n_devices=2):
    if matched:
        sim = Simulation(freq_max=10e9, domain=(20e-3, 1e-3, 1e-3), dx=1e-3, cpml_layers=4,
                         boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                                               y=Boundary(lo="pmc", hi="pmc"),
                                               z=Boundary(lo="pec", hi="pec")))
    else:
        sim = Simulation(freq_max=10e9, domain=(16e-3, 6e-3, 6e-3), dx=1e-3,
                         boundary="cpml", cpml_layers=2)
    grid = sim._build_grid()
    width = (grid.shape[0] + n_devices - 1) // n_devices
    node = width if seam else width - 2
    for x in ([node] if ports == 1 else [node, node - 2]):
        pos = ((x - grid.pad_x_lo) * grid.dx,
               0.0 if matched else 3e-3, 0.0 if matched else 3e-3)
        sim.add_port(pos, component, impedance=188.365156834 if matched else 50,
                     waveform=GaussianPulse(f0=5e9, bandwidth=1.6))
        assert grid.position_to_index(pos)[0] == x
    sim.add_probe(sim._ports[0].position, component)
    return sim


def _parity(*, seam=True, ports=2, mode="explicit", matched=False,
            component="ez", n_devices=2):
    sim = _model(seam, ports, component, matched, n_devices)
    kwargs = {"n_steps": 256, "skip_preflight": True}
    if mode == "explicit":
        kwargs.update(compute_s_params=True, s_param_freqs=FREQS, s_param_n_steps=320)
    elif mode == "true":
        kwargs.update(compute_s_params=True)
    elif mode == "freqs":
        kwargs.update(s_param_freqs=FREQS)
    elif mode == "steps":
        kwargs.update(s_param_n_steps=320)
    reference = sim.run(**kwargs)
    actual = sim.run(devices=jax.devices("cpu")[:n_devices], **kwargs)
    assert actual.s_params is not None
    assert actual.s_params.shape == reference.s_params.shape
    np.testing.assert_array_equal(actual.freqs, reference.freqs)
    error = float(np.max(np.abs(actual.s_params.astype(np.complex128)
                               - reference.s_params.astype(np.complex128))))
    minimum = float(np.min(20 * np.log10(np.abs(reference.s_params[0, 0]))))
    print(f"devices={n_devices} seam={seam} ports={ports} mode={mode} "
          f"component={component} matched={matched} max_delta={error:.12g} "
          f"min_S11_dB={minimum:.9g} gate={S_ATOL:g}")
    assert error <= S_ATOL, f"S parity max_delta={error} gate={S_ATOL}"
    assert actual.time_series.shape == (256, 1)
    assert len(sim._probes) == 1 and all(pe.excite for pe in sim._ports)
    if matched:
        assert minimum <= -10
        assert np.min(np.abs(reference.s_params[1, 0])) > 0.1
    return actual


@pytest.mark.parametrize("seam", [False, True], ids=["interior", "seam"])
@pytest.mark.parametrize("ports", [1, 2])
@pytest.mark.parametrize("mode", ["default", "explicit"])
def test_all_entries_all_bins(seam, ports, mode):
    _parity(seam=seam, ports=ports, mode=mode)


@pytest.mark.parametrize("mode", ["default", "explicit", "true", "freqs", "steps"])
def test_matched_line(mode):
    _parity(matched=True, mode=mode)


@pytest.mark.parametrize("component", ["ex", "ey"])
def test_other_components(component):
    _parity(component=component)


@pytest.mark.parametrize("kind, message", [("wire", "extended lumped port"),
                                            ("passive", "passive port"),
                                            ("graded", "Phase B distributed\\+NU")])
@pytest.mark.parametrize("explicit", [False, True])
def test_refusals(kind, message, explicit):
    kwargs = {"dz_profile": np.array([1e-3] * 3 + [0.5e-3] * 6)} if kind == "graded" else {}
    sim = Simulation(freq_max=10e9, domain=(16e-3, 6e-3, 6e-3),
                     dx=1e-3, boundary="pec", **kwargs)
    extra = {"extent": 1e-3} if kind == "wire" else {"excite": False} if kind == "passive" else {}
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
        _parity(matched=True)
    print(f"mutation={mutation}: parity RED")


@pytest.mark.parametrize("nodes", [5, 9])
def test_closed_matched_line(nodes):
    # Regression for the initial candidate: y-PMC must not clamp the port E.
    from tests.unit.ports.test_lumped_two_port_matched_line import _build
    sim = _build("lumped")
    if nodes == 9:
        sim = Simulation(freq_max=10e9, domain=(8e-3, 1e-3, 1e-3), dx=1e-3,
                         boundary=BoundarySpec(x=Boundary(lo="pmc", hi="pmc"),
                                               y=Boundary(lo="pmc", hi="pmc"),
                                               z=Boundary(lo="pec", hi="pec")))
        for node in (5, 3):
            sim.add_port((node * 1e-3, 0, 0), "ez", impedance=376.730313668,
                         waveform=GaussianPulse(f0=5e9, bandwidth=1.6))
    kwargs = (dict(n_steps=256, skip_preflight=True) if nodes == 9 else
              dict(n_steps=320, s_param_freqs=FREQS, skip_preflight=True))
    expected = sim.run(**kwargs)
    actual = sim.run(devices=jax.devices("cpu")[:2], **kwargs)
    error = float(np.max(np.abs(actual.s_params.astype(np.complex128)
                               - expected.s_params.astype(np.complex128))))
    minimum = float(np.min(20 * np.log10(np.abs(expected.s_params[0, 0]))))
    print(f"closed_matched nodes={nodes} max_delta={error:.12g} min_S11_dB={minimum:.9g}")
    assert error <= S_ATOL
    assert minimum <= -10


def test_one_scan_per_drive_and_shared_replay_bundle(monkeypatch):
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    from rfx.probes.probes import PortVIReplayBundle
    from rfx.runners import distributed_v2 as runner
    calls = []
    original = runner.run_distributed

    def recorded(sim, **kwargs):
        calls.append((kwargs.get("_sparam_drive_idx"),
                      len(kwargs.get("_sparam_probes", ())), kwargs["n_steps"]))
        return original(sim, **kwargs)

    monkeypatch.setattr(runner, "run_distributed", recorded)
    sim = _model(matched=True)
    expected = compute_lumped_wire_s_matrix_via_scan(sim, FREQS, n_steps=320,
                                                   return_vi_dump=True)
    actual = compute_lumped_wire_s_matrix_via_scan(sim, FREQS, n_steps=320,
                                                  return_vi_dump=True,
                                                  devices=jax.devices("cpu")[:2])
    assert calls == [(0, 10, 320), (1, 10, 320)]
    assert isinstance(actual, PortVIReplayBundle)
    assert actual.port_names == expected.port_names
    assert actual.driven_port_indices == expected.driven_port_indices
    np.testing.assert_array_equal(actual.freqs, expected.freqs)
    np.testing.assert_array_equal(actual.port_impedances, expected.port_impedances)
    np.testing.assert_allclose(actual.s_params, expected.s_params, rtol=0, atol=S_ATOL)
    for name in ("voltages", "currents"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                   rtol=1e-5, atol=0)


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
    _parity(n_devices=int(sys.argv[1]), matched=True)
