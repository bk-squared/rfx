"""Graded run routing: single-device parity, not an independent physics oracle.

Unequal faces, graded cells, off-plane probes, off-grid body dimensions and
sources/probes on opposite sides of a slab cut exercise the routed physics.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, LorentzPole, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")


def model(case):
    absorbing = case.startswith("cpml")
    boundary = (BoundarySpec(x=Boundary(lo="pec", hi="cpml"), y="cpml",
                             z=Boundary(lo="cpml", hi="pec"))
                if absorbing else "pec")
    sim = Simulation(
        freq_max=15e9, domain=(15.3e-3, 7.7e-3, 8.2e-3), dx=1e-3,
        dx_profile=np.r_[1e-3, np.linspace(.85e-3, 1.15e-3, 13), 1e-3],
        boundary=boundary, cpml_layers=2 if absorbing else 0,
    )
    sim.add_source((5.1e-3, 3.2e-3, 4.3e-3), "ez", amplitude_kind="field",
                   waveform=lambda t: jnp.exp(-((t-1.2e-11)/5e-12)**2))
    for pos in ((7.3e-3, 2.1e-3, 5.4e-3), (9.6e-3, 4.2e-3, 3.1e-3)):
        for field in FIELDS:
            sim.add_probe(pos, field)
    if case in ("debye", "lorentz"):
        poles = ({"debye_poles": [DebyePole(delta_eps=1.7, tau=1e-11)]}
                 if case == "debye" else
                 {"lorentz_poles": [LorentzPole(
                     omega_0=2*np.pi*8e9, delta=2*np.pi*1e9,
                     kappa=(2*np.pi*8e9)**2)]})
        sim.add_material("ade", eps_r=2.3, **poles)
        sim.add(Box((6.1e-3, 1.4e-3, 2.2e-3), (10.7e-3, 5.6e-3, 6.3e-3)),
                material="ade")
    if case in ("pec_body", "cpml_pec"):
        sim.add(Box((7.1e-3, 4.1e-3, 2.2e-3), (9.4e-3, 6.2e-3, 5.3e-3)),
                material="pec")
    return sim


def arrays(result):
    return dict(time_series=np.asarray(result.time_series), **{
        field: np.asarray(getattr(result.state, field)) for field in FIELDS})


pytestmark = pytest.mark.distributed


@pytest.mark.parametrize("case", ["pec", "pec_body", "cpml", "debye", "lorentz", "cpml_pec"])
@pytest.mark.parametrize("n_devices", [2, 3])
def test_graded_run_matches_single_device(case, n_devices, record_property):
    devices = jax.devices("cpu")[:n_devices]
    if len(devices) < n_devices:
        pytest.skip(f"needs {n_devices} virtual CPU devices")
    single = model(case).run(n_steps=40, skip_preflight=True)
    distributed = model(case).run(n_steps=40, devices=devices, skip_preflight=True)
    want, got = arrays(single), arrays(distributed)
    assert np.max(np.abs(want["time_series"])) > 1e-5
    movements = {}
    for name in want:
        a, b = got[name], want[name]
        assert a.shape == b.shape and a.dtype == b.dtype == np.float32
        assert np.isfinite(a).all() and np.isfinite(b).all()
        peak = np.max(np.abs(b))
        error = np.max(np.abs(a - b))
        ulps = float(error / np.spacing(peak)) if peak else float(error)
        record_property(name + "_peak_ulps", ulps)
        print(f"{case}/{n_devices}/{name}: {ulps:g} peak ULP")
        movements[name] = ulps
    # Summed observable gets the separate cross-trace accumulation bar.
    squared_sum = sum(np.sum(a.astype(np.float64)**2) for a in got.values())
    reference_sum = sum(np.sum(a.astype(np.float64)**2) for a in want.values())
    sum_error = abs(squared_sum-reference_sum) / reference_sum
    record_property("summed_relative_error", sum_error)
    print(f"{case}/{n_devices}/summed: {sum_error:g} relative")
    assert sum_error <= 1e-4
    assert int(distributed.state.step) == 40
    assert distributed.dt == single.dt
    assert distributed.grid.shape == single.grid.shape
    assert distributed.realized_geometry is not None
    assert distributed.settling_witness is not None
    for name in ("s_params", "freqs", "ntff_data", "ntff_box", "dft_planes",
                 "flux_monitors", "snapshots", "snapshot_axes", "wire_port_sparams",
                 "waveguide_ports", "waveguide_sparams", "waveguide_port_flux",
                 "ringdown", "current_moment_data", "current_moment_monitor"):
        assert getattr(distributed, name) is None, name
    assert all(ulp <= 9 for ulp in movements.values()), (case, n_devices, movements)


def test_graded_run_empty_probes_and_resolved_steps(two_devices):
    sim = model("pec")
    sim._probes.clear()
    expected = sim._nu_n_steps(.2)
    result = sim.run(num_periods=.2, devices=two_devices, skip_preflight=True)
    assert result.time_series.shape == (expected, 0)
    assert int(result.state.step) == expected


@pytest.mark.parametrize("feature, message", [
    ("port", "Lumped / wire ports"),
    ("ntff", "NTFF box"),
    ("cpml_kappa", "cpml_kappa_max"),
    ("s_params", "compute_s_params"),
])
def test_graded_run_refuses_unimplemented_inputs(feature, message, two_devices):
    sim = model("cpml")
    kwargs = {}
    if feature == "port":
        sim.add_port((6e-3, 3e-3, 3e-3), "ez", impedance=50.)
        kwargs["compute_s_params"] = False
    elif feature == "ntff":
        sim.add_ntff_box((3e-3, 2e-3, 2e-3), (12e-3, 6e-3, 6e-3), freqs=[5e9])
    elif feature == "cpml_kappa":
        sim._cpml_kappa_max = 2.
    else:
        kwargs["compute_s_params"] = True
    with pytest.raises((NotImplementedError, ValueError), match=message):
        sim.run(n_steps=4, devices=two_devices, skip_preflight=True, **kwargs)
