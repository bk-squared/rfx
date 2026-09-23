"""Refuse four input classes before distributed field updates.

Exercise the public dispatch and both runner entries with two CPU devices.
PEC controls use the relative tolerance from
``test_distributed.TestDistributedRunner.test_distributed_matches_single_pec``.
"""

import warnings

import jax
import numpy as np
import pytest

from rfx import Box, DebyePole, GaussianPulse, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.runners import distributed, distributed_v2


N_STEPS = 40
ENTRIES = ("api", "v2", "v1")
LANES = {
    "api": "distributed multi-device run()",
    "v2": "distributed (v2) runner",
    "v1": "distributed (v1) pmap runner",
}


def _devices():
    devices = jax.devices("cpu")
    assert len(devices) >= 2, "requires the root conftest's two CPU devices"
    return devices[:2]


def _build(*, entry="api", periodic="", boundary="pec", port=None,
           flux=False, ntff=False, kerr=False, debye=False, rlc=False):
    """A PEC box has a source at x=6 mm and Ez probes at x=12 and 22 mm."""
    # v1 requires nx divisible by two; v2 pads the 25-cell x grid.
    domain_x = 23e-3 if entry == "v1" else 24e-3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        sim = Simulation(freq_max=15e9, domain=(domain_x, 12e-3, 12e-3),
                         dx=1e-3, boundary=boundary)
        if periodic:
            sim.set_periodic_axes(periodic)
        if port is None:
            sim.add_source(position=(6e-3, 6e-3, 6e-3), component="ez",
                           amplitude_kind="field")
        else:
            sim.add_port(position=(6e-3, 6e-3, 6e-3), component="ez", **port)
        for x in (12e-3, 22e-3):
            sim.add_probe(position=(x, 6e-3, 6e-3), component="ez")
        if flux:
            sim.add_flux_monitor(axis="x", coordinate=12e-3, n_freqs=3)
        if ntff:
            sim.add_ntff_box((4e-3, 4e-3, 4e-3), (20e-3, 8e-3, 8e-3),
                             n_freqs=3)
        if kerr:
            sim.add_material("kerr", eps_r=2.0, chi3=1e-2)
            sim.add(Box((8e-3, 3e-3, 3e-3), (16e-3, 9e-3, 9e-3)), material="kerr")
        if debye:
            sim.add_material("debye", eps_r=2.0,
                             debye_poles=[DebyePole(delta_eps=1.5, tau=1e-11)])
            sim.add(Box((8e-3, 3e-3, 3e-3), (16e-3, 9e-3, 9e-3)), material="debye")
        if rlc:
            sim.add_lumped_rlc((9e-3, 6e-3, 6e-3), "ez", R=10.0)
    return sim


def _run(sim, entry, **kwargs):
    if entry == "api":
        return sim.run(n_steps=N_STEPS, devices=_devices(), **kwargs)
    runner = distributed_v2 if entry == "v2" else distributed
    return runner.run_distributed(sim, n_steps=N_STEPS, devices=_devices(), **kwargs)


def _assert_refused(sim, entry, *features, **kwargs):
    with pytest.raises(NotImplementedError) as exc:
        _run(sim, entry, **kwargs)
    message = str(exc.value)
    if entry == "api":
        remedy = "omit devices=... (use a single-device run() instead)"
    else:
        remedy = "call sim.run(...) without devices= instead of calling this runner"
        assert "omit devices=..." not in message, message
    for feature in (*features, LANES[entry], remedy, "rfx.runners."):
        assert feature in message, message


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("declaration", ("axes", "boundary_spec"))
def test_periodic_boundaries_are_refused(entry, declaration):
    """A box periodic in y must not run with the declared non-periodic wall."""
    if declaration == "axes":
        sim = _build(entry=entry, periodic="y")
    else:
        spec = BoundarySpec(
            x=Boundary(lo="pec", hi="pec"),
            y=Boundary(lo="periodic", hi="periodic"),
            z=Boundary(lo="pec", hi="pec"),
        )
        sim = _build(entry=entry, boundary=spec)
    assert sim._periodic_axes == "y"
    _assert_refused(sim, entry, "periodic", "'y'", "non-periodic wall")


@pytest.mark.parametrize("entry", ENTRIES)
def test_extended_lumped_port_is_refused(entry):
    """A 3 mm, 50 ohm port requires both its source and resistive termination."""
    sim = _build(entry=entry, port={"impedance": 50.0, "extent": 3e-3})
    _assert_refused(sim, entry, "extended lumped port", "extent=",
                    "neither a source nor its resistive termination")


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("waveform", ("explicit", "default"))
def test_passive_port_is_refused(entry, waveform):
    """A passive 50 ohm port has no applied excitation, including waveform=None."""
    port = {"impedance": 50.0, "excite": False}
    if waveform == "explicit":
        port["waveform"] = GaussianPulse(f0=7.5e9, bandwidth=0.8)
    sim = _build(entry=entry, port=port)
    assert (sim._ports[0].waveform is None) == (waveform == "default")
    _assert_refused(sim, entry, "excite=False", "drive the port", "waveform=None")


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("monitor,feature", (
    ("flux", "add_flux_monitor()"), ("ntff", "add_ntff_box()"),
))
def test_surface_monitor_is_refused(entry, monitor, feature):
    """Flux planes and NTFF boxes request surface fields from the driven box."""
    sim = _build(entry=entry, **{monitor: True})
    _assert_refused(sim, entry, feature, "would be None")


def test_explicit_bloch_phase_is_refused():
    """A Bloch boundary specifies a phase even when periodic axes are unset."""
    sim = _build()
    phase = (0.0, 0.1, 0.0)
    with pytest.raises(NotImplementedError, match="Bloch.*") as exc:
        distributed_v2.refuse_unsupported_distributed_features(
            sim, lane="direct helper", bloch=phase)
    assert ("call sim.run(...) without devices= instead of calling this runner"
            in str(exc.value))
    for entry in ("v2", "v1"):
        _assert_refused(_build(entry=entry), entry, "Bloch", bloch=phase)
    sim._bloch = phase
    _assert_refused(sim, "api", "Bloch")


@pytest.mark.parametrize("entry", ENTRIES)
def test_kerr_material_is_refused(entry):
    """A chi3 block between source and probe: this lane drops chi3 (linear physics)."""
    sim = _build(entry=entry, kerr=True)
    _assert_refused(sim, entry, "Kerr chi3", "'kerr'", "as linear")


@pytest.mark.parametrize("entry", ENTRIES)
def test_lumped_rlc_element_is_refused(entry):
    """A 10 ohm element between source and probe: this lane never applies it (#1239).

    Measured before the refusal (#1239's 12 mm box): the multi-device trace was
    bit-identical to the same box without the element, 43 % from single device.
    """
    sim = _build(entry=entry, rlc=True)
    _assert_refused(sim, entry, "add_lumped_rlc()", "as if the elements were absent")


@pytest.mark.parametrize("entry", ("api", "v2"))
def test_lumped_port_with_debye_block_matches_native(entry):
    """A 50 ohm port drives a box with a Debye block between port and probes.

    The staged dispersion coefficients must come from the materials the run
    actually uses (x-padded, with the port's resistive sigma folded in); a
    staging fed the unpadded or port-less materials moves this trace by more
    than the probe's own peak.
    """
    sim = _build(entry=entry, port={"impedance": 50.0}, debye=True)
    native = np.asarray(sim.run(n_steps=N_STEPS).time_series)
    multi = np.asarray(_run(sim, entry).time_series)
    assert np.max(np.abs(native)) > 0
    relative = np.max(np.abs(native - multi)) / np.max(np.abs(native))
    assert relative < 1e-4, f"{entry}: relative Ez difference {relative:.9e}"


@pytest.mark.parametrize("source", ("point", "lumped_port"))
def test_default_pec_model_matches_native(source):
    """A PEC box contains a point source or an excited single-cell 50 ohm port."""
    port = {"impedance": 50.0} if source == "lumped_port" else None
    for entry in ENTRIES:
        sim = _build(entry=entry, port=port)
        assert not sim._periodic_axes
        assert not sim._flux_monitors and sim._ntff is None
        assert all(p.extent is None and p.excite for p in sim._ports)
        native = np.asarray(sim.run(n_steps=N_STEPS).time_series)
        multi = np.asarray(_run(sim, entry).time_series)
        assert native.shape == multi.shape == (N_STEPS, 2)
        assert np.max(np.abs(native)) > 0
        relative = np.max(np.abs(native - multi)) / (np.max(np.abs(native)) + 1e-30)
        # Same PEC criterion as TestDistributedRunner.test_distributed_matches_single_pec.
        assert relative < 1e-4, f"{entry}: relative Ez difference {relative:.9e}"


def test_tfsf_and_waveguide_models_reach_single_device_fallback(monkeypatch):
    """TFSF or waveguide excitation with surface monitors uses the native fallback."""
    for kind in ("tfsf", "waveguide"):
        for entry in ENTRIES:
            sim = Simulation(freq_max=15e9, domain=(24e-3, 12e-3, 12e-3),
                             dx=1e-3, boundary="cpml")
            sim.add_flux_monitor(axis="x", coordinate=12e-3, n_freqs=3)
            sim.add_ntff_box((4e-3, 4e-3, 4e-3), (20e-3, 8e-3, 8e-3), n_freqs=3)
            if kind == "tfsf":
                sim.add_tfsf_source(f0=7.5e9, bandwidth=0.5)
            else:
                sim.add_waveguide_port(
                    x_position=6e-3, y_range=(2e-3, 10e-3),
                    z_range=(2e-3, 10e-3), mode=(1, 0), mode_type="TE",
                    direction="+x", f0=7.5e9, bandwidth=0.5,
                )
            public_run = sim.run
            calls = []
            sentinel = object()

            def native_run(**kwargs):
                calls.append(kwargs)
                return sentinel

            monkeypatch.setattr(sim, "run", native_run)
            with pytest.warns(UserWarning, match="Falling back to single-device"):
                result = (public_run(n_steps=N_STEPS, devices=_devices(), skip_preflight=True)
                          if entry == "api" else _run(sim, entry))
            assert result is sentinel
            assert calls == [{"n_steps": N_STEPS}]
