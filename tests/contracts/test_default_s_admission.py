"""Default and explicit S requests share admission before any time step."""
from dataclasses import replace
import warnings

import jax
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.errors import RfxUnsupported
from rfx.runners._admission import default_run_forms_s, s_drive_inventory
from tests.unit.sparams.test_mixed_port_sparam import _base_sim, _add_feed, _add_msl


KINDS = ("microstrip", "passive_microstrip", "waveguide", "Floquet", "TFSF", "coaxial")


def _scene(graded, kind, wire=True):
    sim, y = _base_sim(**({"dz_profile": np.full(9, 254e-6 / 3)} if graded else {}))
    if kind == "waveguide":
        sim.add_waveguide_port(.004, name="guide", f0=2.5e9)
    _add_feed(sim, y, extent=254e-6 if wire else 254e-6 / 3)
    if not wire:
        sim._ports[0] = replace(sim._ports[0], extent=None)
    if kind in ("microstrip", "passive_microstrip"):
        _add_msl(sim, y, name="strip", excite=kind == "microstrip")
    elif kind == "Floquet":
        source = Simulation(freq_max=5e9, domain=(.008, .003, .001), dx=.0001)
        source.add_floquet_port(.0005, name="plane", f0=2.5e9)
        # Registration rejects graded z meshes; check stored-entry admission too.
        sim._floquet_ports = source._floquet_ports
    elif kind == "TFSF":
        source = Simulation(freq_max=5e9, domain=(.008, .003, .001), dx=.0001)
        source.add_tfsf_source(f0=2.5e9, margin=1)
        # Public registration already refuses TFSF beside an impedance port.
        sim._tfsf = source._tfsf
    elif kind == "coaxial":
        sim.add_coaxial_port((.002, y, 0.), "bottom")
    return sim


@pytest.fixture
def before_step(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("stepping started before S admission refused the declaration")
    monkeypatch.setattr(jax.lax, "scan", forbidden)
    monkeypatch.setattr(jax.lax, "while_loop", forbidden)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("wire", [False, True], ids=["lumped", "wire"])
@pytest.mark.parametrize("distributed", [False, True], ids=["single", "devices"])
@pytest.mark.parametrize("kind", KINDS)
def test_default_equals_explicit_before_step(before_step, graded, wire, distributed, kind):
    errors = []
    for explicit in (False, True):
        sim = _scene(graded, kind, wire)
        options = {"compute_s_params": True} if explicit else {}
        if distributed:
            options["devices"] = [jax.devices()[0]] * 2
        with pytest.raises((NotImplementedError, ValueError)) as caught:
            sim.run(n_steps=1, skip_preflight=True, **options)
        errors.append(caught.value)
    assert type(errors[0]) is type(errors[1])
    assert str(errors[0]) == str(errors[1])
    assert not hasattr(errors[0], "code")
    assert ("add_coaxial_port()" if kind == "coaxial" else "single result schema") in str(errors[0])


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("distributed", [False, True], ids=["single", "devices"])
@pytest.mark.parametrize("kind", ["microstrip", "waveguide", "Floquet", "TFSF"])
def test_forward_inventory_before_step(before_step, graded, distributed, kind):
    sim = _scene(graded, kind)
    with pytest.raises(ValueError, match="isolated drive") as caught:
        sim.forward(n_steps=1, skip_preflight=True, port_s11_freqs=np.array([2.5e9]),
                    distributed=distributed, devices=[jax.devices()[0]] * 2 if distributed else None)
    message = str(caught.value)
    assert "wire port 0 at" in message
    assert {"microstrip": "microstrip strip", "waveguide": "waveguide guide",
            "Floquet": "Floquet plane", "TFSF": "TFSF source"}[kind] in message
    assert "run(compute_s_params=False)" in message
    if kind == "microstrip":
        assert "compute_mixed_s_matrix()" in message


@pytest.mark.parametrize("kind", ["microstrip", "waveguide", "Floquet", "TFSF", "coaxial"])
def test_scan_inventory_before_step(before_step, kind):
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    with pytest.raises(RfxUnsupported, match="foreign drives") as caught:
        compute_lumped_wire_s_matrix_via_scan(_scene(False, kind), np.array([2.5e9]),
                                             n_steps=1, skip_preflight=True)
    assert kind in str(caught.value)
    assert "run(compute_s_params=False)" in str(caught.value)


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("amplitude", [1.0, 0.0], ids=["driven_guide", "passive_guide"])
@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_waveguide_microstrip_inventory_only_before_step(before_step, entry, amplitude, mesh):
    from tests.contracts.test_waveguide_mixed_admission import _guide
    sim = _guide(mesh + "-" + entry)
    sim._waveguide_ports[0] = replace(sim._waveguide_ports[0], amplitude=amplitude)
    sim.add_msl_port((.020, .010, .002), width=.004, height=.002,
                     direction="-x", name="strip", n_probe_offset=None,
                     n_probe_spacing=None, n_probes=3)
    with pytest.raises(ValueError, match="isolated drive") as caught:
        getattr(sim, entry)(n_steps=1, skip_preflight=True)
    assert ("waveguide guide" in str(caught.value)) is (amplitude != 0.0)
    assert "microstrip strip" in str(caught.value)
    assert "compute_mixed_s_matrix()" in str(caught.value)
    assert "run(compute_s_params=False)" in str(caught.value)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("case,uniform_s,graded_s", [
    ("lumped", True, True), ("wire", True, True),
    ("passive_lumped", True, False), ("passive_wire", False, False),
    ("receiver", True, True), ("two_lumped", True, False),
    ("two_wire", True, True), ("plain_source", False, False),
])
def test_default_prediction_matches_runner(graded, case, uniform_s, graded_s):
    sim = Simulation(freq_max=10e9, domain=(.009, .008, .008), dx=.001,
                     boundary="pec", **({"dz_profile": np.full(8, .001)} if graded else {}))
    wire = "wire" in case
    sim.add_port((.003, .004, .003), "ez", impedance=50.,
                 waveform=GaussianPulse(f0=5e9), excite=not case.startswith("passive"),
                 **({"extent": .002} if wire else {}))
    if case in ("receiver", "two_lumped", "two_wire"):
        sim.add_port((.006, .004, .003), "ez", impedance=50.,
                     waveform=GaussianPulse(f0=5e9), excite=case != "receiver",
                     **({"extent": .002} if wire else {}))
    if case == "plain_source":
        sim.add_source((.006, .004, .003), "ez")
    expected = graded_s if graded else uniform_s
    assert default_run_forms_s(sim) is expected
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sim.run(n_steps=12, skip_preflight=True)
    assert (result.s_params is not None) is expected


@pytest.mark.parametrize("kind", ["waveguide", "Floquet"])
def test_concrete_zero_is_passive_but_traced_amplitude_is_a_drive(kind):
    sim = _scene(False, kind)
    attr = "_waveguide_ports" if kind == "waveguide" else "_floquet_ports"
    entries = getattr(sim, attr)
    original = entries[0]
    entries[0] = replace(original, amplitude=0.0)
    assert s_drive_inventory(sim) == [("wire", "port 0 at (0.002, 0.0015, 0.0)")]
    def traced(amplitude):
        entries[0] = replace(original, amplitude=amplitude)
        assert [kind for kind, _ in s_drive_inventory(sim)] == ["wire", kind]
        return amplitude
    jax.make_jaxpr(traced)(0.0)
    entries[0] = original


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
def test_periodic_wire_default_equals_explicit_before_step(before_step, graded):
    sim = Simulation(freq_max=10e9, domain=(.009, .008, .008), dx=.001,
                     boundary={"x": "periodic", "y": "pec", "z": "pec"},
                     **({"dz_profile": np.full(8, .001)} if graded else {}))
    sim.add_port((.003, .004, .003), "ez", extent=.002)
    errors = []
    for options in ({}, {"compute_s_params": True}):
        with pytest.raises(NotImplementedError) as caught:
            sim.run(n_steps=1, skip_preflight=True, **options)
        errors.append(caught.value)
    assert type(errors[0]) is type(errors[1])
    assert str(errors[0]) == str(errors[1])
    assert "does not honor periodic axes" in str(errors[0])


@pytest.mark.parametrize("kind", KINDS)
def test_subgrid_explicit_family_refusal_before_step(before_step, kind):
    sim = _scene(False, kind)
    sim.add_refinement((.0002, .0006), ratio=2, validation="off")
    assert default_run_forms_s(sim) is False
    with pytest.raises((ValueError, NotImplementedError)) as caught:
        sim.run(n_steps=1, compute_s_params=True, skip_preflight=True)
    assert ("add_coaxial_port()" if kind == "coaxial" else "single result schema") in str(caught.value)


def test_coaxial_entries_are_drives():
    sim = _scene(False, "coaxial")
    assert s_drive_inventory(sim) == [
        ("wire", "port 0 at (0.002, 0.0015, 0.0)"),
        ("coaxial", "port 0 at (0.002, 0.0015, 0.0)"),
    ]


def test_microstrip_only_default_run_keeps_fields_without_impedance_s():
    sim, y = _base_sim()
    _add_msl(sim, y)
    result = sim.run(n_steps=1, skip_preflight=True)
    assert result.s_params is None
    assert result.time_series.shape[0] == 1
