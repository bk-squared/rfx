"""Waveguide plane admission uses the complete declaration (#1584)."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation


MESSAGE = ("Waveguide port measurement planes exceed the physical x-domain "
           "after grid snapping; reduce ref_offset/probe_offset, "
           "flip direction, or move x_position inward")


def _repro(order):
    sim = Simulation(freq_max=10e9, domain=(.03, .02, .01), dx=None,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"},
                     cpml_layers=4)

    def structure():
        sim.add_material("block", eps_r=16)
        sim.add(Box((.012, 0, 0), (.018, .02, .01)), material="block")

    def port():
        sim.add_waveguide_port(.006, f0=8e9, freqs=jnp.array([8e9]),
                               ref_offset=3, probe_offset=18, name="p1")

    for declare in ((port, structure) if order == "port-first"
                    else (structure, port)):
        declare()
    return sim


def test_full_declaration_sets_waveguide_admission_and_realized_planes():
    # Base 0f84f9265: port-first refuses on dx=0.00149896229, source 8,
    # requested ref/probe 11/26 (no config is created). Structure-first
    # accepts on dx=0.0003: built config source/ref/probe = 24/27/42.
    # Expected indices below are measured constants, independent of admission.
    records = []
    for order in ("port-first", "structure-first"):
        sim = _repro(order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.preflight()
        grid = sim._build_grid()
        assert not sim._uses_nonuniform_mesh
        assert grid.dx == pytest.approx(.0003)
        cfg = sim._build_waveguide_port_config(
            sim._waveguide_ports[0], grid, jnp.array([8e9]), 1)
        record = (cfg.x_index, cfg.ref_x, cfg.probe_x)
        assert record == (24, 27, 42)
        records.append(record)
    assert records[0] == records[1]


def _invalid(*, graded=False, two_ports=False):
    extra = {"dz_profile": np.array([.003, .002, .002, .003])} if graded else {}
    sim = Simulation(freq_max=10e9, domain=(.03, .02, .01), dx=.002,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"},
                     cpml_layers=4, **extra)
    sim.add_waveguide_port(.024, f0=8e9, freqs=jnp.array([8e9]),
                           ref_offset=3, probe_offset=6, name="invalid")
    if two_ports:
        sim.add_waveguide_port(.006, f0=8e9, freqs=jnp.array([8e9]),
                               ref_offset=1, probe_offset=2, name="valid")
    return sim


def test_forward_s_request_admits_planes_before_routing_or_stepping(monkeypatch):
    def no_steps(*args, **kwargs):
        pytest.fail("waveguide refusal reached field stepping")

    monkeypatch.setattr(jax.lax, "scan", no_steps)
    sim = _invalid()
    with pytest.raises(ValueError) as caught:
        sim.forward(n_steps=1, port_s11_freqs=jnp.array([8e9]),
                    skip_preflight=True)
    assert str(caught.value) == MESSAGE


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("entry", ["forward", "run_distributed", "forward_distributed",
                                   "s_matrix", "preflight_sparameters"])
def test_plane_refusal_before_any_step(graded, entry, monkeypatch):
    def no_steps(*args, **kwargs):
        pytest.fail("waveguide refusal reached field stepping")

    monkeypatch.setattr(jax.lax, "scan", no_steps)
    sim = _invalid(graded=graded, two_ports=entry in ("s_matrix", "preflight_sparameters"))
    with pytest.raises(ValueError) as caught:
        if entry == "forward":
            sim.forward(n_steps=1, skip_preflight=True)
        elif entry == "run_distributed":
            sim.run(n_steps=1, devices=jax.devices(), skip_preflight=True,
                    compute_s_params=False)
        elif entry == "forward_distributed":
            sim.forward(n_steps=1, distributed=True, devices=jax.devices(),
                        skip_preflight=True)
        elif entry == "s_matrix":
            sim.compute_waveguide_s_matrix(n_steps=1, normalize=True)
        else:
            sim.preflight_sparameters(calculator="waveguide", normalize=True,
                                      include_general=False)
    assert str(caught.value) == MESSAGE


@pytest.mark.parametrize("entry", ["graded_runner", "measurement_plan"])
def test_graded_routes_without_dispatch_admit_planes(entry, monkeypatch):
    # The graded config builder does not check planes itself; the lane
    # admission (direct runner) and the measurement plan must.
    def no_steps(*args, **kwargs):
        pytest.fail("waveguide refusal reached field stepping")

    monkeypatch.setattr(jax.lax, "scan", no_steps)
    sim = _invalid(graded=True)
    with pytest.raises(ValueError) as caught:
        if entry == "graded_runner":
            from rfx.runners.nonuniform import run_nonuniform_path
            run_nonuniform_path(sim, n_steps=1, compute_s_params=False)
        else:
            from rfx.measurement.plan import measurement_plan
            measurement_plan(sim, n_steps=1)
    assert str(caught.value) == MESSAGE
