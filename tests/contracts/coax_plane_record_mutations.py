"""Explicitly selected, in-process mutations of the five calculator route pins."""
import gc
import json
import sys
from dataclasses import replace

import jax
import numpy as np

from tests.contracts.test_coax_plane_record_route import (
    LENGTHS, capture_case, check_p1, check_p2, check_p3, check_p4, check_p5,
)


def _red(monkeypatch, pin, mutation, check, *, row="one", eps_scale=None, install):
    pin_name = pin
    red = False
    receipt = None

    def pin(receipt):
        nonlocal red
        try:
            check(receipt)
        except AssertionError:
            red = True

    try:
        receipt = capture_case(row, LENGTHS[row][0], eps_scale=eps_scale,
                               install=install, check=pin)
        assert red, f"{pin_name}: {mutation} did not turn the pin red"
    finally:
        print("MUTROW " + json.dumps(dict(pin=pin_name, mutation=mutation, red=red)),
              file=sys.stderr, flush=True)
        del receipt
        jax.clear_caches()
        gc.collect()


def _frequency_mutation(mp, kind):
    from rfx.measurement import calculator_plan as module
    from rfx.sparams import coax
    original = module.calculator_plan

    def changed(*args, **kw):
        plan = original(*args, **kw)
        return replace(plan, owners=tuple(replace(owner, frequencies=tuple(
            np.float64(np.nextafter(np.float32(f), np.float32(np.inf)))
            for f in owner.frequencies)) if owner.kind == kind else owner
            for owner in plan.owners))

    mp.setattr(module, "calculator_plan", changed)
    mp.setattr(coax, "calculator_plan", changed)


def _record_all(mp):
    import rfx.simulation as simulation
    original = simulation.run

    def changed(*args, **kw):
        return original(*args, **dict(kw, record_dft=True))

    mp.setattr(simulation, "run", changed)


def _voltage_mutation(mp, transform):
    from rfx.measurement import calculator_plan as module
    from rfx.sparams import coax
    original = module.coax_plane_voltages

    def changed(*args):
        return transform(original(*args))

    mp.setattr(module, "coax_plane_voltages", changed)
    mp.setattr(coax, "coax_plane_voltages", changed)


def test_p1_plane_frequencies_one_float32_ulp(monkeypatch):
    _red(monkeypatch, "P1", "plane_frequencies_one_float32_ulp", check_p1,
         install=lambda mp: _frequency_mutation(mp, "dft_plane"))


def test_p1_flux_frequencies_one_float32_ulp(monkeypatch):
    _red(monkeypatch, "P1", "flux_frequencies_one_float32_ulp", check_p1, row="two",
         install=lambda mp: _frequency_mutation(mp, "flux"))


def test_p2_one_port_records_on(monkeypatch):
    _red(monkeypatch, "P2", "one_port_records_on", check_p2, install=_record_all)


def test_p2_transition_records_all_on(monkeypatch):
    _red(monkeypatch, "P2", "transition_records_all_on", check_p2,
         row="transition", install=_record_all)


def test_p3_plane_indices_plus_one(monkeypatch):
    def install(mp):
        from rfx.measurement import ports
        from rfx.sparams import coax
        original = ports.coax_owner

        def changed(grid, **kw):
            return original(grid, **dict(kw, plane_indices=[p+1 for p in kw["plane_indices"]]))

        mp.setattr(ports, "coax_owner", changed)
        mp.setattr(coax, "coax_owner", changed)

    _red(monkeypatch, "P3", "plane_indices_plus_one", check_p3, install=install)


def test_p3_voltage_columns_reversed(monkeypatch):
    _red(monkeypatch, "P3", "voltage_columns_reversed", check_p3,
         install=lambda mp: _voltage_mutation(mp, lambda v: v[:, ::-1]))


def test_p4_one_port_eps_records_on(monkeypatch):
    _red(monkeypatch, "P4", "one_port_eps_records_on", check_p4, eps_scale=1.01,
         install=_record_all)


def test_p4_two_port_eps_records_on(monkeypatch):
    _red(monkeypatch, "P4", "two_port_eps_records_on", check_p4, row="two", eps_scale=1.01,
         install=_record_all)


def test_p5_missing_voltages(monkeypatch):
    _red(monkeypatch, "P5", "missing_voltages", check_p5,
         install=lambda mp: _voltage_mutation(mp, lambda _: None))


def test_p5_last_coax_plane_dropped(monkeypatch):
    def install(mp):
        from rfx.sources import waveguide_port
        original = waveguide_port.settling_db_from_named_records

        def changed(channels, **kw):
            channels = list(channels)
            last = next(name for name, _ in reversed(channels) if name.startswith("coax/"))
            return original([(name, value) for name, value in channels if name != last], **kw)

        mp.setattr(waveguide_port, "settling_db_from_named_records", changed)

    _red(monkeypatch, "P5", "last_coax_plane_dropped", check_p5, install=install)
