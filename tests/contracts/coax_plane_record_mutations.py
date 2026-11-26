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
from tests.contracts.test_coax_runner_memory_slope import K_BOUND, measure_row


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
    from rfx.sparams import _coax_plane_measurement as measurement
    assert module.calculator_plan is measurement.calculator_plan
    original = module.calculator_plan

    def changed(*args, **kw):
        plan = original(*args, **kw)
        return replace(plan, owners=tuple(replace(owner, frequencies=tuple(
            np.float64(np.nextafter(np.float32(f), np.float32(np.inf)))
            for f in owner.frequencies)) if owner.kind == kind else owner
            for owner in plan.owners))

    mp.setattr(module, "calculator_plan", changed)
    mp.setattr(measurement, "calculator_plan", changed)


def _record_all(mp):
    import rfx.simulation as simulation
    original = simulation.run

    def changed(*args, **kw):
        return original(*args, **dict(kw, record_dft=True))

    mp.setattr(simulation, "run", changed)


def _voltage_mutation(mp, transform):
    from rfx.measurement import calculator_plan as module
    from rfx.sparams import coax
    assert module.coax_plane_voltages is coax.coax_plane_voltages
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
        from rfx.sparams import _coax_plane_measurement as measurement
        assert ports.coax_owner is measurement.coax_owner
        original = ports.coax_owner

        def changed(grid, **kw):
            return original(grid, **dict(kw, plane_indices=[p+1 for p in kw["plane_indices"]]))

        mp.setattr(ports, "coax_owner", changed)
        mp.setattr(measurement, "coax_owner", changed)

    _red(monkeypatch, "P3", "plane_indices_plus_one", check_p3, install=install)


def test_p3_voltage_columns_reversed(monkeypatch):
    _red(monkeypatch, "P3", "voltage_columns_reversed", check_p3,
         install=lambda mp: _voltage_mutation(mp, lambda v: v[:, ::-1]))


def test_p3_voltage_column_zero_repeated(monkeypatch):
    _red(monkeypatch, "P3", "voltage_column_zero_repeated", check_p3,
         install=lambda mp: _voltage_mutation(
             mp, lambda v: np.repeat(v[:, :1], v.shape[1], axis=1)))


def test_p4_one_port_eps_records_on(monkeypatch):
    _red(monkeypatch, "P4", "one_port_eps_records_on", check_p4, eps_scale=1.01,
         install=_record_all)


def test_p4_two_port_eps_records_on(monkeypatch):
    _red(monkeypatch, "P4", "two_port_eps_records_on", check_p4, row="two", eps_scale=1.01,
         install=_record_all)


def test_p4_one_port_eps_measurement_on(monkeypatch):
    def install(mp):
        import rfx.simulation as simulation
        from rfx.measurement.calculator_plan import calculator_plan
        original = simulation.run

        def changed(grid, materials, n_steps, **kw):
            measurement = calculator_plan(
                grid, n_steps, probes=kw["probes"], planes=kw["dft_planes"])
            return original(grid, materials, n_steps, **dict(kw, measurement=measurement))

        mp.setattr(simulation, "run", changed)

    _red(monkeypatch, "P4", "one_port_eps_measurement_on", check_p4, eps_scale=1.01,
         install=install)


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


def test_k_one_port_records_on():
    row = measure_row("one", install=_record_all)
    red = row["k"] > K_BOUND
    print("MUTROW " + json.dumps(dict(pin="k", mutation="k_one_port_records_on",
                                    red=red, k=row["k"])), file=sys.stderr, flush=True)
    assert red, "k: one_port_records_on did not turn the pin red"


def test_k_two_port_records_on():
    row = measure_row("two", install=_record_all)
    red = row["k"] > K_BOUND
    print("MUTROW " + json.dumps(dict(pin="k", mutation="k_two_port_records_on",
                                    red=red, k=row["k"])), file=sys.stderr, flush=True)
    assert red, "k: two_port_records_on did not turn the pin red"


def test_k_one_port_records_moved():
    def install(mp):
        import rfx.simulation as simulation
        _record_all(mp)
        original = simulation.run

        def changed(*args, **kw):
            result = original(*args, **kw)
            return result._replace(snapshots={"moved": result.dft_time_records},
                                   dft_time_records=())

        mp.setattr(simulation, "run", changed)

    row = measure_row("one", install=install)
    red = row["k"] > K_BOUND
    print("MUTROW " + json.dumps(dict(pin="k", mutation="k_one_port_records_moved",
                                    red=red, k=row["k"])), file=sys.stderr, flush=True)
    assert red, "k: one_port_records_moved did not turn the pin red"
