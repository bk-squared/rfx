"""Calculator plane-record routing against independent legacy runner observations."""
from __future__ import annotations

import gc
import json
import sys
from dataclasses import dataclass, field

import jax
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.measurement import calculator_plan as plan_module
from tests._coax_msl_instrument_fixture import build_instrument_junction, instrument_kwargs


LENGTHS = {"one": (1000, 2000, 4000), "two": (600, 1200, 2400),
           "transition": (200, 400, 800)}
COUNTS = {"one": (12, 1, 0), "two": (6, 2, 0), "transition": (6, 2, 9)}


@dataclass
class Receipt:
    row: str
    n: int
    calls: list = field(default_factory=list)
    witnesses: list = field(default_factory=list)
    voltage_kwargs: dict = field(default_factory=dict)


def _case(row, n, eps_scale=None):
    if row == "one":
        sim = Simulation(freq_max=10e9, domain=(.008, .008, .030),
                         dx=.0004, boundary="cpml")
        sim.add_coaxial_port((.004, .004, .015), pin_radius=.000635,
                            outer_radius=.002055)
        return sim.compute_coaxial_line_reflection(
            termination="matched", n_steps=n, freqs=np.linspace(4e9, 6e9, 11),
            eps_scale=eps_scale)
    if row == "two":
        from tests.unit.autodiff.test_coax_two_port_ad import (
            _build_small_two_port_sim, FREQ, PROBE_COUNT, PROBE_START_CELLS,
            PROBE_SPACING_CELLS,
        )
        sim = _build_small_two_port_sim()
        scratch = Simulation(domain=(.008, .008, .012), freq_max=40e9, boundary="cpml")
        scratch.add_flux_monitor(axis="z", coordinate=.006, freqs=np.asarray(FREQ), name="middle")
        return sim.compute_coaxial_two_port(
            n_steps=n, freqs=FREQ, probe_count=PROBE_COUNT,
            probe_start_cells=PROBE_START_CELLS, probe_spacing_cells=PROBE_SPACING_CELLS,
            eps_scale=eps_scale, extra_flux_monitors=scratch._flux_monitors)
    sim = build_instrument_junction()
    scratch = Simulation(freq_max=16e9, domain=(.0125, .0034, .0039), dx=.0001,
                         cpml_layers=8, boundary="cpml")
    scratch.add_flux_monitor(axis="z", coordinate=.003, freqs=np.array([6e9, 8e9, 10e9]),
                             size=(.0013, .0012), center=(.00215, .0017), name="top")
    return sim.compute_coax_msl_transition(
        **instrument_kwargs(n), extra_flux_monitors=scratch._flux_monitors)


def _bits(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, label
    assert actual.tobytes() == expected.tobytes(), (
        label, float(np.max(np.abs(actual - expected), initial=0)))


def _spectra(result):
    from rfx.probes.probes import flux_spectrum
    return [np.asarray(flux_spectrum(m, exact_f64=True)) for m in result.flux_monitors or ()]


def capture_case(row, n, *, eps_scale=None, install=None, check=None):
    import rfx.simulation as simulation
    from rfx.sparams import _coax_oneport_record
    from rfx.sources import coaxial_port, waveguide_port

    receipt = Receipt(row, n)
    original_run = simulation.run
    original_record = _coax_oneport_record.run_reflection_record
    original_voltage = coaxial_port.coaxial_line_plane_voltage
    original_score = waveguide_port.settling_db_from_named_records

    def reflection(grid, materials, n_steps, **kw):
        receipt.voltage_kwargs = dict(center_xy=kw["center_xy"], pin_radius=kw["a"],
            outer_radius=kw["b"], pec_cell_mask=kw["conductors"].pec_cells)
        return original_record(grid, materials, n_steps, **kw)

    def runner(grid, materials, n_steps, **kw):
        call = dict(grid=grid, kwargs=kw, post={}, voltage_kwargs={})
        actual = original_run(grid, materials, n_steps, **kw)
        legacy_kw = {k: v for k, v in kw.items() if k != "measurement"}
        legacy = original_run(grid, materials, n_steps, **dict(legacy_kw, record_dft=True))
        call["legacy_planes"] = [np.asarray(p.accumulator) for p in legacy.dft_planes]
        call["legacy_probes"] = np.asarray(legacy.time_series)
        call["legacy_flux"] = _spectra(legacy)
        del legacy
        if eps_scale is None:
            recorded = original_run(grid, materials, n_steps, **dict(kw, record_dft=True))
            call["recorded"] = recorded
            if row == "one":
                for pi in range(COUNTS[row][0]):
                    kwargs = dict(receipt.voltage_kwargs, plane_index=kw["dft_planes"][2*pi].index)
                    call["post"][pi] = np.asarray(original_voltage(
                        grid, recorded.dft_time_records[2*pi], recorded.dft_time_records[2*pi+1], **kwargs))
        call["actual"] = actual
        receipt.calls.append(call)
        return actual

    def voltage(grid, ex, ey, **kw):
        # Read the exact, unchanged Gamma-path keywords rather than the new owner.
        for call in receipt.calls:
            for pi, plane in enumerate(call["actual"].dft_planes[::2]):
                if plane.accumulator is ex and pi < COUNTS[row][0]:
                    call["voltage_kwargs"][pi] = kw
                    recorded = call.get("recorded")
                    if recorded is not None:
                        call["post"][pi] = np.asarray(original_voltage(
                            grid, recorded.dft_time_records[2*pi], recorded.dft_time_records[2*pi+1], **kw))
        return original_voltage(grid, ex, ey, **kw)

    def score(channels, **kw):
        channels = list(channels)
        result = original_score(channels, **kw)
        receipt.witnesses.append((channels, kw, result))
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simulation, "run", runner)
        mp.setattr(_coax_oneport_record, "run_reflection_record", reflection)
        mp.setattr(coaxial_port, "coaxial_line_plane_voltage", voltage)
        mp.setattr(waveguide_port, "settling_db_from_named_records", score)
        if install is not None:
            install(mp)
        receipt.output = _case(row, n, eps_scale)
        jax.block_until_ready(receipt.calls[-1]["actual"].time_series)
        if check is not None:
            check(receipt)
    return receipt


def check_p1(receipt):
    assert len(receipt.calls) == (1 if receipt.row == "one" else 2)
    for drive, call in enumerate(receipt.calls):
        result = call["actual"]
        assert len(result.dft_planes) == len(call["legacy_planes"])
        for i, (p, expected) in enumerate(zip(result.dft_planes, call["legacy_planes"])):
            _bits(p.accumulator, expected, f"P1 {receipt.row} drive {drive} plane {i}")
        _bits(result.time_series, call["legacy_probes"], f"P1 {receipt.row} probes")
        spectra = _spectra(result)
        assert len(spectra) == len(call["legacy_flux"])
        for i, (actual, expected) in enumerate(zip(spectra, call["legacy_flux"])):
            _bits(actual, expected, f"P1 {receipt.row} flux {i}")


def check_p2(receipt):
    for call in receipt.calls:
        kw, result = call["kwargs"], call["actual"]
        if receipt.row != "transition":
            assert kw["record_dft"] is False
            assert result.dft_time_records == ()
        else:
            n_coax, _, n_msl = COUNTS[receipt.row]
            selection = (False,) * (2*n_coax) + (True,) * n_msl
            assert kw["record_dft"] == selection
            records = result.dft_time_records
            assert len(records) == n_msl
            for record, plane in zip(records, kw["dft_planes"][2*n_coax:]):
                assert record.shape == (receipt.n,) + plane.accumulator.shape[1:]
        records = result.sparam_time_records
        assert records
        assert all(r.samples.shape[0] == receipt.n for r in records)
        assert sum(r.samples.shape[1] for r in records) == COUNTS[receipt.row][0]


def check_p3(receipt):
    for drive, call in enumerate(receipt.calls):
        owner, = (o for o in call["kwargs"]["measurement"].owners if o.kind == "coax")
        vin = plan_module.coax_plane_voltages(call["recorded"], owner)
        assert vin is not None and vin.dtype == np.complex128
        assert len(call["post"]) == COUNTS[receipt.row][0]
        post = np.stack([call["post"][i] for i in range(COUNTS[receipt.row][0])], axis=1)
        assert vin.shape == post.shape == (receipt.n, COUNTS[receipt.row][0])
        diff = np.abs(vin-post)
        step, plane = np.unravel_index(np.argmax(diff), diff.shape)
        d, peak = float(diff[step, plane]), float(np.max(np.abs(post)))
        ulp = float(np.spacing(np.float32(peak)))
        r = d/ulp
        print("P3ROW " + json.dumps(dict(row=receipt.row, n=receipt.n, drive=drive,
            d=d, peak=peak, ulp=ulp, r=r, step=int(step),
            plane=int(call["kwargs"]["dft_planes"][2*plane].index), column=int(plane))), file=sys.stderr, flush=True)
        assert peak > 0 and r <= 9, (receipt.row, receipt.n, drive, d, peak, ulp, r)
    if receipt.row == "one":
        from rfx.sources.waveguide_port import settling_db_from_named_records
        channels, kw, actual = receipt.witnesses[0]
        call = receipt.calls[0]
        post_channels = [(name, call["post"][int(name.split("/")[1][5:])])
                         if name.startswith("coax/") else (name, values) for name, values in channels]
        post_result = settling_db_from_named_records(post_channels, **kw)
        print("SETTLINGROW " + json.dumps(dict(row="one", n=receipt.n,
            in_step=actual, post=post_result), default=_json_value), file=sys.stderr, flush=True)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def check_p4(receipt):
    for call in receipt.calls:
        assert call["kwargs"]["record_dft"] is False
        assert "measurement" not in call["kwargs"]
    check_p1(receipt)


def check_p5(receipt):
    n_coax, n_probes, n_msl = COUNTS[receipt.row]
    expected = ({f"coax/plane{i}/V" for i in range(n_coax)} |
                {f"probe{i}" for i in range(n_probes)} |
                {f"msl/plane{i}/V" for i in range(n_msl)})
    assert len(receipt.witnesses) == len(receipt.calls)
    for channels, _, detail in receipt.witnesses:
        assert {name for name, _ in channels} == expected


@pytest.mark.parametrize("row,n", [(row, n) for row, ns in LENGTHS.items() for n in ns])
def test_pins(row, n):
    receipt = capture_case(row, n)
    try:
        check_p1(receipt)
        check_p2(receipt)
        check_p3(receipt)
        check_p5(receipt)
    finally:
        del receipt
        jax.clear_caches()
        gc.collect()


@pytest.mark.parametrize("row", ["one", "two"])
def test_eps_scale_p4(row):
    receipt = capture_case(row, LENGTHS[row][0], eps_scale=1.01)
    try:
        check_p4(receipt)
    finally:
        del receipt
        jax.clear_caches()
        gc.collect()
