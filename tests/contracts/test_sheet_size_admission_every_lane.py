"""skip_preflight drops advisories only: a sheet-size refusal stays on every lane."""
import warnings

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from tests._coax_msl_instrument_fixture import DX, build_instrument_junction, instrument_kwargs

D = 1e-3
EDGE = .35  # solved PEC sheet edge beyond its last node, in cells (measured rule)


def solved(nodes, lo, hi, free_ends):
    """Node span the drawn interval covers, plus EDGE cell per free end."""
    nodes = np.asarray(nodes)
    inside = nodes[(nodes >= lo - 1e-12) & (nodes <= hi + 1e-12)]
    covered = inside[-1] - inside[0]
    cell = nodes[1] - nodes[0]
    return covered, covered + EDGE*cell*free_ends, hi - lo


def scan_scene(snap):
    sim = Simulation(freq_max=1e9, domain=(10*D,)*3, dx=D, boundary="pec", snap=snap)
    half = 4.7*D/1.05/2
    sim.add(Box((5*D-half, 5*D-half, 5*D), (5*D+half, 5*D+half, 5*D)), material="pec")
    sim.add_port((2*D, 2*D, 2*D), "ez", impedance=50.,
                 waveform=GaussianPulse(f0=5e8, bandwidth=.9))
    return sim, half


def scan_expected(half):
    covered, eff, drawn = solved(np.arange(11)*D, 5*D-half, 5*D+half, 2)
    assert covered == pytest.approx(4*D) and eff == pytest.approx(4.7*D)
    rel = (eff - drawn)/drawn
    line = (f"drawn {drawn/D:.4g}mm, nodes cover {covered/D:.4g}mm, "
            f"solved as {eff/D:.4g}mm ({rel:+.2%})")
    assert line == "drawn 4.476mm, nodes cover 4mm, solved as 4.7mm (+5.00%)"
    return [f"geometry[0] 'pec' {axis}: {line}" for axis in "xy"]


def coax_expected():
    nodes = np.arange(126)*DX
    clear, yc, trace = 4*DX, 17*DX, 10*DX
    # (label, axis, lo, hi, free ends): walls and abutting sheets are not free.
    dims = [("geometry[1]", "x", 0., trace-clear, 1),   # wall | clearance hole
            ("geometry[2]", "x", trace+clear, 125*DX, 1),  # under 1 %, not listed
            ("geometry[3]", "y", 0., yc-clear, 1),
            ("geometry[4]", "y", yc+clear, 34*DX, 1),
            ("geometry[5]", "y", yc-3*DX, yc+3*DX, 2)]
    listed = []
    for label, axis, lo, hi, free in dims:
        _, eff, drawn = solved(nodes, lo, hi, free)
        rel = (eff - drawn)/drawn
        if abs(rel) > .01:
            listed.append(f"{label} 'pec' {axis}:")
            listed.append(f"({rel:+.2%})")
    assert listed[-1] == "(+11.67%)" and len(listed) == 8
    return listed


def run_scan(sim):
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    return compute_lumped_wire_s_matrix_via_scan(sim, [5e8], n_steps=2, skip_preflight=True)


def run_coax(sim, monkeypatch, snap):
    class ReachedRunner(Exception):
        pass

    def runner(*args, **kwargs):
        if snap == "strict":
            pytest.fail("a strict sheet-size refusal must precede stepping")
        raise ReachedRunner

    monkeypatch.setattr("rfx.simulation.run", runner)
    sim._snap = snap
    try:
        sim.compute_coax_msl_transition(**instrument_kwargs(n_steps=1))
    except ReachedRunner:
        pass


@pytest.mark.parametrize("snap", ("strict", "declared"))
@pytest.mark.parametrize("lane", ("lumped_scan", "coax_msl_transition"))
def test_skip_preflight_keeps_sheet_size_admission(lane, snap, monkeypatch):
    if lane == "lumped_scan":
        sim, half = scan_scene(snap)
        expected = scan_expected(half)
        count = 2
        call = lambda: run_scan(sim)
    else:
        sim = build_instrument_junction()
        expected = coax_expected()
        count = 4
        call = lambda: run_coax(sim, monkeypatch, snap)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if snap == "strict":
            with pytest.raises(ValueError) as refused:
                call()
            message = str(refused.value)
        else:
            call()
    notes = [str(w.message) for w in caught if "conductor sheet dimension" in str(w.message)]
    if snap == "strict":
        assert notes == []
    else:
        assert len(notes) == 1
        message = notes[0]
    assert f"{count} conductor sheet dimension(s) are solved more than 1% off" in message
    for text in expected:
        assert text in message
