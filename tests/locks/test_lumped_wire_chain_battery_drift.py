"""Live drift locks on ten internally conducted, asymmetric TEM coax lines.

Re-pin history, 2026-10-05 (#1162 after #1221 B3b): the old 1 mm rung
used PEC domain plates and one-cell PMC side walls; B3b moved its magnetic
image to the E nodes and changed the realized line. The full OLD complex
curves remain, unchanged, in tests/fixtures/lumped_wire_chain_battery/fixture.json,
solves[{lumped,wire}_{short,open,res_half,res_double,matched}_1000um],
producer c3936b3b, 2026-09-23, VESSL 369367263710/369367263711.
OLD endpoint values (1 GHz -> 10 GHz) are recorded below, not overwritten.

The replacement uses internal PEC sheets y=2/5, z=2/(3+gap), a filament
at (3,3), and unequal exterior margins 2/3 and 2/4 cells. The radial feed
is off the cross-section symmetry planes. Only longitudinal PMC ends
remain: their 1-cell/3-cell open stubs enter the TEM closed form.
C' follows the realized transverse Dirichlet cell problem; Zc=eta0/(C'/eps0).
Lumped: gap=1, dx=100 um, shape=(306,9,9), port/load x nodes=1/302,
Zc=eta0/3.75, realized length=30.1 mm (declared 30.108 mm).
Wire: gap=4, dx=25 um, shape=(1206,9,12), x nodes=1/1202,
Zc=111.23520365290427 ohm from the 7-node Dirichlet solve,
realized length=30.025 mm (declared 30.027 mm). The wire still has FOUR
live cells; the load is a series chain of four R/4 resistors.

Fresh CPU float32 records use 20 periods and 91 bins at 1..10 GHz.
The original bars remain: 2 dB magnitude, -20 dB matched floor, 1% on
phase crossings and electrical-length slope, passivity <=1.02, and the
phase direction. Each reflecting solve is also judged directly against
the TEM line plus both open stubs with those same bars. No fitted length
or impedance enters that oracle. The old battery replay records are
historical and remain separate from these new live locks.
"""
from dataclasses import asdict
import json

import numpy as np
import pytest

from tests import _chain_battery_drift as drift
from tests import _electrical_length as EL
from tests._interior_tem_line import chain, input_reflection
from tests._interior_tem_pins import PINS

LOCK_PROVENANCE = {
    "fixture": "tests/_interior_tem_pins.py",
    "generator": "tests._interior_tem_line.chain + Simulation.forward",
    # Unchanged product-code revision; the new fixture is committed with these pins.
    "commit": "0993dc71963d9b274db387f95f9b8962874029d4",
    "date": "2026-10-05",
    "run_id": "local CPU #1162 rebuild",
    "host": "JAX CPU float32",
    "pinned_until": "2027-03-23",
}
FAMILY = "lumped / wire internal TEM coax"
DRIVER = "scripts/diagnostics/lumped_wire_chain_battery_measure.py"
KINDS = ("lumped", "wire")
DUTS = ("short", "open", "res_half", "res_double", "matched")
FREQS = np.linspace(1e9, 10e9, 91)
NUM_PERIODS = 20.
REMEASURE = "re-measure chain(kind, dut).forward(port_s11_freqs=FREQS, num_periods=20) on CPU"


@pytest.fixture(scope="module")
def driver():
    # Only the crossing arithmetic is reused; the old geometry and its
    # assembler's half-cell assumptions do not construct these new records.
    return drift.load_driver(DRIVER)


def _reflecting_findings(driver, reference, live, *, report):
    findings = drift.magnitude_findings("|S11|", FREQS, reference, live, report=report)
    findings += drift.crossing_findings(
        "S11", driver.phase_crossings(FREQS, reference)["crossings"],
        driver.phase_crossings(FREQS, live)["crossings"], FREQS[0], FREQS[-1], report=report)
    findings += drift.electrical_length_findings("S11", FREQS, reference, live, report=report)
    findings += drift.phase_direction_findings("S11", FREQS, reference, live)
    return findings


@pytest.mark.parametrize("dut", DUTS)
@pytest.mark.parametrize("kind", KINDS)
def test_the_coarsest_mesh_still_solves_to_its_stored_s11(driver, kind, dut, record_property):
    key = f"{kind}_{dut}"
    entry = PINS[key]
    sim, line = chain(kind, dut)
    moved = drift.realized_differences(entry["line"], asdict(line))
    assert not moved, drift.stale_record(FAMILY, key, moved, REMEASURE)
    result = sim.forward(port_s11_freqs=FREQS, num_periods=NUM_PERIODS, skip_preflight=True)
    specs = result.lumped_port_sparams if kind == "lumped" else result.wire_port_sparams
    assert len(specs) == 1
    spec = specs[0][0]
    assert spec.component == "ez" and spec.impedance == pytest.approx(line.zc, rel=1e-9)
    if kind == "lumped":
        assert (spec.i, spec.j, spec.k) == line.port
    else:
        assert tuple(spec.live_cells) == tuple((line.port[0], 3, 3 + k) for k in range(4))
        assert spec.excite
    live = np.asarray(result.s_params).reshape(-1)
    stored = drift.complex_array(entry["s11"])
    zload = {"short": 0., "open": np.inf, "res_half": .5 * line.zc,
             "res_double": 2 * line.zc, "matched": line.zc}[dut]
    analytic = input_reflection(line, FREQS, zload)
    record_property("freqs_hz", json.dumps(FREQS.tolist()))
    for name, values in (("stored_s11", stored), ("live_s11", live), ("closed_form", analytic)):
        record_property(name, json.dumps([[float(v.real), float(v.imag)] for v in values]))
    assert np.isfinite(live).all()
    assert max(abs(live)) <= 1.02
    report = []
    if dut == "matched":
        findings = drift.bound_findings("matched |S11|", FREQS, live, report=report)
        findings += drift.bound_findings("matched TEM |S11|", FREQS, analytic)
    else:
        findings = _reflecting_findings(driver, stored, live, report=report)
        findings += _reflecting_findings(driver, analytic, live, report=report)
    print(f"[battery drift] {key}: " + "; ".join(report))
    assert not findings, drift.stale_record(FAMILY, key, findings, REMEASURE)


# OLD lumped_short: (-0.31498000025749207+0.9490987062454224j) -> (-0.9997029900550842+0.024372028186917305j)
# OLD lumped_open: (0.3412729799747467-0.9399644136428833j) -> (0.9250831604003906+0.3797631561756134j)
# OLD lumped_res_half: (-0.1078980565071106+0.32007065415382385j) -> (-0.3335723876953125+0.008125212043523788j)
# OLD lumped_res_double: (0.09750854223966599-0.31413817405700684j) -> (0.33289608359336853-0.008143632672727108j)
# OLD lumped_matched: (-0.004983312916010618+0.0036159285809844732j) -> (-0.0003804727748502046-5.96065319768968e-06j)
# OLD wire_short: (-0.31498000025749207+0.9490987062454224j) -> (-0.9997029900550842+0.024372028186917305j)
# OLD wire_open: (0.3412729799747467-0.9399644136428833j) -> (0.9250831604003906+0.3797631561756134j)
# OLD wire_res_half: (-0.1078980565071106+0.32007065415382385j) -> (-0.3335723876953125+0.008125212043523788j)
# OLD wire_res_double: (0.09750854223966599-0.31413817405700684j) -> (0.33289608359336853-0.008143632672727108j)
# OLD wire_matched: (-0.004983312916010618+0.0036159285809844732j) -> (-0.0003804727748502046-5.96065319768968e-06j)


def test_the_crossing_rule_excuses_only_a_crossing_that_can_have_left_the_sweep():
    """The line's last crossing sits 0.2 % below the top of the 1-10 GHz sweep
    at this mesh (9.98 GHz on the short), so a drift far inside the bar can
    carry it out of the band. That absence is excused; a crossing that moves by
    more than the bar is caught wherever it lands, and so is one the record
    does not have. Arithmetic on crossing lists, no solve."""
    band = (1e9, 10e9)
    stored = [{"hz": 2.5e9, "multiple_of_pi": 0}, {"hz": 5.0e9, "multiple_of_pi": -1},
              {"hz": 7.5e9, "multiple_of_pi": -2}, {"hz": 9.95e9, "multiple_of_pi": -3}]

    def moved(factor):
        return [{**c, "hz": c["hz"] * factor} for c in stored if c["hz"] * factor <= band[1]]

    assert not drift.crossing_findings("S11", stored, moved(1.006), *band)
    down = drift.crossing_findings("S11", stored, moved(0.985), *band)
    assert any("live crossing at 9.80075 GHz" in f for f in down), down
    extra = drift.crossing_findings(
        "S11", stored, stored + [{"hz": 6.2e9, "multiple_of_pi": -2}], *band)
    assert any("live crossing at 6.20000 GHz" in f for f in extra), extra


def test_each_check_fires_just_outside_its_bar_and_not_just_inside():
    """The comparisons all three guards share, on made-up curves straddling
    each bar: 2 dB in magnitude, -20 dB for a deep null, 1 % in frequency, the
    phase turning the other way, 1 % in electrical length, a flipped verdict, a
    moved cell. A check that stops reporting, or reports inside its bar, reds
    here on every pull request, not only in the weekly lane that solves.
    Arithmetic, no solve."""
    freqs = np.linspace(1e9, 10e9, 91)
    line = np.full(freqs.size, 1.0 / 3.0 + 0j)
    assert drift.magnitude_findings("S11", freqs, line, line * 10 ** (2.05 / 20))
    assert not drift.magnitude_findings("S11", freqs, line, line * 10 ** (1.95 / 20))
    null = line.copy()
    null[40:43] = 1e-3                      # -60 dB: a core, compared with nothing
    assert not drift.magnitude_findings("S21", freqs, null, null * [
        10.0 if 40 <= k < 43 else 1.0 for k in range(freqs.size)])
    assert drift.bound_findings("S11", freqs, np.full(freqs.size, 10 ** (-19.9 / 20)))
    assert not drift.bound_findings("S11", freqs, np.full(freqs.size, 10 ** (-20.1 / 20)))
    assert drift.frequency_findings("notch", 3.7e9, 3.7e9 * 1.0105)
    assert not drift.frequency_findings("notch", 3.7e9, 3.7e9 * 0.9905)
    delay = np.exp(-2j * np.pi * freqs * 0.2e-9)     # a 0.2 ns line: -11 rad over the band
    assert drift.phase_direction_findings("S21", freqs, delay, np.conj(delay))
    assert not drift.phase_direction_findings("S21", freqs, delay, 0.5 * delay)
    assert drift.electrical_length_findings(
        "S21", freqs, delay, np.exp(-2j * np.pi * freqs * 0.2e-9 * 1.0105))
    assert drift.electrical_length_findings(
        "S21", freqs, delay, np.exp(-2j * np.pi * freqs * 0.2e-9 * 0.9895))
    assert not drift.electrical_length_findings(
        "S21", freqs, delay, 0.5 * np.exp(-2j * np.pi * freqs * 0.2e-9 * 1.0095 + 0.3j))
    assert not drift.electrical_length_findings(
        "S21", freqs, delay, np.exp(-2j * np.pi * freqs * 0.2e-9 * 0.9905))
    assert drift.verdict_findings({"v": {"ok": True}}, {"v": {"ok": False}}, [("v", "ok")])
    assert drift.verdict_findings({"v": {"ok": False}}, {"v": {"ok": True}}, [("v", "ok")])
    assert not drift.verdict_findings({"v": None}, {"v": {"ok": False}}, [("v", "ok")])
    assert drift.realized_differences({"nodes": [32, 2, 5]}, {"nodes": [33, 2, 5]})
    assert not drift.realized_differences({"dt": 1.9065748695310057e-12},
                                          {"dt": 1.9065748695310057e-12 * (1 + 1e-12)})


def test_the_electrical_length_is_the_ratio_of_two_delays():
    """The measure the batteries and the locks share
    (``tests/_electrical_length.py``), on lines whose answer is known: the
    slope of the unwrapped phase of ``exp(-j 2 pi f tau)`` is ``-2 pi tau``, so
    two such lines compare as ``tau1 / tau2 - 1`` whatever their magnitudes and
    whatever constant phase they carry. Bins at or below -20 dB carry no
    reading, and a gap in the bins is refused, because the phase unwrapped
    across a transmission zero can take its half-turn either way. Arithmetic,
    no solve."""
    freqs = np.linspace(1e9, 7e9, 121)
    tau = 0.16e-9
    line = np.exp(-2j * np.pi * freqs * tau)
    longer = 0.3 * np.exp(-2j * np.pi * freqs * tau * 1.02 + 1.1j)
    all_bins = np.ones(freqs.size, dtype=bool)
    assert EL.electrical_length_ratio(freqs, longer, line, all_bins) == pytest.approx(
        0.02, abs=1e-12)
    assert EL.electrical_length_ratio(freqs, line, longer, all_bins) == pytest.approx(
        1.0 / 1.02 - 1.0, abs=1e-12)
    assert EL.electrical_length_ratio(freqs, line, line, all_bins) == 0.0
    notch = line.copy()
    notch[60:64] *= 1e-3                                   # -60 dB: no phase to read
    kept = EL.transmitting_bins(notch)
    assert not kept[60:64].any() and kept[:60].all() and kept[64:].all()
    with pytest.raises(ValueError, match="contiguous"):
        EL.electrical_length_ratio(freqs, notch, line, kept)
    assert EL.electrical_length_ratio(freqs, longer, line, kept[:60].tolist()
                                      + [False] * 61) == pytest.approx(0.02, abs=1e-12)
    assert EL.ELECTRICAL_LENGTH_FRAC == 0.01
