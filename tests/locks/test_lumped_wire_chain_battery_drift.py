"""Live locks on internal TEM lines continued through both CPML ends.

The previous internal open-stub curves are historical in commit 2a1053f3;
the older terminal battery remains in tests/fixtures/lumped_wire_chain_battery/fixture.json. They cannot serve
as baselines for a different circuit. Expectations now come from the exact
shunt-network formula, not newly pinned measurement records.

The one-cell lumped and four-cell wire cross-sections/separations are retained.
D3 moves the matched -20 dB floor to the load-plane Gamma_L; raw S11 is
reported because a matched shunt plus the line beyond it reflects at the port.
"""
import json

import numpy as np
import pytest

from tests import _chain_battery_drift as drift
from tests import _electrical_length as EL
from tests._interior_tem_line import chain, input_reflection

# The commit identifies the solver revision used to exercise the hand-derived
# oracle; expected values are formulas, not pinned measurements from that commit.
LOCK_PROVENANCE = {
    "fixture": "tests/_interior_tem_line.py",
    "generator": "closed-form shunt network; tests._interior_tem_line.chain",
    "commit": "4c7bbf2fb",
    "date": "2026-10-08",
    "run_id": "local CPU #1162 through-line rebuild; unresolved rows remain red",
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


def load_plane_reflection(line, freqs, s):
    """D3, closed-form inverse, independent of product extractors.

    DFT kernel exp(-j omega t); delay exp(-j beta d), beta=2pi f/c0.
    At the load: r=(ZL||Zc-Zc)/(ZL||Zc+Zc)=-Zc/(2ZL+Zc).
    At the port: q=r exp(-2j beta d), Zport=Zc(1+q)/2,
    S=(q-1)/(q+3), so q=(1+3S)/(1-S), r=q exp(2j beta d).
    Solving gives ZL=-Zc(1+r)/(2r), hence Gamma_L=(1+3r)/(1-r).
    The last form remains finite at an open circuit (r=0).

    Old floor: raw |S11| <= 0.1. New: |Gamma_L| <= 0.1 (same -20 dB).
    Derivation: matched ZL=Zc has |r|=1/3, so raw |S| ranges 1/5..1/2
    (-13.9794..-6.0206 dB); its load-plane Gamma_L is exactly zero.
    """
    q = (1 + 3 * s) / (1 - s)
    r = q * np.exp(4j * np.pi * np.asarray(freqs) * line.length / 299792458.)
    return (1 + 3 * r) / (1 - r)


def matched_findings(line, freqs, s, *, report=None):
    return drift.bound_findings("matched |Gamma_L|", freqs,
                               load_plane_reflection(line, freqs, s), report=report)


def test_matched_floor_rejects_both_resistor_mutations():
    _, line = chain("lumped", "matched")
    assert not matched_findings(line, FREQS, input_reflection(line, FREQS, line.zc))
    for ratio in (.5, 2.):
        s = input_reflection(line, FREQS, ratio * line.zc)
        np.testing.assert_allclose(load_plane_reflection(line, FREQS, s),
                                   (ratio - 1) / (ratio + 1), atol=1e-14)
        assert matched_findings(line, FREQS, s), "removed matched floor check"


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
def test_chain_matches_the_exact_shunt_network(driver, kind, dut, record_property):
    key = f"{kind}_{dut}"
    sim, line = chain(kind, dut)
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
    zload = {"short": 0., "open": np.inf, "res_half": .5 * line.zc,
             "res_double": 2 * line.zc, "matched": line.zc}[dut]
    analytic = input_reflection(line, FREQS, zload)
    record_property("freqs_hz", json.dumps(FREQS.tolist()))
    for name, values in (("live_s11", live), ("closed_form", analytic)):
        record_property(name, json.dumps([[float(v.real), float(v.imag)] for v in values]))
    assert np.isfinite(live).all()
    report = []
    if dut == "matched":
        gamma_load = load_plane_reflection(line, FREQS, live)
        record_property("gamma_load", json.dumps([[v.real, v.imag] for v in gamma_load]))
        record_property("raw_port_db", json.dumps((20 * np.log10(abs(live))).tolist()))
        print(f"{key}: raw port dB {20 * np.log10(abs(live))}")
        findings = matched_findings(line, FREQS, live, report=report)
        findings += matched_findings(line, FREQS, analytic)
    else:
        # The original 1.02 passivity bar stays on reflecting DUTs. D3 judges
        # matched ONLY via |Gamma_L|<=0.1, not raw |S|. That also implies
        # |q|<=1.1/2.9 and |S|<=(1+|q|)/(3-|q|)<=10/19, so removing the
        # redundant matched raw 1.02 guard does not weaken the verdict.
        assert max(abs(live)) <= 1.02
        findings = _reflecting_findings(driver, analytic, live, report=report)
    print(f"[battery drift] {key}: " + "; ".join(report))
    assert not findings, f"{key}: exact through-line network verdict: {findings}"



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
