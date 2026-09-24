"""The lumped / wire chain battery's coarsest mesh, solved again and held to its
stored S11.

The battery (``tests/fixtures/lumped_wire_chain_battery/fixture.json``) is a
30 mm parallel-plate line one cell wide between PEC plates and magnetic side
walls, fed at one end by a one-cell lumped port or a four-cell wire port and
terminated in a short, an open, R = Zc/2, R = 2 Zc and R = Zc. With the port
referenced to the line's own Zc the closed form is S11 = Gamma_L exp(-2j beta L):
|S11| is |Gamma_L| at every bin and the line length lives in the phase, which
crosses the real axis every c / 4L = 2.5 GHz.

Its replay test (``tests/oracle/test_lumped_wire_chain_battery.py``) reads the
stored S11 only. This file builds the 1.0 mm mesh of the same ten lines with
the battery's own driver, refuses to solve unless the grid, the port and the
load land where the record says they did, solves them, and holds the live S11
to the stored one with the bar the battery is judged by:

* |S11| within 2 dB of the stored curve at every bin. The matched line is a
  deep null (the PI's 2026-09-21 ruling): it is held to -20 dB and compared in
  dB with nothing.
* every phase crossing — a frequency where S11 is real — within 1 % of its
  partner of the same sign on the other curve.
* the record's own verdicts at this mesh — passivity at 1.02, |S11| within
  2 dB of the closed form, the crossings against the closed form, the phase
  turning the same way with frequency as the closed form's, the matched floor —
  come out the same when the battery's assembler computes them from the live
  S11. The phase direction is the one a conjugated S11 (the other time
  convention) changes: its magnitude and its real-axis crossings are the same.

A red here means the stored battery describes a different solver from the one
under test. The remedy is to measure the battery again, not to move anything in
this file.

Lane: the default fast suite (no marker), so every pull request that touches
code, every push to main and the weekly CPU lane run it. Wall time: 3.7 s for
all ten lines on four VESSL CPU cores (run 369367264067), 0.2-0.8 s a line.
"""
LOCK_PROVENANCE = {
    "fixture": "tests/fixtures/lumped_wire_chain_battery/fixture.json",
    "generator": "scripts/diagnostics/lumped_wire_chain_battery_measure.py",
    "commit": "c3936b3b",
    "date": "2026-09-23",
    "run_id": "369367263710 (wire), 369367263711 (lumped)",
    "host": "VESSL cpu-32-mem-64, jax 0.6.2 cpu, float32",
    "pinned_until": "2027-03-23",
}

import json

import numpy as np
import pytest

from tests import _chain_battery_drift as drift

FAMILY = "lumped / wire"
DRIVER = LOCK_PROVENANCE["generator"]
FIXTURE = drift.REPO / LOCK_PROVENANCE["fixture"]
REMEASURE = (f"`PYTHONPATH=. python {DRIVER} --stage solve --kind <lumped|wire> "
             "--all-duts --all-rungs --out <dir> --run-id <id>` for both port kinds, "
             "then `--assemble`")

RUNG_UM = 1000
KINDS = ("lumped", "wire")
DUTS = ("short", "open", "res_half", "res_double", "matched")

# The verdicts the record carries at this mesh, by DUT. The replay test gates
# passivity and the closed-form magnitude at every mesh; the crossings against
# the closed form are recorded at this one (1.7-2.8 % out, which is why the
# battery recommends 0.5 mm) and must not flip either. `same_sign` says the
# unwrapped angle falls with frequency as the closed form's does; a conjugated
# S11 keeps every other number here and flips that one.
VERDICTS = {
    "reflecting": (("passivity", "within_bar"),
                   ("magnitude_vs_analytic", "within_bar"),
                   ("phase", "within_bar"),
                   ("phase", "angle_slope_rad_per_hz", "same_sign")),
    "matched": (("passivity", "within_bar"),
                ("matched_floor", "within_bar")),
}


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def driver():
    return drift.load_driver(DRIVER)


def _live_verdicts(driver, entry, live_grid, spec, live_s11) -> dict:
    """The battery's assembler, run on the live S11 as if it were a stage
    record, so the verdicts are computed by the code that computed the stored
    ones."""
    kind, dut = entry["kind"], entry["dut"]
    rec = {
        "kind": kind, "dut": dut, "rung_um": RUNG_UM,
        "drive": entry["drive"], "num_periods": entry["num_periods"],
        "n_steps": entry["n_steps"],
        "declared": driver.declared(kind, dut, RUNG_UM * 1e-6),
        "realized_grid": live_grid, "port_spec": spec,
        "preflight": {"text": []}, "warnings": [], "wall_s": 0.0, "peak_memory": {},
        "s11": driver._c(live_s11),
    }
    return driver._solve_entry(rec)


@pytest.mark.parametrize("dut", DUTS)
@pytest.mark.parametrize("kind", KINDS)
def test_the_coarsest_mesh_still_solves_to_its_stored_s11(fixture, driver, kind, dut):
    key = f"{kind}_{dut}_{RUNG_UM}um"
    entry = fixture["solves"][key]
    assert entry["provenance"]["commit"].startswith(LOCK_PROVENANCE["commit"]), (
        f"{key} was measured at {entry['provenance']['commit']}, but this guard's "
        f"LOCK_PROVENANCE names {LOCK_PROVENANCE['commit']}: the battery was measured "
        "again and the guard was not moved to the new records")
    dx = RUNG_UM * 1e-6

    # Before any step: the line the driver builds today is the recorded line —
    # the same node count, the port and the load on the same nodes, the same
    # walls, the same time step.
    sim = driver.build_sim(kind, dut, dx, drive=entry["drive"])
    live_grid = driver.assert_realized_grid(sim, kind, dut, dx)
    moved = drift.realized_differences(entry["realized_grid"], live_grid)
    assert not moved, drift.stale_record(
        FAMILY, key, ["the channel built today is not the recorded channel: "
                      + "; ".join(moved)], REMEASURE)

    result = driver.solve_s11(sim, num_periods=entry["num_periods"])
    live = driver._s11_of(result)
    spec = driver.port_spec_record(result, kind, dut, dx)

    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    stored = drift.complex_array(entry["s11"])
    findings = drift.realized_differences(entry["port_spec"], spec, "port_spec")
    report: list[str] = []
    if dut == "matched":
        findings += drift.bound_findings("the matched line's |S11|", freqs, live,
                                         report=report)
    else:
        findings += drift.magnitude_findings("|S11|", freqs, stored, live,
                                             report=report)
        findings += drift.crossing_findings(
            "the phase of S11",
            driver.phase_crossings(freqs, stored)["crossings"],
            driver.phase_crossings(freqs, live)["crossings"],
            float(freqs[0]), float(freqs[-1]), report=report)
    findings += drift.verdict_findings(
        entry, _live_verdicts(driver, entry, live_grid, spec, live),
        VERDICTS["matched" if dut == "matched" else "reflecting"])
    print(f"[battery drift] {key}: " + "; ".join(report))
    assert not findings, drift.stale_record(FAMILY, key, findings, REMEASURE)


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
    phase turning the other way, a flipped verdict, a moved cell. A check that
    stops reporting, or reports inside its bar, reds here on every pull
    request, not only in the weekly lane that solves. Arithmetic, no solve."""
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
    assert drift.verdict_findings({"v": {"ok": True}}, {"v": {"ok": False}}, [("v", "ok")])
    assert drift.verdict_findings({"v": {"ok": False}}, {"v": {"ok": True}}, [("v", "ok")])
    assert not drift.verdict_findings({"v": None}, {"v": {"ok": False}}, [("v", "ok")])
    assert drift.realized_differences({"nodes": [32, 2, 5]}, {"nodes": [33, 2, 5]})
    assert not drift.realized_differences({"dt": 1.9065748695310057e-12},
                                          {"dt": 1.9065748695310057e-12 * (1 + 1e-12)})
