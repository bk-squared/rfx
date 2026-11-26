"""The microstrip chain battery's stub notch and thru on their coarsest mesh,
solved again and held to their stored S.

The battery (``tests/fixtures/msl_chain_battery/fixture.json``) is a 600 um
microstrip on 300 um of eps_r = 3.66 over a ground wall, 20 mm between its two
ports, with a 12 mm open stub at the midpoint, solved as drawn on edge-aware
graded x and y cells. The stub is a quarter wavelength long near 3.71 GHz, where
it shorts the line: |S21| falls to a transmission zero and the power reflects.

No other test reads the fixture, so no CI test judges its ladder or its 25 um
column power; this lock is the check that the code still produces the 100 um
rung's numbers. On that rung the stored record puts the notch at 3.698 GHz,
0.35 % below the quarter-wave closed form's 3.711 GHz. The closed form omits the
open-end extension, so that figure places the notch and does not grade it. This
file builds the 100 um board again on the notch's edge-aware profiles with the
battery's own driver under strict admission, refuses to solve unless the
realized trace, stub, substrate and ports are the recorded ones, solves it with
the recorded drive and record length, and holds the live S to the stored S with
the bar the battery is judged by:

* the notch frequency — the vertex of the parabola through |S21|^2 at the
  minimum bin and its two neighbours, the battery's own estimator — within 1 %;
* |S21| and |S11| within 2 dB at every bin outside the notch core, the bins
  where the stored |S21| is at or below -20 dB. Inside the core the depth is a
  difference of two near-zeros and is compared with nothing;
* the phase of S21 turning the same way with frequency as it does in the
  record: the sign of the least-squares slope of its unwrapped phase, fitted
  outside the core (falling, -2.02 rad/GHz on the stored record). A conjugated
  S (the other time convention) keeps every magnitude and the notch frequency
  and flips only this;
* the record's own verdicts at this mesh — the record settles below -40 dB,
  column power at most 1.02, reciprocity at most 0.02 — hold for the live S.

The thru is the same board without the stub, on the notch's cells, built and
checked the same way before any step. It has no frequency feature to place, and
a line that grows longer electrically transmits exactly as before, so its
magnitude says nothing about its length. Its lock holds:

* |S21| within 2 dB at every bin, and |S11| under the -20 dB bound the battery
  holds the thru's reflection to;
* the electrical length of S21 within 1 % of the record's: the least-squares
  slope of its unwrapped phase against frequency, live over stored;
* the phase direction and the record's verdicts, as for the notch.

Live and stored S are compared directly at the feed planes.

The notch's own slope is not read as an electrical length: across its
transmission zero the unwrapped phase can take the zero's half-turn either way,
depending on where the zero falls between bins, so its slope measures the
unwrap, not the line.

A red here means the stored battery describes a different solver from the one
under test. The remedy is to measure the battery again, not to move anything in
this file.

Lane: gpu and slow, the gpu step of the weekly A6000 lane (run by the
merge-train runner on Mac, remi gpu-a6000). Its gpu selection excludes
crossval_ladder. At the recorded 40 source periods each solve takes about 66 s
on an RTX A6000 (the stage records' solve wall time, not the job's).
"""
LOCK_PROVENANCE = {
    "fixture": "tests/fixtures/msl_chain_battery/fixture.json",
    "generator": "scripts/diagnostics/msl_chain_battery_measure.py",
    "commit": "f26760e288",
    "date": "2026-11-26",
    "run_id": "absorber-battery-ea-settle-f2rhs (notch), absorber-battery-ea-p40-100-50-47mmr (thru)",
    "host": "remi gpu-a6000 (RTX A6000), jax 0.10.2 cuda, float32",
    "pinned_until": "2027-05-26",
}

import json

import numpy as np
import pytest

from tests import _chain_battery_drift as drift

FAMILY = "microstrip"
DRIVER = LOCK_PROVENANCE["generator"]
FIXTURE = drift.REPO / LOCK_PROVENANCE["fixture"]
REMEASURE = (f"`PYTHONPATH=. python {DRIVER} --stage solve --dut <notch|thru> "
             "--rung <100|50|25> --out <dir> --run-id <id>` for every rung, then "
             "`--assemble`")

RUNG_UM = 100
KEY = f"notch_{RUNG_UM}um"
THRU_KEY = f"thru_{RUNG_UM}um"


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def driver():
    return drift.load_driver(DRIVER)


def _solve_live(fixture, driver, key: str, dut: str):
    """The record ``key``, the board the driver builds for it today, solved.

    Refuses before any step when the record names another commit than this
    guard, or when the realized trace, stub, substrate, ports or drive differ
    from the record's.

    ``_solve_live`` returns the live and stored S directly at the feed planes,
    with the declared port reference impedances.
    """
    assert fixture["schema_version"] == 2
    assert fixture["reference_plane"] == "feed"
    entry = fixture["solves"][key]
    assert entry["provenance"]["commit"].startswith(LOCK_PROVENANCE["commit"]), (
        f"{key} was measured at {entry['provenance']['commit']}, but this guard's "
        f"LOCK_PROVENANCE names {LOCK_PROVENANCE['commit']}: the battery was measured "
        "again and the guard was not moved to the new records")
    dx = RUNG_UM * 1e-6

    sim = driver.build_sim(dx, dut, drive=entry["drive"], snap="strict")
    live_geometry = driver.assert_realized(sim, dx, dut)
    moved = drift.realized_differences(entry["realized"], live_geometry)
    assert not moved, drift.stale_record(
        FAMILY, key, ["the board built today is not the recorded board: "
                      + "; ".join(moved)], REMEASURE)

    result = driver.solve(sim, num_periods=entry["num_periods"])
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    np.testing.assert_allclose(np.asarray(result.freqs, dtype=float), freqs,
                               rtol=1e-6, atol=0.0)
    return entry, result, freqs, drift.complex_array(entry["S"]), np.asarray(result.S)


def _verdict_findings(driver, entry, result, live) -> list[str]:
    """The record's own verdicts — settled, column power, reciprocity — computed
    from the live S."""
    power = driver.power_metrics(live)
    settling = (None if result.settling_db is None
                else np.asarray(result.settling_db, dtype=float))
    return drift.verdict_findings(entry, {
        "settled": bool(settling is not None
                        and np.all(settling <= drift.SETTLING_DB)),
        "column_power_within_bar": bool(
            power["max_column_power"] <= drift.COLUMN_POWER_MAX),
        "reciprocity_within_bar": bool(
            power["reciprocity_metric"] <= drift.RECIPROCITY_MAX),
    }, (("settled",), ("column_power_within_bar",), ("reciprocity_within_bar",)))


@pytest.mark.gpu
@pytest.mark.slow
def test_the_notch_on_the_coarsest_mesh_still_solves_to_its_stored_s(fixture, driver):
    entry, result, freqs, stored, live = _solve_live(fixture, driver, KEY, "notch")

    notch_stored = driver.parabolic_min(freqs, np.abs(stored[1, 0, :]))
    notch_live = driver.parabolic_min(freqs, np.abs(live[1, 0, :]))
    findings = []
    report: list[str] = []
    if notch_live.get("at_band_edge"):
        findings.append(f"the live |S21| is smallest at the band edge "
                        f"({notch_live['bin_hz'] / 1e9:.4f} GHz): there is no notch "
                        "inside the band to place")
    else:
        findings += drift.frequency_findings(
            "the notch (vertex of |S21|^2)", notch_stored["interp_hz"],
            notch_live["interp_hz"], report=report)
    core = drift.db(stored[1, 0, :]) <= drift.DEEP_NULL_DB
    findings += drift.magnitude_findings("|S21|", freqs, stored[1, 0, :],
                                         live[1, 0, :], deep=core, report=report)
    findings += drift.magnitude_findings("|S11|", freqs, stored[0, 0, :],
                                         live[0, 0, :], deep=core, report=report)
    findings += drift.phase_direction_findings("S21", freqs, stored[1, 0, :],
                                               live[1, 0, :], deep=core, report=report)
    findings += _verdict_findings(driver, entry, result, live)
    print(f"[battery drift] {KEY}: " + "; ".join(report))
    assert not findings, drift.stale_record(FAMILY, KEY, findings, REMEASURE)


@pytest.mark.gpu
@pytest.mark.slow
def test_the_thru_on_the_coarsest_mesh_still_solves_to_its_stored_s(fixture, driver):
    entry, result, freqs, stored, live = _solve_live(fixture, driver, THRU_KEY, "thru")

    report: list[str] = []
    findings = drift.magnitude_findings("|S21|", freqs, stored[1, 0, :], live[1, 0, :],
                                        report=report)
    findings += drift.bound_findings("the thru's |S11|", freqs, live[0, 0, :],
                                     report=report)
    findings += drift.phase_direction_findings("S21", freqs, stored[1, 0, :],
                                               live[1, 0, :], report=report)
    findings += drift.electrical_length_findings("S21", freqs, stored[1, 0, :],
                                                 live[1, 0, :], report=report)
    findings += _verdict_findings(driver, entry, result, live)
    print(f"[battery drift] {THRU_KEY}: " + "; ".join(report))
    assert not findings, drift.stale_record(FAMILY, THRU_KEY, findings, REMEASURE)
