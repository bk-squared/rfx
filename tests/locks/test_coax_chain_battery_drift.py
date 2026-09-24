"""The coaxial chain battery's two-port line on its coarsest mesh, solved again
and held to its stored S.

The battery (``tests/fixtures/coax_chain_battery/fixture.json``) is a PTFE-filled
coaxial line — pin radius 0.635 mm, outer conductor 2.055 mm, 48.6 ohm — with
its two feeds 58.6 mm apart. The thru is the bare line: it reflects nothing, so
|S11| is a deep null at every bin and |S21| is 0 dB. The bead multiplies the
fill's permittivity by 4 over four annulus widths (5.68 mm) midway along the
line: a section of half the line's impedance that reflects about -4 dB at its
worst and passes everything where it is half a wavelength long, near 9.1 GHz.

Its replay test (``tests/oracle/test_coax_chain_battery.py``) reads the stored S
only. This file builds the 4-annulus-cell mesh (0.355 mm cells) of both lines
again with the battery's own driver, refuses to solve unless the realized
cross-section, feeds, probes and bead are the recorded ones and the record is
as long, solves them, and holds the live S to the stored S with the bar the
battery is judged by:

* |S21| within 2 dB at every bin, on both lines, and its phase turning the same
  way with frequency as in the record (falling, 14-16 rad across the band). A
  conjugated S (the other time convention) keeps every magnitude and the
  reflection zero and flips only the phase direction;
* the electrical length of both lines within 1 % of the record's: the
  least-squares slope of S21's unwrapped phase against frequency, live over
  stored. The thru has no frequency feature, so this is the only check here
  that sees it grow longer: with every edge permittivity above vacuum read
  5 % high it transmits as before and delays about 2.4 % more;
* the thru's |S11| and |S22| held to -20 dB, a deep null compared with nothing
  (the PI's 2026-09-21 ruling);
* the bead's reflection zero — the vertex of the parabola through |S11|^2, the
  battery's own estimator — within 1 %, and its |S11| and |S22| within 2 dB
  outside the zero's core: the bins where the stored curve or the closed form
  of the section is at or below -20 dB;
* the record's own verdicts on this mesh — settled below -40 dB where the lane
  emits a witness, column power at most 1.02, reciprocity at most 0.02, the
  thru's reflection under its bound — hold for the live S.

A red here means the stored battery describes a different solver from the one
under test. The remedy is to measure the battery again, not to move anything in
this file.

Lane: slow_physics, the weekly CPU lane (validation.yml, slow-tests). Wall
time: 98 s for the thru and 94 s for the bead on four VESSL CPU cores (run
369367264074); 66 and 68 s on an A6000 (run 369367264076).
"""
LOCK_PROVENANCE = {
    "fixture": "tests/fixtures/coax_chain_battery/fixture.json",
    "generator": "scripts/diagnostics/coax_chain_battery_measure.py",
    "commit": "162a66fa",
    "date": "2026-09-23",
    "run_id": "369367263795 (bead), 369367263551 (thru)",
    "host": "VESSL gpu, jax 0.6.2 cuda, float32",
    "pinned_until": "2027-03-23",
}

import json

import numpy as np
import pytest

from tests import _chain_battery_drift as drift

FAMILY = "coaxial"
DRIVER = LOCK_PROVENANCE["generator"]
FIXTURE = drift.REPO / LOCK_PROVENANCE["fixture"]
REMEASURE = (f"`PYTHONPATH=. python {DRIVER} --stage solve --dut <dut> --rung <4|6|9> "
             "--out <dir> --run-id <id>` for every DUT and rung, then `--assemble`")

RUNG = 4
DUTS = ("thru", "bead")
# The records this guard reads, by the commit each was measured at. The
# fixture's own `record_provenance` block shows the two commits share one
# rfx/ tree.
RECORD_COMMITS = {"thru": "ca6da2b1", "bead": LOCK_PROVENANCE["commit"]}


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def driver():
    return drift.load_driver(DRIVER)


@pytest.mark.slow_physics
@pytest.mark.parametrize("dut", DUTS)
def test_the_coarsest_mesh_still_solves_to_its_stored_s(fixture, driver, dut):
    key = f"{dut}_rung{RUNG}"
    entry = fixture["solves"][key]
    assert entry["provenance"]["commit"].startswith(RECORD_COMMITS[dut]), (
        f"{key} was measured at {entry['provenance']['commit']}, but this guard names "
        f"{RECORD_COMMITS[dut]}: the battery was measured again and the guard was "
        "not moved to the new records")

    # Before any step: the line the driver builds today is the recorded line —
    # the same annulus in cells, pin and shell cells, wall thickness, feed,
    # probe and bead planes — and the record is as many steps long.
    sim = driver.build_sim(RUNG, dut, drive=entry["drive"])
    live_geometry = driver.assert_realized(sim, RUNG, dut)
    moved = drift.realized_differences(entry["realized"], live_geometry)
    n_steps = driver.record_steps(sim._build_grid(), entry["record_units"])
    if n_steps != entry["n_steps"]:
        moved.append(f"the record is {n_steps} steps, recorded {entry['n_steps']}")
    assert not moved, drift.stale_record(
        FAMILY, key, ["the line built today is not the recorded line: "
                      + "; ".join(moved)], REMEASURE)

    result = driver.solve_two_port(sim, n_steps=n_steps,
                                   eps_scale=driver._dut_eps_scale(sim, RUNG, dut))
    findings = drift.realized_differences(
        entry["cross_check"],
        driver.cross_check_result_against_layout(live_geometry, result, "two_port"),
        "cross_check")
    live = np.asarray(result.s_params)
    freqs = np.asarray(entry["freqs_hz"], dtype=float)
    np.testing.assert_allclose(np.asarray(result.freqs, dtype=float), freqs,
                               rtol=1e-6, atol=0.0)
    stored = drift.complex_array(entry["S"])

    report: list[str] = []
    findings += drift.magnitude_findings("|S21|", freqs, stored[1, 0, :], live[1, 0, :],
                                         report=report)
    findings += drift.phase_direction_findings(
        "S21", freqs, stored[1, 0, :], live[1, 0, :],
        deep=drift.db(stored[1, 0, :]) <= drift.DEEP_NULL_DB, report=report)
    findings += drift.electrical_length_findings("S21", freqs, stored[1, 0, :],
                                                 live[1, 0, :], report=report)
    if dut == "thru":
        findings += drift.bound_findings("the thru's |S11|", freqs, live[0, 0, :],
                                         report=report)
        findings += drift.bound_findings("the thru's |S22|", freqs, live[1, 1, :],
                                         report=report)
    else:
        zero_stored = driver.parabolic_min(freqs, np.abs(stored[0, 0, :]))
        zero_live = driver.parabolic_min(freqs, np.abs(live[0, 0, :]))
        if zero_live.get("at_band_edge"):
            findings.append(f"the live |S11| is smallest at the band edge "
                            f"({zero_live['bin_hz'] / 1e9:.4f} GHz): there is no "
                            "reflection zero inside the band to place")
        else:
            findings += drift.frequency_findings(
                "the reflection zero (vertex of |S11|^2)", zero_stored["interp_hz"],
                zero_live["interp_hz"], report=report)
        section = driver.bead_referee(
            freqs, a=live_geometry["pin_radius_m"], b=live_geometry["outer_radius_m"],
            eps_fill=float(driver.PTFE_EPS_R), eps_scale=driver.BEAD_EPS_SCALE,
            length_m=live_geometry["bead_length_realized_m"])
        closed_form_core = drift.db(section["abs_S11"]) <= drift.DEEP_NULL_DB
        for label, i in (("|S11|", 0), ("|S22|", 1)):
            core = closed_form_core | (drift.db(stored[i, i, :]) <= drift.DEEP_NULL_DB)
            findings += drift.magnitude_findings(label, freqs, stored[i, i, :],
                                                 live[i, i, :], deep=core, report=report)

    power = driver.power_metrics(live)
    witness = driver.settling_witness(result)
    live_verdicts = {
        "settled": (bool(np.all(np.asarray(witness["settling_db"], dtype=float)
                                <= drift.SETTLING_DB))
                    if witness["has_energy_witness"] else None),
        "column_power_within_bar": bool(
            power["max_column_power"] <= drift.COLUMN_POWER_MAX),
        "reciprocity_within_bar": bool(
            power["reciprocity_metric"] <= drift.RECIPROCITY_MAX),
    }
    paths = [("settled",), ("column_power_within_bar",), ("reciprocity_within_bar",)]
    if dut == "thru":
        live_verdicts["thru_reflection_within_deep_null_bound"] = bool(
            drift.db(live[0, 0, :]).max() <= drift.DEEP_NULL_DB)
        paths.append(("thru_reflection_within_deep_null_bound",))
    findings += drift.verdict_findings(entry, live_verdicts, paths)
    print(f"[battery drift] {key}: " + "; ".join(report))
    assert not findings, drift.stale_record(FAMILY, key, findings, REMEASURE)
