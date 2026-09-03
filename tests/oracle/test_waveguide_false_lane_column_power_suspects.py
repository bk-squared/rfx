"""Replay gate for the empty-guide column-power suspect comparison (issue #873).

WHAT WAS MEASURED. On the WR-90 chain battery's thru — an empty guide, which
neither reflects nor absorbs — the ``normalize=False`` extractor reports column
power above unity: 1.8253e-02 / 4.0817e-03 / 9.8341e-04 at dx = 2.54 / 1.27 /
0.635 mm (``tests/fixtures/waveguide_chain_battery/fixture.json``, ``cells[]``
with ``dut="thru", lane="false"``, key ``column_power_max``). The excess falls
~4x per halving, so it is second order in dx.

WHAT THIS FILE PINS. The comparison recomputed by
``scripts/diagnostics/waveguide_false_lane_column_power_suspects.py`` and stored
in ``tests/fixtures/waveguide_false_lane_column_power/suspects.json``: how much
of that excess each named suspect accounts for, and the pre-declared verdict.
Decision rule and tolerances: ``docs/design_notes/
waveguide_false_lane_column_power_predeclaration.md`` — written before the
comparison ran.

THE VERDICT PINNED HERE IS NON-CLOSING. The three suspects the issue lists,
even multiplied together, reproduce the ladder's dx-SCALING but only 61-65 % of
its MAGNITUDE at every rung, and the residual is itself first order in dx. So
the suspect list is incomplete; the tests below lock that state so a later claim
of closure has to move a number, not a sentence.

WHAT IS NOT PINNED HERE. No physics gate, no tolerance and no golden moves. The
``normalize=False`` extractor is untouched: nothing here licenses a fix, because
nothing here identifies the whole mechanism.

IF THIS FILE GOES RED BECAUSE THE PORT CUTOFF WAS CORRECTED (issue #868, the
aperture solved on N+1 cells), that is the expected signal, not a regression:
re-run the diagnostic script and re-commit ``suspects.json`` with the new
baseline. Do not relax an assertion to keep it green.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPORT = (
    Path(__file__).resolve().parents[1]
    / "fixtures/waveguide_false_lane_column_power/suspects.json"
)
BATTERY = (
    Path(__file__).resolve().parents[1]
    / "fixtures/waveguide_chain_battery/fixture.json"
)
RUNGS = ("coarse", "mid", "fine")


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text())


@pytest.fixture(scope="module")
def battery() -> dict:
    return json.loads(BATTERY.read_text())


def _ladder(report: dict, key: str) -> np.ndarray:
    return np.asarray(report["ladders"][key], dtype=float)


def test_report_replays_the_frozen_battery_run(report, battery):
    """The comparison is a re-read of ONE stored run, not a new measurement."""
    assert report["source_run_id"] == battery["provenance"]["run_id"]
    assert report["source_commit"] == battery["provenance"]["commit"]
    assert report["predeclaration"] == (
        "docs/design_notes/waveguide_false_lane_column_power_predeclaration.md"
    )
    for rung in RUNGS:
        entry = report["per_rung"][rung]
        cell = next(
            c for c in battery["cells"]
            if c["dut"] == "thru" and c["lane"] == "false" and c["rung"] == rung
        )
        assert entry["measured"]["column_power_max"] == cell["column_power_max"]
        assert entry["settling_db"] == cell["settling_db"]
        assert entry["preflight"] == cell["preflight"] == []
        # rule 10: the only warning on this lane is the documented one
        assert len(entry["warnings"]) == 1
        assert "normalize=False" in entry["warnings"][0]["message"]


def test_settling_witness_holds_on_every_replayed_rung(report):
    """Every |S| quoted here is backed by a ring-down below -40 dB per drive."""
    for rung in RUNGS:
        for drive, db in report["per_rung"][rung]["settling_db"].items():
            assert db <= -40.0, (rung, drive, db)


def test_the_excess_is_a_spurious_reflection_not_a_transmission_error(report):
    """|S11|^2 carries most of the column-power excess at every rung.

    Measured share of the worst bin's excess: 0.726 / 0.690 / 0.702. Locked at
    >= 0.65, a 6 % margin below the smallest of the three. At mid and fine the
    transmission term is NEGATIVE at the worst bin, so the reflection term is
    more than the whole excess there.
    """
    shares = []
    for rung in RUNGS:
        worst = report["per_rung"][rung]["excess_decomposition_worst_bin_col0"]
        share = worst["reflection_term_abs_s11_sq"] / worst["column_power_excess"]
        shares.append(share)
        assert share >= 0.65, (rung, worst)
    assert shares[1] > 1.0 and shares[2] > 1.0


def test_the_guide_propagates_a_cutoff_the_port_config_does_not_carry(report):
    """The S21 phase fit picks the guide's discrete cutoff, never the port's.

    rms residual of ``unwrap(angle S21) = -beta(f; fc)*L`` over the 17 bins:
    0.080 / 0.017 / 0.004 deg at the guide's discrete cutoff against
    8.613 / 5.084 / 2.753 deg at the port config's ``f_cutoff``. Locked at a
    factor 50; the smallest measured separation is 108x.
    """
    for rung in RUNGS:
        c = report["per_rung"][rung]["cutoffs_hz"]
        assert c["rms_deg_at_port_cutoff"] / c["rms_deg_at_discrete_guide"] >= 50.0
        # the fit and the closed form for an N-cell guide agree to 0.1 %
        assert abs(c["guide_fitted_from_s21_phase"] / c["guide_discrete"] - 1.0) < 1e-3
        # the port's effective aperture is one cell wider than the guide
        width = c["port_cutoff_effective_width_cells"]
        assert abs(width - (report["per_rung"][rung]["guide_cells_u"] + 1)) < 0.05


def test_only_the_cutoff_suspect_is_first_order_in_dx(report):
    """S2 and S3 fall ~4x per halving; the measured excess needs a ~2x factor.

    Successive-rung ratios of the band-mean |Gamma|: measured 2.19 / 2.09,
    S1 1.77 / 1.87, S2 4.05 / 4.01, S3 3.88 / 3.94. A suspect whose Gamma is
    second order contributes a FOURTH-order column-power term and cannot carry
    an excess that falls 4x per halving.
    """
    def ratios(key: str) -> np.ndarray:
        v = _ladder(report, key)
        return v[:-1] / v[1:]

    assert np.all((ratios("measured_abs_s11_band_mean") > 1.9)
                  & (ratios("measured_abs_s11_band_mean") < 2.4))
    assert np.all((ratios("S1_modal_cutoff_impedance") > 1.5)
                  & (ratios("S1_modal_cutoff_impedance") < 2.5))
    assert np.all(ratios("S2_normal_half_cell_h_average") >= 3.5)
    assert np.all(ratios("S3_aperture_transverse_pairing") >= 3.5)


def test_no_named_suspect_reproduces_the_ladder(report):
    """The pre-declared verdict: branch (iii), NON-CLOSING.

    Rule (design note, fixed before the numbers): a suspect reproduces the
    ladder when its band-mean |Gamma| is within a factor 1.25 of the measured
    band-mean |S11| at all three rungs AND its successive ratios are within 0.20
    of the measured ones. Measured pred/meas for the product of all three:
    0.645 / 0.620 / 0.607 — the scaling leg passes, the magnitude leg does not.
    """
    assert report["verdict"]["branch"] == "iii_non_closing"
    assert report["verdict"]["reproducing"] == []
    product = report["decision"]["product_S1_S2_S3"]
    assert product["ratio_within_0p20"] is True
    assert product["magnitude_within_1p25"] is False
    assert max(product["pred_over_measured"]) <= 0.70


def test_the_unexplained_remainder_is_itself_first_order_in_dx(report):
    """Measured minus product: 0.0337 / 0.0164 / 0.0081, ratios 2.05 / 2.03.

    That is the load-bearing negative result. A second-order remainder would be
    a rounding story about the three suspects; a first-order one means a fourth
    first-order channel exists in the extraction that none of them names.
    """
    resid = np.asarray(
        [report["per_rung"][r]["residual"]["measured_minus_product_band_mean"]
         for r in RUNGS], dtype=float
    )
    assert np.all(resid > 0.0)
    ratios = resid[:-1] / resid[1:]
    assert np.all((ratios > 1.8) & (ratios < 2.3)), ratios


def test_the_flux_lane_thru_number_is_a_construction_not_a_comparison(report):
    """Why the 550x thru gap overstates the case.

    ``extract_waveguide_s_matrix_flux`` builds the diagonal from
    ``|F_ref - F_dev|``; on an empty guide the reference run IS the device run,
    so |S11| is identically zero and |S21| identically one. The load-bearing
    flux comparison is the slab, where the two runs differ.
    """
    for rung in RUNGS:
        m = report["per_rung"][rung]["measured"]
        assert m["flux_lane_abs_s11_band_mean"] == 0.0
        assert abs(m["flux_lane_column_power_max"] - 1.0) < 1e-4


def test_live_port_config_still_carries_the_cutoff_the_report_compared_against():
    """Ties the committed JSON to live code: rebuild the coarse port config.

    Builds the port configuration only — no FDTD solve. If this fires because
    the port aperture was narrowed to the guide's N cells (issue #868), re-run
    ``scripts/diagnostics/waveguide_false_lane_column_power_suspects.py`` and
    re-commit the report; do not widen the tolerance.
    """
    import jax.numpy as jnp

    from tests import _waveguide_chain_battery_fixture as F

    rep = json.loads(REPORT.read_text())
    dx = rep["per_rung"]["coarse"]["dx_m"]
    sim = F.build_simulation("thru", dx)
    grid = sim._build_grid()
    cfg = sim._build_waveguide_port_config(
        sim._waveguide_ports[0], grid, jnp.asarray(F.FREQS),
        int(grid.num_timesteps(F.NUM_PERIODS)),
    )
    recorded = rep["per_rung"]["coarse"]["cutoffs_hz"]["port_config_f_cutoff"]
    assert float(cfg.f_cutoff) == pytest.approx(recorded, rel=1e-9)
    assert float(cfg.dt) == pytest.approx(rep["per_rung"]["coarse"]["dt_s"], rel=1e-12)
    assert np.asarray(cfg.ez_profile).shape == (
        rep["per_rung"]["coarse"]["aperture_cells_u"],
        rep["per_rung"]["coarse"]["aperture_cells_v"],
    )
