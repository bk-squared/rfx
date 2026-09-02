"""Fast structural companion to the @slow platform-envelope datum for issue
#610 (see tests/test_msl_port_integration.py::
test_msl_thru_line_z0_length_invariance_and_positive_sign's docstring for the
physics and PLATFORM ENVELOPE note). This file runs NO FDTD -- it checks that
the recorded cross-platform datum ledger,
tests/fixtures/msl_z0_length_invariance/platform_datums.json, stays
internally consistent, so a future red on this lock is diagnosable as
platform-vs-code without re-deriving anything.

Bounds are imported from tests/test_msl_port_integration.py, not restated --
a hardcoded 0.007 / 0.15 in this second file would be exactly the "second
copy of the gate that can drift from the mechanism" class of defect the
#610 review flagged.
"""

from __future__ import annotations

LOCK_PROVENANCE = {
    "fixture": "tests/fixtures/msl_z0_length_invariance/platform_datums.json",
    "generator": "hand-recorded (PR #611 docstring; validation.yml run logs; AMD EPYC re-measure at 1f005d0d)",
    "commit": "826f686",
    "date": "2026-08-30",
    "run_id": "gh-actions 31367543755, 32004256088, 32700177994 (witness rows); derivation row unknown",
    "host": "GitHub ubuntu-24.04 jax 0.6.2; AMD EPYC 9654 Linux jax 0.6.2; derivation platform not recorded",
    "pinned_until": "2027-02-26",
}

import json
from pathlib import Path

from tests._gate_policy import gate_from_envelope
from tests.test_msl_port_integration import LEG_S11_BOUND, MEASURED_SPREAD_ENVELOPE

DATUMS_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "msl_z0_length_invariance" / "platform_datums.json"
)

# Recomputing a spread from a record's own `legs` dict does not reproduce the
# recorded `spread` bit-for-bit: the derivation legs (57.3381/57.5778/
# 57.6030) recompute to 0.0046064518, not the recorded 0.004607 (#610 review
# check 5). Exact equality applies only to the recorded `spread` field.
_RECOMPUTE_TOL = 1e-6


def _load():
    with open(DATUMS_PATH) as f:
        return json.load(f)


def test_datums_file_exists_and_identifies_its_test():
    data = _load()
    assert data["schema"] == "rfx.msl_z0_length_invariance_platform_datums"
    assert data["test"] == (
        "tests/test_msl_port_integration.py::"
        "test_msl_thru_line_z0_length_invariance_and_positive_sign"
    )


def test_bounds_are_single_sourced_from_the_slow_test_not_restated():
    """The JSON's bounds must equal what the slow test itself derives/enforces
    -- computed from the SAME module constants the slow test imports, not a
    fresh literal in this file."""
    data = _load()
    spread_tol = gate_from_envelope(MEASURED_SPREAD_ENVELOPE, quantum=1000)
    assert spread_tol == 0.007
    assert data["bounds_unchanged"]["spread_tol"] == spread_tol
    assert data["bounds_unchanged"]["per_leg_mean_s11_max"] == LEG_S11_BOUND == 0.15
    assert data["derivation"]["measured_spread_envelope"] == MEASURED_SPREAD_ENVELOPE


def test_derivation_row_recomputes_within_tolerance_and_has_no_estimated_platform():
    data = _load()
    deriv = next(d for d in data["datums"] if d["role"] == "derivation")
    legs = deriv["legs"]
    vals = [legs["8"]["mean_abs_z0"], legs["10"]["mean_abs_z0"], legs["12"]["mean_abs_z0"]]
    recomputed = (max(vals) - min(vals)) / (sum(vals) / len(vals))
    assert abs(recomputed - deriv["spread"]) < _RECOMPUTE_TOL, (
        f"derivation spread {deriv['spread']} does not recompute from its own "
        f"legs (got {recomputed}) within {_RECOMPUTE_TOL}"
    )
    # "not measured" beats a silent estimate -- do not let a future edit infer
    # a platform (e.g. from an unrelated later golden capture) for this row.
    for field in ("cpu", "os", "jax", "numpy", "blas"):
        assert deriv["platform"][field] == "not recorded", (
            f"derivation platform.{field} should stay 'not recorded' unless a "
            f"dated, sourced measurement backs a real value"
        )


def test_every_passed_datum_is_actually_under_bound():
    """Falsifier (#610 review check 5): tag a record PASSED with the retired
    L=6mm spread (0.049556) or an over-bound L=12 |S11| (0.1513) and this
    reds."""
    data = _load()
    spread_tol = data["bounds_unchanged"]["spread_tol"]
    s11_max = data["bounds_unchanged"]["per_leg_mean_s11_max"]
    for d in data["datums"]:
        if d["verdict"] != "PASSED":
            continue
        assert d["spread"] < spread_tol, (
            f"{d['id']}: recorded spread {d['spread']} is not < bound {spread_tol} "
            f"but is tagged PASSED"
        )
        leg12 = d["legs"]["12"]
        if "mean_s11" in leg12:
            assert leg12["mean_s11"] < s11_max, (
                f"{d['id']}: L=12mm mean|S11| {leg12['mean_s11']} is not < bound "
                f"{s11_max} but is tagged PASSED"
            )


def test_datums_are_unique_by_commit_and_platform():
    data = _load()
    keys = [(d["commit"], d["platform"].get("cpu")) for d in data["datums"]]
    assert len(keys) == len(set(keys)), f"duplicate (commit, cpu) datum: {keys}"
    # Real coverage today: 1 derivation + 3 CI witnesses + 1 pod witness. Not
    # an aspirational target -- a future accidental deletion of a datum reds
    # this against the count that is ACTUALLY on file.
    assert len(data["datums"]) >= 5


def test_every_datum_carries_a_precision_tag():
    data = _load()
    known = {"docstring_4-5dp", "printed_2dp", "float32_full"}
    for d in data["datums"]:
        assert d.get("precision") in known, f"{d['id']} missing/unknown precision tag"


def test_cross_platform_witnesses_agree_within_the_print_quantum():
    """Same-commit-second-platform falsifier check: if a witness datum failed
    to reproduce the derivation's spread to within the print quantum, the
    'platform, not code' attribution in the docstring would be false. All
    recorded witnesses currently agree -- this is the hypothesis surviving a
    test it could have failed, not an assumption."""
    data = _load()
    printed_2dp_witnesses = [
        d for d in data["datums"] if d["role"] == "witness" and d["precision"] == "printed_2dp"
    ]
    assert printed_2dp_witnesses, "no printed_2dp witness datums to cross-check"
    for d in printed_2dp_witnesses:
        # 2 dp print quantum on a ~0.46% spread is +-0.005 pp of slack either side.
        assert abs(d["spread"] - 0.0046) <= 0.0002, (
            f"{d['id']}: spread {d['spread']} outside the printed_2dp agreement "
            f"band around the derivation's 0.0046"
        )
