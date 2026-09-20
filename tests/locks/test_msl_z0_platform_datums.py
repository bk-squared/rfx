"""Fast structural companion to the @slow platform-envelope datum for issue
#610 (see tests/unit/sparams/test_msl_port_integration.py::
test_msl_thru_line_z0_length_invariance_and_positive_sign's docstring for the
physics and PLATFORM ENVELOPE note). This file runs NO FDTD -- it checks that
the recorded cross-platform datum ledger,
tests/fixtures/msl_z0_length_invariance/platform_datums.json, stays
internally consistent, so a future red on this lock is diagnosable as
platform-vs-code without re-deriving anything.

BOARD DIMENSION (#1084, schema_version 2). The ledger used to carry ONE
top-level `recipe`, and that recipe described the RETIRED dx = 80 um board
(trace drawn as a one-cell PEC Box, realized as ONE wall) while its `test`
field named a test that has run the #931 board (DX = H_SUB/3 = 84.67 um,
zero-thickness PEC sheet) since the migration commits. Nothing noticed,
because this file runs no FDTD and checked internal consistency only.

So every datum now carries a `board` key into `boards`, each board carries
its own recipe, and `active_board` says which one the named test runs.
``test_active_board_recipe_matches_the_live_test_constants`` imports DX and
H_SUB from the test module and asserts the active board's dx is that dx --
that is the check whose absence let the ledger drift off its own test. The
pre-#931 rows stand as a labelled historical record; they are not silently
compared against post-contract runs, because the agreement band that would
compare them is now per board.

Bounds are imported from tests/unit/sparams/test_msl_port_integration.py, not restated --
a hardcoded 0.007 / 0.15 in this second file would be exactly the "second
copy of the gate that can drift from the mechanism" class of defect the
#610 review flagged.
"""

from __future__ import annotations

LOCK_PROVENANCE = {
    "fixture": "tests/fixtures/msl_z0_length_invariance/platform_datums.json",
    "generator": "hand-recorded (PR #611 docstring; validation.yml run logs; AMD EPYC re-measure at 1f005d0d) + #796 per-sha bisect + #1084 board rows (VESSL 369367259284 docstring print; AMD EPYC 9654 CPU at 6d721a56 and c6788ef7; VESSL 369367261423 RTX 4090 GPU at c6788ef7; jax-0.6.2 stack control on the EPYC host at c6788ef7)",
    "commit": "c6788ef7",
    "date": "2026-09-16",
    "run_id": "gh-actions 31367543755, 32004256088, 32700177994 (pre-931 witness rows); VESSL 369367259284, 369367261423 (#931-board rows); derivation row unknown",
    "host": "GitHub ubuntu-24.04 jax 0.6.2; AMD EPYC 9654 Linux jax 0.6.2 and jax 0.10.2; RTX 4090 / i7-13700K jax 0.4.33 (VESSL remilab-c0); derivation platform not recorded",
    "pinned_until": "2027-02-26",
}

import json
from pathlib import Path

from tests._gate_policy import gate_from_envelope
from tests.unit.sparams.test_msl_port_integration import (
    DX,
    H_SUB,
    LEG_S11_BOUND,
    MEASURED_SPREAD_ENVELOPE,
)

DATUMS_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "msl_z0_length_invariance" / "platform_datums.json"
)

# Recomputing a spread from a record's own `legs` dict does not reproduce the
# recorded `spread` bit-for-bit: the derivation legs (57.3381/57.5778/
# 57.6030) recompute to 0.0046064518, not the recorded 0.004607 (#610 review
# check 5). Exact equality applies only to the recorded `spread` field.
_RECOMPUTE_TOL = 1e-6

# Two ways the same 2-decimal print reaches this ledger: straight off a CI
# log, or copied into the slow test's docstring. Both carry the same 0.01
# ohm / 0.01 pp quantum, so both answer to a board's printed_2dp_band.
_TWO_DP_PRECISIONS = ("printed_2dp", "docstring_2dp")


def _load():
    with open(DATUMS_PATH) as f:
        return json.load(f)


def test_datums_file_exists_and_identifies_its_test():
    data = _load()
    assert data["schema"] == "rfx.msl_z0_length_invariance_platform_datums"
    assert data["test"] == (
        "tests/unit/sparams/test_msl_port_integration.py::"
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


def test_datums_are_unique_by_board_commit_and_platform():
    """#1084: the key is (board, commit, platform). (commit, cpu) alone could not
    hold a dx=80um row and a #931-board row measured at the same commit on the
    same CPU -- which is exactly the pair the ledger needs to compare boards --
    and `cpu` alone cannot separate two rows that differ only in runtime stack,
    which is exactly the pair the #1084 stack control needs. The whole platform
    block is the identity, so jax/numpy/backend all count."""
    data = _load()
    keys = [
        (d["board"], d["commit"], json.dumps(d["platform"], sort_keys=True))
        for d in data["datums"]
    ]
    assert len(keys) == len(set(keys)), (
        "duplicate (board, commit, platform) datum: "
        + str([(b, c) for b, c, _ in keys])
    )
    # Real coverage today: pre-931 board 1 derivation + 3 CI witnesses + 1 pod
    # witness + 4 bisect rows; #931 board 5 rows (#1084). Not an aspirational
    # target -- a future accidental deletion of a datum reds this against the
    # count that is ACTUALLY on file.
    assert len(data["datums"]) >= 14


def test_every_datum_names_a_declared_board():
    data = _load()
    boards = data["boards"]
    assert data["active_board"] in boards
    for d in data["datums"]:
        assert d["board"] in boards, f"{d['id']}: unknown board {d['board']!r}"
    # Every declared board must actually have rows -- a board block with no
    # datum is a recipe nobody measured.
    used = {d["board"] for d in data["datums"]}
    assert used == set(boards), f"boards without datums: {set(boards) - used}"


def test_active_board_recipe_matches_the_live_test_constants():
    """THE #1084 LOCK. The ledger pointed at a retired board for the whole
    #931 migration because nothing compared its recipe against the test it
    names. This imports the test module's own DX/H_SUB and asserts the active
    board is that board.

    Falsifier: point `active_board` at "pre931-dx80um-onewall" (dx 8e-05) and
    this reds; change DX in the test module without adding a board here and it
    reds too."""
    data = _load()
    active = data["boards"][data["active_board"]]
    assert active["status"] == "active"
    rec = active["recipe"]
    assert rec["dx_m"] == DX, (
        f"active board {data['active_board']!r} declares dx_m={rec['dx_m']} but "
        f"the test it gates runs DX={DX} -- the ledger is describing a different "
        f"board than the one under gate (issue #1084)"
    )
    assert rec["dx_expression"] == "H_SUB / 3" and rec["dx_m"] == H_SUB / 3
    assert rec["h_sub_m"] == H_SUB
    for bid, board in data["boards"].items():
        if bid == data["active_board"]:
            continue
        assert board["status"].startswith("retired"), (
            f"{bid}: a non-active board must be marked retired, not {board['status']!r}"
        )
        assert board["recipe"]["dx_m"] != DX, (
            f"{bid}: a retired board sharing the active dx makes the board ids "
            f"ambiguous"
        )


def test_every_datum_carries_a_precision_tag():
    data = _load()
    known = {"docstring_4-5dp", "docstring_2dp", "printed_2dp", "float32_full"}
    for d in data["datums"]:
        assert d.get("precision") in known, f"{d['id']} missing/unknown precision tag"


def test_two_dp_witnesses_agree_within_their_own_board_print_quantum():
    """Same-board-second-platform falsifier check, per board (#1084).

    The band used to be one global +-0.0002 around 0.0046 -- the pre-#931
    board's number. Any #931-board row reds that, which is why the band is now
    a property of the board: each board declares `printed_2dp_band`, centred on
    its own reference datum. A row that misses its board's band is a real
    disagreement to diagnose, not a tolerance to widen.

    What this does NOT assert: that the boards agree with each other, or that
    2-dp and full-precision rows on one board agree. The #931 board's 2-dp row
    reads 0.0016 while its full-precision rows read 0.00048-0.00057; that gap
    is recorded and classified in the ledger's cross_platform_checks, not
    smoothed over here."""
    data = _load()
    boards = data["boards"]
    seen = set()
    for d in data["datums"]:
        if d["role"] != "witness" or d["precision"] not in _TWO_DP_PRECISIONS:
            continue
        band = boards[d["board"]]["printed_2dp_band"]
        seen.add(d["board"])
        assert abs(d["spread"] - band["center"]) <= band["halfwidth"], (
            f"{d['id']}: spread {d['spread']} outside the 2-dp agreement band "
            f"{band['center']} +- {band['halfwidth']} of its own board "
            f"{d['board']!r}"
        )
    assert seen, "no 2-dp witness datums to cross-check"
    for bid, board in boards.items():
        if "printed_2dp_band" in board:
            assert bid in seen, (
                f"{bid} declares a printed_2dp_band but has no 2-dp witness row "
                f"to hold to it"
            )


def test_each_board_band_is_centred_on_its_own_reference_datum():
    """The band centre must be a number the ledger actually measured on that
    board, not a hand-typed constant: it has to sit within one halfwidth of the
    reference datum's recorded spread, and that datum has to belong to the
    board."""
    data = _load()
    by_id = {d["id"]: d for d in data["datums"]}
    for bid, board in data["boards"].items():
        band = board.get("printed_2dp_band")
        if band is None:
            continue
        ref = by_id[band["reference_datum"]]
        assert ref["board"] == bid, (
            f"{bid}: band reference {ref['id']} belongs to board {ref['board']!r}"
        )
        assert abs(band["center"] - ref["spread"]) <= band["halfwidth"], (
            f"{bid}: band centre {band['center']} is more than one halfwidth "
            f"from its reference datum's spread {ref['spread']}"
        )


def test_recorded_cross_platform_checks_recompute_from_their_own_rows():
    """#1084 fact 2. Each entry in `cross_platform_checks` names two datum rows
    and records the gap between their spreads in percentage points plus the
    verdict the ledger's OWN classification_policy gives that gap. This
    recomputes both from the named rows, so a hand-edited gap or a verdict that
    contradicts the policy threshold reds here."""
    data = _load()
    checks = data["cross_platform_checks"]
    assert checks, "no cross-platform checks recorded"
    by_id = {d["id"]: d for d in data["datums"]}
    threshold = data["classification_policy"]["threshold_pp"]
    for c in checks:
        a, b = (by_id[i] for i in c["rows"])
        assert a["board"] == b["board"] == c["board"], (
            f"{c['id']}: compares rows from different boards"
        )
        gap = abs(a["spread"] - b["spread"]) * 100.0
        assert abs(gap - c["gap_pp"]) < 1e-9, (
            f"{c['id']}: recorded gap_pp {c['gap_pp']} does not recompute from "
            f"{a['id']}/{b['id']} (got {gap})"
        )
        same_commit = a["commit"] == b["commit"] and a["commit"] != "not recorded"
        if not same_commit:
            assert c["verdict"] == "unclassifiable", (
                f"{c['id']}: rows do not share a recorded commit, so the "
                f"same-commit classification policy cannot apply -- verdict must "
                f"be 'unclassifiable', not {c['verdict']!r}"
            )
        elif gap >= threshold:
            assert c["verdict"] == "platform variance", (
                f"{c['id']}: gap {gap} pp >= policy threshold {threshold} pp but "
                f"verdict is {c['verdict']!r}"
            )
        else:
            assert c["verdict"] == "agree", (
                f"{c['id']}: gap {gap} pp < policy threshold {threshold} pp but "
                f"verdict is {c['verdict']!r}"
            )
