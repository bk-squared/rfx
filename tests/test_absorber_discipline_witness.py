"""GATED far-port absorber discipline across every committed broad-E5 envelope.

#496 ask 1 and 2. The depth was RECORDED in all six broad-E5 fixtures and read
by no test — `grep -rn setup_recipe tests/` returned only fixture JSONs — which
is how ten configurations shipped between 0.060 and 0.128 lambda_g under a
discipline whose floor is 0.5. A recorded number no test reads is decoration.
`scripts/diagnostics/i496_unitarity_witness_audit.py --all` already derived the
table on a clean checkout; this file is that derivation as an ASSERTION.

Modelled on the NU broad-E4 witness #576 landed
(``test_waveguide_nu_broad_e4_comparison_gates.py::
test_absorber_depth_witness_is_gated_not_just_recorded``), extended from one
lane to every committed one and given the escape hatch that lane did not need:

  1. the DISCIPLINE — absorber >= 0.5 lambda_g at the lowest measured frequency.
  2. RECOMPUTED, never restated: lambda_g comes from each artifact's own
     ``cutoff_te10_hz`` and each case's own ``dx_m``, so a fixture cannot pass by
     writing a flattering fraction next to a thin absorber. (That failure mode is
     not hypothetical here: until 287b281 the band producer copied a stale
     manifest header into ``setup_recipe.cpml_layers``, and a run at 0.75
     lambda_g recorded — and audited as — 0.060.)
  3. the PHYSICAL witness that the absorber worked: these are lossless slabs, so
     column power must be 1. A thin absorber leaks and pushes it over; a
     truncated record pulls it under. Both directions count.
  4. EXPLICIT ACCEPTANCE or nothing: a lane below the floor passes only if it
     carries an ``absorber_discipline`` block stating the decision and its
     evidence (282934b annotated five band lanes after measuring that
     regenerating them buys 1.10-1.20x for ~30x the wallclock). Silence is a
     failure. This is the part that keeps the check honest as new lanes land —
     a new fixture below the floor fails until someone writes down why.

Discovery is by glob, not a hand-maintained list, so a new broad-E5 lane is
picked up without editing this file (the lesson _gate_policy's `_REAL_CASES`
records). If the glob ever matches nothing, that is itself a failure.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
C0 = 299_792_458.0
FAR_PORT_LAMBDA_G_FRACTION_FLOOR = 0.5          # #496
# Lossless slabs: column power is 1. 1e-3 is the band the E4 witness uses for
# its PEC short; every committed lane here measures at or under 3.2e-3, so this
# is a real bound rather than a rubber stamp — see the per-lane note below.
UNITARITY_EXCESS_CEILING = 5e-3

ENVELOPES = sorted(
    REPO.glob("tests/fixtures/waveguide_*broad_e5/*_broad_e5_envelope.json"))


def _lam_g_low(env: dict) -> float:
    es = env["envelope_summary"]
    f_lo = float(es["freq_range_hz"][0])
    f_c = float(es["cutoff_te10_hz"])
    assert f_lo > f_c, f"band edge {f_lo} is below TE10 cutoff {f_c}"
    return (C0 / f_lo) / math.sqrt(1.0 - (f_c / f_lo) ** 2)


def _absorber_cells(env: dict, case: dict) -> int | None:
    """Per-case first: with a derived absorber the summary depth is wrong for
    most cases, and reading it would report a compliant lane as violating."""
    cells = case.get("cpml_layers")
    if cells is None:
        cells = (env["envelope_summary"].get("setup_recipe") or {}).get(
            "cpml_layers")
    return None if cells is None else int(cells)


def _case_dx(env: dict, case: dict) -> float | None:
    dx = case.get("dx_m")
    if dx is None:
        dx = (env["envelope_summary"].get("setup_recipe") or {}).get("base_dx_m")
    return None if dx is None else float(dx)


def test_the_glob_finds_the_committed_lanes() -> None:
    """A discovery-driven check that discovers nothing is a green light for
    everything."""
    assert len(ENVELOPES) >= 6, [p.name for p in ENVELOPES]


@pytest.mark.parametrize("path", ENVELOPES, ids=lambda p: p.stem)
def test_absorber_discipline_is_gated_not_just_recorded(path: Path) -> None:
    env = json.loads(path.read_text())
    lam_g = _lam_g_low(env)
    ann = env.get("absorber_discipline") or {}
    accepted = ann.get("status") == "below_floor_accepted"

    worst_frac = math.inf
    for case in env["cases"]:
        cells = _absorber_cells(env, case)
        dx = _case_dx(env, case)
        assert cells is not None, (
            f"{path.name} case {case['tag']}: no absorber depth recorded in "
            f"cases[].cpml_layers or setup_recipe — the discipline cannot be "
            f"checked at all, which is the #496 state this test exists to end")
        assert dx is not None, f"{path.name} case {case['tag']}: no dx recorded"
        worst_frac = min(worst_frac, cells * dx / lam_g)

    if worst_frac < FAR_PORT_LAMBDA_G_FRACTION_FLOOR:
        assert accepted, (
            f"{path.name}: absorber is {worst_frac:.3f} lambda_g, below the "
            f"{FAR_PORT_LAMBDA_G_FRACTION_FLOOR} far-port discipline (#496), and "
            f"the fixture carries no `absorber_discipline` acceptance. Either "
            f"fix the absorber or record WHY it is accepted, with the "
            f"measurement behind the decision — silence is what let ten "
            f"configurations ship below the floor")
        assert ann.get("decision"), (
            f"{path.name}: accepted below the floor with no stated decision")
        assert "probe" in ann, (
            f"{path.name}: accepted below the floor without saying whether the "
            f"attribution was PROBED — an unprobed lane must say so rather than "
            f"inherit a probed lane's conclusion")


@pytest.mark.parametrize("path", ENVELOPES, ids=lambda p: p.stem)
def test_passivity_witness_is_present_and_physical(path: Path) -> None:
    """The absorber fraction is only trustworthy to the extent the physical
    witness agrees. Absent is NOT passing: the WR-90 NU lane read `NOT MEASURED`
    for its whole history because its builder wrote no unitarity keys, and an
    earlier revision of the auditor rendered that absence as a perfect 0.0e+00
    (the #303 failure shape)."""
    env = json.loads(path.read_text())
    for case in env["cases"]:
        lo, hi = case.get("unitarity_min"), case.get("unitarity_max")
        assert lo is not None and hi is not None, (
            f"{path.name} case {case['tag']}: no passivity witness. These are "
            f"lossless slabs — column power is measurable and must be 1; a lane "
            f"without it cannot corroborate its own absorber (#496 ask 3)")
        excess = max(abs(1.0 - float(lo)), abs(float(hi) - 1.0))
        assert excess <= UNITARITY_EXCESS_CEILING, (
            f"{path.name} case {case['tag']}: column power departs from unity by "
            f"{excess:.2e} on a LOSSLESS structure — over-unity means the "
            f"absorber leaks, under-unity means the record is truncated, and "
            f"both are co-conditions (#576)")
