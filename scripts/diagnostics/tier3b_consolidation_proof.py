#!/usr/bin/env python
"""Tier-3b test-corpus consolidation: the mechanical nothing-lost proof.

Companion of ``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``.
Given four ``pytest --collect-only -q`` dumps (default lane and full
``-m "gpu or not gpu"`` inventory, BEFORE = tier-4b tip, AFTER = this tip)
and the git ref of the BEFORE tip, it checks:

1. every BEFORE node id maps to exactly one AFTER node id that exists in the
   AFTER collection (``old_path::id -> new_path::id`` through FILE_MAP, test
   id unchanged, plus the one recorded collapse), and every AFTER id is
   mapped from at least one BEFORE id; reports N, M, D, K and N == M + D + K;
2. assertion-statement counts per merged file >= the sum over the files it
   absorbed (counted at the BEFORE ref via ``git show``);
3. ``.test_durations``: every key under an absorbed path rewritten to the new
   node id, none dropped, values identical.

Exit status is non-zero if any check fails. No pytest is run here; it only
reads the dumps, so run it from the repository root of the AFTER tip.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

_ABSORBED = {
    "tests/unit/boundaries/test_cpml_material_aware.py": [
        "tests/unit/nonuniform/test_nonuniform_cpml_dielectric.py",
        "tests/unit/runners/test_distributed_cpml_dielectric.py",
        "tests/unit/runners/test_distributed_nu_cpml_dielectric.py",
        "tests/unit/runners/test_distributed_pmap_cpml_dielectric.py",
        "tests/unit/runners/test_vmap_cpml_dielectric.py",
        "tests/unit/sparams/test_lumped_wire_sparam_cpml_dielectric.py",
        "tests/unit/subgrid/test_subgrid_cpml_dielectric.py",
    ],
    "tests/unit/sparams/test_settling_witness.py": [
        "tests/unit/sparams/test_msl_settling_witness.py",
        "tests/unit/sparams/test_settling_witness_enforcement.py",
        "tests/unit/sparams/test_waveguide_settling_witness.py",
    ],
    "tests/unit/subgrid/test_sbp_sat.py": [
        "tests/unit/subgrid/test_sbp_sat_1d.py",
        "tests/unit/subgrid/test_sbp_sat_2d.py",
        "tests/unit/subgrid/test_sbp_sat_3d.py",
        "tests/unit/subgrid/test_sbp_sat_alpha.py",
        "tests/unit/subgrid/test_sbp_sat_jit.py",
    ],
    "tests/crossval/test_waveguide_broad_e5.py": [
        "tests/crossval/test_waveguide_broad_e5_envelope_gates.py",
        "tests/crossval/test_waveguide_broad_e5_live_anchor.py",
        "tests/crossval/test_waveguide_broad_e5_phase_gates.py",
        "tests/crossval/test_waveguide_broad_e5_phase_tolerance_envelope.py",
        "tests/crossval/test_waveguide_broad_e5_tolerance_envelope.py",
    ],
    "tests/unit/sparams/test_coax_two_port_smatrix.py": [
        "tests/unit/sparams/test_coax_two_port_fdtd.py",
        "tests/unit/sparams/test_coax_two_port_solve.py",
    ],
    "tests/unit/sparams/test_coaxial_line_reflection.py": [
        "tests/unit/sparams/test_coaxial_line_calibration.py",
        "tests/unit/sparams/test_coaxial_line_extraction.py",
    ],
    "tests/unit/preflight/test_preflight_absorber.py": [
        "tests/unit/preflight/test_preflight_absorber_frame.py",
        "tests/unit/preflight/test_preflight_geometry_absorber_aggregation.py",
        "tests/unit/preflight/test_preflight_dispersive_pole_at_absorber.py",
    ],
    "tests/unit/preflight/test_preflight_rasterization.py": [
        "tests/unit/preflight/test_preflight_campaign_statics.py",
        "tests/unit/preflight/test_preflight_graded_rasterization.py",
        "tests/unit/preflight/test_preflight_thin_metal_nu.py",
    ],
    "tests/unit/preflight/test_preflight_guards.py": [
        "tests/unit/preflight/test_preflight_physics_thresholds.py",
        "tests/unit/preflight/test_preflight_false_positives.py",
        "tests/unit/preflight/test_preflight_structured_and_guards.py",
        "tests/unit/preflight/test_preflight_tfsf_lumped.py",
    ],
    "tests/unit/materials/test_sheet_impedance.py": [
        "tests/unit/materials/test_leontovich_sheet_identity.py",
        "tests/unit/materials/test_sheet_impedance_operator.py",
        "tests/unit/materials/test_sheet_lane_fences.py",
        "tests/unit/materials/test_sheet_stacked_adjacent_gap.py",
        "tests/unit/materials/test_thin_conductor_nonbox_sheet.py",
    ],
}
FILE_MAP = {old: new for new, olds in _ABSORBED.items() for old in olds}
# Plain moves (no merge): the four cv22/cv23 tests that landed on main after
# the tier-4b plan and were still at the top level; moved into tests/crossval/
# beside test_cv24_nu_cavity_gates.py.
_MOVED = {f"tests/{b}": f"tests/crossval/{b}" for b in (
    "test_cv22_dispersive_eps_mapping.py", "test_cv22_dispersive_slab_gates.py",
    "test_cv23_lossy_eps_mapping.py", "test_cv23_lossy_slab_gates.py")}
FILE_MAP.update(_MOVED)

_GP = ("tests/contracts/test_gate_policy_is_shared.py::"
       "test_margin_ceil_case_imports_shared_multiplier_not_a_local_literal")
# The one node-id rewrite that is not a plain path substitution: the contract
# case parametrised over _MARGIN_CEIL_FILES follows the merged file.
NODE_MAP = {f"{_GP}[test_waveguide_broad_e5_tolerance_envelope.py]": f"{_GP}[test_waveguide_broad_e5.py]"}
# The one BEFORE id that collapsed into an AFTER id already claimed above
# (same function, same file after the merge) — counted as D.
COLLAPSED = {f"{_GP}[test_waveguide_broad_e5_phase_tolerance_envelope.py]": f"{_GP}[test_waveguide_broad_e5.py]"}

ASSERT_RX = re.compile(r"^\s*assert\b|pytest\.raises\(|pytest\.warns\(|np\.testing\.assert_|npt\.assert_")


def _ids(path: Path) -> list[str]:
    return [l.strip() for l in path.read_text().splitlines() if "::" in l and l.startswith("tests/")]


def _map(before_id: str) -> str:
    if before_id in NODE_MAP:
        return NODE_MAP[before_id]
    if before_id in COLLAPSED:
        return COLLAPSED[before_id]
    path, rest = before_id.split("::", 1)
    return FILE_MAP.get(path, path) + "::" + rest


def check_lane(name: str, before: list[str], after: list[str]) -> bool:
    assert len(set(before)) == len(before) and len(set(after)) == len(after)
    after_set = set(after)
    mapping = {b: _map(b) for b in before}
    missing = [(b, a) for b, a in mapping.items() if a not in after_set]
    reverse = collections.Counter(mapping.values())
    unmapped = [a for a in after if reverse[a] == 0]
    multi = {a: c for a, c in reverse.items() if c > 1 and a not in COLLAPSED.values()}
    d = sum(1 for b in before if b in COLLAPSED)
    k = 0
    ok = not missing and not unmapped and not multi and len(before) == len(after) + d + k
    print(f"[{name}] before N={len(before)} after M={len(after)} deleted D={d} "
          f"moved-to-self-check K={k} -> N == M + D + K: {len(before) == len(after) + d + k}; "
          f"missing={len(missing)} unmapped_after={len(unmapped)} multi_mapped={len(multi)}")
    for x in missing[:20]:
        print("   MISSING", x)
    for x in unmapped[:20]:
        print("   UNMAPPED", x)
    return ok


def _count_asserts(text: str) -> int:
    return sum(1 for l in text.splitlines() if ASSERT_RX.search(l))


def check_assertions(base: str) -> bool:
    ok = True
    for new, olds in sorted(_ABSORBED.items()):
        before = 0
        for old in olds:
            before += _count_asserts(subprocess.check_output(["git", "show", f"{base}:{old}"], text=True))
        after = _count_asserts(Path(new).read_text())
        flag = "" if after >= before else "   <-- DROPPED"
        print(f"[asserts] {new}: after {after} >= before {before}{flag}")
        ok &= after >= before
    return ok


def check_durations(base: str) -> bool:
    before = json.loads(subprocess.check_output(["git", "show", f"{base}:.test_durations"], text=True))
    after = json.loads(Path(".test_durations").read_text())
    expected = {}
    for k, v in before.items():
        nk = NODE_MAP.get(k)
        if nk is None and "::" in k:
            p, rest = k.split("::", 1)
            nk = FILE_MAP.get(p, p) + "::" + rest
        expected[nk or k] = v
    # A BEFORE key whose node id collapsed into an already-present AFTER id
    # (COLLAPSED) would be a dead key after the merge; it is removed, and the
    # removal is the only permitted drop.
    collapsed_keys = sorted(k for k in before if k in COLLAPSED)
    for k in collapsed_keys:
        expected.pop(k, None)
    dropped = sorted(set(expected) - set(after))
    extra = sorted(set(after) - set(expected))
    same = all(after.get(k) == v for k, v in expected.items())
    print(f"[durations] keys before={len(before)} after={len(after)} "
          f"remapped={sum(1 for k in before if k not in after and k not in collapsed_keys)} "
          f"collapsed_removed={len(collapsed_keys)} dropped={len(dropped)} extra={len(extra)} values_identical={same}")
    return not dropped and not extra and same and len(before) == len(after) + len(collapsed_keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="3bafe2f", help="git ref of the BEFORE tip (tier-4b)")
    ap.add_argument("--before-all", required=True, type=Path)
    ap.add_argument("--after-all", required=True, type=Path)
    ap.add_argument("--before-default", type=Path)
    ap.add_argument("--after-default", type=Path)
    a = ap.parse_args()
    ok = check_lane("full inventory", _ids(a.before_all), _ids(a.after_all))
    if a.before_default and a.after_default:
        ok &= check_lane("default lane", _ids(a.before_default), _ids(a.after_default))
    ok &= check_assertions(a.base)
    ok &= check_durations(a.base)
    print("PROOF HOLDS" if ok else "PROOF FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
