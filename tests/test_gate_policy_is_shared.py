"""Issue #528 cross-check: the envelope -> gate multiplier now has exactly
ONE definition (``tests/_gate_policy.py``), and every gated crossval case
must consume it rather than restate ``1.5`` locally.

Two properties are checked:

  1. VISIBILITY -- no consumer file carries a local envelope-multiplier
     literal any more; each imports ``tests._gate_policy`` instead. A
     coherent per-case edit (hard pin + derived relation, both in one file)
     can no longer relax the multiplier without touching a shared,
     reviewer-visible object.
  2. THE FALSIFIER -- a source-level find-replace of the ONE multiplier
     definition (the exact class of edit an adversarial review of PR #499
     caught doubling one case's gate while every existing guard stayed
     green) moves every quantized-gate case's derived value together, and
     reverting reproduces every case's frozen CI-pinned gate bit-for-bit.

No FDTD runs here -- pure-Python, replays committed fixtures.
"""
from __future__ import annotations

import json
import re
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATE_POLICY = REPO / "tests" / "_gate_policy.py"

# The quantized-gate consumers: `gate = ceil(env * MULTIPLIER * quantum) /
# quantum`, hard-pinned in a test AND re-derived in the crossval script's
# own --write-fixture self-check.
_QUANTIZED_GATE_FILES = [
    REPO / "tests" / "test_wr90_iris_modematch_gates.py",
    REPO / "tests" / "test_rcs_mie_ka_sweep_gates.py",
    REPO / "tests" / "test_rcs_dielectric_sphere_mie_gates.py",
    REPO / "validation" / "crossval" / "16_pec_sphere_mie_ka_sweep.py",
    REPO / "validation" / "crossval" / "17_dielectric_sphere_mie.py",
    REPO / "validation" / "crossval" / "18_wr90_iris_modematch.py",
]

# The bounded-margin consumers: a PINNED module constant checked against
# [worst_measured, worst_measured * MULTIPLIER] -- a different formula shape
# (see tests/_gate_policy.py docstring), so these import the multiplier
# directly rather than calling gate_from_envelope.
_MARGIN_CEIL_FILES = [
    REPO / "tests" / "test_waveguide_broad_e5_tolerance_envelope.py",
    REPO / "tests" / "test_waveguide_broad_e5_phase_tolerance_envelope.py",
    REPO / "tests" / "test_waveguide_group_delay_tolerance_envelope.py",
]

# (fixture path, measured-envelope key path, CI-pinned-gate key path,
# quantum) for every quantized-gate case that exists in this repo today.
_REAL_CASES = [
    ("tests/fixtures/wr90_iris_modematch/fixture.json",
     ("gates", "fine_measured_envelope_abs"), ("gates", "fine_gate_abs"), 100),
    ("tests/fixtures/wr90_iris_modematch/fixture.json",
     ("gates", "richardson_measured_envelope_abs"), ("gates", "richardson_gate_abs"), 100),
    ("tests/fixtures/rcs_mie_ka_sweep/fixture.json",
     ("gates", "coarse_measured_envelope_db"), ("gates", "coarse_gate_db"), 10),
    ("tests/fixtures/rcs_mie_ka_sweep/fixture.json",
     ("gates", "fine_measured_envelope_db"), ("gates", "fine_gate_db"), 10),
    ("tests/fixtures/rcs_dielectric_sphere_mie/fixture.json",
     ("gates", "coarse_measured_envelope_db"), ("gates", "coarse_gate_db"), 10),
]


def _load(rel_path: str) -> dict:
    with open(REPO / rel_path) as f:
        return json.load(f)


def _dig(d: dict, path: tuple[str, ...]):
    for k in path:
        d = d[k]
    return d


def test_gate_policy_module_defines_exactly_one_multiplier():
    from tests._gate_policy import ENVELOPE_GATE_MULTIPLIER, gate_from_envelope
    assert ENVELOPE_GATE_MULTIPLIER == 1.5
    assert callable(gate_from_envelope)


@pytest.mark.parametrize(
    "path", _QUANTIZED_GATE_FILES, ids=[p.name for p in _QUANTIZED_GATE_FILES])
def test_quantized_gate_case_imports_shared_helper_not_a_local_literal(path):
    src = path.read_text(encoding="utf-8")
    assert "_gate_policy import" in src and "gate_from_envelope" in src, (
        f"{path.name} does not import the shared gate_from_envelope helper"
    )
    # The exact pattern removed from every case: `<expr> * 1.5 * <quantum>`.
    assert re.search(r"\*\s*1\.5\s*\*", src) is None, (
        f"{path.name} still carries a local `* 1.5 *` gate-derivation "
        f"literal instead of calling the shared helper"
    )


@pytest.mark.parametrize(
    "path", _MARGIN_CEIL_FILES, ids=[p.name for p in _MARGIN_CEIL_FILES])
def test_margin_ceil_case_imports_shared_multiplier_not_a_local_literal(path):
    src = path.read_text(encoding="utf-8")
    assert "_gate_policy import" in src and "ENVELOPE_GATE_MULTIPLIER" in src, (
        f"{path.name} does not import the shared ENVELOPE_GATE_MULTIPLIER"
    )
    assert re.search(r"^MARGIN_CEIL\s*=\s*1\.5", src, re.MULTILINE) is None, (
        f"{path.name} still assigns MARGIN_CEIL = 1.5 as a local literal "
        f"instead of importing the shared constant"
    )


def test_mutating_the_shared_multiplier_moves_every_gated_case_and_reverts():
    """The falsifier: reproduce the EXACT adversarial edit named in the
    issue (a source-level find-replace of the single multiplier
    definition) and show every quantized-gate case's derived value moves
    together, then confirm reverting restores every case's frozen
    CI-pinned gate exactly."""
    original_src = GATE_POLICY.read_text(encoding="utf-8")
    needle = "ENVELOPE_GATE_MULTIPLIER: float = 1.5"
    assert needle in original_src, (
        "expected literal not found -- tests/_gate_policy.py's definition "
        "changed shape; update this falsifier to match"
    )
    mutated_src = original_src.replace(
        needle, "ENVELOPE_GATE_MULTIPLIER: float = 3.0")
    assert mutated_src != original_src

    def _exec_gate_policy(src_text: str) -> types.ModuleType:
        mod = types.ModuleType("gate_policy_under_test")
        exec(compile(src_text, str(GATE_POLICY), "exec"), mod.__dict__)
        return mod

    baseline_mod = _exec_gate_policy(original_src)
    mutated_mod = _exec_gate_policy(mutated_src)
    assert baseline_mod.ENVELOPE_GATE_MULTIPLIER == 1.5
    assert mutated_mod.ENVELOPE_GATE_MULTIPLIER == 3.0

    moved = []
    for fixture_path, env_key, gate_key, quantum in _REAL_CASES:
        data = _load(fixture_path)
        env = float(_dig(data, env_key))
        pinned_gate = float(_dig(data, gate_key))
        # CAUGHT (before the mutation): the shared helper reproduces the
        # frozen CI-pinned gate exactly -- behavior-preserving migration.
        baseline = baseline_mod.gate_from_envelope(env, quantum=quantum)
        assert baseline == pytest.approx(pinned_gate, abs=1e-9), (
            fixture_path, env_key, baseline, pinned_gate)
        # CAUGHT (the falsifier): the mutated multiplier widens this case's
        # derived gate too -- it did not need its own file touched.
        mutated = mutated_mod.gate_from_envelope(env, quantum=quantum)
        assert mutated > baseline, (fixture_path, env_key, baseline, mutated)
        moved.append((fixture_path, env_key, baseline, mutated))

    assert len(moved) == len(_REAL_CASES), "not every real gated case responded"

    # REVERT: re-deriving from the UNMODIFIED module reproduces every
    # case's pinned gate again, bit-for-bit -- nothing was left mutated
    # (this test only ever exec'd source text into throwaway module
    # objects; the file on disk was never touched).
    assert GATE_POLICY.read_text(encoding="utf-8") == original_src
    for fixture_path, env_key, gate_key, quantum in _REAL_CASES:
        data = _load(fixture_path)
        env = float(_dig(data, env_key))
        pinned_gate = float(_dig(data, gate_key))
        reverted = baseline_mod.gate_from_envelope(env, quantum=quantum)
        assert reverted == pytest.approx(pinned_gate, abs=1e-9)
