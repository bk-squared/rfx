"""Issue #528 cross-check: the envelope -> gate multiplier now has exactly
ONE definition (``tests/_gate_policy.py``), and every gated crossval case
must consume it rather than restate ``1.5`` locally.

The guards here are NOT all equally load-bearing (adversarial review of
PR #539 flagged this explicitly -- be honest about it):

  * The source-grep tests below (``..._imports_shared_helper_not_a_local_literal``,
    ``..._imports_shared_multiplier_not_a_local_literal``) only reject the
    LITERAL string ``1.5`` being restated. They are a tripwire, not a
    guarantee: a plant using ``1.50``, an aliased name, or any OTHER value
    (e.g. a local ``MARGIN_CEIL = 3.0``) sails through them untouched.
  * The load-bearing guards are the numeric cross-derivations: the two
    falsifiers for the quantized-gate lanes
    (``test_mutating_the_shared_multiplier_...`` and
    ``test_monkeypatching_the_live_shared_multiplier_...``) and the
    from-outside numeric cross-check for the bounded-margin lanes
    (``test_margin_ceiling_lanes_are_bound_by_the_shared_multiplier_from_outside``).
    These recompute a case's gate/bound from the shared constant and the
    committed data independently of whatever the case's own file claims,
    so a coherent in-file plant (relax the local check AND re-pin the
    constant to fit under it) is still caught.
  * The import-route test
    (``test_crossval_script_import_route_resolves_the_repo_gate_policy_module``)
    guards a DIFFERENT failure mode entirely: not a relaxed multiplier, but
    a site-packages ``tests`` package (grcwa installs a non-namespace one)
    winning the ``import tests`` race and silently -- or loudly, see that
    test's docstring -- resolving to the wrong module.

No FDTD runs here -- pure-Python, replays committed fixtures and artifacts.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
GATE_POLICY = REPO / "tests" / "_gate_policy.py"

# The quantized-gate consumers: `gate = ceil(env * MULTIPLIER * quantum) /
# quantum`, hard-pinned in a test AND re-derived in the crossval script's
# own --write-fixture self-check.
_QUANTIZED_GATE_FILES = [
    REPO / "tests" / "crossval" / "test_wr90_iris_modematch_gates.py",
    REPO / "tests" / "crossval" / "test_rcs_mie_ka_sweep_gates.py",
    REPO / "tests" / "crossval" / "test_rcs_dielectric_sphere_mie_gates.py",
    REPO / "tests" / "crossval" / "test_wr90_iris_filter_gates.py",
    REPO / "validation" / "crossval" / "16_pec_sphere_mie_ka_sweep.py",
    REPO / "validation" / "crossval" / "17_dielectric_sphere_mie.py",
    REPO / "validation" / "crossval" / "18_wr90_iris_modematch.py",
    REPO / "validation" / "crossval" / "19_wr90_iris_filter_aghanim.py",
    # #576 review F5. This lane is in the tripwire list but NOT in _REAL_CASES
    # below, and that is a real coverage gap, not an oversight: the mutation
    # falsifiers discover cases from `tests/fixtures/**/fixture.json` files
    # carrying a `gates` dict with `<prefix>_measured_envelope_<suffix>` /
    # `<prefix>_gate_<suffix>` pairs, and this lane's artifact is a flat
    # comparison JSON keyed `max_mag_abs_tol` / `mean_mag_abs_tol` with a
    # per-pair list — a different schema with its own consumers. It also gates
    # at quantum 1000 (milli-scale residual) where _QUANTUM_BY_SUFFIX maps every
    # `abs` key to 100. So the multiplier-mutation coverage here rests on the
    # test file's own derived-not-pinned tolerances, which is weaker than the
    # discovered lanes' from-outside check.
    REPO / "tests" / "crossval" / "test_waveguide_nu_broad_e4_comparison_gates.py",
    (REPO / "scripts" / "diagnostics"
     / "build_waveguide_wr90_nu_flux_broad_e4_comparison.py"),
    # #574 promotion. Same shape and the same coverage gap as the E4 pair above:
    # a flat envelope JSON keyed `max_mag_abs_tol`, gating at quantum 1000, so
    # the fixture-glob discovery in _REAL_CASES does not reach it either. Its
    # from-outside check is its own lane's
    # test_envelope_is_recomputed_from_the_artifact_and_capped_from_outside,
    # which re-derives MAX_TOL through gate_from_envelope AND caps the measured
    # envelope with a literal pinned outside the artifact (blind below 1.203x,
    # measured in that file).
    REPO / "tests" / "crossval" / "test_waveguide_nu_broad_e5_envelope_gates.py",
    (REPO / "scripts" / "diagnostics"
     / "build_waveguide_wr90_nu_flux_broad_e5_envelope.py"),
]

# The bounded-margin consumers: a PINNED module constant checked against
# [worst_measured, worst_measured * MULTIPLIER] -- a different formula shape
# (see tests/_gate_policy.py docstring), so these import the multiplier
# directly rather than calling gate_from_envelope.
_MARGIN_CEIL_FILES = [
    REPO / "tests" / "crossval" / "test_waveguide_broad_e5_tolerance_envelope.py",
    REPO / "tests" / "crossval" / "test_waveguide_broad_e5_phase_tolerance_envelope.py",
    REPO / "tests" / "test_waveguide_group_delay_tolerance_envelope.py",
]

_CROSSVAL_SCRIPTS = [
    REPO / "validation" / "crossval" / "16_pec_sphere_mie_ka_sweep.py",
    REPO / "validation" / "crossval" / "17_dielectric_sphere_mie.py",
    REPO / "validation" / "crossval" / "18_wr90_iris_modematch.py",
    REPO / "validation" / "crossval" / "19_wr90_iris_filter_aghanim.py",
]

_ENVELOPE_KEY_RE = re.compile(
    r"^(?P<prefix>.+)_measured_envelope_(?P<suffix>abs|db|mhz)$")
_QUANTUM_BY_SUFFIX = {"abs": 100, "db": 10, "mhz": 1}


def _discover_real_cases() -> list[tuple[str, tuple[str, str], tuple[str, str], int]]:
    """Pair every ``<prefix>_measured_envelope_<abs|db>`` gates-key with its
    ``<prefix>_gate_<abs|db>`` sibling across every committed
    ``tests/fixtures/**/fixture.json`` -- derived from the glob rather than
    hand-maintained (#528 review MEDIUM 2), so a new quantized-gate case
    is picked up automatically instead of needing this file edited too --
    modulo the suffix map: #499's case 19 (anticipated here before it
    landed) gates in integer MHz, which required adding ``mhz`` -> 1 to
    ``_QUANTUM_BY_SUFFIX`` in the same PR; a future case reusing an
    existing suffix costs nothing.

    Descriptive, not authoritative: as of this writing this discovers
    exactly 6 cases -- wr90_iris_modematch {fine, richardson},
    rcs_mie_ka_sweep {coarse, fine}, rcs_dielectric_sphere_mie {coarse},
    wr90_iris_filter {f0}.
    Four OTHER committed fixture.json files exist
    (rcs280_reference_subtraction, rcs_cube_bem, rcs_sphere_mie,
    rcs_sphere_three_way) and are correctly excluded: none has a top-level
    ``gates`` dict with a matching envelope/gate key pair. Dielectric's
    ``fine_rung_witness_envelope_db`` is also correctly excluded -- it has
    no ``fine_gate_db`` sibling (that rung is reported, never gated).
    """
    cases: list[tuple[str, tuple[str, str], tuple[str, str], int]] = []
    for fixture_path in sorted(REPO.glob("tests/fixtures/**/fixture.json")):
        data = json.loads(fixture_path.read_text())
        gates = data.get("gates")
        if not isinstance(gates, dict):
            continue
        rel = fixture_path.relative_to(REPO).as_posix()
        for key in gates:
            m = _ENVELOPE_KEY_RE.match(key)
            if not m:
                continue
            gate_key = f"{m.group('prefix')}_gate_{m.group('suffix')}"
            if gate_key not in gates:
                continue
            cases.append((rel, ("gates", key), ("gates", gate_key),
                          _QUANTUM_BY_SUFFIX[m.group("suffix")]))
    return cases


_REAL_CASES = _discover_real_cases()


def _load(rel_path: str) -> dict:
    with open(REPO / rel_path) as f:
        return json.load(f)


def _dig(d: dict, path: tuple[str, ...]):
    for k in path:
        d = d[k]
    return d


# ---------------------------------------------------------------------------
# Independent re-derivation of `worst` for the bounded-margin lanes, from the
# SAME committed artifacts each lane itself reads -- duplicated here on
# purpose (not imported from the lane modules) so a coordinated edit to a
# lane's own worst-computation cannot also blind this outside check.
# ---------------------------------------------------------------------------

_BROAD_E5_FIXTURES = REPO / "tests" / "fixtures" / "waveguide_broad_e5"
_GROUP_DELAY_ENVELOPE = (
    REPO / "tests/fixtures/waveguide_group_delay/wr340_near_cutoff_group_delay_envelope.json")

sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_band_broad_e5_envelope import MAX_TOL  # type: ignore  # noqa: E402
from build_waveguide_band_broad_e5_phase_envelope import MAX_PHASE_TOL_DEG  # type: ignore  # noqa: E402
from build_waveguide_group_delay_envelope import MAX_GROUP_DELAY_TOL_NS  # type: ignore  # noqa: E402


def _worst_broad_e5_magnitude_diff() -> float:
    diffs = [float(c["max_mag_abs_diff"])
             for f in sorted(_BROAD_E5_FIXTURES.glob("waveguide_*_broad_e5_envelope.json"))
             for c in json.loads(f.read_text())["cases"]]
    assert diffs, "no committed broad-E5 magnitude cases found"
    return max(diffs)


def _worst_broad_e5_phase_diff() -> float:
    diffs = [float(c["max_phase_diff_deg"])
             for f in sorted(_BROAD_E5_FIXTURES.glob("waveguide_*_broad_e5_phase_envelope.json"))
             for c in json.loads(f.read_text())["cases"]]
    assert diffs, "no committed broad-E5 phase cases found"
    return max(diffs)


def _worst_group_delay_diff_ns() -> float:
    env = json.loads(_GROUP_DELAY_ENVELOPE.read_text())
    meas = np.array(env["tau_g_measured_ns"])
    ana_stencil = np.array(env["tau_g_analytic_via_stencil_ns"])
    idx = env["interior_indices"]
    return float(np.abs(meas[idx] - ana_stencil[idx]).max())


def test_gate_policy_module_defines_exactly_one_multiplier():
    from tests._gate_policy import ENVELOPE_GATE_MULTIPLIER, gate_from_envelope
    assert ENVELOPE_GATE_MULTIPLIER == 1.5
    assert callable(gate_from_envelope)


def test_real_cases_are_discovered_from_the_fixture_glob_not_hand_maintained():
    """#528 review MEDIUM 2: an empty or broken glob must not silently pass
    every other assertion in this file (they'd vacuously succeed over zero
    cases) -- assert a floor AND that today's 6 known cases are all in it."""
    assert len(_REAL_CASES) >= 6, (
        f"only {len(_REAL_CASES)} real gated cases discovered via the "
        f"fixture glob -- an empty or broken glob would silently pass "
        f"every other case-driven assertion in this file"
    )
    discovered = {(rel, env_key) for rel, env_key, _, _ in _REAL_CASES}
    expected = {
        ("tests/fixtures/wr90_iris_modematch/fixture.json",
         ("gates", "fine_measured_envelope_abs")),
        ("tests/fixtures/wr90_iris_modematch/fixture.json",
         ("gates", "richardson_measured_envelope_abs")),
        ("tests/fixtures/rcs_mie_ka_sweep/fixture.json",
         ("gates", "coarse_measured_envelope_db")),
        ("tests/fixtures/rcs_mie_ka_sweep/fixture.json",
         ("gates", "fine_measured_envelope_db")),
        ("tests/fixtures/rcs_dielectric_sphere_mie/fixture.json",
         ("gates", "coarse_measured_envelope_db")),
        ("tests/fixtures/wr90_iris_filter/fixture.json",
         ("gates", "f0_measured_envelope_mhz")),
    }
    assert expected <= discovered


@pytest.mark.parametrize(
    "path", _QUANTIZED_GATE_FILES, ids=[p.name for p in _QUANTIZED_GATE_FILES])
def test_quantized_gate_case_imports_shared_helper_not_a_local_literal(path):
    """Tripwire only (see module docstring): rejects the literal `1.5`
    restated as a `* 1.5 *` gate-derivation expression. Does not, and
    cannot by grep alone, rule out a different multiplier value -- that is
    what the falsifier tests below are for."""
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
    """Tripwire only (see module docstring): rejects `MARGIN_CEIL = 1.5`
    restated as a local literal. A plant of `MARGIN_CEIL = 3.0` (any OTHER
    value) passes this grep untouched -- caught instead by
    test_margin_ceiling_lanes_are_bound_by_the_shared_multiplier_from_outside,
    which does not read MARGIN_CEIL at all."""
    src = path.read_text(encoding="utf-8")
    assert "_gate_policy import" in src and "ENVELOPE_GATE_MULTIPLIER" in src, (
        f"{path.name} does not import the shared ENVELOPE_GATE_MULTIPLIER"
    )
    assert re.search(r"^MARGIN_CEIL\s*=\s*1\.5", src, re.MULTILINE) is None, (
        f"{path.name} still assigns MARGIN_CEIL = 1.5 as a local literal "
        f"instead of importing the shared constant"
    )


def test_margin_ceiling_lanes_are_bound_by_the_shared_multiplier_from_outside():
    """#528 review MEDIUM 1 -- the load-bearing guard for the three
    bounded-margin lanes. Each lane's OWN test file checks
    `worst <= PINNED <= worst * MARGIN_CEIL` using its OWN `MARGIN_CEIL`
    name -- which a plant can locally repoint to any value with the source
    grep above none the wiser. This test re-derives `worst` independently
    (see the `_worst_*` helpers above, which read the same committed
    artifacts the lanes read but do not call into the lane modules) and
    checks the SAME bound using `ENVELOPE_GATE_MULTIPLIER` imported fresh
    from `tests._gate_policy` -- so the bound is enforced regardless of
    what any lane file's own `MARGIN_CEIL` name is bound to.

    Verified (all three currently well inside the shared 1.5x ceiling):
    magnitude 0.05 / 0.0414 = 1.21, phase 15.0 / 11.99 = 1.25, group delay
    0.042 / 0.0320 = 1.31.

    Reviewer's plant, manually re-run against this exact check and
    confirmed to red (see PR #539 review response / commit body for the
    numbers): planting a local `MARGIN_CEIL = 3.0` in
    test_waveguide_broad_e5_tolerance_envelope.py and re-pinning
    `MAX_TOL` to 0.09 (0.09 / 0.0414 = 2.17 -- passes that file's own
    3.0x-ceiling check) still fails THIS assertion, because
    `0.09 > 0.0414 * ENVELOPE_GATE_MULTIPLIER (0.0621)` regardless of the
    planted local ceiling.
    """
    from tests._gate_policy import ENVELOPE_GATE_MULTIPLIER

    lanes = (
        (_worst_broad_e5_magnitude_diff(), MAX_TOL, "broad-E5 magnitude"),
        (_worst_broad_e5_phase_diff(), MAX_PHASE_TOL_DEG, "broad-E5 phase"),
        (_worst_group_delay_diff_ns(), MAX_GROUP_DELAY_TOL_NS,
         "near-cutoff group delay"),
    )
    for worst, pinned, label in lanes:
        assert worst <= pinned, (
            label, worst, pinned,
            "pinned constant is below the independently-recomputed worst "
            "case -- would fail a validated case")
        assert pinned <= worst * ENVELOPE_GATE_MULTIPLIER, (
            label, worst, pinned, ENVELOPE_GATE_MULTIPLIER,
            "pinned constant exceeds worst * ENVELOPE_GATE_MULTIPLIER -- "
            "slack beyond the shared repo-wide ceiling, regardless of "
            "whatever a local MARGIN_CEIL name in the lane's own file "
            "claims")


def test_margin_ceiling_plant_is_caught_by_the_from_outside_cross_check():
    """Permanent regression coverage for the reviewer's plant described
    above, using the REAL committed broad-E5 magnitude worst case (no file
    is touched -- this only proves the arithmetic mechanism): a re-pin
    that fits under a locally-planted 3.0x ceiling still violates the
    shared 1.5x one."""
    from tests._gate_policy import ENVELOPE_GATE_MULTIPLIER

    worst = _worst_broad_e5_magnitude_diff()
    planted_local_ceiling = 3.0
    re_pinned = 0.09
    assert re_pinned <= worst * planted_local_ceiling, (
        "test setup invalid: the planted re-pin must itself pass the "
        "attacker's own local (widened) ceiling")
    assert re_pinned > worst * ENVELOPE_GATE_MULTIPLIER, (
        "test setup invalid: the planted re-pin must violate the shared "
        "ceiling for this to demonstrate anything")
    with pytest.raises(AssertionError):
        assert re_pinned <= worst * ENVELOPE_GATE_MULTIPLIER


def test_mutating_the_shared_multiplier_moves_every_gated_case_and_reverts():
    """Falsifier 1 of 2 (source-mutation shape): reproduce the EXACT
    adversarial edit named in the issue (a source-level find-replace of
    the single multiplier definition, as if someone had actually edited
    tests/_gate_policy.py on disk) and show every quantized-gate case's
    derived value moves together, then confirm reverting restores every
    case's frozen CI-pinned gate exactly. This execs mutated SOURCE TEXT
    into throwaway module objects -- it never touches any already-imported
    module or the file on disk. See
    test_monkeypatching_the_live_shared_multiplier_moves_every_gated_case
    for the complementary live-module falsifier.
    """
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

    assert _REAL_CASES, "no real cases discovered -- see the glob-discovery test"
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


def test_monkeypatching_the_live_shared_multiplier_moves_every_gated_case():
    """Falsifier 2 of 2 (live-module shape, #528 review L1): patches
    `ENVELOPE_GATE_MULTIPLIER` on the ALREADY-IMPORTED `tests._gate_policy`
    module -- the actual module object every consumer's
    `from tests._gate_policy import gate_from_envelope` is bound to in this
    process -- rather than exec'ing a source copy. This is the cheaper and
    more direct proof that consumers read the constant at CALL time, not
    at import time: `gate_from_envelope` is not re-imported anywhere below,
    only called again after the patch.

    Concretely verified: the dielectric-sphere coarse case moves 6.3 dB ->
    12.6 dB (measured envelope 4.181 dB, multiplier 1.5 -> 3.0).

    Uses `pytest.MonkeyPatch.context()` explicitly (rather than the
    function-scoped `monkeypatch` fixture) so the revert is demonstrated
    INSIDE this test, not merely implied by fixture teardown.
    """
    from tests import _gate_policy

    assert _REAL_CASES, "no real cases discovered -- see the glob-discovery test"
    baseline = {}
    for fixture_path, env_key, gate_key, quantum in _REAL_CASES:
        data = _load(fixture_path)
        env = float(_dig(data, env_key))
        pinned_gate = float(_dig(data, gate_key))
        baseline_gate = _gate_policy.gate_from_envelope(env, quantum=quantum)
        assert baseline_gate == pytest.approx(pinned_gate, abs=1e-9)
        baseline[(fixture_path, env_key)] = baseline_gate

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_gate_policy, "ENVELOPE_GATE_MULTIPLIER", 3.0)
        assert _gate_policy.ENVELOPE_GATE_MULTIPLIER == 3.0
        for fixture_path, env_key, gate_key, quantum in _REAL_CASES:
            data = _load(fixture_path)
            env = float(_dig(data, env_key))
            mutated_gate = _gate_policy.gate_from_envelope(env, quantum=quantum)
            assert mutated_gate > baseline[(fixture_path, env_key)], (
                fixture_path, env_key, baseline[(fixture_path, env_key)],
                mutated_gate)

    # REVERT (outside the context manager): the live module's constant and
    # every case's re-derived gate are back to the pinned values.
    assert _gate_policy.ENVELOPE_GATE_MULTIPLIER == 1.5
    for fixture_path, env_key, gate_key, quantum in _REAL_CASES:
        data = _load(fixture_path)
        env = float(_dig(data, env_key))
        pinned_gate = float(_dig(data, gate_key))
        reverted_gate = _gate_policy.gate_from_envelope(env, quantum=quantum)
        assert reverted_gate == pytest.approx(pinned_gate, abs=1e-9)


_IMPORT_ROUTE_PROBE = """
import runpy, sys, os
script = {script!r}
# Reproduce the exact scenario a bare `python script.py` invoked from an
# arbitrary cwd sets up: sys.path[0] = the SCRIPT'S OWN directory, not the
# repo root and not cwd -- the harshest ordering, since nothing points at
# the repo until the script's own sys.path.insert(0, _REPO_ROOT) line runs.
sys.path.insert(0, os.path.dirname(script))
ns = runpy.run_path(script, run_name="not_main")
gfe = ns["gate_from_envelope"]
print(gfe.__globals__.get("__file__", ""))
"""


@pytest.mark.parametrize(
    "script", _CROSSVAL_SCRIPTS, ids=[p.name for p in _CROSSVAL_SCRIPTS])
def test_crossval_script_import_route_resolves_the_repo_gate_policy_module(script):
    """#528 review L3: guards a DIFFERENT failure mode than the rest of
    this file -- not a relaxed multiplier, an import-route collision.
    `/root/.local/lib/python3.10/site-packages/tests/` is a REGULAR
    (non-namespace) package installed by `grcwa` (used by the Floquet-RCWA
    referee lane). If anything resolves `import tests` before a crossval
    script's own `sys.path.insert(0, _REPO_ROOT)` line runs, that binding
    is cached in `sys.modules['tests']` for the rest of the process and the
    script's own insert cannot undo it -- confirmed by hand: pre-importing
    the site-packages `tests` package and then running one of these
    scripts' prologue raises `ModuleNotFoundError: No module named
    'tests._gate_policy'` (loud, not silently wrong, but still a crash).

    This test reproduces the reviewer's exact method -- `runpy.run_path`
    with `run_name != "__main__"` (so the heavy FDTD `__main__` block never
    executes) and `sys.path[0]` set to the script's own directory -- in a
    FRESH SUBPROCESS per script, not in-process. In-process would be
    non-discriminating: this pytest session has already imported the
    correct `tests._gate_policy` (to collect this very file), so
    `sys.modules['tests']` is already correctly bound before this test
    runs regardless of whether the script's OWN sys.path-insert logic is
    even correct -- an in-process check could not tell a working script
    from a broken one. The subprocess starts with an empty
    `sys.modules` and a `cwd` outside the repo (so `''`/cwd cannot
    accidentally save it either), so passing here is real evidence the
    script's own `sys.path.insert(0, _REPO_ROOT)` -> import ordering is
    what resolves `tests` correctly, not an artifact of the test
    environment.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_ROUTE_PROBE.format(script=str(script))],
        cwd=str(REPO / "validation"),
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"{script.name} import-route probe crashed "
        f"(this is exactly the ModuleNotFoundError shape a site-packages "
        f"`tests` shadow produces):\n{proc.stdout}\n{proc.stderr}"
    )
    resolved_file = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    assert resolved_file == str(GATE_POLICY), (
        f"{script.name} resolved gate_from_envelope to {resolved_file!r}, "
        f"not this repo's {GATE_POLICY} -- a shadowing `tests` package won "
        f"the import race"
    )
