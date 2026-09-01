"""Fast, no-simulation regression tests for crossval gate LOGIC.

Covers ``validation/crossval/11_waveguide_port_wr90.py`` (per-freq band +
ceiling gate, issue #340) and ``validation/crossval/04_multilayer_fresnel.py``
(per-bin conservation ceiling + settling-tail witness, issue #341; the
fringe-resolved gate against the analytic reference and the pointwise gate that
puts Meep's NUMBERS in the verdict, issue #812).

Neither crossval script's actual FDTD gate runs in any automated CI workflow
(confirmed 2026-07-14: no ``.github/workflows/*.yml`` invokes
``scripts/run_crossval_cpu.py`` or either script directly;
``tests/contracts/test_crossval_manifest_contract.py`` only unit-tests the runner's
classification logic against synthetic/mocked subprocess results, and the
manifest's structural self-consistency — never the scripts themselves). This
file pins the GATE MATH against synthetic arrays so a future edit to either
script's ceiling/tail logic reds in the fast CI lane, without paying for a
full (and, for cv04, optional-Meep-dependent) FDTD run.

cv11 is properly guarded (``if __name__ == "__main__":`` at
validation/crossval/11_waveguide_port_wr90.py:837) and its gate helper is a
pure function, so it is imported directly here. cv04 runs its FDTD and gate
computation entirely at MODULE level with no ``__main__`` guard (confirmed
2026-07-14) — importing it would execute the full simulation, so its
ceiling/tail logic and constants are replicated inline instead, with exact
source-line citations.

The #812 fringe gate is NOT replicated: it lives in
``validation/crossval/comparators/fringe_gate.py`` as pure functions precisely
so the falsifiers below run the same code the crossval script runs. Replicated
logic can drift from the gate it claims to pin; imported logic cannot.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CROSSVAL_DIR = REPO_ROOT / "validation" / "crossval"


def _load_cv11():
    """Import cv11 as a module without executing its __main__ block."""
    path = CROSSVAL_DIR / "11_waveguide_port_wr90.py"
    spec = importlib.util.spec_from_file_location("_cv11_gate_logic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cv11_per_freq_band_check_rejects_single_bin_spike():
    """A single-bin 1.5 spike in an otherwise all-ones |S| array must FAIL
    the per-freq band gate (mirrors cv11's own selftest, issue #340)."""
    cv11 = _load_cv11()
    f_fake = np.linspace(8.2e9, 12.4e9, 21)
    spike = np.ones(21)
    spike[10] = 1.5
    assert not cv11.per_freq_band_check(
        "test-spike", f_fake, spike, 0.93, 1.07, ceiling=1.05,
    )


def test_cv11_per_freq_band_check_accepts_healthy_curve():
    """An all-ones |S| array (within the band) must PASS."""
    cv11 = _load_cv11()
    f_fake = np.linspace(8.2e9, 12.4e9, 21)
    assert cv11.per_freq_band_check(
        "test-healthy", f_fake, np.ones(21), 0.93, 1.07, ceiling=1.05,
    )


def test_cv11_per_freq_band_check_rejects_ceiling_violation_within_band():
    """A value inside [lo, hi] can still violate the SEPARATE passivity
    ceiling — the ceiling must be checked independently of the band."""
    cv11 = _load_cv11()
    f_fake = np.linspace(8.2e9, 12.4e9, 21)
    mag = np.ones(21)
    mag[5] = 1.06   # inside [0.93, 1.07] but above ceiling=1.05
    assert not cv11.per_freq_band_check(
        "test-ceiling", f_fake, mag, 0.93, 1.07, ceiling=1.05,
    )


def test_cv11_selftest_runs_without_aborting():
    """cv11's own _selftest_per_freq_gate (validation/crossval/
    11_waveguide_port_wr90.py:357-377) calls sys.exit(1) if either of its two
    synthetic checks fails to bite. A normal return here means the gate is
    genuinely live on the version of the code under test."""
    cv11 = _load_cv11()
    cv11._selftest_per_freq_gate()


# ---------------------------------------------------------------------------
# cv04 gate logic, replicated with exact source-line citations (see module
# docstring for why this can't be a direct import).
# ---------------------------------------------------------------------------

# validation/crossval/04_multilayer_fresnel.py:338 (issue #341)
CV04_CONS_MAX_LIMIT = 0.06
# validation/crossval/04_multilayer_fresnel.py:232-234 (issue #341)
CV04_TAIL_WINDOW = 50
CV04_TAIL_PURITY_LIMIT = 1e-3
CV04_TAIL_LIMIT = 0.10


def _cv04_cons_max_ok(r_plus_t_minus_1: np.ndarray) -> bool:
    """Replicates validation/crossval/04_multilayer_fresnel.py:316,339:
    ``cons_rfx = np.abs(R_rfx + T_rfx - 1)``;
    ``cons_max_ok = bool(cons_rfx.max() <= CONS_MAX_LIMIT)``."""
    cons = np.abs(r_plus_t_minus_1)
    return bool(cons.max() <= CV04_CONS_MAX_LIMIT)


def test_cv04_conservation_ceiling_rejects_single_bin_spike():
    """A single out-of-band |R+T-1| bin above the ceiling must FAIL —
    this is the exact class of silent single-bin spike issue #341 closed."""
    healthy = np.full(21, 0.01)
    spike = healthy.copy()
    spike[10] = 0.10  # > CV04_CONS_MAX_LIMIT
    assert not _cv04_cons_max_ok(spike)


def test_cv04_conservation_ceiling_accepts_healthy_curve():
    healthy = np.full(21, 0.01)
    assert _cv04_cons_max_ok(healthy)


def _cv04_tail_ok(inc_tail: np.ndarray, refl_tail: np.ndarray,
                  trans_tail: np.ndarray, inc_peak: float) -> bool:
    """Replicates validation/crossval/04_multilayer_fresnel.py:236-243:
    the settling-tail witness (issue #341) — the last TAIL_WINDOW samples of
    the incident/reflected/transmitted time series must be clean (incident
    tail negligible relative to its own peak = pulse has passed) and settled
    (reflected/transmitted tails below TAIL_LIMIT of the incident peak)."""
    tail_inc_rel = np.max(np.abs(inc_tail)) / inc_peak
    tail_refl_rel = np.max(np.abs(refl_tail)) / inc_peak
    tail_trans_rel = np.max(np.abs(trans_tail)) / inc_peak
    tail_window_clean = tail_inc_rel < CV04_TAIL_PURITY_LIMIT
    return bool(
        tail_window_clean
        and tail_refl_rel < CV04_TAIL_LIMIT
        and tail_trans_rel < CV04_TAIL_LIMIT
    )


def test_cv04_tail_witness_rejects_contaminated_window():
    """A tail window still carrying the direct pulse (incident tail not
    negligible) must FAIL, regardless of the reflected/transmitted levels."""
    inc_peak = 1.0
    contaminated_inc = np.full(CV04_TAIL_WINDOW, 0.5)  # >> purity limit
    settled = np.zeros(CV04_TAIL_WINDOW)
    assert not _cv04_tail_ok(contaminated_inc, settled, settled, inc_peak)


def test_cv04_tail_witness_rejects_unsettled_reflected_or_transmitted():
    """A clean incident window with a reflected/transmitted tail still above
    TAIL_LIMIT (ringing, not yet settled) must FAIL."""
    inc_peak = 1.0
    clean_inc = np.zeros(CV04_TAIL_WINDOW)
    unsettled = np.full(CV04_TAIL_WINDOW, 0.5)  # >> TAIL_LIMIT
    assert not _cv04_tail_ok(clean_inc, unsettled, unsettled, inc_peak)


def test_cv04_tail_witness_accepts_clean_settled_window():
    inc_peak = 1.0
    clean_inc = np.zeros(CV04_TAIL_WINDOW)
    settled = np.full(CV04_TAIL_WINDOW, 0.01)  # << TAIL_LIMIT
    assert _cv04_tail_ok(clean_inc, settled, settled, inc_peak)


# ---------------------------------------------------------------------------
# cv04 fringe-resolved gate (issue #812).
#
# The audit found that cv04's E4 label was carried by an `import` (no Meep
# number entered any verdict) and that its analytic reference decided only
# through a band mean over an interference pattern (pattern P2). It measured
# the consequence: with the pre-#812 gates the case reports PASS while the true
# slab permittivity is 12.33% away from what rfx built, or the true thickness
# 8.0% away, or the measured R_max 22.3% low.
#
# These tests import the SAME functions the crossval script calls, so they
# cannot drift from the gate they pin. The committed cv04 configuration is
# reproduced here as constants; the falsifiers are the audit's own numbers.
# ---------------------------------------------------------------------------

CV04_EPS = 4.0
CV04_D = 10.0e-3
CV04_N = 2.0
CV04_DX = 1.0e-3
CV04_DT = 2.335067793382187e-12  # Grid(...).dt for the committed cv04 config
CV04_DF_BIN = 1.0 / (8192 * CV04_DT)  # nfft = 2**ceil(log2(719)) * 8
CV04_C0 = 2.998e8
CV04_BAND = (3.0321e9, 11.8666e9)  # the mask's contiguous band, 170 bins


def _load_fringe_gate():
    path = CROSSVAL_DIR / "comparators" / "fringe_gate.py"
    spec = importlib.util.spec_from_file_location("_cv04_fringe_gate_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cv04_band_axis():
    lo, hi = CV04_BAND
    n = int(round((hi - lo) / CV04_DF_BIN)) + 1
    return lo + CV04_DF_BIN * np.arange(n)


def _ideal_slab_R(freqs, eps_r=CV04_EPS, d=CV04_D):
    """Exact lossless-slab R(f) at normal incidence (Airy form).

    R = F sin^2(delta) / (1 + F sin^2(delta)),  F = 4 r / (1 - r)^2,
    r = ((n-1)/(n+1))^2 the single-interface power reflectance,
    delta = 2 pi f n d / c. Peaks at R_max = ((eps-1)/(eps+1))^2 = 0.36 for
    eps = 4, zeros at delta = m pi.
    """
    n = math.sqrt(eps_r)
    delta = 2 * np.pi * freqs * n * d / CV04_C0
    r = ((n - 1.0) / (n + 1.0)) ** 2
    finesse = 4.0 * r / (1.0 - r) ** 2
    s2 = np.sin(delta) ** 2
    return finesse * s2 / (1.0 + finesse * s2)


def _cv04_compare(freqs, r_curve, **overrides):
    fg = _load_fringe_gate()
    kwargs = dict(
        eps_r=CV04_EPS, d=CV04_D, n_index=CV04_N,
        dx=CV04_DX, dt=CV04_DT, df_bin_hz=CV04_DF_BIN, c0=CV04_C0,
        label="test",
    )
    kwargs.update(overrides)
    return fg.compare_fringes(freqs, r_curve, **kwargs)


def test_cv04_fringe_windows_match_the_predeclared_values():
    """The frozen windows are 59.1 / 106.3 / 234.9 MHz at the three gated
    fringes (docs/design_notes/issue812_cv04_fringe_gate_predeclaration.md
    section 4, committed BEFORE the measurement). If this test moves, someone
    widened a pre-declared gate. These are the digits behind
    `_04_fresnel_results/fringe_gate_geometry.json::windows.fringes[]
    .position_window_hz`, which is what the gate's comments cite."""
    fg = _load_fringe_gate()
    windows = [
        fg.position_window_hz(
            f, n_index=CV04_N, dx=CV04_DX, dt=CV04_DT,
            df_bin_hz=CV04_DF_BIN, c0=CV04_C0,
        ) / 1e6
        for f in (3.7475e9, 7.4950e9, 11.2425e9)
    ]
    assert windows == pytest.approx([59.04, 106.37, 234.88], abs=0.05)
    assert fg.FRINGE_VALUE_LIMIT == 0.04
    assert fg.SAFETY == 2.0


def test_cv04_fringe_gate_accepts_the_ideal_slab():
    """(A) An exact analytic slab must PASS -- otherwise the gate is broken,
    not the physics."""
    freqs = _cv04_band_axis()
    verdict = _cv04_compare(freqs, _ideal_slab_R(freqs))
    assert verdict.ok, verdict.reasons
    assert [row.kind for row in verdict.rows] == ["max", "min", "max"]


def test_cv04_fringe_gate_rejects_the_audit_eps_defect():
    """(B) The audit's eps probe: the true slab is eps=4.4933 (+12.33%) and rfx
    measured a 4.0 slab. Every pre-#812 gate passes on this (measured: T mean
    err 0.0494 and R mean err 0.0449 against their 0.05 limits, max|R+T-1|
    0.0487 against 0.06). The fringe gate must fire, and on POSITION."""
    freqs = _cv04_band_axis()
    measured = _ideal_slab_R(freqs, eps_r=4.0)          # what rfx built
    verdict = _cv04_compare(freqs, measured, eps_r=4.4933, n_index=None)
    assert not verdict.ok
    assert any("fringe POSITION" in reason for reason in verdict.reasons)


def test_cv04_fringe_gate_rejects_the_audit_thickness_defect():
    """(B) The audit's thickness probe: the true slab is 8.0% thicker."""
    freqs = _cv04_band_axis()
    measured = _ideal_slab_R(freqs, d=CV04_D)
    verdict = _cv04_compare(freqs, measured, d=CV04_D * 1.08)
    assert not verdict.ok
    assert any("fringe POSITION" in reason for reason in verdict.reasons)


def test_cv04_fringe_gate_rejects_the_audit_rmax_defect():
    """(B) The audit's amplitude probe: R_max reads 22.3% low (0.3600 ->
    0.2797) with the fringe POSITIONS untouched. The value gate must fire and
    the position gate must stay silent -- the two are independent."""
    freqs = _cv04_band_axis()
    measured = _ideal_slab_R(freqs) * (0.2797 / 0.3600)
    verdict = _cv04_compare(freqs, measured)
    assert not verdict.ok
    assert any("fringe VALUE" in reason for reason in verdict.reasons)
    assert not any("fringe POSITION" in reason for reason in verdict.reasons)


def test_cv04_fringe_gate_rejects_a_one_cell_thickness_error():
    """One cell is the smallest thickness error the 1 mm grid can express."""
    freqs = _cv04_band_axis()
    measured = _ideal_slab_R(freqs, d=CV04_D + CV04_DX)
    verdict = _cv04_compare(freqs, measured)
    assert not verdict.ok
    assert any("fringe POSITION" in reason for reason in verdict.reasons)


def test_cv04_fringe_gate_fails_rather_than_reports_a_pinned_extremum():
    """A measured curve with no fringe structure at all (flat R) must FAIL with
    a CONTAINMENT reason, not silently report a cell boundary as an extremum.
    This is the property that keeps the half-fringe search window from
    entailing the verdict."""
    freqs = _cv04_band_axis()
    verdict = _cv04_compare(freqs, np.full(freqs.shape, 0.36))
    assert not verdict.ok
    assert any("CONTAINMENT" in reason for reason in verdict.reasons)


def test_cv04_fringe_gate_is_not_entailed_by_its_own_search_window():
    """The search window (half-fringe cell, +-1873.75 MHz) must be far wider
    than the verdict window (<=234.9 MHz), so 'found in the cell' cannot imply
    'passes the gate'. A 400 MHz shift is comfortably inside every cell and
    must still FAIL."""
    fg = _load_fringe_gate()
    freqs = _cv04_band_axis()
    fsr = CV04_C0 / (2 * CV04_N * CV04_D)
    cell_half = fg.CELL_HALF_WIDTHS_PER_FSR * fsr
    shift = 400e6
    assert shift < cell_half           # inside the search window
    # A frequency-scaled curve shifts every fringe; scale so the top fringe
    # moves by `shift`.
    scale = 11.2425e9 / (11.2425e9 - shift)
    verdict = _cv04_compare(freqs, _ideal_slab_R(freqs * scale))
    assert not verdict.ok
    assert any("fringe POSITION" in reason for reason in verdict.reasons)


def test_cv04_external_pointwise_gate_accepts_an_agreeing_solver():
    """(A) A Meep run that reproduces the analytic slab and rfx must pass."""
    fg = _load_fringe_gate()
    tiny = np.full(64, 0.01)
    assert fg.external_pointwise_reasons(tiny, tiny, tiny, tiny) == []


def test_cv04_external_pointwise_gate_rejects_a_disagreeing_solver():
    """(B) The gate that did not exist before #812: a Meep result that departs
    from the analytic slab, or from rfx, must now FAIL the script instead of
    being printed. Both legs are checked independently."""
    fg = _load_fringe_gate()
    tiny = np.full(64, 0.01)
    bad_vs_analytic = tiny.copy()
    bad_vs_analytic[7] = 0.09          # > MEEP_ABS_LIMIT = 0.08
    reasons = fg.external_pointwise_reasons(bad_vs_analytic, tiny, tiny, tiny)
    assert len(reasons) == 1
    assert "Meep vs analytic" in reasons[0]

    bad_cross = tiny.copy()
    bad_cross[3] = 0.20                # > MEEP_CROSS_LIMIT = 0.16
    reasons = fg.external_pointwise_reasons(tiny, tiny, bad_cross, tiny)
    assert len(reasons) == 1
    assert "rfx vs Meep" in reasons[0]


def test_cv04_external_pointwise_gate_is_a_maximum_not_a_mean():
    """A single bad bin in an otherwise perfect comparison must fire: the P2
    band-mean collapse is exactly what this gate exists to avoid."""
    fg = _load_fringe_gate()
    perfect = np.zeros(4096)
    spike = perfect.copy()
    spike[2048] = 0.5
    assert fg.external_pointwise_reasons(spike, perfect, perfect, perfect)
    assert float(spike.mean()) < fg.MEEP_ABS_LIMIT   # a mean gate would pass


# ---------------------------------------------------------------------------
# cv04 numeric provenance (issue #812 ROUND 2).
#
# Round 1's blocking finding on this lane was not a defective gate: it was a
# source comment claiming the detector is "REFERENCE-BLIND", which the
# implementation has not been since the gate was first measured (a
# reference-blind prominence detector was tried and withdrawn for failing
# criterion (A) on correct code — pre-declaration note section 9). The repair
# is mechanical rather than editorial: every quantity cv04's prose asserts about
# this gate is emitted to a committed artifact and re-derived here, so a stale
# sentence and a stale number both fail a test instead of a reviewer.
# ---------------------------------------------------------------------------

CV04_EVIDENCE_JSON = (
    CROSSVAL_DIR / "_04_fresnel_results" / "fringe_gate_geometry.json"
)


def _load_cv04_evidence_emitter():
    path = CROSSVAL_DIR / "comparators" / "emit_cv04_fringe_gate_evidence.py"
    spec = importlib.util.spec_from_file_location("_cv04_evidence_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_same(actual, expected, path="$"):
    """Recursive compare: exact for str/bool/None, 1e-9 relative for numbers."""
    assert type(actual) is type(expected) or (
        isinstance(actual, (int, float)) and isinstance(expected, (int, float))
    ), f"{path}: type {type(actual)} != {type(expected)}"
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"{path}: key set differs"
        for key in expected:
            _assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: length differs"
        for i, item in enumerate(expected):
            _assert_same(actual[i], item, f"{path}[{i}]")
    elif isinstance(expected, bool) or expected is None:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, (int, float)):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-15), (
            f"{path}: {actual!r} != {expected!r}"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_cv04_committed_evidence_json_is_reproducible():
    """The committed artifact must be exactly what the emitter produces today.

    This is the numeric-provenance leg: cv04's comments and design note point at
    keys in this file instead of restating digits, so a number that drifts from
    the code that produced it fails here rather than surviving in prose."""
    import json

    assert CV04_EVIDENCE_JSON.exists(), (
        f"missing {CV04_EVIDENCE_JSON}; regenerate with "
        "python validation/crossval/comparators/emit_cv04_fringe_gate_evidence.py"
    )
    committed = json.loads(CV04_EVIDENCE_JSON.read_text(encoding="utf-8"))
    fresh = _load_cv04_evidence_emitter().build_evidence()
    _assert_same(fresh, committed)


def test_cv04_evidence_records_every_falsifier_verdict_it_declares():
    """Each falsifier's measured verdict must match its declared expectation —
    criterion (A) passes, every criterion (B) probe fails."""
    ev = _load_cv04_evidence_emitter().build_evidence()
    ids = [f["id"] for f in ev["falsifiers"]]
    assert len(ids) == len(set(ids))
    assert any(f["criterion"] == "A" for f in ev["falsifiers"])
    assert sum(f["criterion"] == "B" for f in ev["falsifiers"]) >= 5
    for f in ev["falsifiers"]:
        assert f["verdict"]["ok"] is f["expect_ok"], f["id"]


def test_cv04_search_window_cannot_entail_the_verdict():
    """The property that replaced the withdrawn reference-blindness claim.

    Non-entailment is a relation between two widths plus the pinning rule, not
    a property of the detector's ignorance. Both halves are asserted here from
    the artifact the comments cite."""
    ev = _load_cv04_evidence_emitter().build_evidence()
    windows = ev["windows"]
    # (i) the search cell is much wider than the widest verdict window
    ratio = windows["cell_half_width_hz"] / windows["max_position_window_hz"]
    assert ratio == pytest.approx(windows["non_entailment_ratio"], rel=1e-12)
    assert ratio > 5.0
    for fringe in windows["fringes"]:
        assert fringe["position_window_over_cell_half_width"] < 0.2
    # (ii) reaching the cell edge is a failure, not a reported boundary
    by_id = {f["id"]: f for f in ev["falsifiers"]}
    pinned = by_id["B_structureless_curve_is_a_containment_failure"]
    assert pinned["verdict"]["reason_classes"] == ["CONTAINMENT"]
    # (iii) a shift the detector CAN find still fails the gate
    inside = by_id["B_shift_inside_the_search_cell_still_fails"]
    assert inside["shift_over_cell_half_width"] < 1.0
    assert inside["verdict"]["reason_classes"] == ["POSITION"]
    assert inside["verdict"]["worst_position_window_utilisation"] > 1.0


def test_cv04_gate_sources_do_not_claim_reference_blindness():
    """Anti-regression for round 1's blocking finding.

    The detector is reference-ANCHORED. Any file that gates cv04 claiming
    otherwise is describing an implementation this repo withdrew, so the claim
    fails a test rather than a reviewer. `reference-blind` may only appear where
    it is explicitly negated or named as the withdrawn alternative."""
    ev = _load_cv04_evidence_emitter().build_evidence()
    assert ev["detector"]["reference_blind"] is False

    for name in ("04_multilayer_fresnel.py", "comparators/fringe_gate.py"):
        lines = (CROSSVAL_DIR / name).read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            if "reference-blind" not in line.lower():
                continue
            # a comment paragraph wraps, so judge the claim on its neighbours
            # with the comment markers stripped
            low = " ".join(
                raw.strip().lstrip("#").strip()
                for raw in lines[max(0, lineno - 2):lineno + 1]
            ).lower()
            negated = any(
                token in low
                for token in (
                    "not reference-blind",          # the direct denial
                    "not load-bearing",             # the reason it is denied
                    "reference-blind prominence",   # the withdrawn detector
                    "previously claimed",           # the correction itself
                    "withdrawn",
                )
            )
            assert negated, (
                f"{name}:{lineno} asserts reference-blindness: {line.strip()!r}"
            )


def test_cv04_evidence_config_still_matches_the_script_it_cites():
    """The evidence artifact is only worth its keys if its config is the
    script's config. Each constant is re-read from the cited source line, so an
    edit to the crossval script that moves eps/d/dx/c0 invalidates the artifact
    here instead of silently invalidating every number that points at it."""
    emitter = _load_cv04_evidence_emitter()
    lines = (CROSSVAL_DIR / "04_multilayer_fresnel.py").read_text(
        encoding="utf-8"
    ).splitlines()

    def value_on(lineno: int, name: str) -> float:
        stmt = lines[lineno - 1].split("#")[0].strip()
        lhs, _, rhs = stmt.partition("=")
        assert lhs.strip() == name, (
            f"04_multilayer_fresnel.py:{lineno} is {stmt!r}, not an assignment "
            f"to {name}"
        )
        return float(eval(rhs.strip(), {"math": math, "np": np}))  # noqa: S307

    assert value_on(63, "eps_slab") == emitter.EPS_R
    assert value_on(65, "d_slab") == emitter.D_M
    assert value_on(67, "dx") == emitter.DX_M
    assert value_on(43, "C0") == emitter.C0
    assert math.sqrt(emitter.EPS_R) == emitter.N_INDEX
    # the FFT length the bin width is derived from
    assert "np.ceil(np.log2(n_steps)) * 8" in lines[290 - 1]
