"""Committed gates for the 3-port H-plane T-junction broad-E4/E5 evidence.

ADDITIVE, shadow-class evidence for the already-``broad_e5_passed`` rectangular
waveguide port family. These fixtures are deliberately NOT wired into the
manifest's ``external_comparison_artifacts`` / ``broad_e5_envelope_artifacts``
lists, so the external-reference auditor's per-family counts are UNCHANGED
(the same precedent as the NU-waveguide broad-E4 lane, PR #257). This evidence
does NOT promote a public multi-port S-matrix API: the public
``compute_waveguide_s_matrix`` still lacks per-port reference plumbing — see the
skipped ``test_waveguide_branch_junction_mixed_normals_reciprocal_through_api``
in ``tests/test_api.py``. The T-junction numbers were produced through the
module-level ``extract_waveguide_s_matrix_flux(ref_materials_per_port=...)`` with
per-port straight-guide PEC references, an internal (not public-surface) path.

Two committed fixtures are locked here, both re-derived (pure NumPy, no FDTD)
from the RAW magnitude arrays embedded in each fixture, using the SAME metric
functions the producer used (imported from
``scripts/diagnostics/build_waveguide_tjunction_committed_fixtures.py``), so a
regression to a fixture's numbers OR a silently-loosened gate goes red:

1. ``waveguide_tjunction_broad_e5_envelope.json`` — rfx-internal reciprocity /
   passivity / mesh-convergence over the converged mesh pair (dx 1.0 / 0.667mm,
   fixed 48mm CPML) across the full single-mode TE10 band 5.0-7.0 GHz.

2. ``waveguide_tjunction_meep_external_comparison.json`` — rfx |S| vs an
   independent matched far-port MEEP FDTD flux reference (res=4000).

Both layers REPLAY frozen numbers; they are not a live-physics anchor. The
gitignored-``.omx`` loss of the June-2026 numbers (the reason for this recommit)
is exactly what committing the raw arrays here prevents.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_waveguide_tjunction_committed_fixtures import (  # noqa: E402
    cross_fdtd_bandmean,
    cross_fdtd_max,
    mesh_convergence,
    passivity,
    reciprocity,
)

FIXTURE_DIR = REPO / "tests" / "fixtures" / "waveguide_tjunction_e4"
_ENVELOPE = "waveguide_tjunction_broad_e5_envelope.json"
_COMPARISON = "waveguide_tjunction_meep_external_comparison.json"

# Pinned gate constants (the producer's tj_finalize_converged.py values). The
# re-derivation below reads tolerances FROM the fixture, so without this pin a
# silently-loosened fixture gate would keep everything green.
EXPECTED_ENVELOPE_GATES = {
    "reciprocity_tol": 0.05,
    "passivity_tol": 1.10,
    "convergence_tol": 0.08,
}
EXPECTED_XFDTD_TOL = 0.11

# Same broad-blocking tokens the external-reference auditor rejects. (This lane
# is not auditor-wired, but the honesty guard is cheap and mirrors the siblings.)
BLOCKING_TOKENS = (
    "narrow", "enabling", "blocked", "partial", "limited", "experimental",
    "shadow",
)


def _load_envelope(fixture_dir: Path = FIXTURE_DIR) -> dict:
    return json.loads((fixture_dir / _ENVELOPE).read_text())


def _load_comparison(fixture_dir: Path = FIXTURE_DIR) -> dict:
    return json.loads((fixture_dir / _COMPARISON).read_text())


# --------------------------------------------------------------------------- #
# Presence / status / pinned-gate guards.                                     #
# --------------------------------------------------------------------------- #
def test_envelope_fixture_present_and_passed() -> None:
    env = _load_envelope()
    assert env["schema"] == "rfx.waveguide_tjunction_broad_e5_envelope"
    assert env["schema_version"] == 2
    assert env["status"] == "passed"
    assert env["evidence_level"].startswith("E5-broad")
    lvl = env["evidence_level"].lower()
    for tok in BLOCKING_TOKENS:
        assert tok not in lvl, f"blocking token {tok!r} in evidence_level"


def test_comparison_fixture_present_and_passed() -> None:
    cmp = _load_comparison()
    assert cmp["schema"] == "rfx.waveguide_tjunction_meep_external_comparison"
    assert cmp["schema_version"] == 2
    assert cmp["status"] == "passed"
    assert cmp["evidence_level"].startswith("E4-broad")
    lvl = cmp["evidence_level"].lower()
    for tok in BLOCKING_TOKENS:
        assert tok not in lvl, f"blocking token {tok!r} in evidence_level"


def test_envelope_gates_pinned() -> None:
    """A silently-loosened envelope gate must go red here."""
    assert _load_envelope()["gates"] == EXPECTED_ENVELOPE_GATES


def test_comparison_gate_pinned() -> None:
    assert _load_comparison()["cross_fdtd_tol"] == EXPECTED_XFDTD_TOL


# --------------------------------------------------------------------------- #
# Re-derivation from the RAW embedded arrays (integrity + gate verdict).       #
# --------------------------------------------------------------------------- #
def test_envelope_rederives_from_raw_and_gates_pass() -> None:
    env = _load_envelope()
    S_coarse = np.asarray(env["S_coarse"], dtype=float)
    S_fine = np.asarray(env["S_fine"], dtype=float)
    # Real 3-port, full band: (recv, drive, freq) = (3, 3, N).
    assert S_coarse.shape[:2] == (3, 3) and S_fine.shape[:2] == (3, 3), (
        S_coarse.shape, S_fine.shape)
    N = S_fine.shape[2]
    assert len(env["band_hz"]) == N

    gates = env["gates"]
    by_mesh = {c["mesh"]: c for c in env["cases"]}
    for label, S in (("coarse", S_coarse), ("fine", S_fine)):
        c = by_mesh[label]
        r = reciprocity(S)
        p = passivity(S)
        # Recorded metrics must equal a fresh re-derivation from the raw arrays.
        assert r == pytest.approx(c["reciprocity"], rel=1e-9, abs=1e-12), c
        assert p == pytest.approx(c["passivity_max"], rel=1e-9, abs=1e-12), c
        # And the gates must genuinely pass.
        assert r <= gates["reciprocity_tol"], c
        assert p <= gates["passivity_tol"], c
        assert c["reciprocity_pass"] and c["passivity_pass"], c

    conv = mesh_convergence(S_coarse, S_fine)
    assert conv == pytest.approx(env["mesh_convergence_max"], rel=1e-9, abs=1e-12)
    assert conv <= gates["convergence_tol"]

    # The per-freq mesh-conv table must re-derive too (no hand-edited row).
    band = np.asarray(env["band_hz"], dtype=float)
    for k in range(N):
        key = f"{band[k] / 1e9:.1f}GHz"
        expected = float(np.abs(S_coarse[:, :, k] - S_fine[:, :, k]).max())
        assert env["per_freq_mesh_conv"][key] == pytest.approx(expected, rel=1e-9, abs=1e-12)

    assert env["status"] == "passed"


def test_comparison_rederives_from_raw_and_gates_pass() -> None:
    cmp = _load_comparison()
    S_fine = np.asarray(cmp["rfx_S_fine"], dtype=float)
    M = np.asarray(cmp["M"], dtype=float)
    assert S_fine.shape == M.shape and S_fine.shape[:2] == (3, 3), (S_fine.shape, M.shape)

    xdev = cross_fdtd_max(S_fine, M)
    xdev_bm = cross_fdtd_bandmean(S_fine, M)
    meep_p = passivity(M)
    meep_r = reciprocity(M)

    assert xdev == pytest.approx(cmp["rfx_vs_meep_max_abs_dev"], rel=1e-9, abs=1e-12)
    assert xdev_bm == pytest.approx(cmp["rfx_vs_meep_bandmean_max_abs_dev"], rel=1e-9, abs=1e-12)
    assert meep_p == pytest.approx(cmp["meep_passivity_max"], rel=1e-9, abs=1e-12)
    assert meep_r == pytest.approx(cmp["meep_reciprocity"], rel=1e-9, abs=1e-12)

    tol = cmp["cross_fdtd_tol"]
    assert xdev <= tol
    assert meep_r <= EXPECTED_ENVELOPE_GATES["reciprocity_tol"]
    assert meep_p <= EXPECTED_ENVELOPE_GATES["passivity_tol"]
    assert cmp["status"] == "passed"


# --------------------------------------------------------------------------- #
# Fail-closed perturbations (in-memory copies of the raw arrays).              #
# --------------------------------------------------------------------------- #
def test_passivity_scale_fails_closed() -> None:
    """Scaling the rfx |S| up by 1.3 must break the passivity gate."""
    S_fine = np.asarray(_load_envelope()["S_fine"], dtype=float)
    tampered = S_fine * 1.3
    assert passivity(S_fine) <= EXPECTED_ENVELOPE_GATES["passivity_tol"]
    assert passivity(tampered) > EXPECTED_ENVELOPE_GATES["passivity_tol"]


def test_reciprocity_asymmetry_fails_closed() -> None:
    """Breaking the S[1,0]/S[0,1] symmetry must break the reciprocity gate."""
    S_fine = np.asarray(_load_envelope()["S_fine"], dtype=float)
    tampered = S_fine.copy()
    tampered[1, 0] += 0.2
    assert reciprocity(S_fine) <= EXPECTED_ENVELOPE_GATES["reciprocity_tol"]
    assert reciprocity(tampered) > EXPECTED_ENVELOPE_GATES["reciprocity_tol"]


def test_cross_fdtd_corruption_fails_closed() -> None:
    """Shifting the MEEP reference by +0.2 must break the cross-FDTD gate."""
    cmp = _load_comparison()
    S_fine = np.asarray(cmp["rfx_S_fine"], dtype=float)
    M = np.asarray(cmp["M"], dtype=float)
    assert cross_fdtd_max(S_fine, M) <= cmp["cross_fdtd_tol"]
    assert cross_fdtd_max(S_fine, M + 0.2) > cmp["cross_fdtd_tol"]


def test_tampered_recorded_metric_detected() -> None:
    """A hand-edited summary metric no longer matches the raw re-derivation."""
    env = copy.deepcopy(_load_envelope())
    S_coarse = np.asarray(env["S_coarse"], dtype=float)
    S_fine = np.asarray(env["S_fine"], dtype=float)
    rederived = mesh_convergence(S_coarse, S_fine)
    # Untampered fixture agrees.
    assert env["mesh_convergence_max"] == pytest.approx(rederived, rel=1e-9, abs=1e-12)
    # Tamper the recorded scalar only (raw arrays untouched) -> integrity break.
    env["mesh_convergence_max"] = rederived + 0.5
    assert env["mesh_convergence_max"] != pytest.approx(rederived, rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- #
# Cross-consistency between the two fixtures.                                  #
# --------------------------------------------------------------------------- #
def test_cross_fixture_consistency() -> None:
    """The two fixtures must describe the same run: identical band, and the
    envelope's S_fine must equal the rfx side used in the comparison."""
    env = _load_envelope()
    cmp = _load_comparison()
    assert env["band_hz"] == cmp["band_hz"], "band mismatch between fixtures"
    S_fine_env = np.asarray(env["S_fine"], dtype=float)
    S_fine_cmp = np.asarray(cmp["rfx_S_fine"], dtype=float)
    assert S_fine_env.shape == S_fine_cmp.shape
    assert np.allclose(S_fine_env, S_fine_cmp, rtol=0, atol=0), (
        "envelope S_fine differs from the comparison's rfx side")


# --------------------------------------------------------------------------- #
# Committed-artifact guard (the whole point of the recommit).                  #
# --------------------------------------------------------------------------- #
def _git_available() -> bool:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=REPO, capture_output=True, text=True,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"
    except OSError:
        return False


def _is_tracked(rel_path: str) -> bool:
    r = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel_path],
        cwd=REPO, capture_output=True, text=True,
    )
    return r.returncode == 0


def test_both_fixtures_git_tracked() -> None:
    """Both fixture files must be genuinely git-tracked — a present-but-untracked
    file (the gitignored-.omx shape that lost the June numbers) does NOT count.
    Skips only where git is unavailable (mirrors the fast-suite manifest guard)."""
    if not _git_available():
        pytest.skip("git unavailable; committed-artifact membership cannot be verified")
    for name in (_ENVELOPE, _COMPARISON):
        rel = str((FIXTURE_DIR / name).relative_to(REPO))
        assert _is_tracked(rel), f"{rel} is not git-tracked (evidence would be lost on clean checkout)"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
