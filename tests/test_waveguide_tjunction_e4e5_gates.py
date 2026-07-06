"""Committed gates for the 3-port H-plane T-junction full-band E4/E5 evidence,
now a TWO-GEOMETRY broad claim.

Evidence for the ``broad_e5_passed`` rectangular waveguide port family. History:
PR #270 wired the SINGLE-geometry T-junction MEEP comparison into the manifest's
``external_comparison_artifacts`` as a claims-bearing passed-E4 artifact and
gated a companion single-geometry envelope — honestly BELOW the auditor's
numeric breadth bar (>= 2 geometry variants), so the envelope stayed UNLISTED.
Those single-geometry honesty locks were the explicit tripwire demanding a
breadth campaign. This campaign COMPLETES the breadth bar with a genuine second
junction geometry (W=0.036 m, arms 92/92/72 mm, band 5.2-7.3 GHz) alongside the
original (W=0.040 m, arms 90/90/70 mm, band 5.0-7.0 GHz):

* The MEEP external COMPARISON now carries ``summary.geometry_count == 2`` with
  zero failed pairs (6 = 2 geometries x 3 driven ports), so the auditor's
  ``_comparison_breadth_ok`` classifies it as a BROAD-E4 artifact.
* The rfx-internal ENVELOPE now carries ``envelope_summary.case_count == 4``
  (2 geometries x 2 converged meshes) across 2 distinct geometry variants and 2
  distinct dx over a >=1.4 frequency span, so ``_envelope_breadth_ok`` accepts
  it and it is now LISTED in ``broad_e5_envelope_artifacts``.

The public junction API (``port_reference_sims`` on
``compute_waveguide_s_matrix``) shipped in PR #269. The T-junction numbers were
produced through ``extract_waveguide_s_matrix_flux(ref_materials_per_port=...)``
with per-port straight-guide PEC references.

Two committed fixtures are locked here, both re-derived (pure NumPy, no FDTD)
from the RAW per-geometry magnitude arrays embedded in each fixture, using the
SAME metric functions the producer used (imported from
``scripts/diagnostics/build_waveguide_tjunction_committed_fixtures.py``), so a
regression to a fixture's numbers OR a silently-loosened gate goes red:

1. ``waveguide_tjunction_broad_e5_envelope.json`` — per-geometry reciprocity /
   passivity / mesh-convergence over each geometry's converged mesh pair
   (dx 1.0 / 0.667mm, fixed 48mm CPML) across its full single-mode TE10 band.

2. ``waveguide_tjunction_meep_external_comparison.json`` — per-geometry rfx |S|
   vs an independent matched far-port MEEP FDTD flux reference (res=4000).

The two-geometry BROAD locks (``test_two_geometry_broad_locks``) require the
claim prose and the machine-readable numerics to agree in BOTH directions
(scope says "broad" AND the summary is numerically broad), and drive the REAL
auditor predicates so the fixtures stay conformant with the auditor's breadth
schema, not just with this test's own reading.

Both layers REPLAY frozen numbers; they are not a live-physics anchor. The
gitignored-``.omx`` loss of the June-2026 numbers (the reason committing the raw
arrays here matters) is exactly what committing the raw arrays here prevents.
"""
from __future__ import annotations

import copy
import json
import os
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
from check_port_external_references import (  # noqa: E402  # type: ignore
    _comparison_breadth_ok,
    _envelope_breadth_ok,
)

# FIXTURE_DIR defaults to the committed fixtures; an env override lets the
# producer's synthetic end-to-end validation point the SAME gate functions at a
# scratch build (the committed-artifact guard skips cleanly when overridden).
FIXTURE_DIR = Path(
    os.environ.get(
        "RFX_TJUNCTION_FIXTURE_DIR",
        str(REPO / "tests" / "fixtures" / "waveguide_tjunction_e4"),
    )
)
_ENVELOPE = "waveguide_tjunction_broad_e5_envelope.json"
_COMPARISON = "waveguide_tjunction_meep_external_comparison.json"

# The two junction geometries of the broad claim.
GEOM_KEYS = ("geom1", "geom2")
EXPECTED_LABELS = {
    "hplane_tee_W40mm_arms90_90_70",
    "hplane_tee_W36mm_arms92_92_72",
}

# Pinned gate constants (the producer's tj_finalize_converged.py values). The
# re-derivation below reads tolerances FROM the fixture, so without this pin a
# silently-loosened fixture gate would keep everything green.
EXPECTED_ENVELOPE_GATES = {
    "reciprocity_tol": 0.05,
    "passivity_tol": 1.10,
    "convergence_tol": 0.08,
}
EXPECTED_XFDTD_TOL = 0.11

# Same broad-blocking tokens the external-reference auditor rejects in an
# evidence_level. (The auditor also scans claim_scope with a superset that
# includes "only"; that channel is exercised via the real predicate below.)
BLOCKING_TOKENS = (
    "narrow", "enabling", "blocked", "partial", "limited", "experimental",
    "shadow",
)


def _load_envelope(fixture_dir: Path = FIXTURE_DIR) -> dict:
    return json.loads((fixture_dir / _ENVELOPE).read_text())


def _load_comparison(fixture_dir: Path = FIXTURE_DIR) -> dict:
    return json.loads((fixture_dir / _COMPARISON).read_text())


def _geom_block(payload: dict, key: str) -> dict:
    return next(b for b in payload["geometry_blocks"] if b["key"] == key)


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
# Two-geometry BROAD honesty locks (the FLIP of the old single-geometry locks).#
# --------------------------------------------------------------------------- #
def test_two_geometry_broad_locks() -> None:
    """The broad claim rests on TWO genuine junction geometries — lock it, and
    require the claim prose and the machine-readable numerics to agree in BOTH
    directions.

    (a) Both fixtures' claim_scope MUST now contain "broad" (the old locks
    forbade it under a single geometry; a broad claim now must be earned). (b)
    The comparison summary records geometry_count == 2 with two DISTINCT labels
    and zero failed pairs (6 = 2 geometries x 3 driven ports). (c) The envelope
    summary records case_count == 4 (2 geometries x 2 converged meshes), all
    passing, two distinct dx and a >=1.4 freq span across two distinct
    geometries. (d) The REAL auditor breadth predicates must ACCEPT both
    fixtures, and each must fail-closed when a geometry is stripped from the
    summary — the genuine numeric-broad verification, not this test's own
    reading."""
    env = _load_envelope()
    cmp = _load_comparison()

    # Claim <-> numerics agreement (both directions): scope says broad AND the
    # summaries are numerically broad.
    assert "broad" in env["claim_scope"].lower(), env["claim_scope"]
    assert "broad" in cmp["claim_scope"].lower(), cmp["claim_scope"]

    s = cmp["summary"]
    assert s["geometry_count"] == 2
    assert len(s["geometries"]) == 2 and len(set(s["geometries"])) == 2
    assert set(s["geometries"]) == EXPECTED_LABELS
    assert s["pair_count"] == 6
    assert s["passed_pair_count"] == 6
    assert s["failed_pair_count"] == 0

    es = env["envelope_summary"]
    assert es["case_count"] == 4 and es["passed_case_count"] == 4
    assert len(set(es["dx_values_m"])) == 2
    assert len(es["geometries"]) == 2 and len(set(es["geometries"])) == 2
    assert set(es["geometries"]) == EXPECTED_LABELS
    lo, hi = es["freq_range_hz"]
    assert lo > 0 and hi / lo >= 1.4, (lo, hi)

    # Drive the ACTUAL auditor predicates — the genuine numeric-broad check.
    assert _envelope_breadth_ok(env)[0] is True, _envelope_breadth_ok(env)[1]
    assert _comparison_breadth_ok(cmp)[0] is True, _comparison_breadth_ok(cmp)[1]

    # Fail-closed: stripping a geometry from either summary must flip both
    # predicates to False.
    pe = copy.deepcopy(env)
    pe["envelope_summary"]["geometries"] = pe["envelope_summary"]["geometries"][:1]
    assert _envelope_breadth_ok(pe)[0] is False, "envelope must fail-closed on single geometry"

    pc = copy.deepcopy(cmp)
    pc["summary"]["geometry_count"] = 1
    assert _comparison_breadth_ok(pc)[0] is False, "comparison must fail-closed on single geometry"


def test_manifest_wiring_state() -> None:
    """Governance lock: BOTH fixtures are now listed in the manifest — the
    comparison in ``external_comparison_artifacts`` (claims-bearing broad-E4),
    and the envelope in ``broad_e5_envelope_artifacts`` (two-geometry broad-E5).
    Silent unwiring goes red here."""
    manifest = json.loads(
        (REPO / "scripts" / "diagnostics" / "port_external_reference_requirements.json").read_text()
    )
    wg = next(e for e in manifest["requirements"]
              if e["family"] == "rectangular_waveguide_port")
    cmp_rel = f"tests/fixtures/waveguide_tjunction_e4/{_COMPARISON}"
    env_rel = f"tests/fixtures/waveguide_tjunction_e4/{_ENVELOPE}"
    assert cmp_rel in wg["external_comparison_artifacts"], (
        "T-junction MEEP comparison silently unwired from the manifest")
    assert env_rel in wg["broad_e5_envelope_artifacts"], (
        "two-geometry T-junction envelope must now be listed as a broad-E5 "
        "envelope artifact (the breadth bar is genuinely met)")


# --------------------------------------------------------------------------- #
# Re-derivation from the RAW embedded per-geometry arrays (integrity + gates).  #
# --------------------------------------------------------------------------- #
def test_envelope_rederives_from_raw_and_gates_pass() -> None:
    env = _load_envelope()
    gates = env["gates"]
    blocks = env["geometry_blocks"]
    assert len(blocks) == 2
    assert {b["key"] for b in blocks} == set(GEOM_KEYS), blocks
    assert {b["label"] for b in blocks} == EXPECTED_LABELS

    conv_over = []
    total_cases = 0
    for b in blocks:
        S_coarse = np.asarray(b["S_coarse"], dtype=float)
        S_fine = np.asarray(b["S_fine"], dtype=float)
        # Real 3-port, full band: (recv, drive, freq) = (3, 3, N).
        assert S_coarse.shape[:2] == (3, 3) and S_fine.shape[:2] == (3, 3), (
            b["key"], S_coarse.shape, S_fine.shape)
        N = S_fine.shape[2]
        assert len(b["band_hz"]) == N
        assert {c["mesh"] for c in b["cases"]} == {"coarse", "fine"}, b["cases"]
        assert N >= 11, f"{b['key']}: expected the full band (>=11 pts), got {N}"

        by_mesh = {c["mesh"]: c for c in b["cases"]}
        for label, S in (("coarse", S_coarse), ("fine", S_fine)):
            c = by_mesh[label]
            r = reciprocity(S)
            p = passivity(S)
            # Recorded metrics must equal a fresh re-derivation from the raw arrays.
            assert r == pytest.approx(c["reciprocity"], rel=1e-9, abs=1e-12), (b["key"], c)
            assert p == pytest.approx(c["passivity_max"], rel=1e-9, abs=1e-12), (b["key"], c)
            # And the gates must genuinely pass.
            assert r <= gates["reciprocity_tol"], (b["key"], c)
            assert p <= gates["passivity_tol"], (b["key"], c)
            assert c["reciprocity_pass"] and c["passivity_pass"], (b["key"], c)
            total_cases += 1

        conv = mesh_convergence(S_coarse, S_fine)
        assert conv == pytest.approx(b["mesh_convergence_max"], rel=1e-9, abs=1e-12)
        assert conv <= gates["convergence_tol"], (b["key"], conv)
        conv_over.append(conv)

        # The per-freq mesh-conv table must re-derive too (no hand-edited row).
        band = np.asarray(b["band_hz"], dtype=float)
        for k in range(N):
            key = f"{band[k] / 1e9:.1f}GHz"
            expected = float(np.abs(S_coarse[:, :, k] - S_fine[:, :, k]).max())
            assert b["per_freq_mesh_conv"][key] == pytest.approx(expected, rel=1e-9, abs=1e-12)

    # Top-level + summary aggregates must re-derive from the per-geometry blocks.
    assert env["mesh_convergence_max"] == pytest.approx(max(conv_over), rel=1e-9, abs=1e-12)
    es = env["envelope_summary"]
    assert total_cases == 4
    assert es["case_count"] == 4 and es["passed_case_count"] == 4
    assert es["mesh_convergence_max"] == pytest.approx(max(conv_over), rel=1e-9, abs=1e-12)
    assert env["status"] == "passed"


def test_comparison_rederives_from_raw_and_gates_pass() -> None:
    cmp = _load_comparison()
    tol = cmp["cross_fdtd_tol"]
    blocks = cmp["geometry_blocks"]
    assert len(blocks) == 2
    assert {b["key"] for b in blocks} == set(GEOM_KEYS), blocks

    xdevs, bandmeans = [], []
    for b in blocks:
        S_fine = np.asarray(b["rfx_S_fine"], dtype=float)
        M = np.asarray(b["M"], dtype=float)
        assert S_fine.shape == M.shape and S_fine.shape[:2] == (3, 3), (
            b["key"], S_fine.shape, M.shape)

        xdev = cross_fdtd_max(S_fine, M)
        xdev_bm = cross_fdtd_bandmean(S_fine, M)
        meep_p = passivity(M)
        meep_r = reciprocity(M)

        assert xdev == pytest.approx(b["cross_fdtd_max"], rel=1e-9, abs=1e-12)
        assert xdev_bm == pytest.approx(b["cross_fdtd_bandmean"], rel=1e-9, abs=1e-12)
        assert meep_p == pytest.approx(b["meep_passivity_max"], rel=1e-9, abs=1e-12)
        assert meep_r == pytest.approx(b["meep_reciprocity"], rel=1e-9, abs=1e-12)

        assert xdev <= tol, (b["key"], xdev)
        assert meep_r <= EXPECTED_ENVELOPE_GATES["reciprocity_tol"], (b["key"], meep_r)
        assert meep_p <= EXPECTED_ENVELOPE_GATES["passivity_tol"], (b["key"], meep_p)
        assert b["status"] == "passed", b["key"]

        # The per-freq cross-FDTD table must re-derive too.
        band = np.asarray(b["band_hz"], dtype=float)
        for k in range(S_fine.shape[2]):
            key = f"{band[k] / 1e9:.1f}GHz"
            expected = float(np.abs(S_fine[:, :, k] - M[:, :, k]).max())
            assert b["per_freq_cross_fdtd"][key] == pytest.approx(expected, rel=1e-9, abs=1e-12)

        xdevs.append(xdev)
        bandmeans.append(xdev_bm)

    s = cmp["summary"]
    assert s["max_mag_abs_diff"] == pytest.approx(max(xdevs), rel=1e-9, abs=1e-12)
    assert s["bandmean_mag_abs_diff"] == pytest.approx(max(bandmeans), rel=1e-9, abs=1e-12)
    assert cmp["status"] == "passed"


# --------------------------------------------------------------------------- #
# Fail-closed perturbations, on EACH geometry block.                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", GEOM_KEYS)
def test_passivity_scale_fails_closed(key: str) -> None:
    """Scaling a geometry's rfx |S| up by 1.3 must break the passivity gate."""
    S_fine = np.asarray(_geom_block(_load_envelope(), key)["S_fine"], dtype=float)
    assert passivity(S_fine) <= EXPECTED_ENVELOPE_GATES["passivity_tol"]
    assert passivity(S_fine * 1.3) > EXPECTED_ENVELOPE_GATES["passivity_tol"]


@pytest.mark.parametrize("key", GEOM_KEYS)
def test_reciprocity_asymmetry_fails_closed(key: str) -> None:
    """Breaking a geometry's S[1,0]/S[0,1] symmetry must break reciprocity."""
    S_fine = np.asarray(_geom_block(_load_envelope(), key)["S_fine"], dtype=float)
    tampered = S_fine.copy()
    tampered[1, 0] += 0.2
    assert reciprocity(S_fine) <= EXPECTED_ENVELOPE_GATES["reciprocity_tol"]
    assert reciprocity(tampered) > EXPECTED_ENVELOPE_GATES["reciprocity_tol"]


@pytest.mark.parametrize("key", GEOM_KEYS)
def test_cross_fdtd_corruption_fails_closed(key: str) -> None:
    """Shifting a geometry's MEEP reference by +0.2 must break the cross-FDTD gate."""
    cmp = _load_comparison()
    b = _geom_block(cmp, key)
    S_fine = np.asarray(b["rfx_S_fine"], dtype=float)
    M = np.asarray(b["M"], dtype=float)
    assert cross_fdtd_max(S_fine, M) <= cmp["cross_fdtd_tol"]
    assert cross_fdtd_max(S_fine, M + 0.2) > cmp["cross_fdtd_tol"]


def test_tampered_recorded_metric_detected() -> None:
    """A hand-edited per-geometry summary metric no longer matches the raw
    re-derivation."""
    env = copy.deepcopy(_load_envelope())
    b = _geom_block(env, "geom1")
    S_coarse = np.asarray(b["S_coarse"], dtype=float)
    S_fine = np.asarray(b["S_fine"], dtype=float)
    rederived = mesh_convergence(S_coarse, S_fine)
    # Untampered fixture agrees.
    assert b["mesh_convergence_max"] == pytest.approx(rederived, rel=1e-9, abs=1e-12)
    # Tamper the recorded scalar only (raw arrays untouched) -> integrity break.
    b["mesh_convergence_max"] = rederived + 0.5
    assert b["mesh_convergence_max"] != pytest.approx(rederived, rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- #
# Cross-consistency between the two fixtures, per geometry.                     #
# --------------------------------------------------------------------------- #
def test_cross_fixture_consistency() -> None:
    """The two fixtures must describe the same runs per geometry: identical band,
    and each geometry's envelope S_fine must equal the rfx side used in that
    geometry's comparison."""
    env = _load_envelope()
    cmp = _load_comparison()
    env_blocks = {b["key"]: b for b in env["geometry_blocks"]}
    cmp_blocks = {b["key"]: b for b in cmp["geometry_blocks"]}
    assert set(env_blocks) == set(cmp_blocks) == set(GEOM_KEYS)
    for key in GEOM_KEYS:
        eb, cb = env_blocks[key], cmp_blocks[key]
        assert eb["band_hz"] == cb["band_hz"], f"{key} band mismatch between fixtures"
        assert eb["label"] == cb["label"], f"{key} label mismatch between fixtures"
        S_fine_env = np.asarray(eb["S_fine"], dtype=float)
        S_fine_cmp = np.asarray(cb["rfx_S_fine"], dtype=float)
        assert S_fine_env.shape == S_fine_cmp.shape
        assert np.allclose(S_fine_env, S_fine_cmp, rtol=0, atol=0), (
            f"{key} envelope S_fine differs from the comparison's rfx side")


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
    Skips where git is unavailable OR where FIXTURE_DIR is overridden outside the
    repo (the producer's scratch synthetic-validation run)."""
    if not _git_available():
        pytest.skip("git unavailable; committed-artifact membership cannot be verified")
    try:
        rels = [str((FIXTURE_DIR / name).relative_to(REPO)) for name in (_ENVELOPE, _COMPARISON)]
    except ValueError:
        pytest.skip("FIXTURE_DIR overridden outside the repo (dev/scratch validation run)")
    for rel in rels:
        assert _is_tracked(rel), f"{rel} is not git-tracked (evidence would be lost on clean checkout)"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
