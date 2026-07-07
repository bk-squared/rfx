"""cv06b MSL open-stub notch — Palace-FEM REFEREE gates (WP 1-B).

Locks the independent-method arbitration committed under
``tests/fixtures/msl_notch_e4/msl_stub_notch_palace_referee.json``. Palace is a
frequency-domain FEM solver on a conformal tetrahedral mesh (no staircase), so it
captures the open-end fringing exactly and REFEREES the committed rfx-vs-openEMS
~5.8% notch split (rfx 3.6273 GHz, openEMS 3.4286 GHz, analytic 3.69 GHz).

Result locked here: Palace lands at ~3.631 GHz (parabolic) at BOTH mesh
densities (coarse->mid shift only -0.006 GHz), closest to rfx (+0.1%; the
openEMS reference is ~5.9% away). Our earlier working interpretation
("openEMS captures more open-end fringing") is revised by this evidence.

No FDTD / no Palace re-run here: the raw Palace port-S arrays are committed as a
fixture, and every gated number is re-derived (no CSV) by
``build_msl_notch_palace_referee.build_referee`` — mirroring the openEMS
comparison-gate + coax broad-E4 evidence-commit pattern.

THE REFEREE LOCK (``test_referee_sides_with_rfx``) reads the rfx and openEMS
notch frequencies from their sibling committed fixtures, so tampering with either
side (or with the Palace mid notch) fails the gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIX = _REPO_ROOT / "tests/fixtures/msl_notch_e4"
_REFEREE = _FIX / "msl_stub_notch_palace_referee.json"
sys.path.insert(0, str(_REPO_ROOT / "scripts/diagnostics"))
from build_msl_notch_palace_referee import build_referee  # noqa: E402


@pytest.fixture(scope="module")
def derived():
    return build_referee(_FIX)


@pytest.fixture(scope="module")
def fixture():
    return json.loads(_REFEREE.read_text())


def test_all_four_blocks_passive(derived, fixture):
    """Passivity witness: the referee is only trustworthy if it conserves energy
    on the matched geometry. Both sweeps AND both 11-pt probes stay well under
    the 1.02 fail-closed ceiling (all ~0.86)."""
    for mesh in ("coarse", "mid"):
        assert derived[mesh]["sweep_max_energy_sum"] <= 1.02, (mesh, "sweep")
        assert derived[mesh]["probe_max_energy_sum"] <= 1.02, (mesh, "probe")
        # committed max_energy_sum agrees with the re-derived value
        assert np.isclose(
            fixture[mesh]["max_energy_sum"], derived[mesh]["sweep_max_energy_sum"], rtol=1e-9
        )
        assert np.isclose(
            fixture[mesh]["probe"]["max_energy_sum"], derived[mesh]["probe_max_energy_sum"], rtol=1e-9
        )


def test_recorded_notch_matches_rederived(derived, fixture):
    """Fixture integrity: the recorded per-mesh notch (bin, parabolic vertex,
    depth, |S11|@notch) equals what ``build_referee`` re-derives from the raw
    Palace arrays."""
    for mesh in ("coarse", "mid"):
        rec = fixture[mesh]["notch"]
        red = derived[mesh]["notch"]
        for key in ("bin_f_ghz", "parabolic_f_ghz", "depth_db", "s11_at_notch"):
            assert np.isclose(rec[key], red[key], rtol=1e-9), (mesh, key, rec[key], red[key])


def test_single_notch_per_sweep(derived):
    """Each Palace sweep has exactly ONE local |S21| minimum — a single clean
    stub notch, not a Fabry-Perot comb."""
    for mesh in ("coarse", "mid"):
        assert derived[mesh]["n_local_minima"] == 1, (mesh, derived[mesh]["n_local_minima"])


def test_mesh_convergence(derived):
    """coarse -> mid (sqrt2 refinement) barely moves the notch: |shift| <= 0.05
    GHz, so the referee is converged, not mesh-dependent."""
    shift = derived["referee"]["convergence_shift_ghz"]
    assert abs(shift) <= 0.05, shift


def test_referee_sides_with_rfx(derived):
    """THE REFEREE LOCK: the Palace mid-mesh notch is CLOSER to the rfx notch
    than to the openEMS notch — both read from the SIBLING committed fixtures, so
    this is a cross-fixture lock (tampering either side flips it)."""
    palace_mid = derived["mid"]["notch"]["parabolic_f_ghz"]
    rfx = json.loads((_FIX / "msl_stub_notch_rfx_dx50.json").read_text())
    oe = json.loads((_FIX / "msl_stub_notch_openems_dx50.json").read_text())
    rfx_notch = float(rfx["notch"]["f_ghz"])
    oe_notch = float(oe["notch"]["f_ghz"])
    assert abs(palace_mid - rfx_notch) < abs(palace_mid - oe_notch), (
        f"Palace mid {palace_mid} must be closer to rfx {rfx_notch} than openEMS {oe_notch}"
    )
    assert derived["referee"]["sides_with"] == "rfx"


def test_committed_referee_matches_rederived(derived, fixture):
    """Fixture integrity: the committed ``referee`` section equals what the
    producer re-derives (sides_with, palace mid notch, three-way distances,
    convergence shift)."""
    assert fixture["referee"] == derived["referee"]


def test_meta_integrity(fixture):
    """Provenance guard: Palace solver, two distinct meshes, and the VESSL run
    ids (incl. the failed-lane record) are all present."""
    meta = fixture["meta"]
    assert meta["solver"] == "palace"
    mesh = meta["mesh"]
    assert set(mesh) == {"coarse", "mid"}
    assert mesh["coarse"]["tets"] != mesh["mid"]["tets"], "meshes must differ"
    runs = meta["vessl_runs"]
    assert runs["coarse"] and runs["mid"]
    assert runs["failed_lanes"], "failed-lane record must be kept"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
