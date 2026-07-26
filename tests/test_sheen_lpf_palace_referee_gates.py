"""cv07 Sheen microstrip LPF — Palace-FEM REFEREE gates.

Locks the independent-method arbitration committed under
``tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json``. Palace is a
frequency-domain FEM solver on a conformal tetrahedral mesh (no staircase), run on
the matched cv07 geometry to referee the committed rfx-vs-openEMS "first null"
split (rfx argmin 7.218 GHz, openEMS argmin 7.983 GHz).

WHAT THE REFEREE FOUND: the Sheen stopband is a DOUBLE transmission-zero
(~7.0 AND ~8.0 GHz), not a single null. openEMS resolves this doublet in close
agreement with the conformal referee; rfx's coarser mesh + frequency sampling
distorts it. The committed argmin "first null" compares different members of the
doublet in each solver, so its ~10% "split" is largely a comparator artifact. The
frozen ``sides_with`` therefore names the FDTD solver whose full stopband STRUCTURE
(both zeros) the referee matches best; the fragile argmin metric is locked
separately and labelled as such. No analytic reference (the stepped-impedance
transmission zeros have no clean fringing-free closed form).

No FDTD / no Palace re-run here: the raw Palace port-S arrays are committed as a
fixture, and every gated number is re-derived (no CSV) by
``build_sheen_lpf_palace_referee.build_referee``.

THE REFEREE LOCK (``test_referee_sides_with_structure``) reads the rfx and openEMS
doublet zeros from their sibling committed cv07 fixtures
(``validation/crossval/_07_sheen_results/{rfx,openems}.json``), so tampering with either
FDTD side (or with the Palace doublet) fails the gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIX = _REPO_ROOT / "tests/fixtures/sheen_lpf_e4"
_REFEREE = _FIX / "sheen_lpf_palace_referee.json"
_SHEEN_RESULTS = _REPO_ROOT / "validation/crossval/_07_sheen_results"
sys.path.insert(0, str(_REPO_ROOT / "scripts/diagnostics"))
from build_sheen_lpf_palace_referee import build_referee  # noqa: E402

pytestmark = pytest.mark.skipif(
    not _REFEREE.exists(),
    reason="Palace Sheen referee fixture absent (VESSL run not yet frozen)",
)


@pytest.fixture(scope="module")
def derived():
    return build_referee(_FIX)


@pytest.fixture(scope="module")
def fixture():
    return json.loads(_REFEREE.read_text())


def test_all_blocks_passive(derived, fixture):
    """Passivity witness: the referee is only trustworthy if it conserves energy
    on the matched geometry. Both sweeps AND both 11-pt probes stay under the 1.02
    fail-closed ceiling (a lumped-driven microstrip LPF radiates, so the sum sits
    below 1)."""
    for mesh in ("coarse", "mid"):
        assert derived[mesh]["sweep_max_energy_sum"] <= 1.02, (mesh, "sweep")
        assert derived[mesh]["probe_max_energy_sum"] <= 1.02, (mesh, "probe")
        assert np.isclose(
            fixture[mesh]["max_energy_sum"], derived[mesh]["sweep_max_energy_sum"], rtol=1e-9
        )
        assert np.isclose(
            fixture[mesh]["probe"]["max_energy_sum"], derived[mesh]["probe_max_energy_sum"], rtol=1e-9
        )


def test_recorded_doublet_matches_rederived(derived, fixture):
    """Fixture integrity: the recorded per-mesh doublet (lower/upper zero: bin,
    parabolic vertex, depth, |S11|@min) and the argmin null equal what
    ``build_referee`` re-derives from the raw Palace arrays."""
    for mesh in ("coarse", "mid"):
        for side in ("lower", "upper"):
            rec = fixture[mesh]["doublet"][side]
            red = derived[mesh]["doublet"][side]
            for key in ("bin_f_ghz", "parabolic_f_ghz", "depth_db", "s11_at_min"):
                assert np.isclose(rec[key], red[key], rtol=1e-9), (mesh, side, key)
        for key in ("bin_f_ghz", "parabolic_f_ghz", "depth_db", "s11_at_min"):
            assert np.isclose(fixture[mesh]["null"][key], derived[mesh]["null"][key], rtol=1e-9)


def test_stopband_is_a_deep_doublet(derived):
    """The referee's core finding: each Palace mesh shows TWO deep transmission
    zeros (lower ~7, upper ~8 GHz), both below -25 dB, and they are distinct
    (>~0.5 GHz apart) — i.e. a genuine double-null, not a single notch."""
    for mesh in ("coarse", "mid"):
        d = derived[mesh]["doublet"]
        assert d["lower"]["depth_db"] < -25.0, (mesh, "lower depth", d["lower"]["depth_db"])
        assert d["upper"]["depth_db"] < -25.0, (mesh, "upper depth", d["upper"]["depth_db"])
        sep = d["upper"]["parabolic_f_ghz"] - d["lower"]["parabolic_f_ghz"]
        assert sep > 0.5, (mesh, "zero separation", sep)


def test_doublet_convergence(derived):
    """coarse -> mid (sqrt2 refinement) barely moves EITHER zero: fractional shift
    <= 3% per zero, so the double-null structure is converged, not mesh-dependent."""
    ref = derived["referee"]
    pd = ref["palace_doublet_mid_ghz"]
    for side in ("lower", "upper"):
        assert abs(ref["convergence_shift_ghz"][side]) / pd[side] <= 0.03, (side,)


def test_referee_sides_with_structure(derived, fixture):
    """THE REFEREE LOCK: the FDTD solver named by ``sides_with`` matches BOTH
    Palace zeros (worst-zero % error) strictly better than the other solver — the
    two FDTD doublets are re-read from the SIBLING committed cv07 fixtures, so this
    is a cross-fixture lock (tampering either side flips it)."""
    ref = derived["referee"]
    winner = ref["sides_with"]
    assert winner == fixture["referee"]["sides_with"]
    struct = ref["structure_distance_pct"]
    other = "openems" if winner == "rfx" else "rfx"
    assert struct[winner] < struct[other], (winner, struct)

    # cross-fixture lock: re-derive the two FDTD doublets from _sheen_results and
    # confirm the winner's zeros are jointly closer to the Palace doublet.
    pd = ref["palace_doublet_mid_ghz"]

    def _doublet_from(path):
        d = json.loads(path.read_text())
        f = np.asarray(d["freqs_hz"], dtype=float) / 1e9
        s21 = np.asarray(d["s21_mag"], dtype=float)

        def _lo_hi(lo, hi):
            m = (f >= lo) & (f <= hi)
            idx = np.where(m)[0]
            return float(f[int(idx[int(np.argmin(s21[idx]))])])
        return _lo_hi(6.3, 7.5), _lo_hi(7.5, 8.6)

    win_lo, win_hi = _doublet_from(_SHEEN_RESULTS / f"{winner}.json")
    oth_lo, oth_hi = _doublet_from(_SHEEN_RESULTS / f"{other}.json")
    win_err = max(abs(win_lo - pd["lower"]) / pd["lower"], abs(win_hi - pd["upper"]) / pd["upper"])
    oth_err = max(abs(oth_lo - pd["lower"]) / pd["lower"], abs(oth_hi - pd["upper"]) / pd["upper"])
    assert win_err < oth_err, (winner, win_err, other, oth_err)


def test_committed_referee_matches_rederived(derived, fixture):
    """Fixture integrity: the committed ``referee`` section equals what the
    producer re-derives."""
    assert fixture["referee"] == derived["referee"]


def test_meta_integrity(fixture):
    """Provenance guard: Palace solver, two distinct meshes, the VESSL run ids, the
    lumped-port leakage note, and the no-analytic statement are all present."""
    meta = fixture["meta"]
    assert meta["solver"] == "palace"
    mesh = meta["mesh"]
    assert set(mesh) == {"coarse", "mid"}
    assert mesh["coarse"]["tets"] != mesh["mid"]["tets"], "meshes must differ"
    assert meta["vessl_runs"], "VESSL run record must be kept"
    assert "n/a" in meta["analytic_ref"].lower()
    assert "level" in meta["port_note"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
