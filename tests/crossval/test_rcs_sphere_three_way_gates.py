"""PEC-sphere three-way RCS gates: exact-Mie / rfx-FDTD / Bempp-BEM (campaign Lane 1).

Adds the FIRST independent integral-equation (surface-BEM) cross-check to rfx's
RCS pipeline, committed under ``tests/fixtures/rcs_sphere_three_way/``. Bempp-cl
meshes the true curved PEC surface (no FDTD staircase), so BEM-vs-Mie agreement
rules out a class of shared-method artefacts that FDTD-vs-FDTD (Meep/openEMS)
cannot, and independently confirms the Mie reference at each ka.

Posture (honest, additive):
  * The exact Mie column is RE-DERIVED here from ``scipy.special`` (does NOT
    import the producer), so a producer-side error cannot self-certify.
  * Bempp values are frozen offline evidence (Bempp is not a CI/runtime dep);
    the gate reads them from the fixture and checks them against re-derived Mie.
  * This lane changes/relaxes NO existing gate. The coarse-ladder envelope stays
    in ``test_rcs_mie_reference_gates.py``; the fine point in
    ``test_rcs_mie_fixture.py``. All cross-solver distances are stated as
    rfx-centric / method-distance facts, never a verdict that a solver is wrong.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

from tests._git_tracked import git_available, is_tracked

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests/fixtures/rcs_sphere_three_way/fixture.json"
_RFX_FINE = _REPO_ROOT / "tests/fixtures/rcs_sphere_mie/fixture.json"

# Measured Bempp-vs-Mie floor is 0.151 dB across the ladder; gate at 0.5 dB
# leaves margin without being loose enough to hide a harness regression.
_BEMPP_SELF_ANCHOR_DB = 0.5
# All three independent methods land within this at ka~1 (measured max 0.063 dB).
_THREE_WAY_CLOSE_DB = 0.30


def _mie_ratio(ka: float) -> float:
    """sigma/(pi a^2) for a PEC sphere, backscatter (Ruck 1970) — independent."""
    x = float(ka)
    n_max = int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2)) + 15
    n = np.arange(1, n_max + 1)
    jn, yn = spherical_jn(n, x), spherical_yn(n, x)
    jnp_, ynp_ = spherical_jn(n, x, derivative=True), spherical_yn(n, x, derivative=True)
    hn, hnp_ = jn + 1j * yn, jnp_ + 1j * ynp_
    a_n = jn / hn
    b_n = (jn + x * jnp_) / (hn + x * hnp_)
    series = np.sum(((-1.0) ** n) * (2 * n + 1) * (a_n - b_n))
    return float(np.abs(series) ** 2 / x ** 2)


@pytest.fixture(scope="module")
def fx():
    return json.loads(_FIXTURE.read_text())


def test_committed_mie_column_is_exact_series(fx):
    """LOAD-BEARING: every committed sigma_mie equals the independently re-derived
    exact series (rtol 1e-9). Tampering with the reference fails here."""
    for row in fx["bempp"]["ladder"]:
        want = _mie_ratio(row["ka"])
        assert np.isclose(row["sigma_mie_over_pi_a2"], want, rtol=1e-9), row["ka"]
    assert np.isclose(fx["three_way_ka1"]["sigma_mie_over_pi_a2"], _mie_ratio(1.0), rtol=1e-9)


def test_bempp_self_anchor_reproduces_mie(fx):
    """The independent BEM reference must pass its own limit first: committed
    Bempp backscatter sigma is within the measured floor of re-derived Mie, and
    the stored dB is self-consistent with the stored sigmas."""
    for row in fx["bempp"]["ladder"]:
        mie = _mie_ratio(row["ka"])
        dB = 10.0 * np.log10(row["sigma_bempp_over_pi_a2"] / mie)
        assert np.isclose(dB, row["dB_bempp_vs_mie"], atol=1e-6), row["ka"]
        assert abs(dB) <= _BEMPP_SELF_ANCHOR_DB, (row["ka"], dB)
    assert fx["bempp"]["floor_measured_db"] <= _BEMPP_SELF_ANCHOR_DB


def test_bempp_h_refinement_converges_to_mie(fx):
    """Discriminating physical witness: as h -> h/2 (N grows) the Bempp residual
    vs INDEPENDENTLY re-derived Mie shrinks monotonically toward 0 (mesh
    convergence), and the finest rung is <= 0.1 dB. dB is recomputed here from the
    committed sigma against re-derived Mie -- the producer cannot self-certify.
    (A fixed normalization error would converge to a nonzero constant, not 0, so
    this + the 4-ka zero-straddle jointly pin the harness normalization.)"""
    mie1 = _mie_ratio(1.0)
    conv = sorted(fx["bempp"]["convergence_ka1"], key=lambda c: c["N_dofs"])
    resid = [abs(10.0 * np.log10(c["sigma_bempp_over_pi_a2"] / mie1)) for c in conv]
    assert resid == sorted(resid, reverse=True), resid  # strictly improving
    assert resid[-1] <= 0.1, resid[-1]


_KEYSTEP = re.compile(r"\[(\d+)\]|([A-Za-z0-9_][A-Za-z0-9_-]*)")


def _resolve(reference: str):
    """Walk a `path.json::a.b[2].c` reference (issue #928).

    The three-way fixture REFERENCES the sibling's committed value instead of
    holding a copy, so the number has one home. A reference is only worth that
    if it cannot point outside the repository or at a file no clone has, so
    this resolver checks both before reading -- the first version accepted any
    path the filesystem would open, including an absolute one and an untracked
    one, which is a reference to this machine rather than to the repository.
    """
    rel, sep, keypath = reference.partition("::")
    assert sep and keypath, f"{reference!r} is not a path::keypath reference"
    assert not Path(rel).is_absolute() and ".." not in Path(rel).parts, (
        f"{reference}: the path must be repo-relative")
    target = _REPO_ROOT / rel
    assert target.is_file(), f"{reference}: {rel} does not exist"
    if git_available(_REPO_ROOT):
        assert is_tracked(rel, _REPO_ROOT), (
            f"{reference}: {rel} exists here but is NOT git-tracked, so the "
            f"reference resolves on this machine and nowhere else.")
    node = json.loads(target.read_text())
    walked = ""
    for step in _KEYSTEP.finditer(keypath):
        index, name = step.group(1), step.group(2)
        if index is not None:
            walked += f"[{index}]"
            assert isinstance(node, list) and int(index) < len(node), (
                f"{reference}: {walked} is not a list index that exists")
            node = node[int(index)]
        else:
            walked = f"{walked}.{name}" if walked else name
            assert isinstance(node, dict) and name in node, (
                f"{reference}: {walked} does not exist")
            node = node[name]
    assert walked.replace(".", "").replace("[", "").replace("]", ""), reference
    return node


def _rfx_fine(fx) -> float:
    return float(_resolve(fx["three_way_ka1"]["rfx_fine_ref"]))


def test_rfx_fine_column_is_a_resolvable_reference_not_a_copy(fx):
    """The rfx column resolves to the sibling fixture's committed monostatic
    value. It carries no copy of that number, and its witness status is stated:
    the reference fixes where the value comes from, not whether the value is
    converged."""
    three_way = fx["three_way_ka1"]
    assert "rfx_fine_over_pi_a2" not in three_way, (
        "the copied value is back; the reference is the only home")
    assert three_way["rfx_fine_ref"].endswith("::monostatic.rfx_sigma_over_pi_a2")
    expected = json.loads(_RFX_FINE.read_text())["monostatic"]["rfx_sigma_over_pi_a2"]
    assert _rfx_fine(fx) == expected
    assert three_way["rfx_fine_witness_status"] == "carried-unwitnessed"
    assert "CPML" in three_way["rfx_fine_witness_note"]


def test_three_way_spread_self_consistent_and_close(fx):
    """At ka~1 the three INDEPENDENT methods (exact analytic / FDTD-fine / BEM)
    mutually agree within _THREE_WAY_CLOSE_DB. Spreads are recomputed from the
    sigmas and checked against the stored values (humble: rfx-centric distances,
    not a pass/fail verdict on any solver)."""
    t = fx["three_way_ka1"]
    mie, rfx, bem = (t["sigma_mie_over_pi_a2"], _rfx_fine(fx), t["bempp_over_pi_a2"])
    want = {
        "rfx_vs_mie": 10 * np.log10(rfx / mie),
        "bempp_vs_mie": 10 * np.log10(bem / mie),
        "rfx_vs_bempp": 10 * np.log10(rfx / bem),
    }
    for key, val in want.items():
        assert np.isclose(t["spread_db"][key], val, atol=1e-6), key
        assert abs(val) <= _THREE_WAY_CLOSE_DB, (key, val)
