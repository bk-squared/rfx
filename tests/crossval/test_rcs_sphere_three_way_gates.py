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
# The WHOLE keypath must be dotted keys and [i] indices and nothing else: the
# first version used findall, so `monostatic/rfx_sigma_over_pi_a2` and
# `monostatic..` tokenized to the same steps as the correct spelling and
# resolved happily (round-2 item 9b).
_KEYPATH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_][A-Za-z0-9_-]*|\[\d+\])*$")

# Where a referenced measurement is allowed to live: its CANONICAL home, not
# "any tracked file with those bytes" (round-2 item 4 -- the reviewer resolved
# a byte-identical duplicate of the sibling fixture and the gate accepted it).
#
# The crossval manifest is the registry for a case's own artifacts, and it is
# consulted first. This fixture is not one: it was produced by its own
# committed generator for issue #276, and no crossval case's script writes it,
# so registering it under cv16 would assert an ownership that does not exist --
# the mistake this whole change exists to stop. It is registered HERE instead,
# with the two things that make the claim checkable: the generator that wrote
# it and the gate that reads it, both committed.
CANONICAL_ARTIFACTS: dict[str, dict[str, str]] = {
    "tests/fixtures/rcs_sphere_mie/fixture.json": {
        "measurement": "PEC-sphere monostatic backscatter RCS at the fine rung (ka~1, dx=lambda/40)",
        "generator": "tests/fixtures/rcs_sphere_mie/generate_fixture.py",
        "gate": "tests/crossval/test_rcs_mie_reference_gates.py",
        "why_not_in_the_manifest": (
            "no crossval case script produces it; it is issue #276's own "
            "fixture, and claiming it under cv16 would be a false ownership"),
    },
}


def _manifest_artifacts() -> set[str]:
    manifest = json.loads((_REPO_ROOT / "validation/crossval/manifest.json")
                          .read_text(encoding="utf-8"))
    return {path for case in manifest["cases"] for path in case["artifact_paths"]}


def _assert_canonical(rel: str, reference: str) -> None:
    """The path must be the registered home of the measurement it names."""
    if rel in _manifest_artifacts():
        return
    entry = CANONICAL_ARTIFACTS.get(rel)
    assert entry is not None, (
        f"{reference}: {rel} is not a registered artifact. A reference must "
        f"name the canonical home of a measurement -- an artifact declared in "
        f"validation/crossval/manifest.json, or one registered in "
        f"CANONICAL_ARTIFACTS with its generator and its gate -- so that a "
        f"byte-identical copy somewhere else cannot stand in for it.")
    for key in ("measurement", "generator", "gate"):
        assert entry.get(key, "").strip(), (rel, key)
    for key in ("generator", "gate"):
        target = entry[key]
        assert (_REPO_ROOT / target).is_file(), (rel, key, target)
        if git_available(_REPO_ROOT):
            assert is_tracked(target, _REPO_ROOT), (rel, key, target)


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
    assert _KEYPATH.match(keypath), (
        f"{reference}: {keypath!r} is not a keypath -- dotted keys and [i] "
        f"indices only")
    assert not Path(rel).is_absolute() and ".." not in Path(rel).parts, (
        f"{reference}: the path must be repo-relative")
    target = _REPO_ROOT / rel
    assert target.is_file(), f"{reference}: {rel} does not exist"
    if git_available(_REPO_ROOT):
        assert is_tracked(rel, _REPO_ROOT), (
            f"{reference}: {rel} exists here but is NOT git-tracked, so the "
            f"reference resolves on this machine and nowhere else.")
    _assert_canonical(rel, reference)
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


def test_the_resolver_refuses_a_duplicate_and_a_malformed_keypath(fx, tmp_path):
    """(B) arm for round-2 items 4 and 9b.

    A byte-identical COPY of the sibling fixture, git-tracked or not, is not
    the measurement's home; and a keypath that is not dotted keys must not
    tokenize into one that is.
    """
    reference = fx["three_way_ka1"]["rfx_fine_ref"]
    rel, _, keypath = reference.partition("::")

    # the real one resolves
    assert _resolve(reference) == json.loads(
        (_REPO_ROOT / rel).read_text())["monostatic"]["rfx_sigma_over_pi_a2"]

    # A byte-identical duplicate is refused because the PATH is not the
    # registered home -- the reviewer's plant was git-tracked, so the tracking
    # check alone would have accepted it. Both halves are exercised: the
    # registry rule on a tracked file that is not the home, and the whole
    # resolver on an untracked copy.
    tracked_but_not_the_home = "tests/fixtures/rcs_sphere_three_way/fixture.json"
    assert is_tracked(tracked_but_not_the_home, _REPO_ROOT)
    with pytest.raises(AssertionError, match="not a registered artifact"):
        _assert_canonical(tracked_but_not_the_home,
                          f"{tracked_but_not_the_home}::{keypath}")

    duplicate_rel = "tests/fixtures/rcs_sphere_mie/fixture_copy_probe.json"
    duplicate = _REPO_ROOT / duplicate_rel
    assert not duplicate.exists(), "probe path is not clean; a previous run leaked"
    try:
        duplicate.write_bytes((_REPO_ROOT / rel).read_bytes())
        with pytest.raises(AssertionError, match="not a registered artifact"):
            _assert_canonical(duplicate_rel, f"{duplicate_rel}::{keypath}")
        with pytest.raises(AssertionError, match="NOT git-tracked"):
            _resolve(f"{duplicate_rel}::{keypath}")
    finally:
        duplicate.unlink(missing_ok=True)

    # malformed keypaths
    for bad in ("monostatic/rfx_sigma_over_pi_a2", "monostatic..",
                "monostatic rfx", "monostatic.[0]"):
        with pytest.raises(AssertionError, match="is not a keypath"):
            _resolve(f"{rel}::{bad}")


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
