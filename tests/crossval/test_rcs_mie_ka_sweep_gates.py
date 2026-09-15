"""PEC-sphere Mie ka-sweep (0.5-4.0) — frozen-fixture gates (campaign item 1).

Locks the committed measurement record of
``validation/crossval/16_pec_sphere_mie_ka_sweep.py --write-fixture``
(``tests/fixtures/rcs_mie_ka_sweep/fixture.json``) against an INDEPENDENT
in-test re-implementation of the exact conducting-sphere Mie backscatter
series (Ruck/Bohren-Huffman convention, from ``scipy.special`` — not the
producer's oracle module, so a shared oracle bug cannot self-certify; a
four-way convention-independent verification is recorded in the PR #475
review).

Two-tier posture (PR #475 revision — the original 3-clearance envelope
ALIASED at fine ka=4.0, which failed a 3.5 dB gate at 9 of 13 clearances):
  * GATED: coarse ka <= 1.25 and fine-rung ka = 2.0 only, at
    gate = round-up(measured clearance-scan envelope x 1.5). The envelope is
    RECOMPUTED HERE from the committed clearance_scan/domain_realizations
    data and asserted against both the recorded envelope field and the gate,
    so none of the three can drift alone (review F6).
  * NOT GATED: coarse ka >= 1.5 and fine ka = 3.0 / ka = 4.0 — domain-size
    unconverged near the deep Mie nulls (domain-to-domain SPREAD 8.0 dB at
    coarse ka=1.75 and 14.5 dB at fine ka=3.0; worst single-point |delta|
    11.1 dB coarse / 9.3 dB fine, both at ka=3.0; fine ka=4.0 max 6.17 dB
    in the review's 13-clearance scan). This module PINS that fence
    (claim-scope docpin + witness-presence + fence-membership assertions) so
    it can be neither silently gated nor silently quoted as validated.

No FDTD runs here — the fixture is frozen evidence; live regeneration is the
crossval script's job. These gates must not be re-tuned to look tighter than
the recorded physics (no-silent-gate-loosening rule).


#931 SCOPE, measured not assumed: the rfx scatterer on this lane is built with
the low-level ``rasterize(grid, [(shape, 1.0, PEC_SIGMA)])`` — a sigma = 1e7
CELL FILL. Design note §1.8 fences that model out of the lattice ownership
contract (it is a lossy volume, not a realized PEC edge set), so nothing here
moves under #931 and no re-run is scheduled. The two models are NOT the same
object: at the cube fixture's mesh, node- and centre-sampling put a sphere at
3023 vs 3082 cells. That difference is pinned in
``tests/crossval/test_rcs_cube_bem_gates.py::test_the_sigma_fill_and_the_pec_contract_are_not_the_same_object``,
so this lane's agreement figures must not be read as evidence about conductor
realization.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.special import spherical_jn, spherical_yn

from tests._gate_policy import gate_from_envelope

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests/fixtures/rcs_mie_ka_sweep/fixture.json"
_ARTIFACT = _REPO_ROOT / "validation/crossval/_16_ka_sweep_results/rfx.json"

KA_GATED_COARSE = [0.5, 0.75, 1.0, 1.25]
KA_FINE_GATED = [2.0]
KA_FINE_REPORTED = [3.0, 4.0]


def _mie_backscatter_over_pi_a2(ka: float) -> float:
    """sigma/(pi a^2), PEC sphere backscatter (Ruck 1970) — independent."""
    x = float(ka)
    n_max = int(np.ceil(x + 4.05 * x ** (1.0 / 3.0) + 2)) + 15
    n = np.arange(1, n_max + 1)
    jn, yn = spherical_jn(n, x), spherical_yn(n, x)
    jnp_ = spherical_jn(n, x, derivative=True)
    ynp_ = spherical_yn(n, x, derivative=True)
    hn, hnp_ = jn + 1j * yn, jnp_ + 1j * ynp_
    a_n = jn / hn
    b_n = (jn + x * jnp_) / (hn + x * hnp_)
    series = np.sum(((-1.0) ** n) * (2 * n + 1) * (a_n - b_n))
    return float(np.abs(series) ** 2 / x ** 2)


@pytest.fixture(scope="module")
def fixture() -> dict:
    with open(_FIXTURE) as f:
        return json.load(f)


def _gated_coarse_deltas(fixture) -> list[float]:
    """Every committed |delta| for the gated coarse bins, across canonical +
    domain realizations + clearance scan (the envelope population)."""
    out = [abs(r["delta_db"]) for r in fixture["gated_coarse"]]
    for c in ("30", "40"):
        out += [abs(r["delta_db"]) for r in fixture["domain_realizations"][c]
                if r["ka"] <= max(KA_GATED_COARSE)]
    for ka in KA_GATED_COARSE:
        out += [abs(r["delta_db"])
                for r in fixture["clearance_scan"]["coarse"][str(ka)]]
    return out


def _gated_fine_deltas(fixture) -> list[float]:
    out = [abs(r["delta_db"]) for r in fixture["gated_fine"]]
    for c in ("30_fine", "40_fine"):
        out += [abs(r["delta_db"]) for r in fixture["domain_realizations"][c]
                if r["ka"] in KA_FINE_GATED]
    for ka in KA_FINE_GATED:
        out += [abs(r["delta_db"])
                for r in fixture["clearance_scan"]["fine"][str(ka)]]
    return out


def test_fixture_and_artifact_are_the_same_record(fixture):
    """The public crossval artifact and the test fixture must not drift apart."""
    with open(_ARTIFACT) as f:
        artifact = json.load(f)
    assert artifact == fixture


def test_script_claim_scope_literal_matches_fixture(fixture):
    """Binds the script-source claim_scope to the committed fixture copy
    (AST-extracted): the PR #476 F1 retraction was applied to this MERGED
    fixture by hand-patching, so without this binding a later
    --write-fixture would silently revert the corrected wording (and any
    script prose edit would leave the fixture stale) while the suite stays
    green — the prose analogue of the D2 constant binding."""
    import ast
    mod = ast.parse((_REPO_ROOT / "validation/crossval/16_pec_sphere_mie_ka_sweep.py"
                     ).read_text(encoding="utf-8"))
    lits = {k.value: ast.literal_eval(v)
            for node in ast.walk(mod) if isinstance(node, ast.Dict)
            for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant)
            and k.value in ("claim_scope", "offline_probes_2026_07_27")}
    assert set(lits) == {"claim_scope", "offline_probes_2026_07_27"}
    assert " ".join(lits["claim_scope"].split()) == " ".join(fixture["claim_scope"].split())
    assert (" ".join(lits["offline_probes_2026_07_27"].split())
            == " ".join(fixture["provenance"]["offline_probes_2026_07_27"].split()))


def test_f1_retraction_is_content_pinned(fixture):
    """PR #476 R2: the leakage-probe retraction (applied retroactively to
    this merged item) must be CONTENT-pinned so a coherent script+fixture
    edit cannot silently undo it."""
    scope = " ".join(fixture["claim_scope"].split()).lower()
    assert "retracted as a same-array tautology" in scope
    prov = " ".join(fixture["provenance"]["offline_probes_2026_07_27"].split()).lower()
    assert "retracted" in prov


def test_gate_equals_recomputed_envelope_times_1p5(fixture):
    """Anti-drift for the AUDIT TRAIL itself (review F6): the recorded
    envelope must equal the envelope recomputed from the committed data, and
    each gate must equal round-up(envelope x 1.5) to 0.1 dB."""
    g = fixture["gates"]
    env_coarse = max(_gated_coarse_deltas(fixture))
    env_fine = max(_gated_fine_deltas(fixture))
    assert abs(g["coarse_measured_envelope_db"] - env_coarse) < 5e-3
    assert abs(g["fine_measured_envelope_db"] - env_fine) < 5e-3
    assert g["coarse_gate_db"] == pytest.approx(
        gate_from_envelope(env_coarse, quantum=10), abs=1e-9)
    assert g["fine_gate_db"] == pytest.approx(
        gate_from_envelope(env_fine, quantum=10), abs=1e-9)
    # D1 (review): HARD numeric ceiling, deliberately redundant with the
    # derived relation above. Without it a physics regression widens its own
    # gate and stays green (the derived assert alone is self-ratifying).
    # Widening either constant requires editing THIS line with a written
    # root-cause — the no-silent-gate-loosening rule.
    assert g["coarse_gate_db"] == 3.3
    assert g["fine_gate_db"] == 4.0


def test_script_live_gate_constants_match_fixture(fixture):
    """D2 (review): the crossval script's LIVE gate constants are what CI
    actually enforces; bind them to the fixture's recorded gates (and hence,
    through the D1 pins, to the hard ceiling) so they cannot diverge."""
    src = (_REPO_ROOT / "validation/crossval/16_pec_sphere_mie_ka_sweep.py"
           ).read_text(encoding="utf-8")
    m_coarse = re.search(r"^GATE_COARSE_DB = ([0-9.]+)", src, re.MULTILINE)
    m_fine = re.search(r"^GATE_FINE_DB = ([0-9.]+)", src, re.MULTILINE)
    assert m_coarse and m_fine, "gate constants not found in script source"
    assert float(m_coarse.group(1)) == fixture["gates"]["coarse_gate_db"]
    assert float(m_fine.group(1)) == fixture["gates"]["fine_gate_db"]


def test_gated_coarse_bins_within_envelope_gate(fixture):
    """ka <= 1.25 coarse: every committed realization within the gate,
    Mie re-derived independently in-test."""
    gate = float(fixture["gates"]["coarse_gate_db"])
    rows = fixture["gated_coarse"]
    assert [r["ka"] for r in rows] == KA_GATED_COARSE
    for r in rows:
        mie_over = _mie_backscatter_over_pi_a2(r["ka"])
        # independent Mie must agree with the recorded Mie leg (oracle check)
        assert abs(10 * np.log10(mie_over / r["mie_sigma_over_pi_a2"])) < 0.01
        delta = 10 * np.log10(r["rfx_sigma_over_pi_a2"] / mie_over)
        assert abs(delta) <= gate, (r["ka"], delta)
    assert max(_gated_coarse_deltas(fixture)) <= gate


def test_gated_fine_rung_within_envelope_gate(fixture):
    """Fine rung (12.8 cells/radius) ka=2.0: within the gate at every
    committed realization; ka=3.0/4.0 must NOT be in the gated list."""
    gate = float(fixture["gates"]["fine_gate_db"])
    rows = fixture["gated_fine"]
    assert [r["ka"] for r in rows] == KA_FINE_GATED
    for r in rows:
        mie_over = _mie_backscatter_over_pi_a2(r["ka"])
        assert abs(10 * np.log10(mie_over / r["mie_sigma_over_pi_a2"])) < 0.01
        delta = 10 * np.log10(r["rfx_sigma_over_pi_a2"] / mie_over)
        assert abs(delta) <= gate, (r["ka"], delta)
    assert max(_gated_fine_deltas(fixture)) <= gate
    # F1 fence: the aliased bin stays out of the gated set
    assert fixture["gates"]["fine_ka"] == KA_FINE_GATED
    assert [r["ka"] for r in fixture["fine_rung_reported"]] == KA_FINE_REPORTED


def test_null_region_is_fenced_not_gated(fixture):
    """The unconverged bins must be present as diagnostics AND fenced in
    prose. If someone converts the null region into a gate (or deletes the
    diagnostics to hide it), this goes red first."""
    scope = " ".join(fixture["claim_scope"].split()).lower()
    assert "not gated" in scope
    assert "domain-size-only change" in scope
    assert "local-minimum positions move" in scope
    assert "ka=4.0 fails a 3.5 db gate" in scope       # the F1 aliasing record
    # D3 (review): metric named explicitly, both tiers, from committed data —
    # docpins on BOTH strings so the fine figure cannot go stale again.
    assert "8.0 db (coarse, at ka=1.75)" in scope
    assert "14.5 db (fine, at ka=3.0" in scope
    # the diagnostic curve actually covers the fenced region (no fine rows
    # smuggled in — review F7: fine reported rows live under their own key)
    kas = [r["ka"] for r in fixture["diagnostic_curve_clear20"]]
    assert len(kas) == len(set(kas)) == 15
    assert set(np.arange(6, 17) * 0.25) <= set(kas)    # 1.5 .. 4.0 present
    assert all(r["cells_per_radius"] == fixture["config"]["coarse_cells_per_radius"]
               for r in fixture["diagnostic_curve_clear20"])
    assert "never gated" in fixture["gates"]["posture"]


def test_attribution_witnesses_are_recorded(fixture):
    """Committed witnesses must ride with the record — truncation ON THE
    GATED BINS (review F4), domain realizations, and the anti-aliasing
    clearance scan (review F1)."""
    trunc = fixture["truncation_witness"]
    assert {t["ka"] for t in trunc} == set(KA_GATED_COARSE + KA_FINE_GATED)
    for t in trunc:
        assert abs(t["delta_1x_db"] - t["delta_2x_db"]) <= 0.3, t
    assert {"30", "40", "30_fine", "40_fine"} <= set(fixture["domain_realizations"])
    assert len(fixture["domain_realizations"]["30"]) == 15
    scan = fixture["clearance_scan"]
    assert len(scan["clearances"]) >= 7
    for ka in KA_GATED_COARSE:
        assert len(scan["coarse"][str(ka)]) == len(scan["clearances"])
    for ka in KA_FINE_GATED + KA_FINE_REPORTED:
        assert len(scan["fine"][str(ka)]) == len(scan["clearances"])
    # the fenced fine bins must show WHY they are fenced, in data:
    ka4 = [abs(r["delta_db"]) for r in scan["fine"]["4.0"]]
    assert max(ka4) > fixture["gates"]["fine_gate_db"], (
        "fine ka=4.0 scan no longer exceeds the gate — if this is a real "
        "physics improvement, promote it with a root-cause note, do not "
        "just delete the fence")
    # offline probes are provenance, not data — the wording must say so
    prov = " ".join(fixture["provenance"]["offline_probes_2026_07_27"].split()).lower()
    assert "not committed as data" in prov


def test_operating_point_is_the_derived_one(fixture):
    """Config pins: cells-per-radius floors, not lambda-resolution floors."""
    cfg = fixture["config"]
    assert cfg["coarse_cells_per_radius"] == 6.4
    assert cfg["fine_cells_per_radius"] == 12.8
    for r in fixture["gated_coarse"] + fixture["diagnostic_curve_clear20"]:
        assert r["a_over_dx"] >= 6.3, r  # the sphere is never cell-starved


# ---------------------------------------------------------------------------
# #931 lattice ownership contract: cv16's conductor is a sigma FILL, fenced
# out of the contract (§1.8). These two tests are the fence, in data.
# ---------------------------------------------------------------------------

def _cv16_module():
    """Import cv16 without executing its __main__ block."""
    import importlib.util
    import sys

    path = _REPO_ROOT / "validation/crossval/16_pec_sphere_mie_ka_sweep.py"
    spec = importlib.util.spec_from_file_location("_cv16_ka_sweep", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cv16_operating_point(ka: float, cpr: float, clear_cells: int = 20):
    """cv16's own geometry arithmetic for one bin, verbatim from run_point."""
    from rfx.geometry.csg import Sphere
    from rfx.grid import C0, Grid

    f0, lam = 3e9, C0 / 3e9
    radius = ka * lam / (2 * np.pi)
    res = max(15, int(np.ceil(2 * np.pi * cpr / ka)))
    dx = lam / res
    domain = 2 * radius + 2 * clear_cells * dx
    grid = Grid(freq_max=f0 * 1.5, domain=(domain,) * 3, dx=dx, cpml_layers=8)
    return grid, Sphere(center=(domain / 2,) * 3, radius=radius), radius


def test_cv16_conductor_is_a_node_sampled_sigma_fill_not_a_pec_volume():
    """cv16's sphere is a high-sigma MATERIAL FILL, and #931 §1.8 fences that
    out of the ownership contract: it stays a lossy volume model on NODE
    samples, while a PEC volume declared through ``Simulation.add`` is sampled
    at CELL CENTRES (§1.1). The two must not be silently equated.

    Asserted here on the real rasterizer at cv16's own gated operating points
    (build-time, no solve):

    * the sigma fill still equals ``Sphere.mask(grid)`` -- the model under
      which every committed ``a_eff``, both gate constants and the whole
      15-point curve were measured. cv16 refuses to quote an RCS otherwise;
    * ``n_occupied`` reproduces the count cv16's own GEOMETRY-FIDELITY
      CONVENTION note records for that bin, parsed out of the module docstring
      so the prose and the raster cannot drift apart (the committed fixture
      predates the ``a_eff`` convention and carries no ``n_occupied`` field --
      the docstring is where that measurement lives);
    * the PEC-volume cell set for the SAME sphere is DIFFERENT, and bigger:
      the node sampler is one cell short on every ``+`` side, so centre
      sampling realizes the sphere symmetric about its centre.

    If cv16 is ever brought under the contract, a_eff moves ~1.2% at ka=0.5,
    the Mie reference leg moves with it, and the fixture plus both gate
    constants need a re-solve. This test is what makes that a decision rather
    than a surprise.
    """
    import numpy as _np
    from rfx.geometry.csg import rasterize
    from rfx.geometry.rasterize_grid import (centres_from_uniform_grid,
                                             pec_volume_cell_mask)

    cv16 = _cv16_module()
    # "ka=0.50 a_eff/a=0.988032 (N=1082, -1.197%)" -- the GEOMETRY-FIDELITY
    # CONVENTION note in cv16's module docstring, the record of what the
    # gated curve's geometry actually was.
    quoted = {float(ka): (float(a), int(n)) for ka, a, n in re.findall(
        r"ka=([\d.]+)\s+a_eff/a=([\d.]+)\s+\(N=(\d+),", cv16.__doc__)}
    assert {0.5, 2.0} <= set(quoted), quoted

    for ka, cpr in [(0.5, 6.4), (2.0, 12.8)]:
        grid, sphere, radius = _cv16_operating_point(ka, cpr)
        _eps, sigma = rasterize(grid, [(sphere, 1.0, cv16.PEC_SIGMA)])
        info = cv16.assert_conductor_model(grid, sphere, sigma)

        a_quoted, n_quoted = quoted[ka]
        assert info["n_occupied_node"] == n_quoted, (
            f"ka={ka}: the node-sampled raster now gives "
            f"{info['n_occupied_node']} cells, cv16's own docstring records "
            f"N={n_quoted} -- the gated curve was measured on that geometry")
        assert info["n_occupied_pec_volume"] > info["n_occupied_node"], (
            f"ka={ka}: the centre-sampled PEC volume ("
            f"{info['n_occupied_pec_volume']}) is not larger than the "
            f"node-sampled fill ({info['n_occupied_node']}) -- the §1.1 "
            "symmetry claim (node sampling is one cell short on every + side) "
            "no longer holds, so this fence's stated consequence is stale")
        assert info["n_cells_differing"] > 0

        # The reason it matters, in the quantity cv16 actually gates on.
        dx = float(grid.dx)
        a_of = lambda n: (3 * n * dx ** 3 / (4 * _np.pi)) ** (1.0 / 3.0)  # noqa: E731
        a_node = a_of(info["n_occupied_node"]) / radius
        a_vol = a_of(info["n_occupied_pec_volume"]) / radius
        assert a_node == pytest.approx(a_quoted, abs=5e-6), (
            f"ka={ka}: realized a_eff/a={a_node:.6f}, docstring says "
            f"{a_quoted:.6f}")
        assert abs(a_vol - a_node) > 0.002, (
            f"ka={ka}: a_eff/a under the two models ({a_node:.6f} node vs "
            f"{a_vol:.6f} centre) now agree to better than 0.2%; the fence's "
            "cost claim needs re-measuring before it is re-quoted")
        # centre sampling is the symmetric one: its occupied bounding box
        # extends one cell further on the - side than the node sampler's.
        centre = _np.asarray(
            pec_volume_cell_mask(sphere, centres_from_uniform_grid(grid)),
            dtype=bool)
        node = _np.asarray(sphere.mask(grid), dtype=bool)
        lo_node = int(_np.nonzero(node.any(axis=(1, 2)))[0].min())
        lo_centre = int(_np.nonzero(centre.any(axis=(1, 2)))[0].min())
        assert lo_centre < lo_node, (ka, lo_centre, lo_node)


def test_cv16_refuses_when_the_fill_stops_matching_the_node_mask():
    """The fence is only evidence if it can fire: hand ``assert_conductor_model``
    a sigma array with one cell knocked out and it must refuse, not warn."""
    from rfx.geometry.csg import rasterize

    cv16 = _cv16_module()
    grid, sphere, _r = _cv16_operating_point(0.5, 6.4)
    _eps, sigma = rasterize(grid, [(sphere, 1.0, cv16.PEC_SIGMA)])
    tampered = np.array(sigma)
    ijk = tuple(int(v[0]) for v in np.nonzero(tampered > 0))
    tampered[ijk] = 0.0
    with pytest.raises(RuntimeError, match="node-sampled shape mask"):
        cv16.assert_conductor_model(grid, sphere, tampered)
