"""cv15's mandatory geometry self-check: the REALIZED electric-wall PLANES
must be exactly the declared ones, measured through the contract's one edge
set.

History. Issue #740: cv15's ``#325 AVOIDANCE (mandatory)`` self-check asserted
only the substrate's z EXTENT (``n_sub_raster == N_SUB``, from ``round(z/DX)``
cell counting) and kept PASSING while the ground conductor realized its
electric wall one full cell BELOW the declared substrate floor (the #693
"vacuum ground cell" trap) -- a live vacuum cell inside the modelled cavity,
undetected, +55.0% of the cavity's electrical thickness.

Issue #931 (the lattice ownership contract): the ground and the patch are FOIL
and are now DECLARED as sheets -- zero-thickness PEC ``Box``es on the two
substrate faces, the same structure both openEMS legs build. The check they
are under is no longer a hand-copied edge rule; ``assert_realized_stack`` reads
``rfx.boundaries.pec.realized_pec_edge_masks`` through ``realized_wall_planes``
and asserts three things: a wall at each declared plane, NO wall anywhere else
(in particular none at ``k_patch + 1``), and that the sheets wrote no material.

The geometry under test is built through cv15's OWN production builder,
``build_rfx_sim(*, do_gain, ground_plane_z, patch_kind)`` -- separated from
``run_rfx()`` for the #740 review (cv15 is classified ``audited`` in
``tests/_example_fidelity_lib.py`` on that builder). The positive tests pass NO
declaration arguments, so the script's own declarations are what is under test.
The first version of this file mirrored the geometry in a test-local copy that
hardcoded the fix, and deleting the fix from the script left it green -- the
reviewer's finding, and why the builder exists. ``assert_realized_stack`` and
``_stack_check_ok`` are called UNMODIFIED from the script itself, never copied.

Both negative controls run through that same production builder:

* ``ground_plane_z = z_sub_lo - DX`` -- the ground sheet declared one node
  plane low. It reproduces the pre-#740 realization (one wall below the floor,
  a live vacuum cell in the cavity) and keeps the geometry behind
  ``_15_patch_results/rfx_one_plane_ground_b29f9de7.json`` -- the +6.09%
  blindness evidence two public documents cite -- reachable through the public
  API after ``two_plane`` is deleted.
* ``patch_kind = "volume_1cell"`` -- the pre-#931 patch spelling, a one-cell
  PEC ``Box``. Under the contract a one-cell Box is a filled slab with BOTH
  faces, so it grows a wall at ``k_patch + 1`` (11.9062 mm) that the openEMS
  zero-thickness patch has no counterpart for. Without this arm the
  no-extra-wall assertion is decoration: nothing reachable would make it fire.

cv15 is guarded by ``if __name__ == "__main__":`` (see its final lines), so
importing it (to reach its module constants and the two functions above)
executes no simulation. Every test here is build-time -- ``_build_grid`` +
``_assemble_materials`` + the edge masks. No solve.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CV15_PATH = REPO_ROOT / "validation" / "crossval" / "15_patch_antenna_rt5880.py"

# The cv15 script is migrated by the crossval-C group in the same phase as this
# file: ground and patch become add_thin_conductor sheets at z_sub_lo / z_sub_hi,
# the two_plane parameter is deleted, assert_realized_stack is re-expressed on
# rfx.boundaries.pec.realized_wall_planes, and the rfx leg is re-solved
# (207 s CPU, num_periods 45 as committed) so validation/crossval/_15_patch_results/
# and the manifest claim_scope can be re-derived. VESSL run: rfx-931-post-cv15.
_MIGRATION_RUN = ("VESSL rfx-931-post-cv15 — 15_patch_antenna_rt5880.py "
                  "migrated to sheet declarations and re-solved (crossval-C)")


def _load_cv15():
    """Import cv15 as a module without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location("_cv15_wall_planes", CV15_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cv15_or_skip():
    """cv15, or a skip naming the migration that has to land first.

    The pre-#931 script imports ``two_plane_extension_masks`` from
    ``rfx.boundaries.pec`` and passes ``two_plane=`` to ``sim.add`` — both
    deleted by the contract — so on an unmigrated tree it does not import at
    all. Saying that once, here, is better than eight identical tracebacks.
    (The crossval-C migration has landed; this guard now only documents the
    dependency and keeps the skip text should the script ever regress.)
    """
    try:
        cv15 = _load_cv15()
    except Exception as exc:                       # ImportError / TypeError
        pytest.skip(f"cv15 has not been migrated to the #931 contract yet "
                    f"({type(exc).__name__}: {exc}); {_MIGRATION_RUN}")
    params = inspect.signature(cv15.build_rfx_sim).parameters
    if "two_plane" in params:
        pytest.skip("cv15's build_rfx_sim still takes two_plane=, which the "
                    f"contract deletes (design note §2); {_MIGRATION_RUN}")
    return cv15


def _build_test_sim(cv15, **kw):
    """Build cv15's geometry through the PRODUCTION builder,
    ``cv15.build_rfx_sim`` -- not a test-local mirror (#740 review, item 1:
    a mirrored copy hardcoded the fix, so deleting it from the script left
    every test green).

    No keyword (what every positive test passes) means the script's own
    declarations are under test: change either conductor's declaration in the
    script and the positive tests below go red. Only the negative controls
    pass a keyword. Returns ``(sim, grid, patch_shape)`` so the caller can run
    the script's OWN ``assert_realized_stack`` against it, cheaply (no solve).
    """
    sim, patch_shape, _geom = cv15.build_rfx_sim(do_gain=False, **kw)
    grid = sim._build_grid()
    return sim, grid, patch_shape


# ---------------------------------------------------------------------------
# assert_realized_stack: on the real rasterized geometry, via the contract's
# one edge set (cheap -- _build_grid + _assemble_materials, no solve).
# ---------------------------------------------------------------------------

def test_cv15_committed_geometry_realizes_declared_walls(capsys):
    """The committed geometry must realize its electric walls exactly at the
    declared z_sub_lo/z_sub_hi planes, across the whole patch footprint."""
    cv15 = _load_cv15()
    sim, grid, patch_shape = _build_test_sim(cv15)
    stack_check = cv15.assert_realized_stack(sim, grid, patch_shape)
    capsys.readouterr()

    assert stack_check["ground_wall_z"] == pytest.approx(cv15.AIR_BELOW, abs=1e-12)
    assert stack_check["patch_wall_z"] == pytest.approx(
        cv15.AIR_BELOW + cv15.H_SUB, abs=1e-12)
    assert stack_check["n_sub_cells"] == cv15.N_SUB
    assert len(stack_check["eps_between"]) == cv15.N_SUB
    assert all(e == pytest.approx(cv15.EPS_R, abs=1e-6)
               for e in stack_check["eps_between"])
    # Recorded provenance, not the thing gated on (see _stack_check_ok).
    assert stack_check["ground_realization"] == "sheet"
    assert stack_check["patch_realization"] == "sheet"


def test_cv15_both_conductors_are_declared_sheets(capsys):
    """#931 §1.3: a sheet owns NO cell. Both conductors must come back as
    DECLARED sheets on the two substrate faces, and NEITHER may contribute a
    cell -- the substrate is the only body that occupies cells here, so a
    conductor leaking into the cell mask would give the realized stack a face
    the openEMS zero-thickness reference has no counterpart for.

    Read through ``tests/_realized_geometry`` -- the branch's one spelling of
    the build-time realization check -- rather than a local ``argwhere`` over a
    mask, which is the drift the single-owner rule (§1.7) exists to stop. The
    sheet-declaration check and the wall-plane check below fail independently:
    one reads the declarations, the other the realized edge set.
    """
    import numpy as np

    from tests._realized_geometry import assert_sheet_planes, realized

    cv15 = _load_cv15()
    sim, _grid, _ps = _build_test_sim(cv15)
    z_sub_lo = cv15.AIR_BELOW
    z_sub_hi = cv15.AIR_BELOW + cv15.H_SUB

    assert_sheet_planes(sim, 2, [z_sub_lo, z_sub_hi],
                        what="cv15 ground and patch foil")
    rz = realized(sim)
    capsys.readouterr()
    assert not rz.wires
    assert rz.pec_mask is None or not bool(np.asarray(rz.pec_mask).any()), (
        "a declared sheet occupied cells -- it must own none (#931 §1.3)")


def test_cv15_no_wall_above_the_patch_plane(capsys):
    """The negative half of the stack check, asserted directly: over the patch
    footprint the realized z-wall planes are EXACTLY the two declared ones. A
    wall at ``k_patch + 1`` is the thing cv15 measured and rejected in 2026-08
    (11.9062 mm, no counterpart in the openEMS zero-thickness patch); before
    #931 its absence rested on a realization DEFAULT, and defaults are not
    evidence.

    Checked at the patch footprint's centre column AND its four rims -- the rim
    is where a closed-vs-half-open footprint disagreement would show, because
    the edge leaving the rim node points out of the patch.
    """
    import numpy as np

    from tests._realized_geometry import assert_wall_planes, node_index, realized

    cv15 = _load_cv15()
    sim, _grid, _ps = _build_test_sim(cv15)
    rz = realized(sim)
    capsys.readouterr()

    z_sub_lo = cv15.AIR_BELOW
    z_sub_hi = cv15.AIR_BELOW + cv15.H_SUB
    k_patch = node_index(rz.grid, 2, z_sub_hi)

    patch_fp = np.zeros(tuple(rz.grid.shape)[:2], dtype=bool)
    for sp in rz.sheets:
        if sp.normal_axis == 2 and sp.plane == k_patch:
            patch_fp |= np.asarray(sp.footprint).any(axis=2)
    assert patch_fp.any()

    ii, jj = np.nonzero(patch_fp)
    columns = {(int(ii.mean()), int(jj.mean())),
               (int(ii.min()), int(jj.min())), (int(ii.max()), int(jj.max())),
               (int(ii.min()), int(jj.max())), (int(ii.max()), int(jj.min()))}
    for (i, j) in columns:
        assert_wall_planes(sim, 2, [z_sub_lo, z_sub_hi], ij=(i, j),
                           what=f"cv15 patch footprint column ({i}, {j})")


def test_cv15_declares_foil_as_sheets_and_owns_no_cell(capsys):
    """#931 §1.3: 35 um copper on a 787 um laminate is foil, so it is a SHEET.

    The falsifier arm of the test above. A ground and patch declared as VOLUMES
    would still put a wall at z_sub_lo and z_sub_hi -- the positive test would
    pass -- while adding a second wall a cell away on each side and shorting the
    normal edge through the metal, which is the geometry the #720 A/B measured
    as the WORST agreement with the external reference (+0.265 correlation
    against +0.829 for the single-plane board, VESSL 369367256724). So the
    check is on the declaration: the conductors own no cell, and no wall stands
    one cell outside the laminate on either side.
    """
    cv15 = _cv15_or_skip()
    sim, _grid, _patch = _build_test_sim(cv15)
    capsys.readouterr()

    from tests._realized_pec import (assert_no_wall_at,
                                     assert_normal_edge_live,
                                     assert_sheet_owns_no_cell, realize)

    realized = realize(sim)
    assert len(realized.sheets) >= 2, (
        "cv15's ground and patch must be sheet declarations "
        f"(add_thin_conductor); the build carries {len(realized.sheets)}")
    assert_sheet_owns_no_cell(realized, what="cv15 foil")
    assert_normal_edge_live(realized, what="cv15 foil")
    z_lo = cv15.AIR_BELOW
    z_hi = cv15.AIR_BELOW + cv15.H_SUB
    assert_no_wall_at(realized, 2, [z_lo - cv15.DX, z_hi + cv15.DX],
                      what="cv15 board")


def test_cv15_negative_control_ground_sheet_one_plane_low_raises(capsys):
    """NEGATIVE CONTROL 1 (issue #740 review, required change 5; #931 respelt):
    declare the ground sheet one node plane BELOW the substrate floor through
    the PRODUCTION builder and confirm the script's OWN
    ``assert_realized_stack`` -- not a test-local copy -- raises, naming
    z_sub_lo.

    This is the fail-before-fix witness. Before #740 this exact geometry -- an
    all-one-plane ground wall a cell below the floor -- silently passed every
    check in the file and produced ``rfx_one_plane_ground_b29f9de7.json``'s
    +6.09% vs openEMS. Under #931 it is reachable as a wrong DECLARATION
    rather than as a different realization rule for the same declaration,
    which is the point of the contract.
    """
    cv15 = _load_cv15()
    sim, grid, patch_shape = _build_test_sim(
        cv15, ground_plane_z=cv15.AIR_BELOW - cv15.DX)
    with pytest.raises(RuntimeError, match="z_sub_lo"):
        cv15.assert_realized_stack(sim, grid, patch_shape)
    capsys.readouterr()


def test_cv15_negative_control_patch_as_one_cell_volume_raises(capsys):
    """NEGATIVE CONTROL 2 (#931): the pre-#931 patch spelling -- a one-cell PEC
    ``Box`` -- is a filled slab with BOTH faces under the contract, so it grows
    an unreferenced wall at ``k_patch + 1``. The check must refuse to quote f0
    and must NAME that plane, because the number a reader needs is which wall
    appeared, not that something was wrong.
    """
    cv15 = _load_cv15()
    sim, grid, patch_shape = _build_test_sim(cv15, patch_kind="volume_1cell")
    with pytest.raises(RuntimeError, match="one-cell VOLUME") as exc:
        cv15.assert_realized_stack(sim, grid, patch_shape)
    capsys.readouterr()
    # 11.9062 mm = z_sub_hi + DX, the plane this file measured and rejected in
    # 2026-08; printed as physical z, with the CPML pad offset removed.
    assert "11.9062" in str(exc.value), str(exc.value)


def test_cv15_builder_rejects_an_unknown_patch_kind():
    """The falsifier arm is a declaration switch, not a free-text field: a
    typo'd value must raise rather than silently fall back to production."""
    cv15 = _load_cv15()
    with pytest.raises(ValueError, match="patch_kind"):
        cv15.build_rfx_sim(patch_kind="one_plane")


# ---------------------------------------------------------------------------
# compare()'s stack-geometry gate: pinned with synthetic dicts (no solve),
# following test_crossval_gate_logic.py's precedent for this directory.
# ---------------------------------------------------------------------------

def _good_stack_check(cv15):
    return dict(
        ground_wall_z=cv15.AIR_BELOW,
        patch_wall_z=cv15.AIR_BELOW + cv15.H_SUB,
        n_sub_cells=cv15.N_SUB,
        eps_between=[cv15.EPS_R] * cv15.N_SUB,
        n_distinct_eps=cv15.N_DISTINCT_EPS_EXPECTED,
        ground_realization="sheet",
        patch_realization="sheet",
    )


def test_stack_check_ok_accepts_matching_measurement():
    cv15 = _cv15_or_skip()
    ok, detail = cv15._stack_check_ok(_good_stack_check(cv15))
    assert ok, detail


def test_stack_check_ok_rejects_missing_leg():
    """A leg from before the #740 fix has no `stack_check` key at all --
    that must FAIL, not be skipped (the whole #740 defect was a leg that
    looked fine without this check)."""
    cv15 = _cv15_or_skip()
    ok, detail = cv15._stack_check_ok(None)
    assert not ok
    assert "missing" in detail


def test_stack_check_ok_rejects_displaced_ground_wall():
    """The defect itself: a ground wall one cell below z_sub_lo must FAIL even
    if n_sub_cells/eps happen to look right."""
    cv15 = _cv15_or_skip()
    sc = _good_stack_check(cv15)
    sc["ground_wall_z"] = cv15.AIR_BELOW - cv15.DX
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail


def test_stack_check_ok_rejects_wrong_eps_between():
    cv15 = _cv15_or_skip()
    sc = _good_stack_check(cv15)
    sc["eps_between"] = [1.0] * cv15.N_SUB  # vacuum, not the declared laminate
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail


def test_stack_check_ok_rejects_a_leg_that_never_measured_the_materials():
    """#931: ``n_distinct_eps`` is the witness that the sheets wrote no
    material (the deleted #702 own-cell resample). A leg recorded before that
    key existed FAILS -- same reasoning as a missing ``stack_check``: the
    property was not measured, so it is not evidence. This is what makes the
    committed pre-#931 ``rfx.json`` fail compare() until it is regenerated,
    rather than passing on a stack nobody checked."""
    cv15 = _load_cv15()
    sc = _good_stack_check(cv15)
    del sc["n_distinct_eps"]
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail
    assert "n_distinct_eps" in detail


def test_stack_check_ok_rejects_a_third_material():
    """A third distinct eps means something re-sampled a conductor's own cell
    or a partial fill appeared -- either way the cavity is not the declared
    stack."""
    cv15 = _load_cv15()
    sc = _good_stack_check(cv15)
    sc["n_distinct_eps"] = cv15.N_DISTINCT_EPS_EXPECTED + 1
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail


def test_stack_check_ok_ignores_realization_label():
    """required change 1: ``ground_realization``/``patch_realization`` are
    recorded PROVENANCE only. A leg whose walls are correct but whose label
    says something else (a different mechanism landed the same planes) must
    still PASS."""
    cv15 = _load_cv15()
    sc = _good_stack_check(cv15)
    sc["ground_realization"] = "some_future_mechanism"
    sc["patch_realization"] = "some_future_mechanism"
    ok, detail = cv15._stack_check_ok(sc)
    assert ok, detail


def test_cv15_declaring_the_sheets_changes_no_material(capsys):
    """#931 §1.3: a sheet owns no cell and writes no material. Asserted the
    strong way -- the assembled ``eps_r`` array with both sheets declared must
    be BIT-IDENTICAL to the same build with the conductors removed, not merely
    "still two distinct values".

    This is cv17's G17-B pattern (``17_dielectric_sphere_mie.py``:
    ``check_realized_material``, exactly two eps values or the run is not about
    the declared material) carried onto the sheet side, and it is the witness
    for the DELETED #702 family -- "re-sample a 1-node sheet's own cell at its
    live edge", which existed precisely because the old ground conductor DID
    own a cell whose material had to be patched afterwards.

    The conductor-free build here is a deliberate test-local control: it is the
    thing WITHOUT the declarations, so it cannot go stale when the script's
    declarations change (the #740 review's objection was to mirroring the thing
    UNDER TEST, which this file does not do -- every other test drives
    ``build_rfx_sim``).
    """
    import numpy as np

    from rfx import Box, Simulation
    from rfx.boundaries.spec import BoundarySpec

    cv15 = _load_cv15()
    sim, grid, _ = _build_test_sim(cv15)
    mats, *_ = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])

    cx, cy = cv15.DOM_X / 2, cv15.DOM_Y / 2
    z_sub_lo = cv15.AIR_BELOW
    z_sub_hi = (10 + cv15.N_SUB) * cv15.DX
    bare = Simulation(
        freq_max=4e9, domain=(cv15.DOM_X, cv15.DOM_Y, cv15.DOM_Z), dx=cv15.DX,
        boundary=BoundarySpec.uniform("cpml"), cpml_layers=cv15.N_CPML,
    )
    bare.add_material("sub", eps_r=cv15.EPS_R, sigma=cv15.SIGMA_SUB)
    bare.add(Box((cx - cv15.GP_X / 2, cy - cv15.GP_Y / 2, z_sub_lo),
                 (cx + cv15.GP_X / 2, cy + cv15.GP_Y / 2, z_sub_hi)),
             material="sub")
    bare_mats, *_ = bare._assemble_materials(
        bare._build_grid(), pec_sheets=[], pec_wires=[])
    capsys.readouterr()

    assert np.array_equal(np.asarray(mats.eps_r), np.asarray(bare_mats.eps_r)), (
        "declaring the two PEC sheets changed the permittivity array -- a "
        "sheet owns no cell and writes no material (#931 §1.3)")
    assert np.array_equal(np.asarray(mats.sigma), np.asarray(bare_mats.sigma)), (
        "declaring the two PEC sheets changed the conductivity array")


def test_cv15_feed_decomposition_arm_reproduces_the_pre931_port(capsys):
    """#931 changed the conductor DECLARATIONS and the FEED in one step, and
    the committed f0 moved 4.71 % (2.313947 -> 2.423039 GHz). Two changes, one
    number: the attribution needs a measurement, not an argument.

    ``build_rfx_sim(feed="pre931")`` is that measurement's other arm -- the
    production sheets with the OLD port, starting 1.0*DX above the substrate
    floor and spanning 2*DX ("cells strictly between GP & patch"). Pinned here
    so the arm cannot rot into something that is no longer the old feed, which
    would make the decomposition meaningless while still producing a number.

    The stack check must pass on BOTH arms: the feed is not supposed to touch
    the realized walls (a port releases only the one component it drives,
    design note §6), and if it did, the decomposition would be measuring two
    things again.
    """
    cv15 = _load_cv15()
    _sim, _ps, geom = cv15.build_rfx_sim(do_gain=False, feed="full_span")
    assert geom["port_z0"] == pytest.approx(cv15.AIR_BELOW, abs=1e-12)
    assert geom["port_extent"] == pytest.approx(cv15.H_SUB, abs=1e-12)

    sim, grid, patch_shape = _build_test_sim(cv15, feed="pre931")
    _s, _p, geom_old = cv15.build_rfx_sim(do_gain=False, feed="pre931")
    assert geom_old["port_z0"] == pytest.approx(
        cv15.AIR_BELOW + 1.0 * cv15.DX, abs=1e-12)
    assert geom_old["port_extent"] == pytest.approx(2.0 * cv15.DX, abs=1e-12)

    sc = cv15.assert_realized_stack(sim, grid, patch_shape)
    capsys.readouterr()
    assert sc["ground_realization"] == "sheet"
    assert sc["patch_realization"] == "sheet"
    assert sc["n_distinct_eps"] == cv15.N_DISTINCT_EPS_EXPECTED


def test_cv15_builder_rejects_an_unknown_feed():
    cv15 = _load_cv15()
    with pytest.raises(ValueError, match="feed"):
        cv15.build_rfx_sim(feed="two_plane")


def test_cv15_current_measurement_prose_follows_committed_legs():
    """Regeneration must update the case banner and its decomposition rationale."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    results = root / "validation/crossval/_15_patch_results"
    before = json.loads((results / "rfx_pre931_two_plane_ground_1f005d0d.json").read_text())
    after = json.loads((results / "rfx.json").read_text())
    script = (root / "validation/crossval/15_patch_antenna_rt5880.py").read_text()
    shift = (after["f_primary_hz"] / before["f_primary_hz"] - 1) * 100
    pair = f'{before["f_primary_hz"] / 1e9:.6f} -> {after["f_primary_hz"] / 1e9:.6f} GHz'
    for carrier in (script, Path(__file__).read_text()):
        assert f"{shift:.2f} % ({pair})" in carrier
    assert f'{after["s11_dip_db"]:.2f} dB at {after["f_dip_hz"] / 1e9:.3f} GHz' in script
    assert f'ring-down frequency is {after["f_primary_hz"] / 1e9:.6f} GHz' in script
# ---------------------------------------------------------------------------
# compare()'s GALVANIC-FEED gate (issue #920, respelt for #931 sheets) -- the
# feed-side twin of the wall-plane gate above. #920's original criterion was
# CELL-based (the port's first/last rasterized cell is PEC, i.e. dead/shorted
# by the conductor it sits inside); under the ownership contract a sheet's
# normal component never goes PEC (design note #931 sec1.3, "normal E through
# a sheet stays live" -- Mz is unchanged by a z-normal sheet), so that
# criterion is unsatisfiable for cv15 (no volume conductor at all in the
# production build) and was respelt as a PLANE criterion: the feed's two
# endpoints must themselves be realized conductor wall planes, read through
# the SAME realized_pec_edge_masks/realized_wall_planes pair
# assert_realized_stack uses. Same synthetic-dict style as the stack gate
# above -- no solve.
# ---------------------------------------------------------------------------

def _good_feed_check(cv15):
    """The classification a galvanic post produces, expressed in the module's
    own constants (never this board's cell indices): the feed spans
    z_sub_lo (the ground sheet's realized plane) to z_sub_hi (the patch
    sheet's), and both ends must be RECORDED as landing on a realized wall
    plane -- which assert_galvanic_feed only sets True after checking the
    real edge set, never assumed from the span numbers alone."""
    z0 = cv15.AIR_BELOW
    z1 = cv15.AIR_BELOW + cv15.H_SUB
    return dict(
        port_z0=z0, port_extent=z1 - z0,
        z0_node_k=0, z1_node_k=0,
        z0_on_realized_plane=True, z1_on_realized_plane=True,
        galvanic=True,
    )


def test_feed_check_ok_accepts_the_galvanic_classification():
    cv15 = _load_cv15()
    ok, detail = cv15._feed_check_ok(_good_feed_check(cv15))
    assert ok, detail


def test_feed_check_ok_accepts_the_committed_leg():
    """The synthetic dict above must be the same shape the SHIPPING leg
    carries -- otherwise the gate math is pinned against a fiction."""
    cv15 = _load_cv15()
    import json
    leg = json.loads(
        (REPO_ROOT / "validation/crossval/_15_patch_results/rfx.json")
        .read_text(encoding="utf-8"))
    ok, detail = cv15._feed_check_ok(leg["feed_check"])
    assert ok, detail


def test_feed_check_ok_rejects_missing_leg():
    """A leg from before #920/#931 has no ``feed_check`` key -- FAIL, not
    skip. This is the gap the item-A review measured: the archived
    floating-post leg (no ``feed_check``) otherwise passes every gate."""
    cv15 = _load_cv15()
    ok, detail = cv15._feed_check_ok(None)
    assert not ok
    assert "missing" in detail


def test_feed_check_ok_rejects_the_archived_floating_post_leg():
    """The measured #920 defect itself, through the committed artifact: the
    archived leg predates the self-check entirely, so it has no
    ``feed_check`` key at all."""
    cv15 = _load_cv15()
    import json
    old = json.loads(
        (REPO_ROOT / "validation/crossval/_15_patch_results"
         / "rfx_floating_post_1f005d0d.json").read_text(encoding="utf-8"))
    ok, detail = cv15._feed_check_ok(old.get("feed_check"))
    assert not ok, detail


def test_feed_check_ok_rejects_a_displaced_span():
    """A leg whose recorded flags claim both ends are on realized planes but
    whose actual span numbers have moved off the two conductors' planes --
    the gate rests on the NUMERIC re-check against this module's own
    constants, not on the recorded flags alone."""
    cv15 = _load_cv15()
    fc = _good_feed_check(cv15)
    fc["port_z0"] = fc["port_z0"] + cv15.DX
    ok, detail = cv15._feed_check_ok(fc)
    assert not ok, detail


def test_feed_check_ok_rejects_a_self_declared_galvanic_flag():
    """``galvanic``/``z0_on_realized_plane``/``z1_on_realized_plane`` are
    recorded labels; the gate must not rest on any one of them alone. A leg
    with the correct span but a dropped plane flag FAILS, and the converse
    -- honest plane flags with the ``galvanic`` label dropped -- also FAILS,
    so no single field can rescue or be rescued by the others."""
    cv15 = _load_cv15()
    fc = _good_feed_check(cv15)
    fc["z0_on_realized_plane"] = False
    ok, _ = cv15._feed_check_ok(fc)
    assert not ok
    fc2 = _good_feed_check(cv15)
    fc2["galvanic"] = False
    ok2, _ = cv15._feed_check_ok(fc2)
    assert not ok2


# ---------------------------------------------------------------------------
# assert_galvanic_feed on the REAL rasterized geometry (no solve), and the
# round trip into the gate.
# ---------------------------------------------------------------------------

def test_cv15_committed_geometry_rasterizes_a_galvanic_feed(capsys):
    """The production builder's port must classify as galvanic on the real
    assembled geometry, and what it records must satisfy ``compare()``'s
    gate -- the two halves of the #920/#931 fix pinned against each other."""
    cv15 = _load_cv15()
    sim, _patch_shape, geom = cv15.build_rfx_sim(do_gain=False)
    grid = sim._build_grid()
    fc = cv15.assert_galvanic_feed(sim, grid, geom)
    capsys.readouterr()

    assert fc["galvanic"] is True
    assert fc["z0_on_realized_plane"] is True
    assert fc["z1_on_realized_plane"] is True
    assert fc["port_z0"] == pytest.approx(cv15.AIR_BELOW, abs=1e-12)
    assert fc["port_extent"] == pytest.approx(cv15.H_SUB, abs=1e-12)
    ok, detail = cv15._feed_check_ok(fc)
    assert ok, detail


def test_cv15_negative_control_pre920_span_is_refused(capsys):
    """FAIL-BEFORE-FIX for #920/#931, through the script's OWN assert: feed
    it the span cv15 shipped before either fix -- the DECOMPOSITION arm's
    ``feed="pre931"``, one cell above the floor and spanning two INTERIOR
    substrate cells, touching neither conductor's plane -- and it must
    refuse. Driven through the PRODUCTION builder (``build_rfx_sim``), not a
    hand-rolled geom dict, for the same #740-review reason every other test
    in this file does: a mirrored copy cannot catch a script regression.
    """
    cv15 = _load_cv15()
    sim, _patch_shape, geom = cv15.build_rfx_sim(do_gain=False, feed="pre931")
    grid = sim._build_grid()
    with pytest.raises(RuntimeError, match="assert_galvanic_feed"):
        cv15.assert_galvanic_feed(sim, grid, geom)
    capsys.readouterr()


# ---------------------------------------------------------------------------
# The archived pre-fix leg has TWO names -- deliberately, not a merge leftover
# to clean up. Guard against them silently drifting apart.
# ---------------------------------------------------------------------------

def test_the_two_archived_pre_fix_leg_names_stay_byte_identical():
    """``rfx_floating_post_1f005d0d.json`` (main's name, cited by CHANGELOG.md)
    and ``rfx_pre931_two_plane_ground_1f005d0d.json`` (this branch's name,
    cited by docs/design_notes/20260908_docs_truth_field_ledger.md,
    RECOMPUTE.md and scripts/diagnostics/cv15_before_after_931.py) are the
    SAME artifact -- the #768 leg, before #920's galvanic-feed fix and before
    #931's sheet declarations, archived independently by two campaigns that
    each named it after the defect they were chasing (a floating feed post;
    a two_plane-ground realization). Both descriptions are accurate; that is
    the point, not an error to resolve by picking one.

    Neither file is to be deleted or renamed: each has its own dated,
    committed document citing it by that name, and rewriting a dated record
    to point at a file that no longer exists under the name it used is worse
    than the duplication. What must never happen instead is the two
    diverging -- someone regenerates one and not the other, and both
    campaigns' evidence silently stops describing the same board. This
    fails loudly, cheaply, the moment that happens.
    """
    import hashlib

    results = REPO_ROOT / "validation/crossval/_15_patch_results"
    a = results / "rfx_floating_post_1f005d0d.json"
    b = results / "rfx_pre931_two_plane_ground_1f005d0d.json"
    assert a.is_file(), f"{a} is missing -- do not delete either archived name"
    assert b.is_file(), f"{b} is missing -- do not delete either archived name"
    ha = hashlib.sha256(a.read_bytes()).hexdigest()
    hb = hashlib.sha256(b.read_bytes()).hexdigest()
    assert ha == hb, (
        f"the two archived names for the #768 pre-fix cv15 leg have "
        f"diverged: {a.name} sha256={ha} vs {b.name} sha256={hb}. They "
        f"must stay byte-identical -- both names are cited by dated "
        f"documents (CHANGELOG.md for the first, "
        f"20260908_docs_truth_field_ledger.md/RECOMPUTE.md/"
        f"cv15_before_after_931.py for the second) as the SAME artifact. "
        f"If the board was legitimately re-measured, regenerate BOTH files "
        f"identically, or retire one name explicitly (updating every "
        f"document that cites it) rather than letting them silently split."
    )

