"""The crossval directory's own witness for the #931 lattice ownership contract.

``tests/contracts/test_lattice_ownership_contract.py`` proves the contract at
the primitive level. This file is the crossval-level statement of the two
sentences every case here now depends on, exercised through the SAME reader the
cases use (``tests/_realized_pec``), so a regression shows up as one named
failure rather than as four unexplained resonance shifts in cv05, cv06b, cv07
and cv15 at once.

  VOLUME (§1.2)  a Box drawn ``z_a -> z_b`` realizes tangential walls at BOTH
                 planes and shorts every normal edge between them; realized
                 thickness equals drawn thickness at t = 1, 2, 3 cells and on
                 every axis.
  SHEET (§1.3)   a declared sheet owns no cell, writes no material, realizes
                 exactly one plane, and leaves the normal edge through it live.

Both come with their falsifier arm, because a check that only looks for the
walls it wants cannot see an extra one — and an extra wall a cell away is
exactly what a foil mis-declared as a volume produces.

The second half of the file is the cross-case guard the inventory asked for:
before #931 this directory carried six independent derivations of "where did
the metal land" (cv15's two_plane, cv19's ±1 pair, cv18's aperture −1, cv09's
mirror identity, cv10's half-cell PMC plane, cv06b's realized trace width),
three of which contradicted each other about the same WR-90 iris and the same
microstrip trace. The contract's "one realized-edge function, no case-specific
knobs" clause is unenforceable without a test, and it is the clause most likely
to rot first, because every new case has a local reason to re-derive.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from rfx import Box, Simulation
from tests._realized_pec import (assert_no_wall_at, assert_normal_edge_live,
                                 assert_sheet_owns_no_cell,
                                 assert_sheet_planes, assert_walls_at, realize,
                                 wall_planes, wall_positions)

REPO_ROOT = Path(__file__).resolve().parents[2]

DX = 1e-3
DOMAIN = (0.02, 0.02, 0.02)
LO = 0.006          # 6 cells in from the domain origin, clear of the CPML pad
FOOTPRINT = (0.005, 0.015)


def _sim():
    return Simulation(freq_max=10e9, domain=DOMAIN, dx=DX, boundary="cpml",
                      cpml_layers=6)


def _body(axis: int, thickness_cells: int, *, sheet: bool = False):
    """A slab normal to ``axis``, drawn on node planes, in a fresh build."""
    lo = [FOOTPRINT[0]] * 3
    hi = [FOOTPRINT[1]] * 3
    lo[axis] = LO
    hi[axis] = LO if sheet else LO + thickness_cells * DX
    sim = _sim()
    shape = Box(tuple(lo), tuple(hi))
    if sheet:
        sim.add_thin_conductor(shape)
    else:
        sim.add(shape, material="pec")
    return sim


@pytest.mark.parametrize("axis", (0, 1, 2), ids=("x", "y", "z"))
@pytest.mark.parametrize("t", (1, 2, 3))
def test_a_volume_realizes_both_drawn_faces_at_every_thickness(axis, t):
    """§1.2, read through the reader the cases use.

    The pre-#931 rule stood one wall per masked cell at that cell's LOWER node
    plane and never zeroed the top face, so a 2-cell body was NOT a repair for
    a 1-cell one — measured, both gave the same 983.7 um cavity for a 787.0 um
    laminate. Hence the sweep over t: a rule that is only right at one
    thickness is the defect this contract replaces.
    """
    sim = _body(axis, t)
    realized = realize(sim)
    z_lo, z_hi = LO, LO + t * DX
    # The footprint is the body's own cells: a wall is asserted where the body
    # is, not across the whole plane.
    assert_walls_at(realized, axis, [z_lo, z_hi], footprint=realized.cells,
                    what=f"t={t} slab")
    got = wall_positions(realized, axis)
    assert got[0] == pytest.approx(z_lo, abs=1e-12)
    assert got[-1] == pytest.approx(z_hi, abs=1e-12)
    assert got[-1] - got[0] == pytest.approx(t * DX, abs=1e-12), (
        f"realized thickness {1e3 * (got[-1] - got[0]):.4f} mm against a drawn "
        f"{t * DX * 1e3:.4f} mm")
    # falsifier: nothing one cell outside either drawn face
    assert_no_wall_at(realized, axis, [z_lo - DX, z_hi + DX],
                      what=f"t={t} slab")
    # the interior is shorted: the normal component is PEC everywhere between
    # the two faces, over the footprint
    m = np.asarray(realized.edges[axis], dtype=bool)
    occupied = np.asarray(realized.cells, dtype=bool)
    assert bool(m[occupied].all()), (
        "a normal edge is live inside the slab — the body is not filled")


@pytest.mark.parametrize("axis", (0, 1, 2), ids=("x", "y", "z"))
def test_a_sheet_realizes_one_plane_and_owns_nothing(axis):
    """§1.3, with the falsifier arm the volume test cannot supply.

    A sheet and a one-cell volume agree about the plane the foil is ON. They
    disagree about everything else: the volume adds a second wall a cell away,
    owns a cell of the mesh, and shorts the normal edge through the metal.
    Those three differences are what moved cv05's resonance by 15 percentage
    points, so they are checked one at a time.
    """
    sim = _body(axis, 0, sheet=True)
    realized = realize(sim)
    assert realized.cells is None or not bool(np.asarray(realized.cells).any())
    assert_sheet_owns_no_cell(realized)
    assert_normal_edge_live(realized)
    assert_sheet_planes(realized, axis, [LO])
    assert len(wall_planes(realized, axis)) == 1, (
        f"a sheet realized {wall_planes(realized, axis)} planes; it declares one")
    assert_no_wall_at(realized, axis, [LO - DX, LO + DX], what="sheet")
    # and it writes no material
    eps = np.asarray(realized.materials.eps_r)
    assert np.allclose(eps, 1.0), (
        "a sheet wrote permittivity into the mesh; it owns no cell (§1.3)")


def test_a_sheet_and_a_one_cell_volume_are_distinguishable():
    """The two declarations of the same drawn foil must NOT realize the same
    thing — otherwise the declaration carries no information and the whole
    contract is a naming exercise."""
    sheet = realize(_body(2, 0, sheet=True))
    volume = realize(_body(2, 1))
    assert wall_planes(sheet, 2) != wall_planes(volume, 2)
    assert len(wall_planes(volume, 2)) == len(wall_planes(sheet, 2)) + 1


# --------------------------------------------------------------------------- #
# Cross-case guard: no case may re-derive the realization for itself
# --------------------------------------------------------------------------- #

# Tokens that only ever appear in a case-specific realization derivation. Each
# is a mechanism the contract deleted, not a coincidence of wording.
_RETIRED = (
    "two_plane",
    "two_plane_extension_masks",
    "tangential_edge_masks",
    "resample_sheet_node_materials",
    "sheet_normal_live_axis_masks",
    "collect_thin_conductor_sheet_inputs",
    "_two_plane_cell_mask",
)

# Files that may name a retired token forever, with the reason. Two kinds only:
# a file that names it in order to REFUSE it, and a file whose subject is the
# history of its removal.
_ALLOWED = {
    "tests/crossval/test_lattice_ownership_crossval_witness.py":
        "this file — it names the tokens in order to forbid them",
    "tests/crossval/test_meep_crossval.py":
        "asserts the two retired symbols are GONE from rfx.boundaries.pec "
        "(`for gone in (...): assert not hasattr(...)`), so it must name them; "
        "it imports neither and re-implements nothing",
    # --- prose only: each names the retired flag to say what it USED to do
    # and why the committed number moves. None imports or re-derives anything;
    # all read realized_pec_edge_masks / realized_wall_planes.
    "validation/crossval/15_patch_antenna_rt5880.py":
        "docstring history of the #740 ground patch (:65, :79, :112) and one "
        "line at :264 recording that the PREVIOUS version OR'd the base "
        "tangential_edge_masks with the extension; the live code calls neither",
    "validation/crossval/18_wr90_iris_modematch.py":
        "two docstring sentences (:516, :967) explaining that the far face was "
        "never a wall and that the flag put it back for t = 1 only",
    "tests/crossval/test_patch_canonical_farfield_e4.py":
        "one comment (:147) citing the #740 arm's -4.7% as the prediction the "
        "post-#931 measurement was checked against",
    "tests/crossval/test_crossval_cv15_wall_planes.py":
        "documents the #740 mechanism as superseded history and skips on a "
        "tree that still exposes it",
    "tests/crossval/test_patch_mode_identification.py":
        "resolves two frozen #740 ring-down legs by role; the old key "
        "spelling is the fallback",
    # --- group X-A (cv05). Prose only: both files name the retired flag to
    # cite the 2026-08-28 A/B verdict that explains WHY cv05's agreement with
    # openEMS is expected to move. Neither imports or re-implements anything;
    # both read realized_pec_edge_masks / realized_wall_planes.
    "tests/crossval/test_cv05_realized_sheet_planes.py":
        "one docstring sentence contrasting a foil with the retired flag and "
        "the one-cell PEC Box, which is what the test exists to distinguish",
    "validation/crossval/05_patch_antenna.py":
        "cites the 2026-08-28 two_plane A/B verdict in its header as the "
        "recorded reason the openEMS agreement moves; no flag, no mechanism",
    "tests/crossval/test_rcs_dielectric_sphere_mie_gates.py":
        "asserts the three #702 helpers are GONE from "
        "rfx.geometry.rasterize_grid (not hasattr) — it names them to "
        "refuse them, which is this guard's own job one level down",
    "tests/crossval/test_cv23_lossy_slab_gates.py":
        "names resample_sheet_node_materials as the retired mechanism its "
        "one-cell-lossy-body pin used to guard against; history in a "
        "docstring, no call",
}


def _scan(root: Path):
    hits = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in _ALLOWED:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in _RETIRED:
            if re.search(rf"\b{re.escape(token)}\b", text):
                hits.append((rel, token))
    return hits


def _tree_is_migrated() -> bool:
    """False while a crossval script still imports a deleted pec helper.

    The crossval-C migration of ``15_patch_antenna_rt5880.py`` is the last
    holder; until it lands the script does not even import, so this guard
    would fire on work that is already scheduled rather than on a regression.
    """
    cv15 = REPO_ROOT / "validation/crossval/15_patch_antenna_rt5880.py"
    if not cv15.exists():
        return True
    return "two_plane_extension_masks" not in cv15.read_text(encoding="utf-8")


def test_no_case_re_derives_the_realization_for_itself():
    """One realized-edge function, no per-case knobs (design note §1.7).

    Six derivations lived in this directory and three of them disagreed. The
    contract forbids that, and a clause with no test is a clause that rots —
    each new case has a local reason to re-derive, and each one is individually
    reasonable. So the retired mechanisms are named and forbidden, with an
    allowlist that has to state a reason.
    """
    if not _tree_is_migrated():
        pytest.skip(
            "validation/crossval/15_patch_antenna_rt5880.py still imports the "
            "deleted two_plane_extension_masks; the crossval-C migration "
            "(VESSL rfx-931-post-cv15) lands before this guard can "
            "distinguish scheduled work from a regression")
    hits = _scan(REPO_ROOT / "tests/crossval")
    hits += _scan(REPO_ROOT / "validation/crossval")
    assert not hits, (
        "case-specific realization mechanics are back:\n"
        + "\n".join(f"  {rel}: {token}" for rel, token in hits)
        + "\nEvery consumer reads rfx.boundaries.pec.realized_pec_edge_masks / "
          "realized_wall_planes / edge_is_pec (#931 §1.7). If a file must name "
          "one of these tokens as history or to refuse it, add it to _ALLOWED "
          "with the reason.")


def test_the_allowlist_entries_still_exist_and_still_need_the_exception():
    """An allowlist that outlives its entries is a hole, not a list."""
    for rel, reason in _ALLOWED.items():
        path = REPO_ROOT / rel
        assert path.exists(), f"allowlisted file {rel} is gone; drop the entry"
        assert reason.strip(), rel
        text = path.read_text(encoding="utf-8")
        assert any(re.search(rf"\b{re.escape(t)}\b", text) for t in _RETIRED), (
            f"{rel} no longer names any retired token; remove its allowlist "
            "entry so the guard covers it again")
