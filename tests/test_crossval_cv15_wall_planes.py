"""Regression test for issue #740: cv15's mandatory geometry self-check must
assert the REALIZED electric-wall PLANES, not the declared Box extents.

``validation/crossval/15_patch_antenna_rt5880.py``'s pre-#740 ``#325
AVOIDANCE (mandatory)`` self-check asserted only the substrate's z EXTENT
(``n_sub_raster == N_SUB``, from ``round(z/DX)`` cell counting) and kept
PASSING while the one-cell one-plane ground ``Box`` realized its electric
wall one full cell BELOW the declared substrate floor (the #693 "vacuum
ground cell" trap, closed on the canonical patch lane by PRs #716/#718) --
a live vacuum cell inside the modelled cavity, undetected.

cv15's ``run_rfx()`` is classified ``builder_fused_with_solve`` in
``tests/_example_fidelity_lib.py`` (build and FDTD-solve share one function,
machine-checked by ``tests/test_example_fidelity_contract.py`` via
``functions_building_simulation`` -- a top-level function whose OWN body
constructs a ``Simulation`` AND calls a solve entrypoint). Introducing a
separate top-level "builder" function would flip that classification (and
this repo's #740 review explicitly warns against silent gate/classification
drift), so this test does NOT call into a extracted builder. Instead, and
following ``tests/test_crossval_gate_logic.py``'s own precedent for cv04
("cv04 runs its FDTD and gate computation entirely at MODULE level ... its
ceiling/tail logic and constants are replicated inline instead"), the
geometry here is rebuilt directly from cv15's PUBLIC module constants
(``EPS_R``, ``H_SUB``, ``L_PATCH``, ... -- all imported, none re-derived) to
mirror exactly what ``run_rfx()`` constructs, cited by FUNCTION NAME rather
than line number (this campaign's own lesson: "freezing by line number
breaks on unrelated edits" -- #753/#754). ``assert_realized_stack`` and
``_stack_check_ok`` -- the actual #740 fix -- are imported and called
UNMODIFIED from the script itself, never copied.

cv15 is guarded by ``if __name__ == "__main__":`` (see its final lines), so
importing it (to reach its module constants and the two functions above)
executes no simulation.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

from rfx import Simulation, Box, GaussianPulse
from rfx.boundaries.spec import BoundarySpec

REPO_ROOT = Path(__file__).resolve().parents[1]
CV15_PATH = REPO_ROOT / "validation" / "crossval" / "15_patch_antenna_rt5880.py"


def _load_cv15():
    """Import cv15 as a module without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location("_cv15_wall_planes", CV15_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_test_sim(cv15):
    """Rebuild the SAME geometry ``run_rfx()`` constructs (mirrored, not
    imported -- that function is fused with the FDTD solve, see module
    docstring), from cv15's own public module constants only. Returns
    ``(sim, grid, patch_shape)`` so the caller can run the script's OWN
    ``assert_realized_stack`` against it, cheaply (no solve)."""
    cx, cy = cv15.DOM_X / 2, cv15.DOM_Y / 2
    z_sub_lo = cv15.AIR_BELOW
    z_sub_hi = z_sub_lo + cv15.H_SUB
    z_patch_lo, z_patch_hi = z_sub_hi, z_sub_hi + cv15.DX
    sim = Simulation(
        freq_max=4e9, domain=(cv15.DOM_X, cv15.DOM_Y, cv15.DOM_Z), dx=cv15.DX,
        boundary=BoundarySpec.uniform("cpml"), cpml_layers=cv15.N_CPML,
    )
    sim.add_material("sub", eps_r=cv15.EPS_R, sigma=cv15.SIGMA_SUB)
    sim.add(Box((cx - cv15.GP_X / 2, cy - cv15.GP_Y / 2, z_sub_lo - cv15.DX),
                (cx + cv15.GP_X / 2, cy + cv15.GP_Y / 2, z_sub_lo)),
            material="pec", two_plane=True)
    sim.add(Box((cx - cv15.GP_X / 2, cy - cv15.GP_Y / 2, z_sub_lo),
                (cx + cv15.GP_X / 2, cy + cv15.GP_Y / 2, z_sub_hi)),
            material="sub")
    patch_shape = Box(
        (cx - cv15.L_PATCH / 2, cy - cv15.W_PATCH / 2, z_patch_lo),
        (cx + cv15.L_PATCH / 2, cy + cv15.W_PATCH / 2, z_patch_hi))
    sim.add(patch_shape, material="pec")
    feed_x = cx + cv15.FEED_OFFSET_X
    port_z0 = z_sub_lo + 1.0 * cv15.DX
    port_extent = 2.0 * cv15.DX
    sim.add_port(position=(feed_x, cy, port_z0), component="ez",
                 impedance=50.0, extent=port_extent,
                 waveform=GaussianPulse(f0=cv15.F_DESIGN, bandwidth=1.0))
    grid = sim._build_grid()
    return sim, grid, patch_shape


# ---------------------------------------------------------------------------
# assert_realized_stack: the actual #740 fix, on the real rasterized
# geometry (cheap -- _build_grid + _assemble_materials, no solve).
# ---------------------------------------------------------------------------

def test_cv15_committed_geometry_realizes_declared_walls(capsys):
    """The committed (post-#740-fix) geometry must realize its electric
    walls exactly at the declared z_sub_lo/z_sub_hi planes, across the
    whole patch footprint -- watched PASS on today's tree."""
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
    assert stack_check["ground_realization"] == "two_plane"


def test_cv15_negative_control_one_plane_ground_raises(capsys):
    """NEGATIVE CONTROL (issue #740 review, required change 5): force the
    ground geometry entry back to the pre-fix one-plane realization (via
    ``dataclasses.replace`` on the frozen ``_GeometryEntry`` built above)
    and confirm the script's OWN ``assert_realized_stack`` -- not a
    test-local copy -- raises, naming z_sub_lo.

    This is the fail-before-fix witness for THIS PR: on the pre-#740-fix
    source (ground ``Box`` built with no ``two_plane=True``, and no
    ``assert_realized_stack`` at all -- confirmed by running this whole
    file against the unmodified script via the copy-aside/git-checkout
    protocol: AttributeError on every test, 0 passed; see this PR's
    fail-before-fix run log) this exact scenario -- an all-one-plane
    ground wall -- is what silently passed. Forcing the flag back off
    here and calling the FIXED script's ``assert_realized_stack``
    reproduces that same displaced-wall geometry and confirms the NEW
    check catches it.
    """
    cv15 = _load_cv15()
    sim, grid, patch_shape = _build_test_sim(cv15)

    ground_entry = sim._geometry[0]
    assert ground_entry.material_name == "pec"
    assert ground_entry.two_plane is True  # the #740 fix, before we undo it
    sim._geometry[0] = dataclasses.replace(ground_entry, two_plane=False)

    with pytest.raises(RuntimeError, match="z_sub_lo"):
        cv15.assert_realized_stack(sim, grid, patch_shape)
    capsys.readouterr()


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
        ground_realization="two_plane",
    )


def test_stack_check_ok_accepts_matching_measurement():
    cv15 = _load_cv15()
    ok, detail = cv15._stack_check_ok(_good_stack_check(cv15))
    assert ok, detail


def test_stack_check_ok_rejects_missing_leg():
    """A leg from before the #740 fix has no `stack_check` key at all --
    that must FAIL, not be skipped (the whole #740 defect was a leg that
    looked fine without this check)."""
    cv15 = _load_cv15()
    ok, detail = cv15._stack_check_ok(None)
    assert not ok
    assert "missing" in detail


def test_stack_check_ok_rejects_displaced_ground_wall():
    """The pre-fix one-plane-ground defect itself: ground wall one cell
    below z_sub_lo must FAIL even if n_sub_cells/eps happen to look right."""
    cv15 = _load_cv15()
    sc = _good_stack_check(cv15)
    sc["ground_wall_z"] = cv15.AIR_BELOW - cv15.DX
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail


def test_stack_check_ok_rejects_wrong_eps_between():
    cv15 = _load_cv15()
    sc = _good_stack_check(cv15)
    sc["eps_between"] = [1.0] * cv15.N_SUB  # vacuum, not the declared laminate
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail


def test_stack_check_ok_ignores_realization_label():
    """required change 1: `ground_realization` is recorded PROVENANCE only.
    A leg whose walls are correct but whose label says something else (a
    different mechanism landed the same planes) must still PASS."""
    cv15 = _load_cv15()
    sc = _good_stack_check(cv15)
    sc["ground_realization"] = "some_future_mechanism"
    ok, detail = cv15._stack_check_ok(sc)
    assert ok, detail
