"""Regression test for issue #740: cv15's mandatory geometry self-check must
assert the REALIZED electric-wall PLANES, not the declared Box extents.

``validation/crossval/15_patch_antenna_rt5880.py``'s pre-#740 ``#325
AVOIDANCE (mandatory)`` self-check asserted only the substrate's z EXTENT
(``n_sub_raster == N_SUB``, from ``round(z/DX)`` cell counting) and kept
PASSING while the one-cell one-plane ground ``Box`` realized its electric
wall one full cell BELOW the declared substrate floor (the #693 "vacuum
ground cell" trap, closed on the canonical patch lane by PRs #716/#718) --
a live vacuum cell inside the modelled cavity, undetected.

The geometry under test is built through cv15's OWN production builder,
``build_rfx_sim(*, do_gain, two_plane)`` -- separated from ``run_rfx()`` for
the #740 review (cv15 is classified ``audited`` in
``tests/_example_fidelity_lib.py`` on that builder). The positive tests pass
NO ``two_plane`` argument, so the script's default is what is under test:
flip that default and ``test_cv15_committed_geometry_realizes_declared_walls``
goes red (verified: 1 failed / 6 passed). The first version of this file
mirrored the geometry in a test-local copy that hardcoded ``two_plane=True``,
and deleting the fix from the script left it green -- the reviewer's
finding, and why the builder exists. ``assert_realized_stack`` and
``_stack_check_ok`` -- the actual #740 fix -- are called UNMODIFIED from the
script itself, never copied.

cv15 is guarded by ``if __name__ == "__main__":`` (see its final lines), so
importing it (to reach its module constants and the two functions above)
executes no simulation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CV15_PATH = REPO_ROOT / "validation" / "crossval" / "15_patch_antenna_rt5880.py"


def _load_cv15():
    """Import cv15 as a module without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location("_cv15_wall_planes", CV15_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_test_sim(cv15, *, two_plane=None):
    """Build cv15's geometry through the PRODUCTION builder,
    ``cv15.build_rfx_sim`` -- not a test-local mirror (#740 review, item 1:
    the mirrored copy hardcoded ``two_plane=True``, so deleting the fix from
    the script left every test green).

    ``two_plane=None`` (the default, used by every positive test) passes NO
    toggle, so the script's own default is what is under test: flip that
    default -- or drop ``two_plane`` from the ground Box -- and the positive
    tests go red (verified: default forced to False -> the positive
    wall-plane test fails, 1 failed / 6 passed; the gate tests use synthetic
    dicts by design and the negative control passes False itself). Only the
    negative control passes ``two_plane=False`` explicitly. Returns
    ``(sim, grid, patch_shape)`` so the caller can run the script's OWN
    ``assert_realized_stack`` against it, cheaply (no solve)."""
    kw = {} if two_plane is None else dict(two_plane=two_plane)
    sim, patch_shape, _geom = cv15.build_rfx_sim(do_gain=False, **kw)
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
    """NEGATIVE CONTROL (issue #740 review, required change 5): build the
    pre-fix one-plane ground through the PRODUCTION builder
    (``build_rfx_sim(two_plane=False)``) and confirm the script's OWN
    ``assert_realized_stack`` -- not a test-local copy -- raises, naming
    z_sub_lo.

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
    # Through the PRODUCTION path: build_rfx_sim(two_plane=False) is the
    # pre-fix one-plane ground. If someone deletes the two_plane default
    # from the script, the positive tests above fail; this one proves the
    # check would have caught the original geometry.
    sim, grid, patch_shape = _build_test_sim(cv15, two_plane=False)
    assert sim._geometry[0].material_name == "pec"
    assert sim._geometry[0].two_plane is False

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
