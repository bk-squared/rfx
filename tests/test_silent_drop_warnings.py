"""Regression locks for the silent-drop class bug (fixed 2026-04).

Previously the distributed, non-uniform, and subgridded dispatch paths
in ``Simulation.run`` silently dropped most of the run-time kwargs
(``checkpoint``, ``snapshot``, ``until_decay``, ``decay_*``,
``conformal_pec``, ``conformal_min_weight``). ``subpixel_smoothing`` on
the NU path was fixed in commit ``1a2e6c5``; this test pins the rest of
the class:

1. **NU path**: emits an explicit ``UserWarning`` for ``snapshot``,
   ``conformal_pec`` when set to non-default values.
   ``subpixel_smoothing`` and ``checkpoint`` are *propagated* on the NU
   path and therefore must NOT warn. ``until_decay`` is propagated on
   ABSORBING (cpml/upml) NU boundaries since #383 (must NOT warn there)
   and still warn-and-drops on closed/PEC NU boundaries, with a
   lane-accurate reason (the interior-energy stop needs an absorber).

2. **Subgridded path**: emits a ``UserWarning`` for each unsupported
   kwarg (including ``subpixel_smoothing`` and ``checkpoint`` which
   the subgridded runner cannot accept).

3. **PMC + CPML preflight (P2.7)**: warns when a PMC/PEC reflector
   face coexists with CPML on the opposite face of the same axis —
   the current ``Grid`` allocates ``pad_{axis}`` symmetrically, so
   the reflector plane is offset by ``pad_{axis}·dx`` from the
   user domain edge. Tracks the per-face grid padding work item.

Distributed-path warnings are not exercised here because they require
multi-device availability; their helper entry is wired identically to
the other paths and is covered by the unit test on
``_warn_unsupported_run_kwargs`` below.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


# --------------------------------------------------------------------
# Helper unit-test: _warn_unsupported_run_kwargs fires only on
# non-default values and stays quiet otherwise.
# --------------------------------------------------------------------

def test_warn_helper_silent_on_defaults():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Simulation._warn_unsupported_run_kwargs("dummy", {
            "subpixel_smoothing": False,
            "checkpoint": False,
            "snapshot": None,
            "until_decay": None,
            "conformal_pec": False,
        })
    assert [str(w.message) for w in caught] == []


def test_warn_helper_fires_on_non_defaults():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Simulation._warn_unsupported_run_kwargs("dummy-path", {
            "subpixel_smoothing": True,
            "checkpoint": True,
            "snapshot": "not-none",
            "until_decay": 1e-3,
            "conformal_pec": True,
        })
    msgs = [str(w.message) for w in caught]
    assert any("subpixel_smoothing" in m for m in msgs)
    assert any("checkpoint" in m for m in msgs)
    assert any("snapshot" in m for m in msgs)
    assert any("until_decay" in m for m in msgs)
    assert any("conformal_pec" in m for m in msgs)
    assert all("dummy-path" in m for m in msgs)


# --------------------------------------------------------------------
# NU-path dispatch: warn for the kwargs that stay dropped after the 2026-04 fix.
# --------------------------------------------------------------------

def _make_nu_sim():
    """Minimal NU sim: 1D cavity with a 4-cell graded dz."""
    sim = Simulation(
        freq_max=10e9,
        domain=(2e-3, 2e-3, 4e-3),
        dx=1e-3,
        boundary="pec",
        cpml_layers=0,
    )
    sim._dz_profile = np.full(4, 1e-3)
    sim.add_source((1e-3, 1e-3, 1e-3), "ex")
    sim.add_probe((1e-3, 1e-3, 3e-3), "ex")
    return sim


@pytest.mark.parametrize(
    "kw,val",
    [
        ("snapshot", "anything_truthy"),
        ("until_decay", 1e-3),
        ("conformal_pec", True),
    ],
)
def test_nu_path_warns_on_dropped_kwargs(kw, val):
    sim = _make_nu_sim()
    # `snapshot` needs a SnapshotSpec-like object to even reach dispatch
    # without type-checking; since we're testing the warn-on-entry
    # path, we only exercise it when the sim can build a snapshot
    # argument by passing a sentinel through. Guard at the test level.
    if kw == "snapshot":
        # The helper only inspects identity vs `None`, so any non-None
        # sentinel fires the warning. The actual run itself will not
        # forward this sentinel anywhere.
        sentinel = object()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Hit the helper directly — the run() path validates the
            # snapshot type at the uniform-path ingress, which we
            # cannot dodge without a real SnapshotSpec.
            sim._warn_unsupported_run_kwargs("non-uniform mesh",
                                             {"snapshot": sentinel})
        assert any("snapshot" in str(w.message) for w in caught)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(n_steps=16, **{kw: val})
    msgs = [str(w.message) for w in caught]
    assert any(kw in m and "non-uniform mesh" in m for m in msgs), (
        f"expected a non-uniform-mesh silent-drop warning for {kw}={val!r}, "
        f"got: {msgs}"
    )
    if kw == "until_decay":
        # #383: _make_nu_sim is a closed (PEC, cpml_layers=0) NU sim, so
        # the drop must carry the lane-accurate reason — the
        # interior-energy stop needs absorbing boundaries.
        assert any("until_decay" in m and "absorbing" in m for m in msgs), (
            f"closed-boundary NU until_decay drop must state the "
            f"absorbing-boundary reason (#383), got: {msgs}"
        )


def test_nu_path_until_decay_not_dropped_on_absorbing_boundary():
    """#383: until_decay is propagated (no drop warning) on a CPML NU sim."""
    dz = np.full(8, 1e-3)
    sim = Simulation(
        freq_max=10e9,
        domain=(4e-3, 4e-3, 8e-3),
        dx=1e-3,
        dz_profile=dz,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_source((2e-3, 2e-3, 2e-3), "ez")
    sim.add_probe((2e-3, 2e-3, 5e-3), "ez")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(
            until_decay=1e-3,
            decay_check_interval=20,
            decay_min_steps=20,
            decay_max_steps=200,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("until_decay" in m and "silently ignored" in m
                   for m in msgs), (
        f"until_decay must be honoured on the absorbing-boundary NU path "
        f"(#383), got: {msgs}"
    )


def test_nu_path_checkpoint_does_not_warn():
    """checkpoint is propagated through _run_nonuniform (wired 2026-04)."""
    sim = _make_nu_sim()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(n_steps=16, checkpoint=True)
    msgs = [str(w.message) for w in caught]
    assert not any("checkpoint=" in m and "non-uniform mesh" in m
                   for m in msgs), (
        f"checkpoint should be propagated on the NU path, not dropped. "
        f"Warnings: {msgs}"
    )


def test_nu_path_subpixel_does_not_warn():
    """subpixel_smoothing is propagated through _run_nonuniform since 1a2e6c5."""
    sim = _make_nu_sim()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(n_steps=16, subpixel_smoothing=True)
    msgs = [str(w.message) for w in caught]
    assert not any("subpixel_smoothing" in m and "non-uniform mesh" in m
                   for m in msgs), (
        f"subpixel_smoothing should be propagated on the NU path. "
        f"Warnings: {msgs}"
    )


# --------------------------------------------------------------------
# P2.7 preflight: PMC/PEC face + CPML on the same axis.
# --------------------------------------------------------------------

def test_preflight_silent_on_pmc_plus_cpml_uniform_path():
    """Uniform mesh with PMC+CPML composition: after the per-face
    grid allocation (2026-04) the reflector wall aligns with the user domain
    edge (pad=0 on the PMC side), so P2.7 does NOT fire."""
    spec = BoundarySpec(
        x="periodic",
        y=Boundary(lo="cpml", hi="pmc"),
        z="periodic",
    )
    sim = Simulation(
        freq_max=10e9,
        domain=(4e-3, 8e-3, 4e-3),
        dx=1e-3,
        boundary=spec,
        cpml_layers=4,
    )
    sim.add_source((2e-3, 4e-3, 2e-3), "ex")
    sim.add_probe((2e-3, 6e-3, 2e-3), "ex")
    issues = sim.preflight()
    assert not any("P2.7" in s for s in issues), (
        f"P2.7 must not fire on uniform path (per-face padding 2026-04 "
        f"closes the gap). Got: {issues}"
    )


def test_preflight_silent_on_pmc_plus_cpml_nu_path():
    """NU path: per-face allocation (2026-04) extended to NonUniformGrid,
    so P2.7 is silent on NU too (gap closed on both paths)."""
    spec = BoundarySpec(
        x="periodic",
        y=Boundary(lo="cpml", hi="pmc"),
        z="periodic",
    )
    sim = Simulation(
        freq_max=10e9,
        domain=(4e-3, 8e-3, 4e-3),
        dx=1e-3,
        boundary=spec,
        cpml_layers=4,
    )
    sim._dz_profile = np.full(4, 1e-3)  # force NU path
    sim.add_source((2e-3, 4e-3, 2e-3), "ex")
    sim.add_probe((2e-3, 6e-3, 2e-3), "ex")
    issues = sim.preflight()
    assert not any("P2.7" in s for s in issues), (
        f"P2.7 must not fire on NU path after per-face fix. Got: {issues}"
    )


def test_nu_grid_asymmetric_pmc_allocation():
    """NonUniformGrid.pad_y_lo must be 0 when y_lo is a PMC face."""
    from rfx.nonuniform import make_nonuniform_grid
    dz_profile = np.full(8, 1e-3)
    g = make_nonuniform_grid(
        domain_xy=(8e-3, 8e-3),
        dz_profile=dz_profile,
        dx=1e-3,
        cpml_layers=4,
        pmc_faces={"y_lo"},
    )
    assert g.pad_y_lo == 0, f"expected pad_y_lo=0, got {g.pad_y_lo}"
    assert g.pad_y_hi == 4
    assert g.pad_x_lo == 4 and g.pad_x_hi == 4
    # ny = interior + pad_y_hi (no lo padding) = 8 + 4 = 12.
    assert g.ny == 12, f"expected ny=12, got {g.ny}"
    # axis_pads property carries the leading (lo) pad per axis.
    assert g.axis_pads == (4, 0, 4)


def test_preflight_silent_on_cpml_without_reflector():
    """Plain all-CPML sim: no P2.7 warning."""
    sim = Simulation(
        freq_max=10e9,
        domain=(4e-3, 4e-3, 4e-3),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_source((2e-3, 2e-3, 2e-3), "ex")
    sim.add_probe((3e-3, 2e-3, 2e-3), "ex")
    issues = sim.preflight()
    assert not any("P2.7" in s for s in issues), (
        f"P2.7 must not fire without a PMC/PEC reflector face. Got: {issues}"
    )


def test_preflight_silent_on_closed_cavity_with_pmc():
    """cpml_layers=0 closed cavity + PMC face: no P2.7 warning (the
    architectural gap only applies when CPML is allocated)."""
    spec = BoundarySpec(
        x="periodic",
        y=Boundary(lo="pec", hi="pmc"),
        z="periodic",
    )
    sim = Simulation(
        freq_max=10e9,
        domain=(2e-3, 8e-3, 2e-3),
        dx=1e-3,
        boundary=spec,
        cpml_layers=0,
    )
    sim.add_source((1e-3, 4e-3, 1e-3), "ex")
    sim.add_probe((1e-3, 6e-3, 1e-3), "ex")
    issues = sim.preflight()
    assert not any("P2.7" in s for s in issues), (
        f"P2.7 must not fire when cpml_layers=0 (no allocated padding). "
        f"Got: {issues}"
    )


def test_uniform_grid_asymmetric_pmc_allocation():
    """Grid.pad_y_lo must be 0 when y_lo is a PMC face (per-face
    allocation). Leading axis_pads tuple also reflects this."""
    from rfx.grid import Grid
    g = Grid(
        freq_max=10e9, domain=(10e-3, 10e-3, 10e-3), dx=1e-3,
        cpml_layers=8, pmc_faces={"y_lo"},
    )
    assert g.pad_y_lo == 0, f"expected pad_y_lo=0 for PMC face, got {g.pad_y_lo}"
    assert g.pad_y_hi == 8
    assert g.pad_x_lo == 8 and g.pad_x_hi == 8
    assert g.axis_pads == (8, 0, 8), (
        f"axis_pads must carry lo pads as leading offset, got {g.axis_pads}"
    )
    # position_to_index: user y=0 maps to array index 0 (the PMC wall).
    assert g.position_to_index((0.0, 0.0, 0.0))[1] == 0
    # Shape: ny = interior + pad_y_hi (no lo padding).
    assert g.shape[1] == int(np.ceil(10e-3 / 1e-3)) + 1 + 8
