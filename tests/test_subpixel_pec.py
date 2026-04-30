"""TDD-first tests for Stage 1 conformal PEC face-shift (issue: WR-90
mesh-conv xfail / staircase-vs-physical-aperture mismatch).

The plan in ``docs/agent-memory/rfx-known-issues.md`` §"Stage 1 redesign"
(2026-04-29) calls for a *boundary-face Box injection into the existing
Dey-Mittra path*. The failing acceptance gates are split here into the
smallest pieces that can be checked without running a full FDTD scan:

1. ``Boundary`` carries an opt-in ``conformal`` flag (default off).
2. ``Boundary(conformal=True)`` is only valid on a PEC face.
3. ``BoundarySpec`` exposes which faces are conformal.
4. ``Simulation._assemble_materials`` injects a half-space ``Box`` on
   each conformal PEC face whose wall coordinate is auto-derived from
   the largest waveguide-port aperture on that face.
5. With the default ``conformal=False`` the assembled ``pec_shapes``
   list is unchanged (regression guard).

The cv11 PEC-short A/B and ``test_mesh_convergence_s21_scaled_cpml``
acceptance tests will be added as a second TDD pass once these
unit-level pieces land — running a full WR-90 scan in this file would
turn the test suite into a multi-minute job.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box


# WR-90 cross-section in metres.
WR_90_A = 0.02286
WR_90_B = 0.01016


# -----------------------------------------------------------------------------
# Boundary dataclass tests
# -----------------------------------------------------------------------------


def test_boundary_conformal_flag_accepted_on_pec_face():
    """Boundary should accept an opt-in ``conformal=True`` flag when both
    sides are PEC. Default is False (bit-identical with current code)."""
    b = Boundary(lo="pec", hi="pec", conformal=True)
    assert b.conformal is True
    # Default off
    b_default = Boundary(lo="pec", hi="pec")
    assert b_default.conformal is False


def test_boundary_conformal_roundtrip_via_dict():
    """``conformal`` survives ``to_dict``/``from_dict`` so legacy specs
    stored in JSON manifests do not silently lose the flag."""
    b = Boundary(lo="pec", hi="pec", conformal=True)
    d = b.to_dict()
    assert d.get("conformal") is True
    b2 = Boundary.from_dict(d)
    assert b2.conformal is True

    # Default flag must not pollute the serialised form (so existing
    # snapshots stay byte-equal).
    d_default = Boundary(lo="pec", hi="pec").to_dict()
    assert "conformal" not in d_default


def test_boundary_conformal_rejects_non_pec_face():
    """``conformal=True`` is meaningless on absorbing or periodic faces
    — the flag drives the *PEC* face-shift path. Mis-typed configs
    should fail loudly rather than silently no-op."""
    with pytest.raises(ValueError, match="conformal"):
        Boundary(lo="cpml", hi="cpml", conformal=True)
    with pytest.raises(ValueError, match="conformal"):
        Boundary(lo="periodic", hi="periodic", conformal=True)
    # Mixed: one PEC face is enough to make the flag meaningful, but it
    # is ambiguous which face it applies to. Reject for clarity.
    with pytest.raises(ValueError, match="conformal"):
        Boundary(lo="cpml", hi="pec", conformal=True)


def test_boundaryspec_conformal_faces_inventory():
    """``BoundarySpec.conformal_faces()`` should return the set of face
    labels (``"y_lo"``/``"y_hi"``/...) whose enclosing axis has the
    conformal flag enabled. Empty when no axis opts in."""
    spec_off = BoundarySpec(x="cpml", y="pec", z="pec")
    assert spec_off.conformal_faces() == set()

    spec_y = BoundarySpec(
        x="cpml",
        y=Boundary(lo="pec", hi="pec", conformal=True),
        z="pec",
    )
    assert spec_y.conformal_faces() == {"y_lo", "y_hi"}

    spec_yz = BoundarySpec(
        x="cpml",
        y=Boundary(lo="pec", hi="pec", conformal=True),
        z=Boundary(lo="pec", hi="pec", conformal=True),
    )
    assert spec_yz.conformal_faces() == {"y_lo", "y_hi", "z_lo", "z_hi"}


# -----------------------------------------------------------------------------
# Simulation._assemble_materials integration
# -----------------------------------------------------------------------------


def _wr90_sim(*, conformal: bool):
    """Minimal WR-90 sim with one +x waveguide port and PEC y/z faces.

    The y/z domains stretch slightly past the physical aperture so that
    the boundary cell is a *fractional* cell — that is the whole reason
    Stage 1 exists. Without that fractional cell the half-space Box has
    no work to do and the test would be vacuous.
    """
    domain_x = 0.06
    domain_y = 0.025  # > WR_90_A so j=22 is interior, j=23..24 sit past wall
    domain_z = 0.012  # > WR_90_B
    sim = Simulation(
        freq_max=10e9,
        domain=(domain_x, domain_y, domain_z),
        dx=0.001,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec", conformal=conformal),
            z=Boundary(lo="pec", hi="pec", conformal=conformal),
        ),
        cpml_layers=8,
    )
    sim.add_waveguide_port(
        0.020,  # Inside the interior, well past the 8-layer x-CPML.
        y_range=(0.0, WR_90_A),
        z_range=(0.0, WR_90_B),
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        f0=8e9,
        bandwidth=0.5,
        name="left",
    )
    return sim


def _half_space_box_lo_y(pec_shapes, axis_idx: int, value: float, *, atol=1e-9):
    """Return Box(es) in ``pec_shapes`` whose ``corner_lo[axis_idx]``
    equals ``value`` (within ``atol``). Used to spot the y_hi/z_hi
    half-space injection."""
    return [
        s for s in pec_shapes
        if isinstance(s, Box) and abs(s.corner_lo[axis_idx] - value) <= atol
    ]


def test_assemble_materials_injects_halfspace_for_conformal_face():
    """With ``Boundary(conformal=True)`` on the y and z PEC faces the
    assembled ``pec_shapes`` list should contain half-space Boxes whose
    inner edge sits at the waveguide aperture (``port.a``/``port.b``).
    The Dey-Mittra path then sees a real PEC half-space and produces
    fractional weights at the boundary cell."""
    sim = _wr90_sim(conformal=True)
    grid = sim._build_grid()
    _, _, _, _, pec_shapes, _ = sim._assemble_materials(grid)

    # y_hi: half-space whose corner_lo[1] == WR_90_A
    y_hi = _half_space_box_lo_y(pec_shapes, axis_idx=1, value=WR_90_A)
    assert len(y_hi) >= 1, (
        f"expected a half-space Box with corner_lo[1]={WR_90_A:.5f} m for "
        f"y_hi conformal PEC; got pec_shapes={pec_shapes!r}"
    )

    # z_hi: half-space whose corner_lo[2] == WR_90_B
    z_hi = _half_space_box_lo_y(pec_shapes, axis_idx=2, value=WR_90_B)
    assert len(z_hi) >= 1, (
        f"expected a half-space Box with corner_lo[2]={WR_90_B:.5f} m for "
        f"z_hi conformal PEC; got pec_shapes={pec_shapes!r}"
    )


def test_assemble_materials_unchanged_without_conformal():
    """Default ``conformal=False`` must keep ``pec_shapes`` identical to
    today's behaviour: boundary-face PEC stays in the binary
    ``apply_pec_faces`` path and is *not* registered as a Shape. This
    is the load-bearing regression guard for the opt-in design."""
    sim = _wr90_sim(conformal=False)
    grid = sim._build_grid()
    _, _, _, _, pec_shapes, _ = sim._assemble_materials(grid)

    # No geometry, no thin conductors — the only PEC is the boundary
    # spec, which should NOT spawn any pec_shapes entries.
    assert pec_shapes == [], (
        f"expected empty pec_shapes when conformal=False; got {pec_shapes!r}"
    )


def test_conformal_weights_fractional_at_wr90_y_boundary_cell():
    """Sanity that the existing Dey-Mittra weight machinery reproduces
    the plan-stated boundary-cell weight (~0.36 at WR-90 dx=1 mm) when
    the half-space Box from Stage 1 is in place. Skipped until the Box
    injection actually lands — without that the assertion is vacuous."""
    from rfx.geometry.conformal import compute_conformal_weights_sdf

    sim = _wr90_sim(conformal=True)
    grid = sim._build_grid()
    _, _, _, _, pec_shapes, _ = sim._assemble_materials(grid)
    if not pec_shapes:
        pytest.skip("conformal injection not yet implemented")

    w_ex, w_ey, w_ez = compute_conformal_weights_sdf(grid, pec_shapes)
    w_ex = np.asarray(w_ex)

    # Expected fractional cell: Yee Ex at (i+0.5, j, k) carries y=j*dx.
    # The SDF half-space starts at y=WR_90_A=22.86 mm; the cell with
    # |sdf| ≈ 0.14 mm sits at j = round(WR_90_A/dx) (=23 for dx=1 mm).
    j_boundary = int(round(WR_90_A / float(grid.dx)))
    # Pick an interior x and the middle-z cell.
    i_mid = grid.shape[0] // 2
    k_mid = grid.shape[2] // 2
    w_at_boundary = float(w_ex[i_mid, j_boundary, k_mid])

    # Plan reproduction: 0.36 ± 0.05 (handover memory entry 4910).
    assert 0.0 < w_at_boundary < 1.0, (
        f"boundary cell weight should be fractional, got {w_at_boundary:.4f}"
    )
    assert abs(w_at_boundary - 0.36) < 0.05, (
        f"WR-90 dx=1 mm boundary-cell weight should be ≈0.36, got "
        f"{w_at_boundary:.4f}"
    )


# -----------------------------------------------------------------------------
# Stage 1 step 2: end-to-end run smoke + auto-routing
# -----------------------------------------------------------------------------


def test_run_smokes_with_conformal_boundary():
    """Stage 1 step 2: end-to-end run with ``Boundary(conformal=True)``
    must not raise and must produce finite fields. Catches plumbing
    bugs where the half-space Box trips up downstream code paths
    (initialisation, JIT compile, scan body)."""
    sim = _wr90_sim(conformal=True)
    result = sim.run(n_steps=20)
    ey = np.asarray(result.state.ey)
    assert np.all(np.isfinite(ey)), "non-finite ey after conformal run"
    assert float(np.max(np.abs(ey))) > 0, (
        "ey is identically zero after 20 steps — source did not fire"
    )


def test_run_conformal_auto_routes_from_boundaryspec():
    """``Boundary(conformal=True)`` alone must activate the Dey-Mittra
    pipeline. Without auto-routing, ``conformal_pec`` stays False at
    ``Simulation.run`` and the half-space Box from Stage 1 step 1
    sits unused in ``pec_shapes`` — a silent no-op footgun.

    Pins down two invariants:
      * conformal=True (default kwargs) ≠ conformal=False (different
        eps at the boundary cell drives different time evolution),
      * conformal=True (default kwargs) == conformal=True with
        explicit ``conformal_pec=True`` (auto-routing is consistent
        with the manual flag, no surprise from setting both)."""
    n = 30
    r_off = _wr90_sim(conformal=False).run(n_steps=n)
    r_on = _wr90_sim(conformal=True).run(n_steps=n)
    r_on_explicit = _wr90_sim(conformal=True).run(
        n_steps=n, conformal_pec=True,
    )

    ey_off = np.asarray(r_off.state.ey)
    ey_on = np.asarray(r_on.state.ey)
    ey_on_e = np.asarray(r_on_explicit.state.ey)

    diff_vs_baseline = float(np.max(np.abs(ey_off - ey_on)))
    assert diff_vs_baseline > 1e-9, (
        "auto-routing failed: conformal=True produced bit-identical "
        f"fields to conformal=False (max|diff|={diff_vs_baseline:g}). "
        "BoundarySpec.conformal_faces() is non-empty but "
        "Simulation.run did not flip conformal_pec to True."
    )

    np.testing.assert_allclose(
        ey_on, ey_on_e, atol=1e-12, rtol=0,
        err_msg="auto-routed conformal differs from explicit "
        "conformal_pec=True — they should be the same path.",
    )


def test_run_explicit_false_overrides_conformal_boundary():
    """Escape hatch: ``conformal_pec=False`` explicit kwarg must keep
    the binary ``apply_pec_faces`` path even when the BoundarySpec
    declares ``conformal=True``. Useful for A/B regression diagnosis
    and for keeping older diagnostic scripts on the legacy path."""
    n = 30
    r_off = _wr90_sim(conformal=False).run(n_steps=n)
    r_forced = _wr90_sim(conformal=True).run(
        n_steps=n, conformal_pec=False,
    )

    np.testing.assert_array_equal(
        np.asarray(r_off.state.ey),
        np.asarray(r_forced.state.ey),
        err_msg="explicit conformal_pec=False did not override the "
        "BoundarySpec.conformal_faces() auto-route.",
    )


# -----------------------------------------------------------------------------
# Stage 1 step 3: battery-geometry support + DROP-skip on conformal +face
# -----------------------------------------------------------------------------
#
# The Stage 1 step 1 hi-side veto silently no-ops on the validation
# battery's geometry: when ``y_range`` / ``z_range`` is omitted on
# ``add_waveguide_port`` (the default), the port aperture spans the
# *full* ``self._domain``, so ``wall_hi == self._domain[axis]`` and
# the original "domain edge already coincides with the physical wall"
# check skipped the Box. But the *grid* extends past ``self._domain``
# due to dx-snap, so a fractional cell still exists at the boundary.
# The fix is to always inject the hi-side Box (the SDF naturally
# produces weight=1 when no fractional cell exists).
#
# Then the binary DROP in ``init_waveguide_port`` becomes wrong on
# conformal axes: the staircase shift is now handled by Dey-Mittra
# eps_correction at the boundary cell, so zeroing the same cell in
# the modal V/I integral is double-counting (the failure mode the
# 2026-04-29 first attempt produced).


def _wr90_battery_sim(*, conformal: bool):
    """Validation-battery-style WR-90 sim: domain edge == port wall,
    ``y_range``/``z_range`` left at the default (full grid). Dx=3 mm
    matches the battery's coarsest mesh."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        dx=0.003,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec", conformal=conformal),
            z=Boundary(lo="pec", hi="pec", conformal=conformal),
        ),
        cpml_layers=4,
    )
    sim.add_waveguide_port(
        0.030,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        f0=8e9,
        bandwidth=0.5,
        name="left",
    )
    return sim


def test_assemble_materials_injects_when_yrange_omitted():
    """Battery-geometry repro: no explicit y_range/z_range, but the
    grid extends past ``self._domain`` due to dx-snap. The hi-side
    Box must still be injected so the Dey-Mittra path kicks in at
    the fractional boundary cell."""
    sim = _wr90_battery_sim(conformal=True)
    grid = sim._build_grid()
    _, _, _, _, pec_shapes, _ = sim._assemble_materials(grid)

    y_hi = _half_space_box_lo_y(pec_shapes, axis_idx=1, value=0.04)
    z_hi = _half_space_box_lo_y(pec_shapes, axis_idx=2, value=0.02)
    assert len(y_hi) >= 1, (
        f"expected y_hi half-space Box at corner_lo[1]=0.04 m for "
        f"battery geometry; got pec_shapes={pec_shapes!r}"
    )
    assert len(z_hi) >= 1, (
        f"expected z_hi half-space Box at corner_lo[2]=0.02 m for "
        f"battery geometry; got pec_shapes={pec_shapes!r}"
    )


def test_grid_carries_conformal_faces_from_boundaryspec():
    """``Grid`` must surface the conformal-face inventory so that
    ``init_waveguide_port`` can decide whether to skip the binary
    DROP. ``apply_pec_faces`` already routes through
    ``Grid.pec_faces``; conformal flag follows the same pattern."""
    sim = _wr90_battery_sim(conformal=True)
    grid = sim._build_grid()
    assert getattr(grid, "conformal_faces", None) == {"y_lo", "y_hi",
                                                       "z_lo", "z_hi"}

    sim_off = _wr90_battery_sim(conformal=False)
    grid_off = sim_off._build_grid()
    assert getattr(grid_off, "conformal_faces", set()) == set()


def test_waveguide_port_skips_drop_on_conformal_face():
    """``init_waveguide_port``'s binary DROP at the +face boundary
    cell must be suppressed when that face is conformal — the
    Dey-Mittra eps_correction at the same cell is the principled
    handler. Otherwise the cell is zeroed twice (DROP in V/I + 1/α
    eps scaling) which over-corrects and caps PEC-short closure
    (the 2026-04-29 first-attempt failure mode)."""
    import jax.numpy as jnp

    freqs = jnp.linspace(5e9, 9e9, 5)

    sim_off = _wr90_battery_sim(conformal=False)
    sim_on = _wr90_battery_sim(conformal=True)
    grid_off = sim_off._build_grid()
    grid_on = sim_on._build_grid()

    cfg_off = sim_off._build_waveguide_port_config(
        sim_off._waveguide_ports[0], grid_off, freqs, n_steps=200,
    )
    cfg_on = sim_on._build_waveguide_port_config(
        sim_on._waveguide_ports[0], grid_on, freqs, n_steps=200,
    )

    dA_off = np.asarray(cfg_off.aperture_dA)
    dA_on = np.asarray(cfg_on.aperture_dA)

    # Without conformal: y_hi DROP zeroes the last u-row.
    assert np.all(dA_off[-1, :] == 0.0), (
        f"expected DROP at u_hi (y_hi) without conformal; got "
        f"dA[-1,:]={dA_off[-1, :]}"
    )
    # With conformal: the DROP is skipped → boundary cell preserved.
    assert np.all(dA_on[-1, :] > 0.0), (
        f"expected aperture preserved at u_hi when conformal=True; "
        f"got dA[-1,:]={dA_on[-1, :]}"
    )
    # Symmetric check on v_hi (z_hi).
    assert np.all(dA_off[:, -1] == 0.0)
    assert np.all(dA_on[:, -1] > 0.0)


def test_run_battery_geometry_auto_routes():
    """End-to-end: with the battery geometry and conformal=True, the
    Dey-Mittra pipeline must produce a different field from the
    binary baseline. Closes the silent-no-op gap that Stage 1 step 1
    left for the most-common WR-90 setup pattern (no y_range)."""
    n = 30
    r_off = _wr90_battery_sim(conformal=False).run(n_steps=n)
    r_on = _wr90_battery_sim(conformal=True).run(n_steps=n)
    diff = float(np.max(np.abs(np.asarray(r_off.state.ey)
                                - np.asarray(r_on.state.ey))))
    assert diff > 1e-9, (
        "battery-geometry conformal=True still bit-identical to "
        "baseline — Stage 1 step 3 did not close the silent no-op."
    )
