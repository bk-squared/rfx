"""TDD-first tests for Stage 1 conformal PEC face-shift (issue: WR-90
mesh-conv xfail / staircase-vs-physical-aperture mismatch).

The Stage 1 redesign plan (2026-04-29) calls for a *boundary-face Box
injection into the existing Dey-Mittra path*. The failing acceptance
gates are split here into the smallest pieces that can be checked
without running a full FDTD scan:

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
    _, _, _, _, pec_shapes, _, _ = sim._assemble_materials(grid)

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
    _, _, _, _, pec_shapes, _, _ = sim._assemble_materials(grid)

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
    _, _, _, _, pec_shapes, _, _ = sim._assemble_materials(grid)
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
    _, _, _, _, pec_shapes, _, _ = sim._assemble_materials(grid)

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
    (the 2026-04-29 first-attempt failure mode).

    TWO PATHS, and after issue #868 they differ. The DROP fires only when
    the aperture slice reaches the array edge, i.e. only when the aperture
    contains a cell PAST the +face wall:

    * the ``Simulation`` builder now hands the port the guide's CELL span,
      so its aperture stops at the last real guide cell and there is nothing
      past the wall to drop — the double correction this test guards against
      cannot arise on that path at all, in either conformal setting. That is
      asserted below as an equality between the two settings plus an exact
      aperture area, which is a STRONGER statement than "the last row is
      zero": under the old node span the area came out right only BECAUSE
      the extra row was dropped.
    * a low-level caller that hands ``WaveguidePort`` the whole node plane
      (``y_slice=(0, grid.ny)``) still has that cell inside its aperture, and
      the suppression logic still has to discriminate there. That is where
      the original assertion is re-run, so the guard keeps a firing path.
    """
    import jax.numpy as jnp
    from rfx.sources.waveguide_port import WaveguidePort, init_waveguide_port

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

    # --- builder path: no cell past the wall, so nothing to drop ---------
    dx = grid_off.dx
    n_u, n_v = grid_off.ny - 1, grid_off.nz - 1     # guide cells, not nodes
    assert dA_off.shape == (n_u, n_v), (dA_off.shape, (n_u, n_v))
    assert np.all(dA_off > 0.0), (
        f"the builder aperture is the guide's {n_u} x {n_v} cells; none of "
        f"them sits past a wall, so none may be dropped; got dA={dA_off}")
    assert np.all(dA_on > 0.0), dA_on
    # Same aperture either way — the conformal branch has nothing to suppress.
    assert np.array_equal(dA_off, dA_on)
    # And the area is the rasterized guide's, exactly.
    assert float(dA_off.sum()) == pytest.approx(n_u * dx * n_v * dx, rel=1e-6)

    # --- low-level path: the node plane still contains the ghost cell -----
    def _lowlevel_dA(grid):
        port = WaveguidePort(
            x_index=int(0.030 / grid.dx) + grid.axis_pads[0],
            y_slice=(0, grid.ny), z_slice=(0, grid.nz),
            a=(grid.ny - 1) * grid.dx, b=(grid.nz - 1) * grid.dx,
            mode=(1, 0), mode_type="TE", direction="+x", normal_axis="x",
            u_slice=(0, grid.ny), v_slice=(0, grid.nz),
        )
        cfg = init_waveguide_port(port, grid.dx, freqs, f0=8e9, bandwidth=0.5,
                                  dft_total_steps=200, dt=float(grid.dt),
                                  grid=grid)
        return np.asarray(cfg.aperture_dA)

    ll_off, ll_on = _lowlevel_dA(grid_off), _lowlevel_dA(grid_on)
    assert np.all(ll_off[-1, :] == 0.0), (
        f"expected DROP at u_hi (y_hi) without conformal on the node-plane "
        f"aperture; got dA[-1,:]={ll_off[-1, :]}")
    assert np.all(ll_on[-1, :] > 0.0), (
        f"expected aperture preserved at u_hi when conformal=True; "
        f"got dA[-1,:]={ll_on[-1, :]}")
    assert np.all(ll_off[:, -1] == 0.0)
    assert np.all(ll_on[:, -1] > 0.0)


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


# -----------------------------------------------------------------------------
# Stage 1 step 4: cv11-style PEC-short acceptance gate via S-matrix path
# -----------------------------------------------------------------------------


def _pec_short_sim(*, conformal: bool):
    """Validation-battery-style PEC-short setup. Two ports + a thin
    PEC wall midway through the guide; with num_periods=40 the round
    trip settles inside the DFT window and |S11| → 1.0 for a correct
    extractor. Mirrors ``tests/oracle/test_waveguide_port_validation_battery
    ::test_pec_short_s11_magnitude`` so the gate is directly comparable."""
    import jax.numpy as jnp

    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        dx=0.003,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec", conformal=conformal),
            z=Boundary(lo="pec", hi="pec", conformal=conformal),
        ),
        cpml_layers=10,
    )
    # #931: the old 2 mm slab on a 3 mm grid is a sub-cell VOLUME the
    # contract refuses (it was realized by the thin branch as one wall at
    # x = 0.087 m). Drawn as one full cell, 0.084 -> 0.087 m: a solid
    # short with walls on both drawn planes, the far one still at 0.087 m.
    # The |S11| magnitude gate below is unchanged.
    sim.add(Box((0.084, 0, 0), (0.087, 0.04, 0.02)), material="pec")
    freqs = jnp.linspace(5e9, 7e9, 6)
    sim.add_waveguide_port(
        0.010, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=6e9, bandwidth=0.5, name="left",
    )
    sim.add_waveguide_port(
        0.090, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=6e9, bandwidth=0.5, name="right",
    )
    return sim


# Two-sided |S11| envelope for the conformal PEC-short. DERIVED FROM
# MEASUREMENT (2026-09-15, #1043 / PR #1047), not chosen:
#   interior bins 1-4, measured |dev| -- head 0.0017/0.0012/0.0019/0.0023,
#                                        pre-fix 0.0007/0.0029/0.0023/0.0033
#   band-edge bins 0 and 5           -- head 0.0145 / 0.0123
# The interior bar is the measured interior worst (0.0033) rounded up to the
# next half-percent.
#
# The all-bin bar is the **40-PERIOD** edge worst (0.0145 here, 0.0278 pre-fix)
# rounded up to the next percent, and it is only a 40-period envelope. The same
# rig reads a max |dev| of 0.0448 at 80 periods and 0.0363 at 160, all of it in
# band-edge bin 5 -- so this bar would NOT hold at a longer record and must not
# be quoted as a record-independent envelope. The gate below runs at 40
# periods, which is what makes it applicable here.
_PEC_SHORT_DEV_INTERIOR = 0.005
_PEC_SHORT_DEV_ALL_BINS_AT_40_PERIODS = 0.03
# Band edges of the 5-7 GHz sweep. Bin 5 (7.0 GHz) sits just under the guide's
# second-mode cutoff (40 mm x 20 mm -> TE20/TE01 at 7.5 GHz), and bin 0 is the
# low edge; both carry the largest extraction deviation on every tree measured.
_PEC_SHORT_INTERIOR_BINS = slice(1, 5)


def test_pec_short_s11_with_conformal_face_pec():
    """Acceptance gate: with ``Boundary(conformal=True)`` on the y/z PEC
    faces, the PEC-short ``|S11|`` must sit within a TWO-SIDED envelope of 1.

    Without the Stage 1 step 4 plumbing through
    ``compute_waveguide_s_matrix``, the Step 3 DROP-skip on the +face
    aperture row contaminates V/I via the PEC-normal Ey component
    (which ``apply_pec_faces`` does NOT zero) and PEC-short collapses
    to ~0.84. The ``conformal_eps_correction`` at the boundary cell
    (eps_eff = eps / α with α≈0.83 at dx=3 mm) compensates by
    suppressing the contaminated cell's contribution before the V/I
    integral, restoring Meep-class closure.

    Pre-implementation measurement (2026-04-30): conformal=False →
    0.996; conformal=True → 0.843.

    **Why the bar is two-sided now (2026-09-15, #1043 / PR #1047).** It used to
    be ``s11.min() >= 0.99``, and a one-sided minimum cannot see the failure
    this rig actually has. A lossless PEC short is ``|S11| = 1`` exactly, so
    ``|S11| > 1`` is a passivity violation and the old bar passed it silently:
    on the pre-fix tree this same rig at 160 periods reads ``|S11|`` 9.36-12.09
    — the field is growing — and ``min >= 0.99`` PASSES that run.

    Root cause of the growth, and why it is gone: ``conformal_eps_correction``
    sets ``aniso_eps`` to ``eps_eff = eps/w`` at wall cells
    (``rfx/runners/uniform.py:279-303``), which is HIGHER than
    ``materials.eps_r``; the x CPML pad spans every y and z, so those wall
    cells sit in the absorber; and before #1043 ``apply_cpml_e`` built its psi
    coefficient from the staircase epsilon while the Yee half used the
    corrected one. That is the amplifying direction of the #1043 inequality.

    Measured across record length, which is the independent axis that settles
    it (``scripts/diagnostics/cpml_subpixel_stability/f1_pec_short_gate.py``):

    ====== ======================== ========================
    periods  pre-fix |S11| range      this tree |S11| range
    ====== ======================== ========================
    40       [0.9942, 1.0278]         [0.9877, 1.0145]
    80       [0.5724, 2.8001]         [0.9552, 1.0014]
    160      [9.3597, 12.0899]        [0.9637, 1.0014]
    ====== ======================== ========================

    and the ring-down witness moves the two ways round: pre-fix it DEGRADES
    with a longer record (-39.2/-5.09 dB at 40 → -10.33/-10.58 at 160, the
    signature of growth), here it improves (-40.05/-15.01 → -76.57/-26.62).
    So the pre-fix 40-period numbers were early exponential growth stopped
    before it showed, which is also why their mean sat ABOVE 1 (1.0032).

    On this tree the residual is **entirely the top band-edge bin 5** (7.0 GHz,
    0.93x this guide's TE20/TE01 cutoff at 7.5 GHz), on a port that never
    reaches the -40 dB settling bar at any record length tested. Its |S11| runs
    0.9877 -> 0.9552 -> 0.9637 across the three records, while the interior
    bins 1-4 fall monotonically, 0.0023 -> 0.0019 -> 0.0018 |dev|. So the
    band-edge residual is a near-cutoff extraction limit on an unsettled port,
    NOT record-length truncation -- lengthening the record does not remove it.
    That is why the edge bins are gated loosely and the interior tightly.

    The gate that catches the growth is
    ``test_pec_short_conformal_stays_bounded_over_a_long_record`` below — this
    one is the 40-period envelope, that one is the physics."""
    sim = _pec_short_sim(conformal=True)
    res = sim.compute_waveguide_s_matrix(num_periods=40, normalize=False)
    s11 = np.abs(np.asarray(res.s_params)[0, 0, :])
    dev = np.abs(s11 - 1.0)
    print(f"\n[step4 pec-short] |S11| range "
          f"[{s11.min():.4f}, {s11.max():.4f}] mean={s11.mean():.4f} "
          f"max|dev|={dev.max():.4f} interior|dev|="
          f"{dev[_PEC_SHORT_INTERIOR_BINS].max():.4f}")
    assert dev.max() <= _PEC_SHORT_DEV_ALL_BINS_AT_40_PERIODS, (
        f"PEC-short |S11| with conformal=True left its two-sided envelope: "
        f"per-bin |S11| = {[round(float(v), 4) for v in s11]}, "
        f"max |dev| = {dev.max():.4f} (bar "
        f"{_PEC_SHORT_DEV_ALL_BINS_AT_40_PERIODS}, a 40-PERIOD envelope — the "
        f"same rig reads 0.0448 at 80 periods and 0.0363 at 160, all in "
        f"band-edge bin 5, so this bar is not record-independent). A lossless "
        f"short is |S11| = 1 exactly, so BOTH directions matter — above 1 is a "
        f"passivity violation, below 1 is loss the structure does not have."
    )
    assert dev[_PEC_SHORT_INTERIOR_BINS].max() <= _PEC_SHORT_DEV_INTERIOR, (
        f"PEC-short |S11| drifted in the INTERIOR bins, where this rig is "
        f"trustworthy: per-bin |dev| = "
        f"{[round(float(v), 4) for v in dev]}, interior max = "
        f"{dev[_PEC_SHORT_INTERIOR_BINS].max():.4f} (bar "
        f"{_PEC_SHORT_DEV_INTERIOR}). The band-edge bins 0 and 5 are excluded "
        f"from this bar by measurement, not by convenience — see the "
        f"docstring's record-length table."
    )


# The growth bar. Provenance, both measured on this rig at 160 periods:
# this tree max |S11| = 1.0014, the pre-fix tree max |S11| = 12.0899. Any
# threshold between them separates them; 1.05 is the passivity bound (|S11|
# <= 1 for a lossless short) plus the measured discretization headroom.
_PEC_SHORT_LONG_RECORD_MAX_S11 = 1.05


@pytest.mark.slow
def test_pec_short_conformal_stays_bounded_over_a_long_record():
    """The conformal PEC-short must not GROW when the record is lengthened.

    This is the gate the old one-sided ``min >= 0.99`` could not express, and
    the reason #1043's CPML coefficient fix is load-bearing here rather than
    cosmetic. On the pre-fix tree this rig is unstable — ``eps_eff = eps/w``
    at the conformal wall cells is higher than the staircase ``materials.eps_r``
    that ``apply_cpml_e`` was building its psi coefficient from, and those
    cells run through the x CPML pads — so the field grows and 160 periods
    reads ``|S11|`` 9.36-12.09 with the ring-down witness DEGRADING as the
    record lengthens.

    40 periods is too short to see it, which is exactly why the sibling gate
    above could sit green over a diverging simulation for months.

    RED on the pre-fix tree (12.0899), green here (1.0014). If this ever goes
    red again, do not relax it: re-read the per-bin trace and the settling
    witness first — a PEC short that returns more than it was given is not a
    tolerance problem."""
    sim = _pec_short_sim(conformal=True)
    res = sim.compute_waveguide_s_matrix(num_periods=160, normalize=False)
    s11 = np.abs(np.asarray(res.s_params)[0, 0, :])
    print(f"\n[pec-short long record] 160 periods |S11| range "
          f"[{s11.min():.4f}, {s11.max():.4f}] mean={s11.mean():.4f}")
    assert np.all(np.isfinite(s11)), (
        f"PEC-short |S11| went non-finite over a 160-period record: {s11}"
    )
    assert s11.max() <= _PEC_SHORT_LONG_RECORD_MAX_S11, (
        f"PEC-short |S11| GREW over a 160-period record: per-bin "
        f"{[round(float(v), 4) for v in s11]}, max = {s11.max():.4f} "
        f"(bar {_PEC_SHORT_LONG_RECORD_MAX_S11}). A lossless short cannot "
        f"return more than it was given; this is the conformal-in-CPML-pad "
        f"instability #1043 fixed, measured at 12.0899 before that fix."
    )


def test_pec_short_s11_baseline_unchanged_with_binary_path():
    """Regression guard: ``Boundary(conformal=False)`` (default)
    PEC-short |S11| stays at the pre-Stage-1 baseline. Catches any
    accidental coupling between the conformal plumbing work and the
    binary path.

    ``num_periods`` is 80 here and 40 in the conformal sibling. That is a
    SETTLING length, not a gate: measured on VESSL 369367259190 after #931
    redrew the short from a sub-cell slab (one wall) to one full cell
    (walls on both drawn faces, 0.084 and 0.087 m, verified below), the
    binary lane at 40 periods reads

        |S11| = [1.0057, 0.9936, 0.9996, 1.0029, 1.0012, 0.9892]

    — three bins ABOVE 1 for a passive reflector, so the record's own
    noise is +-0.6% and the 0.9892 minimum is 1.1% low. At 80 periods the
    same geometry reads

        |S11| = [0.9974, 0.9984, 1.0010, 1.0009, 0.9988, 1.0024]

    — every bin within 0.26% of unity. The short moved one cell toward the
    left port when it stopped being a zero-thickness wall, and the 40-period
    window no longer contains the settled response of the new round trip.
    The 0.99 gate is untouched.

    (The conformal sibling is left at 40 periods on purpose: at 80 it
    diverges, |S11| in [0.57, 2.77] on the same geometry. That is the
    Dey-Mittra face-PEC lane, which #931 §1.8 fences out of the ownership
    contract, and it is recorded here as an observation, not fixed.)
    """
    sim = _pec_short_sim(conformal=False)
    # build-time: the short realizes walls on BOTH drawn faces (#931 §1.2)
    from tests._realized_geometry import assert_wall_planes
    assert_wall_planes(sim, 0, [0.084, 0.087], what="cv11-style PEC short")

    res = sim.compute_waveguide_s_matrix(num_periods=80, normalize=False)
    s11 = np.abs(np.asarray(res.s_params)[0, 0, :])
    assert s11.min() >= 0.99, (
        f"PEC-short |S11| baseline regressed: min={s11.min():.4f} "
        f"(gate 0.99). Stage 1 must not affect the binary path."
    )
    # the record must also not be contaminated the way 40 periods was: a
    # passive short cannot reflect more than it receives.
    assert s11.max() <= 1.01, (
        f"|S11| max={s11.max():.4f} > 1 for a passive short — the DFT "
        "window is contaminated; lengthen num_periods before reading the "
        "minimum as physics")


# -----------------------------------------------------------------------------
# Stage 1 step 5 (nice-to-have): mesh-convergence S21 with Boundary(conformal=True)
# -----------------------------------------------------------------------------
#
# The binary-PEC mesh-conv case
# (``test_mesh_convergence_s21_scaled_cpml`` in the validation battery)
# already passes since the 2026-04-29 Box.mask_on_coords fix; this test
# locks the conformal=True path to the same convergence behaviour so the
# Dey-Mittra coefficient-modifying lane stays Meep-class on
# inverse-design / topology-changing geometries where binary staircase
# would re-introduce the cell-count rounding jitter.


def _wr90_meshconv_sim(*, dx: float, conformal: bool, cpml_layers: int):
    """Two-port WR-90 with one εr=4 obstacle for S21 mesh convergence.

    Mirrors the validation-battery DOMAIN/PORT placement so refinement
    behaviour is comparable to the binary-PEC baseline test, but uses
    ``Boundary(conformal=conformal)`` on the y/z PEC walls so the Stage 1
    Dey-Mittra path can take effect."""
    import jax.numpy as jnp

    sim = Simulation(
        freq_max=10e9,
        domain=(0.12, 0.04, 0.02),
        dx=dx,
        boundary=BoundarySpec(
            x="cpml",
            y=Boundary(lo="pec", hi="pec", conformal=conformal),
            z=Boundary(lo="pec", hi="pec", conformal=conformal),
        ),
        cpml_layers=cpml_layers,
    )
    sim.add_material("diel_4", eps_r=4.0, sigma=0.0)
    sim.add(Box((0.05, 0.0, 0.0), (0.07, 0.04, 0.02)), material="diel_4")

    freqs = jnp.asarray([6.0e9])
    sim.add_waveguide_port(
        0.01,
        direction="+x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=6e9, bandwidth=0.5,
        waveform="modulated_gaussian",
        name="left",
    )
    sim.add_waveguide_port(
        0.09,
        direction="-x", mode=(1, 0), mode_type="TE",
        freqs=freqs, f0=6e9, bandwidth=0.5,
        waveform="modulated_gaussian",
        name="right",
    )
    return sim


@pytest.mark.slow
def test_mesh_convergence_s21_with_conformal_pec_baseline():
    """Baseline: same geometry with ``conformal=False`` should refine
    monotonically — established by the validation battery's
    ``test_mesh_convergence_s21_scaled_cpml`` (2026-04-29 unblocked).
    Mirrors that gate locally so the conformal-path test below has a
    side-by-side reference inside the same file.
    """
    target_cpml_m = 0.030
    resolutions = [0.003, 0.002, 0.0015]
    s21_values: list[float] = []
    for dx in resolutions:
        layers = max(8, int(round(target_cpml_m / dx)))
        sim = _wr90_meshconv_sim(dx=dx, conformal=False, cpml_layers=layers)
        res = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
        s = np.asarray(res.s_params)
        port_idx = {n: i for i, n in enumerate(res.port_names)}
        s21 = float(np.abs(s[port_idx["right"], port_idx["left"], 0]))
        s21_values.append(s21)
        print(f"[meshconv-binary] dx={dx*1e3:.1f}mm cpml={layers} |S21|={s21:.4f}")
    coarse_delta = abs(s21_values[0] - s21_values[1])
    fine_delta = abs(s21_values[1] - s21_values[2])
    print(f"[meshconv-binary] coarse_delta={coarse_delta:.4f} fine_delta={fine_delta:.4f}")
    assert fine_delta < 0.10, (
        f"Baseline binary mesh-conv failed locally — coarse={coarse_delta:.4f}, "
        f"fine={fine_delta:.4f}. Test setup or environment regression."
    )


# Measured envelope for the conformal mesh-convergence ladder, 2026-09-15
# (#1043 / PR #1047). DERIVED, not chosen: the three rungs read
# 0.7796 / 0.6974 / 0.7274 on this tree, so the ladder spans [0.6974, 0.7796]
# and the widening quantum is the ladder's own coarse delta, 0.0822.
_MESHCONV_CONFORMAL_S21_LO = 0.6974 - 0.0822
_MESHCONV_CONFORMAL_S21_HI = 0.7796 + 0.0822


@pytest.mark.slow
def test_mesh_convergence_s21_with_conformal_pec():
    """``Boundary(conformal=True)`` must keep mesh refinement on the S21
    of an εr=4 obstacle within the same fine-delta gate (0.10) as the
    binary baseline. Three resolutions {3, 2, 1.5} mm with
    CPML thickness scaled to a fixed 30 mm physical absorber.

    **This was an ``xfail(strict=True)`` until 2026-09-15 (#1043 / PR #1047)**,
    tracking "the Stage 1 conformal path NaNs at fine dx". It was a
    self-detecting stale check by design — its own reason text named the XPASS
    as the signal to delete the ``_validate_cfg_conformal_fine_dx`` preflight
    guard and the ``rfx/sparams/waveguide.py`` runtime guardrail — and it
    XPASSed once #1043 made ``apply_cpml_e`` build its psi coefficient from the
    permittivity the E update uses. Both guards are gone in the same change.

    The NaN was never a property of the conformal METHOD.
    ``conformal_eps_correction`` sets ``aniso_eps`` to ``eps_eff = eps/w`` at
    wall cells (``rfx/runners/uniform.py:279-303``), higher than
    ``materials.eps_r``; the x CPML pad spans every y and z, so those cells sit
    in the absorber; and the psi half was reading the staircase epsilon. That
    is the amplifying direction of the #1043 inequality, and it is why "dt
    cannot cure it" — a coefficient sign flip is not a CFL problem.

    |S21| at dx 3 / 2 / 1.5 mm, all three measured on this rig:

    ==================================== ==========================
    tree                                  |S21| at each rung
    ==================================== ==========================
    origin/main                           0.7564 / 0.6974 / nan
    parameter present but not threaded    0.7564 / 0.6974 / nan
    psi coefficient threaded (this tree)  0.7796 / 0.6974 / 0.7274
    ==================================== ==========================

    The middle row is the mutation control: adding the parameter without
    passing it at the call sites leaves the NaN, so the threading is what
    fixes it. dx = 2 mm reads identically on every tree because the wall lands
    on a node there and the conformal weight is 1 — conformal is a no-op at
    that rung by construction.

    **NOT claimed here**: that conformal PEC is ACCURATE. The 2026-06-08
    verdicts that falsified four conformal methods were all taken with this
    defect present and have to be re-measured before any accuracy claim. This
    test gates stability and mesh convergence, which is what it always
    measured."""
    target_cpml_m = 0.030
    resolutions = [0.003, 0.002, 0.0015]
    s21_values: list[float] = []
    for dx in resolutions:
        layers = max(8, int(round(target_cpml_m / dx)))
        sim = _wr90_meshconv_sim(dx=dx, conformal=True, cpml_layers=layers)
        res = sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)
        s = np.asarray(res.s_params)
        port_idx = {n: i for i, n in enumerate(res.port_names)}
        s21 = float(np.abs(s[port_idx["right"], port_idx["left"], 0]))
        s21_values.append(s21)
        print(f"[meshconv-conformal] dx={dx*1e3:.1f}mm cpml={layers} |S21|={s21:.4f}")

    # PRIMARY signal: the historical failure mode was a NaN |S21| at fine dx,
    # not a tolerance miss, so finiteness is asserted on its own. It stays the
    # first assertion now that the test is a real gate -- a regression here
    # would come back as a NaN before it came back as a bad tolerance.
    assert np.all(np.isfinite(s21_values)), (
        f"conformal=True produced non-finite |S21| at fine dx: {s21_values} — "
        "this is the pre-#1043 conformal-fine-dx NaN returning. Its cause was "
        "the CPML psi coefficient reading a different permittivity than the "
        "Yee half of the same timestep; check "
        "rfx/boundaries/cpml.py's inv_eps_r_update and the two call sites "
        "that pass it before looking anywhere else."
    )

    coarse_delta = abs(s21_values[0] - s21_values[1])
    fine_delta = abs(s21_values[1] - s21_values[2])
    print(f"[meshconv-conformal] coarse_delta={coarse_delta:.4f} "
          f"fine_delta={fine_delta:.4f}")

    assert fine_delta <= coarse_delta + 0.01, (
        f"Refinement did not reduce |S21| change with conformal=True: "
        f"coarse={coarse_delta:.4f}, fine={fine_delta:.4f}"
    )
    assert fine_delta < 0.10, (
        f"Fine-mesh |S21| change too large with conformal=True: "
        f"{fine_delta:.4f} (gate 0.10)"
    )
    # The measured envelope, so a rung that goes finite-but-wrong is caught
    # too and not just a NaN. Bounds are the ladder's own span widened by its
    # own coarse delta -- see the constants above for the derivation.
    assert all(_MESHCONV_CONFORMAL_S21_LO <= v <= _MESHCONV_CONFORMAL_S21_HI
               for v in s21_values), (
        f"conformal mesh-convergence ladder left its measured envelope "
        f"[{_MESHCONV_CONFORMAL_S21_LO:.4f}, {_MESHCONV_CONFORMAL_S21_HI:.4f}]: "
        f"{[round(v, 4) for v in s21_values]} at dx = 3 / 2 / 1.5 mm "
        f"(measured 0.7796 / 0.6974 / 0.7274 when this gate was written)."
    )


# -----------------------------------------------------------------------------
# G-WI5 guardrail: DELETED 2026-09-15 (#1043 / PR #1047)
# -----------------------------------------------------------------------------


def test_guardrail_conformal_normalize_no_longer_warns():
    """The G-WI5 runtime guardrail must stay deleted.

    It warned that ``compute_waveguide_s_matrix(conformal=True,
    normalize=True)`` "is KNOWN to produce NaN S-parameters at fine mesh" and
    told users to switch to staircase PEC or a coarser mesh. The NaN was the
    #1043 CPML psi coefficient reading a different permittivity than the Yee
    half of the same timestep (``conformal_eps_correction`` sets ``aniso_eps``
    to ``eps_eff = eps/w`` at wall cells, higher than ``materials.eps_r``, and
    those cells run through the x CPML pads). It is fixed, so the warning
    became advice to avoid a working path.

    Measured on the tripwire's own three rungs, |S21| at dx 3 / 2 / 1.5 mm:
    0.7564 / 0.6974 / nan before, 0.7796 / 0.6974 / 0.7274 after — see
    ``test_mesh_convergence_s21_with_conformal_pec``, which is the gate that
    replaced this one and measures the behaviour instead of the message.

    This test is the anti-regression: if the guardrail comes back, either it
    was re-added by mistake or the conformal path regressed, and in the second
    case the convergence gate is red too and is the one to read first.
    """
    import warnings as _warnings

    sim = _wr90_meshconv_sim(dx=0.003, conformal=True, cpml_layers=8)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        sim.compute_waveguide_s_matrix(num_periods=40, normalize=True)

    stale = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "conformal" in str(w.message).lower()
        and "nan" in str(w.message).lower()
    ]
    assert not stale, (
        "the deleted G-WI5 conformal/NaN guardrail is warning again: "
        f"{[str(w.message) for w in stale]}. It was removed by #1043 / "
        "PR #1047 because the NaN it names no longer happens; see the note "
        "where it stood in rfx/sparams/waveguide.py."
    )
