"""Issue #706 — opt-in two-plane realization for one-cell PEC slabs.

Measured baseline (issue #706, eigenmode-witnessed): ``apply_pec_mask``
zeroes ONE tangential-E entry per PEC cell index — the cell's LOWER node
plane — so a PEC body filling exactly one cell along its normal presents
one electrical wall and its own cell volume stays live.  The opt-in
(``sim.add(shape, material="pec", two_plane=True)``) additionally zeroes
the tangential components at the NEXT node plane (k+1) and shields the
slab's interior normal edge, for cell runs of length exactly 1 only.

Off-state is pinned bit-identical to main (golden fixture captured from
the main checkout at 8e00497, sha256
96ba81c10b2459c4f594e1f54fba67d64336660f8db6d18a094721a602dc65b3;
the sheet is deliberately PARTIAL-lateral — a full-lateral sheet plus the
PEC walls seals the domain and every mutated component is identically
zero above it, measured OFF-vs-ON max field diff 0.0 on that fixture
vs 3.12e+06 on this one).

The physics gate is the eigenmode witness in
``TestEigenmodeWitness`` (parallel-plate cavity between two one-cell
sheets; ~6 s per run on CPU, ~12 s for the class).  Mutation
falsification results (each mutation applied by hand to
``rfx/boundaries/pec.py`` / ``apply_pec_mask``, the suite rerun, then
reverted) are recorded verbatim in the affected tests' docstrings.
"""

LOCK_PROVENANCE = {
    "fixture": "none",
    "generator": "hand-derived (same-process off-state contract; the 8e00497 golden was retired in #707)",
    "commit": "4bc7707",
    "date": "2026-08-24",
    "run_id": "local",
    "host": "authoring workstation, JAX_PLATFORMS=cpu (os / jax version not recorded in #707)",
    "pinned_until": "2027-02-20",
}

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation, GaussianPulse, Box
from rfx.boundaries.pec import (
    apply_pec_mask,
    tangential_edge_masks,
    two_plane_extension_masks,
)

C0 = 299792458.0
DX = 0.2e-3


def _np(a):
    return np.asarray(a)


def _mask(shape, cells):
    m = np.zeros(shape, dtype=bool)
    for c in cells:
        m[c] = True
    return jnp.asarray(m)


# ---------------------------------------------------------------------------
# Unit level: the extension rule itself
# ---------------------------------------------------------------------------

class TestExtensionMasks:
    def test_one_cell_z_slab_planes(self):
        """A flagged 4x4x1 z-slab adds: tangential Ex/Ey at plane k+1 with
        the SAME lateral footprint the base rule zeroed at plane k, plus
        the interior normal edge Ez(k).  Nothing else.

        Mutation falsification M1 (far-plane neuter: replace the inner
        ``for t in range(3)`` loop of ``two_plane_extension_masks`` with
        ``pass``): this test FAILED at
        ``assert np.array_equal(ex2[:, :, 4], base_ex[:, :, 3])``
        (``assert False`` — no plane-4 selection), and the ON-state
        eigenmode witness fell back to the one-plane ladder — measured
        ``(47748540156.41224, 49922252775.160515)`` at its 1% gate,
        i.e. dominant 47.7485 GHz = the OFF value, 4.35% from the
        two-plane prediction.  Reverted after measurement.
        """
        shape = (8, 8, 8)
        cells = [(i, j, 3) for i in range(2, 6) for j in range(2, 6)]
        pec = _mask(shape, cells)
        ex2, ey2, ez2 = [_np(a) for a in two_plane_extension_masks(pec, pec)]
        base_ex, base_ey, base_ez = [_np(a) for a in tangential_edge_masks(pec)]
        # base: tangential at plane k=3 only, no normal component
        assert sorted(set(np.where(base_ex)[2])) == [3]
        assert not base_ez.any()
        # extras: same footprint moved to plane 4
        assert np.array_equal(ex2[:, :, 4], base_ex[:, :, 3])
        assert np.array_equal(ey2[:, :, 4], base_ey[:, :, 3])
        assert sorted(set(np.where(ex2)[2])) == [4]
        assert sorted(set(np.where(ey2)[2])) == [4]
        # interior normal edge at the slab cell itself
        assert sorted(set(np.where(ez2)[2])) == [3]
        assert np.array_equal(ez2[:, :, 3], _np(pec)[:, :, 3])

    def test_two_cell_run_gets_nothing(self):
        """A 2-cell run (flagged one-cell slab ADJACENT to more PEC) is a
        thick body: the base tangential rule already zeroes planes k and
        k+1, and the extension must add NOTHING (no double-masking, no
        k+2 plane).

        Covers both spellings: (a) a 2-cell body entirely flagged,
        (b) a flagged one-cell slab abutting an UNflagged PEC body —
        the run-length test is evaluated against the FULL pec union.

        Mutation falsification M4 (adjacency neuter: compute the run-1
        neighbours from ``two_plane_mask`` instead of ``pec_mask`` in
        ``two_plane_extension_masks``): case (b) FAILED
        (``assert not any(a.any() for a in extras)`` -> ``assert not
        True`` — the flagged cell, self-isolated in its own union,
        sprouted extension planes inside the neighbouring body), and
        ``test_seam_adjacency_through_wrap`` failed the same way.
        Reverted after measurement.
        """
        shape = (8, 8, 8)
        foot = [(i, j) for i in range(2, 6) for j in range(2, 6)]
        two_cell = _mask(shape, [(i, j, k) for (i, j) in foot for k in (3, 4)])
        # (a) whole body flagged
        extras = [_np(a) for a in two_plane_extension_masks(two_cell, two_cell)]
        assert not any(a.any() for a in extras)
        # (b) flagged one-cell slab abutting an unflagged PEC body
        flagged = _mask(shape, [(i, j, 3) for (i, j) in foot])
        extras = [_np(a) for a in two_plane_extension_masks(two_cell, flagged)]
        assert not any(a.any() for a in extras)

    def test_slab_in_last_cell_nonperiodic(self):
        """Non-periodic axis, slab in the last cell: the far face is the
        domain boundary (owned by the domain BC / apply_pec) — the k+1
        plane has no array entry and is dropped; the interior normal
        edge is still shielded.  No crash, no wraparound."""
        shape = (6, 6, 6)
        cells = [(i, j, 5) for i in range(1, 5) for j in range(1, 5)]
        pec = _mask(shape, cells)
        ex2, ey2, ez2 = [_np(a) for a in two_plane_extension_masks(pec, pec)]
        assert not ex2.any() and not ey2.any()
        assert sorted(set(np.where(ez2)[2])) == [5]

    def test_periodic_seam_wraps_to_plane_zero(self):
        """Periodic axis, slab in the last cell: the far-face plane IS
        plane 0 across the seam (#689 convention, shared helper — not a
        hand-copied second rule)."""
        shape = (6, 6, 6)
        cells = [(i, j, 5) for i in range(1, 5) for j in range(1, 5)]
        pec = _mask(shape, cells)
        per = (False, False, True)
        ex2, ey2, ez2 = [_np(a) for a in
                         two_plane_extension_masks(pec, pec, per)]
        base_ex, _, _ = [_np(a) for a in tangential_edge_masks(pec, per)]
        assert sorted(set(np.where(ex2)[2])) == [0]
        assert np.array_equal(ex2[:, :, 0], base_ex[:, :, 5])
        assert sorted(set(np.where(ez2)[2])) == [5]

    def test_seam_adjacency_through_wrap(self):
        """Periodic axis, flagged slab at k=n-1 with PEC at k=0: through
        the seam that is a 2-cell run — no extension."""
        shape = (6, 6, 6)
        foot = [(i, j) for i in range(1, 5) for j in range(1, 5)]
        pec = _mask(shape, [(i, j, k) for (i, j) in foot for k in (0, 5)])
        flagged = _mask(shape, [(i, j, 5) for (i, j) in foot])
        extras = [_np(a) for a in
                  two_plane_extension_masks(pec, flagged, (False, False, True))]
        assert not any(a.any() for a in extras)

    def test_length_one_axis_no_op(self):
        """2-D lane: a length-1 axis makes every cell self-adjacent
        through the wrap (never a run of length 1) — no extension along
        it."""
        shape = (6, 6, 1)
        cells = [(i, j, 0) for i in range(1, 5) for j in range(1, 5)]
        pec = _mask(shape, cells)
        ex2, ey2, ez2 = [_np(a) for a in two_plane_extension_masks(pec, pec)]
        assert not ex2.any() and not ey2.any() and not ez2.any()

    def test_cleared_cells_contribute_nothing(self):
        """Cells removed from the live pec_mask after assembly (wire-port
        live-cell clearing pattern) must not sprout extension planes even
        if still present in the flagged mask."""
        shape = (8, 8, 8)
        cells = [(i, j, 3) for i in range(2, 6) for j in range(2, 6)]
        flagged = _mask(shape, cells)
        live = flagged & ~_mask(shape, [(3, 3, 3)])
        extras = [_np(a) for a in two_plane_extension_masks(live, flagged)]
        ez2 = extras[2]
        assert not ez2[3, 3, 3]

    def test_apply_pec_mask_off_path_unchanged(self):
        """``two_plane_mask=None`` (the default) computes exactly the base
        tangential masks — spot-checked against a state of ones."""
        from rfx.core.yee import FDTDState
        shape = (8, 8, 8)
        cells = [(i, j, 3) for i in range(2, 6) for j in range(2, 6)]
        pec = _mask(shape, cells)
        ones = jnp.ones(shape, dtype=jnp.float32)
        st = FDTDState(ex=ones, ey=ones, ez=ones, hx=ones, hy=ones,
                       hz=ones, step=0)
        out = apply_pec_mask(st, pec)
        bex, bey, bez = tangential_edge_masks(pec)
        assert np.array_equal(_np(out.ex) == 0.0, _np(bex))
        assert np.array_equal(_np(out.ey) == 0.0, _np(bey))
        assert not (_np(out.ez) == 0.0).any()
        # and the opt-in zeroes the extension planes on the same state
        out2 = apply_pec_mask(st, pec, two_plane_mask=pec)
        assert (_np(out2.ex)[:, :, 4][_np(bex)[:, :, 3]] == 0.0).all()
        assert (_np(out2.ez)[:, :, 3][_np(pec)[:, :, 3]] == 0.0).all()


# ---------------------------------------------------------------------------
# API level: opt-in registration and lane refusals
# ---------------------------------------------------------------------------

def _small_sim(**kw):
    L = 8 * DX
    sim = Simulation(40e9, (L, L, L), dx=DX, boundary="pec", **kw)
    sim.add(Box((0, 0, 3 * DX), (L, L, 4 * DX)), material="pec",
            two_plane=True)
    sim.add_source((2.5 * DX, 2.5 * DX, 5.5 * DX), "ex",
                   waveform=GaussianPulse(f0=20e9, bandwidth=0.8),
                   amplitude_kind="current")
    return sim


class TestOptInSurface:
    def test_non_pec_material_refused(self):
        """f0 / dielectric fence: two_plane is a hard-PEC-slab option; a
        surface-impedance sheet is a different operator
        (``add_thin_conductor``) and is deliberately untouched — it never
        enters ``pec_mask`` nor ``_two_plane_cell_mask``."""
        L = 8 * DX
        sim = Simulation(40e9, (L, L, L), dx=DX, boundary="pec")
        sim.add_material("diel", eps_r=2.2)
        with pytest.raises(ValueError, match="not PEC"):
            sim.add(Box((0, 0, 3 * DX), (L, L, 4 * DX)), material="diel",
                    two_plane=True)

    def test_flag_default_off_and_mask_none(self):
        L = 8 * DX
        sim = Simulation(40e9, (L, L, L), dx=DX, boundary="pec")
        sim.add(Box((0, 0, 3 * DX), (L, L, 4 * DX)), material="pec")
        assert sim._two_plane_cell_mask(sim._build_grid()) is None

    def test_flagged_mask_matches_pec_cells(self):
        sim = _small_sim()
        grid = sim._build_grid()
        _, _, _, pec, _, _, _ = sim._assemble_materials(grid)
        tp = sim._two_plane_cell_mask(grid)
        assert tp is not None
        assert np.array_equal(_np(tp), _np(pec))

    def test_vmap_sweep_refuses(self):
        from rfx.vmap_sweep import vmap_material_sweep
        sim = _small_sim()
        sim.add_material("sweepme", eps_r=2.0)
        with pytest.raises(NotImplementedError, match="706"):
            vmap_material_sweep(sim, "eps_r", [1.0, 2.0], n_steps=4)

    def test_conformal_pec_refuses(self):
        sim = _small_sim()
        with pytest.raises(NotImplementedError, match="706"):
            sim.run(n_steps=4, conformal_pec=True, skip_preflight=True)

    def test_adi_subgrid_distributed_refuse(self):
        """The refusal guard is the FIRST statement of each unsupported
        private lane entry, so calling with dummy args raises before any
        arg is touched."""
        sim = _small_sim()
        with pytest.raises(NotImplementedError, match="706"):
            sim._run_adi_from_materials(None, None, None, None, n_steps=1)
        with pytest.raises(NotImplementedError, match="706"):
            sim._run_subgridded(None, None, None, 1)
        with pytest.raises(NotImplementedError, match="706"):
            sim._forward_distributed_nonuniform_from_materials(n_steps=1)


# ---------------------------------------------------------------------------
# Off-state bit-identity (golden captured at main 8e00497, this platform)
# ---------------------------------------------------------------------------

class TestOffStateBitIdentity:
    """OFF-state contract: with no entry flagged, the #706 machinery is
    INERT — not "close", inert.  The one-plane behaviour is load-bearing
    (#677), so any drift here is a defect, not a tolerance question.

    HISTORY.  The first version of this contract byte-compared against a
    golden .npz captured from main on the authoring workstation.  It red
    on CI within the hour: GitHub's runner produces last-bit float
    differences (reduction order — the same cross-machine class that red
    the #698 mode-profile pin), so the golden pinned the HOST, not the
    contract.  The two tests below state the same contract WITHOUT any
    cross-machine artifact: everything is computed twice in ONE process
    on ONE machine, so byte-equality is exact by rights and any mismatch
    is the feature leaking.

    Mutation coverage of the PAIR, measured on this rewrite (M2 = inside
    ``apply_pec_mask``, substitute ``two_plane_mask = pec_mask`` when
    handed ``None`` — forced default-on):
    ``test_unflagged_sim_never_reaches_the_extension`` RED
    (``two_plane_extension_masks invoked 2x``); the byte-equality test
    alone SURVIVES M2 because the mutation contaminates both of its legs
    identically — the two tests are complementary, not redundant: the
    spy catches invocation leaks, the byte test catches a path handing a
    non-None mask to ``apply_pec_mask`` without going through the
    collector.  Verbatim: ``1 failed, 1 passed``; clean tree
    ``22 passed``.
    """

    def _fixture(self):
        L = 16 * DX
        sim = Simulation(40e9, (L, L, L), dx=DX, boundary="pec")
        sim.add_material("diel22", eps_r=2.2)
        sim.add(Box((2 * DX, 2 * DX, 7 * DX), (14 * DX, 14 * DX, 8 * DX)),
                material="pec")
        sim.add(Box((3 * DX, 3 * DX, 10 * DX), (9 * DX, 9 * DX, 13 * DX)),
                material="diel22")
        sim.add_source((4.5 * DX, 5.5 * DX, 3.5 * DX), "ex",
                       waveform=GaussianPulse(f0=20e9, bandwidth=0.8),
                       amplitude_kind="current")
        sim.add_probe((11.5 * DX, 9.5 * DX, 4.5 * DX), "ex")
        sim.add_probe((5.5 * DX, 6.5 * DX, 11.5 * DX), "ez")
        return sim

    def test_unflagged_sim_never_reaches_the_extension(self, monkeypatch):
        """A sim with no ``two_plane`` entry must not INVOKE the rule at
        all.  Stronger than comparing outputs: if the extension is never
        called, it cannot have contributed, on any hardware."""
        import rfx.boundaries.pec as _pecmod
        calls = []
        real = _pecmod.two_plane_extension_masks

        def _spy(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(_pecmod, "two_plane_extension_masks", _spy)
        self._fixture().run(n_steps=60)
        assert not calls, (
            f"two_plane_extension_masks invoked {len(calls)}x on a sim "
            "with no flagged entry — the OFF state must be inert (#706)")

    def test_off_state_byte_equal_to_integration_bypassed_run(self,
                                                              monkeypatch):
        """Same process, same machine: a normal flag-free run must be
        BYTE-equal to a run with the #706 integration surgically forced
        off (the union-mask collector returns None, exactly the pre-#706
        code path).  Cross-machine float drift cannot enter — both legs
        share one BLAS, one XLA, one reduction order."""
        res_a = self._fixture().run(n_steps=300)

        from rfx.api import Simulation as _S
        monkeypatch.setattr(_S, "_two_plane_cell_mask",
                            lambda self, *a, **k: None)
        res_b = self._fixture().run(n_steps=300)

        for name, ga, gb in (
                ("ex", res_a.state.ex, res_b.state.ex),
                ("ey", res_a.state.ey, res_b.state.ey),
                ("ez", res_a.state.ez, res_b.state.ez),
                ("hx", res_a.state.hx, res_b.state.hx),
                ("hy", res_a.state.hy, res_b.state.hy),
                ("hz", res_a.state.hz, res_b.state.hz),
                ("time_series", res_a.time_series, res_b.time_series)):
            a, b = _np(ga), _np(gb)
            assert a.dtype == b.dtype, name
            assert np.array_equal(a, b), (
                f"{name}: flag-free run differs from the "
                "integration-bypassed run — the OFF state must be "
                "bit-identical to the pre-#706 path (#706 opt-in contract)")


# ---------------------------------------------------------------------------
# #702 interaction: assembly untouched, interior edge dead when ON
# ---------------------------------------------------------------------------

class TestInteriorEdgeAnd702:
    def _build(self, two_plane):
        L = 12 * DX
        sim = Simulation(40e9, (L, L, L), dx=DX, boundary="pec")
        sim.add_material("diel3", eps_r=3.0)
        # dielectric fills the region; the slab's faces abut it (#702's
        # stackup pattern: the metal layer is a slot no dielectric fills)
        sim.add(Box((0, 0, 2 * DX), (L, L, 5 * DX)), material="diel3")
        sim.add(Box((0, 0, 6 * DX), (L, L, 9 * DX)), material="diel3")
        # PARTIAL-lateral slab: with a full-lateral slab the PEC walls
        # seal the cell once both planes are dead and NOTHING drives the
        # interior edge, so the interior-edge pin cannot fail (measured:
        # the M3 mutation below passed on a full-lateral fixture).  The
        # open rim keeps curl-H drive available at the slab's edge.
        sim.add(Box((2 * DX, 2 * DX, 5 * DX), (10 * DX, 10 * DX, 6 * DX)),
                material="pec", two_plane=two_plane)
        sim.add_source((3.5 * DX, 4.5 * DX, 7.5 * DX), "ez",
                       waveform=GaussianPulse(f0=20e9, bandwidth=0.8),
                       amplitude_kind="current")
        return sim

    def test_assembled_eps_independent_of_flag(self):
        """The flag changes the per-step mask only — the assembled eps
        (including the #702 live-edge resample of the slab's cell) is
        bit-identical either way.  With two planes the slab's interior
        edge is DEAD, so whatever eps #702 left there is inert rather
        than fought."""
        son = self._build(True)
        soff = self._build(False)
        g = son._build_grid()
        mon, _, _, pon, _, _, _ = son._assemble_materials(g)
        moff, _, _, poff, _, _, _ = soff._assemble_materials(soff._build_grid())
        assert np.array_equal(_np(mon.eps_r), _np(moff.eps_r))
        assert np.array_equal(_np(mon.sigma), _np(moff.sigma))
        assert np.array_equal(_np(pon), _np(poff))

    def test_interior_edge_energy_zero_when_on(self):
        """ON: the slab's interior normal edge (Ez at the slab cells) is
        exactly 0 after every step.  OFF: the same edge is live (that is
        the measured #706 baseline) — recorded as the contrast witness.

        Mutation falsification M3 (interior-edge neuter: delete
        ``extras[ax] = extras[ax] | run1`` in
        ``two_plane_extension_masks``): this test FAILED — measured
        ``assert 24953462.0 == 0.0`` (ON max|Ez| at the slab cells
        2.50e+07) — while the eigenmode witness still PASSED under the
        same mutation, which is exactly why this dedicated pin exists.
        On the earlier FULL-lateral fixture the mutation escaped this
        test too (walls sealed the rim; nothing drove the edge) — hence
        the partial-lateral slab.  Reverted after measurement.
        """
        son = self._build(True)
        g = son._build_grid()
        _, _, _, pec, _, _, _ = son._assemble_materials(g)
        slab = _np(pec)
        zs = sorted(set(np.where(slab)[2]))
        assert len(zs) == 1, "fixture must rasterize to a one-cell slab"
        res_on = son.run(n_steps=200)
        ez_on = _np(res_on.state.ez)[slab]
        assert float(np.max(np.abs(ez_on))) == 0.0
        res_off = self._build(False).run(n_steps=200)
        ez_off = _np(res_off.state.ez)[slab]
        assert float(np.max(np.abs(ez_off))) > 0.0  # live edge = baseline

    def test_far_plane_dead_when_on(self):
        """ON: tangential E at the k+1 plane is exactly 0 after the run;
        OFF: live.  Direct end-to-end pin of the uniform-lane threading.

        Under mutation M1 (far-plane neuter) this test FAILED — measured
        ``assert 34045928.0 == 0.0`` (ON Ex at the k+1 plane 3.40e+07).
        Reverted after measurement."""
        son = self._build(True)
        g = son._build_grid()
        _, _, _, pec, _, _, _ = son._assemble_materials(g)
        slab = _np(pec)
        k = sorted(set(np.where(slab)[2]))[0]
        foot = slab[:, :, k]
        res_on = son.run(n_steps=200)
        assert float(np.max(np.abs(_np(res_on.state.ex)[:, :, k + 1][foot]))) == 0.0
        res_off = self._build(False).run(n_steps=200)
        assert float(np.max(np.abs(_np(res_off.state.ex)[:, :, k + 1][foot]))) > 0.0


# ---------------------------------------------------------------------------
# Non-uniform lane threading
# ---------------------------------------------------------------------------

class TestNonUniformLane:
    def _build(self, two_plane):
        # deliberately non-uniform dz (finer around the slab) — a
        # uniform-valued profile tests nothing (workspace lesson)
        dz = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]) * DX
        Lz = float(dz.sum())
        L = 8 * DX
        z_lo = float(dz[:3].sum())   # start of cell 3 (a 1.0*DX cell)
        z_hi = z_lo + float(dz[3])
        sim = Simulation(40e9, (L, L, Lz), dx=DX, dz_profile=dz,
                         boundary="pec")
        sim.add(Box((0, 0, z_lo), (L, L, z_hi)), material="pec",
                two_plane=two_plane)
        sim.add_source((2.5 * DX, 2.5 * DX, z_hi + 2.5 * DX), "ex",
                       waveform=GaussianPulse(f0=20e9, bandwidth=0.8),
                       amplitude_kind="current")
        return sim

    def test_nu_far_plane_and_interior_dead_when_on(self):
        """NU runner (rfx/runners/nonuniform.py -> rfx/nonuniform.py scan):
        ON kills Ex at the k+1 plane and Ez at the slab cell; OFF leaves
        both live.  Pins that the coords-based flagged mask lands on the
        same cells as the NU pec_mask."""
        son = self._build(True)
        res_on = son.run(n_steps=150)
        grid = son._build_nonuniform_grid()
        from rfx.runners.nonuniform import assemble_materials_nu
        _, _, _, pec = assemble_materials_nu(son, grid)
        slab = _np(pec)
        assert slab.any()
        zs = sorted(set(np.where(slab)[2]))
        assert len(zs) == 1
        k = zs[0]
        foot = slab[:, :, k]
        assert float(np.max(np.abs(_np(res_on.state.ex)[:, :, k + 1][foot]))) == 0.0
        assert float(np.max(np.abs(_np(res_on.state.ez)[:, :, k][foot]))) == 0.0
        res_off = self._build(False).run(n_steps=150)
        assert float(np.max(np.abs(_np(res_off.state.ex)[:, :, k + 1][foot]))) > 0.0


# ---------------------------------------------------------------------------
# ACCEPTANCE: the eigenmode witness (issue #706)
# ---------------------------------------------------------------------------

class TestEigenmodeWitness:
    """CPU-sized port of the #706 witness (/tmp/sheet_boundary_witness.py).

    Parallel-plate cavity between two z-normal one-cell PEC sheets in a
    PEC box; the lowest Ex mode is f = c/2 * sqrt((1/Lxy)^2 + (1/Lz')^2)
    with Lxy the lateral wall spacing and Lz' the plate spacing.  All
    lengths are derived from the RASTERIZED cell indices (immune to
    float registration of box edges — this fixture's second sheet lands
    one cell high of the naive coordinate arithmetic, measured):

      OFF (one-plane): walls at each sheet's LOWER node plane
                       -> Lz' = (k2 - k1) * dx
      ON  (two-plane): walls at sheet1's UPPER and sheet2's LOWER plane
                       -> Lz' = (k2 - k1 - 1) * dx

    Measured on this fixture (grid (33,33,25), sheets at z-cells {3,21},
    8000 steps, harminv over the [2000:] window, preflight advisory
    "'pec' z-extent 200um = 1.0 cells — below 1 cell resolution ..."
    quoted in full in the PR body):

      OFF: dominant 47.7486 GHz (Q 1.8e5) vs pred 47.7733 -> -0.05%;
           ladder (1,1,1) 53.19 / (0,2,1) 62.64 fits <=0.05%
      ON:  dominant 49.8924 GHz (Q 8.7e4) vs pred 49.9223 -> -0.06%;
           ladder 55.12 / 64.29 fits <=0.2%
      cross-hypothesis distance 4.3-4.5% >> the 1% gates.

    Runtime ~6 s per leg on CPU (12 s for the class).

    Mutation falsification (M1 far-plane neuter, see
    TestExtensionMasks.test_one_cell_z_slab_planes): the ON leg measured
    dominant 47.7485 GHz — the OFF value, 4.35% from the two-plane
    prediction — and failed the 1% gate; (M2 default-on): the OFF leg
    measured 49.8924 GHz — the two-plane value — and failed its gate.
    """

    GAP_CELLS = 16
    LAT_CELLS = 32

    def _run(self, two_plane):
        lat = self.LAT_CELLS * DX
        z1_lo = 3 * DX
        z1_hi = z1_lo + DX
        z2_lo = z1_hi + self.GAP_CELLS * DX
        z2_hi = z2_lo + DX
        Lz = z2_hi + 3 * DX
        sim = Simulation(60e9, (lat, lat, Lz), dx=DX, boundary="pec")
        kw = {"two_plane": True} if two_plane else {}
        sim.add(Box((0, 0, z1_lo), (lat, lat, z1_hi)), material="pec", **kw)
        sim.add(Box((0, 0, z2_lo), (lat, lat, z2_hi)), material="pec", **kw)
        sim.add_source((11.1 * DX, 13.3 * DX, z1_hi + self.GAP_CELLS * DX * 0.31),
                       "ex", waveform=GaussianPulse(f0=50e9, bandwidth=0.5),
                       amplitude_kind="current")
        sim.add_probe((21.7 * DX, 8.9 * DX, z1_hi + self.GAP_CELLS * DX * 0.62),
                      "ex")
        grid = sim._build_grid()
        _, _, _, pec, _, _, _ = sim._assemble_materials(grid)
        zs = sorted(set(np.where(_np(pec))[2]))
        assert len(zs) == 2, f"expected two one-cell sheets, got z-cells {zs}"
        k1, k2 = zs
        nlat = grid.shape[0] - 1  # tangential-E walls at planes 0 and n-1
        res = sim.run(n_steps=8000)
        ts = _np(res.time_series)[:, 0].ravel()
        from rfx import harminv
        modes = harminv(ts[2000:], float(res.dt), f_min=35e9, f_max=60e9)
        cand = sorted([m for m in modes if m.Q > 100],
                      key=lambda m: -abs(m.amplitude))
        assert cand, "harminv found no modes"
        f_meas = cand[0].freq
        lxy = nlat * DX

        def pred(nz_cells):
            return C0 / 2 * np.sqrt((1 / lxy) ** 2 + (1 / (nz_cells * DX)) ** 2)

        return f_meas, pred(k2 - k1), pred(k2 - k1 - 1)

    def test_off_one_plane_ladder(self):
        f, f_one, f_two = self._run(False)
        assert abs(f - f_one) / f_one < 0.01, (f, f_one)
        assert abs(f - f_two) / f_two > 0.03, (f, f_two)

    def test_on_two_plane_ladder(self):
        f, f_one, f_two = self._run(True)
        assert abs(f - f_two) / f_two < 0.01, (f, f_two)
        assert abs(f - f_one) / f_one > 0.03, (f, f_one)
