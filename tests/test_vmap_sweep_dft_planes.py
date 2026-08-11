"""Tests for DFT plane accumulators on the vmap fast path (#578).

CPU-runnable on tiny grids (intentionally not gpu-marked, matching
``test_vmap_sweep_eligibility.py``, so the PR gate exercises this file).

Comparator-class work (rfx-known-issues.md case ledger; workspace R5): the
equivalence tests below carry a predeclared tolerance derived from an
inspected per-bin dump (not asserted blind), and a one-sided falsifier
ritual (deliberate sign flip in the vmap DFT kernel, which must turn the
equivalence check red) was run before this tolerance was trusted.

CORRECTED (#637): this docstring previously reported a ~5e-4 relative
floor on this exact fixture and attributed it to "ordinary float32
accumulation roundoff between the two independently-coded scan bodies".
That was wrong. The fixture's ``substrate`` box spans the full transverse
(y, z) extent of the domain, i.e. it touches the CPML faces; the
material-NAMED sweep mask (``Shape.mask``, geometry cells only) never
covers the CPML padding, so every swept batch element ran with a padding
absorber matched to the BASE simulation's eps_r (4.0) instead of its own
-- 2040 of 12167 cells wrong on this exact grid (re-measured after the
y/z hi bound below was nudged past the domain edge for issue #627's
rebase interaction -- see that bound's own inline comment; the earlier
780-cell count predates that nudge and undercounted because #627a's own
pre-fix bug ALSO left the y_hi/z_hi pad incorrectly vacuum in the
reference, masking part of this defect's true footprint). The ~5e-4
floor was that defect, not roundoff: moving the same slab off the CPML
faces (so no material lands in the padding) made the identical
comparison exactly ``0.0`` -- not merely smaller, bit-identical -- which
a genuine independent-roundoff floor would not do. #637 fixed the
padding (``_extend_batched_cpml_pad`` in ``rfx/vmap_sweep.py`` re-runs
the same per-face edge-slice-copy ``_assemble_materials`` uses, on the
already batch-correct interior, so each swept batch element's padding
matches what ``Simulation.run()`` would build for that value).

R5 per-bin dump, POST-FIX (batched vmap vs. sequential ``sim.run()``,
plane p1 = axis=x, coordinate=0.010 m, component=ez, n_freqs=4, eps_r=2.0,
n_steps=60 -- eps_r=6.0 and the non-CPML (``pec``) boundary are also
exactly ``0.0`` at every bin and are omitted for brevity):

    freq (Hz)     |run acc|      |vmap acc|     complex maxdiff   rel diff
    5.0000e+08    4.856995e-12   4.856995e-12   0.000000e+00      0.000e+00
    2.0000e+09    4.365797e-12   4.365797e-12   0.000000e+00      0.000e+00
    3.5000e+09    3.553883e-12   3.553883e-12   0.000000e+00      0.000e+00
    5.0000e+09    2.804024e-12   2.804024e-12   0.000000e+00      0.000e+00

Falsifier (temporary sign flip of the vmap DFT kernel to ``exp(+1j...)``,
reverted via ``git checkout`` immediately after each run): relative error
jumped to 0.11-1.79 (O(1)) uniformly across every bin, both when run as a
standalone script and when run against this committed pytest file -- and
ONLY the ``TestVmapDftPlaneFastPath`` tests went red, confirming the
falsifier is localized to the code path it is meant to catch (this part
of the ritual is unchanged by #637 and was re-run against the fixed code
to confirm it still catches a kernel defect). #637's own falsifier
(reverting ONLY the pad-extension fix, i.e. the 2040-cell defect, while
keeping everything else fixed) reproduces a ~9e-4 floor (8.83e-4
measured, up from the pre-rebase ~5e-4 for the same reason the cell
count moved) -- reviewable by reverting ``_extend_batched_cpml_pad``'s
call sites in ``rfx/vmap_sweep.py`` and re-running this file: exactly
``test_dft_plane_matches_run_cpml``,
``TestVmapMaterialSweepCpmlPad::test_material_named_sweep_pad_cells_match_run_materials``,
``TestVmapMaterialSweepCpmlPad::test_dft_plane_matches_run_alternate_geometry``,
and ``test_dft_plane_matches_run_cpml_x64`` go red (the global-sweep pin
and everything unrelated stay green) -- an independent reviewer
reproduced this exact four-test signature separately (on the pre-rebase
fixture; not re-verified against this file's post-rebase numbers, but
the mechanism and the four-test signature are unchanged, only the exact
magnitude moved with the cell count). The gate below (``rtol=1e-6``) is
anchored near the observed floor with margin for cross-machine
floating-point reduction-order jitter (this repo's own experience is
that cross-machine float comparisons are not bit-exact -- see the CI
slow-suite agent-memory entry; an independent re-measurement of the
CHANGELOG's separate 7-configuration representativeness sweep on
different hardware got exactly ``0.0`` on all seven, where this
session's machine measured them at-or-below ~4e-7 -- same conclusion,
different floating-point path), not fitted to hide a defect: it is
~1000x tighter than the old ``2e-3`` and five to six orders of magnitude
below every pre-fix defect measurement (8.83e-4 on this fixture; 4.2e-3
to 5.8e-2 across that representativeness sweep, PR
`fix/637-vmap-sweep-cpml-pad`).

Gate bracketing (why ``1e-6`` and not looser or tighter): from BELOW,
every measured post-fix residual on this file's DFT-plane fixtures is
<=3.3e-8 at n_steps=60 (>=30x margin; see the x64 class docstring for
how this floor grows with ``n_steps`` on the SAME fixture -- it is not a
fixture-independent bound), and the sibling fixture where the two paths
genuinely execute different arithmetic --
``TestVmapAmplitudeKindCurrent`` (dynamic per-batch Cb-normalization,
not a shared code path with the DFT-plane kernel; a raw-time-series
comparison, not this file's per-bin DFT metric) -- measured on this
session's machine at 1.74e-7 (cpml, eps_r=2.0) / 4.17e-7 (eps_r=6.0):
still comfortably under even at the looser value -- a second data point
for the cross-machine-jitter margin above, not just the ``1e-6`` anchor
(an independent, pre-rebase re-measurement on different hardware landed
in the same 1.5e-7 to 3.4e-7 order of magnitude; not re-verified
post-rebase). From ABOVE, the weakest defect signal the gate must still
catch is ``test_dft_plane_matches_run_alternate_geometry``'s ~1.3e-5
pre-fix measurement; a gate at ``1e-5`` would leave that only 1.3x
headroom and destroy it as a falsifier. ``1e-6`` sits inside both
bounds. Thread-count
sweep (1 to 192 CPUs, same fixtures): zero spread -- the compared
quantities (a single batch element's DFT accumulator vs. a single
sequential run's) contain no cross-batch or cross-space reduction for
either code path to reorder differently under a different thread count,
so there is no mechanism for this floor to move with parallelism.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

try:  # modern JAX: scoped x64 promoted to top-level (experimental removed v0.8.0)
    from jax import enable_x64 as _enable_x64
except ImportError:  # older JAX (< ~0.4.31)
    from tests._x64_compat import enable_x64 as _enable_x64

from rfx import Simulation, GaussianPulse, Box
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.vmap_sweep import (
    vmap_material_sweep, VmapSweepResult, _build_batched_materials,
)

_FALLBACK_MATCH = "Falling back to sequential"


def _dft_sim(boundary: str = "cpml", eps_r: float = 4.0,
             amplitude_kind: str | None = None) -> Simulation:
    """CPML/PEC dielectric slab with one DFT plane MID-domain (not near a
    boundary): moving the plane away from the domain edge matters — near
    the edge the accumulated signal is dominated by roundoff noise (near
    zero at these short step counts) and a relative-tolerance check would
    pass vacuously without exercising anything (discovered while building
    this file: a plane at x=0.016 gave accumulator magnitudes ~1e-15 with
    no real signal; x=0.010 gives ~1e-12, a genuine non-trivial spectrum).
    """
    kwargs = {"cpml_layers": 6} if boundary == "cpml" else {}
    sim = Simulation(
        freq_max=5e9, domain=(0.02, 0.02, 0.02), boundary=boundary,
        dx=0.002, **kwargs,
    )
    sim.add_material("substrate", eps_r=eps_r)
    # y/z hi bound is domain-edge + half a cell, not exactly the domain
    # edge: Box's rasterization is half-open ([lo, hi), see Box's own
    # docstring) so a hi bound landing EXACTLY on the domain edge drops
    # that edge node from the box's own mask -- unrelated to #637, but it
    # interacts with it post-#627 (rfx.geometry.rasterize_grid's hi-face
    # vacuum fallback, closes-#627 fce1091). The margin was added because
    # the vmap helper did not reproduce that fallback, so every test
    # built on this fixture would otherwise have failed for a SEPARATE,
    # already-disclosed reason unrelated to what they test. #643 closed
    # that gap (the helper now vmaps the shared rule, so the exact-edge
    # case agrees bit-for-bit too -- see
    # test_exact_hi_face_touch_matches_run_via_shared_fallback and
    # TestVmapBatchedPadByteIdentity's all_six_faces_exact_hi row). The
    # margin is KEPT so the numbers quoted throughout this file stay
    # attached to the geometry they were measured on; it is no longer
    # load-bearing. It is far short of entering the CPML pad itself
    # (half a cell vs. the pad's full cpml_layers=6 cells), so it only
    # recovers the one excluded interior node -- confirmed directly
    # (0 mask cells inside the pad region) before landing this change.
    sim.add(Box((0.005, 0, 0), (0.015, 0.021, 0.021)), material="substrate")
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=GaussianPulse(f0=3e9),
                    amplitude_kind=amplitude_kind)
    sim.add_probe((0.005, 0.01, 0.01), "ez")
    sim.add_dft_plane_probe(axis="x", coordinate=0.010, component="ez",
                             n_freqs=4, name="p1")
    return sim


def _assert_dft_planes_match(vmap_planes: dict, ref_planes: dict, *,
                              rtol: float, ctx: str) -> None:
    """Compare dict[str, DFTPlaneProbe] (a Result) against a single batch
    slice of VmapSweepResult.dft_planes, comparison arithmetic in f64
    (#527 style: the accumulators are complex64-scale; casting up before
    diffing avoids a spurious float32-comparator artefact).

    Per-FREQUENCY-BIN relative error, normalized by that bin's own peak
    magnitude over the plane -- NOT a blind elementwise
    ``assert_allclose``. A DFT plane has near-zero cells far from the
    source/trace where the true signal is dominated by roundoff-level
    cancellation noise; an elementwise rtol check flags those as huge
    relative mismatches even though both paths agree to within float32
    precision on the actual signal (discovered building this test: doing
    the elementwise check first gave 27% "mismatched" elements, all at
    |value| ~ 1e-16 next to a ~1e-12 peak). This is the same max-abs-diff
    / max-abs-reference ratio manually verified and inlined into this
    module's docstring (the R5 per-bin dump) before this gate was written.
    """
    for name, probe in ref_planes.items():
        ref_acc = np.asarray(probe.accumulator, dtype=np.complex128)
        got_acc = np.asarray(vmap_planes[name], dtype=np.complex128)
        assert ref_acc.shape == got_acc.shape, (
            f"DFT plane {name!r} shape mismatch ({ctx}): "
            f"{got_acc.shape} vs {ref_acc.shape}"
        )
        for fi in range(ref_acc.shape[0]):
            mag_ref = np.max(np.abs(ref_acc[fi]))
            diff = np.max(np.abs(got_acc[fi] - ref_acc[fi]))
            assert mag_ref > 0.0, (
                f"DFT plane {name!r} freq bin {fi} is all-zero in the "
                f"reference ({ctx}) -- fixture does not exercise signal "
                "at this bin, tolerance check is meaningless"
            )
            rel = diff / mag_ref
            assert rel < rtol, (
                f"DFT plane {name!r} freq bin {fi} mismatch ({ctx}): "
                f"rel={rel:.3e} >= rtol={rtol:.3e} "
                f"(|ref|_max={mag_ref:.3e}, maxdiff={diff:.3e})"
            )


class TestVmapDftPlaneFastPath:
    """Fast-path (vmap-batched) DFT accumulator vs. direct sim.run(),
    per swept element -- the comparator-class equivalence gate."""

    def test_dft_plane_matches_run_cpml(self):
        eps_values = np.array([2.0, 6.0])
        n_steps = 60

        vmap_res = vmap_material_sweep(
            _dft_sim("cpml", eps_r=4.0), "substrate.eps_r", eps_values,
            n_steps=n_steps,
        )
        assert vmap_res.dft_planes is not None
        assert set(vmap_res.dft_planes.keys()) == {"p1"}
        # axis="x" plane -> (n1, n2) = (Ny, Nz) INCLUDING CPML padding;
        # not hand-computed here (grid padding is an implementation detail)
        # -- cross-checked against the shape run() itself produces below.
        ref0 = _dft_sim("cpml", eps_r=float(eps_values[0])).run(n_steps=n_steps)
        expected_shape = (len(eps_values),) + tuple(
            ref0.dft_planes["p1"].accumulator.shape)
        assert vmap_res.dft_planes["p1"].shape == expected_shape

        for idx, ev in enumerate(eps_values):
            ref = _dft_sim("cpml", eps_r=float(ev)).run(n_steps=n_steps)
            # #637: post-fix this is exactly 0.0 at every bin (measured);
            # rtol=1e-6 anchors near that floor with margin for
            # cross-machine float jitter, not fitted to the old defect.
            _assert_dft_planes_match(
                {"p1": vmap_res.dft_planes["p1"][idx]},
                ref.dft_planes, rtol=1e-6, ctx=f"cpml eps_r={ev}",
            )

    def test_dft_plane_matches_run_pec(self):
        """Non-CPML (plain PEC-wall) boundary takes the OTHER
        ``_build_vmap_scan_fn`` branch (``use_cpml=False``) -- pin it too,
        it is a structurally different scan body."""
        eps_values = np.array([2.0, 6.0])
        n_steps = 60

        vmap_res = vmap_material_sweep(
            _dft_sim("pec", eps_r=4.0), "substrate.eps_r", eps_values,
            n_steps=n_steps,
        )
        for idx, ev in enumerate(eps_values):
            ref = _dft_sim("pec", eps_r=float(ev)).run(n_steps=n_steps)
            # PEC has no CPML padding (grid.pad_*=0), so #637 never applied
            # here -- measured exactly 0.0 at every bin both before and
            # after the fix. Tightened alongside the cpml gate for the
            # same reason (was sitting at a loose rtol=2e-3 with no
            # measurement behind it).
            _assert_dft_planes_match(
                {"p1": vmap_res.dft_planes["p1"][idx]},
                ref.dft_planes, rtol=1e-6, ctx=f"pec eps_r={ev}",
            )

    def test_dft_plane_dtype_matches_run(self):
        """#477/#484 x64-contract pin: the vmap accumulator dtype must be
        the SAME complex dtype run() would produce (complex64 with x64
        off, the default here)."""
        vmap_res = vmap_material_sweep(
            _dft_sim("cpml"), "substrate.eps_r", np.array([2.0, 4.0]),
            n_steps=20,
        )
        ref = _dft_sim("cpml").run(n_steps=20)
        assert vmap_res.dft_planes["p1"].dtype == ref.dft_planes["p1"].accumulator.dtype

    def test_no_dft_plane_registered_leaves_dft_planes_none(self):
        """Control: a sim with no add_dft_plane_probe call must NOT
        fabricate a dft_planes dict."""
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02),
                          boundary="cpml", cpml_layers=6, dx=0.002)
        sim.add_material("substrate", eps_r=4.0)
        sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="substrate")
        sim.add_source((0.01, 0.01, 0.01), "ez", waveform=GaussianPulse(f0=3e9))
        sim.add_probe((0.005, 0.01, 0.01), "ez")

        res = vmap_material_sweep(sim, "substrate.eps_r", np.array([2.0, 4.0]),
                                   n_steps=20)
        assert res.dft_planes is None


class TestVmapMaterialSweepCpmlPad:
    """#637 regression coverage: a material-NAMED sweep must reach the
    CPML padding replicated from that material, not inherit the base
    simulation's padding. Three angles, deliberately not three copies of
    the same fixture (this arc was previously burned by a fixture-specific
    claim -- see the #637 PR body's representativeness table):

    1. Mechanism-level: the swept batch ``MaterialArrays`` themselves must
       match what ``Simulation._assemble_materials`` would build for that
       swept value, cell-for-cell -- no time-stepping involved, so this
       isolates ``_build_batched_materials``/``_extend_batched_cpml_pad``
       from any downstream FDTD-kernel equivalence question.
    2. A STRUCTURALLY DIFFERENT geometry from the module's ``_dft_sim``
       fixture (touches only the x_hi CPML face, not all four transverse
       faces; different domain, dx, cpml_layers, and |swept-base| delta)
       -- chosen adverse to "the fix only happens to work for the
       committed shape".
    3. The GLOBAL (non-material-named) sweep path, which was already
       accidentally correct before #637 (``non_vac = eps_r != 1.0`` in
       ``_build_batched_materials`` happens to select the replicated
       padding too) -- pinned so #637's fix to the material-named branch
       cannot regress the global branch it does not touch.
    """

    def test_material_named_sweep_pad_cells_match_run_materials(self):
        """Direct, bit-exact comparison of the batched material arrays
        against ``sim._assemble_materials`` for each swept value -- the
        2040-wrong-cells measurement from the #637 issue (see the module
        docstring for why this count moved from the issue's original 780
        after the #627 rebase), pinned as a regression instead of an
        ad-hoc script."""
        from rfx.vmap_sweep import _build_batched_materials

        eps_values = np.array([2.0, 6.0])
        base_sim = _dft_sim("cpml", eps_r=4.0)
        grid = base_sim._build_grid()
        base_materials, *_ = base_sim._assemble_materials(grid)

        batched = _build_batched_materials(
            base_sim, grid, base_materials, "substrate.eps_r",
            jnp.asarray(eps_values),
        )

        for idx, ev in enumerate(eps_values):
            want_sim = _dft_sim("cpml", eps_r=float(ev))
            want_materials, *_ = want_sim._assemble_materials(grid)
            npt.assert_array_equal(
                np.asarray(batched.eps_r[idx]),
                np.asarray(want_materials.eps_r),
                err_msg=f"batched eps_r at swept value {ev} disagrees with "
                        f"run()'s own _assemble_materials -- CPML padding "
                        f"not correctly swept (#637)",
            )
            # sigma/mu_r are not swept by this param_name and must be
            # untouched (same as run()'s, since the material only varies
            # eps_r here).
            npt.assert_array_equal(
                np.asarray(batched.sigma[idx]), np.asarray(want_materials.sigma))
            npt.assert_array_equal(
                np.asarray(batched.mu_r[idx]), np.asarray(want_materials.mu_r))

    def test_dft_plane_matches_run_alternate_geometry(self):
        """A geometry NOT sharing the committed fixture's shape: touches
        the x_lo CPML face (the fixture above touches y/z faces, never
        x), domain/dx/cpml_layers/delta all differ from ``_dft_sim`` too.

        Built via a falsifier ritual of its own: an earlier draft of this
        test used a box touching the x_HI face at exactly the domain
        edge and passed identically before and after the #637 fix (a
        vacuous, non-discriminating test). Inspecting the assembled
        materials showed why -- ``Box.mask`` is inclusive on a shape's
        ``lo`` corner but not on its ``hi`` corner at an exact domain-edge
        coordinate, so that box never actually reached the interior edge
        cell CPML padding replicates from (0 non-vacuum pad cells, base
        AND swept alike). Touching via the ``lo`` corner instead (as
        below) reaches the padding: measured pre-fix worst rel err
        2.50e-05 / 1.34e-05 (comfortably above the ``rtol=1e-6`` gate),
        exactly ``0.0`` post-fix. (Re-measured after the y/z hi bound
        below was nudged past the domain edge for issue #627's rebase
        interaction, see that bound's own inline comment -- the earlier
        2.08e-05/1.13e-05 pair predates that nudge and is superseded by
        these numbers, not a separate discrepancy.)
        """
        domain = (0.024, 0.016, 0.016)
        cpml_layers = 5
        dx = 0.002
        eps_values = np.array([2.5, 9.0])
        n_steps = 40

        def sim_fn(eps_r):
            sim = Simulation(freq_max=5e9, domain=domain, boundary="cpml",
                              cpml_layers=cpml_layers, dx=dx)
            sim.add_material("slab", eps_r=eps_r)
            # touches x_lo, full y/z extent -- the committed fixture above
            # touches y/z faces and leaves x alone; this is the mirror.
            # y/z hi bound is domain-edge + half a cell, same reason as
            # _dft_sim above (Box's half-open-hi exclusion at an exact
            # domain-edge coordinate, unrelated to #637 but interacting
            # with #627's hi-face pad fallback post-rebase).
            sim.add(Box((0.0, 0.0, 0.0), (0.008, 0.017, 0.017)),
                    material="slab")
            sim.add_source((0.016, 0.008, 0.008), "ez",
                            waveform=GaussianPulse(f0=3e9))
            sim.add_probe((0.018, 0.008, 0.008), "ez")
            sim.add_dft_plane_probe(axis="x", coordinate=0.016,
                                     component="ez", n_freqs=3, name="p1")
            return sim

        vmap_res = vmap_material_sweep(
            sim_fn(5.0), "slab.eps_r", eps_values, n_steps=n_steps,
        )
        for idx, ev in enumerate(eps_values):
            ref = sim_fn(float(ev)).run(n_steps=n_steps, skip_preflight=True)
            _assert_dft_planes_match(
                {"p1": vmap_res.dft_planes["p1"][idx]},
                ref.dft_planes, rtol=1e-6,
                ctx=f"alternate geometry (x_lo-only touch) eps_r={ev}",
            )

    def test_global_sweep_pad_cells_still_correct(self):
        """SCOPE CHECK (#637): the GLOBAL ``"eps_r"`` sweep (no material
        name) does not go through ``_extend_batched_cpml_pad`` at all --
        it was already correct because ``non_vac = eps_r != 1.0`` is
        evaluated on the already-padded ``base_materials.eps_r``, which
        is non-1.0 in the padding too (replicated from the same
        non-vacuum material). Pin that this stays true post-#637, on the
        SAME edge-touching fixture the material-named tests use, so a
        future change to the material-named branch cannot silently break
        the global branch it must not touch."""
        eps_values = np.array([2.0, 6.0])
        n_steps = 60

        vmap_res = vmap_material_sweep(
            _dft_sim("cpml", eps_r=4.0), "eps_r", eps_values, n_steps=n_steps,
        )
        for idx, ev in enumerate(eps_values):
            ref = _dft_sim("cpml", eps_r=float(ev)).run(n_steps=n_steps)
            _assert_dft_planes_match(
                {"p1": vmap_res.dft_planes["p1"][idx]},
                ref.dft_planes, rtol=1e-6, ctx=f"global sweep eps_r={ev}",
            )

    def test_exact_hi_face_touch_matches_run_via_shared_fallback(self):
        """Regression lock for issue #643, CLOSED -- a slab whose bound
        sits EXACTLY on the domain's hi face (all six CPML faces via a
        `Box((0,0,0), domain)`, mirroring the #582/#627 discovery
        fixture), swept by material name.

        HISTORY: this test shipped with #637 as ``xfail(strict=True)``.
        Its gap was real: ``_extend_batched_cpml_pad`` reproduced
        ``_assemble_materials``' PRE-#627 rule (a straight edge-slice
        copy) and so missed the hi-face inner-column fallback #627 added
        to ``rfx.geometry.rasterize_grid.extend_cpml_pad_materials``.
        ``Simulation.run()`` read the material in its x/y/z-hi pads; the
        vmap path read vacuum. Measured worst per-bin DFT-plane relative
        error against ``run()`` on this exact fixture: 1.66e-2
        (eps_r=2.0) / 4.25e-2 (eps_r=6.0). #643 removed the second copy
        of the rule entirely -- ``_extend_batched_cpml_pad`` now vmaps
        the shared helper over the sweep axis -- and this comparison is
        exactly ``0.0`` at every bin for both elements. The marker is
        gone rather than the test: this is the only fixture in the suite
        that sits exactly on the domain edge (``_dft_sim`` and the
        alternate-geometry test both nudge a half-cell past it on
        purpose), so it is the only thing pinning the fallback's
        reachability from the batched path.

        NON-VACUITY PRECONDITION (assert-first, before the equality
        assert): the same arc already produced one test that passed for
        the wrong reason -- the alternate-geometry test's first draft
        used a hi-face-touching box that never reached the pad at all,
        so it agreed pre- and post-fix while measuring nothing. Here the
        precondition is checked directly against ``run()``'s own
        assembled materials: the x-hi pad must read the slab's eps_r,
        NOT vacuum, which is true only because #627's fallback fires.
        If a future rasterizer change made this fixture stop exercising
        the fallback, the precondition fails loudly instead of the
        equality passing vacuously."""
        domain = (0.02, 0.02, 0.02)
        cpml_layers = 6
        dx = 0.002
        eps_values = np.array([2.0, 6.0])
        n_steps = 60

        def sim_fn(eps_r):
            sim = Simulation(freq_max=5e9, domain=domain, boundary="cpml",
                              cpml_layers=cpml_layers, dx=dx)
            sim.add_material("slab", eps_r=eps_r)
            sim.add(Box((0, 0, 0), domain), material="slab")
            sim.add_source((0.01, 0.01, 0.01), "ez",
                            waveform=GaussianPulse(f0=3e9))
            sim.add_probe((0.006, 0.01, 0.01), "ez")
            sim.add_dft_plane_probe(axis="x", coordinate=0.006,
                                     component="ez", n_freqs=4, name="p1")
            return sim

        # --- precondition: this fixture really does exercise #627's
        # hi-face fallback, i.e. run()'s x-hi pad is NOT vacuum and the
        # naive source column (the last interior column) IS.
        probe_sim = sim_fn(4.0)
        probe_grid = probe_sim._build_grid()
        probe_mats, *_ = probe_sim._assemble_materials(probe_grid)
        eps = np.asarray(probe_mats.eps_r)
        phx = probe_grid.pad_x_hi
        assert phx > 0, "fixture lost its x-hi CPML pad"
        # Sample the transverse CENTRE. Not the whole pad slab: the same
        # half-open rasterization also drops this box's y/z hi nodes, so
        # genuine vacuum survives at those transverse coordinates in
        # run()'s OWN arrays -- asserting "all of the pad is 4.0" would
        # assert something false about run(), not about the vmap path.
        cy, cz = probe_grid.shape[1] // 2, probe_grid.shape[2] // 2
        naive_src = eps[-phx - 1, cy, cz]
        inner_src = eps[-phx - 2, cy, cz]
        hi_pad = eps[-phx:, cy, cz]
        assert naive_src == 1.0 and inner_src == 4.0, (
            "fixture no longer reproduces the half-open-hi vacuum column "
            "the #627 fallback exists for (naive source column "
            f"{naive_src}, inner column {inner_src}) -- this test would "
            "pass vacuously; rebuild the fixture, do not delete the assert"
        )
        assert np.all(hi_pad == 4.0), (
            "run()'s x-hi pad is not the slab material -- #627's hi-face "
            f"fallback did not fire, so there is nothing here for the "
            f"batched path to match (pad column: {hi_pad})"
        )

        vmap_res = vmap_material_sweep(
            sim_fn(4.0), "slab.eps_r", eps_values, n_steps=n_steps,
        )
        for idx, ev in enumerate(eps_values):
            ref = sim_fn(float(ev)).run(n_steps=n_steps, skip_preflight=True)
            _assert_dft_planes_match(
                {"p1": vmap_res.dft_planes["p1"][idx]},
                ref.dft_planes, rtol=1e-6,
                ctx=f"exact-hi-face-touch (#643) eps_r={ev}",
            )


def _matrix_sim(shape_lo, shape_hi, *, boundary="cpml", cpml_layers=6,
                domain=(0.02, 0.02, 0.02), dx=0.002, boundary_spec=None,
                eps_r=4.0, sigma=0.0, mu_r=1.0, second_material=False):
    """Builder for the #643 byte-identity matrix. One knob per matrix
    dimension; the sweep itself is applied by the caller through
    ``add_material``'s three material fields."""
    kwargs = {"boundary": boundary_spec if boundary_spec is not None
              else boundary}
    if cpml_layers is not None:
        kwargs["cpml_layers"] = cpml_layers
    sim = Simulation(freq_max=5e9, domain=domain, dx=dx, **kwargs)
    sim.add_material("slab", eps_r=eps_r, sigma=sigma, mu_r=mu_r)
    sim.add(Box(shape_lo, shape_hi), material="slab")
    if second_material:
        # A lossy, magnetic second material -- the ONLY way the shared
        # helper's JOINT vacuum predicate can differ from a per-field one
        # (#637's own note: none of its fixtures carried such a material,
        # so the two predicates were indistinguishable there).
        sim.add_material("lossy", eps_r=1.0, sigma=0.02, mu_r=2.0)
        sim.add(Box((0.0, 0.0, 0.0), (0.006, 0.02, 0.02)), material="lossy")
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=GaussianPulse(f0=3e9))
    sim.add_probe((0.006, 0.01, 0.01), "ez")
    return sim


_D = (0.02, 0.02, 0.02)

# (id, builder-kwargs) -- the geometry/boundary matrix from issue #643's
# acceptance criterion: #637's own matrix PLUS the exact-hi-face case.
_BYTE_IDENTITY_MATRIX = [
    ("single_face_xlo", dict(shape_lo=(0.0, 0.006, 0.006),
                             shape_hi=(0.008, 0.014, 0.014))),
    ("all_six_faces_exact_hi", dict(shape_lo=(0.0, 0.0, 0.0), shape_hi=_D)),
    ("all_six_faces_past_hi", dict(shape_lo=(0.0, 0.0, 0.0),
                                   shape_hi=(0.021, 0.021, 0.021))),
    ("corner_touch", dict(shape_lo=(0.0, 0.0, 0.0),
                          shape_hi=(0.008, 0.008, 0.008))),
    ("inset_no_face_touch", dict(shape_lo=(0.006, 0.006, 0.006),
                                 shape_hi=(0.014, 0.014, 0.014))),
    ("transverse_span_yz", dict(shape_lo=(0.005, 0.0, 0.0),
                                shape_hi=(0.015, 0.02, 0.02))),
    ("cpml_layers_0", dict(shape_lo=(0.0, 0.0, 0.0), shape_hi=_D,
                           cpml_layers=0)),
    ("pec", dict(shape_lo=(0.0, 0.0, 0.0), shape_hi=_D, boundary="pec",
                 cpml_layers=None)),
    ("upml", dict(shape_lo=(0.0, 0.0, 0.0), shape_hi=_D, boundary="upml")),
    # periodic is only expressible through a BoundarySpec, and only
    # symmetrically per axis -- x periodic (pad 0 on both x faces) with
    # y/z still absorbing is the mixed case worth pinning.
    ("periodic_x_cpml_yz", dict(
        shape_lo=(0.0, 0.0, 0.0), shape_hi=_D,
        boundary_spec=BoundarySpec(
            x=Boundary(lo="periodic", hi="periodic"), y="cpml", z="cpml"))),
    ("two_materials", dict(shape_lo=(0.006, 0.0, 0.0), shape_hi=(0.02, 0.02, 0.02),
                           second_material=True)),
    ("asymmetric_per_face_pads", dict(
        shape_lo=(0.0, 0.0, 0.0), shape_hi=_D, cpml_layers=None,
        boundary_spec=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=9),
            y=Boundary(lo="pec", hi="cpml", hi_thickness=6),
            z=Boundary(lo="cpml", hi="pec", lo_thickness=7),
        ))),
]


class TestVmapBatchedPadByteIdentity:
    """Issue #643's acceptance criterion, as a committed matrix.

    What the batched path builds for element *b* must be BYTE-IDENTICAL
    to what ``Simulation._assemble_materials`` builds for a simulation
    carrying that swept value -- all three material arrays, every cell,
    across the geometry/boundary matrix #637 used (single face, all six
    faces, corner, inset, ``cpml_layers=0``, pec, upml, periodic, two
    materials, asymmetric per-face pads) PLUS the exact-hi-face-touching
    case #643 is about.

    This is the mechanism-level lock. It costs no time-stepping, so it
    can afford to be wide where the DFT-plane equivalence tests (which
    do step) have to be narrow. #637 shipped ONE such comparison on ONE
    fixture; the reason #643 existed at all is that a rule can be right
    on the one fixture that is checked and wrong on the class.

    Both directions are exercised on every row: the swept field is
    compared for equality (it must MOVE with the sweep and still match
    run()), and the two unswept fields are compared too (they must NOT
    drift -- the shared helper takes one joint decision across all
    three, so a swept value that empties a column changes the other two
    arrays' pads as well, and that coupling has to match run() rather
    than merely be self-consistent)."""

    @staticmethod
    def _assert_identical(kwargs, param, values, base_kwargs):
        base_sim = _matrix_sim(**{**kwargs, **base_kwargs})
        grid = base_sim._build_grid()
        base_materials, *_ = base_sim._assemble_materials(grid)
        batched = _build_batched_materials(
            base_sim, grid, base_materials, param, jnp.asarray(values))

        field = param.split(".")[1]
        for idx, v in enumerate(values):
            want_sim = _matrix_sim(**{**kwargs, **{field: float(v)}})
            want, *_ = want_sim._assemble_materials(grid)
            for name in ("eps_r", "sigma", "mu_r"):
                npt.assert_array_equal(
                    np.asarray(getattr(batched, name)[idx]),
                    np.asarray(getattr(want, name)),
                    err_msg=(
                        f"batched {name} at {param}={v} disagrees with "
                        f"run()'s own _assemble_materials -- the batched "
                        f"pad rule and the shared rule have drifted "
                        f"again (#643)"
                    ),
                )

    @pytest.mark.parametrize(
        "case_id,kwargs",
        _BYTE_IDENTITY_MATRIX,
        ids=[c[0] for c in _BYTE_IDENTITY_MATRIX],
    )
    def test_eps_r_sweep_byte_identical_to_assemble_materials(
            self, case_id, kwargs):
        self._assert_identical(
            kwargs, "slab.eps_r", np.array([2.0, 4.0, 9.0]),
            base_kwargs=dict(eps_r=4.0))

    @pytest.mark.parametrize(
        "case_id,kwargs",
        _BYTE_IDENTITY_MATRIX,
        ids=[c[0] for c in _BYTE_IDENTITY_MATRIX],
    )
    def test_sigma_sweep_byte_identical_to_assemble_materials(
            self, case_id, kwargs):
        self._assert_identical(
            kwargs, "slab.sigma", np.array([0.0, 0.05]),
            base_kwargs=dict(sigma=0.01))

    @pytest.mark.parametrize(
        "case_id,kwargs",
        _BYTE_IDENTITY_MATRIX,
        ids=[c[0] for c in _BYTE_IDENTITY_MATRIX],
    )
    def test_mu_r_sweep_byte_identical_to_assemble_materials(
            self, case_id, kwargs):
        self._assert_identical(
            kwargs, "slab.mu_r", np.array([1.0, 3.0]),
            base_kwargs=dict(mu_r=2.0))

    @pytest.mark.parametrize(
        "sigma_bulk,conductor_kind",
        [(1.0e4, "non-PEC"), (5.8e7, "PEC")],
        ids=["non_pec_conductor", "pec_conductor"],
    )
    @pytest.mark.parametrize(
        "param,values,base_value",
        [("slab.eps_r", (2.0, 9.0), 4.0),
         ("slab.sigma", (0.0, 0.05), 0.01),
         ("slab.mu_r", (1.0, 3.0), 2.0)],
        ids=["eps_r", "sigma", "mu_r"],
    )
    def test_thin_conductor_pad_matches_run(
            self, sigma_bulk, conductor_kind, param, values, base_value):
        """Issue #642: ``run()`` applies thin conductors AFTER the pad
        extension (``rfx/api/_compile.py``: ``extend_cpml_pad_materials``,
        then the ``_thin_conductors`` loop), so its padding carries the
        background material and never the conductor.

        HISTORY: this shipped with #643 as ``xfail(strict=True)``, because
        #643's scope was the pad-extension RULE while this is the pipeline
        ORDER. #643 made the batched path reuse the shared rule and could
        not close this, since the batched path was handed the wrong INPUT
        (``base_materials``, already post-conductor) rather than running
        the wrong algorithm. Measured on the ``slab.eps_r`` row below:
        88 mismatched cells per swept element of 12167 (40 ``eps_r`` +
        48 ``sigma``, all inside the x pads) on #643's tree, 6592 on the
        pre-#637 tree; **0** once ``_build_batched_materials`` re-derives
        the PRE-conductor arrays and re-applies ``apply_thin_conductor``
        after the extension. The ``sigma``/``mu_r`` sweeps were not in
        the issue's own measurement and leak the same way (96 + 80 and
        96 + 96 cells per pair respectively on #643's tree) -- the leak
        is the conductor being replicated outward, so it does not depend
        on which field is swept.

        The PEC row is the must-pass companion, not filler: it measured
        0 mismatched cells on every tree (a PEC conductor routes to
        ``pec_mask``, not to the material arrays), so it is the case that
        had to keep passing while the non-PEC case went from red to
        green. An equality test can be satisfied by a fixture where
        nothing could differ; ``test_thin_conductor_fixture_is_live``
        below rules that out for the non-PEC row."""
        field = param.split(".")[1]

        def sim_fn(val):
            kw = {"eps_r": 4.0, "sigma": 0.0, "mu_r": 1.0}
            kw[field] = val
            sim = _matrix_sim((0.0, 0.0, 0.0), (0.02, 0.02, 0.02), **kw)
            sim.add_thin_conductor(
                Box((0.0, 0.008, 0.008), (0.02, 0.012, 0.012)),
                sigma_bulk=sigma_bulk, thickness=35e-6)
            return sim

        base_sim = sim_fn(base_value)
        grid = base_sim._build_grid()
        base_materials, *_ = base_sim._assemble_materials(grid)
        vals = np.asarray(values, dtype=np.float32)
        batched = _build_batched_materials(
            base_sim, grid, base_materials, param, jnp.asarray(vals))
        for idx, v in enumerate(vals):
            want, *_ = sim_fn(float(v))._assemble_materials(grid)
            for name in ("eps_r", "sigma", "mu_r"):
                npt.assert_array_equal(
                    np.asarray(getattr(batched, name)[idx]),
                    np.asarray(getattr(want, name)),
                    err_msg=(
                        f"{conductor_kind} thin-conductor {name} pad "
                        f"disagrees with run() at {param}={v} (#642)"),
                )

    def test_thin_conductor_fixture_is_live(self):
        """Control for the non-PEC rows above: the conductor must sit ON
        the column the pad replicates FROM, and the pad must nevertheless
        carry the background material. Without this the equality above
        could hold because the conductor is nowhere near a pad -- exactly
        the vacuous-fixture failure mode #643's own matrix needed a
        control for.

        Asserts the two halves separately so a future failure says which
        one moved: the conductor IS at the x-lo interior edge (so the old
        code had something to replicate), and the x-lo pad is NOT the
        conductor (so the new code does not replicate it)."""
        def sim_fn(eps_r):
            sim = _matrix_sim((0.0, 0.0, 0.0), (0.02, 0.02, 0.02),
                              eps_r=eps_r)
            sim.add_thin_conductor(
                Box((0.0, 0.008, 0.008), (0.02, 0.012, 0.012)),
                sigma_bulk=1.0e4, thickness=35e-6)
            return sim

        base_sim = sim_fn(4.0)
        grid = base_sim._build_grid()
        base_materials, *_ = base_sim._assemble_materials(grid)
        values = np.array([2.0, 9.0], dtype=np.float32)
        batched = _build_batched_materials(
            base_sim, grid, base_materials, "slab.eps_r",
            jnp.asarray(values))

        plx = grid.pad_x_lo
        assert plx > 0
        # (y, z) inside the conductor bar: 0.008..0.012 m at dx=0.002 is
        # interior nodes 4 and 5, i.e. grid index pad + 4.
        jy = grid.pad_y_lo + 4
        kz = grid.pad_z_lo + 4
        # sigma_eff = sigma_bulk * thickness / dx
        sigma_eff = 1.0e4 * 35e-6 / 0.002

        for idx, v in enumerate(values):
            eps = np.asarray(batched.eps_r[idx])
            sig = np.asarray(batched.sigma[idx])
            # Half 1: the conductor really is on the replication source.
            assert sig[plx, jy, kz] == pytest.approx(sigma_eff), (
                f"conductor absent from the x-lo interior edge column "
                f"(sigma={sig[plx, jy, kz]}, expected {sigma_eff}) -- the "
                f"fixture no longer exercises #642 and the equality test "
                f"above is vacuous")
            assert eps[plx, jy, kz] == pytest.approx(1.0)
            # Half 2: and the pad is the background material anyway.
            npt.assert_array_equal(
                sig[:plx, jy, kz], np.zeros(plx, dtype=sig.dtype),
                err_msg="conductor sigma leaked into the x-lo pad (#642)")
            npt.assert_array_equal(
                eps[:plx, jy, kz], np.full(plx, v, dtype=eps.dtype),
                err_msg=("x-lo pad does not carry the swept slab eps_r -- "
                         "either the conductor leaked (#642) or the pad "
                         "extension regressed (#637/#643)"))

    def test_matrix_is_not_vacuous_swept_value_reaches_the_pad(self):
        """Control for the whole matrix above: at least one row must have
        the swept value actually LANDING in the CPML pad, otherwise every
        equality in this class could hold trivially (both sides vacuum).

        This is the same failure the alternate-geometry test's first
        draft had -- a fixture that agreed before and after the fix
        because it never reached the pad. Pinned as its own test so the
        matrix cannot silently become decorative."""
        kwargs = dict(_BYTE_IDENTITY_MATRIX[1][1])   # all_six_faces_exact_hi
        base_sim = _matrix_sim(**kwargs, eps_r=4.0)
        grid = base_sim._build_grid()
        base_materials, *_ = base_sim._assemble_materials(grid)
        batched = _build_batched_materials(
            base_sim, grid, base_materials, "slab.eps_r",
            jnp.asarray([2.0, 9.0]))

        phx = grid.pad_x_hi
        assert phx > 0
        # Sample the transverse CENTRE of the x-hi pad. Not the whole
        # slab: the same half-open rasterization that motivates #627's
        # fallback also leaves genuine vacuum at the y/z hi nodes this
        # box excludes, and run() reproduces that exactly -- asserting
        # "all of the pad is the material" would be asserting something
        # false about run() too, not about the batched path.
        cy, cz = grid.shape[1] // 2, grid.shape[2] // 2
        hi_col_0 = np.asarray(batched.eps_r[0])[-phx:, cy, cz]
        hi_col_1 = np.asarray(batched.eps_r[1])[-phx:, cy, cz]
        assert np.all(hi_col_0 == 2.0), (
            "swept value 2.0 did not reach the x-hi pad "
            f"(got {hi_col_0}) -- the matrix above would be comparing "
            "vacuum against vacuum"
        )
        assert np.all(hi_col_1 == 9.0), (
            f"swept value 9.0 did not reach the x-hi pad (got {hi_col_1})"
        )
        # ...and the two elements must actually DIFFER there, which is
        # the property #637 + #643 together buy. On main this column read
        # 1.0 for BOTH elements (the #643 defect).
        assert not np.array_equal(hi_col_0, hi_col_1)


class TestVmapDftPlaneX64:
    """#477/#484 x64-contract pin, scoped (never module-level, per repo
    rule): under ``jax_enable_x64``, ``init_dft_plane_probe`` (called
    identically by both the vmap fast path and ``run()``) selects
    ``complex128`` instead of ``complex64`` (rfx/probes/probes.py:441).
    This is a separate axis from the default-precision equivalence tests
    above -- confirm the promoted dtype AND that equivalence still holds.

    CORRECTED (#637): this docstring previously claimed the x64 floor was
    "the same ~5e-4 relative floor" as default precision and that it
    "does not tighten" under x64. Both halves were wrong -- that 5e-4 was
    the #637 CPML-padding defect (see the module docstring), not a
    genuine precision floor, so there was nothing for x64 to fail to
    tighten. Post-fix, on this fixture AT n_steps=60: default precision
    is exactly 0.0 at every bin (bit-identical vmap-vs-run); x64 is NOT
    bit-identical, worst observed 3.30e-08 (eps_r=6.0, highest-frequency
    bin) -- i.e. x64 promotes the DFT accumulator to complex128 but the
    underlying field state is still produced by the same float32-pinned
    Yee kernels, so accumulating in higher precision surfaces a genuine
    (tiny) float32-vs-float64 promotion-order residual that default
    precision's exact bit-identity can't show.

    That 3.30e-08 is a property of THIS fixture AT n_steps=60, not a
    fixture-independent precision bound -- it is not safe to read as "the
    x64 floor is ~3e-8" in general. Sweeping n_steps on the same fixture
    (signal-bearing bins only): 1.06e-07 (120 steps), 3.80e-07 (200) --
    growing smoothly as the DFT window covers more of the pulse's
    post-source evolution, more steps of float32-pinned Yee accumulation
    to promote into the complex128 sum. Past a fixture- and bin-dependent
    point (this fixture's highest-frequency bin, past n_steps~=300) the
    reference magnitude itself decays toward the numerical noise floor as
    that bin's spectral content leaves the window; the ratio then blows
    up on a shrinking denominator, not a growing numerator -- 2.26e-03 at
    n_steps=400 on this fixture's weakest bin is that artefact (verified:
    the ABSOLUTE difference barely moves between n_steps=300 and 400
    (1.40e-17 to 9.13e-18 -- if anything it drops slightly), only the
    reference magnitude collapses, ~28x (1.21e-13 to 4.04e-15)), the same
    near-zero-denominator failure mode ``_assert_dft_planes_match``'s own
    docstring warns about for an elementwise check -- not evidence of
    divergence. ``rtol=1e-6`` covers the genuine n_steps=60 floor with
    ~30x margin (see the module docstring's gate-bracketing note for the
    full argument); it is not claimed to generalize to arbitrarily long
    ``n_steps`` on other fixtures, and a future test built on a
    longer-duration fixture should re-derive its own floor rather than
    assume this one."""

    def test_dft_plane_matches_run_cpml_x64(self):
        eps_values = np.array([2.0, 6.0])
        n_steps = 60

        with _enable_x64(True):
            vmap_res = vmap_material_sweep(
                _dft_sim("cpml", eps_r=4.0), "substrate.eps_r", eps_values,
                n_steps=n_steps,
            )
            assert vmap_res.dft_planes["p1"].dtype == np.complex128

            for idx, ev in enumerate(eps_values):
                ref = _dft_sim("cpml", eps_r=float(ev)).run(n_steps=n_steps)
                assert ref.dft_planes["p1"].accumulator.dtype == np.complex128
                _assert_dft_planes_match(
                    {"p1": vmap_res.dft_planes["p1"][idx]},
                    ref.dft_planes, rtol=1e-6, ctx=f"x64 cpml eps_r={ev}",
                )


class TestVmapDftPlaneFallbackCarries:
    """#578: the sequential fallback must ALSO populate ``.dft_planes`` —
    no path asymmetry (design v2 PR B.1)."""

    def test_upml_fallback_carries_dft_planes(self):
        """UPML (not fast-path eligible) + a registered DFT plane: the
        fallback must return the SAME accumulator run() would (exact
        match expected -- the fallback literally calls sim.run() per
        swept value, so this is not a numerical-tolerance comparator, it
        is the same call twice)."""
        eps_values = np.array([2.0, 6.0])
        n_steps = 20

        with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
            res = vmap_material_sweep(
                _dft_sim("upml", eps_r=4.0), "substrate.eps_r", eps_values,
                n_steps=n_steps,
            )
        assert res.dft_planes is not None
        assert res.dft_planes["p1"].shape[0] == len(eps_values)

        for idx, ev in enumerate(eps_values):
            ref = _dft_sim("upml", eps_r=float(ev)).run(n_steps=n_steps)
            npt.assert_allclose(
                res.dft_planes["p1"][idx],
                np.asarray(ref.dft_planes["p1"].accumulator),
                rtol=1e-10, atol=1e-20,
                err_msg=f"fallback dft_planes should exactly reproduce "
                        f"run() at eps_r={ev} (same call)",
            )


class TestVmapPortFamilyEligibility:
    """MSL/floquet/coaxial ports are separate registries from
    ``sim._ports`` (lumped/soft-source) and were never consulted by the
    fast-path eligibility guard before #578 -- a genuine silent-drop gap
    for MSL/floquet (run()-consumed) and an honesty gap for coaxial (not
    consumed by plain run() at all). These pins belong here rather than
    ``test_vmap_sweep_eligibility.py`` because they combine port families
    with a registered DFT plane to also exercise the fallback-carries-DFT
    path for a non-UPML fallback reason."""

    def test_floquet_port_takes_sequential_fallback(self):
        Lx, Ly, Lz = 0.015, 0.015, 0.03
        sim = Simulation(freq_max=15e9, domain=(Lx, Ly, Lz), boundary="cpml",
                          cpml_layers=8, dx=0.001)
        sim.add_material("substrate", eps_r=2.2)
        sim.add(Box((0, 0, Lz / 2 - 0.001), (Lx, Ly, Lz / 2)), material="substrate")
        patch_w = 0.008
        x0 = (Lx - patch_w) / 2
        y0 = (Ly - patch_w) / 2
        sim.add(Box((x0, y0, Lz / 2), (x0 + patch_w, y0 + patch_w, Lz / 2 + 0.001)),
                material="pec")
        sim.add_floquet_port(Lz * 0.25, axis="z", scan_theta=0.0, scan_phi=0.0,
                              polarization="te", n_freqs=10)
        sim.add_probe((Lx / 2, Ly / 2, Lz / 2 + 0.002), component="ex")

        with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
            res = vmap_material_sweep(
                sim, "substrate.eps_r", np.array([2.0, 3.0]), n_steps=20,
            )
        assert res.time_series.shape[0] == 2

    def test_msl_port_takes_sequential_fallback(self):
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.01, 0.006),
                          boundary="cpml", cpml_layers=6, dx=0.001)
        sim.add_material("sub", eps_r=4.0)
        sim.add(Box((0, 0, 0), (0.02, 0.01, 0.002)), material="sub")
        sim.add(Box((0, 0.004, 0.002), (0.02, 0.006, 0.003)), material="pec")
        sim.add_msl_port(position=(0.003, 0.005, 0.0), width=0.002,
                          height=0.002, direction="+x", impedance=50.0)

        with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
            res = vmap_material_sweep(
                sim, "sub.eps_r", np.array([2.0, 3.0]), n_steps=15,
            )
        assert res.time_series.shape[0] == 2

    def test_coaxial_port_takes_sequential_fallback_and_fails_loud(self):
        """Coaxial ports are not consumed by plain ``sim.run()`` at all
        (pre-existing NotImplementedError there); before #578 the vmap
        fast-path eligibility guard did not check ``_coaxial_ports``, so
        a coax-port sim silently took the fast path and just ignored the
        port (no error, wrong/incomplete physics). Post-#578 it routes to
        the sequential fallback, which surfaces run()'s existing honest
        NotImplementedError instead of a silent no-op -- this is the
        'honesty claim distinguishes the families' guard from the design
        doc, verified by an actual raise, not just a routing check."""
        sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02),
                          boundary="pec", dx=0.002)
        sim.add_material("sub", eps_r=4.0)
        sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="sub")
        sim.add_coaxial_port((0.010, 0.010, 0.015), face="top")
        sim.add_source((0.01, 0.01, 0.005), "ez", waveform=GaussianPulse(f0=3e9))
        sim.add_probe((0.005, 0.01, 0.01), "ez")

        with pytest.warns(UserWarning, match=_FALLBACK_MATCH):
            with pytest.raises(NotImplementedError, match="add_coaxial_port"):
                vmap_material_sweep(
                    sim, "sub.eps_r", np.array([2.0, 3.0]), n_steps=15,
                )


class TestVmapAmplitudeKindCurrent:
    """PR #617 (issue #571) added ``amplitude_kind='current'`` soft
    sources with a dynamic per-batch Cb-normalization path in
    ``_build_vmap_scan_fn`` (rfx/vmap_sweep.py:450-468) -- until this
    file, no test exercised it at the ``vmap_material_sweep`` level (the
    #617 coverage gap named in the #578 design doc). Tolerance is looser
    than the DFT-plane gate above: ``rtol=1e-3`` on the raw time series.

    PARTIALLY CORRECTED (#637): this docstring previously attributed the
    whole observed floor to "ordinary float32 FDTD chaos, not a
    phase/sign defect". That was only half right. Measured directly
    (max|diff| / max|ref| on the raw time series, same fixture, cpml
    eps_r=2.0, n_steps=20): pre-#637-fix 6.88e-05, post-fix 1.74e-07 --
    a ~396x drop (re-measured post-#627-rebase; the exact figures moved
    from an earlier 4.89e-05/2.60e-07 pair for the same reason the
    module docstring's cell count moved, but the conclusion is
    unchanged), so part of this test's own floor WAS the #637
    CPML-padding defect leaking into the probe reading through the
    mismatched absorber, not pure roundoff. The remaining post-fix floor
    (1.74e-07) IS genuine float32 dynamic-Cb arithmetic noise: it now
    matches the ``boundary="pec"`` case (1.87e-07, no CPML padding to be
    wrong in the first place, and unchanged by the rebase since it has no
    CPML pad to interact with #627 either) to within the same order of
    magnitude, and #637's fix does not touch this code path's Cb
    computation. The ``rtol=1e-3`` gate below sits ~5700x above that
    genuine floor, so it was never a tight fit to any defect and is left
    unchanged."""

    def test_amplitude_kind_current_matches_run_cpml(self):
        eps_values = np.array([2.0, 6.0])
        n_steps = 20

        vmap_res = vmap_material_sweep(
            _dft_sim("cpml", eps_r=4.0, amplitude_kind="current"),
            "substrate.eps_r", eps_values, n_steps=n_steps,
        )
        for idx, ev in enumerate(eps_values):
            ref = _dft_sim(
                "cpml", eps_r=float(ev), amplitude_kind="current",
            ).run(n_steps=n_steps)
            npt.assert_allclose(
                vmap_res.time_series[idx],
                np.asarray(ref.time_series, dtype=np.float64),
                rtol=1e-3, atol=1e-6,
                err_msg=f"amplitude_kind='current' vmap-vs-run at eps_r={ev}",
            )

    def test_amplitude_kind_current_matches_run_pec(self):
        eps_values = np.array([2.0, 6.0])
        n_steps = 20

        vmap_res = vmap_material_sweep(
            _dft_sim("pec", eps_r=4.0, amplitude_kind="current"),
            "substrate.eps_r", eps_values, n_steps=n_steps,
        )
        for idx, ev in enumerate(eps_values):
            ref = _dft_sim(
                "pec", eps_r=float(ev), amplitude_kind="current",
            ).run(n_steps=n_steps)
            npt.assert_allclose(
                vmap_res.time_series[idx],
                np.asarray(ref.time_series, dtype=np.float64),
                rtol=1e-3, atol=1e-6,
                err_msg=f"amplitude_kind='current' vmap-vs-run (pec) at eps_r={ev}",
            )


class TestVmapReturnFieldsRaise:
    """#578: return_fields=True was documented-but-never-implemented
    (VmapSweepResult.final_fields was never populated) -- it now raises
    ValueError instead of silently returning final_fields=None."""

    def test_return_fields_true_raises(self):
        sim = _dft_sim("cpml")
        with pytest.raises(ValueError, match="return_fields=True"):
            vmap_material_sweep(
                sim, "substrate.eps_r", np.array([2.0, 4.0]), n_steps=10,
                return_fields=True,
            )

    def test_return_fields_false_is_unaffected(self):
        """Control: the default (False) must not raise and must behave
        exactly as before."""
        sim = _dft_sim("cpml")
        res = vmap_material_sweep(
            sim, "substrate.eps_r", np.array([2.0, 4.0]), n_steps=10,
            return_fields=False,
        )
        assert isinstance(res, VmapSweepResult)
        assert res.final_fields is None
