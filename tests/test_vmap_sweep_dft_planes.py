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
from rfx.vmap_sweep import vmap_material_sweep, VmapSweepResult

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
    # vacuum fallback, closes-#627 fce1091): run() now correctly
    # recovers that node via the fallback, this file's vmap helper
    # doesn't reproduce that fallback (out of scope here, see the #637
    # CHANGELOG entry's "Overlap with issue #627" paragraph), so without
    # this margin every test built on this fixture would fail for a
    # SEPARATE, already-disclosed reason unrelated to what they test.
    # The margin is far short of entering the CPML pad itself (half a
    # cell vs. the pad's full cpml_layers=6 cells), so it only recovers
    # the one excluded interior node -- confirmed directly (0 mask cells
    # inside the pad region) before landing this change.
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
