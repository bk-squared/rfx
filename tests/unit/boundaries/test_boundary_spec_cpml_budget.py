"""Issue #647 — ``cpml_layers`` as an allocation BUDGET vs the grid it lands on.

``cpml_layers`` is a per-simulation allocation budget shared by all six
faces; each face's ACTIVE thickness is its own ``lo_thickness`` /
``hi_thickness`` (or the budget), no-op-padded up to the budget. The CPML
scratch buffers were cut from that budget on EVERY axis, including axes
that allocate no absorber at all, so a per-face ``BoundarySpec`` whose
budget exceeded an axis's cell count died inside the scan with
``mul got incompatible shapes for broadcasting`` — at every precision,
through both ``run()`` and ``forward()``, with preflight reporting
"All checks passed".

Two crashing families, both fixed here and both covered below:

1. an axis whose faces are ALL PEC/PMC, so it gets no padding at all and
   its extent is just the interior (the reported fixture), and
2. an ABSORBING axis whose per-face thickness is well below the budget,
   so its padding is thinner than the buffer (not in the report).

Every threshold in this file is DERIVED from the built grid
(``min(grid.shape)``), never written as a literal: the reported "fails at
16, works at 8" is a property of one domain and one frequency, not of the
code.

Coverage note: per-face ``BoundarySpec`` had almost no executing CPML
coverage before this file. The two pre-existing tests that ran FDTD with
asymmetric per-face thickness asserted only shape + finiteness, and
``test_precision_lane_guard.py`` documents having had to pin
``cpml_layers=4`` to reach the CPML compute at all. So this file also adds
the missing physics: an absorbing per-face configuration whose interior
energy is measured against an all-PEC control.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import jax

from rfx import Grid, Simulation
from rfx.boundaries.cpml import (
    _assert_absorber_fits, _axis_buffer_depths, apply_cpml_e, apply_cpml_h,
    init_cpml,
)
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.core.yee import init_state

# The reported fixture: 20 mm cube, 5 GHz, five PEC faces + one CPML face.
_FREQ = 5.0e9
_CUBE = (0.02, 0.02, 0.02)
_CENTRE = (0.01, 0.01, 0.008)
_PROBE = (0.007, 0.007, 0.006)


def _cube_sim(cpml_layers, *, hi_thickness=None, hi_token="cpml",
              precision="float32"):
    z = Boundary(lo="pec", hi=hi_token,
                 hi_thickness=hi_thickness if hi_token == "cpml" else None)
    sim = Simulation(
        freq_max=_FREQ, domain=_CUBE, cpml_layers=cpml_layers,
        boundary=BoundarySpec(x="pec", y="pec", z=z),
        precision=precision,
    )
    sim.add_source(_CENTRE, "ez", amplitude_kind="field")
    sim.add_probe(_PROBE, "ez")
    return sim


def _field_digest(result) -> str:
    h = hashlib.sha256()
    for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
        h.update(np.asarray(getattr(result.state, comp)).tobytes())
    return h.hexdigest()


def _cpml_program(sim):
    """The CPML update this sim COMPILES: psi shapes + optimized HLO text.

    ``lower(...).compile().as_text()`` is the post-optimization HLO, i.e.
    the program XLA will actually run — the level at which #876 lived (two
    budgets over the same absorber fused, and therefore contracted, their
    ``ey + cb*(d1*inv - d2*inv)`` differently). The PRE-optimization text
    would not do: the profile arrays are still budget-length constants
    there, and only constant-folding the ``_clip_lo`` / ``_clip_hi`` slices
    reveals that the two budgets describe one program.
    """
    grid = sim._build_grid()
    params, cpml_state = init_cpml(grid)

    def step(state, cs):
        state, cs = apply_cpml_e(state, params, cs, grid)
        return apply_cpml_h(state, params, cs, grid)

    text = jax.jit(step).lower(init_state(grid.shape), cpml_state) \
        .compile().as_text()
    shapes = tuple(np.shape(v) for v in jax.tree_util.tree_leaves(cpml_state))
    return shapes, text


def _interior_energy(result, grid) -> float:
    return sum(
        float(np.sum(np.asarray(getattr(result.state, comp),
                                dtype=np.float64)[grid.interior] ** 2))
        for comp in ("ex", "ey", "ez")
    )


# --------------------------------------------------------------------------
# 1. The crash, and the threshold it sits at.
# --------------------------------------------------------------------------

def test_budget_over_a_pec_axis_extent_runs_instead_of_broadcasting():
    """Family 1: the reported fixture. Budget > the PEC axes' extent."""
    sim = _cube_sim(16)
    grid = sim._build_grid()
    # Confirm the fixture really is in the failing regime, derived from the
    # grid rather than assumed from the report's numbers.
    assert sim._cpml_layers > min(grid.shape)

    result = sim.run(n_steps=60, skip_preflight=True)
    trace = np.asarray(result.time_series)[:, 0]
    assert trace.shape[0] == 60
    assert np.all(np.isfinite(trace))
    assert np.max(np.abs(trace)) > 0.0


def test_budget_over_an_absorbing_axis_extent_runs():
    """Family 2: the ABSORBING axis is the narrow one.

    ``hi_thickness`` below the budget makes z's padding — and so its whole
    extent — smaller than the buffer, on the axis that actually absorbs.
    Not in the issue report; found by asking which axes the crash condition
    can reach.
    """
    sim = _cube_sim(16, hi_thickness=2)
    grid = sim._build_grid()
    assert sim._cpml_layers > grid.nz
    assert grid.pad_z_hi == 2

    result = sim.run(n_steps=60, skip_preflight=True)
    assert np.all(np.isfinite(np.asarray(result.time_series)))


@pytest.mark.parametrize("precision", ["float32", "mixed"])
def test_crash_family_is_precision_independent(precision):
    """The report measured this at every precision; pin both lanes."""
    sim = _cube_sim(16, precision=precision)
    result = sim.forward(n_steps=5, skip_preflight=True)
    assert np.all(np.isfinite(np.asarray(result.time_series)))


def test_budget_at_the_axis_extent_still_works():
    """Sibling of the two above: the same fixture just under the threshold.

    Without this the "it raises / it crashes" cases prove nothing about the
    fixture — a fixture that can only fail is not evidence.
    """
    grid = _cube_sim(4)._build_grid()
    fitting = min(grid.shape)
    sim = _cube_sim(fitting)
    assert sim._cpml_layers <= min(sim._build_grid().shape)
    trace = np.asarray(sim.run(n_steps=60, skip_preflight=True).time_series)
    assert np.all(np.isfinite(trace))
    assert np.max(np.abs(trace)) > 0.0


# --------------------------------------------------------------------------
# 2. The clamp is exact: saturation in the budget.
# --------------------------------------------------------------------------

def test_result_saturates_once_the_budget_covers_the_narrowest_axis():
    """Fixing the absorber and raising ONLY the budget must stop mattering.

    ``hi_thickness`` is pinned, so every budget here builds the SAME grid
    with the SAME active absorber; only the scratch-buffer allocation
    differs. The clamp holds that allocation at the ACTIVE absorber depth,
    so every one of these budgets must compile to the SAME PROGRAM — the
    psi buffer shapes and the optimized HLO — and therefore reproduce the
    fitting-budget fields BIT for BIT. That fitting budget is a
    configuration the pre-fix code also ran, which is what makes this an
    invariance witness rather than a self-consistency check: the clamped
    answer is the unclamped answer.

    Program identity is asserted directly and not left to the digest
    because the digest alone cannot say WHY two runs agree. Issue #876:
    before the clamp reached the active depth, budgets 8 and 9 over the
    same 4-layer absorber emitted two different fusions, and on arm64 the
    XLA fusion emitter contracted one of them into an FMA — 1 ulp on three
    ``ey`` cells at step 6, a different sha at step 120. Nothing physical
    distinguished the two runs; the allocation budget had leaked into the
    emitted arithmetic. Asserting the program closes that leak at the
    place it happens, on every platform, instead of noticing it only where
    the fusion happens to differ.
    """
    grid = _cube_sim(4, hi_thickness=4)._build_grid()
    saturate_at = min(grid.shape)

    reference = digest = shapes = text = None
    for budget in (saturate_at, saturate_at + 1, saturate_at + 5,
                   4 * saturate_at, 8 * saturate_at):
        sim = _cube_sim(budget, hi_thickness=4)
        assert sim._build_grid().shape == grid.shape, (
            "pinning hi_thickness must keep the grid fixed"
        )
        shapes, text = _cpml_program(sim)
        digest = _field_digest(sim.run(n_steps=120, skip_preflight=True))
        if reference is None:
            reference = (shapes, text, digest)
        assert shapes == reference[0], (
            f"budget={budget} changed the CPML buffer shapes although the "
            f"active absorber is identical to budget={saturate_at}: "
            f"{shapes} vs {reference[0]}"
        )
        assert text == reference[1], (
            f"budget={budget} compiled a different CPML program than "
            f"budget={saturate_at} although the active absorber is "
            f"identical — the allocation budget is leaking into the "
            f"emitted arithmetic (#876)"
        )
        assert digest == reference[2], (
            f"budget={budget} changed the fields although the grid and the "
            f"active absorber are identical to budget={saturate_at}"
        )


def test_the_clamp_saturates_at_the_active_absorber_not_the_axis_extent():
    """Where saturation SITS: the active absorber, not the grid (#876).

    The buffer depth is what the emitted program is cut from, so the depth
    itself is the invariant worth pinning — the test above would still pass
    if the clamp saturated one layer late on every axis, and the fields
    would then still depend on the budget between the two thresholds.
    """
    sim = _cube_sim(16, hi_thickness=4)
    grid = sim._build_grid()
    assert grid.pad_z_hi == 4 and grid.pad_z_lo == 0
    assert _axis_buffer_depths(grid, 16) == (1, 1, 4), (
        "z is clamped to its own 4 active layers; x and y are PEC-closed "
        "and keep a single no-op layer (depth 0 would make the hi-face "
        "slice arr[-0:] the whole axis)"
    )
    # ...and a UNIFORM absorber is untouched: every face allocates the full
    # budget, so the depths stay the budget and every currently-compiled
    # program — every committed digest — is the same program as before.
    uni = Simulation(freq_max=_FREQ, domain=_CUBE, cpml_layers=4,
                     boundary="cpml")
    uni_grid = uni._build_grid()
    assert _axis_buffer_depths(uni_grid, 4) == (4, 4, 4)


def test_a_grid_that_does_not_state_its_face_pads_keeps_the_full_buffer():
    """The active-depth clamp is licensed by the grid, not assumed (#876).

    ``apply_cpml_e/h`` are also driven with duck-typed grids that carry only
    a whole-axis ``pad_<axis>`` and a single legacy ``CPMLParams`` — there
    EVERY face gets a real absorbing profile of length ``cpml_layers``, and
    an absent ``pad_x_lo`` means "this grid does not say", not "no
    absorber". Reading it as the latter thinned a real 10-layer waveguide
    absorber to one layer: ``test_overlap_extraction.py`` measured the
    overlap-vs-V/I S21 agreement going from < 1e-4 to 1.3e-2 (S11 4.0e-2).
    Such an axis must keep the pre-#876 ``min(cpml_layers, extent)`` depth.

    The shape below is ``_WgGrid`` from that file, reduced to what
    ``_axis_buffer_depths`` reads.
    """
    class _WholeAxisPadGrid:
        shape = (40, 21, 11)
        cpml_layers = 10
        pad_x, pad_y, pad_z = 10, 0, 0

    assert _axis_buffer_depths(_WholeAxisPadGrid(), 10) == (10, 10, 10), (
        "an axis whose per-face pads are not stated must not be clamped: "
        "nothing there is KNOWN to be no-op padding"
    )
    # a grid that states one face still gets the clamp on that axis
    class _OneFaceStated(_WholeAxisPadGrid):
        pad_x_hi = 4

    assert _axis_buffer_depths(_OneFaceStated(), 10) == (4, 10, 10)


def test_an_axis_outside_cpml_axes_keeps_its_pre_876_applied_depth():
    """The licensing guard is two-sided: pads of 0 is not "no absorber".

    ``Grid`` gives every axis outside ``cpml_axes`` per-face pads of 0
    (``Grid._face_pad``), so those pads ARE stated and the first half of the
    guard lets them through. But ``init_cpml`` never reads ``cpml_axes``:
    unless the face is PEC/PMC it builds a REAL absorbing profile of length
    ``cpml_layers`` for it, and ``rfx.simulation.run`` applies it whenever
    its own ``cpml_axes`` argument — a separate knob defaulting to ``"xyz"``
    — names the axis. Clamping there read "no padding cells" as "no
    absorber" and cut a real 8-layer absorber to 1.

    Measured on this fixture through ``rfx.run`` for 200 steps: field digest
    ``d5ffa08245a4cb82`` and residual E energy 1.6400655163e+01 with the
    absorber intact, against ``4da793698805a454`` / 8.3652918424e+00 (-49%)
    while the clamp applied to x and y. The first pair is also what
    ``origin/main`` produces, i.e. this axis's answer is unchanged by #876.

    The three assertions below are the applied depth at the three places it
    is observable: the clamp's own return, the psi buffers those slices are
    cut from, and the profile that proves the absorber was real.
    """
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.02),
                cpml_axes="z", cpml_layers=8)
    assert grid.cpml_axes == "z"
    assert (grid.pad_x_lo, grid.pad_x_hi) == (0, 0)
    assert (grid.pad_y_lo, grid.pad_y_hi) == (0, 0)

    # 1. the pre-#876 depth: min(cpml_layers, extent), NOT max(pad_lo, pad_hi)
    assert _axis_buffer_depths(grid, 8) == (8, 8, 8)

    params, state = init_cpml(grid)

    # 2. the buffers the apply_cpml_e/h face slices are cut from
    assert state.psi_ey_xlo.shape[0] == 8
    assert state.psi_ez_xhi.shape[0] == 8
    assert state.psi_ex_ylo.shape[0] == 8
    assert state.psi_ez_yhi.shape[0] == 8

    # 3. and the profile on those faces really absorbs — this is what makes
    #    the depth matter at all. A no-op layer is exactly (b=1, c=0); the
    #    innermost layer of a CPML profile has sigma=0 by construction, so a
    #    full-budget profile carries 7 active layers out of the 8 allocated
    #    — seven times what the one-sided clamp left standing.
    for face in ("x_lo", "x_hi", "y_lo", "y_hi"):
        prof = getattr(params, face)
        b = np.asarray(prof.b).ravel()
        c = np.asarray(prof.c).ravel()
        assert b.size == 8 and c.size == 8
        assert np.count_nonzero(b != 1.0) == 8, f"{face} b is no-op padding"
        # the innermost layer has sigma=0 by construction, so its c is -0.0;
        # the other 7 drive a real recursive convolution.
        assert np.count_nonzero(c != 0.0) == 7, f"{face} c is no-op padding"

    # The guard is not a blanket disable: an axis that IS in cpml_axes but
    # allocates nothing because both its faces are PEC still clamps to the
    # single no-op layer — and there "pad 0" really does mean "no absorber",
    # because ``init_cpml`` gives a PEC face ``_cpml_noop_profile``. That is
    # the #876 behaviour this file exists for.
    pec_closed = _cube_sim(16, hi_thickness=4)._build_grid()
    assert pec_closed.cpml_axes == "xyz"
    assert {"x_lo", "x_hi", "y_lo", "y_hi"} <= set(pec_closed.pec_faces)
    pec_params, _ = init_cpml(pec_closed)
    assert np.all(np.asarray(pec_params.x_lo.b) == 1.0)
    assert np.all(np.asarray(pec_params.x_lo.c) == 0.0)
    assert _axis_buffer_depths(pec_closed, 16) == (1, 1, 4)


def test_thinning_the_active_absorber_is_a_different_answer():
    """Negative control for the saturation test.

    The digest must respond to the ABSORBER. Without this the saturation
    assertion would also pass on a harness that simply cannot tell two runs
    apart.

    This is the sibling that used to compare two BUDGETS below saturation
    (4 vs 8 with ``hi_thickness=4`` pinned) and assert they differed. They
    did differ — measured here at 418/768 ``ex`` cells and max |Δ| 5.6e-7
    after 120 steps — but not for a reason that survives inspection: both
    runs build the same grid and the same 4-layer absorber, and every layer
    the deeper buffer added was no-op padding (b=1, c=0, kappa=1). The
    entire signal was the allocation budget leaking into the emitted
    arithmetic, i.e. the #876 defect itself, so once the clamp removes it
    those two runs are correctly identical. The control is therefore
    re-pointed at a degree of freedom that is real: same budget, thinner
    absorber.
    """
    budget = 8
    thick = _field_digest(
        _cube_sim(budget, hi_thickness=4).run(n_steps=120, skip_preflight=True))
    thin = _field_digest(
        _cube_sim(budget, hi_thickness=2).run(n_steps=120, skip_preflight=True))
    assert thin != thick
    # and the change is in the absorber, not in the budget: the buffer
    # depth follows hi_thickness at a FIXED cpml_layers
    assert _axis_buffer_depths(
        _cube_sim(budget, hi_thickness=4)._build_grid(), budget) == (1, 1, 4)
    assert _axis_buffer_depths(
        _cube_sim(budget, hi_thickness=2)._build_grid(), budget) == (1, 1, 2)


# --------------------------------------------------------------------------
# 3. Physics: a clamped per-face absorber still absorbs.
# --------------------------------------------------------------------------

_PIPE_FREQ = 20e9
_PIPE_DX = 2.5e-3
_PIPE_DOMAIN = (0.02, 0.02, 0.08)


def _pipe_sim(hi_token, cpml_layers):
    """Square PEC pipe along z, terminated by ``hi_token`` on z_hi.

    TE10 cutoff of the 20 mm cross-section is c/(2a) = 7.5 GHz, so the
    10 GHz source content propagates down the pipe and reaches the
    termination. The cross-section is only 9 cells, so a budget above that
    exercises the clamp on x and y while the z absorber does real work.
    """
    sim = Simulation(
        freq_max=_PIPE_FREQ, domain=_PIPE_DOMAIN, dx=_PIPE_DX,
        cpml_layers=cpml_layers,
        boundary=BoundarySpec(x="pec", y="pec",
                              z=Boundary(lo="pec", hi=hi_token)),
    )
    sim.add_source((0.01, 0.01, 0.01), "ey", amplitude_kind="field")
    sim.add_probe((0.01, 0.01, 0.04), "ey")
    return sim


@pytest.mark.slow
def test_clamped_per_face_absorber_actually_absorbs():
    """The absorber survives the clamp — measured against an all-PEC control.

    Both runs are the same pipe with the same excitation; only the z_hi
    face differs. The PEC control is lossless, so its interior energy must
    NOT decay; the CPML termination must take the guided energy out. The
    bound below is two orders of magnitude inside the measured separation.
    """
    open_sim = _pipe_sim("cpml", 16)
    open_grid = open_sim._build_grid()
    assert open_sim._cpml_layers > min(open_grid.nx, open_grid.ny), (
        "fixture must exercise the clamp on the transverse axes"
    )

    closed_sim = _pipe_sim("pec", 16)
    closed_grid = closed_sim._build_grid()

    e_open = _interior_energy(
        open_sim.run(n_steps=1200, skip_preflight=True), open_grid)
    e_closed = _interior_energy(
        closed_sim.run(n_steps=1200, skip_preflight=True), closed_grid)

    assert np.isfinite(e_open) and np.isfinite(e_closed)
    # Measured on this fixture: e_open/e_closed ~ 1e-4 at 1200 steps
    # (7.6e-2 at 200 steps, falling monotonically). Anything above 1e-2
    # means the termination stopped absorbing.
    assert e_open / e_closed < 1e-2, (
        f"open-ended pipe retained {e_open:.4e} vs closed {e_closed:.4e}"
    )
    # The lossless control must not have quietly decayed instead.
    e_closed_early = _interior_energy(
        _pipe_sim("pec", 16).run(n_steps=300, skip_preflight=True),
        closed_grid)
    assert e_closed > 0.1 * e_closed_early


# --------------------------------------------------------------------------
# 4. Preflight: the advisory, and the per-face thickness it rests on.
# --------------------------------------------------------------------------

def _advisories(sim) -> list[str]:
    return [m for m in sim.preflight()]


def test_budget_advisory_fires_exactly_above_the_axis_extent():
    """Advisory boundary derived from the grid, not from the report.

    Issue #737/#742: re-pinned onto a TRUE positive. This test's original
    form swept ``_cube_sim(budget)`` (x=pec, y=pec, z=Boundary(hi='cpml'),
    no explicit hi_thickness) and asserted ``expected = budget >
    min(grid.shape)``. Measured: every hit in that sweep landed on the
    x/y axes (pad_lo=pad_hi=0, both PEC) — the exact false positive #742
    reports, since a PEC-closed axis has no absorber for cpml_layers to
    exceed. The z axis, whose hi face absorbs the FULL budget, never fired
    (its own padding grows with the budget, so it can never exceed itself).
    So this test's entire prior signal was the false positive, and the
    allocation>0 guard added for #742 makes it fail permanently as
    written (0 hits at every budget) — not a loosened gate, but a test
    whose fixture never exercised a real absorber.
    An explicit small ``hi_thickness=2`` gives z a genuine, budget
    -independent allocation, so once ``cpml_layers`` exceeds z's own cell
    count the advisory is a true positive; x/y stay PEC-closed and must
    never fire, both before and after the fix.
    """
    for budget in (2, 4, 8, 9, 16, 40):
        sim = _cube_sim(budget, hi_thickness=2)
        grid = sim._build_grid()
        expected = budget > grid.shape[2]
        hits = [m for m in _advisories(sim) if "exceeds the" in m]
        assert bool(hits) == expected, (
            f"cpml_layers={budget} on grid {grid.shape}: expected "
            f"advisory={expected}, got {hits}"
        )
        if expected:
            assert all("z-axis" in m for m in hits), hits
        assert not any("x-axis" in m or "y-axis" in m for m in hits), (
            "x/y are PEC-closed on both faces (#742): they allocate no "
            f"absorber, so cpml_layers has nothing to exceed there: {hits}"
        )


def test_scalar_cpml_boundary_never_trips_the_budget_advisory():
    """Sibling: with every face absorbing, the axis is padded by 2*budget.

    So the budget cannot exceed an axis extent however large it gets, and
    the advisory must stay silent — it is a per-face-only condition.
    """
    sim = Simulation(freq_max=_FREQ, domain=_CUBE, cpml_layers=64,
                     boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), "ez", amplitude_kind="field")
    assert [m for m in _advisories(sim) if "exceeds the" in m] == []


def test_absorber_budget_advisory_silent_on_a_pec_closed_axis():
    """Issue #742: the false positive, at the pinned repro shape.

    One axis (x) genuinely absorbs, so ``cpml_layers`` is a real budget;
    y and z are PEC on both faces (pad_lo=pad_hi=0 — no absorber to
    budget there at all). Measured on the UNFIXED code: this exact shape
    (one absorbing axis + two PEC-closed axes, budget >
    min(y,z grid extent)) fires on both y and z — the class reported 6x
    on cv11's three legs and 2x on cv18.
    """
    sim = Simulation(
        freq_max=_FREQ, domain=_CUBE, cpml_layers=40,
        boundary=BoundarySpec(x=Boundary(lo="pec", hi="cpml"),
                              y="pec", z="pec"),
    )
    sim.add_source((0.01, 0.01, 0.01), "ez", amplitude_kind="field")
    grid = sim._build_grid()
    # Confirm we are in the would-be-false-positive regime for y/z.
    assert sim._cpml_layers > grid.shape[1]
    assert sim._cpml_layers > grid.shape[2]
    hits = [m for m in _advisories(sim) if "exceeds the" in m]
    assert hits == [], hits


def test_absorber_budget_advisory_silent_on_a_periodic_closed_axis():
    """Sibling: a periodic-closed axis (the cv18/cv19 idiom) is the other
    #742 false-positive shape — also pad_lo=pad_hi=0 on both faces, also
    must stay silent regardless of how far the budget exceeds it.
    """
    sim = Simulation(
        freq_max=_FREQ, domain=_CUBE, cpml_layers=40,
        boundary=BoundarySpec(x=Boundary(lo="pec", hi="cpml"),
                              y=Boundary(lo="periodic", hi="periodic"),
                              z="pec"),
    )
    sim.add_source((0.01, 0.01, 0.01), "ez", amplitude_kind="field")
    grid = sim._build_grid()
    assert sim._cpml_layers > grid.shape[1]
    hits = [m for m in _advisories(sim) if "exceeds the" in m]
    assert hits == [], hits


def test_preflight_thickness_follows_the_allocated_per_face_layers():
    """Preflight must report the ALLOCATED absorber, not the budget.

    ``cpml_thick_*`` is consumed as a calibrated clearance buffer (MSL and
    waveguide port geometry advisories), so reporting the budget on a face
    that allocates less biases those distances by the whole ratio.
    """
    budget, thickness = 16, 2
    sim = Simulation(
        freq_max=_FREQ, domain=(0.06, 0.06, 0.06), cpml_layers=budget,
        boundary=BoundarySpec(
            x="pec", y="pec",
            z=Boundary(lo="pec", hi="cpml", hi_thickness=thickness)),
    )
    grid = sim._build_grid()
    _, thick_hi, _ = sim._validate_cfg_compute_cpml_thickness(budget * grid.dx)
    assert thick_hi[2] == pytest.approx(grid.pad_z_hi * grid.dx, rel=1e-9)
    assert thick_hi[2] == pytest.approx(thickness * grid.dx, rel=1e-9)


def test_preflight_thickness_is_the_budget_without_an_override():
    """Sibling of the test above: no override, no change from before."""
    budget = 16
    sim = Simulation(
        freq_max=_FREQ, domain=(0.06, 0.06, 0.06), cpml_layers=budget,
        boundary=BoundarySpec(x="pec", y="pec",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    grid = sim._build_grid()
    _, thick_hi, _ = sim._validate_cfg_compute_cpml_thickness(budget * grid.dx)
    assert thick_hi[2] == pytest.approx(budget * grid.dx, rel=1e-9)
    assert thick_hi[2] == pytest.approx(grid.pad_z_hi * grid.dx, rel=1e-9)


def test_preflight_thickness_is_zero_on_every_reflector_face():
    """PEC / PMC / periodic faces allocate nothing, on either side."""
    sim = Simulation(
        freq_max=_FREQ, domain=(0.06, 0.06, 0.06), cpml_layers=8,
        boundary=BoundarySpec(x="pmc", y="pec",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    grid = sim._build_grid()
    thick_lo, thick_hi, _ = sim._validate_cfg_compute_cpml_thickness(
        8 * grid.dx)
    assert thick_lo == [0.0, 0.0, 0.0]
    assert thick_hi[0] == 0.0 and thick_hi[1] == 0.0
    assert thick_hi[2] > 0.0


_NU_DZ = np.full(40, 2.0e-4)
_NU_DOMAIN = (0.06, 0.06, float(np.sum(_NU_DZ)))


def _nu_thin_z_hits(boundary) -> list[str]:
    sim = Simulation(
        freq_max=_FREQ, domain=_NU_DOMAIN, dx=3.0e-3, cpml_layers=8,
        boundary=boundary, dz_profile=_NU_DZ,
    )
    sim.add_source((0.03, 0.03, _NU_DOMAIN[2] / 2), "ez",
                   amplitude_kind="field")
    return [m for m in sim.preflight() if "z-thickness" in m]


def test_thin_nu_z_absorber_advisory_is_silent_when_z_has_no_absorber():
    """The P2.6 advisory measured the BUDGET against the z cell sizes.

    With per-face PEC z faces there is no z absorber to be thin, but the
    advisory still fired — the same scalar-attribute blindness, one check
    over.
    """
    assert _nu_thin_z_hits(BoundarySpec(x="cpml", y="cpml", z="pec")) == []


@pytest.mark.parametrize(
    "boundary", ["cpml", BoundarySpec(x="cpml", y="cpml", z="cpml")])
def test_thin_nu_z_absorber_advisory_still_fires_when_z_absorbs(boundary):
    """Sibling: the genuine thin-z case must be untouched.

    Same profile, same budget, same domain — only the z faces differ, so a
    silent result here would mean the fix above disabled the check rather
    than scoping it.
    """
    assert _nu_thin_z_hits(boundary)


# --------------------------------------------------------------------------
# 5. The guard that keeps the clamp honest.
# --------------------------------------------------------------------------

class _StubGrid:
    """Minimal duck-typed grid carrying only the per-face pads."""

    def __init__(self, **pads):
        for face in ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"):
            setattr(self, f"pad_{face}", pads.get(face, 0))


def test_absorber_wider_than_its_own_axis_is_rejected():
    """The clamp may only ever drop no-op padding.

    A grid whose ALLOCATED pad exceeds its own axis would make the clamp
    silently thin a real absorber, so it is rejected instead. No public
    constructor can build one today (every builder sizes an axis as
    ``interior + pad_lo + pad_hi``); the guard exists so that a future one
    cannot reintroduce a quietly weakened absorber through this path, and
    it is fed the malformed grid directly here rather than assumed.
    """
    with pytest.raises(ValueError, match="absorbing layers"):
        _assert_absorber_fits(_StubGrid(z_hi=12), (8, 8, 10))


def test_absorber_exactly_filling_its_axis_is_accepted():
    """Sibling: the boundary case must NOT raise."""
    _assert_absorber_fits(_StubGrid(z_hi=10), (8, 8, 10))
    _assert_absorber_fits(_StubGrid(z_lo=4, z_hi=6), (8, 8, 10))


# --------------------------------------------------------------------------
# 6. Clones must not collapse the per-face layout (found by the grep sweep).
# --------------------------------------------------------------------------

def test_quick_convergence_clone_keeps_the_per_face_layout():
    """``quick_convergence`` rebuilds the Simulation at each dx.

    It used to pass the collapsed ``_boundary`` string, so a five-PEC-face
    cavity came back with an absorber on all six faces at every refinement
    step — the swept geometry was not the geometry under study.
    """
    from rfx.convergence import quick_convergence

    sim = _cube_sim(4)

    factory = None
    # quick_convergence builds its factory internally; reach it by monkey-
    # patching convergence_study to capture the factory instead of running
    # the (expensive) sweep.
    import rfx.convergence as conv
    original = conv.convergence_study
    try:
        def _capture(*, sim_factory, **_kw):
            nonlocal factory
            factory = sim_factory
            return None
        conv.convergence_study = _capture
        quick_convergence(sim, dx_factors=[1.0])
    finally:
        conv.convergence_study = original

    assert factory is not None
    clone = factory(sim._build_grid().dx)
    assert clone._boundary_spec == sim._boundary_spec
    assert clone._pec_faces == sim._pec_faces
    assert clone._build_grid().face_pads == sim._build_grid().face_pads


def test_quick_convergence_clone_of_a_scalar_sim_is_unchanged():
    """Sibling: a legacy scalar-boundary simulation clones as before."""
    from rfx.convergence import quick_convergence

    sim = Simulation(freq_max=_FREQ, domain=_CUBE, cpml_layers=4,
                     boundary="cpml")
    sim.add_source((0.01, 0.01, 0.01), "ez", amplitude_kind="field")
    sim.add_probe(_PROBE, "ez")

    factory = None
    import rfx.convergence as conv
    original = conv.convergence_study
    try:
        def _capture(*, sim_factory, **_kw):
            nonlocal factory
            factory = sim_factory
            return None
        conv.convergence_study = _capture
        quick_convergence(sim, dx_factors=[1.0])
    finally:
        conv.convergence_study = original

    clone = factory(sim._build_grid().dx)
    assert clone._boundary == "cpml"
    assert clone._build_grid().face_pads == sim._build_grid().face_pads


# --------------------------------------------------------------------------
# 7. ADI has no per-face absorber (found by the #647 grep sweep).
# --------------------------------------------------------------------------

def test_adi_rejects_a_non_uniform_absorber_layout():
    """``solver='adi'`` stamps its absorbing sigma on all six faces.

    It reaches that path off ``self._boundary == 'cpml'``, which is true as
    soon as ANY face absorbs, so a per-face spec used to get an absorber
    exactly where the caller asked for a reflector — with nothing said.
    """
    with pytest.raises(ValueError, match="uniform absorber"):
        Simulation(
            freq_max=_FREQ, domain=(0.06, 0.06, 0.06), cpml_layers=8,
            boundary=BoundarySpec(x="cpml", y="cpml",
                                  z=Boundary(lo="pec", hi="cpml")),
            solver="adi",
        )


def test_adi_rejects_a_per_face_thickness_override():
    """Same blindness through the thickness knob rather than the token."""
    with pytest.raises(ValueError, match="uniform absorber"):
        Simulation(
            freq_max=_FREQ, domain=(0.06, 0.06, 0.06), cpml_layers=8,
            boundary=BoundarySpec(
                x="cpml", y="cpml",
                z=Boundary(lo="cpml", hi="cpml", hi_thickness=2)),
            solver="adi",
        )


@pytest.mark.parametrize("boundary", ["cpml", "pec", BoundarySpec.uniform("cpml")])
def test_adi_accepts_a_uniform_absorber_and_runs(boundary):
    """Sibling of both rejections: same solver, same domain, same budget.

    A uniform layout — scalar or spelled out as a BoundarySpec — must still
    construct AND produce a finite non-zero trace, so the rejection above
    is about the layout and not about ADI plus BoundarySpec in general.
    """
    sim = Simulation(
        freq_max=_FREQ, domain=(0.06, 0.06, 0.06), cpml_layers=8,
        boundary=boundary, solver="adi",
    )
    sim.add_source((0.03, 0.03, 0.03), "ez", amplitude_kind="field")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    trace = np.asarray(sim.run(n_steps=6, skip_preflight=True).time_series)
    assert np.all(np.isfinite(trace))
    assert np.max(np.abs(trace)) > 0.0
