"""#690 — two same-normal f0 sheets on ADJACENT layers must not load the gap.

Defect (found by external review against 4eb7fa4).
``build_sheet_impedance_ctx`` classified the UNION of every sheet's cell
mask with ``rfx.boundaries.pec.tangential_edge_masks``. That helper knows
only adjacency, so two one-layer sheets sharing a normal axis and sitting
on neighbouring cell layers read as one two-cell body, and the SHEET-NORMAL
E edge between them was handed to the resistive sheet update. Since #677 an
f0 sheet is node-thin, so the two layers are two films one dual spacing
apart with vacuum in between — that edge is the dielectric gap of a
two-layer board and must stay lossless.

Fix: each spec already declares its ``normal_axis``; keep component ``c``
only where some covering sheet has ``c`` tangential.

ORACLE 1 (exact, hand-enumerated). An f0 sheet is realized on ONE E node
along its normal (``check_sheet_occupancy``), so it loads only its two
in-plane components, one edge per footprint cell per layer. For n adjacent
one-layer z-normal sheets over a footprint of |F| cells (>= 2 cells wide in
both in-plane directions):

    |mask_ex| = |mask_ey| = |F| * n      |mask_ez| = 0

|F| = 16 (4x4), n = 2, 6x6x8 grid:

    expected                 (32, 32,  0)
    measured at 4eb7fa4      (32, 32, 32)   <- 32 spurious gap edges
    measured with the fix    (32, 32,  0)

ORACLE 2 (physics witness, independent of the mask code). The gap edge is
vacuum, so it must ring, not be clamped. With the defect the gap sees
sigma = 1/(Rs0*dz) = 9.0e4 S/m, giving x2 = sigma*dt/eps0 ~ 4.8e3 and
A = exp(-x2) = 0.0 in float32 — the edge is pinned to curlH/sigma every
step. Fixture below: 4x4x4 mm PEC box, dx = 0.25 mm, two 2x2 mm copper f0
films on adjacent layers (z = 2.00 and 2.25 mm), 1500 steps. The source is
an Ex pulse at z = 1 mm, OFF the films and off the probed edge — an Ez
source sitting on the gap edge writes that edge directly every step and
masks the loss it is supposed to reveal (measured: peak 8.70e-01 with the
defect vs 1.24e+00 fixed, only 1.4x, useless as a gate). Driving from
outside and letting the field couple into the gap through the open plate
perimeter separates them cleanly:

    peak|Ez_gap|         tail_rms(last 300)
    4eb7fa4   1.038701e-09         3.379344e-11
    fixed     4.037870e-03         4.693612e-06

Separation 3.9e6 (~132 dB), so the 1e-5 gate is not tuned: the fixed value
clears it by 400x and the defect value misses it by 1e4.

Preflight for that fixture, verbatim: ``[PREFLIGHT] All checks passed.``

NEGATIVE CONTROL in this module: the same two sheets moved to NON-adjacent
layers must give a bit-identical ctx to the pre-fix code, and the coplanar
abutting 1-cell-strip case must keep both in-plane components (the literal
per-spec-OR alternative loses the transverse one, which would be a new
composition bug for meanders/combs drawn as several boxes).
"""

import warnings

import numpy as np
import jax.numpy as jnp

from rfx import Box, GaussianPulse, Simulation
from rfx.materials.thin_conductor import (
    SheetImpedanceSpec,
    build_sheet_impedance_ctx,
)

F0 = 10e9
DX = 0.25e-3
NX, NY, NZ = 6, 6, 8
FOOT = range(1, 5)          # 4 cells wide in both in-plane directions


def _spec(xs, ys, zs, normal, g=1.0):
    m = np.zeros((NX, NY, NZ), bool)
    m[np.ix_(list(xs), list(ys), list(zs))] = True
    m = jnp.asarray(m)
    return SheetImpedanceSpec(mask=m, normal_axis=normal, g_sheet=g,
                              sigma_sheet=jnp.where(m, g / DX, 0.0))


def _counts(ctx):
    return (int(jnp.sum(ctx.mask_ex)),
            int(jnp.sum(ctx.mask_ey)),
            int(jnp.sum(ctx.mask_ez)))


def test_adjacent_same_normal_sheets_leave_the_gap_edge_unloaded():
    """ORACLE 1: |mask_ex| = |mask_ey| = |F|*n, |mask_ez| = 0."""
    ctx = build_sheet_impedance_ctx([
        _spec(FOOT, FOOT, [3], 2),
        _spec(FOOT, FOOT, [4], 2, g=2.0),
    ])
    assert _counts(ctx) == (32, 32, 0), _counts(ctx)

    # the in-plane sets are exactly the union of each film's own classification
    from rfx.boundaries.pec import tangential_edge_masks
    a = _spec(FOOT, FOOT, [3], 2)
    b = _spec(FOOT, FOOT, [4], 2, g=2.0)
    ax, ay, _ = tangential_edge_masks(a.mask)
    bx, by, _ = tangential_edge_masks(b.mask)
    assert bool(jnp.all(ctx.mask_ex == (ax | bx)))
    assert bool(jnp.all(ctx.mask_ey == (ay | by)))

    # sigma is still the per-node sum; the two films occupy disjoint cells,
    # so no mixing happens and each layer keeps its own value.
    sig = np.asarray(ctx.sigma_sheet)
    assert np.isclose(sig[2, 2, 3], 1.0 / DX)
    assert np.isclose(sig[2, 2, 4], 2.0 / DX)


def test_deeper_same_normal_stack_leaves_every_gap_edge_unloaded():
    """Three adjacent layers: |mask_ez| stays 0, in-plane scales with n."""
    ctx = build_sheet_impedance_ctx([
        _spec(FOOT, FOOT, [3], 2),
        _spec(FOOT, FOOT, [4], 2),
        _spec(FOOT, FOOT, [5], 2),
    ])
    assert _counts(ctx) == (48, 48, 0), _counts(ctx)


def test_non_adjacent_stack_is_unchanged_negative_control():
    """The pre-fix code already got this right; it must not move."""
    ctx = build_sheet_impedance_ctx([
        _spec(FOOT, FOOT, [2], 2),
        _spec(FOOT, FOOT, [5], 2, g=2.0),
    ])
    assert _counts(ctx) == (32, 32, 0), _counts(ctx)


def test_single_sheet_ctx_is_unchanged_negative_control():
    ctx = build_sheet_impedance_ctx([_spec(FOOT, FOOT, [3], 2)])
    assert _counts(ctx) == (16, 16, 0), _counts(ctx)


def test_coincident_same_normal_sheets_still_add_conductance():
    """Parallel sheet admittances on one node: 1/Rs_tot = 1/Rs_1 + 1/Rs_2."""
    ctx = build_sheet_impedance_ctx([
        _spec(FOOT, FOOT, [3], 2, g=1.0),
        _spec(FOOT, FOOT, [3], 2, g=2.0),
    ])
    assert _counts(ctx) == (16, 16, 0), _counts(ctx)
    assert np.isclose(float(np.asarray(ctx.sigma_sheet)[2, 2, 3]), 3.0 / DX)


def test_coplanar_abutting_one_cell_strips_keep_both_in_plane_components():
    """A patterned plane drawn as abutting 1-cell boxes must classify the
    same as the single Box that covers them. The literal per-spec-OR
    alternative to the #690 fix drops mask_ex here (measured 8 -> 0)."""
    ctx = build_sheet_impedance_ctx([
        _spec([3], FOOT, [3], 2),
        _spec([4], FOOT, [3], 2),
    ])
    one_box = build_sheet_impedance_ctx([_spec([3, 4], FOOT, [3], 2)])
    assert _counts(ctx) == _counts(one_box), (_counts(ctx), _counts(one_box))
    assert bool(jnp.all(ctx.mask_ex == one_box.mask_ex))
    assert bool(jnp.all(ctx.mask_ey == one_box.mask_ey))
    assert bool(jnp.all(ctx.mask_ez == one_box.mask_ez))


def _stacked_gap_sim():
    sim = Simulation(freq_max=20e9, domain=(4e-3, 4e-3, 4e-3), dx=DX,
                     boundary="pec")
    for zc in (2.0e-3, 2.25e-3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.add_thin_conductor(
                Box((1e-3, 1e-3, zc), (3e-3, 3e-3, zc)),
                sigma_bulk=5.8e7, surface_impedance_f0=F0)
    # Source OFF the films and OFF the probed edge — see the module
    # docstring: an Ez source on the gap edge writes it directly and hides
    # the very loss under test.
    sim.add_source((2e-3, 2e-3, 1.0e-3), "ex",
                   waveform=GaussianPulse(f0=F0), amplitude_kind="field")
    sim.add_probe((2e-3, 2e-3, 2.125e-3), "ez")
    return sim


def test_gap_between_stacked_films_rings_instead_of_being_clamped():
    """ORACLE 2 — end-to-end field witness. Preflight on this fixture reads
    ``[PREFLIGHT] All checks passed.`` verbatim."""
    res = _stacked_gap_sim().run(n_steps=1500)
    ts = np.abs(np.asarray(res.time_series).ravel())
    peak = float(ts.max())
    # 4eb7fa4 measured 1.038701e-09 here; fixed measures 4.037870e-03.
    assert peak > 1e-5, peak
