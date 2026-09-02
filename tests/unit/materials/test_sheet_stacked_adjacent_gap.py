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

Preflight for that fixture, verbatim (copied from stdout, leading
indent included): ``  [PREFLIGHT] All checks passed (NTFF advisory tier;
the PEC-overlap error check runs on forward()/preflight()).``

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
    """ORACLE 2 — end-to-end field witness. Preflight on this fixture reads,
    verbatim: ``  [PREFLIGHT] All checks passed (NTFF advisory tier; the
    PEC-overlap error check runs on forward()/preflight()).``"""
    res = _stacked_gap_sim().run(n_steps=1500)
    ts = np.abs(np.asarray(res.time_series).ravel())
    peak = float(ts.max())
    # 4eb7fa4 measured 1.038701e-09 here; fixed measures 4.037870e-03.
    assert peak > 1e-5, peak


# ---------------------------------------------------------------------------
# The veto's exception: a LENGTH-1 axis is not a normal direction.
#
# The #690 veto above zeroes component ``c`` wherever no covering sheet has
# ``c`` tangential. On a 2-D grid (nz == 1) ``sheet_normal_axis`` calls every
# flat patch z-normal, because z IS its thinnest bounding-box axis — there is
# no other z extent available. The veto then zeroed ``mask_ez``, and in
# ``mode="2d_tmz"`` Ez is the only live E component, so the sheet became
# bit-identically inert: an f0 copper patch behaved exactly like vacuum.
#
# ``tangential_edge_masks`` already keeps the wrap on a length-1 axis for
# exactly this reason (#689) — the veto ran after it and threw the result
# away. Measured on the fixture below (20x20x1, dx = 1 mm, 6x6-cell copper
# patch, 400 steps):
#
#     ctx nnz (ex, ey, ez)          (36, 36,  0)  ->  (36, 36, 36)
#     G4 PEC/f0 identity, 2-D mask   False        ->  True
#     peak|Ez| at the patch node    7.536002e-02  ->  8.477738e-13
#     f0/none, patch node            1.000000     ->  0.000000
#     f0/none, shadow probe          1.000000     ->  0.051592
#     pec/none, shadow probe         0.051638 (reference, unchanged)
#
# The shadow ratio is the physics check, not just a mask count: at 15 GHz
# copper's Rs is 0.032 ohm/square, so a copper film must shadow like a PEC
# plate. Fixed f0 and PEC agree to 0.09%; the inert sheet is off by 19x.
#
# Preflight for the f0 and none fixtures, verbatim:
#   ``  [PREFLIGHT] All checks passed (NTFF advisory tier; the PEC-overlap
#   error check runs on forward()/preflight()).``
# and for the PEC reference, whose first probe sits inside the plate on
# purpose (that IS the measurement — Ez must be zero there):
#   ``  [PREFLIGHT] Port/source at (0.011, 0.011, 0) is inside PEC geometry
#   'pec'. Field will be zero. Move source outside PEC.``
#
# The exception is the DEGENERATE axis only, NOT ``periodic[c]``: see
# test_periodic_seam_gap_edge_stays_vetoed.
# ---------------------------------------------------------------------------

TMZ_SHAPE = (20, 20, 1)
TMZ_PATCH = Box((0.008, 0.008, 0.0), (0.014, 0.014, 0.001))


def _flat_spec(shape, xs, ys, normal, g=1.0):
    m = np.zeros(shape, bool)
    m[np.ix_(list(xs), list(ys), [0])] = True
    m = jnp.asarray(m)
    return SheetImpedanceSpec(mask=m, normal_axis=normal, g_sheet=g,
                              sigma_sheet=jnp.where(m, g / DX, 0.0))


def test_two_d_sheet_keeps_its_out_of_plane_component():
    """nz == 1: the veto must not fire on z. Without the exception the
    only live 2d_tmz component is zeroed and the sheet is inert."""
    spec = _flat_spec(TMZ_SHAPE, range(8, 14), range(8, 14), 2)
    ctx = build_sheet_impedance_ctx([spec], periodic=(False, False, True))
    assert _counts(ctx) == (36, 36, 36), _counts(ctx)

    # ...and it holds on the DEFAULT flags too, which is what the callers
    # that never pass ``periodic`` (forward(), the NU runner, the eager
    # S-parameter re-runs) get.
    ctx_def = build_sheet_impedance_ctx([spec])
    assert _counts(ctx_def) == (36, 36, 36), _counts(ctx_def)


def test_two_d_sheet_keeps_the_g4_pec_footprint_identity():
    """#677 G4 on a 2-D mask: the f0 footprint is what apply_pec_mask zeroes."""
    from rfx.boundaries.pec import apply_pec_mask
    from rfx.core.yee import init_state

    spec = _flat_spec(TMZ_SHAPE, range(8, 14), range(8, 14), 2)
    per = (False, False, True)
    ctx = build_sheet_impedance_ctx([spec], periodic=per)
    st = init_state(TMZ_SHAPE)
    ones = jnp.ones(TMZ_SHAPE, jnp.float32)
    out = apply_pec_mask(st._replace(ex=ones, ey=ones, ez=ones),
                         spec.mask, per)
    for zeroed, m in ((out.ex, ctx.mask_ex), (out.ey, ctx.mask_ey),
                      (out.ez, ctx.mask_ez)):
        np.testing.assert_array_equal(np.asarray(zeroed) == 0.0,
                                      np.asarray(m))


def test_periodic_seam_gap_edge_stays_vetoed():
    """The exception is the degenerate axis, NOT a periodic axis.

    Two one-layer z-normal films on cells 0 and n-1 of a z-PERIODIC domain
    are two films with the seam between them, exactly the #690 geometry —
    ``tangential_edge_masks`` fuses them through the wrap (32 Ez edges) and
    the veto must still remove them. Broadening the exception to
    ``union.shape[c] == 1 or periodic[c]`` reds this test with ez = 32.
    """
    from rfx.boundaries.pec import tangential_edge_masks
    per = (False, False, True)
    a = _spec(FOOT, FOOT, [0], 2)
    b = _spec(FOOT, FOOT, [NZ - 1], 2)
    fused = tangential_edge_masks(a.mask | b.mask, per)
    assert int(jnp.sum(fused[2])) == 32, int(jnp.sum(fused[2]))
    ctx = build_sheet_impedance_ctx([a, b], periodic=per)
    assert _counts(ctx) == (32, 32, 0), _counts(ctx)


def _tmz_sim(kind):
    sim = Simulation(freq_max=30e9, domain=(0.02, 0.02, 0.001), dx=1e-3,
                     boundary="pec", mode="2d_tmz")
    if kind == "pec":
        sim.add(TMZ_PATCH, material="pec")
    elif kind == "f0":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.add_thin_conductor(TMZ_PATCH, sigma_bulk=5.8e7,
                                   surface_impedance_f0=15e9)
    sim.add_source((0.004, 0.010, 0), "ez",
                   waveform=GaussianPulse(f0=15e9), amplitude_kind="field")
    sim.add_probe((0.011, 0.011, 0), "ez")   # inside the patch footprint
    sim.add_probe((0.017, 0.010, 0), "ez")   # shadow, behind the patch
    return sim


def test_two_d_f0_patch_shadows_like_pec_end_to_end():
    """Physics witness: copper at 15 GHz must shadow like PEC. Preflight
    for each fixture is quoted verbatim in the section comment above."""
    peaks = {}
    for kind in ("none", "f0", "pec"):
        res = _tmz_sim(kind).run(n_steps=400, skip_preflight=(kind == "pec"))
        ts = np.abs(np.asarray(res.time_series))
        peaks[kind] = (float(ts[:, 0].max()), float(ts[:, 1].max()))

    node_ratio = peaks["f0"][0] / peaks["none"][0]
    shadow_f0 = peaks["f0"][1] / peaks["none"][1]
    shadow_pec = peaks["pec"][1] / peaks["none"][1]
    # inert sheet measured 1.000000 / 1.000000; fixed 1.1e-11 / 0.051592
    assert node_ratio < 1e-6, (node_ratio, peaks)
    assert abs(shadow_f0 - shadow_pec) / shadow_pec < 0.02, \
        (shadow_f0, shadow_pec, peaks)
