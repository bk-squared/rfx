"""Source / wire-port normalization by the E-node DUAL spacing (issue #672).

Sibling of ``tests/unit/materials/test_thin_conductor_nu_dual_spacing.py`` (issue #671): the
same primal-vs-dual confusion, at two more surfaces on the non-uniform lane.

An ``E_a`` component is an EDGE along its own axis ``a`` and sits ON a node on
the two transverse axes. So its control volume is MIXED — primal per-cell
width on ``a``, dual spacing ``(d[k-1]+d[k])/2`` on the other two — and the
Ampere loop that measures its enclosed current runs on the DUAL face, each leg
weighted by the dual spacing along THAT H component's own axis. The
non-uniform E update makes this concrete: the Ex update divides by ``inv_dy``
and ``inv_dz`` and never by ``inv_dx`` (``_profile_to_inv_arrays``, CORE-C2).

Both surfaces used the primal per-cell width on all three axes, and the
wire-port loop additionally weighted each leg by a width from an unrelated
axis — invisible while ``dx == dy == dz``, which is why the uniform lane can
spell the whole loop with one ``dx`` and still be right there.

MEASURED at 13de212 (before) and on this branch (after).

**Source control volume.** Fixture: dx and dz profiles
``[0.5 mm]*5 + [1.0 mm]*6 + [0.5 mm]*5``, dy uniform 0.5 mm, node
i = k = 5 (the 0.5 -> 1.0 transition: primal 1.0 mm, dual 0.75 mm), j = 5.

  component   dV realized (before)   dV correct     before/correct
  ex          5.000001e-10           3.750001e-10   1.333333
  ey          5.000001e-10           2.812500e-10   1.777778
  ez          5.000001e-10           3.750000e-10   1.333333

After: 1.000000 on all three (re-measured by the tests below).

**Wire-port Ampere loop.** With distinct grading per axis so no leg can hide
behind another, the pre-#672 expression is off by 1.33x to 3.33x on the six
discriminating legs while the fixed helper reproduces the closed form exactly.

**Amplitude witness (diagnostic, not a gate).** Source on a 2:1 step node vs a
locally uniform node on the SAME mesh, probe 3 fine cells away through
identical cells both times, peak |Ez|:

  node                   before        after
  step (dual 0.75 mm)    2.344130e+08  1.562754e+08
  control (dual 0.5 mm)  1.801345e+08  1.801345e+08
  step/control ratio     1.301322      0.867548

The control node is untouched by the fix to all 7 printed digits — a node
whose neighbours are equal cannot move — and the step node changes by exactly
1.500000, the ratio the closed form predicts. The step/control ratio does NOT
land on 1.0 because the case source sits ON the transition and the control
does not, so the two launch into different local neighbourhoods; that residual
is a fixture confounder, not a normalization error, and the exact oracles
above are what bind. Recorded here rather than gated, per the #671 lesson that
a bare ref-vs-graded ratio is not a matched control.
"""

from __future__ import annotations

import warnings

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.core.yee import MaterialArrays, EPS_0
from rfx.nonuniform import (
    e_node_dual_spacings,
    e_node_dual_spacing_at,
    make_current_source,
    make_nonuniform_grid,
    wire_port_current,
)

D = 0.5e-3
# 0.5 -> 1.0 -> 0.5 mm; index 5 is the 2:1 transition node.
PROF = np.concatenate([np.full(5, D), np.full(6, 2.0 * D), np.full(5, D)])
K_STEP = 5
AXIS_OF = {"ex": 0, "ey": 1, "ez": 2}
RATIO_GATE = 1e-5        # float32 product of three widths + one divide
BLIND_GUARD = 0.30       # the wrong form must be at least this far off


def _dual_np(prof):
    """Dual spacings, written longhand in float64 so the oracle does not
    route through the helper the fix depends on."""
    p = np.asarray(prof, dtype=np.float64)
    return np.concatenate([p[:1], 0.5 * (p[:-1] + p[1:])])


# ---------------------------------------------------------------------------
# The two spellings of the dual rule must agree, and be exact on a uniform mesh
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prof", [
    PROF,
    np.full(9, D),
    np.concatenate([np.full(3, D), np.full(4, 4.0 * D), np.full(3, D)]),
])
def test_scalar_dual_spacing_matches_the_vector_helper(prof):
    vec = np.asarray(e_node_dual_spacings(prof))
    for k in range(prof.size):
        got = float(e_node_dual_spacing_at(np.asarray(prof, dtype=np.float64), k))
        # The vector helper goes through jnp (float32 by default), the scalar
        # one keeps the caller's dtype — agreement is to float32 roundoff.
        assert abs(got - float(vec[k])) <= 1e-6 * abs(got), (k, got, vec[k])


def test_dual_equals_primal_bit_exactly_on_a_uniform_profile():
    """This is what guarantees no uniform-mesh number can move."""
    prof = np.full(11, D)
    # Same dtype on both sides: the helper's jnp round-trip is float32 by
    # default, so compare against the float32 image of the profile.
    assert np.array_equal(np.asarray(e_node_dual_spacings(prof)),
                          prof.astype(np.float32))
    for k in range(prof.size):
        assert float(e_node_dual_spacing_at(prof, k)) == float(prof[k])


# ---------------------------------------------------------------------------
# Site 1 — the source control volume
# ---------------------------------------------------------------------------

def _grid(prof_x, prof_z):
    return make_nonuniform_grid((8e-3, 8e-3), prof_z, D, cpml_layers=0,
                                dx_profile=prof_x)


def _realized_dv(grid, idx, component):
    """Recover dV from the emitted waveform: w[0] = (Cb/dV) * waveform(0),
    and with eps_r = 1, sigma = 0 the analytic Cb is dt/EPS_0."""
    shape = grid.shape
    mats = MaterialArrays(eps_r=np.ones(shape, np.float32),
                          sigma=np.zeros(shape, np.float32),
                          mu_r=np.ones(shape, np.float32))
    src = make_current_source(grid, idx, component, lambda t: 1.0, 4, mats)
    return (grid.dt / EPS_0) / float(np.asarray(src[4])[0])


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
def test_source_control_volume_is_the_mixed_primal_dual_volume(component):
    grid = _grid(PROF, PROF)
    idx = (K_STEP, 5, K_STEP)          # j = 5 sits on the uniform y mesh
    prim = [np.asarray(grid.dx_arr, dtype=np.float64),
            np.asarray(grid.dy_arr, dtype=np.float64),
            np.asarray(grid.dz, dtype=np.float64)]
    dual = [_dual_np(p) for p in prim]
    p_axis = AXIS_OF[component]

    expected = 1.0
    primal_only = 1.0
    for a in range(3):
        expected *= (prim[a][idx[a]] if a == p_axis else dual[a][idx[a]])
        primal_only *= prim[a][idx[a]]

    got = _realized_dv(grid, idx, component)
    assert abs(got / expected - 1.0) <= RATIO_GATE, (
        f"{component}: realized dV {got:.6e} vs mixed primal/dual "
        f"{expected:.6e} (ratio {got / expected:.6f})")
    # Blindness guard: the fixture must still be graded enough that the
    # all-primal form is clearly excluded (it is what shipped before #672).
    assert abs(primal_only / expected - 1.0) > BLIND_GUARD, (
        f"{component}: fixture stopped discriminating — all-primal dV "
        f"{primal_only:.6e} is within {BLIND_GUARD:.0%} of the correct one")


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
def test_source_control_volume_is_exact_on_a_uniform_profile(component):
    """No uniform-mesh amplitude may move: dual == primal bit-exactly there."""
    prof = np.full(11, D)
    grid = _grid(prof, prof)
    got = _realized_dv(grid, (5, 5, 5), component)
    expected = (float(np.asarray(grid.dx_arr)[5])
                * float(np.asarray(grid.dy_arr)[5])
                * float(np.asarray(grid.dz)[5]))
    assert abs(got / expected - 1.0) <= RATIO_GATE, (got, expected)


def test_unknown_component_raises_instead_of_falling_back():
    """A silent primal fallback is the #369 vaporized-metal class."""
    grid = _grid(PROF, PROF)
    with pytest.raises(ValueError, match="unknown component"):
        _realized_dv(grid, (5, 5, 5), "hz")


def test_amplitude_kind_field_is_unaffected_by_the_control_volume():
    """``amplitude_kind='field'`` multiplies by dV/Cb, cancelling the Cb/dV
    applied to the waveform — for ANY dV. Pinned explicitly rather than
    assumed, because the cancellation only holds while both use the same
    ``dV`` object."""
    grid = _grid(PROF, PROF)
    shape = grid.shape
    mats = MaterialArrays(eps_r=np.ones(shape, np.float32),
                          sigma=np.zeros(shape, np.float32),
                          mu_r=np.ones(shape, np.float32))
    w = make_current_source(grid, (K_STEP, 5, K_STEP), "ez",
                            lambda t: 1.0, 4, mats, amplitude_kind="field")[4]
    np.testing.assert_allclose(np.asarray(w), 1.0, rtol=1e-6)


def test_source_control_volume_stays_on_the_ad_path():
    """``jax.grad`` w.r.t. a traced dz_profile must survive the fix.

    The failure mode this guards is a refactor that resolves the widths with
    ``float()`` on the traced branch, which kills the mesh-as-design-variable
    (GEO-C3) path with no forward-value change.
    """
    def dv(prof_z):
        g = make_nonuniform_grid((8e-3, 8e-3), prof_z, D, cpml_layers=0)
        shape = g.shape
        mats = MaterialArrays(eps_r=jnp.ones(shape), sigma=jnp.zeros(shape),
                              mu_r=jnp.ones(shape))
        src = make_current_source(g, (5, 5, K_STEP), "ez", lambda t: 1.0, 3,
                                  mats)
        return (g.dt / EPS_0) / src[4][0]

    g = np.asarray(jax.grad(dv)(jnp.asarray(PROF)))
    # ez at k: dV = dual_x * dual_y * d_z[k], so d(dV)/d(d_z[k]) is the
    # transverse dual area (x and y are uniform here, spacing D).
    analytic = D * D
    assert np.all(np.isfinite(g))
    assert abs(float(g[K_STEP]) / analytic - 1.0) < 1e-3, (g[K_STEP], analytic)


# ---------------------------------------------------------------------------
# Site 2 — the wire-port Ampere loop
# ---------------------------------------------------------------------------

# Distinct grading per axis, so a leg weighted by the WRONG axis cannot
# coincide with the right one.
# Ratios chosen so EVERY leg's wrong weight (the pre-#672 primal/wrong-axis
# one) is at least 30% off its correct dual weight — see the blindness guard.
LOOP_PROFS = [
    np.concatenate([np.full(5, D), np.full(6, 4.0 * D), np.full(5, D)]),
    np.concatenate([np.full(5, D), np.full(6, 2.0 * D), np.full(5, D)]),
    np.concatenate([np.full(5, D), np.full(6, 7.0 * D), np.full(5, D)]),
]
LOOP_IDX = (K_STEP, K_STEP, K_STEP)
# comp -> (H_c field, its own axis, H_b field, its own axis), (a, b, c) cyclic
LOOP_CASES = {"ez": ("hy", 0, "hx", 1),
              "ex": ("hz", 1, "hy", 2),
              "ey": ("hx", 2, "hz", 0)}


def _ramp_fields(cname, cax, a, bname, bax, b, n):
    """H fields linear in each component's own staggered coordinate, so the
    discrete difference across a node is EXACTLY slope * dual spacing."""
    shape = (n, n, n)
    flds = {"hx": np.zeros(shape), "hy": np.zeros(shape),
            "hz": np.zeros(shape)}
    for name, axis, slope in ((cname, cax, a), (bname, bax, b)):
        prof = LOOP_PROFS[axis]
        centres = np.concatenate([[0.0], np.cumsum(prof)])[:-1] + prof / 2
        arr = flds[name]
        for m in range(prof.size):
            sl = [slice(None)] * 3
            sl[axis] = m
            arr[tuple(sl)] = slope * centres[m]
    return {k: jnp.asarray(v) for k, v in flds.items()}


def _old_loop(hx, hy, hz, comp, mi, mj, mk, dxi, dyj, dz_local):
    """The pre-#672 expression, verbatim from 13de212."""
    if comp == "ez":
        return (hy[mi, mj, mk] - hy[mi - 1, mj, mk]
                - hx[mi, mj, mk] + hx[mi, mj - 1, mk]) * dxi
    if comp == "ex":
        return (hz[mi, mj, mk] - hz[mi, mj - 1, mk]
                - hy[mi, mj, mk] + hy[mi, mj, mk - 1]) * dz_local
    return (hx[mi, mj, mk] - hx[mi, mj, mk - 1]
            - hz[mi, mj, mk] + hz[mi - 1, mj, mk]) * dz_local


@pytest.mark.parametrize("comp", ["ex", "ey", "ez"])
@pytest.mark.parametrize("ab", [(1.0, 0.0), (0.0, 1.0)])
def test_wire_port_current_matches_the_closed_form_loop(comp, ab):
    """Closed form: with H_c = a*x_c and H_b = b*x_b, the enclosed current
    through the dual face is exactly ``(a - b) * dual_c * dual_b``.

    Each leg is exercised on its own so a fix that gets the SUM right by
    cancellation still fails.
    """
    a, b = ab
    n = LOOP_PROFS[0].size + 1
    cname, cax, bname, bax = LOOP_CASES[comp]
    flds = _ramp_fields(cname, cax, a, bname, bax, b, n)
    duals = [_dual_np(p)[LOOP_IDX[i]] for i, p in enumerate(LOOP_PROFS)]
    prims = [float(p[LOOP_IDX[i]]) for i, p in enumerate(LOOP_PROFS)]

    got = float(wire_port_current(flds["hx"], flds["hy"], flds["hz"],
                                  comp, *LOOP_IDX, *duals))
    closed = a * duals[cax] * duals[bax] - b * duals[bax] * duals[cax]
    assert abs(got / closed - 1.0) <= RATIO_GATE, (comp, ab, got, closed)

    old = float(_old_loop(flds["hx"], flds["hy"], flds["hz"], comp,
                          *LOOP_IDX, prims[0], prims[1], prims[2]))
    assert abs(old / closed - 1.0) > BLIND_GUARD, (
        f"{comp} {ab}: the pre-#672 expression is within {BLIND_GUARD:.0%} of "
        f"the closed form on this fixture — it no longer discriminates")


@pytest.mark.parametrize("comp", ["ex", "ey", "ez"])
def test_wire_port_current_matches_the_cubic_spelling(comp):
    """On dx == dy == dz the helper must reproduce the single-``dx`` uniform
    spelling to float roundoff.

    Not bit-identical: the old expression summed the four H terms and applied
    one multiply, the helper multiplies each leg by its own metric, so the
    two differ in the multiply order (measured ~1e-8 relative in float32).
    On a uniform mesh the METRICS are identical, which is the claim that
    matters — no uniform-mesh number moves beyond roundoff.
    """
    n = 12
    rng = np.random.default_rng(0)
    flds = {k: jnp.asarray(rng.standard_normal((n, n, n)))
            for k in ("hx", "hy", "hz")}
    idx = (6, 6, 6)
    got = float(wire_port_current(flds["hx"], flds["hy"], flds["hz"],
                                  comp, *idx, D, D, D))
    old = float(_old_loop(flds["hx"], flds["hy"], flds["hz"], comp, *idx,
                          D, D, D))
    # The loop is a difference of comparable terms, so the tolerance is
    # scaled to the TERM magnitude, not to the (possibly cancelling) result.
    scale = D * max(float(jnp.max(jnp.abs(flds[k]))) for k in flds)
    assert abs(got - old) <= 1e-5 * scale, (comp, got, old, scale)


def test_wire_port_current_rejects_an_unknown_component():
    z = jnp.zeros((4, 4, 4))
    with pytest.raises(ValueError, match="unknown component"):
        wire_port_current(z, z, z, "hz", 2, 2, 2, D, D, D)


# ---------------------------------------------------------------------------
# Preflight advisories
# ---------------------------------------------------------------------------

GRADED = [0.5e-3] * 8 + [1.5e-3] * 8      # node 8 (z = 4 mm) is the step
GRADED_DOWN = [1.5e-3] * 8 + [0.5e-3] * 8  # node 8 (z = 12 mm), DOWN step
UNIFORM = [0.5e-3] * 16
DXA = 0.5e-3
LXY = 24 * DXA


def _preflight(dz, add, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=10e9,
                         domain=(LXY, LXY, 0.0 if dz is not None else 8.0e-3),
                         dx=DXA, dz_profile=dz, boundary="cpml",
                         cpml_layers=6)
        add(sim, **kw)
        return " ".join(sim.preflight())


def test_preflight_advises_on_a_source_on_a_graded_node():
    """Fires for an ``ex`` source whose TRANSVERSE z axis is graded, stays
    silent when the graded axis is the component's own (exact there), on a
    matched node, on a uniform profile and on the uniform lane."""
    def _src(sim, z, comp):
        sim.add_source(position=(12 * DXA, 12 * DXA, z), component=comp,
                       waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
                       amplitude_kind="current")

    fired = _preflight(GRADED, _src, z=4.0e-3, comp="ex")
    assert "current source at" in fired, fired
    assert "TRANSVERSE" in fired and "DUAL spacing" in fired, fired
    assert "500µm below" in fired and "1.5mm above" in fired, fired

    # ez: z is the component's OWN axis -> primal, exact, no advisory
    assert "current source at" not in _preflight(GRADED, _src, z=4.0e-3,
                                                 comp="ez")
    # matched node on the same graded mesh / uniform profile / uniform lane
    assert "current source at" not in _preflight(GRADED, _src, z=2.0e-3,
                                                 comp="ex")
    assert "current source at" not in _preflight(UNIFORM, _src, z=4.0e-3,
                                                comp="ex")
    assert "current source at" not in _preflight(None, _src, z=4.0e-3,
                                                 comp="ex")


def test_source_advisory_ratio_names_the_cell_its_sentence_names():
    """The counterfactual ratio must be ``d[k]/dual`` — the primal width the
    pre-#672 code actually used — on DOWN steps as well as UP steps.

    On an UP step the two candidate spellings coincide, which is why every
    fixture above missed this. ``GRADED`` steps 0.5 -> 1.5 mm at node 8, so
    ``d[k] = 1.5mm`` IS ``max(d_below, d_above)`` and both give 1.500x.
    ``GRADED_DOWN`` steps 1.5 -> 0.5 mm at the same node: ``d[k] = 0.5mm``
    and ``dual = 1mm``, so the honest ratio is 0.500x while
    ``max(d_below, d_above)/dual`` would print 1.500x next to a sentence that
    has just named the 500µm cell as the primal one.
    """
    def _src(sim, z, comp):
        sim.add_source(position=(12 * DXA, 12 * DXA, z), component=comp,
                       waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
                       amplitude_kind="current")

    up = _preflight(GRADED, _src, z=4.0e-3, comp="ex")
    assert "not the primal cell 1.5mm" in up, up
    assert "off by 1.500x" in up, up

    down = _preflight(GRADED_DOWN, _src, z=12.0e-3, comp="ex")
    assert "not the primal cell 500µm" in down, down
    assert "off by 0.500x" in down, (
        "the advisory's ratio names a different cell from its sentence on a "
        f"DOWN step: {down}")


def test_preflight_advises_on_a_wire_port_on_a_graded_node():
    """Fires for an ``ex`` wire port whose z Ampere-loop axis is graded."""
    def _port(sim, z, comp):
        sim.add_port(position=(12 * DXA, 12 * DXA, z), component=comp,
                     impedance=50.0, extent=2 * DXA,
                     waveform=GaussianPulse(f0=5e9, bandwidth=0.8))

    fired = _preflight(GRADED, _port, z=4.0e-3, comp="ex")
    assert "wire port at" in fired and "Ampere-loop axes" in fired, fired
    assert "DUAL spacing" in fired, fired

    assert "Ampere-loop axes" not in _preflight(GRADED, _port, z=4.0e-3,
                                                comp="ez")
    assert "Ampere-loop axes" not in _preflight(GRADED, _port, z=2.0e-3,
                                                comp="ex")
    assert "Ampere-loop axes" not in _preflight(UNIFORM, _port, z=4.0e-3,
                                                comp="ex")
    assert "Ampere-loop axes" not in _preflight(None, _port, z=4.0e-3,
                                                comp="ex")


# ---------------------------------------------------------------------------
# Amplitude witness (diagnostic; see the module docstring for the confounder)
# ---------------------------------------------------------------------------

W_PROF = np.concatenate([np.full(4, D), np.full(8, 2 * D), np.full(12, D)])
W_NODES = np.concatenate([[0.0], np.cumsum(W_PROF)])
W_STEP, W_CTRL, W_OFF = 12, 16, 3


def _peak_at_probe(k_src):
    domx = float(W_PROF.sum())
    domy = domz = 8e-3
    sim = Simulation(freq_max=20e9, domain=(domx, domy, domz), dx=D,
                     dx_profile=W_PROF, boundary="cpml", cpml_layers=6)
    sim.add_source(position=(float(W_NODES[k_src]), domy / 2, domz / 2),
                   component="ez",
                   waveform=GaussianPulse(f0=8e9, bandwidth=0.8),
                   amplitude_kind="current")
    sim.add_probe(position=(float(W_NODES[k_src + W_OFF]), domy / 2, domz / 2),
                  component="ez")
    r = sim.run(n_steps=600, skip_preflight=True)
    return float(np.max(np.abs(np.asarray(r.time_series))))


def test_graded_step_source_amplitude_witness():
    """Regression lock on the measured step-vs-control amplitude ratio.

    NOT an invariance gate — the case source sits ON the mesh transition and
    the control does not, so the two launch into different neighbourhoods and
    the ratio does not reduce to 1. What the fix moved, measured across
    13de212 and this branch: the CONTROL node is unchanged to all 7 printed
    digits (1.801345e+08 both times — a node whose neighbours are equal
    cannot move) while the STEP node changed by exactly 1.500000
    (2.344130e+08 -> 1.562754e+08), which is the ratio the closed form
    predicts for one graded transverse axis.
    """
    step, ctrl = _peak_at_probe(W_STEP), _peak_at_probe(W_CTRL)
    ratio = step / ctrl
    print(f"[witness] step={step:.6e} ctrl={ctrl:.6e} ratio={ratio:.6f}")
    assert abs(ratio / 0.867548 - 1.0) < 0.02, (
        f"step/control peak ratio {ratio:.6f} moved off the recorded "
        f"0.867548 (pre-#672 it was 1.301322)")


# ---------------------------------------------------------------------------
# wp_meta slot-order guard
# ---------------------------------------------------------------------------

# Three DIFFERENT gradings, one per axis, each STRICTLY varying, so at every
# node the three dual spacings differ from each other AND from the primal
# width. No permutation of them can be silently self-consistent.
# The first and last cell must equal the boundary ``dx`` (CPML cells are
# uniform), so the varying part sits in the interior.
def _graded(slope):
    return np.concatenate([np.full(5, D),
                           D * (1.0 + slope * np.arange(1.0, 11.0)),
                           np.full(5, D)])


G_X, G_Y, G_Z = _graded(0.30), _graded(0.17), _graded(0.41)


def _nodes(prof):
    return np.concatenate([[0.0], np.cumsum(np.asarray(prof, dtype=np.float64))])


def test_wire_port_metrics_reach_the_ampere_loop_on_the_right_axes(monkeypatch):
    """The dual spacings must arrive at ``wire_port_current`` on the axes
    they were computed for.

    ``wp_meta`` is a flat 14-tuple built at one place in
    ``rfx/nonuniform.py`` and unpacked at another. Nothing about a positional
    tuple makes the two agree: swapping the ``dual_x`` and ``dual_y`` slots
    at the build site leaves the code importable, every other test green, and
    every uniform-mesh number bit-identical — while silently weighting each
    Ampere leg of every graded-mesh wire port by the wrong axis's spacing.
    This test is the guard for that class.

    It observes the PRODUCTION call rather than re-deriving the metrics. One
    spy records the padded grid the runner actually built (CPML padding
    shifts every index, so the unpadded input profiles are NOT the right
    oracle); a second records what the scan body passes to
    ``wire_port_current``. The expected duals then come from the longhand
    float64 ``_dual_np`` oracle applied to the padded profiles at the very
    indices production reported.

    Falsified by mutation (2026-08-20): swapping the ``dual_x`` / ``dual_y``
    build slots reds this test with ``dual_x: production passed 0.0008825,
    expected 0.001175 at cell (13, 13, 14)`` while the other 26 tests in this
    module all still pass. Restoring the slots turns it green again.

    A ``dz_local`` / ``dual_zk`` swap is caught from the other side too — the
    primal z width would arrive where the dual is expected, and the fixture
    keeps those two apart at every node.
    """
    import rfx.nonuniform as _nu
    import rfx.runners.nonuniform as _run

    grids = []
    orig_grid = _run.make_nonuniform_grid

    def grid_spy(*a, **kw):
        g = orig_grid(*a, **kw)
        grids.append(g)
        return g

    monkeypatch.setattr(_run, "make_nonuniform_grid", grid_spy)

    seen = []
    orig = _nu.wire_port_current

    def spy(hx, hy, hz, comp, mi, mj, mk, dual_x, dual_y, dual_z):
        seen.append((comp, int(mi), int(mj), int(mk),
                     float(dual_x), float(dual_y), float(dual_z)))
        return orig(hx, hy, hz, comp, mi, mj, mk, dual_x, dual_y, dual_z)

    monkeypatch.setattr(_nu, "wire_port_current", spy)

    nx, ny, nz = _nodes(G_X), _nodes(G_Y), _nodes(G_Z)
    sim = Simulation(freq_max=10e9,
                     domain=(float(G_X.sum()), float(G_Y.sum()),
                             float(G_Z.sum())),
                     dx=D, dx_profile=G_X, dy_profile=G_Y, dz_profile=G_Z,
                     boundary="cpml", cpml_layers=4)
    sim.add_port(position=(float(nx[9]), float(ny[9]), float(nz[9])),
                 component="ez", impedance=50.0,
                 extent=float(G_Z[9] + G_Z[10]),
                 excite=True, waveform=GaussianPulse(f0=2e9, bandwidth=0.9))
    sim.run(n_steps=8, compute_s_params=True,
            s_param_freqs=jnp.array([2e9]), skip_preflight=True)

    assert grids, "make_nonuniform_grid was never called"
    assert seen, ("wire_port_current was never called — the fixture stopped "
                  "exercising the NU wire-port path")
    grid = grids[0]
    comp, mi, mj, mk, got_x, got_y, got_z = seen[0]
    assert comp == "ez", comp

    px = np.asarray(grid.dx_arr, dtype=np.float64)
    py = np.asarray(grid.dy_arr, dtype=np.float64)
    pz = np.asarray(grid.dz, dtype=np.float64)
    want_x = float(_dual_np(px)[mi])
    want_y = float(_dual_np(py)[mj])
    want_z = float(_dual_np(pz)[mk])
    primal_z = float(pz[mk])

    # The fixture must actually discriminate: if these ever collapse onto one
    # another a permutation would slip through and this test would assert
    # nothing.
    assert len({round(v, 12) for v in (want_x, want_y, want_z, primal_z)}) == 4, (
        f"fixture no longer discriminates: dual_x={want_x} dual_y={want_y} "
        f"dual_z={want_z} primal_z={primal_z} at cell ({mi}, {mj}, {mk})")

    for name, got, want in (("dual_x", got_x, want_x),
                            ("dual_y", got_y, want_y),
                            ("dual_z", got_z, want_z)):
        assert abs(got - want) <= 1e-6 * want, (
            f"{name}: production passed {got:.6g}, expected {want:.6g} at "
            f"cell ({mi}, {mj}, {mk}). The wp_meta slots are crossed — an "
            f"Ampere leg is weighted by another axis's spacing.")


def test_wp_meta_live_cell_slots_carry_primal_widths(monkeypatch):
    """Issue #764 wp_meta slot guard: slots 13/14 (live cells, per-cell d_par).

    The whole-port gap voltage V_port = sum_live(-E_c * d_par,c) weights
    each live cell's E by the PRIMAL width on the component's own axis at
    THAT cell (#672 metric family) — never the dual spacing, and never the
    midpoint cell's width broadcast over the run. The graded-z fixture
    keeps primal and dual apart at every node and makes every interior
    primal width unique, so either substitution reds this test.

    Also pins the slot ORDER extension itself: slot 13 is the static
    live-cell index tuple whose midpoint is slots 0..2, slot 14 the
    matching per-cell primal widths (same order as slot 13).
    """
    import rfx.nonuniform as _nu
    import rfx.runners.nonuniform as _run

    grids = []
    orig_grid = _run.make_nonuniform_grid

    def grid_spy(*a, **kw):
        g = orig_grid(*a, **kw)
        grids.append(g)
        return g

    monkeypatch.setattr(_run, "make_nonuniform_grid", grid_spy)

    metas = []
    orig_build = _nu._build_wp_meta

    def build_spy(wire_ports, grid):
        m = orig_build(wire_ports, grid)
        metas.append(m)
        return m

    monkeypatch.setattr(_nu, "_build_wp_meta", build_spy)

    nx, ny, nz = _nodes(G_X), _nodes(G_Y), _nodes(G_Z)
    sim = Simulation(freq_max=10e9,
                     domain=(float(G_X.sum()), float(G_Y.sum()),
                             float(G_Z.sum())),
                     dx=D, dx_profile=G_X, dy_profile=G_Y, dz_profile=G_Z,
                     boundary="cpml", cpml_layers=4)
    sim.add_port(position=(float(nx[9]), float(ny[9]), float(nz[9])),
                 component="ez", impedance=50.0,
                 extent=float(G_Z[9] + G_Z[10]),
                 excite=True, waveform=GaussianPulse(f0=2e9, bandwidth=0.9))
    sim.run(n_steps=8, compute_s_params=True,
            s_param_freqs=jnp.array([2e9]), skip_preflight=True)

    assert grids, "make_nonuniform_grid was never called"
    assert metas, "_build_wp_meta was never called — the NU wire-port " \
                  "S-param path stopped routing through it"
    grid = grids[0]
    meta = metas[0][0]
    assert len(meta) == 15, (
        f"wp_meta is a {len(meta)}-tuple, expected 15 (slots 13/14 added "
        f"by issue #764)")

    live_cells = meta[13]
    d_par = meta[14]
    assert isinstance(live_cells, tuple) and len(live_cells) >= 2, live_cells
    assert len(d_par) == len(live_cells)

    # Slots 0..2 are the midpoint of the LIVE run (issue #764).
    assert tuple(meta[0:3]) == tuple(live_cells[len(live_cells) // 2]), (
        f"mid slots {meta[0:3]} are not the live-run midpoint of {live_cells}")

    pz = np.asarray(grid.dz, dtype=np.float64)
    dual_z = _dual_np(pz)
    for (ci, cj, ck), dp in zip(live_cells, d_par):
        want = float(pz[ck])          # PRIMAL width at THIS cell
        wrong_dual = float(dual_z[ck])
        assert abs(float(dp) - want) <= 1e-9 * want, (
            f"slot-14 d_par at cell ({ci},{cj},{ck}) is {float(dp):.6g}, "
            f"expected the primal dz {want:.6g} (dual would be "
            f"{wrong_dual:.6g}) — the V_port metric family moved off #672 "
            f"PRIMAL")
    # The fixture must discriminate: primal != dual and the two live-cell
    # primal widths differ (strictly graded interior).
    ks = [c[2] for c in live_cells]
    assert len({round(float(pz[k]), 15) for k in ks}) == len(ks), (
        "fixture no longer discriminates: equal primal dz across the live "
        "run")
    assert all(abs(float(pz[k]) - float(dual_z[k])) > 1e-12 for k in ks)
