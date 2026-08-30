"""Issue #469: the auto n_probe_offset solves BOTH clearances, not one edge.

``add_msl_port``'s auto-default is the upstream-only lower edge
(``max(3, λ/(4π·dx), 5·h_sub/dx)``); ``compute_msl_s_matrix`` now solves the
downstream reflector constraint too (``_resolve_msl_auto_offsets``): midpoint
of the compliant interval when a reflector bounds it, unchanged when none
does, loud warning + upstream-priority fallback when the interval is empty
(the honest "this feed line is too short for a clean measurement" case —
the pre-#469 advisory recommended INCREASING the offset, which moves the
probes toward the reflector).

Geometry mirrors the #469 measurement (Sheen 1990 LPF rfx leg: dx=200 µm,
h_sub=0.794 mm, εr=2.2, feed ≈ 10 mm, f_max=20 GHz). Hand arithmetic for
the pinned expectations (kept independent of the implementation):
  offset_min = round(5·0.794/0.2) = 20 cells (fringing term dominates)
  spacing    = round(λ_min_eff/8/4/dx) = 2 cells, span = (5−1)·2 = 8 cells
  λ_g/4 clearance = 0.25·c/(20 GHz·√5) = 1.676 mm
  d(feed→patch) = 12.466 − 2.5 = 9.966 mm
  offset_max = int((9.966 − 1.676)/0.2) − 8 = 41 − 8 = 33
  midpoint   = (20 + 33)//2 = 26  (inside the measured-clean window;
               the old default 20 sat at the contaminated near edge)
"""
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.grid import Grid
from rfx.api._sparams import _resolve_msl_auto_offsets

DX = 2e-4
DOMAIN = (0.020, 0.02632, 0.0038)
Y_C = 0.01316
W_TRACE = 0.002413
H_SUB = 0.000794


def _sim_with_feed_and_patch(patch_x0=0.012466, patch_x1=0.015006):
    sim = Simulation(freq_max=20e9, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (DOMAIN[0], DOMAIN[1], H_SUB)), material="sub")
    # the port's own feed trace (contains the feed plane -> excluded)
    sim.add(Box((0.001, Y_C - W_TRACE / 2, H_SUB),
                (patch_x0, Y_C + W_TRACE / 2, H_SUB + DX)), material="pec")
    # the patch = downstream reflector (wide, but < 80 % of domain y)
    sim.add(Box((patch_x0, 0.003, H_SUB),
                (patch_x1, 0.02332, H_SUB + DX)), material="pec")
    sim.add_msl_port(position=(0.0025, Y_C, 0.0), width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    return sim


def _grid():
    return Grid(freq_max=20e9, domain=DOMAIN, dx=DX, cpml_layers=8)


def test_auto_offset_moves_to_interval_midpoint_with_reflector():
    """Sheen-parameter geometry: auto offset resolves to the midpoint 26 of
    the compliant interval [20, 33] (hand arithmetic in the module
    docstring), not the contaminated lower edge 20."""
    sim = _sim_with_feed_and_patch()
    (pe,) = sim._msl_ports
    assert pe.n_probe_offset == 20  # stored lower edge (pre-#469 default)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (resolved,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports),
                                                _grid())
    assert resolved.n_probe_offset == 26
    # the stored entry and the auto bookkeeping are untouched (idempotency)
    assert sim._msl_ports[0].n_probe_offset == 20
    (again,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), _grid())
    assert again.n_probe_offset == 26


def test_explicit_offset_is_never_adjusted():
    sim = _sim_with_feed_and_patch()
    sim.add_msl_port(position=(0.0025, Y_C, 0.0), width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0, eps_r_sub=2.2,
                     name="pexp", n_probe_offset=30)
    resolved = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), _grid())
    by_name = {pe.name: pe for pe in resolved}
    assert by_name["pexp"].n_probe_offset == 30


def test_no_reflector_keeps_the_pre469_default():
    """Only the port's own trace (and substrate) registered: the solve must
    return the stored lower edge unchanged — byte-identity with the
    pre-#469 default on open thru lines."""
    sim = Simulation(freq_max=20e9, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (DOMAIN[0], DOMAIN[1], H_SUB)), material="sub")
    sim.add(Box((0.001, Y_C - W_TRACE / 2, H_SUB),
                (0.019, Y_C + W_TRACE / 2, H_SUB + DX)), material="pec")
    sim.add_msl_port(position=(0.0025, Y_C, 0.0), width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (resolved,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports),
                                                _grid())
    assert resolved.n_probe_offset == 20


def test_empty_interval_warns_and_keeps_upstream_priority():
    """Patch 5 mm from the feed: offset_max = int((5−1.676)/0.2)−8 = 8 < 20
    -> the two clearances are mutually unsatisfiable; warn loudly and keep
    the upstream edge (the pre-#469 behavior was a silent compromise)."""
    sim = _sim_with_feed_and_patch(patch_x0=0.0075, patch_x1=0.010)
    with pytest.warns(UserWarning, match="mutually unsatisfiable"):
        (resolved,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports),
                                                _grid())
    assert resolved.n_probe_offset == 20


def test_own_trace_of_minus_x_port_is_not_a_reflector():
    """'-x' port with only its own output feed trace: the pre-#469
    heuristic (x-extent >= 80 % of a broken inter-port-extent estimate)
    read the port's OWN trace as a reflector at 0 µm (the cv07 p2 false
    positive, validation/crossval/07_sheen_lpf.py known-residual note);
    the feed-containment exclusion must leave this geometry
    reflector-free."""
    sim = Simulation(freq_max=20e9, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (DOMAIN[0], DOMAIN[1], H_SUB)), material="sub")
    sim.add(Box((0.008, Y_C - W_TRACE / 2, H_SUB),
                (0.018, Y_C + W_TRACE / 2, H_SUB + DX)), material="pec")
    sim.add_msl_port(position=(0.0175, Y_C, 0.0), width=W_TRACE, height=H_SUB,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (resolved,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports),
                                                _grid())
    assert resolved.n_probe_offset == 20


def test_driver_applies_the_solve_end_to_end():
    """The empty-interval warning must surface through the PUBLIC
    compute_msl_s_matrix call (n_steps=1: the solve runs before the FDTD,
    so a minimal-step run proves the wiring)."""
    sim = _sim_with_feed_and_patch(patch_x0=0.0075, patch_x1=0.010)
    with pytest.warns(UserWarning, match="mutually unsatisfiable"):
        sim.compute_msl_s_matrix(freqs=np.array([5e9, 10e9]), n_steps=1)




# ---------------------------------------------------------------------------
# Issue #686 — the bail-out is about the PROPAGATION axis, not "any axis".
#
# The pre-#686 gate was ``getattr(grid, "dz", None) is not None``: it fired on
# every NonUniformGrid. Its stated reason ("cell-counted intervals are
# ill-defined under graded dx") is a statement about the axis the probes march
# along, and a dz_profile does not grade dx. For an x-propagating port the
# interval is well defined on a z-graded mesh and the solve simply never ran.
#
# EVERY test below runs over all four supported directions. The first cut of
# these tests used '+x' throughout — including both halves of the
# anti-permutation test — and an implementation that ignored the port's axis
# entirely (``(dx_arr, dy_arr, dz)[axis]`` -> ``[0]``, i.e. "always read dx")
# passed the whole file unchanged while re-instating the exact #686 defect for
# '+y'/'-y' ports. A direction-blind axis gate cannot be gated by
# direction-blind fixtures.
# ---------------------------------------------------------------------------

#: The board is defined in the port's own (propagation, width) frame and
#: mapped onto (x, y) per direction, so all four directions get the SAME
#: geometry in the frame that matters and therefore the same hand arithmetic
#: as the uniform case above: stored edge 20, solved midpoint 26.
_L_P = DOMAIN[0]          # 0.020 m along the propagation axis
_L_W = DOMAIN[1]          # 0.02632 m across it
_W_C = Y_C                # trace centre on the width axis
_N_P = int(round(_L_P / DX))    # 100 interior cells along propagation
_N_W = int(round(_L_W / DX))    # 132 interior cells across

_ALL_DIRECTIONS = ("+x", "-x", "+y", "-y")


def _board_domain(direction):
    """Domain for a board whose feed runs along ``direction``'s axis."""
    return ((_L_P, _L_W, DOMAIN[2]) if direction[1] == "x"
            else (_L_W, _L_P, DOMAIN[2]))


def _pw(direction, p, w):
    """Map a (propagation, width) coordinate to (x, y) for ``direction``.

    A '-' direction mirrors the propagation coordinate, so the port still
    launches INTO the board and the patch is still downstream of the feed.
    """
    q = p if direction[0] == "+" else _L_P - p
    return (q, w) if direction[1] == "x" else (w, q)


def _pw_box(direction, p0, p1, w0, w1, z0, z1):
    x0, y0 = _pw(direction, p0, w0)
    x1, y1 = _pw(direction, p1, w1)
    return Box((min(x0, x1), min(y0, y1), z0),
               (max(x0, x1), max(y0, y1), z1))


def _nu_sim(direction="+x", **profiles):
    """The Sheen-parameter feed+patch board, oriented along ``direction``."""
    dom = _board_domain(direction)
    sim = Simulation(freq_max=20e9, domain=dom, dx=DX,
                     boundary="cpml", cpml_layers=8, **profiles)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (dom[0], dom[1], H_SUB)), material="sub")
    # the port's own feed trace (contains the feed plane -> excluded)
    sim.add(_pw_box(direction, 0.001, 0.012466,
                    _W_C - W_TRACE / 2, _W_C + W_TRACE / 2,
                    H_SUB, H_SUB + DX), material="pec")
    # the patch = downstream reflector
    sim.add(_pw_box(direction, 0.012466, 0.015006, 0.003, 0.02332,
                    H_SUB, H_SUB + DX), material="pec")
    px, py = _pw(direction, 0.0025, _W_C)
    sim.add_msl_port(position=(px, py, 0.0), width=W_TRACE, height=H_SUB,
                     direction=direction, impedance=50.0, eps_r_sub=2.2,
                     name="p1")
    return sim


def _graded(n, d0, ratio):
    """Graded interior with both boundary cells back at ``d0``.

    make_nonuniform_grid requires profile[0] == profile[-1] == dx for the
    x/y axes (the CPML cells use the boundary spacing). ``ratio == 1.0``
    yields a uniform-VALUED profile: the NU code path runs, nothing grades.
    """
    prof = np.full(n, d0, dtype=float)
    prof[n // 4:3 * n // 4] = d0 * ratio
    return prof


def _profiles(direction, *, prop_ratio=1.0, width_ratio=1.0):
    """dx/dy profiles grading this port's PROPAGATION and WIDTH axes.

    Both in-board axes always carry an explicit profile, so which one is
    graded is the only difference between the cases below — and the two
    ratios are never equal, so an implementation that read the wrong axis
    would have to agree by coincidence rather than by construction.
    """
    prop = _graded(_N_P, DX, prop_ratio)
    width = _graded(_N_W, DX, width_ratio)
    return ({"dx_profile": prop, "dy_profile": width}
            if direction[1] == "x"
            else {"dx_profile": width, "dy_profile": prop})


def _resolved(sim):
    grid = sim._build_nonuniform_grid()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entries = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), grid)
    return entries[0].n_probe_offset, caught


def _skips(caught):
    return [str(w.message) for w in caught if "SKIPPED" in str(w.message)]


@pytest.mark.parametrize("direction", _ALL_DIRECTIONS)
def test_z_graded_stackup_now_runs_the_solve(direction):
    """The headline case: graded dz, in-board port -> the solve runs.

    26 is the same midpoint the uniform grid resolves to (see
    ``test_auto_offset_moves_to_interval_midpoint_with_reflector``); the
    propagation axis is ungraded, so the cell count is the same one.
    """
    dz = 0.2e-3 * 1.07 ** np.arange(14, dtype=float)
    assert float(dz.max() / dz.min()) > 2.0, "fixture dz is not graded"
    sim = _nu_sim(direction, dz_profile=dz)
    assert sim._msl_auto_offset_min == {"p1": 20}
    off, caught = _resolved(sim)
    assert off == 26, f"expected the solved midpoint 26, got {off}"
    assert not _skips(caught)


@pytest.mark.parametrize("direction", _ALL_DIRECTIONS)
def test_uniform_valued_profile_is_not_grading(direction):
    """Uniform-valued dx AND dy profiles take the NU path, grade nothing."""
    sim = _nu_sim(direction, **_profiles(direction))
    off, caught = _resolved(sim)
    assert off == 26
    assert not _skips(caught)


@pytest.mark.parametrize("direction", _ALL_DIRECTIONS)
def test_graded_width_axis_does_not_block_the_solve(direction):
    """Grading the WIDTH axis must not bail: the probes do not march
    along it, so the cell count they need is still single-valued."""
    sim = _nu_sim(direction, **_profiles(direction, width_ratio=1.25))
    off, caught = _resolved(sim)
    assert off == 26
    assert not _skips(caught)


@pytest.mark.parametrize("direction", _ALL_DIRECTIONS)
def test_graded_propagation_axis_bails_out_and_says_so(direction):
    """Graded propagation axis: keep the stored edge, but WARN."""
    sim = _nu_sim(direction, **_profiles(direction, prop_ratio=1.5))
    off, caught = _resolved(sim)
    assert off == 20, "a graded propagation axis must keep the stored edge"
    msgs = _skips(caught)
    assert len(msgs) == 1, msgs
    assert f"propagation axis {direction[1]} is GRADED" in msgs[0]
    assert f"direction={direction!r}" in msgs[0]
    assert "'p1'" in msgs[0]
    assert "#686" in msgs[0]


@pytest.mark.parametrize("direction", _ALL_DIRECTIONS)
def test_axis_roles_are_not_permutable(direction):
    """The gate must read THIS port's propagation axis — not a fixed one.

    Both cases below grade exactly one in-board axis, at DIFFERENT ratios,
    and differ only in WHICH role that axis plays for this port. Any
    implementation that reads a hard-coded axis instead of the port's own
    gets one of the two backwards for at least one direction: '[0]'
    (always dx) inverts both '+y' and '-y', '[1]' (always dy) inverts both
    '+x' and '-x', and '[2]' (always dz) fails the z-graded test above.
    """
    graded_prop, _ = _resolved(
        _nu_sim(direction, **_profiles(direction, prop_ratio=1.5)))
    assert graded_prop == 20, (
        f"{direction}: grading the propagation axis must bail out")

    graded_width, _ = _resolved(
        _nu_sim(direction, **_profiles(direction, width_ratio=1.25)))
    assert graded_width == 26, (
        f"{direction}: grading the width axis must NOT bail out")


@pytest.mark.parametrize("direction", _ALL_DIRECTIONS)
def test_traced_profile_is_reported_as_unevaluable(direction):
    """A mesh-as-design-variable profile cannot be inspected host-side —
    keep the stored edge and say WHY, rather than silently."""
    import jax
    import jax.numpy as jnp

    base = np.full(_N_P, DX)
    axis_kw = "dx_profile" if direction[1] == "x" else "dy_profile"

    def _probe(prof):
        sim = _nu_sim(direction, **{axis_kw: prof})
        grid = sim._build_nonuniform_grid()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            entries = _resolve_msl_auto_offsets(
                sim, list(sim._msl_ports), grid)
        _probe.result = (entries[0].n_probe_offset,
                         [str(w.message) for w in caught])
        return jnp.sum(prof)

    jax.jit(_probe)(jnp.asarray(base))
    off, msgs = _probe.result
    assert off == 20
    skipped = [m for m in msgs if "SKIPPED" in m]
    assert len(skipped) == 1, msgs
    assert "traced" in skipped[0]
    assert f"the {direction[1]}-axis cell sizes are a traced" in skipped[0]


# ---------------------------------------------------------------------------
# Issue #681: auto probe-SPACING widening (span solve)
# ---------------------------------------------------------------------------
#
# A ~0.1 lambda_g probe span leaves the N-probe beta fit noise-fragile
# (measured, 500-trial Monte-Carlo at 1% probe noise, N=5: median beta
# error 5.5% at a 0.10 lambda_g span vs 0.81% at 0.30 lambda_g). For
# AUTO-spacing ports _resolve_msl_auto_offsets widens the spacing toward
# lambda_g(f_max)/4 per probe step, capped by half the reflector interval
# (the other half stays with the #469 midpoint rule) and by the absorber
# clearance. Hand arithmetic for the open-thru geometry below (dx=200um,
# eps_eff_HJ(w=2.413mm, h=0.794mm, eps_r=2.2) = 1.8697):
#   lambda_g(20 GHz) = c/(20e9*sqrt(1.8697)) = 10.96 mm
#   target spacing   = round(10.96/4/0.2) = 14 cells
#   feed at 2.5 mm, '+x', domain 20 mm -> dist_edge = 17.5 mm
#   clear = 1.676 mm (lambda_g/4 at f_max, eps-proxy 5), cpml = 8 cells
#   span budget = int((17.5-1.676)/0.2) - 8 - 20 = 51 cells
#   spacing = min(14, 51//4) = 12, span = 48 cells = 9.6 mm = 0.88 lambda_g
#   (the Sheen REFLECTOR geometry stays byte-identical: budget
#    21//2 = 10 cells -> spacing max(2, 10//4) = 2 = the registration
#    default, offset midpoint 26 -- covered by the #469 tests above.)


def _open_thru_sim():
    sim = Simulation(freq_max=20e9, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (DOMAIN[0], DOMAIN[1], H_SUB)), material="sub")
    sim.add(Box((0.001, Y_C - W_TRACE / 2, H_SUB),
                (0.019, Y_C + W_TRACE / 2, H_SUB + DX)), material="pec")
    return sim


def test_auto_spacing_widens_on_open_thru():
    """No reflector, long feed: spacing widens 2 -> 12 cells (hand
    arithmetic above); the deepest probe stays clear of the absorber."""
    sim = _open_thru_sim()
    sim.add_msl_port(position=(0.0025, Y_C, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0,
                     eps_r_sub=2.2, name="p1")
    (pe,) = sim._msl_ports
    assert pe.n_probe_spacing == 2      # conservative registration default
    assert sim._msl_auto_probe_spacing["p1"] == pytest.approx(1.8697, abs=1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (r,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), _grid())
    assert r.n_probe_offset == 20       # no reflector: offset untouched
    assert r.n_probe_spacing == 12
    deepest = r.n_probe_offset + (r.n_probes - 1) * r.n_probe_spacing
    n_cells = int(DOMAIN[0] / DX)
    clear_cells = int(0.001676 / DX)
    assert deepest <= n_cells - 8 - clear_cells
    # span in guide wavelengths at f_max: >= 0.3 (the measured-adequate
    # regime), <= 1.0 + rounding (the (N-1)/4 anti-alias ceiling)
    lam_g = 2.998e8 / (20e9 * np.sqrt(1.8697))
    span = (r.n_probes - 1) * r.n_probe_spacing * DX
    assert 0.3 * lam_g <= span <= 1.05 * lam_g


def test_auto_spacing_is_idempotent_including_on_resolved_entries():
    sim = _open_thru_sim()
    sim.add_msl_port(position=(0.0025, Y_C, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0,
                     eps_r_sub=2.2, name="p1")
    (r,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), _grid())
    (r2,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), _grid())
    (r3,) = _resolve_msl_auto_offsets(sim, [r], _grid())
    assert (r.n_probe_offset, r.n_probe_spacing) \
        == (r2.n_probe_offset, r2.n_probe_spacing) \
        == (r3.n_probe_offset, r3.n_probe_spacing) == (20, 12)
    # stored entry untouched
    assert sim._msl_ports[0].n_probe_spacing == 2


def test_explicit_spacing_is_never_widened():
    sim = _open_thru_sim()
    sim.add_msl_port(position=(0.0025, Y_C, 0.0), width=W_TRACE,
                     height=H_SUB, direction="+x", impedance=50.0,
                     eps_r_sub=2.2, name="pexp", n_probe_spacing=3)
    assert "pexp" not in sim._msl_auto_probe_spacing
    (r,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), _grid())
    assert r.n_probe_spacing == 3


def test_reflector_geometry_spacing_stays_byte_identical():
    """The Sheen interval geometry has only 21 slack cells; half of it
    (10) cannot fit even one widened probe step, so the resolved spacing
    equals the registration default and the #469 midpoint is unchanged
    (this is the pre-#681 byte-identity claim, asserted directly)."""
    sim = _sim_with_feed_and_patch()
    (r,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), _grid())
    assert r.n_probe_spacing == sim._msl_ports[0].n_probe_spacing == 2
    assert r.n_probe_offset == 26


def test_graded_propagation_axis_keeps_registration_spacing():
    """Resolver-skip path: spacing must stay the conservative stored
    default (a widened value could overrun a feed the solve cannot
    measure)."""
    sim = _nu_sim("+x", **_profiles("+x", prop_ratio=1.5))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        (r,) = _resolve_msl_auto_offsets(
            sim, list(sim._msl_ports),
            sim._build_nonuniform_grid(),
        )
    assert r.n_probe_spacing == sim._msl_ports[0].n_probe_spacing
