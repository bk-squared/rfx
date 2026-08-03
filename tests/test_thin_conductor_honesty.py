"""What `add_thin_conductor` actually models, pinned (issue #504).

The API accepts `thickness` and `eps_r` for every conductor but reads neither
when the shape is routed to PEC — which is every real metal, because the
routing predicate is `sigma_bulk >= 1e6` alone. These tests pin the behaviour
AND the honesty of the message about it, because the previous message stated
a false inequality: at `sigma_bulk=3.5e7, thickness=17e-6, dx=1e-3` it printed
"sigma_eff=5.95e+05 S/m exceeds PEC threshold (1e6)" — 5.95e5 does not exceed
1e6, and the conductor was routed to PEC regardless, on a predicate the
message never named.
"""

import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.materials.thin_conductor import _PEC_SIGMA_THRESHOLD, ThinConductor


def _sim(dx=1e-3):
    return Simulation(freq_max=10e9, domain=(6e-3, 6e-3, 3e-3), dx=dx)


def _sheet():
    return Box((1e-3, 1e-3, 1e-3), (5e-3, 5e-3, 1e-3))


def _warn_for(**kw):
    sim = _sim()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim.add_thin_conductor(_sheet(), **kw)
    msgs = [str(w.message) for w in rec if "add_thin_conductor" in str(w.message)]
    return msgs


# ---------------------------------------------------------------------------
# The behaviour being disclosed
# ---------------------------------------------------------------------------

def test_thickness_does_not_affect_a_metal_sheet():
    """Six orders of magnitude of `thickness` produce the SAME PEC mask.

    This is the fact the warning exists to disclose. If a future change makes
    thickness load-bearing on the PEC path, this test fails and the warning
    must be rewritten rather than silently left in place.

    Deliberately NOT parametrized. The first version of this test was, and
    compared each case against a `ref` initialised inside the same call — so
    every case compared itself to itself and the test could not fail. A
    reviewer proved it by making the mask grow with thickness; all five cases
    still passed. The comparison has to happen ACROSS thicknesses, in one
    call, which is what this does.
    """
    masks = {}
    for thickness in (1e-9, 17e-6, 35e-6, 68e-6, 1e-3):
        sim = _sim()
        sim.add_thin_conductor(_sheet(), sigma_bulk=5.8e7, thickness=thickness)
        mask = np.asarray(sim._assemble_materials(sim._build_grid())[3])
        assert mask.sum() > 0, f"sheet did not rasterise at t={thickness}"
        masks[thickness] = mask

    ref_t, ref = next(iter(masks.items()))
    for t, m in masks.items():
        assert np.array_equal(m, ref), (
            f"thickness became load-bearing on the PEC path: t={t} gives "
            f"{int(m.sum())} cells on z-layers "
            f"{np.unique(np.nonzero(m)[2]).tolist()}, but t={ref_t} gives "
            f"{int(ref.sum())} on {np.unique(np.nonzero(ref)[2]).tolist()}. "
            "The warning text now understates what the parameter does."
        )


def test_every_real_metal_is_on_the_pec_side():
    """The threshold sits below every metal, so 'lower sigma_bulk' is not a knob."""
    shape = _sheet()
    for sigma in (5.8e7, 3.5e7, 4.1e7, 1.4e6):      # Cu, Al, Ag, stainless
        assert ThinConductor(shape=shape, sigma_bulk=sigma, thickness=35e-6).is_pec
    for sigma in (9.9e5, 1e5, 1e4):                  # below threshold only
        assert not ThinConductor(shape=shape, sigma_bulk=sigma, thickness=35e-6).is_pec


def test_lossy_path_still_depends_on_thickness():
    """The one place `thickness` is load-bearing must not regress.

    Below the threshold the sheet becomes a conductivity, `sigma_eff =
    sigma_bulk * thickness / dx`, so doubling the thickness doubles it.
    """
    from rfx.materials.thin_conductor import apply_thin_conductor
    from rfx.core.yee import MaterialArrays

    sim = _sim()
    grid = sim._build_grid()
    shape = _sheet()
    ones = np.ones(grid.shape, dtype=np.float32)
    got = []
    for t in (35e-6, 70e-6):
        tc = ThinConductor(shape=shape, sigma_bulk=1e3, thickness=t)
        mats, _ = apply_thin_conductor(
            grid, tc,
            MaterialArrays(eps_r=ones.copy(), sigma=np.zeros_like(ones),
                           mu_r=ones.copy()),
            None,
        )
        got.append(float(np.asarray(mats.sigma).max()))
    assert got[0] > 0.0
    assert got[1] == pytest.approx(2.0 * got[0], rel=1e-9)


# ---------------------------------------------------------------------------
# The message about it — fire, clear, and never false
# ---------------------------------------------------------------------------

def test_warning_fires_for_a_metal():
    msgs = _warn_for(sigma_bulk=5.8e7, thickness=35e-6)
    assert len(msgs) == 1, msgs
    m = msgs[0]
    assert "LOSSLESS PEC sheet" in m
    assert "thickness is not used" in m


def test_warning_is_silent_below_the_threshold():
    """The lossy path honours both parameters, so it must not be warned about."""
    assert _warn_for(sigma_bulk=1e4, thickness=35e-6) == []


def test_warning_names_the_deciding_quantity_and_states_no_false_inequality():
    """Regression lock on the specific defect this change fixed.

    The old message printed `sigma_bulk*thickness/dx` and asserted it exceeded
    1e6. On these very inputs that product is 5.95e5, so the sentence was
    false while the routing decision (on `sigma_bulk`) was correct. The
    message must quote the quantity it is actually testing, and every
    inequality it states must hold.
    """
    sb, t, dx = 3.5e7, 17e-6, 1e-3
    sigma_eff = sb * t / dx
    assert sigma_eff < _PEC_SIGMA_THRESHOLD, "fixture no longer exercises the trap"

    msgs = _warn_for(sigma_bulk=sb, thickness=t)
    assert len(msgs) == 1
    m = msgs[0]
    # the deciding quantity is named with its real value
    assert f"sigma_bulk={sb:.2e}" in m
    # and the misleading derived quantity is not presented as the test
    assert "sigma_eff" not in m
    assert f"{sigma_eff:.2e}" not in m
    # the stated relation is the true one
    assert "at or above" in m and sb >= _PEC_SIGMA_THRESHOLD


def test_warning_does_not_invent_an_effective_thickness():
    """It must not report a cell-derived thickness as if it were the model.

    A one-cell PEC layer is a SURFACE (rfx/boundaries/pec.py zeroes tangential
    E only where the mask has a neighbour on that axis), so quoting
    "dx thick = N oz" would replace a silent falsehood with a loud one.
    """
    m = _warn_for(sigma_bulk=5.8e7, thickness=35e-6)[0]
    for forbidden in ("oz", "effective thickness", "modelled thickness"):
        assert forbidden not in m.lower(), forbidden


def test_warning_does_not_offer_advice_that_does_not_work():
    """'Use a lower sigma_bulk' alone would send a user to a different metal.

    The message may mention it, but only while saying it changes the material
    rather than the copper thickness.
    """
    m = _warn_for(sigma_bulk=5.8e7, thickness=35e-6)[0]
    if "lower sigma_bulk" in m:
        assert "different material" in m


def test_warning_stays_short_enough_to_read():
    """Length is part of usability, and the LONG branch is the one to guard.

    The message this replaced was 526 characters, long enough to bury its own
    first sentence — the advisory-burial problem (#470). But the first version
    of this check only rendered the DEFAULT-thickness branch (399 chars) while
    the branch nearest the limit is the one that echoes a caller's
    non-default thickness back at them. A reviewer caught that the assertion
    was not guarding the case it was meant to.

    Limit derived from measurement, not picked: the long branch measures 428
    characters today, so 460 leaves ~7% headroom while staying far below the
    526 that caused the complaint.
    """
    default_branch = _warn_for(sigma_bulk=5.8e7, thickness=35e-6)[0]
    long_branch = _warn_for(sigma_bulk=5.8e7, thickness=9.99e-6)[0]

    assert "You passed thickness=" in long_branch, (
        "the long branch no longer echoes the caller's thickness — this test "
        "is measuring the wrong branch again"
    )
    assert len(long_branch) > len(default_branch), "branches did not diverge"
    for m in (default_branch, long_branch):
        assert len(m) < 460, f"message grew to {len(m)} chars (was 526 once)"


def test_preflight_hint_does_not_recommend_a_no_op():
    """The mesh-resolution hint must not send a user on a pointless detour.

    Measured: for a PEC material a sub-cell `Box` and `add_thin_conductor()`
    on the same footprint produce a BIT-IDENTICAL `pec_mask`, so advising the
    swap changed nothing. This pins both halves — the equivalence, and the
    hint no longer claiming otherwise.
    """
    def _mask(use_thin):
        sim = _sim()
        box = Box((1e-3, 1e-3, 1e-3),
                  (5e-3, 5e-3, 1e-3 + (0.0 if use_thin else 35e-6)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if use_thin:
                sim.add_thin_conductor(box, thickness=35e-6)
            else:
                sim.add(box, material="pec")
        return np.asarray(sim._assemble_materials(sim._build_grid())[3])

    assert np.array_equal(_mask(False), _mask(True)), (
        "a sub-cell PEC Box and add_thin_conductor no longer agree — the "
        "hint's rationale changed and its wording must be revisited"
    )

    sim = _sim()
    sim.add(Box((1e-3, 1e-3, 1e-3), (5e-3, 5e-3, 1e-3 + 35e-6)), material="pec")
    # preflight() RETURNS the advisory messages (imitating
    # tests/test_msl_port_preflight.py::_msl_warnings), it does not warn.
    hints = [m for m in sim.preflight() if "below 1 cell resolution" in m]
    assert hints, "the sub-cell resolution advisory stopped firing"
    joined = " ".join(hints)
    assert "Use add_thin_conductor() for sub-cell PEC sheet" not in joined
    assert "one-cell PEC surface" in joined
