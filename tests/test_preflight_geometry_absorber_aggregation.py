"""Issue #660 — ``_validate_cfg_geometry_in_cpml`` reporting contract.

The pre-#660 check emitted one warning per geometry entry whose message
carried only the material name and the axis. Measured on a fixture
reproducing the reported CAD import (61 solids, 56 sharing one material
name): **61 warnings, 56 byte-identical**, and the overshoot distance —
already computed as ``c2[ax] - hi_b`` at the warn site — was discarded, so
a one-cell rounding overshoot and an 11mm coordinate-origin error printed
the same text.

This file pins the two things that fixed it:

1. **Aggregation** — one warning per crossed AXIS, naming the entry count
   and the worst offender, instead of N copies. Same fixture: 61 -> 1.
2. **The distance** — the overshoot, the crossed boundary side, the
   boundary coordinate and the offending bbox face are all in the message.

Plus the two controls that matter more than either: a legitimate SINGLE
shape overshoot must still warn, and geometry entirely inside the domain
must still warn not at all. A check that stopped firing would be worse
than a noisy one.

The ``code="geometry_in_absorber"`` slug is unchanged (it is the only
machine-readable key on this finding; grep confirms this file and the
check site are its only referents), and the message keeps the
``"extends into CPML region along <axis>-axis"`` substring that
``test_preflight_false_positives.py``, ``test_periodic_cpml.py`` and
``test_preflight_absorber_frame.py`` match on.

Frame reminder (issue #500, see ``_absorber_boundary_for_axis``): the
absorber is padded EXTERIOR to the requested domain, so the hi-side
boundary sits at ``domain_extent`` and ``[0, domain_extent]`` is
absorber-free. Every "crossing" fixture below therefore puts a bbox face
at a coordinate genuinely past ``domain_extent`` (or below 0), not merely
close to the edge.
"""

from __future__ import annotations

import pytest

from rfx import Box, Simulation

LX, LY, LZ = 0.030, 0.030, 0.004
DX = 0.5e-3
CODE = "geometry_in_absorber"


def _sim() -> Simulation:
    sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=DX,
                     cpml_layers=8, boundary="cpml")
    sim.add_material("metal", eps_r=1.0)
    sim.add_material("ro4003c", eps_r=3.55)
    return sim


def _absorber_issues(sim: Simulation) -> list:
    """Every ``geometry_in_absorber`` finding, as structured issues."""
    sim.add_source((LX / 2, LY / 2, LZ / 2), "ez")
    return [i for i in sim.preflight(strict=False)
            if getattr(i, "code", None) == CODE]


def _x_overshoot_box(overshoot: float, y0: float = 0.001) -> Box:
    """A box whose hi-x face sits ``overshoot`` past the x-hi absorber
    boundary (which is at ``LX``, the requested domain extent)."""
    hi = LX + overshoot
    return Box((hi - 0.002, y0, 0.001), (hi, y0 + 0.002, 0.002))


# --------------------------------------------------------------------- #
# 1. Aggregation: N entries -> one warning per axis.
# --------------------------------------------------------------------- #

def test_issue660_sixty_one_entries_collapse_to_one_warning():
    """The reported CAD-import shape: 61 solids displaced past x-hi, 56 of
    them sharing the material name 'metal'. Pre-#660 this emitted 61
    warnings (56 byte-identical); it must now emit exactly one."""
    sim = _sim()
    for i in range(56):
        sim.add(_x_overshoot_box(0.013 + i * 1e-5), material="metal")
    for i in range(5):
        sim.add(_x_overshoot_box(0.013 + i * 1e-5, y0=0.005),
                material="ro4003c")

    issues = _absorber_issues(sim)
    assert len(issues) == 1, (
        f"61 crossing entries on one axis must aggregate to ONE warning; "
        f"got {len(issues)}:\n" + "\n".join(str(i) for i in issues[:5])
    )
    msg = str(issues[0])
    assert "61 geometry entries cross the x-axis absorber" in msg, (
        f"the aggregate must name the entry count; got: {msg}"
    )


def test_issue660_worst_offender_is_the_deepest_not_the_first():
    """The one entry the aggregate names must be the deepest crossing —
    that is the one that distinguishes a rounding artefact from a
    misplaced model. Entry #1 here overshoots 20x further than #0."""
    sim = _sim()
    sim.add(_x_overshoot_box(0.0005), material="metal")          # #0
    sim.add(_x_overshoot_box(0.011, y0=0.005), material="metal")  # #1
    sim.add(_x_overshoot_box(0.001, y0=0.009), material="metal")  # #2

    issues = _absorber_issues(sim)
    assert len(issues) == 1, f"expected one aggregated warning, got {issues!r}"
    msg = str(issues[0])
    assert "geometry entry #1" in msg, (
        f"the deepest crossing (#1, 11mm) must be the one named; got: {msg}"
    )
    assert "11mm past" in msg, f"worst overshoot must be quoted; got: {msg}"
    assert "overshoot 500µm to 11mm" in msg, (
        f"the aggregate must give the overshoot RANGE so a uniform "
        f"displacement is distinguishable from a scatter; got: {msg}"
    )


def test_issue660_loc_carries_per_entry_index_face_and_overshoot():
    """Per-entry detail moves to the structured finding, not to N warning
    lines. ``loc`` must therefore carry more than the count: every crossing
    entry's index, crossed face and its own overshoot — so nothing a
    pre-#660 reader could have learned from N lines is lost."""
    sim = _sim()
    sim.add(_x_overshoot_box(0.002), material="metal")
    sim.add(_x_overshoot_box(0.011, y0=0.005), material="metal")
    sim.add(Box((0.010, 0.010, -0.0008), (0.012, 0.012, 0.001)),
            material="ro4003c")

    issues = _absorber_issues(sim)
    assert len(issues) == 2, f"expected one warning per axis, got {issues!r}"
    by_axis = {("x" if " x-axis" in str(i) else "z"): i for i in issues}
    assert by_axis["x"].loc == "geometry[#0 hi 2mm,#1 hi 11mm]", (
        f"loc must give each entry's index, face and overshoot; "
        f"got {by_axis['x'].loc!r}"
    )
    assert by_axis["z"].loc == "geometry[#2 lo 800µm]", (
        f"lo-side loc must record the lo face; got {by_axis['z'].loc!r}"
    )


def test_issue660_separate_axes_get_separate_warnings():
    """Aggregation is per-axis, so an x crossing and a z crossing stay
    distinguishable — ``test_periodic_cpml.py`` matches on the axis token."""
    sim = _sim()
    sim.add(_x_overshoot_box(0.002), material="metal")
    # Straddles z=0: c1[2] is genuinely in the exterior z-lo absorber.
    sim.add(Box((0.010, 0.010, -0.0008), (0.012, 0.012, 0.001)),
            material="ro4003c")

    issues = _absorber_issues(sim)
    assert len(issues) == 2, f"expected one warning per axis, got {issues!r}"
    by_axis = {("x" if " x-axis" in str(i) else "z"): str(i) for i in issues}
    assert set(by_axis) == {"x", "z"}, f"axes not separated: {issues!r}"
    assert " z-axis" not in by_axis["x"] and " x-axis" not in by_axis["z"], (
        f"each per-axis warning must name only its own axis: {issues!r}"
    )


# --------------------------------------------------------------------- #
# 2. The distance.
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("overshoot,expected", [
    (0.011, "11mm past the x-hi absorber boundary at 30mm"),
    (0.0005, "500µm past the x-hi absorber boundary at 30mm"),
])
def test_issue660_message_states_overshoot_and_crossed_boundary(overshoot,
                                                               expected):
    """``c2[ax] - hi_b`` and ``hi_b`` are both in scope at the warn site;
    both must be printed, in the repo's unit-adaptive form."""
    sim = _sim()
    sim.add(_x_overshoot_box(overshoot), material="metal")

    issues = _absorber_issues(sim)
    assert len(issues) == 1, f"expected exactly one warning, got {issues!r}"
    msg = str(issues[0])
    assert expected in msg, f"expected {expected!r} in message; got: {msg}"
    assert "bbox hi face at" in msg, (
        f"the offending bbox face coordinate must be printed; got: {msg}"
    )


def test_issue660_lo_side_crossing_names_the_lo_boundary():
    """The lo-side branch reports ``lo_b - c1[ax]`` against the lo face."""
    sim = _sim()
    sim.add(Box((0.010, 0.010, -0.0008), (0.012, 0.012, 0.001)),
            material="ro4003c")

    issues = _absorber_issues(sim)
    assert len(issues) == 1, f"expected exactly one warning, got {issues!r}"
    msg = str(issues[0])
    assert "800µm past the z-lo absorber boundary at 0mm" in msg, (
        f"lo-side overshoot and boundary must be printed; got: {msg}"
    )
    assert "bbox lo face at -800µm" in msg, (
        f"the negative bbox lo face must be printed; got: {msg}"
    )


# --------------------------------------------------------------------- #
# 3. Controls — the check must not stop firing, and must not start.
# --------------------------------------------------------------------- #

def test_issue660_single_shape_overshoot_still_warns():
    """One legitimately misplaced shape still draws exactly one warning,
    and the aggregate clause is omitted when there is nothing to
    aggregate."""
    sim = _sim()
    sim.add(_x_overshoot_box(0.011), material="metal")

    issues = _absorber_issues(sim)
    assert len(issues) == 1, f"a single overshoot must still warn; got {issues!r}"
    msg = str(issues[0])
    assert "extends into CPML region along x-axis" in msg, (
        f"the substring other test files match on must survive; got: {msg}"
    )
    assert "geometry entries cross" not in msg, (
        f"no plural aggregate clause for a single entry; got: {msg}"
    )
    assert "issue #61" in msg, f"the physics explanation must survive; got: {msg}"


def test_issue660_geometry_fully_inside_the_domain_stays_silent():
    """Non-firing control. The absorber is padded EXTERIOR to the
    requested domain (#500), so 61 boxes packed against the domain edges
    are absorber-free and must produce zero warnings — an aggregation
    change must not turn a silent case into a firing one."""
    sim = _sim()
    for i in range(61):
        y0 = 0.001 + (i % 10) * 0.002
        sim.add(Box((LX - 0.002, y0, 0.0), (LX, y0 + 0.001, LZ)),
                material="metal")

    issues = _absorber_issues(sim)
    assert not issues, (
        f"interior geometry must draw no absorber warning; got {issues!r}"
    )
