"""Preflight — the absorber stage: CPML frame, geometry-in-absorber reporting,
dispersive pole at an absorbing face.

One file per preflight stage (tier 3b of the 2026-09 test-corpus
reorganisation, see ``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``).
Sections, each formerly its own file:

1. **Absorber-overlap validator frame (issue #500)** — was
   ``test_preflight_absorber_frame.py``. The validator family compared
   user-frame coordinates against an INTERIOR-CPML reading while every rfx
   grid builder pads the absorber EXTERIOR to the requested domain, so five
   ``_validate_cfg_*`` / ``_check_msl_port_geometry`` consumers of
   ``cpml_thick_lo`` / ``cpml_thick_hi`` false-fired on geometry nowhere
   near the absorber. This section (a) proves the exterior-padding ground
   truth directly off the real grid builders (uniform ``Grid`` and
   ``make_nonuniform_grid``) — the single fact ``_absorber_boundary_for_axis``
   encodes; (b) gives each consumer a firing test and a non-firing control;
   (c) regression-locks the exact PR #499 false positive (WR-90 ports inside
   a valid 90.678mm domain). Post-review additions: the distinct
   ``absorber_proximity`` advisory (H1/M3: a coordinate interior but within
   2 cells of an active absorber boundary) and the MSL consumer's explicit,
   separately-calibrated ``cpml_layers*dx`` buffer (MH2/M5: the 2026-05-04
   ledger rule ``LY >= W + 2*(2*h_sub + 8*dx)``, restored ON TOP OF the
   exterior-frame boundary). The manual mutation falsifier (swapping the
   lo/hi roles in ``_absorber_boundary_for_axis`` reds 22 tests, the
   NON-firing controls) is recorded in the pre-merge file's git history.
2. **``_validate_cfg_geometry_in_cpml`` reporting contract (issue #660)** —
   was ``test_preflight_geometry_absorber_aggregation.py``. Pre-#660 the
   check emitted one warning per geometry entry (61 warnings, 56
   byte-identical on the reported CAD import) and discarded the overshoot
   distance. Pinned: aggregation (one warning per crossed AXIS naming the
   entry count and the worst offender), the distance / crossed side /
   boundary coordinate / offending bbox face in the message, ``loc`` carrying
   every entry's index, face and overshoot; plus the two controls (a single
   overshoot still warns, interior geometry stays silent). The
   ``code="geometry_in_absorber"`` slug and the ``"extends into CPML region
   along <axis>-axis"`` substring other tests match on are unchanged.
3. **Dispersive pole material touching a CPML face (issues #636/#808)** —
   was ``test_preflight_dispersive_pole_at_absorber.py``. The shipped pad
   extension replicates only the statics (#627a); pole masks are not
   replicated (#636 factorial measured naive re-addition divergent, Drude
   divergent even under the CFS alpha rule); since #808 a pole-carrying
   column's statics are not promoted by the hi-face fallback either. The
   advisory ``dispersive_pole_at_absorber_face`` flags EVERY pole family
   touching a face that carries an absorber (#808 broadened it from
   {high-Q in-band Lorentz, Drude}: the quiet Debye configuration silently
   moved a committed recovery observable past its gate). Inset structures
   and non-absorbing boundaries stay quiet. Module constants of this
   section carry a ``DP_`` prefix (``DP_DX``, ``_dp_sim``) to coexist with
   section 2's ``DX`` / ``_sim``; values are unchanged.

Every assertion, tolerance, fixture value and parametrisation of the
absorbed files is kept verbatim.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from rfx import Simulation, Box
from rfx.api._preflight import (
    _PreflightMixin,
    _absorber_boundary_for_axis,
    _coord_in_absorber,
    _coord_near_absorber,
    PreflightConfigError,
)
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.grid import Grid
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole, drude_pole


# ===========================================================================
# formerly tests/unit/preflight/test_preflight_absorber.py
# ===========================================================================

# --------------------------------------------------------------------- #
# 1. Ground truth: CPML pads EXTERIOR to the requested domain.
# --------------------------------------------------------------------- #

def test_uniform_grid_pads_absorber_exterior_to_requested_domain():
    """rfx.grid.Grid — reproduces issue #500's own repro-1 numbers
    (WR-90 stage-S3: dx=254um, cpml_layers=110, x-extent 90.678mm)."""
    dx = 254e-6
    domain = (90.678e-3, 22.86e-3, 10.16e-3)
    cpml_layers = 110
    g = Grid(freq_max=10e9, domain=domain, dx=dx, cpml_layers=cpml_layers)

    assert (g.nx, g.pad_x_lo, g.pad_x_hi) == (578, 110, 110)

    # Grid.position_to_index maps node pad_x_lo to user x=0 (i = round(pos/dx)
    # + pad_x_lo), so node 0's user coordinate is -pad_x_lo*dx and node
    # nx-1's is (nx-1-pad_x_lo)*dx.
    x_node0 = (0 - g.pad_x_lo) * dx
    x_node_last = (g.nx - 1 - g.pad_x_lo) * dx
    assert x_node0 == pytest.approx(-27.940e-3, abs=1e-6)
    assert x_node_last == pytest.approx(118.618e-3, abs=1e-6)

    # The requested domain [0, 90.678mm] sits strictly inside the grid's
    # user-coordinate span -> it is absorber-free; the absorber occupies
    # the EXTERIOR bands [x_node0, 0) and (domain[0], x_node_last].
    assert x_node0 < 0.0 < domain[0] < x_node_last


def test_nonuniform_grid_pads_absorber_exterior_to_requested_domain():
    """rfx.nonuniform.make_nonuniform_grid — the dz_profile-covered
    physical z-domain is likewise absorber-free; CPML is appended
    exterior to it via ``_pad_profile``, same convention as the uniform
    lane above."""
    from rfx.nonuniform import make_nonuniform_grid

    dz_profile = np.array([0.5e-3] * 20 + [1.0e-3] * 10)  # 20mm physical z
    cpml_layers = 8
    ng = make_nonuniform_grid((0.03, 0.03), dz_profile, 0.5e-3, cpml_layers=cpml_layers)

    assert (ng.pad_z_lo, ng.pad_z_hi) == (cpml_layers, cpml_layers)
    # N cells are bounded by N+1 nodes (#562) — the outermost node is the
    # absorber's outer face, which is why the interior extent below is exact.
    assert ng.nz == len(dz_profile) + ng.pad_z_lo + ng.pad_z_hi + 1

    dz_full = np.asarray(ng.dz)
    edges = np.concatenate(([0.0], np.cumsum(dz_full)))
    # Node pad_z_lo is user z=0 (mirrors Grid's node pad_x_lo convention).
    # The LAST node is edges[nz-1], not edges[-1]: the cell array carries one
    # trailing duplicate whose only job is to bound the last real cell with a
    # node (#562), and whose own H term the stencil zeroes — it is not physical
    # extent, so extents must be read from the node positions, never from
    # sum(dz).
    z_node0 = edges[0] - edges[ng.pad_z_lo]
    z_node_last = edges[ng.nz - 1] - edges[ng.pad_z_lo]
    physical_extent = float(np.sum(dz_profile))

    # float32 storage (NonUniformGrid.dz) — loosen tolerance accordingly.
    assert z_node0 == pytest.approx(-4.0e-3, abs=1e-6)
    assert z_node_last == pytest.approx(28.0e-3, abs=1e-6)
    assert z_node0 < 0.0 < physical_extent < z_node_last


def test_absorber_boundary_helper_matches_ground_truth():
    """Unit check on the canonical helper itself: active boundary is
    always exactly 0.0 / domain_extent, never offset inward by the
    thickness (the pre-#500 bug), and inactive sides return None."""
    lo, hi = _absorber_boundary_for_axis(0.090678, 27.94e-3, 27.94e-3)
    assert lo == 0.0
    assert hi == 0.090678
    # No active absorber on either side (e.g. PEC/PMC/periodic face).
    assert _absorber_boundary_for_axis(0.090678, 0.0, 0.0) == (None, None)
    # Membership: interior coordinates are never "in" the absorber.
    assert not _coord_in_absorber(0.020, 0.090678, 27.94e-3, 27.94e-3)
    assert not _coord_in_absorber(0.070678, 0.090678, 27.94e-3, 27.94e-3)
    # Exterior coordinates are.
    assert _coord_in_absorber(-0.001, 0.090678, 27.94e-3, 27.94e-3)
    assert _coord_in_absorber(0.091, 0.090678, 27.94e-3, 27.94e-3)


def test_last_interior_node_reads_as_overlap_not_proximity_h1_conservatism():
    """Issue #510 nit 3: the docstring's own example (domain_extent=0.0101,
    dx=1e-3 -> ceil(10.1)=11 -> the true last interior node sits at 0.011,
    one cell BEYOND the nominal domain_extent, yet the real grid still
    treats it as interior) reads as ``absorber_overlap`` here — the more
    SEVERE membership finding — rather than the lower-severity
    ``absorber_proximity`` a genuinely-interior placement gets. This is the
    user-visible face of ``_absorber_boundary_for_axis``'s documented
    "conservative by up to one cell" hi-side design (see its docstring):
    the boundary is deliberately allowed to read slightly early rather
    than risk missing a genuine overlap. Pinned here rather than "fixed"
    — every other absorber_overlap/absorber_proximity consumer relies on
    this exact membership frame (see the module docstring's mutation
    falsifier list), so loosening it to reclassify this one case as
    proximity would change shared semantics, not just this corner case."""
    dx = 1e-3
    domain_extent = 0.0101
    cpml_layers = 2
    ct_hi = cpml_layers * dx
    g = Grid(freq_max=10e9, domain=(domain_extent, 0.01, 0.01), dx=dx,
             cpml_layers=cpml_layers)
    # Issue #510 review (non-blocking #7): derive the true last interior
    # node from rfx.grid.Grid's OWN reported sizing (like the rest of
    # this file's ground-truth tests do), not a hand-transcribed number.
    # nx = ceil(domain/dx) + 1 + pad_lo + pad_hi, so
    # ceil(domain/dx) = nx - 1 - pad_lo - pad_hi is the interior cell
    # count, and that many cells beyond user x=0 is the last interior
    # node's user coordinate.
    n_interior_cells = g.nx - 1 - g.pad_x_lo - g.pad_x_hi
    true_last_interior_node = n_interior_cells * dx
    assert true_last_interior_node == pytest.approx(0.011)
    assert _coord_in_absorber(true_last_interior_node, domain_extent, 0.0, ct_hi)
    # _coord_near_absorber is not even reached by a caller once membership
    # fires (callers check overlap first, per both helpers' docstrings),
    # but confirm directly it would not independently classify this as
    # proximity either — the two really are mutually exclusive here.
    assert not _coord_near_absorber(
        true_last_interior_node, domain_extent, 0.0, ct_hi, dx
    )


# --------------------------------------------------------------------- #
# 2. Consumer 1 — _validate_cfg_absorber_placement (probe/port position).
# --------------------------------------------------------------------- #

def _absorber_placement_sim(z_probe: float) -> Simulation:
    # Issue #500 repro-2 domain: cpml_thickness (16mm) exceeds the whole
    # 10.16mm z-extent, so the pre-#500 interior-frame check flagged
    # EVERY z position, including the domain centre.
    sim = Simulation(freq_max=10e9, domain=(50e-3, 22.86e-3, 10.16e-3),
                     dx=1e-3, cpml_layers=16, boundary="cpml")
    sim.add_source((0.001, 0.01143, 0.005), "ez")
    sim.add_probe((0.025, 0.01143, z_probe), "ez")
    return sim


def test_absorber_placement_silent_on_domain_centre_probe():
    """Issue #500 repro-2, regression-locked: a probe at the exact
    geometric centre of the domain must not warn — CPML z-thickness
    (16mm) exceeding the 10.16mm z-extent used to make this fire
    unconditionally."""
    issues = _absorber_placement_sim(0.00508).preflight(strict=False)
    hits = [m for m in issues if "is near/inside" in m and "Probe" in m]
    assert not hits, f"domain-centre probe must not warn; got {hits}"


def test_absorber_placement_fires_on_probe_genuinely_in_absorber():
    """Positive control: a probe at a genuinely negative z (past the
    z=0 domain edge, inside the exterior absorber) must still warn."""
    issues = _absorber_placement_sim(-0.001).preflight(strict=False)
    hits = [m for m in issues if "is near/inside" in m and "Probe" in m]
    assert hits, f"probe in the exterior absorber must warn; got {issues!r}"


def test_absorber_placement_proximity_advisory_fires_within_2_cells():
    """Review finding H1: a probe genuinely INSIDE the domain but within
    _ABSORBER_PROXIMITY_CELLS=2 cells (dx=1mm -> 2mm margin here) of the
    z=0 boundary gets the distinct absorber_proximity advisory, not
    silence and not the absorber_overlap membership warning. This is the
    coverage tests/unit/preflight/test_run_preflight_parity.py and
    tests/unit/sparams/test_msl_internal_probe_advisories.py's #470 lock both turned
    out to depend on."""
    issues = _absorber_placement_sim(0.0005).preflight(strict=False)
    overlap = [m for m in issues if "is near/inside" in m and "Probe" in m]
    proximity = [
        m for m in issues
        if "Probe" in m and "within 2 cells" in m and "CPML absorber" in m
    ]
    assert not overlap, f"a probe within 2mm of the edge is not IN the absorber; got {overlap}"
    assert proximity, f"expected a proximity advisory for a probe 0.5mm from the edge; got {issues!r}"


def test_absorber_placement_silent_past_the_proximity_margin():
    """Non-firing control: a probe 3mm from the z=0 edge (past the 2mm /
    2-cell proximity margin) draws neither the overlap nor the proximity
    advisory."""
    issues = _absorber_placement_sim(0.003).preflight(strict=False)
    hits = [m for m in issues if "Probe" in m and ("near/inside" in m or "within 2 cells" in m)]
    assert not hits, f"a probe 3mm from the edge must draw neither advisory; got {hits}"


# --------------------------------------------------------------------- #
# 3. Consumer 2 — _validate_cfg_geometry_in_cpml (geometry bounding box).
# --------------------------------------------------------------------- #

def _geometry_in_cpml_sim(c1z: float) -> Simulation:
    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
                     cpml_layers=4, boundary="cpml")
    sim.add_material("diel", eps_r=2.0)
    sim.add(Box((0.001, 0.001, c1z), (0.009, 0.009, 0.009)), material="diel")
    sim.add_source((0.005, 0.005, 0.005), "ez")
    return sim


def test_geometry_in_cpml_silent_when_entirely_interior():
    """A Box entirely within [0, domain_extent] can never touch the
    exterior absorber, however close to the edge it sits."""
    issues = _geometry_in_cpml_sim(0.001).preflight(strict=False)
    hits = [m for m in issues if "extends into CPML" in m]
    assert not hits, f"interior Box must not warn; got {hits}"


def test_geometry_in_cpml_fires_when_bbox_crosses_domain_edge():
    """Positive control: a Box whose low z-edge is genuinely negative
    (crosses z=0 into the exterior absorber) must still warn (issue
    #61's original footgun)."""
    issues = _geometry_in_cpml_sim(-0.001).preflight(strict=False)
    hits = [m for m in issues if "extends into CPML" in m]
    assert hits, f"Box crossing into the absorber must warn; got {issues!r}"


# --------------------------------------------------------------------- #
# 4. Consumer 3 — _validate_cfg_ntff_absorber_overlap (NTFF box corners).
# --------------------------------------------------------------------- #

def _ntff_sim(corner_lo_z: float) -> Simulation:
    sim = Simulation(freq_max=10e9, domain=(0.06, 0.06, 0.06), dx=2e-3,
                     cpml_layers=8, boundary="cpml")
    sim.add_source((0.03, 0.03, 0.03), "ez")
    sim.add_ntff_box((0.01, 0.01, corner_lo_z), (0.05, 0.05, 0.05), freqs=(10e9,))
    return sim


def test_ntff_absorber_overlap_silent_when_box_interior():
    issues = _ntff_sim(0.01).preflight(strict=False)
    hits = [m for m in issues if "NTFF box extends" in m]
    assert not hits, f"interior NTFF box must not warn; got {hits}"


def test_ntff_absorber_overlap_fires_when_corner_crosses_domain_edge():
    """Positive control: lo-corner z genuinely negative (5mm past the
    z=0 edge, inside the 16mm-thick absorber) must still warn."""
    issues = _ntff_sim(-0.005).preflight(strict=False)
    hits = [m for m in issues if "NTFF box extends" in m]
    assert hits, f"NTFF box crossing into the absorber must warn; got {issues!r}"


# --------------------------------------------------------------------- #
# 5. Consumer 4 — _validate_cfg_waveguide_reference_plane.
#
# This is the exact check issue #500 repro-1 documents. Its absorber-
# overlap branch is now provably UNREACHABLE through the public API: the
# hard bounds check immediately above it (`effective < 0 or
# effective > domain_ext`) already raises PreflightConfigError for any
# `effective` this branch could otherwise catch (their thresholds are
# now identical — both 0.0 / domain_ext under the corrected frame), and
# add_waveguide_port() itself rejects an out-of-domain x_position /
# reference_plane before preflight ever runs. Exercised at the mixin
# level (SimpleNamespace fake `self`, matching this file's existing
# _validate_cfg_conformal_fine_dx pattern) so the unreachable branch is
# still under direct test.
# --------------------------------------------------------------------- #

def _fake_wg_port(direction: str, x_position: float, reference_plane=None):
    return SimpleNamespace(direction=direction, x_position=x_position,
                           reference_plane=reference_plane)


def test_waveguide_reference_plane_silent_on_wr90_ports_in_valid_domain():
    """Regression lock for issue #500 repro-1 / PR #499 (the stage-S3
    iris filter review that triggered #500): WR-90 ports at 20mm and
    70.678mm on a 90.678mm domain (dx=254um, cpml_layers=110) are
    comfortably interior and must not warn — this exact configuration
    used to emit two 'inside the CPML absorbing region' false positives
    (also visible, pre-fix, on every run of test_waveguide_forward.py)."""
    domain = (90.678e-3, 22.86e-3, 10.16e-3)
    dx = 254e-6
    sim = Simulation(freq_max=10e9, domain=domain, dx=dx, cpml_layers=110,
                     boundary="cpml")
    sim.add_waveguide_port(0.020, direction="+x", name="p1")
    sim.add_waveguide_port(0.070678, direction="-x", name="p2")
    issues = sim.preflight(strict=False)
    hits = [m for m in issues if "inside the CPML absorbing region" in m]
    assert not hits, f"WR-90 ports in a valid domain must not warn; got {hits}"


def test_waveguide_reference_plane_absorber_branch_is_dead_given_hard_check():
    """Direct mixin-level check: an `effective` genuinely outside
    [0, domain_ext] raises PreflightConfigError from the hard check
    BEFORE the absorber-overlap warning branch (now equally at 0.0 /
    domain_ext) could ever fire."""
    domain = (90.678e-3, 22.86e-3, 10.16e-3)
    ct_lo = [110 * 254e-6, 0.0, 0.0]
    ct_hi = [110 * 254e-6, 0.0, 0.0]
    fake = SimpleNamespace(
        _domain=domain, _geometry=[],
        _waveguide_ports=[_fake_wg_port("+x", 0.0, reference_plane=-0.005)],
    )
    with pytest.raises(PreflightConfigError, match="outside the x-domain"):
        _PreflightMixin._validate_cfg_waveguide_reference_plane(fake, warnings, ct_lo, ct_hi)


def test_waveguide_reference_plane_silent_at_mixin_level_near_edge():
    """Mixin-level non-firing control mirroring the repro-1 numbers
    directly (no add_waveguide_port() guard in the way)."""
    domain = (90.678e-3, 22.86e-3, 10.16e-3)
    ct_lo = [110 * 254e-6, 0.0, 0.0]
    ct_hi = [110 * 254e-6, 0.0, 0.0]
    fake = SimpleNamespace(
        _domain=domain, _geometry=[],
        _waveguide_ports=[_fake_wg_port("+x", 0.020), _fake_wg_port("-x", 0.070678)],
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _PreflightMixin._validate_cfg_waveguide_reference_plane(fake, warnings, ct_lo, ct_hi)
    assert not rec, f"ports comfortably inside the domain must not warn; got {rec}"


# --------------------------------------------------------------------- #
# 6. Consumer 5 — _check_msl_port_geometry (checks 1 & 3: distance to
#    the nearest absorbing/reflecting boundary).
#
# Review finding MH2: unlike consumers 1-4, this one is NOT a plain
# instance of the #500 interior-frame bug. The reference position is the
# exterior-frame domain edge (_absorber_boundary_for_axis) PLUS an
# EXPLICIT, separately-calibrated buffer (cpml_thick_{lo,hi} = n_cpml*dx)
# — docs/agent-memory/rfx-known-issues.md "Status 2026-05-04 (CALIBRATED,
# OpenEMS-class)": with LY = W + 6*dx at dx=80um/cpml_layers=8 "the trace
# ended up INSIDE the CPML overlap region (negative clearance)" and Z0
# drifted UP with refinement instead of converging to Hammerstad's
# 47.89 Ohm; the fix requires LY >= W + 2*(2*h_sub + 8*dx). An earlier
# pass of this PR dropped the buffer entirely (treating this consumer as
# the same bug class as the other four), which silently re-admitted that
# exact negative-clearance configuration — verified below against the
# ledger's own numbers.
# --------------------------------------------------------------------- #

_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
_LX = 14e-3


def _msl_x_clearance_sim(dx: float, port_x: float, cpml_layers: int = 8) -> Simulation:
    ly = _W_TRACE + 8 * _H_SUB
    sim = Simulation(
        freq_max=5e9, domain=(_LX, ly, _H_SUB + 1.5e-3), dx=dx, cpml_layers=cpml_layers,
        boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=_EPS_R)
    sim.add(Box((0, 0, 0), (_LX, ly, _H_SUB)), material="ro4350b")
    y_c = ly / 2.0
    sim.add(Box((0, y_c - _W_TRACE / 2, _H_SUB), (_LX, y_c + _W_TRACE / 2, _H_SUB + dx)),
            material="pec")
    sim.add_msl_port(position=(port_x, y_c, 0), width=_W_TRACE, height=_H_SUB,
                     direction="+x", impedance=50.0)
    return sim


def _msl_y_clearance_sim(dx: float, ly: float, cpml_layers: int = 8) -> Simulation:
    """Mirrors docs/agent-memory/rfx-known-issues.md's 2026-05-04 mesh-conv
    fixture (LY parametrized, port_x fixed comfortably clear of x-CPML)."""
    sim = Simulation(
        freq_max=5e9, domain=(_LX, ly, _H_SUB + 1.5e-3), dx=dx, cpml_layers=cpml_layers,
        boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=_EPS_R)
    sim.add(Box((0, 0, 0), (_LX, ly, _H_SUB)), material="ro4350b")
    y_c = ly / 2.0
    sim.add(Box((0, y_c - _W_TRACE / 2, _H_SUB), (_LX, y_c + _W_TRACE / 2, _H_SUB + dx)),
            material="pec")
    sim.add_msl_port(position=(2e-3, y_c, 0), width=_W_TRACE, height=_H_SUB,
                     direction="+x", impedance=50.0)
    return sim


def test_msl_x_cpml_clearance_fires_on_ledger_negative_clearance_case():
    """The exact 2026-05-04 ledger regression: dx=80um, cpml_layers=8 ->
    calibrated buffer 640um; port at x=600um is INSIDE the buffered
    margin (clearance = 600 - 640 = -40um < 0), the "trace ended up
    INSIDE the CPML overlap region" case the calibration fixed. Dropping
    the buffer (an earlier pass of this PR) silently went silent here."""
    issues = _msl_x_clearance_sim(dx=80e-6, port_x=0.6e-3).preflight(strict=False)
    hits = [m for m in issues if "x-CPML" in m]
    assert hits, f"port inside the calibrated buffer must warn; got {issues!r}"
    assert any("calibrated CPML buffer" in m for m in hits), hits


def test_msl_x_cpml_clearance_silent_once_past_buffer_plus_recommended():
    """Non-firing control: dx=100um, cpml_layers=8 -> buffer=800um,
    recommended=2*h_sub=508um, total=1308um. Port at x=1400um clears
    both -> silent."""
    issues = _msl_x_clearance_sim(dx=100e-6, port_x=1.4e-3).preflight(strict=False)
    hits = [m for m in issues if "x-CPML" in m]
    assert not hits, f"port past buffer+recommended must not warn; got {hits}"


def test_msl_x_cpml_clearance_fires_when_genuinely_too_close():
    """Positive control: port at x=300um is inside the buffer alone
    (dx=100um, cpml_layers=8 -> buffer=800um) — must warn."""
    issues = _msl_x_clearance_sim(dx=100e-6, port_x=0.3e-3).preflight(strict=False)
    hits = [m for m in issues if "x-CPML" in m]
    assert hits, f"port with real 300um clearance must warn; got {issues!r}"


def test_msl_y_clearance_fires_on_ledger_ly_w_plus_6dx():
    """Ledger reproduction: the PRE-calibration ``LY = W + 6*dx`` geometry
    (dx=80um, cpml_layers=8) that motivated the 2026-05-04 fix must still
    warn under the restored buffer."""
    ly = _W_TRACE + 6 * 80e-6
    issues = _msl_y_clearance_sim(dx=80e-6, ly=ly).preflight(strict=False)
    hits = [m for m in issues if "lateral clearance" in m]
    assert hits, f"pre-calibration LY=W+6dx must warn; got {issues!r}"
    assert any("calibrated CPML buffer" in m for m in hits), hits


def test_msl_y_clearance_silent_on_ledger_calibrated_ly():
    """Ledger reproduction: the CALIBRATED ``LY >= W + 2*(2*h_sub + 8*dx)``
    geometry must be silent."""
    ly = _W_TRACE + 2 * (2 * _H_SUB + 8 * 80e-6)
    issues = _msl_y_clearance_sim(dx=80e-6, ly=ly).preflight(strict=False)
    hits = [m for m in issues if "lateral clearance" in m]
    assert not hits, f"calibrated LY must not warn; got {hits}"


def test_msl_clearance_buffer_scales_with_cpml_layers_not_hardcoded_8():
    """MH2: the ledger's "8*dx" is that specific fixture's cpml_layers=8,
    not a hardcoded constant — doubling cpml_layers must double the
    buffer and therefore still fire at a clearance that would clear a
    fixed-800um buffer."""
    # dx=100um, cpml_layers=16 -> buffer=1600um. Port at x=1400um cleared
    # an 800um buffer (see the silent-control above) but not a 1600um one.
    issues = _msl_x_clearance_sim(dx=100e-6, port_x=1.4e-3, cpml_layers=16).preflight(
        strict=False
    )
    hits = [m for m in issues if "x-CPML" in m]
    assert hits, f"doubled cpml_layers must double the buffer and warn; got {issues!r}"
    assert any("1.6mm calibrated CPML buffer" in m for m in hits), hits


# ===========================================================================
# formerly tests/unit/preflight/test_preflight_absorber.py
# ===========================================================================

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


# ===========================================================================
# formerly tests/unit/preflight/test_preflight_absorber.py
# ===========================================================================

DP_DX = 1e-3
NA, NB, NZ = 20, 16, 10
F0 = 3e9
W0 = 2 * np.pi * F0


def _dp_sim(boundary="cpml"):
    return Simulation(freq_max=2.5 * F0, domain=(NA * DP_DX, NB * DP_DX, NZ * DP_DX),
                      dx=DP_DX, boundary=boundary, cpml_layers=8)


def _findings(sim):
    report = sim.preflight()
    return report.by_code("dispersive_pole_at_absorber_face")


def _touching_box():
    # touches x-lo, x-hi and y-lo; interior in z
    return Box((0.0, 0.0, 3 * DP_DX), (NA * DP_DX, 8 * DP_DX, 7 * DP_DX))


def _inset_box():
    # >= 2 cells of vacuum before every face
    return Box((3 * DP_DX, 3 * DP_DX, 3 * DP_DX), (10 * DP_DX, 8 * DP_DX, 7 * DP_DX))


def test_high_q_lorentz_touching_face_warns():
    sim = _dp_sim()
    sim.add_material("slab", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=W0, delta=W0 / 120.0,
                                                kappa=3.0 * W0 ** 2)])
    sim.add(_touching_box(), material="slab")
    found = _findings(sim)
    assert len(found) == 1, found
    issue = found[0]
    assert issue.severity == "warning"
    assert "Q=60" in str(issue)
    assert "x-lo" in issue.loc and "y-lo" in issue.loc
    assert "#636" in str(issue)


def test_drude_touching_face_warns():
    sim = _dp_sim()
    sim.add_material("metalish", eps_r=1.0,
                     lorentz_poles=[drude_pole(omega_p=W0, gamma=W0 / 100.0)])
    sim.add(_touching_box(), material="metalish")
    found = _findings(sim)
    assert len(found) == 1, found
    assert "Drude" in str(found[0])


def test_inset_structure_stays_quiet():
    sim = _dp_sim()
    sim.add_material("slab", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=W0, delta=W0 / 120.0,
                                                kappa=3.0 * W0 ** 2)])
    sim.add(_inset_box(), material="slab")
    assert _findings(sim) == []


def test_low_q_lorentz_touching_face_warns():
    """Inverted from stays-quiet by #808 (module docstring): the
    statics-without-pole pad state is a property of every pole family."""
    sim = _dp_sim()
    sim.add_material("lossy_pole", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=W0, delta=W0 / 4.0,
                                                kappa=3.0 * W0 ** 2)])  # Q=2
    sim.add(_touching_box(), material="lossy_pole")
    found = _findings(sim)
    assert len(found) == 1, found
    assert "Q=2" in str(found[0])
    assert "#808" in str(found[0])


def test_out_of_band_high_q_lorentz_touching_face_warns():
    """Inverted from stays-quiet by #808: out-of-band only removes the
    resonance-sharp mismatch, not the undeclared pad material."""
    sim = _dp_sim()
    w_hi = 2 * np.pi * 40e9  # far above 1.5 * 2*pi*freq_max (7.5e9 band)
    sim.add_material("uv_pole", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=w_hi,
                                                delta=w_hi / 120.0,
                                                kappa=0.5 * w_hi ** 2)])
    sim.add(_touching_box(), material="uv_pole")
    found = _findings(sim)
    assert len(found) == 1, found
    assert "out-of-band" in str(found[0])


def test_debye_touching_face_warns():
    """Inverted from stays-quiet by #808 — the #808 fixture IS a Debye
    face-toucher, and the regression it took (recovery 11% -> 32% error
    when the pad surround changed under it) is exactly what the original
    "warning on every lossy PCB substrate would be noise" scoping said
    could not matter. Written rationale:
    docs/design_notes/issue808_debye_pad_predeclaration.md."""
    sim = _dp_sim()
    sim.add_material("fr4ish", eps_r=4.0,
                     debye_poles=[DebyePole(delta_eps=0.4, tau=1e-11)])
    sim.add(_touching_box(), material="fr4ish")
    found = _findings(sim)
    assert len(found) == 1, found
    issue = found[0]
    assert issue.severity == "warning"
    assert "Debye" in str(issue)
    assert "#808" in str(issue)
    # exact-touch branch (this box ends AT x-hi): the background-pad
    # wording is the truthful one and must stay
    assert "the pad stays background" in str(issue)
    assert "drawn PAST the domain" not in str(issue)
    # the resonance-risk (#636 divergence-sharp) clause must NOT claim
    # this family is in the divergence class
    assert "divergence-risk class" not in str(issue)


def test_overdrawn_hi_face_names_the_realized_pad_truthfully():
    """A dispersive box drawn PAST a hi face rasterizes its own statics —
    and pole cells up to the overdraw depth — into the absorber (measured
    2026-09-01: the whole hi pad reads eps_r 4.0 and the first overdraw
    layers carry the pole mask). The advisory must say that, not the
    exact-touch branch's 'the pad stays background'."""
    sim = _dp_sim()
    sim.add_material("fr4ish", eps_r=4.0,
                     debye_poles=[DebyePole(delta_eps=0.4, tau=1e-11)])
    # drawn 3 cells PAST x-hi; clear of every other face
    sim.add(Box((10 * DP_DX, 3 * DP_DX, 3 * DP_DX),
                ((NA + 3) * DP_DX, 8 * DP_DX, 7 * DP_DX)), material="fr4ish")
    found = _findings(sim)
    assert len(found) == 1, found
    msg = str(found[0])
    assert "x-hi" in found[0].loc
    assert "drawn PAST the domain" in msg, msg
    assert "the pad carries the material's statics" in msg, msg
    assert "the pad stays background" not in msg, (
        "overdrawn hi face still claims a background pad — the realized "
        "pad there carries the statics and pole cells: " + msg)


def test_pec_boundary_stays_quiet():
    sim = _dp_sim(boundary="pec")
    sim.add_material("slab", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=W0, delta=W0 / 120.0,
                                                kappa=3.0 * W0 ** 2)])
    sim.add(_touching_box(), material="slab")
    assert _findings(sim) == []


def test_two_touching_entries_aggregate_into_one_finding():
    sim = _dp_sim()
    sim.add_material("slab", eps_r=4.0,
                     lorentz_poles=[LorentzPole(omega_0=W0, delta=W0 / 120.0,
                                                kappa=3.0 * W0 ** 2)])
    sim.add(Box((0.0, 0.0, 3 * DP_DX), (5 * DP_DX, 8 * DP_DX, 7 * DP_DX)),
            material="slab")  # x-lo
    sim.add(Box((15 * DP_DX, 0.0, 3 * DP_DX), (NA * DP_DX, 8 * DP_DX, 7 * DP_DX)),
            material="slab")  # x-hi
    found = _findings(sim)
    assert len(found) == 1, found
    assert "#0" in found[0].loc and "#1" in found[0].loc
