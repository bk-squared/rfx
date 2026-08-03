"""Issue #500 — the preflight absorber-overlap validator family compared
user-frame coordinates against an INTERIOR-CPML reading while every rfx
grid builder pads the absorber EXTERIOR to the requested domain. That made
five ``_validate_cfg_*`` / ``_check_msl_port_geometry`` consumers of
``cpml_thick_lo`` / ``cpml_thick_hi`` false-fire on geometry nowhere near
the absorber (verified: waveguide reference planes comfortably inside a
90.678mm domain, a probe at the domain centre), and the whole family had
NO tests at all before this file (the #303 mis-gated-guard class).

This file:

1. Proves the exterior-padding ground truth directly off the real grid
   builders (``rfx.grid.Grid`` for the uniform lane,
   ``rfx.nonuniform.make_nonuniform_grid`` for the non-uniform/dz_profile
   lane) — the single fact ``_absorber_boundary_for_axis``
   (``rfx/api/_preflight.py``) encodes and every consumer now goes
   through.
2. Gives each of the five consumers a firing test (geometry genuinely
   past the requested-domain edge, into the exterior absorber) and a
   non-firing control (geometry near an edge but still inside the
   requested domain — the shape the pre-#500 code false-fired on).
3. Regression-locks the exact false positive PR #499 (the stage-S3 iris
   filter review that triggered #500) hit: WR-90 waveguide ports
   comfortably inside a valid 90.678mm domain must not warn.

Non-uniform lane (issue #500 "Not verified" item): ``_validate_simulation_config``
(``rfx/api/_preflight.py``) calls all five consumers unconditionally —
there is no ``dz_profile`` gate on the whole check family, unlike the
unrelated #494/#502 two-port absorber advisory (a different, self-contained
check in ``compute_waveguide_s_matrix`` that PR #502 found silent on the NU
lane). ``_validate_cfg_compute_cpml_thickness`` already sums the leading
``dz_profile`` entries for the z-axis thickness, and
``make_nonuniform_grid`` pads CPML exterior to the ``dz_profile``-covered
physical domain exactly like the uniform ``Grid`` — proved below — so the
same frame and the same fix apply on both lanes; there is no separate NU
code path to special-case.

Mutation falsifier (manual, recorded here rather than left as a permanent
CI assertion): with ``_absorber_boundary_for_axis`` locally edited so
``lo_boundary = domain_extent if ct_lo > 0 else None`` (swapping the lo/hi
roles), every firing test in this file went RED — each one no longer
detected its geometry as "in the absorber" on the side it actually put it
on. Reverted; see the PR body / rfx-known-issues.md #500 entry for the
recorded run.
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
    PreflightConfigError,
)
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.grid import Grid


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
    assert ng.nz == len(dz_profile) + ng.pad_z_lo + ng.pad_z_hi

    dz_full = np.asarray(ng.dz)
    edges = np.concatenate(([0.0], np.cumsum(dz_full)))
    # Node pad_z_lo is user z=0 (mirrors Grid's node pad_x_lo convention).
    z_node0 = edges[0] - edges[ng.pad_z_lo]
    z_node_last = edges[-1] - edges[ng.pad_z_lo]
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
# --------------------------------------------------------------------- #

_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
_LX = 14e-3


def _msl_x_clearance_sim(dx: float, port_x: float) -> Simulation:
    ly = _W_TRACE + 8 * _H_SUB
    sim = Simulation(
        freq_max=5e9, domain=(_LX, ly, _H_SUB + 1.5e-3), dx=dx, cpml_layers=8,
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


def test_msl_x_cpml_clearance_silent_on_former_false_positive():
    """Former false positive: dx=100um, cpml_layers=8 -> CPML thickness
    800um. Port at x=600um: the pre-#500 interior-frame check measured
    clearance = 600 - 800 = -200um (< the 508um=2*h_sub recommendation)
    and fired. The real clearance to the domain edge (x=0, where the
    absorber actually begins) is 600um > 508um -> must be silent."""
    issues = _msl_x_clearance_sim(dx=100e-6, port_x=0.6e-3).preflight(strict=False)
    hits = [m for m in issues if "x-CPML" in m]
    assert not hits, f"port with real 600um clearance must not warn; got {hits}"


def test_msl_x_cpml_clearance_fires_when_genuinely_too_close():
    """Positive control: port at x=300um genuinely has only 300um of
    real clearance to the x=0 domain edge (< 508um recommended) —
    must still warn under both the old and corrected frame."""
    issues = _msl_x_clearance_sim(dx=100e-6, port_x=0.3e-3).preflight(strict=False)
    hits = [m for m in issues if "x-CPML" in m]
    assert hits, f"port with real 300um clearance must warn; got {issues!r}"
