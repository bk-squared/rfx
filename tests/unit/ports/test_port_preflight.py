"""Issue #71 — preflight warning for single-cell port in dielectric.

Pins the P1.9 preflight check in ``rfx/api.py`` that catches the
patch-antenna feed footgun: a single-cell ``LumpedPort`` placed
mid-substrate with no adjacent PEC pin cannot couple to the patch
TM mode. The optimiser reads a nonsense loss from the floating Ez
source.

Positive case: bad setup fires UserWarning citing ``extent=``.
Negatives: WirePort extent, port in air, coax-style port with
adjacent PEC pin, strict-mode ValueError.
"""

from __future__ import annotations

import warnings

import pytest

from rfx import Box, Simulation


def _build_patch(*, feed_extent: float | None, add_pec_pin: bool = False):
    """Minimal patch-antenna-style geometry mirroring issue #71's repro.

    Substrate spans 5 dx in z so a mid-substrate port sits genuinely
    inside the dielectric with 2 cells of substrate above and below —
    no PEC cell is adjacent along the z-axis, matching the #71 footgun.
    """
    F0, EPS_R, DX = 10e9, 3.38, 0.4e-3
    H_SUB = 5 * DX  # 2 mm — ensures port+/-1 cell stays in dielectric
    # PEC outer walls: keeps the preflight focus on the port-coupling
    # warning and avoids CPML-proximity noise on a small test domain.
    domain = (0.02, 0.015, 0.02)
    sim = Simulation(
        freq_max=2 * F0, domain=domain, dx=DX,
        boundary="pec",
    )
    sim.add_material("sub", eps_r=EPS_R)
    # Substrate sits interior to the domain so nothing is near the PEC walls.
    sub_z0 = 5e-3
    sim.add(Box((0.005, 0.0, sub_z0),
                (domain[0] - 0.005, domain[1], sub_z0 + H_SUB)),
            material="sub")
    # Patch sits at the top of the substrate (>= 2 cells above the
    # mid-substrate port cell).
    patch_z = sub_z0 + H_SUB
    sim.add(Box(
        (0.005, 0.004, patch_z),
        (0.015, 0.011, patch_z + DX),
    ), material="pec")
    feed_pos = (0.010, 0.007, sub_z0 + H_SUB * 0.5)
    if add_pec_pin:
        # Coax-style pin: PEC box immediately below the port cell (one
        # cell along the Ez axis) simulating a vertical probe feed.
        sim.add(Box(
            (feed_pos[0] - DX / 2, feed_pos[1] - DX / 2, sub_z0),
            (feed_pos[0] + DX / 2, feed_pos[1] + DX / 2, feed_pos[2] - DX / 2),
        ), material="pec")
    sim.add_port(feed_pos, "ez", impedance=50.0, extent=feed_extent)
    sim.add_probe(feed_pos, "ez")
    return sim


def _build_port_in_air():
    """Port floating in vacuum — nothing to warn about."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.5e-3,
                     boundary="cpml", cpml_layers=6)
    sim.add_port((0.01, 0.01, 0.01), "ez", impedance=50.0)
    sim.add_probe((0.01, 0.01, 0.01), "ez")
    return sim


def _capture_preflight(sim) -> list[str]:
    # preflight() consumes its own inner `catch_warnings` context and
    # returns the captured strings directly as a list.
    return sim.preflight()


def test_single_cell_port_in_dielectric_warns():
    """Bad setup: single-cell Ez port mid-substrate, no adjacent PEC pin."""
    sim = _build_patch(feed_extent=None, add_pec_pin=False)
    msgs = _capture_preflight(sim)
    matching = [m for m in msgs if "extent" in m and "sub" in m]
    assert matching, (
        f"expected UserWarning mentioning 'extent' and the enclosing "
        f"dielectric, got captured warnings: {msgs!r}"
    )


def test_wire_port_extent_no_warning():
    """Good setup: same geometry with extent=H_SUB promotes to WirePort."""
    sim = _build_patch(feed_extent=0.8e-3)
    msgs = _capture_preflight(sim)
    bad = [m for m in msgs if "floating single-cell port" in m]
    assert not bad, f"unexpected #71 warning for WirePort setup: {bad!r}"


def test_port_in_air_no_warning():
    """Port in vacuum has no enclosing dielectric, no warning."""
    sim = _build_port_in_air()
    msgs = _capture_preflight(sim)
    bad = [m for m in msgs if "floating single-cell port" in m]
    assert not bad, f"unexpected #71 warning for port-in-air: {bad!r}"


def test_port_adjacent_to_pec_no_warning():
    """Coax-style port with PEC pin one cell below — no warning."""
    sim = _build_patch(feed_extent=None, add_pec_pin=True)
    msgs = _capture_preflight(sim)
    bad = [m for m in msgs if "floating single-cell port" in m]
    assert not bad, f"unexpected #71 warning for coax-pin setup: {bad!r}"


def test_strict_mode_raises_valueerror():
    """Under strict=True the UserWarning is escalated to ValueError."""
    sim = _build_patch(feed_extent=None, add_pec_pin=False)
    with pytest.raises(ValueError, match="extent"):
        sim.preflight(strict=True)
