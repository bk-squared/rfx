"""Preflight false-positive refinements (2026-05-06).

The Y2 MSL stub-notch demo (`examples/inverse_design/msl_stub_notch_tuning.py`)
exposed three preflight checks that were firing on canonical
transmission-line geometry while still leaving the original footgun cases
covered.  These tests pin both halves of each refinement: the FP case must
go silent, and the original footgun case must still warn.

* FP1 — `PEC ?-extent N cells volume under-resolved` was firing on
  microstrip-trace strips (`LX × W_trace × dx` ≈ many × 4.7 × 1 cells)
  because the per-axis loop saw `4.7 cells` on `y` and ignored that
  `z = 1 cell` makes the object a thin sheet, not a volume.
* FP3 — `Material extends into CPML region along x-axis` was firing on
  `Box((0, 0, 0), (LX, LY, H_SUB))` substrates and traces because the
  original issue #61 check did not distinguish "intentional full-domain
  extension" from "accidental leak past the absorber".
* FP4 — `Port/source ... inside PEC geometry. Field will be zero.` was
  firing on MSL diagnostic Hy probes positioned at `z = h_sub + 0.5·dx`
  (the centre of a 1-cell trace PEC, where tangential H is non-zero
  by physics).

See `rfx/api.py::_validate_mesh_quality` and
`rfx/api.py::_validate_simulation_config` for the refined conditions.
"""

from __future__ import annotations

from rfx import Simulation, Box


def _issues(sim):
    return sim.preflight()


def _has(issues, substring):
    return any(substring in i for i in issues)


# ---------------------------------------------------------------------------
# FP1 — thin-sheet PEC strip should not trigger PEC-volume warning
# ---------------------------------------------------------------------------
def test_thin_pec_strip_with_4_cell_y_silent_on_volume_warning():
    """Strip-shape PEC with z = 1 cell is a thin sheet, not a volume.
    The 4-cells-along-y signal must not fire the volume warning."""
    DX = 0.5e-3
    LX, LY, LZ = 0.030, 0.005, 0.002
    sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=DX,
                     cpml_layers=4)
    sim.add_source((LX/2, LY/2, 0.0005), "ez")
    # 30mm × 2mm × 0.5mm = 60 × 4 × 1 cells (y-axis is the FP zone)
    sim.add(Box((0.0, 0.0015, 0.001), (LX, 0.0035, 0.0015)),
            material="pec")
    issues = _issues(sim)
    assert not _has(issues, "PEC volume"), (
        f"thin PEC strip (1 cell in z) must not fire volume warning; "
        f"issues: {issues!r}"
    )


def test_pec_volume_partial_in_all_axes_still_warns():
    """A PEC that is 3-5 cells in EVERY axis is the original target of
    the volume warning — must still fire."""
    DX = 1e-3
    sim = Simulation(freq_max=10e9, domain=(0.020, 0.020, 0.020), dx=DX,
                     cpml_layers=4)
    sim.add_source((0.010, 0.010, 0.002), "ez")
    # 4mm × 4mm × 4mm = 4 × 4 × 4 cells: every axis in [3, 5).
    sim.add(Box((0.005, 0.005, 0.005), (0.009, 0.009, 0.009)),
            material="pec")
    issues = _issues(sim)
    assert _has(issues, "PEC volume"), (
        f"true 4-cell PEC volume must still warn; issues: {issues!r}"
    )


# ---------------------------------------------------------------------------
# FP3 — explicit full-domain Box edge is not a CPML-extension footgun
# ---------------------------------------------------------------------------
def test_full_domain_dielectric_silent_on_cpml_extension():
    """Box((0, 0, 0), (LX, LY, ...)) is the canonical MSL substrate
    pattern — must not trigger the issue #61 CPML-extension warning."""
    LX, LY, LZ = 0.030, 0.005, 0.002
    sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=0.5e-3,
                     cpml_layers=4)
    sim.add_material("fr4", eps_r=4.3)
    sim.add(Box((0, 0, 0), (LX, LY, 0.0005)), material="fr4")
    sim.add_source((LX/2, LY/2, 0.0002), "ez")
    issues = _issues(sim)
    assert not _has(issues, "extends into CPML"), (
        f"full-domain Box must not fire CPML-extension warning; "
        f"issues: {issues!r}"
    )


def test_inset_box_leaking_into_cpml_still_warns():
    """A Box that inset short of the domain edge (so the user clearly
    did NOT intend full-domain extension) but still drifts into the
    CPML region must still warn — this is the original issue #61
    leak-into-absorber case."""
    LX, LY, LZ = 0.030, 0.005, 0.002
    DX = 0.5e-3
    sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=DX,
                     cpml_layers=4)
    sim.add_material("fr4", eps_r=4.3)
    # CPML thickness = 4 × 0.5mm = 2mm.  Box inset 0.5mm on each side
    # — well short of the domain edge (5·dx away from LX) so the
    # intentional-edge heuristic does not exempt, AND inside the
    # 30 % CPML penetration threshold (0.5mm < 0.6mm).
    sim.add(Box((0.0005, 0, 0), (LX - 0.0005, LY, 0.0005)),
            material="fr4")
    sim.add_source((LX/2, LY/2, 0.0002), "ez")
    issues = _issues(sim)
    assert _has(issues, "extends into CPML"), (
        f"Box inset and leaking into CPML must still warn; "
        f"issues: {issues!r}"
    )


# ---------------------------------------------------------------------------
# FP4 — H-component probe at thin-PEC-sheet centre is valid
# ---------------------------------------------------------------------------
def _msl_sim_with_probe(component: str) -> Simulation:
    """Tiny MSL geometry with one diagnostic probe at the centre of the
    1-cell trace PEC.  Used to test FP4 component-aware exemption."""
    EPS_R = 3.66
    H_SUB = 254e-6
    W_TRACE = 600e-6
    DX = 127e-6
    LX, LY, LZ = 0.010, 0.005, H_SUB + 1.0e-3
    sim = Simulation(freq_max=9e9, domain=(LX, LY, LZ), dx=DX,
                     cpml_layers=4)
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_trace = LY / 2.0
    sim.add(
        Box((0, y_trace - W_TRACE / 2, H_SUB),
            (LX, y_trace + W_TRACE / 2, H_SUB + DX)),
        material="pec",
    )
    sim.add_source((0.5e-3, y_trace, 0.5 * H_SUB), "ez")
    # Probe at trace cell centre: z = H_SUB + 0.5·dx, inside the PEC
    # trace bbox by construction.
    sim.add_probe((LX / 2, y_trace, H_SUB + 0.5 * DX), component)
    return sim


def test_hy_probe_at_thin_trace_pec_silent_on_inside_pec():
    """An Hy diagnostic probe placed at the centre of a 1-cell trace
    PEC measures tangential H — physically non-zero — and must not
    trigger the inside-PEC warning."""
    sim = _msl_sim_with_probe("hy")
    issues = _issues(sim)
    assert not _has(issues, "is inside PEC geometry"), (
        f"Hy probe at thin-trace PEC centre must not warn; "
        f"issues: {issues!r}"
    )


def test_ez_probe_at_thin_trace_pec_still_warns():
    """An Ez probe at the same position is killed by the PEC update —
    the warning must still fire (only H components are exempt)."""
    sim = _msl_sim_with_probe("ez")
    issues = _issues(sim)
    assert _has(issues, "is inside PEC geometry"), (
        f"Ez probe inside thin PEC must still warn; issues: {issues!r}"
    )


def test_hy_probe_inside_thick_pec_volume_still_warns():
    """H decays to zero deep inside a thick PEC volume.  An Hy probe
    placed at the centre of a 5-cell PEC cube must still warn — the
    thin-sheet exemption applies only to ≤ 1.5·dx-thick PEC."""
    DX = 1.0e-3
    LX, LY, LZ = 0.020, 0.020, 0.020
    sim = Simulation(freq_max=10e9, domain=(LX, LY, LZ), dx=DX,
                     cpml_layers=4)
    sim.add_source((0.002, 0.002, 0.002), "ez")
    # 5 × 5 × 5 mm = 5 × 5 × 5 cells PEC volume.
    sim.add(Box((0.005, 0.005, 0.005), (0.010, 0.010, 0.010)),
            material="pec")
    sim.add_probe((0.0075, 0.0075, 0.0075), "hy")  # cell centre, deep
    issues = _issues(sim)
    assert _has(issues, "is inside PEC geometry"), (
        f"Hy probe in thick PEC volume must still warn; "
        f"issues: {issues!r}"
    )
