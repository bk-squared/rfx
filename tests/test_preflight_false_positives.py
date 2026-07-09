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


def _codes(sim):
    return {getattr(i, "code", None) for i in sim.preflight()}


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


# ---------------------------------------------------------------------------
# Item #3 (LLM-naive-usage audit) — pec_boundary_open advisory: R2-STOP lock.
#
# The audit asked whether the ``pec_boundary_open`` advisory
# (``_validate_cfg_pec_boundary_open_structure``) should be UNGATED from its
# ``self._ntff is not None`` condition so an open radiator read via a near-field
# probe / S11 alone (no NTFF box) also warns. Investigation (2026-07-09)
# R2-STOPPED that ungating: NTFF is the SOLE radiation-intent signal on the
# Simulation config (there is no directivity / far-field / "radiate" flag), so
# a source (and/or a finite PEC object) inside a ``boundary="pec"`` domain is
# config-IDENTICAL between an open radiator that mistakenly used PEC and a
# legitimate closed cavity / internal-PEC numerics test. The committed suite is
# full of the latter — e.g. ``test_adi.py::
# test_simulation_adi_internal_pec_geometry_masks_ez`` (PEC Box in a pec box +
# source), ``test_extract_s_matrix_pec_mask.py``, ``test_conformal.py::
# test_api_conformal_flag`` (PEC cylinder in a pec box + port). Any broadening
# that catches the footgun would false-alarm all of them, and a false-alarming
# preflight erodes trust worse than the silent gap. These tests LOCK that
# decision: the NTFF-declared open radiator still warns; the valid closed
# structures must stay silent so a future well-meaning ungating cannot regress
# them unnoticed.
# ---------------------------------------------------------------------------
def test_pec_boundary_open_still_warns_when_ntff_declared():
    """Radiation intent (an NTFF box) + boundary='pec' must still warn —
    the existing, principled gate is preserved by the R2-STOP."""
    sim = Simulation(freq_max=10e9, domain=(0.06, 0.06, 0.06), dx=2e-3,
                     boundary="pec")
    sim.add_source((0.03, 0.03, 0.03), "ez")
    sim.add(Box((0.028, 0.028, 0.020), (0.032, 0.032, 0.024)), material="pec")
    sim.add_ntff_box((0.01, 0.01, 0.01), (0.05, 0.05, 0.05))
    assert "pec_boundary_open" in _codes(sim), (
        "NTFF-declared open radiator on a PEC boundary must still warn"
    )


def test_pec_cavity_with_internal_pec_object_stays_silent():
    """FALSE-POSITIVE lock: a source + finite PEC object inside a pec box
    (the ``test_adi`` internal-PEC-masks-Ez / ``test_conformal`` patterns) is a
    VALID closed structure and must NOT emit pec_boundary_open. This is the
    population that any ntff-ungating would false-alarm — the reason #3 was
    R2-STOPPED."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     boundary="pec")
    sim.add(Box((0.008, 0.008, 0.0), (0.012, 0.012, 0.01)), material="pec")
    sim.add_source((0.01, 0.01, 0.0), "ez")
    sim.add_probe((0.01, 0.01, 0.0), "ez")
    assert "pec_boundary_open" not in _codes(sim), (
        "internal-PEC-object closed cavity must not warn (R2-STOP rationale)"
    )


def test_pec_empty_cavity_with_source_stays_silent():
    """FALSE-POSITIVE lock: a bare source in a pec box (empty resonant cavity)
    is config-identical to an open radiator and must NOT warn — there is no
    discriminator, hence the R2-STOP."""
    sim = Simulation(freq_max=12e9, domain=(0.03, 0.03, 0.03), dx=1.5e-3,
                     boundary="pec")
    sim.add_source((0.015, 0.015, 0.015), "ez")
    assert "pec_boundary_open" not in _codes(sim), (
        "empty PEC cavity with a source must not warn (R2-STOP rationale)"
    )
