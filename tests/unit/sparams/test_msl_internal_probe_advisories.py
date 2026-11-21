"""Issue #470: library-internal witness probes stay out of the user record.

``compute_msl_s_matrix`` registers N point Ez settling-witness probes per
port (PR #468). Pre-#470 they were indistinguishable from user probes, so
(1) the internal run's auto-preflight reported ~10 self-inflicted
"Probe at ... near/inside CPML" advisories per driven run on the 2-port
thru — burying the genuine MSL port-clearance advisories that must be
quoted next to every claims-bearing number — and (2) the generic #332
probe-tail advisory double-fired next to the MSL-specific ``settling_db``
witness (two differently-defined settling numbers per run).

Containment: ``Simulation._internal_probe_indices`` runtime bookkeeping
(deliberately NOT an ``_ProbeEntry`` field — the entry field set is pinned
by the interop design-IR contract). Probe-placement preflight advisories
and #332/#336 skip internal indices. USER probes keep placement advisories;
the MSL runs read DFT planes, so their per-bin settling witness replaces
#332's envelope advisory even when user probes are present.
The sheet lands exactly on the substrate face's node line at h_sub / dx = 0.8 mm / 0.2 mm = 4.
"""
import warnings

import jax.numpy as jnp

from rfx import Box, Simulation
from tests._realized_geometry import assert_sheet_planes, assert_wall_planes

FREQS = jnp.linspace(2e9, 18e9, 8)


def _thru():
    """Drawn across the whole domain, the strip left 2 mm open stubs whose effective quarter wave (23.8 GHz) is within 1.5x of the 2-18 GHz read."""
    sim = Simulation(snap="declared", freq_max=20e9, domain=(0.012, 0.008, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0.0008)), material="sub")
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0)), material="pec")
    sim.add(Box((0.002, 0.0034, 0.0008), (0.010, 0.0046, 0.0008)),
            material="pec")
    sim.add_msl_port(position=(0.002, 0.004, 0.0), width=0.0012,
                     height=0.0008, direction="+x", impedance=50.0,
                     eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, 0.004, 0.0), width=0.0012,
                     height=0.0008, direction="-x", impedance=50.0,
                     eps_r_sub=2.2, name="p2")
    return sim


def test_witness_probes_produce_no_probe_advisories_and_no_332():
    """A user-probe-free thru run must emit ZERO probe-placement advisories
    ('Probe at ...') and ZERO #332 tail advisories — pre-#470 it emitted 10
    'Probe at ... near/inside CPML' self-advisories per driven run and
    double-fired #332 next to settling_db. The MSL-specific settling
    witness itself must still fire on this truncated record (the
    consolidation keeps ONE witness, it does not silence the diagnosis)."""
    sim = _thru()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)
    msgs = [str(w.message) for w in caught]
    print(f'MSL user_probes={len(sim._probes)}: '
          f'envelope={[m for m in msgs if "ring-down truncated" in m]}, '
          f'witness={[m for m in msgs if "settling witness" in m]}')
    assert not any("Probe at" in m for m in msgs), msgs
    assert not any("ring-down truncated" in m for m in msgs), msgs
    assert any("settling witness" in m for m in msgs), msgs
    assert res.settling_db is not None
    assert sim._internal_probe_indices == set()
    assert len(sim._probes) == 0


def test_user_probe_advisories_and_332_still_fire():
    """User probes keep their placement advisory during the same MSL run.
    The truncated record emits the S settling-witness warning. Its DFT
    planes exclude the run from the envelope advisory. The interior probe
    above the trace sees real signal and a hot tail at num_periods=2.

    Issue #500 / review finding H1: on this fixture (dx=2e-4, cpml_layers=8
    -> pad=8) the second probe sits at x=200um = node 9, one cell inside
    the domain (pad=8 is node 8, the first interior node) -- genuinely
    interior under the #500 exterior-padding frame, so it no longer trips
    the absorber_overlap ("near/inside") advisory the pre-#500 code fired
    (for the wrong, interior-frame reason). It is exactly the
    _ABSORBER_PROXIMITY_CELLS=2 case the #500 review's H1 finding restored:
    within 2 cells of the boundary, so it draws the distinct
    absorber_proximity advisory instead. This test and
    tests/unit/preflight/test_run_preflight_parity.py's near-identical fixture were the
    two regressions that surfaced the removed proximity coverage as
    load-bearing (see _validate_cfg_absorber_placement's docstring).
    The consolidated preflight UserWarning hides absorber_proximity's own code, so the warning is matched by text.
    """
    sim = _thru()
    sim.add_probe(position=(0.006, 0.004, 0.0012), component="ez")
    sim.add_probe(position=(0.0002, 0.004, 0.0012), component="ez")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)
    msgs = [str(w.message) for w in caught]
    print(f'MSL user_probes={len(sim._probes)}: '
          f'envelope={[m for m in msgs if "ring-down truncated" in m]}, '
          f'witness={[m for m in msgs if "settling witness" in m]}')
    assert any(
        "within 2 cells" in m and "CPML absorber" in m and "fringe/reflection" in m
        for m in msgs
    ), msgs
    assert not any("ring-down truncated" in m for m in msgs), msgs
    assert any("settling witness" in m for m in msgs), msgs
    assert len(sim._probes) == 2
    assert sim._internal_probe_indices == set()


def test_realized_conductor_planes_equal_the_declaration():
    """Build-time witness (no solve) for the #931 ownership contract.

    Ground and trace are declared as SHEETS, each with exactly one wall
    plane on its own laminate face,
    with the normal Ez edge through it left live. Drawn one cell thick it
    was a volume: two walls, and the Ez edge between them shorted. This
    assertion is what keeps the declaration and the realization the same
    statement.
    """
    sim = _thru()
    assert_sheet_planes(sim, 2, [0., 0.0008], what="MSL thru ground and trace")
    assert_wall_planes(sim, 2, [0., 0.0008], what="MSL thru ground and trace")
