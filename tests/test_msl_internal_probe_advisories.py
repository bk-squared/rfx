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
and #332/#336 skip internal indices; USER probes keep pre-#470 behavior
exactly (the #332 fail-safe in test_issue80_patch_s11_regression relies on
user-probe series still being evaluated during MSL runs).
"""
import warnings

import jax.numpy as jnp
import numpy as np

from rfx import Box, Simulation

FREQS = jnp.linspace(2e9, 18e9, 8)


def _thru():
    sim = Simulation(freq_max=20e9, domain=(0.012, 0.008, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0.0008)), material="sub")
    sim.add(Box((0.0, 0.0034, 0.0008), (0.012, 0.0046, 0.0010)),
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
    assert not any("Probe at" in m for m in msgs), msgs
    assert not any("issue #332" in m for m in msgs), msgs
    assert any("settling witness" in m for m in msgs), msgs
    assert res.settling_db is not None
    # bookkeeping fully restored: no internal probes leak out of the call
    assert sim._internal_probe_indices == set()
    assert len(sim._probes) == 0


def test_user_probe_advisories_and_332_still_fire():
    """USER probes keep pre-#470 behavior during the same MSL run: a probe
    near the x-CPML still draws a placement advisory, and an interior
    user probe's hot tail still fires #332 on the truncated record (the
    fail-safe test_issue80_patch_s11_regression depends on).

    Issue #500 / review finding H1: on this fixture (dx=2e-4, cpml_layers=8
    -> pad=8) the second probe sits at x=200um = node 9, one cell inside
    the domain (pad=8 is node 8, the first interior node) -- genuinely
    interior under the #500 exterior-padding frame, so it no longer trips
    the absorber_overlap ("near/inside") advisory the pre-#500 code fired
    (for the wrong, interior-frame reason). It is exactly the
    _ABSORBER_PROXIMITY_CELLS=2 case the #500 review's H1 finding restored:
    within 2 cells of the boundary, so it draws the distinct
    absorber_proximity advisory instead. This test and
    tests/test_run_preflight_parity.py's near-identical fixture were the
    two regressions that surfaced the removed proximity coverage as
    load-bearing (see _validate_cfg_absorber_placement's docstring)."""
    sim = _thru()
    # interior probe above the trace: sees real signal -> hot tail at
    # num_periods=2 -> #332 must fire on the USER column
    sim.add_probe(position=(0.006, 0.004, 0.0012), component="ez")
    # one cell inside the domain (node 9 of pad=8) -> absorber_proximity
    sim.add_probe(position=(0.0002, 0.004, 0.0012), component="ez")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.compute_msl_s_matrix(freqs=FREQS, num_periods=2.0)
    msgs = [str(w.message) for w in caught]
    # compute_msl_s_matrix folds individual preflight findings into ONE
    # consolidated "[run] preflight found N advisory issue(s)" UserWarning
    # (issue #66 parity), so absorber_proximity's own .code is not visible
    # on the outer warning object here -- match its text instead, the way
    # this test always has.
    assert any(
        "within 2 cells" in m and "CPML absorber" in m and "fringe/reflection" in m
        for m in msgs
    ), msgs
    assert any("issue #332" in m for m in msgs), msgs
    # user probes survive the call
    assert len(sim._probes) == 2
    assert sim._internal_probe_indices == set()
