"""Regression: lumped/wire S-parameter extraction must be CPML-material-aware.

Issue #203. ``run(compute_s_params=True)`` for a single-cell lumped port (or a
wire port) runs a *separate* eager FDTD re-run inside
``rfx.probes.probes.extract_s_matrix`` / ``extract_s_matrix_wire`` to accumulate
the port V/I DFTs. Those re-runs called ``apply_cpml_h`` / ``apply_cpml_e``
WITHOUT the ``materials=`` argument, so the CPML fell back to free-space eps_0
coefficients. Inside a dielectric (eps_r > 1) the absorber update is then
unstable, the field diverges exponentially to float32 overflow (first NaN
~step 300-400), the V/I DFTs are poisoned, and every S-parameter comes back NaN.

The production JIT scan body passes ``materials=materials`` to both CPML calls
and is stable — which is why the *same* run with ``compute_s_params=False`` is
healthy. The fix threads ``materials=mats`` into the CPML calls of BOTH
``extract_s_matrix`` (lumped) and ``extract_s_matrix_wire`` (wire), so the
extractor and the production scan handle dielectric-in-CPML identically.

``test_lumped_...`` is the strict #203 regression: it triggers the divergence
deterministically (dielectric spanning the full transverse cross-section so its
y/z faces sit in the CPML, run ~700 steps so the pre-fix field blow-up reaches
NaN), and was confirmed to FAIL on the pre-fix code and pass after.
``test_wire_...`` exercises the sibling wire extractor — which carried the same
missing-``materials=`` omission — on the same geometry; empirically this case
stayed finite even pre-fix (the single-cell lumped excitation is what seeded the
instability), so it is a forward-looking finiteness/passivity guard, not a
proven before/after regression. Both assert only finiteness and passivity
(|S| <= 1); the absolute |S11| of a lossless dielectric block on an open domain
is a near-total reflector and is not a validated number.
"""

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.sources.sources import GaussianPulse

# Enough steps for the pre-fix CPML divergence to reach NaN (it first appears
# ~step 300-400; fewer steps would let the buggy code pass spuriously).
_N_STEPS = 700


def _dielectric_in_cpml_sim():
    """Small open-domain sim with an eps_r=4 block spanning the transverse
    cross-section, so the dielectric occupies CPML cells (the #203 trigger)."""
    sim = Simulation(
        freq_max=5e9,
        domain=(0.06, 0.03, 0.02),
        dx=1.5e-3,
        boundary="cpml",
        cpml_layers=8,
    )
    sim.add_material("diel", eps_r=4.0)
    # Spans full y/z extent -> the block's transverse faces sit in the CPML.
    sim.add(Box((0.02, 0.0, 0.0), (0.04, 0.03, 0.02)), material="diel")
    return sim


def test_lumped_port_sparam_cpml_dielectric_finite_passive():
    """Single-cell lumped port + CPML + dielectric must give finite, passive S11.

    Pre-fix this returned all-NaN s_params (issue #203 as-filed symptom).
    """
    sim = _dielectric_in_cpml_sim()
    # Single-cell lumped port (extent=None), interior in x (clear of x-CPML).
    sim.add_port(
        position=(0.03, 0.015, 0.01),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8),
    )
    result = sim.run(n_steps=_N_STEPS, compute_s_params=True)

    assert result.s_params is not None
    s = np.asarray(result.s_params)
    assert s.shape == (1, 1, result.freqs.shape[0])
    assert np.all(np.isfinite(s)), "lumped S-params must be finite (issue #203)"
    max_s11 = float(np.max(np.abs(s[0, 0, :])))
    assert max_s11 <= 1.0 + 1e-3, f"passivity: max|S11|={max_s11:.4f} > 1"


def test_wire_port_sparam_cpml_dielectric_finite_passive():
    """Wire port (extent=) + CPML + dielectric stays finite and passive.

    ``extract_s_matrix_wire`` carried the identical missing-``materials=`` CPML
    omission and is fixed alongside the lumped extractor. This geometry did not
    by itself reproduce the pre-fix lumped divergence (the wire excitation does
    not seed it the same way), so this is a forward-looking finiteness/passivity
    guard on the wire S-param path rather than a proven before/after regression.
    """
    sim = _dielectric_in_cpml_sim()
    sim.add_port(
        position=(0.03, 0.015, 0.01),
        component="ez",
        impedance=50.0,
        waveform=GaussianPulse(f0=3e9, bandwidth=0.8),
        extent=0.006,
    )
    result = sim.run(n_steps=_N_STEPS, compute_s_params=True)

    assert result.s_params is not None
    s = np.asarray(result.s_params)
    assert np.all(np.isfinite(s)), "wire S-params must be finite (issue #203)"
    max_s11 = float(np.max(np.abs(s[0, 0, :])))
    assert max_s11 <= 1.0 + 1e-3, f"passivity: max|S11|={max_s11:.4f} > 1"
