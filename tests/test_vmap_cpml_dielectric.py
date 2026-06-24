"""Regression: material-aware CPML on the vmap_sweep scan body (issue #205).

``rfx/vmap_sweep.py``'s batched scan called ``apply_cpml_e/h`` WITHOUT
``materials=``, so the absorber used free-space ε₀/μ₀ regardless of the local
dielectric. When a dielectric fills the CPML region, the free-space coefficient
is ε_r× too strong and the scan diverges to NaN/inf — the same mechanism as the
uniform extractor bug (#203/#204) and the non-uniform scan body (#205→#208).

This is the vmap sibling of ``test_nonuniform_cpml_dielectric.py``. The fix is a
passthrough: ``materials`` is already the per-batch-element ``MaterialArrays``
inside the vmapped ``run_one``, so passing ``materials=materials`` lets vmap
batch it transparently (no batch-axis slicing). The witness must fill the whole
domain (all six CPML faces) so energy is absorbed THROUGH dielectric-filled CPML.

Witness measured on the buggy tree: vmap diverges (max|.|~1e24-1e26) while the
material-aware ``run()`` stays finite (~3e-2). Post-fix the vmap result is finite
and matches ``run()``.
"""

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.vmap_sweep import vmap_material_sweep


def _full_dielectric_cpml_sim(eps_r: float):
    """CPML sim whose dielectric fills the ENTIRE domain (incl. all 6 CPML
    faces), so the absorber sees the dielectric — the geometry that exposes a
    free-space-CPML divergence."""
    sim = Simulation(
        freq_max=5e9, domain=(0.02, 0.02, 0.02),
        boundary="cpml", cpml_layers=6, dx=0.002,
    )
    sim.add_material("d", eps_r=eps_r)
    sim.add(Box((0.0, 0.0, 0.0), (0.02, 0.02, 0.02)), material="d")
    sim.add_source((0.01, 0.01, 0.01), "ez", waveform=GaussianPulse(f0=3e9))
    sim.add_probe((0.006, 0.01, 0.01), "ez")
    return sim


def test_vmap_cpml_dielectric_is_finite_and_matches_run():
    """vmap over a dielectric that fills the CPML must stay finite (material-
    aware absorber) and reproduce the uniform material-aware run() path.

    Pre-#205-fix this diverges to NaN/inf because the vmap scan used free-space
    CPML inside the eps_r dielectric.
    """
    eps_values = np.array([4.0, 10.0])
    n_steps = 300  # buggy tree reaches ~1e26 well before this

    res = vmap_material_sweep(
        _full_dielectric_cpml_sim(10.0), "d.eps_r", eps_values, n_steps=n_steps,
    )
    ts = np.asarray(res.time_series)  # (2, n_steps, 1)

    # Primary regression guard: a free-space absorber in the dielectric diverges.
    assert np.isfinite(ts).all(), (
        "vmap CPML sweep produced non-finite fields — the dielectric-filled "
        "absorber is not material-aware (issue #205 regression). "
        f"max|.|={np.nanmax(np.abs(ts)):.3e}"
    )
    # The whole batch must be bounded (passive), not just non-NaN.
    assert float(np.max(np.abs(ts))) < 1.0, (
        f"vmap CPML sweep fields are implausibly large ({np.max(np.abs(ts)):.3e}) "
        "— absorber likely mismatched."
    )

    # Correctness: the eps_r=10 sweep element must reproduce the material-aware
    # uniform run() path (the supported reference) on the same geometry.
    ref = np.asarray(
        _full_dielectric_cpml_sim(10.0).run(n_steps=n_steps).time_series
    )
    np.testing.assert_allclose(
        ts[1], ref, atol=1e-5, rtol=1e-4,
        err_msg="vmap CPML sweep (eps_r=10) disagrees with material-aware run()",
    )


def test_vmap_cpml_distinct_eps_change_response():
    """Sanity: different eps in the (now finite) CPML sweep give distinct, finite
    responses — guards against a degenerate 'all-clamped-to-the-same' fix."""
    eps_values = np.array([2.0, 8.0])
    res = vmap_material_sweep(
        _full_dielectric_cpml_sim(8.0), "d.eps_r", eps_values, n_steps=250,
    )
    ts = np.asarray(res.time_series)
    assert np.isfinite(ts).all()
    assert np.max(np.abs(ts[0] - ts[1])) > 0.0, "eps_r had no effect on the sweep"
