"""T7-E Phase 2 PR3 — PMC runtime mechanism (not a physics oracle).

Pins the mechanism of ``rfx.boundaries.pmc.apply_pmc_faces``. The
quantitative PMC physics (λ/4 mode-ladder vs PEC-PEC λ/2 ladder) is
asserted separately in
``tests/test_boundary_pmc_oracle.py::test_pmc_lambda_quarter_two_peak_ladder``
— that oracle is what makes the PMC claim load-bearing; this file
only guarantees the mechanism wiring.

Mechanism pins:

1. **Tangential-H-zero at the PMC face** — direct state-field sample
   asserts ``|H_tan|`` < 1e-20 on PMC-designated face cells after the
   scan body runs.
2. **Finite non-zero trace** — PMC + CPML mix produces a valid
   probe time series.
3. **Mixed PMC/CPML seam finite output** — no NaN/Inf when PMC and
   CPML share an axis.
4. **Dual-boundary Hx sample** — PMC zeros Hx at the face cell while
   PEC leaves it free, confirming the two code paths are engaged.
5. **Distributed-PMC acceptance** (T8, 2026-04) — the sharded NU forward
   path now wires ``apply_pmc_faces`` via ``_apply_pmc_face_nu_shmap``
   (and dually in ``distributed_v2.py`` / ``distributed.py``).
   ``forward(distributed=True)`` with any PMC face must NOT raise
   ``NotImplementedError``. Full sharded-state evidence is in
   ``tests/test_boundary_pmc_distributed.py``.

Additional quantitative oracles beyond the λ/4 ladder (closed-box
energy conservation, analytic impedance matching) are tracked as a
follow-up harness — they need a source-calibrated long run.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


def test_tangential_h_is_zero_at_pmc_face():
    """Sample H at the PMC face at the end of a short run and assert
    that the tangential components are zero (or below numerical dust)."""
    spec = BoundarySpec(
        x="cpml", y="cpml",
        z=Boundary(lo="pmc", hi="cpml"),
    )
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
        boundary=spec,
    )
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.005), "ez")
    res = sim.run(n_steps=40)

    # state.hx and state.hy at k=0 are tangential to the z_lo face,
    # so PMC must zero them. state.hz is normal — skip.
    hx_at_z_lo = np.asarray(res.state.hx)[:, :, 0]
    hy_at_z_lo = np.asarray(res.state.hy)[:, :, 0]
    max_hx = float(np.max(np.abs(hx_at_z_lo)))
    max_hy = float(np.max(np.abs(hy_at_z_lo)))
    # PMC is enforced at every scan step; post-scan max_|H_tan| must
    # be zero (exactly, in float32 arithmetic).
    assert max_hx < 1e-20, f"PMC z_lo failed: max |Hx| = {max_hx:.3e}"
    assert max_hy < 1e-20, f"PMC z_lo failed: max |Hy| = {max_hy:.3e}"


def test_pmc_runtime_produces_finite_nonzero_trace():
    """Sanity: a PMC + CPML sim injects energy and produces a
    non-zero finite probe trace."""
    spec = BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pmc", hi="cpml"))
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
        boundary=spec,
    )
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.006), "ez")
    ts = np.asarray(sim.run(n_steps=80).time_series)
    assert np.all(np.isfinite(ts))
    assert float(np.max(np.abs(ts))) > 1e-9


def test_mixed_pmc_cpml_seam_is_finite():
    """Mixed-face regression: PMC on z_lo, CPML on z_hi. No NaN / Inf
    in the probe trace. The late-time stability bound + quantitative
    reflection measurements live in the physics harness rather than
    as unit tests (too sensitive to source waveform + probe placement)."""
    spec = BoundarySpec(
        x="cpml", y="cpml",
        z=Boundary(lo="pmc", hi="cpml"),
    )
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
        boundary=spec,
    )
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.006), "ez")
    ts = np.asarray(sim.run(n_steps=100).time_series)[:, 0]
    assert np.all(np.isfinite(ts))


def test_pmc_plus_distributed_forward_accepts():
    """T8 (2026-04): PMC is now wired into the sharded NU forward path.

    The reject guard that used to live at ``rfx/api.py:4264-4274`` was
    removed in T8 once ``_apply_pmc_face_nu_shmap`` landed in the
    sharded scan body. Regression pin: ``forward(distributed=True)``
    with a PMC face must NOT raise ``NotImplementedError`` anymore.

    Quantitative PMC-on-sharded-state evidence is asserted in
    ``tests/test_boundary_pmc_distributed.py``. Here we only pin that
    the dispatch no longer blocks; a full sharded run requires a
    multi-device fixture and is covered separately.
    """
    import jax
    dz = np.array([0.5e-3] * 10, dtype=np.float64)
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, float(np.sum(dz))),
        dx=0.5e-3, dz_profile=dz, cpml_layers=4,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pmc", hi="cpml")),
    )
    sim.add_source((0.005, 0.005, 0.002), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    # Use a real single-device list. Multi-device sharded-PMC evidence
    # lives in tests/test_boundary_pmc_distributed.py.
    devices = [jax.devices()[0]]
    # Should NOT raise the removed PMC NotImplementedError. If the
    # guard is ever restored by mistake, this line raises.
    res = sim.forward(n_steps=20, distributed=True, devices=devices,
                      skip_preflight=True)
    # Sanity: probe time series is finite.
    ts = np.asarray(res.time_series)
    assert np.all(np.isfinite(ts))


def test_pmc_and_pec_produce_physically_different_h_at_face():
    """Dual-boundary evidence: PEC zeros tangential E on z_lo, PMC zeros
    tangential H on z_lo. The direct field-sample evidence pins the
    duality without relying on probe-time-series sensitivity that
    depends on source waveform and propagation time.
    """
    def _final_state(spec):
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
            boundary=spec,
        )
        sim.add_source((0.005, 0.005, 0.005), "ez")
        sim.add_probe((0.005, 0.005, 0.005), "ez")
        return sim.run(n_steps=40).state

    # PMC z_lo must zero Hx, Hy at z=0 (tangential H). PEC z_lo
    # leaves Hx, Hy free (PEC zeros tangential E instead — see
    # apply_pec_faces).
    st_pmc = _final_state(BoundarySpec(x="cpml", y="cpml",
                                       z=Boundary(lo="pmc", hi="cpml")))
    st_pec = _final_state(BoundarySpec(x="cpml", y="cpml",
                                       z=Boundary(lo="pec", hi="cpml")))
    hx_pmc_face = float(np.max(np.abs(np.asarray(st_pmc.hx)[:, :, 0])))
    hx_pec_face = float(np.max(np.abs(np.asarray(st_pec.hx)[:, :, 0])))
    assert hx_pmc_face < 1e-20, f"PMC z_lo failed to zero Hx: {hx_pmc_face:.3e}"
    # PEC should have non-zero Hx at z_lo (nothing constrains it there).
    # The threshold is loose — we just assert the PEC path does NOT
    # zero H (i.e. Hx_pec is meaningfully > machine-zero).
    assert hx_pec_face > hx_pmc_face, (
        f"PEC z_lo unexpectedly left Hx ≈ 0 ({hx_pec_face:.3e}); "
        f"PEC and PMC may share a code path"
    )
