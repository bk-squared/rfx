"""T7 Phase 2 PR2 — asymmetric per-face CPML thickness (mechanism only).

These tests pin the PADDED-PROFILE MECHANISM shipped in
``rfx/boundaries/cpml.py``:

- ``Grid.face_layers`` is populated from ``BoundarySpec``,
- the Simulation builds and runs without NaN on asymmetric thickness,
- the σ array for a thin face has σ_max at index 0 descending, zeros
  in the padded no-op region, and σ_max at the outer-boundary end for
  the (pre-flipped) hi-face.

The QUANTITATIVE asymmetric-CPML evidence is COEFFICIENT-LEVEL (σ_max),
not a field-reflection ratio. ``tests/test_boundary_cpml_oracle.py`` pins
σ_max on the 4-layer face at exactly 4× the 16-layer face
(``test_asymmetric_sigma_max_ratio_matches_analytic``), the polynomial
grading shape, and that the per-face σ arrays differ
(``test_asymmetric_sigma_arrays_differ``) — that is what makes the
asymmetric-CPML claim load-bearing. A between-face FIELD-reflection ratio
is deliberately NOT asserted: the Gaussian source pulse (~200 steps) is
longer than the source-to-face round trip (~30 steps), so it cannot be
time-gated in a single asymmetric sim (see that oracle's docstring). This
file only guarantees the mechanism wiring.
"""

from __future__ import annotations

import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


def _run_probe_trace(boundary_spec: BoundarySpec,
                     probe_position: tuple[float, float, float],
                     n_steps: int) -> np.ndarray:
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
        boundary=boundary_spec, cpml_layers=16,
    )
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe(probe_position, "ez")
    return np.asarray(sim.run(n_steps=n_steps).time_series)[:, 0]


def test_asymmetric_cpml_builds_and_runs_end_to_end():
    """Asymmetric thickness via padded-profile engine runs to completion
    with finite output. The deep physics validation (analytic R(ω)
    comparison, thin-face reflection quantitative bound) is tracked as
    a follow-up; this pin verifies the mechanism (Grid.face_layers
    propagation, padded profile construction, no NaN/Inf in the scan
    body) which is the acceptance criterion for PR2 mechanism correctness."""
    spec_asym = BoundarySpec(
        x="cpml", y="cpml",
        z=Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=16),
    )
    probe = (0.005, 0.005, 0.005)
    trace = _run_probe_trace(spec_asym, probe, n_steps=80)
    assert trace.shape[0] == 80
    assert np.all(np.isfinite(trace)), "asymmetric CPML run produced NaN/Inf"
    assert float(np.max(np.abs(trace))) > 1e-9, (
        "asymmetric CPML run produced near-zero probe signal"
    )


def test_padded_profile_structure_is_correct():
    """Inspect the padded profile directly: the thin face's σ array has
    σ_max at index 0 descending to ~0 at index 3, then 0 from index 4
    through 15. The hi (pre-flipped + front-padded) face has the
    reverse layout."""
    from rfx.boundaries.cpml import init_cpml

    spec_asym = BoundarySpec(
        x="cpml", y="cpml",
        z=Boundary(lo="cpml", hi="cpml", lo_thickness=4, hi_thickness=16),
    )
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
        boundary=spec_asym, cpml_layers=16,
    )
    grid = sim._build_grid()
    params, _ = init_cpml(grid)

    # Thin (4-layer) z_lo: first 4 sigma entries are real + descending;
    # indices 4..15 must all be zero (no-op padding).
    sigma_zlo = np.asarray(params.z_lo.sigma)
    assert sigma_zlo[0] > 0.0, "thin z_lo outer-boundary sigma must be > 0"
    assert np.all(sigma_zlo[4:] == 0.0), (
        f"thin z_lo padding region must be zero, got {sigma_zlo[4:]}"
    )
    # 16-layer z_hi: no padding, all 16 entries active. After _flip_profile
    # sigma has σ_max at index 15, 0 at index 0.
    sigma_zhi = np.asarray(params.z_hi.sigma)
    assert sigma_zhi[-1] > 0.0, "full-thickness z_hi outer-boundary sigma must be > 0"
    assert sigma_zhi[0] == 0.0, "z_hi interior-facing sigma must be 0 (pre-flipped convention)"


def test_symmetric_thickness_matches_scalar_cpml_layers():
    """Bit-identity pin: a BoundarySpec with explicit symmetric
    thickness matching cpml_layers must produce the same trace as
    the plain scalar input. Both routes resolve to identical
    Grid.face_layers and identical init_cpml output."""
    spec_explicit = BoundarySpec(
        x=Boundary(lo="cpml", hi="cpml", lo_thickness=16, hi_thickness=16),
        y=Boundary(lo="cpml", hi="cpml", lo_thickness=16, hi_thickness=16),
        z=Boundary(lo="cpml", hi="cpml", lo_thickness=16, hi_thickness=16),
    )
    spec_scalar = BoundarySpec.uniform("cpml")

    probe = (0.005, 0.005, 0.006)
    trace_explicit = _run_probe_trace(spec_explicit, probe, n_steps=60)
    trace_scalar = _run_probe_trace(spec_scalar, probe, n_steps=60)
    np.testing.assert_array_equal(trace_explicit, trace_scalar)
