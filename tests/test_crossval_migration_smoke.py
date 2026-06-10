"""T9 (2026-04) — crossval BoundarySpec migration smoke tests.

Per-script rfx-only smoke for the 6 migrated crossval scripts. Pins
that the BoundarySpec-migrated scripts still build a valid Simulation
and produce finite probe output on a 5-step run. The full cross-solver
drift gate (Meep/OpenEMS pre/post numerical bit-identity) is deferred
to a VESSL session per the C.5 precedent; local-env gate is structural
(V173-A bit-identity + shim equivalence tests).

Each test mirrors the `rfx.Simulation(...)` construction of one
migrated crossval script but uses a tiny domain so the full test file
runs in <30 s. The point is to pin:

1. BoundarySpec-migrated constructors build without exception.
2. No DeprecationWarning fires on construction (pec_faces / set_periodic_axes).
3. `sim._boundary_spec` is a BoundarySpec instance after construction.
4. A 5-step run produces finite probe values.

Crossval 04 is deliberately omitted — it does not instantiate
`rfx.Simulation(...)`; it drives the raw Yee loop directly. See
`docs/research_notes/2026-04-18_v174_crossval04_skip_justification.md`.
"""
from __future__ import annotations

import warnings

import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.grid import C0


def _assert_finite_probe(result) -> None:
    ts = np.asarray(result.time_series)
    assert np.all(np.isfinite(ts)), f"probe has non-finite values: {ts}"


def _assert_no_legacy_deprecation(records) -> None:
    """Pin that no pec_faces / set_periodic_axes DeprecationWarning fires."""
    for w in records:
        msg = str(w.message)
        assert "pec_faces" not in msg, (
            f"migration left pec_faces DeprecationWarning: {msg}"
        )
        assert "set_periodic_axes" not in msg, (
            f"migration left set_periodic_axes DeprecationWarning: {msg}"
        )


def test_crossval_01_bend_smoke():
    """01_waveguide_bend.py — boundary=BoundarySpec.uniform(boundary)."""
    a = 1.0e-6
    dx = a / 10
    sx, sy = 10 * a, 10 * a
    boundary = "cpml"  # script parameter
    sim = Simulation(
        freq_max=0.25 * C0 / a,
        domain=(sx, sy, dx),
        dx=dx,
        boundary=BoundarySpec.uniform(boundary),
        cpml_layers=6,
        mode="2d_tmz",
    )
    sim.add_source((sx / 2, sy / 2, dx / 2), "ez")
    sim.add_probe((sx / 2 + dx, sy / 2, dx / 2), "ez")
    assert isinstance(sim._boundary_spec, BoundarySpec)
    res = sim.run(n_steps=5)
    _assert_finite_probe(res)


def test_crossval_02_ring_smoke():
    """02_ring_resonator.py — boundary=BoundarySpec.uniform('upml')."""
    a = 1.0e-6
    dx = a / 10
    domain = 10 * a
    sim = Simulation(
        freq_max=0.25 * C0 / a,
        domain=(domain, domain, dx),
        dx=dx,
        boundary=BoundarySpec.uniform("upml"),
        cpml_layers=6,
        mode="2d_tmz",
    )
    sim.add_source((domain / 2, domain / 2, dx / 2), "ez")
    sim.add_probe((domain / 2 + dx, domain / 2, dx / 2), "ez")
    assert isinstance(sim._boundary_spec, BoundarySpec)
    assert sim._boundary_spec.absorber_type == "upml"
    res = sim.run(n_steps=5)
    _assert_finite_probe(res)


def test_crossval_03_straight_flux_smoke():
    """03_straight_waveguide_flux.py — boundary=BoundarySpec.uniform('upml')."""
    a = 1.0e-6
    dx = a / 10
    dom_x, dom_y = 12 * a, 10 * a
    sim = Simulation(
        freq_max=0.25 * C0 / a,
        domain=(dom_x, dom_y, dx),
        dx=dx,
        boundary=BoundarySpec.uniform("upml"),
        cpml_layers=6,
        mode="2d_tmz",
    )
    sim.add_source((dom_x / 2, dom_y / 2, dx / 2), "ez")
    sim.add_probe((dom_x / 2 + dx, dom_y / 2, dx / 2), "ez")
    assert isinstance(sim._boundary_spec, BoundarySpec)
    res = sim.run(n_steps=5)
    _assert_finite_probe(res)


def test_crossval_05_patch_smoke():
    """05_patch_antenna.py — boundary=BoundarySpec.uniform('cpml')."""
    sim = Simulation(
        freq_max=4e9,
        domain=(0.02, 0.02, 0.005),
        dx=0.5e-3,
        boundary=BoundarySpec.uniform("cpml"),
        cpml_layers=6,
    )
    sim.add_source((0.01, 0.01, 0.0025), "ez")
    sim.add_probe((0.012, 0.01, 0.0025), "ez")
    assert isinstance(sim._boundary_spec, BoundarySpec)
    res = sim.run(n_steps=5)
    _assert_finite_probe(res)


def test_crossval_06_notch_smoke():
    """06_msl_notch_filter.py — pec_faces={'z_lo'} migrated to
    BoundarySpec(x='cpml', y='cpml', z=Boundary(lo='pec', hi='cpml')).
    Explicitly asserts no DeprecationWarning about pec_faces fires."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim = Simulation(
            freq_max=10e9,
            domain=(0.02, 0.01, 0.002),
            dx=0.5e-3,
            boundary=BoundarySpec(
                x="cpml", y="cpml",
                z=Boundary(lo="pec", hi="cpml"),
            ),
            cpml_layers=6,
        )
        _assert_no_legacy_deprecation(w)
    sim.add_source((0.01, 0.005, 0.001), "ez")
    sim.add_probe((0.012, 0.005, 0.001), "ez")
    assert isinstance(sim._boundary_spec, BoundarySpec)
    # Legacy view must still expose the pec_faces set for downstream code.
    assert "z_lo" in sim._boundary_spec.pec_faces()
    assert "z_lo" in sim._pec_faces  # legacy attribute derived by shim
    res = sim.run(n_steps=5)
    _assert_finite_probe(res)


def test_crossval_07_inverse_smoke():
    """07_inverse_design_demo.py — boundary=BoundarySpec.uniform('pec').
    The script later reads back sim._boundary (a scalar string, via the
    shim's legacy-view) to pass through _run(); that pattern still works
    after BoundarySpec migration. This test pins the shim equivalence."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.06, 0.02, 0.02),
        boundary=BoundarySpec.uniform("pec"),
    )
    sim.add_source((0.01, 0.01, 0.01), "ez")
    sim.add_probe((0.01, 0.01, 0.01), "ez")
    sim.add_probe((0.05, 0.01, 0.01), "ez")
    assert isinstance(sim._boundary_spec, BoundarySpec)
    # Shim: sim._boundary stays a scalar for legacy _run(boundary=...) call.
    assert sim._boundary == "pec"
    res = sim.run(n_steps=5)
    _assert_finite_probe(res)


def test_crossval_08_progressive_smoke():
    """08_progressive_inverse_design.py — default became explicit
    boundary=BoundarySpec.uniform('cpml')."""
    sim = Simulation(
        freq_max=12e9,
        domain=(0.024, 0.024, 0.024),
        dx=1.0e-3,
        boundary=BoundarySpec.uniform("cpml"),
        cpml_layers=6,
    )
    sim.add_source((0.012, 0.012, 0.004), "ez")
    sim.add_probe((0.012, 0.012, 0.020), "ez")
    assert isinstance(sim._boundary_spec, BoundarySpec)
    res = sim.run(n_steps=5)
    _assert_finite_probe(res)
