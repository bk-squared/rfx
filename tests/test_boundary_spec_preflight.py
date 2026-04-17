"""T7-D: preflight compatibility with explicit BoundarySpec inputs.

The T7-B shim keeps self._boundary / self._pec_faces / self._periodic_axes
in sync with self._boundary_spec, so all existing preflight checks read
the same values whether the user passed the legacy triad or a
BoundarySpec. These tests pin that equivalence for three representative
per-face mix scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec


@pytest.fixture(autouse=True)
def _silence_legacy_warnings(monkeypatch):
    # The explicit-BoundarySpec path itself emits no DeprecationWarning;
    # this fixture ensures nothing else surfaces one either.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def test_patch_antenna_mix_preflight_no_error():
    """pec z_lo + cpml z_hi + cpml on x/y — the classic patch-antenna
    boundary configuration must build and run cleanly."""
    spec = BoundarySpec(
        x="cpml",
        y="cpml",
        z=Boundary(lo="pec", hi="cpml"),
    )
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
        boundary=spec,
    )
    sim.add_source((0.005, 0.005, 0.0025), "ez")
    sim.add_probe((0.005, 0.005, 0.003), "ez")
    # preflight + run must complete without unexpected errors.
    res = sim.run(n_steps=20)
    assert np.asarray(res.time_series).shape[0] == 20
    assert sim._boundary_spec.pec_faces() == {"z_lo"}


def test_periodic_cpml_mix_preflight_suppresses_cpml_on_periodic_axis():
    """periodic x + cpml y + cpml z — the #68 fix still applies: CPML
    is not allocated on the periodic axis."""
    spec = BoundarySpec(x="periodic", y="cpml", z="cpml")
    sim = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
        boundary=spec,
    )
    # periodic must be reflected in the legacy view for downstream code.
    assert sim._periodic_axes == "x"
    # Allocator-level check: the Grid built from this sim must have no
    # CPML on the periodic x axis (issue #68 fix via BoundarySpec shim).
    grid = sim._build_grid()
    assert "x" not in grid.cpml_axes, (
        f"CPML was allocated on the periodic x axis; got cpml_axes="
        f"{grid.cpml_axes!r}"
    )
    assert grid.pad_x == 0, f"expected pad_x=0 on periodic axis, got {grid.pad_x}"
    assert grid.pad_y > 0 and grid.pad_z > 0, (
        f"non-periodic axes must retain CPML padding "
        f"(pad_y={grid.pad_y}, pad_z={grid.pad_z})"
    )
    sim.add_source((0.005, 0.005, 0.005), "ez")
    sim.add_probe((0.005, 0.005, 0.006), "ez")
    res = sim.run(n_steps=20)
    assert np.all(np.isfinite(np.asarray(res.time_series)))


def test_explicit_uniform_cpml_matches_legacy_scalar_cpml():
    """BoundarySpec.uniform('cpml') and boundary='cpml' produce the same
    canonical self._boundary_spec. Pins T7-B round-trip equivalence."""
    spec = BoundarySpec.uniform("cpml")
    sim_explicit = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
        boundary=spec,
    )
    sim_legacy = Simulation(
        freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
        boundary="cpml",
    )
    assert sim_explicit._boundary_spec == sim_legacy._boundary_spec
    assert sim_explicit._boundary == sim_legacy._boundary
    assert sim_explicit._pec_faces == sim_legacy._pec_faces
    assert sim_explicit._periodic_axes == sim_legacy._periodic_axes


def test_absorber_type_view_matches_legacy_boundary():
    """sim._boundary_spec.absorber_type lines up with the legacy
    scalar self._boundary when there is any absorbing face."""
    for tok in ("cpml", "upml"):
        sim = Simulation(
            freq_max=10e9, domain=(0.01, 0.01, 0.005), dx=0.5e-3,
            boundary=tok,
        )
        assert sim._boundary_spec.absorber_type == tok
        assert sim._boundary == tok
