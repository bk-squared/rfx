"""Both uniform entry points pass the TF/SF boundary policy to the solver.

Capture the solver inputs before stepping, including a deliberately changed
Method-B policy: a helper that either caller ignores must fail this contract.
"""

import pytest

from rfx import Simulation
from rfx import simulation
@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("kind,expected", [
    ("normal", ((False, True, True), "x")),
    ("bloch", ((False, True, True), "x")),
    ("methodB", ((False, False, True), "xy")),
    ("closed", ((False, False, False), "xyz")),
])
def test_uniform_entry_points_use_tfsf_boundary_policy(monkeypatch, entry, kind, expected):
    _capture_policy(monkeypatch, entry, kind, expected)


def _capture_policy(monkeypatch, entry, kind, expected):
    sim = Simulation(freq_max=30e9, domain=(0.0243, 0.0187, 0.0141),
                     dx=0.001, cpml_layers=4)
    sim.add_tfsf_source(
        f0=15e9, bandwidth=0.15, margin=3,
        angle_deg=20 if kind in ("bloch", "methodB") else 0,
        method="methodB" if kind == "methodB" else "bloch",
        closed_box=kind == "closed",
    )
    captured = {}

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        captured.update(kwargs)
        raise Captured

    monkeypatch.setattr(simulation, "run", capture)
    with pytest.raises(Captured):
        getattr(sim, entry)(n_steps=2, skip_preflight=True)
    assert (captured["periodic"], captured["cpml_axes"]) == expected
    # #1549: forward passes no wall selection of its own on any kind; the walls come from the realized
    # boundary record (resolve_wall_faces), which never walls a face the plane wave wraps.
    if entry == "forward":
        assert captured["pec_axes"] is None
