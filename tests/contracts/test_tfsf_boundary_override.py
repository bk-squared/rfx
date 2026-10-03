"""Both uniform entry points pass the TF/SF boundary policy to the solver.

Capture the solver inputs before stepping, including a deliberately changed
Method-B policy: a helper that either caller ignores must fail this contract.
"""

import pytest

from rfx import Simulation
from rfx import simulation
from rfx.sources import tfsf


@pytest.mark.parametrize("entry", ["run", "forward"])
@pytest.mark.parametrize("kind,expected", [
    ("normal", ((False, True, True), "x")),
    ("bloch", ((False, True, True), "x")),
    ("methodB", ((False, False, True), "xy")),
    ("closed", ((False, False, False), "xyz")),
])
def test_uniform_entry_points_use_tfsf_boundary_policy(monkeypatch, entry, kind, expected):
    _capture_policy(monkeypatch, entry, kind, expected)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_methodb_helper_change_reaches_both_entry_points(monkeypatch, entry):
    original = tfsf.tfsf_boundary_flags

    def changed(cfg):
        if tfsf.is_tfsf_methodB(cfg):
            return (False, True, False), "xz"
        return original(cfg)

    monkeypatch.setattr(tfsf, "tfsf_boundary_flags", changed)
    _capture_policy(monkeypatch, entry, "methodB", ((False, True, False), "xz"))


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
    # Preserve forward's existing wall policy as well as the shared flags.
    if entry == "forward":
        assert captured["pec_axes"] == (None if kind == "closed" else "")
