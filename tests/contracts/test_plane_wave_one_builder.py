"""Declaration-based plane waves share one builder, including flux owners."""
import ast
from pathlib import Path

import numpy as np
import pytest

from rfx import Simulation
from rfx.measurement import flux_weights
from rfx.measurement.plan import measurement_plan


def test_no_plane_wave_builder_bypass():
    root = Path(__file__).resolve().parents[2]
    offenders = []
    for path in sorted((root / "rfx").rglob("*.py")):
        # compute_rcs builds an open source from calculator arguments, not a
        # Simulation declaration; it never builds a closed box.
        if path.relative_to(root).as_posix() in {"rfx/sources/tfsf.py", "rfx/rcs.py"}:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        names = {"init_tfsf"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.update(alias.asname or alias.name for alias in node.names
                             if alias.name == "init_tfsf")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if ((isinstance(fn, ast.Name) and fn.id in names)
                    or (isinstance(fn, ast.Attribute) and fn.attr == "init_tfsf")):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, "plane-wave builder bypasses: " + ", ".join(offenders)


@pytest.mark.parametrize("declaration,expected_periodic", [
    pytest.param(dict(closed_box=True, polarization="ez"),
                 (False, False, False), id="closed-normal-ez"),
    pytest.param(dict(closed_box=True, polarization="s", theta_deg=30.0, phi_deg=0.0),
                 (False, False, False), id="closed-oblique-s"),
    pytest.param(dict(closed_box=False, polarization="ez"),
                 (False, True, True), id="open-normal-ez"),
])
def test_measurement_plan_flux_owner_periodic(declaration, expected_periodic, monkeypatch):
    sim = Simulation(freq_max=15e9, domain=(0.036, 0.024, 0.024), dx=1e-3,
                     boundary="cpml", cpml_layers=8)
    sim.add_tfsf_source(f0=8e9, margin=4, direction="+x", **declaration)
    sim.add_flux_monitor(axis="x", coordinate=0.0305, freqs=np.array([6e9, 8e9]))
    observed = []
    make_monitor = flux_weights.grid_flux_monitor

    def record_periodic(*args, **kwargs):
        observed.append(kwargs["periodic"])
        return make_monitor(*args, **kwargs)

    monkeypatch.setattr(flux_weights, "grid_flux_monitor", record_periodic)
    plan = measurement_plan(sim, n_steps=60)
    owner, = plan.owners
    assert owner.kind == "flux"
    assert owner.channels
    assert observed == [expected_periodic]
