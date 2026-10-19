"""Cell 1: uniform forward poles, lossy sheet and PMC versus one device."""
from unittest.mock import patch

import numpy as np
import pytest

from rfx import simulation
from rfx.runners import distributed_v2
from tests._multi_device_scene import host_devices
from tests._one_device_parity import (
    assert_multi_device,
    COMPONENTS, STEPS, field_probes, perturb_cell, record, scene, trace_metrics,
)

pytestmark = pytest.mark.distributed


def evaluate(sim, devices=None):
    """Observe the actual forward solve's final fields before result packaging."""
    states, solvers = [], []
    original_scan = simulation.run
    original_run = distributed_v2.run_distributed

    def scan(*args, **kwargs):
        # Forward normally discards the final state. Retain the same solve's
        # state for observation; this flag does not alter the time step.
        kwargs["return_state"] = True
        result = original_scan(*args, **kwargs)
        states.append(result.state)
        solvers.append("one_device")
        return result

    def run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        states.append(result.state)
        solvers.append("run_distributed")
        return result

    options = dict(checkpoint=False) if devices is None else dict(distributed=True, devices=devices)
    with patch.object(simulation, "run", scan), \
            patch.object(distributed_v2, "run_distributed", run):
        result = sim.forward(n_steps=STEPS, skip_preflight=True, **options)
    assert len(states) == 1
    assert solvers == ["one_device" if devices is None else "run_distributed"], solvers
    if devices is not None:
        assert_multi_device(sim, result)
    return result, {c: np.asarray(getattr(states[0], c)) for c in COMPONENTS}


@pytest.mark.parametrize("n_devices", [2, 3])
@pytest.mark.parametrize("kind", ["debye", "lorentz", "film", "pmc"])
def test_uniform_forward_matches_one_device(kind, n_devices):
    devices = host_devices(n_devices)
    model = scene(kind)
    exact = kind in ("film", "pmc")
    labels = field_probes(model, magnetic=exact)
    one, fields = evaluate(model)
    rows = []
    for mutation in ("none", "drop-input", "eps-seam-plus-1-percent"):
        actual_model = scene(kind, drop=mutation == "drop-input")
        if mutation == "eps-seam-plus-1-percent":
            perturb_cell(actual_model, n_devices)
        assert field_probes(actual_model, magnetic=exact) == labels
        multi, actual_fields = evaluate(actual_model, devices)
        measured = trace_metrics(one.time_series, multi.time_series, float(one.dt), labels, exact=exact)
        if exact:
            measured["fields"] = {
                c: dict(bit_equal=bool(np.array_equal(fields[c], actual_fields[c])),
                        max_abs=float(np.abs(actual_fields[c].astype(np.float64) - fields[c]).max()))
                for c in COMPONENTS}
            measured["passed"] &= all(v["bit_equal"] for v in measured["fields"].values())
        rows.append(record(dict(cell=1, model=kind, boundary="pmc-y-lo/pec" if kind == "pmc" else "pec",
                                devices=n_devices, steps=STEPS), mutation, measured))
    assert all(not row["passed"] for row in rows[1:]), ("falsifier not resolved", rows)
    assert rows[0]["passed"], rows[0]
