"""Cell 2: graded CPML run with poles and no ports versus one device."""
import numpy as np
import pytest

from tests._multi_device_scene import host_devices
from tests._one_device_parity import STEPS, field_probes, perturb_cell, record, scene, trace_metrics, assert_multi_device

pytestmark = pytest.mark.distributed


@pytest.mark.parametrize("n_devices", [2, 3])
@pytest.mark.parametrize("kind", ["debye", "lorentz"])
def test_graded_cpml_poles_match_one_device(kind, n_devices):
    devices = host_devices(n_devices)
    model = scene(kind, boundary="cpml", graded=True)
    labels = field_probes(model, graded=True)
    one = model.run(n_steps=STEPS, skip_preflight=True)
    rows = []
    for mutation in ("none", "drop-pole", "eps-seam-plus-1-percent"):
        actual_model = scene(kind, boundary="cpml", graded=True, drop=mutation == "drop-pole")
        if mutation == "eps-seam-plus-1-percent":
            perturb_cell(actual_model, n_devices, graded=True)
        assert field_probes(actual_model, graded=True) == labels
        multi = actual_model.run(n_steps=STEPS, skip_preflight=True, devices=devices)
        assert_multi_device(actual_model, multi)
        assert np.asarray(multi.time_series).shape[0] == STEPS
        rows.append(record(dict(cell=2, model=kind, boundary="cpml", devices=n_devices, steps=STEPS),
                           mutation, trace_metrics(one.time_series, multi.time_series,
                                                   float(one.dt), labels)))
    assert all(not row["passed"] for row in rows[1:]), ("falsifier not resolved", rows)
    assert rows[0]["passed"], rows[0]
