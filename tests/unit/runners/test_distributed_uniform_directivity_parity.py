"""Cell 4: directivity and radiation pattern from each uniform public entry."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement.farfield import compute_far_field
from rfx.farfield import directivity, radiation_pattern
from tests._multi_device_scene import host_devices
from tests._one_device_parity import STEPS, perturb_cell, record, scene, assert_multi_device

pytestmark = pytest.mark.distributed


def model(boundary="cpml", *, drop=False):
    sim = scene("vacuum", boundary=boundary)
    if not drop:
        sim.add_source((.008, .005, .003), "ey", amplitude_kind="field",
                       waveform=lambda t: .7 * jnp.exp(-((t - 11e-12) / 4e-12) ** 2))
    sim.add_ntff_box((.001, .001, .001), (.011, .007, .007), freqs=[5e9, 10e9])
    return sim


def quantities(result):
    ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid,
                          np.linspace(0, np.pi, 19), np.linspace(0, 2*np.pi, 37))
    return {"directivity": directivity(ff), "radiation_pattern": radiation_pattern(ff)}


@pytest.mark.parametrize("n_devices", [2, 3])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_uniform_farfield_matches_one_device(entry, n_devices):
    devices = host_devices(n_devices)

    def evaluate(sim, selected=None):
        options = dict(n_steps=STEPS, skip_preflight=True)
        if entry == "forward":
            options.update(dict(checkpoint=False) if selected is None else
                           dict(distributed=True, devices=selected))
        elif selected is not None:
            options["devices"] = selected
        result = getattr(sim, entry)(**options)
        if selected is not None:
            assert_multi_device(sim, result)
        return result

    # Check the open-box assumption on both public entries and both device counts.
    for selected in (None, devices):
        with pytest.raises(ValueError, match="NTFF|ntff") as refused:
            evaluate(model("pec"), selected)
        record(dict(cell=4, entry=entry, boundary="pec", devices=1 if selected is None else n_devices),
               "admission", dict(refusal=str(refused.value)))
    one = quantities(evaluate(model()))
    assert all(np.isfinite(v).all() and np.any(v) for v in one.values())
    rows = []
    for mutation in ("none", "drop-source", "eps-seam-plus-1-percent"):
        actual_model = model(drop=mutation == "drop-source")
        if mutation == "eps-seam-plus-1-percent":
            perturb_cell(actual_model, n_devices)
        actual = quantities(evaluate(actual_model, devices))
        measurements = {}
        for name, expected in one.items():
            value = np.asarray(actual[name])
            delta = np.abs(value.astype(np.float64) - expected)
            first = np.argwhere(value != expected)
            index = tuple(int(v) for v in first[0]) if len(first) else None
            measurements[name] = dict(bit_equal=bool(np.array_equal(value, expected)), bar="array_equal",
                                      finite=bool(np.isfinite(value).all()), max_abs=float(delta.max()),
                                      first_index=index, one=float(expected[index]) if index else None,
                                      multi=float(value[index]) if index else None)
        rows.append(record(dict(cell=4, entry=entry, model="two-source-vacuum", boundary="cpml",
                                devices=n_devices, steps=STEPS), mutation,
                           dict(quantities=measurements, passed=all(
                               v["bit_equal"] and v["finite"] for v in measurements.values()))))
    assert all(not row["passed"] for row in rows[1:]), ("falsifier not resolved", rows)
    assert rows[0]["passed"], rows[0]
