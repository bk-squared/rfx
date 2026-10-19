"""Owned volume sums and plain forward result propagation."""
import json
from time import perf_counter

import jax
from tests._multi_device_scene import host_devices
import numpy as np
import pytest

from rfx.current_moments import current_moment_far_field
from tests.unit.runners.test_distributed_record_planes import model, add_planes
from rfx.probes.probes import flux_spectrum


@pytest.mark.parametrize("n_devices", [2, 3])
@pytest.mark.parametrize("entry", ["run", "forward"])
def test_moments_and_forward(n_devices, entry, tmp_path):
    sim, grid, seam = model("cpml", n_devices)
    sim.add_current_moment_monitor((3e-3, 2e-3, 2e-3), (9e-3, 7e-3, 6e-3),
                                   block_size=2e-3, freqs=[3e9, 7e9])
    add_planes(sim, grid, seam)
    sim.add_flux_monitor(axis="x", coordinate=(seam - grid.pad_x_lo) * grid.dx,
                         freqs=[3e9, 7e9], name="seam")
    start = perf_counter()
    run = getattr(sim, entry)
    reference = run(n_steps=40, skip_preflight=True)
    actual = run(n_steps=40, skip_preflight=True, devices=host_devices(n_devices),
                 **({"distributed": True} if entry == "forward" else {}))
    jax.block_until_ready(actual.current_moment_data)
    theta, phi = np.linspace(0, np.pi, 9), np.linspace(0, 2 * np.pi, 11)
    expected_ff = current_moment_far_field(reference, theta, phi)
    actual_ff = current_moment_far_field(actual, theta, phi)
    comparisons = {"accumulator": (actual.current_moment_data[0], reference.current_moment_data[0]),
                   "compensation": (actual.current_moment_data[1], reference.current_moment_data[1]),
                   "E_theta": (actual_ff.E_theta, expected_ff.E_theta),
                   "E_phi": (actual_ff.E_phi, expected_ff.E_phi)}
    measurements = {}
    for name, (a, b) in comparisons.items():
        a, b = np.asarray(a), np.asarray(b)
        delta = float(np.max(np.abs(a.astype(np.complex128) - b.astype(np.complex128))))
        peak = float(np.max(np.abs(b)))
        measurements[name] = dict(max_abs=delta, peak=peak, relative=delta / peak if peak else delta)
    report = dict(boundary="cpml", devices=n_devices, grid_shape=grid.shape, n_steps=40,
                  entry=entry, seconds=perf_counter() - start, moments=measurements)
    (tmp_path / "measurements.json").write_text(json.dumps(report))
    print(report)
    for name, values in measurements.items():
        assert values["max_abs"] <= 1e-4 * values["peak"], (name, values)
    for name, expected in reference.dft_planes.items():
        assert np.array_equal(actual.dft_planes[name].accumulator, expected.accumulator), name
    for component in ("e1_dft", "e2_dft", "h1_dft", "h2_dft"):
        assert np.array_equal(getattr(actual.flux_monitors["seam"], component),
                              getattr(reference.flux_monitors["seam"], component)), component
    assert np.array_equal(flux_spectrum(actual.flux_monitors["seam"]),
                          flux_spectrum(reference.flux_monitors["seam"]))


def _weight_ownership(layout, monitor):
    ratios = {}
    for name in ("w_ex", "w_ey", "w_ez"):
        expected = np.sum(np.asarray(getattr(monitor, name)), axis=1)
        actual = sum(np.sum(np.asarray(getattr(part, name)), axis=1) for part in layout.parts)
        ratios[name] = float(np.sum(actual[0]) / np.sum(expected[0]))
        delta, peak = np.max(np.abs(actual - expected)), np.max(np.abs(expected))
        assert delta <= 1e-4 * peak, ("moment weight ownership", name, ratios[name], float(delta), float(peak))
    return ratios


def test_moment_weights_not_cut_mutation(tmp_path):
    from jax.sharding import Mesh
    from rfx.runners._distributed_records import MomentRecords
    sim, grid, seam = model("cpml", 3)
    sim.add_current_moment_monitor((3e-3, 2e-3, 2e-3), (9e-3, 7e-3, 6e-3),
                                   block_size=2e-3, freqs=[3e9, 7e9])
    native = sim.run(n_steps=4, skip_preflight=True)
    layout = MomentRecords(Mesh(np.array(host_devices(3)), ("x",)),
                           native.current_moment_monitor, seam, native.state.ex.dtype)
    ratios = _weight_ownership(layout, native.current_moment_monitor)
    # Remove the cuts on the actual monitor weights. The ownership invariant
    # is checked before stepping, where incompatible array shapes would hide
    # the duplicated volume this mutation is meant to expose.
    layout.parts = [part._replace(**{name: getattr(native.current_moment_monitor, name)
                                     for name in ("w_ex", "w_ey", "w_ez")}) for part in layout.parts]
    with pytest.raises(AssertionError, match="moment weight ownership") as failure:
        _weight_ownership(layout, native.current_moment_monitor)
    report = dict(normal_ratios=ratios, mutation="weights not cut", assertion=str(failure.value))
    (tmp_path / "measurements.json").write_text(json.dumps(report))
    print(report)


def test_moment_previous_e_mutation(monkeypatch, tmp_path):
    from rfx.runners import _distributed_records as records
    original = records.MomentRecords.update

    def wrong(self, buffer, state, dt, step, *, ranks, previous):
        return original(self, buffer, state, dt, step, ranks=ranks,
                        previous=self.prepare(state, ranks=ranks))
    monkeypatch.setattr(records.MomentRecords, "update", wrong)
    with pytest.raises(AssertionError, match="accumulator") as failure:
        test_moments_and_forward(2, "run", tmp_path)
    print(f"mutation=post-step E used as previous E: RED {failure.value}")
