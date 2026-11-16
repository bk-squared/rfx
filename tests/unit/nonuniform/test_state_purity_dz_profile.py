"""Execution order leaves graded probe records and public mesh reports unchanged."""
import sys

import numpy as np

from rfx import GaussianPulse, Simulation


def _dx_only_nu_sim():
    sim = Simulation(
        freq_max=5e9, domain=(0.020, 0.010, 0.0043), dx=1e-3,
        boundary="cpml", cpml_layers=4, dx_profile=np.full(20, 1e-3),
    )
    sim.add_source((0.008, 0.004, 0.001), "ez", waveform=GaussianPulse(f0=3e9))
    sim.add_probe((0.009, 0.004, 0.001), "ez")
    return sim


def _byte_equal(row, actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    same_layout = a.shape == b.shape and a.dtype == b.dtype
    difference = float(np.max(np.abs(a - b))) if same_layout else float("inf")
    print(f"{row}: max_abs={difference:.12g}, bar=0 (byte equality), "
          f"expected_shape={b.shape}, measured_shape={a.shape}", file=sys.stderr)
    assert same_layout and a.tobytes() == b.tobytes(), (
        f"{row}: entry order changed output; max_abs={difference}, bar=0")


def _same_result(row, actual, expected):
    assert actual.grid is not None and expected.grid is not None
    print(f"{row}: steps=5, expected_mesh={expected.grid.shape}, "
          f"measured_mesh={actual.grid.shape}, expected_nz={expected.grid.nz}, "
          f"measured_nz={actual.grid.nz}, dz={np.asarray(actual.grid.dz).tolist()}", file=sys.stderr)
    _byte_equal(row + "/nz", actual.grid.nz, expected.grid.nz)
    _byte_equal(row + "/dz", actual.grid.dz, expected.grid.dz)
    _byte_equal(row + "/dz_f64", actual.grid.dz_f64, expected.grid.dz_f64)
    _byte_equal(row + "/trace", actual.time_series, expected.time_series)


def test_entry_order_preserves_graded_records_and_mesh_summary():
    # Existing run/forward bitwise bar: tests/contracts/test_run_forward_one_pipeline.py:809-810.
    # The entry-order spec also explicitly requires byte equality.
    sim = _dx_only_nu_sim()
    before = sim.artifact_report(n_steps=5)["mesh"]
    reference = _dx_only_nu_sim().run(n_steps=5)
    fresh_forward = _dx_only_nu_sim().forward(n_steps=5)
    trace = np.asarray(reference.time_series)
    print(f"reference: peak={np.max(np.abs(trace)):.12g}, steps=5, "
          f"mesh={reference.grid.shape}", file=sys.stderr)
    assert np.isfinite(trace).all() and np.any(trace), "entry-order reference trace must be finite and nonzero"
    _same_result("fresh-forward/fresh-run", fresh_forward, reference)
    # Two run calls expose runner persistence; forward twice also exposes a
    # mutation in its grid builder when the first call's returned grid is intact.
    for row, entry in (("first-run", "run"), ("second-run", "run"),
                       ("forward-after-run", "forward"), ("second-forward", "forward")):
        actual = getattr(sim, entry)(n_steps=5)
        _same_result(row + "/fresh-run", actual, reference)
        if entry == "forward":
            _same_result(row + "/fresh-forward", actual, fresh_forward)
        after = sim.artifact_report(n_steps=5)["mesh"]
        changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
        print(f"{row}/artifact-mesh: changed_keys={changed}, count={len(changed)}, bar=0, "
              f"before_profiles={before['profiles']}, after_profiles={after['profiles']}", file=sys.stderr)
        assert after == before, f"{row}: artifact mesh summary changed: {changed}; bar=0"
