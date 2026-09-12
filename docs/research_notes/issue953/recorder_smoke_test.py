"""Synthetic backend test of the recorder; never an RF result."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import record_falsifiers as recorder


def main(out):
    original_compute = recorder.rfx.Simulation.compute_msl_s_matrix
    original_preflight = recorder.rfx.Simulation.preflight
    returned = []

    def fake_compute(self, **kwargs):
        assert kwargs["n_freqs"] == 100 and kwargs["num_periods"] == 20.0
        f = np.linspace(.7e9, 7e9, 100)
        theta = np.pi / 2 * f / 3.679e9
        s = np.zeros((2, 2, 100), dtype=np.complex64)
        s[0, 1] = s[1, 0] = 2 / (2 + 1j * np.tan(theta))
        raw = s * 1.02
        index = len(returned)
        settling = ([-80., -81.], [-40., -40.], [np.nan, -80.])[index]
        if index == 2:
            raw[0, 0, 0] = np.nan
        result = SimpleNamespace(
            freqs=f, S=s, S_raw=raw, Z0=np.full((2, 100), 50.), beta=np.ones(100),
            settling_db=np.array(settling), reliable=np.ones(100, dtype=bool),
            reference_impedances=np.array([47.9, 47.9]), cond_a=np.ones(100),
            passivity_correction=np.full(100, .02),
            beta_railed=np.full((2, 100), index == 1), assembly="synthetic_wave_solve",
        )
        np.savez(kwargs["raw_3probe_dump_path"], synthetic_marker=True)
        returned.append(result)
        return result

    recorder.rfx.Simulation.compute_msl_s_matrix = fake_compute
    recorder.rfx.Simulation.preflight = lambda *a, **k: None
    try:
        rc = recorder.main(out)
    finally:
        recorder.rfx.Simulation.compute_msl_s_matrix = original_compute
        recorder.rfx.Simulation.preflight = original_preflight
    assert len(returned) == 3
    for label, expected in zip(recorder.LABELS, returned):
        with np.load(out / f"{label}-result.npz") as data:
            for name in ("S", "S_raw", "reference_impedances", "settling_db", "beta_railed", "assembly"):
                value = np.asarray(getattr(expected, name))
                assert data[name].shape == value.shape and data[name].dtype == value.dtype
                assert data[name].tobytes() == value.tobytes(), name
    observations = json.loads((out / "observations.json").read_text())
    assert observations[0]["settling_screen_pass"]
    assert observations[1]["settling_status_by_drive"] == ["pass", "pass"]
    assert observations[1]["settling_screen_pass"]  # exact -40 dB is inclusive
    assert observations[2]["settling_status_by_drive"] == ["absent", "pass"]
    assert not observations[2]["settling_screen_pass"]
    assert observations[0]["raw_max_coherent_power_gain"] > 1
    assert observations[2]["nonfinite_frequency_count"] == 1
    assert observations[2]["raw_max_coherent_power_gain"] is None
    outcome = json.loads((out / "outcome.json").read_text())
    assert outcome["complete"] and not outcome["all_settling_screens_pass"]
    summary = dict(scope="SYNTHETIC FIELD BACKEND ONLY; no FDTD", calls=3,
                   result_array_bytes_identical=True, exact_minus40_passes=True,
                   absent_witness_not_accepted=True, nonfinite_raw_bin_not_filtered=True,
                   raw_gain_not_clipped=True, producer_return_code=rc)
    (out / "smoke-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
