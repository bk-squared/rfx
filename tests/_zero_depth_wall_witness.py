"""A pulse reflected by a zero-layer absorber's backing on one/two devices."""
import json
from pathlib import Path
import sys
import warnings

import jax
import numpy as np

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from tests.contracts.path_equivalence.comparison import compare


def trace(nonuniform, distributed, explicit_pec=False):
    profiles = {"dz_profile": np.full(16, 1e-3)} if nonuniform else {}
    wall = Boundary("pec", "cpml") if explicit_pec else Boundary("cpml", "cpml", 0, 4)
    sim = Simulation(freq_max=15e9, domain=(0.025, 0.024, 0.016), dx=1e-3,
                     cpml_layers=4, boundary=BoundarySpec(x="cpml", y=wall, z="cpml"), **profiles)
    sim.add_source((0.007, 0.006, 0.010), "ez", amplitude_kind="current")
    sim.add_probe((0.007, 0.003, 0.010), "ez")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 64 steps include the first wall return; later cavity round trips accumulate rounding.
        result = sim.run(n_steps=64, skip_preflight=True,
                         devices=jax.devices("cpu")[:2] if distributed else None)
    return np.asarray(result.time_series[:, 0])


def measure():
    assert len(jax.devices("cpu")) == 2
    return {f"{mesh}_{devices}_{kind}": trace(mesh == "graded", devices == "two", kind == "pec")
            for mesh in ("uniform", "graded") for devices in ("one", "two")
            for kind in ("zero", "pec")}


def judge(traces, measurements):
    for mesh in ("uniform", "graded"):
        for devices in ("one", "two"):
            zero, pec = traces[f"{mesh}_{devices}_zero"], traces[f"{mesh}_{devices}_pec"]
            assert zero.dtype == np.float32 and np.max(np.abs(zero)) > 0
            # §7 Addendum 4: on every lane a zero-depth CPML face is the same
            # electric wall as a declared PEC face, bit for bit.
            np.testing.assert_array_equal(zero, pec, err_msg=f"{mesh} {devices} device(s)")
    # Across device counts only the uniform lane is judged at the cross-trace
    # bar: the defect was uniform-only, and the graded two-device seam rounding
    # already sits at the 9-ULP bar for a declared PEC face as well.
    reference = traces["uniform_one_zero"]
    actual = traces["uniform_two_zero"]
    compare(reference, actual, record="uniform_two_zero.probe", kind="step", measurements=measurements)
    compare(np.fft.rfft(reference), np.fft.rfft(actual),
            record="uniform_two_zero.probe_spectrum", kind="accumulated", measurements=measurements)


def main():
    traces = measure()
    measurements = []
    try:
        judge(traces, measurements)
    finally:
        Path(sys.argv[1]).write_text(json.dumps({
            "traces": {key: value.tolist() for key, value in traces.items()},
            "measurements": measurements}, indent=2) + "\n")


if __name__ == "__main__":
    main()
