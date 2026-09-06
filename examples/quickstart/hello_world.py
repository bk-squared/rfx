"""hello_world.py — the smallest rfx run: an install check.

Run it::

    python examples/quickstart/hello_world.py

A 20 mm vacuum box with PEC walls, one Gaussian pulse source at the centre, one
Ez probe two cells away, 120 time steps, then a short summary. It finishes in
well under ten seconds on a laptop CPU. This checks the install; it does not
model a device.

Uses only the stable public API: ``rfx.Simulation`` and ``GaussianPulse``.
"""

from __future__ import annotations

import time

import numpy as np

from rfx import Simulation
from rfx.sources.sources import GaussianPulse


def main() -> None:
    t_start = time.time()

    # 1. Build the domain.
    #    A 20 mm vacuum cube. `freq_max` is the highest frequency of interest,
    #    here 10 GHz. `dx` is the cell size (2 mm), so the box is about 10
    #    cells per axis. `boundary="pec"` closes it with perfect electric
    #    conductor walls — the cheapest boundary, with no absorber to size.
    sim = Simulation(
        freq_max=10e9,            # 10 GHz upper frequency of interest
        domain=(0.02, 0.02, 0.02),  # 20 mm x 20 mm x 20 mm, in metres
        dx=2e-3,                  # 2 mm cells -> ~10 cells per axis
        boundary="pec",           # closed perfectly-conducting box
    )

    # 2. Add a source.
    #    A soft point source at the box centre driving Ez with a Gaussian
    #    pulse centred at 5 GHz: broadband enough to excite the box modes
    #    across the band.
    #    `amplitude_kind` fixes what the waveform amplitude means. "current"
    #    is a drive current in amperes, which is why the peak |Ez| printed
    #    below is large: 1 A into a 2 mm cell is a strong drive. "field" is a
    #    raw volts-per-metre increment per step instead. Pass one explicitly:
    #    1.8 warns when it is omitted, 1.9 requires it, and 2.0 makes
    #    "current" the default (#914).
    sim.add_source(
        (0.01, 0.01, 0.01),                  # centre of the box, in metres
        "ez",                                # drive the z-component of E
        waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
        amplitude_kind="current",              # amplitude is a current in
                                               # amperes (see step 2 note)
    )

    # 3. Add a probe.
    #    Records the Ez field at one point over time, two cells from the
    #    source so the pulse arrival is visible in the trace.
    sim.add_probe((0.014, 0.01, 0.01), "ez")

    # 4. Run the time-stepping.
    #    120 steps is enough for the pulse to reach the probe and ring inside
    #    the PEC box, so the trace peak is distinct from its final value.
    #    `compute_s_params=False` because this run has no ports: raw field
    #    data only.
    n_steps = 120
    result = sim.run(n_steps=n_steps, compute_s_params=False)

    # 5. Look at the result.
    #    `result.time_series` has shape (n_steps, n_probes). We have one
    #    probe, so column 0 is its recorded Ez trace.
    trace = np.asarray(result.time_series)[:, 0]
    grid_shape = result.grid.shape  # (nx, ny, nz) including boundary padding

    elapsed = time.time() - t_start

    # 6. Print a short, human-readable summary.
    print("rfx hello world")
    print("-" * 40)
    print(f"grid size      : {tuple(int(n) for n in grid_shape)} cells")
    print(f"time steps     : {n_steps}")
    print(f"probe samples  : {trace.shape[0]}")
    print(f"peak |Ez|      : {float(np.max(np.abs(trace))):.4e}")
    print(f"final Ez       : {float(trace[-1]):+.4e}")
    print(f"all finite     : {bool(np.all(np.isfinite(trace)))}")
    print(f"wall time      : {elapsed:.2f} s")
    print("-" * 40)
    print("If you see finite numbers above, rfx is working. Next: try")
    print("examples/tutorials/ — the ordered learning path (see examples/README.md).")


if __name__ == "__main__":
    main()
