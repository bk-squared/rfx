"""hello_world.py — the simplest possible rfx simulation.

Run it::

    python examples/quickstart/hello_world.py

This is the canonical first thing to run after installing rfx. It builds a
tiny empty box, drops a pulse source in the middle, watches one field point,
steps the simulation a few dozen times, and prints a short summary. It is
deliberately small so it finishes in well under ten seconds on a laptop CPU —
the point is to see rfx actually run, not to model a real device.

Everything below uses only the stable public API
(``rfx.Simulation`` + ``GaussianPulse``).
"""

from __future__ import annotations

import time

import numpy as np

from rfx import Simulation
from rfx.sources.sources import GaussianPulse


def main() -> None:
    t_start = time.time()

    # 1. Build the domain.
    #    A 20 mm cube of empty space (vacuum). `freq_max` tells rfx the
    #    highest frequency we care about; here 10 GHz. `dx` is the cell size
    #    (2 mm), so the box is about 10 cells on each side — tiny on purpose.
    #    `boundary="pec"` wraps the box in perfect electric conductor walls
    #    (a closed metal box), which is the cheapest boundary to simulate.
    sim = Simulation(
        freq_max=10e9,            # 10 GHz upper frequency of interest
        domain=(0.02, 0.02, 0.02),  # 20 mm x 20 mm x 20 mm, in metres
        dx=2e-3,                  # 2 mm cells -> ~10 cells per axis
        boundary="pec",           # closed perfectly-conducting box
    )

    # 2. Add a source.
    #    A soft point source at the centre of the box that injects an Ez
    #    (vertical electric field) pulse. The `GaussianPulse` is a short
    #    broadband "ping" centred at 5 GHz — like tapping the box to see how
    #    it rings.
    sim.add_source(
        (0.01, 0.01, 0.01),                  # centre of the box, in metres
        "ez",                                # drive the z-component of E
        waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
    )

    # 3. Add a probe.
    #    A point "microphone" that records the Ez field at one location over
    #    time, two cells away from the source so we see the pulse arrive.
    sim.add_probe((0.014, 0.01, 0.01), "ez")

    # 4. Run the time-stepping.
    #    Step the FDTD update enough times for the pulse to reach the probe
    #    and ring inside the little PEC box, so the trace shows a peak that is
    #    distinct from its final value. We pass `compute_s_params=False`
    #    because this toy run has no ports — we just want raw field data.
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
    print("examples/crossval/ for real RF device validation cases.")


if __name__ == "__main__":
    main()
