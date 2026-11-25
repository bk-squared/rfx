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

from rfx import GaussianPulse, Simulation


def build_simulation() -> Simulation:
    """Build the 20 mm PEC box with its source and probe — no time stepping.

    Separated from ``main()`` so the #737 example-fidelity gate
    (``tests/contracts/test_example_fidelity_contract.py``) can build this
    example and pin its preflight/realization output without solving it.
    ``main()`` calls it, so what the gate builds is what running the script
    runs.
    """
    sim = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.02),
        dx=2e-3,
        boundary="pec",
    )

    sim.add_source(
        (0.01, 0.01, 0.01),
        "ez",
        waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
        amplitude_kind="current",
    )

    sim.add_probe((0.014, 0.01, 0.01), "ez")

    return sim


def main() -> None:
    t_start = time.time()

    sim = build_simulation()

    # docs-run-start
    preflight = sim.preflight()
    print(preflight.format())
    preflight.raise_for_failure()
    n_steps = 120
    result = sim.run(n_steps=n_steps, compute_s_params=False)
    print(result.time_series.shape)
    # docs-run-end

    trace = np.asarray(result.time_series)[:, 0]
    grid_shape = result.grid.shape

    elapsed = time.time() - t_start

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
