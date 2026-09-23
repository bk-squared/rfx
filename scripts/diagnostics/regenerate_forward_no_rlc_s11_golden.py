#!/usr/bin/env python3
"""Regenerate tests/fixtures/golden_forward_no_rlc_s11.npy.

The golden is the |S11| vector of a bare 50-ohm lumped port in an empty CPML
box — the falsifier that a sim with NO lumped RLC element is unchanged through
``forward()``. Its values are extractor-convention-bearing, so they move when
the convention does.

It moved on 2026-09-21: a driven lumped port now reads its S11 from the
terminal V/I pair sampled after source injection, with the Yee half-step
current phase, instead of the passive port-branch algebra on a pre-injection
sample (scripts/diagnostics/lumped_port_known_load_line.py). Run this, and
write the printed before/after into the PR body.

Run with no arguments. Writes the .npy to the path above, relative to the
repository this script is in, and prints both vectors.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from rfx import GaussianPulse, Simulation  # noqa: E402

GOLDEN = (Path(__file__).resolve().parents[2]
          / "tests" / "fixtures" / "golden_forward_no_rlc_s11.npy")
FREQS_HZ = np.array([3.0, 4.0, 5.0, 6.0, 7.0]) * 1e9


def main() -> None:
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 15,
                     boundary="cpml", cpml_layers=6)
    sim.add_port(position=(0.0093, 0.0093, 0.0093), component="ez",
                 impedance=50.0, waveform=GaussianPulse(f0=5e9, bandwidth=0.9))
    got = np.asarray(sim.forward(port_s11_freqs=FREQS_HZ, n_steps=1200).s_params)

    if GOLDEN.exists():
        old = np.load(GOLDEN)
        print(f"before  {np.array2string(old, precision=6)}")
        print(f"before |S11| {np.array2string(np.abs(old), precision=6)}")
        if old.shape == got.shape:
            print(f"max|after - before| = {np.max(np.abs(got - old)):.6f}")
    print(f"after   {np.array2string(got, precision=6)}")
    print(f"after  |S11| {np.array2string(np.abs(got), precision=6)}")

    np.save(GOLDEN, got)
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    main()
