"""A matched coax sheet preserves the physical end-capacitance floor.

The vacuum cell formerly written by the annular resistor removed a shunt
capacitance, cancelling part of the end reflection. The independent reference is
31.49 fF across a 48.5914 ohm line: |Gamma| = x / sqrt(4 + x**2), x = omega C Z.
Its 11-bin 4–6 GHz median is -32.386 dB; the declared mesh allowance is 0.717 dB.

The leader's resumed layout uses an 8 x 8 x 30 mm board, 0.4 mm cells, all
12 default probes and the default end offset, with a 3000-step record.
"""
from __future__ import annotations

import time

import numpy as np

from rfx.api import Simulation


def test_matched_sheet_keeps_the_physical_end_capacitance():
    start = time.monotonic()
    freqs = np.linspace(4e9, 6e9, 11)
    x = 2 * np.pi * freqs * 31.49e-15 * 48.5914
    expected_magnitude = x / np.sqrt(4 + x**2)
    expected_floor = 20 * np.log10(np.median(expected_magnitude))
    sim = Simulation(freq_max=10e9, domain=(0.008, 0.008, 0.030),
                     dx=0.0004, boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.015),
                        pin_radius=0.000635, outer_radius=0.002055)
    result = sim.compute_coaxial_line_reflection(
        termination="matched", n_steps=3000, freqs=freqs)
    magnitude = np.abs(np.asarray(result.s11))
    floor = float(20 * np.log10(np.median(magnitude)))
    slope = float(20 * np.log10(magnitude[-1] / magnitude[0]))
    print(f"F={floor:.9f} dB; F_cont={expected_floor:.9f} dB; "
          f"S={slope:.9f} dB; settling={result.settling_db:.9f} dB; "
          f"witness={result.settling_witness['status']}; "
          f"wall={time.monotonic() - start:.3f} s")
    assert result.status == "passed", result.status
    assert result.settling_witness["status"] == "pass", result.settling_witness
    assert result.settling_db <= -40.0, result.settling_db
    assert np.all(np.isfinite(magnitude)) and np.max(magnitude) <= 1.0
    assert abs(floor - expected_floor) <= 0.717, (floor, expected_floor)
    assert abs(slope - 3.522) <= 0.2, slope
