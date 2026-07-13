"""Cavity ring-down -- choose the expected mode before reading its frequency.

A closed rectangular PEC cavity stores electromagnetic energy.  With vacuum
inside and no resistive load, the field rings forever because there is nowhere
for the energy to go.  That is the physics.  A frequency extracted from a
finite record is meaningful, but its reported Q is only a window-length
number, not the physical Q of this lossless cavity.

The textbook cavity frequencies are

``f_mnp = (c/2)*sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)``.

This tutorial excites several modes, prints every mode returned by
``harminv()``, and selects the one nearest the analytic TE101 frequency.
Source and probe positions weight each field pattern differently, so the
loudest mode is not necessarily the requested mode.

The simulation is run with a short record and then a record four times longer.
Frequency resolution scales as one over the record length.  Long records are
now inexpensive to analyze because ``decimate='auto'`` is the harminv default
for oversampled, band-limited data.

Run as::

    python examples/tutorials/resonance_harminv.py
"""

from __future__ import annotations

import numpy as np

from rfx import GaussianPulse, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.harminv import HarminvMode, harminv


C0 = 299_792_458.0
A = 24.0e-3
B = 12.0e-3
D = 36.0e-3
DX = 0.75e-3
FREQ_MAX = 12.0e9

M, N, P = 1, 0, 1
F_TE101 = (C0 / 2.0) * np.sqrt((M / A) ** 2 + (N / B) ** 2 + (P / D) ** 2)

SHORT_STEPS = 600
LONG_STEPS = 4 * SHORT_STEPS


def build_cavity() -> Simulation:
    """Build the closed vacuum cavity, broadband source, and field probe."""
    sim = Simulation(
        freq_max=FREQ_MAX,
        domain=(A, B, D),
        dx=DX,
        boundary=BoundarySpec.uniform("pec"),
    )

    # These off-centre positions couple to TE101 and TE102.  TE102 is made a
    # little louder on purpose, so amplitude order cannot choose TE101 for us.
    sim.add_source(
        (0.37 * A, 0.41 * B, 0.20 * D),
        component="ey",
        waveform=GaussianPulse(f0=F_TE101, bandwidth=0.9),
    )
    sim.add_probe((0.63 * A, 0.58 * B, 0.80 * D), component="ey")
    return sim


def extract_modes(result) -> list[HarminvMode]:
    """Remove the drive portion and analyze the remaining ring-down."""
    if result.dt is None:
        raise RuntimeError("Simulation result did not include its timestep")

    trace = np.asarray(result.time_series)[:, 0]
    ring_down = trace[len(trace) // 4 :]
    ring_down = ring_down - np.mean(ring_down)
    return harminv(
        ring_down,
        float(result.dt),
        0.70 * F_TE101,
        1.50 * F_TE101,
        min_Q=1.0,
        max_modes=12,
        # decimate="auto" is the default; it keeps the full time span while
        # removing samples that the band-limited analysis does not need.
    )


def print_mode_list(label: str, modes: list[HarminvMode]) -> None:
    """Print every returned frequency, Q, and amplitude."""
    print(f"{label} full mode list:")
    for index, mode in enumerate(modes, start=1):
        print(
            f"  {index}: f={mode.freq / 1e9:.6f} GHz, "
            f"Q={mode.Q:.6g}, amplitude={mode.amplitude:.6e}"
        )


def nearest_te101(modes: list[HarminvMode]) -> HarminvMode:
    """Return the extracted mode nearest the textbook TE101 value."""
    if not modes:
        raise RuntimeError("Harminv returned no cavity modes")
    return min(modes, key=lambda mode: abs(mode.freq - F_TE101))


def relative_error_percent(mode: HarminvMode) -> float:
    """Return absolute frequency error as a percentage."""
    return 100.0 * abs(mode.freq - F_TE101) / F_TE101


def main() -> None:
    sim = build_cavity()

    # Expect "All checks passed" and no lossless-dielectric advisory.  This is
    # a vacuum cavity with closed PEC walls.  That advisory would appear for a
    # perfectly lossless dielectric placed in an open CPML model, where an
    # idealized material can make a resonance Q misleading.
    report = sim.preflight()
    loss_advisory = bool(report.by_code("lossless_q"))
    if report:
        raise RuntimeError("Vacuum cavity has unexpected preflight advisories")
    print(f"Vacuum-cavity loss advisory observed: {loss_advisory}")

    short_result = sim.run(
        n_steps=SHORT_STEPS,
        compute_s_params=False,
        skip_preflight=True,
    )
    long_result = sim.run(
        n_steps=LONG_STEPS,
        compute_s_params=False,
        skip_preflight=True,
    )

    short_modes = extract_modes(short_result)
    long_modes = extract_modes(long_result)
    print_mode_list("Short record", short_modes)
    print_mode_list("Long record", long_modes)

    # harminv returns amplitude order, strongest first.  We deliberately use
    # analytic proximity instead: source and probe placement decide loudness.
    short_te101 = nearest_te101(short_modes)
    long_te101 = nearest_te101(long_modes)
    short_error = relative_error_percent(short_te101)
    long_error = relative_error_percent(long_te101)

    print(f"Analytic TE101 frequency: {F_TE101 / 1e9:.6f} GHz")
    print(f"Short-record TE101 frequency: {short_te101.freq / 1e9:.6f} GHz")
    print(f"Short-record TE101 error: {short_error:.6f}%")
    print(f"Long-record TE101 frequency: {long_te101.freq / 1e9:.6f} GHz")
    print(f"Long-record TE101 error: {long_error:.6f}%")
    print(f"Long/short record-length ratio: {LONG_STEPS / SHORT_STEPS:.1f}")
    print("Mode selection: nearest analytic frequency, not strongest amplitude")
    print("Harminv sampling: decimate='auto' is the default")
    print(
        "Lossless-cavity Q note: energy is conserved, so the printed Q values "
        "depend on record length and are not physical."
    )


if __name__ == "__main__":
    main()
