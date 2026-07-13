"""Short-dipole far field -- place the Huygens box before trusting a pattern.

One rule matters most for near-to-far-field work: keep every monitor face at
least half the shortest monitored wavelength from radiating or scattering
structures.  A closer face samples the reactive near field, so its pattern and
directivity may be inaccurate.

This tutorial first places a deliberately close box so ``preflight()`` can
show the warning.  It then moves the box beyond half a wavelength, runs one
small simulation, compares the result with the textbook short-dipole value,
and saves an E-plane cut.

Run as::

    python examples/tutorials/antenna_farfield_pattern.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfx import GaussianPulse, Simulation, compute_far_field, directivity


C0 = 299_792_458.0
F0 = 8.0e9
FREQ_MAX = 10.0e9
DOMAIN = 60.0e-3
DX = 1.5e-3
CPML_LAYERS = 6
N_STEPS = 400

CENTER = (DOMAIN / 2.0,) * 3
CLOSE_GAP = 3.0e-3
VALID_GAP = 19.5e-3
HALF_WAVELENGTH = C0 / (2.0 * F0)

OUTPUT_PATH = Path(__file__).with_name("output") / "short_dipole_e_plane.png"


def build_simulation() -> Simulation:
    """Build a small open-domain model of a z-directed short dipole."""
    sim = Simulation(
        freq_max=FREQ_MAX,
        domain=(DOMAIN, DOMAIN, DOMAIN),
        dx=DX,
        boundary="cpml",
        cpml_layers=CPML_LAYERS,
    )
    sim.add_source(
        CENTER,
        component="ez",
        waveform=GaussianPulse(f0=F0, bandwidth=0.5),
    )
    return sim


def save_e_plane_cut(far_field) -> None:
    """Save the phi=0 electric-field cut, normalized to its peak."""
    magnitude = np.abs(np.asarray(far_field.E_theta[0, :, 0]))
    normalized = magnitude / np.max(magnitude)
    magnitude_db = 20.0 * np.log10(np.maximum(normalized, 1.0e-2))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    ax.plot(np.degrees(far_field.theta), magnitude_db, color="tab:blue")
    ax.set(
        xlabel=r"Polar angle $\theta$ (degrees)",
        ylabel="Normalized |E-theta| (dB)",
        title="Short-dipole E-plane cut",
        xlim=(0.0, 180.0),
        ylim=(-40.0, 1.0),
    )
    ax.grid(True, alpha=0.3)
    fig.savefig(OUTPUT_PATH, dpi=130)
    plt.close(fig)


def main() -> None:
    sim = build_simulation()

    # At 8 GHz, half a wavelength is 18.74 mm.  These first faces are only
    # 3 mm from the source, so the next full preflight call should print six
    # near-field advisories.  That warning is the useful result of this build;
    # do not compute a pattern from this monitor placement.
    sim.add_ntff_box(
        corner_lo=tuple(coordinate - CLOSE_GAP for coordinate in CENTER),
        corner_hi=tuple(coordinate + CLOSE_GAP for coordinate in CENTER),
        freqs=[F0],
    )
    close_report = sim.preflight()
    close_advisory_observed = bool(close_report.by_code("ntff_near_field"))
    print(f"Close-box advisory observed: {close_advisory_observed}")

    # Move every face 19.5 mm from the source: beyond lambda/2, and one cell
    # inside the CPML-free region.  The next full preflight should print that
    # all checks passed.  Its full check also catches a conductor crossing a
    # monitor face, which run()'s automatic advisory tier does not check.
    sim.add_ntff_box(
        corner_lo=tuple(coordinate - VALID_GAP for coordinate in CENTER),
        corner_hi=tuple(coordinate + VALID_GAP for coordinate in CENTER),
        freqs=[F0],
    )
    corrected_report = sim.preflight()
    if corrected_report:
        raise RuntimeError("Corrected far-field setup still has preflight advisories")

    print(
        f"Corrected face spacing: {VALID_GAP * 1e3:.2f} mm "
        f"(lambda/2 = {HALF_WAVELENGTH * 1e3:.2f} mm)"
    )

    # run() repeats the near-field advisory tier, so the next message should
    # again say that its checks passed.  For far-field work, call preflight()
    # yourself first, as above, to include the full monitor-face checks.
    result = sim.run(n_steps=N_STEPS, compute_s_params=False)

    # A z-directed point dipole is rotationally symmetric.  Its phi=0 E-plane
    # is therefore the full angular description, and directivity() integrates
    # this single azimuth over 2*pi.  This uses the source symmetry; it is not
    # parameter tuning.
    theta = np.linspace(0.01, np.pi - 0.01, 73)
    phi = np.array([0.0])
    far_field = compute_far_field(
        result.ntff_data,
        result.ntff_box,
        result.grid,
        theta,
        phi,
    )
    peak_directivity = float(directivity(far_field)[0])
    if not np.isfinite(peak_directivity):
        raise RuntimeError("Far-field directivity is not finite")

    # This coarse grid has a small, known numerical offset.  Do not tune the
    # physical setup against it; refine a production model instead.
    textbook_directivity = 1.76
    print(f"Peak directivity: {peak_directivity:.3f} dBi")
    print(
        f"Textbook short-dipole value: {textbook_directivity:.2f} dBi; "
        f"difference: {peak_directivity - textbook_directivity:+.3f} dB"
    )

    save_e_plane_cut(far_field)
    print(f"Saved E-plane cut: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
