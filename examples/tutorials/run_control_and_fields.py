"""Run control and fields -- choose when a simulation should finish.

This tutorial applies all three run controls to one small open simulation:

* ``n_steps`` executes an exact number of FDTD updates.
* ``num_periods`` converts periods at ``freq_max`` to an update count.
* ``until_decay`` continues until total interior field energy has fallen by
  the requested factor.  It is the recommended choice for open-domain
  ring-down work.

An intentionally tiny ``num_periods`` value clips the pulse response so the
post-run advisory can report the remaining envelope level.  The simulation is
then repeated with ``until_decay``.  Finally, the probe samples and final field
arrays are inspected, and a two-dimensional ``|Ez|`` slice is saved.

Run as::

    python examples/tutorials/run_control_and_fields.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfx import GaussianPulse, Simulation


DOMAIN = (0.032, 0.032, 0.032)
DX = 1.0e-3
FREQ_MAX = 8.0e9
F0 = 4.0e9
FIXED_STEPS = 120
CLIPPED_PERIODS = 0.1
DECAY_FACTOR = 1.0e-3
DECAY_MAX_STEPS = 1_200

OUTPUT_PATH = Path(__file__).with_name("output") / "run_control_ez_slice.png"


def build_simulation() -> Simulation:
    """Build the open 41-by-41-by-41-cell model used for every run."""
    sim = Simulation(
        freq_max=FREQ_MAX,
        domain=DOMAIN,
        dx=DX,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_source(
        (0.016, 0.016, 0.016),
        component="ez",
        waveform=GaussianPulse(f0=F0, bandwidth=0.8),
    )
    sim.add_probe((0.019, 0.016, 0.016), component="ez")
    return sim


def truncation_messages(recorded: list[warnings.WarningMessage]) -> list[str]:
    """Return only the post-run messages about a clipped ring-down."""
    return [
        str(item.message)
        for item in recorded
        if "ring-down truncated" in str(item.message)
    ]


def save_ez_slice(plane: np.ndarray) -> tuple[int, int]:
    """Save the middle-z plane of the final electric field magnitude."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
    image = ax.imshow(plane.T, origin="lower", cmap="magma")
    ax.set(
        xlabel="x cell",
        ylabel="y cell",
        title="Final middle-plane |Ez|",
    )
    fig.colorbar(image, ax=ax, label="|Ez|")
    fig.savefig(OUTPUT_PATH, dpi=130)
    plt.close(fig)
    return plane.shape


def main() -> None:
    sim = build_simulation()

    # Expect "All checks passed": the source and probe are inside the open
    # domain, and no geometry overlaps the absorbing cells.  The explicit call
    # keeps the full report visible; each run below skips the duplicate check.
    report = sim.preflight()
    if report:
        raise RuntimeError("Run-control setup has unexpected advisories")

    # Fixed n_steps means exactly 120 updates.  It makes no promise that the
    # pulse has left the domain, so use it when an update count is the decision.
    fixed_result = sim.run(
        n_steps=FIXED_STEPS,
        compute_s_params=False,
        skip_preflight=True,
    )
    print(f"Fixed n_steps samples: {fixed_result.time_series.shape[0]}")

    # num_periods counts periods of freq_max.  A value of 0.1 is absurdly
    # short here.  Expect a warning such as "run ended at -0.1 dB of peak":
    # a level near 0 dB says the response was clipped near its largest value.
    with warnings.catch_warnings(record=True) as clipped_warnings:
        warnings.simplefilter("always")
        clipped_result = sim.run(
            num_periods=CLIPPED_PERIODS,
            compute_s_params=False,
            skip_preflight=True,
        )
    clipped_messages = truncation_messages(clipped_warnings)
    if not clipped_messages:
        raise RuntimeError("The deliberately clipped run emitted no advisory")
    print(f"Source-period samples: {clipped_result.time_series.shape[0]}")
    print(f"Truncation advisory: {clipped_messages[0]}")

    # until_decay=1e-3 finishes when total field energy in the non-absorbing
    # interior stays below one thousandth of its peak on consecutive checks.
    # It overrides the fixed-count choices and is recommended for ring-down.
    with warnings.catch_warnings(record=True) as decay_warnings:
        warnings.simplefilter("always")
        decay_result = sim.run(
            until_decay=DECAY_FACTOR,
            decay_check_interval=20,
            decay_min_steps=100,
            decay_max_steps=DECAY_MAX_STEPS,
            compute_s_params=False,
            skip_preflight=True,
        )
    decay_messages = truncation_messages(decay_warnings)
    print(f"Until-decay samples: {decay_result.time_series.shape[0]}")
    print(f"Until-decay truncation advisory observed: {bool(decay_messages)}")

    # Probe samples have shape (time updates, probes).  The final state keeps
    # complete Yee-grid arrays.  Slice each electric component through the
    # middle z plane; a different plane is only a different array index.
    print(f"Probe time_series shape: {decay_result.time_series.shape}")
    ex = np.abs(np.asarray(decay_result.state.ex))
    ey = np.abs(np.asarray(decay_result.state.ey))
    ez = np.abs(np.asarray(decay_result.state.ez))
    middle_z = ez.shape[2] // 2
    ex_slice = ex[:, :, middle_z]
    ey_slice = ey[:, :, middle_z]
    ez_slice = ez[:, :, middle_z]
    print(
        "Final field slice shapes: "
        f"Ex={ex_slice.shape}, Ey={ey_slice.shape}, Ez={ez_slice.shape}"
    )

    slice_shape = save_ez_slice(ez_slice)
    print(f"Saved |Ez| slice {slice_shape}: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
