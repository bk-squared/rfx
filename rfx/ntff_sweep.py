"""Streaming NTFF multi-frequency accumulation (issue #43).

NTFF DFT arrays scale as ``n_freqs × face_cells × 4`` complex64. For
broadband antenna sweeps (100+ frequency points) this dominates the
working set. ``ntff_sweep`` splits the frequency list into small
batches, reruns the simulation once per batch, and concatenates the
per-batch NTFFData along the frequency axis — trading wall-time for
memory.

Not JAX-differentiable; this is an orchestrator for post-processing
runs where the Simulation object is reused. For gradient-based
inverse design at a single frequency, keep using ``sim.forward`` with
a 1-frequency NTFF box.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from rfx.farfield import NTFFData


def _concat_faces(per_batch: list[NTFFData]) -> NTFFData:
    """Stack per-batch NTFFData along the frequency axis (axis 0)."""
    def _cat(field):
        arrs = [getattr(b, field) for b in per_batch]
        arrs = [np.asarray(a) for a in arrs]
        return np.concatenate(arrs, axis=0)
    return NTFFData(
        x_lo=_cat("x_lo"), x_hi=_cat("x_hi"),
        y_lo=_cat("y_lo"), y_hi=_cat("y_hi"),
        z_lo=_cat("z_lo"), z_hi=_cat("z_hi"),
        c_x_lo=_cat("c_x_lo"), c_x_hi=_cat("c_x_hi"),
        c_y_lo=_cat("c_y_lo"), c_y_hi=_cat("c_y_hi"),
        c_z_lo=_cat("c_z_lo"), c_z_hi=_cat("c_z_hi"),
    )


def ntff_sweep(sim, freqs, *, batch_size: int | None = None,
               run_kwargs: dict | None = None) -> tuple[NTFFData, np.ndarray]:
    """Accumulate NTFF data over ``freqs`` in memory-bounded batches.

    Parameters
    ----------
    sim : Simulation
        Must already have ``add_ntff_box(...)`` called so the Huygens
        box corners are known. The frequency list is overridden per
        batch; the original list is restored before return.
    freqs : array-like of Hz
        Full sweep frequency list.
    batch_size : int, optional
        Per-batch frequency count. Default: full list (single run).
        Pick smaller values when the NTFF accumulator dominates memory.
    run_kwargs : dict, optional
        Forwarded to ``sim.run``. Example: ``{"num_periods": 60,
        "compute_s_params": False}``.

    Returns
    -------
    (NTFFData, np.ndarray)
        Concatenated NTFFData plus the original freqs array.

    Notes
    -----
    - Each sub-run is fully independent; the per-batch cost is one
      full forward simulation.
    - Side effect: the sim's NTFF box's freqs field is restored on
      success or exception (try/finally).
    """
    if getattr(sim, "_ntff", None) is None:
        raise ValueError(
            "ntff_sweep requires sim.add_ntff_box(...) first — it "
            "inherits the box corners but overrides the freq list."
        )
    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("freqs must be a non-empty 1-D array of Hz")
    if batch_size is None or batch_size <= 0 or batch_size > len(freqs):
        batch_size = len(freqs)

    corner_lo, corner_hi, orig_freqs = sim._ntff
    run_kwargs = run_kwargs or {}

    batches: list[np.ndarray] = [
        freqs[i:i + batch_size] for i in range(0, len(freqs), batch_size)
    ]
    per_batch_data: list[NTFFData] = []
    try:
        for batch in batches:
            sim._ntff = (corner_lo, corner_hi, batch)
            result = sim.run(**run_kwargs)
            if result.ntff_data is None:
                raise RuntimeError(
                    "ntff_sweep: sim.run returned no ntff_data — "
                    "check that the NTFF box is valid."
                )
            per_batch_data.append(result.ntff_data)
    finally:
        sim._ntff = (corner_lo, corner_hi, orig_freqs)

    combined = _concat_faces(per_batch_data)
    return combined, freqs
