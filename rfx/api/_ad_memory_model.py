"""Saved carry and formulation-specific AD tape accounting (#1786)."""
from __future__ import annotations

import math

from rfx.api._ad_memory_types import ADMemoryUnavailable
from rfx.model.design_box import design_box_bounds, design_box_window


# Calibrated (GPU, jax 0.10.2): maximum G1 intercept/cell, rounded up to 8 B.
# rfx-archive:rfx/records/20261016-ad-memory-tape-terms/
# G1 acc-admem-g1-20261016t043022z-648d9; both maxima are the 21^3 rows.
AD_MEMORY_C0_PER_CELL = {"whole-grid": 320, "box": 192}


def tape_accounting(sim, accounting, *, design_box, design, checkpoint, segmented,
                    distributed=False, n_steps=1, checkpoint_every=None, checkpoint_segments=None, n_warmup=0):
    """Resolve the shared window, then read the requested forward carry."""
    if design not in ("permittivity", "occupancy"):
        raise ValueError("design must be 'permittivity' or 'occupancy'")
    if not isinstance(checkpoint, bool):
        raise ValueError("checkpoint must be a bool")
    if sim._precision != "float32":
        raise ADMemoryUnavailable("AD memory's fixed term and tape are calibrated on float32 only; "
                                  f"precision={sim._precision!r} is not supported")
    grid = accounting.get("grid")
    if distributed:
        raise ADMemoryUnavailable("AD memory carry could not be read: distributed forward is not supported by this trace")
    window = None
    bounds = None
    if design_box is not None:
        if distributed or sim._solver != "yee":
            raise NotImplementedError("design_box requires a single-device Yee forward lane")
        if design == "occupancy" and accounting["grid_kind"] == "nonuniform":
            raise NotImplementedError("design_box occupancy is not supported on a graded mesh")
        if grid is None:
            raise ValueError("design_box requires a successfully built grid")
        bounds, shape = design_box_bounds(sim._design_box_bounds_from_corners(grid, design_box), grid)
        _, slices, _, _ = design_box_window(bounds, grid.shape)
        window = math.prod(s.stop - s.start for s in slices)
    else:
        shape = tuple(grid.shape) if grid is not None else ()
    from rfx.api._ad_carry import read_carry
    saved, excluded, carry_warnings = read_carry(sim, grid, n_steps, design_box=design_box, design=design,
        shape=shape, checkpoint=checkpoint, checkpoint_every=checkpoint_every,
        checkpoint_segments=checkpoint_segments, n_warmup=n_warmup)
    carry = sum(x["bytes"] for x in saved)
    # Only the float32 fixed term and tape have been calibrated.
    tape = carry if window is None else (6 if design == "permittivity" else 3) * window * 4
    bounded = window is not None and checkpoint and not segmented
    if bounded:
        # The rematted box's no-segment mechanism is not identified: bound
        # it by the carry plus six window arrays, rather than claiming a fit.
        tape = carry + 6 * window * 4
    return dict(saved=saved, excluded_linear=excluded, carry_warnings=carry_warnings,
                modelled=True, unmodelled_features=[],
                carry_bytes=carry,
                tape_step_bytes=tape, tape_window_cells=window,
                design_box_bounds=bounds, tape_is_bound=bounded,
                c0_bytes=AD_MEMORY_C0_PER_CELL["box" if window is not None else "whole-grid"] * accounting["cells"])


def segment_counts(n, warmup, every, segments):
    """Return active boundary count and live tape length for either runner."""
    active = n - warmup
    if segments is not None:
        length = n // segments
        return segments - warmup // length, min(length, active)
    if every is not None:
        from rfx.stepping.remat import effective_chunk
        length = effective_chunk(active, every) or every
        return math.ceil(active / length), min(length, active)
    return None, None


def tape_warning(model, *, checkpoint, segments, live, active):
    """Explain the checkpoint assumption and the boundary/live-tape balance."""
    messages = list(model["carry_warnings"])
    if not model["modelled"]:
        messages.append("AD carry modelled=False; missing components: " + ", ".join(model["unmodelled_features"]) + ".")
    if model["tape_is_bound"]:
        messages.append("The checkpoint=True box tape is a bound; pass checkpoint=False without segments for the window-only tape.")
    if segments is not None:
        boundary = segments * model["carry_bytes"]
        tape = live * model["tape_step_bytes"]
        direction = "shrink the live tape" if tape > boundary else "shrink boundary storage"
        messages.append(f"Balance K_active*carry_bytes = s*tape_step_bytes to {direction} in the segmented plan.")
    return " ".join(messages) or None


def validate_counts(n_steps_i, n_warmup_i, checkpoint_every_i, checkpoint_segments_i):
    """Validate request counts without tracing or allocating a simulation."""
    if n_steps_i <= 0:
        raise ValueError("n_steps must be positive")
    if n_warmup_i < 0:
        raise ValueError("n_warmup must be >= 0")
    if n_warmup_i >= n_steps_i:
        raise ValueError(f"n_warmup ({n_warmup_i}) must be < n_steps ({n_steps_i})")
    if checkpoint_every_i is not None and checkpoint_segments_i is not None:
        raise ValueError(
            "checkpoint_every and checkpoint_segments are mutually exclusive"
        )
    if checkpoint_every_i is not None and checkpoint_every_i <= 0:
        raise ValueError(
            f"checkpoint_every must be positive when provided, got {checkpoint_every_i}"
        )
    if checkpoint_segments_i is not None:
        if checkpoint_segments_i < 1:
            raise ValueError(
                f"checkpoint_segments must be ≥ 1, got {checkpoint_segments_i}"
            )
        if n_steps_i % checkpoint_segments_i != 0:
            raise ValueError(
                f"checkpoint_segments={checkpoint_segments_i} does not divide "
                f"n_steps={n_steps_i}"
            )
