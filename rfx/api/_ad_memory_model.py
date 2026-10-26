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
    resident = face_record_resident_bytes(sim, grid, n_steps, checkpoint=checkpoint,
        checkpoint_every=checkpoint_every, checkpoint_segments=checkpoint_segments, n_warmup=n_warmup)
    # Only the float32 fixed term and tape have been calibrated.
    tape = carry if window is None else (6 if design == "permittivity" else 3) * window * 4
    if window is None and not resident and not (checkpoint and not segmented):
        # Plain AD saves no per-step residual of the stop_gradient line.
        # Saved boundaries and per-step remat still retain all six leaves.
        tape -= carried_closed_box_line_bytes(sim, grid, n_steps, saved)
    bounded = window is not None and checkpoint and not segmented
    if bounded:
        # The rematted box's no-segment mechanism is not identified: bound
        # it by the carry plus six window arrays, rather than claiming a fit.
        tape = carry + 6 * window * 4
    return dict(saved=saved, excluded_linear=excluded, carry_warnings=carry_warnings,
                modelled=True, unmodelled_features=[],
                carry_bytes=carry,
                resident_bytes=resident,
                tape_step_bytes=tape, tape_window_cells=window,
                design_box_bounds=bounds, tape_is_bound=bounded,
                c0_bytes=AD_MEMORY_C0_PER_CELL["box" if window is not None else "whole-grid"] * accounting["cells"])


def carried_closed_box_line_bytes(sim, grid, n_steps, saved):
    """Identify six carried line leaves by the one builder's shapes and dtypes.

    Closed boxes use 3D Yee fields and CPML slabs; their rank-three leaves
    cannot match the rank-one line arrays. Require the exact multiplicity
    rather than silently subtracting ambiguous or missing carry leaves.
    """
    if sim._tfsf is None or not sim._tfsf.closed_box:
        return 0
    from collections import Counter
    import jax
    from rfx.sources.tfsf import tfsf_from_declaration
    from rfx.sources.tfsf_discrete_plane_wave import COMPONENTS

    state = jax.eval_shape(lambda: tfsf_from_declaration(
        sim._tfsf, grid, sim._freq_max, n_steps=n_steps)[1])
    signatures = Counter((tuple(getattr(state, name).shape), str(getattr(state, name).dtype))
                         for name in COMPONENTS)
    matching = [leaf for leaf in saved if (tuple(leaf["shape"]), leaf["dtype"]) in signatures]
    if Counter((tuple(leaf["shape"]), leaf["dtype"]) for leaf in matching) != signatures:
        raise ADMemoryUnavailable("AD memory could not uniquely identify the six carried closed-box line leaves")
    return sum(leaf["bytes"] for leaf in matching)


def segment_counts(n, warmup, every, segments):
    """Return active boundary count and live tape length for either runner."""
    active = n - warmup
    if isinstance(every, tuple):
        from rfx.stepping.remat import checkpoint_lengths, checkpoint_memory
        lengths = checkpoint_lengths(every)
        # Levels the clamp turns off are skipped, as the graded runner does.
        memory = checkpoint_memory(active, lengths,
                                   boundary_bytes=0, step_bytes=0, static_bytes=0)
        return memory.held_boundaries, memory.live_steps
    if segments is not None:
        length = n // segments
        return segments - warmup // length, min(length, active)
    if every is not None:
        from rfx.stepping.remat import effective_chunk
        length = effective_chunk(active, every) or every
        return math.ceil(active / length), min(length, active)
    return None, None


def face_record_bytes(cfg, n_steps):
    """Float32 storage for the unique H and E table operands at every step."""
    from rfx.stepping.incident_line import _face_reads
    return 4 * n_steps * sum(len(indices) for tables in (cfg.tables_h, cfg.tables_e)
                             for indices in _face_reads(tables)[0])


def face_record_pays(cfg, n_steps, *, checkpoint, checkpoint_segments=None,
                     checkpoint_every=None, n_warmup=0):
    """Choose a record only when it costs less than the saved incident lines.

    record_bytes = 4 x steps x (unique H reads + unique E reads). The reads
    grow with the box's face area for oblique incidence. A checkpointed
    gradient saves the six float32 line arrays once per saved boundary B:
    segment_counts(...)[0] for segments/intervals (every segment is
    rematerialised, whatever ``checkpoint`` says, #1815), steps - warmup
    for per-step remat (checkpoint=True alone), and zero otherwise.
    Choose B * line_bytes > record_bytes. The resident record is not
    itself a saved carry.

    Review measurements (theta=30 deg; decimal MB; cells, phi, steps,
    H/E reads, record MB, one line MB, record/line):
      43   0  1000    835/835       6.7   4.76    1.4
      43   0  4000    835/835      26.7  18.73    1.4
      43  45  1000   2240/2240     17.9   2.41    7.4
      43  45  4000   2240/2240     71.7   9.46    7.6
      83   0  1000   7315/7315     58.5   4.91   11.9
      83   0  4000   7315/7315    234.1  18.87   12.4
      83  45  1000  12909/12909   103.3   2.49   41.4
      83  45  4000  12909/12909   413.1   9.55   43.3
     163   0  1000  32376/32376   259.0   5.20   49.8
     163   0  4000  32376/32376  1036.0  19.17   54.1
     163  45  1000  34269/34269   274.2   2.66  102.9
     163  45  4000  34269/34269  1096.6   9.72  112.8

    A trace-time constant is invisible to memory_analysis: the selected
    record must be built with recorded_scan so the compiled program counts it.
    """
    if not getattr(cfg, "closed_box", False):
        return False
    if checkpoint_segments is not None or checkpoint_every is not None:
        boundaries = segment_counts(n_steps, n_warmup, checkpoint_every, checkpoint_segments)[0]
    else:
        boundaries = n_steps - n_warmup if checkpoint else 0
    line_bytes = 6 * cfg.n_nodes * 4
    return boundaries * line_bytes > face_record_bytes(cfg, n_steps)


def face_record_resident_bytes(sim, grid, n_steps, **knobs):
    """Read the selected record's static cost without allocating the line."""
    if sim._tfsf is None or not sim._tfsf.closed_box:
        return 0
    import jax
    from rfx.sources.tfsf import tfsf_from_declaration
    resident = []

    def abstract_cost():
        cfg, _ = tfsf_from_declaration(sim._tfsf, grid, sim._freq_max, n_steps=n_steps)
        resident.append(face_record_bytes(cfg, n_steps) if face_record_pays(cfg, n_steps, **knobs) else 0)
        return ()

    jax.eval_shape(abstract_cost)
    return resident[0]


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
    if isinstance(checkpoint_every_i, tuple):
        from rfx.stepping.remat import checkpoint_lengths
        checkpoint_lengths(checkpoint_every_i)
    elif checkpoint_every_i is not None and checkpoint_every_i <= 0:
        raise ValueError(
            f"checkpoint_every must be positive when provided, got {checkpoint_every_i}"
        )
    if checkpoint_segments_i is not None:
        if isinstance(checkpoint_segments_i, tuple):
            raise ValueError("AD memory checkpoint_segments must be an int; use checkpoint_every for levels")
        if checkpoint_segments_i < 1:
            raise ValueError(
                f"checkpoint_segments must be ≥ 1, got {checkpoint_segments_i}"
            )
        if n_steps_i % checkpoint_segments_i != 0:
            raise ValueError(
                f"checkpoint_segments={checkpoint_segments_i} does not divide "
                f"n_steps={n_steps_i}"
            )
