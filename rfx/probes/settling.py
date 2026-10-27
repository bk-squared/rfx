"""Host diagnostics for already-recorded user probes, shared by run/forward."""
from __future__ import annotations

from rfx._grid_metric import field_index

import jax
import numpy as np

from rfx.core.jax_utils import is_tracer


_COMPONENTS = ("ex", "ey", "ez", "hx", "hy", "hz")

_COMPONENT_AXIS = {"ex": 0, "ey": 1, "ez": 2}

SOURCE_DOMINATED_QUALIFIER = (
    "SOURCE-DOMINATED: every scored record sits on a registered "
    "source/port drive cell; not an independent ring-down witness. Add a "
    "probe away from every drive cell, somewhere the field is live "
    "(issue #1090)."
)


def _cell_of(grid, position, component=None):
    """Yee cell of a physical position, or None when it is not resolvable.

    Point components use ``field_index``, as their source/probe constructors
    do. Extents and geometry retain the component-free node lookup. The
    co-location test is therefore "the grid puts them in the same cell",
    exactly; it is never a ``dx``-relative distance, which would be wrong
    on a graded mesh and arbitrary on a uniform one.
    """
    if grid is None or position is None:
        return None
    from rfx.lumped import _resolve_position_to_index
    try:
        position = tuple(float(c) for c in position)
        idx = (field_index(grid, position, component) if component is not None
               else _resolve_position_to_index(grid, position))
    except Exception:
        # A drive or probe declared outside the built grid cannot be
        # co-located with anything; position_to_index raises there.
        return None
    return tuple(int(v) for v in idx)


def drive_cells(grid, *entry_groups):
    """Yee cells occupied by registered POINT drives (sources / ports).

    Each entry is duck-typed: ``position`` is the declared drive point,
    and an entry that also carries ``extent`` (``add_port(extent=...)``,
    a wire port) drives every cell its extent spans along its
    ``component`` axis -- the same ``position``-to-``position + extent``
    span ``rfx/api/_execute.py`` builds its ``WirePort`` from.

    Covered: ``Simulation._ports`` (``add_source`` soft sources and
    ``add_port`` lumped/wire ports) and ``Simulation._msl_ports`` (the
    registered feed point). NOT covered: drives declared as a PLANE or a
    VOLUME rather than a point -- waveguide ports (``add_waveguide_port``)
    and TFSF. Those need a plane/box membership test rather than a cell
    test; a probe inside such a source region is still source-dominated in
    the #1090 sense and is not flagged here (issue #1102).
    """
    cells: set[tuple[int, int, int]] = set()
    for entries in entry_groups:
        for entry in entries or ():
            start = _cell_of(grid, getattr(entry, "position", None),
                             getattr(entry, "component", None) if getattr(entry, "extent", None) is None else None)
            if start is None:
                continue
            cells.add(start)
            extent = getattr(entry, "extent", None)
            axis = _COMPONENT_AXIS.get(getattr(entry, "component", None))
            if not extent or axis is None:
                continue
            end_position = [float(c) for c in entry.position]
            end_position[axis] += float(extent)
            end = _cell_of(grid, tuple(end_position))
            if end is None:
                continue
            lo, hi = sorted((start[axis], end[axis]))
            for index in range(lo, hi + 1):
                spanned = list(start)
                spanned[axis] = index
                cells.add(tuple(spanned))
    return frozenset(cells)


def source_dominated_columns(grid, entries, cells):
    """Probe columns whose cell is a drive cell (issue #1090).

    Cell co-location alone, deliberately without a component match: the
    H components of a source cell are the direct curl of the E edge the
    drive writes, so every component recorded there peaks on the drive
    pulse, not on the structure's response.
    """
    if grid is None or not cells:
        return frozenset()
    dominated = set()
    for col, entry in enumerate(entries or ()):
        cell = _cell_of(grid, getattr(entry, "position", None),
                        getattr(entry, "component", None))
        if cell is not None and cell in cells:
            dominated.add(col)
    return frozenset(dominated)


def probe_record_info(time_series, entries, internal_indices=(),
                      source_dominated=(), grid=None):
    """Numeric provenance survives JAX result trees without string leaves.

    Each item is ``(column, component_code, source_dominated_flag, i, j, k)``;
    the flag is 1 when that probe shares a Yee cell with a registered drive
    (see :func:`source_dominated_columns`), and ``(i, j, k)`` is the cell of
    the declared position (absent when unresolved), which tells co-located
    records apart (#1909). All stay integers so the selection metadata can
    ride inside a JAX result tree.
    """
    shape = getattr(time_series, "shape", ())
    count = 1 if len(shape) == 1 else shape[1] if len(shape) == 2 else 0
    internal = internal_indices or ()
    dominated = source_dominated or ()
    result = []
    # forward() may create a source-position fallback record when the user
    # registered no probe. Only declared probe columns have user provenance;
    # an extra backend record must not silently provide this witness.
    for col in range(min(count, len(entries))):
        if col in internal:
            continue
        component = getattr(entries[col], "component", "?") if col < len(entries) else "?"
        code = _COMPONENTS.index(component) if component in _COMPONENTS else -1
        cell = _cell_of(grid, getattr(entries[col], "position", None)) or ()
        result.append((col, code, 1 if col in dominated else 0, *cell))
    return tuple(result)


def probe_record_settling_witness(time_series, probe_info=None, *, warn=True,
                                  source_end_index=None, dt=None, freqs=None, freq_max=None):
    """Return (dB or None, provenance) using the canonical record arithmetic.

    None provenance selects all columns with unknown component labels; an
    empty tuple selects none. Traced data are explicitly unavailable. A
    concrete result can be scored after JIT without adding this host work to
    the differentiated solver. ``warn=False`` makes repeated property reads
    quiet; the execution entry point retains the canonical coverage warning.

    Records flagged source-dominated (issue #1090 -- the probe shares a Yee
    cell with a registered drive) do not carry the verdict while any
    independent scored record exists. When they are the only scored records,
    ``qualifier`` / ``source_dominated`` retain that spatial coverage caveat.
    sensitivities are judged by a separate witness
    """
    from rfx.sources.waveguide_port import settling_db_from_named_records

    def absent(reason, skipped=()):
        _, detail = settling_db_from_named_records(
            (), dt=dt, freqs=freqs, freq_max=freq_max, return_detail=True)
        # ``source_dominated``/``qualifier`` describe the record that
        # CARRIES the verdict; an absent witness has none, so they are
        # empty here rather than describing records nothing rests on.
        # No NaN in an absent witness (#885): no number rests on it.
        return None, {**detail, "db": None, "worst_freq_hz": None,
                      "status": "absent", "route": None,
                      "worst_record": None, "per_record_db": {},
                      "skipped_records": list(skipped), "reason": reason,
                      "source_dominated": False,
                      "source_dominated_records": [], "qualifier": ""}

    advice = ("add a point probe (sim.add_probe(position, component)) away "
              "from every source/port drive cell — a record on the drive "
              "cell measures source turn-off, not ring-down, and is not an "
              "independent witness (#1090) — and retain its time series, or "
              "use run(until_decay=...) to bound the ring-down through its "
              "stop criterion")
    if is_tracer(source_end_index) or is_tracer(time_series) or any(is_tracer(leaf) for leaf in jax.tree_util.tree_leaves(probe_info)):
        return absent("probe records or their selection are traced; inspect "
                      "the concrete result after JIT/AD evaluation")
    if time_series is None:
        return absent("this result returned no probe time series: " + advice)
    series = np.asarray(time_series)
    if series.ndim == 1:
        series = series[:, None]
    if series.ndim != 2 or series.size == 0:
        return absent("this result recorded no probe time series: " + advice)
    info = (tuple((i, -1, 0) for i in range(series.shape[1]))
            if probe_info is None else probe_info)
    if not info:
        return absent("no user probe records are selected; library-internal "
                      "witness probes are judged by their own driver: " + advice)
    named = []
    groups = {}
    dominated_names = set()
    for item in info:
        # Pre-#1090 metadata is a 2-tuple; a stored result predating the
        # flag scores exactly as it did, with nothing marked dominated.
        col, component = int(item[0]), int(item[1])
        flag = int(item[2]) if len(item) > 2 else 0
        if not 0 <= col < series.shape[1]:
            raise ValueError("settling probe metadata indexes a missing record column")
        label = _COMPONENTS[component] if 0 <= component < len(_COMPONENTS) else "?"
        name = f"probe{col}({label})"
        named.append((name, series[:, col]))
        # A rounding record's level is eps times the fields that cancel at
        # its own point, so its reference is the same-kind records there.
        cell = tuple(int(v) for v in item[3:6]) if len(item) > 5 else None
        if label != "?" and cell is not None and min(cell) >= 0:
            groups[name] = f"{label[0].upper()} at cell {cell}"
        if flag:
            dominated_names.add(name)
    independent = [(name, record) for name, record in named if name not in dominated_names]
    selected = independent or named
    worst, detail = settling_db_from_named_records(
        selected, return_detail=True, dt=dt, freqs=freqs, freq_max=freq_max,
        source_end_index=source_end_index, rounding_groups=groups)
    per_record = detail["per_record_db"]
    finite = {name: value for name, value in per_record.items() if np.isfinite(value)}
    qualifier = "" if independent else SOURCE_DOMINATED_QUALIFIER
    return worst, {
        **detail, "route": "probe_records",
        "worst_record": max(finite, key=finite.get) if finite else None,
        "source_dominated": bool(qualifier),
        "source_dominated_records": sorted(dominated_names),
        # Lone: its cell is known and no other scored record shares it.
        "lone_records": sorted(name for name, _ in selected if name in groups
                               and [groups.get(other) for other, _ in selected]
                               .count(groups[name]) < 2),
        "qualifier": qualifier}
