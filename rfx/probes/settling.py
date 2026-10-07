"""Host diagnostics for already-recorded user probes, shared by run/forward."""
from __future__ import annotations


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


def source_end_step(drives, n_steps, dt, *, tolerance=1e-6,
                    return_detail=False):
    """Last active drive sample + 1 + free-space transit, or no source end.

    ``drives`` contains ``(waveform_or_samples, distance_metres)`` pairs.
    Callables are sampled exactly like the runners: float32(n) * dt.
    Precomputed incident tables use the same threshold on their own peak.
    Detail retains the sampled magnitudes for ringdown's window checks;
    those checks intentionally retain their configurable tolerance and no
    transit delay. An all-zero drive ends at step zero.
    """
    import jax.numpy as jnp
    from rfx.grid import C0

    if dt is None or is_tracer(dt) or not np.isfinite(dt) or dt <= 0:
        return (None, []) if return_detail else None
    times = None
    rows = []
    end = 0
    available = True
    for waveform, distance in drives:
        if callable(waveform):
            with jax.ensure_compile_time_eval():
                if times is None:
                    times = jnp.arange(n_steps, dtype=jnp.float32) * dt
                samples = jax.vmap(waveform)(times)
        else:
            samples = waveform
        if samples is None or is_tracer(samples):
            available = False
            continue
        raw = np.asarray(samples)
        w = (np.abs(raw.astype(np.complex128)) if np.iscomplexobj(raw)
             else np.abs(np.asarray(raw, dtype=np.float64)))
        # A recorded table may lack the scan's last step (the waveguide
        # port does not write it); it then speaks for the steps it holds.
        if w.ndim != 1 or not n_steps - 1 <= len(w) <= n_steps:
            available = False
            continue
        if not np.isfinite(w).all():
            available = False
        peak = float(np.max(w)) if w.size else 0.0
        on = np.flatnonzero(w > float(tolerance) * peak)
        off = int(on[-1]) + 1 if on.size else 0
        rows.append((w, peak, off))
        if off >= len(w) and peak > 0:
            available = False
        if distance is None or not np.isfinite(distance) or distance < 0:
            available = False
        else:
            end = max(end, off + int(np.ceil(float(distance) / (C0 * float(dt)))))
    value = end if available else None
    return (value, rows) if return_detail else value


def simulation_source_end_step(sim, n_steps, dt, records, grid=None,
                               waveguide_configs=None):
    """Source end for registered point drives and selected physical probes.

    Unknown plane/volume drives must not silently disappear from coverage.
    TF/SF uses its waveform and box diagonal in addition to point drives.
    """
    from rfx.measurement.modal import recorded
    from rfx.sources.sources import GaussianPulse

    if dt is None or is_tracer(dt):
        return None
    records = tuple(records)
    positions = [getattr(p, "position", None) for p in records]
    if not positions or any(p is None for p in positions):
        return None
    for record in records:
        if getattr(record, "extent", None):
            end = np.asarray(record.position, dtype=float).copy()
            end[_COMPONENT_AXIS[record.component]] += record.extent
            positions.append(end)
    drives = []
    if getattr(sim, "_waveguide_ports", ()):
        if not waveguide_configs:
            return None
        for cfg in waveguide_configs.values():
            if cfg.src_amp:
                axis = "xyz".index(cfg.normal_axis)
                distance = max(abs(float(p[axis]) - float(cfg.source_x_m))
                               for p in positions)
                drives.append((recorded(cfg, "v_inc_t"), distance))
    for pe in (*getattr(sim, "_ports", ()), *getattr(sim, "_msl_ports", ())):
        if getattr(pe, "impedance", None) != 0 and not getattr(pe, "excite", True):
            continue
        waveform = pe.waveform
        if waveform is None:
            waveform = GaussianPulse(f0=sim._freq_max / 2, bandwidth=0.8)
        if getattr(pe, "position", None) is None:
            return None
        starts = [np.asarray(pe.position, dtype=float)]
        if getattr(pe, "extent", None):
            end = starts[0].copy()
            end[_COMPONENT_AXIS[pe.component]] += pe.extent
            starts.append(end)
        distance = max(float(np.linalg.norm(start - p))
                       for start in starts for p in positions)
        drives.append((waveform, distance))
    for pe in getattr(sim, "_floquet_ports", ()):
        waveform = GaussianPulse(
            f0=pe.f0 if pe.f0 is not None else sim._freq_max / 2,
            bandwidth=pe.bandwidth, amplitude=pe.amplitude)
        center = np.asarray(sim._domain, dtype=float) / 2
        center["xyz".index(pe.axis)] = pe.position
        distance = max(float(np.linalg.norm(center - p)) for p in positions)
        drives.append((waveform, distance))
    tf = getattr(sim, "_tfsf", None)
    if tf is not None:
        if grid is None:
            return None
        from rfx.sources.tfsf import init_tfsf
        cfg, _ = init_tfsf(
            grid.nx, grid.boundary_cell("x", "lo"), dt, cpml_layers=grid.cpml_layers,
            ny=grid.ny, nz=grid.nz, tfsf_margin=tf.margin,
            f0=tf.f0 if tf.f0 is not None else sim._freq_max / 2,
            bandwidth=tf.bandwidth, amplitude=tf.amplitude,
            polarization=tf.polarization, direction=tf.direction,
            angle_deg=tf.angle_deg, waveform=tf.waveform,
            method=tf.method, closed_box=tf.closed_box)
        drives.append(tfsf_source_drive(cfg, grid))
    return source_end_step(drives, n_steps, dt) if drives else None


def tfsf_source_drive(cfg, grid):
    """TF/SF source waveform and transit across its box diagonal."""
    import jax.numpy as jnp

    def waveform(t):
        custom = getattr(cfg, "custom_waveform", None)
        if custom is not None:
            return cfg.src_amp * custom(t)
        arg = (t - cfg.src_t0) / cfg.src_tau
        env = jnp.exp(-(arg ** 2))
        kind = getattr(cfg, "src_waveform", "analytic")
        if kind == "modulated_gaussian":
            return cfg.src_amp * env * jnp.cos(2 * jnp.pi * cfg.src_fcen * (t - cfg.src_t0))
        if kind == "continuous_wave":
            ramp = .5 * (1 - jnp.cos(jnp.pi * jnp.clip(t / cfg.src_t0, 0., 1.)))
            return cfg.src_amp * ramp * jnp.sin(2 * jnp.pi * cfg.src_fcen * t)
        if kind == "analytic":
            return cfg.src_amp * env * jnp.exp(-1j * 2 * jnp.pi * cfg.src_fcen * (t - cfg.src_t0))
        return cfg.src_amp * (-2 * arg) * env

    lengths = []
    for axis in "xyz":
        lo = getattr(cfg, axis + "_lo", 0)
        hi = getattr(cfg, axis + "_hi", grid.shape["xyz".index(axis)] - 1)
        # A slab has no transverse box indices (both default to zero),
        # but the illumination spans the transverse domain.
        if axis != "x" and hi == lo and grid.shape["xyz".index(axis)] > 1:
            lo, hi = 0, grid.shape["xyz".index(axis)] - 1
        lengths.append(grid.node_of(axis, hi) - grid.node_of(axis, lo))
    return waveform, float(np.linalg.norm(lengths))


def sampled_source_end_step(sources, probes, grid, n_steps):
    """Already sampled E/H drive tables and low-level probe indices."""
    def position(entry):
        return np.array([grid.node_of(axis, getattr(entry, idx))
                         for axis, idx in zip("xyz", "ijk")])

    if not sources or not probes:
        return None
    positions = [position(p) for p in probes]
    return source_end_step(
        [(src.waveform, max(float(np.linalg.norm(position(src) - p))
                            for p in positions)) for src in sources],
        n_steps, grid.dt)


def _cell_of(grid, position):
    """Yee cell of a physical position, or None when it is not resolvable.

    Uses the SAME mapping that PLACES sources and probes --
    ``grid.position_to_index`` (uniform ``Grid``) / ``rfx.nonuniform.
    position_to_index`` (``NonUniformGrid``), duck-typed through
    ``rfx.lumped._resolve_position_to_index``, which is what
    ``rfx.simulation.make_source`` and ``make_probe`` both call. The
    co-location test is therefore "the grid puts them in the same cell",
    exactly; it is never a ``dx``-relative distance, which would be wrong
    on a graded mesh and arbitrary on a uniform one.
    """
    if grid is None or position is None:
        return None
    from rfx.lumped import _resolve_position_to_index
    try:
        idx = _resolve_position_to_index(
            grid, tuple(float(c) for c in position))
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
            start = _cell_of(grid, getattr(entry, "position", None))
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
        cell = _cell_of(grid, getattr(entry, "position", None))
        if cell is not None and cell in cells:
            dominated.add(col)
    return frozenset(dominated)


def probe_record_info(time_series, entries, internal_indices=(),
                      source_dominated=()):
    """Numeric provenance survives JAX result trees without string leaves.

    Each item is ``(column, component_code, source_dominated_flag)``; the
    flag is 1 when that probe shares a Yee cell with a registered drive
    (see :func:`source_dominated_columns`). All three stay integers so the
    selection metadata can ride inside a JAX result tree.
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
        result.append((col, code, 1 if col in dominated else 0))
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
                      "floor_amplitude": None,
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
        if flag:
            dominated_names.add(name)
    independent = [(name, record) for name, record in named if name not in dominated_names]
    selected = independent or named
    worst, detail = settling_db_from_named_records(
        selected, return_detail=True, dt=dt, freqs=freqs, freq_max=freq_max,
        source_end_index=source_end_index)
    per_record = detail["per_record_db"]
    finite = {name: value for name, value in per_record.items() if np.isfinite(value)}
    qualifier = "" if independent else SOURCE_DOMINATED_QUALIFIER
    return worst, {
        **detail, "route": "probe_records",
        "worst_record": max(finite, key=finite.get) if finite else None,
        "source_dominated": bool(qualifier),
        "source_dominated_records": sorted(dominated_names),
        "qualifier": qualifier}
