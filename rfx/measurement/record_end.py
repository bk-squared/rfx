"""The known drive beyond a record: one input of the record-end judgement (#2030).

A drive whose samples are known past the record (a waveguide port's H companion
near cutoff decays as t^-3/2 and outlives any practical record) adds, to every
recorded quantity, the response to its own beyond-record part. By causality that
response lies wholly after the record, and its share of a record's in-record DFT
per bin is ``|DFT d[N:K]| / |DFT d[:N]|``: the structure's transfer cancels, so
it is a property of the drive alone. The record check adds it to the pole tail
in one per-bin budget (rfx.sparams._tail_witness); until_decay waits for it
(rfx.measurement.decay).

Limits (separate-instance review, #2022): the max over a port's E and H drives
bounds the combined share where their responses add, downstream of a one-way
source (there it is about 2x high); behind the source they cancel and it is not
a bound. Every scored plane of a waveguide port is downstream of its source.
The power-sum allowance 2s + s^2 built on this share is an estimate (e = s per
S entry), adopted after a check against settled records, not a proven bound.

The source end has one rule (``source_end_step``) and one collector
(``injection_end``) here; no other module computes a source end (#2030 S1).
The collector reads the drive tables the time step applies: the sampled port,
MSL and point waveforms, the Floquet table its runner builds, a driven
waveguide port's ``e_inc_table`` (never the analytic ``v_inc_t``; its H
companion does not set the end, its beyond-record part is
``drive_tail_share``'s), coaxial E and H tables, and TF/SF's waveform plus
its launch-to-box reach.
"""
from __future__ import annotations

import numpy as np

TRACED = "cannot judge: traced"
STILL_ON = "cannot judge: {source} still injecting at the record end"
_COMPONENT_AXIS = {"ex": 0, "ey": 1, "ez": 2}


def drive_tail_share(drives, dt, n_record, freqs) -> np.ndarray:
    """Known drive beyond a record / inside it, per bin, maximized over drives.

    Host-only float64 plain DFTs; the tail's time-origin phase cancels in abs.
    A drive ending at the record contributes zero, including an all-zero drive.
    """
    from rfx.measurement.ringdown_tail import plain_dft

    bins = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if int(n_record) != n_record or n_record < 0:
        raise ValueError("n_record must be a nonnegative integer")
    n_record = int(n_record)
    share = np.zeros(bins.size, dtype=np.float64)
    for raw in drives:
        drive = np.asarray(raw, dtype=np.float64)
        if drive.ndim != 1 or len(drive) < n_record or not np.isfinite(drive).all():
            raise ValueError("known drives must be finite 1-D samples of length >= n_record")
        if len(drive) == n_record:
            continue
        denominator = np.abs(plain_dft(drive[:n_record], dt, bins))
        numerator = np.abs(plain_dft(drive[n_record:], dt, bins))
        ratio = np.zeros(bins.size, dtype=np.float64)
        np.divide(numerator, denominator, out=ratio, where=denominator != 0)
        ratio[(denominator == 0) & (numerator != 0)] = np.inf
        share = np.maximum(share, ratio)
    return share


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
    import jax
    import jax.numpy as jnp
    from rfx.core.jax_utils import is_tracer
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


def injection_end(n_steps, dt, *, records=None, grid=None, sim=None,
                  sources=(), mag_sources=(), waveguide_ports=(), tfsf=None,
                  return_detail=False):
    """The step after which every applied drive has ended, plus transit.

    ``records=None`` reads incident transit through TF/SF boxes (zero for other drives). Each
    drive adds the free-space transit from its farthest injecting point to
    the farthest record (an object with ``position`` and optional ``extent``,
    a low-level probe with ``i, j, k``, or a point). Drives:

    * ``sources``, ``mag_sources``: built tables (``.waveform`` or entry 4),
      zero padded or cut to ``n_steps`` as the loops read them, at their nodes;
    * ``waveguide_ports`` (sequence or mapping of configs): a driven port's
      ``e_inc_table``, held at its last entry past its end, from its source
      plane;
    * ``tfsf``: a built TF/SF config, its waveform plus launch and scattered transit;
    * ``sim``: what a run builds from the declarations: port and MSL
      waveforms sampled as the runners sample them (a port without one gets
      the default pulse), each Floquet port's table from its runner's builder,
      the TF/SF built from the declaration. A declared waveguide port needs
      its built config in ``waveguide_ports``; a coaxial port is not built on
      this path (run() and forward() refuse it).

    Returns the step, or None when it cannot be judged. ``return_detail``
    returns ``(step, reason, rows)``, a row ``(name, kind, value)`` per drive,
    kind "ok", "still_on", "traced" or "unavailable". On a traced mesh (a
    design variable) a record at a realized node and the TF/SF box cannot be
    judged; declared positions still can.
    """
    from rfx.core.jax_utils import is_tracer

    # (name, samples, reach), or (name, None, (kind, reason)) for a drive
    # whose table cannot be read; evaluated in this order.
    drives = []

    def refuse(name, kind, reason):
        drives.append((name, None, (kind, reason)))

    for kind, entries in (("source", sources), ("magnetic source", mag_sources)):
        for index, src in enumerate(entries or ()):
            table = src.waveform if hasattr(src, "waveform") else src[4]
            if table is not None and not is_tracer(table):
                table = np.asarray(table)[:n_steps]
                table = np.pad(table, (0, n_steps - len(table)))
            drives.append((f"{kind}[{index}]", table, _node_reach(grid, src)))
    if tfsf is not None:
        from rfx.measurement.settling import tfsf_source_drive
        drives.append(("TF/SF source", *tfsf_source_drive(tfsf, grid)))
    configs = (list(waveguide_ports.values()) if hasattr(waveguide_ports, "values")
               else list(waveguide_ports or ()))
    for index, cfg in enumerate(configs):
        name = f"waveguide port[{index}]"
        table = getattr(cfg, "e_inc_table", None)
        if any(is_tracer(v) for v in (cfg.src_amp, table, getattr(cfg, "source_x_m", 0.))):
            refuse(name, "traced", TRACED)
            continue
        if not cfg.src_amp:
            continue
        if table is None or np.shape(table)[0] <= 1:
            refuse(name, "unavailable", f"cannot judge: {name} was initialised without dt, "
                   "so the incident table it applies is not available")
            continue
        table = np.asarray(table)[:n_steps]
        drives.append((name, np.pad(table, (0, n_steps - len(table)), mode="edge"),
                       _plane_reach("xyz".index(getattr(cfg, "normal_axis", "x")),
                                    float(getattr(cfg, "source_x_m", 0.)))))
    if sim is not None:
        _declared_drives(sim, n_steps, grid, bool(configs), drives, refuse)

    rows, reasons = [], []
    points = None if records is None else _record_points(records, grid)
    if points is TRACED:
        reasons.append(TRACED)
        points = None
    elif records is not None and not points:
        reasons.append("cannot judge: a record has no position")
    end = 0
    for name, samples, reach in drives:
        if samples is None and isinstance(reach, tuple):
            rows.append((name, reach[0], None))
            reasons.append(reach[1])
            continue
        distance = (reach if not callable(reach) else
                    (reach(None) if name == "TF/SF source" else 0.0)
                    if points is None else reach(points) if points else None)
        value, detail = source_end_step([(samples, distance)], n_steps, dt, return_detail=True)
        if value is not None:
            rows.append((name, "ok", value))
            end = max(end, value)
        else:
            if (detail and detail[0][1] > 0 and detail[0][2] >= len(detail[0][0])
                    and np.isfinite(detail[0][0]).all()):
                kind, why = "still_on", STILL_ON.format(source=name)
            elif is_tracer(dt) or is_tracer(samples):
                kind, why = "traced", TRACED
            else:
                kind, why = "unavailable", (
                    f"cannot judge: the end of {name} cannot be computed (its samples or "
                    "position are missing, nonfinite or of the wrong length)")
            rows.append((name, kind, None))
            reasons.append(why)
    if not rows:
        reasons.append("cannot judge: no drive")
    step = None if reasons else end
    reason = reasons[0] if reasons else ""
    return (step, reason, rows) if return_detail else step


def drive_end(grid, max_steps, *, sources=(), mag_sources=(), tfsf=None,
              waveguide_ports=()):
    """The until_decay reader: incident TF/SF transit, zero for other drives.

    Point, port and Floquet drives (``sources``, ``mag_sources``) and a
    driven waveguide port's applied incident table (``cfg.e_inc_table``,
    held at its last entry past its end, as the port reads it) are read at
    distance zero; TF/SF uses its waveform and incident transit through the box
    (``tfsf`` is the runner's (config, state) pair). Returns None when a
    drive has not ended by ``max_steps`` (the stop then never fires). Raises
    ValueError naming a drive whose end cannot be computed before the loop.
    """
    _, _, rows = injection_end(
        max_steps, grid.dt, grid=grid, sources=sources, mag_sources=mag_sources,
        tfsf=None if tfsf is None else tfsf[0], waveguide_ports=waveguide_ports,
        return_detail=True)
    end = 0
    for name, kind, value in rows:
        if kind == "ok":
            end = max(end, value)
        elif kind == "still_on":
            return None   # still on at max_steps
        elif name.startswith("waveguide port"):
            raise ValueError(
                f"until_decay: {name} was initialised without dt, so "
                "the incident table it applies is not available before the loop; "
                "initialise it with dt and dft_total_steps, or use a fixed n_steps")
        else:
            raise ValueError(
                f"until_decay: the drive end of {name} cannot be computed before the "
                "loop (its samples are missing, traced, nonfinite or of the wrong length)")
    return None if end > max_steps else end


def _declared_drives(sim, n_steps, grid, have_configs, drives, refuse):
    """The drives a run builds from ``sim``'s declarations, for ``injection_end``."""
    from rfx.sources.sources import GaussianPulse

    if getattr(sim, "_waveguide_ports", ()) and not have_configs:
        refuse("waveguide port", "unavailable",
               "cannot judge: the waveguide port's built configuration is not available")
    for pe in (*getattr(sim, "_ports", ()), *getattr(sim, "_msl_ports", ())):
        if getattr(pe, "impedance", None) != 0 and not getattr(pe, "excite", True):
            continue
        waveform = pe.waveform
        if waveform is None:
            waveform = GaussianPulse(f0=sim._freq_max / 2, bandwidth=0.8)
        name = f"port at {getattr(pe, 'position', None)}"
        if getattr(pe, "position", None) is None:
            refuse(name, "unavailable", f"cannot judge: {name} has no position")
            continue
        starts = [np.asarray(pe.position, dtype=float)]
        if getattr(pe, "extent", None):
            end = starts[0].copy()
            end[_COMPONENT_AXIS[pe.component]] += pe.extent
            starts.append(end)
        drives.append((name, waveform, _points_reach(starts)))
    for index, pe in enumerate(getattr(sim, "_floquet_ports", ())):
        name = f"Floquet port[{index}]"
        if grid is None:
            refuse(name, "unavailable", f"cannot judge: {name} needs the grid its table is built on")
            continue
        import jax
        from rfx.floquet import floquet_port_source
        with jax.ensure_compile_time_eval():  # the declared pulse is concrete under a trace
            center, spec = floquet_port_source(grid, pe, sim._freq_max, sim._domain, n_steps)
            table = np.asarray(spec.waveform)
        drives.append((name, table, _points_reach([np.asarray(center)])))
    tf = getattr(sim, "_tfsf", None)
    if tf is not None:
        if grid is None:
            refuse("TF/SF source", "unavailable", "cannot judge: TF/SF needs the grid it is built on")
        else:
            from rfx.measurement.settling import tfsf_source_drive
            from rfx.sources.tfsf import tfsf_from_declaration
            cfg, _ = tfsf_from_declaration(tf, grid, sim._freq_max, line_steps=n_steps)
            drives.append(("TF/SF source", *tfsf_source_drive(cfg, grid)))
    for index, _ in enumerate(getattr(sim, "_coaxial_ports", ()) or ()):
        refuse(f"coaxial port[{index}]", "unavailable",
               f"cannot judge: coaxial port[{index}] is not built on this path "
               "(its tables are read by the coaxial drivers)")


def _record_points(records, grid):
    """Physical points of the named records (extents add their far end), None
    when one has no position, TRACED when a realized node is on a traced mesh."""
    from rfx.core.jax_utils import is_tracer

    points = []
    for record in records:
        if hasattr(record, "position"):
            if record.position is None:
                return None
            point = np.asarray(record.position, dtype=float)
            points.append(point)
            if getattr(record, "extent", None):
                end = point.copy()
                end[_COMPONENT_AXIS[record.component]] += record.extent
                points.append(end)
            continue
        values = ([grid.node_of(axis, getattr(record, index)) for axis, index in zip("xyz", "ijk")]
                  if all(hasattr(record, index) for index in "ijk") else list(record))
        if any(is_tracer(value) for value in values):
            return TRACED
        points.append(np.asarray(values, dtype=float))
    return points


def _node(grid, entry):
    return np.array([grid.node_of(axis, getattr(entry, index))
                     for axis, index in zip("xyz", "ijk")])


def _node_reach(grid, src):
    def reach(points):
        start = (_node(grid, src) if hasattr(src, "i")
                 else np.array([grid.node_of(axis, src[n]) for n, axis in enumerate("xyz")]))
        return max(float(np.linalg.norm(start - p)) for p in points)
    return reach


def _points_reach(starts):
    return lambda points: max(float(np.linalg.norm(start - p))
                              for start in starts for p in points)


def _plane_reach(axis, coordinate):
    return lambda points: max(abs(float(p[axis]) - coordinate) for p in points)
