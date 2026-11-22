"""The known drive beyond a record: one input of the record-end judgement.

A drive whose samples are known past the record (a waveguide port's H companion
near cutoff decays as t^-3/2 and outlives any practical record) adds, to every
recorded quantity, the response to its own beyond-record part. By causality that
response lies wholly after the record, and its share of a record's in-record DFT
per bin is ``|DFT d[N:K]| / |DFT d[:N]|``: the structure's transfer cancels, so
it is a property of the drive alone. The record check adds it to the pole tail
in one per-bin budget (rfx.sparams._tail_witness); until_decay waits for it
(rfx.measurement.decay).

Limits: the max over a port's E and H drives
bounds the combined share where their responses add, downstream of a one-way
source (there it is about 2x high); behind the source they cancel and it is not
a bound. Every scored plane of a waveguide port is downstream of its source.
The power-sum allowance 2s + s^2 built on this share is an estimate (e = s per
S entry), adopted after a check against settled records, not a proven bound.

The source end has one rule (``source_end_step``) and one collector
(``injection_end``) here; no other module computes a source end.
The collector reads the drive tables the time step applies: the sampled port,
MSL and point waveforms, the Floquet table its runner builds, a driven
waveguide port's ``e_inc_table`` (never the analytic ``v_inc_t``; its H
companion does not set the end, its beyond-record part is
``drive_tail_share``'s), coaxial E and H tables, and TF/SF's waveform plus
its launch-to-box reach. Ring-down's source-off check reads a closed-box
plane wave instead by the level its faces read from the incident line
(``plane_wave_face_level``), because it asks whether the box still injects
inside the identification windows, while the record-end judgement keeps the
waveform plus its launch-to-box reach, because it asks whether the drive's
response can reach the record end.
"""
from __future__ import annotations

from rfx._internal._precision import resolve_coefficient_dtype

import numpy as np

TRACED = "cannot judge: traced"
STILL_ON = "cannot judge: {source} still injecting at the record end"
_COMPONENT_AXIS = {"ex": 0, "ey": 1, "ez": 2}

ECHO_BOUND_FACTOR = 2.05
"""Minimum longest round trips inside an identification window."""


def _echo_bound_unavailable_reason(exc):
    """Keep tracer refusals independent of the trace's argument names."""
    from jax.errors import JAXTypeError

    detail = "traced operands" if isinstance(exc, JAXTypeError) else str(exc)
    return f"cannot judge: echo delay bound unavailable: {detail}; pass echo_bound="


def echo_group_factor(configs):
    """Only each guide's propagating read bins carry its delayed echo."""
    g = 1.0
    for cfg in configs:
        read = np.asarray(cfg.freqs, dtype=np.float64)
        cutoff = float(cfg.f_cutoff)
        propagating = read[read > cutoff]
        if propagating.size:
            g = min(g, float(np.sqrt(1 - (cutoff / np.min(propagating))**2)))
    return g


def material_echo_index(specs, bins, *, declared, debye=None, lorentz=None,
                        override_maxima=()):
    """Form the declared maximum index in host float64 for every caller."""
    n_max = 1.0
    for material in specs:
        eps = np.asarray(material.eps_r, dtype=np.float64)
        mu = np.asarray(material.mu_r, dtype=np.float64)
        dp = getattr(material, 'debye_poles', None) or ()
        lp = getattr(material, 'lorentz_poles', None) or ()
        if (dp or lp) and not bins.size:
            raise ValueError("the dispersive material read band is unavailable")
        if not declared and (debye is not None or lorentz is not None or dp or lp):
            raise ValueError("low-level dispersive material declarations are unavailable")
        if not declared:
            if not np.all(np.isfinite(eps)) or not np.all(np.isfinite(mu)) or np.any(eps <= 0) or np.any(mu <= 0):
                raise ValueError("nonfinite or nonpositive material index")
            n_max = max(n_max, float(np.sqrt(np.max(eps) * np.max(mu))))
            if not np.isfinite(n_max):
                raise ValueError("nonfinite material index")
            continue
        lp = tuple(p for p in lp if float(p.omega_0) != 0.)
        for pole in lp:
            resonance = float(pole.omega_0) / (2 * np.pi)
            if (float(pole.omega_0) > 0 and float(pole.delta) == 0
                    and np.min(bins) <= resonance <= np.max(bins)):
                raise ValueError("an undamped Lorentz resonance lies inside the read band")
        peaks = [np.sqrt(float(p.omega_0)**2 - 2 * float(p.delta)**2) / (2 * np.pi)
                 for p in lp if float(p.omega_0)**2 > 2 * float(p.delta)**2]
        samples = np.r_[0., bins, [f for f in peaks if np.min(bins) <= f <= np.max(bins)]]
        for frequency in samples:
            omega = 2 * np.pi * frequency
            response = eps.astype(np.complex128)
            for pole in dp:
                response = response + float(pole.delta_eps) / (1 + 1j * omega * float(pole.tau))
            for pole in lp:
                strength = getattr(pole.kappa, 'si', pole.kappa)
                response = response + np.asarray(strength) / (
                    float(pole.omega_0)**2 - omega**2 + 2j * float(pole.delta) * omega)
            values = np.sqrt(np.abs(response * mu))
            if not np.all(np.isfinite(values)):
                raise ValueError("nonfinite material index")
            n_max = max(n_max, float(np.max(values)))
    if override_maxima:
        eps_max, design_max, mu_max = override_maxima
        eps_max = max(1., float(eps_max), float(design_max),
                      *(float(np.max(np.asarray(spec.eps_r, dtype=np.float64))) for spec in specs))
        mu_max = max(1., float(mu_max),
                     *(float(np.max(np.asarray(spec.mu_r, dtype=np.float64))) for spec in specs))
        index = np.sqrt(eps_max * mu_max)
        if not np.isfinite(index):
            raise ValueError("nonfinite material override index")
        n_max = max(n_max, float(index))
    return n_max


def echo_delay_bound(grid, *, sim=None, materials=None, freqs=None, waveguide_ports=(),
                     debye=None, lorentz=None):
    """Return ``tau_max, D, n_max, g`` from the realized physical domain.

    An unavailable host operand returns a reason instead of a numerical bound.
    Pads and the nonperiodic fence-post cell never contribute to D. Declared
    materials include vacuum and every material used by the geometry. For
    dispersion, the static response, band edges and Lorentz peaks enter the
    maximum, excluding Drude poles: a conductor's interior is not an echo
    path. A low-level caller supplies concrete nondispersive arrays or an
    explicit host bound.
    """
    try:
        bins = np.asarray(() if freqs is None else freqs, dtype=np.float64).reshape(-1)
        configs = tuple(waveguide_ports.values() if hasattr(waveguide_ports, 'values')
                        else waveguide_ports or ())
        if not bins.size and configs:
            bins = np.unique(np.concatenate([np.asarray(cfg.freqs, dtype=np.float64)
                                             for cfg in configs]))
        if sim is not None:
            specs = [sim._resolve_material(name) for name in
                     dict.fromkeys(entry.material_name for entry in sim._geometry)]
            if not bins.size:
                bins = np.asarray([0., sim._freq_max], dtype=np.float64)
        elif materials is not None:
            specs = (materials if isinstance(materials, (list, tuple)) and not hasattr(materials, 'eps_r')
                     else [materials])
        else:
            raise ValueError("material declarations are unavailable")
        g = echo_group_factor(configs)
        n_max = material_echo_index(specs, bins, declared=sim is not None,
                                    debye=debye, lorentz=lorentz,
                                    override_maxima=getattr(sim, '_echo_override_maxima', ()))
        return echo_delay_bound_on_grid(grid, dict(n_max=n_max, g=g))
    except (TypeError, ValueError, AttributeError, OverflowError) as exc:
        return dict(tau_max=None, D=None, n_max=None, g=locals().get('g'),
                    reason=_echo_bound_unavailable_reason(exc))


def echo_delay_bound_on_grid(grid, bound):
    """Reuse the declared index and group factor with realized host widths.

    Only D and tau_max change. Declaration refusals remain unchanged; a
    width refusal keeps the declared factors so realized widths can repair it.
    """
    if bound.get('reason') and (bound.get('n_max') is None or bound.get('g') is None):
        return bound
    try:
        lengths = []
        for axis in "xyz":
            if axis == 'z' and getattr(grid, 'is_2d', False):
                lengths.append(0.)
                continue
            cells = np.asarray(grid.cells(axis), dtype=np.float64)
            if axis not in getattr(grid, "periodic_axes", ()):
                cells = cells[:-1]
            lo, hi = getattr(grid, f"pad_{axis}_lo"), getattr(grid, f"pad_{axis}_hi")
            interior = cells[lo:len(cells) - hi]
            if not interior.size or not np.all(np.isfinite(interior)) or np.any(interior <= 0):
                raise ValueError(f"unavailable physical {axis} cell widths")
            lengths.append(float(np.sum(interior)))
        D = float(np.linalg.norm(lengths))
        return dict(tau_max=2 * D * bound['n_max'] / (299792458.0 * bound['g']),
                    D=D, n_max=bound['n_max'], g=bound['g'])
    except (TypeError, ValueError, AttributeError, OverflowError) as exc:
        return dict(tau_max=None, D=None, n_max=bound['n_max'], g=bound['g'],
                    reason=_echo_bound_unavailable_reason(exc))


def echo_bound_reason(bound):
    """Explain a refused identification window using the shared bound."""
    if bound.get('reason'):
        return bound['reason']
    return (f"cannot judge: the record ends less than {ECHO_BOUND_FACTOR:g} longest round trips "
            f"({ECHO_BOUND_FACTOR:g}·τ_max = {ECHO_BOUND_FACTOR * bound['tau_max'] * 1e9:.6g} ns) "
            "after the identification window starts; a reflection still on its way, "
            "or one seen fewer than twice, is not visible in the record")


def with_echo_bound(db, witness, bound, dt):
    """Close every passing row using its own recorded identification windows.

    Coverage may already have demoted a pass; its reason is retained. Synthetic
    records with no geometry explicitly carry ``echo_bound=None``.
    """
    row = {**witness, 'echo_bound': bound}
    refused = bool(row.get('echo_bound_refused'))
    reasons = []
    descendants = False
    for key in ('runs', 'groups'):
        if row.get(key):
            descendants = True
            children = [with_echo_bound(child.get('db'), child, bound, dt)[1]
                        for child in row[key]]
            row[key] = children
            refused |= any(child.get('echo_bound_refused') for child in children)
            reasons.extend(child.get('echo_bound_reason') for child in children
                           if child.get('echo_bound_refused'))
            entries = {name: status for child in children
                       for name, status in (child.get('status_per_entry') or {}).items()}
            if entries:
                row['status_per_entry'] = entries
    if not descendants and bound is not None and (row.get('status') == 'pass' or
                              'pass' in (row.get('status_per_entry') or {}).values()):
        windows = row.get('window_starts') or {}
        unavailable = bound.get('reason') or dt is None or not windows
        short = unavailable or any((int(n) - starts[0]) * float(dt)
                                   < ECHO_BOUND_FACTOR * bound['tau_max']
                                   for n, starts in windows.items())
        if short:
            refused = True
            reasons.append(echo_bound_reason(bound) if not unavailable or bound.get('reason') else
                           'cannot judge: echo delay bound identification window unavailable')
    if refused:
        reason = row.get('reason') or ''
        for extra in dict.fromkeys(r for r in reasons if r):
            if extra not in reason:
                reason = '; '.join(filter(None, (reason, extra)))
        row.update(reason=reason, echo_bound_refused=True,
                   echo_bound_reason=next((r for r in reasons if r), row.get('echo_bound_reason')))
        if row.get('status') == 'pass':
            row.update(status='undetermined', db=float('nan'))
            db = float('nan')
        if not descendants and row.get('status_per_entry'):
            row['status_per_entry'] = {name: 'undetermined' if value == 'pass' else value
                                       for name, value in row['status_per_entry'].items()}
    return db, row


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


def plane_wave_face_level(i_h, i_e) -> np.ndarray:
    """The two-scale face-read level ``r(n)`` of a closed-box plane wave, on the host.

    ``r(n) = max(I_E(n) / P_E, I_H(n) / P_H)``: each kind's per-step maxima
    (from ``rfx.stepping.incident_line.face_read_maxima``) divided by its own
    peak ``P_k = max_n I_k(n)``, in float64, the peak cast from the input's
    dtype. A kind with no operand (an empty array) or a zero peak is left out;
    with both left out ``r`` is zero. ``r`` enters :func:`source_end_step` as a
    precomputed table, whose peak is 1 when a kind is kept.
    """
    terms = []
    for raw in (i_e, i_h):
        values = np.asarray(raw)
        if values.size == 0:
            continue
        peak = np.float64(np.max(values))
        if peak == 0.0:
            continue
        terms.append(values.astype(np.float64) / peak)
    if not terms:
        return np.zeros(max(np.size(i_h), np.size(i_e)), dtype=np.float64)
    return terms[0] if len(terms) == 1 else np.maximum(terms[0], terms[1])


def source_end_step(drives, n_steps, dt, *, tolerance=1e-6,
                    return_detail=False):
    """Last active drive sample + 1 + free-space transit, or no source end.

    ``drives`` contains ``(waveform_or_samples, distance_metres)`` pairs.
    Callables are sampled exactly like the runners: float32(n) * dt.
    Precomputed incident tables use the same threshold on their own peak.
    Detail has one row per drive (None when unavailable), retaining sampled
    magnitudes for ringdown's window checks;
    those checks intentionally retain their configurable tolerance and no
    transit delay. An all-zero drive ends at step zero.
    """
    import jax
    import jax.numpy as jnp
    from rfx.core.jax_utils import is_tracer
    from rfx.grid import C0

    if dt is None or is_tracer(dt) or not np.isfinite(dt) or dt <= 0:
        return (None, [None for _ in drives]) if return_detail else None
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
            rows.append(None)
            available = False
            continue
        raw = np.asarray(samples)
        w = (np.abs(raw.astype(np.complex128)) if np.iscomplexobj(raw)
             else np.abs(np.asarray(raw, dtype=np.float64)))
        # A recorded table may lack the scan's last step (the waveguide
        # port does not write it); it then speaks for the steps it holds.
        if w.ndim != 1 or not n_steps - 1 <= len(w) <= n_steps:
            rows.append(None)
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
                  sources=(), mag_sources=(), waveguide_ports=(), tfsf=None, tfsf_angle_deg=None,
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
    from rfx.sources._sampling import field_time_step
    dt = field_time_step(dt, getattr(sim, "_precision", None))
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
        if is_tracer(tfsf_angle_deg):
            refuse("TF/SF source", "traced", TRACED)
        else:
            if tfsf_angle_deg is not None:
                from rfx.sources.tfsf_2d import with_angle
                tfsf = with_angle(tfsf, tfsf_angle_deg)
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
        _declared_drives(sim, n_steps, grid, bool(configs), drives, refuse, tfsf_angle_deg)

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
            if (detail and detail[0] is not None and detail[0][1] > 0 and detail[0][2] >= len(detail[0][0])
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


def _declared_drives(sim, n_steps, grid, have_configs, drives, refuse, tfsf_angle_deg):
    """The drives a run builds from ``sim``'s declarations, for ``injection_end``."""
    from rfx.sources.sources import GaussianPulse
    from rfx.core.jax_utils import is_tracer

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
            center, spec = floquet_port_source(grid, pe, sim._freq_max, sim._domain, n_steps, dtype=resolve_coefficient_dtype(sim._precision))
            table = np.asarray(spec.waveform)
        drives.append((name, table, _points_reach([np.asarray(center)])))
    tf = getattr(sim, "_tfsf", None)
    if tf is not None:
        if is_tracer(tfsf_angle_deg):
            refuse("TF/SF source", "traced", TRACED)
        elif grid is None:
            refuse("TF/SF source", "unavailable", "cannot judge: TF/SF needs the grid it is built on")
        else:
            from rfx.measurement.settling import tfsf_source_drive
            from rfx.sources.tfsf import tfsf_from_declaration
            cfg, _ = tfsf_from_declaration(tf, grid, sim._freq_max, line_steps=n_steps, dtype=resolve_coefficient_dtype(sim._precision))
            if tfsf_angle_deg is not None:
                from rfx.sources.tfsf_2d import with_angle
                cfg = with_angle(cfg, tfsf_angle_deg)
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


from dataclasses import dataclass
from typing import Callable, NamedTuple
from rfx.measurement.ringdown_resolution import in_window_shares, pole_classes

_ROUNDING_ONLY_EPS_MULTIPLE = 2e4
_ROUNDING_TAIL_EPS_MULTIPLE = 2.4e3
_MINUS_40_DB_AMPLITUDE = 1e-2


def _rounding_only_records(records, groups):
    """Name finite records below the rounding margin of their co-located group.

    A record is skipped only when both its peak and its end-of-record level are
    at the rounding level of its cell; a weak record still ringing above that
    level is scored. Measured rounding records' peaks reach about 1.2e3 eps of
    the co-located same-kind peak, and ``_ROUNDING_TAIL_EPS_MULTIPLE`` (2.4e3)
    is twice that; their last-10 % levels on the box are 120 and 235 eps.
    Under float16 (precision='mixed') the margin reaches the peak itself and
    separates nothing, so those records are always scored.
    """
    from rfx.core.jax_utils import is_tracer

    peaks = []
    largest = {}
    for name, raw in records:
        if name not in groups or is_tracer(raw):
            continue
        y = np.asarray(raw)
        if not y.size or not np.isfinite(y).all():
            continue
        group = groups[name]
        peak = float(np.max(np.maximum(np.abs(y.real), np.abs(y.imag))))
        peaks.append((name, group, peak, y))
        largest[group] = max(largest.get(group, 0.), peak)
    skipped = []
    for name, group, peak, y in peaks:
        dtype = y.real.dtype
        if not np.issubdtype(dtype, np.floating):
            continue
        threshold = _ROUNDING_ONLY_EPS_MULTIPLE * float(np.finfo(dtype).eps)
        if threshold < 1. and peak < threshold * largest[group]:
            end = np.atleast_1d(y)
            end = end[-((len(end) + 9) // 10):]
            tail = float(np.max(np.maximum(np.abs(end.real), np.abs(end.imag))))
            if tail > _ROUNDING_TAIL_EPS_MULTIPLE * float(np.finfo(dtype).eps) * largest[group]:
                continue
            ratio = peak / largest[group]
            skipped.append(dict(
                name=name, group=group, peak_ratio=ratio, threshold=float(threshold),
                tail_ratio=tail / largest[group],
                reason=f"peak {ratio:.3g} of the largest co-located record ({group}), below "
                       f"{_ROUNDING_ONLY_EPS_MULTIPLE:g} x eps({dtype}): rounding only"))
    return tuple(skipped)


_POLE_RULE = (
    "The symmetric resolution edge is |Re(s) dt| = log1p(unit_tol)/D. "
    "Unresolved poles of either sign use the judge's tail bound, never an aggregate tail. "
    "Kept non-static growing poles use judge_growing_poles: tail share through "
    "twice the model decay horizon (at least the record length), bar 1e-2, "
    "with (1 + unit_tol)^(2H/D) <= 2. For each discarded growing or non-decaying "
    "sinusoidal pole: abs(DFT of its fitted term inside the identification "
    "window) / abs(whole-record plain DFT), per read bin and channel; "
    "> 1e-2 is undetermined. Static poles are exempt from these checks."
)


@dataclass(frozen=True)
class IdentificationOps:
    """Identification operations supplied by the caller above measurement."""

    identify: Callable
    judge_growing_poles: Callable
    _static_pole: Callable
    _fit_residues: Callable
    RingdownPole: Callable


def _pole_contributions(model, series, bins, denominator, record_s, *, identification: IdentificationOps, growth=None):
    """Return per-bin maxima over fitted pole terms and channels by rule."""
    from rfx.measurement.ringdown_tail import plain_dft

    judge_growing_poles = identification.judge_growing_poles
    _static_pole = identification._static_pole
    _fit_residues = identification._fit_residues
    RingdownPole = identification.RingdownPole

    window = series[model.n_ref:model.n_ref + model.n_window]
    indices = {'discarded growing': [], 'non-decaying sinusoidal': []}
    for k in range(len(model.s)):
        pole = model.pole(k)
        if _static_pole(pole, record_s, 1e-6):
            continue
        if abs(model.s[k].real * model.dt) < 1e-12:
            indices['non-decaying sinusoidal'].append(k)
    for k, pole_s in enumerate(model.s_growing):
        pole = RingdownPole(float(pole_s.imag / (2*np.pi)), 0.,
                            float(-pole_s.real), float(abs(np.exp(pole_s*model.dt))), 0.)
        if not _static_pole(pole, record_s, 1e-6):
            indices['discarded growing'].append(k)
    joint = None
    if indices['discarded growing']:
        joint = _fit_residues(window, np.concatenate([model.s, model.s_growing]),
                              model.dt, column_reference=True)[len(model.s):]
    time = np.arange(len(window)) * model.dt
    if growth is None:
        growth = judge_growing_poles(model, round(record_s / model.dt), bins, denominator)
    values = {"kept non-static growing": growth.growing_share_per_bin}
    for rule, selected in indices.items():
        value = np.zeros(len(bins))
        if rule == 'non-decaying sinusoidal':
            if selected:
                value = np.max(in_window_shares(model, selected, bins, denominator), axis=(1, 2))
            values[rule] = value
            continue
        for k in selected:
            samples = np.exp((time - time[-1]) * model.s_growing[k])[:, None] * joint[k]
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.abs(plain_dft(samples, model.dt, bins)) / denominator
            value = np.maximum(value, np.max(ratio, axis=1))
        values[rule] = value
    return values


class TailShareWitness(NamedTuple):
    db: float
    share_per_bin: np.ndarray
    worst_freq_hz: float
    status: str
    reason: str
    error_per_bin: np.ndarray
    pole_contributions: dict | None = None
    pole_rule: str = _POLE_RULE
    skipped_records: tuple = ()
    drive_share_per_bin: tuple | np.ndarray = ()
    dropped_per_bin: list = []
    record_end_error_per_bin: object = ()
    kappa_per_bin: object = ()
    status_per_entry: object = None
    record_end_error_per_entry: object = None
    record_end_floor_per_bin: object = ()
    tails_per_channel: object = None
    second_tails_per_channel: object = None
    window_starts: object = None
    record_end_error_second_per_entry: object = None
    echo_bound: object = None


def _channel_tail_witness(records, dt, source_end, freqs, *, identification: IdentificationOps, freq_max, rounding_groups=None,
                       _record_results=None, known_drives=None, n_record=None,
                       scale_groups=None, _channel_labels=None, _tails=None,
                       _window_starts=None):
    """Judge per-bin pole tails by agreement between two source-free windows.

    ``rounding_groups`` names co-located same-kind records whose peaks share
    a rounding margin; ungrouped records are never skipped.
    ``scale_groups`` gives a same-quantity key, or one key per input column;
    a missing key or None keeps that column's own level.
    A column is dropped from a bin when its in-record DFT is below
    ``_MINUS_40_DB_AMPLITUDE`` of its group's largest: the -40 dB amplitude
    bar itself, not a tuned null cutoff. A null does not resolve an
    undetermined growing-pole verdict, even when the null's bound alone caused
    that verdict.
    sensitivities are judged by a separate witness
    """
    from dataclasses import replace

    from rfx.core.jax_utils import is_tracer
    from rfx.measurement.ringdown_tail import plain_dft, tail_dft
    from rfx.sparams._settling_groups import column_metadata

    identify = identification.identify
    judge_growing_poles = identification.judge_growing_poles
    _static_pole = identification._static_pole

    records = tuple(records)
    skipped = _rounding_only_records(records, rounding_groups or {})
    skipped_names = {row['name'] for row in skipped}
    if _record_results is not None:
        _record_results.update((name, float('nan')) for name in skipped_names)
    if is_tracer(freqs):
        return TailShareWitness(float('nan'), np.array([]), float('nan'),
                                'undetermined', 'read bins are traced', np.array([]),
                                skipped_records=skipped)
    bins = np.asarray([] if freqs is None else freqs, dtype=float).reshape(-1)
    share = np.zeros(bins.size)
    error = np.zeros(bins.size)
    contributions = {}
    drive_share = np.zeros(bins.size)
    dropped_names = [[] for _ in bins]
    kept_count = np.zeros(bins.size, dtype=int)

    def dropped_rows():
        return ([dict(freq_hz=float(f), count=len(names), channels=names)
                 for f, names in zip(bins, dropped_names)]
                if any(dropped_names) else [])

    def unavailable(status, reason):
        return TailShareWitness(float('nan'), share + drive_share, float(bins[np.argmax(share + drive_share)]) if bins.size and np.any(share + drive_share) else float('nan'), status, reason, error, contributions, skipped_records=skipped, drive_share_per_bin=() if known_drives is None else drive_share, dropped_per_bin=dropped_rows())

    if records and len(skipped) == len(records):
        return unavailable('undetermined', 'every record is rounding only: ' + ', '.join(row['name'] for row in skipped))
    if not records or not bins.size:
        return unavailable('absent', 'no records or no read bins')
    if dt is None or freq_max is None or is_tracer(dt) or is_tracer(freq_max):
        return unavailable('undetermined', 'dt or identification freq_max is unavailable')
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(freq_max) or freq_max <= 0 or not np.isfinite(bins).all():
        return unavailable('undetermined', 'invalid time step, read bins, or identification freq_max')
    if isinstance(known_drives, ValueError):
        reason, known_drives = str(known_drives), ()
        return unavailable('undetermined', reason)
    if known_drives is not None:
        from rfx.measurement.record_end import drive_tail_share
        known_drives = tuple(known_drives)
        if n_record is None or is_tracer(n_record) or any(is_tracer(d) for d in known_drives):
            return unavailable('undetermined', 'known drives or record length are unavailable or traced')
        try:
            drive_share = drive_tail_share(known_drives, dt, n_record, bins)
        except (ValueError, FloatingPointError) as exc:
            return unavailable('undetermined', str(exc))
        if not np.isfinite(drive_share).all():
            return unavailable('undetermined', 'non-finite known drive tail or zero in-record DFT')
    if source_end is None or is_tracer(source_end):
        return unavailable('undetermined', 'source end is unavailable')
    reasons = []
    zero_reasons = []
    groups = {}
    for name, raw in sorted(records, key=lambda item: item[0]):
        if name in skipped_names:
            continue
        if _record_results is not None:
            _record_results[name] = float("nan")
        if is_tracer(raw):
            reasons.append(f"{name}: record is traced")
            continue
        try:
            y = np.asarray(raw)
            if y.ndim == 0 or not y.size or not np.isfinite(y).all():
                raise ValueError("empty or non-finite record")
            y = y.reshape(y.shape[0], -1).astype(np.complex128)
            groups.setdefault(len(y), []).append((name, y))
        except (ValueError, FloatingPointError) as exc:
            reasons.append(f"{name}: {exc}")

    for n, group in sorted(groups.items()):
        label = ", ".join(name for name, _ in group)
        n_reasons = len(reasons)
        try:
            if int(source_end) != source_end or not 0 <= source_end < n:
                raise ValueError("source end outside record")
            start = max(int(source_end), round(.5 * n))
            use_earlier = start // 2 >= source_end
            check_start = start // 2 if use_earlier else start
            check_stop = n if use_earlier else start + round(.9 * (n - start))
            if _window_starts is not None:
                _window_starts[n] = (start, check_start, check_stop)
            if min(n - start, check_stop - check_start) < 10:
                raise ValueError("post-source window too short for the identification check")
            filtered, keys, channel_names = [], [], []
            for name, array in group:
                column_keys, names = column_metadata(
                    name, array.shape[1], scale_groups, _channel_labels)
                active = np.any(array[int(source_end):] != array[int(source_end)], axis=0)
                if _tails is not None:
                    for channel in names:
                        _tails[channel] = (np.zeros(bins.size, complex), np.zeros(bins.size, complex))
                filtered.append((name, array[:, active]))
                keys.extend(key for key, keep in zip(column_keys, active) if keep)
                channel_names.extend(value for value, keep in zip(names, active) if keep)
            group = filtered
            y = np.concatenate([array for _, array in group], axis=1)
            if y.shape[1] == 0:
                zero_reasons.append(f"{label}: no post-source variation")
                if _record_results is not None:
                    for name, _ in group:
                        _record_results[name] = float("-inf")
                continue
            scale = np.max(np.maximum(np.abs(y.real), np.abs(y.imag)), axis=0)
            y = y / scale
            denom = np.abs(plain_dft(y, dt, bins))
            raw_dft = denom * scale
            group_scale = raw_dft.copy()
            columns_by_key = {}
            for col, key in enumerate(keys):
                if key is not None:
                    columns_by_key.setdefault(key, []).append(col)
            for columns in columns_by_key.values():
                group_scale[:, columns] = raw_dft[:, columns].max(axis=1)[:, None]
            dropped = raw_dft < _MINUS_40_DB_AMPLITUDE * group_scale
            kept_count += np.sum(~dropped, axis=1)
            for k in range(len(bins)):
                dropped_names[k].extend(channel_names[c] for c in np.flatnonzero(dropped[k]))

            def identify_window(lo, hi):
                centered = y - np.mean(y[lo:hi], axis=0)
                active = np.any(centered[lo:hi] != 0, axis=0)
                if not np.any(active):
                    raise ValueError("no post-source variation in identification window only")
                model = identify(centered[:, active], dt, lo, hi,
                                 freq_max=max(freq_max, float(np.max(bins))))
                residues = np.zeros((len(model.s), y.shape[1]), dtype=complex)
                residues[:, active] = model.c
                model = replace(model, c=residues)
                unmasked = np.where(active, denom, np.inf)
                return model, centered, np.where(dropped, np.inf, unmasked), unmasked

            main, main_series, main_denom, main_horizon = identify_window(start, n)
            other, other_series, other_denom, other_horizon = identify_window(check_start, check_stop)
            for model, series, window_denom, horizon_denom in (
                    (main, main_series, main_denom, main_horizon),
                    (other, other_series, other_denom, other_horizon)):
                if not len(model.s):
                    reasons.append(f"{label}: no kept poles identified")
                growth = judge_growing_poles(model, n, bins, window_denom,
                                             horizon_denominator=horizon_denom)
                growth_status = (judge_growing_poles(model, n, bins, horizon_denom)
                                 if np.any(dropped) else growth)
                if not growth_status.ok:
                    reasons.append(f"{label}: growing-pole judge {growth_status.status}: {growth_status.reason}")
                for rule, value in _pole_contributions(model, series, bins, window_denom, n*dt, identification=identification, growth=growth).items():
                    contributions[rule] = np.maximum(contributions.get(rule, np.zeros(len(bins))), value)
                    if rule == "kept non-static growing":
                        continue
                    for k in np.flatnonzero((value > 1e-2) | ~np.isfinite(value)):
                        comparison = '> 1e-2' if np.isfinite(value[k]) else 'is non-finite'
                        reasons.append(
                            f"{label}: {rule} pole at read bin {bins[k]:.12g} Hz; "
                            f"identification-window term DFT / whole-record DFT = {value[k]:.12g} {comparison}")
            def resolved_nonstatic(model):
                classes = pole_classes(model)
                keep = np.array([classes[k] != 0 and not _static_pole(model.pole(k), n*dt, 1e-6)
                                 for k in range(len(model.s))], dtype=bool)
                return replace(model, s=model.s[keep], c=model.c[keep])

            main, other = resolved_nonstatic(main), resolved_nonstatic(other)
            tail = tail_dft(main, n - 1, bins)
            alt = tail_dft(other, n - 1, bins)
            if _tails is not None:
                for col, channel in enumerate(channel_names):
                    _tails[channel] = (tail[:, col] * scale[col], alt[:, col] * scale[col])
            with np.errstate(divide="ignore", invalid="ignore"):
                per_channel = np.abs(tail) / main_denom
                other_channel = np.abs(alt) / other_denom
                err_channel = np.abs(tail - alt) / np.minimum(main_denom, other_denom)
            per_channel = np.where(dropped, 0., per_channel)
            other_channel = np.where(dropped, 0., other_channel)
            err_channel = np.where(dropped, 0., err_channel)
            if not np.isfinite(per_channel).all() or not np.isfinite(err_channel).all() or not np.isfinite(other_channel).all():
                reasons.append(f"{label}: non-finite tail or zero in-record DFT")
                continue
            share = np.maximum(share, per_channel.max(axis=1))
            error = np.maximum(error, err_channel.max(axis=1))
            uncertain = (per_channel > 1e-2) != (other_channel > 1e-2)
            if _record_results is not None and len(reasons) == n_reasons:
                offset = 0
                for name, array in group:
                    if array.shape[1] == 0:
                        _record_results[name] = float('-inf')
                        continue
                    columns = slice(offset, offset + array.shape[1])
                    peak = np.max(per_channel[:, columns])
                    if np.any(~dropped[:, columns]) and not np.any(uncertain[:, columns]):
                        _record_results[name] = float(20*np.log10(max(peak, np.finfo(float).tiny)))
                    offset += array.shape[1]
            if np.any(uncertain):
                for k in np.flatnonzero(np.any(uncertain, axis=1)):
                    reasons.append(f"{label}: identification windows disagree at {bins[k]:.12g} Hz")
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            reasons.append(f"{label}: {exc}")
    for k in np.flatnonzero(kept_count == 0):
        if dropped_names[k]:
            reasons.append(f"every channel dropped at read bin {bins[k]:.12g} Hz")
    if reasons:
        return unavailable('undetermined', '; '.join(reasons))
    share = share + drive_share
    worst = int(np.argmax(share))
    db = float(20 * np.log10(max(share[worst], np.finfo(float).tiny)))
    return TailShareWitness(db, share, float(bins[worst]),
                            'pass' if share[worst] <= 1e-2 else 'fail', '; '.join(zero_reasons), error, contributions, skipped_records=skipped, drive_share_per_bin=() if known_drives is None else drive_share, dropped_per_bin=dropped_rows())



_SETTLING_WITNESS_DB = -40.0


def settling_verdict(settling_db) -> str:
    """The ONE -40 dB ring-down comparison in this codebase.

    Returns ``"pass"``, ``"fail"`` or ``"absent"``. ``"absent"`` covers
    ``None`` and every non-finite value (NaN, +/-inf) -- the states that mean
    "no witness was established", which a bare ``> -40`` comparison silently
    turns into a pass. That is not hypothetical: a campaign driver read a
    missing witness as ``np.nan`` through ``getattr(result, "settling_db",
    np.nan)``, compared it to the bar, and a completed 3-hour two-arm
    experiment came back INCONCLUSIVE for an instrument reason.

    Every caller that needs the decision calls this instead of writing the
    comparison again -- including the aggregate warner below and the
    ``run()`` post-run advisories in ``rfx/api/_execute.py``. A value exactly
    at the bar passes (the bar is the inclusive limit, unchanged).
    """
    if settling_db is None:
        return "absent"
    try:
        value = float(settling_db)
    except (TypeError, ValueError):
        return "absent"
    if not np.isfinite(value):
        return "absent"
    return "fail" if value > _SETTLING_WITNESS_DB else "pass"


def record_end_error(records, dt, source_end, freqs, *, identification: IdentificationOps, freq_max,
                     rounding_groups=None, _record_results=None, known_drives=None,
                     n_record=None, scale_groups=None, _channel_labels=None, reader=None):
    """One record-end rule, with an identity or linear/ratio quantity reader.

    Reader entries contain one linear operand, or (b, a) for a ratio. Each
    operand maps column names to weights. Identification and its channel
    diagnostics retain the original denominators, independently of the reader.
    Returned tails use the whole-record plain DFT origin, including exp(-iwNdt).
    A reader aligns each tail with the reader's transform of a unit impulse at
    slot zero, which carries the kernel's sampling-time factor without a second
    copy of its E/H phase formula.
    Identification operations are required; measurement imports no implementation.
    """
    from rfx.core.jax_utils import is_tracer
    from rfx.sparams._settling_groups import column_metadata
    from rfx.measurement.dft import transform

    records = tuple(records)
    tails: dict[str, tuple] = {}
    windows: dict[int, tuple] = {}
    result = _channel_tail_witness(
        records, dt, source_end, freqs, identification=identification, freq_max=freq_max,
        rounding_groups=rounding_groups if reader is None else None,
        _record_results=_record_results, known_drives=known_drives,
        n_record=n_record, scale_groups=scale_groups, _channel_labels=_channel_labels,
        _tails=tails, _window_starts=windows)
    result = result._replace(tails_per_channel={k: v[0] for k, v in tails.items()},
                             second_tails_per_channel={k: v[1] for k, v in tails.items()},
                             window_starts=windows)
    if reader is None:
        return result
    if reader.get('reason'):
        return result._replace(status='undetermined', db=float('nan'), reason=reader['reason'])
    if (is_tracer(freqs) or is_tracer(dt) or dt is None
            or any(is_tracer(raw) for _, raw in records)):
        return result
    bins = np.asarray(freqs, dtype=float).reshape(-1)
    if not bins.size or not windows:
        return result
    columns, groups = {}, {}
    for name, raw in records:
        y = np.asarray(raw)
        if not y.size or not np.isfinite(y).all():
            return result
        y = y.reshape(len(y), -1)
        keys, labels = column_metadata(name, y.shape[1], scale_groups, _channel_labels)
        for col, (key, label) in enumerate(zip(keys, labels)):
            columns[label] = y[:, col]
            groups[label] = key if key is not None else label
    skipped = _rounding_only_records(tuple(columns.items()), groups)
    skip = {row['name'] for row in skipped}
    peaks = {name: float(np.max(np.maximum(abs(y.real), abs(y.imag))))
             for name, y in columns.items()}
    largest = {key: max(peaks[name] for name in columns if groups[name] == key)
               for key in groups.values()}
    spectra, aligned_tails, sigma = {}, {}, {}
    for name, y in columns.items():
        kind, factor = reader['sampling'].get(name, ('E', 1.))
        spectra[name] = factor * np.asarray(transform(y, bins, dt, kind))
        stamp = factor * np.asarray(transform(np.ones(1, dtype=y.dtype), bins, dt, kind)) / dt
        pair = tails.get(name, (np.zeros(bins.size, complex),) * 2)
        aligned_tails[name] = tuple(stamp * t for t in pair)
        eps = np.finfo(y.real.dtype).eps
        nw = len(y) - windows[len(y)][0]
        sigma[name] = factor * _ROUNDING_TAIL_EPS_MULTIPLE * eps * largest[groups[name]] * dt * np.sqrt(nw)
    errors, second_errors, kappas, floors, statuses = {}, {}, {}, {}, {}
    drive = np.asarray(result.drive_share_per_bin)
    if not drive.size:
        drive = np.zeros(bins.size)
    reasons = []
    for entry, operands in reader['entries'].items():
        error, other, kappa, floor = (np.zeros(bins.size) for _ in range(4))
        for weights in operands:
            q = sum(w * spectra[name] for name, w in weights.items())
            numerator = sum(abs(w) * abs(aligned_tails[name][0])
                            for name, w in weights.items() if name not in skip)
            alternate = sum(abs(w) * abs(aligned_tails[name][1])
                            for name, w in weights.items() if name not in skip)
            with np.errstate(divide='ignore', invalid='ignore'):
                error += numerator / abs(q)
                other += alternate / abs(q)
                kappa += sum(abs(w) * abs(spectra[name]) for name, w in weights.items()) / abs(q)
                floor += sum(abs(w) * sigma[name] for name, w in weights.items()) / abs(q)
        error += kappa * drive
        other += kappa * drive
        errors[entry], kappas[entry], floors[entry] = error, kappa, floor
        second_errors[entry] = other
        entry_reason = ''
        if result.status in {'undetermined', 'absent'}:
            status = 'undetermined'
        elif not np.isfinite(error).all() or not np.isfinite(other).all():
            status = 'undetermined'
            entry_reason = 'non-finite tail or zero in-record DFT'
        elif np.any((error > 1e-2) != (other > 1e-2)):
            status = 'undetermined'
            entry_reason = 'windows disagree'
        else:
            status = 'fail' if np.any(error > 1e-2) else 'pass'
        if all(np.all(columns[name] == 0) for name, weight in operands[0].items()
               if np.any(np.asarray(weight) != 0)):
            entry_reason = ('no field recorded at the receiving port over the whole record; '
                            'the record cannot tell an isolated port from a wave that has not arrived')
        if entry_reason:
            reasons.append(f'{entry}: {entry_reason}')
        statuses[entry] = status
    if not statuses:
        return result._replace(status='undetermined', db=float('nan'), reason='no S reader entries')
    error = np.maximum.reduce(list(errors.values()))
    worst = int(np.argmax(error))
    status = max(statuses.values(), key=('pass', 'fail', 'undetermined').index)
    db = (float('nan') if status == 'undetermined' else
          float(20 * np.log10(max(error[worst], np.finfo(float).tiny))))
    return result._replace(
        status=status, db=db, worst_freq_hz=float(bins[worst]),
        reason='; '.join(filter(None, [result.reason, *reasons])), skipped_records=skipped,
        record_end_error_per_bin=error, kappa_per_bin=np.maximum.reduce(list(kappas.values())),
        record_end_floor_per_bin=np.maximum.reduce(list(floors.values())),
        record_end_error_per_entry=errors, status_per_entry=statuses,
        record_end_error_second_per_entry=second_errors,
        tails_per_channel={k: v[0] for k, v in aligned_tails.items()},
        second_tails_per_channel={k: v[1] for k, v in aligned_tails.items()})
