"""The decay stop of ``until_decay``: conserved field energy after the drive.

Both time loops (``simulation.run_until_decay`` on the uniform mesh,
``nonuniform.run_nonuniform_until_decay`` on the graded mesh) call this
module at their chunk boundaries ``c = k * interval``; neither scan
program contains any of it.

* ``drive_end``: the step at which every built drive has ended, with the
  settling check's definition (``rfx.probes.settling.source_end_step``).
* ``boundary_energy``: the leapfrog-conserved interior energy
  ``W_c = sum((eps E^c . E^c + (mu0/eps0) mu H^(c-1/2) . H^(c+1/2)) dV)``.
  ``H^(c+1/2)`` comes from one side step of the loop's own chunk program on
  the boundary state (scan length 1); that step's output is discarded except
  its H fields, so the measure reads whatever the step does (UPML, PEC
  damping, cut-face stages) and the run's trajectory is untouched.
* ``DecayStop``: reference = running max of W over boundaries at or after
  the drive end; a boundary is below when ``W_c < max(thr * ref,
  KAPPA * Wmax_c)``, ``Wmax_c`` the largest W at any boundary up to ``c``,
  drive included; stop at the first boundary at or after ``min_steps`` that
  ends ``consecutive`` boundaries below.
* ``finish_decay``: what ended the run, as one ``until_decay.stop`` Diagnostic.
* The DC advisories: ``warn_dc_floor`` before the run (``run()``) and
  ``static_remnant_advisory`` after a capped run (both loops, via ``finish_decay``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0

AUTO_END = object()   # the loop resolves the drive end from its own tables
MU0_OVER_EPS0 = MU_0 / EPS_0   # a vacuum constant (Z0 squared); no permittivity divides by it

#: The float32 floor of the decay stop. About ten float32 units in the last
#: place below the peak field; a drive leaves a rounding residue near 1e-13 of
#: its peak energy (measured 7.5e-14 on the graded until_decay case), so a
#: threshold relative to a small after-drive reference can ask for less than
#: single precision holds. A boundary also counts as below on the floor when its
#: W is under KAPPA times the largest W at any boundary so far (drive included)
#: and has stopped falling: W_c >= FLOOR_FLAT ** (I / FLOOR_SPAN) * W_(c-I), i.e. less
#: than 5 % decay per 50 steps whatever the check interval I. Risk: a mode whose
#: energy is below 1e-12 of the drive's peak and decays more slowly than 5 % per
#: 50 steps would be stopped as rounding.
KAPPA = 1e-12
FLOOR_FLAT = 0.95
FLOOR_SPAN = 50
FLOOR_TEXT = ("below about 1e-12 of the drive's peak energy the remaining field is at the "
              "single-precision rounding level of the drive's peak field; the requested "
              "threshold was not met")
#: Relative DC (|sum s dt| / sum |s| dt) above which a soft drive is named.
DC_ADVISE = 1e-6
REMEDIES = (
    "GaussianPulse deposits no charge (it starts at zero and its time integral is "
    "zero); ModulatedGaussian keeps a DC part unless bandwidth <= 0.4, and even then "
    "some; radiated_flux_box=((x0,y0,z0),(x1,y1,z1)) enclosing the radiator stops on "
    "the outgoing flux, which a static field does not carry; or run a fixed n_steps")


def drive_end(grid, max_steps, *, sources=(), mag_sources=(), tfsf=None,
              waveguide_ports=()):
    """Last step any built drive is on, plus the TF/SF box transit.

    Point, port and Floquet drives (``sources``, ``mag_sources``) and a
    driven waveguide port's applied incident table (``cfg.e_inc_table``,
    held at its last entry past its end, as the port reads it) are read at
    distance zero; TF/SF uses its waveform and the total-field box diagonal
    (``tfsf_source_drive``). Returns None when a drive has not ended by
    ``max_steps`` (the stop then never fires). Raises ValueError naming a
    drive whose end cannot be computed before the loop.
    """
    from rfx.core.jax_utils import is_tracer
    from rfx.probes.settling import source_end_step, tfsf_source_drive

    drives = []
    for kind, entries in (("source", sources), ("magnetic source", mag_sources)):
        for index, src in enumerate(entries or ()):
            table = src.waveform if hasattr(src, "waveform") else src[4]
            if table is not None and not is_tracer(table):
                # Zero padded or cut to the loop's length, as the loops do.
                table = np.asarray(table)[:max_steps]
                table = np.pad(table, (0, max_steps - len(table)))
            drives.append((f"{kind}[{index}]", table, 0.0))
    if tfsf is not None:
        drives.append(("TF/SF source", *tfsf_source_drive(tfsf[0], grid)))
    for index, cfg in enumerate(waveguide_ports or ()):
        if cfg.src_amp:
            # e_inc_table only: the paired h_inc_table carries an FFT tail near 1.4e-6
            # of its peak that never falls below the end rule's 1e-6 of the peak.
            table = cfg.e_inc_table
            if is_tracer(table) or table.shape[0] <= 1:
                raise ValueError(
                    f"until_decay: waveguide port[{index}] was initialised without dt, so "
                    "the incident table it applies is not available before the loop; "
                    "initialise it with dt and dft_total_steps, or use a fixed n_steps")
            table = np.asarray(table)[:max_steps]
            drives.append((f"waveguide port[{index}]",
                           np.pad(table, (0, max_steps - len(table)), mode="edge"), 0.0))
    end = 0
    for name, samples, distance in drives:
        value, rows = source_end_step([(samples, distance)], max_steps, grid.dt,
                                      return_detail=True)
        if value is not None:
            end = max(end, value)
            continue
        if rows and rows[0][1] > 0 and rows[0][2] >= len(rows[0][0]) \
                and np.isfinite(rows[0][0]).all():
            return None   # still on at max_steps
        raise ValueError(
            f"until_decay: the drive end of {name} cannot be computed before the "
            "loop (its samples are missing, traced, nonfinite or of the wrong length)")
    return None if end > max_steps else end


def refuse_windowed_flux(sim, path):
    """Refuse a flux monitor with a non-rectangular DFT window under
    ``until_decay`` (both meshes): a streaming windowed DFT needs the total
    step count in advance, which a decay-terminated run does not know."""
    for monitor in getattr(sim, "_flux_monitors", None) or []:
        window = getattr(monitor, "dft_window", "rect")
        if window != "rect":
            raise ValueError(
                f"flux monitor {monitor.name!r} uses dft_window={window!r}, which is not "
                f"supported with until_decay on the {path} path: a streaming windowed DFT "
                "needs the true total step count in advance, and a decay-terminated run "
                "does not know it (sizing by decay_max_steps would silently mis-weight "
                "every sample). Use dft_window='rect' (the default) or a fixed n_steps run.")


def one_step_xs(count, *tables):
    """The scan input of step ``count`` alone (step index, then one row per
    table); past a table's end, the zero row the loops pad their tables with."""
    rows = tuple(t[count:count + 1] if count < t.shape[0]
                 else jnp.zeros((1,) + tuple(t.shape[1:]), t.dtype) for t in tables)
    return (jnp.arange(count, count + 1, dtype=jnp.int32),) + rows


def _interior(grid):
    """Non-absorber slice bounds; 2-D TMz keeps its single z cell."""
    z = (0, grid.nz) if getattr(grid, "is_2d", False) else (
        grid.pad_z_lo, grid.nz - grid.pad_z_hi)
    return ((grid.pad_x_lo, grid.nx - grid.pad_x_hi),
            (grid.pad_y_lo, grid.ny - grid.pad_y_hi), z)


def field_axes(grid):
    """(E axes, H axes) the grid's mode carries: 2-D TMz Ez, Hx, Hy; 2-D TEz
    Ex, Ey, Hz; 3-D all six."""
    mode = getattr(grid, "mode", "3d")
    if mode == "2d_tmz":
        return (2,), (0, 1)
    if mode == "2d_tez":
        return (0, 1), (2,)
    return (0, 1, 2), (0, 1, 2)


def _wide(x):
    """At least float32 (a float16 field of precision='mixed' squared overflows)."""
    x = jnp.asarray(x)
    return x.astype(jnp.promote_types(x.dtype, jnp.float32))


def _dot(a, b):
    # Real fields: the product itself. Complex (Bloch) envelopes: Re(a* b).
    return a * b if not jnp.iscomplexobj(a) else jnp.real(jnp.conj(a) * b)


def conserved_energy(state, next_h, eps, mu, volume, region, *, axes=((0, 1, 2), (0, 1, 2))):
    """``sum((eps E.E + (mu0/eps0) mu H.H') dV)`` over ``region``.

    ``state`` holds E^c and H^(c-1/2), ``next_h`` holds H^(c+1/2); ``eps``
    and ``mu`` are per-component relative arrays; ``volume`` is the cell
    volume (scalar or broadcastable to ``region``) or a pair (E volumes, H
    volumes) of per-component control volumes; ``axes`` = (E axes, H axes)
    of the grid's mode (``field_axes``).
    """
    e_axes, h_axes = axes
    field = lambda s, n, a: _wide(getattr(s, n + "xyz"[a])[region])
    e2 = [_wide(eps[a][region]) * _dot(field(state, "e", a), field(state, "e", a)) for a in e_axes]
    hh = [_wide(mu[a][region]) * _dot(field(state, "h", a), field(next_h, "h", a)) for a in h_axes]
    if isinstance(volume, tuple):
        ve, vh = volume
        return (sum(jnp.sum(x * _wide(ve[a])) for x, a in zip(e2, e_axes))
                + MU0_OVER_EPS0 * sum(jnp.sum(x * _wide(vh[a])) for x, a in zip(hh, h_axes)))
    return jnp.sum((sum(e2) + MU0_OVER_EPS0 * sum(hh)) * _wide(volume))


def yee_volumes(grid, region):
    """Graded mesh: each component's own Yee control volume on ``region``, as
    (E volumes, H volumes). E_a: primal width along a, dual widths across;
    H_a: dual width along a, primal widths across (``grid.cells``,
    ``grid.duals``). The single primal cell volume for all six left W
    drifting by 5e-3 in a closed lossless graded box."""
    primal = [jnp.asarray(np.asarray(grid.cells(a))[region[a]], jnp.float32) for a in range(3)]
    dual = [jnp.asarray(np.asarray(grid.duals(a))[region[a]], jnp.float32) for a in range(3)]
    box = lambda w: w[0][:, None, None] * w[1][None, :, None] * w[2][None, None, :]
    e = tuple(box([primal[b] if b == a else dual[b] for b in range(3)]) for a in range(3))
    h = tuple(box([dual[b] if b == a else primal[b] for b in range(3)]) for a in range(3))
    return e, h


def boundary_energy(grid, materials, side_step, *, dx=None,
                    periodic=(False, False, False)):
    """``measure(carry, count) -> float``: W at the chunk boundary ``count``.

    ``side_step = (program, xs_of, extra)``: ``program(carry, xs, extra)`` is
    the loop's own chunk program, run for one step on the scan input
    ``xs_of(count)`` of step ``count``; only the advanced carry's H fields,
    H^(c+1/2), are read. The step and the reduction are one jitted program
    that returns the scalar W, so the advanced carry never leaves it. ``eps`` and ``mu`` are the arrays the E and H
    updates read (the realized components, through ``rfx._realized`` when it
    observes); dV is ``dx**3`` with the loop's cell size on the uniform mesh
    and each component's Yee control volume on the graded mesh (``yee_volumes``). The sum covers the
    non-absorber interior. The per-cell arrays are arguments of the jitted
    reduction, not constants.
    """
    from rfx import _realized
    from rfx.core.jax_utils import split_loop_invariants
    from rfx.model.materials import with_components

    graded = hasattr(grid, "dx_arr")
    if not graded and dx is None:
        raise ValueError("the uniform energy measure needs the loop's cell size dx")
    periodic = tuple(a in grid.periodic_axes for a in "xyz") if graded else tuple(periodic)
    materials = with_components(materials, grid, periodic=periodic)
    sizes = (grid.dx_arr, grid.dy_arr, grid.dz) if graded else None
    region = tuple(slice(lo, hi) for lo, hi in _interior(grid))
    axes = field_axes(grid)
    arrays, rebuild = split_loop_invariants(dict(materials=materials), (grid.nx, grid.ny, grid.nz))
    if graded:
        arrays = dict(arrays=arrays, volumes=yee_volumes(grid, region))
        rebuild_inner = rebuild
        rebuild = lambda a: rebuild_inner(a["arrays"])

    def energy(state, next_h, arrays):
        m = rebuild(arrays)["materials"]
        eps, sigma, mu = (m.components.eps_update, m.components.sigma_update,
                          m.components.mu_update)
        if _realized.ACTIVE is not None:
            kw = dict(cell_sizes=sizes, periodic=periodic) if graded else dict(periodic=periodic)
            eps, _ = _realized.electric(m, eps, sigma, "yee.E", periodic=periodic)
            mu = _realized.magnetic(m, mu, "yee.H", **kw)
        volume = arrays["volumes"] if graded else dx ** 3
        return conserved_energy(state, next_h, eps, mu, volume, region, axes=axes)

    program, xs_of, extra = side_step

    @jax.jit
    def stepped_energy(carry, xs, extra, arrays):
        return energy(carry["fdtd"], program(carry, xs, extra)["fdtd"], arrays)

    def measure(carry, count):
        return float(stepped_energy(carry, xs_of(count), extra, arrays))

    # For inspection (tests): the lowered fused program at one boundary.
    measure.lower = lambda carry, count: stepped_energy.lower(carry, xs_of(count), extra, arrays)
    return measure


@dataclass
class DecayStop:
    """Host rule at chunk boundaries ``count = k * interval``.

    With the energy criterion as the run's stop rule (``enabled``), W is
    formed at every boundary to track ``wmax`` (drive included); ``ref`` is
    the running max of W over boundaries at or after ``n_end``. A boundary
    ``count >= n_end`` is below when ``W < thr * ref`` (on the threshold) or
    when ``W < KAPPA * wmax`` and ``W >= FLOOR_FLAT ** (interval / FLOOR_SPAN) *
    W_previous`` (on the floor: less than 5 % decay per 50 steps); ``below`` counts consecutive ones and resets
    otherwise. ``stop`` is the
    first boundary with ``below >= consecutive`` and ``count >= min_steps``;
    ``criterion`` is then "threshold" when every boundary of that final
    consecutive run had ``W < thr * ref``, else "float32_floor". Not enabled (a forced-length run that still
    writes rows), W is formed only where a ``decay_checks`` row is written.
    With ``n_end`` None (a drive still on) nothing is read and nothing
    stops. ``enabled`` refuses ``consecutive < 2``.
    """

    thr: float
    min_steps: int
    interval: int
    consecutive: int
    n_end: int | None
    enabled: bool = True
    ref: float = 0.0
    wmax: float = 0.0
    below: int = 0
    below_thr: int = 0   # consecutive boundaries with W < thr * ref
    w_prev: float | None = None   # W at the previous boundary read
    stop: int | None = None
    criterion: str | None = None
    last: tuple | None = None   # (count, W, ref, wmax) at the last boundary >= n_end
    decay_checks: list = field(default_factory=list)

    def __post_init__(self):
        if self.enabled and self.consecutive < 2:
            raise ValueError(
                "until_decay: decay_energy_consecutive must be >= 2 with the energy "
                f"criterion, got {self.consecutive} (one sub-threshold check stops on "
                "a transient dip; zero or less stops at min_steps whatever the energy)")

    def check(self, count, energy):
        """``energy()`` is called only at a boundary the rule reads."""
        if self.n_end is None or count % self.interval:
            return self.stop
        if not self.enabled:
            if count >= max(self.n_end, self.min_steps):
                w = float(energy())
                self.ref = max(self.ref, w)
                self.decay_checks.append((count, w, self.ref))
            return self.stop
        w = float(energy())
        self.wmax = max(self.wmax, w)
        previous, self.w_prev = self.w_prev, w
        if count < self.n_end:
            return self.stop
        self.ref = max(self.ref, w)
        on_floor = (w < KAPPA * self.wmax and previous is not None
                    and w >= FLOOR_FLAT ** (self.interval / FLOOR_SPAN) * previous)
        self.below = self.below + 1 if (w < self.thr * self.ref or on_floor) else 0
        self.below_thr = self.below_thr + 1 if w < self.thr * self.ref else 0
        self.last = (count, w, self.ref, self.wmax)
        if count >= self.min_steps:
            self.decay_checks.append((count, w, self.ref))
            if self.stop is None and self.below >= self.consecutive:
                self.stop = count
                self.criterion = ("threshold" if self.below_thr >= self.below
                                  else "float32_floor")
        return self.stop


def make_decay_stop(grid, materials, *, thr, min_steps, interval, consecutive,
                    criterion, max_steps, side_step, n_end=AUTO_END, rows=False, dx=None,
                    sources=(), mag_sources=(), tfsf=None, waveguide_ports=(),
                    periodic=(False, False, False)):
    """The rule and its W measurement for one loop.

    ``criterion``: the energy criterion is this run's stop rule (absorbing
    boundary, no flux box). ``rows``: the loop returns ``decay_checks``
    (graded), so a ``thr = 0`` run still writes its rows. ``n_end`` is the
    runner's value; direct loop callers leave it to ``drive_end`` here.
    ``side_step = (program, xs_of, extra)``: one step of the loop's own chunk
    program (see ``boundary_energy``). ``dx``: the uniform loop's cell size.
    """
    enabled = criterion and thr > 0
    if not criterion or not (enabled or (rows and min_steps <= max_steps)):
        n_end = None
    elif n_end is AUTO_END:
        try:
            n_end = drive_end(grid, max_steps, sources=sources, mag_sources=mag_sources,
                              tfsf=tfsf, waveguide_ports=waveguide_ports)
        except ValueError:
            if enabled:
                raise
            n_end = None   # rows only: a forced-length run is never refused
    rule = DecayStop(thr, min_steps, interval, consecutive, n_end, enabled)
    measure = (boundary_energy(grid, materials, side_step, dx=dx, periodic=periodic)
               if n_end is not None else None)
    return rule, measure


# --- how the run ended ---------------------------------------------------------

def _stop_record(rule, steps, *, fired, flux, point):
    """(quantity, criterion, reached, floor_ratio) of one run end."""
    nan = float("nan")
    if flux is not None or point is not None:
        quantity = "radiated_flux" if flux is not None else "point_field"
        if flux is not None:
            history, checks, peak = flux
            reached = max(history[-checks:]) / peak if history and peak > 0 else nan
        else:
            value, peak = point
            reached = value / peak if peak > 0 else nan
        return quantity, "threshold" if fired else "cap", reached, nan
    if rule.stop is not None:
        criterion = rule.criterion
    else:
        criterion = "cap"
    if rule.last is None or rule.last[2] <= 0:
        return "conserved_energy", criterion, nan, nan
    _, w, ref, wmax = rule.last
    return "conserved_energy", criterion, w / ref, KAPPA * wmax / ref


def finish_decay(rule, steps, *, fired=False, flux=None, point=None, reporter=None,
           state=None, materials=None, grid=None, advisory=None, sources=()):
    """One ``until_decay.stop`` Diagnostic for a run that ended at ``steps``.

    ``rule`` is the loop's DecayStop. ``flux`` = (|P| history, envelope
    checks, peak) when the radiated-flux criterion was the stop rule;
    ``point`` = (last squared sample, peak) for the closed-box point stop;
    ``fired`` says whether either of them stopped the run. Data:
    ``criterion`` ("threshold", "float32_floor" or "cap"),
    ``requested_threshold``, ``reached`` (W_c/ref at the stop, or at the last
    boundary read for a cap; that rule's own ratio for flux and point),
    ``floor_ratio`` (KAPPA * Wmax_c / ref), ``stop_step``,
    ``drive_end_step`` and ``quantity``; NaN where a value does not exist.
    A float32-floor end also warns once (UserWarning). An energy "cap" end
    runs the static-remnant advisory with ``advisory`` (``advisory_inputs``;
    None: from the loop's own ``sources`` tables and material arrays). With a progress reporter the
    criterion line is printed as the last line. Returns None for a
    forced-length run (``thr = 0``).
    """
    import warnings

    from rfx.diagnostic_records import Diagnostic

    if not rule.thr > 0:
        return None
    quantity, criterion, reached, floor_ratio = _stop_record(
        rule, steps, fired=fired, flux=flux, point=point)
    nan = float("nan")
    head = f"until_decay ended at step {steps} ({quantity})"
    if criterion == "threshold":
        message = (f"{head}: requested threshold {rule.thr:.3g}, reached {reached:.3g}; "
                   "the requested threshold was met")
    elif criterion == "float32_floor":
        _, w, _, wmax = rule.last
        message = (f"{head}: requested threshold {rule.thr:.3g}; the energy stopped falling at "
                   f"{reached:.3g} of the after-drive reference ({w / wmax:.3g} of the drive's "
                   f"peak energy); {FLOOR_TEXT}")
        warnings.warn(message, UserWarning, stacklevel=3)
    elif quantity == "conserved_energy" and rule.n_end is None:
        message = (f"{head}: requested threshold {rule.thr:.3g} not met at decay_max_steps; "
                   "a drive was still on at decay_max_steps, so the energy stop could not fire")
    else:
        message = (f"{head}: requested threshold {rule.thr:.3g} not met at decay_max_steps; "
                   f"last level reached {reached:.3g}")
    if criterion == "cap" and quantity == "conserved_energy" and state is not None:
        if advisory is None:
            advisory = dict(drives=table_dc(sources), lossy=None)
        static_remnant_advisory(state, materials, grid, advisory["drives"], advisory["lossy"])
    if reporter is not None:
        reporter.note(f"{criterion} | {message}")
    return Diagnostic(
        "until_decay.stop", "info" if criterion == "threshold" else "warning", criterion,
        message,
        {"criterion": criterion, "requested_threshold": float(rule.thr),
         "reached": float(reached), "floor_ratio": float(floor_ratio),
         "stop_step": int(steps),
         "drive_end_step": nan if rule.n_end is None else int(rule.n_end),
         "quantity": quantity})


def decay_stop_first(result, settling_db, witness):
    """The record-length verdict (``Result.settling_witness``) of an until_decay
    run that ended on the float32 floor or the cap: the stop line goes first in
    its ``reason``, which run()'s settling warnings quote."""
    stops = [d for d in getattr(result, "diagnostics", ()) if d.code == "until_decay.stop"]
    if (isinstance(witness, dict) and stops
            and stops[0].values.get("criterion") in ("float32_floor", "cap")):
        witness = {**witness, "reason": f"{stops[0].message}; {witness.get('reason', '')}"}
    return settling_db, witness


# --- DC advisories ---------------------------------------------------------------

def _relative_dc(samples, dt):
    """(|sum s dt| / sum |s| dt, sum s dt), or None for an empty/nonfinite drive."""
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 1 or not np.all(np.isfinite(s)):
        return None
    denom = float(np.sum(np.abs(s))) * dt
    if denom == 0.0:
        return None
    s_int = float(np.sum(s)) * dt
    return abs(s_int) / denom, s_int


def dc_census(sim, *, dt, n_table):
    """Each soft drive's waveform DC over the planned table.

    Rows ``(label, class name, relative DC, sum s dt, t_done)`` for every
    soft source (``impedance == 0`` excited port entries) and excited MSL
    port (the same soft-Ez deposit), with ``relative DC = |sum s(t_n) dt| /
    sum |s(t_n)| dt`` over ``n_table`` steps and ``t_done = t0 + 4.5 tau``
    (None when the waveform has no ``tau``). A soft current source deposits
    ``sum s dt`` as a static end-charge (issue #388).
    """
    from rfx.core.jax_utils import is_tracer

    if is_tracer(dt):
        return []   # traced dt: advisory host math is impossible here
    dt = float(dt)
    if not dt > 0.0 or n_table <= 0:
        return []
    t = np.arange(int(n_table), dtype=np.float64) * dt
    soft = [pe for pe in sim._ports if pe.impedance == 0.0 and getattr(pe, "excite", True)]
    soft += [pe for pe in sim._msl_ports if getattr(pe, "excite", True)]
    rows = []
    for pe in soft:
        wf = getattr(pe, "waveform", None)
        if wf is None or isinstance(wf, str):
            continue
        try:
            s = np.asarray(wf(jnp.asarray(t)), dtype=np.float64)
        except Exception:
            continue  # non-vectorizable CustomWaveform etc. -- skip
        dc = _relative_dc(s, dt) if s.shape == t.shape else None
        if dc is None:
            continue
        t_done = None
        try:
            tau = float(wf.tau)
        except (AttributeError, TypeError, ValueError):
            tau = None
        if tau is not None and math.isfinite(tau) and tau > 0.0:
            try:
                t0 = float(wf.t0)
            except (AttributeError, TypeError, ValueError):
                t0 = 3.0 * tau
            # The differentiated Gaussian is below 1.4e-8 of its amplitude there.
            t_done = t0 + 4.5 * tau
        label = f"{pe.position} ({getattr(pe, 'component', 'msl port')})"
        rows.append((label, type(wf).__name__, dc[0], dc[1], t_done))
    return rows


def table_dc(sources):
    """``dc_census`` rows from a loop's own source tables (direct loop calls,
    which have no waveform objects): label ``source[i]``, class "drive table"."""
    from rfx.core.jax_utils import is_tracer

    rows = []
    for index, src in enumerate(sources or ()):
        table = src.waveform if hasattr(src, "waveform") else src[4]
        if table is None or is_tracer(table):
            continue
        dc = _relative_dc(np.asarray(table), 1.0)
        if dc is not None:
            rows.append((f"source[{index}]", "drive table", dc[0], dc[1], None))
    return rows


def warn_dc_floor(sim, *, dt, n_table):
    """Pre-run predictor (issue #388): warn for each soft drive whose waveform's
    relative DC over the planned ``n_table``-step table exceeds ``DC_ADVISE``.

    The deposited static charge's field neither decays nor is absorbed by
    CPML; under the after-drive reference a relative DC of 6e-5 already
    blocked the stop, so the threshold is 1e-6. When the table ends inside
    the pulse (``n_table * dt < t0 + 4.5 tau``) the DC is the truncation's,
    and the message says so instead.
    """
    import warnings

    for label, name, rel_dc, s_int, t_done in dc_census(sim, dt=dt, n_table=n_table):
        if rel_dc <= DC_ADVISE:
            continue
        if t_done is not None and n_table * float(dt) < t_done:
            warnings.warn(
                f"soft source at {label}: waveform {name} shows relative DC {rel_dc:.2e} "
                f"over the planned {int(n_table)}-step table, but the planned "
                f"decay_max_steps window ends inside the source pulse (table "
                f"{n_table * float(dt):.3e} s < pulse completion t0+4.5*tau ~ {t_done:.3e} s) "
                "— the DC estimate is truncation-dominated, not intrinsic waveform DC; "
                "until_decay may cap-hit (issue #388). Increase decay_max_steps (or run "
                "with a fixed n_steps) so the table covers the full pulse; raising the "
                "waveform cutoff makes this WORSE (it lengthens the onset).",
                UserWarning, stacklevel=3)
        else:
            warnings.warn(
                f"soft source at {label}: waveform {name} has relative DC {rel_dc:.2e} "
                f"(|sum s*dt| = {abs(s_int):.3e} over the planned {int(n_table)}-step "
                "table): it deposits a static charge whose field neither decays nor is "
                "absorbed by CPML, so the energy left after the drive may never fall to "
                "the until_decay threshold and the run may cap-hit (issue #388); thin "
                "source cells make a small charge's field large, so this warns from "
                f"relative DC {DC_ADVISE:g}. Remedies: {REMEDIES}.",
                UserWarning, stacklevel=3)


def advisory_inputs(sim, *, dt, n_table):
    """What the cap advisory names: ``drives``, the soft drives' DC
    (``dc_census`` rows), and ``lossy``, every material the model places
    with ``sigma > 0`` as ``(name, eps_r, sigma)``."""
    lossy, seen = [], set()
    for entry in getattr(sim, "_geometry", ()):
        name = entry.material_name
        if name not in seen:
            seen.add(name)
            spec = sim._resolve_material(name)
            if spec.sigma > 0:
                lossy.append((repr(name), float(spec.eps_r), float(spec.sigma)))
    return dict(drives=dc_census(sim, dt=dt, n_table=n_table), lossy=lossy)


def array_lossy(materials):
    """``advisory_inputs`` lossy rows from a loop's material arrays (direct
    loop calls, which have no material names): distinct (eps_r, sigma > 0)."""
    sigma = np.atleast_1d(np.asarray(materials.sigma, dtype=np.float64))
    eps = np.broadcast_to(np.asarray(materials.eps_r, dtype=np.float64), sigma.shape)
    lossy = sigma > 0
    if not lossy.any():
        return []
    pairs = np.unique(np.stack([eps[lossy], sigma[lossy]], axis=1), axis=0)
    return [(f"cells with eps_r {e:g}, sigma {g:g} S/m", float(e), float(g)) for e, g in pairs[:5]]


def static_remnant_advisory(state, materials, grid, soft_drives=(), lossy=None) -> None:
    """Post-run advisory on an energy "cap" end (issue #388).

    Warns when the final interior energy is electrostatic: physical energy
    (1/2 eps |E|^2 against 1/2 mu |H|^2) with ``H-share < 1e-2``. A
    propagating field is equipartitioned (H-share about 0.5, and about 0.2
    even at an E-max instant), a static remnant is E-only (about 1e-10).
    Names each soft drive in ``soft_drives`` (``dc_census`` rows) whose
    relative DC exceeds ``DC_ADVISE``, with its class and value. When none
    does, the remnant is quasi-static charge left by the drive or relaxing in
    a lossy medium: each lossy material (``lossy`` rows; None: from the
    material arrays) whose eps/sigma is at least one step is named with that
    relaxation time in steps.
    """
    import warnings

    from rfx.core.yee import component_h_materials

    e_axes, h_axes = field_axes(grid)
    (ix0, ix1), (iy0, iy1), (iz0, iz1) = _interior(grid)
    sl = (slice(ix0, ix1), slice(iy0, iy1), slice(iz0, iz1))
    eps_r = _wide(materials.eps_r[sl])
    state = state._replace(**{n: _wide(getattr(state, n)) for n in ("ex", "ey", "ez", "hx", "hy", "hz")})
    mu_h = component_h_materials(
        materials, cell_sizes=(grid.dx_arr, grid.dy_arr, grid.dz)
        if hasattr(grid, "dx_arr") else None)
    mu_h = tuple(_wide(m[sl]) if jnp.ndim(m) else m for m in mu_h)
    e2 = sum(getattr(state, "e" + "xyz"[a])[sl] ** 2 for a in e_axes)
    u_e = 0.5 * EPS_0 * float(jnp.sum(eps_r * e2))
    if h_axes == (0, 1, 2):
        h2 = state.hx[sl] ** 2 + state.hy[sl] ** 2 + state.hz[sl] ** 2
        weighted_h2 = (mu_h[0] * state.hx[sl] ** 2 + mu_h[1] * state.hy[sl] ** 2
                       + mu_h[2] * state.hz[sl] ** 2)
        same = (mu_h[0] == mu_h[1]) & (mu_h[0] == mu_h[2])
        # Retain the homogeneous diagnostic's operation order and bits.
        weighted_h2 = jnp.where(same, mu_h[0] * h2, weighted_h2)
    else:
        weighted_h2 = sum(mu_h[a] * getattr(state, "h" + "xyz"[a])[sl] ** 2 for a in h_axes)
    u_h = 0.5 * MU_0 * float(jnp.sum(weighted_h2))
    tot = u_e + u_h
    if tot <= 0.0:
        return
    h_share = u_h / tot
    if h_share >= 1e-2:
        return
    named = [f"{label}: {name}, relative DC {rel:.2e}"
             for label, name, rel, _, _ in soft_drives if rel > DC_ADVISE]
    if not named:
        lossy = array_lossy(materials) if lossy is None else lossy
        # A conductor relaxes within a step (eps/sigma < dt): not a lossy medium here.
        media = [f"{name} (eps_r {eps:g}, sigma {sig:g} S/m): eps/sigma = {EPS_0 * eps / sig:.3g} s = "
                 f"{EPS_0 * eps / sig / float(grid.dt):.0f} steps" for name, eps, sig in lossy
                 if EPS_0 * eps / sig >= float(grid.dt)]
        warnings.warn(
            f"run(until_decay=...) cap-hit at max_steps without the energy criterion firing, "
            f"and {100 * (1 - h_share):.1f}% of the final interior energy is electrostatic "
            f"(H-share {h_share:.1e}). No soft drive has relative DC above {DC_ADVISE:g}: the "
            "remnant is quasi-static charge left by the drive, or charge relaxing in a lossy "
            "medium (relaxation time eps/sigma). Lossy materials: "
            f"{'; '.join(media) if media else 'none (no sigma > 0 in the model)'}. Its field is "
            "out of band and the record stays correct (issue #388). Remedies: a fixed n_steps, "
            "radiated_flux_box=((x0,y0,z0),(x1,y1,z1)) enclosing the radiator, or a larger "
            "until_decay threshold.",
            UserWarning, stacklevel=3)
        return
    warnings.warn(
        f"run(until_decay=...) cap-hit at max_steps without the energy criterion firing, "
        f"and {100 * (1 - h_share):.1f}% of the final interior energy is electrostatic "
        f"(H-share {h_share:.1e}): a static charge left by a soft drive holds the energy "
        "above the threshold; it neither radiates nor is absorbed by CPML (issue #388). "
        "Soft drives with relative DC above "
        f"{DC_ADVISE:g}: {'; '.join(named) if named else 'none found'}. "
        f"Remedies: {REMEDIES}.",
        UserWarning, stacklevel=3)
