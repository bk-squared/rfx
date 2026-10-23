"""Compiled FDTD simulation runner.

Composes Yee updates, boundaries, sources, probes, and optional TFSF
plane-wave injection into a single JIT-compiled time loop via
``jax.lax.scan``. All subsystem selection (CPML, dispersion, TFSF) is
resolved at Python trace-time so the compiled function contains only
the needed code paths.
"""

from __future__ import annotations

from rfx._uniform_specs import (
    DesignBoxSpec,
    DesignOccupancySpec,
    ProbeSpec,
    SimResult,
    SnapshotSpec,
    SourceSpec,
    _ORDER4_CFL_FACTOR,
    _StepContext,
    _nearest_divisor,
    resolve_periodic,
)


from rfx.measurement import init_uniform_records, uniform_records

from dataclasses import dataclass, replace
from typing import Any, Callable, NamedTuple

from rfx import _realized, _grid_metric

import jax
import jax.numpy as jnp

from rfx.core.drives import StepDrives, drive_layout, inject_drives
from rfx.measurement import AUTO_END, finish_decay, make_decay_stop, one_step_xs
from rfx.model.source_coefficients import drive_table
from rfx.model import with_edge_keep
import numpy as np

from rfx.boundaries.axes import drop_periodic_axes, resolve_cpml_axes
from rfx.grid import Grid
from rfx.stepping import recorded_scan, update_e, update_e_aniso, update_e_aniso_inv, update_e_box, update_h
from rfx.core.yee import (
    FDTDState, MaterialArrays, init_state,
    e_update_coeffs, component_e_materials,
    EPS_0, MU_0, curl_h, CurlBoundary,
    map_lumped, lumped_total,
    precompute_coeffs, update_he_fast,
)
from rfx.boundaries.pec import (
    apply_pec,  # noqa: F401 -- re-exported; tests import it from here (not used by the step since #1164)
    resolve_wall_faces,
    apply_pec_edges,
    apply_pec_faces,
    kottke_fenced_edge_masks,
    realized_pec_edge_masks,
)
from rfx.progress import (
    ProgressReporter, check_not_traced, concat_chunks, scan_with_progress,
    validate_report_every,
)
from rfx.snapshots import (
    plan_snapshot_pieces, snapshot_axes, snapshot_extractor,
    validate_snapshot_spec,
)


# CFL derating for the (2,4) fourth-order-in-space stencil.  The wider
# staggered difference raises the maximum spatial wavenumber the scheme
# resolves, so the explicit-leapfrog stability bound tightens: the (2,4)
# stable timestep is ~0.857x the (2,2) Courant limit (the 1D bound is
# 6/7 ≈ 0.857; see the kernel header in rfx/core/yee.py).  Applied only
# when stencil_order == 4; order=2 keeps grid.dt unchanged (byte-identical).


# ---------------------------------------------------------------------------
# Source / probe specifications
# ---------------------------------------------------------------------------



class MagneticSourceSpec(NamedTuple):
    """Precomputed H-component current source (Schelkunoff M/J magnetic).

    Used for the Schelkunoff J+M one-sided MSL port launch. The waveform
    is precomputed to include the -dt/mu coefficient so injection is a
    plain add (same pattern as SourceSpec for E fields).

    component : "hx", "hy", or "hz"
    waveform  : (n_steps,) float array — values added to the H field
        component at (i, j, k) each timestep (after the H update).
    """
    i: int
    j: int
    k: int
    component: str
    waveform: jnp.ndarray




class WirePortSParamSpec(NamedTuple):
    """Wire port S-param DFT specification for the compiled runner.

    Pre-computed metadata for inline V/I DFT accumulation inside
    jax.lax.scan, avoiding the separate Python-loop S-param extraction.

    mid_i, mid_j, mid_k: midpoint cell for V and I measurement — the
        midpoint of the LIVE wire run (issue #764; identical to the
        historical all-extent midpoint when no extent cell is dead).
    component: 'ez', 'ex', or 'ey'.
    freqs: (n_freqs,) frequency array.
    impedance: port reference impedance (Z0).
    live_cells: static tuple of (i, j, k) LIVE wire cells (issue #764) for
        the whole-port gap-voltage accumulator
        V_port = sum over live cells of -E_c*dx.  Empty tuple = degenerate
        fallback to the single midpoint cell (pre-#764 constructors).
    excite: whether this port is genuinely driven in THIS pass.  The
        whole-port driven-diagonal formula
        S_kk = (V_port - Z0*I)/(V_port + Z0*I) applies ONLY when True;
        a passive port's diagonal keeps the legacy per-cell convention
        (a load-independent port-cell diagnostic, not S_jj — issue #764).
    """
    mid_i: int
    mid_j: int
    mid_k: int
    component: str
    freqs: jnp.ndarray
    impedance: float
    live_cells: tuple = ()
    excite: bool = True


class LumpedPortSParamSpec(NamedTuple):
    """Lumped port S-param DFT specification for the compiled runner.

    Pre-computed metadata for inline V/I DFT accumulation inside
    jax.lax.scan, providing an AD-compatible alternative to the
    Python-loop ``extract_s_matrix`` path for single-cell lumped ports.

    Attributes
    ----------
    i, j, k : int
        Lumped port cell index.
    component : str
        E-field component ("ex", "ey", or "ez").
    freqs : (n_freqs,) jnp.ndarray
        Frequencies (Hz) at which to accumulate V and I DFTs.
    impedance : float
        Port reference impedance Z0 (ohms).
    excite : bool
        Whether this port is genuinely driven in THIS pass.  A driven port
        reads its diagonal from the terminal V/I pair
        (:func:`rfx.probes.probes.driven_port_reflection`); a passive one
        keeps the load-independent port-branch reading
        (:func:`rfx.probes.probes.extract_lumped_s11`).  Mirrors
        :class:`WireSParamSpec.excite`.

    Notes
    -----
    V is sampled as ``-E·dx`` and I as the curl-H loop integral times
    ``dx`` at the port cell, both AFTER source injection, with the Yee
    half-step phase on I.  Issue #72; sampling slot and diagonal formula
    decided by scripts/diagnostics/lumped_port_known_load_line.py.
    """
    i: int
    j: int
    k: int
    component: str
    freqs: jnp.ndarray
    impedance: float
    excite: bool = True










# ---------------------------------------------------------------------------
# Helpers to build source / probe specs
# ---------------------------------------------------------------------------

def _source_cell_cb(grid: Grid, idx, materials, component,
                    periodic=(False, False, False)):
    """Cb of the E update for ``component`` at the source cell.

    Single spelling of the coefficient shared by ``make_source`` and
    ``make_j_source``. Since #1210 it is the PER-COMPONENT coefficient:
    ``make_j_source`` exists to turn a current into the field increment the
    update itself would have produced, and the update multiplies each
    component by the mean of eps and sigma over the four cells its edge
    touches. Reading the owning cell here instead put 0.556x the declared
    current into an Ey source standing on an eps 1|9 step along y.
    ``materials.eps_r``/``sigma`` may be JAX tracers (forward/AD path) —
    The model accessor reads the finished edge and retains JAX derivatives.
    """
    from rfx.model.materials import e_update_coefficient_at, electric_grid_kwargs
    return e_update_coefficient_at(materials, idx, component, grid.dt, periodic, **electric_grid_kwargs(grid))


def _uniform_cell_volume(grid: Grid) -> float:
    """dV = dx*dy*dz with missing axes duck-typed to dx (engineering
    principle 3). A 2D grid is treated as ONE CELL DEEP (dz -> dx), so
    dV = dx**3 there too — the documented #571 convention
    (``rfx/api/_source_semantics.py``), not per-unit-length dx**2."""
    return grid.dx * getattr(grid, "dy", grid.dx) * getattr(grid, "dz", grid.dx)


def make_source(grid: Grid, position, component, waveform_fn, n_steps,
                materials=None, amplitude_kind=None):
    """Create a SourceSpec by precomputing a waveform function (raw E-field add).

    WARNING: raw field source causes DC accumulation on PEC surfaces.
    Prefer ``make_j_source`` for resonance detection.

    Native amplitude convention: ``'field'`` — ``E += w``.
    ``amplitude_kind='current'`` uses the shared source-table builder's
    ``Cb/dV`` coefficient and REQUIRES
    ``materials`` (for Cb at the source cell). ``amplitude_kind`` of
    None or ``'field'`` is bit-identical to the historical output (no
    multiply is applied at all). ``materials`` is accepted and unused in
    that legacy/native case.
    """
    from rfx.model.source_coefficients import uniform_source_table
    idx = _grid_metric.field_index(grid, position, component)
    if amplitude_kind == "current" and materials is None:
        raise ValueError(
            "make_source(amplitude_kind='current') needs materials "
            "for the Cb normalization at the source cell")
    waveform = uniform_source_table(
        grid, idx, component, waveform_fn, n_steps, materials, amplitude_kind)
    return SourceSpec(i=idx[0], j=idx[1], k=idx[2],
                      component=component, waveform=waveform)


def make_j_source(grid: Grid, position, component, waveform_fn, n_steps, materials,
                  amplitude_kind=None):
    """Create a SourceSpec using current density injection (J source).

    Unlike ``make_source``, the waveform is Cb-normalized so that
    the source enters through the Ampere update equation:

    E += Cb * waveform(t)

    where Cb = dt / (eps * (1 + sigma*dt/2eps)).

    Note (scoped to ``amplitude_kind=None``): this is a soft *field*
    source, not a current-density source. Injected amplitude depends on
    local material (eps, sigma) but NOT on cell size. Power coupling
    varies with resolution; use Harminv/FFT for mode extraction rather
    than absolute amplitude.

    Native amplitude convention (issue #571): ``'cb'`` — ``E += Cb*w`` —
    which is NEITHER named kind (the legacy open-uniform third contract).
    ``amplitude_kind='current'`` uses ``Cb*w/dV``; ``'field'`` returns the
    waveform directly through the shared source-table builder.
    None is bit-identical to the historical output (no multiply applied).

    Parameters
    ----------
    grid : Grid
    position : (x, y, z) in metres
    component : "ex", "ey", or "ez"
    waveform_fn : callable(t) -> value
    n_steps : int
    materials : MaterialArrays (for Cb computation at source cell)
    amplitude_kind : 'field' | 'current' | None (issue #571, above)
    """
    from rfx.model.source_coefficients import uniform_source_table
    idx = _grid_metric.field_index(grid, position, component)
    i, j, k = idx
    waveform = uniform_source_table(
        grid, idx, component, waveform_fn, n_steps, materials, amplitude_kind,
        native="cb")
    return SourceSpec(i=i, j=j, k=k,
                      component=component, waveform=waveform)


def make_port_source(grid: Grid, port, materials: MaterialArrays, n_steps):
    """Create a SourceSpec for a lumped port (Cb-corrected waveform).

    The port impedance must already be folded into *materials* via
    ``setup_lumped_port()``.
    """
    from rfx.sources.port_drive import stamped_drive, port_drive_waveform
    idx = _grid_metric.field_index(grid, port.position, port.component)
    i, j, k = idx

    waveform = port_drive_waveform(
        grid, idx, port.component, port.excitation, n_steps, materials,
        sigma_port=stamped_drive(port, idx)[0],
        unit_field=stamped_drive(port, idx)[1])
    return SourceSpec(i=i, j=j, k=k,
                      component=port.component, waveform=waveform)


def make_wire_port_sources(grid, port, materials, n_steps, pec_edge_masks=None):
    """Create a list of SourceSpec for a multi-cell WirePort.

    Each LIVE cell gets a Thevenin source voltage w/n_live and resistance
    R/n_live: its Norton current is w/R. Dead extent cells inside PEC get
    no source (issue #318). The port impedance must
    already be folded into *materials* via ``setup_wire_port()``.

    Returns
    -------
    list[SourceSpec]
    """
    from rfx.sources.sources import _wire_port_live_cells

    cells, live_flags, n_live = _wire_port_live_cells(grid, port, pec_edge_masks)
    from rfx.sources.port_drive import stamped_drive, port_drive_waveform

    specs = []
    for cell, live in zip(cells, live_flags):
        if not live:
            continue
        i, j, k = cell
        waveform = port_drive_waveform(
            grid, cell, port.component, port.excitation, n_steps, materials,
            sigma_port=stamped_drive(port, cell)[0],
            unit_field=stamped_drive(port, cell)[1])
        specs.append(SourceSpec(i=i, j=j, k=k,
                                component=port.component, waveform=waveform))
    return specs


def make_probe(grid: Grid, position, component):
    """Create a ProbeSpec from a physical position."""
    idx = _grid_metric.field_index(grid, position, component)
    return ProbeSpec(i=idx[0], j=idx[1], k=idx[2], component=component)




# ---------------------------------------------------------------------------
# Compiled runner
# ---------------------------------------------------------------------------





def _suggest_checkpoint_segments(n_steps: int) -> int:
    """Auto pick K ≈ √n_steps that divides n_steps."""
    import math
    return _nearest_divisor(n_steps, max(1, int(math.isqrt(n_steps))))


# ---------------------------------------------------------------------------
# Shared setup helper (W6.2)
# ---------------------------------------------------------------------------
#
# ``run()`` and ``run_until_decay()`` share ~73% of their setup code: boundary
# resolution, subsystem flags, carry_init assembly, metadata extraction, and
# the _StepContext field population.  ``_build_step_setup`` centralises that
# shared work and returns a ``_SimSetup`` NamedTuple.  Each driver then
# applies its handful of driver-specific overrides before constructing the
# final ``_StepContext``.
#
# Intentional behavioural differences preserved:
#   * ``flux_meta_11`` (11-field, with window name/alpha) used by ``run()``;
#     ``flux_meta_8`` (8-field, no window) used by ``run_until_decay()``
#     (use_flux_window=False).  Unifying decay onto windowed DFT is a
#     deliberate future task.
#   * ``use_fast_he`` / ``fast_coeffs`` — ``run()`` only; decay path is a
#     chunked scan with the fast path deliberately OFF to preserve its
#     standalone-step arithmetic (no GPU fast-path change in #1322).







from rfx.model import design_box_bounds, resolve_design_box




# ---------------------------------------------------------------------------
# Shared Yee scan body (W6.1)
# ---------------------------------------------------------------------------
#
# ``run()`` and ``run_until_decay()`` (formerly Python loop + jax.jit)
# ran two ~85%-identical copies of the per-step Yee kernel.  ``make_core_step``
# is the single source of truth.  Both call sites build a ``_StepContext`` from
# their own setup code, then:
#   * ``run()``           wraps ``core`` in a scan body that unpacks ``xs`` and
#                         assembles the scan output tuple (probe + snapshot).
#   * ``run_until_decay`` scans ``core`` in chunks and reads every step
#                         of ``extras["monitor_val"]`` at chunk boundaries.
#
# Every free variable of the old closures is passed explicitly via the context
# (no capture of caller locals) so the builder is unit-testable.  Numerics,
# Yee sub-step ordering, and dtype casts are unchanged — this is pure code
# motion.  The two documented behavioural differences are parameterised:
#   * ``use_fast_he`` / ``fast_coeffs`` — GPU baked-PEC path (run() only today).
#   * ``use_flux_window`` + 11-field ``flux_meta`` — streaming DFT window for
#     flux monitors.  ``run_until_decay`` passes ``use_flux_window=False`` and
#     an 8-field ``flux_meta`` to keep its historical rect-window (no weight)
#     behaviour bit-identical.  Unifying the decay path onto windows is a
#     deliberate follow-up, not part of this refactor.










# ---------------------------------------------------------------------------
# Field-decay-based stopping criterion (host checks + compiled scan chunks)
# ---------------------------------------------------------------------------


from rfx.stepping import make_core_step
from rfx.stepping import run_uniform_loop as run
from rfx.stepping import run_uniform_until_decay as run_until_decay
from rfx.stepping import core_step_invariants
