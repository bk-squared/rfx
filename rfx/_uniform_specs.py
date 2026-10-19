"""Shared uniform-runner records and scalar helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax.numpy as jnp

from rfx.core.drives import StepDrives
from rfx.core.yee import FDTDState
from rfx.grid import Grid


_ORDER4_CFL_FACTOR = 0.857


class SourceSpec(NamedTuple):
    """Precomputed point source for the compiled runner.

    waveform : (n_steps,) float array — precomputed values added to the
        field component at (i, j, k) each timestep.
    """
    i: int
    j: int
    k: int
    component: str
    waveform: jnp.ndarray


class ProbeSpec(NamedTuple):
    """Point probe that records a field component each timestep."""
    i: int
    j: int
    k: int
    component: str


class DesignBoxSpec(NamedTuple):
    """One static box whose E update runs on its own permittivity (#1179).

    The box is a design region: its permittivity is the quantity being
    differentiated. Handing it here instead of writing it into
    ``materials.eps_r`` keeps the grid-wide update coefficients constant, so
    reverse-mode AD stores box-shaped arrays per timestep instead of
    grid-shaped ones. The physics is unchanged — ``update_e_box`` recomputes
    exactly what ``update_e`` would have computed at those cells.

    bounds : (i0, i1, j0, j1, k0, k1)
        Half-open cell index bounds of the box, resolved from the declared
        corners against the grid the solve builds.
    eps_r : box-shaped array (usually a tracer)
        Relative permittivity at the box cells.
    sigma : box-shaped array, a 3-tuple of them, or None
        Conductivity at the box cells. ``None`` (default) takes the run's own
        ``materials.sigma`` slice, so a lossy background inside the box is
        carried exactly rather than silently dropped.

        A single array is the one conductivity every E component in the box
        sees, exactly as ``materials.sigma`` is read by ``update_e``. A
        3-tuple ``(sigma_x, sigma_y, sigma_z)`` of box-shaped arrays gives
        each component its OWN conductivity, and therefore its own Ca/Cb.
        What needs it: a conducting SHEET has current only along its two
        in-plane edges, so a design variable per in-plane edge must not
        also load the edge through the sheet. ``(s, s, s)`` reproduces the
        single-array result bit for bit.
    held_edges : tuple of ``(axis, i, j, k)``, default ``()``
        E edges in the box's update window that KEEP the coefficients the
        run's own ``materials`` give them: a port's edges, whose load and
        drive were built from those materials before the time loop. ``axis``
        is the edge's component (0, 1, 2 for Ex, Ey, Ez), ``(i, j, k)`` its
        cell. The design values do not reach a held edge — its derivative
        with respect to them is exactly zero — so the port keeps the DRAWN
        material in its own feed gap: the modelled structure differs from
        the same design written as a whole-grid ``eps_override`` by exactly
        that gap. At any cell carrying a lumped load a cell design value is
        the cell's VOLUME material, the load staying on its own edge.
        Listing an edge is the opt-in: a port or source cell in the box that
        is not on a listed edge is refused, and so is a per-edge design
        conductivity over an unlisted load.
        ``Simulation.forward(design_box_holds_ports=True)`` lists every
        edge of every lumped and wire port in the window, passive ports
        included.
    """
    bounds: tuple
    eps_r: Any
    sigma: Any = None
    held_edges: tuple = ()


class DesignOccupancySpec(NamedTuple):
    """One static box whose PEC occupancy is the design variable (#1183).

    The relaxed-conductor rule scales every E component by ``1 - M``, ``M``
    the noisy-OR of the component's incident cells' occupancy. With a traced
    WHOLE-GRID occupancy that multiply keeps three grid-sized primal E arrays
    per timestep; with the traced values confined to a box the arrays it
    keeps are window-sized. Same physics — the window is recomputed from the
    field before the grid-wide scaling, so it reproduces exactly what
    ``apply_pec_occupancy`` would have written with the design occupancy in
    the array.

    bounds : (i0, i1, j0, j1, k0, k1)
        Half-open cell index bounds of the box.
    occupancy : box-shaped array (usually a tracer)
        Occupancy in [0, 1] at the box cells; REPLACES the run's own static
        occupancy there. Outside the box the static array is carried.
    """
    bounds: tuple
    occupancy: Any


class SimResult(NamedTuple):
    """Compiled simulation output.

    time_series : (n_steps, n_probes) float array, or (n_steps, 0) if
        no probes were specified.
    ntff_data : NTFFData or None
        Accumulated near-to-far-field DFT data (if NTFF box was used).
    dft_planes : tuple[DFTPlaneProbe, ...] or None
        Final accumulated DFT plane probes.
    waveguide_ports : tuple[WaveguidePortConfig, ...] or None
        Final accumulated waveguide-port configs.
    wire_port_sparams : tuple | None
        Final V/I/V_inc DFT accumulators for wire port S-params.
    lumped_port_sparams : tuple | None
        Final V/I DFT accumulators for lumped port S-params (issue #72).
        Each entry is ``(LumpedPortSParamSpec, (v_dft, i_dft, v_ref_dft))``;
        v/i are POST-injection (i with the Yee half-step phase) and
        v_ref is the PRE-injection drive sample.
    wire_refplane_sparams : tuple | None
        Final reference-plane V/I DFT accumulators for the opt-in wire
        S-matrix plane path (issue #313).  Each entry is
        ``(WireRefPlaneSpec, (v_dft, i_minus_dft, i_plus_dft))``; two
        entries per opted port (plane slots 0 and 1).
    snapshots : dict[str, ndarray] or None
        Field snapshots keyed by component name, each
        ``(n_steps // interval, ...)``; see ``SnapshotSpec``.
    ntff_box : NTFFBox or None
        NTFF box specification used for accumulation.
    grid : Grid or None
        Grid metadata needed by post-processing / objective helpers.
    snapshot_axes : dict[str, rfx.snapshots.SnapshotAxes] or None
        Per snapshot component: sample coordinates in metres, the slice
        plane, and the step count and time of each frame (#1259). ``None``
        when no snapshot was recorded.
    dt : float or None
        The time step the scan actually advanced by, in seconds. It is
        ``grid.dt`` except under ``stencil_order=4``, whose step is derated
        by ``_ORDER4_CFL_FACTOR``; the time series and every frame are
        sampled at this step, so time and frequency must be read with it,
        not with ``grid.dt``.
    current_moment_data : (accumulator, kahan) or None
        Block current moments accumulated inside the time loop
        (``rfx.current_moments``). The accumulator is
        ``(n_freqs, n_blocks, 3, n_weights)``.
    current_moment_monitor : CurrentMomentMonitor or None
        The slab, block map and centres those moments belong to.
    """
    state: FDTDState | None
    time_series: jnp.ndarray
    ntff_data: object = None
    dft_planes: tuple | None = None
    flux_monitors: tuple | None = None
    waveguide_ports: tuple | None = None
    wire_port_sparams: tuple | None = None
    lumped_port_sparams: tuple | None = None
    snapshots: dict | None = None
    ntff_box: object = None
    grid: Grid | None = None
    wire_refplane_sparams: tuple | None = None
    snapshot_axes: dict | None = None
    dt: float | None = None
    current_moment_data: object = None
    current_moment_monitor: object = None
    adjoint_settling: object = None
    sparam_time_records: tuple | None = None
    dft_time_records: tuple | None = None
    decay_stop: object = None   # run_until_decay: the until_decay.stop Diagnostic


class SnapshotSpec(NamedTuple):
    """Mid-simulation field snapshot specification.

    interval : int
        Record a frame every *interval* time steps; an integer >= 1, anything
        else is refused. Counting is 1-based over completed steps: a run of
        ``n_steps`` records ``n_steps // interval`` frames, frame ``k``
        (0-based) holding the fields after step ``(k + 1) * interval``, i.e.
        after steps ``interval, 2*interval, ...``. The last
        ``n_steps % interval`` steps record no frame. ``interval=1`` records
        every step. Memory scales with the frame count.
    components : tuple of str
        Field components to capture, e.g. ("ez",) or ("ex", "hy").
    slice_axis : int or None
        0, 1 or 2. If not None and *slice_index* is not None, capture a 2-D
        slice at *slice_index* along this axis instead of the full 3-D field
        (saves memory). With *slice_index* None the full field is recorded.
    slice_index : int or None
        PADDED lattice index along *slice_axis*: the same index space as
        ``grid.shape`` and ``grid.position_to_index``, CPML cells counted
        (``grid.position_to_index(pos)[axis]`` gives the index of a physical
        position).

    Where each recorded sample sits (metres) and when each frame was taken
    (seconds): ``Result.snapshot_axes`` after a run, or
    ``rfx.snapshot_axes(grid, spec, n_steps)`` before one (``rfx/snapshots.py``
    states the lattice layout and the E/H time convention).
    """
    interval: int = 10
    components: tuple = ("ez",)
    slice_axis: int | None = None
    slice_index: int | None = None


def _nearest_divisor(n: int, target: int) -> int:
    """Return the divisor of `n` closest to `target` (≥1).

    Used by the segmented-checkpointing path (issue #73) to suggest a
    valid `checkpoint_segments` value when the user passes one that does
    not divide `n_steps` evenly.
    """
    target = max(1, int(target))
    best = 1
    best_diff = abs(target - 1)
    # Search divisors in [1, sqrt(n)] and their cofactors.
    k = 1
    while k * k <= n:
        if n % k == 0:
            for d in (k, n // k):
                diff = abs(d - target)
                if diff < best_diff:
                    best, best_diff = d, diff
        k += 1
    return best


def resolve_periodic(grid, periodic):
    """The run's per-axis periodic flags, after the 2-D lane override (#689).

    One spelling, shared by :func:`_build_step_setup` and by any lane that
    must realize the PEC edge masks (§1.7) before calling ``run`` — those
    masks are only correct under the flags the step function will use.
    """
    if periodic is None:
        periodic = (False, False, False)
    else:
        if len(periodic) != 3:
            raise ValueError(f"periodic must have length 3, got {periodic!r}")
        periodic = tuple(bool(v) for v in periodic)
    if getattr(grid, "is_2d", False):
        periodic = (periodic[0], periodic[1], True)
    return periodic


@dataclass(frozen=True)
class _StepContext:
    """Static + array context for the shared Yee step kernel (W6.1).

    Holds every free variable the per-step kernel needs: physical constants,
    the resolved ``use_*`` subsystem flags, pre-extracted metadata tuples, and
    the boundary/dispersion helper callables that each caller imports during
    setup.  Lazily-imported helpers (waveguide-port, PMC, conformal, pec-mask,
    NTFF accumulate, RLC update) are re-imported inside the kernel body exactly
    as the original closures did and are therefore NOT stored here.
    """

    # ---- physical constants / grid ----
    grid: Any
    materials: Any
    dt: Any
    dx: Any
    periodic: tuple
    pec_axes: str
    # (2,4) fourth-order-in-space stencil order: 2 (default, byte-identical)
    # or 4. Threaded into the H / E kernel calls in core_step.
    stencil_order: int

    # ---- subsystem flags ----
    use_fast_he: bool
    use_upml: bool
    use_cpml: bool
    use_tfsf: bool
    use_debye: bool
    use_lorentz: bool
    use_ntff: bool
    use_dft_planes: bool
    use_flux_monitors: bool
    use_waveguide_ports: bool
    use_pec_faces: bool
    use_pmc_faces: bool
    use_aniso_inv: bool
    aniso_inv_eps_smooth: bool
    use_pec_edges: bool
    use_pec_occupancy: bool
    use_conformal: bool
    use_wire_sparams: bool
    use_lumped_sparams: bool
    use_lumped_rlc: bool
    use_kerr: bool

    # ---- output behaviour ----
    use_snapshot: bool
    use_monitor: bool
    use_flux_window: bool

    # ---- arrays / coeffs / configs ----
    fast_coeffs: Any = None
    cpml_params: Any = None
    cpml_axes: str = ""
    upml_coeffs: Any = None
    tfsf_cfg: Any = None
    tfsf_is_2d: bool = False
    bloch: Any = None  # per-axis Bloch phase for oblique-periodic TFSF (#404); None = real path
    debye_coeffs: Any = None
    lorentz_coeffs: Any = None
    pec_edge_masks: Any = None
    pec_static_edge_masks: Any = None
    kerr_chi3: Any = None
    ntff: Any = None
    # In-loop block current moments (rfx.current_moments). Default OFF so
    # every existing caller's context is unchanged.
    use_current_moments: bool = False
    current_moments: Any = None
    curl_boundary: Any = None
    pec_faces_frozen: Any = frozenset()
    pmc_faces_frozen: Any = frozenset()

    # ---- pre-extracted metadata ----
    # Sole source authority: None suppresses BOTH stages, including when xs
    # still contains primal samples (the reciprocity adjoint pass).
    drives: StepDrives | None = None
    electric_injector: Callable | None = None
    prb_meta: tuple = ()
    dft_meta: tuple = ()
    flux_meta: tuple = ()
    waveguide_meta: tuple = ()
    port_record_specs: tuple | None = None
    wire_sparam_meta: tuple = ()
    lumped_sparam_meta: tuple = ()
    rlc_meta: tuple = ()
    # issue #313 opt-in reference-plane channel (defaults keep every
    # existing caller byte-identical)
    use_wire_refplanes: bool = False
    wire_refplane_meta: tuple = ()
    # issue #677 node-thin surface-impedance sheet operator (defaults keep
    # every existing caller byte-identical)
    use_sheet_impedance: bool = False
    sheet_impedance: Any = None
    # issue #1179 design-box permittivity (defaults keep every existing
    # caller byte-identical)
    use_design_box: bool = False
    design_box: Any = None
    # issue #1183 design-box PEC occupancy (same default discipline)
    use_design_occupancy: bool = False
    design_occupancy: Any = None

    # ---- output extractors ----
    monitor_component: str = "ez"
    mon_idx: Any = None
    snapshot_extractor: Callable | None = None

    # ---- caller-imported helper callables ----
    apply_cpml_h: Callable | None = None
    apply_cpml_e: Callable | None = None
    apply_upml_h: Callable | None = None
    apply_upml_e: Callable | None = None
    apply_tfsf_h: Callable | None = None
    apply_tfsf_e: Callable | None = None
    update_tfsf_1d_h: Callable | None = None
    update_tfsf_1d_e: Callable | None = None
    update_tfsf_2d_h: Callable | None = None
    update_tfsf_2d_e: Callable | None = None
    init_ntff_data: Callable | None = None
    accumulate_ntff: Callable | None = None
    accumulate_current_moments: Callable | None = None
    apply_kerr_ade: Callable | None = None
    update_rlc_elements: Callable | None = None

    @property
    def src_meta(self):
        """Read-only aperture view for legacy diagnostics, never a kernel input."""
        if self.drives is None:
            return ()
        return tuple((i, j, k, component)
                     for component, nodes in self.drives.electric.nodes.items()
                     for i, j, k in zip(*nodes))
