"""Batched material-sweep time stepping."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from rfx.core.drives import StepDrives, drive_layout, inject_drives
from rfx.core.yee import MaterialArrays, init_state
from rfx.measurement.accumulators import plane_metadata
from rfx.probes.probes import DFTPlaneProbe
from rfx._uniform_specs import SourceSpec, ProbeSpec


def _build_vmap_scan_fn(
    grid,
    n_steps: int,
    *,
    use_cpml: bool = False,
    sources: list[SourceSpec] | None = None,
    j_source_meta: list[tuple] | None = None,
    probes: list[ProbeSpec] | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    cpml_axes: str = "xyz",
    pec_axes: str = "xyz",
    pec_mask=None,
    pec_sheets=(),
    pec_wires=(),
    dft_probes: list[DFTPlaneProbe] | None = None,
    conductors=None,
):
    """Build a pure function ``f(materials, drives) -> (time_series, dft_accs)``
    suitable for vmap.

    Composes the uniform FDTD step, including CPML, PEC and observations, without
    dispersion, TFSF, or waveguide ports.  For the common material-sweep use
    case (PEC cavity, CPML absorber, DFT-plane probe, or simple probe
    measurement) this covers the needed physics.

    Parameters
    ----------
    use_cpml : bool
        If True, the scan carry additionally holds the CPML psi-state and the
        CPML H/E sub-steps are applied. Source samples passed to the returned
        function already include their material-dependent normalization.
    j_source_meta : list of (i, j, k, component)
        Populated for legacy CPML J-sources AND for ``amplitude_kind='current'``
        soft sources on any boundary. Their waveforms are prepared with the
        scalar run() helpers for each material batch element, then passed to
        the returned function alongside materials.
    dft_probes : list of DFTPlaneProbe or None
        Zero-initialized DFT plane probes (from ``init_dft_plane_probe``,
        the SAME helper ``run()`` uses — rfx/runners/uniform.py:489-497).
        Their ``.accumulator`` seeds the scan-carry accumulator for that
        plane; ``.component``/``.axis``/``.index``/``.freqs`` drive the
        per-step kernel shared with ``Simulation.run()``. Scan slot n reads
        E at (n+1)*dt and H half a step earlier (#404).
        The returned ``run_one`` always returns a 2-tuple
        ``(time_series, dft_accs)`` where ``dft_accs`` is a tuple of final
        accumulators in the SAME order as ``dft_probes`` (empty tuple if
        no planes were registered).
    """
    dt = grid.dt
    dx = grid.dx
    sources = sources or []
    j_source_meta = j_source_meta or []
    probes = probes or []
    dft_probes = dft_probes or []

    # #931 §1.7: realize (Mx, My, Mz) ONCE here, at build time, under the
    # flags this scan body will use.  Sheets and wires own no cell, so
    # this is the only place the batched lane can see them.
    pec_sheets = tuple(pec_sheets or ())
    pec_wires = tuple(pec_wires or ())
    pec_edge_masks = None if conductors is None else conductors.pec_edges
    if conductors is None and (pec_mask is not None or pec_sheets or pec_wires):
        from rfx.boundaries.pec import realized_pec_edge_masks
        pec_edge_masks = realized_pec_edge_masks(
            pec_mask, sheets=pec_sheets, wires=pec_wires, periodic=periodic)
    use_pec_edges = pec_edge_masks is not None
    use_dft = len(dft_probes) > 0
    dft_meta = plane_metadata(dft_probes)
    dft_acc_init = tuple(p.accumulator for p in dft_probes)

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h

    # Precompute source waveform matrix (these don't depend on materials)
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    src_meta = [(s.i, s.j, s.k, s.component) for s in sources]
    prb_meta = [(p.i, p.j, p.k, p.component) for p in probes]

    # Material-dependent source locations; samples are prepared before vmap.
    j_src_meta = j_source_meta

    # Initialize CPML once (shared across batch)
    if use_cpml:
        cpml_params, cpml_state_init = init_cpml(grid)

    def run_one(materials: MaterialArrays, prepared_j_waveforms) -> tuple[jnp.ndarray, tuple]:
        """Run a single FDTD simulation with the given materials.

        Returns ``(time_series, dft_accs)``: time_series shaped
        ``(n_steps, n_probes)``; dft_accs a tuple of final ``(n_freqs, n1,
        n2)`` complex accumulators, one per registered DFT plane (empty
        tuple if none).
        """
        from rfx.model.materials import with_components
        materials = with_components(materials, grid, periodic=periodic)
        fdtd = init_state(grid.shape)
        drives = drive_layout(src_meta + list(j_src_meta), fdtd.ex.dtype)

        j_src_waveforms = prepared_j_waveforms

        from rfx._uniform_specs import _StepContext
        from rfx.stepping.uniform_setup import make_core_step
        ctx = _StepContext(
            grid=grid, materials=materials, dt=dt, dx=dx, periodic=periodic,
            pec_axes=pec_axes, stencil_order=2, use_fast_he=False,
            use_upml=False, use_cpml=use_cpml, use_tfsf=False,
            use_debye=False, use_lorentz=False, use_ntff=False,
            use_dft_planes=use_dft, use_flux_monitors=False,
            use_waveguide_ports=False, use_pec_faces=bool(pec_axes),
            use_pmc_faces=False, use_aniso_inv=False, aniso_inv_eps_smooth=False,
            use_pec_edges=use_pec_edges, use_pec_occupancy=False,
            use_conformal=False, use_wire_sparams=False, use_lumped_sparams=False,
            use_lumped_rlc=False, use_kerr=False, use_snapshot=False,
            use_monitor=False, use_flux_window=False,
            drives=StepDrives(electric=drives, magnetic=None), electric_injector=inject_drives,
            prb_meta=tuple(prb_meta), dft_meta=dft_meta,
            pec_faces_frozen=frozenset(f"{axis}_{side}" for axis in pec_axes
                                       for side in ("lo", "hi")),
            pec_edge_masks=pec_edge_masks,
            cpml_params=cpml_params if use_cpml else None, cpml_axes=cpml_axes,
            apply_cpml_h=apply_cpml_h if use_cpml else None,
            apply_cpml_e=apply_cpml_e if use_cpml else None,
        )
        core = make_core_step(ctx)

        def step_fn(carry, xs):
            step_idx, src_vals, j_src_vals = xs
            values = jnp.concatenate((src_vals.astype(fdtd.ex.dtype),
                                      j_src_vals.astype(fdtd.ex.dtype)))
            carry, probe_out, _ = core(carry, step_idx, values, jnp.zeros(0))
            return carry, probe_out

        xs = (
            jnp.arange(n_steps, dtype=jnp.int32),
            src_waveforms,
            j_src_waveforms,
        )
        init_carry = {"fdtd": fdtd}
        if use_cpml:
            init_carry["cpml"] = cpml_state_init
        if use_dft:
            init_carry["dft_planes"] = dft_acc_init
        final_carry, time_series = jax.lax.scan(step_fn, init_carry, xs)
        final_dft_accs = final_carry.get("dft_planes", ())
        return time_series, final_dft_accs

    return run_one
