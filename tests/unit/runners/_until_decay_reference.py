"""Frozen uniform decay loop from origin/main 4725b748, before #1322.

Captured with ast.parse/ast.unparse (docstrings/comments removed only).
The setup and step kernel are shared; the driver and stop checks below
are the original implementation, independent of the chunked driver.
Retained DFT and S-parameter records follow the current runner interface.
"""

from __future__ import annotations

from rfx.simulation import (
    Grid,
    MaterialArrays,
    SourceSpec,
    ProbeSpec,
    SnapshotSpec,
    SimResult,
    _build_step_setup,
    _StepContext,
    make_core_step,
    validate_snapshot_spec,
    snapshot_extractor,
    snapshot_axes,
    validate_report_every,
    check_not_traced,
    ProgressReporter,
    _warn_static_remnant_cap_hit,
)
import jax
import jax.numpy as jnp


def run_until_decay_reference(
    grid: Grid,
    materials: MaterialArrays,
    *,
    decay_by: float = 0.001,
    check_interval: int = 50,
    min_steps: int = 100,
    max_steps: int = 50000,
    decay_energy_consecutive: int = 2,
    radiated_flux_box: tuple | None = None,
    flux_env_checks: int = 4,
    monitor_component: str = "ez",
    monitor_position: tuple[float, float, float] | None = None,
    boundary: str = "pec",
    cpml_axes: str = "xyz",
    pec_axes: str | None = None,
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    tfsf: tuple | None = None,
    sources: list[SourceSpec] | None = None,
    probes: list[ProbeSpec] | None = None,
    dft_planes: list | None = None,
    flux_monitors: list | None = None,
    waveguide_ports: list | None = None,
    ntff: object | None = None,
    current_moments: object | None = None,
    snapshot: SnapshotSpec | None = None,
    checkpoint: bool = False,
    aniso_eps: tuple | None = None,
    aniso_inv_eps: tuple | None = None,
    aniso_inv_eps_smooth: bool = False,
    pec_mask: object | None = None,
    pec_sheets: object = (),
    pec_wires: object = (),
    pec_edge_masks: object | None = None,
    pec_occupancy: object | None = None,
    conformal_weights: tuple | None = None,
    wire_port_sparams: list | None = None,
    lumped_port_sparams: list | None = None,
    lumped_rlc: list | None = None,
    kerr_chi3: jnp.ndarray | None = None,
    field_dtype=None,
    return_state: bool = True,
    mag_sources: list | None = None,
    checkpoint_segments: int | None = None,
    stencil_order: int = 2,
    report_every: int | None = None,
    report_label: str = "",
    sheet_impedance: object | None = None,
    record_dft: bool = False,
) -> SimResult:
    if checkpoint_segments is not None:
        raise NotImplementedError(
            "checkpoint_segments is not supported by run_until_decay: this function uses a Python loop, not jax.lax.scan, so scan-level gradient checkpointing does not apply. Use run() with checkpoint_segments if you need scan-level checkpointing."
        )
    sources = sources or []
    probes = probes or []
    dft_planes = dft_planes or []
    flux_monitors = flux_monitors or []
    waveguide_ports = waveguide_ports or []
    wire_port_sparams = wire_port_sparams or []
    lumped_port_sparams = lumped_port_sparams or []
    lumped_rlc = lumped_rlc or []
    mag_sources = mag_sources or []
    _setup = _build_step_setup(
        grid=grid,
        materials=materials,
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
        debye=debye,
        lorentz=lorentz,
        tfsf=tfsf,
        sources=sources,
        probes=probes,
        dft_planes=dft_planes,
        flux_monitors=flux_monitors,
        waveguide_ports=waveguide_ports,
        ntff=ntff,
        current_moments=current_moments,
        aniso_eps=aniso_eps,
        aniso_inv_eps=aniso_inv_eps,
        aniso_inv_eps_smooth=aniso_inv_eps_smooth,
        pec_mask=pec_mask,
        pec_sheets=pec_sheets,
        pec_wires=pec_wires,
        pec_edge_masks=pec_edge_masks,
        pec_occupancy=pec_occupancy,
        conformal_weights=conformal_weights,
        wire_port_sparams=wire_port_sparams,
        lumped_port_sparams=lumped_port_sparams,
        lumped_rlc=lumped_rlc,
        kerr_chi3=kerr_chi3,
        field_dtype=field_dtype,
        mag_sources=mag_sources,
        stencil_order=stencil_order,
        sheet_impedance=sheet_impedance,
    )
    carry = _setup.carry_init
    if record_dft and dft_planes:
        carry["dft_time_records"] = tuple(
            jnp.zeros((max_steps,) + plane.accumulator.shape[1:], dtype=carry["fdtd"].ex.dtype)
            for plane in dft_planes)
    if wire_port_sparams or lumped_port_sparams:
        carry["sparam_time_records"] = tuple(
            jnp.zeros((max_steps, count), dtype=field_dtype)
            for count in ([4] * len(wire_port_sparams) + [3] * len(lumped_port_sparams)))
    dx = _setup.dx
    waveguide_meta = _setup.waveguide_meta
    wire_sparam_meta = _setup.wire_sparam_meta
    lumped_sparam_meta = _setup.lumped_sparam_meta
    flux_meta_decay = _setup.flux_meta_8
    use_flux_monitors = len(flux_monitors) > 0
    use_wire_sparams = len(wire_port_sparams) > 0
    use_lumped_sparams = len(lumped_port_sparams) > 0
    use_dft_planes = len(dft_planes) > 0
    use_waveguide_ports = len(waveguide_ports) > 0
    if monitor_position is None:
        cx = (grid.nx - 1) * dx / 2.0
        cy = (grid.ny - 1) * dx / 2.0
        cz = 0.0 if grid.is_2d else (grid.nz - 1) * dx / 2.0
        monitor_position = (cx, cy, cz)
    mon_idx = grid.position_to_index(monitor_position)
    _step_ctx = _StepContext(
        **_setup.ctx_kwargs,
        use_fast_he=False,
        use_snapshot=False,
        use_monitor=True,
        use_flux_window=False,
        fast_coeffs=None,
        flux_meta=flux_meta_decay if use_flux_monitors else (),
        monitor_component=monitor_component,
        mon_idx=mon_idx,
        snapshot_extractor=None,
    )
    _core_step = make_core_step(_step_ctx)

    @jax.jit
    def _single_step(carry_in, step_idx, src_vals, mag_src_vals):
        new_carry, probe_out, extras = _core_step(carry_in, step_idx, src_vals, mag_src_vals)
        return (new_carry, probe_out, extras["monitor_val"])

    if snapshot is not None:
        snap_interval = validate_snapshot_spec(snapshot)
        _take_snapshot = snapshot_extractor(snapshot)
        snap_frames: list = []
    if sources:
        src_waveforms = jnp.stack(
            [
                s.waveform[:max_steps]
                if s.waveform.shape[0] >= max_steps
                else jnp.pad(s.waveform, (0, max_steps - s.waveform.shape[0]))
                for s in sources
            ],
            axis=-1,
        )
    else:
        src_waveforms = jnp.zeros((max_steps, 0), dtype=jnp.float32)
    if mag_sources:
        mag_src_waveforms = jnp.stack(
            [
                s.waveform[:max_steps]
                if s.waveform.shape[0] >= max_steps
                else jnp.pad(s.waveform, (0, max_steps - s.waveform.shape[0]))
                for s in mag_sources
            ],
            axis=-1,
        )
    else:
        mag_src_waveforms = jnp.zeros((max_steps, 0), dtype=jnp.float32)
    use_absorbing = boundary in ("cpml", "upml")
    _ix0, _ix1 = (grid.pad_x_lo, grid.nx - grid.pad_x_hi)
    _iy0, _iy1 = (grid.pad_y_lo, grid.ny - grid.pad_y_hi)
    if grid.is_2d:
        _iz0, _iz1 = (0, grid.nz)
    else:
        _iz0, _iz1 = (grid.pad_z_lo, grid.nz - grid.pad_z_hi)

    def _interior_energy(state) -> float:
        sx, sy, sz = (slice(_ix0, _ix1), slice(_iy0, _iy1), slice(_iz0, _iz1))
        if grid.is_2d:
            u = state.ez[sx, sy, sz] ** 2 + state.hx[sx, sy, sz] ** 2 + state.hy[sx, sy, sz] ** 2
        else:
            u = (
                state.ex[sx, sy, sz] ** 2
                + state.ey[sx, sy, sz] ** 2
                + state.ez[sx, sy, sz] ** 2
                + state.hx[sx, sy, sz] ** 2
                + state.hy[sx, sy, sz] ** 2
                + state.hz[sx, sy, sz] ** 2
            )
        return float(jnp.sum(u))

    use_flux_stop = radiated_flux_box is not None and use_absorbing
    if use_flux_stop:
        _flo = grid.position_to_index(radiated_flux_box[0])
        _fhi = grid.position_to_index(radiated_flux_box[1])
        _bl = (
            min(_flo[0], _fhi[0]),
            max(_flo[0], _fhi[0]),
            min(_flo[1], _fhi[1]),
            max(_flo[1], _fhi[1]),
            min(_flo[2], _fhi[2]),
            max(_flo[2], _fhi[2]),
        )

    def _radiated_power(state) -> float:
        ex, ey, ez = (state.ex, state.ey, state.ez)
        hx, hy, hz = (state.hx, state.hy, state.hz)
        il, ih, jl, jh, kl, kh = _bl
        jj, kk = (slice(jl, jh), slice(kl, kh))
        ii = slice(il, ih)
        p = jnp.sum(ey[ih, jj, kk] * hz[ih, jj, kk] - ez[ih, jj, kk] * hy[ih, jj, kk])
        p -= jnp.sum(ey[il, jj, kk] * hz[il, jj, kk] - ez[il, jj, kk] * hy[il, jj, kk])
        p += jnp.sum(ez[ii, jh, kk] * hx[ii, jh, kk] - ex[ii, jh, kk] * hz[ii, jh, kk])
        p -= jnp.sum(ez[ii, jl, kk] * hx[ii, jl, kk] - ex[ii, jl, kk] * hz[ii, jl, kk])
        p += jnp.sum(ex[ii, jj, kh] * hy[ii, jj, kh] - ey[ii, jj, kh] * hx[ii, jj, kh])
        p -= jnp.sum(ex[ii, jj, kl] * hy[ii, jj, kl] - ey[ii, jj, kl] * hx[ii, jj, kl])
        return float(p)

    peak_sq = 0.0
    peak_U = 0.0
    energy_below = 0
    peak_flux = 0.0
    flux_below = 0
    flux_hist: list[float] = []
    decayed_fired = False
    all_probes = []
    actual_steps = 0
    _reporter = None
    if report_every is not None:
        _report_every = validate_report_every(report_every, n_steps=max_steps)
        check_not_traced(
            carry,
            materials,
            aniso_eps,
            aniso_inv_eps,
            pec_mask,
            pec_edge_masks,
            pec_occupancy,
            conformal_weights,
            kerr_chi3,
            debye,
            lorentz,
            tfsf,
        )
        _reporter = ProgressReporter(max_steps, label=report_label, total_is_cap=True)
    for step in range(max_steps):
        step_idx = jnp.array(step, dtype=jnp.int32)
        src_vals = src_waveforms[step]
        mag_src_vals = mag_src_waveforms[step]
        carry, probe_out, monitor_val = _single_step(carry, step_idx, src_vals, mag_src_vals)
        all_probes.append(probe_out)
        actual_steps = step + 1
        if snapshot is not None and actual_steps % snap_interval == 0:
            snap_frames.append(_take_snapshot(carry["fdtd"]))
        if _reporter is not None and actual_steps % _report_every == 0:
            jax.block_until_ready(carry["fdtd"])
            _reporter.report(actual_steps)
        if use_absorbing and use_flux_stop:
            if step % check_interval == 0:
                flux_hist.append(abs(_radiated_power(carry["fdtd"])))
                env = max(flux_hist[-flux_env_checks:])
                if env > peak_flux:
                    peak_flux = env
                if actual_steps >= min_steps and peak_flux > 0.0 and (env < decay_by * peak_flux):
                    flux_below += 1
                    if flux_below >= decay_energy_consecutive:
                        decayed_fired = True
                        break
                else:
                    flux_below = 0
        elif use_absorbing:
            if decay_by > 0.0 and step % check_interval == 0:
                U = _interior_energy(carry["fdtd"])
                if U > peak_U:
                    peak_U = U
            if decay_by > 0.0 and actual_steps >= min_steps and (step % check_interval == 0):
                if U < decay_by * peak_U:
                    energy_below += 1
                    if energy_below >= decay_energy_consecutive:
                        decayed_fired = True
                        break
                else:
                    energy_below = 0
        else:
            val_sq = float(monitor_val) ** 2
            if val_sq > peak_sq:
                peak_sq = val_sq
            if actual_steps >= min_steps and step % check_interval == 0 and (peak_sq > 0.0):
                if val_sq < decay_by * peak_sq:
                    break
    if _reporter is not None and _reporter.last_reported != actual_steps:
        jax.block_until_ready(carry["fdtd"])
        _reporter.report(actual_steps)
    if use_absorbing and (not decayed_fired) and (not use_flux_stop):
        _warn_static_remnant_cap_hit(carry["fdtd"], materials, grid)
    time_series = jnp.stack(all_probes, axis=0)
    final_dft_planes = None
    if use_dft_planes:
        final_dft_planes = tuple(
            (probe._replace(accumulator=acc) for probe, acc in zip(dft_planes, carry["dft_planes"]))
        )
    final_waveguide_ports = None
    if use_waveguide_ports:
        final_waveguide_ports = tuple(
            (
                cfg_meta._replace(
                    dt=float(_setup.dt),
                    v_probe_t=accs[0],
                    v_ref_t=accs[1],
                    i_probe_t=accs[2],
                    i_ref_t=accs[3],
                    v_inc_t=accs[4],
                    n_steps_recorded=accs[5],
                )
                for cfg_meta, accs in zip(waveguide_meta, carry["waveguide_port_accs"])
            )
        )
    final_wire_sparams = None
    if use_wire_sparams:
        final_wire_sparams = tuple(
            ((wp_meta, accs) for wp_meta, accs in zip(wire_sparam_meta, carry["wire_sparam_accs"]))
        )
    final_lumped_sparams = None
    if use_lumped_sparams:
        final_lumped_sparams = tuple(
            (
                (lp_meta, accs)
                for lp_meta, accs in zip(lumped_sparam_meta, carry["lumped_sparam_accs"])
            )
        )
    final_flux_monitors = None
    if use_flux_monitors:
        final_flux_monitors = tuple(
            (
                fm._replace(e1_dft=accs[0], e2_dft=accs[1], h1_dft=accs[2], h2_dft=accs[3])
                for fm, accs in zip(flux_monitors, carry["flux_monitors"])
            )
        )
    snapshots = snap_axes = None
    if snapshot is not None:
        if snap_frames:
            snapshots = {
                comp: jnp.stack([frame[i] for frame in snap_frames])
                for i, comp in enumerate(snapshot.components)
            }
        else:
            snapshots = {
                comp: jnp.zeros((0,) + s.shape, s.dtype)
                for comp, s in zip(
                    snapshot.components, jax.eval_shape(_take_snapshot, carry["fdtd"])
                )
            }
        snap_axes = snapshot_axes(grid, snapshot, actual_steps, dt=_setup.dt)
    return SimResult(
        state=carry["fdtd"] if return_state else None,
        time_series=time_series,
        ntff_data=carry.get("ntff"),
        dft_planes=final_dft_planes,
        flux_monitors=final_flux_monitors,
        waveguide_ports=final_waveguide_ports,
        wire_port_sparams=final_wire_sparams,
        lumped_port_sparams=final_lumped_sparams,
        snapshots=snapshots,
        ntff_box=ntff,
        grid=grid,
        snapshot_axes=snap_axes,
        dt=_setup.dt,
        current_moment_data=carry.get("current_moments"),
        current_moment_monitor=current_moments,
        dft_time_records=tuple(r[:actual_steps] for r in carry.get("dft_time_records", ())),
        sparam_time_records=tuple(r[:actual_steps] for r in carry.get("sparam_time_records", ())),
    )
