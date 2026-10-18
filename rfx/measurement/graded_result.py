"""Result assembly for graded-grid solves and ringdown records."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from rfx.measurement.time_records import with_graded_records

if TYPE_CHECKING:
    from rfx.stepping.graded_setup import _NUScanSetup


def assemble_graded_result(setup: _NUScanSetup, final: dict, time_series, n_record=None) -> dict:
    """Assemble the NU result dict (pure code motion from run_nonuniform,
    #383). Shared by :func:`run_nonuniform` and
    :func:`run_nonuniform_until_decay` so the two paths return the exact
    same schema (state, time_series, dt, conditional dft_planes /
    flux_monitors / ntff_data / waveguide_ports / s_params...)."""
    dt = setup.dt
    dft_planes = setup.dft_planes
    flux_monitors = setup.flux_monitors
    waveguide_meta = setup.waveguide_meta
    wp_meta = setup.wp_meta
    use_dft_planes = setup.use_dft_planes
    use_flux_monitors = setup.use_flux_monitors
    use_lumped_rlc = setup.use_lumped_rlc
    use_ntff = setup.use_ntff
    use_waveguide_ports = setup.use_waveguide_ports

    result = {
        "dft_time_records": tuple(r[:n_record] for r in final.get("dft_time_records", ())),
        "state": final["fdtd"],
        "time_series": time_series,
        "dt": dt,
    }

    # Surface final RLC ADE states (per element).
    if use_lumped_rlc:
        result["rlc_states"] = tuple(final["rlc_states"])

    # Repack DFT plane probes with their final accumulators.
    if use_dft_planes:
        result["dft_planes"] = tuple(
            probe._replace(accumulator=acc)
            for probe, acc in zip(dft_planes, final["dft_planes"])
        )

    # Repack flux monitors with their final E/H DFT accumulators so the
    # caller can call flux_spectrum() on them (same schema as uniform).
    if use_flux_monitors:
        result["flux_monitors"] = tuple(
            mon._replace(e1_dft=e1, e2_dft=e2, h1_dft=h1, h2_dft=h2)
            for mon, (e1, e2, h1, h2) in zip(
                flux_monitors, final["flux_monitors"]
            )
        )

    # Surface final NTFF DFT accumulators
    if use_ntff:
        result["ntff_data"] = final["ntff"]

    # Surface the in-loop block current moments and the slab they belong to.
    # getattr: run(ringdown=) assembles S through here with a stand-in setup
    # (rfx.ringdown._assemble) that carries no monitor.
    if getattr(setup, "use_current_moments", False):
        result["current_moment_data"] = final["current_moments"]
        result["current_moment_monitor"] = setup.current_moments

    # Surface final waveguide-port configs (with recorded modal V/I
    # time series; spectra are extracted post-scan via rect-DFT).
    if use_waveguide_ports:
        result["waveguide_ports"] = tuple(
            cfg_meta._replace(
                v_probe_t=accs[0],
                v_ref_t=accs[1],
                i_probe_t=accs[2],
                i_ref_t=accs[3],
                v_inc_t=accs[4],
                n_steps_recorded=accs[5],
            )
            for cfg_meta, accs in zip(
                waveguide_meta, final["waveguide_port_accs"]
            )
        )

    from rfx.measurement.port_s import assemble_graded_ports_result
    from types import SimpleNamespace
    request = SimpleNamespace(context=setup, port_owners=setup.step_fn_inv.port_owners,
                              port_acc_indices=setup.step_fn_inv.port_acc_indices)
    result.update(assemble_graded_ports_result(final.get('wire_sparams'), wp_meta,
        tuple(r[:n_record] for r in final.get('sparam_time_records', ())), request))
    return result
