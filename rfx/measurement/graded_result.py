"""Result assembly for graded-grid solves and ringdown records."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from rfx.measurement.accumulators import plane_result, flux_result
from rfx.measurement.time_records import with_graded_records

if TYPE_CHECKING:
    from rfx.stepping.graded_setup import _NUScanSetup


def assemble_graded_result(setup: _NUScanSetup, final: dict, time_series, n_record=None) -> dict:
    """Assemble the NU result dict (pure code motion from run_nonuniform).
    Shared by :func:`run_nonuniform` and
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
        "dft_time_records": tuple(r[:n_record] for r in final.get("dft_time_records", ()) if r is not None),
        "state": final["fdtd"],
        "time_series": time_series,
        "dt": dt,
    }

    if use_lumped_rlc:
        result["rlc_states"] = tuple(final["rlc_states"])

    if use_dft_planes:
        result["dft_planes"] = tuple(
            plane_result(probe, acc[0])
            for probe, acc in zip(dft_planes, final["dft_planes"])
        )

    if use_flux_monitors:
        result["flux_monitors"] = tuple(
            flux_result(mon, (e1[0], e2[0], h1[0], h2[0]))
            for mon, (e1, e2, h1, h2) in zip(
                flux_monitors, final["flux_monitors"]
            )
        )

    if use_ntff:
        result["ntff_data"] = final["ntff"]

    if getattr(setup, "use_current_moments", False):
        result["current_moment_data"] = final["current_moments"]
        result["current_moment_monitor"] = setup.current_moments

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
