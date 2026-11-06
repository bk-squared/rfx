"""Recorded coax one-port drive and its two-port-compatible settling witness."""

import numpy as np


def run_reflection_record(grid, materials, n_steps, *, spec, conductors, cpml_axes,
                          planes, probes_z, center_xy, a, b, eps_scale, freqs, freq_max):
    """Return the unchanged runner result plus one settling value and witness."""
    from rfx.simulation import run as _run, ProbeSpec
    from rfx.sources.coaxial_port import coaxial_line_plane_voltage

    dz = float(grid.dx)
    # Match the two-port witness: an annular point at the ladder's middle plane.
    witness_probes = [ProbeSpec(
        i=int(round((center_xy[0] + 0.5 * (a + b)) / dz)) + int(grid.pad_x_lo),
        j=int(round(center_xy[1] / dz)) + int(grid.pad_y_lo),
        k=int(probes_z[len(probes_z) // 2]), component="ex",
    )]
    result = _run(
        grid, materials, int(n_steps), boundary="cpml", cpml_axes=cpml_axes,
        sources=list(spec.electric_sources), mag_sources=list(spec.magnetic_sources),
        probes=witness_probes, dft_planes=planes, record_dft=True,
        pec_edge_masks=conductors.pec_edges,
        return_state=False,
    )
    if result.dft_planes is None:
        raise RuntimeError("compute_coaxial_line_reflection(): runner returned no DFT planes")

    settling_db = float("nan")
    settling_witness = {"status": "undetermined", "reason": "traced or unavailable S channels"}
    if eps_scale is None:
        ts = np.asarray(result.time_series)
        if ts.ndim == 2 and ts.shape[0] >= 10 and ts.shape[1] == len(witness_probes):
            from rfx.measurement.record_end import injection_end
            from rfx.sources.waveguide_port import settling_db_from_named_records

            source_end = injection_end(
                len(ts), grid.dt, records=witness_probes, grid=grid,
                sources=spec.electric_sources, mag_sources=spec.magnetic_sources,
            )
            plane_records = getattr(result, "dft_time_records", None)
            channels = [
                (f"coax/plane{pi}/V", np.asarray(coaxial_line_plane_voltage(
                    grid, plane_records[2*pi], plane_records[2*pi+1],
                    center_xy=center_xy, pin_radius=a, outer_radius=b)))
                for pi in range(len(probes_z))
            ] if plane_records else []
            if channels:
                channels.extend((f"probe{i}", ts[:, i]) for i in range(ts.shape[1]))
            settling_db, settling_witness = settling_db_from_named_records(
                channels, source_end_index=source_end, dt=result.dt,
                freqs=freqs, freq_max=freq_max, return_detail=True,
            )
    return result, settling_db, settling_witness
