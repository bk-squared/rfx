"""Uniform coax voltage spectra, reference-plane reflection and record witnesses."""
import jax
import jax.numpy as jnp
import numpy as np


def port_owner(sim, grid, conductors, frequencies):
    from rfx.model.coax_line import coax_port_layout
    from .ports import coax_owner
    port = sim._coaxial_ports[0]
    release = getattr(conductors, 'coax_bore_release', None)
    layout = release.layout if release is not None else coax_port_layout(
        grid, port, absorber_pads=tuple(getattr(grid, f'pad_{a}_{s}')
            for a in 'xyz' for s in ('lo', 'hi')), ntff=sim._ntff)
    return coax_owner(grid, port_id=0, plane_indices=layout.plane_indices,
        center_xy=port.position[:2], pin_radius=port.pin_radius,
        outer_radius=port.outer_radius, frequencies=() if frequencies is None else frequencies,
        reference_index=layout.reference_index)


def init_coax_records(carry, plan, n_steps):
    owners = tuple(o for o in plan.owners if o.kind == 'coax' and o.channels) if plan else ()
    if not owners:
        return
    owner, = owners
    dtype = carry['fdtd'].ex.dtype
    carry['coax_sparam_accs'] = jnp.zeros((len(owner.channels), len(owner.frequencies)),
                                        dtype=jnp.result_type(dtype, jnp.complex64))
    carry['coax_time_records'] = jnp.zeros((n_steps, len(owner.channels)), dtype=dtype)


def update_coax(carry, state, step, dt, table, freqs):
    from .projector import project
    from rfx.core.dft_utils import port_dft_phase
    volts = project(state, table)
    acc = carry['coax_sparam_accs']
    phase = port_dft_phase(step, freqs, dt, dtype=acc.dtype)
    return dict(coax_sparam_accs=acc + volts[:, None] * phase[None, :],
                coax_time_records=jax.lax.stop_gradient(
                    carry['coax_time_records'].at[step].set(volts)))


def coax_records(carry, plan, n_steps=None):
    from .time_records import from_owners
    if 'coax_time_records' not in carry:
        return ()
    owners = tuple(o for o in plan.owners if o.kind == 'coax' and o.channels)
    return from_owners((carry['coax_time_records'],), owners, n_steps)


def coax_port_s11(voltages, *, owner, z_tem_ohm, impedance, traced=False):
    """Return declared-impedance S11 and native Gamma from in-scan plane DFTs."""
    from .coax_records import coaxial_line_reflection_from_plane_voltages
    from rfx.sparams._port_reference import renormalize_power_waves
    xp = jnp if traced else np
    planes = [p.coordinate_m for p in owner.reference_planes if p.name.startswith('probe:')]
    reference = next(p.coordinate_m for p in owner.reference_planes if p.name == 'deembedding')
    planes = np.abs(np.asarray(planes) - reference)
    reference = 0.0
    order = np.argsort(planes)
    voltage = xp.asarray(voltages)[order]
    native = xp.stack([coaxial_line_reflection_from_plane_voltages(
        np.asarray(planes)[order], voltage[:, i], reference_plane_m=reference,
        _prefer_jnp=traced).reflection for i in range(voltage.shape[1])])
    s11 = renormalize_power_waves(native[None, None, :], [z_tem_ohm], [impedance])
    return s11, native


def port_result(sim, product, result, *, traced=False):
    plan = product._prepared_solve.measurement
    owner, = (o for o in plan.owners if o.kind == 'coax')
    if not len(owner.frequencies):
        return None, None, None
    if result.coax_sparam_accs is None:
        raise ValueError('coax port has no in-scan voltage accumulators')
    z_tem = product.coax_bore_release.z_tem_ohm
    s11, native = coax_port_s11(result.coax_sparam_accs, owner=owner,
        z_tem_ohm=z_tem, impedance=sim._coaxial_ports[0].impedance, traced=traced)
    return s11, owner.frequencies, {owner.id: dict(native_gamma=native, z_tem_ohm=z_tem)}


def declared_drive(sim, grid, port):
    from rfx.model.coax_line import coax_port_layout
    from rfx.sources.sources import GaussianPulse
    layout = coax_port_layout(grid, port, absorber_pads=tuple(getattr(grid, f'pad_{a}_{s}')
        for a in 'xyz' for s in ('lo', 'hi')), ntff=sim._ntff)
    waveform = port.excitation or GaussianPulse(f0=sim._freq_max / 2, bandwidth=0.8)
    delta = .5 * grid.dt + .5 * grid.dx * np.sqrt(2.1) / 299792458
    e_plane = layout.source_index + (port.face == 'top')
    h_plane = layout.source_index - (port.face == 'bottom')
    return ((waveform, (*port.position[:2], grid.node_of(2, e_plane))),
            (lambda t: waveform(t + jnp.float32(delta)),
             (*port.position[:2], grid.node_of(2, h_plane))))


def record_witness(records, dt, source_end, freqs, freq_max):
    from .settling import settling_db_from_named_records
    channels = [(f'{r.id}/{name}', np.asarray(r.samples[:, i]))
                for r in records for i, name in enumerate(r.channels)]
    db, detail = settling_db_from_named_records(channels, dt=dt, freqs=freqs,
        source_end_index=source_end, freq_max=freq_max,
        scale_groups={name: 'coax/V' for name, _ in channels}, return_detail=True)
    return db, {**detail, 'scored_port_record_ids': [r.id for r in records]}
