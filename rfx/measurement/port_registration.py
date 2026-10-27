"""Register time observations independently of requests for port spectra."""
from rfx import _grid_metric


def ordered_specs(specs):
    return tuple(s for kind in ('wire', 'lumped') for s in specs if s[2] == kind)


def register_wire(records, spectra, component, cells, live, freqs, impedance, excite, port_id):
    from rfx._uniform_specs import WirePortSParamSpec
    if not cells:
        return
    live_cells = tuple(tuple(int(x) for x in c) for c, yes in zip(cells, live) if yes)
    mid = live_cells[len(live_cells) // 2]
    records.append((component, live_cells, 'wire', mid, port_id))
    if freqs is not None:
        spectra.append(WirePortSParamSpec(*mid, component, freqs,
                                          float(impedance), live_cells, bool(excite)))


def register_lumped(records, spectra, component, idx, freqs, impedance, excite, port_id):
    from rfx._uniform_specs import LumpedPortSParamSpec
    mid = tuple(int(x) for x in idx)
    records.append((component, (mid,), 'lumped', mid, port_id))
    if freqs is not None:
        spectra.append(LumpedPortSParamSpec(*mid, component, freqs, float(impedance), bool(excite)))


def uniform_run_specs(grid, wires, live_cells, lumped, ports):
    specs = [(p.component, live, 'wire', live[len(live)//2])
             for p, live in zip(wires, live_cells)]
    for p in lumped:
        mid = tuple(_grid_metric.field_index(grid, p.position, p.component))
        specs.append((p.component, (mid,), 'lumped', mid))
    ids = [i for wire in (True, False) for i, p in enumerate(ports)
           if p.impedance != 0.0 and (p.extent is not None) == wire]
    return tuple((*spec, port_id) for spec, port_id in zip(specs, ids, strict=True))


def graded_specs(ports):
    return tuple((p['component'], p.get('live_cells', ((p['mid_i'], p['mid_j'], p['mid_k']),)), p.get('kind', 'wire'),
                  (p['mid_i'], p['mid_j'], p['mid_k']), p.get('port_id', i)) for i, p in enumerate(ports))


def graded_s_ports(ports):
    return [p for p in ports if not p.get('record_only', False)]


def graded_accumulators(ports, grid, freqs, carry):
    import jax.numpy as jnp
    from rfx.measurement.graded_port_meta import _build_wp_meta
    ports = graded_s_ports(ports)
    if not ports or freqs is None:
        return None, [], False
    bins = jnp.asarray(freqs, dtype=jnp.float32)
    carry['wire_sparams'] = tuple(tuple(jnp.zeros(len(bins), dtype=jnp.complex64)
                                      for _ in range(5)) for _ in ports)
    return bins, _build_wp_meta(ports, grid), True


def graded_acc_indices(ports):
    return tuple(i for i, p in enumerate(ports) if not p.get("record_only", False))


def graded_s_voltage(accumulators, ports, index):
    slot = 0 if ports[index].get('kind', 'wire') == 'lumped' else 3
    return accumulators[index][slot]
