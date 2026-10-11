"""Register time observations independently of requests for port spectra."""
from rfx import _grid_metric


def ordered_specs(specs):
    return tuple(s for kind in ('wire', 'lumped') for s in specs if s[2] == kind)


def register_wire(records, spectra, component, cells, live, freqs, impedance, excite):
    from rfx.simulation import WirePortSParamSpec
    if not cells:
        return
    live_cells = tuple(tuple(int(x) for x in c) for c, yes in zip(cells, live) if yes)
    mid = live_cells[len(live_cells) // 2]
    records.append((component, live_cells, 'wire', mid))
    if freqs is not None:
        spectra.append(WirePortSParamSpec(*mid, component, freqs,
                                          float(impedance), live_cells, bool(excite)))


def register_lumped(records, spectra, component, idx, freqs, impedance, excite):
    from rfx.simulation import LumpedPortSParamSpec
    mid = tuple(int(x) for x in idx)
    records.append((component, (mid,), 'lumped', mid))
    if freqs is not None:
        spectra.append(LumpedPortSParamSpec(*mid, component, freqs, float(impedance), bool(excite)))


def uniform_run_specs(grid, wires, live_cells, lumped):
    specs = [(p.component, live, 'wire', live[len(live)//2])
             for p, live in zip(wires, live_cells)]
    for p in lumped:
        mid = tuple(_grid_metric.field_index(grid, p.position, p.component))
        specs.append((p.component, (mid,), 'lumped', mid))
    return tuple(specs)


def graded_specs(ports):
    return tuple((p['component'], p.get('live_cells', ((p['mid_i'], p['mid_j'], p['mid_k']),)), p.get('kind', 'wire'),
                  (p['mid_i'], p['mid_j'], p['mid_k'])) for p in ports)


def graded_s_ports(ports):
    return [p for p in ports if not p.get('record_only', False)]


def graded_accumulators(ports, grid, freqs, carry):
    import jax.numpy as jnp
    from rfx.nonuniform import _build_wp_meta
    ports = graded_s_ports(ports)
    if not ports or freqs is None:
        return None, [], False
    bins = jnp.asarray(freqs, dtype=jnp.float32)
    carry['wire_sparams'] = tuple(tuple(jnp.zeros(len(bins), dtype=jnp.complex64)
                                      for _ in range(4)) for _ in ports)
    return bins, _build_wp_meta(ports, grid), True


def graded_acc_indices(ports):
    return tuple(i for i, p in enumerate(ports) if not p.get("record_only", False))


def graded_s_voltage(accumulators, ports, index):
    slot = 0 if ports[index].get('kind', 'wire') == 'lumped' else 3
    return accumulators[index][slot]
