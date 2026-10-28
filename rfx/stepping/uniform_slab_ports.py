"""Uniform main-record ports: owner-local samples and in-loop spectra."""
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.core.yee import FDTDState
from rfx.measurement.port_records import pack_port_records
from rfx.measurement.port_sampler import (
    pre_injection_samples, ref_dft_update, wire_dft_update, lumped_dft_update,
    stack_samples,
)
from rfx.measurement.projector import ComponentTerms, SampleTable, port_table, port_owners, project
from rfx.measurement.time_records import from_owners
from rfx.stepping.rank import rank_shard_map
from rfx.stepping.slab import place
from rfx.stepping.slab_measurements import cut_probe_table, sum_partials


_STATE_SPEC = FDTDState(*(P('x'),) * 6, step=P())


class UniformSlabPorts:
    """Small P('x') payloads; fields are sampled only on their owning slab."""

    def __init__(self, grid, entries, specs, boundary, layout, mesh, n_steps,
                 dtype, freqs, *, record_vref=True):
        from rfx._uniform_specs import WirePortSParamSpec, LumpedPortSParamSpec
        self.specs = tuple(specs)
        self.mesh, self.dt = mesh, grid.dt
        self.record_vref = record_vref
        self.owners = port_owners(grid, specs, nu=False, boundary=boundary)
        self.table = port_table(grid, specs, nu=False, boundary=boundary, dtype=dtype)
        self.payload = cut_probe_table(self.table, layout, mesh) if specs else ()
        self.metadata = []
        if freqs is not None:
            bins = jnp.asarray(freqs, dtype=jnp.float32)
            for pe, spec in zip(entries, specs, strict=True):
                _, cells, kind, mid, _ = spec
                self.metadata.append(
                    WirePortSParamSpec(*mid, pe.component, bins, pe.impedance, cells, pe.excite)
                    if kind == 'wire' else
                    LumpedPortSParamSpec(*mid, pe.component, bins, pe.impedance, pe.excite))
        initial = {}
        if specs:
            initial['sparam_time_records'] = tuple(jnp.zeros(
                (n_steps, 4 if s[2] == 'wire' else 3), dtype=dtype) for s in specs)
        for kind, count in (('wire', 5), ('lumped', 3)):
            metas = [m for m, s in zip(self.metadata, specs) if s[2] == kind]
            if metas:
                initial[kind + '_sparam_accs'] = tuple(tuple(jnp.zeros(
                    len(m.freqs), dtype=jnp.result_type(jnp.complex64, jnp.float64))
                    for _ in range(count)) for m in metas)
        self.initial = jax.tree.map(lambda a: place(
            jnp.broadcast_to(a, (layout.n_devices,) + a.shape), NamedSharding(mesh, P('x'))), initial)

    def local_table(self, payload):
        parts, order, ids = jax.tree.map(lambda a: a[0], payload)
        return SampleTable(tuple(ComponentTerms(t.component, *p)
                           for t, p in zip(self.table.components, parts)),
                           self.table.n_channels, self.table.channel_names, order, ids)

    def references(self, state, payload, *, ranks):
        if not self.specs:
            return ()
        def local(st, data, *, rank):
            refs = pre_injection_samples(st, self.specs, None, table=self.local_table(data))
            return tuple(a[None] for a in refs)
        return rank_shard_map(local, mesh=self.mesh, in_specs=(_STATE_SPEC, P('x')),
                              out_specs=P('x'))(ranks, state, payload)

    def update(self, carry, state, refs, step, payload, *, ranks):
        if not self.specs:
            return carry
        def local(values, st, references, index, data, *, rank):
            values, references = jax.tree.map(lambda a: a[0], (values, references))
            physical = project(st, self.local_table(data))
            recorded_refs = references if self.record_vref else tuple(jnp.zeros_like(a) for a in references)
            samples = stack_samples(physical, self.specs, recorded_refs)
            result = {'sparam_time_records': tuple(record.at[index].set(jax.lax.stop_gradient(sample))
                      for record, sample in zip(values['sparam_time_records'], samples, strict=True))}
            offsets = dict(wire=0, lumped=0)
            channel = 0
            for meta, spec, ref in zip(self.metadata, self.specs, references):
                kind = spec[2]
                key = kind + '_sparam_accs'
                acc = values[key][offsets[kind]]
                reference, phase = ref_dft_update(acc[-1], ref, index, meta.freqs, self.dt)
                width = 3 if kind == 'wire' else 2
                update = wire_dft_update if kind == 'wire' else lumped_dft_update
                result.setdefault(key, []).append((*update(
                    acc, physical[channel:channel + width], index, meta.freqs, self.dt, phase), reference))
                offsets[kind] += 1
                channel += width
            result = {k: tuple(v) for k, v in result.items()}
            return jax.tree.map(lambda a: a[None], result)
        return rank_shard_map(local, mesh=self.mesh,
                              in_specs=(P('x'), _STATE_SPEC, P('x'), P(), P('x')),
                              out_specs=P('x'))(ranks, carry, state, refs, step, payload)

    def reduce(self, final, *, ranks):
        return sum_partials(final, self.mesh, ranks)

    def assemble(self, final, *, multi_process=False):
        if multi_process and final:
            from jax.experimental import multihost_utils
            final = multihost_utils.process_allgather(final, tiled=True)
        accs = final.get('wire_sparam_accs', ()) + final.get('lumped_sparam_accs', ())
        wire, lumped = pack_port_records(
            [s[2] for s in self.specs] if self.metadata else (), accs, self.metadata)
        return dict(wire_port_sparams=wire, lumped_port_sparams=lumped,
                    sparam_time_records=from_owners(final.get('sparam_time_records', ()), self.owners))
