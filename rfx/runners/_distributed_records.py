"""Owned records at the uniform and graded distributed sampling slots.

Ranks are data, and assembly replicates before indexing by rank (#1441).
Only placement is performed for cell records; moment pairs are summed once.
"""

from rfx.errors import RfxUnsupported
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.runners._rank import rank_shard_map
from rfx.runners._distributed_ntff import SlabNTFF
from rfx.stepping.slab import place
from rfx.measurement import accumulators


def refuse_record_options(sim, options):
    names = [name for name in ("record_dft", "dft_time_records", "dft_planes",
                               "flux_monitors", "current_moments")
             if options.get(name) is not None and options.get(name) is not False]
    if sim._dft_planes and getattr(sim, "_internal_probe_indices", ()):
        names.append("dft_time_records (record_dft from _internal_probe_indices)")
    if names:
        raise RfxUnsupported(f"{', '.join(names)} is not supported on the distributed record path")


def _field_specs(state):
    return state._replace(**{name: P("x") for name in
                             ("ex", "ey", "ez", "hx", "hy", "hz")}, step=P())


class Packed:
    """Largest-owner capacity, zero tail padding, and rank-local operations."""

    def __init__(self, mesh, shapes, dtype):
        self.mesh, self.shapes, self.dtype = mesh, shapes, dtype
        self.sizes = [[int(np.prod(s)) for s in part] for part in shapes]
        self.capacity = max(map(sum, self.sizes))

    def initial(self):
        return place(np.zeros((self.mesh.size, self.capacity), self.dtype),
                              NamedSharding(self.mesh, P("x")))

    def unpack(self, buffer, rank):
        arrays, offset = [], 0
        for shape, size in zip(self.shapes[rank], self.sizes[rank]):
            arrays.append(buffer[offset:offset + size].reshape(shape))
            offset += size
        return arrays

    def pack(self, arrays):
        flat = jnp.concatenate([a.ravel() for a in arrays]) if arrays else jnp.zeros(0, self.dtype)
        return jnp.pad(flat, (0, self.capacity - flat.size))

    def mapped(self, buffer, state, step, branches, *, ranks, previous=None):
        inputs = (buffer, state, step)
        specs = (P("x"), _field_specs(state), P())
        if previous is not None:
            inputs += (previous,)
            specs += (P("x"),)

        @partial(rank_shard_map, mesh=self.mesh, in_specs=specs,
                 out_specs=P("x"), check_rep=False)
        def apply(*args, rank):
            return lax.switch(rank, branches, args)[None]
        return apply(ranks, *inputs)

    def replicated(self, buffer):
        return (place(buffer, NamedSharding(self.mesh, P()))
                if isinstance(buffer, (jax.Array, jax.core.Tracer)) else buffer)


def _cut(axis, index, region, start, end, nx_per):
    """Local sampling region and destination rows of a plane component."""
    lo, hi, lo2, hi2 = region
    if axis == 0:
        if index // nx_per != start // nx_per:
            return None
        return index - (start - 1), region, slice(None)
    left, right = max(lo, start), min(hi, end)
    if right <= left:
        return None
    return index, (left - start + 1, right - start + 1, lo2, hi2), slice(left - lo, right - lo)


class PlaneRecords(Packed):
    def __init__(self, mesh, templates, shape, nx_per, *, flux=False, window_from_state=True):
        self.templates, self.flux = templates, flux
        self.window_from_state = window_from_state
        metadata = (accumulators.flux_metadata(templates, window=True) if flux
                    else accumulators.plane_metadata(templates))
        self.parts, shapes = [], []
        for rank in range(mesh.size):
            start, end = rank * nx_per, min((rank + 1) * nx_per, shape[0])
            part, local_shapes = [], []
            for number, (template, meta) in enumerate(zip(templates, metadata)):
                if flux:
                    axis, index, freqs, channels, regions, *rest = meta
                    arrays = template[:4]
                else:
                    channel, axis, index, freqs, region = meta
                    arrays = (template.accumulator,)
                    regions = (region or (0, arrays[0].shape[1], 0, arrays[0].shape[2]),)
                cuts = [_cut(axis, index, region, start, end, nx_per) for region in regions]
                if all(cut is None for cut in cuts):
                    continue
                # Finite flux products can differ by one endpoint. Empty
                # products keep a zero-width region, preserving metadata order.
                local_regions, destinations = [], []
                for array, region, cut in zip(arrays, regions, cuts):
                    if cut is None:
                        cut = (index, (1, 1, region[2], region[3]), slice(0, 0))
                    local_index, local_region, destination = cut
                    local_regions.append(local_region)
                    destinations.append(destination)
                    local_shapes.append((array.shape[0], local_region[1] - local_region[0],
                                         local_region[3] - local_region[2]))
                local_meta = ((axis, local_index, freqs, channels, tuple(local_regions), *rest)
                              if flux else (channel, axis, local_index, freqs, local_regions[0]))
                part.append((number, local_meta, destinations))
            self.parts.append(part)
            shapes.append(local_shapes)
        dtype = templates[0][0].dtype
        super().__init__(mesh, shapes, dtype)

    def update(self, buffer, state, dt, step, *, ranks):
        branches = []
        for rank, part in enumerate(self.parts):
            def branch(args, rank=rank, part=part):
                local, fields, n = args
                arrays = self.unpack(local[0], rank)
                meta = [entry[1] for entry in part]
                if self.flux:
                    grouped = [tuple(arrays[i:i + 4]) for i in range(0, len(arrays), 4)]
                    values = accumulators.flux(fields, grouped, meta, dt, n,
                                               window_step=fields.step if self.window_from_state else None)
                    arrays = [a for group in values for a in group]
                else:
                    arrays, _ = accumulators.planes(fields, arrays, meta, dt, n)
                return self.pack(arrays)
            branches.append(branch)
        return self.mapped(buffer, state, step, branches, ranks=ranks)

    def assemble(self, buffer):
        buffer = self.replicated(buffer)
        result = [[jnp.zeros(a.shape, a.dtype) for a in (t[:4] if self.flux else (t.accumulator,))]
                  for t in self.templates]
        for rank, part in enumerate(self.parts):
            arrays = iter(self.unpack(buffer[rank], rank))
            for number, _, destinations in part:
                for component, destination in enumerate(destinations):
                    result[number][component] = result[number][component].at[:, destination].set(next(arrays))
        return tuple(t._replace(**dict(zip(("e1_dft", "e2_dft", "h1_dft", "h2_dft")
                                           if self.flux else ("accumulator",), values)))
                     for t, values in zip(self.templates, result))


class SnapshotRecords:
    """Separate component histories; only one frame is written per step."""
    def __init__(self, mesh, grid, nx_per, spec, n_steps, dtype):
        from rfx.snapshots import snapshot_axes
        self.axes = snapshot_axes(grid, spec, n_steps)
        self.spec, self.frames = spec, n_steps // spec.interval
        sample = next(iter(self.axes.values()))
        self.axis = None if sample.slice_axis is None else "xyz".index(sample.slice_axis)
        self.index = sample.slice_index
        self.mesh, self.dtype = mesh, dtype
        self.parts = []
        self.full_shape = tuple(n for a, n in enumerate(grid.shape) if a != self.axis)
        for rank in range(mesh.size):
            start, end = rank * nx_per, min((rank + 1) * nx_per, grid.shape[0])
            owned = self.axis != 0 or self.index // nx_per == rank
            self.parts.append((start, end, owned))
        self.frame_shape = (self.full_shape if self.axis == 0 else
                            (min(nx_per, grid.shape[0]),) + self.full_shape[1:])

    def initial(self):
        shape = (self.mesh.size, self.frames) + self.frame_shape
        return tuple(place(np.zeros(shape, self.dtype), NamedSharding(self.mesh, P("x")))
                     for _ in self.spec.components)

    def update(self, buffer, state, dt, step, *, ranks):
        if not self.frames:
            return buffer
        # The rank branches return only the new frame. The history never
        # enters the conditional: passed through it, XLA copies the whole
        # history every step (cost grows with frames x steps).
        branches = []
        for start, end, owned in self.parts:
            def branch(fields, start=start, end=end, owned=owned):
                if not owned:
                    return (jnp.zeros((), bool),
                            tuple(jnp.zeros(self.frame_shape, self.dtype) for _ in self.spec.components))
                frames = []
                for component in self.spec.components:
                    slices = [slice(1, 1 + max(0, end - start)), slice(None), slice(None)]
                    if self.axis is not None:
                        slices[self.axis] = self.index - (start - 1) if self.axis == 0 else self.index
                    value = getattr(fields, component)[tuple(slices)]
                    # Pad the new frame only, never the history in the carry.
                    frames.append(jnp.pad(value, tuple((0, cap - size) for cap, size in
                                                       zip(self.frame_shape, value.shape))).astype(self.dtype))
                return jnp.ones((), bool), tuple(frames)
            branches.append(branch)

        @partial(rank_shard_map, mesh=self.mesh,
                 in_specs=(P("x"), _field_specs(state), P()), out_specs=P("x"), check_rep=False)
        def apply(local, fields, n, *, rank):
            owned, frames = lax.switch(rank, branches, fields)
            write = owned & ((n + 1) % self.spec.interval == 0)
            row = jnp.minimum(n // self.spec.interval, self.frames - 1)
            arrays = []
            for history, value in zip(local, frames):
                old = lax.dynamic_index_in_dim(history[0], row, 0, keepdims=False)
                value = jnp.where(write, value, old)
                arrays.append(lax.dynamic_update_slice(
                    history, value[None, None], (0, row) + (0,) * value.ndim))
            return tuple(arrays)
        return apply(ranks, buffer, state, step)

    def assemble(self, buffer):
        buffer = jax.tree.map(lambda a: (place(a, NamedSharding(self.mesh, P()))
                                        if isinstance(a, (jax.Array, jax.core.Tracer)) else a), buffer)
        result = {c: jnp.zeros((self.frames,) + self.full_shape, self.dtype) for c in self.spec.components}
        for rank, (start, end, owned) in enumerate(self.parts):
            if owned:
                for component, history in zip(self.spec.components, buffer):
                    value = history[rank]
                    result[component] = (value if self.axis == 0 else
                                         result[component].at[:, start:end].set(value[:, :end - start]))
        return result


class MomentRecords(Packed):
    def __init__(self, mesh, monitor, nx_per, field_dtype):
        from rfx.current_moments import current_moment_accum_dtype
        self.monitor, self.parts = monitor, []
        m = monitor
        for rank in range(mesh.size):
            start = rank * nx_per
            lo = min(m.i_hi, max(m.i_lo, start))
            hi = max(lo, min(m.i_hi, start + nx_per))
            rows = slice(lo - m.i_lo, hi - m.i_lo)
            self.parts.append(m._replace(
                i_lo=lo - start + 1, i_hi=hi - start + 1,
                w_ex=m.w_ex[:, rows], w_ey=m.w_ey[:, rows], w_ez=m.w_ez[:, rows],
                inv_dx_e=m.inv_dx_e[rows],
                seg=m.seg.reshape(m.i_hi - m.i_lo, m.j_hi - m.j_lo)[rows].reshape(-1)))
        shape = (len(m.freqs), m.n_blocks, 3, m.n_weights)
        super().__init__(mesh, [[shape, shape] for _ in self.parts], current_moment_accum_dtype(field_dtype))
        self.previous = Packed(mesh, [[(p.i_hi - p.i_lo, p.j_hi - p.j_lo, p.k_hi - p.k_lo + d)
                                       for d in (1, 1, 0)] for p in self.parts], field_dtype)

    def prepare(self, state, *, ranks):
        from rfx.current_moments import slab_e_snapshot
        branches = []
        for m in self.parts:
            def branch(args, m=m):
                _, fields, _ = args
                return self.previous.pack(slab_e_snapshot(fields, m))
            branches.append(branch)
        return self.previous.mapped(self.previous.initial(), state, 0, branches, ranks=ranks)

    def update(self, buffer, state, dt, step, *, ranks, previous):
        from rfx.current_moments import accumulate_current_moments
        branches = []
        for rank, m in enumerate(self.parts):
            def branch(args, rank=rank, m=m):
                local, fields, n, old = args
                return self.pack(accumulate_current_moments(
                    self.unpack(local[0], rank), fields, self.previous.unpack(old[0], rank), m, dt, n))
            branches.append(branch)
        return self.mapped(buffer, state, step, branches, ranks=ranks, previous=previous)

    def assemble(self, buffer):
        buffer = self.replicated(buffer)
        summed = jnp.sum(buffer, axis=0)
        return tuple(self.unpack(summed, 0))


def _host_arrays(value):
    """Keep global result templates off devices during the scan."""
    return jax.tree.map(lambda x: np.array(x, copy=True) if isinstance(x, jax.Array) else x, value)


class SlabRecords:
    def __init__(self, sim, grid, nx_per, mesh, field_dtype, n_steps, snapshot=None):
        from rfx.measurement.setup import build_planes
        from rfx.measurement.flux_weights import build_flux_monitor_cfgs
        from rfx.current_moments import monitor_for_simulation
        self.layouts, self.sim, self.grid = {}, sim, grid
        self.ntff_box = None
        if sim._ntff is not None:
            from rfx.farfield import make_ntff_box
            self.ntff_box = make_ntff_box(grid, *sim._ntff)
            self.layouts["ntff"] = SlabNTFF(self.ntff_box, grid.shape, nx_per, mesh, field_dtype)
        with jax.ensure_compile_time_eval():
            planes = build_planes(sim, grid, n_steps)
            flux = build_flux_monitor_cfgs(sim, grid, n_steps)
            monitor = monitor_for_simulation(sim, grid, sim._periodic_flags())
        for what, template in (("a DFT plane probe", planes), ("a flux monitor", flux),
                               ("a current-moment monitor", monitor)):
            if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(template)):
                raise NotImplementedError(
                    f"{what} on the distributed multi-device path needs concrete monitor arguments "
                    "(frequencies, positions) inside jax.jit: declare them outside the traced function.")
        planes = _host_arrays(planes)
        flux = _host_arrays(flux)
        if any(m.axis == 0 and m.index == 0 for m in flux):
            raise NotImplementedError("x-normal flux at global x index 0 is not supported on the distributed path; use one device")
        if planes:
            self.layouts["planes"] = PlaneRecords(mesh, planes, grid.shape, nx_per)
        if flux:
            self.layouts["flux"] = PlaneRecords(mesh, flux, grid.shape, nx_per, flux=True,
                window_from_state=not sim._uses_nonuniform_mesh)
        self.monitor = _host_arrays(monitor)
        if self.monitor is not None:
            self.layouts["moments"] = MomentRecords(mesh, self.monitor, nx_per, field_dtype)
        if snapshot is not None:
            self.layouts["snapshots"] = SnapshotRecords(mesh, grid, nx_per, snapshot, n_steps, field_dtype)

    def initial(self):
        return {kind: layout.initial() for kind, layout in self.layouts.items()}

    def prepare(self, state, *, ranks):
        return self.layouts["moments"].prepare(state, ranks=ranks) if "moments" in self.layouts else None

    def update(self, buffer, state, dt, step, *, ranks, previous):
        return {kind: layout.update(buffer[kind], state, dt, step, ranks=ranks,
                                    **({"previous": previous} if kind == "moments" else {}))
                for kind, layout in self.layouts.items()}

    def assemble(self, buffers, *, multi_process):
        values = {}
        for kind, layout in self.layouts.items():
            buffer = buffers[kind]
            if multi_process:
                from jax.experimental import multihost_utils
                buffer = multihost_utils.process_allgather(buffer, tiled=True)
            values[kind] = layout.assemble(buffer)
        result = dict(ntff_box=self.ntff_box, ntff_data=values.get("ntff"),
                      current_moment_data=values.get("moments"), current_moment_monitor=self.monitor)
        if "planes" in values:
            result["dft_planes"] = {e.name: p for e, p in zip(self.sim._dft_planes, values["planes"])}
        if "flux" in values:
            result["flux_monitors"] = {e.name: m for e, m in zip(self.sim._flux_monitors, values["flux"])}
        if "snapshots" in values:
            result.update(snapshots=values["snapshots"], snapshot_axes=self.layouts["snapshots"].axes)
        return result
