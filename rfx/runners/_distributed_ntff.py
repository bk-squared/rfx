"""Owned surface records for the uniform distributed run lane.

Each rank packs its six face parts and six compensations. A shard has room
for the largest owner's record (zero tail padding only), independent of time.
The global face record is allocated only when assembling the result.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.farfield import (NTFFData, accumulate_ntff, init_ntff_data,
                          ntff_accum_dtype, _require_face_centre_margin)


class SlabNTFF:
    """Static ownership/layout and shared-arithmetic scan adapter."""

    def __init__(self, box, shape, nx_per, mesh, field_dtype):
        _require_face_centre_margin(box, shape)
        self.box, self.mesh = box, mesh
        self.n_devices = mesh.size
        self.dtype = ntff_accum_dtype(field_dtype)
        self.parts = []
        nf, nj, nk = len(box.freqs), box.j_hi - box.j_lo, box.k_hi - box.k_lo
        for rank in range(self.n_devices):
            start = rank * nx_per
            lo = min(box.i_hi, max(box.i_lo, start))
            hi = max(lo, min(box.i_hi, start + nx_per))
            faces = tuple(index // nx_per == rank for index in (box.i_lo, box.i_hi))
            shapes = [(nf, nj if owned else 0, nk, 4) for owned in faces]
            shapes += [(nf, hi - lo, nk, 4)] * 2 + [(nf, hi - lo, nj, 4)] * 2
            shapes *= 2
            sizes = [int(np.prod(shape)) for shape in shapes]
            self.parts.append((lo, hi, faces, start - 1, shapes, sizes))
        self.capacity = max(sum(part[-1]) for part in self.parts)

    def initial(self):
        return jax.device_put(jnp.zeros((self.n_devices, self.capacity), self.dtype),
                              NamedSharding(self.mesh, P("x")))

    @staticmethod
    def unpack(buffer, shapes, sizes):
        offset = 0
        arrays = []
        for shape, size in zip(shapes, sizes):
            arrays.append(buffer[offset:offset + size].reshape(shape))
            offset += size
        return NTFFData(*arrays)

    def update(self, buffer, state, dt, step):
        # The right ghost Hx equals the neighbour's first real Hx: the H
        # update computes ghost rows from exchanged E, including CPML
        # corrections. Face-centre y/z samples rely on this invariant.
        branches = []
        for lo, hi, faces, offset, shapes, sizes in self.parts:
            def branch(args, lo=lo, hi=hi, faces=faces, offset=offset,
                       shapes=shapes, sizes=sizes):
                local, fields, n = args
                data = self.unpack(local, shapes, sizes)
                data = accumulate_ntff(data, fields, self.box, dt, n,
                                       x_offset=offset, owned_x=(lo, hi),
                                       owned_x_faces=faces)
                packed = jnp.concatenate([a.ravel() for a in data])
                return jnp.pad(packed, (0, self.capacity - sum(sizes)))
            branches.append(branch)

        field_specs = state._replace(**{name: P("x") for name in
                                      ("ex", "ey", "ez", "hx", "hy", "hz")}, step=P())

        @partial(shard_map, mesh=self.mesh, in_specs=(P("x"), field_specs, P()),
                 out_specs=P("x"), check_rep=False)
        def update_local(local, fields, n):
            return lax.switch(lax.axis_index("x"), branches,
                              (local[0], fields, n))[None]
        return update_local(buffer, state, step)

    def assemble(self, buffer):
        """Place each cell once, preserving compensation; no spatial sum."""
        result = list(init_ntff_data(self.box, field_dtype=jnp.float64
                                    if self.dtype == jnp.complex128 else jnp.float32))
        for rank, (lo, hi, faces, _, shapes, sizes) in enumerate(self.parts):
            part = self.unpack(buffer[rank], shapes, sizes)
            for index, value in enumerate(part):
                face = index % 6
                if face < 2:
                    if faces[face]:
                        result[index] = value
                else:
                    result[index] = result[index].at[:, lo - self.box.i_lo:hi - self.box.i_lo].set(value)
        return NTFFData(*result)
