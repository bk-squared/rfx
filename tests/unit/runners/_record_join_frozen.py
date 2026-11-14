"""Frozen joins from origin/main 064695dff, independent of the product join."""
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from rfx.stepping.slab import place
from rfx.farfield import init_ntff_data, NTFFData

def PlaneRecords(self, buffer):
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


def SnapshotRecords(self, buffer):
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


def MomentRecords(self, buffer):
    buffer = self.replicated(buffer)
    summed = jnp.sum(buffer, axis=0)
    return tuple(self.unpack(summed, 0))


def SlabNTFF(self, buffer):
    """Place each cell once, preserving compensation; no spatial sum."""
    result = list(init_ntff_data(self.box, field_dtype=jnp.float64
                                if self.dtype == jnp.complex128 else jnp.float32))
    # Indexing the x-sharded buffer by rank compiles a gather with an
    # XLA partition-id; replicate it first, which stays
    # differentiable. The multi-process caller passes an already
    # gathered host array, which needs neither.
    if isinstance(buffer, (jax.Array, jax.core.Tracer)):
        buffer = place(buffer, NamedSharding(self.mesh, P()))
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


def TimePlaneRecords(self, buffer):
    # New per-step oracle: place each rank's owned rectangle using NumPy.
    # This deliberately does not call the production join or host assembler.
    import numpy as np
    result = [np.zeros((self.frames,) + t.accumulator.shape[1:], self.dtype)
              for t in self.templates]
    for rank, part in enumerate(self.parts):
        for number, meta, destinations in part:
            width = meta[-1][1] - meta[-1][0]
            result[number][:, destinations[0]] = np.asarray(buffer[number])[rank, :, :width]
    return tuple(result)


def assemble(layout, buffer):
    return globals()[type(layout).__name__](layout, buffer)
