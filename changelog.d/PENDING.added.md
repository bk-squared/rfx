### Added — `forward(distributed=True)` runs across several processes (nodes) and takes its design variables as x-sharded arrays, so gradients no longer need a whole-domain copy anywhere (#PENDING)

- The differentiable multi-device forward ran in one process only: its time loop closed over
  the grid-spacing arrays, its setup built global arrays from one process's data, and its
  design inputs had to be whole-domain arrays. Every array the loop reads is now a jit
  argument, geometry is placed slab by slab on whichever process owns each slab, and the
  loop runs on a device mesh that spans processes (one process per GPU on several nodes).
- `eps_override`, `sigma_override` and `pec_occupancy_override` may be jax.Arrays sharded
  along x with the layout from `Simulation.distributed_override_layout(devices)`; ghost rows
  are filled inside the jitted program, and `jax.grad` returns the gradient in the same
  sharded layout. `Simulation.shard_distributed_override(...)` places a local array in that
  layout. A local whole-domain override is still accepted in one process and refused, with
  that layout named, when the devices span processes.
- Two processes (CPU) give the same probe trace and permittivity gradient as one process,
  bit for bit; the existing local-override path is unchanged, bit for bit.
- Forward no longer gathers the whole final field state after the loop (it never used it).
