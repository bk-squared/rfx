### Fixed — on the distributed graded forward, a current source standing in material supplied by `eps_override` / `sigma_override` received `Cb_drawn/Cb_override` times its current (3.02x for eps_r 3.38 → 10.2) and a permittivity gradient through the override missed the drive's own derivative; it now receives the drive the single-device lane gives it (#1279)

- A soft current source (`add_source`, `amplitude_kind='current'` or the
  default) adds `Cb * I(t) / dV` to its edge every step, and
  `Cb = dt/(eps + sigma*dt/2)` carries the permittivity of that edge.
  `forward(distributed=True)` stepped the field with the override but built
  `Cb` from the permittivity as drawn: with a drawn eps_r 3.38 raised to 10.2
  by the override, the probe field came out 3.04 times the single-device
  lane's, and a slab raised by e^0.1 through the override read 1.105 times the
  same slab drawn that way. Both now agree to within 1.5e-7 (record ratio),
  and d ln|E(8 GHz)|^2 / d ln eps_r through a traced override reads -2.9314
  on both lanes (it read -0.9314: the drive's own share, -2, was missing).
- The drive coefficient is now built inside the jitted program from the slabs
  the field update receives, so every override form `forward` accepts reaches
  it: a local array, an x-sharded global array (also across processes — a
  source on a device's first cell reads its neighbouring cells from the
  device that owns them), and a traced one under `jax.grad` or `jax.vmap`.
- Unchanged, bit for bit: runs without an eps/sigma override, and
  `amplitude_kind='field'` sources (their amplitude is divided by the same
  Cb, which cancels). The single-device lanes are untouched.
