### Changed — `forward(distributed=True)` exchanges only the four boundary faces the Yee update reads, and sums the probes once after the time loop

- The differentiable multi-device forward sent all six field components across every slab
  cut in both directions every step (with wrap-around) and all-reduced the probe samples
  every step: for two devices, 24 face transfers and 13 collectives per step, and the same
  again in the reverse loop of a gradient. It now does what `sim.run(devices=...)` does
  since #1222: Hy and Hz go one cell to the right, Ey and Ez one cell to the left, one
  packed transfer each way, and the probe samples stay on their owner device until one sum
  after the loop. That is 2 collectives per step in the forward loop and 2 in the reverse
  loop.
- Probe traces are bit-identical to the full exchange, and the permittivity, conductivity
  and occupancy gradients agree with it within 3.5 float32 ULP of each gradient's peak (JAX
  0.10.2, 0.6.2 and 0.4.33 on CPU; 2, 3 and 4 devices with uneven slabs; a model with a
  lossy Debye + Lorentz block, a PEC block, PMC and PEC faces, CPML and a graded x mesh).
