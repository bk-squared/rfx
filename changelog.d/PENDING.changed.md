### Changed — `sim.run(devices=...)` exchanges only the four boundary faces the Yee update reads, and sums the probes once after the time loop (#PENDING)

- Every step used to send all six field components across every slab cut in both
  directions (with wrap-around), then all-reduce the probe samples: for two devices, 24
  face transfers and 13 collectives per step. The E update reads only Hy and Hz from the
  left neighbour and the H update only Ey and Ez from the right one, so the runner now
  sends those two pairs, one packed transfer each way: 4 face transfers and 2 collectives
  per step. Probe samples stay on their owner device and are summed once after the loop.
- Results are bit-identical to the full exchange on JAX 0.10.2 (PEC and CPML; 2, 3 and
  4 devices with uneven slabs; seam-cell sources and probes; a PMC face, a PEC volume, a
  lossy block, Debye and Lorentz blocks and a lumped port). On JAX 0.4.33 (CPU) one
  model with all of those features moves by 1–2 float32 ULP because the compiler fuses
  the shorter program differently; with fusion disabled it is identical there too.
- The multi-device `forward(distributed=True)` path still uses the full exchange.
