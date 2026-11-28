### Fixed — One device: the whole-grid setup arrays are released before the time loop starts, so they no longer occupy memory beside it (#PR)

- Before, a one-device `sim.run()` or `forward()` kept the whole-grid setup arrays alive while the time loop ran, beside the loop's own operands. These are the assembled materials, the conductor masks, the initial field carry and the stored step context.
- Now the run hands them to the time step and drops its own references first, on the uniform and graded paths. With `until_decay`, the run keeps the permittivity, permeability and conductivity arrays that its decay check reads. A lumped RLC shorted by a conductor is still refused, during setup.
- On one RTX A6000, with a 1024 × 64 × 192 float32 grid, a wire port, a DFT plane and a probe over 200 steps, the device memory peak falls from 1.680 GB to 1.290 GB. The 390 MB difference is the size of 7.75 grid-sized arrays.
- Fields, probes, port records and DFT planes are bit-identical to before, in seven CPU configurations: uniform, graded, `until_decay`, S parameters, `forward()` on both meshes, and a scene with conductors. On the A6000 scene, the final fields, the port record and the DFT plane are bit-identical as well.
- `Simulation.realized_geometry()` after a run still returns the full conductor arrays. It rebuilds them when the run has released its own.
