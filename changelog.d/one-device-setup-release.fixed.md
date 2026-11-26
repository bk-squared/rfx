### Fixed — One device: the whole-grid setup arrays are released before the time loop starts, so they no longer occupy memory beside it (#PR)

- Before, a one-device `sim.run()` or `forward()` kept the whole-grid setup arrays alive while the time loop ran, beside the loop's own operands. These are the assembled materials, the conductor masks, the initial field carry and the stored step context.
- Now the run hands them to the time step and drops its own references first. This holds on the uniform and graded paths, with and without `until_decay`. A lumped RLC shorted by a conductor is still refused, during setup.
- Fields, probes, port records and DFT planes are bit-identical to before.
- `Simulation.realized_geometry()` after a run still returns the full conductor arrays. It rebuilds them when the run has released its own.
