### Fixed — `forward(ringdown=...).ringdown.report` can be read after `jax.grad`, `jax.jvp` or `jax.jacfwd` on a graded mesh with traced cell widths (#PR)

Before, every such read raised `UnexpectedTracerError`, so a design step needed one extra concrete `forward()` to get
its record-end rows (W0, W1, WE, passivity). The report now judges the record the gradient came from, on the mesh that
design step realized, and a concrete run's report is unchanged.
