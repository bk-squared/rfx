Non-uniform forward simulations with `checkpoint_every` now run exactly the
requested number of time steps, including when the checkpoint chunk does not
divide the record length. This prevents extra zero-source steps from changing
final fields and accumulated port, DFT, flux, and NTFF measurements. The same
fix applies to the distributed non-uniform forward runner.
