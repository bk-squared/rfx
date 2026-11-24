### Changed — the `GaussianPulse.t0` and `LorentzPoleMap` docstrings state what the code computes (#PR)

- `GaussianPulse.t0` is the centre of the Gaussian, `cutoff * tau`. The differentiated pulse is zero
  there, with its extrema near `t0 ± tau/√2`; the docstring called `t0` the peak.
- A `LorentzPoleMap` callable is read only at occupied cells, also under `jax.grad`, so it need not be
  finite outside its material. The docstring still asked for its argument to be clamped there.
