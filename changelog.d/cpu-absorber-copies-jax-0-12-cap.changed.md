### Changed — on jax 0.11.2 the CPU time step no longer copies every field at each absorber update, and rfx now installs jax below 0.12 (#PR)

- On CPU with jax/jaxlib 0.11.2 one time step took 1.8 to 2.9 times as long as on jax 0.10.2 (8 cores, 0.53 M and 3.05 M cells), because the compiler copied each field between the absorber's partial updates. An optimization barrier after each update removes the copies: the step now takes 0.98 to 1.03 times as long as on jax 0.10.2. Results are unchanged on jax 0.10.2 (bit for bit) and agree to 2.3e-6 of the peak gradient on 0.11.2.
- The install cap moves from jax < 0.11.2 to jax < 0.12. jax 0.12 still takes 1.24 to 1.29 times as long per step at 0.53 M cells on CPU and GPU (see known limitations, #2125).
