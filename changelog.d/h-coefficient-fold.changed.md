### Changed — the uniform one-device H update forms its coefficients once per call (#PR)

- The magnetic update coefficients dt/(μ·μ0), with their high-face row factors, and the CPML H face-window products are formed once per call before the time loop instead of every step. Fields are bit-identical to the previous code in the tested cases; the gradient with respect to mu_r and the checkpointed gradients differ in the last bits only (at most 1.04e-5 of their largest component).
- No result and no measured speed changes for a user: on the 81³ uniform one-device gradient at JAX 0.10.2 (RTX A6000) the step time was 3.5 µs longer with this change, inside its standard error of 6.6 µs.
