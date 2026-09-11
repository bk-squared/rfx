# GPU consumer failure: passivity reconstruction precision

2026-09-11. Measured on RTX 4090, JAX 0.4.33.dev20241023+e3c6d6430,
NumPy 1.26.4. This is a numerical defect in the existing measurement
postprocessor, found while validating the #729 source correction.

The unchanged x/y rotation test failed on both the candidate `ea8c8f3f`
(run 369367260415) and current main `e32e386d` (369367260416). Candidate
maximum complex S difference was 9.785e-4; main was 9.093e-4, against the
existing 1e-4 bound. Both drives in both orientations settled below -100 dB.
The sequential driver stopped before the AD and NU cases.

The candidate's raw S matrices differed by only 4.242e-6. Its passivity
projection amplified that difference and returned matrices with maximum
singular values 1.000354 and 1.000278, violating its own strict bound.
The raw V/I, raw S and returned S are retained in the two run directories.

## Controlled replay, no further FDTD

Run 369367260417 replayed exactly the candidate's retained matrices on
current main. The SVD factors, singular-value clipping radius, dtype and
inputs were held fixed. Only the reconstruction multiply's precision
changed to `jax.lax.Precision.HIGHEST`.

| Quantity | Default reconstruction | Highest reconstruction |
|---|---:|---:|
| Worst reconstruction error before clipping | 4.31e-4 | 5.65e-7 |
| Difference from independent host-f64 projection, same radius | 5.43e-4 | 4.45e-7 |
| Largest singular value after clipping | 1.000354 | 0.99999264 |
| x/y maximum complex S difference | 9.785e-4 | 3.919e-6 |

The f32 SVD factors themselves reconstruct on the host to about 6e-7 and
are orthogonal to about 7e-7. Replacing SVD or enabling f64 for the whole
simulation is therefore unnecessary for this failure. JAX exposes local
contraction precision through [`einsum(precision=...)`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.einsum.html).

The existing radius `1 - 64*eps(float32)` leaves approximately 7.63e-6 for
reconstruction. A 4e-4 multiply error cannot respect that budget; increasing
the clipping margin would hide the arithmetic defect by perturbing the
measured S further. The correction specifies full multiplication precision
only in `_project_passive`, preserving its dtype, clipping rule, raw-data
retention, and exclusion from the `eps_override`/AD channel.

The new small near-unitary matrix tests compare 2-, 8-, and 32-port
projections against an independent host-f64 reference and the strict
passivity bound. CPU and explicitly GPU-marked variants keep this contract
in both execution environments. The GPU variant deliberately exercises a
lower ambient matmul precision, which the local physical operation must
override. Full candidate consumer acceptance is still pending after these
diagnostic runs; their replay result does not claim the AD or NU case ran.

Evidence:

- `gpu-consumers-369367260415/`: original candidate failure and raw records.
- `gpu-axis-baseline-369367260416/`: same failure on unmodified main, plus
  the first pure-matrix attribution.
- `gpu-projection-precision-369367260417/`: controlled precision replay.
- Corresponding complete provider logs and hashes are under
  `docs/research_notes/vessl_logs/` and each run's `files.json`.
