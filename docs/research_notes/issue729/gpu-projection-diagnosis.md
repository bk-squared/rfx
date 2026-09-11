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
measured S further. The initial correction specified full multiplication
precision only in `_project_passive`. The broader test below showed why
this alone was not sufficient for the existing multiport contract.

The new small near-unitary matrix tests compare 2-, 8-, and 32-port
projections against an independent host-f64 reference and the strict
passivity bound. CPU and explicitly GPU-marked variants keep this contract
in both execution environments. The GPU variant deliberately exercises a
lower ambient matmul precision, which the local physical operation must
override. Full candidate consumer acceptance is still pending after these
diagnostic runs; their replay result does not claim the AD or NU case ran.

## Follow-up: factor accuracy at 32 ports

Run 369367260422 used the committed highest-precision correction `089ada4f`.
Both CPU/GPU cases passed at 2 and 8 ports, but a near-unitary 32-port GPU
case returned sigma_max=1.000003074. It failed the existing strict bound;
the test and clipping radius were not relaxed. The driver stopped there.

Run 369367260427 isolated the two operations on near-unitary and strongly
nonpassive matrices of sizes 2, 8 and 32, in complex64 and complex128:

- At 32 ports the f32 GPU SVD factors had orthogonality error up to 1.7e-5
  and singular-value error about 1e-5 on the near-unitary input. Full
  multiplication precision cannot recover accuracy absent from the factors.
- Applying only the clipped correction to the original matrix improved
  near-unitary inputs, but a strongly nonpassive 32-port case then returned
  sigma_max=1.000089. This algebraically equivalent spelling was rejected.
- Local f64 factorization/reconstruction and conversion to f32 fixed the
  f32 cases. Native f64 output on a near-unitary 32-port case still exceeded
  the strict bound by 7e-15. This image's JAX SVD has no algorithm-selection
  argument, so it cannot explicitly request a different GPU factorization.

The final implementation uses host LAPACK in double precision for this
small concrete S matrix. Both production call sites already exclude traced
S; MSL also excludes a concrete eps_override so FD and AD remain the same
unprojected function. This introduces no host conversion into the
differentiable chain and changes no FDTD field dtype. An independent
read-only caller audit confirmed these two call sites and found no
JIT/grad/vmap caller of the private projection helper.

The radius still uses the input/output dtype's 64*eps. Output arrays return
to their original dtype and device placement. Host-to-device restoration
uses a scoped x64 context so a complex128 array created in an earlier scope
does not silently become complex64. Only finite bins enter LAPACK; invalid
bins retain NaN values and NaN corrections for the caller's finiteness audit.
The final tests construct their analytic answer from known singular values
and orthonormal factors, without sharing the production SVD as their oracle.
A separate Hermitian power-operator eigensolve checks the passive bound.
They cover both dtypes, near-unitary 2/8/32-port matrices, and existing
strongly nonpassive 2/8/32-port inputs. Context and invalid-bin tests pin
the compatibility requirements. GPU acceptance of the final implementation
is pending at this checkpoint.

Evidence:

- `gpu-consumers-369367260415/`: original candidate failure and raw records.
- `gpu-axis-baseline-369367260416/`: same failure on unmodified main, plus
  the first pure-matrix attribution.
- `gpu-projection-precision-369367260417/`: controlled precision replay.
- `gpu-corrected-consumers-369367260422/`: the broader strict-bound failure.
- `gpu-projection-size-369367260427/`: factor and reconstruction attribution.
- Corresponding complete provider logs and hashes are under
  `docs/research_notes/vessl_logs/` and each run's `files.json`.
