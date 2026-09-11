# First full CI inventory and repairs

2026-09-11, PR #981. Initial head c2dd484e completed with 15 successful
checks and 6 failed shard jobs. Their failures reduce to four causes:

- The settling, progress-reporting and vmap MSL fixtures omitted their
  explicit ground. The mixed-port wave fallback also needs a physical
  ground even when its domain z_lo face is CPML. Add the missing conductors
  and update the independent sheet/plane census. Preserve the original
  field, settling, bit-identity and eligibility assertions.
- Two new blocking preflight emission sites share the conductor-plane
  code. Audit and register 109 sites / 74 literal codes, with the existing
  dynamic-site and entrypoint classifications unchanged.
- JAX 0.10.2 removed the experimental x64 contexts. Production now prefers
  the public jax.enable_x64 API, falling back to the experimental import
  on older supported JAX. Projection math, dtype/device preservation and
  precision scope are unchanged. See the official JAX changelog:
  https://docs.jax.dev/en/latest/changelog.html
- The separate short-record AD smoke fixture still declared a 254 um port
  under a trace at 320 um. Its alignment repair is checked separately;
  it must not inherit the settled gate's accuracy claims.

Verified locally at this checkpoint: 5 settling/census tests, 7 emission/
progress/vmap/mixed tests, and on a separate Python 3.11 environment pinned
to CI's JAX 0.10.2 and NumPy 2.4.6, 15 projection/precision cases plus 3
actual MSL extraction cases. Logs are losslessly compressed with hashes.
The original GPU image's experimental context remains the fallback; the
new import changes no numerical operation on that image.

The initial failed CI is not acceptance. Final head CI remains required.

The aligned short-record AD smoke and its independent trace-plane check
also passed (2 tests): loss 1.944183, max|S| 1.0128, gradient -0.02061103.
The objective, three-period record and all assertions were retained.
The unrelated probe-only control keeps its old 80 um pitch. Comments now
state the limits correctly: a negative finite-record gradient is an
empirical fixture fingerprint, not a law that epsilon increases loss;
short records do not excuse AD/FD disagreement. The separate settled MSL
gate carries the accuracy claim. No new FD comparison is claimed here.
