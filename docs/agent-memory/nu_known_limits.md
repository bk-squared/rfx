# Non-Uniform Mesh Path — Known Limits

Sentinel tests use `xfail(strict=True)` so XPASS will trip CI when a
limit is closed. Update this doc when that happens.

## 1. `pec_occupancy_override` (soft-PEC continuous occupancy)

- **Status**: Not supported on NU path. Hard PEC mask only.
- **Sentinel**: `tests/test_nonuniform_forward_grad.py` — xfail strict.
- **Cause**: `run_nonuniform` scan body has no occupancy field plumbed
  through. Uniform path supports it.
- **Use case unblocked when fixed**: density-based topology optimization
  on graded meshes.

## 2. `dz_profile` / mesh-as-design-variable gradient

- **Status**: `jax.grad` w.r.t. `dz_profile` raises
  `TracerArrayConversionError`.
- **Sentinel**: `tests/test_nonuniform_gradient.py` — xfail strict.
- **Cause**: `make_nonuniform_grid` calls `np.asarray` and
  `float(np.min)` on `dz_profile` before any JAX op — host boundary.
- **Scope**: source/eps/sigma gradients on NU mesh work fine; only the
  mesh geometry itself is non-differentiable.
- **Future research**: adaptive mesh refinement / mesh inverse design.
  Out of scope for issue #31.
