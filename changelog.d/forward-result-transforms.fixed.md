### Fixed — Return whole forward results through JAX transforms (#PR)

Whole `forward()` results can pass through `jax.jit`, `jax.vmap`, `jax.jvp`
and `jax.eval_shape`, and serve as auxiliary output for `jax.vjp`, `jax.grad`,
`jax.value_and_grad` and `jax.jacfwd` with `has_aux=True`. Host metadata stays
as computed during the trace. Read batched results one element at a time using
`jax.tree_util.tree_map(lambda a: a[i], result)`; settling diagnostics refuse
results whose array shapes changed. Read diagnostics on primal results, since
JVP tangents and Jacobians retain the result type without diagnostic meaning.
