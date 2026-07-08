# WP 4-E — Traceability spike: lumped R/L/C component values as a differentiable DoF

Status: **R2-STOP (design note is the deliverable).** No production source was changed.
Date: 2026-07-08. Branch: `worktree-agent-accfde1ea794a65c2`.

## Question

Can a lumped element's component value (R, L, or C from
`Simulation.add_lumped_rlc(...)`) be a differentiable design variable, i.e. is
there a working `jax.grad(|S11|^2)` (or any scalar port objective) w.r.t. R or C
through a public rfx path? This is the "unserved-novel" DoF class: no adjoint-EM
stack ships lumped-component-value gradients.

## Verdict

**Not through any public path today, and closing the gap is a scope-(b)
structural change to the differentiable `forward()` lane — R2-STOP.** The
underlying FDTD/ADE scan physics *is* differentiable w.r.t. a component value
(positive witness below, L-DoF, AD-vs-FD rel err 8.3e-6), so the DoF class is
viable; the blockers are entirely in the setup layer and in the fact that the
differentiable lane never processes lumped RLC at all.

## Where the trace dies (empirical, this branch, jax 0.6.2, rfx 1.6.6)

Two independent, load-bearing blockers. Neither is a one-line fix.

### Blocker 1 (PRIMARY, structural) — `forward()` never processes lumped RLC

`Simulation.forward()` is the AD lane (designed for `jax.grad` /
`jax.value_and_grad`). Its uniform path `_forward_from_materials`
(`rfx/api/_execute.py:520`) iterates `self._ports` and `self._msl_ports` and
folds them into materials, but **never iterates `self._lumped_rlc`** — it does
not call `setup_rlc_materials` / `build_rlc_meta`, and it does not pass
`lumped_rlc=` to the differentiable driver `_run(...)` (`rfx/api/_execute.py:932`).
The distributed forward lane at least *rejects* RLC
(`rfx/api/_execute.py:1429`, `NotImplementedError`); the uniform lane silently
ignores it.

Empirical proof (byte-identical): building the same sim with vs without an
`add_lumped_rlc(...)` element and calling `forward(port_s11_freqs=...)` gives
`max|Δ|S11|| = 0.0` — the RLC element has **zero** effect on the differentiable
result, so `∂|S11|²/∂R ≡ 0` structurally (no path to differentiate).

Consequence: making `build_rlc_meta` jnp-native (Blocker 2) is **necessary but
not sufficient**. Without threading `_lumped_rlc` into `_forward_from_materials`,
R/C never enter the differentiable forward at all.

### Blocker 2 (setup layer) — `build_rlc_meta` / `setup_rlc_materials` concretize

Even at the low-level driver, the meta builder concretizes component values:

- `rfx/lumped.py:227-228`: `eps = float(materials.eps_r[i,j,k]) * EPS_0`,
  `sigma = float(materials.sigma[i,j,k])`. When R (or C) is a tracer folded into
  `materials.sigma` / `materials.eps_r`, `float(...)` raises
  **`jax.errors.ConcretizationTypeError` at `rfx/lumped.py:228`**. Confirmed for
  pure-R series and R+C series.
- `rfx/lumped.py:232-246`: `has_inductor = spec.L > 0`, then Python
  `if has_inductor:` / `if has_capacitor:`. Under `jax.grad` (JVPTracer) a
  comparison returns the concrete primal bool, so the `if` *passes*; under
  `jax.jit` (abstract DynamicJaxprTracer) it raises
  **`TracerBoolConversionError`**. The production `optimize` path is jit-wrapped,
  so these gates bite there.
- `rfx/lumped.py:158-159` `_series_needs_ade`:
  `n_components = (spec.R>0) + (spec.L>0) + (spec.C>0)`. With traced component
  values, `spec.R>0` is a JAX **bool** array and `bool + bool` has OR semantics
  (not an integer count), so a genuine 2-component R+C series wrongly returns
  `False` → the fold path folds R into `sigma` → Blocker-2 `float(sigma)` death.
  Confirmed empirically (`_series_needs_ade = False` under a traced R+C).
- `rfx/lumped.py:198` `setup_rlc_materials`: the `if spec.topology=="series" and
  _series_needs_ade(spec):` guard also raises **`TracerBoolConversionError`**
  under jit for a traced component value.
- `rfx/lumped.py:120-122` `init_rlc_state`: the ADE carry
  (`inductor_current`, `capacitor_charge`) is pinned to `jnp.float32`. A tracer
  DoF under `jax_enable_x64` promotes the update to float64 and trips the
  **scan-carry dtype contract** ("carry input ... float32[] but output ...
  float64[]"). So a scoped-x64 DoF also needs the ADE state dtype threaded.

## Positive witness — the scan physics IS differentiable w.r.t. a component value

Pure-L parallel element, L as the DoF, through the **real** `build_rlc_meta`
plus the low-level differentiable driver `rfx.simulation.run(lumped_rlc=[meta])`,
under `jax.grad` (float32). Pure-L is not folded into materials, so
`float(sigma)`/`float(eps)` hit concrete values and the `has_inductor` gate is
concrete under a JVPTracer:

```
energy(L0=10nH)   = 1.191247e-02
AD  d(energy)/dL  = 4.318479e+06
FD  d(energy)/dL  = 4.318515e+06
rel_err(AD,FD)    = 8.3e-06      finite=True  nonzero=True   -> PASS
```

This establishes that (a) the scan-carry contract already supports RLC
(`rfx/simulation.py:800-805` builds `carry_init["rlc_states"]`, the scan body
updates it at `rfx/simulation.py:1204-1207`), and (b) reverse-mode AD flows
cleanly through the ADE update once a component value reaches the meta as a
tracer. The DoF class is real; only the plumbing is missing. (A working end-to-end
`|S11|²`-vs-R witness could not be produced because the `|S11|` path lives behind
`forward()`, which is exactly Blocker 1.)

## Scope of a real fix — (b), not (a) alone, not (c)

- **(a) jnp-native setup layer** — NECESSARY: drop the `float()` at
  `lumped.py:227-228`; replace the Python `if` gates (`lumped.py:236-246`,
  `:198`) with `jnp.where` / static-topology dispatch; fix `_series_needs_ade`
  to count with integer casts not bool `+`; thread the ADE-state dtype in
  `init_rlc_state`. INSUFFICIENT on its own (Blocker 1).
- **(b) thread `_lumped_rlc` into the differentiable forward lane** — REQUIRED
  and the crux: in `_forward_from_materials` iterate `self._lumped_rlc`, call
  `setup_rlc_materials` (already jnp; folds R/C into the jnp materials arrays),
  build a **traced** meta, and pass `lumped_rlc=metas` to the `_run(...)` call at
  `rfx/api/_execute.py:932`. Plus an injection surface so a component value can
  be supplied AS a tracer (e.g. a `rlc_values_override` mapping, mirroring
  `eps_override` / `sigma_override`), since `LumpedRLCSpec` stores plain floats.
- **(c) change the `lax.scan` carry** — NOT required. The carry already carries
  `rlc_states` (`rfx/simulation.py:800-805`). The only carry-adjacent friction is
  the float32 dtype pin in `init_rlc_state` (a dtype thread, not a new carry
  field). `ForwardResult`'s return pytree is unchanged.

Because a working R/C gradient requires (b) — a structural change to the
`forward()` lane and a new tracer-injection surface — this trips the WP 4-E
falsifier / R2 rule. STOP; do not force it in a spike.

## Why the "harmless enabling half" was declined

The task permitted optionally shipping a jnp-native `build_rlc_meta` alone IFF
byte-identical for plain-float inputs. It is **not** byte-identical: the current
path computes `D0`/`gamma`/`dt_dx_over_L` in Python float64 and passes them as
weakly-typed scalars into the scan; dropping `float()` computes them in float32.
Measured `D0`: float64 path `539.1448218627913` vs float32 path
`539.144775390625` (abs diff `6.1e-5` after casting both to float32). Bit-identity
of the concrete path is non-negotiable here, and the enabling half delivers zero
functional value on its own (Blocker 1). A byte-identity-safe version requires a
*separate* traced code path (not a mutation of the concrete setup path), which is
part of the real (b) fix, not a harmless standalone step. So `rfx/lumped.py` was
left untouched.

## Recommended path (if/when prioritized)

1. Add a traced meta builder used only by the differentiable lane (leave the
   concrete `build_rlc_meta` byte-identical for `run()`), plus a
   `rlc_values_override` injection surface on `forward()`.
2. Thread `_lumped_rlc` through `_forward_from_materials` → `_run(lumped_rlc=)`.
3. Thread the ADE-state dtype in `init_rlc_state` for scoped-x64 FD checks.
4. Gate with: (i) AD-vs-FD `∂|S11|²/∂R` and `∂/∂C` on a stable lumped fixture
   (rel < 5%, scoped x64, freqs pinned where the source has energy), and
   (ii) a byte-identity test proving `run()` and the plain-float `forward()` path
   are unchanged. Byte-identity of the concrete path is the falsifier.

## Reproduction

Diagnostics used for this note (scratchpad, not committed): traced-R death at
`lumped.py:228`; jit `TracerBoolConversionError` at `lumped.py:198`;
forward()-ignores-RLC byte-identical check; `_series_needs_ade` bool-`+`
mis-count; `D0` float64-vs-float32 byte-identity; and the pure-L AD-vs-FD witness
above.
