# Inverse Design Cookbook

This page is the **safe starting point** for using `rfx` as an RF research
backend. It focuses on the combinations that are currently reliable enough to
use without repeated trial-and-error, and it labels the combinations that are
still experimental.

For the step-by-step execution order, see
[RF Backend Workflow](rf_backend_workflow.md). For a runnable supported example,
see `examples/09_agent_friendly_workflow.py` (a small impedance-mismatch
removal workflow with a forward physics sanity check).

## Workflow status

| Workflow | Status | Notes |
|---|---|---|
| `Simulation.run()` + ports + `result.s_params` | Supported | Use for forward RF analysis and validation |
| `optimize()` with time-domain proxy objectives | Supported with guardrails | Add probes explicitly |
| `topology_optimize()` with dielectric foreground/background | Experimental-supported | Use proxy objectives; keep problems small first |
| `topology_optimize(material_fg="pec")` | Experimental / caveat | Gradient behavior is still evolving |
| NTFF post-processing on canonical radiators | Supported with setup rules | Margin + domain sizing matter |
| NTFF-driven inverse design on practical antennas | Experimental | Treat as research workflow, not turnkey |
| `estimated_memory_mb` as a hard optimizer memory guarantee | Not supported | Use it only as a coarse planning estimate |

## 1. Start from the forward problem first

Before running optimization, prove that the corresponding **forward** problem is
healthy:

1. build the geometry
2. run `Simulation.run()`
3. verify the expected probe signal / resonance / S-parameter / far-field trend
4. only then move to `optimize()` or `topology_optimize()`

If the forward case is not yet believable, optimization will usually turn a
setup problem into a harder-to-debug gradient problem.

## 2. Objective compatibility rules

`rfx` has two different objective families.

Before launching a costly run, you can ask `rfx` for structured diagnostics:

```python
report = sim.preflight_optimize(region, obj, n_steps=500)
print(report.summary())
```

The preflight report combines:
- objective / probe / NTFF compatibility checks
- bounded physics-aware checks for common setup mistakes
- optional compile-time memory-budget enforcement for the supported-safe proxy lane

For autonomous/agentic workflows, use strict enforcement at the optimizer call
site:

```python
result = optimize(sim, region, obj, n_iters=20, n_steps=500, preflight_mode="strict")
```

In `strict` mode, experimental warning lanes are blocked as well. That is the
recommended setting for autonomous agents.

If you want `rfx` to reject supported-safe runs that exceed a compile-time
gradient-memory budget, pass `memory_budget_mb`:

```python
result = optimize(
    sim,
    region,
    obj,
    n_iters=20,
    n_steps=500,
    preflight_mode="strict",
    memory_budget_mb=18000,
)
```

This budget gate currently targets only the tagged built-in supported-safe proxy
objectives. It is not a universal guarantee for arbitrary custom objectives or
experimental NTFF-driven paths. It is also backend-dependent: if a backend reports
compile-time memory as unavailable (for example, zero-valued
`memory_analysis()` on some GPU/JAX combinations), preflight will reject the
budget request instead of pretending the gate is enforced.

### A. Forward-analysis objectives
These require `result.s_params`, so they belong to **forward runs**:

- `minimize_s11()`
- `maximize_s21()`
- `target_impedance()`
- `maximize_bandwidth()`

Typical usage:

```python
result = sim.run(n_steps=1200, compute_s_params=True)
loss = minimize_s11(jnp.array([5e9]), target_db=-10)(result)
```

### B. Gradient-optimization objectives
These are the objectives intended for `optimize()` / `topology_optimize()`:

- `minimize_reflected_energy()`
- `maximize_transmitted_energy()`
- `steer_probe_array()`
- `maximize_directivity()` (experimental; requires NTFF and uses a coarse normalized-directivity surrogate)

Typical usage:

```python
sim.add_probe(port_position, "ez")
sim.add_probe(output_position, "ez")

obj = minimize_reflected_energy(port_probe_idx=0)
result = optimize(sim, region, obj, n_iters=20, n_steps=500)
```

## 3. Probe rules for optimization

Time-domain proxy objectives use `result.time_series`, so **you must provide the
probes they consume**.

### Reflection minimization
For `minimize_reflected_energy()`:
- add a probe at the excited port location
- use that probe index as `port_probe_idx`

### Transmission maximization
For `maximize_transmitted_energy()`:
- add a probe at the output / transmitted-field location
- use that probe index as `output_probe_idx`

### Probe-array steering
For `steer_probe_array()`:
- add all target/suppression probes explicitly
- verify the indexing order before optimization

If you do not add probes, optimization may produce empty or misindexed
`time_series` data. Recent guardrails reject the most common built-in misuses
before the solver starts.

## 4. NTFF setup rules

For CPML-bounded simulations, the following rules are the practical baseline.

### Rule 1 — keep the NTFF box away from CPML
Use at least:

```python
ntff_margin = (cpml_layers + 3) * dx
```

from **each physical domain edge**.

If the Huygens box overlaps or hugs the CPML region, the far-field result can
collapse toward zero even though the simulation itself still runs.

### Rule 2 — domain size must be large enough
For meaningful NTFF on broadband RF problems, use approximately:

```python
domain_axis >= 1.5 * c0 / freq_min
```

per axis as a starting rule.

If the domain is too small, the fields may not develop adequately before they
reach CPML.

### Rule 3 — prefer float64 for NTFF work
`rfx` can run in lower precision, but NTFF is more reliable in `float64`.
When possible, set:

```bash
export JAX_ENABLE_X64=1
```

especially for antenna / far-field optimization experiments.

## 5. Memory budgeting for inverse design

For gradient runs, **`n_steps` is the primary memory knob**.

Recommended workflow:
1. start with a small forward-valid problem
2. set `n_steps=300` to `500`
3. confirm gradients/loss behave sensibly
4. only then increase `n_steps` for better fidelity

Both `optimize()` and `topology_optimize()` now expose `n_steps` directly so
you can control gradient-run memory at the call site.

`auto_configure(...).estimated_memory_mb` is useful for rough planning, but it
is **not a hard guarantee** for `optimize()` / `topology_optimize()`. XLA
compilation buffers and reverse-mode intermediates can still exceed the estimate
substantially.

## 6. PEC topology caveat

`topology_optimize(material_fg="pec")` is still an active research area in
`rfx`.

Current caveat:
- high-conductivity interpolation can lead to weak or flat gradients
- treat PEC topology optimization as **experimental** until the conductor
  abstraction is improved
- `topology_optimize()` uses a gentler default beta continuation when PEC is in
  the design region, but that reduces risk rather than guaranteeing
  convergence

If you need a stable starting point today, begin with dielectric-vs-dielectric
or dielectric-vs-air topology problems first.

## 7. Recommended progression for RF research use

### Stage A — forward validation
- `Simulation.run()`
- verify resonances / S-parameters / fields

### Stage B — proxy-objective optimization
- add explicit probes
- use `minimize_reflected_energy()` or `maximize_transmitted_energy()`
- keep `n_steps` conservative

### Stage C — far-field workflow
- add NTFF box with CPML clearance
- verify nontrivial NTFF accumulation on the forward run
- only then try directivity-oriented optimization

### Stage D — experimental workflows
- PEC topology optimization
- large CPML inverse-design problems near memory limits
- practical antenna NTFF optimization

Treat Stage D as research work with additional diagnostics, not as turnkey API
usage.
