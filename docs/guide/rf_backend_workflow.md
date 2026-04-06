# RF Backend Workflow

This guide describes the recommended workflow for using `rfx` as an
**agent-friendly RF research backend**.

If you only remember one rule, use this one:

> **forward validation first, optimization second, experimental workflows last**

## Canonical supported workflow

### Step 1 — choose a supported lane
Use the readiness matrix in the README / cookbook:
- **supported**: forward runs, S-parameter analysis, small proxy-objective optimization
- **supported with guardrails**: proxy-objective inverse design, dielectric topology
- **experimental**: PEC topology, NTFF-driven optimization on practical antennas

### Step 2 — run structured preflight
Before launching an optimizer run:

```python
report = sim.preflight_optimize(region, objective, n_steps=500)
print(report.summary())
```

For autonomous or CI-like runs:

```python
result = optimize(..., preflight_mode="strict")
```

When you need explicit protection against supported-safe optimizer blowups, add
a compile-memory budget:

```python
result = optimize(..., preflight_mode="strict", memory_budget_mb=18000)
```

This memory budget gate currently applies only to the tagged built-in
supported-safe proxy objectives, which are the recommended lane for agentic use.
If the active
backend does not expose usable compile-memory stats, preflight will mark the
budget gate unavailable instead of silently accepting it.

Use `guided` mode for exploratory human work, and `strict` mode for agentic
execution that should stop on warnings.

In addition to objective compatibility, preflight now checks a bounded
physics-aware rule set for common setup mistakes such as:
- probes / ports too close to CPML-backed boundaries
- under-resolved thin dielectric layers
- zero-thickness PEC on non-uniform meshes
- explicit NTFF usage with PEC boundaries

That includes experimental lanes: in strict mode, workflows like NTFF/directivity
optimization are intentionally blocked by preflight warnings instead of being
treated as production-safe.

### Step 3 — verify the forward problem first
Before any optimization, confirm that the forward problem behaves sensibly:
- probe signals are nontrivial
- resonances / spectra are plausible
- S-parameters exist where expected
- NTFF accumulation is nontrivial if you plan to use far-field logic later

If the forward problem is not believable, optimization usually just magnifies a
setup mistake.

### Step 4 — start with the safe optimization path
For the most reliable current workflow:
- use **explicit probes**
- use a **built-in proxy objective**
- choose **conservative `n_steps`**
- keep the problem small first

Recommended first choices:
- `maximize_transmitted_energy()`
- `minimize_reflected_energy()`

### Step 5 — escalate only after success at the lower level
Order of escalation:
1. forward simulation
2. proxy-objective optimization
3. dielectric topology optimization
4. NTFF-assisted workflows
5. practical antenna NTFF optimization / PEC topology research

## Canonical example

Use this example as the current recommended starting point:

- `examples/09_agent_friendly_workflow.py`

It demonstrates:
- structured preflight
- a forward physics anchor (uniform slab sweep)
- forward validation
- strict-mode optimization
- a built-in proxy objective in a supported-safe lane

## Why issue #13 and #17 are different

### Issue #13
Issue #13 is the **broader optimizer/backend issue**.
It covers:
- optimizer memory explosion
- PEC topology flat/zero-gradient behavior
- NTFF behavior inside optimization paths

In short, #13 is about the **differentiable backend contract** and whether
large inverse-design workflows are robust.

### Issue #17
Issue #17 is the **narrower practical NTFF validation issue**.
It asks why a practical patch-antenna forward setup can still produce near-zero
far-field even when it appears similar to the dipole example.

In short, #17 is about **forward NTFF reliability on realistic antenna setups**.

### Relationship
- **#13** = systemic optimizer/differentiable backend blocker
- **#17** = narrower forward far-field validation/problem characterization issue

They overlap in symptoms, but they are not the same closure target.

## Recommended closure policy

- Close workflow-hardening issues like #18 when guidance, preflight, and safe
  workflow examples exist.
- Keep solver-blocker issues like #13 and validation-specific issues like #17
  open until their backend behavior is genuinely resolved.

## Maintainer validation harnesses

For issue-13-style runtime validation beyond unit tests:

- `scripts/supported_lane_memory_sweep.py` — compile-memory scaling sweep
- `scripts/issue13_runtime_validation.py` — preflight + one-step runtime check
- `examples/vessl_issue13_runtime_validation.yaml` — GPU/VESSL execution entry
- `scripts/vessl_cli_compat.py` — local compatibility wrapper for the installed VESSL CLI on modern NumPy
