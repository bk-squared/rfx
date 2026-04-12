# Auto-Subgrid Design Spec

**Date**: 2026-04-12
**Project**: rfx FDTD simulator
**Author**: Byungkwan Kim + AI Agent
**Status**: Approved

## Motivation

rfx needs automatic mesh refinement so that:
1. Users don't manually guess where fine mesh is needed
2. LLM agents can programmatically inspect refinement plans and decide whether to apply them
3. Simulation accuracy is validated, not assumed

Two design pillars of rfx drive this: **autograd** (differentiable simulation) and **RF design with LLM agents**. The auto-subgrid system must be both human-inspectable (visualizations) and machine-readable (structured JSON specs).

## Roadmap: C → A → B

### Phase C: 3D SBP-SAT Stabilization + Validation

**Goal**: Make 3D subgridding trustworthy before building automation on top.

**Scope**:

1. **Implement Cheng et al. 2025 penalty coefficients**
   - File: `rfx/subgridding/sbp_sat_3d.py`
   - Current: `tau` is a user parameter with default 0.5, applied uniformly
   - Target: Derive energy-stable alpha from Cheng et al. IEEE TAP (DOI: 10836194) equations for 3D Yee grid SBP-SAT coupling
   - Change: `_shared_node_coupling_3d()` — replace ad-hoc `tau * ratio / (ratio + 1)` with paper-derived coefficients
   - Preserve: `tau` parameter as a user-facing scaling factor (1.0 = paper default)

2. **Stability validation**
   - Test: empty PEC cavity, Gaussian pulse, 100k steps
   - Criterion: total energy is non-increasing (monotone) to within float32 tolerance (~1e-6 relative growth per step)
   - Test: CPML domain, 50k steps, energy decays (no late-time growth)

3. **Accuracy validation (crossval-grade)**
   - Use crossval 04 (multilayer Fresnel slab) as benchmark
   - Run 1: uniform fine grid (dx = dx_coarse / ratio)
   - Run 2: coarse grid + subgridded slab region
   - Criterion: R(f) and T(f) mean error < 5% between subgridded and uniform-fine
   - Produces PNG with side-by-side comparison (geometry + R/T curves)

4. **Test updates**
   - Remove energy growth warning from `test_sbp_sat_3d.py`
   - Add quantitative energy conservation test (non-increasing)
   - Add accuracy crossval test (vs uniform fine)

**Exit criteria**: 3D SBP-SAT passes energy conservation AND <5% accuracy vs uniform fine on Fresnel slab.

---

### Phase A: RefinePlan Pipeline

**Goal**: Post-hoc diagnostic + one-shot refinement with human/agent-readable output.

#### A.1 Indicator System

**Architecture**: Plugin-style indicators, each independent. Two strategies for combining.

```python
# Base interface
class RefinementIndicator:
    name: str
    def evaluate(self, sim, result) -> IndicatorResult:
        """Return (error_map: ndarray, regions: List[RefineRegion])"""
        ...
```

**Built-in indicators**:

| Indicator | Source | What it detects |
|-----------|--------|-----------------|
| `GradientIndicator` | `result.state` | High field gradients (under-resolved features) |
| `MaterialBoundaryIndicator` | `sim._geometry` | Dielectric interfaces (subpixel accuracy needed) |
| `PMLProximityIndicator` | `sim._cpml_layers` | Structures near PML (absorption artifacts) |
| `SourceVicinityIndicator` | `sim._sources, sim._ports` | Near-field around sources/ports |

Each indicator returns regions with:
```python
class RefineRegion:
    box: Box                    # physical coordinates (x_lo, y_lo, z_lo, x_hi, y_hi, z_hi)
    ratio: int                  # suggested refinement ratio (per-region)
    reason: str                 # indicator name
    max_error: float            # peak error in this region
    confidence: float           # 0-1, how certain the indicator is
```

**Combining strategies**:

- `strategy="rule_based"` (default): Each indicator runs independently. Results are unioned. Overlapping regions merged by taking the larger box and higher ratio. Reason field lists all contributing indicators.
- `strategy="composite"`: Indicators produce normalized score maps [0,1]. Weighted sum: `score = sum(w_i * indicator_i)`. Regions extracted by thresholding composite score. Weights configurable via `weights={"gradient": 1.0, "material": 0.5, ...}`.

**Merge logic**:
- Overlapping boxes: union bounding box, max ratio, concatenated reasons
- Regions < `min_cells` (default 4^3 = 64): dropped
- Total refined cells > `max_fraction` (default 0.3) of domain: return `{"recommendation": "use_uniform_fine"}` instead of regions

#### A.2 RefinePlan Object

```python
class RefinePlan:
    regions: List[RefineRegion]
    error_maps: Dict[str, ndarray]   # per-indicator error maps
    grid: Grid
    strategy: str
    
    def report(self, path=None):
        """Human-readable: PNG with eps_r + error overlay + region boxes.
        Prints text summary. Saves PNG if path given."""
    
    def to_spec(self) -> dict:
        """Agent-readable: JSON-serializable structured output."""
        return {
            "status": "refinement_suggested" | "uniform_fine_recommended" | "no_refinement_needed",
            "regions": [
                {
                    "corner_lo": [x, y, z],
                    "corner_hi": [x, y, z],
                    "ratio": 3,
                    "reason": ["gradient", "material_boundary"],
                    "max_error": 0.42,
                    "confidence": 0.85
                }, ...
            ],
            "metrics": {
                "max_error": 0.42,
                "mean_error": 0.08,
                "cells_to_refine": 1200,
                "total_cells": 50000,
                "memory_increase_estimate": 2.1,
                "indicators_triggered": ["gradient", "material_boundary"]
            },
            "action": "sim.run_refined(plan)"
        }
```

#### A.3 API

```python
# Step 1: Run coarse simulation
result = sim.run(n_steps=1000)

# Step 2: Get refinement plan
plan = sim.suggest_refinement(
    result,
    indicators=["gradient", "material_boundary", "source_vicinity"],  # or "all"
    strategy="rule_based",          # or "composite"
    threshold=0.3,                  # error threshold for region detection
    min_ratio=2,                    # minimum refinement ratio
    max_ratio=5,                    # maximum refinement ratio
)

# Step 3: Inspect
plan.report(path="refinement_plan.png")   # human
spec = plan.to_spec()                      # agent

# Step 4: Execute
result_refined = sim.run_refined(plan)
```

#### A.4 API Extension: Multi-Region Refinement

Current `add_refinement(z_range=...)` replaced with:

```python
sim.add_refinement(
    region=Box(corner_lo, corner_hi),   # 3D box, not z-only
    ratio=3,
    tau=0.5,
)
# Can be called multiple times for multiple regions
```

Implementation:
- `sim._refinements: List[RefinementSpec]` (was single `_refinement`)
- Runner iterates regions, creates SubgridConfig3D for each
- Non-overlapping regions enforced (raise ValueError if overlap detected)
- Nested refinement (subgrid inside subgrid) deferred to Phase B

#### A.5 Visualization (`plan.report()`)

2x2 figure:
- Top-left: eps_r geometry map with refinement region boxes overlaid
- Top-right: composite error map (or dominant indicator map)
- Bottom-left: per-indicator breakdown (bar chart or small multiples)
- Bottom-right: text summary (region count, cells, memory estimate, recommendations)

---

### Phase B: Automatic Convergence (Future)

**Goal**: `sim.run(auto_refine=True)` — iterative refinement until convergence.

**Sketch** (not fully designed — depends on Phase A validation):

```python
result = sim.run(
    n_steps=1000,
    auto_refine=True,
    convergence_target=0.05,     # max acceptable error
    max_refinement_levels=3,     # prevent infinite recursion
    max_memory_factor=10.0,      # abort if memory > 10x base
)
```

Internal loop:
1. Run coarse
2. `suggest_refinement()` → plan
3. If `plan.status == "no_refinement_needed"`: done
4. `run_refined(plan)` → result
5. Compare refined vs coarse (probe signals or flux)
6. If change < `convergence_target`: done
7. Else: use refined result as new "coarse", goto 2

Requires:
- Multi-level nested subgrids (subgrid inside subgrid)
- Memory budget tracking
- Convergence metric definition

**Deferred** until Phase A is validated on real problems.

---

## File Changes Summary

### Phase C (new/modified files)
| File | Change |
|------|--------|
| `rfx/subgridding/sbp_sat_3d.py` | Implement Cheng et al. penalty coefficients |
| `tests/test_sbp_sat_3d.py` | Add energy conservation + accuracy crossval tests |
| `tests/test_subgrid_crossval.py` | New: Fresnel slab subgridded vs uniform-fine |

### Phase A (new/modified files)
| File | Change |
|------|--------|
| `rfx/indicators.py` | New: indicator base class + 4 built-in indicators |
| `rfx/refine_plan.py` | New: RefinePlan, RefineRegion, merge logic |
| `rfx/api.py` | Extend `add_refinement()` to 3D box, add `suggest_refinement()`, `run_refined()` |
| `rfx/runners/subgridded.py` | Multi-region support |
| `rfx/amr.py` | Deprecate in favor of `indicators.py` (keep as alias) |
| `tests/test_indicators.py` | New: unit tests for each indicator |
| `tests/test_refine_plan.py` | New: merge logic, report/to_spec output |
| `tests/test_auto_refine_e2e.py` | New: E2E test (coarse → plan → refined) |

## Dependencies

- Phase C has no external dependencies (Cheng et al. paper is the reference)
- Phase A depends on Phase C completion (3D must be stable)
- Phase B depends on Phase A validation
- scipy is optional (for connected-component labeling in region suggestion)

## Success Criteria

### Phase C
- [ ] 3D energy non-increasing over 100k steps (PEC cavity)
- [ ] 3D energy decays in CPML domain (50k steps)
- [ ] Fresnel slab: subgridded R/T within 5% of uniform-fine

### Phase A
- [ ] `suggest_refinement()` returns valid plan for patch antenna geometry
- [ ] `plan.report()` produces informative PNG
- [ ] `plan.to_spec()` returns valid JSON that an LLM agent can parse and act on
- [ ] `run_refined(plan)` produces result with lower error than coarse-only
- [ ] Multi-region refinement works (2+ non-overlapping regions)
- [ ] Per-region ratio works (region A at ratio=2, region B at ratio=3)

### Phase B (future)
- [ ] Iterative refinement converges on waveguide bend problem
- [ ] Memory budget respected
- [ ] Nested subgrids stable
