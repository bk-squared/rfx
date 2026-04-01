# FDTD Subgridding: State-of-the-Art Survey (2024)

## 1. The Problem

Uniform Yee grids waste cells: a 200³ domain at dx=0.5mm for a small antenna
feature means 8M cells, when only the antenna region needs fine resolution.
Subgridding uses fine dx only where needed, with coarse dx elsewhere.

**Challenge**: The coarse-fine interface introduces:
- **Late-time instability** (the dominant problem)
- **Spurious reflections** at the interface
- **Phase error accumulation** from interpolation

## 2. Classical Approaches

### 2.1 Direct Interpolation (Chevalier & Luebbers, 1997)
- Linear interpolation of fields at coarse-fine boundary
- Simple but **unconditionally unstable** in long runs
- Reflection coefficient ~10% at interface
- Abandoned by most researchers

### 2.2 Huygens Subgridding (HSG) (Bérenger, 2000-2006)
- Uses TFSF-like Huygens surfaces as the coarse-fine interface
- Coarse grid drives the fine grid via equivalent currents
- Fine grid drives the coarse grid via anti-Huygens correction
- Supports **arbitrary spatial ratios** (tested up to 99:1)
- **Main weakness**: late-time instability from imperfect cancellation
  between Huygens and anti-Huygens surfaces

### 2.3 Conservative Subgridding (Monk & Süli, 2007)
- Provably stable via energy-conserving discrete operators
- Requires non-standard grid stencils
- Complex implementation, limited practical adoption

## 3. Modern Solutions (2023-2025)

### 3.1 SBP-SAT Method (Cheng et al., J. Comput. Phys., 2023) ★★★

**Most promising for rfx.**

- **Summation-By-Parts (SBP)** operators mimic continuous integration-by-parts
  at the discrete level
- **Simultaneous Approximation Terms (SAT)** weakly enforce interface conditions
- **Provably stable**: energy estimate rigorously proved, guaranteed stable in
  arbitrarily long simulations
- Adds boundary field components to Yee grid for 2nd/4th order SBP properties
- Derived interpolation matrices couple mesh blocks of different sizes
- **Easy to integrate**: only simple modifications to existing FDTD codes
- Extended to 2D with arbitrary grid ratios (IEEE TAP, 2023)
- 3D development underway (Cheng et al., 2024)

**Key insight**: Instead of trying to make interpolation stable (impossible in
general), SBP-SAT reformulates the interface as a penalty term that
*provably* dissipates energy, preventing late-time growth.

**References**:
- [Cheng et al., J. Comput. Phys. 494, 112510, 2023](https://arxiv.org/abs/2110.09054)
- [Cheng et al., IEEE TAP, 2023 — 2D arbitrary ratio](https://ieeexplore.ieee.org/document/10242252/)
- [Cheng et al., IEEE 2024 — arbitrary ratios + material transitions](https://ieeexplore.ieee.org/document/10686530)

### 3.2 Switched Huygens Subgridding (SHSG) (Chilton, 2009)
- Fundamental modification to HSG that switches which surface drives which
- **143x more stable** than standard HSG for 3D dipole problem
- **10x more stable** for 1D resonant problem
- Still not *provably* stable, but practically much better

### 3.3 Iteration-Based Temporal Subgridding (2024)
- Uses iterative updating equations at the temporal coarse-fine interface
- Different from spatial interpolation — converges to consistent solution
- Published in Mathematics, Jan 2024
- [MDPI Mathematics 12(2), 302, 2024](https://www.mdpi.com/2227-7390/12/2/302)

### 3.4 Unconditionally Stable Temporal Decomposition (USTD)
- Uses unconditionally stable (implicit) FDTD in the fine region
- Synchronous updates eliminate temporal interpolation entirely
- Combines arbitrary spatial ratio + zero late-time instability
- Trade-off: implicit solve is more expensive per step

## 4. Competitor: FDTDX (JAX-based FDTD, Dec 2024)

[FDTDX](https://github.com/ymahlau/fdtdx) is an open-source JAX FDTD for
nanophotonics inverse design (Mahlau et al., arXiv:2412.12360).

| Feature | FDTDX | rfx |
|---------|-------|-----|
| Target | Nanophotonics | RF/microwave |
| Autodiff | Time-reversal adjoint | jax.checkpoint reverse-mode |
| Multi-GPU | Yes (4×H100 = 2.3B cells) | Not yet |
| Subgridding | No | Not yet |
| S-parameters | No | Yes (full N-port) |
| Waveguide ports | No | Yes (multi-axis) |
| Performance | 10x faster than Meep (288M cells) | 1,310 Mcells/s (RTX 4090) |

**Key difference**: FDTDX uses time-reversal adjoint (memory-efficient but
limited to linear, lossless media). rfx uses jax.checkpoint reverse-mode AD
(works with any differentiable physics including lossy/dispersive).

**FDTDX does NOT have subgridding.** Neither does Meep or OpenEMS.
Implementing subgridding in rfx would be a unique differentiator.

## 5. Recommendation for rfx

### Approach: SBP-SAT (Cheng et al., 2023)

**Why SBP-SAT over Huygens**:
- **Provably stable** (not just "more stable")
- Simple interface modifications (not a complete rewrite)
- Supports arbitrary grid ratios
- Works with standard Yee grids
- Active research group with 3D extension underway

### Implementation plan

```
Phase 1: 1D SBP-SAT prototype (validate stability)
  - Two 1D Yee grids with different dx
  - SBP operators at interface
  - SAT penalty terms
  - Run for 10⁶ steps, verify no growth

Phase 2: 2D TM extension
  - Apply to rfx 2D mode
  - Validate against uniform fine grid
  - Measure reflection at interface

Phase 3: 3D integration with jax.lax.scan
  - Two scan bodies (coarse + fine) coupled via SAT
  - Verify jax.grad flows through interface
  - Benchmark memory savings

Phase 4: High-level API
  - Simulation.add_refinement_region(box, ratio=3)
  - Auto-generate SBP operators for the given ratio
```

### JAX compatibility assessment

- **SBP operators**: sparse matrices → `jax.experimental.sparse` or dense for small interfaces
- **SAT penalty**: simple field-dependent correction → fully differentiable
- **Two-grid scan**: can use nested `jax.lax.scan` or single scan with carry for both grids
- **Gradient flow**: SAT terms are differentiable (linear operations on fields)
- **Conclusion**: SBP-SAT is JAX-compatible and should be differentiable

### When to use subgridding (decision guide)

| Scenario | Subgridding worth it? |
|----------|----------------------|
| Small feature in large domain (wire antenna) | **Yes** — 10-100x memory saving |
| Uniform medium with boundary detail | **Yes** — fine grid only at boundaries |
| Whole domain needs fine resolution | **No** — just use fine uniform |
| Already fits in GPU memory | **No** — overhead not worth it |
| Need > 200³ on single GPU | **Yes** — critical enabler |

## 6. Latest Updates (2025-2026)

### 3D SBP-SAT now available
Cheng et al. published the **3D extension** of SBP-SAT FDTD, theoretically
stable with multiple mesh blocks of different sizes. Also a variant using
staggered Yee grids **without modifying field components** (via boundary
extrapolation to satisfy SBP). This significantly simplifies integration
into existing Yee-based codes like rfx.

### Time-reversal AD (ACS Photonics, 2024)
98% memory reduction vs standard AD by recording fields only at lossy
boundaries and replaying time-reversed FDTD. Relevant for rfx: could
complement jax.checkpoint for even larger simulations with subgridding.

### GPU-native AMR (CFD, 2025-2026)
Block-based octree AMR running entirely on GPU (no CPU-GPU transfers)
is now standard in CFD (lattice Boltzmann, compressible flows). The
pattern — forest of octrees with CUDA kernels — could be adapted for
FDTD subgridding on GPU. No FDTD-specific GPU AMR published yet.

### No one else has differentiable subgridding
Neither FDTDX, Meep, OpenEMS, nor any published framework combines
differentiable FDTD with subgridding. **rfx implementing SBP-SAT +
jax.grad would be a genuine first.**

## Sources

- [SBP-SAT FDTD Subgridding (Cheng et al., 2023)](https://arxiv.org/abs/2110.09054)
- [2D SBP-SAT with Arbitrary Grid Ratio (IEEE TAP, 2023)](https://ieeexplore.ieee.org/document/10242252/)
- [Stable Subgridding with Material Transitions (IEEE, 2024)](https://ieeexplore.ieee.org/document/10686530)
- [Huygens Subgridding (Bérenger, 2006)](https://ui.adsabs.harvard.edu/abs/2006ITAP...54.3797B/abstract)
- [Switched Huygens Subgridding](https://www.research.ed.ac.uk/en/publications/switched-huygens-subgridding-for-the-fdtd-method)
- [Iteration-Based Temporal Subgridding (2024)](https://www.mdpi.com/2227-7390/12/2/302)
- [Conservative FDTD Subgridding (Monk & Süli)](https://ieeexplore.ieee.org/document/4298171/)
- [FDTDX: JAX FDTD for Nanophotonics](https://github.com/ymahlau/fdtdx)
- [FDTDX Paper (arXiv:2412.12360)](https://arxiv.org/abs/2412.12360)
