# Comparison with Other FDTD Tools

rfx exists in an ecosystem of excellent FDTD simulators. Each tool makes
different design trade-offs suited to different communities. This page helps
you decide which tool fits your workflow, and acknowledges the prior art that
rfx builds upon.

---

## Acknowledgements

rfx would not exist without the decades of open-source FDTD research that
came before it:

- **Meep** (MIT) established the gold standard for open-source FDTD, proving
  that a well-designed community tool can rival commercial codes. Its
  eigenmode solver, adjoint optimization framework, and extensive validation
  suite set the bar for every project that followed.
- **OpenEMS** brought robust FDTD simulation to the RF and antenna community
  with a mature C++ engine, rich geometry primitives, and deep MATLAB/Octave
  integration. Many of rfx's RF-oriented design choices were informed by
  OpenEMS workflows.
- **FDTDX** demonstrated that JAX-native FDTD can scale to billion-cell
  photonic structures with memory-efficient time-reversal differentiation,
  showing what is possible when modern ML frameworks meet electromagnetics.
- **Commercial tools** (Tidy3D, CST, HFSS, XFdtd) continue to push
  performance and usability boundaries, providing the industry-grade
  validation references that open-source tools benchmark against.

---

## Design Philosophy

Each tool reflects the priorities of its primary user community:

| Tool | Primary audience | Core philosophy |
|------|-----------------|-----------------|
| **Meep** | Photonics researchers | General-purpose FDTD with broad physics coverage |
| **OpenEMS** | RF/antenna engineers | Practical RF simulation with MATLAB integration |
| **FDTDX** | Photonic inverse design | Memory-efficient large-scale JAX differentiation |
| **rfx** | RF/microwave inverse design | Differentiable RF simulation with lossy materials |

rfx focuses specifically on the intersection of RF/microwave engineering and
gradient-based optimization. Unlike Meep and OpenEMS, which target broad
simulation needs, rfx was designed from the ground up around `jax.grad` and
lossy material support. Unlike FDTDX, which targets large-scale photonic
structures, rfx prioritizes RF-specific features like lumped ports, waveguide
modal excitation, and dispersive substrate models common in PCB and antenna
work.

---

## Feature Comparison

| Feature | Meep | OpenEMS | FDTDX | rfx |
|---------|------|---------|-------|-----|
| **License** | GPL-2.0 | GPL-3.0 | MIT | MIT |
| **Language** | C++ / Python | C++ / MATLAB | Python / JAX | Python / JAX |
| **GPU acceleration** | -- | -- | JAX (multi-GPU) | JAX (multi-GPU) |
| **Autodiff** | Adjoint (selected objectives) | -- | Time-reversal | `jax.grad` (checkpointed) |
| **Target domain** | Photonics / general | RF / antenna | Photonics | RF / microwave |
| **Lossy media + gradients** | Partial | Simulation only | Limited (lossless focus) | Full support |
| **Dispersive models** | Lorentz, Drude, custom | Debye, Lorentz, Drude | -- | Debye, Lorentz, Drude |
| **Lumped ports** | -- | Yes | -- | Yes |
| **Waveguide ports** | Eigenmode source | Yes | -- | Modal S-matrix extraction |
| **Lumped RLC elements** | -- | Yes | -- | ADE-based series/parallel |
| **Non-uniform mesh** | Subpixel smoothing | Graded mesh | Uniform | Non-uniform Yee mesh |
| **Subgridding** | -- | -- | -- | SBP-SAT |
| **PML** | Multilayer PML | PML / MUR | PML | CFS-CPML |
| **Near-to-far field** | Built-in | Built-in | -- | Built-in |
| **Topology optimization** | Via adjoint | -- | Via time-reversal | Density-based with `jax.grad` |
| **Community size** | Large (est. 2006) | Large (est. 2010) | Growing (est. 2024) | Early (est. 2026) |

*Entries marked "--" indicate the feature is not natively supported at the
time of writing. Capabilities may have been added since this comparison was
last updated.*

---

## Shared Strengths

All four open-source tools share important qualities:

- **Free and open-source** -- anyone can inspect, modify, and redistribute
  the code.
- **FDTD fundamentals** -- Yee grid, explicit time-stepping, broadband
  results from a single run.
- **PML absorbing boundaries** -- all tools implement some form of perfectly
  matched layers.
- **Active development** -- each project has maintainers who respond to
  issues and improve the code.

---

## When to Use Which

Choosing the right tool depends on your problem, your existing workflow, and
your team's expertise:

### Meep

Meep is the natural choice when you need a **battle-tested, general-purpose
FDTD** with decades of community validation. It excels at photonic crystal
simulations, eigenmode analysis, and problems where its rich physics library
(nonlinear materials, custom sources, mode decomposition) saves significant
development time. If your team already uses Meep, there is rarely a reason to
switch for pure forward simulation.

### OpenEMS

OpenEMS is ideal for **practical RF and antenna design** where you want a
mature, well-documented workflow with extensive example coverage. Its MATLAB
integration makes it popular in academic RF labs, and its geometry engine
handles complex antenna structures naturally. For teams comfortable with
MATLAB/Octave, OpenEMS offers the shortest path from concept to validated
S-parameter results.

### FDTDX

FDTDX is the tool to reach for when you need **JAX-native differentiation at
very large scale** -- photonic structures with billions of cells where
memory-efficient time-reversal gradients are essential. Its design prioritizes
throughput and memory efficiency for the photonic inverse design community.

### rfx

rfx is designed for **RF/microwave inverse design** workflows where you need:

- `jax.grad` through lossy, dispersive materials (FR4, Rogers substrates,
  copper with skin effect)
- Lumped and waveguide port S-parameters as differentiable objectives
- Non-uniform meshing for thin PCB substrates
- A pure-Python stack that integrates with JAX-based ML pipelines

If your problem is "optimize this PCB antenna's S11 using gradient descent on
a GPU," rfx was built for exactly that workflow.

---

## Limitations and Honest Caveats

rfx is a young project. In the interest of transparency:

- **Community size**: Meep and OpenEMS have thousands of users and years of
  edge-case fixes. rfx has been validated against analytical benchmarks and
  cross-checked with both tools, but it has not yet seen the breadth of
  real-world use that builds deep confidence.
- **Physics coverage**: Meep supports nonlinear materials, Kerr media,
  cylindrical coordinates, and many source types that rfx does not yet offer.
- **Geometry engine**: OpenEMS has a sophisticated geometry engine with CSG,
  curves, and import from CAD tools. rfx currently provides basic box, via,
  and curved-patch primitives.
- **Documentation and examples**: Meep's documentation (and the photonics
  tutorials by its community) is extensive. rfx's docs are growing but not
  yet at that level.
- **Stability**: Commercial tools like CST and HFSS have decades of
  engineering behind their solvers. rfx targets a different niche (open,
  differentiable, GPU-accelerated), but it does not replace a commercial
  solver for certification-grade work.

---

## Migration Resources

If you are coming from Meep or OpenEMS, the
[Migration Guide](migration.md) maps common workflow patterns into rfx
equivalents.

---

*Last updated: 2026-04-03. Feature comparisons reflect publicly documented
capabilities at the time of writing. If you notice an inaccuracy, please open
an issue.*
