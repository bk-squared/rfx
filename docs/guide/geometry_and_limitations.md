# Geometry Definition and Simulation Limitations

This guide covers the geometry primitives available in rfx, how to use them,
what rfx is well-suited for, what it cannot do well, how to choose the right
simulation setup, and how rfx compares to other electromagnetic solvers.

---

## 1. Geometry Definition

### 1.1 Coordinate System

rfx uses a right-handed Cartesian coordinate system with coordinates
`(x, y, z)` specified in **metres**. The origin sits at the corner of the
computational domain — not at the centre. When you create a `Simulation` with
`domain=(0.05, 0.05, 0.025)`, the physical domain spans from `(0, 0, 0)` to
`(0.05, 0.05, 0.025)` metres. All geometry positions, port locations, and
probe positions use this absolute coordinate frame.

### 1.2 Available Primitives

rfx provides three core CSG primitives in `rfx.geometry.csg`: **Box**,
**Sphere**, and **Cylinder**. It also provides higher-level geometry helpers
`Via` and `CurvedPatch` in `rfx.geometry`. All five shapes now work on all
runner paths: uniform, nonuniform, subgridded, and distributed.

All shapes implement a unified rasterization protocol based on
`Shape.mask_on_coords(x, y, z)` and `Shape.bounding_box()`. The grid-facing
`mask(grid)` entry point remains available and delegates to
`mask_on_coords()`, while geometry utilities such as `auto_configure()` use
`bounding_box()` for extent detection.

#### Box

An axis-aligned rectangular parallelepiped defined by two opposite corners.

```python
from rfx.geometry.csg import Box

# A 5 cm × 5 cm × 1 mm dielectric slab at the bottom of the domain
slab = Box(corner_lo=(0, 0, 0), corner_hi=(0.05, 0.05, 0.001))

# A small cube (1 cm side) centred at (2.5 cm, 2.5 cm, 1.25 cm)
cube = Box(corner_lo=(0.02, 0.02, 0.007), corner_hi=(0.03, 0.03, 0.018))
```

#### Sphere

Defined by a centre point and a radius.

```python
from rfx.geometry.csg import Sphere

# A 5 mm radius sphere at the centre of a 4 cm cubic domain
sphere = Sphere(center=(0.02, 0.02, 0.02), radius=0.005)
```

#### Cylinder

Defined by a centre, radius, height, and axis of alignment (`"x"`, `"y"`, or
`"z"`). The height is measured symmetrically about the centre along the
specified axis.

```python
from rfx.geometry.csg import Cylinder

# A z-oriented cylinder: radius 3 mm, height 10 mm, centred at domain centre
cyl = Cylinder(center=(0.02, 0.02, 0.02), radius=0.003, height=0.01, axis="z")

# An x-oriented cylinder for a via or pin
via = Cylinder(center=(0.01, 0.01, 0.005), radius=0.0005, height=0.001, axis="x")
```

#### Via

`Via` is a plated-through-hole helper that participates in the same shape
protocol as the core CSG primitives. Its rasterization delegates to an
internal Box decomposition, so it works on every runner path.

#### CurvedPatch

`CurvedPatch` is a higher-level curved conductor helper. Under the unified
shape protocol it also works on every runner path, using a staircase
decomposition for rasterization.

> **Implementation note:** `Box` retains an optimized slice-based path on
> nonuniform and subgridded runners. Other shapes rasterize through
> `mask_on_coords()` on the runner-provided coordinate arrays.

### 1.3 Boolean CSG Operations

The module provides three boolean operations. Each takes two shapes and a
`grid`, returning a combined boolean mask:

```python
from rfx.geometry.csg import Box, Cylinder, union, difference, intersection

grid = sim._build_grid()  # internal; normally the API handles this

# Union: waveguide body with a flange
body = Box(corner_lo=(0, 0.005, 0.005), corner_hi=(0.05, 0.015, 0.015))
flange = Box(corner_lo=(0, 0, 0), corner_hi=(0.002, 0.02, 0.02))
combined_mask = union(body, flange, grid)

# Difference: box with a cylindrical hole drilled through it
block = Box(corner_lo=(0.01, 0.01, 0.01), corner_hi=(0.03, 0.03, 0.03))
hole = Cylinder(center=(0.02, 0.02, 0.02), radius=0.003, height=0.025, axis="z")
drilled_mask = difference(block, hole, grid)

# Intersection: only the overlap region of two shapes
mask = intersection(body, flange, grid)
```

> **Note:** The boolean functions return raw masks on the grid. For typical
> workflows you do not call these directly — you add shapes via `Simulation.add()`
> and the rasterizer applies them in painter-order (last shape wins where they
> overlap). Use `rasterize()` for lower-level control.

### 1.4 Material Assignment

#### Registering Materials

Materials are registered by name with `Simulation.add_material()`. rfx also
ships a built-in library (vacuum, air, fr4, rogers4003c, alumina, silicon,
ptfe, copper, aluminum, pec, water\_20c) that can be used by name without
prior registration.

```python
from rfx.api import Simulation

sim = Simulation(freq_max=10e9, domain=(0.05, 0.05, 0.025))

# Register a custom substrate
sim.add_material("my_substrate", eps_r=3.2, sigma=0.001)

# Use built-in library materials directly — no registration needed
sim.add(Box((0, 0, 0), (0.05, 0.05, 0.001)), material="fr4")
sim.add(Box((0, 0, 0.001), (0.05, 0.05, 0.00103)), material="copper")
```

Materials support optional Debye and Lorentz dispersion poles for
frequency-dependent permittivity:

```python
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole

sim.add_material("water", eps_r=4.9, sigma=0.0,
                 debye_poles=[DebyePole(delta_eps=74.1, tau=8.3e-12)])

sim.add_material("gold_drude", eps_r=1.0, sigma=0.0,
                 lorentz_poles=[LorentzPole(delta_eps=-1e4, omega0=0, gamma=4.08e13)])
```

#### Assigning Materials to Geometry

Use `Simulation.add(shape, material=name)` to place geometry:

```python
sim.add(Box((0, 0, 0), (0.05, 0.05, 0.001)), material="my_substrate")
sim.add(Sphere((0.025, 0.025, 0.005), radius=0.003), material="alumina")
```

Shapes are rasterized in the order they are added — **later shapes overwrite
earlier ones** in overlapping regions (painter's algorithm). This is how you
build composite structures such as a dielectric block with a drilled-out air
hole: add the block first, then add the hole with `material="vacuum"`.

### 1.5 Grid Snapping (Stairstepping)

All geometry is rasterized onto the active Yee grid used by the selected
runner. A cell is either fully inside or fully outside a shape — there is no
partial cell assignment in the default mode. This produces **staircase
approximations** of curved and angled surfaces. The staircasing error scales
as O(dx) and is the dominant source of error for smooth geometries on coarse
grids.

### 1.6 Subpixel Smoothing

rfx implements **anisotropic subpixel smoothing** at dielectric interfaces to
reduce stairstepping error. When enabled (`subpixel_smoothing=True` in
`sim.run()`), the solver computes a signed distance function (SDF) for each
supported shape (Box, Sphere, Cylinder) and assigns effective per-component
permittivity at boundary voxels:

- **Parallel** to the interface: arithmetic mean — `eps_par = f * eps1 + (1 - f) * eps2`
- **Perpendicular** to the interface: harmonic mean — `eps_perp = [f/eps1 + (1 - f)/eps2]^{-1}`

where `f` is the volumetric filling fraction estimated from the SDF. This
yields **second-order convergence** instead of first-order stairstepping.

```python
result = sim.run(subpixel_smoothing=True)
```

> **Important:** Subpixel smoothing applies only to **dielectric** interfaces.
> It does not help with PEC (perfect electric conductor) stairstepping, because
> PEC is enforced as a field zeroing condition, not via permittivity.

---

## 2. What rfx Is Good For

rfx is a JAX-based FDTD solver designed for **differentiable electromagnetic
simulation**. Its sweet spot is small- to medium-scale problems where
automatic differentiation via `jax.grad` is the primary value proposition.

### 2.1 Rectangular Waveguide Filter / Coupler Design

rfx has first-class waveguide port support with TE/TM modal excitation and
S-parameter extraction. Axis-aligned rectangular waveguides map perfectly to
the Yee grid (no stairstepping), making this the highest-accuracy use case.

```python
sim = Simulation(freq_max=15e9, domain=(0.08, 0.03, 0.015), boundary="cpml")
sim.add(Box((0, 0.005, 0), (0.08, 0.025, 0.01)), material="pec")
sim.add_waveguide_port(x_position=0.005, direction="+x", mode=(1, 0), mode_type="TE")
sim.add_waveguide_port(x_position=0.075, direction="-x", mode=(1, 0), mode_type="TE")
```

### 2.2 Lumped-Port Antenna Impedance Matching

Lumped ports with configurable impedance and Gaussian pulse excitation allow
rapid impedance sweep studies. The S-parameter extraction gives reflection
coefficient (S11) directly.

```python
sim.add_port((0.025, 0.025, 0.001), "ez", impedance=50,
             waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
result = sim.run(compute_s_params=True)
# result.s_params[0, 0, :] is S11 vs frequency
```

### 2.3 Dielectric Structure Inverse Design

Because the entire simulation loop is written in JAX, you can compute
`jax.grad` of any scalar loss through the simulation — e.g., minimize
|S11|^2 at a target frequency by optimizing permittivity distributions.

```python
import jax

def loss_fn(eps_array):
    # Build sim with parameterised eps, run, return |S11|^2
    ...

grad_eps = jax.grad(loss_fn)(initial_eps)
```

### 2.4 PEC Cavity Resonance Analysis

Closed PEC cavities with `boundary="pec"` give analytical-quality resonance
frequencies. The Stage 1 validation achieved **0.22% error** on a rectangular
cavity, confirming the core solver accuracy.

### 2.5 Normal-Incidence RCS

The TFSF (Total-Field/Scattered-Field) plane-wave source supports
normal-incidence scattering. Combined with the NTFF (Near-to-Far-Field)
transform, you can extract RCS of plates and spheres.

```python
sim.add_tfsf_source(polarization="ez", direction="+x")
sim.add_ntff_box(corner_lo=(0.01, 0.01, 0.01), corner_hi=(0.04, 0.04, 0.04))
result = sim.run()
ff = compute_far_field(result.ntff_data, result.ntff_box, ...)
```

### 2.6 Waveguide S-Parameter Extraction

Single-mode TE/TM S-parameter extraction uses modal decomposition at
waveguide port planes, with optional calibration presets for reference-plane
de-embedding.

### 2.7 Microstrip / Stripline on Planar Substrates

Planar transmission-line structures align naturally with the Cartesian grid.
Use 2D mode (`mode="2d_tmz"` or `mode="2d_tez"`) for fast cross-section
analysis, or thin 3D domains for full-wave modelling.

```python
sim = Simulation(freq_max=20e9, domain=(0.04, 0.02, 0.003), mode="3d")
sim.add(Box((0, 0, 0), (0.04, 0.02, 0.0008)), material="fr4")      # substrate
sim.add(Box((0, 0.008, 0.0008), (0.04, 0.012, 0.00083)), material="copper")  # trace
sim.add(Box((0, 0, -0.0001), (0.04, 0.02, 0.0)), material="pec")   # ground
```

---

## 3. What rfx Cannot Do Well

### 3.1 Curved Surfaces (Stairstepping)

rfx uses Cartesian Yee grids, including uniform, nonuniform, and subgridded
runner paths. Curved PEC surfaces still suffer from staircase approximation
with O(dx) error, and there is no conformal mesh. Subpixel smoothing
mitigates this for **dielectric** interfaces (second-order convergence), but
**PEC curves** remain staircased. For accurate PEC cylinder/sphere
scattering, you need a very fine grid or a solver with conformal meshing.

### 3.2 Oblique Incidence Scattering

The TFSF implementation currently supports only x-directed propagation with
ez/ey polarization. Oblique incidence beyond about 10 degrees introduces TFSF
leakage artefacts. This is under active development.

### 3.3 Very Large Domains (> 200^3 Cells)

rfx runs on a single GPU with no MPI domain decomposition. A 200^3 grid
consumes roughly 6 GB of GPU memory for the six field components plus
material arrays. Beyond this, you will hit GPU memory limits on consumer-grade
GPUs (24 GB). For domains requiring millions of cells, use a distributed
solver.

### 3.4 Thin Wires and Filaments

Structures much thinner than the cell size (e.g., wire antennas) need subcell
wire models (Holland's thin-wire formalism). rfx has a `ThinConductor` model
for planar conductors (sheet resistance correction), but does not have a
general 1D thin-wire subcell model. Resolving thin wires by brute-force mesh
refinement is impractical.

### 3.5 Frequency-Dependent Port Impedance

Lumped ports use a fixed real impedance Z0 (default 50 ohms). There are no
Floquet ports, no complex-impedance ports, and no frequency-dependent port
models. For waveguide ports, the modal impedance is computed analytically for
the dominant mode, but higher-order mode port impedance is not supported.

### 3.6 Photonic Crystal Band Structure

rfx has no eigenmode solver or Bloch-periodic boundary conditions for band
structure computation. It is a time-domain transient solver. While you can
extract resonance peaks from transient ringdown, this is far less efficient
than a dedicated eigenmode solver (e.g., MPB).

### 3.7 Nonlinear Materials

Only linear isotropic dielectric and magnetic materials are supported, with
optional Debye and Lorentz frequency dispersion. Nonlinear constitutive
relations (Kerr, saturable absorbers, ferrites with hysteresis) are not
implemented.

### 3.8 Multi-Physics Coupling

rfx is a pure electromagnetic solver. There is no thermal solver, no
mechanical stress solver, and no fluid coupling. If your problem requires
coupled EM-thermal analysis (e.g., microwave heating), you need an external
coupling loop or a multi-physics platform.

### 3.9 Broadband CW Steady-State

For narrow-band CW excitation, the FDTD time-marching must run long enough
for transients to decay. The `until_decay` parameter helps
(`sim.run(until_decay=1e-4)`), but convergence can be very slow for
high-Q resonators. In such cases, prefer a frequency-domain solver or use
the field decay criterion with a generous step budget.

---

## 4. Choosing the Right Simulation Setup

### 4.1 Boundary Conditions: CPML vs PEC

| Use case | Boundary | Why |
|---|---|---|
| Open-region scattering, antennas | `boundary="cpml"` | Absorbs outgoing waves; simulates infinite free space |
| Closed metallic cavity | `boundary="pec"` | Perfect reflector walls; no absorbing layers needed |
| Waveguide with ports | `boundary="cpml"` | Ports require CPML on the propagation axis |
| Resonance of a closed structure | `boundary="pec"` | Exact boundary condition, no CPML reflection artefacts |

CPML (Complex Frequency-Shifted Perfectly Matched Layer) adds absorbing layers
around the domain. The default 10 layers per face is adequate for most cases.
PEC boundaries use zero tangential E-field and add no extra cells.

### 4.2 2D vs 3D Mode

Use `mode="2d_tmz"` (Ez, Hx, Hy) or `mode="2d_tez"` (Hz, Ex, Ey) when:

- The problem is uniform along one axis (e.g., waveguide cross-section analysis).
- You want fast iteration during design exploration.
- Memory is limited and the 3D domain would be too large.

Use `mode="3d"` when:

- The structure varies in all three dimensions.
- You need full vector field interactions (e.g., patch antenna radiation).
- Accurate S-parameter extraction requires 3D modal decomposition.

### 4.3 Choosing Cell Size (dx)

The standard rule of thumb is:

```
dx <= lambda_min / 20
```

where `lambda_min = c / freq_max` is the shortest wavelength in the simulation
(in free space). In dielectric media, the wavelength is shorter by a factor of
`sqrt(eps_r)`, so:

```
dx <= c / (freq_max * sqrt(eps_r_max)) / 20
```

If `dx` is not specified, rfx auto-computes it from `freq_max` using 20 cells
per minimum wavelength. Override with `dx=...` when you need finer resolution
(e.g., thin films, fine features) or coarser grids for rapid prototyping.

### 4.4 Simulation Length

**Fixed steps:** Specify `n_steps` or use the default `num_periods` (20
periods at `freq_max`). Good for quick checks.

**Field decay criterion (recommended):** Use `until_decay=1e-4` (or `1e-3`
for faster, coarser results). The solver monitors a field component and stops
when it decays to the specified fraction of peak amplitude. This is essential
for accurate frequency-domain quantities (S-parameters, DFT probes).

```python
result = sim.run(until_decay=1e-4, decay_max_steps=30000)
```

For high-Q structures, increase `decay_max_steps` and expect long runtimes.

### 4.5 When to Use Subpixel Smoothing

- **Use it** when your geometry has curved or angled dielectric interfaces
  (spheres, cylinders, diagonal dielectric boundaries). It provides
  second-order convergence and significantly reduces the grid resolution
  needed for a given accuracy target.
- **Do not rely on it** for PEC boundaries. PEC stairstepping is not mitigated
  by permittivity smoothing — only conformal meshing (not available in rfx)
  helps.
- Subpixel smoothing adds a modest one-time preprocessing cost. It is
  negligible compared to the time-stepping cost for large simulations.

```python
# Recommended for any simulation with curved dielectric shapes
result = sim.run(subpixel_smoothing=True, until_decay=1e-4)
```

### 4.6 Port Placement Guidelines

- **Distance from CPML:** Place ports at least 15 cells (or `15 * dx`) from
  the CPML boundary to avoid evanescent coupling to the absorbing layers.
- **Distance from scatterers:** Place ports at least 10-20 cells from
  discontinuities so that only the dominant propagating mode is measured.
- **Between ports:** For two-port waveguide S-parameters, ensure the ports
  are far enough apart that the structure under test is fully between them.
- **Waveguide port aperture:** The `y_range` and `z_range` should match the
  physical waveguide cross-section exactly. The modal profile is computed
  analytically from these dimensions.

---

## 5. Comparison with Other Tools

### 5.1 rfx vs Meep

| Feature | rfx | Meep |
|---|---|---|
| Autodiff (jax.grad) | Yes — full reverse-mode AD through the FDTD loop | No native AD |
| CPML | CFS-CPML (complex frequency-shifted) | Standard UPML |
| Subpixel smoothing | Dielectric only (SDF-based anisotropic averaging) | Dielectric and PEC (superior for curved PEC) |
| Eigenmode solver | No | Yes (MPB integration, eigenmode sources) |
| GPU acceleration | Yes (JAX XLA) | Limited (no native GPU backend) |
| Language | Python / JAX | Python / C++ |

**When to choose Meep:** You need eigenmode sources, PEC subpixel smoothing,
band-structure computation, or a mature community ecosystem.

**When to choose rfx:** You need differentiable simulation for inverse design,
GPU acceleration, or CFS-CPML for challenging absorbing-boundary problems.

### 5.2 rfx vs OpenEMS

| Feature | rfx | OpenEMS |
|---|---|---|
| Autodiff | Yes | No |
| Modal decomposition | Waveguide port TE/TM decomposition | Waveguide port support |
| Mesh import | No (CSG primitives only) | Yes (STEP, STL via CSXCAD) |
| GUI | No | AppCSXCAD visual editor |
| Nonuniform / subgridded grid | Yes | Yes (graded mesh) |
| Language | Python / JAX | MATLAB / Python / C++ |

**When to choose OpenEMS:** You have complex CAD geometry, need a nonuniform
mesh, or prefer a visual workflow with AppCSXCAD.

**When to choose rfx:** You need gradient-based optimisation or want to stay
in a pure Python/JAX ecosystem.

### 5.3 rfx vs CST / HFSS (Commercial)

| Feature | rfx | CST / HFSS |
|---|---|---|
| License | Open-source (MIT) | Commercial (expensive) |
| Autodiff | Yes | No (finite-difference sensitivities only) |
| Conformal mesh | No | Yes (PBA in CST, curvilinear in HFSS) |
| MPI / distributed | No (single GPU) | Yes (multi-node, multi-GPU) |
| GUI / CAD integration | No | Full 3D modeller, parametric sweeps |
| Solver variety | FDTD only | FDTD, FEM, MoM, asymptotic, hybrid |
| Post-processing | Basic (Python scripts) | Extensive built-in (Smith charts, 3D patterns) |

**When to choose CST/HFSS:** You need conformal meshing for curved PEC, very
large problems, MPI parallelism, regulatory-grade validation, or a
click-and-solve workflow.

**When to choose rfx:** You need differentiable simulation for topology/shape
optimisation, open-source reproducibility, tight JAX/ML integration, or
cannot afford commercial licences.

### 5.4 Decision Flowchart

1. **Need jax.grad through the EM solver?** → rfx is your best option in the
   open-source space.
2. **Complex curved PEC geometry?** → Meep (subpixel PEC) or commercial
   (conformal mesh).
3. **Domain > 200^3 cells?** → Commercial solver with MPI, or Meep on CPU
   clusters.
4. **CAD import (STEP/STL)?** → OpenEMS or commercial.
5. **Band structure / eigenmodes?** → Meep + MPB.
6. **Quick parametric sweep of a planar/waveguide structure?** → rfx (fast
   GPU iteration + autodiff).

---

## Summary

rfx excels at **small- to medium-scale, differentiable electromagnetic
simulations** on a single GPU. Its JAX foundation makes it uniquely suited for
inverse design and gradient-based optimisation of RF/microwave structures. The
geometry system is intentionally simple — core CSG primitives, higher-level
`Via` / `CurvedPatch` helpers, and boolean operations — which keeps the API
clean and the rasterisation fully differentiable.

The key trade-off is generality: rfx does not have conformal meshing, MPI
parallelism, eigenmode solvers, or CAD import. For problems that need those
capabilities, use Meep, OpenEMS, or a commercial tool. For problems where
`jax.grad` through a full-wave EM solver is the critical enabler, rfx is the
right choice.
