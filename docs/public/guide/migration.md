---
title: "Migration Guide"
sidebar:
  order: 90
---

This page helps you move a model from **Meep** or **openEMS** to rfx. All
three are Yee-cell FDTD solvers, so the physics is familiar; the API is
different. It maps the concepts, translates three typical scripts and lists
the differences that most often catch newcomers.

Before you port a production model, check that its mesh, boundaries, ports
and observables are inside the
[Recommended Configuration](/rfx/validation/recommended-configuration/) and
[Support Boundaries](/rfx/api/support-boundaries/).

## Concept mapping

| Concept | Meep | openEMS | rfx |
|---------|------|---------|-----|
| Grid setup | `Simulation(resolution=N)` | `InitCSX()` + `InitFDTD()` | `Simulation(freq_max=..., domain=...)` or `Simulation.auto(...)` |
| Cell size | `resolution` (cells per unit length) | `SetDeltaUnit(1e-3)` and mesh lines | `dx=` in metres; derived from `freq_max` if omitted |
| Source | `Source`, `EigenModeSource` | `AddExcitation` | `add_source()` (no impedance), `add_port()` and the other port types |
| S-parameters | `add_flux()` + post-processing | `CalcPort` | one calculator per port family: lumped/wire `run(compute_s_params=True)`, microstrip `compute_msl_s_matrix()`, waveguide `compute_waveguide_s_matrix()`, coax `compute_coaxial_line_reflection()` / `compute_coaxial_two_port()` |
| Resonances | `harminv(...)` | manual FFT | `result.find_resonances()` |
| Auto-stop | `stop_when_fields_decayed` | `EndCriteria` | `run(until_decay=1e-3)` for open problems; a fixed `n_steps` for a closed PEC cavity |
| Materials | `Medium(epsilon=...)` | `AddMaterial` | `sim.add_material(...)`, or a library name such as `"fr4"` |
| Metal | `perfect_electric_conductor` | `AddMetal` | `sim.add(shape, material="pec")` for a **volume**; `sim.add_thin_conductor(...)` or a zero-thickness `Box` for a **sheet** |
| Absorber | `PML(thickness)` | `AddPML` | `boundary="cpml"`, `cpml_layers=` |
| Dispersive media | `LorentzianSusceptibility` | `AddLorentzMaterial` | `DebyePole`, `LorentzPole`, `drude_pole()` |
| Gradients | adjoint solver for design-region objectives | not native | `jax.grad` through the solver |
| Inverse design | `meep.adjoint.OptimizationProblem` | not native | `rfx.optimize(sim, region, objective)` |
| Non-uniform mesh | not native | `SmoothMeshLines` | `dz_profile=` (and `dx_profile`/`dy_profile`) with the limits in [Non-Uniform Mesh](/rfx/guide/nonuniform-mesh/) |

An openEMS `AddMetal` box with one zero-length side is a sheet, and a solid box
is a volume. rfx follows the same split: the declaration decides it. See
[Core Concepts](/rfx/guide/concepts/).

## Translating scripts

### Meep: cavity resonance

```text
# Meep (for comparison only)
import meep as mp

sim = mp.Simulation(
    cell_size=mp.Vector3(0.1, 0.1, 0.05),
    resolution=50,
    boundary_layers=[],
)
sim.sources = [mp.Source(
    mp.GaussianSource(frequency=2.0, fwidth=0.5),
    component=mp.Ez,
    center=mp.Vector3(0.03, 0.03, 0.02),
)]
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(), 1e-6))
```

```python
# rfx equivalent
from rfx import Simulation

sim = Simulation(freq_max=5e9, domain=(0.1, 0.1, 0.05), boundary="pec")
sim.add_source(position=(0.03, 0.03, 0.02), component="ez",
               amplitude_kind="current")
sim.add_probe(position=(0.06, 0.06, 0.02), component="ez")
result = sim.run(n_steps=5000)
modes = result.find_resonances()   # list of modes with .freq, .Q, ...
print([f"{m.freq / 1e9:.3f} GHz" for m in modes[:3]])
```

Coordinates are in metres and `freq_max` in hertz; there are no normalized
units. With no `dx`, rfx picks a cell size from `freq_max`, and with no
waveform the source is a `GaussianPulse` centred at `freq_max / 2`.

A closed PEC box has no loss, so its field never decays and `until_decay` is
the wrong stop. Choose a fixed record long enough for the frequency resolution
you need, repeat with a longer one, and keep only the modes whose frequency
does not move. The Q of a lossless cavity is infinite; add material or
conductor loss before you read a Q.

### openEMS: waveguide S-parameters

```matlab
% openEMS (MATLAB, for comparison only)
CSX = InitCSX();
FDTD = InitFDTD('EndCriteria', 1e-5);
[CSX, port{1}] = AddRectWaveGuidePort(CSX, 0, 1, ...);
RunOpenEMS(Sim_Path, Sim_CSX);
port = calcPort(port, Sim_Path, freq);
s11 = port{1}.uf.ref ./ port{1}.uf.inc;
```

```python
# rfx equivalent
import jax.numpy as jnp
from rfx import Simulation

# WR-90: the transverse domain is the guide cross-section, and the waveguide
# calculator bounds it with PEC walls, so no metal box is needed.
freqs = jnp.linspace(8e9, 11.5e9, 8)   # above the 6.56 GHz TE10 cutoff
sim = Simulation(freq_max=12e9, domain=(0.10, 0.02286, 0.01016),
                 dx=2e-3, boundary="cpml", cpml_layers=16)
sim.add_waveguide_port(0.024, direction="+x", freqs=freqs, f0=9.75e9, name="in")
sim.add_waveguide_port(0.076, direction="-x", freqs=freqs, f0=9.75e9, name="out")

result = sim.compute_waveguide_s_matrix(num_periods=45)
s11 = result.s_params[0, 0, :]   # shape (n_ports, n_ports, n_freqs)
```

The mesh here is coarse so the example runs quickly. The calculator's
warnings name the absorber depth and record length it needs; 16 layers and 45
periods satisfy them here. `normalize=False` (the default) is the modal V/I
decomposition; `normalize="flux"` gives a power-normalized transmission
magnitude. There is no single `CalcPort`
equivalent: `run(compute_s_params=True)` is only for lumped and wire
`add_port(...)` ports. See [Probes and S-Parameters](/rfx/guide/probes-sparams/).

### Meep adjoint: inverse design

```text
# Meep (for comparison only)
opt = mpa.OptimizationProblem(...)
opt.update_design([design_params])
f, g = opt()  # forward + adjoint
```

```python
# rfx: optimize() runs jax.grad through the solver; no adjoint code to write
import jax.numpy as jnp
from rfx import Simulation, Box, DesignRegion, GaussianPulse
from rfx.optimize import optimize
from rfx.optimize_objectives import minimize_s11_at_freq_wave_decomp

sim = Simulation(freq_max=4e9, domain=(0.05, 0.05, 0.025), dx=2.5e-3,
                 boundary="pec")
sim.add_material("slab_init", eps_r=4.0)
sim.add(Box((0.015, 0.015, 0.005), (0.035, 0.035, 0.020)), material="slab_init")
sim.add_port((0.025, 0.025, 0.0025), "ez",   # in the air below the slab
             waveform=GaussianPulse(f0=3e9, bandwidth=0.8))

region = DesignRegion(
    corner_lo=(0.015, 0.015, 0.005),
    corner_hi=(0.035, 0.035, 0.020),
    eps_range=(1.0, 12.0),
)
objective = minimize_s11_at_freq_wave_decomp(target_freq=3e9, port_idx=0)

# This objective needs the port DFT at the target frequency.
opt = optimize(sim, region, objective, n_iters=3, lr=0.01, n_steps=400,
               port_s11_freqs=jnp.asarray([3e9]), verbose=False)
print(opt.loss_history)   # opt.eps_design holds the optimized permittivity
```

Three iterations on a coarse mesh keep the example short, and preflight warns
that 2.5 mm cells are at the edge of phase accuracy in the slab. A real design
needs a finer mesh, many more iterations and a converged forward run to check
the final result. See
[Inverse Design](/rfx/guide/inverse-design/).

## What is different in rfx

- **Everything runs in Python through JAX.** There is no solver binary and no
  file exchange. The same script runs on CPU, or on an NVIDIA GPU with a
  CUDA-enabled JAX build.
- **Gradients come from the solver itself.** Supported workflows let
  `jax.grad` differentiate through the time stepping, so an ordinary Python
  loss function can drive an optimizer. Check the final design's RF result
  with a converged run; the loss is only a proxy.
- **One `Simulation` object holds the model.** Add materials, shapes, sources,
  ports and observables to it, then call `run()` or a port calculator. `run()`
  computes lumped and wire port S-parameters automatically; other port
  families use their own calculators.
- **Preflight checks the setup before you spend time on a solve.** Read
  `sim.preflight()`. `run()` repeats it and prints the warnings.
- **Snapshots.** `run(snapshot=SnapshotSpec(...))` records field frames on
  the uniform grid, with a fixed `n_steps` or with `until_decay`. The frames
  are `interval` steps apart (default 10); read their times from
  `result.snapshot_axes`. A graded mesh, multi-device runs and the ADI solver
  do not record snapshots and raise if you ask for them.
- **A built-in material library.** `fr4`, `rogers4003c`, `rogers4350b`,
  `rt_duroid_5880`, `alumina`, `ptfe`, `silicon`, `copper`, `aluminum`, `pec`,
  `air`, `vacuum` and `water_20c` work by name. Their values are nominal: use
  your laminate's data for a real design.
- **Automatic defaults.** Without `dx`, the cell size comes from `freq_max`.
  The CPML is 16 layers unless you set `cpml_layers`.
  `Simulation.auto(freq_range=(f_min, f_max))` proposes a domain and mesh from
  a band.

## Common gotchas

| Gotcha | What to do |
|-------|----------|
| Positions are in metres, not millimetres or cells | Write `0.012`, not `12` |
| `freq_max` is in hertz, not normalized frequency | `freq_max=5e9` for 5 GHz |
| The absorber cells are added outside `domain=` | `domain=` is the region you model. Geometry drawn to a face continues into the absorber, as a board running off the edge should |
| `run()` returns a `Result`, not files | Read `result.time_series`, `result.s_params`, `result.find_resonances()` |
| There is no mesh file | The solved grid is `result.grid`; `rfx.plan_simulation_mesh(sim)` inspects the mesh before a run |
| An argument a path does not support raises | `run()` refuses, for example, `until_decay` on a closed graded mesh before the first step; the message says what to change |

## Next

[Quick Start](/rfx/guide/quickstart/) runs a first simulation, and
[Core Concepts](/rfx/guide/concepts/) explains the model behind it.
