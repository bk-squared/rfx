# rfx Examples

## Start here — the learning path

New to rfx? Run these in order; each teaches one decision a real design needs.

| Step | Script | You learn |
| --- | --- | --- |
| 1 | `quickstart/hello_world.py` | install check, the minimal Simulation -> run -> probe loop |
| 2 | `tutorials/boundary_spec_demo.py` | choosing boundaries: CPML=open, PEC=closed (never mix roles), PMC symmetry, periodic cells |
| 3 | `tutorials/slab_rt_flux_monitor.py` | R(f)/T(f) the right way: `add_flux_monitor` + two-run reference (why probe-FFT lies) |
| 4 | `tutorials/nonuniform_patch_demo.py` | graded z-mesh for thin substrates + verifying where your layers actually landed |
| 5 | `tutorials/materials_and_dispersion.py` | library + custom materials, why loss matters (the infinite-Q trap), Debye/Lorentz dispersion |
| 6 | `tutorials/run_control_and_fields.py` | `n_steps` vs `num_periods` vs `until_decay`, reading the truncation warning, extracting field slices |
| 7 | `tutorials/ports_and_sparams_101.py` | which port for which structure (all five), S11 basics, real-world pitfalls |
| 8 | `tutorials/resonance_harminv.py` | ring-down resonance extraction, picking modes by physics (not loudness), record-length vs resolution |
| 9 | `tutorials/antenna_farfield_pattern.py` | far-field boxes done right (half-wavelength rule), directivity vs the textbook dipole |
| 10 | `tutorials/patch_antenna_demo.py` | a real antenna end to end: mesh-registered stack, picking the radiating mode by its far field (not loudness), settling witness, honest error budget vs openEMS |
| 11 | `tutorials/rcs_scattering.py` | radar cross-section with incident-reference subtraction |
| 12 | `inverse_design/differentiable_s11_design.py` | the differentiable design loop: `forward` + `jax.grad` + optimizer |
| 13 | `tutorials/artifact_report_demo.py` | exporting a shareable scene/mesh/report bundle |

`config/microstrip_thru.yaml` shows the declarative YAML front-end for the same
Simulation API.

Read the preflight output every time. Its warnings identify conditions that can
invalidate the result, including lossless-dielectric Q traps, graded-mesh
rasterization errors, and incomplete ring-down.


---

## Validation scripts

Solver checks are under `validation/crossval/`; they are not tutorials. Start
with the learning path above. Use `validation/README.md` when you need to
reproduce a comparison with an analytic result or another solver.
