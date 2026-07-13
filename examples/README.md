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
| 10 | `tutorials/rcs_scattering.py` | radar cross-section with incident-reference subtraction |
| 11 | `inverse_design/differentiable_s11_design.py` | the differentiable design loop: `forward` + `jax.grad` + optimizer |
| 12 | `tutorials/artifact_report_demo.py` | exporting a shareable scene/mesh/report bundle |

`config/microstrip_thru.yaml` shows the declarative YAML front-end for the same
Simulation API.

Read the preflight output every time — the advisories are part of the result,
and several of them encode hard-won physics rules (lossless-dielectric Q traps,
graded-mesh rasterization, ring-down truncation).


---

## Where the validation evidence went

The cross-validation scripts that used to live under `examples/crossval/` are
internal measurement fixtures, not tutorials. They now live in
`validation/` (see `validation/README.md` for the script table and the
machine-readable registry `validation/crossval/manifest.json`). Start with the
learning path above; consult the validation tree only when you need the
evidence behind a specific accuracy statement.
