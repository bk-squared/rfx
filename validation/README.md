# rfx Validation Scripts

These scripts compare rfx results with analytic calculations, other solvers,
or a second rfx configuration. They are test cases, not tutorials. Start with
`examples/README.md` if you are learning the software.

Each comparison covers the geometry, mesh, frequency range, source, and output
implemented by that script. A passing result does not establish accuracy for a
different setup.

## Run a validation case

Run scripts from the repository root. The straight-waveguide case is the
shortest place to start after completing the examples:

```bash
python validation/crossval/03_straight_waveguide_flux.py
```

The process exit code has a specific meaning:

- `0`: every configured numerical check passed, including a required external
  comparison when the case has one.
- `1`: a simulation, numerical check, or required execution step failed.
- `2`: the rfx checks completed, but a required reference file or external
  solver was unavailable. This result is inconclusive, not a pass.

See `crossval/manifest.json` for each script's dependencies, expected exit
codes, numerical-check files, and generated artifacts.

## Available cases

| Script | What it checks | Reference and interpretation limit |
| --- | --- | --- |
| `crossval/01_waveguide_bend.py` | 2-D dielectric-bend transmission and straight-guide normalization | Meep and unit transmission; applies to the scripted geometry and band |
| `crossval/02_ring_resonator.py` | 2-D ring-resonator mode frequencies | Meep; checks the scripted mode count and mean frequency error |
| `crossval/03_straight_waveguide_flux.py` | Straight-guide flux and band-mean transmission | Meep and unit transmission within the measured window |
| `crossval/04_multilayer_fresnel.py` | Normal-incidence multilayer reflection, transmission, and energy balance | Transfer-matrix calculation and Meep over the scripted band |
| `crossval/05_patch_antenna.py` | 2.4 GHz finite-ground-plane patch resonance (coarse-mesh integration check) | Diagnostic-reporter only: reports a coarse-mesh resonance vs a Balanis estimate and OpenEMS, both reported not gated. Its 20% openEMS check is a smoke bound that passes over two different substrate geometries (#325), so it does not establish accuracy. Patch-accuracy evidence lives in `tests/test_issue80_patch_resonance_harminv.py`, `tests/test_issue80_patch_s11_regression.py`, and `tests/test_patch_cavity_eps_oracle.py`. |
| `crossval/06b_msl_notch_filter_uniform.py` | Uniform-mesh microstrip-notch response | The script checks a quarter-wave estimate at dx=80 µm. A separate dx=50 µm OpenEMS result is checked by `tests/test_msl_notch_e4_comparison_gates.py`; the shipped dx=80 µm run is not an external-solver comparison. Neither establishes general MSL-port accuracy. |
| `crossval/09_half_symmetric_waveguide.py` | PMC half-symmetry against a full cavity | Pozar TE101 frequency and a full-domain rfx calculation for the scripted mesh |
| `crossval/10_pmc_cpml_half_symmetric.py` | Stability of combined PMC and CPML boundaries | rfx self-consistency check only; it is not an external-solver comparison |
| `crossval/11_waveguide_port_wr90.py` | WR-90 empty guide, PEC short, and dielectric slab | Analytic checks, including the Airy slab result, determine the exit status. Optional Meep, OpenEMS, and Palace files add diagnostic table columns but are not required for exit code `0`. Use `docs/guides/sparameter_support_matrix.md` for the supported range and authoritative checks. |

When reporting a result, include the script path, commit, geometry, mesh,
frequency range, comparator, and numerical thresholds. Do not generalize a
successful case to a different port, material model, boundary, or mesh without
an applicable comparison.

## Other validation directories

- `research/subgrid/12_subgrid_disjoint_prototype.py` is a short-duration
  research check for a topology that is not long-time energy-stable.
- `research/subgrid/13_subgrid_material_validation.py` demonstrates the guarded
  subgrid boundary and checks the same material setup on a uniform mesh.
  SBP-SAT/subgridding is not a supported feature.
- `tap_paper/` contains paper-support scripts. Values identified there as
  targets are experiment goals, not measured results. See
  `tap_paper/README.md` for the status of each script.
