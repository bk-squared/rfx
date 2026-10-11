# rfx inverse-design examples (T-MTT paper)

These scripts reproduce the inverse-design results of the paper

> **rfx: An End-to-End Differentiable 3-D FDTD Simulator for RF and Microwave
> Engineering**, B. Kim, *IEEE Transactions on Microwave Theory and
> Techniques* (T-MTT).

## Paper version and main

The numbers in the paper, and the numbers this README quotes from it, were produced at the tagged
release `paper-tmtt-2026` (2026-08-10). To reproduce them, run the scripts at that tag:

```bash
git checkout paper-tmtt-2026
```

The scripts in this directory on `main` are the CURRENT code. The solver's port, material and conductor
models were corrected after the tag, so some numbers differ; each script's header says what it shows on
`main`. The lumped-port gradient check on `main` runs a different scene from the paper's, because the
paper's scene no longer exercises the gradient (see that script's header).

## Paper materials map (start here)

Everything the paper and its response letter reference lives in five places:

| Material | Location |
|---|---|
| Worked inverse-design examples (notch, taper, beam steering, gradient check) | this directory |
| Solver cross-validation suite (18 numbered studies with embedded Meep/openEMS configs; Palace setups in `palace/`) | [`validation/crossval/`](../crossval/) |
| Four-solver X-band patch record (geometry + port figure, protocol, results, exclusions) | [`docs/crossval/patch_xband_4solver.md`](../../docs/crossval/patch_xband_4solver.md) |
| Raw patch-campaign data (CST Touchstone files, falsification ledger, `REPRODUCE.md`) | branch [`research/calibration-inverse`](https://github.com/bk-squared/rfx/tree/research/calibration-inverse/scripts/research/calibration/crossval), `scripts/research/calibration/crossval/` |
| Release accompanying the paper | tag [`paper-tmtt-2026`](https://github.com/bk-squared/rfx/tree/paper-tmtt-2026) |

Each script is self-contained and runs a real reverse-mode FDTD gradient.
The smoke/full split differs per script — check before running:

| Script | `SMOKE` env var | CPU expectation |
|---|---|---|
| `waveguide_dielectric_taper.py` | honored, `SMOKE=1` default (coarse grid, short run) | minutes |
| `beam_steering_superstrate.py` | honored, `SMOKE=1` default (coarse grid, few iterations) | minutes |
| `msl_stub_notch_tuning.py` | **not implemented — always runs its full multi-start workload** | ~12–15 min on a fast workstation; can exceed 45 min on slower CPUs |
| `lumped_port_gradient_check.py` | not used (float32 adjoint against float64 finite differences) | about 20 min |

`SMOKE=1` does not reproduce the headline numbers; `SMOKE=0` reproduces the
paper's headline numbers and is GPU-backed in practice for the taper and the
beam-steering superstrate.

> **Expected preflight advisories (notch).** `msl_stub_notch_tuning.py`
> prints four preflight warnings in its validated configuration (an infinite
> `z_lo` PEC face coexisting with finite PEC objects, a lossless dielectric
> in an open CPML domain, and two-substrate-cell MSL ports); they are
> expected for this example and do not indicate failure.

> **Install.** These scripts use the optional optimization extra (for
> `optax`). From the clone root:
>
> ```bash
> python3 -m venv .venv
> .venv/bin/python -m pip install -e '.[optimization]'
> source .venv/bin/activate
> ```
>
> (or `pip install rfx-fdtd[optimization]` with the repo on `PYTHONPATH`).

> **Notch filter (Example 1)** is `msl_stub_notch_tuning.py` in this directory
> (cross-validation companion:
> `validation/crossval/06b_msl_notch_filter_uniform.py`).

> **GPU note.** `SMOKE=0` for the dielectric taper and the beam-steering
> superstrate runs a long, full-resolution reverse-mode scan and is impractical
> without a GPU.

## Worked examples

**Example 1 - Microstrip notch filter (1 variable).** A transmission notch is
placed at 6 GHz by descending a single stub-length design variable. The
single-variable descent reaches a -46.1 dB in-band objective, and the validated
optimized null is -55.7 dB at 5.924 GHz, within 3.1% of the analytic
quarter-wave length. Not duplicated here; see
`validation/tmtt_paper/msl_stub_notch_tuning.py` (cross-validation companion:
`validation/crossval/06b_msl_notch_filter_uniform.py`).

> The -46.1 dB and -55.7 dB figures are the paper's, produced at tag
> `paper-tmtt-2026`. On `main` the descent reaches a different minimum; see
> the header of `msl_stub_notch_tuning.py` for what `main` shows.

**Example 2 - Waveguide dielectric taper (30 sections).**
`waveguide_dielectric_taper.py` matches a WR-90 guide to a high-permittivity
load over the X-band by optimizing a graded N-section dielectric taper through
the differentiable modal S-matrix. The 30-section taper reaches a band-mean
|S11| of -26.7 dB at dx = 0.5 mm, and -38.0 dB re-optimized at the production
resolution dx = 0.25 mm, versus a discretized Klopfenstein taper of the same
electrical length at -36.6 dB (dx = 0.25 mm). At a comparable coarse-grid solve
budget, particle-swarm and genetic search trail the gradient by at least
11.6 dB. Those figures were measured on the 23.0 mm-wide guide the 0.5 mm and
0.25 mm meshes rasterize (23.00 x 10.50 and 23.00 x 10.25 mm), not on WR-90;
decided 2026-09-19 in #1122, disclosed rather than corrected. The default SMOKE lane
meshes at a commensurate dx = 1.27 mm and does solve WR-90 exactly. Run:
`SMOKE=1 JAX_PLATFORMS=cpu python validation/tmtt_paper/waveguide_dielectric_taper.py`
(CPU, ~1-3 min); `SMOKE=0 python validation/tmtt_paper/waveguide_dielectric_taper.py`
(full, GPU).

**Example 3 - Beam-steering superstrate (441-param latent / 2883-cell).**
`beam_steering_superstrate.py` tilts the main beam of a reflector-backed dipole
toward 30 deg by optimizing a graded per-cell dielectric superstrate, with
reverse-mode AD through the FDTD solve and the near-to-far-field (NTFF)
transform. At the lambda/40 mesh-converged recut, D(30 deg) = 9.5 dBi for the
441-parameter latent parameterization and 9.45 dBi for the full 2883-cell
superstrate, a +3.6 dB gain over the 5.9 dBi bare plate-backed dipole. A
laterally uniform slab of the same aperture reaches at most 5.4 dBi toward
30 deg. An independent openEMS run corroborates the steered direction (8.9 dBi
toward 30 deg, with the pattern peak near 30 deg). Run:
`SMOKE=1 JAX_PLATFORMS=cpu python validation/tmtt_paper/beam_steering_superstrate.py`
(CPU, ~1-3 min); `SMOKE=0 python validation/tmtt_paper/beam_steering_superstrate.py`
(full, GPU).

## Gradient verification

`lumped_port_gradient_check.py` verifies the lumped voltage-current S11 path.
In the paper (tag `paper-tmtt-2026`) the directional derivative of |S11|^2
agreed with a central finite difference to 0.2 % over a 24-cell block beside a
lone port. On `main` that scene no longer exercises the gradient, so the script
runs a port feeding a resistor-terminated line and checks the float32 adjoint
against float64 finite differences per cell, plus a record-length check; its
header gives the scene, the declared bars and the numbers on `main`. Runs on
CPU in about 20 min:
`JAX_PLATFORMS=cpu python validation/tmtt_paper/lumped_port_gradient_check.py`.

The modal-S gradient check (taper) and the NTFF log-ratio gradient check
(superstrate) are exercised by their example scripts. AD-vs-FD agreement is
2.0% for the modal-S path and 1.1% for the NTFF path.

## Forward validation

The solver's forward accuracy is cross-checked against analytic references and,
for the notch filter and the beam-steering case, against openEMS: PEC cavity
eigenfrequencies to 0.008%, and WR-90 dielectric-step and Debye S-parameters to
~0.01 in |S11|.
