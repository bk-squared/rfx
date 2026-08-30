# rfx inverse-design examples (T-MTT paper)

These scripts reproduce the inverse-design results of the paper

> **rfx: An End-to-End Differentiable 3-D FDTD Simulator for RF and Microwave
> Engineering**, B. Kim, submitted to *IEEE Transactions on Microwave Theory and
> Techniques* (T-MTT).

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
| `lumped_port_gradient_check.py` | not used (single forward + adjoint + FD check) | a few minutes |

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

> **-46.1 dB pending re-derivation (issue #514).** The descent objective
> above is `|S21|²` from the plane-lane N-probe extractor
> (`rfx.probes.msl_wave_decomp` via `register_msl_plane_probes` /
> `_v_from_plane` / `_i_from_plane`), which was reproduced by the
> **pre-#514** V/I definitions and is reproducible as-is at tag
> `paper-tmtt-2026`. #514 pointed that lane's V/I integrals at
> `compute_msl_s_matrix`'s own primitives instead of a drifted copy, so
> a re-run after this change can move the -46.1 dB figure; it is pending
> re-derivation and re-optimization (a PI decision + a lead-lane run, not
> a pod task). Amendment threshold: any change beyond the quoted 0.1 dB
> precision. The **-55.7 dB validated optimized-null** figure is the
> unchanged reference — it comes from the production S-matrix path
> (`compute_msl_s_matrix`, already flux-validated, #520/#549), which #514
> does not touch.

**Example 2 - Waveguide dielectric taper (30 sections).**
`waveguide_dielectric_taper.py` matches a WR-90 guide to a high-permittivity
load over the X-band by optimizing a graded N-section dielectric taper through
the differentiable modal S-matrix. The 30-section taper reaches a band-mean
|S11| of -26.7 dB at dx = 0.5 mm, and -38.0 dB re-optimized at the production
resolution dx = 0.25 mm, versus a discretized Klopfenstein taper of the same
electrical length at -36.6 dB (dx = 0.25 mm). At a comparable coarse-grid solve
budget, particle-swarm and genetic search trail the gradient by at least
11.6 dB. Run:
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

`lumped_port_gradient_check.py` verifies the lumped voltage-current S11 path:
the analytic directional derivative of |S11|^2 (one reverse-mode pass) agrees
with a central finite difference (two forward passes) to 0.2% over the 24-cell
design region. Runs on CPU in ~5-10 min (no GPU needed):
`JAX_PLATFORMS=cpu python validation/tmtt_paper/lumped_port_gradient_check.py`.

The modal-S gradient check (taper) and the NTFF log-ratio gradient check
(superstrate) are exercised by their example scripts. AD-vs-FD agreement is
2.0% for the modal-S path and 1.1% for the NTFF path.

## Forward validation

The solver's forward accuracy is cross-checked against analytic references and,
for the notch filter and the beam-steering case, against openEMS: PEC cavity
eigenfrequencies to 0.008%, and WR-90 dielectric-step and Debye S-parameters to
~0.01 in |S11|.
