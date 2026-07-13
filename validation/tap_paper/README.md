# rfx inverse-design examples (T-MTT paper)

These scripts reproduce the inverse-design results of the paper

> **rfx: An End-to-End Differentiable 3-D FDTD Simulator for RF and Microwave
> Engineering**, B. Kim, submitted to *IEEE Transactions on Microwave Theory and
> Techniques* (T-MTT).

Each script is self-contained and runs a real reverse-mode FDTD gradient.
`SMOKE=1` (the default) verifies the pipeline end to end on CPU in ~1-3 min;
it does not reproduce the headline numbers. `SMOKE=0` reproduces the paper's
headline numbers and is GPU-backed in practice for the taper and the
beam-steering superstrate.

> **Install.** These scripts use the optional optimization extra (for `optax`):
> `pip install rfx-fdtd[optimization]` (with the repo on `PYTHONPATH`, or an
> editable install).

> **Notch filter (Example 1) lives elsewhere in the repo**, not duplicated here:
> `validation/tap_paper/msl_stub_notch_tuning.py` (cross-validation companion:
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
`validation/tap_paper/msl_stub_notch_tuning.py` (cross-validation companion:
`validation/crossval/06b_msl_notch_filter_uniform.py`).

**Example 2 - Waveguide dielectric taper (30 sections).**
`waveguide_dielectric_taper.py` matches a WR-90 guide to a high-permittivity
load over the X-band by optimizing a graded N-section dielectric taper through
the differentiable modal S-matrix. The 30-section taper reaches a band-mean
|S11| of -26.7 dB at dx = 0.5 mm, and -38.0 dB re-optimized at the production
resolution dx = 0.25 mm, versus a discretized Klopfenstein taper of the same
electrical length at -36.6 dB (dx = 0.25 mm). At a comparable coarse-grid solve
budget, particle-swarm and genetic search trail the gradient by at least
11.6 dB. Run:
`SMOKE=1 JAX_PLATFORMS=cpu python validation/tap_paper/waveguide_dielectric_taper.py`
(CPU, ~1-3 min); `SMOKE=0 python validation/tap_paper/waveguide_dielectric_taper.py`
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
`SMOKE=1 JAX_PLATFORMS=cpu python validation/tap_paper/beam_steering_superstrate.py`
(CPU, ~1-3 min); `SMOKE=0 python validation/tap_paper/beam_steering_superstrate.py`
(full, GPU).

## Gradient verification

`lumped_port_gradient_check.py` verifies the lumped voltage-current S11 path:
the analytic directional derivative of |S11|^2 (one reverse-mode pass) agrees
with a central finite difference (two forward passes) to 0.2% over the 24-cell
design region. Runs on CPU in ~5-10 min (no GPU needed):
`JAX_PLATFORMS=cpu python validation/tap_paper/lumped_port_gradient_check.py`.

The modal-S gradient check (taper) and the NTFF log-ratio gradient check
(superstrate) are exercised by their example scripts. AD-vs-FD agreement is
2.0% for the modal-S path and 1.1% for the NTFF path.

## Forward validation

The solver's forward accuracy is cross-checked against analytic references and,
for the notch filter and the beam-steering case, against openEMS: PEC cavity
eigenfrequencies to 0.008%, and WR-90 dielectric-step and Debye S-parameters to
~0.01 in |S11|.
