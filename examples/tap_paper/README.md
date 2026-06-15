# rfx inverse-design examples

Illustrative inverse-design examples for an in-progress internal rfx manuscript
(working title *"rfx: A JAX-Native Differentiable 3-D FDTD Simulator with a
Trusted-Gradient Design Loop"* — not yet written or submitted). Each script is
self-contained, runs a real reverse-mode FDTD gradient, and defaults to a fast
`SMOKE=1` CPU sanity mode; set `SMOKE=0` for the full-resolution settings (GPU
recommended, and required in practice for the taper).

The full-resolution numbers quoted below are **projected targets**, not yet
locked by a committed run — treat them as design goals, not validated results.
The `SMOKE=1` runs only exercise the pipeline end to end; they do not reproduce
the headline figures.

> **Install.** These scripts use the optional optimization extra (for `optax`):
> `pip install rfx-fdtd[optimization]` (with the repo on `PYTHONPATH`, or an
> editable install). `gradient_tolerance_analysis.py` does not need `optax`.

> **Related examples elsewhere in the repo.** The notch-filter and patch-antenna
> examples are not duplicated here:
> - Notch filter (MSL stub tuning): `examples/inverse_design/msl_stub_notch_tuning.py`
>   (cross-validation companion: `examples/crossval/06_msl_notch_filter.py`).
> - Patch antenna (cross-solver validation): `examples/crossval/05_patch_antenna.py`.

> **GPU note.** `SMOKE=0` for the dielectric taper does a long, full-resolution
> reverse-mode scan and is impractical without a GPU.

## Scripts

- **`waveguide_dielectric_taper.py`** — broadband WR-90 waveguide-to-dielectric-load matching by optimizing a graded N-section dielectric taper via the differentiable modal S-matrix. *Projected full-res target:* a 30-section taper reaching band-averaged reflection near **-27.5 dB at dx=0.5 mm** (and near **-38 dB** re-optimized at dx=0.25 mm, cf. a discretized Klopfenstein taper at about -36.6 dB) — design goals, not yet locked by a committed run. Run: `SMOKE=1 JAX_PLATFORMS=cpu python examples/tap_paper/waveguide_dielectric_taper.py` (CPU, ~1-3 min); `SMOKE=0 python examples/tap_paper/waveguide_dielectric_taper.py` (full, GPU recommended).

- **`grin_superstrate_directivity.py`** — maximizes broadside directivity of a dipole by optimizing a per-cell GRIN dielectric-superstrate permittivity field, with reverse-mode AD through the FDTD solve and the NTFF transform (`maximize_directivity(..., log_ratio=True)`, the unbiased power-changing-DoF form). *Projected full-res target:* broadside directivity rising from the bare dipole's **~1.1 dBi to ~6.4 dBi**, with D = 4*pi*U_max/P_rad over the full (theta,phi) sphere using the sin(theta) solid-angle weight. Run: `SMOKE=1 JAX_PLATFORMS=cpu python examples/tap_paper/grin_superstrate_directivity.py` (CPU, ~1-3 min); `SMOKE=0 python examples/tap_paper/grin_superstrate_directivity.py` (full, GPU).

- **`beam_steering_superstrate.py`** — tilts the main beam off broadside (toward 30 deg) by optimizing a graded per-cell dielectric superstrate over a reflector-backed dipole. *Projected full-res target:* **D(30 deg) ~ 11 dBi** with the realized peak at ~30-32 deg, versus ~5.4 dBi for a laterally uniform slab of the same aperture and ~5.9 dBi for the bare plate-backed dipole (dx=lambda/20, 31x31x3-cell superstrate ~= 2.9k DOF, 140 Adam iters). Run: `SMOKE=1 JAX_PLATFORMS=cpu python examples/tap_paper/beam_steering_superstrate.py` (CPU, ~1-3 min); `SMOKE=0 python examples/tap_paper/beam_steering_superstrate.py` (full, GPU).

- **`gradient_tolerance_analysis.py`** — reuses the FDTD gradient for analysis (not design): one reverse pass yields the full per-cell sensitivity field dD/d(eps), first-order tolerance propagation predicts sigma_D (cross-checked in-script against a small Monte-Carlo ensemble), and projected gradient ascent finds the worst-case fabrication corner; re-optimizes a small GRIN slab inline so no saved eps is needed. *Projected full-res target:* at dx=lambda/20 with 507 design cells, **507 per-cell sensitivities from ONE backward pass** (vs 1014 finite differences = 2 forwards/cell); first-order sigma_D matching an 80-sample Monte-Carlo at sigma_eps=0.10; a worst-case corner ~0.16 dB below nominal. Run: `SMOKE=1 JAX_PLATFORMS=cpu python examples/tap_paper/gradient_tolerance_analysis.py` (CPU, ~1-3 min; corner -0.034 dB below random worst at the coarse SMOKE settings); `SMOKE=0 python examples/tap_paper/gradient_tolerance_analysis.py` (full, GPU).
