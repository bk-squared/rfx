# S-Parameter Calculation Support Matrix

This matrix answers three separate questions for each public primitive:

1. Does it define a port impedance and reference plane?
2. Which API computes its S-parameters?
3. What RF evidence and restrictions apply to that calculation now?

The machine-readable companion is
`docs/guides/sparameter_support_matrix.json`. Evidence levels are defined in
`docs/guides/physics_validation_evidence_rule.md`; raw voltage/current replay is
defined in `docs/guides/sparameter_dump_replay.md`.

Status terms on this page are **supported**, **limited**, **experimental**,
**not documented**, and **unsupported** as defined in
`docs/guides/support_matrix.md`.

## Result and metric convention

Full matrices use
`S[receiver_port, driven_port, frequency_index]` with shape
`(n_ports, n_ports, n_freqs)`. Frequencies are in Hz with shape `(n_freqs,)`.
Unless a metric name or sentence explicitly says dB, `*_mag_abs_diff` is an
absolute difference in linear magnitude.

Passivity, reciprocity, replay, and warning checks are necessary diagnostics;
none alone proves calibrated RF accuracy. Apply only the evidence listed for
the selected port, mode, mesh, geometry, and frequency range.

Automatic guards also differ by calculator. MSL and waveguide matrices receive
the full column-power check. Lumped/wire results are checked only for non-finite
or excessive individual `|S|` values, not column power. The public
`compute_coaxial_line_reflection(...)` result has no shared automatic passivity
guard. An absent warning therefore cannot be compared across port families.
(`compute_coaxial_two_port(...)`, below, does route through the shared guard.)

## API summary

| Primitive | Calculation API | Result | Current status |
|---|---|---|---|
| Lumped `add_port(..., extent=None)` | `run(compute_s_params=True, s_param_freqs=...)` | `Result.s_params`, `Result.freqs` | **limited** — one-cell impedance model; E2/E3/E4-partial evidence |
| Lumped `add_port(..., extent=None)` | `forward(port_s11_freqs=...)` | `ForwardResult.s_params`, `.freqs` (S11 vectors) | **limited** — uniform, single-device AD path; inherits the lumped-port RF limits |
| Wire `add_port(..., extent=...)` | `run(compute_s_params=True, s_param_freqs=...)` | `Result.s_params`, `Result.freqs` | **limited** — multi-cell discrete feed across `extent`; magnitude evidence is stronger than absolute calibration evidence |
| Wire `add_port(..., extent=...)` | `forward(port_s11_freqs=...)` | `ForwardResult.s_params`, `.freqs` (S11 vectors) | **limited** — uniform, single-device AD path |
| `add_msl_port(...)` | `compute_msl_s_matrix(...)` | `MSLSMatrixResult.S`, `.freqs`, `.Z0`, `.beta`, `.port_names`, `.reliable` | **limited** — E5-narrow / eigenmode-blocked; external notch agreement is characterized, not tight; `eps_override` AD validated (converged f64 referee 0.000110) |
| `add_waveguide_port(...)` | `compute_waveguide_s_matrix(...)` | `WaveguideSMatrixResult.s_params`, `.freqs`, `.port_names`, `.port_directions`, `.reference_planes` | **limited** — broad magnitude evidence for documented uniform single-mode rectangular guides; phase and junction evidence are narrower |
| `add_waveguide_port(...)` | `run(...)` | `Result.waveguide_sparams[name]` | **limited diagnostic** — per-port output, not the full multi-port matrix API |
| `add_coaxial_port(...)` | `compute_coaxial_line_reflection(...)` | `CoaxialLineReflectionResult` | **limited** — exactly one `face="top"` port; broad-E5 analytic and broad-E4 MEEP evidence for the documented TEM-line result |
| `add_coaxial_port(...)` | `compute_coaxial_s_matrix(...)` | `CoaxialSMatrixResult` | **experimental and deprecated** — older single-plane V/I path; can produce non-physical `\|S11\| > 1` for a lossless short |
| `add_coaxial_port(...)` | `compute_coaxial_two_port(...)` | `CoaxialTwoPortResult` | **experimental** (issue #489 stage 2) — two-drive through-line 2-port solve; every DUT it can currently gate against is azimuthally symmetric (TM0n only); no external referee; no phase claim |
| `add_floquet_port(...)` | no documented high-level S-parameter API | none | **experimental** — broadside diagnostic helpers only; no calibrated Floquet-port result |
| Sources, TFSF, probes, DFT planes, flux monitors | none | field, resonance, or flux results | **not a port** — no impedance or S-matrix reference plane |

## Lumped port

**Use:** a one-cell feed or load with a positive scalar reference impedance.
It is not a transmission-line mode. R/L/C/RLC entries below are synthetic
extractor checks, not capabilities of `Simulation.add_port(...)`; circuit
elements use the separate `add_lumped_rlc(...)` API and do not themselves
define an S-parameter port.

**RF evidence (E2/E3/E4-partial):**

- Closed-form open, short, matched, resistive, capacitor, inductor, series-RLC,
  and parallel-RLC extractor checks have `max_abs_diff 7.91e-8` against a
  `2.20e-6` tolerance.
- A real two-port V/I replay covers 9 frequencies and 2 ports with
  `max_abs_diff 1.13e-7` against `9.84e-7`.
- A three-case uniform-grid replay/passivity/reciprocity check has maximum replay
  difference `1.58e-7`, maximum column power `0.971`, and maximum reciprocity
  difference `3.02e-7`.
- The rfx/OpenEMS PEC-box magnitude checks cover three port-position cases. The
  largest per-case linear-magnitude differences are `0.11835` maximum and
  `0.06466` mean. These cases do not cover a broad matched/open/short/load set.

**Restrictions:**

- `forward(port_s11_freqs=...)` is uniform and single-device only.
- Nonuniform lumped-port S-parameter extraction is unsupported.
- TFSF, waveguide ports, and mixed port families cannot share this S-parameter
  calculation.
- Analytic extractor and V/I replay checks validate algebra and reproducibility;
  they do not establish a generally calibrated lumped-port result.

Relevant implementations and tests include `tests/test_sparam.py`,
`tests/test_port_dump_replay.py`,
`scripts/diagnostics/report_lumped_analytic_oracles.py`, and
`scripts/diagnostics/build_lumped_openems_sweep_comparison.py`.

## Wire port

**Use:** a one-cell transverse probe/wire feed. Use `add_msl_port(...)` when the
intended model is a distributed microstrip line.

**RF evidence (E2/E3/E4-partial):**

- A real midpoint-cell two-port V/I replay covers 7 frequencies and 2 ports with
  `max_abs_diff 7.82e-8` against `9.80e-7`.
- A three-case uniform-grid replay/passivity/reciprocity check has maximum replay
  difference `8.20e-8`, maximum column power `0.979`, and maximum reciprocity
  difference `1.24e-6`.
- The patch/OpenEMS comparison over 1.5--3.4 GHz has
  `max_mag_abs_diff 0.05318` and `mean_mag_abs_diff 0.02750`. Phase is not gated
  because the reference conventions differ.
- A three-case OpenEMS mesh/length comparison covers `dx` of 1--2 mm, wire
  lengths of 4--8 mm, and 0.8--1.8 GHz, with
  `max_mag_abs_diff_across_cases 0.05212`.

**Restrictions:**

- Absolute S-matrix calibration remains limited by the one-cell feed and current
  per-cell impedance convention. Do not treat the replay as modal-port
  calibration.
- The nonuniform wire calculation is experimental; regression and AD coverage
  are not external RF validation.
- `forward(port_s11_freqs=...)` is uniform and single-device only.

Relevant checks include `validation/crossval/05_patch_antenna.py`,
`tests/test_twoport_wire_port.py`, `tests/test_wire_port_sparams_forward.py`, and
`scripts/diagnostics/report_wire_replay_sweep.py`.

## Microstrip-line port

**API:** `compute_msl_s_matrix(...)` with the laplace/quasi-TEM model.

**RF evidence (E5-narrow / eigenmode-blocked):**

- The uniform thru-line check uses `|S21|` in `(0.90, 1.05)` and
  `Re(Z0)` in `(40, 65) ohm` for the cited `dx=80 um` setup.
- The analytic quarter-wave-notch case reports `1.63%` frequency error,
  `-34.3 dB` notch depth, and median `Re(Z0)=48.6 ohm` for its cited run.
- The committed matched-geometry OpenEMS comparison at `dx=50 um` reports a
  `5.8%` notch-frequency difference, linear `|S21|` mean difference `0.105`,
  and maximum difference `0.2172` over 2.5--6 GHz. This is a characterized
  external check, not a tight cross-solver match. See
  `tests/fixtures/msl_notch_e4/comparison_summary.json`.
- Raw three-probe replay matches the production matrix with
  `max_abs_diff=0` over 30 frequencies.
- The `eps_override` gradient is validated by a converged f64 AD-vs-FD
  referee: rel_err `0.000110` at `num_periods=20` through the full
  extraction (#483/#486). The launch fixture derives from registered
  materials on both the FD and AD sides; staticness is regression-locked
  by `tests/test_msl_source_fixture_static.py`.

`MSLSMatrixResult.reliable` is available during normal execution and is `None`
during JAX tracing. Under the multi-drive solve (issue #507) a `False` entry at
any port contaminates the entire frequency slice `S[:, :, k]`, so the only safe
per-bin filter is `np.all(reliable, axis=0)`; the per-port index tells you which
probe plane collapsed. Details and the filtering example:
[Low-signal MSL bins](../public/guide/probes-sparams.mdx#low-signal-msl-bins).
A `True` entry is not an accuracy guarantee.

**Restrictions:**

- Nonuniform `mode="laplace"` and `mode="uniform"` have internal settled-S11
  regression coverage but no external nonuniform comparison; treat them as
  experimental.
- `mode="eigenmode"` is unsupported and raises `NotImplementedError`.
- SBP-SAT subgridding, ADI, TFSF, and mixed port families are unsupported for
  this calculation.
- Strong-reflector `|S11|`'s roughly 0.16--0.22 "staircase-Z0 floor" this
  document used to cite is **RETIRED** (issue #487) — it was substantially
  the #511 modal-voltage span and #507 far-port-echo single-ratio assembly
  extractor defects, fixed in PR #516 (`f95240f`), not a mesh property. A
  smaller floor survives whose mechanism is the mismatch between the
  rasterized line's own Z0 and the analytic Hammerstad-Jensen anchor `S` is
  normalized against, tracking `|Gamma_implied| = |(Z0-Z0_HJ)/(Z0+Z0_HJ)|`
  within ~1.3x over 5 of 6 points of the #487 re-sweep
  (`scripts/diagnostics/msl_z0_bias_floor_sweep.py`, committed JSON). No
  single envelope number is published, for two reasons: (1) below
  `|Gamma_implied| ~ 0.006` (the finest aligned sweep point) the sweep
  cannot resolve whether the mechanism still holds — it compares one
  band-mean Z0 against a band-mean `|S11|(f)`, with no per-bin trace to
  exclude a Jensen's-inequality artifact, against a fitted-Z0 estimator
  the library's own honesty guard calls healthy only to +/-10% — so this
  is reported as a resolution limit of the sweep, not a confirmed second
  mechanism (see the script for the full breakdown); (2) even where the
  mechanism does hold, the measured floors are specific to that one thru
  fixture, and generalizing them to a dB promise for arbitrary MSL ports
  would itself be the overclaim the next sentence forbids. Do not
  generalize the matched/thru/notch evidence.
- The auto `n_probe_offset` solves the upstream/downstream clearance interval
  at driver time (midpoint with a reflector, unchanged without one) and warns
  loudly when a short feed cannot satisfy both clearances (#469). Library
  witness probes are excluded from preflight advisories (#470).

## Rectangular-waveguide port

**API:** use `compute_waveguide_s_matrix(...)` for a full matrix. At least two
ports are required. `run()` provides only per-port diagnostics.

**Uniform single-mode magnitude evidence:**

- Analytic Airy checks cover WR-28, WR-62, WR-15, WR-340, and WR-10 dielectric
  slabs. With `normalize="flux"`, maximum per-band linear `|S11|` differences
  are 0.005--0.041 for the cited cases.
- The Palace WR-90 comparison covers empty guide, PEC short, and dielectric slab
  from 8.2--12.4 GHz. Across five compared terms, the maximum and mean
  linear-magnitude differences are `0.0707` and `0.00943`.
- The validation battery requires empty-guide `max |S11| < 0.02`, maximum column
  power `< 1.02`, symmetric-obstacle mean reciprocity error `< 0.01`, and a
  PEC-short result with `min |S11| >= 0.99` and `max |S11| < 1.03`.
- The cv11 cross-solver gates use a band-mean linear-magnitude difference of
  `0.10` for S11 and `0.07` for S21, a masked band-mean phase difference of
  `60 degrees` where reference magnitude is at least `0.30`, and a maximum
  complex-S difference of `0.30`.

This supports broad magnitude use inside the documented uniform, single-mode
rectangular-guide limits. Phase evidence covers fewer configurations; do not
infer equally broad phase accuracy from the magnitude status.

**Nonuniform transverse mesh:** single-mode `normalize=True` and
`normalize="flux"` run. Analytic Airy fixtures cover grading ratios 1--3,
relative permittivity 2 and 4, and 8.2--12.4 GHz with a maximum
linear-magnitude difference of `0.01561`. A passed Palace magnitude comparison
covers `normalize="flux"`, a graded-`dy` ratio of 2, and WR-90
empty/PEC-short/dielectric-slab cases over 8.2--12.4 GHz; its maximum and mean
linear-magnitude differences are `0.07009` and `0.01042`. This is external RF
evidence for that configuration, not for other profiles, bands, phase,
multimode extraction, or arbitrary junctions. The calculation remains
experimental outside those stated results. `eps_override` and `sigma_override`
differentiation is implemented only with `normalize="flux"`.
`tests/test_waveguide_nu_flux_ad.py` finite-difference-checks `eps_override`;
there is no corresponding nonuniform `sigma_override` AD-vs-FD test. Neither
implementation nor gradient regression is RF validation.

**Setup restrictions:**

- Prefer `normalize="flux"`; it uses a matched reference run for Poynting-flux
  normalization and modal V/I phase. It costs `2 * n_ports` FDTD runs.
- The cited five-band fixtures use 24 CPML cells and band-specific `dx` values
  from 25 um to 1.5 mm. Every case has at least 60 cells per vacuum wavelength
  at the highest sampled frequency. This is not a 60-cell guarantee inside the
  dielectric: the coarsest WR-340, `eps_r=4` case has about 30 cells per bulk
  dielectric wavelength.
- Port, reference-plane, and discontinuity distances are band-specific in the
  fixtures. Do not infer a one-guided-wavelength rule from them; preserve those
  coordinates or establish mesh, domain, and port-placement convergence for a
  different geometry.
- Choose slab length and frequency samples away from Airy reflection nulls;
  otherwise relative error is dominated by the numerical noise floor.
- Choose `dx` so the slab length is an integer number of cells; staircase
  quantization directly perturbs the round-trip phase.
- Draw interior PEC obstacles (irises, septa, posts) with their interior faces
  on **cell midpoints**, keep the metal depth an exact number of cells, and
  assert the realized footprint. `Box` rasterizes half-open `[lo, hi)` over node
  coordinates, so a box drawn between two node planes occupies one cell fewer
  than drawn, asymmetrically at the `hi` face. Two facing fins drawn to leave a
  nominal opening `d` therefore leave an electrical opening of `d + dx` **or
  `d + 2*dx`, and which one is not predictable from the nominal dimensions**.
  The pair's two interior faces are different corner types: the lo fin's is a
  `hi` corner, which half-openness always drops, so it always retreats one
  cell; the hi fin's is a `lo` corner, which is kept unless float32 rounding
  puts the node just below it. One retreat gives `d + dx` with the opening
  **asymmetric** (centre `dx/2` low); two retreats give `d + 2*dx`, **centred**.
  Measured on WR-90 at both a/30 and a/60: 7.620 mm and 18.288 mm give
  `d + dx` off-centre, 12.192 mm gives `d + 2*dx` centred. **Transverse** to
  the propagation direction the electrical dimension is the span between the
  innermost zeroed planes, `(n_open + 1) * dx` — the measure that reproduces
  `a = cells * dx` exactly, and an independent refit of 16 committed
  single-iris configurations across two meshes pins the realized aperture to
  within 1/20 of a cell of it (a stage-S3 / issue #499 review observation; no
  committed record carries the refit yet — a caution-grade number, like the
  longitudinal one below). **That identity does not carry into the
  propagation direction:** an obstacle's electrical *thickness* is set by field
  interaction with the discontinuity rather than by a cutoff, is measured to
  fall between `t_cells * dx` and `(t_cells - 1) * dx` so neither integer rule
  holds, and is not settled — treat it as an unknown of order half a cell and
  fold the sensitivity into the reported envelope instead of picking a rule.
  (This measured *effective* thickness is a different quantity from a cascade
  comparator's electrical-length bookkeeping — issue #499's comparator draws
  `t_c = round(t/dx) + 1` so `(t_c - 1)*dx` conserves total electrical
  length; that choice answers a different question and is not contradicted
  here.)
  Offsetting each
  interior face half a cell the wrong way retreats both faces by construction
  rather than by luck, giving `d + 2*dx` deterministically at every aperture;
  that is the drawing case 18's blocked revision used. In
  the WR-90 single-iris lane this inflated the `|S11|` difference against an
  analytic mode-matching oracle by 4-6x (0.0193 to 0.1262 at `d = 7.620 mm`,
  a/30). Because the error scales with `dx` it mimics first-order convergence,
  and on a resonant structure it shifts the passband instead of widening a
  magnitude tolerance. Midpoint corners are rounding-independent, and with
  `(cells - d_cells)` even the realized opening equals the nominal one exactly;
  at odd parity a symmetric opening of that width is not representable on the
  grid and costs one cell **more** however it is drawn. Odd parity is a fork
  rather than a dead end — change `dx`/the aperture so the parity works, or
  place the fins asymmetrically on purpose and accept a recorded half-cell
  offset instead of rounding the aperture (the quantity that sets the cutoff)
  to the wrong parity. Neither is recommended here, because the cost of the
  offset has not been measured; what is required is that the offset be recorded
  and representable by the comparator, since an off-centre aperture compared
  against a centred oracle silently becomes comparator error. See the `Box`
  docstring for the
  arithmetic and `run_point` in
  `validation/crossval/18_wr90_iris_modematch.py` for the assert pattern.
- Size the absorber from the guide wavelength at the **lowest** measured
  frequency, where `lambda_g` is longest and the `cpml_layers=16` default is
  weakest. `compute_waveguide_s_matrix` documents `>= 0.5 * lambda_g` and now
  emits an advisory when the configured stack is thinner. That floor is a
  minimum, not a target: at WR-90 band edge (8.2 GHz, `lambda_g` about 61 mm)
  a 0.30 `lambda_g` stack left residual `|S11|` ripple 0.0706, 0.50 `lambda_g`
  left 0.0366, and 0.75 `lambda_g` left 0.0093, so a 0.5 `lambda_g` absorber
  can still set the accuracy envelope instead of discretization. Case 18
  derives its depth as `ceil(0.75 * lambda_g(band edge) / dx)`; the validated
  fixtures use 20-24 cells rather than the default 16.
- Multimode `normalize=True` is unsupported.
- Branch, T-junction, and septum calculations require per-port matched straight-
  guide references and far-port placement. Compact arbitrary junctions are not
  covered by the broad uniform-guide magnitude result.

## Coaxial port

Use `compute_coaxial_line_reflection(...)` for the documented TEM-line result.
The simulation must contain exactly one coaxial port, it must use `face="top"`,
and no other port family may be registered. It also requires `mode="3d"`,
`solver="yee"`, `precision="float32"`, `stencil_order=2`, a uniform grid, and
`boundary="cpml"` with `cpml_layers > 0`. Other settings raise before grid
construction. Both z faces must have positive CPML thickness, the method must
use its default `cpml_axes="z"`, all six `BoundarySpec` face tokens must be
`cpml`, and periodic boundary axes are unsupported.
The calculator constructs the line, TEM source, DFT planes, and termination.
Do not register separate geometry, thin conductors, lumped RLC elements,
probes, field monitors, NTFF boxes, or `add_coaxial_*` termination helpers.
The registered port contributes x/y, `face`, radii, and waveform. Its z
coordinate, `pin_length`, and `impedance` do not set the internally derived
line layout or loads; use `feed_impedance=` and use `dut_impedance=` only for
`termination="matched"`. `probe_count` must be an integer of at least three,
and all requested planes must fit between the DUT and source. Increase the z
domain or reduce the count, start, or spacing if the method reports that they
do not fit; it does not silently use fewer planes.

**RF evidence (broad-E5 analytic, broad-E4 external):**

- The analytic check covers 4--12 GHz, short/open/matched/resistive 25 and
  100 ohm terminations, characteristic impedances 48.6 and 63 ohm, and a mesh
  sweep. For method-gated cases, maximum `|Gamma|` deviation is `0.0372` against
  a `0.05` tolerance and maximum recurrence residual is `0.00588` against
  `0.03`.
- Use about four or more annulus cells; the committed gate requires at least
  3.5. Coarser cases are reported as under-resolved.
- The matched-load fixture reaches `|Gamma|` deviation `0.0929` because of the
  single-cell annular resistor and is reported separately rather than used as a
  method gate.
- The MEEP short/open comparison over 4--12 GHz has maximum and mean
  linear-magnitude differences `0.0628` and `0.0235`.
- End-to-end differentiation is checked through the `eps_scale` dielectric
  channel; the cited AD-versus-finite-difference discrepancy is `2.6%`.

See `tests/fixtures/coax_broad_e5/`, `tests/fixtures/coax_broad_e4/`, and
`tests/test_coax_end_to_end_ad.py`.

This API is not a general multi-port coaxial-network solver and does not cover
arbitrary launches, mixed port families, nonuniform meshes, TFSF, Floquet, or
SBP-SAT. PEC, UPML, zero-layer CPML, ADI, two-dimensional, and fourth-order
configurations are also unsupported. Mixed precision is unsupported.
Boundary specifications without positive CPML on both z faces, non-z
`cpml_axes` selections, mixed boundary-face tokens, and periodic axes are
unsupported. `run()` and `forward()` reject high-level coaxial S-parameter
requests.
The older `compute_coaxial_s_matrix(...)` path is deprecated and experimental.

`compute_coaxial_two_port(...)` (issue #489 stage 2) extends the same
transmission-line method to two ports: it builds a single through line with a
matched annular-resistor feed near each z end, drives each end's own TEM TFSF
source in turn (two FDTD runs), and assembles the S-matrix via a two-drive
solve that does not assume the non-driven port sees zero incident wave. It is
**experimental**: every DUT it can currently gate against (none, a matched
feed, or a coaxial dielectric plug) is azimuthally symmetric and excites only
TM0n modes, while the transition discontinuities issue #489 targets excite
TE11 (cutoff 25.17 GHz on the validated SMA line, evanescently surviving to
the first probe plane). No external referee has run against this method and
no phase claim is made. See `tests/test_coax_two_port_fdtd.py` for the
measured single-run envelope and its provenance.

Numerical line attenuation at the validated 3.79-cell annulus gives `|S21|`
0.96 -> 0.74 on the 60 mm / 40 GHz fixture over 4-12 GHz even though `|S11|`
stays at or below 0.05 throughout. A post-hoc consistency check (run after
this measurement; the estimator was chosen after seeing an own/other-drive
split described below, not predeclared) found the `|S21|` deficit equals
what the extractor's own matrix-pencil-fitted `Re(gamma)` predicts over the
reported port separation (within about 2% at every measured frequency,
`|S21|` against `exp(-Re(gamma)*L12)`). That check is sensitive to
SCALE-type deficits (amplitude mis-normalization, mode conversion, a bad
wave split — `gamma` is fit from shape, not scale) and is structurally
BLIND to reference-plane referral errors (a referral error scales the wave
amplitude and `L12` by the same factor, so the compensation cancels it
exactly — verified to five decimal places at +30 cells of injected error).
The under-resolved-annulus recipe (at least 4 cells) that this repo already
documents for reflection accuracy applies to transmission magnitude here
too, even when `status` reports `"passed"` (`annulus_cells`
only gates below 3.5).

## Floquet/Bloch and non-port observables

`add_floquet_port(...)` has broadside modal bookkeeping, field-dump replay, and
analytic empty-space/slab diagnostics. Representative internal differences are
`max_abs_diff 0.06067` for the empty-space analytic-null check and
`max_mag_abs_diff 0.06212` for the three-frequency homogeneous-slab magnitude
check. These are not RCWA or independent full-wave validation, and there is no
documented high-level Floquet S-parameter API. Treat the result as experimental.

`add_source(...)`, polarized sources, TFSF, point probes, DFT plane probes, and
flux monitors do not define a port impedance or S-matrix reference. Validate
their field, resonance, far-field, or flux observable directly; do not document
them as port substitutes.

## Rejection and preflight behavior

Explicit S-parameter requests outside the matching API must fail rather than
returning `None` or silently omitting a feature:

- `run(compute_s_params=True)` accepts only lumped/wire `add_port(...)`.
- `compute_msl_s_matrix(...)` accepts only MSL-port simulations.
- `compute_waveguide_s_matrix(...)` accepts only waveguide-port simulations.
- `compute_coaxial_line_reflection(...)` accepts only exactly one `face="top"`
  coaxial port with no mixed port family, within its documented line setup.
- `forward(port_s11_freqs=...)` accepts only uniform, single-device lumped/wire
  port setups.

Check routing before an expensive run:

```python
sim.preflight_sparameters(calculator="run")
sim.preflight_sparameters(calculator="forward")
sim.preflight_sparameters(calculator="msl")
sim.preflight_sparameters(calculator="waveguide")
```

The returned issues check API compatibility, not mesh/time convergence or RF
accuracy. Use `strict=True` when the setup should fail on any reported issue.

## Mixed port families — EXPERIMENTAL, not in the validated set

The per-extractor restrictions above are accurate: `compute_msl_s_matrix`, the
lumped/wire scan driver, and the coaxial calculators each reject foreign port
families. They should not be read as "rfx cannot compute across port families
at all" — a separate, explicitly experimental method exists:

```python
res = sim.compute_mixed_s_matrix(freqs=freqs)   # lumped/wire + MSL, uniform mesh
```

`compute_mixed_s_matrix` drives each port in turn, takes its diagonals from the
per-family validated extractors, and takes off-diagonal **magnitudes** from
Poynting flux (`magnitude_channel="flux"`, the default), normalizing incident
power as `P_net / (1 - |S_jj|^2)`. No characteristic-impedance anchor enters the
magnitude.

What this is and is not:

| aspect | status |
|---|---|
| off-diagonal magnitude | experimental; internal reciprocity witness 9% on the probe-fed MSL fixture (55% on the wave channel) |
| absolute magnitude | **NOT validated** — no external-solver referee has been run |
| off-diagonal phase | provisional (mixes two reference-plane conventions) |
| per-column power | an algebraic identity on the flux channel — **not** a passivity check |

Because the flux normalization makes column power identically 1 whenever the
arriving power equals the net launched power, a green passivity result on this
lane carries no information. Reciprocity is the only independent internal
witness, and the method warns when it exceeds `reciprocity_tol`.

Everything outside the first supported pair is rejected loudly: waveguide,
Floquet, coaxial and TFSF registrations, bare sources and 0-ohm ports, mixed
lumped+wire sets, `reference_plane_cells`, non-uniform meshes, SBP-SAT, and ADI.
The flux channel additionally requires a PEC `z_lo` ground and vertical
(`component="ez"`) lumped/wire ports, since the per-port flux box omits its
bottom face and treats the port extent as a height.

Two further cautions on this lane, both measured rather than anticipated:

- **Neither diagonal is verified.** The wire port-cell V·I accounting was measured
  undercounting delivered power ~3× against an independent Poynting referee, and on an
  end-fed fixture the MSL probe plane's local `V/I` was ~591 Ω and strongly reactive
  while the reported `|S22|` was 0.03. Those cannot both be right, and which one is
  wrong is open.
- **The returned diagonal is not always the measured one.** With
  `enforce_passivity=True` (the default), `_project_passive` is a joint SVD clip: when
  any entry is non-passive it rewrites others as a side effect. On the committed test
  fixture the shipped MSL diagonal came out ~4× its unprojected value. Read `S_raw` and
  `passivity_correction` for what was actually measured.
