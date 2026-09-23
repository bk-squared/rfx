# B0b measurement record

A lossless epsilon_r = 4 slab, 7.49481145 mm thick, was excited through a TE Floquet port at 5 GHz in an air/slab/air periodic cell.
The analytic TE slab magnitudes at 5 GHz are 0.600000, 0.616486, 0.666206, 0.748306 at 0, 15, 30 and 45 degrees.
The public `run()` and `forward()` records contain `s_params=None` and `freqs=None`; native |S11| and phase are absent.

**F1: public-output premise CONTRADICTED; native angle-accuracy assertion NOT MEASURED.** No native Floquet S11 was returned at any tested angle, including 0 degrees.
**F2: CONFIRMED.** `rfx/` has zero syntactic callers and only the definition of `apply_bloch_periodic_x`; every instrumented run recorded zero calls. The direct-call observer sentinel recorded one.
Source record: [SOURCE_AUDIT.md](SOURCE_AUDIT.md). Observer check: [verification/checks.json](verification/checks.json); six final fields and probe records exactly equal with/without the observer.

The table contains **diagnostic** `extract_floquet_modes` values from captured fields at the source plane, outside the public run; all 17 bins from 4–6 GHz, both planes and both analytic references are in [floquet_band.csv](floquet_band.csv) and [TABLES.md](TABLES.md).

| requested angle deg | diagnostic magnitude / phase deg at 4 GHz | at 5 GHz | at 6 GHz | analytic at angle, 5 GHz magnitude / phase deg | analytic normal, 5 GHz |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.607186 / -3.904 | 0.620398 / -86.783 | 0.468017 / -152.201 | 0.600000 / -90.000 | 0.600000 / -90.000 |
| 15 | 0.596154 / -4.060 | 0.620005 / -88.998 | 0.479969 / -153.374 | 0.616486 / -80.204 | 0.600000 / -90.000 |
| 30 | 0.559929 / -4.605 | 0.621461 / -95.969 | 0.517148 / -156.678 | 0.666206 / -51.696 | 0.600000 / -90.000 |
| 45 | 0.486744 / -5.892 | 0.634532 / -108.552 | 0.583130 / -161.560 | 0.748306 / -7.067 | 0.600000 / -90.000 |

Phase convention: exp(+jwt); analytic reference is the source plane z = 14.9896229 mm, slab front z = 37.47405725 mm. The ahead-plane table uses z = 22.48443435 mm. No extra E/H staggering correction was applied.
At each mesh, recorded fields are bit-identical between tested scan angles: 0/15/30/45 degrees at the coarse mesh, 0/30 degrees at each refinement. [angle_equalities.json](angle_equalities.json).

| dx mm | grid | slab nodes along normal | diagnostic 5 GHz magnitude / phase deg, 0 deg | at 30 deg | final / post-source peak energy dB |
|---:|---|---:|---:|---:|---:|
| 0.936851431 | 9x9x117 | 8 | 0.620398 / -86.783 | 0.621461 / -95.969 | -31.658 |
| 0.468425716 | 17x17x213 | 16 | 0.655886 / -88.524 | 0.658032 / -97.485 | -22.805 |
| 0.234212858 | 33x33x405 | 32 | 0.669118 / -89.204 | 0.671639 / -98.089 | -14.014 |

Floquet pads (x_lo,x_hi,y_lo,y_hi,z_lo,z_hi) = (0,0,0,0,10,10); kernel periodic = (True,True,False), Bloch = None, CPML axes = z. run PEC axes = z; forward PEC axes = empty.
Declared cell = 7.49481145 x 7.49481145 x 89.9377374 mm. Full slab planes contain 81/81, 289/289, 1089/1089 epsilon_r=4 nodes; realized thickness = 7.49481145 mm. [full_material_check.json](full_material_check.json).
Derived transverse array lengths N*dx = 8.431662881, 7.963237166, 7.729024308 mm; each periodic Yee difference rolls the full array (rfx/core/yee.py:172-218).
Initial nominal-span Box measurements are retained under `results/floquet_*`; their slab planes contained 17/81, 33/289, 65/1089 air-valued nodes. They are excluded from the tables above.
Energy = 0.5*dx^3*sum_interior(eps0*eps_r*sum(abs(E)^2)+mu0*mu_r*sum(abs(H)^2)); post-source begins at t0+4*tau. All tabulated Floquet energy ratios are above -40 dB: **truncation-suspect**. [energy_and_grid.csv](energy_and_grid.csv).

The same quarter-wave slab through oblique TFSF at 30 degrees gave |reflection| = 0.675677 at 5 GHz; analytic = 0.666206, difference = +0.123 dB; normal analytic = 0.600000.
Method: `forward()` vacuum/slab pair, `oblique_reflection_magnitude`, +j kernel, 3200 steps, full record. The repository oracle uses the same magnitude helper with a thick-slab front-face gate; no phase is supplied by this helper.
TFSF grid = 117x29x29, dx = 0.936851431 mm, six pads = 10 each; kernel periodic = (False,True,True), CPML = x, complex Bloch present. [tfsf_band.csv](tfsf_band.csv).
TFSF energy ratios: vacuum -10.198 dB (truncation-suspect), slab -46.029 dB. Off-f0 rows retain the source's single-f0 transverse Bloch phase.

| Public entry at 30 deg | execution | native S11 | angle-specific warning/finding | other warning |
|---|---|---|---:|---|
| run() | accepted | None | 0 | lossless slab; extended Box geometry |
| forward() | accepted | None | 0 | lossless slab; extended Box geometry |
| RISUnitCell.sweep_angle([30]) | accepted | returned amplitudes/phases via FFT fallback | 0 | lossless substrate |
| run(compute_s_params=True) | refused before kernel | absent | generic Floquet refusal | exact exception below |

`ValueError: run(compute_s_params=True) computes Result.s_params only for add_port(...) lumped or wire ports; add_floquet_port(...) is experimental and has no claims-bearing run(compute_s_params=True) S-matrix path.`
All preflight banners and warning text are quoted verbatim in [PREFLIGHT_AND_WARNINGS.md](PREFLIGHT_AND_WARNINGS.md). RIS entry case uses the wrapper's lossless rogers5880 substrate, no ground plane; its arrays are in `results/ris_theta30_r1/summary.json`.

FACT correction record: `floquet_port_configs` has only its creation/append sites; `update_floquet_dft`, `inject_floquet_source`, and `compute_floquet_s_params` each have zero callers under rfx/. `extract_floquet_modes` has three calls, all inside `compute_floquet_s_params`. RIS calls `np.fft.rfft` and peak-normalizes when native s_params is None.
Commit: `798ec64e5cda057318664bc431e528819f9e371a`. VESSL run IDs: **none**; all measurements local, summed process CPU = 488.602 s (observer/material/source audits additional short reads). Local case IDs are the result directory names in [measurements.json](measurements.json).
Commands/last lines: [COMMANDS.md](COMMANDS.md); `campaign_full.py` → `CAMPAIGN_FULL_FINAL 11 cases`; `source_audit.py` → Bloch callers 0; `verify.py` → observer exact-equality, sentinel 1.
Repository status/removal: see [cleanup.json](cleanup.json). No repository edits, push, PR, or GitHub comments.

Conclusion: leader fills.
