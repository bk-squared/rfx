This is a measurement-comparator and residual-field diagnostic; no product code was changed.
A point Ez pulse excited a realized 62.956416 mm vacuum cube, declared as 60 mm, with absorbing faces; the reference prescription requires the PEC-wall echo to arrive after the record.
With the committed reference declared as 200 mm and 400 samples, s=1/N=8 measured -47.331259 dB; with a 713.506 mm reference and the same record it measured -68.251294 dB.

**Scope and provenance.** Source export `e7f7e02704fd46ea7e21f127b19fc81cb66d6148`; observed fields float32; all solves on remilab-c0, gpu-rtx4090. Every job copied `PROVENANCE.txt` and verified [driver hashes](job_source_hashes_3.json). All 2744 exported source files and both prior reports remain byte-for-byte unchanged. All task writes stayed under `B/`; no repository edits, Git operations, commits, pushes, PRs or GitHub comments. Snapshot: 2026-09-22T13:21:13.780657+00:00.

**Item 1 — nine exact-oracle values.** The original point source, GaussianPulse(f0=2 GHz, bandwidth=0.5, cutoff=3), stepping loops, window, reference and peak time-domain ratio were retained. The s=1/N=8 numerical method is AST-identical to the committed test after excluding only result capture before its unchanged assertion; all 30 CPML profile arrays at s=1 are bit-identical to stock. [Method check](static_verification_3.json); [profile check](scale1_profile_verification_3.json).
Realized CPML grids for N=4/8/16: 30³/38³/54³, N pads on all six faces; reference 68³ with zero pads and realized side 200.860947 mm. dx=2.99792458 mm, dt=5.715767665 ps, 400 samples, window 0–2.280591298 ns. The original reference's nearest geometric wall round trip is 0.630 ns; no alternative window/reference was substituted in this table. Grid, source/probe indices and actual CPML arrays are recorded per arm under `raw_3/exact_*`.
The [replacement rig-4 table](TABLE_plane.md) contains the committed oracle's peak difference / peak reference, in dB. The variant tables, crossings and per-arm tables are under `failed_instrument/` and are not cited as rig-4 results.

| N \ s | 0 | 1 | 3 |
| --- | --- | --- | --- |
| 4 | -47.346676 | -47.351574 | -47.441456 |
| 8 | -47.563676 | -47.331259 | -47.298867 |
| 16 | -47.419697 | -47.322361 | -47.319560 |


**Item 2 — centre-frequency values, N=8.** Each number uses one source-centre setting and the same peak time-domain method. dx and the realized 62.956416 mm CPML cube remain fixed; only f0, record length and reference size vary. No wide-band spectral crossing is used.

| Source centre (GHz) \ s | 0 | 1 | 3 |
| --- | --- | --- | --- |
| 0.5 | STOP | STOP | STOP |
| 1 | -64.516387 | -68.586788 | -71.286951 |
| 2 | -66.245165 | -68.251294 | -70.354434 |
| 4 | -61.432629 | -62.472239 | -63.691977 |

The following are GPU-received reference sizes for 1/2/4 GHz; 0.5 GHz is an unrun sizing calculation. References have zero pads; CPML grids remain 38³ with eight pads per face. All realized geometric wall-echo times in the completed frequency sweep exceed its window end. [Full frequency table and dimensions](TABLE_plane_frequency.md).

| f0 (GHz) | Samples | Window end (ns) | Reference side (m) | Reference grid | Geometric wall echo (ns) |
|---|---|---|---|---|---|
| 0.5 | 1600 planned | 9.139512496 | 2.764086463 planned | 923³ planned | not run |
| 1 | 800 | 4.566898364 | 1.397032854 | 467³ | 4.630 |
| 2 | 400 | 2.280591298 | 0.713506050 | 239³ | 2.350 |
| 4 | 200 | 1.137437765 | 0.365746799 | 123³ | 1.190 |

**ASSUMPTION and STOP — 0.5 GHz.** The driver chooses `ceil(400 × 2 GHz / f0)` samples to preserve the original pulse-relative duration; Addendum 2 does not explicitly fix that duration. At 0.5 GHz this calls for a 923³ reference: six float32 fields require 17.576 GiB; fields plus three non-donated updated components require 26.364 GiB before materials; two distinct material buffers bring that estimate to 32.222 GiB, against nominal 24 GiB on RTX 4090. These are buffer-size calculations, not measured allocations. No 0.5 GHz allocation or solve was attempted. Options: retain the STOP; leader-authorized shorter window with tail measurements; frequency-scaled mesh; or another reference implementation/resource. None was selected. [Sizing record](reference_sizing_3.json); three explicit STOP records under `results_3/`. No fitted f_alpha or coefficient c is reported; formula values for s=0/1/3 are 0/0.898755/2.696266 GHz.

**Item 3 — one-difference explanation of 48 dB: not found.** The measured source comparison changes the single Ez node to a centred 5×5-node Ez patch; all other original-oracle settings remain fixed. The reference-size comparison uses identical CPML probe samples. These are the recorded contrasts, not a cause assignment for the full 48 dB. [Comparison data](COMPARATOR_COMPARISONS_3.json).

| s=1/N=8, f0=2 GHz | Peak time-domain ratio (dB) | DFT ratio at 2 GHz (dB) |
|---|---|---|
| Point source, original 200 mm reference | -47.331259 | -30.626387 |
| Point source, 713.506 mm reference | -68.251294 | -57.321308 |
| 5×5 source, original 200 mm reference | -36.029747 | -18.816763 |

The reference-size-only difference is -20.920035 dB; the source-patch-only difference is 11.301512 dB. The brief's −68.3 dB is not the number returned by the unchanged committed reference/window here; the resized 2 GHz record is −68.251294 dB. A replay of retained variant traces, without another solve, gives −15.213596 dB under the peak ratio and −19.988515 dB under the 2 GHz full-record DFT ratio. [Replay record](variant_trace_replay_3.json).

**Rigs 1 and 2 — final fields and spatial energy.** The earlier energy witness sums interior nodes only; absorber field energy is measured separately here. The first energy column below is the witness's interior sum reconstructed from the dump. Port means one realized feed/source x plane across the MSL width and ground-to-strip span, or the whole WR-90 modal aperture. These masks, six field components, coordinates, material arrays and available PEC masks are stored with every dump. H and E use the same-step Yee samples; CPML auxiliary variables are excluded. [Full partition table](TABLE_fields_3.md); [component energies and spatial intervals](FIELD_SPATIAL_DETAIL_3.json).

| Drive | Interior end energy (J) | H / interior (%) | At port / interior (%) | Absorber / full grid (%) |
| --- | --- | --- | --- | --- |
| MSL two-port drive 1 | 8.338517867e-24 | 97.170823 | 0.028912 | 14.622733 |
| MSL two-port drive 2 | 7.781864318e-24 | 96.809682 | 0.023066 | 16.126479 |
| MSL one-port drive 1 | 5.367125183e-24 | 92.437626 | 0.013271 | 14.835648 |
| WR-90 drive 1 | 3.833448293e-16 | 79.374880 | 0.525868 | 1.036751 |

For the MSL one-port drive, the 2.5–97.5% longitudinal interval of interior energy is 0.150–13.900 mm. For WR-90 it is 0.762–91.948 mm. The component totals, field extrema with coordinates, and fractions within 0/1/2/4/8/16 cells of each feed plane are retained; no mode or charge mechanism is assigned.
MSL dumps: [one-port](raw_3/diagnostic_msl/one_port/fields_end_00.npz), [two-port drive 1](raw_3/diagnostic_msl/two_port/fields_end_00.npz), [two-port drive 2](raw_3/diagnostic_msl/two_port/fields_end_01.npz). WR-90: [dump](raw_3/diagnostic_waveguide/fields_end_00.npz). Projected maps: [MSL](raw_3/diagnostic_msl/one_port/fields_end_00.png), [WR-90](raw_3/diagnostic_waveguide/fields_end_00.png).
**Witness discrepancy recorded.** The requested s=0/N=8 MSL one-port repeat gives end/post-source-peak -23.990014 dB, last-5%-maximum/post-source-peak -21.751347 dB, and probe settling -81.018486 dB. The retained record has those same values, not the quoted −39.3/−36.4/−104 dB. Both MSL S arrays are bit-identical to their retained arrays. WR-90's source-end index still equals all 6884 samples; its post-source energy peak remains undefined. All four drive energy witnesses retain `truncation-suspect`. [Repeat comparisons](repeat_comparison_3.json); [MSL record](raw_3/diagnostic_msl/result.json); [WR-90 record](raw_3/diagnostic_waveguide/result.json).

**WR-90 passivity warning.** The s=0/N=4 repeat returns maximum |S11|=+24.270875 dB at 7 GHz, maximum column power 267.354461670, and modal settling -27.291651 dB. Prior/new maximum |S11|: +24.271002/+24.270875 dB; maximum per-bin amplitude difference 0.000128756912 dB and phase difference 0.00120854621 degrees. No cause for the repeat difference is assigned.
The old driver calls `extract_waveguide_s11` (`src/rfx/sources/waveguide_port.py:1810–1815`), which emitted no warning in this repeat and does not call the shared guard. The returned matrix was passed, without another solve, through `_warn_if_nonpassive_smatrix` (`src/rfx/sparams/_common.py:1159`), with the unnormalized waveguide tolerance 2.0. Its exact emitted warning is [retained here](raw_3/diagnostic_waveguide/passivity_warning.txt):

> extract_waveguide_s11 (diagnostic replay through shared guard): extracted S-matrix failed a passivity/finiteness self-check — passivity_violation: max column power 267.354 exceeds limit 3 at driven port 0, frequency index 0. A passive structure cannot have column power > 1; this almost always means the extractor (current sign/scale or reference plane) is wrong and the S-parameters are UNRELIABLE. Inspect the V/I dump via rfx.validation.validate_port_smatrix / replay_smatrix_from_vi_dump before trusting or optimizing against these numbers. extract_waveguide_s11 (diagnostic replay through shared guard): per-frequency amplitude advisory at frequency index 0: max |S| = 16.35; passivity violated: extraction/normalization artifact — do not interpret as physics; see the normalize parameter docstring and issue #337

**Jobs and STOP history.** All six jobs were submitted once and completed. Initial provider-capacity queue messages are retained in their logs; no job was relaunched. The only unmeasured requested values are the three 0.5 GHz values listed above. No other rig-1/rig-2 configuration, patch configuration or failed plane variant was rerun.

| Measurement | Run ID | Provider status | Cases |
| --- | --- | --- | --- |
| oracle | [369367263428](jobs_3/oracle/provider.log) | completed | 10 |
| msl | [369367263429](jobs_3/msl/provider.log) | completed | 1 |
| waveguide | [369367263430](jobs_3/waveguide/provider.log) | completed | 1 |
| frequency_1 | [369367263447](jobs_3/frequency_1/provider.log) | completed | 3 |
| frequency_2 | [369367263449](jobs_3/frequency_2/provider.log) | completed | 3 |
| frequency_4 | [369367263450](jobs_3/frequency_4/provider.log) | completed | 3 |

**Checks and commands.** 18 requested oracle values, one source-control value, four final-field dumps; 42 GPU solves and 98815 timesteps. Original nine oracle assertions passed; the source-control's unchanged −40 dB assertion failed at −36.029747 dB and is retained as a numeric measurement. CPU FDTD solves: 0. Build-only, POSIX-shell, source/driver hashes, finite arrays, energy-partition, dump/witness and prior-report checks are recorded in [verification_3.json](verification_3.json). High-level preflight text is retained verbatim per diagnostic; the committed low-level oracle has no preflight call. [Commands and last lines](COMMANDS_3.md).
The initial exact-equality check of a local NumPy 2.4.6 logarithm replay failed in one case, by 7.105427357601002e-15 dB. The saved peak amplitudes match exactly; an independent `math.log10` replay matches all 19 stored dB values exactly. No tolerance or GPU result was changed. [All replay values](log_replay_precision_3.json).
**Commit list and cleanup.** No commits and no Git operations. No worktree was created; no repository files were edited. No VESSL runs were deleted; prior and new evidence-bearing runs are retained. `src/`, `REPORT.md` and `REPORT_2.md` are unchanged.

Conclusion: leader fills after the sweep.
