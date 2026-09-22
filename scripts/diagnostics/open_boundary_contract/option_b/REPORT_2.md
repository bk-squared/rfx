Gaussian pulses excited a continued microstrip line (3–4.5 GHz and 0.5–4.5 GHz), WR-90 (7–12 GHz), a grounded lossless patch (150 periods), and a normal-incidence plane-wave box (0.2–6 GHz).
The specified witness requires end energy below −40 dB of the post-source peak; other records retain the label `truncation-suspect`.
The recorded patch settling spans -46.870108 to -43.615181 dB; the other recorded S-parameters and reflection amplitudes appear below.

**Addendum 1.** Source exported once to `B/src`, commit `e7f7e02704fd46ea7e21f127b19fc81cb66d6148`. Every job copied `src/PROVENANCE.txt` and verified driver SHA-256 values against [job_source_hashes.json](job_source_hashes.json). No Git command ran in a job. The removed worktree was not recreated.
`REPORT.md` is byte-for-byte unchanged. The original jobs remain under `jobs/`; original drivers, job files, tables and unavailable-result records are copied under `failed_launch/`. New job records are under `jobs_2/`. All task writes stayed under `B/`; no repository edits, commits, pushes, PRs or GitHub comments.

**B1 — calibration measurements.** References: 10.4 m and 10.72 m long, each 10 mm × 10 mm transversely; grids 4161×5×5 and 4289×5×5. Window 0–19.5185602268 ns, 4096 samples, dt 4.76643717383 ps. Stencil-distance wall round trips: 4157 and 4285 steps. Injected echo: amplitude 0.001, delay 64 steps; maximum error from −60 dB = 1.07724805076e-06 dB. Reference-difference maximum = -600 dB (the stored zero floor); minimum incident-spectrum magnitude / band maximum = 0.115972049772. [Full calibration](raw/plane_reference/calibration.json); `extra_rigs.py:74–105, 145–171`.
**B2 — extractor execution recorded; independent numerical validity untested.** Far port removed; the remaining port retains `terminates=()`. The two-port rig retains it on both ports. All one-port calls returned a 1×1 S-matrix. The extractor rejects zero ports (`src/rfx/sparams/msl.py:260`), uses the registered port count (`:311`), and checks PEC under every retained port (`:496–521`). Received strip-edge records include both x pads. [Source excerpts](SOURCE_EXCERPTS.md); [one-port record, s=0/N=4](raw/msl/0_4/one_port/result.json). Per-arm energy witnesses remain attached; no independent one-port calibration was added.
**Other assumption.** A main-versus-#1012 comparison was not run. The plane table records both mirror-probe reflection measurements and their differences; no cause is assigned. No measured c, derived |S11| bar from the 2 dB |S21| bar, minimum N, or best-alpha statement is asserted.

**STOP record.** No new infrastructure STOP or arm exception occurred. The prior Git-ownership STOP remains in `REPORT.md`; all four new jobs reached GPU measurements. Numerical records marked truncation-suspect are retained at the requested durations; duration extension or accepting the present records remains a leader decision.
**Design discrepancies retained from the first report.** DESIGN.md:31/:50 specifies N=4/8/16; :42 specifies six patch layers. Both are recorded: 84 common configurations, 21 additional low-frequency configurations, and 7 additional N=6 patch configurations. The cited waveguide builder declares 40×20 mm; the local variant records 22.86×10.16 mm, 90×40 aperture cells, cutoff 6.556807478 GHz. The cited reflectivity oracle uses a point Ez source; the local variant uses a uniform 5×5-node transverse Ez source with periodic transverse boundaries. Its R is the two-ended-box residual divided by the incident reference. These variants are unchanged from the prepared first launch; alternatives remain those listed in `REPORT.md`.

Each job was relaunched once. Provider status and sweep configuration counts follow; GPU identity-check solves and the two plane-reference solves are additional.

| Structure | New run ID | Provider status | Configurations |
|---|---|---|---|
| Microstrip line, both frequency bands | [369367263409](jobs_2/msl/provider.log) | completed | 42 |
| WR-90, 7–12 GHz | [369367263410](jobs_2/waveguide/provider.log) | completed | 21 |
| Grounded lossless patch, 150 periods | [369367263411](jobs_2/patch/provider.log) | completed | 28 |
| Normal-incidence plane-wave box, 0.2–6 GHz | [369367263412](jobs_2/plane/provider.log) | completed | 21 |

Tables contain the recorded headline quantities. Linked tables additionally contain energy, settling and secondary measurements; no truncation-suspect value is removed.

[Microstrip line, 3–4.5 GHz — Maximum one-port |S11| (dB)](TABLE_msl.md)

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | -9.437 | -9.437 | -9.439 | -9.454 | -9.449 | -9.243 | -8.951 |
| 8 | -14.99 | -14.99 | -14.99 | -14.99 | -14.99 | -15.08 | -14.1 |
| 16 | -22.27 | -22.27 | -22.27 | -22.26 | -22.24 | -22.18 | -22.11 |

[Microstrip line, 0.5–4.5 GHz — Maximum one-port |S11| (dB)](TABLE_msl_low.md)

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | -8.062 | -8.062 | -8.069 | -8.102 | -8.4 | -7.508 | -1.627 |
| 8 | -12.4 | -12.4 | -12.39 | -12.38 | -12.41 | -12.76 | -9.493 |
| 16 | -17.28 | -17.28 | -17.28 | -17.28 | -17.31 | -17.55 | -18.13 |

[WR-90, 7–12 GHz — Maximum |S11| (dB)](TABLE_waveguide.md)

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | 24.27 | 24.2 | 24.07 | 23.58 | 22.14 | 17.82 | 11.13 |
| 8 | -13.85 | -13.85 | -13.85 | -13.85 | -13.86 | -13.89 | -13.98 |
| 16 | -22.27 | -22.27 | -22.27 | -22.28 | -22.29 | -22.34 | -22.5 |

[Grounded lossless patch, 150 periods — Worst-probe settling (dB)](TABLE_patch.md)

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | -45.05 | -45.04 | -45.03 | -44.99 | -44.87 | -44.5 | -43.62 |
| 8 | -45.6 | -45.6 | -45.6 | -45.61 | -45.61 | -45.63 | -45.71 |
| 16 | -46.76 | -46.76 | -46.76 | -46.76 | -46.77 | -46.8 | -46.87 |
| 6 | -44.94 | -44.94 | -44.94 | -44.92 | -44.89 | -44.8 | -44.67 |

[Normal-incidence plane-wave box, 0.2–6 GHz — Maximum clean-reference |R| (dB)](TABLE_plane.md)

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | -5.895 | -5.905 | -5.924 | -5.994 | -6.207 | -1.342 | 11.36 |
| 8 | -13.21 | -13.22 | -13.23 | -13.26 | -13.35 | -13.68 | -1.964 |
| 16 | -24.06 | -24.07 | -24.08 | -24.15 | -24.33 | -24.96 | -26.24 |

**Energy witnesses.** Counts of configurations with at least one truncation-suspect drive: msl: 21/21; msl_low: 21/21; waveguide: 21/21; patch: 0/28; plane: 1/21. Every WR-90 source-end index equals the 6884-sample record length, so its post-source peak is undefined; end energy in joules and modal settling remain recorded. [All per-drive witnesses](WITNESSES_2.csv).
**Plane frequencies.** [CROSSINGS_plane.csv](CROSSINGS_plane.csv) gives every interpolated −40/−60 dB crossing, sampled values below each threshold, and 0.05 s/(2π ε0). All 21 −60 dB crossing lists are empty; no sampled R is at or below −60 dB. Per-arm reflection CSVs each contain 291 frequencies (20 MHz spacing). A unique f_alpha is not assigned.
**Counts and checks.** 112/112 configurations returned complete records; 196 sweep solves, 3962714 sweep timesteps, plus 2×4096 plane-reference timesteps and 8×96 identity-check timesteps. All four GPU instrument checks report identical final field arrays and probe records with/without energy recording. Requested/received dtype: float32. CPML arrays and realized grids, pads and apertures are stored per solve; preflight text is verbatim. No CPU FDTD solve or repository test suite ran.
**Commands and last lines.** [COMMANDS_2.md](COMMANDS_2.md). Four launches only; short provider reads; full available logs retained. The four first-launch runs remain referenced by `REPORT.md` and were retained under the reference-run exception. No runs were deleted.
**Commit list and cleanup.** No commits created. The source export has no `.git`; the former worktree remains absent. The first-launch empty commit list remains in [commit_list.txt](commit_list.txt). Driver hash, provenance, 112-result, 21-spectrum, report-link and prior-report checks are recorded in [verification_2.json](verification_2.json).
**FACT discrepancies.** The named battery dimensions and oracle source differ from the design descriptions as recorded above. The historical alpha-sweep values and #1012 effects were not remeasured.

Conclusion: leader fills after the sweep.
