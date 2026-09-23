This is a measurement-run infrastructure STOP; no microwave result was produced.
The requested structures were a microstrip line (3–4.5 GHz and 0.5–4.5 GHz), WR-90 (7–12 GHz), a grounded lossless patch (150 periods), and a normal-incidence plane-wave box (0.2–6 GHz).
FDTD solves: 0; recorded timesteps: 0; measured S-parameters, decay, energy, and reflection crossings: unavailable.

Local detached source: `e7f7e02704fd46ea7e21f127b19fc81cb66d6148`. GPU commit reads failed; the four empty `commit.txt` files are retained. No GPU commit or dtype is asserted.
All writes are under `B/` apart from creating/removing the requested worktree. No repository edits, commits, pushes, PRs, or GitHub comments.

**B1 — unmeasured.** Build-only plane reference: 10.4 m × 10 mm × 10 mm, grid 4161×5×5; check reference 10.72 m, grid 4289×5×5. Window 0–19.5185602268 ns, 4096 samples, dt 4.76643717383 ps. The recorded stencil-distance wall round trips are 4157 and 4285 steps. The −60 dB injected-echo and reference-difference measurements never ran. [Build evidence](dry/plane/0_4/result.json); `extra_rigs.py:74–105`.
**B2 — build acceptance observed; numerical validity unmeasured.** Far port removed. The two-port variant registers `terminates=()` on both ports; the one-port variant retains it on its sole port. `rfx/sparams/msl.py:260` rejects zero ports, `:311` uses their count, and `:496–521` requires PEC under each retained port. Both builds reached the solver context; the one-port context contains strip edges in both x pads. Grid 289×58×41, dx 50 µm, pads x=4/4, y=4/4, z=0/4. [Build evidence](dry/msl/0_4/result.json); [pinned source excerpts](SOURCE_EXCERPTS.md).
**Other assumption — unmeasured.** No #1012 comparison or mirror-probe reflection measurement ran. No measured f_alpha, minimum N, derived |S11| bar, or best-alpha statement is available.

**STOP-1 (all jobs).** At line 27 of every YAML, `git -c safe.directory="$src" -C "$src" rev-parse HEAD` printed `fatal: detected dubious ownership in repository at '/root/workspace/bk-workspace/rfx-wt-B-sweep'`. No Python command was reached. Options: retain these STOP records; or authorize new attempts after testing the configured safe-directory mechanism inside this image. No option requiring a new launch was executed; brief lines 13–14 say to launch each job once.
**STOP-2 (local build, resolved before launch).** The first waveguide metadata recorder unpacked each config as a pair and raised `ValueError: too many values to unpack (expected 2)`. The recorder was changed to iterate configs; subsequent build reached the solver context. Options were correcting that iteration (done) or omitting aperture capture (not done). [Initial](smoke_waveguide_1_8.log); [repeat](smoke_waveguide_03_8.log).
**Design discrepancies recorded.** DESIGN.md:31/:50 specifies N=4/8/16 and 84 primary arms; :42 specifies six patch layers. Prepared schedule contains the common rows plus seven N=6 patch arms and 21 low-frequency arms: 112 scheduled configurations, zero executed. Options: common patch rows, N=6 only, or both (prepared).
The named battery actually declares 40×20 mm (`tests/oracle/test_waveguide_port_validation_battery.py:49–61`); the prepared WR-90 variant realizes 22.86×10.16 mm, 90×40 aperture cells, cutoff 6.556807478 GHz (`dry/waveguide/0.3_8/result.json`). Options: that local WR-90 variant (prepared), or the original 40×20 mm guide.
The named reflectivity oracle uses a point Ez source (`test_pml_reflectivity.py:34–60`). The prepared variant uses uniform Ez injection over 5×5 periodic transverse nodes, retaining clean-reference subtraction; its R is the two-ended box residual. Options: that plane variant (prepared), or the original 3D point source. Clarification questions received no reply.

Each job was submitted once. The provider logs and terminal status reads are retained:

| Structure | Run ID | Provider status | GPU solves |
|---|---|---|---|
| Microstrip line, 3–4.5 GHz | [369367263402](jobs/msl/provider.log) | failed | 0 |
| WR-90, 7–12 GHz | [369367263403](jobs/waveguide/provider.log) | failed | 0 |
| Grounded lossless patch, 150 periods | [369367263404](jobs/patch/provider.log) | failed | 0 |
| Normal-incidence plane-wave box, 0.2–6 GHz | [369367263405](jobs/plane/provider.log) | failed | 0 |

The following tables have columns s and rows N. STOP denotes no measurement; the linked files also contain witness tables and secondary-quantity tables.

[Microstrip line, 3–4.5 GHz — Maximum one-port |S11| (dB)](TABLE_msl.md)

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |
| 8 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |
| 16 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |

[Microstrip line, 0.5–4.5 GHz — Maximum one-port |S11| (dB)](TABLE_msl_low.md)

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |
| 8 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |
| 16 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |

[WR-90, 7–12 GHz — Maximum |S11| (dB)](TABLE_waveguide.md)

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |
| 8 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |
| 16 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |

[Grounded lossless patch, 150 periods — Worst-probe settling (dB)](TABLE_patch.md)

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |
| 8 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |
| 16 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |
| 6 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |

[Normal-incidence plane-wave box, 0.2–6 GHz — Maximum clean-reference |R| (dB)](TABLE_plane.md)

| N \ s | 0 | 0.01 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|---|
| 4 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |
| 8 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |
| 16 | STOP | STOP | STOP | STOP | STOP | STOP | STOP |

Energy end/post-source peak and settling witnesses: unavailable for every arm. The [plane crossing CSV](CROSSINGS_plane.csv) has explicit STOP statuses; per-arm reflection CSVs contain headers only.
The 112 per-arm JSON files under `results/` contain null measurements, realized geometry, actual commit, and actual dtype, plus the recorded run ID and STOP. CPU build-only records remain separately under `dry/`; their preflight text is verbatim.

**Commands and last lines.** [COMMANDS.md](COMMANDS.md) records worktree creation, all five build attempts, parse/shell checks, four submissions, terminal reads and provider-log last lines. No prior run had this sweep prefix; the cited alpha runs were retained as references. No runs were deleted.
**Checks.** Four successful build-only cases; one corrected recorder exception. GPU field/probe identity checks and all sweep solves were not reached. No CPU FDTD solve ran. No product tests or CI suites were run for this measurement-only task.
**Commit list.** `git log --oneline origin/main..`: empty. `git status --porcelain=v1 --untracked-files=all`: empty before removal.
**FACT discrepancies.** The literal alpha profile and `terminates=()` build behavior were observed. The named battery’s dimensions and the reflectivity oracle’s point excitation differ from DESIGN.md as listed above. Historical alpha-sweep numerical claims were not remeasured.
**Cleanup.** The requested `git worktree remove` returned 0; `test ! -e /root/workspace/bk-workspace/rfx-wt-B-sweep` returned 0. The worktree is removed. [Artifact verification](verification.json): 112 explicit STOP records, four nonempty provider logs, zero solves, and all report links present.

Conclusion: leader fills after the sweep.
