# B0c PMC cavity measurements

The 24 x 20 x 4 mm vacuum cavity was excited by the specified Ez differentiated Gaussian at (7,6,2) mm and probed at (17,13,2) mm; x faces were declared PMC and y/z faces PEC, without absorbers. The supplied formula gives f01=7.494811450 GHz and f11=9.756058116 GHz at a'=24 mm or 9.932091974 GHz at a'=23 mm (the brief gives 9.7547 GHz by hand at 24 mm). The table records the returned frequencies and whether a mode appeared in the fixed (0,1) frequency window.

| Entry | dx mm | f01 GHz; error; Q | f11 GHz; error; Q | derived a' mm | (0,1) appears | Operators |
|---|---:|---|---|---:|---|---|
| run-cpu | 1 | 7.489619558; 1.875e-08; 2.392e+07 | 9.928806111; 1.628e-07; 3.653e+06 | 23.017689728 | yes | M(x); P(xyz); F(yz) |
| run-cpu | 0.5 | 7.493514190; 1.013e-09; 4.65e+08 | 9.840846749; 9.81e-09; 6.309e+07 | 23.504621839 | yes | M(x); P(xyz); F(yz) |
| run-cpu | 0.25 | 7.494487008; 4.637e-09; 1.016e+08 | 9.798072887; 6.699e-10; 9.198e+08 | 23.751189827 | yes | M(x); P(xyz); F(yz) |
| forward-cpu | 1 | 7.489619557; 1.872e-08; 2.396e+07 | 9.928806112; 1.63e-07; 3.649e+06 | 23.017689725 | yes | M(x); P(xyz); F(yz) |
| forward-cpu | 0.5 | 7.493514189; 9.663e-10; 4.877e+08 | 9.840846752; 1.007e-08; 6.145e+07 | 23.504621821 | yes | M(x); P(xyz); F(yz) |
| forward-cpu | 0.25 | 7.494487008; 4.364e-09; 1.08e+08 | 9.798072883; 8.168e-10; 7.545e+08 | 23.751189849 | yes | M(x); P(xyz); F(yz) |
| sweep-cpu | 1 | none returned | 9.752832409; 2.17e-08; 2.693e+07 | 24.019382318 | no | P(xyz) |
| sweep-cpu | 0.5 | none returned | 9.755252174; 5.408e-08; 1.134e+07 | 24.004838873 | no | P(xyz) |
| sweep-cpu | 0.25 | none returned | 9.755856392; 8.66e-08; 7.085e+06 | 24.001210915 | no | P(xyz) |
| nonuniform-cpu | 1 | 7.489619557; 1.86e-08; 2.412e+07 | 9.928806113; 1.63e-07; 3.648e+06 | 23.017689719 | yes | M(x); P(xyz) |
| nonuniform-cpu | 0.5 | 7.493514190; 1.109e-09; 4.25e+08 | 9.840846750; 9.63e-09; 6.427e+07 | 23.504621832 | yes | M(x); P(xyz) |
| nonuniform-cpu | 0.25 | 7.494487008; 4.546e-09; 1.037e+08 | 9.798072885; 6.902e-10; 8.929e+08 | 23.751189840 | yes | M(x); P(xyz) |
| run-gpu | 1 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | B(xyz); update_he_fast |
| run-gpu | 0.5 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | B(xyz); update_he_fast |
| run-gpu | 0.25 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | B(xyz); update_he_fast |

P=apply_pec; F=apply_pec_faces; M=apply_pmc_faces; B=precompute_coeffs(pec_axes). P/F zero tangential E at 0/N-1; M zeros tangential H at 0/N-2; B zeros tangential E coefficients at 0/N-1. Full call traces and measured zero planes: [DETAILS.md](DETAILS.md), [operator_measurement.json](operator_measurement.json).
Derived: a'=1/sqrt((2*f11/c)^2-(1/b)^2), using b=20 mm and c=299792458 m/s. Error and Q are raw Harminv fields; error is dimensionless. Detection windows: (0,1) 6.8-8.2 GHz; (1,1) 9.0-10.7 GHz; no returned mode is reported as "no".
All entries: node spans 24/20/4 mm, zero pads on all faces; dx 1/0.5/0.25 mm gives 25x21x5 / 49x41x9 / 97x81x17 nodes and 4096/8192/16384 steps. Constant-dz arrays, coordinates, timesteps, source/probe indices, complete mode lists and energy records are in [RESULTS.json](RESULTS.json) and linked raw NPZ files.
Preflight stdout verbatim for every arm: `  [PREFLIGHT] All checks passed.` Return representation: `[]`. Each `.log` preserves the original output; extracted stdout is in each `.preflight.stdout.txt`.
Source: `a4aaa86276598410d9eb66eb373cef65918a83cd`; export checked against all 3003 tracked files, zero mismatches. 12 frequency records; three original GPU callback failures retained.
GPU frequencies and (0,1) presence: NOT MEASURED. All three original GPU traces record `precompute_coeffs` and `update_he_fast`; energy callback failed because JAX_PLATFORMS=cuda excluded the CPU backend. No corrected job was submitted.

| Job | Run id | Last job line / status |
|---|---|---|
| run-cpu | 369367263573 | JOB_FINAL: COMPLETE, returncode 0 |
| forward-cpu | 369367263574 | JOB_FINAL: COMPLETE, returncode 0 |
| sweep-cpu | 369367263576 | JOB_FINAL: COMPLETE, returncode 0 |
| nonuniform-cpu | 369367263577 | JOB_FINAL: COMPLETE, returncode 0 |
| run-gpu | 369367263578 | JOB_FINAL: COMPLETE, returncode 1 |

Commands and last lines: `prepare_jobs.py run-cpu forward-cpu sweep-cpu nonuniform-cpu run-gpu` → `run-gpu sh -n OK`; `submit_one.py <entry>` → `RECORDED <entry> <id>` above; `job_runner.py <entry>` → `JOB_FINAL` above; `verify_observation.py` → `OPERATOR_PLANES_RECORDED 54`; `reduce.py` → counts in [artifact_verification.json](artifact_verification.json). Exact invocation records: [COMMANDS.md](COMMANDS.md).
The four 64-step CPU observation checks have maximum probe difference 0 V/m; the original and plain arrays are retained. Provider logs are under `jobs/<entry>/provider.log`. Each job was launched once; no repository files were edited and no push, PR or GitHub comment was made.
The detached worktree was removed; command exit 0, path and registration absent ([cleanup.json](cleanup.json)). All written measurement artifacts are under B0c/.

Conclusion: leader fills.

## GPU arm (relaunch)

The same 24 x 20 x 4 mm vacuum cavity was excited and probed at the specified Ez locations, with x faces declared PMC and y/z faces PEC, without absorbers. The supplied formula gives f01=7.494811450 GHz and f11=9.756058116/9.932091974 GHz for a'=24/23 mm. No mode was returned in the fixed (0,1) window for any of the three meshes; the returned (1,1) records are below.
Run `369367263579` completed, returncode 0; the single corrected job uses `JAX_PLATFORMS=cuda,cpu`. Each trace records `precompute_coeffs(pec_axes="xyz")` and `update_he_fast`; all eight call events equal the corresponding original GPU trace.

| Entry | dx mm | f01 GHz; error; Q | f11 GHz; error; Q | derived a' mm | (0,1) appears | Operators |
|---|---:|---|---|---:|---|---|
| run-gpu | 1 | none returned | 9.752832534; 2.16e-08; 2.704e+07 | 24.019381565 | no | B(xyz); update_he_fast |
| run-gpu | 0.5 | none returned | 9.755252299; 5.419e-08; 1.132e+07 | 24.004838121 | no | B(xyz); update_he_fast |
| run-gpu | 0.25 | none returned | 9.755856520; 8.679e-08; 7.069e+06 | 24.001210144 | no | B(xyz); update_he_fast |

Derived: a'=1/sqrt((2*f11/c)^2-(1/b)^2), b=20 mm, c=299792458 m/s. Error and Q are raw Harminv fields; error is dimensionless. Detection windows remain (0,1) 6.8-8.2 GHz and (1,1) 9.0-10.7 GHz; B has the coefficient zero planes defined above.
For dx 1/0.5/0.25 mm: 25x21x5 / 49x41x9 / 97x81x17 nodes, 24/20/4 mm spans, all pads zero, 4096/8192/16384 steps. Preflight stdout verbatim for each: `  [PREFLIGHT] All checks passed.` Return representation: `[]`.
Energy final/post-source-peak at dx 1/0.5/0.25 mm: -0.333513903 / -0.203203843 / -0.106878101 dB; full energy arrays and peak/final joules are retained.
Evidence and commands: [relaunch details](gpu_callback_recovery/DETAILS.md), [full records](gpu_callback_recovery/RESULTS.json), [provider log](gpu_callback_recovery/jobs/run-gpu/provider.log). `gpu_recovery_runner.py` ended `JOB_FINAL` COMPLETE, returncode 0; source export and original artifacts verified unchanged. All three meshes ran once in this one relaunch.
Conclusion: leader fills.
