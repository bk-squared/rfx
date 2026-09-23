# GPU cavity relaunch records

The 24 x 20 x 4 mm vacuum box was excited by an Ez differentiated Gaussian at (7,6,2) mm, f0=8.5 GHz, fractional bandwidth 0.8, and probed at (17,13,2) mm. The declared x faces are PMC and y/z faces PEC, without absorbers; the supplied formula gives f01=7.494811450 GHz. No returned Harminv mode lies in the fixed 6.8-8.2 GHz window in these three records.

The following table records the volume-summed energy at stored staggered field levels: 0.5 sum(epsilon0 |E|^2 + mu0 |H|^2) dx^3. Peak means the maximum after the source ends; no decay or boundary interpretation is assigned.

| dx mm | Post-source peak J | Final J | Final/peak dB | Raw JSON (NPZ and log share the stem) |
|---|---:|---:|---:|---|
| 1 | 1.0527097102e-21 | 9.74893643231e-22 | -0.333513903 | [run-gpu__1.json](witness/run-gpu__1.json) |
| 0.5 | 6.36800708219e-23 | 6.07691488381e-23 | -0.203203843 | [run-gpu__0p5.json](witness/run-gpu__0p5.json) |
| 0.25 | 3.92132185249e-24 | 3.82599747843e-24 | -0.106878101 | [run-gpu__0p25.json](witness/run-gpu__0p25.json) |

FACT: each trace records precompute_coeffs(pec_axes="xyz") at rfx/core/yee.py:604, called from simulation.py:2554, and update_he_fast at rfx/core/yee.py:718, called from simulation.py:1714. The eight recorded call events equal the original failed GPU trace at each dx; no apply_pmc_faces, apply_pec or apply_pec_faces call was recorded.
FACT: all 3003 source-export files match source_manifest.json, and all original report, measurement and job artifacts match their prelaunch hashes. No source files were edited; the prepared job sets JAX_PLATFORMS=cuda,cpu. The existing export was reused without creating a worktree.
Derived quantity: a' is calculated from the supplied frequency inversion with b=20 mm and c=299792458 m/s.

| Command | Last line / status |
|---|---|
| Local source/script checks, CPU callback smoke and YAML block `sh -n` | `RELAUNCH_PRECHECK source_files 3003 mismatches 0 script_hashes 4 CPU callback OK; sh -n OK` |
| `vessl run list` before submission | `PRESUBMIT listed; five cited B0c reference runs retained with existing nonempty provider logs; no active B0c run` (filtered listing and retention record retained) |
| `vessl run create -f /root/workspace/bk-workspace/.boundary-model/B0c/gpu_callback_recovery/vessl_run-gpu.yaml` (exclusive marker before the single invocation) | `RECORDED run-gpu relaunch 369367263579`; exit 0 |
| VESSL: `python -B /root/workspace/bk-workspace/.boundary-model/B0c/gpu_recovery_runner.py` | `JOB_FINAL` status COMPLETE, returncode 0, all three subprocesses exit 0; exact final JSON is in jobs/run-gpu/provider.log |
| `vessl run logs 369367263579 --tail 100000` | exit 0; nonempty jobs/run-gpu/provider.log |
| `vessl run read 369367263579` | exit 0; Status completed, remilab-c0, gpu-rtx4090, 14 CPU / 48 GiB / 1 GPU |
| `/root/workspace/bk-workspace/rfx/.venv/bin/python -B /root/workspace/bk-workspace/.boundary-model/B0c/gpu_callback_recovery/reduce_relaunch.py` | `RELAUNCH_VERIFIED 3 GPU records; traces unchanged; report draft 59 lines` |

Full precision frequencies, errors, Q, signed decay, grid coordinates, pads, source/probe indices, energy and call events: [RESULTS.json](RESULTS.json). Original-report bytes: [REPORT.before-relaunch.md](REPORT.before-relaunch.md). Submission marker, provider response and run id: jobs/run-gpu/. The measurement code and solver export are byte-identical to those used for the original GPU job.

Conclusion: leader fills.
