# Issue 752 re-review after absolute-face corrections

No remaining material blocker was found in the reviewed working source. The two original alignment refutations are closed. Final exact-head review, completion of the remaining required local checks and applicable CI are still required; this is not a claim that a quantitative Z0 campaign is complete.

The new face helper locates the absolute declared trace position against the canonical node line independently of epsilon-column availability. Material compatibility fields delegate to it. The reported conductor gap still requires valid attachment to original realized conductor geometry; no alternate-wall repair or override inspection was added. Uniform spacing hints are limited to an origin-zero ground; offset and NU hints name the absolute declared ground/trace planes.

Independent reviewer execution used `/usr/bin/python3.10` with `JAX_PLATFORMS=cpu` and `PYTHONPATH=/root/rfx-752`. It reran only the two original build-only counterexamples plus 150 additional pure face-location cases (uniform/NU, fractions 0, 0.125, 0.25, 0.5, 0.875), exact terminal-node checks, and out-of-range/NaN/Inf refusal checks. No full author suite, field simulation or RF campaign was rerun. The exact executed heredoc body is preserved as `absolute_face_recheck.py`; the observed stdout is transcribed from completed tool chunk 0645ce. It was not rerun just to package this record.

The offset-ground case now yields fraction 0.20000000000000018 and an absolute-plane alignment message. The valid NU/vacuum-material case yields fraction 0, retains NU identity, and emits no false scalar-fallback warning. No new endpoint/range/fraction error was observed on the additional cases. Exact-node right-bracket conventions do not change a warning because their fraction is zero.

A structural recursive comparison of the full fidelity snapshot against base19aa found exactly two changed leaves, both `message` strings under `variants/validation/tmtt_paper/msl_stub_notch_tuning.py::build_sim::f_target/preflight` at indices 1 and 2. Every numeric value, geometry/grid entry, key and other finding is unchanged. Old/new file hashes match `snapshot-change-final-receipt.json`.

The current preflight source SHA-256 is 0bac79f7528efe60a7035c1ab703c4727b98e8c7fc6ace5af714fb5d86d2d152, matching the parent's AD-audit source receipt. Parent AD/FD evidence remains scoped to its recorded tests and synthetic S-assembly; it is not new general Z0 accuracy evidence. The reviewer inspected that receipt without rerunning those field tests.

Archive CLI conclusions remain unchanged: default generation refuses before generator work; explicit inspection returns hash-verified historical records with current-validation false. Both frozen JSON hashes, original script gzip/source hashes and unchanged non-main numerical helper ASTs were verified in the earlier review. No new current-geometry/current-Z0 claim was found in the updated guide.
