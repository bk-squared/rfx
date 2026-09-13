# CPML coefficient placement on Yee nodes

Ported from research branch commit 7b6c33d5 (2026-09-13, omo pod lane,
`accel/pod-distributed-usable-20260912`). The two tests were relocated into
main's layout (`tests/unit/boundaries/test_cpml_yee_stagger.py`,
`tests/unit/runners/test_distributed_cpml_stagger.py`); the NU arm of the
distributed test now enters a named-axis map because main gates the x faces
with `lax.axis_index("x")` instead of the research branch's `rank_index=`
kwarg (a separate lane). No solver change was made during the port.

The GPU evidence below was produced on the research branch and is cited by
digest, not re-run here. Experiment68 (pre-fix): solver source
7bc0e275b1ca741d506ac0da40be3ecbc0dd7145, input manifest SHA256
81c4745c8c529e6b45943e421cf3afd79b8e53bcb286373814394bfc9f4eb329,
authenticated receipt
b2b49489195e8d27f2903b08a97e46d922ac3474559ea0f1f5bc8b54b116697f.
Experiment69 (post-fix, this change): solver source
7b6c33d52e4bc199de90df5ee43adddb35d310af, input manifest SHA256
9bb2a2300f460f0f11f353f22238970078233cd48d3a37ad6b2661283c46b675,
source receipt
b9dee996c992046cc68c783666ed1dd5d5edb24a4050772e69d9534acb914a0b.
Against the fixed 5% far-channel requirement the far-Hy full peak-relative
residual was 0.13937570732 (68) and 0.000388194010609 (69). No GPU run has
been made from this branch; `scripts/cpml_dipole_waveform_witness.py` and
`scripts/vessl_cpml_stagger_witness.yaml` exist to reproduce both arms.

Experiment68's independently bounded infinite-lattice comparison rejects
lattice/source-table-only sufficiency. A separate operator inspection then
identified incorrect coefficient placement: E and transverse H reused the
same integer-sampled CPML profile despite their half-cell displacement.
E reflection maps i to N-1-i; H maps i to N-2-i. This is a coordinate
invariant, not an absorber-strength fit.

Before edits, actual native CPML one-update tests passed all three E
mirrors and failed all three H mirrors. The correction adds explicit
magnetic profiles sampled at +0.5 on low faces and -0.5 before high-face
reversal. Original E coefficient computation is unchanged. Profile support
is clipped, and existing field/psi slicing and material factors are retained.

The uniform distributed initializer/applier and its v2 wrappers receive the
same profiles. An independent review found the separate NU distributed
applier still selected E coefficients; new three-axis RED tests reproduced
that mismatch before selecting the explicit six magnetic face profiles.
True NU distributed material/AD regressions subsequently passed.

One-active-layer faces retain their original single-sample coefficients:
there is no resolved grading interval. New stagger-matching claims require
at least two active layers. This avoids the new formula inventing a
different origin for that legacy configuration, but does not qualify a
one-layer matched absorber. Explicit legacy CPMLParams/AxisParams without
magnetic profiles retain their legacy behavior.

## Evidence

Everything in this section is **ported from research commit 7b6c33d5 and not
re-measured on this branch.** The counts and timings below -- 136 passed, the
5 RED to 16 pass remedy count, the 45-test compatibility check at 67.59 s, the
20 byte-identical NPY pairs -- were recorded on
`accel/pod-distributed-usable-20260912`, and this bundle carries **no committed
receipt** for any of them; the same holds for the infinite-lattice assertion in
the header above. They are kept as the research lane's own record, reported and
not gated, rather than deleted. What is gated on this branch is the
operator-placement claim itself
(`tests/unit/boundaries/test_cpml_yee_stagger.py` -- the exact mirror residual,
plus the absolute orientation pin and the swap falsifier added 2026-09-14 after
an independent review showed the mirrors alone pass on a reversed pair) and the
GPU-witness section below, where every number resolves to a committed receipt.

- Native3H RED/3E pass to6pass; distributed3RED to combined9pass.
- Initial broad related CPU suite:136passed,1marker-deselected,51warnings.
  Warnings were read: existing amplitude/preflight/grading advisories and
  onset warnings remain visible.
- Review remedies:5RED to16pass, including NU forward/AD;5existing warnings.
- Actual small lossy CPML public native and N2 CLI: comparison_pass.
  All25artifacts per run verified; all20NPY pairs byte-identical.
- Actual one-update legacy-profile toggle: each axis mirror residual
  0.0002238516463 A/m before, exactly0 after. Complete signed corrections
  and residuals were rendered and inspected in canonical shared evidence
  `cpml-stagger-surface/operator-toggle.png`; raw NPZ retained.

Ruff passes. CPML module/new tests pass mypy; one uniform distributed port
error and five NU type errors reproduce on byte-identical pre-edit files
in the preserved worktree. They are not suppressed or called new failures.
LSP remains unavailable. Final compatibility checks pass45tests
(1marker deselection,8existing warnings) in67.59s; Ruff/mypy on the clean
three-file subset pass and wheel/sdist build exits0.

No full corrected GPU trajectory has run. This increment establishes
operator placement and CPU integration, not removal of all Experiment68
waveform error. The full5%/late1% requirements and native/distributed1e-6/0
contracts remain frozen. No other numerical lane or old STOP is promoted.

## GPU witness (2026-09-14, remilab-c0)

The section above says no GPU run had been made from this branch. That is now
superseded: the two-arm witness ran as VESSL run 369367260765
(`cpml-stagger-witness-r1`) on cluster remilab-c0, preset `gpu-rtx4090`, image
`nvcr.io/nvidia/jax:24.10-py3`, 2026-09-14 01:02-01:03 KST, job exit 0. The
receipts are committed under `docs/research_notes/cpml_stagger/gpu-369367260765/`:
`manifest.json` lists every committed file with its SHA256, `launch.json` and
`environment.json` carry the run provenance (jax 0.4.33.dev20241023+e3c6d6430,
Python 3.10.12, one NVIDIA GeForce RTX 4090, float32), `run.yaml` is the
submitted spec, and the full job log is
`docs/research_notes/vessl_logs/369367260765-cpml-stagger-witness-completed.log.gz`.

Both arms ran inside one job, on one GPU, against one comparator. The GREEN
tree's `scripts/diagnostics/cpml_dipole_waveform_witness.py` was copied into the
RED tree before either arm ran, so the only difference between the arms is the
three CPML solver files; the two identical digests
(c48933f56f26a5057f90365dd151f6cfe91b79b6f432d420a0b9b5fc82279834) are in
`artifacts/comparator.sha256`. The arms are RED = fa3929136159e0a644783b56012b21c1bce21340
(main, pre-fix) and GREEN = 47a586b136c7a2cf68595f6058bd3e456354d013 (this
branch); `artifacts/commits.txt` holds both. Tolerances were predeclared and
fixed, not fitted: full
`docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/green_witness.json::full_tolerance = 0.05`,
late
`docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/green_witness.json::late_tolerance = 0.01`,
source
`docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/green_witness.json::source_tolerance = 1e-05`.

Full-window peak-relative residual per channel, read out of the committed
receipts rather than retyped from the job log:

| channel | RED (fa392913, main) | GREEN (47a586b1, this branch) |
| --- | --- | --- |
| `middle_ez` | `docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/red_witness.json::channels.middle_ez.full_relative_max = 0.0391754` | `docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/green_witness.json::channels.middle_ez.full_relative_max = 0.00430682` |
| `middle_hy` | `docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/red_witness.json::channels.middle_hy.full_relative_max = 0.037326776` | `docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/green_witness.json::channels.middle_hy.full_relative_max = 0.00148394` |
| `far_ez` | `docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/red_witness.json::channels.far_ez.full_relative_max = 0.0404082` | `docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/green_witness.json::channels.far_ez.full_relative_max = 0.0004422755` |
| `far_hy` | `docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/red_witness.json::channels.far_hy.full_relative_max = 0.13937570731810509` FAIL | `docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/green_witness.json::channels.far_hy.full_relative_max = 0.0003881940106090643` PASS |

RED's failure is far-Hy in the full window only: its three other channels pass
the 5% requirement, its source check passes
(`docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/red_witness.json::source_relative_peak_error = 6.5534537e-07`,
the same value GREEN reports), and its late window passes on all four channels
(`docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/red_witness.json::channels.far_hy.late_relative_max = 0.00125`).
So the arms are separated by one channel, not by a broken run, and that
separation is **consistent with** the operator-placement argument rather than
predicted by it: far-Hy falls 359x with the bulk and source code unchanged
between the arms, which implicates CPML placement. The mirror derivation
establishes that E and transverse H must be sampled half a cell apart; it does
not on its own say which observation channel a violation shows up in, so this
is corroboration, not a channel-level prediction the run confirmed. GREEN passes every full and late
requirement; its worst late channel is
`docs/research_notes/cpml_stagger/gpu-369367260765/artifacts/green_witness.json::channels.far_ez.late_relative_max = 0.000174`.
The witness verdicts as printed per arm are in `artifacts/summary.txt`,
`artifacts/red_witness.log` and `artifacts/green_witness.log`.

This reproduces the research-branch numbers on different hardware. The pod's
Experiment 68 (pre-fix, A6000) reported far-Hy 0.13938; RED here gives
0.13937570731810509 on an RTX 4090 -- agreement to four digits. Experiment 69
(post-fix) reported 0.000388; GREEN gives 0.0003881940106090643. The digest-cited
Experiment 68/69 receipts in the header above are therefore not the only GPU
evidence for this fix any more, and the residual is not a property of one GPU.

GREEN's lane tests were re-run on the GPU in the same job: 14 passed
(`artifacts/green_lane_tests.log`). RED's lane-test log is a collection error,
not a result -- the two test files do not exist on main -- so the witness
residual, not a test count, is what separates the arms
(`artifacts/red_lane_tests.log`).

Scope, unchanged from the witness spec's own declaration: a PASS here is a
bounded total-waveform observation only. It does not separate bulk, source-cell
and CPML error; it does not certify a reflection coefficient
(`bulk_error_certified`, `cpml_reflection_coefficient_certified` and
`onset_impulse_resolved` are all false in both receipts); and it qualifies no
other mesh, angle or material than the one fixture it ran -- 96x72x60, h=1/512,
dt=3.723779042052746e-12, 1536 steps, 12 CPML layers, kappa_max=1, vacuum,
float32. The fixed 5%/1% requirements and the native/distributed contracts
remain frozen; no other numerical lane or old STOP is promoted by this run.

The two raw trace arrays are deliberately NOT committed. They are recorded by
digest in `manifest.json` under `not_committed`: red_trace.npy sha256
8390e4cef6a2d01269eadb60d55cd61ae5ac9ecc7a6c8a1ca370440ec60c43bb, green_trace.npy
sha256 d179ac30d0c2ba91f8ae54a84546cedbd48d52d61f17307e28be291c344fe283 (110720
bytes each). Every scalar **this GPU-witness section** cites lives in the two
witness JSONs, which also carry the per-channel reference/observed/residual
series the traces hold. That claim is scoped to this section only: the Evidence
section above is the ported research-lane record and has no receipt in this
bundle, as its own disclaimer says.
The originals are retained on remilab-fs at
`personal-workspaces/claude-workspace/rfx/runs/cpml-stagger-20260913T160213Z/`.

## Pre-existing limitation (out of scope)

The NU distributed applier synthesizes the x and y high-face **E** profiles by
flipping the corresponding low-face ones -- `b_xr`/`c_xr`/`k_xr` at
`rfx/runners/distributed_nu.py` line 1466 and `b_yr`/`c_yr`/`k_yr` at line 1472
(grep the names; the line numbers rot) -- so it does not honor an arbitrary
explicit high-face E profile;
the public NU adapter also does not forward arbitrary per-face counts. An
independent review measured this directly on a kernel probe -- h = 1/512,
budget 4, kappa_max 3, face counts (1, 4, 2, 3, 3, 1), random unit-scale
fields, seed 5 -- and found a native/NU maximum Ey difference of 1122.99 V/m
while the H differences stayed at or below 2.38e-7 A/m, i.e. float32 arithmetic
noise; with every face count set to 1 the H differences stayed at that same
floor. Those numbers are the reviewer's measurement, reported and not gated:
they have no committed receipt here.

This behavior **predates this lane** and is untouched by it. What it limits is
any general per-face **E** claim on the NU distributed path; it does not touch
the corrected magnetic-profile selection, which is what this note qualifies --
the H differences above are the evidence that the H selection is unaffected.
Fixing the E synthesis is a separate change on a separate lane.
