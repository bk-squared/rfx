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
