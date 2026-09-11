# Issue #729 site 3: source span exceeds the actual conductor interval

Status: diagnosis and partial correction, not ready to merge. Baseline main is
f85ed767. These are static profile/load checks, not fresh FDTD S-parameter or
line-input-impedance measurements. The PI's out-of-range declaration-versus-
actual-wall policy is pending.

## Physical derivation and reproduction

For substrate-normal Ez, index k denotes the primal edge [z_k,z_(k+1)]. Ground
at node g and trace at node t contain edges g <= k < t, exactly t-g cells.
The production voltage extractor already uses that exclusive trace bound.
The old source profile instead used t-g+1 cells and normalized over all of
them. Its Laplace Dirichlet trace was also one coarse cell too high. This
changes the source shape and the load sigma, not just a reported scalar.

`rfx729-local.py` constructs an aligned 254 um, eps=3.66 substrate, 8-cell-wide
sheet trace and a sheet ground. The #931 realized wall query independently
returns planes 8 and 12. The baseline profile occupies indices 8..12 and
claims five cells / 317.5 um. The actual four-cell interval carries only
0.7331298467 V although the full profile integrates to 1 V. With its generated
sigma, V_actual^2 / integral(sigma |E|^2 dV) is 26.8739676 ohm, rather than the
specified 50 ohm. This is a modal-load consistency calculation, not an RF
input impedance extracted from a propagating FDTD line.

The count-only temporary prototype uses four cells: source indices 8..11,
V_actual=1 V, V_actual^2/P=50.0000006 ohm. Its z0_static changes from 60.7307 to
53.0136 ohm. The old and new scalar values are recorded, not used as physical
acceptance thresholds. The narrow prototype can be recreated from the base
source by replacing the single `n_z_sub = n_hi - n_lo + 1` calculation with
`n_hi - n_lo`; its hash is in receipt.json.

The current worktree also corrects the legacy uniform source/termination and
PEC-clearing edge list. `msl_cross_section_span.n_hi` remains the upper NODE
index so existing trace searches and voltage metadata keep their meaning;
`cells` now excludes the Ez edge above that plane. Width-axis nodes remain
inclusive. No blanket removal of +1 from width quantities is intended.

## Tests, before touching the historical profile pins

The new six tests obtain ground/trace bounds from the realized PEC edge set,
not from the mode profile. They cover ground offsets, three substrate cell
counts and +/-x,+/-y. They assert source support, actual-wall voltage and the
existing physical load identity, including the uniform primitive. All six
fail on baseline and pass with the core span correction.

The existing port/dual-metric/axis selection yields 104 passed, 2 failed,
2 deselected, 1 xfailed after the core correction. The two failures are:

1. test_msl_yz_cells_covers_full_cross_section asserts the old inclusive
   normal endpoint. That asserted convention conflicts with Ez's edge support.
2. test_msl_mode_profile_on_a_uniform_grid_is_bit_identical_to_the_scalar_form
   pins three moments of the old, too-tall mode. The observed first sum is
   73361.79312530009 -> 69888.63021121288. This is a visible geometry/source
   correction, not reduction-order noise. This root cause is recorded BEFORE
   changing or re-pinning that test. The historical number must not be treated
   as an oracle for the corrected physical span.

The NU 1V and V^2/P tests stay important, but currently choose their upper
integration bound from the profile's own length, making them blind to the
original excessive support. Independent physical wall bounds must supplement
those checks. Primal normal lengths and transverse dual Joule volumes stay.

## Remaining validation and scope

The affected production routes are uniform run, uniform forward (also MSL and
mixed S extraction/topology), nonuniform run/forward (even mode='uniform'
uses a Laplace profile there), and the direct coax-to-MSL driver. Both driven
and passive ports change. General runner clearing reads the edge-list helper;
the coax transition uses its own direct scan. Their actual source/termination
support and tangential PEC preservation must be checked explicitly.

Physical S/settling/refinement checks and current before/after records are
still required; no old gate is widened to absorb this change. The previous
review's off-tree 95/98-minute run estimates do not determine the new campaign.
Use small independent invariants first, then run each needed physical case
once on matching current-main geometry, preserving the raw records.

No claim is made here that a scalar-spacing Laplace box is an exact
nonuniform-cross-section eigensolver, or that repairing this count closes the
mixed-port calibration or all other MSL issues.
