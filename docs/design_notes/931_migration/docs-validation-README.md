# For the ingest agent — `validation/README.md`, rows owned by other #931 groups

The Docs group edited exactly ONE row of `validation/README.md` directly: the
**cv15** row (the critic assigned it; no other group had it). The three rows
below state the pre-2.0 realization as current fact, but they are the subject
matter of the crossval groups that are re-solving those cases, so the exact
numbers must come from THEIR recomputed runs, not from a docs edit. Apply
these as text once each owner's numbers land.

---

## cv06b row (`crossval/06b_msl_notch_filter_uniform.py`) — owner: crossval-B

**APPLIED 2026-09-07 (ingest).** The wording below landed with the merge; the
numbers landed after it, from crossval-B's run 369367259191 — 2.16 % / −39.4 dB
/ 48.2 Ω, with the run log committed as
`_06b_notch_uniform_logs/20260907T124851Z_run.log`. The width convention is NOT
settled by that run (Z0 median 48.2 Ω against a pre-declared 46.48 ± 1.0 Ω) and
the carriers say so; the regenerated falsifier summary and estimator fixture are
still crossval-B's to commit.

Current text contains:

> it shipped at dx=80 µm through 2026-08, where the declared 254 µm substrate
> rasterized to 320 µm

Replace with:

> it shipped at dx = 80 µm through 2026-08, where the declared 254 µm
> substrate rasterized to 320 µm — a **pre-2.0** realization. From 2.0 (#931)
> a conductor is declared a volume, a sheet or a wire and drawn extent equals
> realized extent, so the same drawing on the same mesh no longer produces
> that 320 µm board; what remains is that 254 µm is not an integer number of
> 80 µm cells, which is why the case ships at dx = 63.5 µm = h_sub/4.

and, in the same row, replace

> The analytic reference is evaluated on the realized 635.0 µm trace width,
> read live from `sim.fidelity_report()`.

with the same sentence plus:

> (Under the contract a sheet footprint is sampled **closed** on the in-plane
> axes, so the realized width is re-read from the recomputed run's own
> `fidelity_report()` rather than carried over.)

**Do not hand-edit 635.0.** It comes from the recomputed run.

---

## cv18 row (`crossval/18_wr90_iris_modematch.py`) — owner: crossval-D

**WORDING APPLIED, DIGITS PENDING (2026-09-07 ingest).** cv18/cv19 pass 2 has not
run, so every gate constant in the row is still the pre-2.0 one; the row now says
that and states the re-derivation rule — round-UP(envelope × 1.5), the only
arithmetic allowed to move those digits.

Current text contains:

> Three setup defects (parasitic wall-slot, half-ulp node-plane corners, an
> electrical aperture of d + 2·dx together with a too-thin absorber) were
> found and are each fenced

Replace the parenthesis with:

> (parasitic wall-slot, half-ulp node-plane corners, an electrical aperture of
> d + 2·dx together with a too-thin absorber — the last of these is a
> **pre-2.0** defect: under the lattice ownership contract a PEC volume
> realizes walls at both drawn faces, so the electrical aperture is the drawn
> `d` and the `+ 2·dx` fence is retired rather than re-tuned)

and re-read every gate number in the row from crossval-D's recomputed
`aperture_resolution.json`; the per-configuration windows were derived from
runs on the old realization.

---

## cv20 row (`crossval/20_msl_phase_referee.py`) — owner: crossval-E

**APPLIED 2026-09-07 (ingest).** Both carriers — `validation/README.md` and
`docs/guides/sparameter_support_matrix.md` — now label 0.94 % / 0.31 % as a
pre-2.0 realization re-measured under #931. The new digits wait on cv20's Stage B
re-run.

The row quotes `beta` residuals measured with the trace as a one-cell PEC Box
("the realized board"). crossval-E migrates that trace to a sheet, which moves
the realized board, so `0.94% rfx` and `0.31% openEMS` are re-read from the
recomputed run. Add "(pre-2.0 realization; re-measured under #931)" to the
measured-values parenthesis until the new numbers land.

---

## Not changed anywhere in this file

The dielectric-only rows — cv01, cv02, cv03, cv04, cv17, cv22, cv23 — are
untouched by #931 on purpose: dielectric sampling is unchanged (node
coordinates, half-open), and their bit-identity is the change's own falsifier.
If any of those rows needs an edit, something is wrong with the change, not
with the row.
