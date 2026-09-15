# thru feed-post two-segment lane — post-#931 record

One file per run, named by its VESSL run id, copied verbatim from the harvest
directory. Nothing here is retyped or edited.

| file | run | arm | rc |
|---|---|---|---|
| `post931_verify_369367259204.log` | 369367259204 | `--verify` (predeclaration section 9) | 0 |

## What the record shows

`--verify` is the only arm this lane's VESSL yaml implements
(`validation/research/thru_feedpost_singlepost_vessl.yaml`; its fine-dx arms
are reserved names, not code). It is **pure synthetic apparatus algebra** — no
`Simulation`, no grid, no FDTD — so it is realization-independent: it is the
same check before and after the lattice ownership contract, and it is not
evidence about the post-#931 board.

That is exactly why it is worth committing: it is the regression witness that
the identification apparatus still inverts its own generator after the branch's
changes to `build_thru` / `build_singlepost`. All of V0-V5 pass, at the same
1e-16-class residuals as before:

    generator x-check vs rfx.deembed inverse  max delta 6.47e-16
    V1 thru fit (0.38000000 nH, 4.00000000 ps) err 2.22e-16
    V2b attempt-3 pipeline recovers truth     err 2.22e-16 <= 1%
    V3b synthetic band arm F-D1 fires         True
    APPARATUS VERIFICATION: ALL PASS

## What it does NOT decide

No pre-declared window in
`docs/design_notes/thru_feedpost_twoseg_predeclaration.md` is re-derived by this
run, and none is edited in code. The measured line constants those windows are
built from come from the `--extract` arm, which this lane did not submit, and
the windows' own input is #313 instrument provenance measured on a DIFFERENT
fixture — see `docs/design_notes/931_migration/XE-windows-2b.md` for the
procedure, the arithmetic that reproduces the committed window from it, and why
the input is stale but must not be replaced with this fixture's own numbers
(the predeclaration's section-1 hygiene rule burns exactly that data).
