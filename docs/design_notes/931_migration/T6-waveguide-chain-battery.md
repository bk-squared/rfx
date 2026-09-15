# T6 handover — the waveguide chain battery's device lane changed operator

Owner of the fix: whoever owns `scripts/diagnostics/waveguide_chain_battery_measure.py`
and `tests/fixtures/waveguide_chain_battery/` (scripts / docs group + the ingest
phase). T6 measured it and is handing over the number, not the re-pin.

## What was measured

`tests/oracle/test_waveguide_chain_battery_v18_close.py::test_live_cells_reproduce_the_fixture_cpu`,
coarse and mid rungs, on the T6 worktree at `b884b83f`, CPU, shared pod:

```
[live thru-coarse-false]      max|S_live - S_fixture| = 1.193e-06
[live thru-coarse-flux]       max|S_live - S_fixture| = 2.455e-06
[live pec_short-coarse-false] max|S_live - S_fixture| = 9.381e-01
[live thru-mid-false]         max|S_live - S_fixture| = 1.792e-06
[live thru-mid-flux]          max|S_live - S_fixture| = 1.199e-06
[live pec_short-mid-false]    max|S_live - S_fixture| = 4.979e-01
```

`LIVE_ABS_S_TOL = gate_from_envelope(5.000e-6, quantum=10000) = 1e-4`.

## What it means

The empty guide reproduces to 2.5e-6, so the port, the absorber, the extraction
and the whole plane machinery are exactly where they were. The entire delta is
the `pec_short` DUT.

That DUT is built in `tests/_waveguide_chain_battery_fixture.py` as a Box of
material `pec_like` (eps_r 1, sigma 1e10) spanning the full cross-section, and
`_assemble_materials` moves anything above the PEC threshold OUT of
`materials.sigma` and INTO `pec_mask` — the file's own comment says so. The
waveguide S-matrix lane then used to fold that mask BACK into a sigma = 1e10
cell fill for the device run (`rfx/sources/waveguide_port.py`). Stage C
(`0184d64c`, "the S-matrix lane applies the realized PEC edges instead of a
sigma=1e10 cell fill") replaced the fold with the realized edges.

A hard electric wall and a 1e10 S/m lossy volume are different operators. Both
reflect with magnitude ~1; they do not reflect with the same phase, and 0.938 is
the size of that rotation on a unit-magnitude S11 — not a tolerance failure and
not noise.

Note this is NOT the §1.2 far-face change. The short spans the full cross
section, so under the old rule its tangential walls already stood on every
masked cell plane and its LEADING face — the one the incident wave meets — did
not move. The operator changed, not the geometry.

## What must happen, and what must not

Per the inventory's own instruction for this row: **do not widen
`LIVE_ABS_S_TOL`.** The family needs a fourth pre-declared measurement run:

1. add a section to `docs/design_notes/waveguide_chain_battery_predeclaration.md`
   recording that the device lane's operator changed at `0184d64c`, with the
   0.938 above as the measured size, BEFORE the run;
2. re-measure through `scripts/diagnostics/waveguide_chain_battery_measure.py`
   on the contract build (the producer and its VESSL yaml are both in tree);
3. write a new fixture rather than editing `fixture_v18_close.json` — the
   filename and its VESSL id are the provenance, and mixing operators inside one
   artifact is what the ladder rule forbids;
4. re-point the three live tests at the new artifact, as was already done once
   when PR #889 moved the port aperture.

`test_waveguide_chain_battery_guide_cell_aperture.py` is unaffected: its §7
discriminator is about the port's transverse eigenproblem, which the empty-guide
agreement above shows is intact.

## Ingest follow-through — 2026-09-08

The PI-adjudicated run **369367259427** is now ingested byte-for-byte as
`fixture_931_realized_pec_forward2_run369367259427.json`, and the live cell
comparisons use it. The old sigma-fill artifact remains historical. See the
[ingest record](T6-chain-fixture-ingest-20260908.md) for provenance, CPU validation,
and the **unexplained slab coarse/flux drift (1.042e-5 against the old fixture)**.
The fine GPU live test still requires its own run; this ingest claims no GPU pass.

## Why the sigma fold is worth a second look while this is open

Design note §1.8 fences sigma fills out of the ownership contract, and
`tests/contracts/test_lattice_ownership_contract.py::test_a_sigma_fill_conductor_is_not_a_pec_body`
now pins that the two models are not equated. But the chain battery reaches the
fence from the other side: a `pec_like` MATERIAL on a geometry entry crosses the
1e6 threshold in `rfx/api/_compile.py` and becomes a PEC body, while a raw
`rasterize(grid, [(shape, eps, 1e10)])` stays a sigma fill. Two files that both
say "PEC" therefore mean different operators, and which one you get depends on
whether the conductivity arrived through a material or through a rasterize call.
That distinction is real and deliberate, but nothing states it where a fixture
author would read it. Worth one sentence in the design note's §1.8.

## The pre-declared thickness separation ran (2026-09-07, VESSL 369367259233)

`scripts/vessl_931/pec_short_thickness_sweep.py` re-solved
`test_pec_short_s11_magnitude`'s fixture at SHORT_CELLS = 1, 2 and 4. Leading
face at 85.655 mm in all three arms; only the thickness changes.

```
SHORT_CELLS=1  thickness 2.1414 mm  min|S11| 0.95721  mean 0.97479
SHORT_CELLS=2  thickness 4.2827 mm  min|S11| 0.96705  mean 0.98188   <- the module's own
SHORT_CELLS=4  thickness 8.5655 mm  min|S11| 0.97607  mean 0.98637
spread across thickness = 0.01886
```

The 2-cell arm reproduces the earlier standalone measurement (0.9670) exactly,
so the sweep and the gate are reading the same thing.

**Read it as physics, not as a curve.** Everything behind a total reflector's
leading face is dark, so |S11| CANNOT depend on how many cells of PEC sit
behind that face. It does, monotonically, and it rises toward 1 as the stack
gets thicker. That is not "the redraw moved the reflector by half a cell" — it
says the realized short is **not opaque**, and leaks less the more of it there
is. The deficit also does not close: 8.57 mm of solid PEC still reads 0.97607
against the 0.99 gate, and the thickest arm's first bin reads |S11| = 1.00467,
which is not passive either.

So the pre-declared separation did not come out as a clean either/or, and the
honest conclusion is the one it rules OUT: re-pinning this module at the
thickness it happens to declare would pin a leak. The gate stays red and
un-widened.

**What this leaves for the core/ingest pass.** Same class as the `pec_short`
DUT above (max|dS| 0.938 coarse / 0.498 mid while the empty guide reproduced to
2.5e-6). The cheap next witness, not run here because it is a core-side
question rather than a fixture one: put the same short in as a domain-boundary
PEC face (`BoundarySpec`), which design note §1.8 fences OUT of the ownership
contract and leaves unchanged. If the BoundarySpec wall closes the guide to
|S11| >= 0.99 while the body-realized wall does not, the leak is in the
body-realization or in the S-matrix lane that reads it, and the guide, the
port and the extraction are all cleared in one measurement.
