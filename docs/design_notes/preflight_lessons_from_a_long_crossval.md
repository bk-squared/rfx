# What a long cross-validation taught us about preflight

Written 2026-08-22, from a multi-week cross-validation of an imported multi-layer board
against a commercial reference. No board-specific data appears here; every number is about
rfx's own diagnostics.

## The headline

**Preflight's problem was not too few checks. It was that findings had no aggregation, no
severity, no coverage statement, and no persistence.**

One real run emitted **84 advisories drawn from 6 distinct messages — 93 % duplication**:

```
x42  surface_impedance_f0 thin conductor #N: f0 is more than N% away from the source centre
x30  lossy thin conductor #N sits at z = N mm, an E node whose adjacent cells differ by N%
 x6  'metal' z-extent 17 um = 0.5 cells — below 1 cell resolution
 x4  'ro4450f' z-extent 17 um = 0.6 cells — below 1 cell resolution
 x1  'ro4003c' z-extent 17 um = 0.6 cells — below 1 cell resolution
 x1  PEC 'metal' x-extent N um = N cells — volume under-resolved
```

Another run repeated a single identical advisory **153 times**. The `x30` line above is the
graded-node advisory — genuinely relevant to that model, and invisible inside its own
repetition.

Separately, the one warning that would have redirected the whole investigation *did fire*
and was lost in the same stream: a reported port impedance deviating **46 %** from its
analytic anchor. It was correct, it was printed, and nobody read it for days.

## What actually went wrong, by class

### A. The check existed, was correct, and could not see the object

- The MSL downstream-reflector scan tests `isinstance(shape, Box) and material_name == "pec"`.
  Imported CAD registers every conductor under one name such as `"metal"` (PEC-promoted by
  sigma), and mesh-imported conductors are not `Box`es. The scan examined **zero** of the
  model's conductors and reported clean, while a probe sat at 45 % of the clearance the
  repo's own rule demands. (#685)
- After #677 a `surface_impedance_f0` sheet is a node-thin operator and appears in neither
  `pec_mask` nor `materials.sigma`. A conductor-connectivity check written the obvious way
  found nothing and declared a healthy model disconnected. (#695)

**The general defect: a guard that cannot evaluate a shape reports the same thing as a guard
that found nothing wrong.** Silence has two meanings and the caller cannot tell them apart.

### B. The number was reported and nobody read it

`sheet_residual_um = 8.500006` appeared in every saved sizing artifact of the campaign. It
is exactly half a copper thickness, and it is the signature of the registration defect that
turned out to inflate every cavity in the stack by 8 %. It was emitted for weeks.

`reliable = True` was set on 46 % of in-band bins whose fit was pinned at a scan limit. The
flag is a standing-wave-null mask, not a fit-quality flag — but it is named as though it
answers the question a reader is asking. (#681)

### C. Silence is indistinguishable from success

A blanket `warnings.filterwarnings("ignore")` in the harness muzzled every rfx honesty guard
for most of the campaign, and **nothing in any saved artifact recorded that the guards had
been silenced**. The runs looked identical to clean ones.

Related, outside preflight: two committed physics envelopes were red and default-deselected
by a pytest marker, so they read as coverage while asserting nothing. (#693)

### D. The quantity moved and the consumer did not follow

#672 moved the port current onto dual spacings and left the termination sigma on primal
(#688, #691). #677 moved the sheet out of the material arrays (#695). The memory estimate
reads the uniform grid on a non-uniform run and green-lit a job that OOM'd after queueing
(#696). In each case a concept has more than one spelling and the spellings drifted.

### E. The artifact's headline is not the statistic the reader needs

A summary artifact led with a band **minimum**. It was quoted as the configuration's
performance for the whole campaign; the band **mean** was 9.7 dB worse, and that one
substitution exonerated the wrong component and redirected weeks of work.

## What to change, in order of leverage

**1. Aggregate and rank before printing.** Group identical advisories (`42 conductors:
f0 is more than N% from the source centre`), and order the block by a severity the validator
declares rather than by registration order. This is the largest single win and the smallest
change: it takes a 84-line block to 6 lines and puts the load-bearing one where it is read.

**2. Every validator reports coverage, not just findings.** `reflector scan: examined 0 of 52
conductors (48 skipped: material name != 'pec'; 4 skipped: shape is not a Box)` would have
made #685 self-evident on first run. A validator that examined nothing must never contribute
to "All checks passed".

**3. Findings persist into the result, not only stdout.** Attach the structured findings to
the returned `Result` and let harnesses write them into their artifacts. A warnings filter,
a lost log, or a background job then cannot erase them — and a saved run carries the evidence
needed to re-read it later. This directly answers class C.

**4. Guards test the assembled state, never registration metadata.** One accessor per concept
— a conductor footprint that is `pec_mask | (sigma > threshold) | union(sheet masks)`, and
one answer to "which grid will the solve use" — used by rfx's own validators as well as
offered to callers. This is the structural fix for classes A and D.

**5. One new check, very specific, because it cost the most.** For every one-cell PEC sheet,
inspect the permittivity on its live normal edge. If that edge carries `eps_r = 1.0` while
the cells on both sides are dielectric, the model almost certainly has an unfilled gap at a
metal level, and it puts a vacuum layer in series across whatever cavity that sheet bounds.
In this campaign that single condition removed two resonances from the answer and was
invisible for weeks.

**6. Postflight deserves the same structure as preflight.** Several of the most misleading
conditions were only observable after the solve: a fit pinned at its scan bound, bins that
needed passivity projection, a settling witness that did not reach its bar. These are emitted
today as loose warnings. A structured *extraction health* block attached to the result —
what was measured, how it was measured, and which bins are not quotable — is worth more than
another entry in the pre-run banner.

**7. Make the estimate a real gate.** Size it from the grid the solve will use, include the
operators that allocate (the sheet context did not exist when the estimator was written), and
label which grid the number describes. An estimate that is silently low is worse than none,
because it is read as a green light.

**8. A finding must carry its basis, not just its verdict.** This one was pointed out while
the note was being written, and it is the principle the other seven serve.

A guard that says *"this is a known limitation, do not fix it with a dielectric box"* invites
the next reader either to ignore it or to distrust it, because nothing in the message lets
them check the claim. Compare the same finding written out:

```
NOTE driver-parasitic: boundary node k=157 carries eps_r = 1.000.
  OBSERVED  a 31.4 um cell of vacuum in series across this cavity, at its own boundary node.
  WHY       the mesher registers each 17 um sheet at its MID-PLANE, so this cell spans
            copper-zone + core with no air in it, but the rasteriser samples material at the
            cell's LOWER EDGE -- inside the copper zone -- and assigns eps_r = 1.
  COSTS     depends on the measure, so quote the measure. As a series capacitance
            (sum dz/eps_r, which governs coupling across a gap far shorter than a
            wavelength) this one cell makes the cavity look 17.3% wider and drops the
            coupling capacitance 14.8% -- weaker coupling, so a smaller mode split. As
            phase length (sum dz*sqrt(eps_r)) the same cell is worth only 3.2%. A reader
            who checks the phase measure alone will call this benign.
  DO NOT    fill it with a dielectric box. Three reasons, any one sufficient: the designer's
            stackup says this outer level IS vacuum; the STEP's own slab there is in the
            deliberate exclusion list for that reason; and the patches cover ~11% of the
            plane, so a full-board fill puts dielectric across ~89% of it where the board has
            air. This campaign did fill it, it was load-bearing for a reported result, and it
            was retracted.
  INSTEAD   this is a RASTERIZATION change -- sample the material where the live component
            actually sits (node + d/2 along the sheet normal), not at the node inside the
            copper. (The first draft of this note said "register the sheet on a FACE"; that
            was measured and refuted the next day -- the boundary tie resolves downward, PEC
            lands on the cell above, and the cavity gains a second vacuum cell. A guard's
            INSTEAD line is a claim like any other and ages like one.)
  WRONG IF  the level turns out to be buried after all, or the rasteriser stops sampling at
            the lower edge. Either makes this note obsolete rather than merely inconvenient.
```

Five things the second version has and the first does not: the **measurement**, the
**mechanism**, the **cost in the units the reader cares about**, the **alternative that is
actually legitimate**, and a **falsifier for the guard's own claim**. The last is the one most
often missing: a guard that cannot say what would prove it wrong is an assertion, not a
finding, and a reader has no way to tell a stale guard from a live one.

This is also the answer to "why not just add more checks". A check whose output a reader
cannot evaluate does not reduce the number of ways to be wrong; it adds one.

## What not to do

Do not add more checks to the banner. This campaign ran with a preflight suite of 41
validators and the failure mode was never a missing check — it was a correct check that could
not see, a correct warning that was buried, and a correct number that no artifact preserved.
Adding an eighty-fifth advisory to an eighty-four-advisory block makes the situation worse.

## The one-line version

Preflight should be judged on whether its output changes what the user does next. Measured
against that, an 84-line block with 6 distinct messages, no coverage statement, and no
persistence is failing regardless of how many conditions it correctly detects.


**9. What the mesh does to SYMMETRY is a first-class check, not an emergent surprise.**
The single most accurate configuration of the whole campaign was found not by fixing
physics but by sliding the lattice origin 13 um: a mirror-image conductor pair, identical
in the design, rasterized 173 vs 183 cells (5.6 %) because its mirror plane sat 0.26 cells
off the lattice — repeated with the same sign in every such pair on the board. An A/B run
pair differing ONLY by the slide moved |S11| up to 3.5 dB per bin and improved every
aggregate agreement metric against the external reference. Everything about this was
computable before the first time step: congruence grouping is arithmetic on Box extents,
the cell counts already exist in the assembly masks, and the best origin shift is a
one-dimensional search. A preflight that checks clearances and mesh resolution but never
asks "did the lattice break a symmetry the design has?" silently converts a symmetric
design into an asymmetric model. (Generalized, with three sibling checks the same campaign
motivated, as issue #703.)
