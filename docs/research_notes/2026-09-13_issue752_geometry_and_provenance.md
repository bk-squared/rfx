# #752: distinguish conductor geometry from historical material records

The old quantitative Z0 claims were already retracted by #766/#895. Two
remaining defects could still reproduce the same incorrect interpretation:
preflight treated a dielectric column extent as the RF gap, and the legacy
anchor generator combined historical Z0 values with current geometry before
overwriting the protected record. This repair makes neither operation a
current accuracy claim. This note covers the core repair; the separate fresh
six-point study is specified in [its frozen protocol](issue752/fresh/protocol.md).

## A material extent is not a conductor gap

On main `19aa9e06`, build-only inspection of the historical one-cell PEC
volume fixture gives:

| dx | Same-permittivity material extent | Validated ground–trace gap |
|---:|---:|---:|
| 80µm | 4 slots, 320µm | 3 normal intervals, 240µm |
| 60µm | 5 slots, 300µm | 4 normal intervals, 240µm |

Both ports pass the existing attachment validator. These are not invalid
inputs rejected by #729. The dielectric walk continues into slots also
occupied by the PEC trace; the source interval ends at its substrate-facing
conductor surface. Calling 320/300µm a thicker RF board, or predicting a Z0
change from those extents, therefore misidentifies what the port spans.

Preflight now obtains the conductor gap only after the existing validator
accepts the declared port against the original PEC/surface-impedance
carrier. It reads canonical node positions and the same source interval
as the port. Failed attachment leaves the gap unavailable and preserves
the error; no nearby wall is selected automatically. The one cached
assembly serves all ports. The material extent remains a separately named
observation of the declared material assembly, not a later override.

The four-interval recommendation, [0.10, 0.40] fraction window and 8.3%
geometry trigger keep their values. They are described as heuristics,
without a Z0 accuracy guarantee. The port source, field stepping,
observation and S normalization implementations are unchanged.

## Alignment must use absolute geometry, independently of epsilon

Four cheap counterexamples establish the required distinction:

1. Extending the same epsilon beyond the trace changed the old suggested
   spacings to 50.8/63.5µm or 31.8/36.3µm. The declared face is unchanged;
   its nearest interval-count choices remain 4/3, giving 63.5/84.7µm for
   an origin-aligned ground. Suggestions now use the face's bracket,
   rather than the material-column length.
2. A valid port declared at ground `4.2U`, height `3U`, snaps to conductor
   planes `4U/7U`. Its declared trace is at `7.2U`, so the fraction is 0.2.
   Measuring height from the snapped ground instead reported zero and
   suppressed the alignment warning.
3. A nonuniform grid with a four-U gap over three intervals has its declared
   trace exactly on a node. If its first material slot is vacuum, the old
   material helper returns absence; the caller then incorrectly uses the
   scalar base spacing `3U`, fabricating fraction 0.333 and a uniform-grid
   remedy while claiming the run grid is unavailable.

4. At an exact half-cell ground declaration, the generic grid indexer can
   choose the upper even node while the port's #931 conductor rule chooses
   the lower node. With two dielectric layers, the material walk then skips
   the first layer and reports the second. Its normal start now uses the
   same conductor mapping as the port; transverse lookup and the existing
   volume-ground skip remain unchanged.

Declared-face geometry is now computed independently of the epsilon walk
from absolute `position_z + height` and canonical nodes. Offset and
nonuniform cases name both absolute faces in their remedies: dividing a
height by an integer alone does not align an entire translated board.
The fraction by itself does not establish material/PEC overlap.

The new build suite covers 43 cases, including all propagation directions,
uniform/graded grids, offset grounds, PEC sheets/volumes, an f0 sheet ground,
invalid/unavailable attachment, material continuation, half-cell ground
ties and assembly reuse.
Independent review reproduced the two absolute-coordinate counterexamples
as fixed and checked 150 additional face positions, terminal nodes,
out-of-range inputs and NaN/Inf refusal.

## Retire the unsafe generator while preserving its evidence

The unchanged old generator was exercised with its output redirected to a
temporary path. It returned success and reported a maximum deviation of
**8.6234%**, obtained from old Z0 and new widths. This is an invalid
mixed-provenance comparison, not a measurement of current RF error. With
its default path it would overwrite the historical anchor JSON.

Both historical JSON files remain byte-identical. Both legacy commands now
refuse implicit generation before work begins. Explicit `--show-archive`
verifies their hashes and prints a labelled historical record without
rebuilding geometry or generating experiment output. The old inferred
precision remains inherited metadata; it is not independently verified
provenance of the original field solve. The pre-declared numerical helpers
remain executable-AST identical, and exact original scripts are retained
as compressed source records.

See the stable [MSL geometry guide](../guides/msl_geometry_diagnostics.md)
for the quantity definitions, inspection commands and requirements for any
future matched-record numerical comparison. No current “within 0.4%”
guarantee is restored by this change.

## Validation and scope

- Python 3.10/JAX 0.6.2: 43 new geometry cases; 59 final focused legacy,
  archive, emission and snapshot checks; 18 AD checks. The AD selection
  includes the real MSL forward mini-referee (its existing 3% central-FD
  criterion), source/load work against an independent scalar reference
  in both precisions and all directions, and public S assembly on
  prescribed synthetic fields. The latter also agrees with two central
  differences to relative 0.0000918/0.0001088; this is not an RF passivity
  certificate for the synthetic fields.
- Python 3.11/JAX 0.10.2: 102 final geometry, legacy, archive, emission and
  snapshot checks pass. An earlier 66-case compatibility selection also
  verified source/assembly AD and the two synthetic finite differences;
  the final material-origin correction was followed by the 18-case
  Python 3.10 AD audit on its exact source hash.
- The earlier full preflight/example run passed 483 cases and exposed one
  snapshot mismatch. Exactly two MSL message strings were updated; all
  grid/geometry data and other findings stayed identical. The final focused
  run verifies that snapshot and the unchanged emission inventory.
- Actual command-line checks reject default generation with exit 2 and
  accept explicit historical inspection with exit 0; frozen hashes and
  historical content are unchanged.

Evidence is in [issue752/](issue752/), with full logs, the deliberately
invalid old-generator output, source hashes, original script bytes,
geometry/negative-control records, AD receipts and snapshot-change receipts.
The scripts retaining absolute workspace paths are execution provenance.

## Fresh six-point quantitative assessment

The separate matched-geometry campaign requested for this issue ran all six
cases sequentially on VESSL run `369367260699` using the source archive at
`cbdc0976`. Each case used a zero-thickness node-aligned PEC foil, with the
dielectric top, port top and foil plane coincident. The replacement
differentiated-Gaussian parameter was fixed before the run (`f0=3.75 GHz`,
bandwidth `0.8`); the original `0.4%` comparison and all quality screens were
unchanged. Every case passed the finite, settling (worst `-102.83 dB`),
relative-signal and beta-rail screens. The independent evidence validator
also verified per-drive material/PEC/source arrays, probe/DFT registration,
current-plane metadata, and result/dump identities. Its synthetic mutation
tests reject inconsistent records before fitting.

The primary band is the inclusive 3–4.5 GHz mean of the production
`Re(Z0)` at port 0. The values and deviations are:

| dx | realized h | realized W | mean Re(Z0) | vs repository reference | vs full HJ1980 reference |
|---:|---:|---:|---:|---:|---:|
| 84.667 µm | 254.0 µm | 592.667 µm | 46.36530 Ω | −3.947% | −3.488% |
| 63.500 µm | 254.0 µm | 571.500 µm | 47.95702 Ω | −2.903% | −2.436% |
| 50.800 µm | 254.0 µm | 609.600 µm | 46.39500 Ω | −2.145% | −1.680% |
| 42.333 µm | 254.0 µm | 592.667 µm | 47.41100 Ω | −1.781% | −1.311% |
| 80.000 µm | 240.0 µm | 560.000 µm | 46.35699 Ω | −3.964% | −3.505% |
| 60.000 µm | 240.0 µm | 600.000 µm | 44.92623 Ω | −2.720% | −2.263% |

The all-six `≤0.4%` hypothesis is therefore **refuted** for both references;
the negative result is the conclusion, not a reason to relax a gate. The
independent float64 fit and beta-prior sensitivity differ from production
by only about `0.03%` on the first usable point, so the multi-percent
deviations are not explained by the production extractor's float32 fit alone.
Agreement with either quasi-static model remains a model comparison, not an
absolute electromagnetic accuracy certificate. The residual MSL power issue
tracked by #726 is outside this assessment.

The complete raw records, terminal logs and analysis outputs are under
[`issue752/fresh/`](issue752/fresh/). No field accuracy, source formulation
or reference-impedance replacement is claimed beyond this bounded result.
