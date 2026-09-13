# #940: separate invalid fixtures from invalid test verdicts

This addresses the three owner actions remaining in the September 7 weekly
report: the two-slab guide, the import-order subprocess, and the settled
coax silence control. Production simulator, port and autodiff code is
unchanged. The changes do not certify the whole weekly suite as green.

## Two-slab guide: an override was addressed without padding

The test declares source planes at 24/76 mm and intends epsilon=2.5 in
40–44 mm and epsilon=4 in 56–60 mm. It creates the override array on the
**padded** grid but originally indexes it with `round(x/dx)`. With dx=2 mm
and eight lower-x pad cells, the actual slabs occupy 24–28 and 40–44 mm.
The first slab starts on the vacuum-mode source plane.

This was present in the original #164 test (`b2f133b6`, June 12), whose
grid position conversion already included padding. The old prose's 16 mm
source clearance described intended geometry, not the constructed array.
The override is passed into device permittivity, whereas the waveguide
source/mode configuration is built separately without that override. A
vacuum-guide source on this dielectric interface violates the fixture's
own modal assumption.

On main `45481c6c`, the unmodified test fails at maximum column power
**1.413979** for 2,400 steps. Adding `grid.pad_x_lo` to the two slab indices
alone produces **1.010083 / 1.012403** at 2,400/4,800 steps and passes the
unchanged growth, 1.15 column-power and S-drift gates. The residual roughly
1% excess is not exact passivity; no S clipping or normalization change is
introduced. The historical 1.1071 calibration belongs to the misaddressed
fixture and is labelled accordingly.

The final shared fixture helper produces arrays identical to the measured
two-line repair. A fast check reads the full epsilon array and verifies
vacuum at the actual source planes and reference/current-stencil planes
(30/70 mm). It intentionally does not require every distant diagnostic
probe to be in vacuum: one probe at 56 mm lies in the corrected slab, while
this S-matrix path uses the near reference records.

## Import-order check: a warning word is not a process verdict

The retained September 7
[shard 3 log](https://github.com/bk-squared/rfx/actions/runs/34120118201/job/101736003819)
shows the child pytest process returning zero with **83 passed**. A normal
degenerate-drive warning contains “one drive that failed to excite”; the
parent's substring search for `failed` therefore reports a false failure.

Use the subprocess return code, preserving the existing assertion and full
stdout/stderr on a nonzero code. Three fast controls verify codes 0, 1 and 5
with identical warning-containing stdout. The actual ordered child suite
also passes; all five tests in the parent file pass in 35.47 seconds.

## Coax silence control: use the result's eligibility contract

The public result, shared verdict and module's original warning contract
use **-40 dB, inclusive**. The quiet control had an additional -60 dB
fixture margin, introduced to put a historical record far below the bar.
That margin is removed explicitly; the physical warning threshold remains
-40 dB, and the same 3,000-step solve remains.

The current records are **-58.87838726 / -64.84236209 dB**, respectively
18.88/24.84 dB below the actual bar. Both original drive records are
retained; an independent long-double evaluation of last-10%-mean/peak
power reproduces them. The old -67.26/-68.09 dB record remains historical.
**The physical change between those records has not been attributed.**
The question this test answers is whether an eligible settled record emits
a truncation warning, not whether its decay matches a historical value.

Require exactly two finite witnesses satisfying the inclusive threshold
before checking silence. The former `all(sd < -60)` also accepted empty
arrays and negative infinity; those no longer qualify. Fast controls call
the actual quiet-test body with -50/-55 dB, the -40 dB boundary, a hot
record, NaN, negative infinity and an empty record. The real 400-step
warning and differentiated `eps_scale` absence/NaN tests are unchanged.

The selected real controls and synthetic checks pass (9 tests, 105.44 s).
The subsequent inclusive-boundary correction changes only the auxiliary
assertion and its boundary control; all six fast controls pass (1.28 s).
No additional field solve was needed to establish that a value below -40
also satisfies its inclusive limit.

## Disposition of the other reported entries

Both named RAM runner-kill cases were already moved by #941/#943. Current
collection under the actual a6000 selection, `highmem and not gpu`, includes
the sigma analytic-gradient case and inverse-design optimum case. This
verifies routing; it is not a fresh memory/performance qualification.

The exact golden failure named in #940 used the removed pre-#931 test.
The owner closed #797 on September 11 when that test and geometry were
replaced by #931. That disposition does not establish that the new golden
passes on every platform; no new golden capture or tolerance change is
part of this repair.

Evidence is in [issue940/](issue940/): failure and verification logs,
the corrected guide matrices/overrides, original coax records, exact
source snapshots for the controlled measurements, and hashes. The capture
scripts retain their original workspace paths as execution provenance.
The final helper/array parity receipt prevents the small fixture extraction
from silently changing the inputs that were measured.
