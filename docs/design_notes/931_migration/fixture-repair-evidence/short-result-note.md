# PEC-short fixture repair: observable ports, failed historical interval

**Finding:** the repaired coarse short produces maximum column power
**1.0444802045822144**, outside the unchanged **(2.25, 3]** advisory-witness
interval. The left-drive settling witness is **-25.494890460744337 dB**, which
fails the unchanged -40 dB requirement; the right drive reaches
**-49.59556966120925 dB**. These finite-window S values are inspection data,
not qualified physical-accuracy results. No interval is widened or re-pinned.

Named CPU run: **local-931-fixture-repair-short-coarse**, 2026-09-08,
`PYTHONPATH=$PWD JAX_PLATFORMS=cpu`. Source HEAD was
`949e1845b84b26826d4e5309363bea4c2ffa5920` with the repair uncommitted.
The [capture JSON](short-coarse.json) records the local worktree import,
fixture SHA256 `3c1d0a135933f09136ce00e8e234ad8e00304fcafa52fdfbc746ff39da1b040e`,
material and port-input hashes, both drive records and all warnings. Its
additional provenance block explicitly labels post-capture artifact and
diagnostic hashes. [Compact records](short-coarse-traces.npz) retain all
voltage/current traces, full complex S, both driven incident/reflected
spectra, frequency coordinates and settling readouts.

## Geometry and measurement repair

The common class is a historical fixture lock whose physical premise or
measurement is wrong. This instance had an off-lattice short and right
measurement planes across the opaque plate from the right source. The old
all-zero right-drive column was a disconnected measurement, not a passive
matched-port result.

The corrected guide is 120 x 40 x 20 mm, explicitly not WR-90. Its full
cross-section PEC closing plate occupies **x=84--86 mm**, a machined 2 mm
plate idealized as PEC. The coarse mesh uses **dx=2 mm**; the fine control
uses **dx=1 mm**. Both divide every domain/plate face, and the fixed drawing
preserves the coarse fixture's previous realized body. Coarse realization
has 200 occupied cells and Ex/Ey/Ez edge counts 231/440/420, with both
tangential walls at 84 and 86 mm. Historical node and current center
occupancy agree for this repaired drawing.

| Port | Source x | Reference x | Probe x | Connected region |
|---|---:|---:|---:|---|
| Left, +x drive | 10 mm | 16 mm | 30 mm | Entirely left of 84 mm |
| Right, -x drive | 110 mm | 104 mm | 90 mm | Entirely right of 86 mm |

Distances are fixed in metres across meshes: 6 mm source-to-reference and
20 mm source-to-probe, converted to integer cells. This removes the old
ten-cell offset's change of physical network during refinement. The
coarse diagnostic retains 4--6 GHz in six bins, eight CPML cells (16 mm)
per propagation side, `normalize=False`, and 30 periods: 1312 steps at
3.8131497390620115 ps, approximately 5.003 ns.

## Full-record inspection

Both drives have finite, nonzero incident **and reflected** spectra in all
six bins. Both driven ports have nonzero voltage and current records.
The other port's time traces and both transmission terms are identically
zero, as expected when a full-cross-section PEC plate separates the two
halves. Neither complete drive column is zero anymore. Column powers were
independently recomputed from the saved complex S entries.

| Frequency, GHz | Left reflection magnitude | Right reflection magnitude | Left column power | Right column power |
|---|---:|---:|---:|---:|
| 4.0 | 0.55135012 | 1.01481569 | 0.30398694 | 1.02985084 |
| 4.4 | 1.02199817 | 1.00075626 | 1.04448020 | 1.00151312 |
| 4.8 | 1.00598729 | 0.99971336 | 1.01201046 | 0.99942678 |
| 5.2 | 0.99963307 | 0.99906415 | 0.99926627 | 0.99812919 |
| 5.6 | 0.99564600 | 0.99842089 | 0.99131095 | 0.99684429 |
| 6.0 | 0.94418657 | 0.99301034 | 0.89148825 | 0.98606956 |

The [visually inspected figure](short-coarse-inspection.png) shows every
frequency bin, both drive powers against the old interval, both incident
and reflected spectra, and the reference-voltage envelopes. The left
low-frequency deficit is visible and is not hidden by the near-unity
maximum. The left voltage envelope retains a late secondary lobe while
the right envelope decays further. The plotted voltage block maxima are
an independent trace view, not the API's end/peak energy settling metric;
their dB values must not be substituted for the settling values above.

This removes the disconnected measurement defect and falsifies the
historical advisory-witness interval at the retained run settings. It
does **not** prove accurate near-unity reflection. In addition to failed
left settling, production reports insufficient absorber depth (16 mm
versus its documented 106.4 mm floor at 4 GHz), a short far-boundary
round-trip record and the known source-plane mirror offset. Preserve
those warnings; do not hide them behind the zero-transmission witness.

## Disposition

Keep the original interval assertion failing as the requested finding.
The synthetic passivity-guard tests independently cover the advisory's
warning/silence thresholds. Reclassifying or replacing this physical
advisory witness is a coverage decision, not permission to tune its
plate or intervals until it over-energizes again.

The retained fine control uses the same drawing and measurement planes
but 5--7 GHz, 40 periods and ten 1 mm CPML cells. It changes frequency,
window and absorber thickness as well as mesh, so a passing fine result
is not a fixed-configuration convergence study. No simulation or pytest
was rerun to produce these inspection artifacts.

## Fine control, separately captured

Named run **local-931-fixture-repair-short-fine** subsequently completed
with the same source HEAD and fixture hash, 2998 steps at
1.9065748695310057 ps. The [fine capture](short-fine.json),
[full compact records](short-fine-traces.npz) and
[visually inspected fine figure](short-fine-inspection.png) preserve both
drive records. Maximum column power **1.049412727355957** passes the
unchanged fine-control bound <=2.25. Both settling values pass -40 dB:
left **-47.47585398397359 dB**, right **-103.15757071046333 dB**.

| Frequency, GHz | Left reflection magnitude | Right reflection magnitude | Left column power | Right column power |
|---|---:|---:|---:|---:|
| 5.0 | 0.93019384 | 0.99981475 | 0.86526060 | 0.99962956 |
| 5.4 | 0.99654722 | 0.99966216 | 0.99310637 | 0.99932444 |
| 5.8 | 0.99964553 | 0.99954814 | 0.99929118 | 0.99909645 |
| 6.2 | 1.00016177 | 0.99947476 | 1.00032353 | 0.99894983 |
| 6.6 | 1.00131583 | 0.99942851 | 1.00263345 | 0.99885732 |
| 7.0 | 1.02440846 | 0.99949831 | 1.04941273 | 0.99899685 |

All fine voltage/current traces are finite; both driven ports have
nonzero incident/reflected waves in all bins. Opposite-side traces and
both transmission terms remain identically zero. The figure exposes the
left-drive 5 GHz deficit and 7 GHz excess rather than summarizing them
as a single passing maximum. Column powers were independently recomputed
from the complex S arrays. Production still warns that the 10 mm absorber
is below its documented 45.3 mm floor at 5 GHz. Settling passes, but this
retained advisory control is not an absolute reflection-accuracy oracle.
