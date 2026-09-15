# Coarse advisory witness: numerical attribution and PI stop

The unchanged `(2.25, 3]` interval is tied to the retired realization and an
invalid reference-plane placement. Leave this test failing pending a PI
choice of a qualified live advisory witness. No gate, fixture or operator was
changed in the test or production code.

Four serial diagnostic calls used the current worktree and current solver;
these are diagnostic calls, not four additional pytest passes. All coarse
arms have identical compiled-port fingerprints, incoming material-array
fingerprints, grid `(77,21,11)`, dt `3.8131497390620115e-12` s and 1312 steps.
The only runtime intervention is the conductor realization supplied at the
extractor input. The fixture builder is extracted directly from the unchanged
test AST, and its hash is preserved in every JSON file.
All captured complex a/b bins are finite and reconstruct every complex S
entry within `7.824e-8`; stored a/b magnitudes also agree with those complex
values. `advisory-capture-review.json` records this capture-consistency check.

| Arm | Largest column power | Frequency / driven port | Soft column-power advisory |
|---|---:|---|---|
| Current centre-sampled volume, 84–86 mm | 1.2999107837677002 | 4.4 GHz / left | absent |
| Legacy node-sampled `sigma=1e10`, at 86 mm | 2.5279102325439453 | 6 GHz / left | present |
| Current volume owner on node mask, 86–88 mm | 2.5279037952423096 | 6 GHz / left | present |
| Fine current volume, 85–87 mm | 1.0372081995010376 | 7 GHz / left | absent |

The predeclared falsifier was reached: restoring only the old sigma fold
restores the original interval. More specifically, restoring the node
sampling while retaining the current volume-edge operator also restores it.
The latter bridge differs from the old sigma arm by at most
`8.916872787585057e-6` in any complex first-column S entry, and
`9.179115295410156e-6` in any first-column power bin. The occupancy/sampling
shift is therefore sufficient to recover this scalar gate; the old damping
operator is not required. This is causal attribution of the historical gate,
not validation of its physical premise.

The bridge translates the current volume from 84–86 to 86–88 mm. Its front
wall moves from the right reference plane (84 mm) to 2 mm beyond that plane,
so both references now sample the same left-hand region. This is why the
result must not be described merely as an irrelevant far wall being added.
The per-bin first-column powers at 4, 4.4, 4.8, 5.2, 5.6 and 6 GHz are:

* Current: `[0.57092959, 1.29991078, 1.26020932, 1.24667907, 1.24225235, 1.12598395]`.
* Legacy sigma: `[0.78910649, 2.01916766, 1.99932337, 1.99348330, 2.03118992, 2.52791023]`.
* Node volume: `[0.78911567, 2.01916599, 1.99932384, 1.99348330, 2.03119040, 2.52790380]`.

## What the records show

The right source is at 90 mm but its coarse reference is at 84 mm and its
probe at 70 mm. Both measurement planes are across the short from their
own source. The modal extractor uses the reference V/I records
(`rfx/sources/waveguide_port.py:1834-1837`).

With the current volume and left drive, the right reference has exactly
zero voltage, but a nonzero magnetic/current record (peak
`0.0016590618761256337` in native record units). At a PEC wall this produces
incident and outgoing modal amplitudes with equal magnitudes: the receiving
port's `|a|` and `|b|` at 4.8 GHz are approximately `4.735098e-10` each. This
is a wall standing-field decomposition, not transmitted power through the
short. Its reported `|S21|` is about 0.5 despite full conductor shielding.

For the legacy-sigma and node-volume arms, the right reference lies in front
of the front wall, and the left-drive right-reference voltage peaks are
`0.26585668325424194` and `0.26585662364959717`. Their time traces nearly
coincide. The same-side modal field now produces reported `|S21|` near one,
and summing it with reflection creates the roughly two-unit column power.
At the weakly excited 6 GHz band edge the first-column power reaches 2.528;
the incident denominator is only about `8.6942e-12`, compared with
`9.9508e-10` at 5.2 GHz. It is not a second independent outgoing power port.

With right drive, **all eight captured coarse port records (10,496 samples)
are exactly zero**
for both exact-volume arms: their measurement planes cannot see the source
on the opposite side of the fully shielding short. The extractor's
`safe_a = where(abs(a_drive) > 0, a_drive, 1)` at
`waveguide_port.py:2181` then yields an all-zero second column. These zeros
are not a physical zero-reflection result and do not mean the source failed
to launch in the unobserved region. Settling is NaN because there is no
measurable record coverage, not because the run passed a settling test.

The legacy finite-sigma arm leaves tiny leakage: right-drive right-reference
voltage peak `1.953382455788244e-10` and current peak
`3.193574690720652e-13`. The incident spectrum near 5.2 GHz is only
`2.1902109e-19`, over nine orders below ordinary left-drive incident waves.
Dividing similarly tiny received spectra by this denominator produces a
nonzero second S column with power 0.778–1.356. The different second columns
show why the bridge is **first-column/scalar attribution**, not a claim of
full historical S-matrix equivalence.

Coarse settling is also inadequate independently of the geometry problem:
current left drive `-25.49489046` dB; node-volume left drive `-24.87408732` dB;
legacy left/right `-24.87408239/-35.34062052` dB, against the existing -40 dB
criterion. These records cannot qualify a new accuracy baseline. No window,
absorber or geometry sweep was launched to tune this warning.

## Hidden fine control

The diagnostic independently executed the exact fine branch that the coarse
assertion prevents pytest from reaching. Its maximum column power is
`1.0372081995010376 <= 2.25`; none of its recorded warnings contains
`ADVISORY`, so both original fine assertions hold. Its two settling values
are `-45.51188301` and `-114.21793975` dB. The two off-diagonal entries are
zero at every frequency, while right reflection has exactly unit magnitude.
That right reference is itself on the volume's far wall at 87 mm: its zero
voltage and nonzero current force equal modal wave magnitudes, so this is
not an independent qualification of a general far-port extraction setup.
The original pytest node remains failing at the coarse assertion.

A further diagnostic distinction: the test's `_soft_fired` helper searches
for any `ADVISORY`. Current coarse results do contain a **reciprocity**
advisory even though the soft **column-power** advisory is absent. The table
above identifies the actual column-power warning, not this broad substring.
No helper change is made while the physical witness awaits a PI decision.

## Artifacts and inspection

* `advisory_current.json`, `advisory_legacy_sigma.json`,
  `advisory_node_volume.json`, `advisory_current_fine.json`: complete complex
  S matrices, every column-power bin, a/b spectra, warnings, input hashes,
  geometry and record summaries.
* Matching `_records.npz` files: all captured port V/I traces.
* `advisory-attribution.png`: both driven-column spectra including the fine
  control, and coarse right-reference V/I traces under both drives. Rendered
  with Matplotlib and visually inspected using `view_image`; legacy and bridge
  first-column curves/traces overlap, while the legacy-only tiny right-drive
  leakage is visible with its actual scale.
* `plot_advisory_attribution.py`: reproduces that figure from saved results.

The decision needed is a qualified physical advisory witness or a redesigned
purpose for this lock. Restoring the old realization or moving its source,
mesh, gate or witness until a warning appears would evade that decision.
