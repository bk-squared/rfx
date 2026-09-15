# T1 → fidelity: rule (i) goes silent on a sheet ground

Owner: `rfx/fidelity.py` (core). T1 owns only the tests, so this is written,
not applied.

## Fact

`rfx/fidelity.py:605-607`:

```python
        # A SHEET owns no cell (#931 §1.3), so it claims none: a dielectric
        # drawn after it is not overwritten by it.
        if pec_assembled and sheet_fp is None:
            pec_before |= mask
```

`pec_before` is the accumulator the ordered `dielectric-after-conductor-no-op`
finding (issue #589) reads. A sheet never enters it, so under a sheet-declared
ground the finding cannot fire.

## Why that matters

#589 is not about eps ownership. It is: *a dielectric declared after a
conductor cannot carve a hole in it*; the attempt-2 coax pin passed through a
solid ground and the run measured S00 = (-0.9928, -0.0048), a short. A sheet is
exactly as uncarvable as a slab — the PTFE Cylinder does not remove one edge of
it — so the case the finding exists for is the case it now misses.

The eps statement in the comment is right and should stay: a sheet writes no
`eps_r`, so `claimed-by-conductor` (an eps-fraction finding) correctly does not
fire. The two findings need to part company.

## Recommended shape of the fix

Give the ordered check a REALIZED-metal accumulator instead of a cell
accumulator: for a volume, its cells; for a sheet, its footprint nodes (or,
equivalently and in the contract's own vocabulary, the nodes whose incident
in-plane E edges the sheet zeroes). Overlap for a sheet = the dielectric's
node/cell set intersected with the footprint on the sheet's plane. Report the
count in nodes and say so, so a sheet row is not read as a cell count.

`test_fidelity_topology_findings.py::test_junction_plane_metal_under_a_sheet_ground`
already pins the geometry side of this on the #589 fixture's own numbers
(annulus 36/36, first ring 16/16 at the junction plane) so the finding has a
checked oracle to fire against.
