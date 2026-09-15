# Handoff from group T (tests) — two preflight findings that fire on CORRECT declarations (group P)

Observed while migrating group T's fixtures onto the contract. Both are
advisories, so nothing is red, but both now fire on models that are drawn
exactly the way §1.5 says to draw them.

## 1. `_validate_mesh_quality`'s zero-thickness advisory contradicts §1.5

Every migrated foil now produces, once per entry:

    Zero-thickness geometry 'pec' along z-axis. On non-uniform mesh this may
    produce empty rasterization. Consider giving it at least one cell of
    thickness (1mm).

Under the contract a zero-extent axis on a PEC Box IS the sheet declaration,
and "give it one cell of thickness" is the exact move the contract stopped —
it turns foil into a filled slab with a wall on each face. P's brief already
covers this ("zero-thickness → 'declare a sheet with add_thin_conductor'");
recording it here as live, and noting the wording should distinguish PEC (a
sheet declaration, correct) from a zero-thickness DIELECTRIC (still nothing).

## 2. The buried-sheet warning fires under a full ground plane on the domain floor

`rfx/api/_execute.py` (assembly, stage C) emits:

    PEC sheet pec realizes on node plane N of axis z, and the cells on BOTH
    sides of that plane carry the same dielectric ... The sheet is therefore
    buried half a cell inside the dielectric ...

on a board whose substrate starts at `z = 0` with a ground sheet on that same
plane, because `include_cpml_pad_extension` replicates the substrate DOWN into
the z pad. The declaration is right — the foil is on the laminate's bottom
face — and nothing propagates below a full ground plane anyway, so the finding
is a false positive for this shape.

Two candidate narrowings, for whoever owns the check: ignore cells inside the
CPML pad when deciding "the same dielectric on both sides", or skip the finding
when the sheet's footprint spans the whole transverse domain (a ground plane
has no other side). T worked around it in `tests/unit/api/test_visualize3d.py`
by lifting the board one cell off the domain floor, which is a fixture fix, not
an answer.
