# Handoff from group T (tests) — `_validate_ntff_small_ground_plane` (group P)

`tests/unit/farfield/test_ntff_small_gp_advisory.py` is T's; the validator it
pins is P's.

The advisory still classifies sheet-vs-volume from the DECLARED bounding box:

    ext = [c2[a] - c1[a] for a in range(3)]
    thin = min(range(3), key=lambda a: ext[a])
    if ext[thin] > max(lam / 20.0, l_small / 10.0):
        continue

That is the heuristic the contract replaces with a declaration. It happens to
keep working after T's migration — a zero-thickness Box has `ext[thin] = 0`, so
the sheet still qualifies and the thick-volume control still does not — so
**this file is green either way and P is not blocked by it**. The change P
should still make, when the other validators move: ask the classifier
(volume / sheet / wire) instead of measuring the drawn box, so a one-cell PEC
Box that IS a volume stops being called a ground-plane sheet.

Fixture note for P: the file's stack was redrawn on-lattice (`h = dx = 2.5 mm`
instead of cv05's 1.5 mm) because the old 1.0 mm ground on a 2.5 mm mesh is now
a hard `ValueError` from `classify_pec_entry` (0.4 of a cell), and 1.5 mm has
no node on that mesh. Both foils are sheets on adjacent node planes.
