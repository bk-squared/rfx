# cv05 native constructor evidence — 2026-09-13

VESSL run `369367260761`, source `5979748c04c77ccfdcacea596949442f287e6273`.
Runtime: JAX 0.6.2 CPU, openEMS 0.37.0, CSXCAD 0.7.0. Both native tests passed.
No FDTD time stepping or MSL experiment was run.

The test builds the real rfx case in its build-only mode, passes its realized
conductor/material/terminal record through the external builder, smooths the
mesh with native CSXCAD, reads the native grid back and checks all prescribed
features and PML clearance. It then reads the native XML's material/metal
boxes back against the transferred record. The second arm moves both sheet
planes by one node while leaving the laminate fixed; the resulting air gaps
and shifted material bounds must survive the transfer and the mesh.

Default native board, in mm:

| Part | Lower corner | Upper corner |
|---|---|---|
| Ground sheet | (50, 50, 25) | (110, 105, 25) |
| FR4 | (50, 50, 25) | (110, 105, 26.5) |
| Patch sheet | (66, 59, 26.5) | (94, 96, 26.5) |

The patch remains 28 x 37 mm. The entire board and actual wire terminals share
one translation. The lower air region is chosen from the eight PML cells and
2.5 mm maximum spacing, plus two additional cells; the final native mesh is
checked rather than assuming that choice suffices. The old ground at z=0 put
the feed in the lower absorbing region and is rejected by a separate falsifier.

The four-file legacy cache is no longer adopted. Each full external solve
gets its own retained output directory; old and interrupted outputs remain
available but cannot silently provide a reference for a changed board.

`manifest.json` binds the source, runtime, recipe, XML, JUnit result and full
provider log by hashes. These checks establish constructor/mesh consistency,
not electromagnetic accuracy. Existing rfx ring-down fixtures, mode gates and
tolerances are unchanged. The cv07/#964 part of #959 remains outside this fix.
