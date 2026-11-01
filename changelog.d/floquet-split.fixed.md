### Fixed — Floquet reflection uses E and H at the same position and time (#PR)

Floquet accumulation centres H on E and corrects the y-normal wave sign.
Across 1.5, 1.0 and 0.75 mm meshes, the worst empty-cell |S11| falls from 0.05963 to 0.00224,
and the maximum PEC |S11| from 1.04463 to 1.00323 (worst deviation from 1: 0.00475).
The slab's worst complex Airy error falls from 0.07885 to 0.00931; rotated ports agree within 0.0000062.
Record-length changes still exceed 0.001. Place reflection planes between the source and structure;
call accumulation after the E update, with port_index at least 1.
