### Changed — Keep port edges free of design metal (#PENDING)

Lumped, wire and microstrip ports reserve their incident design cells on
uniform and graded forward solves, with one warning naming the ports and cells.
The former six-face occupancy guard is removed: cells not sharing a port edge
retain their design metal. Occupancy design boxes still refuse port overlap.
Waveguide design-metal exclusion remains an open question; waveguide ports
publish no released cells and their occupancy behavior is unchanged.
