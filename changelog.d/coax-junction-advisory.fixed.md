### Fixed — Plate-fed coaxial ports no longer get a false pin-short advisory (#PR)

A coaxial port fed through a hole in a ground plate no longer gets the false
pin-short advisory. At 1 mm spacing, the check counted the plate's metal on all
12 nodes of the pin's first ring; the run clears that metal from the bore.

`coaxial_port_junction_short` is retired. The port clears its bore through user
metal and refuses unnamed metal in front of its reference plane, requiring the
pin contact to be named in `terminates=`.
