### Fixed — High-wall sources, probes and lumped elements resolve inside the domain (#PENDING)

Sources, E/H probes, single-cell lumped ports and RLC elements declared on a high PEC or PMC wall, or within half a cell of it along a displaced component axis, now use the entry half a cell inside, as at the low wall. Previously sources could radiate nothing, probes could read zero, and a lumped port could report the voltage of an uncoupled edge.
