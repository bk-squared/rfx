### BREAKING — Refuse sources shorted by PEC walls (#PENDING)

Soft sources and lumped ports tangential to a declared PEC wall now raise before stepping.
Their tangential electric edges in the PEC wall are shorted by the conductor.
Wire ports raise when all their driven edges are shorted by PEC geometry or walls.
Preflight reports wall-short declarations as `source_decoupled` errors.
Move the source at least one cell off the wall, or orient it normal to the wall.
