### Changed — Grid-declared absorbing axes (#PENDING)
Low-level `rfx.run` and non-uniform runners default to the grid's absorbing axes.
Explicit axes outside that declaration now raise an error; explicit subsets remain accepted.
Previously a z-only uniform grid could absorb x/y faces inside the physical domain.
