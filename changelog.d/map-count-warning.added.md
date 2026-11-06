### Added — Warn once on Linux hosts with a low memory mapping limit (#PR)

- rfx warns once per process when vm.max_map_count is below 262,144, with remedies for compiling many distinct grids. Numerical results are unchanged.
