### Changed — every public function and `Simulation` method carries full type annotations (#PR)

- All 131 functions in `rfx.__all__` and all public `Simulation` methods now annotate every parameter and the return
  value, so type checkers and IDEs see rfx's real argument and result types (rfx ships `py.typed`). No signature,
  default or behaviour changed. A contract test keeps new public functions annotated.
