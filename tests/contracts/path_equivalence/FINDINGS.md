# Path-equivalence findings

`findings.json` is the source; this page only names its entries. Each tag names
a difference **observed between the result records of two paths**. None states
a mechanism: why a record is absent or shaped differently has not been derived.
The matrix tests accept only these texts as known failures, and check each text
on its own: a listed text that stops occurring in its cell record fails the
test, also when another listed text of the same record still occurs.

Path A is the uniform path (`run()` or `forward()`); path B is the constant-profile
non-uniform path, or the two-device uniform path where stated.

No active findings remain. The three uniform two-device port cells now return
their per-step records, and both wire cells return `wire_port_sparams`.

The exact text per cell and record is in `findings.json` (`causes.*.fingerprints`, `cells`).

Geometry keys that start with `declared_` are excluded from the realized-geometry
equality: they record what the script declared, not what the grid built.
