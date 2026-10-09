# Path-equivalence findings

`findings.json` is the source; this page only names its entries. Each tag names
a difference **observed between the result records of two paths**. None states
a mechanism: why a record is absent or shaped differently has not been derived.
The matrix tests accept only these texts as known failures, and check each text
on its own: a listed text that stops occurring in its cell record fails the
test, also when another listed text of the same record still occurs.

Path A is the uniform path (`run()` or `forward()`); path B is the constant-profile
non-uniform path, or the two-device uniform path where stated.

| Tag | Observed record difference | Cells |
|---|---|---|
| `flux-dA-shape` | flux monitor `dA` has shape `(1, 1)` on A and `(13, 12)` on B | flux, `run()`, uniform vs non-uniform |
| `flux-dA2-missing` | flux monitor `dA2` is absent on A | flux, `run()`, uniform vs non-uniform |
| `forward-flux-record-missing` | `flux_monitors` is absent from both `forward()` results | flux, `forward()`, uniform vs non-uniform |
| `nu-lumped-dft-record-missing` | `lumped_port_sparams` is absent on B | lumped and passive port, `forward()`, uniform vs non-uniform |
| `nu-missing-vref` | `V_ref` is absent on B, in the per-step port record and in the wire-port DFT record | lumped, passive and wire port, uniform vs non-uniform |
| `port-time-record-missing` | the per-step port record is absent on A and B, or on B only, or its `V_port` is absent on A, depending on the cell | lumped, passive and wire port; uniform vs non-uniform and vs two-device |
| `s-parameter-shape` | `s_params` has shape `(2,)` on A and `(1, 1, 2)` on B | lumped, passive and wire port, `forward()`, uniform vs non-uniform |
| `uniform-wire-dft-record-missing` | `wire_port_sparams` is absent on A | lumped, passive and wire port, uniform vs non-uniform |

The exact text per cell and record is in `findings.json` (`causes.*.fingerprints`, `cells`).

Geometry keys that start with `declared_` are excluded from the realized-geometry
equality: they record what the script declared, not what the grid built.
