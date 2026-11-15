### Fixed — Unify unsupported S-parameter request refusals (#PR)

Explicit S requests through `run()` and `forward()` now share the checks for
missing impedance ports and incompatible source or port families. Soft sources
alone, including those beside a closed TF/SF box, cannot supply port S-parameters.
Both calls give the same guidance for the declared families. Forward retains its
MSL near-field probe details, and the specialized calculators remain unchanged.
