### Added — Uniform multi-device port records (#PR)

Uniform `run(devices=...)` now returns lumped and wire `sparam_time_records`
from the main run, with port ids and `V`, `I`, `V_ref`, and wire `V_port`
channels matching one-device runs. Wire S-parameter requests also return
`wire_port_sparams`, including the pre-injection reference spectrum.
