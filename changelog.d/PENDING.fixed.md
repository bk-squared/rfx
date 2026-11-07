### Fixed — a soft point source is no longer listed as a lumped port

- `Simulation.realized_geometry().ports` (and the run result's record) lists only `add_port` ports.
- A soft point source from `add_source` appears once, as a `soft_point` row in `.sources`;
  before, it also appeared as a `lumped` port on the same edge, under the same `port[i]` label.
- Field and S-parameter numbers are unchanged: the solver already treated it as a soft source.
