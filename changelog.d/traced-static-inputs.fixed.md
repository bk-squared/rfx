### Fixed — Name traced static inputs before simulation setup (#PR)

- A traced static input now raises an rfx refusal naming its method and parameter,
  instead of a raw JAX conversion error or a silently dropped tangent.
