### Changed — optimisation loops share one internal Adam interface (#PR)

- Optimisation, multi-start, progressive and topology loops now run Adam through one internal ask/tell optimiser interface, the step that MMA will use.
- Adam results remain unchanged bit for bit, pinned by a CPU contract against the previous loops.
