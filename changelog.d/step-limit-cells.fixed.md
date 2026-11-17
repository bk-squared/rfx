### Fixed — Cell time-step limits (#PR)

Runs now refuse cells whose ε (or ε∞) times μr is below the local time-step
limit, with their bounds, count, and largest stable step. Graded meshes still
accept a pinned `dt=` inside that limit. Subgridded coarse cells and conformal
`cut_cell_floor` use conservative limits. JIT-traced ε or μr on several devices
remain unjudged (#2755).
