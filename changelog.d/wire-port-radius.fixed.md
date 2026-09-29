Wire ports accept an opt-in `radius=` in metres with `extent=`, using local
magnetic and electric self-field corrections through shared component materials.
`radius=None` preserves the existing mesh-sized probe (effective radius about
`0.20 * dx`). The declared-radius model requires a locally square, uniform
transverse mesh, uniform spacing along the pin, and `radius <= 0.20 * dx`;
larger pins must be resolved
geometrically, for example with a coax feed. Single-device nondispersive Yee
`run()` and `forward()` carry the correction. Unsupported solver/material
paths, including distributed, subgridded, ADI and Debye/Lorentz, refuse it
before stepping.
