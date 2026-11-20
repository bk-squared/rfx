### Fixed — a thin element on the far wall of a waveguide port's section is refused like one on the near wall (#PR)

- A surface-impedance sheet, thin film, lumped element or port stamp on the hi transverse wall node of a waveguide port's section reached the run when the guide's walls were PEC volumes. The same element on the lo wall was refused. The port has no model for it, so the run treated that wall as plain PEC.
- Both wall nodes are now judged the same way. PEC edges on either wall stay accepted. A guide whose walls are PEC domain faces was already refused on both walls by the domain-face check.
