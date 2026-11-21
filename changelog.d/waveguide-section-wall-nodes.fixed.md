### Fixed — wall loss on the far wall of a waveguide port's section is refused like wall loss on the near wall (#PR)

- Wall loss (`add_wall_loss`) on a guide wall inside a waveguide port's section was refused on the port's lower transverse wall but reached the run on the upper one. The port's mode assumes a lossless PEC wall there, so that run used a wall the port does not model.
- Both wall nodes are now judged the same way for elements on the wall's E edges: wall loss, surface-impedance sheets, thin films and lumped elements. Plain PEC walls on either side stay accepted.
