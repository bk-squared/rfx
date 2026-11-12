### Fixed — whole-node-plane waveguide ports report the required cell span (#PR)

- Low-level waveguide ports that include the node row beyond a high electric wall now raise `ValueError` with the cell span to use.
- Previously, their second-mode S-parameter entries could be wrong by more than their magnitude.
