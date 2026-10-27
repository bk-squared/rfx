### Fixed — planned z margins and declared auto-mesh lengths agree with the solved grid (#PR)

- Geometry plans include both z margins and expose `offset`; add it to each shape's coordinates when placing the model.
- Automatic meshes keep the declared domain instead of replacing its z length with geometry height plus margin.
- A concrete z profile more than 1% off the declared length is refused; `snap="declared"` keeps it with a warning.
- Preflight warns when a dielectric body's thin face sits flush on an absorber and its material continues beyond the face.
