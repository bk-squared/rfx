### Fixed — On the graded mesh, a plane wave at 0 < |angle_deg| <= 0.01° over a structure finite along the tilt axis is refused as at 0°, instead of being solved as an array of the padded period (#PR)

The source realizes |angle_deg| <= 0.01° as normal incidence. The graded-mesh admission
still treated these angles as oblique and skipped the invariance check along y (`ez`) or
z (`ey`), so a structure finite along that axis was solved, without a message, as an array
whose period includes the absorber pads. It is now refused with the message and remedy
given at 0°. On the uniform mesh these angles now give the 0° refusal text and remedy.
`compute_rcs` treats these angles as normal incidence too; before, it refused or failed on them.
Scenes admitted before run unchanged.
