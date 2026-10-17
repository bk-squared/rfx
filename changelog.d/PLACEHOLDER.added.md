### Added — Full tensor dielectric smoothing (#1632)

Use `subpixel_smoothing="full_tensor"` for full anisotropic interface averaging
on uniform single-device meshes. Unsupported paths raise an explicit error.
`True` retains diagonal smoothing. Full tensor construction follows the run precision.
Refs #1632.
