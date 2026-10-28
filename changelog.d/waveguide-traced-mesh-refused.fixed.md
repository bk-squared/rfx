### Fixed — A waveguide port on a graded mesh whose cell widths are traced is refused by name (#PR)

Differentiating a waveguide S-parameter, `run()` or `forward()` with respect to graded cell widths (`dx_profile`,
`dy_profile`, `dz_profile` as JAX tracers) used to stop with a raw `TracerArrayConversionError` before the first
time step. It now raises `TracedStructureRefusal` naming the ports and the traced axis. The port reads its planes,
mode aperture and flux planes from host cell widths, so no cell-width derivative reaches its S; a permittivity
can still be differentiated through `eps_override`.
