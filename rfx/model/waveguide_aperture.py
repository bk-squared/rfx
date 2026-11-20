"""Refuse transverse node rows outside the guide's high electric walls."""


def validate_waveguide_aperture(normal_axis, u_span, v_span, shape, pec_faces, *, name):
    """Require the cell span on each transverse high wall; never clip a port."""
    axes = {"x": "yz", "y": "xz", "z": "xy"}[normal_axis]
    for axis, span in zip(axes, (u_span, v_span)):
        lo, hi = span
        if hi == shape["xyz".index(axis)] and f"{axis}_hi" in pec_faces:
            raise ValueError(
                f"Waveguide port {name}: {axis}_slice={(lo, hi)} includes the "
                f"node row outside the high electric wall {axis}_hi; use "
                f"cell span {axis}_slice={(lo, hi - 1)}, as Simulation builds it."
            )


def validate_waveguide_ports(grid, ports, pec_faces):
    """Check prebuilt configurations against the electric walls of a run."""
    for index, cfg in enumerate(ports or ()):
        validate_waveguide_aperture(
            cfg.normal_axis, (cfg.u_lo, cfg.u_hi), (cfg.v_lo, cfg.v_hi),
            grid.shape, pec_faces, name=f"{index} ({cfg.direction})",
        )


def validate_uniform_waveguide_ports(grid, ports, pec_axes=None, periodic=None):
    """Use the uniform step's face and periodic overrides."""
    if not ports:
        return
    from rfx._internal._uniform_specs import resolve_periodic
    from rfx.boundaries.pec import resolve_wall_faces

    faces, _ = resolve_wall_faces(grid, resolve_periodic(grid, periodic), pec_axes)
    validate_waveguide_ports(grid, ports, faces)


def validate_graded_waveguide_ports(grid, ports, pec_faces=None, pmc_faces=None):
    """Use the face declarations passed to the graded step."""
    if not ports:
        return
    from types import SimpleNamespace
    from rfx.boundaries.pec import resolve_wall_faces

    walls = SimpleNamespace(
        pec_faces=set(pec_faces or ()), pmc_faces=set(pmc_faces or ()), shape=grid.shape,
    )
    faces, _ = resolve_wall_faces(
        walls, tuple(axis in grid.periodic_axes for axis in "xyz"), None,
    )
    validate_waveguide_ports(grid, ports, faces)
