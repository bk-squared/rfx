"""PMC image-face resolution and assembly-time magnetic coefficients."""

from __future__ import annotations

from rfx.errors import RfxRefusal, RfxUnsupported


def magnetic_image_faces(faces, shape, periodic=(False, False, False)):
    """Resolved magnetic image faces: no periodic or collapsed-axis walls.

    Shared by wall resolution, H reads/storage and flux quadrature. A
    one-node invariant axis has neither a face control-volume half nor a
    high H ghost, even when the declaration names its two faces PMC.
    """
    return frozenset(face for face in faces
                     if shape["xyz".index(face[0])] > 1
                     and not periodic["xyz".index(face[0])])


def refuse_waveguide_pmc(sim):
    """The waveguide mode solver only implements electric aperture walls."""
    faces = sorted(sim._boundary_spec.pmc_faces())
    if faces and sim._waveguide_ports:
        raise RfxUnsupported(
            f"PMC magnetic face(s) {', '.join(faces)}: waveguide-port feature "
            "in waveguide_port.init_waveguide_port has no magnetic aperture "
            "mode solver. Use a full PEC-walled guide until that kernel "
            "implements magnetic symmetry.")


def refuse_port_on_magnetic_face(sim, caller, *, reason):
    """Refuse full-object port power compared with physical-half flux.

    An edge whose node touches the face counts as on the face.
    """
    from rfx._grid_metric import field_index
    from rfx.measurement.plan import _pec_edges
    from rfx.sources.sources import WirePort, _wire_port_live_cells

    grid = sim._build_realized_grid()
    faces = magnetic_image_faces(sim._boundary_spec.pmc_faces(),
                                 grid.shape, sim._periodic_flags())
    if not faces:
        return
    edges = None
    if any(p.impedance and p.extent is not None for p in sim._ports):
        edges = _pec_edges(sim, grid)
    for index, port in enumerate(sim._ports):
        if not port.impedance:
            continue
        live = (field_index(grid, port.position, port.component),)
        if port.extent is not None:
            end = list(port.position)
            end["xyz".index(port.component[1])] += port.extent
            wire = WirePort(port.position, tuple(end), port.component,
                            port.impedance, port.waveform, radius=port.radius)
            cells, flags, _ = _wire_port_live_cells(grid, wire, edges)
            live = tuple(cell for cell, flag in zip(cells, flags) if flag)
        for face in sorted(faces):
            axis = "xyz".index(face[0])
            node = 0 if face.endswith("lo") else grid.shape[axis] - 1
            if any(cell[axis] == node for cell in live):
                raise RfxRefusal(
                    f"{caller}: port {index} at position {port.position} "
                    f"lies on magnetic face {face}. A port on a magnetic "
                    "face describes the full symmetric object while the "
                    f"flux box integrates only the physical half. {reason} "
                    "Use magnitude_channel='wave', or move the port off the face.")


def magnetic_update_factors(faces, shape, periodic=(False, False, False)):
    """Assembly-time 1-D factors holding only declared high H ghosts at zero."""
    import numpy as np

    faces = magnetic_image_faces(faces, shape, periodic)
    factors = []
    for axis, letter in enumerate("xyz"):
        factor = None
        if f"{letter}_hi" in faces:
            factor = np.ones(shape[axis], dtype=np.float32)
            factor[-1] = 0.
        factors.append(factor)
    return tuple(factors)


def magnetic_update_coefficients(coefficients, factors):
    """Apply each axis factor to its two tangential H coefficients."""
    result = list(coefficients)
    for axis, factor in enumerate(factors):
        if factor is not None:
            shape = [1, 1, 1]
            shape[axis] = -1
            for component in range(3):
                if component != axis:
                    result[component] = result[component] * factor.reshape(shape)
    return tuple(result)


def magnetic_update_fields(fields, factors):
    """Factor the whole H update; select positive zero on high ghost rows.

    Multiplication alone retains the bracket's sign on zero. The select
    preserves the old ghost write's +0.0 without a scatter or changing any
    non-ghost bits. Absent factors add no operations to the traced program.
    """
    import jax.numpy as jnp

    result = list(fields)
    for axis, factor in enumerate(factors):
        if factor is not None:
            shape = [1, 1, 1]
            shape[axis] = -1
            factor = factor.reshape(shape)
            for component in range(3):
                if component != axis:
                    value = result[component] * factor.astype(result[component].dtype)
                    result[component] = jnp.where(factor == 0, jnp.zeros((), value.dtype), value)
    return tuple(result)


def apply_pmc_faces(state, faces: set[str]) -> object:
    """Legacy half-cell zeros for explicit low-level absorber witnesses only.

    Production image paths hold high ghosts through update coefficients.
    tests._absorber_witness and test_cpml_localization still use this legacy
    boundary with their manually assembled, non-image kernels.
    """
    # Legacy half-cell enforcement for kernels not yet using the shared image.
    hx, hy, hz = state.hx, state.hy, state.hz

    if "x_lo" in faces:
        hy = hy.at[0, :, :].set(0.0)
        hz = hz.at[0, :, :].set(0.0)
    if "x_hi" in faces:
        hy = hy.at[-2, :, :].set(0.0)
        hz = hz.at[-2, :, :].set(0.0)
    if "y_lo" in faces:
        hx = hx.at[:, 0, :].set(0.0)
        hz = hz.at[:, 0, :].set(0.0)
    if "y_hi" in faces:
        hx = hx.at[:, -2, :].set(0.0)
        hz = hz.at[:, -2, :].set(0.0)
    if "z_lo" in faces:
        hx = hx.at[:, :, 0].set(0.0)
        hy = hy.at[:, :, 0].set(0.0)
    if "z_hi" in faces:
        hx = hx.at[:, :, -2].set(0.0)
        hy = hy.at[:, :, -2].set(0.0)

    return state._replace(hx=hx, hy=hy, hz=hz)
