"""PMC storage handling during the B3/B4 kernel migration.

Single-device shared-image paths pass image=True: physical H is retained,
and only unused high ghosts are canonicalized. The odd image is evaluated
by yee.h_neighbor after every H change. The default retains the old half-cell
zeroing solely for kernels that have not migrated; public distributed and
subgridded/ADI magnetic requests refuse rather than exposing that rule.
"""

from __future__ import annotations


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
        raise NotImplementedError(
            f"PMC magnetic face(s) {', '.join(faces)}: waveguide-port feature "
            "in waveguide_port.init_waveguide_port has no magnetic aperture "
            "mode solver. Use a full PEC-walled guide until that kernel "
            "implements magnetic symmetry.")


def apply_pmc_faces(state, faces: set[str], *, image: bool = False) -> object:
    """Canonicalize ghosts for image paths, or apply legacy half-cell zeros.

    Parameters
    ----------
    state : FDTDState
    faces : set of str
        Which faces to enforce PMC on. Valid names:
        ``"x_lo"``, ``"x_hi"``, ``"y_lo"``, ``"y_hi"``,
        ``"z_lo"``, ``"z_hi"``.

    Notes
    -----
    Yee-grid index convention: H_tan at a ``_lo`` face sits at array
    index 0 (physical position 0.5·dx inside the wall), and at a
    ``_hi`` face sits at array index ``-2`` (physical position
    0.5·dx inside the wall at ``(nx-1)·dx``). Index ``-1`` on the hi
    side is the ghost half-cell 0.5·dx OUTSIDE the wall, which does
    not participate in the interior E-curl stencil and was previously
    zeroed with no effect — causing ``_hi`` PMC to be a silent no-op
    that let the wall behave as PEC. Fixed 2026-04 (see
    tests/unit/boundaries/test_boundary_pmc_hi_faces.py for the regression lock).
    """
    if not faces:
        return state
    if image:
        faces = magnetic_image_faces(faces, state.hx.shape)
        # Physical H samples survive. Shared curl/port reads impose the odd
        # image at consumption; keep the unused stored high ghosts canonical.
        fields = dict(hx=state.hx, hy=state.hy, hz=state.hz)
        for axis, letter in enumerate("xyz"):
            if f"{letter}_hi" in faces:
                edge = [slice(None)] * 3
                edge[axis] = -1
                for component in "xyz":
                    if component != letter:
                        key = "h" + component
                        fields[key] = fields[key].at[tuple(edge)].set(0.0)
        return state._replace(**fields)
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
