"""Perfect Magnetic Conductor (PMC) boundary condition (T7 Phase 2 PR3).

Zeros tangential H-field at boundary faces. The electromagnetic dual
of :mod:`rfx.boundaries.pec`: PEC enforces E_tangential = 0, PMC
enforces H_tangential = 0. PMC is the boundary condition imposed by a
"magnetic wall" — physically rare in practice but essential as a
symmetry-plane image source for structures with mirror symmetry and
as a building block for cavities / waveguides with mixed-BC modes.

For a face normal to axis ``a`` at side ``s ∈ {lo, hi}`` the
tangential H components are the two H components with indices
``≠ a``. On ``x_lo``: Hy[0, :, :] and Hz[0, :, :]. The order point in
the scan body mirrors :func:`rfx.boundaries.pec.apply_pec_faces`
(after the H update, before the next E update).
"""

from __future__ import annotations


def apply_pmc_faces(state, faces: set[str]) -> object:
    """Apply PMC (``H_tan = 0``) on specific boundary faces.

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
    tests/test_boundary_pmc_hi_faces.py for the regression lock).
    """
    if not faces:
        return state
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
