"""Graded-mesh wire-port current: the discrete Ampere loop at the port cell (moved from rfx.nonuniform)."""


def _bwd_neighbor(h, idx, axis, boundary=None):
    """``h`` one cell back along ``axis``, using the SAME out-of-domain
    convention the NU E-update uses.

    ``rfx.core.yee._shift_bwd`` (rfx/core/yee.py) pads explicitly with zero, so ``_curl_h_nu`` reads 0 for H outside the
    domain. A raw ``h[i - 1]`` at ``i == 0`` instead reads the opposite domain face, making a local Ampere loop depend
    on a field cell it does not enclose.

    A length-1 axis carries no variation along itself, so the difference
    along it is 0 (return the cell itself, matching the uniform lane's
    2-D behaviour). NU grids are 3-D today, so that branch is defensive.
    """
    from rfx.core.yee import h_neighbor
    return h_neighbor(h, axis, index=idx, boundary=boundary)


def wire_port_current(hx, hy, hz, comp, mi, mj, mk,
                      dual_x, dual_y, dual_z, boundary=None):
    """Enclosed current from the discrete Ampere loop at a wire-port cell.

    For component ``a`` with ``(a, b, c)`` cyclic,
    ``curl_a = dH_c/db - dH_b/dc`` and the pierced DUAL face has area
    ``dual_b * dual_c``, so

        I_a = (H_c[b] - H_c[b-1]) * dual_c - (H_b[c] - H_b[c-1]) * dual_b

    i.e. each H-difference leg is weighted by the DUAL spacing along THAT H
    COMPONENT'S OWN axis (the leg's own length as a contour segment) — never
    by the axis it is differenced along, and never by a primal cell width.

    Using a primal width on one leg and a width from an unrelated axis on the other violates both metric requirements.
    Both errors are invisible on a uniform cubic
    mesh, which is why the uniform lane (``rfx/simulation.py``) can spell the
    whole thing with a single ``dx`` and still be correct there.



    Out-of-domain H is ZERO. Each backward read goes through
    :func:`_bwd_neighbor`, which pads the way ``_curl_h_nu`` /
    ``rfx.core.yee._shift_bwd`` pad, so the port current is the loop
    integral of the same curl the E-update integrates. The previous raw
    ``[mi - 1]`` spelling was a NEGATIVE index at ``mi == 0``, i.e. H at
    the opposite domain face, which a 4-term LOCAL loop must never touch.

    Scope: this is a stencil-correctness requirement. The opposite-face cell at the last index of the axis holds 0.0 at
    runtime — ``inv_d_h[N-1] = 0`` by construction
    (``_profile_to_inv_arrays``) and the last row is grid pad. Measured on
    a 21-cell NU run with an ez wire port flush against x_lo (``mi == 0``),
    sampling every H component's last plane on every axis at every one of
    300 steps: max|H| there was 0.000000e+00 throughout, under
    ``boundary="pec"`` and under ``boundary="cpml"``, against global
    max|H| of 1.9e+07 and 1.8e+04. So the two spellings agree on any state
    the stepper can produce, and the +1000 perturbation the oracles
    use is not reachable by the solver. Keep the fix — a local loop should
    not read a cell it does not enclose, and the two spellings agreeing
    only by a coincidence of the pad arrays is not a property worth
    depending on — but do not describe it as a live defect.
    """
    idx = (mi, mj, mk)
    if comp == "ez":
        return ((hy[idx] - _bwd_neighbor(hy, idx, 0, boundary)) * dual_y
                - (hx[idx] - _bwd_neighbor(hx, idx, 1, boundary)) * dual_x)
    if comp == "ex":
        return ((hz[idx] - _bwd_neighbor(hz, idx, 1, boundary)) * dual_z
                - (hy[idx] - _bwd_neighbor(hy, idx, 2, boundary)) * dual_y)
    if comp == "ey":
        return ((hx[idx] - _bwd_neighbor(hx, idx, 2, boundary)) * dual_x
                - (hz[idx] - _bwd_neighbor(hz, idx, 0, boundary)) * dual_z)
    raise ValueError(f"wire_port_current: unknown component {comp!r}")
