"""Boundary setup shared by the unchanged uniform and graded operators."""


def oblique_bloch(tfsf, *, use_ntff, use_dft_planes, use_flux_monitors,
                  aniso_eps=None, aniso_inv_eps=None, conformal_weights=None,
                  aniso_inv_eps_smooth=False, subpixel_smoothing=False,
                  node_smoothing=False, pec_present=False, use_upml=False):
    """On the oblique Bloch path, physical fields are ``Re(P * exp(-j * k_t * u))`` with ``u`` the transverse coordinate."""
    use_tfsf = tfsf is not None
    _oblique_bloch = False
    if use_tfsf:
        from rfx.sources.tfsf import is_tfsf_2d as _is_tfsf_2d_early
        _oblique_bloch = _is_tfsf_2d_early(tfsf[0])
    if _oblique_bloch:
        from rfx.errors import RfxUnsupported
        clauses, routes = [], []
        if use_upml:
            clauses.append("the UPML absorber")
            routes.append(", or the CPML absorber")
        if node_smoothing:
            clauses.append("off-diagonal node coupling (node_smoothing)")
            routes.append(", or no off-diagonal node coupling")
        if aniso_inv_eps is not None and not aniso_inv_eps_smooth and pec_present:
            if subpixel_smoothing == "kottke_pec":
                clauses.append("subpixel smoothing with a PEC shape")
                routes.append(", or subpixel_smoothing=False")
            else:
                clauses.append("an inverse permittivity tensor (aniso_inv_eps) with a PEC mask or occupancy")
                routes.append(", or no aniso_inv_eps")
        if aniso_eps is not None and conformal_weights is not None:
            clauses.append("conformal PEC")
            routes.append(", or conformal PEC off")
        if aniso_inv_eps is not None and aniso_inv_eps_smooth:
            clauses.append("a PEC occupancy override with pec_occupancy_model='kottke'")
            routes.append(", or no pec_occupancy_override")
        if clauses:
            what = " and ".join(clauses)
            it = "it" if len(clauses) == 1 else "them"
            route = "".join(routes)
            operation = "the off-diagonal node coupling" if node_smoothing else f"the electric-field update for {it}"
            raise RfxUnsupported(
                f"An oblique Bloch plane wave (angle_deg={tfsf[0].angle_deg}, method='bloch') "
                f"cannot be combined with {what}: {operation} "
                "does not apply the Bloch phase, so the fields would be wrong. "
                f"Use normal incidence{route}."
            )
        _unsupported_ob = []
        if use_ntff:
            _unsupported_ob.append("NTFF box")
        if use_dft_planes:
            _unsupported_ob.append("DFT plane probe")
        if use_flux_monitors:
            _unsupported_ob.append("flux monitor")
        if _unsupported_ob:
            raise NotImplementedError(
                "Oblique (angle_deg != 0) TFSF uses the complex Bloch path; "
                "frequency-domain monitors are not yet transform-aware on it. "
                "Unsupported: " + ", ".join(_unsupported_ob) + ". Use field "
                "snapshots / final state (returned as physical fields), or "
                "compute_rcs for open-domain oblique scattering."
            )

    return _oblique_bloch


def nonuniform_boundaries(grid, cpml_axes, pec_faces, pmc_faces, *, field_dtype=None):
    """PMC faces use odd tangential-magnetic-field images in the nonuniform curl."""
    from types import SimpleNamespace
    from rfx.boundaries.axes import padded_axes
    use_cpml = grid.cpml_layers > 0

    cpml_params = None
    cpml_state_init = None
    cpml_grid = None
    cpml_axes_eff = cpml_axes

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml

        cpml_params, cpml_state_init = init_cpml(
            grid, pec_faces=pec_faces, pmc_faces=pmc_faces,
            **({} if field_dtype is None else {"field_dtype": field_dtype}),
        )
        cpml_grid = grid
        cpml_axes_eff = padded_axes(grid, cpml_axes)

    from rfx.boundaries.pec import resolve_wall_faces as _resolve_walls
    _pec_faces_frozen, _pmc_faces_frozen = _resolve_walls(
        SimpleNamespace(pec_faces=set(pec_faces or ()), pmc_faces=set(pmc_faces or ()),
                        shape=(grid.nx, grid.ny, grid.nz)),
        tuple(a in grid.periodic_axes for a in "xyz"), None)
    from rfx.core.yee import CurlBoundary
    curl_boundary = CurlBoundary(_pec_faces_frozen, _pmc_faces_frozen, tuple(a in grid.periodic_axes for a in "xyz"))
    use_pmc_faces = bool(_pmc_faces_frozen)

    return (use_cpml, cpml_params, cpml_state_init, cpml_grid, cpml_axes_eff,
            curl_boundary, _pec_faces_frozen, _pmc_faces_frozen, use_pmc_faces)
