"""Boundary setup shared by the unchanged uniform and graded operators."""


def oblique_bloch(tfsf, *, use_ntff, use_dft_planes, use_flux_monitors):
    use_tfsf = tfsf is not None
    # An oblique (2D-aux) TFSF injects a tilted plane wave that a plain-periodic
    # REAL grid cannot sustain (the pre-#404 under-tilt). Drive the shared solver
    # on a complex Bloch-envelope path instead: fields carry the envelope P and a
    # per-axis phase rides the periodic roll; the physical field is
    # Re(P·exp(-j k_t·y)). Gated strictly on the 2D-aux discriminator, so normal
    # incidence and every non-TFSF run stay real float32 and byte-identical.
    # NOTE: gated strictly on the 2D-aux (Bloch) discriminator. Open-domain
    # oblique Method B (MethodBConfig, NOT TFSF2DConfig) intentionally bypasses
    # this lane: it stays real float32 and its NTFF/DFT/flux monitors read
    # physical fields, so the fail-loud below must NOT be widened to angle != 0.
    _oblique_bloch = False
    if use_tfsf:
        from rfx.sources.tfsf import is_tfsf_2d as _is_tfsf_2d_early
        _oblique_bloch = _is_tfsf_2d_early(tfsf[0])
    if _oblique_bloch:
        # Frequency-domain monitors accumulate the complex envelope P, not the
        # physical spectrum — fail loud rather than return truncated garbage.
        _unsupported_ob = []
        if use_ntff:
            _unsupported_ob.append("NTFF box")
        if use_dft_planes:
            _unsupported_ob.append("DFT plane probe")
        if use_flux_monitors:
            _unsupported_ob.append("flux monitor")
        if _unsupported_ob:
            raise NotImplementedError(
                "Oblique (angle_deg != 0) TFSF uses the #404 complex Bloch path; "
                "frequency-domain monitors are not yet transform-aware on it. "
                "Unsupported: " + ", ".join(_unsupported_ob) + ". Use field "
                "snapshots / final state (returned as physical fields), or "
                "compute_rcs for open-domain oblique scattering."
            )

    return _oblique_bloch


def nonuniform_boundaries(grid, cpml_axes, pec_faces, pmc_faces):
    from types import SimpleNamespace
    from rfx.boundaries.axes import padded_axes
    # CPML: only initialize when cpml_layers > 0 (skip for PEC boundary)
    use_cpml = grid.cpml_layers > 0

    cpml_params = None
    cpml_state_init = None
    cpml_grid = None
    cpml_axes_eff = cpml_axes

    if use_cpml:
        from rfx.boundaries.cpml import init_cpml

        # Pass NonUniformGrid directly — init_cpml duck-types dx/dy/dz.
        # NonUniformGrid does not carry pmc_faces / pec_faces attrs
        # (frozen dataclass, pytree-registered), so the sets must be
        # threaded through from the caller.
        cpml_params, cpml_state_init = init_cpml(
            grid, pec_faces=pec_faces, pmc_faces=pmc_faces,
        )
        cpml_grid = grid
        cpml_axes_eff = padded_axes(grid, cpml_axes)

    # PMC enforcement (2026-04). The NU scan body previously never
    # zeroed H_tan on PMC faces, so a half-symmetric configuration
    # that relied on the mirror plane was running with an effectively
    # free boundary. Frozen set gives JIT cache a stable hash; empty
    # set short-circuits the apply to a no-op.
    # The electric walls, per face, by the one rule the uniform scan uses
    # (#1164): the NU grid carries no face attributes, so the declared sets
    # threaded in from the caller stand in for them. With no magnetic face
    # this is the six faces ``apply_pec`` zeroed before, plane for plane.
    from rfx.boundaries.pec import resolve_wall_faces as _resolve_walls
    _pec_faces_frozen, _pmc_faces_frozen = _resolve_walls(
        SimpleNamespace(pec_faces=set(pec_faces or ()), pmc_faces=set(pmc_faces or ()),
                        shape=(grid.nx, grid.ny, grid.nz)),
        (False, False, False), None)
    from rfx.core.yee import CurlBoundary
    curl_boundary = CurlBoundary(_pec_faces_frozen, _pmc_faces_frozen)
    use_pmc_faces = bool(_pmc_faces_frozen)

    return (use_cpml, cpml_params, cpml_state_init, cpml_grid, cpml_axes_eff,
            curl_boundary, _pec_faces_frozen, _pmc_faces_frozen, use_pmc_faces)
