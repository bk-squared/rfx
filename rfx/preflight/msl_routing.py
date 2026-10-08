"""MSL calculator routing checks used by scoped preflight."""

def _validate_msl_sparameter_request_for_preflight(self) -> None:
    """Mirror ``compute_msl_s_matrix`` family-routing checks."""

    if not self._msl_ports:
        raise ValueError("No MSL ports registered. Call add_msl_port() first.")
    if self._ports or self._waveguide_ports or self._floquet_ports:
        raise NotImplementedError(
            "compute_msl_s_matrix() is defined only for add_msl_port(...) "
            "families in the current simulation. Use separate simulations "
            "for add_port(...), add_waveguide_port(...), or "
            "add_floquet_port(...) S-parameter workflows."
        )
    if self._tfsf is not None:
        raise NotImplementedError(
            "compute_msl_s_matrix() is not supported together with TFSF; "
            "TFSF is a plane-wave source, not an MSL port."
        )
    if self._coaxial_ports:
        raise NotImplementedError(
            "compute_msl_s_matrix() does not include add_coaxial_port(...); "
            "coaxial-port S-parameters need a separate validated V/I "
            "extraction and calibration contract."
        )
    if (
        self._dz_profile is not None
        or self._dx_profile is not None
        or self._dy_profile is not None
    ) and any(
        getattr(pe, "mode", "laplace") == "eigenmode"
        for pe in self._msl_ports
    ):
        raise NotImplementedError(
            "compute_msl_s_matrix() on a non-uniform mesh supports "
            "mode='laplace'/'uniform' (Ez static-Laplace feed) only; the "
            "eigenmode J+M launch needs the magnetic-source channel that "
            "the non-uniform runner does not carry. Use mode='laplace' "
            "(the add_msl_port default) on the graded-mesh lane."
        )
    if self._refinement is not None:
        raise NotImplementedError(
            "compute_msl_s_matrix() is not supported with SBP-SAT "
            "subgridding."
        )
    if self._solver == "adi":
        raise NotImplementedError(
            "compute_msl_s_matrix() is not supported with solver='adi'; "
            "use the uniform Yee solver."
        )


_validate_msl_sparameter_request_for_preflight.__qualname__ = (
    "_PreflightMixin._validate_msl_sparameter_request_for_preflight"
)
