"""Derived simulation state omitted from portable design documents."""

#: Attributes deliberately absent from the document.  These are derived,
#: transient, or run-time state — see the module docstring, rule 4.
#: None of these attributes is set by the constructor.
EXCLUDED_SIMULATION_ATTRS: tuple[str, ...] = (
    "_realization_early_outcomes",  # derived findings; rebuilt from the declarations
    "_mesh_resolution",
    "_boundary_default_warned",
    # crop rectangles for the internal MSL DFT planes; exists only while
    # compute_msl_s_matrix runs and is removed on exit (never a design input)
    "_dft_plane_regions",
)
