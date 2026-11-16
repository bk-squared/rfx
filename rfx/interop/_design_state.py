"""Derived simulation state omitted from portable design documents."""

#: Attributes deliberately absent from the document.  These are derived,
#: cache or run-time state, never a user declaration.
#: None of these attributes is set by the constructor.
EXCLUDED_SIMULATION_ATTRS: tuple[str, ...] = (
    "_realization_early_outcomes",  # derived findings; rebuilt from the declarations
    "_mesh_resolution",
    "_boundary_default_warned",
)
