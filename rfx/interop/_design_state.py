"""Derived simulation state omitted from portable design documents.

Each DFT plane region in ``_dft_plane_regions`` is the crop rectangle the user passed to ``add_dft_plane_probe(region=)``.
"""

EXCLUDED_SIMULATION_ATTRS: tuple[str, ...] = (
    "_realization_early_outcomes",
    "_mesh_resolution",
    "_boundary_default_warned",
)


EXPORTED_LAZY_SIMULATION_ATTRS: tuple[str, ...] = ("_dft_plane_regions",)
