from __future__ import annotations

from typing import NamedTuple


class SubgridRunOptions(NamedTuple):
    """Bundled keyword inputs for :func:`run_subgridded_jit`.

    Groups the optional PEC masks, sources/probes, lumped-S-parameter, NTFF,
    interface flags, and material-SAT scale knobs into one immutable container
    so the runner entrypoint stays within a reasonable parameter count.
    ``run_subgridded_jit`` unpacks every field into a same-named local, so the
    field names match the historical keyword arguments exactly.
    """

    pec_mask_c: object | None = None
    pec_mask_f: object | None = None
    sources_f: list | None = None
    sources_c: list | None = None
    probe_indices_f: list | None = None
    probe_components: list | None = None
    probe_indices_c: list | None = None
    probe_components_c: list | None = None
    lumped_sparam_indices_f: list | None = None
    lumped_sparam_components_f: list | None = None
    lumped_sparam_impedances_f: list | None = None
    lumped_sparam_cell_counts_f: list | None = None
    lumped_sparam_freqs_f: object | None = None
    ntff_box_f: object | None = None
    ntff_data_f: object | None = None
    use_material_sat: bool | None = None
    sync_coarse_interface_from_fine: bool = False
    sync_coarse_shadow_from_fine: bool = False
    sync_box_coarse_shadow_from_fine: bool = False
    mask_coarse_shadow_interior: bool = False
    use_exterior_z_interfaces: bool = False
    use_boundary_terminated_exterior_z_interfaces: bool = False
    ghost_exterior_coarse_shadow_from_fine: bool = False
    material_sat_scale: float = 1.0
    material_sat_coarse_scale: float = 1.0
    material_sat_fine_scale: float = 1.0
    material_sat_e_coarse_scale: float = 1.0
    material_sat_e_fine_scale: float = 1.0
    material_sat_h_coarse_scale: float = 1.0
    material_sat_h_fine_scale: float = 1.0
    material_sat_zlo_scale: float = 1.0
    material_sat_zhi_scale: float = 1.0
    material_sat_e_zlo_scale: float = 1.0
    material_sat_e_zhi_scale: float = 1.0
    material_sat_h_zlo_scale: float = 1.0
    material_sat_h_zhi_scale: float = 1.0
    material_sat_pair_a_zlo_scale: float = 1.0
    material_sat_pair_b_zlo_scale: float = 1.0
    material_sat_zlo_common_trace_projection: str = "dual"
    material_sat_zhi_common_trace_projection: str = "dual"
    material_sat_normal_e_scale: float = 0.0
    material_sat_zhi_coarse_eps_blend: float = 0.0
    defer_material_h_sat_until_after_e: bool = False
    material_sat_face_projection: str = "node_adjoint"
    inject_sources_before_e_coupling: bool = False
    use_exterior_box_interfaces: bool = False
    inject_sources_on_coarse_shadow: bool = False


