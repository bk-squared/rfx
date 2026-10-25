"""The realized model (S1): one declaration + one grid -> one object every path reads.

``conductors`` holds the PEC edges, sheets, lossy-sheet operators and wires; materials and the
realized-model object follow (plan: rfx-archive rfx/plans/20261004-s1-realized-model/).
"""

# The conductor layer's interface to the five large callers (2.0 layer rule): the
# assembly and stage entry points they call. Preflight reporting, run-record timing and
# solve binding (auto_preflight, early_run_record, bind_solve_call) are not model assembly
# and stay out until they move to their own layer.
from rfx.model.conductors import (
    admit_adi_sources,
    distributed_kernel_inputs,
    distributed_solve_inputs,
    drawn_launch_eps,
    forward_kernel_inputs,
    forward_lumped_port_stage,
    forward_port_stage,
    forward_products,
    forward_sheet_context,
    prepare_port_occupancy,
    prepare_sheet_layout,
    PreparedSolve,
    stage_distributed_edges,
    stage_distributed_occupancy,
    uniform_solve_inputs,
    with_sheet_layout_reports,
)
from rfx.model.design_box import (
    design_box_bounds,
    design_box_window,
    DesignBoxCoeffs,
    held_edge_masks,
    resolve_design_box,
)
from rfx.model.graded_admission import (
    admit_graded_profiles,
    NominalProfile,
    traced_cell_mask,
)
from rfx.model.occupancy import (
    graded_conductor_masks,
    select_occupancy_operator,
    with_edge_keep,
    with_graded_edge_keep,
)
from rfx.model.source_coefficients import (
    current_source_samples,
)
from rfx.model.thin_conductors import (
    admit_thin_conductor,
    refuse_design_films,
)
from rfx.model.waveguide_modes import (
    C0_LOCAL,
    cutoff_frequency,
    init_multimode_waveguide_port,
    init_waveguide_port,
    PortMedium,
    solve_rectangular_modes,
    WaveguidePort,
    WaveguidePortConfig,
)
from rfx.model.waveguide_waveform import (
    waveguide_waveform_error,
)

__all__ = [
    "admit_adi_sources",
    "admit_graded_profiles",
    "admit_thin_conductor",
    "C0_LOCAL",
    "current_source_samples",
    "cutoff_frequency",
    "design_box_bounds",
    "design_box_window",
    "DesignBoxCoeffs",
    "distributed_kernel_inputs",
    "distributed_solve_inputs",
    "drawn_launch_eps",
    "forward_kernel_inputs",
    "forward_lumped_port_stage",
    "forward_port_stage",
    "forward_products",
    "forward_sheet_context",
    "graded_conductor_masks",
    "held_edge_masks",
    "init_multimode_waveguide_port",
    "init_waveguide_port",
    "NominalProfile",
    "PortMedium",
    "prepare_port_occupancy",
    "prepare_sheet_layout",
    "PreparedSolve",
    "refuse_design_films",
    "resolve_design_box",
    "select_occupancy_operator",
    "solve_rectangular_modes",
    "stage_distributed_edges",
    "stage_distributed_occupancy",
    "traced_cell_mask",
    "uniform_solve_inputs",
    "waveguide_waveform_error",
    "WaveguidePort",
    "WaveguidePortConfig",
    "with_edge_keep",
    "with_graded_edge_keep",
    "with_sheet_layout_reports",
]
