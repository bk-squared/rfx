"""The realized model (S1): one declaration + one grid -> one object every path reads.

``conductors`` holds the PEC edges, sheets, lossy-sheet operators and wires; materials and the
realized-model object follow (plan: rfx-archive rfx/plans/20261004-s1-realized-model/).
"""

# The conductor layer's interface to the five large callers (2.0 layer rule): the
# assembly and stage entry points they call. Preflight reporting, run-record timing and
# solve binding (auto_preflight, early_run_record, bind_solve_call) are not model assembly
# and stay out until they move to their own layer.
from rfx.model.conductors import (
    admit_adi_sources, distributed_kernel_inputs,
    distributed_solve_inputs, drawn_launch_eps, forward_kernel_inputs,
    forward_lumped_port_stage, forward_port_stage, forward_products, forward_sheet_context,
    prepare_port_occupancy, stage_distributed_edges, uniform_solve_inputs)
from rfx.model.conductors import PreparedSolve
from rfx.model.design_box import (
    DesignBoxCoeffs, design_box_bounds, design_box_window, held_edge_masks,
    resolve_design_box)
from rfx.model.graded_admission import (
    NominalProfile, admit_graded_profiles, traced_cell_mask,
)
from rfx.model.occupancy import select_occupancy_operator, with_edge_keep
from rfx.model.thin_conductors import refuse_design_films

from rfx.model.waveguide_modes import (
    WaveguidePort,
    WaveguidePortConfig,
    cutoff_frequency,
    init_waveguide_port,
    solve_rectangular_modes,
    init_multimode_waveguide_port,
    C0_LOCAL,
)

__all__ = [
    "DesignBoxCoeffs", "PreparedSolve", "design_box_bounds", "design_box_window",
    "held_edge_masks", "resolve_design_box",
    "NominalProfile", "admit_adi_sources", "admit_graded_profiles", "distributed_kernel_inputs",
    "distributed_solve_inputs", "drawn_launch_eps", "forward_kernel_inputs",
    "forward_lumped_port_stage", "forward_port_stage", "forward_products", "forward_sheet_context",
    "prepare_port_occupancy", "refuse_design_films", "select_occupancy_operator",
    "stage_distributed_edges", "traced_cell_mask", "uniform_solve_inputs", "with_edge_keep",
    "WaveguidePort",
    "WaveguidePortConfig",
    "cutoff_frequency",
    "init_waveguide_port",
    "solve_rectangular_modes",
    "init_multimode_waveguide_port",
    "C0_LOCAL",
]
