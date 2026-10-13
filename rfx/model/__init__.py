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
from rfx.model.graded_admission import admit_graded_profiles
from rfx.model.occupancy import select_occupancy_operator, with_edge_keep
from rfx.model.thin_conductors import refuse_design_films

__all__ = [
    "admit_adi_sources", "admit_graded_profiles", "distributed_kernel_inputs",
    "distributed_solve_inputs", "drawn_launch_eps", "forward_kernel_inputs",
    "forward_lumped_port_stage", "forward_port_stage", "forward_products", "forward_sheet_context",
    "prepare_port_occupancy", "refuse_design_films", "select_occupancy_operator",
    "stage_distributed_edges", "uniform_solve_inputs", "with_edge_keep",
]
