"""P0 material operands and drive units, by the nine admitted lanes.

For single-device E cells, the predicate is: the operand the lane's kernel
receives equals the shared helper applied to the lane's own cell array.
Placement of the cell array and defects inside the helper are not checked
by P0. Distributed E cells use the full-domain single-device helper result,
sliced to the rank's owned rows. Relative epsilon/mu are dimensionless, sigma is S/m, and drive_scale
multiplies the declared waveform to produce an E increment in V/m.
No numerical convention is changed by this table.
"""
from typing import NamedTuple

from tests.contracts.path_disposition import LANES


class Cell(NamedTuple):
    kind: str = "compares"
    issue: str = ""
    note: str = ""


ROWS = ("eps", "sigma", "mu", "conformal", "sat", "override_drive",
        "soft_field", "soft_current", "soft_none",
        "open_field", "open_current", "open_none",
        "lumped_none", "wire_none", "lumped_field", "lumped_current",
        "wire_field", "wire_current")
TABLE = {row: {lane: Cell() for lane in LANES} for row in ROWS}

# This note is source-reading evidence only, not another numerical cell.
NOTES = (("#1373", "READ-ONLY: no dual mu averaging rule exists in rfx/; "
          "the P0 reference is component_h_materials of the lane's own "
          "cell array, including its per-component wire-contour record "
          "(#1398, rfx/core/yee.py:component_h_materials)."),)

for row, feature in (("eps", "dielectric"), ("sigma", "lossy"), ("mu", "mu")):
    # Filled explicitly below where admission refuses a declared material.
    for lane in LANES:
        if lane in ("run_adi", "fwd_adi") and row == "mu":
            TABLE[row][lane] = Cell("refuses", note="ADI refuses magnetic materials")
        elif row == "mu":
            TABLE[row][lane] = Cell("call-site", note="no μ interface rule exists yet; P0's μ reference is component_h_materials of the lane's own cell array")
        elif row in ("eps", "sigma") and lane in ("run_adi", "fwd_adi"):
            TABLE[row][lane] = Cell(issue="#1373", note="per-cell E material")
        elif row in ("eps", "sigma") and lane in ("run_distributed", "fwd_distributed_nu"):
            TABLE[row][lane] = Cell(note="#1303, fixed by #1326")

for lane in LANES:
    TABLE["override_drive"][lane] = (Cell() if lane in ("fwd_uniform", "fwd_nonuniform", "fwd_distributed_nu")
                                     else Cell("not reachable",
                                               note="P0 override fixture exercises the three Yee forward lanes"))
    TABLE["conformal"][lane] = (Cell(note="#1373: four-cell dielectric epsilon before the 1/w scaling")
                                if lane == "run_uniform" else Cell("refuses"))
    TABLE["sat"][lane] = (Cell(issue="#1373", note="per-cell SAT face epsilon")
                          if lane == "run_subgridded" else Cell("not reachable",
                          note="no coarse/fine interface on this lane"))
    # ADI admits only explicit field declarations (#1373).
    if lane in ("run_adi", "fwd_adi"):
        for source in ("soft_none", "open_none", "soft_current", "open_current"):
            TABLE[source][lane] = Cell("refuses", note="ADI implements only field sources")
    for kind in ("field", "current", "none"):
        TABLE["open_" + kind]["run_subgridded"] = Cell("refuses",
            note="all-face absorber outside subgrid production envelope")
    for port in ("lumped", "wire"):
        if lane in ("run_adi", "fwd_adi", "fwd_distributed_nu"):
            TABLE[port + "_none"][lane] = Cell("refuses")
        for kind in ("field", "current"):
            TABLE[port + "_" + kind][lane] = Cell("not reachable",
                note="add_port has no amplitude_kind argument")

# Consumption witnesses are separate from comparison / admission kinds.
# J4 replays and perturbs each listed material quantity at non-SAT sites.
CONSUMPTION = {row: {lane: (
    "J4 replay and perturbation" if row in ("eps", "sigma") or
    (row == "mu" and lane not in ("run_adi", "fwd_adi")) else
    "no consumption witness: ADI refuses mu" if row == "mu" else
    "no consumption witness: SAT apply path not exercised" if row == "sat" else
    "J4 kernel replay: aniso.E sigma_e; no eps_e bit-identity witness" if row == "conformal" else
    "no consumption witness: drive tables not replayed"
) for lane in LANES} for row in ROWS}
for row in ("eps", "sigma"):
    for lane in LANES:
        old = TABLE[row][lane]
        predicate = ("full-domain single-device helper sliced to owned rows"
                     if lane in ("run_distributed", "fwd_distributed_nu") else
                     "the operand the lane's kernel receives equals the shared helper applied to the lane's own cell array")
        TABLE[row][lane] = old._replace(note=(old.note + "; " if old.note else "") + predicate)

# Site-level coverage: apply capability alone is not a consumption witness.
CONSUMPTION_SITES = {
    "yee.E": "J4: eps_e, sigma_e",
    "precompute.E": "J4 kernel replay: eps_e, sigma_e",
    "adi.E": "J4: eps_e, sigma_e",
    "distributed.E": "J4: eps_e, sigma_e",
    "distributed_nu.E": "J4: eps_e, sigma_e",
    "yee.H": "J4: mu_h",
    "precompute.H": "J4 kernel replay: mu_h",
    "aniso.E": "J4 kernel replay: sigma_e; no eps_e bit-identity witness",
    "sat.c": "no witness: apply exists; SAT faces not replayed",
    "sat.f": "no witness: apply exists; SAT faces not replayed",
    "distributed_nu.drive": "no witness: apply exists; runtime scales not replayed",
    **{site + ".sources": "no witness: source tables have no apply path"
       for site in ("uniform", "graded", "adi", "distributed",
                    "distributed_nu", "subgrid")},
}
