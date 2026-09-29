"""P0 material operands and drive units, by the nine admitted lanes.

The reference is the uniform lane's material functions on each lane's own
grid. Relative epsilon/mu are dimensionless, sigma is S/m, and drive_scale
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
          "the P0 reference repeats materials.mu_r for Hx/Hy/Hz "
          "(base dce96837, rfx/core/yee.py:423,442-444)."),)

for row, feature in (("eps", "dielectric"), ("sigma", "lossy"), ("mu", "mu")):
    # Filled explicitly below where admission refuses a declared material.
    for lane in LANES:
        if lane in ("run_adi", "fwd_adi") and row == "mu":
            TABLE[row][lane] = Cell("refuses", note="ADI refuses magnetic materials")
        elif row in ("eps", "sigma") and lane in ("run_adi", "fwd_adi", "fwd_distributed_nu"):
            TABLE[row][lane] = Cell(issue="#1373", note="per-cell E material")
        elif row in ("eps", "sigma") and lane == "run_distributed":
            TABLE[row][lane] = Cell(issue="#1303", note="per-cell E material")

for lane in LANES:
    TABLE["override_drive"][lane] = (Cell() if lane == "fwd_distributed_nu"
                                     else Cell("not reachable",
                                               note="distributed graded runtime drive"))
    TABLE["conformal"][lane] = (Cell(issue="#1306", note="per-cell dielectric epsilon")
                                if lane == "run_uniform" else Cell("refuses"))
    TABLE["sat"][lane] = (Cell(issue="#1373", note="per-cell SAT face epsilon")
                          if lane == "run_subgridded" else Cell("not reachable",
                          note="no coarse/fine interface on this lane"))
    for prefix in ("soft", "open"):
        if lane in ("run_adi", "fwd_adi"):
            TABLE[prefix + "_current"][lane] = Cell("refuses")
        if lane in ("run_nonuniform", "fwd_nonuniform", "fwd_distributed_nu"):
            TABLE[prefix + "_none"][lane] = Cell(issue="#1373", note="graded legacy Cb/dV")
    TABLE["soft_none"]["fwd_uniform"] = Cell(issue="#1373", note="forward legacy Cb")
    for adi in ("run_adi", "fwd_adi"):
        TABLE["open_none"][adi] = Cell(issue="#1373", note="ADI legacy raw waveform")
    for kind in ("field", "current", "none"):
        TABLE["open_" + kind]["run_subgridded"] = Cell("refuses",
            note="all-face absorber outside subgrid production envelope")
    for port in ("lumped", "wire"):
        if lane in ("run_adi", "fwd_adi", "fwd_distributed_nu") or (
                port == "wire" and lane == "run_distributed"):
            TABLE[port + "_none"][lane] = Cell("refuses")
        elif lane in ("run_nonuniform", "fwd_nonuniform"):
            TABLE[port + "_none"][lane] = Cell(issue="#1266", note="Cb/dV rather than Cb/d_parallel")
        for kind in ("field", "current"):
            TABLE[port + "_" + kind][lane] = Cell("not reachable",
                note="add_port has no amplitude_kind argument")
