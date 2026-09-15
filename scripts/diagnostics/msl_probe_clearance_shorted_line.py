"""Historical shorted-line geometry from the #726 experiment.

Retired as an executable accuracy experiment. The finite ground and short
with open CPML boundaries do not establish exact single-mode |S11|=1.
The two arms also moved the source and load. On current main the historical
near arm puts its last probe ON the realized short front, despite the old
continuous-coordinate gap check passing. Neither the proposed bias nor an
unsettleable near-field mechanism follows from these runs.

Historical 2026-08-28 observations (DX=80 um): clean settling -45.5 dB;
near settling -6.1/-5.7/-4.7 dB at 150/300/600 periods. Those are dated
observations, not evidence of failure at every possible record length.
The source revision below uses DX=H_SUB/3 after #931 and must not inherit
those old numbers as current measurements.

The build function is retained only to reproduce the invalid geometry;
see docs/research_notes/issue726/audit_original_short.py. Use
msl_probe_clearance_bias.py for a current, source-fixed cv06b comparison.
"""
from __future__ import annotations

import numpy as np

from rfx import Box, Simulation

C0 = 299792458.0
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
# #931 §1.3, off-lattice interfaces: DX was 80 um, which is 3.175 cells of
# substrate. The trace sheet then snaps to the nearest node — 240 um, 14 um
# INSIDE the laminate — and the realized board is not the declared one. The
# contract does not paper that over with a tie rule, so the mesh is redrawn
# on-lattice at H_SUB/3 (84.67 um, the rung nearest the old 80 um, so the cell
# count and the run cost barely move). Both faces are now exact node planes.
# This is a fixture change: the 2026-08-28 verdict recorded above was measured
# at 80 um and is a dated result, not a prediction for the re-run.
DX = H_SUB / 3
FREQ_MAX = 6e9


N_PROBE_OFFSET = 10      # cells; same comb in both arms
N_PROBE_SPACING = 2      # cells
N_PROBES = 5


def _assert_realized_stack(sim, x_probe, y_probe):
    """Build-time gate (#931), no solve: on a trace column far from the
    shorting wall the realized z wall planes are the two declared ones —
    ground at z = 0 and trace at z = H_SUB — and nothing else.

    Under the pre-2.0 rule the ground's wall sat at z = -DX unless the
    drawing compensated for it, and that is exactly the class this fixture
    is meant to be free of. The shorting wall is a VOLUME and stands a wall
    on every plane it spans, by design, so the column is chosen away from it.

    The check is ``tests/_realized_geometry.assert_wall_planes``, the one
    spelling on this branch.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "tests"))
    from _realized_geometry import assert_wall_planes, node_index, realized
    rz = realized(sim)
    i = node_index(rz.grid, 0, x_probe)
    j = node_index(rz.grid, 1, y_probe)
    got = assert_wall_planes(sim, 2, expected_m=[0.0, H_SUB], ij=(i, j),
                             what="shorted-line stack")
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    z = np.asarray(coords_from_uniform_grid(rz.grid).z)
    print(f"  realized z wall planes: {z[got[0]]*1e3:.4f}, "
          f"{z[got[1]]*1e3:.4f} mm (declared ground 0.0000, trace "
          f"{H_SUB*1e3:.4f})  [#931 build-time gate, no solve]")


def build(line_len: float, port_x: float, short_x: float):
    """Reproduce historical input, including its invalid near-arm probe."""
    margin = 2e-3
    LX = short_x + margin
    clearance = 2 * (2 * H_SUB + 8 * DX)
    LY = W_TRACE + 2 * clearance
    LZ = H_SUB + 1.5e-3
    sim = Simulation(freq_max=FREQ_MAX, domain=(LX, LY, LZ), dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_c = LY / 2
    y_lo, y_hi = y_c - W_TRACE / 2, y_c + W_TRACE / 2
    # #931: printed copper is a SHEET on the substrate top plane. It used to
    # be drawn as a one-cell Box, whose single realized wall was its lo plane
    # — the same plane, by accident of the old rule.
    sim.add_thin_conductor(Box((0, y_lo, H_SUB), (short_x, y_hi, H_SUB)))
    # PEC short: a VOLUME wall from the ground plane up to the trace plane at
    # short_x. Under the contract it realizes tangential walls at BOTH z = 0
    # and z = H_SUB and shorts every normal edge between them, which is what
    # "short" means; the old drawing had to run one cell past the trace to get
    # one wall at the top.
    sim.add(Box((short_x - DX, y_lo, 0.0), (short_x, y_hi, H_SUB)),
            material="pec")
    # #931 migration rule 2: the ground plane used to be drawn ONE CELL BELOW
    # the board (`Box((0,0,-DX),(LX,LY,0))`) so that its only realized wall —
    # the lo node plane — landed at z = 0. That put the wall a cell below the
    # board it was meant to bound whenever the compensation was forgotten.
    # Declared as a sheet AT z = 0, no offset is needed.
    sim.add_thin_conductor(Box((0, 0, 0.0), (LX, LY, 0.0)))
    # Historical geometry: moving this comb ALSO moves the physical source
    # and load. It is not an observation-only comparison. With the auto-resolved
    # comb the near arm's deepest probe landed PAST the short and outside
    # the domain (2026-08-27) — the comb geometry must be pinned, and its
    # extent checked, before the solve.
    sim.add_msl_port(position=(port_x, y_c, 0.0), width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0,
                     n_probe_offset=N_PROBE_OFFSET,
                     n_probe_spacing=N_PROBE_SPACING, n_probes=N_PROBES)
    _assert_realized_stack(sim, 0.5 * short_x, y_c)
    deepest = port_x + (N_PROBE_OFFSET + (N_PROBES - 1) * N_PROBE_SPACING) * DX
    if deepest >= short_x:
        raise SystemExit(
            f"design error: deepest probe {deepest * 1e3:.2f} mm is at or "
            f"past the short {short_x * 1e3:.2f} mm — the comb must fit "
            "between the port and the reflector")
    return sim, deepest



def main():
    raise SystemExit(
        "Retired #726 accuracy fixture: its historical near probe touches "
        "the realized short and its unit-reflection premise is unproven. "
        "Use scripts/diagnostics/msl_probe_clearance_bias.py --out-dir <new-dir>."
    )


if __name__ == "__main__":
    main()
