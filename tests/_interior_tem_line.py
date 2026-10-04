"""Asymmetric internal PEC lattice coax, independent of magnetic side walls.

Outer sheets: y=2,5 and z=2,3+gap. Inner PEC x filament: (y,z)=(3,3).
All coordinates are in cells. Exterior margins are 2/3 cells in y and
2/4 in z. The radial Ez feed is off the cross-section's symmetry planes.
The Dirichlet transverse-cell Laplacian gives C'/eps0=3.75 for gap=1
(the single free node has potential 1/4), hence Zc=eta0/3.75. For gap=4
we solve the same finite Dirichlet system; no S measurement enters Zc.
This is the TEM solution of the realized cells, not a round-wire formula.

Only the longitudinal ends are PMC (open circuits), on declared E nodes.
They are not transverse side walls. Both open stubs enter the line oracle.
Non-integral port/load declarations are snapped; the realized coordinates,
not the declarations, enter the longitudinal transform.
"""
from dataclasses import dataclass

import numpy as np

from rfx import Box, GaussianPulse, PolylineWire, Simulation, realized_pec_edge_masks
from rfx.boundaries.spec import BoundarySpec

ETA0 = 376.730313668
C0 = 299792458.0


@dataclass(frozen=True)
class Line:
    zc: float
    left: float
    length: float
    right: float
    port: tuple
    load: tuple
    gap: int
    dx: float
    shape: tuple


def impedance(gap):
    # Discrete Dirichlet problem on the realized transverse cells. The inner
    # filament is held at 1 V, the four outer sheets at 0 V. Unknown nodes
    # are y=3,4 and z=3..2+gap, excluding the filament at (3,3).
    unknown = [(y, z) for y in (3, 4) for z in range(3, 3 + gap) if (y, z) != (3, 3)]
    index = {node: k for k, node in enumerate(unknown)}
    a = 4 * np.eye(len(unknown))
    rhs = np.zeros(len(unknown))
    for node, k in index.items():
        y, z = node
        for neighbor in ((y-1, z), (y+1, z), (y, z-1), (y, z+1)):
            if neighbor == (3, 3):
                rhs[k] += 1
            elif neighbor in index:
                a[k, index[neighbor]] -= 1
    potentials = dict(zip(unknown, np.linalg.solve(a, rhs)))
    capacitance_over_eps0 = sum(1 - potentials.get(node, 0.)
                                for node in ((2, 3), (4, 3), (3, 2), (3, 4)))
    return float(ETA0 / capacitance_over_eps0)


def build(kind="lumped", *, dx=0.25e-3, cells=9, gap=1, ratio=2.,
          two=False, load="resistor", rlc=None, profile=None, port_factor=1.,
          load_shift=0):
    """One radial lumped edge or a gap-cell wire, away from exterior walls.

    The port is declared at x=1.23 dx and the load at (cells-2.69) dx;
    x snaps to nodes 1 and cells-3 on uniform grids. Open stubs are 1 and
    3 cells long. Domain z extends four cells past the upper conductor.
    """
    domain = (cells * dx, 8 * dx, (7 + gap) * dx)
    profiles = {} if profile is None else dict(
        dx_profile=np.asarray(profile), dy_profile=np.full(8, dx),
        dz_profile=np.full(7 + gap, dx))
    sim = Simulation(freq_max=10e9, domain=domain, dx=dx,
                     boundary=BoundarySpec(x="pmc", y="pec", z="pec"), **profiles)
    for y in (2 * dx, 5 * dx):
        sim.add(Box((0, y, 2 * dx), (domain[0], y, (3 + gap) * dx)), material="pec")
    for z in (2 * dx, (3 + gap) * dx):
        sim.add(Box((0, 2 * dx, z), (domain[0], 5 * dx, z)), material="pec")
    sim.add(PolylineWire(((0, 3 * dx, 3 * dx), (domain[0], 3 * dx, 3 * dx)),
                         radius=0), material="pec")
    port = (1.23 * dx, 3 * dx, 3 * dx)
    load_pos = ((cells - 2.69 + load_shift) * dx, 3 * dx, 3 * dx)
    zc = impedance(gap)
    for pos in ((port, load_pos) if two else (port,)):
        sim.add_port(position=pos, component="ez", impedance=zc * port_factor,
                     waveform=GaussianPulse(f0=5e9, bandwidth=1.6),
                     **({"extent": gap * dx} if kind == "wire" else {}))
    if not two:
        if load == "short":
            sim.add(PolylineWire((load_pos, (load_pos[0], load_pos[1], (3 + gap) * dx)),
                                 radius=0), material="pec")
        elif load != "open":
            resistance, inductance, capacitance = (ratio * zc, 0., 0.) if rlc is None else rlc
            for k in range(gap):
                sim.add_lumped_rlc(position=(load_pos[0], 3 * dx, (3 + k) * dx),
                                   component="ez", R=resistance / gap, L=inductance / gap, C=capacitance * gap,
                                   topology="parallel" if rlc is None else "series")
    grid = sim._build_grid() if profile is None else sim._build_nonuniform_grid()
    if profile is None:
        index = grid.position_to_index
        xs = np.arange(grid.shape[0]) * float(grid.dx)
    else:
        from rfx.nonuniform import position_to_index
        from rfx.geometry.rasterize_grid import coords_from_nonuniform_grid

        def index(position):
            return position_to_index(grid, position)

        xs = np.asarray(coords_from_nonuniform_grid(grid).x)
    # Inspect realized conductor edges before solving the line. No
    # material parameter or measured S value enters the electrostatic oracle.
    sheets, wires = [], []
    assemble = sim._assemble_materials if profile is None else sim._assemble_materials_nu
    materials = assemble(grid, pec_sheets=sheets, pec_wires=wires)
    hard = realized_pec_edge_masks(materials[3], sheets, wires)
    mid = int(index(((port[0] + load_pos[0]) / 2, 3 * dx, 3 * dx))[0])
    ex, ey, ez = (np.asarray(h)[mid] for h in hard)
    expected_ex = np.ones((4, gap + 2), dtype=bool)
    expected_ex[1:3, 1:-1] = False
    expected_ex[1, 1] = True
    np.testing.assert_array_equal(ex[2:6, 2:4 + gap], expected_ex)
    assert not ey[2:5, 3:3 + gap].any()
    assert not ez[3:5, 2:3 + gap].any()
    assert ey[2:5, 2].all() and ey[2:5, 3 + gap].all()
    assert ez[2, 2:3 + gap].all() and ez[5, 2:3 + gap].all()
    ip, il = index(port), index(load_pos)
    assert ip[1:] == il[1:] == (3, 3)
    assert grid.shape[1:] == (9, 8 + gap)
    line = Line(zc, float(xs[ip[0]]), float(xs[il[0]] - xs[ip[0]]),
                float(xs[-1] - xs[il[0]]), tuple(ip), tuple(il), gap, dx, tuple(grid.shape))
    return sim, line


def input_reflection(line, freqs, zload):
    """RLC in parallel with the far open stub, transformed, plus near stub."""
    beta = 2 * np.pi * np.asarray(freqs) / C0
    if np.all(np.asarray(zload) == 0):
        yin = (-1j / np.tan(beta * line.length) + 1j * np.tan(beta * line.left)) / line.zc
        return (1 - line.zc * yin) / (1 + line.zc * yin)
    yload = 1 / np.asarray(zload, dtype=complex) + 1j * np.tan(beta * line.right) / line.zc
    t = np.tan(beta * line.length)
    yin = (yload + 1j * t / line.zc) / (1 + 1j * line.zc * yload * t)
    yin += 1j * np.tan(beta * line.left) / line.zc
    return (1 - line.zc * yin) / (1 + line.zc * yin)


def two_port(line, freqs):
    """Shunt-open-stub / TEM line / shunt-open-stub ABCD network."""
    beta = 2 * np.pi * np.asarray(freqs) / C0
    out = np.empty((2, 2, len(beta)), dtype=complex)
    for k, b in enumerate(beta):
        near = np.array([[1, 0], [1j * np.tan(b * line.left) / line.zc, 1]])
        far = np.array([[1, 0], [1j * np.tan(b * line.right) / line.zc, 1]])
        tl = np.array([[np.cos(b * line.length), 1j * line.zc * np.sin(b * line.length)],
                       [1j * np.sin(b * line.length) / line.zc, np.cos(b * line.length)]])
        a, bb, c, d = (near @ tl @ far).ravel()
        den = a + bb / line.zc + c * line.zc + d
        out[:, :, k] = [[(a + bb / line.zc - c * line.zc - d) / den, 2 / den],
                        [2 / den, (-a + bb / line.zc - c * line.zc + d) / den]]
    return out


def extract_load(line, freqs, s11):
    """Undo the near stub, line transform, then the far stub (no calibration)."""
    beta = 2 * np.pi * np.asarray(freqs) / C0
    yin = (1 - s11) / (line.zc * (1 + s11))
    yin -= 1j * np.tan(beta * line.left) / line.zc
    t = np.tan(beta * line.length)
    yload = (yin - 1j * t / line.zc) / (1 - 1j * line.zc * yin * t)
    yload -= 1j * np.tan(beta * line.right) / line.zc
    return 1 / yload


def chain(kind, dut):
    """30 mm-class line; preserve a one-cell lumped and four-cell wire feed.

    Realized length is 30.1 mm at 100 um for lumped and 30.025 mm at
    25 um for wire. These resolutions keep the original 1% phase bars;
    the unequal open end stubs are respectively 1 and 3 local cells.
    """
    dx, cells, gap = (0.1e-3, 305, 1) if kind == "lumped" else (0.025e-3, 1205, 4)
    return build(kind, dx=dx, cells=cells, gap=gap,
                 ratio={"res_half": .5, "res_double": 2., "matched": 1.}.get(dut, 1.),
                 load=dut if dut in ("short", "open") else "resistor")
