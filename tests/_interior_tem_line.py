"""Asymmetric internal PEC lattice coax, independent of magnetic side walls.

Outer sheets: y=2,5 and z=2,3+gap. Inner PEC x filament: (y,z)=(3,3).
All coordinates are in cells. Exterior margins are 2/3 cells in y and
2/4 in z. The radial Ez feed is off the cross-section's symmetry planes.
The Dirichlet transverse-cell Laplacian gives C'/eps0=3.75 for gap=1
(the single free node has potential 1/4), hence Zc=eta0/3.75. For gap=4
we solve the same finite Dirichlet system; no S measurement enters Zc.
This is the TEM solution of the realized cells, not a round-wire formula.

The line continues through 20 CPML cells at each longitudinal end.
The port and load are shunts across a through line, not terminal loads.
Non-integral port/load declarations are snapped; the realized coordinates,
not the declarations, enter the longitudinal transform.
"""
from dataclasses import dataclass
import sys

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
          load_shift=0, cpml_layers=20, axial_positions=None, declared_separation=None,
          precision="float32"):
    """One radial lumped edge or a gap-cell wire, away from exterior walls.

    Default declarations snap to x nodes 1 and cells-3 before padding.
    axial_positions holds fixed physical declarations for mesh ladders.
    load_shift moves only the fixture; a mutation must retain its nominal oracle.
    """
    domain = (cells * dx, 8 * dx, (7 + gap) * dx)
    profiles = {} if profile is None else dict(
        dx_profile=np.asarray(profile), dy_profile=np.full(8, dx),
        dz_profile=np.full(7 + gap, dx))
    sim = Simulation(freq_max=10e9, domain=domain, dx=dx,
                     boundary=BoundarySpec(x="cpml", y="pec", z="pec"),
                     cpml_layers=cpml_layers, precision=precision, **profiles)
    for y in (2 * dx, 5 * dx):
        sim.add(Box((0, y, 2 * dx), (domain[0], y, (3 + gap) * dx)), material="pec")
    for z in (2 * dx, (3 + gap) * dx):
        sim.add(Box((0, 2 * dx, z), (domain[0], 5 * dx, z)), material="pec")
    sim.add(PolylineWire(((-(cpml_layers + 1) * dx, 3 * dx, 3 * dx),
                          (domain[0] + (cpml_layers + 1) * dx, 3 * dx, 3 * dx)),
                         radius=0), material="pec")
    xp, xl = axial_positions or (1.23 * dx, (cells - 2.69) * dx)
    port = (xp, 3 * dx, 3 * dx)
    load_pos = (xl + load_shift * dx, 3 * dx, 3 * dx)
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
        xs = (np.arange(grid.shape[0]) - grid.pad_x_lo) * float(grid.dx)
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
    # Every physical Ex layer, including both CPML pads, has the same
    # internal inner/outer conductor cross-section. Last Ex slot is unused.
    for ix in range(grid.shape[0] - 1):
        np.testing.assert_array_equal(np.asarray(hard[0])[ix, 2:6, 2:4 + gap], expected_ex)
    assert grid.pad_x_lo == grid.pad_x_hi == cpml_layers
    # The load index is the declaration's snap: Result/ForwardResult expose
    # no assembled RLC cell record. A moved load is independently caught by
    # the measured residual check. Ports are checked against solved specs by
    # assert_solved_ports after the scan, not certified by this recomputation.
    ip, il = index(port), index(load_pos)
    if not two and load == "short":
        # A PEC short DOES expose assembled edges: read the actual radial
        # bridge from the realized Ez mask, independently of its declaration.
        short_x = np.flatnonzero(np.asarray(hard[2])[:, 3, 3])
        assert len(short_x) == 1, f"expected one assembled short bridge, got {short_x}"
        il = (int(short_x[0]), 3, 3)
    assert ip[1:] == il[1:] == (3, 3)
    assert grid.shape[1:] == (9, 8 + gap)
    line = Line(zc, float(xs[ip[0]]), float(xs[il[0]] - xs[ip[0]]),
                float(xs[-1] - xs[il[0]]), tuple(ip), tuple(il), gap, dx, tuple(grid.shape))
    if declared_separation is not None:
        assert_realized_separation(line, declared_separation, axial_positions)
    return sim, line


def assert_realized_separation(line, declared, positions=None):
    """Fixed physical circuit across a ladder; tolerance is floating roundoff.

    1e-14 m is over nine orders below the smallest cell, not a mesh allowance.
    """
    print(f"dx={line.dx:.9g} m: declared d={declared:.12g} m; realized d={line.length:.12g} m", file=sys.stderr)
    np.testing.assert_allclose(line.length, declared, rtol=0, atol=1e-14)
    if positions is not None:
        np.testing.assert_allclose([line.left, line.left + line.length],
                                   positions, rtol=0, atol=1e-14)


# Fourier convention: DFT kernel exp(-j omega t); a delay has exp(-j beta d),
# and a series inductor has impedance +j omega L. Frequencies and reference
# planes below are exactly those requested from the solve (snapped Ez nodes).
MU0 = 1.25663706212e-6
CELL_L_COEFFICIENT = 0.214


def element_inductance(line):
    """Measured gap=1 lattice coefficient, not a fitted reference-plane shift."""
    assert line.gap == 1, "0.214 was measured only for the one-cell cross-section"
    return CELL_L_COEFFICIENT * MU0 * line.dx


def input_reflection(line, freqs, zload, *, element_l=0.):
    """Pure-load oracle by default; optional series L predicts its residual.

    Zt=Zload||Zc, r=(Zt-Zc)/(Zt+Zc)=-Zc/(2 Zload+Zc),
    q=r exp(-2j beta d). The near through branch gives Zjunction=Zc(1+q)/2.
    Add the source cell's series L at its reference terminal. L=0 yields
    S=(q-1)/(q+3). No product extraction or measured calibration enters here.
    """
    w = 2 * np.pi * np.asarray(freqs)
    zl = np.asarray(zload) + 1j * w * element_l
    # Complex infinity arithmetic otherwise produces nan in 2*zl. An absent
    # shunt has exactly zero load-junction reflection.
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(np.isinf(zl), 0., -line.zc / (2 * zl + line.zc))
    q = r * np.exp(-2j * w * line.length / C0)
    zin = line.zc * (1 + q) / 2 + 1j * w * element_l
    return (zin - line.zc) / (zin + line.zc)


def two_port(line, freqs, *, element_l=0.):
    """ABCD = series(jwL) shunt(1/Zc) TEM(d) shunt(1/Zc) series(jwL).

    With L=0: S11=(-2 cos(theta)-j sin(theta))/(4 cos(theta)+5j sin(theta)),
    S21=2/(4 cos(theta)+5j sin(theta)). Both references are Zc.
    """
    out = []
    for f in freqs:
        w = 2 * np.pi * f
        theta = w * line.length / C0
        series = np.array([[1, 1j * w * element_l], [0, 1]])
        shunt = np.array([[1, 0], [1 / line.zc, 1]])
        tl = np.array([[np.cos(theta), 1j * line.zc * np.sin(theta)],
                       [1j * np.sin(theta) / line.zc, np.cos(theta)]])
        a, b, c, d = (series @ shunt @ tl @ shunt @ series).ravel()
        den = a + b / line.zc + c * line.zc + d
        out.append([[(a + b / line.zc - c * line.zc - d) / den, 2 / den],
                    [2 / den, (-a + b / line.zc - c * line.zc + d) / den]])
    return np.moveaxis(np.array(out), 0, -1)


def extract_load(line, freqs, s11):
    """Invert the pure shunt network, without subtracting element inductance.

    q=(1+3S)/(1-S), r=q exp(2j beta d), Zload=-Zc(1+r)/(2r).
    Thus the extracted residual includes BOTH source and load cell effects.
    """
    q = (1 + 3 * s11) / (1 - s11)
    r = q * np.exp(4j * np.pi * np.asarray(freqs) * line.length / C0)
    return -line.zc * (1 + r) / (2 * r)


def residuals(measured, pure, predicted):
    return dict(measured_phase_deg=np.degrees(np.angle(measured / pure)).tolist(),
                predicted_phase_deg=np.degrees(np.angle(predicted / pure)).tolist(),
                measured_abs_ds=np.abs(measured - pure).tolist(),
                predicted_abs_ds=np.abs(predicted - pure).tolist())


def assert_predicted_residual(measured, pure, predicted):
    """D1(ii): each complex and signed phase residual within 10% of prediction.

    Replaces the old magnitude-only 0.05 bar with the measured cell model:
    |(Smeas-Spure)-(Spred-Spure)| <= 0.1 |Spred-Spure| in every bin.
    A zero-inductance prediction cannot erase the observed mesh residual.
    """
    delta = predicted - pure
    assert np.all(np.abs(measured - predicted) <= .1 * np.abs(delta)), (
        f"complex residual differs by >10% of prediction: {residuals(measured, pure, predicted)}")
    phase = np.angle(measured / pure)
    expected_phase = np.angle(predicted / pure)
    assert np.all(np.abs(phase - expected_phase) <= .1 * np.abs(expected_phase)), (
        f"phase residual differs by >10% of prediction: {residuals(measured, pure, predicted)}")


def assert_first_order(dxs, errors):
    """Report the fitted order and judge BOTH adjacent mesh ratios, per bin.

    Retain the existing |p-1| <= log(1.1/.9)/log(4) interval. Applying it
    separately to both adjacent pairs strengthens the old endpoint-only bar;
    a middle-mesh excursion can no longer disappear from the regression.
    """
    dxs, errors = np.asarray(dxs, dtype=float), np.asarray(errors, dtype=float)
    assert np.all(np.isfinite(errors) & (errors > 0)), "mesh errors must be finite and strictly positive"
    assert dxs.ndim == 1 and len(dxs) == len(errors) >= 3, "need at least three matching meshes"
    assert np.all(np.isfinite(dxs) & (dxs > 0)) and np.all(np.diff(dxs) < 0), "meshes must decrease strictly"
    denominator = np.log(dxs[:-1] / dxs[1:]).reshape((-1,) + (1,) * (errors.ndim - 1))
    pairs = np.log(errors[:-1] / errors[1:]) / denominator
    order = np.polyfit(np.log(dxs), np.log(errors), 1)[0]
    print(f"fitted orders: {order}; successive orders: {pairs}", file=sys.stderr)
    assert np.all(np.abs(pairs - 1) <= np.log(1.1 / .9) / np.log(4)), f"successive mesh orders: {pairs}"
    return order


def assert_solved_ports(result, line, kind="lumped", *, two=False):
    """Read cells from the scan's actual DFT specs, including graded metadata.

    run's S-matrix driver does not retain lumped specs on Result; its raw
    scan result does, so that fixture checks each raw record before extraction.
    """
    if isinstance(result, dict):
        entries = result[kind]
    else:
        entries = getattr(result, f"{kind}_port_sparams", None)
        if entries is None:
            entries = result.wire_port_sparams  # graded lumped lane uses wire metadata
    assert entries and len(entries) == (2 if two else 1), "missing solved port specs"
    expected = (line.port, line.load) if two else (line.port,)
    for (spec, _), cell in zip(entries, expected):
        if hasattr(spec, "i"):
            actual = (spec.i, spec.j, spec.k)
        elif hasattr(spec, "live_cells"):
            actual = tuple(spec.live_cells[0])
        else:
            actual = tuple(spec[13][0])
        assert actual == cell, f"solved port cell {actual} != declared snap {cell}"
    print(f"solved {kind} port cells={expected}; realized d={line.length:.12g} m", file=sys.stderr)


def chain(kind, dut, *, dx=None, separation=.0301):
    """Fixed lumped circuit across meshes; original four-cell wire unchanged."""
    if kind == "wire":
        assert dx is None and separation == .0301
        sim, line = build(kind, dx=.025e-3, cells=1205, gap=4,
                          ratio={"res_half": .5, "res_double": 2., "matched": 1.}.get(dut, 1.),
                          load=dut if dut in ("short", "open") else "resistor")
        # Preserve the exact declarations/geometry of the tracker-1549 case.
        assert_realized_separation(line, .030025)
        return sim, line
    dx = .1e-3 if dx is None else dx
    positions = (.1e-3, .1e-3 + separation)
    return build(kind, dx=dx, cells=round((separation + .4e-3) / dx), gap=1,
                 axial_positions=positions, declared_separation=separation,
                 ratio={"res_half": .5, "res_double": 2., "matched": 1.}.get(dut, 1.),
                 load=dut if dut in ("short", "open") else "resistor")
