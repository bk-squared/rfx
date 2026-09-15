"""rfx.fdfd.ports3d and rfx.fdfd.conductor: a lumped port against a lumped
resistor at low frequency, a TEM wire section between two lumped ports,
passivity of every S-matrix, the Leontovich lossy-wall guide against
Pozar's TE10 attenuation, and the gradients w.r.t. a load resistance and
a wall conductivity."""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.fdfd import conductor as cd
from rfx.fdfd import ports3d as p
from rfx.fdfd import yee3d as y
from tests._x64_compat import enable_x64

Z0 = 50.0
H_MM = 1e-3
A_WR90 = 22.86e-3
F0 = 10e9
SIGMA_CU = 5.8e7


def _fd4(f, x0, d):
    return (-f(x0 + 2 * d) + 8 * f(x0 + d) - 8 * f(x0 - d) + f(x0 - 2 * d)) / (12 * d)


def _steps(nx, ny, nz, h=H_MM):
    return jnp.full(nx, h), jnp.full(ny, h), jnp.full(nz, h)


def _max_sv(s) -> float:
    return float(jnp.linalg.svd(s, compute_uv=False).max())


# ----------------------------------------------------------------------------
# P1: a floating PEC plate above the bottom wall, fed by a lumped port and
# loaded by a lumped resistor through the same gap

def _plate_box(gap_cells: int):
    """8 x 8 x (gap + 3) mm box, a PEC plate one cell thick at ``gap_cells``
    above the bottom wall, not touching the side walls. Returns the model,
    steps, PEC masks and the port / resistor elements spanning the gap."""
    nx = ny = 8
    nz = gap_cells + 3
    spec = y.Yee3DSpec(nx, ny, nz)
    m = y.build(spec)
    cells = np.zeros((nx, ny, nz), dtype=bool)
    cells[1:nx - 1, 1:ny - 1, gap_cells] = True
    pec = y.pec_edges_from_cells(spec, cells)
    port = p.LumpedElement(2, (2, 2, 0), (3, 3, gap_cells))
    resistor = p.LumpedElement(2, (5, 5, 0), (6, 6, gap_cells))
    return m, _steps(nx, ny, nz), pec, port, resistor


def test_lumped_port_against_lumped_resistor_at_low_frequency():
    """P1. At 10 MHz the plate's capacitance to the walls (~0.3 pF, in
    parallel with R) and the loop inductance are ~1e-3 effects, so
    S11 = (R - Z0)/(R + Z0). Measured gaps 5.9e-4 / 1.1e-3 / 2.9e-3 for
    R = 25 / 50 / 100 (one-cell gap, purely imaginary, growing linearly
    with R and with frequency as the shunt capacitance predicts -- the
    ratio of the imaginary parts at 100 and 10 MHz is 9.99); the two-cell
    gap (series division of R and Z0 over two edges: gaps 1.9e-3 / 1.0e-4 /
    2.2e-3) and a two-column resistor (parallel division: 3.7e-4 / 1.2e-3 /
    3.0e-3) agree to the same level. Every S is passive (|S11| < 1)."""
    with enable_x64():
        f = 10e6
        m1, (dx, dy, dz), pec1, port1, res1 = _plate_box(1)
        res_2col = p.LumpedElement(2, (5, 4, 0), (6, 6, 1))
        m2, (dx2, dy2, dz2), pec2, port2, res2 = _plate_box(2)
        assert m2.n_unknowns < 2000
        for r in (25.0, 50.0, 100.0):
            want = (r - Z0) / (r + Z0)
            s = p.s_matrix(m1, f, None, dx, dy, dz, [port1], Z0, pec=pec1,
                           loads=[res1], load_impedances=[r])
            assert s.shape == (1, 1)
            gap = abs(complex(s[0, 0]) - want)
            assert gap < 2e-2, (r, complex(s[0, 0]))
            assert gap < 5e-3                       # measured <= 2.9e-3
            assert _max_sv(s) <= 1.0 + 1e-6
            s_2col = p.s_matrix(m1, f, None, dx, dy, dz, [port1], Z0, pec=pec1,
                                loads=[res_2col], load_impedances=[r])
            assert abs(complex(s_2col[0, 0]) - want) < 5e-3
            s_gap2 = p.s_matrix(m2, f, None, dx2, dy2, dz2, [port2], Z0, pec=pec2,
                                loads=[res2], load_impedances=[r])
            assert abs(complex(s_gap2[0, 0]) - want) < 5e-3, complex(s_gap2[0, 0])
        # the mismatch grows linearly with frequency (a shunt C, not a bug)
        s_hi = p.s_matrix(m1, 10 * f, None, dx, dy, dz, [port1], Z0, pec=pec1,
                          loads=[res1], load_impedances=[100.0])
        s_lo = p.s_matrix(m1, f, None, dx, dy, dz, [port1], Z0, pec=pec1,
                          loads=[res1], load_impedances=[100.0])
        ratio = abs(jnp.imag(s_hi[0, 0])) / abs(jnp.imag(s_lo[0, 0]))
        assert 9.0 < float(ratio) < 11.0, float(ratio)


# ----------------------------------------------------------------------------
# P2 / P3: a thin PEC wire (or a 2-cell strip) one cell above the bottom
# wall, fed by lumped ports at its two ends

def _line(width: int, nx=10, ny=30, nz=6, j_a=3, k_w=1):
    """Wire (``width = 1``: PEC Ey edges on one node column) or strip
    (``width`` node columns joined by Ex edges) at height ``k_w`` cells,
    from y-node ``j_a`` to ``ny - j_a``; single-edge (wire) or multi-column
    (strip) lumped ports at the two ends. Returns model, steps, pec, ports
    and the physical length between the ports."""
    m = y.build(y.Yee3DSpec(nx, ny, nz))
    j_b = ny - j_a
    i0 = nx // 2 - (width - 1) // 2
    ex, ey, ez = (np.zeros(s, dtype=bool) for s in y.edge_shapes(nx, ny, nz))
    ey[i0:i0 + width, j_a:j_b, k_w] = True
    if width > 1:
        ex[i0:i0 + width - 1, j_a:j_b + 1, k_w] = True
    ports = [p.LumpedElement(2, (i0, j_a, 0), (i0 + width, j_a + 1, k_w)),
             p.LumpedElement(2, (i0, j_b, 0), (i0 + width, j_b + 1, k_w))]
    return m, _steps(nx, ny, nz), (ex, ey, ez), ports, (j_b - j_a) * H_MM


def test_tem_wire_section_between_matched_lumped_ports_is_passive_and_reciprocal():
    """P2 + P3. A 24 mm bare-Yee-edge wire 1 mm above ground in a 10 x 6 mm
    PEC channel at 2 GHz (~lambda/6). A wire of radius r over ground has
    Zc = 60 acosh(h/r); with the usual ~0.2 h equivalent radius of a bare
    Yee edge that is ~130 ohm, so the ports are set to Z0 = 130 ohm. (The
    image impedance of the discrete line measured from its own Z-matrix is
    133.7 ohm; it is NOT used as Z0 because renormalising any symmetric
    reciprocal two-port to its image impedance gives S11 = 0 identically.)
    Measured: |S21| = 0.9997, |S11| = 0.024, |S21 - S12| = 3e-15, the
    largest singular value is 1 + 2e-15 (single-edge ports read the
    discrete power exactly) and the Z-matrix is invariant under a change
    of the ports' internal impedance (130 -> 50 ohm) to 5e-14. The residual S11 comes from
    the 133.7 vs 130 ohm mismatch (Gamma = 0.014) and from the port
    discontinuity: the vertical feed edge's own inductance and the gap and
    wire-end capacitances, which also stretch the electrical length by
    ~one cell (measured beta*l / (k0 L) = 1.046).

    The 2-cell strip with 3-column ports (Z0 = 50): reciprocity 5e-15,
    Z-matrix invariance 2.6e-6, max singular value 1 - 4.6e-6 (the
    multi-edge readout shows the non-uniformity as apparent loss)."""
    with enable_x64():
        f = 2e9
        m, (dx, dy, dz), pec, ports, length = _line(1)
        assert m.n_unknowns < 10000
        z0 = 130.0
        sol = p.s_matrix(m, f, None, dx, dy, dz, ports, z0, pec=pec, return_solution=True)
        s = sol.s
        assert abs(complex(s[1, 0])) >= 0.95
        assert abs(complex(s[0, 0])) <= 0.2
        assert abs(complex(s[1, 0] - s[0, 1])) <= 1e-6
        assert abs(_max_sv(s) - 1.0) <= 1e-6                 # measured 1e-15
        assert abs(complex(s[1, 0])) > 0.99 and abs(complex(s[0, 0])) < 0.05   # measured 0.9997 / 0.024
        z = p.z_matrix(sol)
        zc = jnp.sqrt(z[0, 0] ** 2 - z[0, 1] ** 2)
        assert abs(float(jnp.real(zc)) - 133.7) < 1.0 and abs(float(jnp.imag(zc))) < 1e-6
        k0 = 2 * np.pi * f / y.C0
        bl = jnp.arccos(z[0, 0] / z[0, 1])
        assert 1.0 < float(jnp.real(bl)) / (k0 * length) < 1.1
        sol50 = p.s_matrix(m, f, None, dx, dy, dz, ports, Z0, pec=pec, return_solution=True)
        z50 = p.z_matrix(sol50)
        assert float(jnp.max(jnp.abs(z50 - z))) < 1e-10 * float(jnp.max(jnp.abs(z)))
        assert float(jnp.max(jnp.abs(p.renormalize(z50, z0) - s))) < 1e-10
        assert abs(_max_sv(sol50.s) - 1.0) <= 1e-6

        # multi-column ports on a strip
        m, (dx, dy, dz), pec, ports, _ = _line(3)
        sol_a = p.s_matrix(m, f, None, dx, dy, dz, ports, Z0, pec=pec, return_solution=True)
        sol_b = p.s_matrix(m, f, None, dx, dy, dz, ports, 70.0, pec=pec, return_solution=True)
        assert abs(complex(sol_a.s[1, 0] - sol_a.s[0, 1])) <= 1e-6
        assert _max_sv(sol_a.s) <= 1.0 + 1e-6                # measured 1 - 4.6e-6
        assert _max_sv(sol_b.s) <= 1.0 + 1e-6
        za, zb = p.z_matrix(sol_a), p.z_matrix(sol_b)
        assert float(jnp.max(jnp.abs(za - zb))) < 1e-5 * float(jnp.max(jnp.abs(za)))   # measured 2.6e-6


# ----------------------------------------------------------------------------
# C1: lossy-wall rectangular guide

def _guide(nx: int, ny: int, nz: int, npml: int, pad: int = 0):
    """WR90 guide (a = 22.86 mm, nx cells across, b = ny cells) along z with
    PML at both z ends; ``pad = 1`` embeds it in a box one cell bigger so
    the walls can be an interior cell-mask conductor."""
    h = A_WR90 / nx
    n_x, n_y = nx + 2 * pad, ny + 2 * pad
    m = y.build(y.Yee3DSpec(n_x, n_y, nz, pml=(0, 0, 0, 0, npml, npml)))
    return m, (jnp.full(n_x, h), jnp.full(n_y, h), jnp.full(nz, h)), h


def _te10_source_and_profile(m, h, pad, k_src):
    """J_y = sin(pi x / a) on the guide nodes (zero on the pad) and the
    profile used for the modal projection."""
    n_x, n_y, _ = m.shape
    x = np.arange(n_x + 1) * h - pad * h
    phi = np.where((x >= -1e-12) & (x <= A_WR90 + 1e-12), np.sin(np.pi * x / A_WR90), 0.0)
    shapes = y.edge_shapes(*m.shape)
    jy = jnp.zeros(shapes[1], jnp.complex128).at[:, pad:n_y - pad, k_src].set(jnp.asarray(phi)[:, None])
    src = (jnp.zeros(shapes[0], jnp.complex128), jy, jnp.zeros(shapes[2], jnp.complex128))
    return src, jnp.asarray(phi)


def _amplitudes(ey, phi, pad, ks):
    n_y = ey.shape[1]
    return jnp.stack([jnp.sum(phi * jnp.mean(ey[:, pad:n_y - pad, k], axis=1)) for k in ks])


def _gamma_from_amplitudes(amps, h) -> complex:
    """Propagation constant from the three-point recurrence
    A[k-1] + A[k+1] = 2 cosh(gamma h) A[k], exact for any mix of the
    forward and backward wave (so the PML's residual reflection does not
    bias it); least squares over the planes."""
    a = np.asarray(amps)
    c = np.sum((a[:-2] + a[2:]) * np.conj(a[1:-1])) / (2 * np.sum(np.abs(a[1:-1]) ** 2))
    return complex(np.arccosh(c) / h)


def _pozar_alpha_c(f, a, b, sigma):
    """Pozar, Microwave Engineering, TE10 conductor attenuation (Np/m)."""
    w = 2 * np.pi * f
    k = w / y.C0
    beta = np.sqrt(k * k - (np.pi / a) ** 2)
    rs = np.sqrt(w * y.MU0 / (2 * sigma))
    return rs * (2 * b * np.pi ** 2 + a ** 3 * k * k) / (a ** 3 * b * beta * k * y.ETA0)


def _lossy_gamma(nx, ny, nz, npml, sigma, interior: bool):
    pad = 1 if interior else 0
    m, (dx, dy, dz), h = _guide(nx, ny, nz, npml, pad)
    if interior:
        cells = np.zeros(m.shape, dtype=bool)
        cells[0], cells[-1], cells[:, 0], cells[:, -1] = True, True, True, True
        cond = cd.Conductor(cells=cells)
    else:
        cond = cd.box_walls(x=True, y=True)
    terms = cd.surface_terms(m, F0, cond, jnp.asarray(sigma), dx, dy, dz)
    k_src = npml + 3
    src, phi = _te10_source_and_profile(m, h, pad, k_src)
    e = y.solve(m, F0, None, dx, dy, dz, src, None, terms)
    ks = np.arange(k_src + 4, nz - npml - 1)
    return _gamma_from_amplitudes(_amplitudes(e[1], phi, pad, ks), h), m.n_unknowns


def test_lossy_wall_guide_matches_pozar_te10_attenuation_and_converges():
    """C1. Copper walls (5.8e7 S/m, R_s = 26 mohm) at 10 GHz on a WR90 guide
    with b = 3 cells; alpha from the modal amplitudes on 20 planes vs
    Pozar's alpha_c = 0.03555 Np/m. Measured relative error +1.50e-3 with
    24 cells across a and +7.25e-4 with 32 (ratio 0.48 ~ (24/32)^2: the
    H_tan-at-half-a-cell sampling and the grid dispersion are both second
    order for the even TE10 wall field). The same walls built as an
    interior cell-mask conductor in a box one cell larger reproduce the
    outer-wall gamma to 5e-15 (the cut-cell rows reduce to the native
    half-dual-cell rows)."""
    with enable_x64():
        errs = []
        for nx, ny in ((24, 3), (32, 4)):
            g, n = _lossy_gamma(nx, ny, 44, 10, SIGMA_CU, interior=False)
            assert n <= 60000
            b = ny * A_WR90 / nx
            alpha_ref = _pozar_alpha_c(F0, A_WR90, b, SIGMA_CU)
            errs.append(abs(g.real - alpha_ref) / alpha_ref)
            beta_disc = y.te10_discrete_beta(F0, A_WR90, A_WR90 / nx, A_WR90 / nx)
            assert abs(g.imag - beta_disc) < 2e-3 * beta_disc   # walls perturb beta slightly
        assert errs[0] < 0.10 and errs[1] < 0.10, errs
        assert errs[1] < errs[0], errs
        assert errs[0] < 5e-3                                  # measured 1.5e-3
        g_out, _ = _lossy_gamma(24, 3, 44, 10, 1e6, interior=False)
        g_in, _ = _lossy_gamma(24, 3, 44, 10, 1e6, interior=True)
        assert abs(g_out - g_in) < 1e-10 * abs(g_out), (g_out, g_in)


# ----------------------------------------------------------------------------
# C2: gradients

def test_gradients_wrt_load_resistance_and_wall_conductivity_match_fd4_and_jit():
    """C2. dS11/dR of the P1 box (forward mode on the complex S11 and
    reverse mode on Re S11) vs FD4 with a 2 ohm step (moves S11 by ~1e-2;
    measured 2.6e-7 relative, the FD4 truncation of (R - Z0)/(R + Z0)),
    and d|S21|^2/d sigma of the copper-wall guide section (|S21|^2 =
    |a2|^2 / |a2_cal|^2 against a PEC-wall calibration solve) vs FD4 with
    a 0.05 sigma step (moves |S21|^2 by 3.0e-5 >> the 1e-9 LU floor; the
    step is small enough that the sigma^-1/2 FD4 truncation is ~1.2e-5,
    which is the measured agreement 1.24e-5 -- the FD, not the gradient,
    is the limit). jvp vs grad 3.9e-14. jit of both values and gradients
    equals eager to the LU noise (see the inline numbers)."""
    with enable_x64():
        f = 10e6
        m, (dx, dy, dz), pec, port, res = _plate_box(1)

        def s11(r):
            return p.s_matrix(m, f, None, dx, dy, dz, [port], Z0, pec=pec,
                              loads=[res], load_impedances=[r])[0, 0]

        r0 = jnp.asarray(75.0)
        _, jv = jax.jvp(s11, (r0,), (jnp.asarray(1.0),))
        fd = _fd4(lambda r: complex(s11(r)), 75.0, 2.0)
        assert abs(complex(jv) - fd) < 1e-4 * abs(fd), (complex(jv), fd)
        assert abs(complex(jv) - 2 * Z0 / (75.0 + Z0) ** 2) < 1e-2 * abs(fd)
        g = jax.grad(lambda r: jnp.real(s11(r)))(r0)
        assert abs(float(g) - fd.real) < 1e-4 * abs(fd.real)
        # jit re-orders the assembly's floating ops; at 10 MHz (cond ~1e8) the
        # LU noise shows up at 1.6e-12 on S11 and 1.7e-14 relative on dS11/dR
        assert abs(complex(jax.jit(s11)(r0)) - complex(s11(r0))) < 1e-10
        g_jit = jax.jit(jax.grad(lambda r: jnp.real(s11(r))))(r0)
        assert abs(float(g_jit) - float(g)) < 1e-12 * abs(float(g))

        nx, ny, nz, npml = 24, 3, 44, 10
        mg, (gx, gy, gz), h = _guide(nx, ny, nz, npml)
        k_src, k_ref = npml + 3, nz - npml - 3
        src, phi = _te10_source_and_profile(mg, h, 0, k_src)
        cond = cd.box_walls(x=True, y=True)
        geo = cd.conductor_geometry(mg, cond)
        e_cal = y.solve(mg, F0, None, gx, gy, gz, src)
        a_cal = _amplitudes(e_cal[1], phi, 0, [k_ref])[0]

        def t21(sigma):
            terms = cd.surface_terms(mg, F0, cond, sigma, gx, gy, gz, geo=geo)
            e = y.solve(mg, F0, None, gx, gy, gz, src, None, terms)
            return jnp.abs(_amplitudes(e[1], phi, 0, [k_ref])[0] / a_cal) ** 2

        s0 = jnp.asarray(SIGMA_CU)
        val = float(t21(s0))
        assert 0.99 < val < 1.0                                    # ~ exp(-2 alpha L)
        gs = jax.grad(t21)(s0)
        fd_s = _fd4(lambda s: float(t21(s)), SIGMA_CU, 0.05 * SIGMA_CU)
        assert abs(fd_s) * 0.05 * SIGMA_CU > 1e-6                  # the step moves the objective
        assert abs(float(gs) - fd_s) < 1e-4 * abs(fd_s), (float(gs), fd_s)   # measured 1.24e-5
        _, jv_s = jax.jvp(t21, (s0,), (jnp.asarray(1.0),))
        assert abs(float(jv_s) - float(gs)) < 1e-10 * abs(float(gs))          # measured 3.9e-14
        v_jit = jax.jit(t21)(s0)
        assert abs(float(v_jit) - val) < 1e-12                                 # measured 2.6e-14
        g_jit = jax.jit(jax.grad(t21))(s0)
        assert abs(float(g_jit) - float(gs)) < 1e-10 * abs(float(gs))         # measured 4.7e-13


def test_input_checks():
    with enable_x64():
        m, (dx, dy, dz), pec, port, res = _plate_box(1)
        with pytest.raises(ValueError, match="outside"):
            p.element_edges(m, p.LumpedElement(2, (2, 2, 0), (3, 3, 9)))
        with pytest.raises(ValueError, match="axis"):
            p.element_edges(m, p.LumpedElement(3, (0, 0, 0), (1, 1, 1)))
        with pytest.raises(ValueError, match="load"):
            p.s_matrix(m, 1e7, None, dx, dy, dz, [port], Z0, pec=pec, loads=[res])
        with pytest.raises(ValueError, match="cells"):
            cd.conductor_geometry(m, cd.Conductor(cells=np.zeros((2, 2, 2), bool)))
        with pytest.raises(ValueError, match="diag_add"):
            y.assemble(m, 1e7, None, dx, dy, dz, p.current_source(m, port, dx, dy, dz),
                       terms=y.BoundaryTerms(diag_add=jnp.zeros(3)))


if __name__ == "__main__":
    t = time.time()
    pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    print(f"{time.time() - t:.1f} s")
