"""Unit locks on the two MSL extractor defects fixed in issues #511 and #507.

Both were invisible to every existing test because both were *systematic*: the
modal voltage was biased by a constant factor on every fixture, and the
single-ratio assembly was self-consistent with itself. So these tests pin the
two definitions directly, with no FDTD, and each has a mutation twin that fails
if the defect is reintroduced.

#511 — ``msl_modal_voltage`` summed ``k_lo..k_hi`` (n+1 Ez edges over an n-cell
substrate). ``k_hi`` is the TRACE cell: ``_msl_yz_cells`` builds an inclusive
span up to ``position_to_index(h_sub) = round(h_sub/dx)``, and ``k_top = k_hi``
is where the PEC search starts. The extra edge lives inside the one-cell PEC
trace, where ``apply_pec_mask`` deliberately preserves normal E as surface
charge, so it was live and contributed about -12%.

#507 — ``S[j, d] = b_j / a_d`` is the d-th column of the true S only when
``a_j = 0`` at every passive port. Measured ``|a_passive/a_driven| = 0.07-0.51``.
"""

import json
import warnings

import jax
import numpy as np
import pytest

try:
    # Modern JAX (scoped x64 context manager promoted to top-level).
    from jax import enable_x64
except ImportError:  # older JAX — jax.experimental.enable_x64 removed in 0.8
    from jax.experimental import enable_x64

from rfx.api._sparams import msl_modal_voltage, msl_solve_s_from_waves

# x64 is scoped PER TEST via the context manager below, never at module level:
# jax.config.update at import time flips at pytest collection and reds every
# same-process pytest-split shard.
#
# The algebra claims (exact recovery, the exact single-ratio error term) are
# statements about the SOLVE, so they are checked in f64 where "exact" is
# meaningful. One test pins the production f32 path separately, because that is
# what ships.
F64_TOL = 1e-12
F32_TOL = 1e-5

# --------------------------------------------------------------------------
# #511 — modal voltage spans the substrate, not the trace cell
# --------------------------------------------------------------------------

N_FREQS = 4
NY, NZ = 9, 8
J_CENTRE = 4
K_LO, K_HI = 0, 3           # 3-cell substrate, trace cell at k=3


def _ez_plane(values_by_k, *, ny=NY, nz=NZ, n_freqs=N_FREQS):
    """(n_freqs, ny, nz) plane whose centre column carries ``values_by_k``."""
    ez = np.zeros((n_freqs, ny, nz), dtype=np.complex128)
    for k, val in values_by_k.items():
        ez[:, J_CENTRE, k] = val
    return ez


def test_modal_voltage_sums_exactly_the_substrate_edges():
    """V = sum over k_lo..k_hi-1, each weighted by its own dz."""
    dz = np.full(NZ, 2.0)
    ez = _ez_plane({0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0})
    v = np.asarray(msl_modal_voltage(
        ez, j_centre=J_CENTRE, k_lo=K_LO, k_hi=K_HI, dz_arr=dz))
    # (1 + 2 + 4) * 2.0 — the k=3 trace cell's 8.0 must NOT appear.
    np.testing.assert_allclose(v, np.full(N_FREQS, 14.0))
    assert v.shape == (N_FREQS,)


def test_trace_cell_field_cannot_influence_the_modal_voltage():
    """MUTATION TWIN for #511.

    The trace cell's Ez is a real, live field (normal E survives on a one-cell
    PEC sheet), so a test that merely checks V is 'reasonable' passes both
    before and after the fix. This one makes the trace cell's value enormous
    and demands V be BIT-IDENTICAL. It fails if ``k_hi`` is ever summed again.
    """
    dz = np.full(NZ, 1.0)
    base = {0: 1.0, 1: 1.0, 2: 1.0}
    v_quiet = np.asarray(msl_modal_voltage(
        _ez_plane({**base, 3: 0.0}),
        j_centre=J_CENTRE, k_lo=K_LO, k_hi=K_HI, dz_arr=dz))
    v_loud = np.asarray(msl_modal_voltage(
        _ez_plane({**base, 3: -1.0e6}),
        j_centre=J_CENTRE, k_lo=K_LO, k_hi=K_HI, dz_arr=dz))
    assert np.array_equal(v_quiet, v_loud), (
        f"the trace cell leaked into V: {v_quiet[0]} vs {v_loud[0]}. "
        "This is issue #511 — the integral must span k_lo..k_hi-1."
    )
    # And the sign is the one that mattered: a negative trace-cell Ez is what
    # made the shipped V read ~12% LOW.
    v_bad = v_quiet + (-1.0e6) * 1.0
    assert not np.allclose(v_loud, v_bad)


@pytest.mark.parametrize("n_sub", [1, 3, 5, 7])
def test_edge_count_equals_substrate_cell_count(n_sub):
    """n substrate cells must contribute exactly n edges, at any mesh."""
    nz = n_sub + 4
    dz = np.full(nz, 1.0)
    ez = _ez_plane({k: 1.0 for k in range(nz)}, nz=nz)
    v = np.asarray(msl_modal_voltage(
        ez, j_centre=J_CENTRE, k_lo=0, k_hi=n_sub, dz_arr=dz))
    np.testing.assert_allclose(v, np.full(N_FREQS, float(n_sub)))


def test_modal_voltage_honours_per_cell_dz():
    """Graded meshes: each edge carries its own dz, not a uniform dx."""
    dz = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 1.0, 1.0, 1.0])
    ez = _ez_plane({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
    v = np.asarray(msl_modal_voltage(
        ez, j_centre=J_CENTRE, k_lo=0, k_hi=3, dz_arr=dz))
    np.testing.assert_allclose(v, np.full(N_FREQS, 0.5 + 1.0 + 2.0))


def test_zero_substrate_span_fails_loudly():
    """A port whose height rasterises to no substrate cell must raise."""
    dz = np.full(NZ, 1.0)
    ez = _ez_plane({0: 1.0})
    with pytest.raises(ValueError, match="trace"):
        msl_modal_voltage(ez, j_centre=J_CENTRE, k_lo=2, k_hi=2, dz_arr=dz)


# --------------------------------------------------------------------------
# #507 — the multi-drive solve, and what the single-ratio rule gets wrong
# --------------------------------------------------------------------------

def _planted_s(n_freqs=N_FREQS):
    """A reciprocal, lossless-ish 2-port with a non-trivial reflection."""
    s11 = 0.25 * np.exp(1j * np.linspace(0.2, 1.1, n_freqs))
    s21 = np.sqrt(1.0 - np.abs(s11) ** 2) * np.exp(
        1j * np.linspace(-0.7, -2.3, n_freqs))
    S = np.zeros((2, 2, n_freqs), dtype=np.complex128)
    S[0, 0], S[1, 1] = s11, s11
    S[1, 0], S[0, 1] = s21, s21
    return S


def _waves_from(S, gamma_passive):
    """Forward-model (a, b) per drive with a REFLECTING passive port.

    For drive ``d`` the incident vector is ``e_d + gamma * e_{1-d}``: the
    passive port is not matched, which is exactly the condition the
    single-ratio rule assumes away.
    """
    n_f = S.shape[-1]
    wave_a = [[None, None], [None, None]]
    wave_b = [[None, None], [None, None]]
    for d in (0, 1):
        a_vec = np.zeros((2, n_f), dtype=np.complex128)
        a_vec[d] = 1.0
        a_vec[1 - d] = gamma_passive
        b_vec = np.einsum("ijf,jf->if", S, a_vec)
        for j in (0, 1):
            wave_a[d][j] = a_vec[j]
            wave_b[d][j] = b_vec[j]
    return wave_a, wave_b


@pytest.mark.parametrize("gamma", [0.0, 0.07, 0.2, 0.51])
def test_multi_drive_solve_recovers_the_planted_s(gamma):
    """Exact recovery at every passive-port reflection, including the
    0.07-0.51 range measured on real fixtures."""
    S = _planted_s()
    wave_a, wave_b = _waves_from(S, gamma)
    with enable_x64():
        S_out, cond_a = msl_solve_s_from_waves(wave_a, wave_b)
        S_out = np.asarray(S_out)
    np.testing.assert_allclose(S_out, S, rtol=F64_TOL, atol=F64_TOL)
    assert cond_a is not None and np.all(np.isfinite(cond_a))


def test_single_ratio_rule_is_wrong_by_gamma_times_the_other_column():
    """MUTATION TWIN for #507, as an exact closed form.

    The superseded rule gives ``S_sr[j,d] = S[j,d] + gamma * S[j,1-d]``. Pinning
    the error exactly means this test fails if the assembly ever reverts, and
    also fails if someone 'fixes' it by fudging a tolerance.
    """
    gamma = 0.2
    S = _planted_s()
    wave_a, wave_b = _waves_from(S, gamma)
    with enable_x64():
        S_solved = np.asarray(msl_solve_s_from_waves(wave_a, wave_b)[0])

    S_sr = np.zeros_like(S)
    for d in (0, 1):
        for j in (0, 1):
            S_sr[j, d] = wave_b[d][j] / wave_a[d][d]

    for d in (0, 1):
        for j in (0, 1):
            np.testing.assert_allclose(
                S_sr[j, d] - S[j, d], gamma * S[j, 1 - d],
                rtol=F64_TOL, atol=F64_TOL,
            )
    # ... and the solve does not have that error.
    assert np.max(np.abs(S_solved - S)) < F64_TOL
    assert np.max(np.abs(S_sr - S)) > 0.1 * gamma


def _single_ratio(wave_a, wave_b, n_ports=2, n_freqs=N_FREQS):
    """The superseded assembly, for comparison only."""
    S_sr = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex128)
    for d in range(n_ports):
        for j in range(n_ports):
            S_sr[j, d] = wave_b[d][j] / wave_a[d][d]
    return S_sr


def test_single_ratio_rule_breaks_unitarity_while_the_solve_preserves_it():
    """General statement: the old rule loses unitarity, the solve keeps it.

    Deliberately NOT "the old rule inflates column power". Expanding the error
    term for a symmetric planted S gives

        col_sr = (1 + gamma^2)*(|S11|^2 + |S21|^2) + 4*gamma*Re(S11*conj(S21))

    whose last term carries the relative phase, so the old rule can push column
    power either side of 1. It inflated on the measured fixtures; that is a
    property of those fixtures, not of the rule. What IS general is that
    unitarity is lost.
    """
    gamma = 0.2
    S = _planted_s()
    np.testing.assert_allclose(
        np.abs(S[0, 0]) ** 2 + np.abs(S[1, 0]) ** 2, 1.0, rtol=F64_TOL)

    wave_a, wave_b = _waves_from(S, gamma)
    with enable_x64():
        S_solved = np.asarray(msl_solve_s_from_waves(wave_a, wave_b)[0])
    col_solved = np.abs(S_solved[0, 0]) ** 2 + np.abs(S_solved[1, 0]) ** 2
    np.testing.assert_allclose(col_solved, 1.0, rtol=1e-10)

    col_sr = np.sum(np.abs(_single_ratio(wave_a, wave_b)[:, 0]) ** 2, axis=0)
    assert np.all(np.abs(col_sr - 1.0) > 1e-3), (
        f"the single-ratio rule should lose unitarity here, got {col_sr}"
    )


def test_near_matched_line_reproduces_the_measured_one_plus_gamma_squared():
    """The relation actually measured on the thru fixtures.

    When the true S11 is ~0 (a uniform line referenced near its own Zc), the
    old rule gives |S_sr[0,0]| = gamma and |S_sr[1,0]| = |S21| = 1, so

        col_sr = 1 + gamma^2 = 1 + |S_sr[0,0]|^2

    which is the ``col - (1 + |S11|^2) = -1.4e-4 .. +3.0e-5`` recorded in #507.
    The excess is the far port's echo counted a second time. This is the
    NEAR-MATCHED case, not a general identity.
    """
    n_f = N_FREQS
    S = np.zeros((2, 2, n_f), dtype=np.complex128)
    phase = np.exp(1j * np.linspace(-0.4, -2.9, n_f))
    S[1, 0] = S[0, 1] = phase          # perfect thru, zero reflection
    for gamma in (0.05, 0.2, 0.45):
        wave_a, wave_b = _waves_from(S, gamma)
        S_sr = _single_ratio(wave_a, wave_b)
        col_sr = np.sum(np.abs(S_sr[:, 0]) ** 2, axis=0)
        np.testing.assert_allclose(col_sr, 1.0 + gamma ** 2, rtol=1e-12)
        np.testing.assert_allclose(
            col_sr, 1.0 + np.abs(S_sr[0, 0]) ** 2, rtol=1e-12)
        with enable_x64():
            S_solved = np.asarray(msl_solve_s_from_waves(wave_a, wave_b)[0])
        np.testing.assert_allclose(
            np.sum(np.abs(S_solved[:, 0]) ** 2, axis=0), 1.0, rtol=1e-10)


def test_production_f32_path_recovers_the_planted_s_to_f32_epsilon():
    """What actually ships runs in complex64; pin that it is good enough."""
    S = _planted_s()
    S_out = np.asarray(msl_solve_s_from_waves(*_waves_from(S, 0.2))[0])
    assert S_out.dtype == np.complex64, (
        f"expected the default f32 path, got {S_out.dtype}"
    )
    np.testing.assert_allclose(S_out, S, rtol=F32_TOL, atol=F32_TOL)


def test_cond_a_reports_a_degenerate_drive_system():
    """gamma -> 1 makes both drives identical; cond(A) must blow up.

    cond_a bounds DEGENERACY only, not accuracy — same contract as the coax
    lane's solve_two_port_from_wave_amplitudes (#489).
    """
    S = _planted_s()
    with enable_x64():
        _, cond_ok = msl_solve_s_from_waves(*_waves_from(S, 0.2))
        _, cond_bad = msl_solve_s_from_waves(*_waves_from(S, 0.999))
    assert float(np.max(cond_ok)) < 10.0
    assert float(np.max(cond_bad)) > 100.0


def test_solve_handles_a_one_port_system():
    """n_ports=1 degenerates to b/a and must not raise."""
    n_f = N_FREQS
    a = [[np.ones(n_f, dtype=np.complex128)]]
    b = [[0.3 * np.ones(n_f, dtype=np.complex128)]]
    with enable_x64():
        S_out, _ = msl_solve_s_from_waves(a, b)
        S_out = np.asarray(S_out)
    assert S_out.shape == (1, 1, n_f)
    np.testing.assert_allclose(S_out[0, 0], 0.3, rtol=F64_TOL)


# --------------------------------------------------------------------------
# F2 (PR #516 review) — the WIRING: the assembly must anchor the V span on
# the RASTERIZED trace node, not on round(h_sub/dx)
# --------------------------------------------------------------------------
#
# The helper tests above cannot catch a call-site defect: msl_modal_voltage
# sums whatever span it is given. These two tests drive the REAL
# compute_msl_s_matrix setup (real grid, real PEC rasterization, real trace
# search) with synthetic z-profiled Ez planes and read back raw_v, pinning
# which Ez edges the production V actually contains on both mesh classes:
#
#   dx = 84.67 um (h_sub/dx = 2.9999, aligned):   trace node 3 -> edges 0..2
#   dx = 80.00 um (h_sub/dx = 3.1750, bisecting): trace node 4 -> edges 0..3
#
# Together they kill both known bad rules: the pre-#511 inclusive span
# (includes the trace-node edge -> fails the ALIGNED case) and the
# round(h_sub/dx) proxy (drops edge 3 -> fails the BISECTING case).

from types import MethodType, SimpleNamespace

import jax.numpy as jnp

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box
from rfx.probes.probes import DFTPlaneProbe

_EPS_R, _H_SUB, _W_TRACE = 3.66, 254e-6, 600e-6
_L_LINE, _MARGIN, _F_MAX = 10e-3, 2e-3, 5e9
# Loud, distinct markers per z-node: edges 0..2 quiet, edge 3 loud positive,
# the node-4 edge loud negative. Inclusion/exclusion is unambiguous.
_MARKER = {0: 1.0, 1: 1.0, 2: 1.0, 3: 10.0, 4: -1000.0}


def _fake_run_z_profile(nz_markers):
    """sim.run stub: every ez plane carries a fixed per-z-node profile."""

    def fake_run(self, *, n_steps=None, num_periods=1.0,
                 compute_s_params=False):
        del n_steps, num_periods, compute_s_params
        grid = self._build_grid()
        planes = {}
        for entry in self._dft_planes:
            acc = jnp.zeros((1, grid.ny, grid.nz), dtype=jnp.complex64)
            if entry.component == "ez":
                prof = jnp.asarray(
                    [complex(nz_markers.get(k, 0.0)) for k in range(grid.nz)],
                    dtype=jnp.complex64,
                )
                acc = jnp.broadcast_to(
                    prof[None, None, :], (1, grid.ny, grid.nz)
                )
            elif entry.component == "hy":
                acc = acc + (0.018 + 0.004j)
            else:  # hz
                acc = acc + (0.005 + 0.001j)
            planes[entry.name] = DFTPlaneProbe(
                accumulator=acc, freqs=entry.freqs,
                component=entry.component, axis=0, index=0,
                total_steps=1, window="rect", window_alpha=0.25,
            )
        return SimpleNamespace(dft_planes=planes)

    return fake_run


def _raw_v_for_dx(dx, tmp_path):
    lx = _L_LINE + 2 * _MARGIN
    ly = _W_TRACE + 2 * (2 * _H_SUB + 8 * dx)
    lz = _H_SUB + 1.5e-3
    sim = Simulation(
        freq_max=_F_MAX, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB + dx)), material="pec")
    for x, d in ((_MARGIN, "+x"), (_MARGIN + _L_LINE, "-x")):
        sim.add_msl_port(position=(x, y_c, 0.0), width=_W_TRACE,
                         height=_H_SUB, direction=d, impedance=50.0)
    sim.run = MethodType(_fake_run_z_profile(_MARKER), sim)
    dump = str(tmp_path / f"wiring_{int(dx*1e9)}.npz")
    with pytest.warns(UserWarning):
        sim.compute_msl_s_matrix(
            n_steps=1, freqs=jnp.asarray([1.0e9], dtype=jnp.float32),
            num_periods=1.0, enforce_passivity=False,
            raw_3probe_dump_path=dump,
        )
    d = np.load(dump, allow_pickle=True)
    return complex(np.asarray(d["raw_v"])[0, 0, 0, 0]), dx


def test_v_span_on_aligned_mesh_stops_below_the_trace_node(tmp_path):
    """dx = h_sub/3: trace at node 3 -> V = edges 0..2 exactly.

    Fails if the pre-#511 inclusive span ever returns (it would add the
    -1000-weighted node... at this mesh the trace node is 3, so the
    inclusive span adds marker[3] = +10).
    """
    v, dx = _raw_v_for_dx(84.67e-6, tmp_path)
    expected = (1.0 + 1.0 + 1.0) * dx
    np.testing.assert_allclose(v.real, expected, rtol=1e-5)
    assert abs(v.imag) < 1e-12


def test_v_span_on_bisecting_mesh_reaches_the_rasterized_trace(tmp_path):
    """dx = 80 um: trace rasterizes at node 4 -> V = edges 0..3.

    This is the F2 case. The round(h_sub/dx) proxy gives k_hi = 3 and
    silently drops edge 3; the correct anchor is the PEC-search trace node.
    A wrong span here is off by the loud marker[3] = 10 x dz, not a
    tolerance-level slip.
    """
    v, dx = _raw_v_for_dx(80e-6, tmp_path)
    expected = (1.0 + 1.0 + 1.0 + 10.0) * dx
    np.testing.assert_allclose(v.real, expected, rtol=1e-5)
    assert abs(v.imag) < 1e-12


# --------------------------------------------------------------------------
# #523 — which assembly produced S must be RECORDED, not just warned about
# --------------------------------------------------------------------------
#
# The fallback is invisible in the numbers: with the default
# enforce_passivity=True the projection clips exactly the >1 column power that
# the fallback warning tells the user to expect. So a caller who reads only S
# cannot tell a solved result from a fallback one. These tests pin the marker,
# and — the gap the post-merge sweep found — give the SOLVE branch its first
# fast end-to-end coverage: the pre-existing wiring fixture feeds identical
# planes to every drive, so A is exactly singular and only the fallback ran.


def _fake_run_drive_dependent(nz_markers, scale_by_run):
    """sim.run stub whose ez planes DIFFER per driven run.

    Distinct drives make A non-singular, so the multi-drive solve is the
    branch under test. ``scale_by_run[i]`` scales run i's ez plane.
    """
    import re
    name_re = re.compile(r"_msl_run(?P<run>\d+)_p(?P<port>\d+)_(?P<kind>ez\d+|hy|hz)")

    def fake_run(self, *, n_steps=None, num_periods=1.0, compute_s_params=False):
        del n_steps, num_periods, compute_s_params
        grid = self._build_grid()
        planes = {}
        for entry in self._dft_planes:
            m = name_re.fullmatch(entry.name)
            run_i = int(m.group("run")) if m else 0
            port_i = int(m.group("port")) if m else 0
            acc = jnp.zeros((1, grid.ny, grid.nz), dtype=jnp.complex64)
            if entry.component == "ez":
                prof = jnp.asarray(
                    [complex(nz_markers.get(k, 0.0)) for k in range(grid.nz)],
                    dtype=jnp.complex64,
                )
                # per-(run, port) amplitude => independent drive columns
                amp = scale_by_run[run_i] * (1.0 + 0.37 * port_i)
                acc = jnp.broadcast_to(
                    (prof * amp)[None, None, :], (1, grid.ny, grid.nz))
            elif entry.component == "hy":
                acc = acc + (0.018 + 0.004j) * scale_by_run[run_i]
            else:
                acc = acc + (0.005 + 0.001j) * scale_by_run[run_i]
            planes[entry.name] = DFTPlaneProbe(
                accumulator=acc, freqs=entry.freqs,
                component=entry.component, axis=0, index=0,
                total_steps=1, window="rect", window_alpha=0.25,
            )
        return SimpleNamespace(dft_planes=planes)

    return fake_run


def _run_with(fake_run, tmp_path, tag, **kw):
    dx = 84.67e-6
    lx = _L_LINE + 2 * _MARGIN
    ly = _W_TRACE + 2 * (2 * _H_SUB + 8 * dx)
    lz = _H_SUB + 1.5e-3
    sim = Simulation(
        freq_max=_F_MAX, domain=(lx, ly, lz), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB + dx)), material="pec")
    for x, d in ((_MARGIN, "+x"), (_MARGIN + _L_LINE, "-x")):
        sim.add_msl_port(position=(x, y_c, 0.0), width=_W_TRACE,
                         height=_H_SUB, direction=d, impedance=50.0)
    sim.run = MethodType(fake_run, sim)
    dump = str(tmp_path / f"{tag}.npz")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_msl_s_matrix(
            n_steps=1, freqs=jnp.asarray([1.0e9], dtype=jnp.float32),
            num_periods=1.0, raw_3probe_dump_path=dump, **kw,
        )
    return res, dump


def test_solve_branch_is_recorded_and_is_the_default(tmp_path):
    """Drive-dependent planes -> A non-singular -> the SOLVE runs.

    First fast end-to-end coverage of the solve branch inside
    compute_msl_s_matrix (the older wiring fixture feeds identical planes to
    every drive, so A is exactly singular and only the fallback ever ran).
    """
    res, dump = _run_with(
        _fake_run_drive_dependent(_MARKER, {0: 1.0, 1: 0.41 - 0.23j}),
        tmp_path, "solved", enforce_passivity=False)
    assert res.assembly == "multi_drive_solve", (
        f"expected the solve branch, got {res.assembly!r}"
    )
    assert res.cond_a is not None and np.all(np.isfinite(res.cond_a))
    meta = json.loads(str(np.load(dump, allow_pickle=True)["metadata_json"]))
    assert meta["production_smatrix_assembly"] == "multi_drive_solve"
    assert meta["schema_version"] >= 3


def test_fallback_is_recorded_even_though_the_numbers_hide_it(tmp_path):
    """Identical planes per drive -> A singular -> fallback, and it SHOWS.

    The mutation twin for #523: with the default passivity projection the
    fallback's own symptom (column power > 1) is clipped away, so this test
    asserts the MARKER, not the numbers — and checks that the numbers really
    do look innocent, which is why the marker has to exist.
    """
    res, dump = _run_with(_fake_run_z_profile(_MARKER), tmp_path, "fell_back")
    assert res.assembly == "single_ratio_fallback", (
        f"expected the fallback, got {res.assembly!r}"
    )
    meta = json.loads(str(np.load(dump, allow_pickle=True)["metadata_json"]))
    assert meta["production_smatrix_assembly"] == "single_ratio_fallback"

    # ... and the reason a marker is needed: after the default projection the
    # returned S carries no >1 column power to betray the fallback.
    col = np.abs(np.asarray(res.S)[:, 0, :]) ** 2
    assert float(np.sum(col, axis=0).max()) <= 1.0 + 1e-5, (
        "if this ever fails the projection stopped hiding the symptom — "
        "good news, but this test's premise needs rewriting"
    )


def test_assembly_marker_is_none_under_tracing(tmp_path):
    """Under a tracer the finiteness test cannot run, so no claim is made.

    Mirrors `reliable`'s documented None-while-tracing contract rather than
    asserting a rule that was never checked.
    """
    from rfx.api._sparams import msl_solve_s_from_waves
    n_f = 2
    a = [[jnp.ones(n_f, dtype=jnp.complex64) * (1.0 + 0.2 * d + 0.1 * j)
          for j in range(2)] for d in range(2)]
    b = [[jnp.ones(n_f, dtype=jnp.complex64) * (0.3 - 0.05 * d + 0.02 * j)
          for j in range(2)] for d in range(2)]

    def f(scale):
        aa = [[x * scale for x in row] for row in a]
        _, cond = msl_solve_s_from_waves(aa, b)
        # cond_a is exactly what gates the assembly marker upstream
        assert cond is None, "cond_a must be None under tracing"
        return jnp.array(0.0)

    jax.grad(f)(jnp.asarray(1.0, dtype=jnp.float32))
