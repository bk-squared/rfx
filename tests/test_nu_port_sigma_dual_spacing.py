"""#688 — the NU port termination sigma must use the DUAL transverse widths.

Defect (external review against 4eb7fa4). Both impedance-port branches of
``rfx/runners/nonuniform.py`` sized the termination conductance with the
PRIMAL cell widths on the two axes transverse to the port component:

    sigma = n_live * d_parallel / (Z0 * d_perp1 * d_perp2)

while the current that measures the port, ``wire_port_current``, weights
its Ampere legs by the DUAL spacings (#672, merged one commit earlier).
Sizing the load on one metric family and measuring it with the other means
the realized Z is not Z0 wherever the two differ — i.e. anywhere the
transverse axes are graded. d_parallel is correctly primal: it is the
length of the E edge V is taken over.

ORACLE 1 — STAMPED-SIGMA CONDUCTANCE IDENTITY (static arithmetic on the
stamped array; does not route through the sizing expression).
Multiplying the NU E update ``eps dE_a/dt + sigma E_a = (curl H)_a`` by the
dual face area ``dual_b*dual_c`` gives ``I_cond = sigma*E_a*dual_b*dual_c``
against ``V = E_a*d_a_primal``, hence ``R_cell = d_a_primal /
(sigma*dual_b*dual_c)``. For an n_live-cell series port of total Z0:

    G_realized = sigma * dual_b * dual_c / d_a_primal  ==  n_live / Z0

The dual spacings are recomputed LONGHAND here off the cell-size profiles,
NOT through ``e_node_dual_spacing_at`` — that helper is what the fix
depends on, so using it would make the oracle circular.

Fixture: 8 fine | 8 coarse | 8 fine cells of 0.5 mm / 1.0 mm on ALL THREE
axes, port on the first coarse cell, Z0 = 50. The node's dual spacing is
0.75 mm against a 1.0 mm primal, so each transverse axis contributes 0.75
and the conductance error is 0.75^2 = 0.5625 (i.e. the realized resistance
was 1.7778x too high).

    component  branch               4eb7fa4 sigma  ratio     fixed sigma  ratio
    ex/ey/ez   single-cell lumped        20.0000  0.562500      35.5556  1.000000
    ex/ey/ez   wire, n_live = 3          60.0000  0.562500     106.6667  1.000000

ORACLE 2 — END-TO-END QUASI-STATIC S11 (a time-stepped field measurement
through the #672 Ampere-loop extractor; shares no suspect quantity with
oracle 1). A correctly terminated PASSIVE n_live-cell port reads

    S11 -> (1 - n_live) / (1 + n_live)

in the quasi-static bins. That form is independent of Z0, so it cannot be
satisfied by rescaling Z0 — only by the port cell realizing Z0/n_live.
Measured on the doubly-graded fixture, Re S11 at 0.2 / 0.4 / 0.6 GHz:

    component  n_live   expected   4eb7fa4     fixed
    ez              2   -0.33333   -0.05882   -0.33333
    ez              4   -0.60000   -0.38462   -0.60000
    ex              2   -0.33333   -0.05889   -0.33343
    ey              2   -0.33333   -0.05882   -0.33333

The 4eb7fa4 column is exactly what a 1.7778x-too-high cell resistance
predicts: Z_in = 1.7778*Z0/2 = 44.44 gives (44.44-50)/(44.44+50) = -0.0588.

PASSIVE port + separate ``add_source`` deliberately: a passive port's I
reads H only while the source loop writes E only, which keeps this clear of
the still-open #683 source-injection ordering question. ``n_live`` is
MEASURED from the stamped array, never derived from ``extent`` — the port
sits in the 2x coarse region, so ``extent/dx`` overcounts.

Single-cell lumped S-parameters are refused on the NU lane (preflight
``_validate_run_sparameter_request``), so oracle 2 reaches only the wire
branch; oracle 1 covers both.

Preflight on the oracle-2 fixture, verbatim (one line, and it is about the
extent rasterization, not about this fix):
``[PREFLIGHT] Wire port at (0.004, 0.004, 0.004) (extent 0.001): dead-cell
classification unavailable (non-uniform mesh (dz_profile/dx_profile/
dy_profile set) -- the shared ground-truth primitive only covers the
uniform-grid path (issue #544)). ...``
"""

import warnings

import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
import rfx.runners.nonuniform as _rn

Z0 = 50.0
D = 0.5e-3
FREQS = np.array([0.2e9, 0.4e9, 0.6e9])
AXIS_OF = {"ex": 0, "ey": 1, "ez": 2}

# The blind guard below is derived FROM this fixture: a 2:1 step on both
# transverse axes gives dual/primal = 0.75 per axis, so the pre-fix
# conductance is 0.5625 of the correct one — 43.75% low. If the fixture is
# ever softened (CFL, runtime), re-derive the guard from the new profile
# rather than carrying 0.30 over.
BLIND_GUARD = 0.30


def _graded(nfine=8, ncoarse=8):
    """fine | coarse | fine, symmetric so profile[0] == profile[-1] == D
    (``make_nonuniform_grid`` requires the boundary cell size on both ends)."""
    return np.array([D] * nfine + [2 * D] * ncoarse + [D] * nfine)


def _dual_np(prof):
    """Longhand dual spacing — NOT ``e_node_dual_spacing_at``."""
    p = np.asarray(prof, dtype=np.float64)
    return np.concatenate([p[:1], 0.5 * (p[:-1] + p[1:])])


def _preflight_text(sim):
    """Preflight advisories as text. They are printed, not re-raised, so
    stdout is the surface (same as tests/test_nonuniform_source_port_dual_
    spacing.py's ``_preflight`` helper)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.preflight()
    return buf.getvalue()


def _sim(component, extent=None, pos_mult=9.0):
    g = _graded()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=20e9, dx=D, domain=(float(g.sum()),) * 3,
                         dx_profile=g, dy_profile=g, dz_profile=g,
                         boundary="pec")
        kw = dict(impedance=Z0, component=component, excite=False)
        if extent is not None:
            kw["extent"] = extent
        sim.add_port((pos_mult * D,) * 3, **kw)
        sim.add_source((3.5 * D,) * 3, "ex", waveform=GaussianPulse(f0=5e9),
                       amplitude_kind="field")
        sim.add_probe((5.5 * D,) * 3, "ex")
    return sim


def _capture(sim):
    """Grab (grid, materials) just before the stepper runs.

    Same monkeypatch-spy pattern tests/test_nonuniform_source_port_dual_spacing
    uses on ``wire_port_current``. The ``assert seen`` guard makes a fixture
    that stops exercising the path fail loudly instead of vacuously passing.
    """
    seen = {}
    orig = _rn.run_nonuniform

    def spy(grid, materials, *a, **kw):
        seen["grid"] = grid
        seen["materials"] = materials
        raise SystemExit

    _rn.run_nonuniform = spy
    try:
        sim.run(n_steps=4, skip_preflight=True)
    except SystemExit:
        pass
    finally:
        _rn.run_nonuniform = orig
    assert seen, "spy never fired — the fixture stopped exercising the NU port path"
    return seen["grid"], seen["materials"]


def _realized_conductance(component, extent):
    grid, mats = _capture(_sim(component, extent))
    sig = np.asarray(mats.sigma)
    cells = np.argwhere(sig > 0)
    assert len(cells), (component, extent, "no port sigma was stamped")
    prof = [np.asarray(grid.dx_arr, dtype=np.float64),
            np.asarray(grid.dy_arr, dtype=np.float64),
            np.asarray(grid.dz, dtype=np.float64)]
    dual = [_dual_np(p) for p in prof]
    a = AXIS_OF[component]
    b, c = [x for x in range(3) if x != a]
    i, j, k = cells[0]
    idx = (i, j, k)
    g = (float(sig[idx]) * dual[b][idx[b]] * dual[c][idx[c]]
         / prof[a][idx[a]])
    return g, len(cells), float(sig[idx]), dual, prof, idx, a, b, c


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("extent,label", [(None, "lumped"), (4 * D, "wire")])
def test_stamped_sigma_realizes_z0_on_a_doubly_graded_node(component, extent,
                                                           label):
    """ORACLE 1 — G_realized == n_live / Z0."""
    g, n_live, _sig, *_ = _realized_conductance(component, extent)
    expect = n_live / Z0
    assert g == pytest.approx(expect, rel=1e-6), (component, label, g, expect)


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("extent,label", [(None, "lumped"), (4 * D, "wire")])
def test_the_primal_spelling_is_blind_on_this_fixture(component, extent,
                                                      label):
    """Blind guard — the fixture must still DISCRIMINATE.

    Without this, the day someone flattens the grading the oracle above
    passes on the broken code too. Measured error 43.75%.
    """
    g, n_live, _sig, dual, prof, idx, a, b, c = _realized_conductance(
        component, extent)
    g_primal = g * (dual[b][idx[b]] * dual[c][idx[c]]) / (
        prof[b][idx[b]] * prof[c][idx[c]])
    rel = abs(g_primal - n_live / Z0) / (n_live / Z0)
    assert rel > BLIND_GUARD, (component, label, rel)


def test_uniform_transverse_mesh_is_bit_identical_to_the_primal_spelling():
    """The fix cannot be a silent global renormalization.

    ``0.5*(d + d) == d`` exactly in IEEE, so on a transversely uniform
    profile the stamped sigma must equal the pre-#688 primal value BIT for
    bit — not merely within a tolerance.
    """
    flat = np.full(24, D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=20e9, dx=D, domain=(float(flat.sum()),) * 3,
                         dx_profile=flat, dy_profile=flat, dz_profile=flat,
                         boundary="pec")
        sim.add_port((9 * D,) * 3, impedance=Z0, component="ez", excite=False)
        sim.add_source((3.5 * D,) * 3, "ex", waveform=GaussianPulse(f0=5e9),
                       amplitude_kind="field")
        sim.add_probe((5.5 * D,) * 3, "ex")
    grid, mats = _capture(sim)
    sig = np.asarray(mats.sigma)
    i, j, k = np.argwhere(sig > 0)[0]
    dz = np.asarray(grid.dz, dtype=np.float64)
    dx = np.asarray(grid.dx_arr, dtype=np.float64)
    dy = np.asarray(grid.dy_arr, dtype=np.float64)
    primal = float(dz[k]) / (Z0 * float(dx[i]) * float(dy[j]))
    assert float(sig[i, j, k]) == np.float32(primal), (
        float(sig[i, j, k]), primal)


def _s11(component, extent):
    sim = _sim(component, extent)
    _grid, mats = _capture(_sim(component, extent))
    n_live = int((np.asarray(mats.sigma) > 0).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.run(n_steps=1200, compute_s_params=True,
                      s_param_freqs=FREQS, skip_preflight=True)
    return np.real(np.asarray(res.s_params)[0, 0, :]), n_live


@pytest.mark.parametrize("component,extent", [("ez", 2 * D), ("ez", 6 * D),
                                              ("ex", 2 * D), ("ey", 2 * D)])
def test_passive_port_quasi_static_s11_matches_the_closed_form(component,
                                                               extent):
    """ORACLE 2 — S11 -> (1 - n_live)/(1 + n_live), independent of Z0."""
    s11, n_live = _s11(component, extent)
    expect = (1.0 - n_live) / (1.0 + n_live)
    assert n_live >= 2, n_live
    np.testing.assert_allclose(s11, expect, rtol=2e-3,
                               err_msg=f"{component} extent={extent} "
                                       f"n_live={n_live} S11={s11}")


def test_preflight_flags_a_lumped_port_on_a_graded_node():
    """#688 widened ``_validate_cfg_wire_port_on_graded_node`` to cover
    single-cell lumped ports: they carry no ``extent``, so the old filter
    skipped them while they sat on the identical metric."""
    # pos 8D is a genuine grading node; pos 9D lands in a rounding band
    # where the advisory's node lookup and the runner's stamped cell index
    # disagree by one (see the module note in the PR body).
    sim = _sim("ez", extent=None, pos_mult=8.0)
    fired = _preflight_text(sim)
    assert "an E node whose" in fired, fired
    assert "lumped port at" in fired, fired
    assert "issue #688" in fired, fired


def test_preflight_is_silent_for_a_lumped_port_on_a_uniform_node():
    """Negative control — the widened filter must not warn everywhere."""
    flat = np.full(24, D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=20e9, dx=D, domain=(float(flat.sum()),) * 3,
                         dx_profile=flat, dy_profile=flat, dz_profile=flat,
                         boundary="pec")
        sim.add_port((8 * D,) * 3, impedance=Z0, component="ez", excite=False)
        sim.add_source((3.5 * D,) * 3, "ex", waveform=GaussianPulse(f0=5e9),
                       amplitude_kind="field")
        sim.add_probe((5.5 * D,) * 3, "ex")
    assert "an E node whose" not in _preflight_text(sim), _preflight_text(sim)
