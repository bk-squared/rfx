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

THE FIXTURE IS ANISOTROPIC ON PURPOSE, and that is the second thing this
module gates. Its first version graded all three axes with the SAME profile
and put the port at the same multiple of D on each, so
``dual_x == dual_y == dual_z`` and ``primal_x == primal_y == primal_z`` at
the port node — and every permutation of the transverse-axis assignment
passed. A reviewer mutated both branches to read the transverse duals off
the WRONG axes and all six parameters stayed green. A conductance oracle
cannot see an axis swap unless the axes carry different numbers, so this
fixture grades x 2:1, y 3:1 and z 4:1, with the fine run a different length
on each so the port lands on a different NODE INDEX per axis:

    axis  profile               port cell  dual/D  primal/D
    x     6 x D | 6 x 2D | 6 x D    6        1.5      2.0
    y     8 x D | 6 x 3D | 8 x D    8        2.0      3.0
    z    10 x D | 6 x 4D | 10 x D  10        2.5      4.0

All six numbers differ, so no permutation of the three axes reproduces any
other's product. ``test_the_fixture_axes_are_pairwise_distinct`` asserts
that directly, so flattening the fixture reds instead of silently
un-gating the module.

ORACLE 1 — STAMPED-SIGMA CONDUCTANCE IDENTITY (static arithmetic on the
stamped array; does not route through the sizing expression).
Multiplying the NU E update ``eps dE_a/dt + sigma E_a = (curl H)_a`` by the
dual face area ``dual_b*dual_c`` gives ``I_cond = sigma*E_a*dual_b*dual_c``
against ``V = E_a*d_a_primal``, hence ``R_cell = d_a_primal /
(sigma*dual_b*dual_c)``. For an n_live-cell series port of total Z0:

    G_realized = sigma * dual_b * dual_c / d_a_primal  ==  n_live / Z0

The dual spacings are recomputed LONGHAND here off the cell-size profiles,
NOT through ``e_node_dual_spacing_at`` — that helper is what the fix
depends on, so using it would make the oracle circular. Every stamped cell
is checked, not just the first.

    component  branch               G*Z0/n_live
    ex/ey/ez   single-cell lumped     1.000000
    ex/ey/ez   wire                   1.000000  (every stamped cell)

ORACLE 2 — END-TO-END QUASI-STATIC S11 (a time-stepped field measurement
through the #672 Ampere-loop extractor; shares no suspect quantity with
oracle 1). A correctly terminated PASSIVE n_live-cell port reads

    S11 -> (1 - n_live) / (1 + n_live)

in the quasi-static bins. That form is independent of Z0, so it cannot be
satisfied by rescaling Z0 — only by the port cell realizing Z0/n_live.
Measured on THIS fixture, Re S11 at 0.2 / 0.4 / 0.6 GHz, worst relative
deviation over the three bins:

    component  extent  n_live   expected   measured             worst rel
    ez            2D      2     -0.33333   -0.33334 ... -0.33333  2.5e-05
    ez            6D      3     -0.50000   -0.50000 ... -0.50000  6.9e-06
    ex            2D      2     -0.33333   -0.33336 ... -0.33334  7.8e-05
    ey            2D      2     -0.33333   -0.33333 ... -0.33333  1.9e-05

For reference, the pre-fix code on the ORIGINAL isotropic fixture read
-0.05882 against the same -0.33333 expectation: a 1.7778x-too-high cell
resistance gives Z_in = 44.44 and (44.44-50)/(44.44+50) = -0.0588.

PASSIVE port + separate ``add_source`` deliberately: a passive port's I
reads H only while the source loop writes E only, which keeps this clear of
the still-open #683 source-injection ordering question. ``n_live`` is
MEASURED from the stamped array, never derived from ``extent`` — the port
sits in a coarse region, so ``extent/dx`` overcounts.

Single-cell lumped S-parameters are refused on the NU lane (preflight
``_validate_run_sparameter_request``), so oracle 2 reaches only the wire
branch. Oracle 1 covers both, which is the only gate the lumped branch's
axis assignment has.

MUTATION RECORD. Both branches of ``run_nonuniform_path`` mutated to read
the two transverse duals off the wrong axes (the assignment rotated by one
axis) — the mutant that survived the isotropic fixture on all six oracle-1
parameters. Measured G*Z0/n_live here, per stamped cell:

    component  lumped      wire
    ex         1.333333    1.333333, 1.000000, 1.000000
    ey         1.250000    1.250000, 0.833333, 0.833333
    ez         0.600000    0.600000, 0.375000

All six oracle-1 parameters red. Note the wire rows: only the FIRST cell is
wrong for ex, because that mutant reads ``dual_x``, which happens to equal
``dual_y`` two cells further into the coarse run. Checking cells[0] alone
would have held here by luck; ``_conductances`` checks every stamped cell
so it does not have to be luck.

Oracle 2 reds on 3 of its 4 parameters under the same mutant
(ez/2D: +0.14286 against -0.33333; ez/6D: -0.05882 against -0.50000;
ey/2D: -0.25000 against -0.33333). The ex leg does NOT move (-0.33336,
within the 2e-3 gate) — its series combination of one wrong and two correct
cells happens to land back on the closed form. Oracle 1 is what catches
that one, which is the reason it had to be made to bind.

Preflight on this fixture, verbatim — two lines, one per Ampere-loop axis
(z is the port's parallel axis for an ez port, so it is correctly absent):
``  [PREFLIGHT] lumped port at (0.0035, 0.00475, 0.006000000000000001)
(component ez) sits at x = 3mm, an E node whose adjacent cells differ by
100% (500µm below, 1mm above). x is one of the port's two Ampere-loop axes,
so the DUAL spacing 750µm weights BOTH that leg of the loop that measures I
(issue #672) and the termination conductance that realizes Z0 (issue #688).
The extracted Z_in = -V/I, and every S-parameter built on it, is most
mesh-sensitive here: move the port onto a locally uniform node (or flatten
the grading there) if its S-parameters are claims-bearing.``
and the same sentence for ``y = 4mm``, ``200% (500µm below, 1.5mm above)``,
``DUAL spacing 1mm``.
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


def _prof(ratio, nfine, ncoarse=6):
    """fine | coarse | fine, symmetric so profile[0] == profile[-1] == D
    (``make_nonuniform_grid`` requires the boundary cell size on both ends).
    The port node is the FIRST coarse cell, where dual = 0.5*(1+ratio)*D
    against a primal of ratio*D — so a different ``ratio`` per axis gives a
    different dual AND a different primal, and a different ``nfine`` gives a
    different node index."""
    return np.array([D] * nfine + [ratio * D] * ncoarse + [D] * nfine)


GX, GY, GZ = _prof(2, 6), _prof(3, 8), _prof(4, 10)
PORT_CELL = (6, 8, 10)
DOMAIN = (float(GX.sum()), float(GY.sum()), float(GZ.sum()))


def _cell_centre(prof, i):
    return float(np.sum(prof[:i]) + 0.5 * prof[i])


PORT_POS = tuple(_cell_centre(p, i)
                 for p, i in zip((GX, GY, GZ), PORT_CELL))

# The blind guard below is derived FROM this fixture. Per-axis dual/primal
# is 0.75 (x), 0.667 (y), 0.625 (z), so the primal spelling of the two
# transverse widths is low by 0.5 (ez), 0.4167 (ex) or 0.4688 (ey) — a
# relative conductance error of 50% at worst-case-best. If the fixture is
# ever softened (CFL, runtime), re-derive this from the new profiles rather
# than carrying 0.40 over.
BLIND_GUARD = 0.40


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


def _sim(component, extent=None, pos=PORT_POS):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=20e9, dx=D, domain=DOMAIN,
                         dx_profile=GX, dy_profile=GY, dz_profile=GZ,
                         boundary="pec")
        kw = dict(impedance=Z0, component=component, excite=False)
        if extent is not None:
            kw["extent"] = extent
        sim.add_port(pos, **kw)
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


def _stamped(component, extent):
    """Everything the oracles need, read back off the stamped arrays."""
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
    return dict(sig=sig, cells=[tuple(int(v) for v in x) for x in cells],
                prof=prof, dual=dual, a=a, b=b, c=c, n_live=len(cells))


def _conductances(st):
    """G_realized per stamped cell — every cell, not just the first."""
    return [float(st["sig"][idx]) * st["dual"][st["b"]][idx[st["b"]]]
            * st["dual"][st["c"]][idx[st["c"]]] / st["prof"][st["a"]][idx[st["a"]]]
            for idx in st["cells"]]


def test_the_fixture_axes_are_pairwise_distinct():
    """The anti-permutation tripwire.

    Oracle 1 can only see a transverse-axis SWAP if the three axes carry
    different metrics at the port node. The first version of this fixture
    used one profile and one position on all three axes, and a mutant that
    read the duals off the wrong axes passed all six parameters.
    """
    st = _stamped("ez", None)
    idx = st["cells"][0]
    assert idx == PORT_CELL, idx
    pairs = [(st["dual"][ax][idx[ax]] / D, st["prof"][ax][idx[ax]] / D)
             for ax in range(3)]
    np.testing.assert_allclose(pairs, [(1.5, 2.0), (2.0, 3.0), (2.5, 4.0)],
                               rtol=1e-6)
    flat = [v for p in pairs for v in p]
    assert len(set(np.round(flat, 9))) == 6, pairs


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("extent,label", [(None, "lumped"), (4 * D, "wire")])
def test_stamped_sigma_realizes_z0_on_a_doubly_graded_node(component, extent,
                                                           label):
    """ORACLE 1 — G_realized == n_live / Z0, on every stamped cell."""
    st = _stamped(component, extent)
    expect = st["n_live"] / Z0
    for idx, g in zip(st["cells"], _conductances(st)):
        assert g == pytest.approx(expect, rel=1e-6), (component, label, idx,
                                                      g, expect)


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("extent,label", [(None, "lumped"), (4 * D, "wire")])
def test_the_primal_spelling_is_blind_on_this_fixture(component, extent,
                                                      label):
    """Blind guard — the fixture must still DISCRIMINATE.

    Without this, the day someone flattens the grading the oracle above
    passes on the broken code too. Worst-case measured error 50%.
    """
    st = _stamped(component, extent)
    idx = st["cells"][0]
    b, c = st["b"], st["c"]
    g = _conductances(st)[0]
    g_primal = g * (st["dual"][b][idx[b]] * st["dual"][c][idx[c]]) / (
        st["prof"][b][idx[b]] * st["prof"][c][idx[c]])
    rel = abs(g_primal - st["n_live"] / Z0) / (st["n_live"] / Z0)
    assert rel > BLIND_GUARD, (component, label, rel)


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("extent,label", [(None, "lumped"), (4 * D, "wire")])
def test_a_transverse_axis_swap_would_be_caught(component, extent, label):
    """The mutation this module exists to catch, computed statically.

    Re-derives what the stamped sigma WOULD have been had the two transverse
    duals been read off any other pair of axes, and asserts every such
    spelling misses ``n_live/Z0``. This is the fixture's discriminating
    power stated as an assertion rather than left to a hand-run mutant.
    """
    st = _stamped(component, extent)
    idx = st["cells"][0]
    a, b, c = st["a"], st["b"], st["c"]
    right = st["dual"][b][idx[b]] * st["dual"][c][idx[c]]
    for pb in range(3):
        for pc in range(pb + 1, 3):
            if (pb, pc) == (min(b, c), max(b, c)):
                continue
            wrong = st["dual"][pb][idx[pb]] * st["dual"][pc][idx[pc]]
            # sigma scales as 1/(dp1*dp2), so G scales as right/wrong
            ratio = right / wrong
            assert abs(ratio - 1.0) > 0.1, (component, label, a, pb, pc,
                                            ratio)


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
    n_live = _stamped(component, extent)["n_live"]
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
    skipped them while they sat on the identical metric.

    One line per AMPERE-LOOP axis. For an ez port those are x and y; z is
    the parallel axis and must NOT be reported, which also checks the
    advisory is axis-aware rather than firing on any grading.
    """
    fired = _preflight_text(_sim("ez", extent=None))
    assert "an E node whose" in fired, fired
    assert "lumped port at" in fired, fired
    assert "issue #688" in fired, fired
    assert "sits at x = " in fired, fired
    assert "sits at y = " in fired, fired
    assert "sits at z = " not in fired, fired


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
