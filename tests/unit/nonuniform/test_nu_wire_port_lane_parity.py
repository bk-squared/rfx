"""Non-uniform vs uniform wire-port S-parameter lane parity (issue #673).

Covers ONE defect, D1. A second candidate, D2, was pulled from this branch
before merge and is now issue #683 — see "The D2 question" below.

D1 (FIXED HERE) — the (a, b) wave split in ``rfx/nonuniform.py`` branched on
     the port's ``direction``. The ``"-x"`` / ``"-y"`` branch is the exact
     RECIPROCAL of the ``"+x"`` / ``"+y"`` one, so every port in the low-x
     half of a domain (and every centred port, which hits Python's dict
     tie-break in ``_auto_direction``) reported ``1/S11``. ``direction`` is
     not a degree of freedom for a lumped gap V/I pair: V and I are taken
     about the SAME axis (V along the E edge, I from the Ampere loop
     encircling that same edge), so there is no second axis to reverse and
     no outward normal in either expression. See the derivation in the block
     comment at the extraction site.

MEASURED, at 13de212 (before) and on this branch (after), all with
``.venv/bin/python -P``, ``rfx.__file__`` asserted inside the checkout,
``JAX_PLATFORMS=cpu``, float32 default precision:

  fixture                            uniform S11(0.2 GHz)   NU S11(0.2 GHz)
  PEC-plate gap, passive, n_live=6   -0.71429+0.00027j      BEFORE -1.40000-0.00053j
                                                            AFTER  -0.71429+0.00027j
  vacuum, passive, n_live=4          -0.60000+0.00034j      BEFORE -1.66667-0.00094j
                                                            AFTER  -0.60000+0.00034j
  max|S11| on the passive vacuum fixture: BEFORE 1.66667, AFTER 0.60000.

That block is HISTORY for the #673 sign fix and is quoted at the
rasterization of its day. Its ``n_live`` labels (6, 4) and the values that
follow from them are the endpoint-inclusive counts retired by #931 R8; the
current numbers are the re-measured table further down. The BEFORE/AFTER
ratio the block exists to show — a reciprocal, 1/S11 — is unaffected.

WHAT THESE TESTS DO AND DO NOT VALIDATE
---------------------------------------
They validate the CONVENTION — the sign and the non-reciprocity of the (a, b)
split, and that the two lanes spell it identically. They do NOT validate any
structure's physics, and the closed form below is NOT an independent physics
oracle for ``S11``.

The reason is that the passive reading is a property of the port CELL, not of
what is attached across it. From the discrete Ampere law at the port edge with
no impressed current, ``-V/I = 1/(G + jwC)`` identically, where G and C are the
port cell's own conductance and capacitance. Measured, three fixtures that
differ only in what sits at the gap — vacuum, PEC plates bracketing the gap,
and an ``eps_r = 10`` slab filling it — all return the SAME
``S11(0.2 GHz)`` at ``n_live = 3``. The load does not move the reading at
all. A convention test is exactly what that supports.

#931 NOTE ON THE THREE-LOAD WITNESS. The table above was measured before
the lattice ownership contract, when a one-cell PEC body realized a single
wall plane and left its normal E live — so the "PEC plates" arm was two
films and could not short anything even in principle. Under the contract
each plate realizes both of its drawn faces and shorts Ez between them, so
the arm is a stronger witness: genuinely conducting bodies one cell from
the port.

RE-MEASURED under the contract (VESSL run 369367259274, commit 770c4e6c,
``JAX_PLATFORMS=cpu``; producer ``docs/design_notes/931_migration/
t3_remeasure.py``, output ``/root/workspace/claude-workspace/rfx/runs/
issue931-post-t3-measure-20260907T191140Z/t3_remeasure.json``), S11 at
0.2 GHz, uniform lane and NU lane agreeing to ~1e-6 on every row:

  gap extent   load          n_live   S11(0.2 GHz)
  5 mm         vacuum        5        -0.6666666 - 3.131e-05j
  5 mm         PEC plates    5        -0.6666666 - 3.075e-05j
  3 mm         vacuum        3        -0.4999999 - 6.952e-05j
  3 mm         eps_r=10 slab 3        -0.5000007 - 6.955e-04j

The reading is set by ``n_live`` and by nothing else: ``(1-n)/(1+n)`` is
-4/6 = -0.6666667 at n = 5 and -2/4 = -0.5 at n = 3, and the measurement
sits on those to seven digits. Shorting the plates' interior did not move
it, which is the module's whole claim. What DID move against the pre-#931
table is the small imaginary part (+2.7e-04 -> -3.1e-05 on the PEC-plate
arm): it is the residual the closed form does not carry, and it is not
gated anywhere. The dielectric arm's -7.0e-04 is the one load-dependent
digit in the table, three orders of magnitude below the real part.

WHY ``n_live`` IS 5 / 3 AND NOT 6 / 4. The counts in the pre-#931 table
above are the endpoint-INCLUSIVE rasterization that #931 R8 retired: a
5 mm extent on a 1 mm mesh drove six Ez edges, the sixth spanning a cell
ABOVE the declared end. The extent is half-open in edges now
(``rfx.sources.sources.wire_port_edge_span``), so it drives five, and the
closed form follows the count. The gates in this file are computed from
``_n_live`` rather than from a literal, so they moved with it — and
``test_the_analytic_oracle_reads_the_runners_own_span`` pins ``_n_live``
against the production rasterizer, because for one commit it did NOT move
with it: the helper held its own copy of the retired rule, and the oracle
sat one cell off the lane it was judging.

The step from the raw ratio to ``S11`` additionally runs through the
extractor's own mixed normalization (V and I are sampled at ONE cell in
``rfx/probes/probes.py:721`` / ``:746``, so the measured Z is the per-cell
``Z0/n_live`` while the reflection formula references the whole-port ``Z0``).
That is the known-wrong #313 / #318 normalization family. Only the FIRST half
— the raw ``V/I -> -Z0/n_live`` in
``test_raw_port_ratio_matches_the_analytic_admittance`` — is oracle-grade.
``test_passive_s11_regression_lock_on_the_known_wrong_normalization`` is named
for what it is: a lock on that artifact, so that a future CORRECT fix reds it
loudly instead of reading as a regression.

Derivation of the raw-ratio oracle: the E update at the port edge enforces
``eps dE/dt + sigma E = (curl H)_c``; with the runner's
``sigma = n_live * d / (Z0 * dp1 * dp2)`` this gives, in phasor form,

    r == V/I = -1 / (n_live/Z0 + j w C),

so ``Re(r) <= 0`` and ``|r| <= Z0/n_live`` — again, both properties of the port
cell — and in the quasi-static limit ``r -> -Z0/n_live``, i.e.
``S11 -> (1 - n_live) / (1 + n_live)``.

THE D2 QUESTION (issue #683 — DECIDED 2026-08-29, by measurement)
------------------------------------------------------------
This lane accumulates the wire-port V/I DFT AFTER source injection; the
uniform lane used to accumulate BEFORE.  The question was decided by the
#683 known-load protocol run (one attempt, gates G0-G2 passed;
docs/design_notes/issue683_sampling_order_decision_protocol.md section 9):
POST-injection sampling is the terminal-consistent, physically correct
order (n*a = +0.9987/+0.9950, n*|b| = 0.08/0.32 Ohm on a six-point R_L
sweep; Ampere-identity residual 2.3e-7 vs 3.25 pre).  The once-CONTESTED
known-load sweep was settled by that run (the earlier independent repro's
fixture had failed to load the port), and the once-UNEXAMINED Ampere
identity was examined in the same run.

The uniform lane then flipped its physical V/I/V_port sampling to POST —
with the pre-injection drive sample kept as the calibration reference for
its #308/#313 off-diagonal decomposition (issue #683 x #764,
docs/design_notes/issue683_decomposer_flip_predeclaration.md) — and the
excited-port disagreement this module kept visible collapsed to float
noise.  The witness is now the LOCKED parity test
``test_excited_port_lane_ordering_disagreement_is_open_683`` (name kept
for git-history greppability).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse

DX = 1e-3
DOMAIN = (16e-3, 12e-3, 14e-3)
NZ = int(round(DOMAIN[2] / DX))
PORT_Y = DOMAIN[1] / 2
PORT_Z = 5e-3
N_STEPS = 800
FREQS = jnp.array([0.2e9, 0.5e9, 1.0e9])

# Pre-declared gates (see module docstring for the measured values).
LANE_PARITY_ATOL = 1e-4      # |S_NU - S_uniform|, per bin
ANALYTIC_ATOL = 1e-3         # |S11 - (1-n)/(1+n)| in the quasi-static bins
PASSIVITY_MAX = 1.0 + 1e-3   # UNDRIVEN ports only: |r| <= Z0/n_live is a
                             # property of the port cell, so this bounds
                             # the passive reading and NOT a driven one


def _build(nu, *, extent=3e-3, excite=False, z0=50.0, port_x=8e-3,
           direction=None, load="vacuum"):
    """Identical physical grid on both lanes.

    ``nu=True`` passes a UNIFORM-VALUED ``dz_profile`` of the same spacing,
    so the two lanes discretise bit-identical geometry and any S-parameter
    difference is the extractor, not the mesh.
    """
    kw = {"dz_profile": np.full(NZ, DX)} if nu else {}
    sim = Simulation(freq_max=10e9, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=6, **kw)
    if load == "pec_plates":
        # Two one-cell PEC plates, one below the port and one above it,
        # each separated from the port edge by one cell of vacuum. They
        # are VOLUMES (#931 §1.2), drawn on node planes, so each realizes
        # walls at BOTH of its drawn faces and shorts the normal E between
        # them — the object the name "plate" always meant. Before #931 a
        # one-cell body realized ONE wall and left its normal E live, so
        # the plates were films, not plates. Neither realization shorts the
        # PORT: the vacuum cell on each side is what makes this a "load
        # near the port", and the module docstring's point is that the
        # passive reading does not move for ANY of the three loads.
        w = 2e-3
        sim.add(Box((port_x - w, PORT_Y - w, PORT_Z - 2 * DX),
                    (port_x + w, PORT_Y + w, PORT_Z - DX)), material="pec")
        sim.add(Box((port_x - w, PORT_Y - w, PORT_Z + extent + DX),
                    (port_x + w, PORT_Y + w, PORT_Z + extent + 2 * DX)),
                material="pec")
    elif load == "dielectric":
        w = 2e-3
        sim.add_material("slab", eps_r=10.0)
        sim.add(Box((port_x - w, PORT_Y - w, PORT_Z),
                    (port_x + w, PORT_Y + w, PORT_Z + extent)),
                material="slab")
    elif load != "vacuum":
        raise ValueError(load)

    pulse = GaussianPulse(f0=2e9, bandwidth=0.9)
    sim.add_port(position=(port_x, PORT_Y, PORT_Z), component="ez",
                 impedance=z0, extent=extent, excite=excite,
                 waveform=(pulse if excite else None), direction=direction)
    if not excite:
        # Passive port: illuminate it with a separate source, which keeps
        # D1 clear of the #683 ordering question entirely — a passive port's
        # V/I is untouched by the source-injection ordering, since I reads H
        # only and the source loop writes E only, at the SOURCE cell.
        sim.add_source(position=(port_x + 3 * DX, PORT_Y, PORT_Z + DX),
                       component="ez", waveform=pulse)
    return sim


def _s11(sim, n_steps=N_STEPS):
    r = sim.run(n_steps=n_steps, compute_s_params=True,
                s_param_freqs=FREQS, skip_preflight=True)
    return np.asarray(r.s_params)[0, 0, :]


def _n_live(extent):
    """Live wire cells, read from the runner's OWN spelling of the rule.

    #931 R8: a wire port's extent is HALF-OPEN in edges (``lo .. hi - 1``),
    spelled once in :func:`rfx.sources.sources.wire_port_edge_span` and
    called by all three lanes. This helper used to carry a second copy of
    the rule — ``int(round(extent / DX)) + 1``, the endpoint-INCLUSIVE
    spelling — which survived the R8 fix and left the analytic oracle
    disagreeing with the lane it judges (measured: VESSL 369367259274, the
    solved ``S11`` moved to -0.666667 / -0.500000 while this helper still
    said n = 6 / 4). Reading the shared spelling is what stops that from
    recurring; ``test_the_analytic_oracle_reads_the_runners_own_span``
    pins it against the production rasterizer.

    None of these fixtures' extents is sub-cell, so
    ``wire_port_edge_span``'s sub-cell branch — the only one that needs a
    real grid — is never taken; the assert below keeps it that way.
    """
    from rfx.sources.sources import wire_port_edge_span

    lo = int(round(PORT_Z / DX))
    hi = int(round((PORT_Z + extent) / DX))
    assert hi > lo, (
        f"extent {extent} is sub-cell at dx = {DX}: wire_port_edge_span "
        "would take its sub-cell branch, which needs a real grid that this "
        "build-free helper does not have")
    first, last = wire_port_edge_span(None, 2, lo, hi,
                                      PORT_Z, PORT_Z + extent)
    return last - first + 1


def _analytic_s11(extent):
    n = _n_live(extent)
    return (1.0 - n) / (1.0 + n)


# --------------------------------------------------------------------------
# Independent oracle: the RAW V/I ratio, no wave decomposition involved.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nu", [False, True])
def test_the_pec_plates_load_realizes_two_solid_plates(nu):
    """Build-time gate (no solve): each plate realizes walls at BOTH of its
    drawn faces, on both lanes, and the port gap stays clear (#931 §1.2).

    The plates are one cell thick. Before the contract that meant ONE wall
    plane with the normal E live between the two drawn faces — a film, not
    a plate — so the fixture's name and the docstring's "should read
    Gamma = -1" described an object rfx did not build. This says what is
    built, without solving.
    """
    from tests.unit._wall_planes_m import wall_planes_m

    extent = 5e-3
    rz, zs = wall_planes_m(_build(nu, extent=extent, load="pec_plates"), 2,
                           nonuniform=nu)
    assert not rz.sheets, "the plates are volumes"
    lower = [PORT_Z - 2 * DX, PORT_Z - DX]
    upper = [PORT_Z + extent + DX, PORT_Z + extent + 2 * DX]
    want = lower + upper
    assert len(zs) == len(want), f"realized z wall planes {zs}, want {want}"
    for got, exp in zip(zs, want):
        assert abs(got - exp) < 1e-9, (zs, want)
    # the port edge itself is not inside either plate: the load sits one
    # cell away on each side, which is what makes it a LOAD and not a short.
    assert all(not (PORT_Z - 1e-12 <= z <= PORT_Z + extent + 1e-12)
               for z in zs)


@pytest.mark.parametrize("nu", [False, True])
@pytest.mark.parametrize("extent", [1e-3, 3e-3, 5e-3])
def test_the_analytic_oracle_reads_the_runners_own_span(nu, extent):
    """Build-time gate (no solve): ``_n_live`` equals the number of cells
    the PRODUCTION rasterizer makes for the same declaration.

    The analytic oracle in this module is ``(1 - n)/(1 + n)`` with ``n``
    from :func:`_n_live`. That is only an oracle while ``n`` is the count
    the runner actually drives. It stopped being one under #931 R8: the
    extent became half-open in edges, the helper kept the
    endpoint-inclusive ``+ 1``, and the oracle went one cell wrong on
    every row while still looking self-consistent. A gate that compares
    the oracle's own count against the rasterizer is what makes that a
    red rather than a quiet mis-measurement, so this file cannot drift
    from the rule again without saying so.
    """
    sim = _build(nu, extent=extent)
    centers, _mid = sim._wire_port_cell_centers(sim._ports[0])
    assert len(centers) == _n_live(extent), (
        f"the rasterizer drives {len(centers)} cells for extent {extent} "
        f"but the analytic oracle assumes {_n_live(extent)}; the oracle is "
        "the one that must follow rfx.sources.sources.wire_port_edge_span")


@pytest.mark.parametrize("extent", [1e-3, 3e-3, 5e-3])
def test_raw_port_ratio_matches_the_analytic_admittance(extent):
    """``V/I`` off the raw DFT accumulators must equal ``-Z0/n_live``.

    This is the only oracle-grade check in the file: it reads the accumulators
    directly (``forward(port_s11_freqs=...)``), so it cannot be satisfied by
    any (a, b) convention, and it stops short of the extractor's mixed
    normalization. It also pins ``Re(r) <= 0`` and ``|r| <= Z0/n_live``.

    Both of those are properties of the port CELL, not of any structure
    attached across the gap (module docstring) — do not read them as a
    passivity result for the fixture.
    """
    z0 = 50.0
    sim = _build(False, extent=extent, z0=z0)
    fr = sim.forward(n_steps=N_STEPS, port_s11_freqs=FREQS)
    _spec, accs = fr.wire_port_sparams[0]
    # Issue #764: the accumulator tuple grew a 4th slot (whole-port
    # v_port_dft). This test's channels — the single-cell v_dft/i_dft and
    # every asserted value — are byte-identical; only the unpack widens.
    v_dft, i_dft = accs[0], accs[1]
    r = np.asarray(v_dft) / np.asarray(i_dft)
    n = _n_live(extent)
    expected = -z0 / n
    assert np.all(np.real(r) <= 0.0), f"Re(V/I) must be <= 0, got {np.real(r)}"
    assert np.all(np.abs(r) <= z0 / n * (1 + 1e-3)), (
        f"|V/I| must be <= Z0/n_live = {z0/n}, got {np.abs(r)}")
    np.testing.assert_allclose(np.real(r), expected, rtol=1e-4,
                               err_msg=f"raw V/I = {r}")


# --------------------------------------------------------------------------
# D1 — lane parity and passivity on a PASSIVE port.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("extent,load", [
    (1e-3, "vacuum"),
    (3e-3, "vacuum"),
    (5e-3, "pec_plates"),
    (3e-3, "dielectric"),
])
def test_passive_lane_parity_and_passivity(extent, load):
    """NU must reproduce the uniform lane bin-for-bin on the same grid.

    Before the fix the NU value was the exact reciprocal (measured
    -1.40000 vs -0.71429 on the PEC-plate gap), i.e. |S11| > 1 at every bin
    on a passive one-port.
    """
    s_uni = _s11(_build(False, extent=extent, load=load))
    s_nu = _s11(_build(True, extent=extent, load=load))
    resid = float(np.max(np.abs(s_nu - s_uni)))
    print(f"[parity] extent={extent} load={load} "
          f"max|S_NU - S_uni|={resid:.3e} "
          f"S_uni(0.2GHz)={s_uni[0]:.6f} S_NU(0.2GHz)={s_nu[0]:.6f}")
    assert resid <= LANE_PARITY_ATOL, (
        f"lane parity broken: max|S_NU - S_uni| = {resid:.3e} "
        f"(NU {s_nu}, uniform {s_uni})")
    assert np.max(np.abs(s_nu)) <= PASSIVITY_MAX, (
        f"passive one-port with |S11| > 1: {np.abs(s_nu)}")
    assert np.max(np.abs(s_uni)) <= PASSIVITY_MAX


@pytest.mark.parametrize("extent", [1e-3, 3e-3, 5e-3])
def test_passive_s11_regression_lock_on_the_known_wrong_normalization(extent):
    """Self-consistency lock. NOT a physics validation of ``S11``.

    Both lanes must land on ``(1 - n_live)/(1 + n_live)``. That value is
    derived analytically from the discrete Ampere law, but only its FIRST
    half is an oracle: the raw ratio ``V/I -> -Z0/n_live``, which
    ``test_raw_port_ratio_matches_the_analytic_admittance`` pins directly off
    the accumulators.

    The step from that ratio to ``S11`` runs through the extractor's own
    mixed normalization — V and I are sampled at ONE cell
    (``rfx/probes/probes.py:721`` / ``:746``), so the measured Z is the
    per-cell ``Z0/n_live`` while the reflection formula references the
    whole-port ``Z0``. That mismatch is the known-wrong #313 / #318 family
    (#318: "physical termination Z0*(n_live/n)", not Z0; #313 do-not-repeat:
    "the port-cell |S21| envelope ... is a REGRESSION LOCK, not physics").

    So this test freezes an ARTIFACT on purpose, to keep both lanes spelling
    it identically. A future CORRECT normalization fix WILL red it. When that
    happens the fix is not a regression — re-derive the expected value here
    and say so in the commit. Do not "restore" this number.

    Witness that it is an artifact and not the structure's reflection: with
    ``n_live = 4`` this asserts ``S11 = -0.6`` for a vacuum gap, for PEC
    plates bracketing the gap and for an ``eps_r = 10`` slab filling it.
    (#931: the plates realize both faces now and short their own interior;
    the reading is unchanged by the load either way — that IS the finding.)
    """
    expected = _analytic_s11(extent)
    s_uni = _s11(_build(False, extent=extent))
    s_nu = _s11(_build(True, extent=extent))
    print(f"[closed-form] n_live={_n_live(extent)} expected={expected:.6f} "
          f"uni={s_uni[0]:.6f} nu={s_nu[0]:.6f}")
    for label, s in (("uniform", s_uni), ("nonuniform", s_nu)):
        np.testing.assert_allclose(
            np.real(s[:2]), expected, atol=ANALYTIC_ATOL,
            err_msg=f"{label} lane off the closed form: {s}")


def test_nu_wave_split_is_direction_free():
    """S must not depend on ``direction`` at all.

    ``direction`` stays on the port spec (the reference-plane path needs it
    for the outboard sign) but it must not reach the (a, b) split. Before
    the fix "-x"/"-y" gave the reciprocal of "+x"/"+y".
    """
    ref = _s11(_build(True, direction="+x"))
    for d in ("-x", "+y", "-y", None):
        s = _s11(_build(True, direction=d))
        np.testing.assert_array_equal(
            s, ref, err_msg=f"direction={d!r} changed the wave split")


@pytest.mark.parametrize("port_x", [4e-3, 12e-3])
def test_passive_lane_parity_across_the_domain(port_x):
    """Low-x and high-x placements must agree with the uniform lane.

    ``_auto_direction`` returns "-x" for the first and "+x" for the second,
    which is exactly what used to split the two into reciprocals.
    """
    s_uni = _s11(_build(False, port_x=port_x))
    s_nu = _s11(_build(True, port_x=port_x))
    resid = float(np.max(np.abs(s_nu - s_uni)))
    print(f"[parity-x] port_x={port_x} max|S_NU - S_uni|={resid:.3e} "
          f"S_uni(0.2GHz)={s_uni[0]:.6f}")
    assert resid <= LANE_PARITY_ATOL, (
        f"max|S_NU - S_uni| = {resid:.3e} (NU {s_nu}, uniform {s_uni})")
    assert np.max(np.abs(s_nu)) <= PASSIVITY_MAX


# --------------------------------------------------------------------------
# The excited port — the #683 sampling-ordering question, DECIDED and locked.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("load", ["vacuum", "pec_plates"])
def test_excited_port_lane_ordering_disagreement_is_open_683(load):
    """An EXCITED port reads the same on both lanes — LOCKED parity.

    CONVERTED 2026-08-29 from the strict-xfail #683 witness (written
    provenance — docs/design_notes/issue683_decomposer_flip_predeclaration.md
    P4; verdict in issue683_sampling_order_decision_protocol.md section 9):
    the uniform lane now samples the physical V/I/V_port POST-injection —
    the measured-correct side — so the ordering disagreement this witness
    kept visible is RESOLVED, not absorbed.  The name is kept so the git
    history of the open question stays greppable.

    Measured residual history (max over the 3 bins of |S_NU - S_uniform|):

        pre-#764 (per-cell diagonal)    1.983e-01 / 6.109e-01
        #764 base (whole-port, PRE)     4.975e-02 / 1.051e-01
        #683 flip (whole-port, POST)    6.839e-08 / 1.767e-07   <- locked

    At 0.2 GHz both lanes read vacuum +0.999983-0.004385j (bit-close);
    the pre-flip near-conjugate signature (uniform +1.000004+0.005537j)
    was exactly the same-step injection increment the flip removed.
    Both lanes read the unloaded driven column as an open (|S11| ~ 1),
    the physically expected reading.

    ``direction`` is NOT a confounder here: the fixture pins ``direction`` and
    D1 is fixed, so the split is direction-free on both lanes.
    """
    extent = 3e-3 if load == "vacuum" else 5e-3
    s_uni = _s11(_build(False, extent=extent, excite=True, load=load))
    s_nu = _s11(_build(True, extent=extent, excite=True, load=load))
    resid = float(np.max(np.abs(s_nu - s_uni)))
    print(f"[parity-excited] load={load} max|S_NU - S_uni|={resid:.3e} "
          f"uni={s_uni[0]:.6f} nu={s_nu[0]:.6f} "
          f"max|S_NU|={np.max(np.abs(s_nu)):.5f}")
    assert resid <= LANE_PARITY_ATOL, (
        f"excited-port lane parity broken: max|S_NU - S_uni| = {resid:.3e} "
        f"(NU {s_nu}, uniform {s_uni})")
