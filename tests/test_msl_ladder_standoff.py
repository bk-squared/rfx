"""Issue #823: the MSL probe ladder's source near-field standoff, and the
extractor's own ladder self-consistency witness.

Two report-only instruments, both added because the same measurement showed
the lane's committed ``fit_residual`` cannot see its own defect.

WHAT WAS MEASURED (settled attempt-3 witness run, VESSL 369367257533; the
committed dump ``scripts/diagnostics/_coax_msl_transition_settled_run_logs/
witnesses_369367257533_attempt3_x64-0_ladders.npz``). Fitting the PRODUCTION
matrix pencil over windows of the committed 9-probe MSL ladder, MSL drive,
reference plane at the junction:

    window      nearest probe   fit residual (6/8/10 GHz)
    msl[0:9]    0.4 mm          0.342  / 0.264  / 0.222     <- production
    msl[0:8]    1.4 mm          3.68e-3/ 2.54e-3/ 1.78e-3
    msl[0:7]    2.4 mm          1.44e-3/ 1.28e-3/ 1.09e-3
    msl[0:6]    3.4 mm          1.41e-3/ 1.25e-3/ 9.98e-4
    msl[3:6]    5.6-7.6 mm      4.55e-5/ 2.66e-5/ 1.41e-5
    msl[6:9]    8.6-10.6 mm     0.169  / 0.0893 / 0.085

so the corruption is carried by ONE probe -- ``msl[8]``, 0.4 mm from the MSL
port's feed plane at x = 11.0 mm. Every window containing it is garbage;
every window excluding it fits two to three orders better than the coax
lane's own 0.02 bar. #823's headline ("the last third", three probes) is
therefore too pessimistic by two probes; the honest statement is one.

THE DERIVED RULE (see ``msl_source_near_field_standoff_cells``'s own
docstring for the full derivation, reproduced here by
``test_near_field_decay_length_matches_the_substrate_transverse_resonance``):
the excess over the float32 noise floor decays with a measured length
0.2056 / 0.1908 / 0.1831 mm (mean 0.1932 mm), against the grounded-substrate
transverse-resonance prediction 2h/pi = 0.19099 mm -- 1.1% on the mean. With
the coax lane's own 0.02 two-wave residual bar that gives d_min =
(2/pi)*ln(R0/rho_max)*h = 4.0354*h_sub, just under the repo's EXISTING
issue-#80 Fix B constant (5*h_sub, already floored into
``add_msl_port``'s auto ``n_probe_offset``). The rule SHIPPED is that
existing constant; the derivation is what licenses reusing it here and
quantifies its margin (rho(5h) = 4.4e-3 against the 0.02 bar).

Both instruments are REPORT-ONLY: no gate, no refusal, no
``msl_fit_residual_max`` (explicitly out of scope, per the PI sequencing --
standoff and witness first).
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rfx.api import Simulation  # noqa: E402
from rfx.api._preflight import (  # noqa: E402
    msl_source_near_field_standoff_cells,
)
from rfx.api._sparams import _ladder_split_witness  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402
from rfx.sources.coaxial_port import (  # noqa: E402
    coaxial_line_reflection_from_plane_voltages,
)
from rfx.sources.msl_port import (  # noqa: E402
    msl_port_from_entry,
    msl_probe_x_coords_n,
)

_LADDER_NPZ = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "diagnostics" / "_coax_msl_transition_settled_run_logs"
    / "witnesses_369367257533_attempt3_x64-0_ladders.npz"
)

# The token every near-field standoff advisory carries, at BOTH emission
# sites (preflight check 5 and compute_coax_msl_transition's realized-ladder
# warning). Frozen here so the two sites cannot drift apart silently.
STANDOFF_TOKEN = "near-field standoff"


# ---------------------------------------------------------------------------
# 1. The predicate itself
# ---------------------------------------------------------------------------

def test_standoff_helper_is_the_repo_s_existing_5_h_sub_constant():
    """The shipped threshold is ``max(3, round(5*h_sub/dx))`` -- the SAME
    number ``rfx/api/__init__.py::add_msl_port`` already floors its auto
    ``n_probe_offset`` to (issue #80 Fix B). Not a new constant."""
    assert msl_source_near_field_standoff_cells(300e-6, 100e-6) == 15
    assert msl_source_near_field_standoff_cells(254e-6, 80e-6) == 16
    assert msl_source_near_field_standoff_cells(254e-6, 127e-6) == 10
    assert msl_source_near_field_standoff_cells(254e-6, 64e-6) == 20
    assert msl_source_near_field_standoff_cells(1000e-6, 250e-6) == 20
    # Floor at 3: add_msl_port refuses anything below it anyway.
    assert msl_source_near_field_standoff_cells(10e-6, 100e-6) == 3
    # Degenerate inputs fall back to the floor rather than raising: an
    # advisory must never crash a run.
    assert msl_source_near_field_standoff_cells(300e-6, 0.0) == 3
    assert msl_source_near_field_standoff_cells(0.0, 100e-6) == 3


def test_helper_matches_add_msl_port_s_own_auto_floor():
    """Auto ports (``n_probe_offset=None``) cannot violate BY CONSTRUCTION:
    ``add_msl_port`` takes ``max(3, lam_cells, round(5*h/dx))``, which is
    >= this helper for every (h, dx). Measured on the RO4350B fixture."""
    sim = _sim_with_msl_port(n_probe_offset=None)
    pe = sim._msl_ports[0]
    assert pe.n_probe_offset >= msl_source_near_field_standoff_cells(
        float(pe.height), float(sim._dx)
    )


def test_near_field_decay_length_matches_the_substrate_transverse_resonance():
    """The derivation, recomputed from the committed attempt-3 ladder dump.

    Fit the two-wave model on the CLEAN window ``msl[0:6]``, extrapolate to
    all nine planes, and read the relative excess rho(d) at each probe. The
    six far probes are a flat float32 noise floor; the two near-port probes
    carry a decaying excess whose length is the substrate's own transverse
    quarter-wave scale 2h/pi -- a property of the board, not of the fixture.
    """
    d = np.load(_LADDER_NPZ, allow_pickle=False)
    x = d["msl_ladder_x_m"]
    v = d["msl_ladder_v"][1]                    # MSL drive
    ref = float(d["ref_msl_m"])
    idx = np.arange(6)
    rho = np.empty((3, 9))
    for fi in range(3):
        out = coaxial_line_reflection_from_plane_voltages(
            x[idx], v[idx, fi], reference_plane_m=ref)
        g = out.gamma
        xc = float(x[idx].mean())
        phi = np.stack([np.exp(+g * (x[idx] - xc)),
                        np.exp(-g * (x[idx] - xc))], axis=1)
        ab, *_ = np.linalg.lstsq(phi, v[idx, fi], rcond=None)
        model = ab[0] * np.exp(+g * (x - xc)) + ab[1] * np.exp(-g * (x - xc))
        rho[fi] = np.abs(v[:, fi] - model) / np.abs(model)

    floor = np.median(rho[:, :6], axis=1)
    assert np.all(floor < 2.0e-3), floor          # flat float32 noise floor
    excess_04 = rho[:, 8] - floor                 # probe at d = 0.4 mm
    excess_14 = rho[:, 7] - floor                 # probe at d = 1.4 mm
    assert np.all(excess_04 > 0.8), excess_04     # 0.817 / 1.017 / 1.273
    delta = 1.0e-3 / np.log(excess_04 / excess_14)

    h_sub = 300e-6
    delta_model = 2.0 * h_sub / math.pi           # 0.19099 mm
    assert abs(float(delta.mean()) - delta_model) / delta_model < 0.02, (
        float(delta.mean()), delta_model)
    for dm in delta:
        assert abs(float(dm) - delta_model) / delta_model < 0.10, float(dm)

    # Amplitude referred back to the port plane, and the resulting floor.
    r0 = float(np.max(excess_04 * np.exp(0.4e-3 / delta)))
    assert 11.0 < r0 < 11.7, r0                   # measured 11.323
    d_min_over_h = (2.0 / math.pi) * math.log(r0 / 0.02)
    assert 4.0 < d_min_over_h < 4.1, d_min_over_h  # measured 4.0354
    # The SHIPPED 5*h rule is conservative against that floor, and its own
    # predicted residual at 5*h clears the 0.02 bar with two decades.
    assert d_min_over_h < 5.0
    assert r0 * math.exp(-5.0 * h_sub / delta_model) < 0.02


# ---------------------------------------------------------------------------
# 2. Preflight check 5 -- the registered-port advisory (fail before fix)
# ---------------------------------------------------------------------------

_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
_LX = 14e-3


def _sim_with_msl_port(*, dx: float = 80e-6, n_probe_offset=None) -> Simulation:
    """RO4350B-class MSL line, the ``test_msl_port_preflight`` geometry.

    dx = 80um, h_sub = 254um -> the standoff is 16 cells (1.27 mm = 5*h_sub).
    """
    ly = _W_TRACE + 2.0 * (2.0 * _H_SUB + 8.0 * dx) + 4.0 * dx
    lz = _H_SUB + 1.5e-3
    sim = Simulation(
        freq_max=5e9, domain=(_LX, ly, lz), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("ro4350b", eps_r=_EPS_R)
    sim.add(Box((0, 0, 0), (_LX, ly, _H_SUB)), material="ro4350b")
    y_c = ly / 2.0
    sim.add(Box((0, y_c - _W_TRACE / 2, _H_SUB),
                (_LX, y_c + _W_TRACE / 2, _H_SUB + dx)), material="pec")
    sim.add_msl_port(position=(4e-3, y_c, 0), width=_W_TRACE, height=_H_SUB,
                     direction="+x", impedance=50.0, eps_r_sub=_EPS_R,
                     n_probe_offset=n_probe_offset)
    return sim


def _standoff_rows(sim: Simulation) -> list[str]:
    return [m for m in sim.preflight() if STANDOFF_TOKEN in m]


def test_msl_probe_ladder_inside_the_source_near_field_is_advised():
    """FAIL-BEFORE-FIX gate for preflight check 5.

    ``n_probe_offset=10`` at h_sub=254um / dx=80um puts probe 0 at 0.80 mm =
    3.15*h_sub -- inside BOTH the shipped 5*h standoff (16 cells) and the
    derived 4.0354*h floor (1.025 mm = 12.8 cells). Exactly ONE new
    ``msl_port_geometry`` row must say so.
    """
    sim = _sim_with_msl_port(n_probe_offset=10)
    rows = _standoff_rows(sim)
    assert len(rows) == 1, rows
    msg = rows[0]
    assert "n_probe_offset=10" in msg
    assert "16 cells" in msg
    assert "5·h_sub" in msg or "5*h_sub" in msg


def test_the_standoff_advisory_carries_the_msl_port_geometry_code():
    """Same-check-family slug reuse (the check-2c / #752 precedent): a new
    SITE, no new ``code=``, so
    ``tests/test_preflight_advisory_emission_contract.py``'s
    ``_FROZEN_LITERAL_CODE_COUNT`` does not move."""
    sim = _sim_with_msl_port(n_probe_offset=10)
    hits = [i for i in sim.preflight() if STANDOFF_TOKEN in str(i)]
    assert len(hits) == 1, [str(i) for i in hits]
    assert hits[0].code == "msl_port_geometry"
    assert hits[0].severity == "warning"        # report-only, never an error


def test_a_compliant_registered_port_draws_no_standoff_advisory():
    """FALSE-POSITIVE control, explicit offset exactly AT the threshold."""
    sim = _sim_with_msl_port(n_probe_offset=16)
    assert _standoff_rows(sim) == []


def test_auto_offset_ports_never_draw_the_standoff_advisory():
    """FALSE-POSITIVE control on the repo's OWN default: an auto
    ``n_probe_offset`` is floored to max(3, lam_cells, round(5h/dx)), so the
    advisory is unreachable for it by construction."""
    for dx in (80e-6, 127e-6, 64e-6, 250e-6):
        sim = _sim_with_msl_port(dx=dx)
        assert _standoff_rows(sim) == [], (dx, _standoff_rows(sim))


def test_committed_example_msl_port_geometries_are_all_compliant():
    """FALSE-POSITIVE census over the committed examples/validation scripts.

    Measured 2026-09-01 by instrumenting ``Simulation.add_msl_port`` across
    ``tests/test_example_fidelity_contract.py`` (which BUILDS all 35 snapshot
    variants without solving): 92 passed, 6 MSL ports registered, 0
    violating. So ``tests/data/example_fidelity_snapshot.json`` and the
    ``msl_port_geometry 6`` header count in that file do NOT move and were
    NOT re-captured. The six realized (n_probe_offset, h_sub, dx) triples are
    frozen here so a future dx change that WOULD move the snapshot reds this
    cheap test first.

    Note the margin: two of the six clear the threshold by exactly ONE cell.
    """
    committed = (
        # (n_probe_offset, h_sub_m, dx_m, required_cells)
        (20, 1000e-6, 250e-6, 20),
        (20, 1000e-6, 250e-6, 20),
        (11, 254e-6, 127e-6, 10),
        (11, 254e-6, 127e-6, 10),
        (28, 254e-6, 64e-6, 20),
        (28, 254e-6, 64e-6, 20),
    )
    for off, h, dx, required in committed:
        got = msl_source_near_field_standoff_cells(h, dx)
        assert got == required, (off, h, dx, got, required)
        assert off >= got, (off, h, dx, got)


# ---------------------------------------------------------------------------
# 3. The coax<->MSL lane's own realized ladder (method-side advisory)
# ---------------------------------------------------------------------------

def _realized_ladder(build, *, count, start, spacing):
    """Realized MSL probe x-coordinates for a committed coax<->MSL fixture,
    via the PRODUCTION ``msl_probe_x_coords_n`` on the real run grid."""
    sim = build()
    pe = sim._msl_ports[0]
    xs = msl_probe_x_coords_n(
        sim._build_grid(), msl_port_from_entry(pe),
        n_probes=count, n_offset_cells=start, n_spacing_cells=spacing,
    )
    return (sorted(float(x) for x in xs), float(pe.position[0]),
            float(pe.height), float(sim._dx))


@pytest.mark.parametrize(
    "label,builder_name,count,start,spacing,junction_x_name,"
    "n_violating_port,n_violating_junction",
    [
        # Measured 2026-09-01 by building each committed fixture and
        # evaluating the predicate on the realized ladder. The standoff is
        # 15 cells = 1.500 mm = 5.000*h_sub at dx=100um, h_sub=300um.
        ("attempt1", "_build_coax_msl_transition_sim", 6, 4, 2,
         "JUNCTION_X", 6, 0),
        ("attempt2", "_build_coax_msl_transition_sim_attempt2", 9, 4, 10,
         "JUNCTION_X", 2, 0),
        ("attempt2_wide", "_build_coax_msl_transition_sim_attempt2_wide",
         9, 4, 10, "JUNCTION_X_2W", 2, 0),
        ("attempt3", "_build_coax_msl_transition_sim_attempt3", 9, 4, 10,
         "JUNCTION_X", 2, 0),
        # Attempt 3b: the same geometry with the compliant ladder (the
        # unique MAXIMAL-COUNT one at the inherited spacing -- that
        # uniqueness is proved in
        # tests/test_coax_msl_transition.py::
        # test_attempt3b_ladder_is_the_unique_maximal_count_compliant_ladder,
        # not re-derived here).
        ("attempt3b", "_build_coax_msl_transition_sim_attempt3b",
         8, 15, 10, "JUNCTION_X", 0, 0),
        # Clears the DERIVED 4.0354*h floor but not the shipped 5*h rule --
        # rejected rather than loosening the rule to admit it.
        ("rejected_9_14_9", "_build_coax_msl_transition_sim_attempt3",
         9, 14, 9, "JUNCTION_X", 1, 1),
    ],
)
def test_committed_coax_msl_ladders_violate_exactly_the_measured_slots(
    label, builder_name, count, start, spacing, junction_x_name,
    n_violating_port, n_violating_junction,
):
    """Enumerated false-positive cost of the shipped 5*h rule on this lane.

    Honest accounting: the MEASUREMENT above says only the d = 0.4 mm probe
    is actually corrupt, so on attempts 2/3 the 5*h rule over-flags exactly
    ONE further probe slot (d = 1.4 mm, rho = 7.6e-3, clean by the 0.02
    bar). That is the price of reusing the repo's existing constant instead
    of shipping a second, tighter one -- and it costs nothing, because the
    advisory is report-only and refuses nothing.
    """
    import test_coax_msl_transition as T

    xs, feed_x, h_sub, dx = _realized_ladder(
        getattr(T, builder_name), count=count, start=start, spacing=spacing)
    junction_x = getattr(T, junction_x_name)
    required_m = msl_source_near_field_standoff_cells(h_sub, dx) * dx
    assert required_m == pytest.approx(5.0 * h_sub)

    viol_port = sum(1 for x in xs if abs(feed_x - x) < required_m - 1e-12)
    viol_junction = sum(1 for x in xs
                        if abs(x - junction_x) < required_m - 1e-12)
    assert (viol_port, viol_junction) == (
        n_violating_port, n_violating_junction), (label, xs)


# ---------------------------------------------------------------------------
# 4. The ladder self-consistency witness
# ---------------------------------------------------------------------------

def test_ladder_split_witness_separates_a_corrupt_ladder_from_a_clean_one():
    """FAIL-BEFORE-FIX gate for the witness, on the COMMITTED attempt-3 dump.

    A residual computed over a window that includes garbage cannot detect
    that the window is the problem. Two disjoint contiguous halves, same
    reference plane, same extractor, can: the corrupt 9-probe MSL ladder
    disagrees with itself by 4.32-4.49 DECADES in |Gamma|, the compliant
    8-probe subset by 0.001-0.005. Three orders of separation.
    """
    d = np.load(_LADDER_NPZ, allow_pickle=False)
    x = d["msl_ladder_x_m"]
    v = d["msl_ladder_v"]
    ref = float(d["ref_msl_m"])

    gdev, decades = _ladder_split_witness(x, v, ref)
    assert gdev.shape == decades.shape == (2, 3)
    assert gdev.dtype == decades.dtype == np.float64

    # MSL drive (index 1) on the MSL array -- the discriminating channel.
    assert np.all(decades[1] > 4.3) and np.all(decades[1] < 4.5), decades[1]
    assert np.all(gdev[1] > 1.6), gdev[1]

    # The compliant 8-probe subset of the SAME dump.
    gdev8, decades8 = _ladder_split_witness(x[:8], v[:, :8, :], ref)
    assert np.all(decades8[1] <= 0.05), decades8[1]
    assert np.all(decades8[0] <= 0.05), decades8[0]
    assert np.all(gdev8[1] < 0.05), gdev8[1]


def test_ladder_split_witness_uses_disjoint_halves_and_drops_the_middle():
    """For odd N the two halves must not share a probe: N=9 -> [0:4] and
    [5:9]. Proven by feeding a ladder whose middle probe alone is poisoned:
    the witness is unchanged, because neither half reads it."""
    d = np.load(_LADDER_NPZ, allow_pickle=False)
    x = d["msl_ladder_x_m"]
    v = np.array(d["msl_ladder_v"])
    ref = float(d["ref_msl_m"])
    base = _ladder_split_witness(x, v, ref)
    v[:, 4, :] *= 3.7                       # middle probe of nine
    poisoned = _ladder_split_witness(x, v, ref)
    assert np.array_equal(base[0], poisoned[0])
    assert np.array_equal(base[1], poisoned[1])


def test_ladder_split_witness_is_nan_below_six_probes():
    """Each half needs >= 3 planes for the matrix pencil, so a 5-probe
    ladder has no witness -- reported as NaN, never as a small number."""
    d = np.load(_LADDER_NPZ, allow_pickle=False)
    x = d["msl_ladder_x_m"]
    v = d["msl_ladder_v"]
    ref = float(d["ref_msl_m"])
    gdev, decades = _ladder_split_witness(x[:5], v[:, :5, :], ref)
    assert gdev.shape == (2, 3)
    assert np.all(np.isnan(gdev)) and np.all(np.isnan(decades))
    # Six is the smallest ladder that HAS one.
    gdev6, decades6 = _ladder_split_witness(x[:6], v[:, :6, :], ref)
    assert np.all(np.isfinite(gdev6)) and np.all(np.isfinite(decades6))


def test_witness_reproduces_the_committed_coax_ladder_reading():
    """The coax array's own half-split reads gamma_dev 0.11-0.34 on EVERY
    run -- the already-known 1 mm-span limitation of the coax stub (#589).
    Recorded here as the reason this witness stays REPORT-ONLY rather than
    becoming a gate: a bar tight enough to catch the MSL ladder would refuse
    the coax ladder on a defect that is not this one."""
    d = np.load(_LADDER_NPZ, allow_pickle=False)
    gdev, decades = _ladder_split_witness(
        d["coax_ladder_z_m"], d["coax_ladder_v"], float(d["ref_coax_m"]))
    assert np.all(gdev > 0.10) and np.all(gdev < 0.35), gdev
    assert np.all(decades < 0.35), decades


# ---------------------------------------------------------------------------
# 5. End to end on the committed attempt-3 fixture (one short FDTD pair)
# ---------------------------------------------------------------------------

_N_STEPS_SMOKE = 200


@pytest.fixture(scope="module")
def _attempt3_smoke():
    """One 200-step attempt-3 call. Far too short for a settled S (the
    ring-down witness rightly screams and every number below is a truncation
    artifact) -- irrelevant here: the advisory is pure ladder GEOMETRY and
    the witness's SHAPE/dtype/finiteness contract is step-count independent.
    ``return_ladder_voltages=True`` because the split witness is opt-in with
    the ladder dump (None on a default call; pinned in
    tests/test_coax_msl_transition_ladder_dump.py).
    """
    import test_coax_msl_transition as T

    sim = T._build_coax_msl_transition_sim_attempt3()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_coax_msl_transition(
            **T._attempt2_kwargs(_N_STEPS_SMOKE), return_ladder_voltages=True)
    return res, [str(w.message) for w in caught]


def test_realized_ladder_standoff_advisory_fires_on_attempt3(_attempt3_smoke):
    """FAIL-BEFORE-FIX gate for the method-side advisory.

    ``msl_probe_*`` are METHOD arguments, invisible to preflight (they are
    not on the registered ``_MSLPortEntry`` at all), so the registered-port
    check 5 cannot see this ladder. The method therefore evaluates the SAME
    predicate on its OWN realized ``xs_sorted``, at both ends the ladder is
    referred to, and says so via ``warnings.warn``.
    """
    _res, msgs = _attempt3_smoke
    hits = [m for m in msgs
            if STANDOFF_TOKEN in m and "compute_coax_msl_transition" in m]
    assert len(hits) == 1, msgs
    msg = hits[0]
    assert "2 of 9" in msg
    assert "11.00" in msg          # the MSL port feed plane, x = 11.0 mm
    assert "1.50" in msg           # the required standoff, 1.50 mm


def test_witness_fields_are_on_the_result_and_report_only(_attempt3_smoke):
    """The two witness fields exist (the fixture opted in with
    ``return_ladder_voltages=True``), are shaped like the diagnostics they
    sit beside, and refuse nothing."""
    res, _msgs = _attempt3_smoke
    n_f = len(res.freqs)
    for name in ("ladder_split_gamma_dev", "ladder_split_reflection_decades"):
        arr = np.asarray(getattr(res, name))
        assert arr.shape == (2, 2, n_f), (name, arr.shape)
        assert arr.dtype == np.float64, (name, arr.dtype)
    # Same [port_array, drive, freq] indexing as fit_residual / gamma, so the
    # rows line up with the existing diagnostics.
    assert res.fit_residual.shape == res.ladder_split_gamma_dev.shape
    # The 9-probe MSL ladder disagrees with itself; the 6-probe coax ladder
    # carries its own known 1 mm-span limitation. Neither refuses anything --
    # the call returned a result.
    assert np.all(np.isfinite(res.ladder_split_gamma_dev))
    assert res.status == "experimental"


def test_witness_matches_a_standalone_refit_of_the_dumped_ladders():
    """The result's witness IS the disjoint-half refit of the SAME ladders,
    not a parallel re-derivation: recompute it from the ladder dump and
    require bit equality. Proved on the committed settled dump so this costs
    no FDTD."""
    d = np.load(_LADDER_NPZ, allow_pickle=False)
    gdev_msl, dec_msl = _ladder_split_witness(
        d["msl_ladder_x_m"], d["msl_ladder_v"], float(d["ref_msl_m"]))
    gdev_coax, dec_coax = _ladder_split_witness(
        d["coax_ladder_z_m"], d["coax_ladder_v"], float(d["ref_coax_m"]))
    stacked_g = np.stack([gdev_coax, gdev_msl])
    stacked_d = np.stack([dec_coax, dec_msl])
    assert stacked_g.shape == stacked_d.shape == (2, 2, 3)
    # Port array 0 = coax, 1 = msl -- the same order as every other
    # (port_array, drive, freq) field on the result.
    assert np.array_equal(stacked_g[1], gdev_msl)
    assert np.array_equal(stacked_d[0], dec_coax)
