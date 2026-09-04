"""The auxiliary-echo record invariant (#888), as a computed witness and a gate.

THE FINDING THIS TURNS INTO AN INSTRUMENT. Every TF/SF injection in this repo
reads its incident field from an auxiliary grid whose own absorber reflects 4 to
6 percent in amplitude. Because R and T are normalised by that same auxiliary
field the contamination cancels identically in vacuum, so the leakage and purity
witnesses cannot see it; it enters the measured R and T only once the record is
long enough for the echo to reach the probes. All 13 committed slab-family rungs
happen to be clean because they run at roughly 0.5 to 0.6 of their own echo
arrival -- a property of the geometry (the record law counts from the probe, the
echo's path counts from the auxiliary source) rather than a margin anyone chose,
and documented in neither ``derive_record_length`` nor ``t_safe_steps``. cv26
above 34 degrees is what happens when a record law grows past it.

Sources: ``docs/design_notes/20260903_cv26_oblique_defect_diagnosis.md`` and
``docs/design_notes/20260903_cv04_envelope_decomposition.md``; this lane's own
note is ``docs/design_notes/20260904_aux_echo_record_invariant.md``.

No FDTD runs here. The arrival is GEOMETRY -- it is never measured on the run it
is meant to guard, which is exactly what lets it bound that run.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("ae_cv22_gates", "validation/crossval/comparators/cv22_dispersive_gates.py")
import rfx.sources.tfsf as T  # noqa: E402  -- the module that OWNS the aux layout
LW = _load("ae_lattice_witness", "validation/crossval/comparators/lattice_witness.py")

# The rig's dt at dx = 1 mm (Courant 0.700); the same constant
# tests/crossval/test_lattice_witness_gates.py pins.
_DT_DX = 2.335067793382187e-12

_CASES = {
    "cv04": _REPO / "validation/crossval/_04_fresnel_results",
    "cv22": _REPO / "validation/crossval/_22_dispersive_results",
    "cv23": _REPO / "validation/crossval/_23_lossy_results",
}

# --------------------------------------------------------------------------
# The measured arrivals the two notes report. These are MEASUREMENTS, and the
# guard's job is to sit at or before each of them -- never after.
# --------------------------------------------------------------------------
# cv04, measured as the first step at which the rig and an echo-free control
# (the auxiliary array padded 4000 cells at its hi end) diverge in float32
# (cv04 note section 2, "Measured, not predicted").
#
# RE-MEASURED 2026-09-04 on the derived 200-cell absorber (#888), because the
# 20-cell numbers -- trans 1230, refl 1350 -- are a measurement of a rig that no
# longer exists, and a bound checked against a stale measurement is not checked.
# Same instrument, cv04's own reference indices and its 2-D TMz timestep. The
# echo the divergence carries is now 1.22e-04 of the incident peak, against
# 5.78e-02 on the 20-cell absorber.
CV04_MEASURED_ARRIVAL = {"trans": 1267, "refl": 1366}
CV04_MEASURED_ARRIVAL_AT_20_CELLS = {"trans": 1230, "refl": 1350}
# cv26 te_45 at its declared dx/2 rung: the arrival the #888 diagnosis tabulates
# for the reflection probe (note section 0, "Records against echo arrival:
# ... te_45 18083 vs 9358").
CV26_TE45_ARRIVAL_REFL = 9358
# The 2-D Bloch auxiliary grid at te_45 (#888 note sections 0 and 3): n2x =
# 3092 with a 30-cell CFS-CPML, source at 33, the phase-slope fit localising
# the reflector at index 3070 = 8.0 cells inside the layer, and the reflection
# probe's auxiliary reference index 1475. The speed is the lattice group
# velocity v_gx(f0) = 0.7071 c the note itself uses, in cells per step.
CV26_TE45 = dict(n_aux=3092, src_idx=33, aux_n_cpml=30, reflector_depth_cells=8.0,
                 probe_aux_index=1475, v_cells=0.4949954837618946)


def _witness_docs():
    out = {}
    for case, results in _CASES.items():
        path = results / "lattice_witness.json"
        if not path.is_file():
            continue
        out[case] = json.loads(path.read_text())
    return out


# ==========================================================================
# 1. The computed quantity, against the arrivals that were MEASURED
# ==========================================================================

def test_the_arrival_is_the_LIVE_auxiliary_geometry_not_a_copy_of_it():
    """The geometry is re-derived from the module that owns it, every run.

    It used to be pinned: ``n_1d == 652``, source at 23, the phase-slope fit's
    reflector at 638.88, paths 894.8 / 964.8 cells -- the 20-cell absorber's
    numbers, from cv04 note section 2. #888's fix moved the absorber to 200
    cells, and a pinned copy would have gone on computing an arrival for a grid
    that no longer exists while passing every record put to it. So the
    expectation is COMPUTED from ``rfx.sources.tfsf``'s own constants here, and
    the only literals left are the ones the arithmetic cannot supply: the rig's
    interior and its probe offsets.
    """
    n_cpml, margin, off = T.AUX_N_CPML_1D, T.AUX_N_MARGIN_1D, T.AUX_SRC_OFFSET_1D
    e = G.slab_aux_echo(600, _DT_DX, dx_div=1, n_steps=719)
    cells = G.rig_cells(600, 1)
    x_lo = cells["x_lo"]
    x_hi = cells["nx"] - x_lo - 1
    n_1d = 2 * n_cpml + 2 * margin + (x_hi - x_lo + 2)
    assert e["aux_n_1d"] == n_1d
    assert e["aux_src_idx"] == n_cpml + off
    # The reflector is now a BOUND at the absorber's inner edge, not a fitted
    # position: at |B/A| ~ 1e-06 the residual backward wave is not one specular
    # echo from one place (2026-09-04 note section 4).
    assert G.AUX_REFLECTOR_DEPTH_CELLS == 0.0
    assert G.AUX_REFLECTOR_DEPTH_IS_BOUND is True
    assert e["aux_reflector_index"] == pytest.approx(n_1d - n_cpml, abs=1e-9)
    for probe, key in (("probe_trans", "path_cells_trans"), ("probe_refl", "path_cells_refl")):
        probe_aux = (n_cpml + margin) + (cells[probe] - x_lo)
        expect = (n_1d - n_cpml) - (n_cpml + off) + ((n_1d - n_cpml) - probe_aux)
        assert e[key] == pytest.approx(expect, abs=1e-9)


def test_a_stale_copy_of_the_absorber_depth_would_be_caught():
    """FALSIFIER for the test above: the 20-cell geometry it used to pin must
    now DISAGREE with the live one, or re-deriving bought nothing."""
    e = G.slab_aux_echo(600, _DT_DX, dx_div=1, n_steps=719)
    assert e["aux_n_1d"] != 652
    assert e["aux_src_idx"] != 23
    assert e["aux_reflector_index"] != pytest.approx(638.88, abs=0.01)


@pytest.mark.parametrize("probe", ["trans", "refl"])
def test_the_computed_arrival_bounds_the_measured_cv04_arrival_from_below(probe):
    """The guard must fire BEFORE the contamination lands, never after.

    Measured by first float32 divergence against an echo-free control: 1230
    steps at the transmission probe, 1350 at the reflection probe. The computed
    arrival is the leading edge (the path arithmetic starts the clock at t = 0
    while the injected pulse peaks at t0, so t0/dt is subtracted), so it must
    sit at or before each measurement -- and close enough to be useful.
    """
    e = G.slab_aux_echo(600, _DT_DX, dx_div=1, n_steps=719)
    got = e[f"arrival_steps_{probe}"]
    measured = CV04_MEASURED_ARRIVAL[probe]
    assert got <= measured, (
        f"the computed arrival {got} is LATER than the measured {measured}: a guard that "
        "fires after the echo has landed is not a guard")
    assert got >= 0.9 * measured, (
        f"the computed arrival {got} is more than 10 % early against the measured "
        f"{measured}; a bound that loose would reject clean records")


def test_the_computed_arrival_reproduces_the_cv26_te45_number():
    """The same helper, driven with the 2-D Bloch auxiliary geometry of #888.

    cv26 is not on main, so its geometry is supplied as the numbers the
    diagnosis note states; what is tested is that the arrival arithmetic --
    source to reflector, reflector back to the probe's auxiliary reference,
    divided by the propagation speed -- reproduces the note's 9358 steps
    exactly. That is the second, independent rig this quantity has to describe.
    """
    got = G.aux_echo_arrival(**CV26_TE45)
    assert got["reflector_index"] == 3070.0
    assert got["path_cells"] == 4632.0
    assert got["arrival_centre_steps"] == CV26_TE45_ARRIVAL_REFL


def test_the_arrival_is_geometry_and_not_a_measurement_of_the_run_it_guards():
    """Change only the record; the arrival must not move.

    A witness derived from the record it bounds cannot bound it. This is the
    property that separates this quantity from ``predict_settling``'s
    ``e_absorber``, which #888 §8 found cancelling the very echo it was
    supposed to bound.
    """
    base = G.slab_aux_echo(1000, _DT_DX, dx_div=1, n_steps=1108)
    for n in (100, 1108, 5000, 100000):
        e = G.slab_aux_echo(1000, _DT_DX, dx_div=1, n_steps=n)
        for k in ("echo_arrival_steps", "echo_arrival_centre_steps", "aux_n_1d",
                  "aux_reflector_index", "path_cells_trans", "path_cells_refl"):
            assert e[k] == base[k], (k, n)
        assert e["record_steps"] == n


def test_the_bounding_probe_is_the_transmission_probe_at_every_slab_rung():
    """The echo reaches the transmission reference first (it sits nearer the
    auxiliary absorber), so the record is bounded by that probe. Stated as a
    test because the invariant takes the minimum and a rig change could move it."""
    for nxi, K in ((600, 1), (1000, 1), (1000, 2), (1000, 4)):
        e = G.slab_aux_echo(nxi, _DT_DX / K, dx_div=K, n_steps=1)
        assert e["echo_arrival_probe"] == "trans"
        assert e["arrival_steps_trans"] < e["arrival_steps_refl"]


# ==========================================================================
# 2. The invariant, recorded on every committed slab-family rung
# ==========================================================================

@pytest.mark.parametrize("case", sorted(_CASES))
def test_every_committed_rung_records_the_invariant(case):
    docs = _witness_docs()
    if case not in docs:
        pytest.skip(f"{case} lattice_witness.json absent")
    for name, rung in docs[case]["rungs"].items():
        e = rung.get("aux_echo")
        assert e is not None, (case, name, "the rung carries no auxiliary-echo witness")
        assert e["schema"] == G.AUX_ECHO_SCHEMA and e["issue"] == 888
        assert e["record_steps"] == rung["n_steps"], (case, name)
        assert e["record_over_echo_arrival"] == pytest.approx(
            e["record_steps"] / e["echo_arrival_steps"], rel=1e-12)
        assert e["limit"] == G.AUX_ECHO_RATIO_LIMIT
        assert rung["gates"]["precond_aux_echo_record"] is e["ok"]


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_committed_witness_recomputes_from_the_declared_geometry(case):
    """The block in the artifact is what the helper computes from the rung's own
    declared geometry -- so it cannot go stale against a rig change."""
    docs = _witness_docs()
    if case not in docs:
        pytest.skip(f"{case} lattice_witness.json absent")
    for name, rung in docs[case]["rungs"].items():
        e = rung["aux_echo"]
        again = G.slab_aux_echo(e["nx_interior"] // e["dx_div"], rung["dt_s"],
                                dx_div=e["dx_div"], n_steps=rung["n_steps"])
        assert again == e, (case, name)


def test_no_committed_rung_is_anywhere_near_the_arrival():
    """FALSIFIER COMPLEMENT: the guard is silent at every committed rung, and the
    margin is a number rather than a colour. The two notes report 0.50-0.57 with
    their own convention (path/v_g at f0, no leading-edge margin); this witness
    subtracts the pulse's t0 and uses the Courant cell speed, both of which make
    the arrival EARLIER, so the same rungs read 0.52-0.60 here."""
    docs = _witness_docs()
    ratios = {}
    for case, doc in docs.items():
        for name, rung in doc["rungs"].items():
            e = rung["aux_echo"]
            ratios[f"{case}:{name}"] = e["record_over_echo_arrival"]
            assert e["ok"] is True, (case, name, e["record_over_echo_arrival"])
            assert rung["gates"]["precond_aux_echo_record"] is True, (case, name)
    assert len(ratios) == 13, sorted(ratios)
    lo, hi = min(ratios.values()), max(ratios.values())
    assert 0.50 <= lo and hi <= 0.62, ratios
    for k in sorted(ratios):
        print(f"aux-echo-invariant {k}: record/arrival = {ratios[k]:.3f}")


# ==========================================================================
# 3. Falsifiers
# ==========================================================================

def _cv04_arrival() -> int:
    """cv04's arrival, computed rather than pinned -- it moves with the absorber
    (1196 at 20 cells, 1176 at 200: a deeper absorber pushes the reflector out
    but pushes the source in by more, so it does NOT buy flight time)."""
    return int(G.slab_aux_echo(600, _DT_DX, dx_div=1, n_steps=719)["echo_arrival_steps"])


@pytest.mark.parametrize("offset,expect_fire", [
    (None, False),   # cv04's committed record, 719 steps
    (-1, False),     # the last admissible step
    (0, True),       # the arrival itself: equality is already a failure
    (+104, True),    # the note's first contaminated record class (mean|dR| 0.0149)
    (+204, True),    # the falsifier this lane declares
    (+1804, True),
])
def test_the_guard_fires_on_a_cv04_record_pushed_past_the_arrival(offset, expect_fire):
    """FALSIFIER: cv04's rig, nothing changed but the record.

    Past its own arrival the case reproduces cv26's failure at normal incidence
    (cv04 note §3), so a guard that stayed silent there would be worthless. The
    probe steps are taken RELATIVE to the computed arrival, so the boundary is
    still exercised exactly at, one before and one after -- a pinned 1195/1196
    pair would have drifted off the boundary when the geometry moved and gone on
    passing.
    """
    n_steps = 719 if offset is None else _cv04_arrival() + offset
    e = G.slab_aux_echo(600, _DT_DX, dx_div=1, n_steps=n_steps)
    assert (not e["ok"]) is expect_fire, (n_steps, e["record_over_echo_arrival"])


def test_the_failure_message_names_the_mechanism_and_the_issue():
    """A red gate has to hand its reader the mechanism, not only a ratio: the
    same defect passed every witness cv26 had for two rounds."""
    n_steps = _cv04_arrival() + 204
    e = G.slab_aux_echo(600, _DT_DX, dx_div=1, n_steps=n_steps)
    msg = G.aux_echo_failure_message(e)
    assert "#888" in msg
    for phrase in ("auxiliary", "absorber", "cancels in vacuum",
                   str(n_steps), str(_cv04_arrival())):
        assert phrase in msg, (phrase, msg)


def test_the_gate_in_the_lattice_witness_is_this_same_quantity():
    """The gate the artifact carries is computed by the same helper, through
    ``lattice_witness.aux_echo_witness`` -- one definition, not two."""
    arm = {"dt_s": _DT_DX,
           "run": {"n_steps": 1400, "dx_div": 1, "nx_interior": 600, "record": {}}}
    e = LW.aux_echo_witness(arm)
    assert e == G.slab_aux_echo(600, _DT_DX, dx_div=1, n_steps=1400)
    assert e["ok"] is False


def test_an_arm_with_no_declared_geometry_is_refused_rather_than_ungated():
    """Silently skipping the invariant is how the defect survived: refuse."""
    with pytest.raises(KeyError):
        LW.aux_echo_witness({"dt_s": _DT_DX, "run": {"n_steps": 719, "record": {}}})


def test_a_rig_whose_probes_moved_is_refused_rather_than_guarded_wrongly():
    """The arrival is only a bound for the geometry it was derived for."""
    arm = {"dt_s": _DT_DX,
           "run": {"n_steps": 719, "dx_div": 1, "nx_interior": 600,
                   "record": {"nx_interior": 600, "probe_trans": 355 + 40}}}
    with pytest.raises(ValueError, match="rig bookkeeping drift"):
        LW.aux_echo_witness(arm)


# ==========================================================================
# 4. What the guard cannot do -- recorded as a test so it cannot be forgotten
# ==========================================================================

def test_the_guard_bounds_when_the_echo_arrives_and_not_how_large_it_is():
    """THE LIMITATION, stated executably.

    The invariant is a function of geometry and the record alone. Nothing in it
    reads the absorber's reflection coefficient, so a rig whose auxiliary
    absorber were ten times worse -- or ten times better -- would produce the
    identical verdict. The actual fix is #888's fix candidate 1 (a deeper
    auxiliary absorber with sigma re-derived from a reflection target: measured
    |B/A| = 6.98e-05 against the shipped 4.40e-02), which is undecided; this
    guard does not substitute for it, and it would NOT have caught cv26 had the
    record law been correct but the absorber worse.
    """
    e = G.slab_aux_echo(600, _DT_DX, dx_div=1, n_steps=719)
    assert not any("B/A" in k or "reflection" in k or "rho" in k for k in e), sorted(e)
    # The only absorber property the arrival uses is WHERE the reflection is
    # generated, never HOW MUCH of it there is.
    assert e["aux_reflector_depth_cells"] == G.AUX_REFLECTOR_DEPTH_CELLS
