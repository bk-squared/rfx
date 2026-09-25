"""A ``Simulation.run`` argument a lane does not implement is refused.

Before 2.0 the distributed, non-uniform, subgridded and ADI lanes warned
that an argument "is silently ignored" and ran without it, so the result
was a different computation from the one requested: a staircase PEC
instead of the Dey-Mittra conformal boundary (``conformal_pec``), scalar
eps at interfaces instead of the smoothed tensor (``subpixel_smoothing``),
a fixed ``n_steps`` instead of a run to decay (``until_decay``), no field
snapshot (``snapshot``), a full AD tape instead of checkpointing
(``checkpoint``), an S-parameter record of the wrong length
(``s_param_n_steps``). Each of those now raises ``NotImplementedError``
naming the argument and the lane, before any time step. The rows below pin
every (lane x argument) pair that refuses; the controls pin that the same
argument still runs on a lane that implements it.

``report_every`` only prints progress, so it stays a warning.

The file keeps its old name because other files cite it. It also carries
the P2.7 preflight checks (PMC/PEC face plus CPML on one axis).
"""

from __future__ import annotations

import re
import sys
import warnings

import jax
import numpy as np
import pytest

from rfx import Box, Cylinder, DebyePole, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.simulation import SnapshotSpec


# The value each argument takes when it asks for its feature.
ASKS = {
    "subpixel_smoothing": True,
    "checkpoint": True,
    "snapshot": SnapshotSpec(interval=2, components=("ez",)),
    "until_decay": 1e-3,
    "conformal_pec": True,
}


def _quiet(build):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build()


def _nu_sim():
    """Closed (PEC) cavity with a 4-cell graded dz."""
    def build():
        sim = Simulation(freq_max=10e9, domain=(2e-3, 2e-3, 4e-3),
                         dx=1e-3, boundary="pec", cpml_layers=0)
        sim._dz_profile = np.full(4, 1e-3)
        sim.add_source((1e-3, 1e-3, 1e-3), "ex")
        sim.add_probe((1e-3, 1e-3, 3e-3), "ex")
        return sim
    return _quiet(build)


def _distributed_sim():
    def build():
        sim = Simulation(freq_max=15e9, domain=(24e-3, 12e-3, 12e-3),
                         dx=1e-3, boundary="pec")
        sim.add_source((6e-3, 6e-3, 6e-3), "ez", amplitude_kind="field")
        sim.add_probe((12e-3, 6e-3, 6e-3), "ez")
        return sim
    return _quiet(build)


def _subgrid_sim():
    def build():
        sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02),
                         dx=1e-3, boundary="pec")
        sim.add_refinement((0.012, 0.02), ratio=2)
        sim.add_source((0.01, 0.01, 0.017), "ez")
        sim.add_probe((0.012, 0.01, 0.017), "ez")
        return sim
    return _quiet(build)


def _adi_sim():
    def build():
        sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.01),
                         boundary="pec", mode="2d_tmz", solver="adi")
        sim.add_source((0.01, 0.01, 0.0), "ez")
        sim.add_probe((0.012, 0.01, 0.0), "ez")
        return sim
    return _quiet(build)


def _devices():
    devices = jax.devices("cpu")
    assert len(devices) >= 2, "requires the root conftest's two CPU devices"
    return devices[:2]


LANES = {
    # lane: (fixture, label in the message, extra run kwargs)
    "non-uniform": (_nu_sim, "non-uniform mesh", {}),
    "distributed": (_distributed_sim, "distributed multi-device", None),
    "subgridded": (_subgrid_sim, "subgridded (SBP-SAT)", {}),
    "adi": (_adi_sim, "ADI (solver='adi')", {}),
}

REFUSED = [
    ("non-uniform", "snapshot"),
    ("non-uniform", "until_decay"),      # closed boundary: no decay stop
    ("non-uniform", "conformal_pec"),
    *[(lane, kw) for lane in ("distributed", "subgridded", "adi")
      for kw in ASKS],
]


def _refusal(label, kw):
    return rf"(?s)refuses .*the {re.escape(label)} lane.*\b{kw}="


@pytest.mark.parametrize("lane,kw", REFUSED, ids=[f"{a}-{b}" for a, b in REFUSED])
def test_lane_refuses_an_argument_it_does_not_implement(lane, kw):
    build, label, extra = LANES[lane]
    sim = build()
    kwargs = {"n_steps": 4, "skip_preflight": True, kw: ASKS[kw]}
    if extra is None:
        kwargs["devices"] = _devices()
    with pytest.raises(NotImplementedError, match=_refusal(label, kw)) as info:
        sim.run(**kwargs)
    print(f"[{lane} x {kw}] {info.value}", file=sys.stderr)
    assert "Instead:" in str(info.value)


def test_closed_nu_until_decay_refusal_states_the_absorber_reason():
    """#383: the closed-boundary NU refusal says why, and what works."""
    with pytest.raises(NotImplementedError) as info:
        _nu_sim().run(n_steps=4, until_decay=1e-3, skip_preflight=True)
    msg = str(info.value)
    assert "absorbing" in msg and "point-field fallback" in msg, msg
    assert "set n_steps or num_periods" in msg, msg


def test_conformal_boundary_declaration_is_refused_on_the_nu_lane():
    """``Boundary(conformal=True)`` turns conformal_pec on by itself."""
    def build():
        spec = BoundarySpec(x="pec", y=Boundary(lo="pec", hi="pec",
                                                conformal=True), z="pec")
        sim = Simulation(freq_max=10e9, domain=(2e-3, 2e-3, 4e-3), dx=1e-3,
                         boundary=spec, cpml_layers=0)
        sim._dz_profile = np.full(4, 1e-3)
        sim.add_source((1e-3, 1e-3, 1e-3), "ex")
        sim.add_probe((1e-3, 1e-3, 3e-3), "ex")
        return sim
    sim = _quiet(build)
    with pytest.raises(NotImplementedError,
                       match=_refusal("non-uniform mesh", "conformal_pec")):
        sim.run(n_steps=4, skip_preflight=True)


def test_every_refused_argument_is_named_in_one_error():
    sim = _subgrid_sim()
    with pytest.raises(NotImplementedError) as info:
        sim.run(n_steps=4, skip_preflight=True, **ASKS)
    msg = str(info.value)
    assert f"refuses {len(ASKS)} argument(s)" in msg, msg
    for kw in ASKS:
        assert f"  {kw}=" in msg, (kw, msg)


# --------------------------------------------------------------------
# S-parameter record length: a lane that reads S from the main run's own
# port record cannot honour an s_param_n_steps other than that record.
# --------------------------------------------------------------------

def _wire_sim(nonuniform):
    def build():
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.01), dx=1e-3,
                         boundary="pec", cpml_layers=0,
                         **({"dz_profile": np.full(10, 1e-3)}
                            if nonuniform else {}))
        sim.add_port((0.01, 0.01, 0.002), "ez", impedance=50.0, extent=0.006)
        sim.add_probe((0.01, 0.01, 0.005), "ez")
        return sim
    return _quiet(build)


@pytest.mark.parametrize("nonuniform,label", [
    (True, "non-uniform mesh"),
    (False, "uniform single-wire-port S-parameter"),
])
def test_one_record_lanes_refuse_a_different_s_param_n_steps(nonuniform, label):
    sim = _wire_sim(nonuniform)
    with pytest.raises(NotImplementedError,
                       match=_refusal(label, "s_param_n_steps")):
        sim.run(n_steps=20, compute_s_params=True,
                s_param_freqs=np.array([2e9, 3e9]), s_param_n_steps=40,
                skip_preflight=True)


@pytest.mark.parametrize("nonuniform", [True, False],
                         ids=["non-uniform", "uniform"])
def test_s_param_n_steps_equal_to_the_record_still_runs(nonuniform):
    sim = _wire_sim(nonuniform)
    res = _quiet(lambda: sim.run(
        n_steps=20, compute_s_params=True,
        s_param_freqs=np.array([2e9, 3e9]), s_param_n_steps=20,
        skip_preflight=True))
    assert res.s_params is not None and res.s_params.shape[-1] == 2


# --------------------------------------------------------------------
# Debye/Lorentz materials: the dispersive E update reads neither the
# smoothed tensor nor the Dey-Mittra eps correction.
# --------------------------------------------------------------------

def _debye_sim(nonuniform, pec_cylinder=False):
    def build():
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02), dx=2e-3,
                         boundary="pec", cpml_layers=0,
                         **({"dz_profile": np.full(10, 2e-3)}
                            if nonuniform else {}))
        sim.add_material("disp", eps_r=4.0,
                         debye_poles=[DebyePole(delta_eps=1.0, tau=1e-11)])
        sim.add(Box((0.004, 0.004, 0.004), (0.012, 0.012, 0.012)),
                material="disp")
        if pec_cylinder:
            sim.add(Cylinder((0.014, 0.014, 0.01), 0.003, 0.012, axis="z"),
                    material="pec")
        sim.add_source((0.006, 0.006, 0.01), "ez")
        sim.add_probe((0.016, 0.006, 0.01), "ez")
        return sim
    return _quiet(build)


@pytest.mark.parametrize("nonuniform,kw,val,pec", [
    (False, "subpixel_smoothing", True, False),
    (False, "subpixel_smoothing", "kottke_pec", False),
    (False, "conformal_pec", True, True),
    (True, "subpixel_smoothing", True, False),
], ids=["uniform-subpixel", "uniform-kottke", "uniform-conformal",
        "nonuniform-subpixel"])
def test_dispersive_lanes_refuse_smoothing_and_conformal(nonuniform, kw, val, pec):
    sim = _debye_sim(nonuniform, pec_cylinder=pec)
    label = ("non-uniform" if nonuniform else "uniform") + (
        " mesh with Debye/Lorentz materials")
    with pytest.raises(NotImplementedError, match=_refusal(label, kw)):
        sim.run(n_steps=4, skip_preflight=True, **{kw: val})


def _wr90_sim(debye, conformal=False, dz=None):
    """A short WR-90 section, optionally with a Debye slab inside."""
    def build():
        walls = Boundary(lo="pec", hi="pec", conformal=conformal)
        sim = Simulation(freq_max=8e9, domain=(0.06, 0.02286, 0.01016),
                         dx=3e-3, cpml_layers=6,
                         boundary=BoundarySpec(x="cpml", y=walls, z=walls),
                         **({"dz_profile": dz} if dz is not None else {}))
        if debye:
            sim.add_material("disp", eps_r=2.0, debye_poles=[
                DebyePole(delta_eps=1.0, tau=1e-11)])
            sim.add(Box((0.024, 0.0, 0.0), (0.036, 0.02286, 0.01016)),
                    material="disp")
        freqs = np.linspace(6e9, 7e9, 3)
        for x, d, name in ((0.009, "+x", "l"), (0.051, "-x", "r")):
            sim.add_waveguide_port(x, direction=d, mode=(1, 0),
                                   mode_type="TE", freqs=freqs, f0=6.5e9,
                                   bandwidth=0.4, name=name)
        return sim
    return _quiet(build)


@pytest.mark.parametrize("kw,val,conformal", [
    ("subpixel_smoothing", True, False),
    ("subpixel_smoothing", "kottke_pec", False),
    ("conformal_pec", None, True),
], ids=["subpixel", "kottke", "conformal-boundary"])
def test_waveguide_s_matrix_refuses_smoothing_and_conformal_with_debye(
        kw, val, conformal):
    """compute_waveguide_s_matrix() shares the uniform lane's dispersive
    E update, so it refuses the same two requests."""
    sim = _wr90_sim(debye=True, conformal=conformal)
    kwargs = {} if val is None else {kw: val}
    with pytest.raises(NotImplementedError, match=_refusal(
            "waveguide S-matrix with Debye/Lorentz materials", kw)) as info:
        sim.compute_waveguide_s_matrix(n_steps=20, normalize=True, **kwargs)
    msg = str(info.value)
    assert msg.startswith("compute_waveguide_s_matrix() refuses"), msg
    if conformal:
        assert "Boundary(conformal=True)" in msg, msg


def test_waveguide_s_matrix_still_smooths_without_dispersion():
    """Control: the same call with a plain dielectric slab runs."""
    def build():
        sim = _wr90_sim(debye=False)
        sim.add(Box((0.024, 0.0, 0.0), (0.036, 0.02286, 0.01016)),
                material="fr4")
        return sim
    sim = _quiet(build)
    res = _quiet(lambda: sim.compute_waveguide_s_matrix(
        n_steps=20, normalize=True, subpixel_smoothing=True))
    assert np.asarray(res.s_params).shape[:2] == (2, 2)


# --------------------------------------------------------------------
# Boundary(conformal=True) off run(): forward() (so optimize()) and the
# non-uniform waveguide S-matrix have no Dey-Mittra update. Measured before
# the refusal: flag on vs off gave bit-identical forward() time series
# (uniform and non-uniform) and a bit-identical non-uniform waveguide S,
# while run() and the uniform waveguide lane moved.
# --------------------------------------------------------------------

def _conformal_cavity(conformal, nonuniform=False):
    def build():
        walls = Boundary(lo="pec", hi="pec", conformal=conformal)
        sim = Simulation(freq_max=10e9, domain=(0.012, 0.0125, 0.012),
                         dx=1e-3, cpml_layers=0,
                         boundary=BoundarySpec(x="pec", y=walls, z="pec"),
                         **({"dz_profile": np.full(12, 1e-3)}
                            if nonuniform else {}))
        sim.add(Cylinder((0.0085, 0.0085, 0.006), 0.0017, 0.008, axis="z"),
                material="pec")
        sim.add_source((0.004, 0.009, 0.006), "ez", amplitude_kind="field")
        sim.add_probe((0.008, 0.004, 0.006), "ez")
        return sim
    return _quiet(build)


def _optimize(sim, **kw):
    import jax.numpy as jnp
    from rfx.optimize import DesignRegion, optimize
    region = DesignRegion(corner_lo=(0.002, 0.002, 0.004),
                          corner_hi=(0.005, 0.005, 0.007), eps_range=(1.0, 4.0))
    return optimize(sim, region, lambda r: jnp.sum(r.time_series ** 2),
                    n_iters=1, n_steps=8, verbose=False, **kw)


CONFORMAL_PATHS = {
    # path: (call, entry, lane label)
    "forward-uniform": (
        lambda: _conformal_cavity(True).forward(
            n_steps=8, checkpoint=False, skip_preflight=True),
        "Simulation.forward()", "uniform forward"),
    "forward-nonuniform": (
        lambda: _conformal_cavity(True, nonuniform=True).forward(
            n_steps=8, checkpoint=False, skip_preflight=True),
        "Simulation.forward()", "non-uniform forward"),
    "forward-distributed-nonuniform": (
        lambda: _conformal_cavity(True, nonuniform=True).forward(
            n_steps=8, checkpoint=False, skip_preflight=True,
            distributed=True, devices=_devices()),
        "Simulation.forward()", "distributed non-uniform forward"),
    "optimize": (
        lambda: _optimize(_conformal_cavity(True), skip_preflight=True),
        "Simulation.forward()", "uniform forward"),
    "waveguide-nonuniform": (
        lambda: _wr90_sim(debye=False, conformal=True,
                          dz=np.full(4, 0.00254)).compute_waveguide_s_matrix(
            n_steps=20, normalize=True),
        "compute_waveguide_s_matrix()", "non-uniform waveguide S-matrix"),
}


@pytest.mark.parametrize("path", list(CONFORMAL_PATHS))
def test_paths_without_dey_mittra_refuse_a_conformal_boundary(path):
    call, entry, label = CONFORMAL_PATHS[path]
    with pytest.raises(NotImplementedError,
                       match=_refusal(label, "conformal_pec")) as info:
        call()
    msg = str(info.value)
    assert msg.startswith(f"{entry} refuses"), msg
    assert "drop Boundary(conformal=True)" in msg, msg


@pytest.mark.parametrize("nonuniform", [False, True],
                         ids=["uniform", "non-uniform"])
def test_forward_without_a_conformal_boundary_still_runs(nonuniform):
    res = _quiet(lambda: _conformal_cavity(False, nonuniform).forward(
        n_steps=8, checkpoint=False, skip_preflight=True))
    assert np.all(np.isfinite(np.asarray(res.time_series)))


# --------------------------------------------------------------------
# Controls: each argument still runs on a lane that implements it.
# --------------------------------------------------------------------

def _uniform_sim(boundary="pec"):
    def build():
        sim = Simulation(freq_max=10e9, domain=(0.012, 0.012, 0.012),
                         dx=1e-3, boundary=boundary,
                         cpml_layers=4 if boundary == "cpml" else 0)
        sim.add(Box((0.003, 0.003, 0.003), (0.006, 0.006, 0.006)),
                material="fr4")
        sim.add(Cylinder((0.009, 0.009, 0.006), 0.0015, 0.008, axis="z"),
                material="pec")
        sim.add_source((0.004, 0.009, 0.006), "ez")
        sim.add_probe((0.008, 0.004, 0.006), "ez")
        return sim
    return _quiet(build)


@pytest.mark.parametrize("kw", list(ASKS))
def test_uniform_lane_still_runs_each_argument(kw):
    boundary = "cpml" if kw == "until_decay" else "pec"
    sim = _uniform_sim(boundary)
    extra = ({"decay_check_interval": 10, "decay_min_steps": 10,
              "decay_max_steps": 40} if kw == "until_decay" else {})
    res = _quiet(lambda: sim.run(n_steps=8, skip_preflight=True,
                                 **{kw: ASKS[kw]}, **extra))
    assert np.all(np.isfinite(np.asarray(res.time_series)))
    if kw == "snapshot":
        assert res.snapshots is not None and "ez" in res.snapshots


def test_uniform_lane_still_runs_a_separate_s_parameter_record():
    """Two lumped ports: the S-matrix is driven in its own runs, so a
    different s_param_n_steps is honoured there."""
    def build():
        sim = Simulation(freq_max=5e9, domain=(0.01, 0.01, 0.01),
                         boundary="pec")
        sim.add_port((0.002, 0.005, 0.005), "ez")
        sim.add_port((0.008, 0.005, 0.005), "ez")
        return sim
    sim = _quiet(build)
    res = _quiet(lambda: sim.run(n_steps=10, compute_s_params=True,
                                 s_param_freqs=np.array([2e9, 3e9, 4e9]),
                                 s_param_n_steps=12, skip_preflight=True))
    assert res.s_params is not None and res.s_params.shape == (2, 2, 3)


def test_nu_path_until_decay_runs_on_absorbing_boundary():
    """#383: until_decay is honoured on a CPML NU sim, with no refusal."""
    dz = np.full(8, 1e-3)
    sim = _quiet(lambda: Simulation(
        freq_max=10e9, domain=(4e-3, 4e-3, 8e-3), dx=1e-3, dz_profile=dz,
        boundary="cpml", cpml_layers=4))
    sim.add_source((2e-3, 2e-3, 2e-3), "ez")
    sim.add_probe((2e-3, 2e-3, 5e-3), "ez")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(
            until_decay=1e-3,
            decay_check_interval=20,
            decay_min_steps=20,
            decay_max_steps=200,
        )
    msgs = [str(w.message) for w in caught]
    # Post-#392 review: this fixture's planned 200-step table (0.38 ns)
    # ends inside the default source pulse (completion t0+3*tau ~
    # 0.48 ns), so its high rel_DC (measured 2.2e-2) is a pure
    # truncation artifact. The #388 advisory must classify it as
    # truncation-dominated (longer decay_max_steps / fixed n_steps) —
    # NOT recommend a higher cutoff, which lengthens the onset and
    # makes truncation worse.
    dc_msgs = [m for m in msgs if "issue #388" in m]
    assert dc_msgs, "the #388 advisory should fire on this fixture"
    assert "truncation-dominated" in dc_msgs[0], (
        f"table-truncated rel_DC must take the truncation branch, "
        f"got: {dc_msgs[0]}"
    )
    assert "higher GaussianPulse cutoff" not in dc_msgs[0]


@pytest.mark.parametrize("kw", ["checkpoint", "subpixel_smoothing"])
def test_nu_lane_still_runs_what_it_implements(kw):
    """checkpoint and subpixel_smoothing are carried by the NU runner."""
    res = _quiet(lambda: _nu_sim().run(n_steps=16, **{kw: True}))
    assert np.all(np.isfinite(np.asarray(res.time_series)))


@pytest.mark.parametrize("lane", ["adi", "subgridded"])
def test_report_every_stays_a_warning(lane):
    """Progress lines change no result: the lane says so and runs on."""
    build, label, _ = LANES[lane]
    sim = build()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.run(n_steps=4, report_every=2, skip_preflight=True)
    msgs = [str(w.message) for w in caught if "report_every" in str(w.message)]
    assert any(label in m and "result is unchanged" in m for m in msgs), msgs
    assert res is not None


# --------------------------------------------------------------------
# Helper unit tests.
# --------------------------------------------------------------------

def test_refusal_helper_passes_values_that_ask_for_nothing():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Simulation._refuse_unsupported_run_kwargs("dummy", {
            "subpixel_smoothing": False,
            "checkpoint": False,
            "snapshot": None,
            "until_decay": None,
            "conformal_pec": False,
            "compute_s_params": None,
            "s_param_freqs": None,
            "s_param_n_steps": None,
        }, instead="go elsewhere")
    assert [str(w.message) for w in caught] == []


def test_refusal_helper_warns_only_for_report_every():
    with pytest.warns(UserWarning, match=r"report_every=5 .*dummy-path"):
        Simulation._refuse_unsupported_run_kwargs(
            "dummy-path", {"report_every": 5}, instead="go elsewhere")
    with pytest.raises(NotImplementedError, match="dummy-path"):
        Simulation._refuse_unsupported_run_kwargs(
            "dummy-path", {"report_every": 5, "checkpoint": True},
            instead="go elsewhere")


# --------------------------------------------------------------------
# P2.7 preflight: PMC/PEC face + CPML on the same axis.
# --------------------------------------------------------------------

def test_preflight_silent_on_pmc_plus_cpml_uniform_path():
    """Uniform mesh with PMC+CPML composition: after the per-face
    grid allocation (2026-04) the reflector wall aligns with the user domain
    edge (pad=0 on the PMC side), so P2.7 does NOT fire."""
    spec = BoundarySpec(
        x="periodic",
        y=Boundary(lo="cpml", hi="pmc"),
        z="periodic",
    )
    sim = Simulation(
        freq_max=10e9,
        domain=(4e-3, 8e-3, 4e-3),
        dx=1e-3,
        boundary=spec,
        cpml_layers=4,
    )
    sim.add_source((2e-3, 4e-3, 2e-3), "ex")
    sim.add_probe((2e-3, 6e-3, 2e-3), "ex")
    issues = sim.preflight()
    assert not any("P2.7" in s for s in issues), (
        f"P2.7 must not fire on uniform path (per-face padding 2026-04 "
        f"closes the gap). Got: {issues}"
    )


def test_preflight_silent_on_pmc_plus_cpml_nu_path():
    """NU path: per-face allocation (2026-04) extended to NonUniformGrid,
    so P2.7 is silent on NU too (gap closed on both paths)."""
    spec = BoundarySpec(
        x="periodic",
        y=Boundary(lo="cpml", hi="pmc"),
        z="periodic",
    )
    sim = Simulation(
        freq_max=10e9,
        domain=(4e-3, 8e-3, 4e-3),
        dx=1e-3,
        boundary=spec,
        cpml_layers=4,
    )
    sim._dz_profile = np.full(4, 1e-3)  # force NU path
    sim.add_source((2e-3, 4e-3, 2e-3), "ex")
    sim.add_probe((2e-3, 6e-3, 2e-3), "ex")
    issues = sim.preflight()
    assert not any("P2.7" in s for s in issues), (
        f"P2.7 must not fire on NU path after per-face fix. Got: {issues}"
    )


def test_nu_grid_asymmetric_pmc_allocation():
    """NonUniformGrid.pad_y_lo must be 0 when y_lo is a PMC face."""
    from rfx.nonuniform import make_nonuniform_grid
    dz_profile = np.full(8, 1e-3)
    g = make_nonuniform_grid(
        domain_xy=(8e-3, 8e-3),
        dz_profile=dz_profile,
        dx=1e-3,
        cpml_layers=4,
        pmc_faces={"y_lo"},
    )
    assert g.pad_y_lo == 0, f"expected pad_y_lo=0, got {g.pad_y_lo}"
    assert g.pad_y_hi == 4
    assert g.pad_x_lo == 4 and g.pad_x_hi == 4
    # ny = interior + pad_y_hi (no lo padding) = 8 + 4 = 12.
    # 8 interior cells + 0 lo pad (PMC face) + 4 hi pad = 12 cells,
    # bounded by 13 nodes (#562).
    assert g.ny == 13, f"expected ny=13, got {g.ny}"
    # axis_pads property carries the leading (lo) pad per axis.
    assert g.axis_pads == (4, 0, 4)


def test_preflight_silent_on_cpml_without_reflector():
    """Plain all-CPML sim: no P2.7 warning."""
    sim = Simulation(
        freq_max=10e9,
        domain=(4e-3, 4e-3, 4e-3),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=4,
    )
    sim.add_source((2e-3, 2e-3, 2e-3), "ex")
    sim.add_probe((3e-3, 2e-3, 2e-3), "ex")
    issues = sim.preflight()
    assert not any("P2.7" in s for s in issues), (
        f"P2.7 must not fire without a PMC/PEC reflector face. Got: {issues}"
    )


def test_preflight_silent_on_closed_cavity_with_pmc():
    """cpml_layers=0 closed cavity + PMC face: no P2.7 warning (the
    architectural gap only applies when CPML is allocated)."""
    spec = BoundarySpec(
        x="periodic",
        y=Boundary(lo="pec", hi="pmc"),
        z="periodic",
    )
    sim = Simulation(
        freq_max=10e9,
        domain=(2e-3, 8e-3, 2e-3),
        dx=1e-3,
        boundary=spec,
        cpml_layers=0,
    )
    sim.add_source((1e-3, 4e-3, 1e-3), "ex")
    sim.add_probe((1e-3, 6e-3, 1e-3), "ex")
    issues = sim.preflight()
    assert not any("P2.7" in s for s in issues), (
        f"P2.7 must not fire when cpml_layers=0 (no allocated padding). "
        f"Got: {issues}"
    )


def test_uniform_grid_asymmetric_pmc_allocation():
    """Grid.pad_y_lo must be 0 when y_lo is a PMC face (per-face
    allocation). Leading axis_pads tuple also reflects this."""
    from rfx.grid import Grid
    g = Grid(
        freq_max=10e9, domain=(10e-3, 10e-3, 10e-3), dx=1e-3,
        cpml_layers=8, pmc_faces={"y_lo"},
    )
    assert g.pad_y_lo == 0, f"expected pad_y_lo=0 for PMC face, got {g.pad_y_lo}"
    assert g.pad_y_hi == 8
    assert g.pad_x_lo == 8 and g.pad_x_hi == 8
    assert g.axis_pads == (8, 0, 8), (
        f"axis_pads must carry lo pads as leading offset, got {g.axis_pads}"
    )
    # position_to_index: user y=0 maps to array index 0 (the PMC wall).
    assert g.position_to_index((0.0, 0.0, 0.0))[1] == 0
    # Shape: ny = interior + pad_y_hi (no lo padding).
    assert g.shape[1] == int(np.ceil(10e-3 / 1e-3)) + 1 + 8
