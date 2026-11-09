"""Every unsupported wall-loss entry refuses the public method by name."""
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.errors import RfxRefusal
from rfx.model.conductors import realized_conductors
from rfx.model.wall_loss import build_wall_operator


API_ROWS = ["selector-none", "selector-both", "sigma-zero", "sigma-negative", "f0-zero", "f0-negative",
            "shape-not-added", "shape-equal-not-identical", "shape-copper", "shape-pec-like", "face-cpml", "face-invalid", "face-empty"]
LANE_ROWS = ["optimize-uniform", "optimize-graded", "gradient-check", "topology", "material-fit", "vmap-scan",
             "coax-msl", "mixed-msl-junction", "ad-memory", "distributed-v2", "distributed-entry-run",
             "distributed-entry-forward", "distributed-nu-forward", "distributed-dispatch-run", "distributed-dispatch-forward",
             "subgrid-dispatch", "subgrid-run", "waveguide", "multimode-waveguide", "lumped-scan", "eager-port-probe",
             "current-moments", "upml", "kottke", "conformal", "traced-mesh", "solid-cpml-pad",
             "debye-overlap", "lorentz-overlap", "drude-overlap", "shape-sheet"]


def model(graded=False, boundary="pec"):
    sim = Simulation(freq_max=10e9, domain=(.012,)*3, dx=.0015, boundary=boundary,
                     cpml_layers=2, snap="declared", **({"dy_profile": np.full(8, .0015)} if graded else {}))
    return sim


def api_call(row):
    sim = model(boundary="cpml" if row == "face-cpml" else "pec")
    kwargs = dict(sigma_bulk=5.8e5, f0=8e9, faces=("y_lo", "y_hi"))
    if row == "selector-none":
        kwargs.pop("faces")
    elif row == "selector-both":
        kwargs["shape"] = Box((.003,)*3, (.006,)*3)
    elif row.startswith("sigma-"):
        kwargs["sigma_bulk"] = 0 if row.endswith("zero") else -1
    elif row.startswith("f0-"):
        kwargs["f0"] = 0 if row.endswith("zero") else -1
    elif row.startswith("shape-"):
        kwargs.pop("faces")
        shape = Box((.003,)*3, (.006,)*3)
        kwargs["shape"] = shape
        if row == "shape-equal-not-identical":
            sim.add(Box((.003,)*3, (.006,)*3), material="pec")
        elif row == "shape-copper":
            sim.add(shape, material="copper")
        elif row == "shape-pec-like":
            sim.add_material("metal", sigma=1e10)
            sim.add(shape, material="metal")
    elif row == "face-invalid":
        kwargs["faces"] = ("y_top",)
    elif row == "face-empty":
        kwargs["faces"] = ()
    return lambda: sim.add_wall_loss(**kwargs)


def lane_call(row, monkeypatch):
    sim = model(graded=row == "optimize-graded")
    sim.add_wall_loss(sigma_bulk=5.8e5, f0=8e9, faces=("y_lo", "y_hi"))
    if row.startswith("optimize-"):
        from rfx.optimize import optimize, DesignRegion
        return lambda: optimize(sim, DesignRegion((.003,)*3, (.006,)*3), lambda result: 0., skip_preflight=True)
    if row == "gradient-check":
        from rfx.optimize import gradient_check
        return lambda: gradient_check(sim, np.ones(1), lambda result: 0.)
    if row == "topology":
        from rfx.topology import topology_optimize
        return lambda: topology_optimize(sim, None, lambda result: 0., skip_preflight=True)
    if row == "material-fit":
        from rfx.differentiable_material_fit import differentiable_material_fit
        return lambda: differentiable_material_fit(lambda *args: sim, np.ones((3, 1)), np.array([7e9, 8e9, 9e9]), n_iterations=1)
    if row == "vmap-scan":
        from rfx.vmap_sweep import _build_full_scan_fn
        return lambda: _build_full_scan_fn(sim, None, None, 1)
    if row == "coax-msl":
        return lambda: sim.compute_coax_msl_transition(junction_x=.006, n_steps=1, skip_preflight=True)
    if row == "mixed-msl-junction":
        from rfx.sparams import _mixed_stages
        monkeypatch.setattr(_mixed_stages, "_validate_mixed_family", lambda *args: ([], False))
        return lambda: sim.compute_mixed_s_matrix(n_steps=1, skip_preflight=True)
    if row == "ad-memory":
        return sim._ad_memory_static_accounting
    if row == "distributed-v2":
        from rfx.runners.distributed_v2 import run_distributed
        return lambda: run_distributed(sim, n_steps=1)
    if row.startswith("distributed-entry-"):
        from rfx.runners._distributed_entry import execute_distributed
        return lambda: execute_distributed(sim, plan=SimpleNamespace(lane="run_distributed"), _solve_assembly=None,
            _diagnostics=(), n_steps=1, num_periods=1, devices=None, _forward_port_s11=row.endswith("forward"))
    if row == "distributed-nu-forward":
        from rfx.runners._distributed_nu_execute import dispatch_forward
        return lambda: dispatch_forward(sim, plan=None, _solve_assembly=SimpleNamespace(diagnostics=()), _call_args={})
    if row.startswith("distributed-dispatch-"):
        mode = row.rsplit("-", 1)[-1]
        return lambda: sim._dispatch_plan(mode=mode, n_steps=1, num_periods=1, distributed=mode=="forward", devices=[] if mode=="run" else None)
    if row.startswith("subgrid-"):
        sim.add_refinement(((.003, .009),), ratio=2, validation="research")
        if row == "subgrid-run":
            # Reach the runner's second guard after the dispatch guard is bypassed.
            monkeypatch.setattr(sim, "_dispatch_plan", lambda **kwargs: SimpleNamespace(lane="run_subgridded", n_steps=1))
        return lambda: sim.run(n_steps=1, compute_s_params=False, skip_preflight=True)
    if row in ("waveguide", "multimode-waveguide"):
        return lambda: sim.compute_waveguide_s_matrix(n_steps=1, skip_preflight=True)
    if row in ("lumped-scan", "eager-port-probe"):
        from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
        return lambda: compute_lumped_wire_s_matrix_via_scan(sim, np.array([8e9]), n_steps=1)
    if row == "current-moments":
        from rfx.current_moments import monitor_for_simulation
        sim.add_current_moment_monitor((.003,)*3, (.009,)*3, .003, np.array([8e9]))
        return lambda: monitor_for_simulation(sim, None)
    if row in ("upml", "kottke", "conformal"):
        if row == "upml":
            sim._boundary = "upml"
        return lambda: sim.run(n_steps=1, compute_s_params=False, skip_preflight=True,
                               subpixel_smoothing=row=="kottke", conformal_pec=True if row=="conformal" else None)
    if row == "traced-mesh":
        def traced(profile):
            sim._dy_profile = profile
            sim.forward(n_steps=1, skip_preflight=True)
            return profile.sum()
        return lambda: jax.make_jaxpr(traced)(np.full(8, .0015))
    if row in ("solid-cpml-pad", "shape-sheet"):
        sim = model(boundary="cpml")
        solid = Box((-.003, .003, .003), (.003, .009, .009)) if row == "solid-cpml-pad" else Box((.003, .003, .006), (.009, .009, .006))
        sim.add(solid, material="pec")
        sim.add_wall_loss(shape=solid, sigma_bulk=5.8e5, f0=8e9)
    elif row.endswith("-overlap"):
        from rfx.materials.debye import DebyePole
        from rfx.materials.lorentz import LorentzPole
        poles = {"debye_poles": [DebyePole(delta_eps=1., tau=1e-11)]} if row == "debye-overlap" else {
            "lorentz_poles": [LorentzPole(omega_0=0. if row == "drude-overlap" else 2*np.pi*8e9, delta=1e9, kappa=1e20)]}
        sim.add_material("dispersive", eps_r=2., **poles)
        sim.add(Box((.003, 0., .003), (.009, .003, .009)), material="dispersive")
    else:
        raise AssertionError(row)
    def assemble():
        grid = sim._build_grid()
        root = realized_conductors(sim, grid)
        build_wall_operator(sim, root, None, lane=row)
    return assemble


@pytest.mark.parametrize("site", API_ROWS + LANE_ROWS)
def test_wall_loss_refused_by_name(site, monkeypatch):
    call = api_call(site) if site in API_ROWS else lane_call(site, monkeypatch)
    with pytest.raises(RfxRefusal, match="add_wall_loss"):
        call()
