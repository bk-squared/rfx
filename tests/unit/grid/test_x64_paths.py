"""Public API x64 carry census: small models and explicit lane refusals."""
import re
import pytest

import jax
import jax.numpy as jnp
import numpy as np
from rfx import Simulation
from rfx.geometry.csg import Box
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import lorentz_pole, drude_pole
from rfx.measurement.incident import measure_normal_incident_spectrum
from rfx.sources.tfsf import init_tfsf
from rfx.ringdown import RingdownSpec

PATHS = ('uniform', 'graded', 'multi_uniform', 'multi_graded', 'subgridded')
FEATURES = ('run', 'forward', 'grad', 'rlc', 'debye', 'lorentz', 'drude', 'ntff', 'flux', 'waveguide', 'normal_open', 'closed_box', 'oblique_open', 'oblique_bloch', 'ringdown')


def exercise(path, feature):
    graded = 'graded' in path and path != 'subgridded'
    multi = path.startswith('multi')
    tfsf = feature in ('normal_open', 'closed_box', 'oblique_open', 'oblique_bloch')
    kw = {'dz_profile': np.array([.001] * 4 + [.0008] * 5 + [.0012] * 5)} if graded else {}
    # 14 mm graded z; all allocated axes <= 23 nodes, fine slab <= 23.
    domain = (.010, .010, .014 if graded else .010)
    boundary = 'cpml' if tfsf or feature in ('ntff', 'waveguide') else 'pec'
    if feature == 'waveguide':
        boundary = dict(x='cpml', y='pec', z='pec')
    sim = Simulation(freq_max=30e9, domain=domain, dx=.001,
                     boundary=boundary, cpml_layers=4, **kw)
    if path == 'subgridded':
        sim.add_refinement(z_range=(.003, .007), ratio=2, validation='research')
    if tfsf:
        sim.add_tfsf_source(f0=20e9, bandwidth=.5, margin=2,
                            closed_box=feature == 'closed_box',
                            angle_deg=30. if 'oblique' in feature else 0.,
                            method='methodB' if feature == 'oblique_open' else 'bloch')
    elif feature == 'waveguide':
        sim.add_waveguide_port(.003, y_range=(0., domain[1]), z_range=(0., domain[2]),
                              f0=20e9, freqs=np.array([20e9]), probe_offset=2, ref_offset=1)
    elif feature == 'ringdown':
        sim.add_port((.004, .004, .001), 'ez', extent=.002,
                     waveform=lambda t: jnp.where(t < 1e-12, 1., 0.))
    else:
        sim.add_source((.004, .005, .005), 'ez', amplitude_kind='field')
    sim.add_probe((.006, .005, .005), 'ez')
    if feature == 'rlc':
        sim.add_lumped_rlc(position=(.005, .005, .005), component='ez',
                           R=50., L=1e-8, C=1e-12, topology='series')
    if feature in ('debye', 'lorentz', 'drude'):
        pole = {'debye': dict(debye_poles=[DebyePole(2., 1e-10)]),
                'lorentz': dict(lorentz_poles=[lorentz_pole(2., 2*np.pi*20e9, 1e9)]),
                'drude': dict(lorentz_poles=[drude_pole(2*np.pi*20e9, 1e9)])}[feature]
        sim.add_material('m', eps_r=2., **pole)
        sim.add(Box((.003, .003, .003), (.007, .007, .007)), material='m')
    if feature == 'ntff':
        sim.add_ntff_box((.002,)*3, (.008,)*3, freqs=np.array([20e9]))
    if feature == 'flux':
        sim.add_flux_monitor(axis='x', coordinate=.006, freqs=np.array([20e9]))
    solve = dict(n_steps=8 if feature == 'ringdown' else 4, skip_preflight=True)
    if multi:
        solve['devices'] = jax.devices('cpu')[:2]
    if feature in ('forward', 'grad', 'ringdown'):
        if multi:
            solve['distributed'] = True
        if feature == 'ringdown':
            solve.update(ringdown=RingdownSpec(), port_s11_freqs=np.array([20e9]))
        if feature == 'grad':
            shape = (11, 11, 15 if graded else 11)
            def objective(eps):
                result = sim.forward(**solve, checkpoint=True,
                                     eps_override=jnp.ones(shape) * eps)
                return jnp.sum(result.time_series ** 2)
            return jax.grad(objective)(jnp.asarray(2.))
        return sim.forward(**solve).time_series
    result = sim.run(**solve)
    if feature == 'closed_box':
        grid = sim._build_grid()
        cfg, state = init_tfsf(grid.nx, grid.dx, grid.dt, ny=grid.ny, nz=grid.nz,
                              cpml_layers=4, tfsf_margin=2, f0=20e9,
                              bandwidth=.5, polarization='ez', direction='+x',
                              closed_box=True, n_steps=4)
        return measure_normal_incident_spectrum(cfg, state, 4, np.array([20e9]), grid.dt)
    return result.time_series


# Product refusals observed on main; expected values do not use solver helpers.
REFUSALS = {
    ("graded", "closed_box"): "CustomWaveform and closed_box TFSF require a uniform, 3D second-order Yee run/forward with no subgrids",
    ("graded", "oblique_open"): "TFSF oblique incidence (angle_deg != 0) is not yet supported on nonuniform z mesh — the 2D auxiliary grid is uniform-only. Use angle_deg=0 or run on the uniform lane.",
    ("graded", "oblique_bloch"): "TFSF oblique incidence (angle_deg != 0) is not yet supported on nonuniform z mesh — the 2D auxiliary grid is uniform-only. Use angle_deg=0 or run on the uniform lane.",
    ("multi_uniform", "grad"): "Simulation.forward() refuses 2 argument(s) the uniform distributed forward lane does not implement; running on without them would not be the run that was asked for:",
    ("multi_uniform", "rlc"): "add_lumped_rlc() element(s) (1): not supported on the distributed multi-device run() path; the lane never applies them and would solve the structure as if the elements were absent, with no warning (rfx.runners.distributed_v2.run_distributed / rfx.runners.distributed.run_distributed have no lumped-element update). Remove the elements, or omit devices=... (use a single-device run() instead).",
    ("multi_uniform", "waveguide"): "waveguide ports (add_waveguide_port) are not supported on run_distributed; use one device",
    ("multi_uniform", "normal_open"): "open-slab normal-incidence TFSF (periodic y/z is not on slabs) is not supported on run_distributed; use one device",
    ("multi_uniform", "oblique_open"): "TFSF Method B oblique (is_tfsf_methodB) is not supported on run_distributed; use one device",
    ("multi_uniform", "oblique_bloch"): "TFSF 2-D Bloch line (is_tfsf_2d) is not supported on run_distributed; use one device",
    ("multi_uniform", "ringdown"): "forward(ringdown=...) is not supported with distributed=True: the port-channel probes and their check against the port accumulators are single-device only.",
    ("multi_graded", "rlc"): "add_lumped_rlc() element(s) (1): not supported on the distributed multi-device run() path; the lane never applies them and would solve the structure as if the elements were absent, with no warning (rfx.runners.distributed_v2.run_distributed / rfx.runners.distributed.run_distributed have no lumped-element update). Remove the elements, or omit devices=... (use a single-device run() instead).",
    ("multi_graded", "waveguide"): "waveguide ports (add_waveguide_port) are not supported on run_distributed_nu; use one device",
    ("multi_graded", "normal_open"): "TFSF plane-wave sources (add_tfsf_source) are not supported on the graded multi-device lane (run_distributed_nu); use one device",
    ("multi_graded", "closed_box"): "CustomWaveform and closed_box TFSF require a uniform, 3D second-order Yee run/forward with no subgrids",
    ("multi_graded", "oblique_open"): "TFSF plane-wave sources (add_tfsf_source) are not supported on the graded multi-device lane (run_distributed_nu); use one device",
    ("multi_graded", "oblique_bloch"): "TFSF plane-wave sources (add_tfsf_source) are not supported on the graded multi-device lane (run_distributed_nu); use one device",
    ("multi_graded", "ringdown"): "forward(ringdown=...) is not supported with distributed=True: the port-channel probes and their check against the port accumulators are single-device only.",
    ("subgridded", "forward"): "add_refinement(z_range=(3, 7) mm, ratio=2) is refused on forward()/optimize()/topology_optimize()/compute_lumped_wire_s_matrix_via_scan(): there is no subgridded lane behind forward()/optimize()/topology_optimize()/compute_lumped_wire_s_matrix_via_scan(), so the refinement would be ignored and the unrefined grid solved, bit-identical to the same model without it. add_refinement takes effect in run(), on a uniform mesh. Remove the refinement to use forward()/optimize()/topology_optimize()/compute_lumped_wire_s_matrix_via_scan(). forward(), optimize(), compute_waveguide_s_matrix() and vmap_material_sweep() also take a non-uniform mesh: a dz_profile whose cells between z = 3 and 7 mm are as fine as the refinement asked for (the coarse cell there divided by 2) gives them that resolution.",
    ("subgridded", "grad"): "add_refinement(z_range=(3, 7) mm, ratio=2) is refused on forward()/optimize()/topology_optimize()/compute_lumped_wire_s_matrix_via_scan(): there is no subgridded lane behind forward()/optimize()/topology_optimize()/compute_lumped_wire_s_matrix_via_scan(), so the refinement would be ignored and the unrefined grid solved, bit-identical to the same model without it. add_refinement takes effect in run(), on a uniform mesh. Remove the refinement to use forward()/optimize()/topology_optimize()/compute_lumped_wire_s_matrix_via_scan(). forward(), optimize(), compute_waveguide_s_matrix() and vmap_material_sweep() also take a non-uniform mesh: a dz_profile whose cells between z = 3 and 7 mm are as fine as the refinement asked for (the coarse cell there divided by 2) gives them that resolution.",
    ("subgridded", "rlc"): "The subgridded run() lane would solve this Simulation as if these inputs were not declared, so it is refused before the first time step:",
    ("subgridded", "debye"): "The subgridded run() lane would solve this Simulation as if these inputs were not declared, so it is refused before the first time step:",
    ("subgridded", "lorentz"): "The subgridded run() lane would solve this Simulation as if these inputs were not declared, so it is refused before the first time step:",
    ("subgridded", "drude"): "The subgridded run() lane would solve this Simulation as if these inputs were not declared, so it is refused before the first time step:",
    ("subgridded", "ntff"): "The subgridded run() lane would solve this Simulation as if these inputs were not declared, so it is refused before the first time step:",
    ("subgridded", "flux"): "The subgridded run() lane would solve this Simulation as if this input was not declared, so it is refused before the first time step:",
    ("subgridded", "waveguide"): "The subgridded run() lane would solve this Simulation as if these inputs were not declared, so it is refused before the first time step:",
    ("subgridded", "normal_open"): "The subgridded run() lane would solve this Simulation as if these inputs were not declared, so it is refused before the first time step:",
    ("subgridded", "closed_box"): "CustomWaveform and closed_box TFSF require a uniform, 3D second-order Yee run/forward with no subgrids",
    ("subgridded", "oblique_open"): "The subgridded run() lane would solve this Simulation as if these inputs were not declared, so it is refused before the first time step:",
    ("subgridded", "oblique_bloch"): "The subgridded run() lane would solve this Simulation as if these inputs were not declared, so it is refused before the first time step:",
    ("subgridded", "ringdown"): "forward(ringdown=...) is not supported on the subgridded lane.",
}

# >5 s in either Mac census; include explicitly with -o addopts= when validating.
SLOW_ROWS = {('multi_graded', 'run'), ('multi_graded', 'grad'),
             ('multi_graded', 'ntff'), ('subgridded', 'run')}
ROWS = [pytest.param(path, feature, id=f'{path}-{feature}',
                     marks=[pytest.mark.slow] if (path, feature) in SLOW_ROWS else [])
        for path in PATHS for feature in FEATURES]


@pytest.mark.parametrize('path,feature', ROWS)
def test_x64_path(path, feature):
    if path.startswith('multi') and len(jax.devices('cpu')) < 2:
        pytest.skip('requires two host CPU devices')
    with jax.enable_x64(True):
        refusal = REFUSALS.get((path, feature))
        if refusal is not None:
            with pytest.raises((ValueError, NotImplementedError), match=re.escape(refusal)):
                exercise(path, feature)
        else:
            value = np.asarray(exercise(path, feature))
            assert value.size > 0
            assert np.isfinite(value).all()
