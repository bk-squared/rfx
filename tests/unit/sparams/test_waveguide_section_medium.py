"""Modal ports resolve one lossless homogeneous section medium (#1994)."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.features import waveguide_segment_cells

DX, A, B, L = .002, .024, .012, .12


def guide(graded=False, *, direction="+x", axis="x", **port):
    domain = [A, B]
    domain.insert("xyz".index(axis), L)
    boundary = {a: "cpml" if a == axis else "pec" for a in "xyz"}
    sim = Simulation(freq_max=12e9, domain=tuple(domain), dx=DX,
                     boundary=boundary, cpml_layers=8,
                     **({f"d{axis}_profile": np.full(60, DX)} if graded else {}))
    sim.add_waveguide_port(.02 if direction.startswith("+") else .10,
                          direction=direction[0]+axis, name="launch", f0=9e9,
                          freqs=np.array([8e9, 10e9]), ref_offset=3, probe_offset=10, **port)
    return sim


def execute(sim, route="run", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if route == "compute_waveguide_s_matrix" and sim._uses_nonuniform_mesh:
            kw.setdefault("normalize", "flux")
        return getattr(sim, route)(n_steps=2, skip_preflight=True, **kw)


def fill(sim, lo, hi, material="fill", **props):
    if material != "pec":
        sim.add_material(material, **(props or dict(eps_r=2.2)))
    sim.add(Box((lo, 0, 0), (hi, A, B)), material=material)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("route", ["run", "forward", "compute_waveguide_s_matrix"])
def test_filled_guide_admitted(graded, route):
    sim = guide(graded)
    fill(sim, -.1, L+.1)
    if route == "compute_waveguide_s_matrix":
        sim.add_waveguide_port(.10, direction="-x", name="receive", f0=9e9,
                              freqs=np.array([8e9, 10e9]))
    result = execute(sim, route)
    if route != "compute_waveguide_s_matrix":
        assert result.waveguide_ports["launch"].medium.eps_r == pytest.approx(2.2)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("placement", ["h", "declared", "pec"])
def test_segment_extents(graded, placement):
    sim = guide(graded, **({"reference_plane": .07} if placement == "declared" else {}))
    lo, hi = {"h": (.018, .0199), "declared": (.06, .062), "pec": (.022, .024)}[placement]
    fill(sim, lo, hi, "pec" if placement == "pec" else "fill")
    with pytest.raises(ValueError, match=r"launch.*(eps|PEC).*cell.*rfx#1994"):
        execute(sim)


@pytest.mark.parametrize("graded", [False, True])
def test_between_reference_and_probe_admitted(graded):
    """S-matrix extraction reads reference waves only, not the probe plane."""
    sim = guide(graded)
    fill(sim, .032, .034)
    execute(sim)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("medium", ["empty", "vacuum", "empty_poles", "beyond"])
def test_vacuum_and_beyond_admitted(graded, medium, record_property, monkeypatch):
    from rfx.boundaries import features
    sim = guide(graded)
    if medium == "beyond":
        fill(sim, .028, .036)
    elif medium == "vacuum":
        sim.add(Box((0, 0, 0), (L, A, B)), material="vacuum")
    elif medium == "empty_poles":
        fill(sim, 0, L, eps_r=1., debye_poles=[], lorentz_poles=[])
    original = features.resolve_waveguide_section_medium
    measured = []

    def observe(ports, grid, materials, *args, **kwargs):
        cfg = ports[0]
        measured.append(float(np.asarray(materials.eps_r)[cfg.ref_x, cfg.u_lo, cfg.v_lo]))
        return original(ports, grid, materials, *args, **kwargs)

    monkeypatch.setattr(features, "resolve_waveguide_section_medium", observe)
    execute(sim)
    assert measured == [1.0]
    record_property("last_segment_eps", measured)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("operand", ["eps_override", "sigma_override"])
def test_override_refused(graded, operand):
    sim = guide(graded)
    grid = sim._build_realized_grid()
    vacuum = 1. if operand == "eps_override" else 0.
    value = jnp.full(grid.shape, vacuum).at[grid.index_of("x", .022), :, :].set(2.2)
    with pytest.raises(ValueError, match=r"launch.*(eps|sigma).*2\.2"):
        execute(sim, "forward", **{operand: value})


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("near", [True, False])
def test_design_box_window(graded, near):
    sim = guide(graded)
    box = ((.022 if near else .06, .004, .004),
           (.024 if near else .064, .006, .006))
    grid = sim._build_realized_grid()
    lo, hi = [tuple(grid.index_of(axis, value) for axis, value in zip("xyz", p)) for p in box]
    eps = jnp.full(tuple(b-a+1 for a, b in zip(lo, hi)), 2.2)
    if graded:
        with pytest.raises(NotImplementedError, match="design box.*waveguide"):
            execute(sim, "forward", design_box=box, design_eps_override=eps)
    elif near:
        with pytest.raises(ValueError, match=r"launch.*design_box update window"):
            execute(sim, "forward", design_box=box, design_eps_override=eps)
    else:
        execute(sim, "forward", design_box=box, design_eps_override=eps)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("direction", ["+x", "-x"])
def test_axes_and_h_index(graded, axis, direction):
    sim = guide(graded, axis=axis, direction=direction)
    grid = sim._build_realized_grid()
    entry = sim._waveguide_ports[0]
    if graded:
        from rfx.runners.nonuniform import _build_waveguide_port_config_nu
        cfg = _build_waveguide_port_config_nu(sim, entry, grid, entry.freqs, 2)
    else:
        cfg = sim._build_waveguide_port_config(entry, grid, entry.freqs, 2)
    bounds = waveguide_segment_cells(cfg, grid)
    a = "xyz".index(axis)
    assert bounds[a] == ((cfg.x_index-1, cfg.ref_x+1) if direction.startswith("+")
                         else (cfg.ref_x, cfg.x_index+1))
    execute(sim)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("profile", ["analytic", "discrete"])
def test_profiles_and_multimode(graded, profile):
    sim = guide(graded, n_modes=2, mode_profile=profile)
    fill(sim, 0, L)
    with pytest.raises(ValueError, match="launch.*eps"):
        execute(sim, "forward")


def test_traced_override_retains_callback():
    sim = guide()
    grid = sim._build_realized_grid()
    # jit executes a non-vacuum traced operand, rather than merely inspecting jaxpr.
    def call(eps):
        return execute(sim, "forward", eps_override=eps).time_series
    with pytest.raises(Exception, match="traced override over the segment"):
        jax.jit(call)(jnp.full(grid.shape, 2.2)).block_until_ready()


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("behind", [False, True])
def test_declared_reference_extends_segment(graded, behind):
    sim = guide(graded, reference_plane=.006 if behind else .07)
    fill(sim, .01 if behind else .06, .012 if behind else .062)
    with pytest.raises(ValueError, match="launch.*eps"):
        execute(sim)


@pytest.mark.parametrize("graded", [False, True])
def test_declared_probe_does_not_extend_segment(graded):
    sim = guide(graded, probe_plane=.07)
    fill(sim, .06, .062)
    execute(sim)
    sim.add_waveguide_port(.10, direction="-x", f0=9e9, freqs=np.array([8e9, 10e9]))
    with pytest.raises(ValueError, match="does not use per-port probe_plane"):
        execute(sim, "compute_waveguide_s_matrix")


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("behind", [False, True])
def test_declared_reference_excludes_its_cell_centre(graded, behind):
    from rfx.geometry.rasterize_grid import _axis_node_positions
    sim = guide(graded)
    grid = sim._build_realized_grid()
    entry = sim._waveguide_ports[0]
    if graded:
        from rfx.runners.nonuniform import _build_waveguide_port_config_nu
        cfg = _build_waveguide_port_config_nu(sim, entry, grid, entry.freqs, 2)
    else:
        cfg = sim._build_waveguide_port_config(entry, grid, entry.freqs, 2)
    nodes = np.asarray(_axis_node_positions(grid.cells("x"), grid.pad_x_lo))
    index = cfg.x_index - 4 if behind else cfg.ref_x + 4
    plane = nodes[index] + grid.cells("x")[index] / 2
    cfg = cfg._replace(segment_planes=(plane,))
    lo, hi = waveguide_segment_cells(cfg, grid)[0]
    assert (lo if behind else hi) == (index+1 if behind else index)


@pytest.mark.parametrize("graded", [False, True])
def test_cpml_material_sigma_only(graded):
    from dataclasses import replace
    sim = guide(graded)
    sim._waveguide_ports[0] = replace(sim._waveguide_ports[0], x_position=.002)
    execute(sim)


@pytest.mark.parametrize("graded", [False, True])
def test_isolated_h_operand(graded):
    """Direct setup material isolates H from the assembler's E-edge averaging."""
    from rfx.boundaries.features import resolve_waveguide_section_medium
    from rfx.core.yee import MaterialArrays
    sim = guide(graded)
    grid = sim._build_realized_grid()
    entry = sim._waveguide_ports[0]
    if graded:
        from rfx.runners.nonuniform import _build_waveguide_port_config_nu
        cfg = _build_waveguide_port_config_nu(sim, entry, grid, entry.freqs, 2)
    else:
        cfg = sim._build_waveguide_port_config(entry, grid, entry.freqs, 2)
    eps = jnp.ones(grid.shape).at[cfg.x_index-1, :, :].set(2.2)
    materials = MaterialArrays(eps, jnp.zeros(grid.shape), jnp.ones(grid.shape))
    with pytest.raises(ValueError, match="launch.*eps"):
        resolve_waveguide_section_medium([cfg], grid, materials, {}, ())


def test_scalar_low_level_medium():
    from rfx.boundaries.features import resolve_waveguide_section_medium
    from rfx.core.yee import MaterialArrays
    sim = guide()
    grid = sim._build_realized_grid()
    entry = sim._waveguide_ports[0]
    cfg = sim._build_waveguide_port_config(entry, grid, entry.freqs, 2)
    resolved = resolve_waveguide_section_medium([cfg], grid, MaterialArrays(1., 0., 2.), {}, ())
    assert resolved[0].medium.mu_r == 2.


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("kind", ["mu", "sigma", "debye", "lorentz", "chi3"])
def test_other_operands(graded, kind):
    from rfx import DebyePole
    from rfx.materials.lorentz import lorentz_pole
    properties = {
        "mu": dict(mu_r=2.), "sigma": dict(sigma=.1),
        "debye": dict(debye_poles=[DebyePole(delta_eps=1., tau=1e-10)]),
        "lorentz": dict(lorentz_poles=[lorentz_pole(delta_eps=1., omega_0=2*np.pi*5e9, delta=1e10)]),
        "chi3": dict(chi3=1e-3),
    }
    sim = guide(graded)
    fill(sim, .022, .024, eps_r=1., **properties[kind])
    if graded and kind == "chi3":
        with pytest.raises(NotImplementedError, match="Kerr|chi3"):
            execute(sim)
    else:
        with pytest.raises(ValueError, match="launch.*rfx#1994"):
            execute(sim)


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("route", ["run", "forward"])
def test_two_device_routes(graded, route, record_property):
    sim = guide(graded)
    fill(sim, 0, L)
    devices = jax.devices("cpu")
    assert len(devices) >= 2
    kwargs = dict(devices=devices[:2])
    if route == "forward":
        kwargs["distributed"] = True
    result = execute(sim, route, **kwargs)
    assert result.waveguide_ports["launch"].medium.eps_r == pytest.approx(2.2)


@pytest.mark.parametrize("validation", ["production", "research"])
def test_subgridded_refuses(validation, record_property):
    sim = guide()
    sim.add_refinement(z_range=(.004, .008), ratio=2, validation=validation)
    with pytest.raises((ValueError, NotImplementedError), match="waveguide|Waveguide|subgridded") as caught:
        execute(sim)
    record_property("refusal", str(caught.value))

@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('profile', ['analytic', 'discrete'])
def test_filled_profiles_and_smoothing(graded, profile):
    sim = guide(graded, mode_profile=profile)
    fill(sim, -.1, .22)
    result = execute(sim, subpixel_smoothing=True)
    assert result.waveguide_ports['launch'].medium.eps_r == pytest.approx(2.2)


@pytest.mark.parametrize('graded', [False, True])
def test_probe_medium_refuses_s21_only(graded):
    from rfx.measurement.waveguide_s import extract_waveguide_s11, extract_waveguide_s21
    from rfx.preflight.waveguide import resolve_preflight_port_configs
    sim = guide(graded)
    fill(sim, .032, .08)
    with pytest.raises(ValueError, match='probe plane.*port medium'):
        execute(sim)
    grid = sim._build_realized_grid()
    entry = sim._waveguide_ports[0]
    if graded:
        from rfx.runners.nonuniform import _build_waveguide_port_config_nu
        cfg = _build_waveguide_port_config_nu(sim, entry, grid, entry.freqs, 2)
    else:
        cfg = sim._build_waveguide_port_config(entry, grid, entry.freqs, 2)
    cfg = resolve_preflight_port_configs(sim, grid, [cfg])[0]
    assert not cfg.probe_medium_matches
    assert np.all(np.isfinite(extract_waveguide_s11(cfg)))
    with pytest.raises(ValueError, match='probe plane'):
        extract_waveguide_s21(cfg)


@pytest.mark.parametrize('graded', [False, True])
@pytest.mark.parametrize('normalize', [True, 'flux'])
def test_unequal_normalized_media_refused(graded, normalize):
    sim = guide(graded)
    fill(sim, .06, .22)
    sim.add_waveguide_port(.10, direction='-x', name='receive', f0=9e9,
                          freqs=np.array([8e9,10e9]))
    with pytest.raises(ValueError, match='share one medium'):
        execute(sim, 'compute_waveguide_s_matrix', normalize=normalize)


@pytest.mark.parametrize('graded', [False, True])
def test_concrete_fill_under_jit_vmap(graded):
    sim = guide(graded)
    fill(sim, -.1, .22)
    def call(dummy):
        cfg = execute(sim, 'forward').waveguide_ports['launch']
        return cfg.v_ref_t + dummy * 0
    plain = call(0.)
    compiled = jax.jit(call)(0.)
    np.testing.assert_array_equal(compiled, plain)
    np.testing.assert_array_equal(jax.jit(jax.vmap(call))(jnp.zeros(2)), jnp.stack([plain,plain]))


def test_injection_purity():
    from tests._filled_waveguide import measure
    from tests.oracle.test_filled_waveguide_port import dump
    empty, filled = measure('W2_empty'), measure('W2')
    dump(empty); dump(filled)
    assert np.all(np.array(filled['reflection']) <= np.maximum(2*np.array(empty['reflection']), .005)), 'W2 upstream injection leakage'


@pytest.mark.parametrize('profile',['analytic','discrete'])
@pytest.mark.parametrize('mode',['TE','TM'])
@pytest.mark.parametrize('x64',[False,True])
def test_vacuum_bit_identical_to_hand_written_main(profile,mode,x64):
    with jax.enable_x64(x64):
        _assert_vacuum_main(profile,mode,jnp.float64 if x64 else jnp.float32)


def _assert_vacuum_main(profile,mode,dtype):
    from rfx.model.waveguide_modes import PortMedium, _compute_beta, _compute_mode_impedance, _incident_tables
    from tests._waveguide_vacuum_main import main_beta, main_impedance, main_tables
    fc,dt,dx,n,t0,tau,f0 = 3e9, .5e-12, .5e-3, 1024, 200e-12, 40e-12, 8e9
    freqs=jnp.asarray([0.,2e9,3e9,5e9,8e9,11e9])
    modal_dt=dt if profile=='discrete' else 0.
    np.testing.assert_array_equal(_compute_beta(freqs,fc,medium=PortMedium(),dt=modal_dt,dx=dx),
                                  main_beta(freqs,fc,modal_dt,dx))
    np.testing.assert_array_equal(_compute_mode_impedance(freqs,fc,mode,medium=PortMedium(),dt=modal_dt,dx=dx),
                                  main_impedance(freqs,fc,mode,modal_dt,dx))
    actual=_incident_tables(f_c=fc,medium=PortMedium(),dt=dt,dx=dx,dft_total_steps=n,
                            t0=t0,tau=tau,f0=f0,waveform='modulated_gaussian',
                            mode_profile=profile,mode_type=mode,field_dtype=dtype)
    expected=main_tables(fc,dt,dx,n,t0,tau,f0,profile,mode,dtype)
    for got,want in zip(actual,expected):
        np.testing.assert_array_equal(got,want)


@pytest.mark.parametrize('budget',[0,2000])
def test_analytic_cutoff_reading_uses_section_medium(monkeypatch,budget):
    from rfx.preflight import waveguide as preflight
    sim=guide()
    fill(sim,-.1,.22)
    grid=sim._build_realized_grid()
    entry=sim._waveguide_ports[0]
    spans=sim._port_transverse_spans(entry,grid,None)
    monkeypatch.setattr(preflight,'WAVEGUIDE_PREFLIGHT_DISCRETE_CELL_BUDGET',budget)
    row=preflight._waveguide_port_cutoff_reading(sim,entry,grid,spans)
    speed=1/np.sqrt(8.8541878128e-12*1.25663706212e-6*np.float32(2.2))
    for key,width in [('declared_hz',spans['y']['declared']),('realized_hz',spans['y']['rasterized'])]:
        assert row[key] == pytest.approx(speed/(2*width),rel=1e-7)

@pytest.mark.parametrize('graded', [False, True])
def test_preflight_uses_fill_cutoff(graded):
    from rfx.preflight.waveguide import _preflight_entry_config
    sim = guide(graded)
    fill(sim, -.1, .22)
    cfg = _preflight_entry_config(sim, sim._waveguide_ports[0])
    assert cfg.f_cutoff < 4.3e9, 'preflight must use filled cutoff, not vacuum cutoff'
    assert cfg.medium.eps_r == pytest.approx(2.2)


def test_interface_between_source_and_reference_refused():
    sim = guide()
    fill(sim, .022, .026)
    with pytest.raises(ValueError, match='section cells.*eps.*cell'):
        execute(sim)


def test_injection_coefficients_use_medium():
    from rfx.model import PortMedium
    from rfx.core.yee import init_state
    from rfx.stepping.waveguide_source import apply_waveguide_port_h, apply_waveguide_port_e
    sim=guide()
    grid=sim._build_realized_grid()
    entry=sim._waveguide_ports[0]
    cfg=sim._build_waveguide_port_config(entry,grid,entry.freqs,4)
    cfg=cfg._replace(e_inc_table=jnp.ones(4),h_inc_table=jnp.ones(4))
    filled=cfg._replace(medium=PortMedium(4.,2.))
    state=init_state(grid.shape)
    for update,field,factor in ((apply_waveguide_port_h,'hy',.5),(apply_waveguide_port_e,'ez',.25)):
        vacuum=getattr(update(state,cfg,0,grid.dt,grid.dx),field)
        actual=getattr(update(state,filled,0,grid.dt,grid.dx),field)
        assert np.max(np.abs(vacuum))>0
        np.testing.assert_allclose(actual,vacuum*factor,rtol=1e-7,err_msg='filled TF/SF coefficient')


@pytest.mark.parametrize('graded',[False,True])
def test_preflight_concrete_override_is_call_local(graded):
    sim=guide(graded)
    grid=sim._build_realized_grid()
    eps=jnp.full(grid.shape,2.2)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        result=sim.forward(n_steps=2,eps_override=eps)
    assert result.waveguide_ports['launch'].medium.eps_r == pytest.approx(2.2)
    assert not hasattr(sim,'_waveguide_call_overrides')
    from rfx.preflight.waveguide import _preflight_entry_config
    assert _preflight_entry_config(sim,sim._waveguide_ports[0]).medium.eps_r == 1.


def test_per_component_conductivity_refused():
    from rfx.boundaries.features import resolve_waveguide_section_medium
    from rfx.core.yee import MaterialArrays
    sim=guide();grid=sim._build_realized_grid();entry=sim._waveguide_ports[0]
    cfg=sim._build_waveguide_port_config(entry,grid,entry.freqs,2)
    zero=jnp.zeros(grid.shape)
    materials=MaterialArrays(jnp.ones(grid.shape),(zero,zero,zero.at[cfg.ref_x,2,2].set(.1)),jnp.ones(grid.shape))
    with pytest.raises(ValueError,match='conductivity.*cell'):
        resolve_waveguide_section_medium([cfg],grid,materials,{},())



@pytest.mark.parametrize('field',['eps_r','mu_r'])
def test_per_component_permittivity_refused_by_name(field):
    """The shared operand list does not classify per-component leaves; the port names the field instead."""
    from rfx.boundaries.features import resolve_waveguide_section_medium
    from rfx.core.yee import MaterialArrays
    sim=guide();grid=sim._build_realized_grid();entry=sim._waveguide_ports[0]
    cfg=sim._build_waveguide_port_config(entry,grid,entry.freqs,2)
    one=jnp.ones(grid.shape)
    fields=dict(eps_r=one,sigma=jnp.zeros(grid.shape),mu_r=one);fields[field]=(one,one,one)
    with pytest.raises(ValueError,match=f'per-component {field} is unsupported'):
        resolve_waveguide_section_medium([cfg],grid,MaterialArrays(**fields),{},())

@pytest.mark.parametrize('graded,normalize,explicit_reference',[
    (False,True,False),(False,'flux',False),(True,True,False),
    (True,'flux',False),(False,'flux',True)])
def test_filled_reference_runs_resolve_medium(graded,normalize,explicit_reference,monkeypatch):
    from rfx.measurement import waveguide_s
    sim=guide(graded);fill(sim,-.1,.22)
    sim.add_waveguide_port(.10,direction='-x',name='receive',f0=9e9,freqs=np.array([8e9,10e9]))
    seen=[]
    original=waveguide_s._extract_global_waves
    def observe(cfg,*args,**kw):
        seen.append(cfg.medium.eps_r)
        return original(cfg,*args,**kw)
    monkeypatch.setattr(waveguide_s,'_extract_global_waves',observe)
    kwargs={}
    if explicit_reference:
        reference=guide();fill(reference,-.1,.22)
        kwargs['port_reference_sims']=[reference,reference]
    execute(sim,'compute_waveguide_s_matrix',normalize=normalize,**kwargs)
    assert len(seen)>=4
    np.testing.assert_allclose(seen,2.2,rtol=1e-7)


def test_filled_overlap_decomposition_and_probe_guard():
    from types import SimpleNamespace
    from rfx.measurement.waveguide_modal import extract_waveguide_sparams_overlap
    from rfx.preflight.waveguide import _preflight_entry_config
    sim=guide();fill(sim,-.1,.22)
    cfg=_preflight_entry_config(sim,sim._waveguide_ports[0])
    omega=2*np.pi*np.asarray(cfg.freqs,dtype=float)
    c=1/np.sqrt(8.8541878128e-12*1.25663706212e-6*float(np.float32(2.2)))
    kc=2*np.pi*cfg.f_cutoff/c
    beta=2/cfg.dx*np.arcsin(cfg.dx/2*np.sqrt((np.sin(omega*cfg.dt/2)/(c*cfg.dt/2))**2-kc**2))
    z=1.25663706212e-6*cfg.dx*np.sin(omega*cfg.dt/2)/(cfg.dt*np.sin(beta*cfg.dx/2))
    phase=np.exp(-1j*beta*.04)
    acc=SimpleNamespace(p1_ref_dft=jnp.ones_like(cfg.freqs)*2,p2_ref_dft=jnp.asarray(2/z),
                        p1_probe_dft=jnp.asarray(2*phase),p2_probe_dft=jnp.asarray(2*phase/z))
    s11,s21=extract_waveguide_sparams_overlap(acc,cfg,ref_shift=.003)
    np.testing.assert_allclose(s11,0,atol=2e-6)
    np.testing.assert_allclose(s21,np.exp(-1j*beta*.037),atol=2e-6)
    with pytest.raises(ValueError,match='probe plane'):
        extract_waveguide_sparams_overlap(acc,cfg._replace(probe_medium_matches=False))


@pytest.mark.parametrize('extra_anisotropy',[False,True])
def test_conformal_geometry_factor_does_not_hide_material_anisotropy(extra_anisotropy):
    from rfx.boundaries.features import resolve_waveguide_section_medium
    from rfx.model.materials import MaterialArrays
    sim=guide()
    grid=sim._build_realized_grid()
    entry=sim._waveguide_ports[0]
    cfg=sim._build_waveguide_port_config(entry,grid,entry.freqs,2)
    one=jnp.ones(grid.shape)
    weights=(one,one.at[cfg.ref_x,2,2].set(.5),one)
    eps=(one,one.at[cfg.ref_x,2,2].set(2.1 if extra_anisotropy else 2.),one)
    updates=dict(aniso_eps=eps,conformal_weights=weights)
    if extra_anisotropy:
        with pytest.raises(ValueError,match='anisotropic permittivity'):
            resolve_waveguide_section_medium([cfg],grid,MaterialArrays(one,0.,one),updates,())
    else:
        resolved=resolve_waveguide_section_medium([cfg],grid,MaterialArrays(one,0.,one),updates,())[0]
        assert resolved.medium.eps_r == 1.
        assert resolved.probe_medium_matches


def test_power_wave_entries_below_either_cutoff_are_nan():
    # Vacuum launch (TE10 cutoff 6.25 GHz) facing an eps_r 2.2 receive port (4.21 GHz): at 5 GHz only the filled port
    # propagates, its partner's Z is the evanescent sentinel, and S21/S12 have no power meaning.
    sim = Simulation(freq_max=12e9, domain=(L, A, B), dx=DX, boundary={"x": "cpml", "y": "pec", "z": "pec"},
                     cpml_layers=8)
    fill(sim, .06, .22)
    freqs = np.array([5e9, 8e9])
    sim.add_waveguide_port(.02, direction="+x", name="launch", f0=8e9, freqs=freqs)
    sim.add_waveguide_port(.10, direction="-x", name="receive", f0=8e9, freqs=freqs)
    s = np.asarray(execute(sim, "compute_waveguide_s_matrix", normalize=False).s_params)
    assert np.all(np.isnan(s[1, 0, 0])) and np.all(np.isnan(s[0, 1, 0])), "transmission below the vacuum cutoff"
    assert np.all(np.isfinite(s[:, :, 1])), "both ports propagate at 8 GHz"
    assert np.isfinite(s[0, 0, 0]) and np.isfinite(s[1, 1, 0]), "reflections are not renormalized"


def test_traced_override_at_probe_plane_refused():
    # The section stays vacuum; only the probe plane sees eps_r 2.2, so a traced S21 split would use the wrong Z.
    sim = guide()
    grid = sim._build_realized_grid()
    eps = np.ones(grid.shape)
    eps[grid.index_of('x', .032):grid.index_of('x', .08)] = 2.2
    def call(eps):
        return execute(sim, "forward", eps_override=eps).time_series
    with pytest.raises(Exception, match="probe plane"):
        jax.jit(call)(jnp.asarray(eps)).block_until_ready()


def test_vacuum_reference_sim_for_filled_ports_refused():
    # A vacuum reference guide would give P_inc and a_inc at the wrong modal impedance, scaling |S| without a word.
    sim=guide();fill(sim,-.1,.22)
    sim.add_waveguide_port(.10,direction='-x',name='receive',f0=9e9,freqs=np.array([8e9,10e9]))
    reference=guide()
    with pytest.raises(ValueError,match="share its port's medium"):
        execute(sim,'compute_waveguide_s_matrix',normalize='flux',port_reference_sims=[reference,reference])
