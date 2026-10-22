"""Step B: VALUE transforms, named STRUCTURE refusal and threshold callbacks."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.api._spec import MaterialSpec
from rfx.model import declaration_roles as D
from rfx.sources.sources import GaussianPulse, ModulatedGaussian, CWSource, CustomWaveform
from tests.unit.runners import test_path_disposition_cells as B

LANES = ("fwd_uniform", "fwd_nonuniform")

# Explicit witnesses, independent of the role registry. Dimensionful deltas
# are additive, including declarations whose default value is zero.
VALUE_CASES = [
    ("MaterialSpec", name, "material", theta, delta)
    for name, theta, delta in (("eps_r", 4., .2), ("sigma", .5, .05),
                              ("sigma", 0., .5), ("mu_r", 2., .1),
                              ("chi3", 0., 100.), ("chi3", 1e3, 100.))
] + [
    ("DebyePole", "delta_eps", "debye", 2., .2),
    ("DebyePole", "tau", "debye", 1e-11, 1e-12),
    ("LorentzPole", "omega_0", "lorentz", 5e10, 1e9),
    ("LorentzPole", "delta", "lorentz", 6e9, 1e8),
    ("LorentzPole", "kappa", "lorentz", 5e21, 1e20),
    ("_PortEntry", "impedance", "port", 50., 2.),
    ("_MSLPortEntry", "impedance", "msl_port", 50., 2.),
    ("CoaxialPort", "impedance", "coax_port", 50., 2.),
    ("ThinConductor", "sigma_bulk", "thin", 1e3, 50.),
    ("ThinConductor", "thickness", "thin", 1e-4, 1e-5),
    ("ThinConductor", "surface_impedance_f0", "thin_f0", 5e9, 1e8),
] + [("LumpedRLCSpec", name, "rlc", theta, delta)
     for name, theta, delta in (("R", 10., 1.), ("L", 1e-9, 1e-10), ("C", 1e-12, 1e-13))
] + [(record, name, feature, theta, delta)
     for record, feature in (("_WaveguidePortEntry", "waveguide_port"),
                             ("_FloquetPortEntry", "floquet_port"), ("_TFSFEntry", "plane_wave"))
     for name, theta, delta in (("f0", 5e9, 1e8), ("bandwidth", .5, .05), ("amplitude", 1., .1))
] + [(cls.__name__, name, "waveform", theta, delta)
     for cls in (GaussianPulse, ModulatedGaussian, CWSource)
     for name, theta, delta in (("f0", 5e9, 1e8), ("bandwidth", .8, .05), ("amplitude", 1., .1))
     if name != "bandwidth" or cls is not CWSource
] + [("_TFSFEntry", "waveform", "custom_tfsf", 1., .1)
] + [("Simulation", field, "profile", 0., 1e-5)
     for field in ("_dx_profile", "_dy_profile", "_dz_profile")]


def value_objective(record, field, feature, lane):
    def objective(theta):
        if feature == "profile":
            nominal = np.full(12, 1e-3)
            nominal[4:8] = [9e-4, 11e-4, 11e-4, 9e-4]
            if isinstance(theta, jax.core.Tracer):
                widths = jnp.asarray(nominal).at[4].add(theta).at[5].add(-theta)
            else:
                widths = nominal.copy()
                widths[4] += float(theta)
                widths[5] -= float(theta)
            sim = B._base(lane, **{field.lstrip("_"): widths},
                          nominal_profiles={field[2]: nominal} if isinstance(widths, jax.core.Tracer) else None,
                          dt=1e-12, dt_min_cell=5e-4)
            B._block(sim, eps_r=4.)
        elif feature in ("material", "debye", "lorentz"):
            sim = B._base(lane)
            kwargs = dict(eps_r=4., sigma=.5, mu_r=2.)
            if feature == "material":
                kwargs[field] = theta
            elif feature == "debye":
                pole = D.DebyePole(2., 1e-11)._replace(**{field: theta})
                kwargs["debye_poles"] = [pole]
            else:
                pole = D.LorentzPole(5e10, 6e9, 5e21)._replace(**{field: theta})
                kwargs["lorentz_poles"] = [pole]
            B._block(sim, **kwargs)
        elif feature == "waveform":
            sim = B._base(lane, source=None)
            cls = {c.__name__: c for c in (GaussianPulse, ModulatedGaussian, CWSource)}[record]
            kwargs = dict(f0=5e9, amplitude=1.)
            if cls is not CWSource:
                kwargs.update(bandwidth=.8, cutoff=3.)
            else:
                kwargs["ramp_steps"] = 0
            kwargs[field] = theta
            sim.add_source((.004, .006, .006), "ez", waveform=cls(**kwargs), amplitude_kind="field")
        elif feature == "port":
            sim = B._base(lane, source=None)
            sim.add_port((.006, .008, .006), "ez", impedance=theta, waveform=B.WAVEFORM)
        elif feature == "rlc":
            sim = B._base(lane)
            kwargs = dict(R=10., L=1e-9, C=1e-12)
            kwargs[field] = theta
            sim.add_lumped_rlc((.006,) * 3, topology="parallel", **kwargs)
        elif feature in ("thin", "thin_f0"):
            sim = B._base(lane)
            kwargs = dict(sigma_bulk=1e3, thickness=1e-4)
            if feature == "thin_f0":
                kwargs["surface_impedance_f0"] = theta
            else:
                kwargs[field] = theta
            sim.add_thin_conductor(Box((.006, .003, .003), (.006, .009, .009)),
                                   **kwargs)
        else:
            attr = {"waveguide_port": "_waveguide_ports", "floquet_port": "_floquet_ports",
                    "plane_wave": "_tfsf", "custom_tfsf": "_tfsf",
                    "coax_port": "_coaxial_ports", "msl_port": "_msl_ports"}[feature]
            key = "plane_wave" if feature == "custom_tfsf" else feature
            sim = B._build(B.FEATURES[attr, key], lane, True)
            if feature == "custom_tfsf":
                waveform = CustomWaveform(lambda t: theta * B.WAVEFORM(t))
                sim._tfsf = replace(sim._tfsf, waveform=waveform)
            elif attr == "_tfsf":
                sim._tfsf = replace(sim._tfsf, **{field: theta})
            else:
                entries = getattr(sim, attr)
                entry = entries[0]
                entries[0] = entry._replace(**{field: theta}) if hasattr(entry, "_fields") else replace(entry, **{field: theta})
        return jnp.sum(sim.forward(n_steps=40, checkpoint=False, skip_preflight=True).time_series ** 2)
    return objective


@pytest.mark.parametrize("record,field,feature,theta,delta,lane", [
    (*case, lane) for case in VALUE_CASES
    for lane in (("fwd_nonuniform",) if case[2] == "profile" else LANES)
])
def test_value_transforms(record, field, feature, theta, delta, lane):
    objective = value_objective(record, field, feature, lane)
    # Existing lane refusals remain part of admission, as B3 specifies for Kerr.
    refusal = ("add_coaxial_port" if record == "CoaxialPort" else
               "Kerr" if record == "MaterialSpec" and field == "chi3" and lane == "fwd_nonuniform" else
               "Floquet" if record == "_FloquetPortEntry" and lane == "fwd_nonuniform" else
               "CustomWaveform and closed_box" if feature == "custom_tfsf" and lane == "fwd_nonuniform" else None)
    if refusal:
        with pytest.raises(NotImplementedError, match=refusal):
            jax.jit(objective)(jnp.float32(theta))
        return
    if record == "_WaveguidePortEntry" and field == "f0":
        theta = 15e9
    before = objective(theta)
    after = objective(theta + delta)
    compiled = jax.jit(objective)(jnp.float32(theta))
    grad = jax.grad(objective)(jnp.float32(theta))
    batched = jax.vmap(objective)(jnp.array([theta, theta + delta], jnp.float32))
    assert delta != 0
    assert np.isfinite(np.asarray([before, after, compiled, grad, *batched])).all()
    assert before != after, (record, field, lane, "dependence witness")
    assert grad != 0, (record, field, lane, "nonzero gradient")


@pytest.mark.parametrize("cls,field", [(c, f) for (c, f), role in D.ROLES.items() if role == D.STRUCTURE]
                         + [(Simulation, f) for f in D.SIMULATION_STRUCTURE])
def test_structure_fields_refused_by_name(cls, field):
    # Construct a record without running its concrete constructor validation;
    # entry admission must cover every field, including optional fields.
    def objective(theta):
        if cls is Simulation:
            sim = object.__new__(Simulation)
            vars(sim)[field] = theta
            D.refuse_traced_structure(sim)
            return theta
        if hasattr(cls, "_fields"):
            record = cls(**{name: theta if name == field else None for name in cls._fields})
        else:
            record = object.__new__(cls)
            for name in D.record_fields(cls):
                object.__setattr__(record, name, theta if name == field else None)
        if cls is D.ThinConductor:
            object.__setattr__(record, "sigma_bulk", 5.8e7)
        sim = SimpleNamespace(_geometry=[record])
        D.refuse_traced_structure(sim)
        return theta
    with pytest.raises(NotImplementedError) as caught:
        jax.jit(objective)(jnp.float32(1.))
    assert cls.__name__ in str(caught.value) and field in str(caught.value)
    exc = caught.value
    while exc is not None:
        assert not isinstance(exc, jax.errors.ConcretizationTypeError)
        exc = exc.__cause__ or exc.__context__


@pytest.mark.parametrize("family", ["material", "thin"])
def test_traced_sigma_threshold(family):
    objective = value_objective("MaterialSpec" if family == "material" else "ThinConductor",
                                "sigma" if family == "material" else "sigma_bulk", family, "fwd_uniform")
    compiled = jax.jit(objective)
    assert np.isfinite(compiled(jnp.float32(.5)))
    name = "block" if family == "material" else "ThinConductor.sigma_bulk"
    with pytest.raises(Exception, match=name + ".*PEC threshold"):
        jax.block_until_ready(compiled(jnp.float32(2e6)))


@pytest.mark.parametrize("cls,field", [(D.Sphere, "radius"), (D.Cylinder, "radius"), (D.Cylinder, "height")])
@pytest.mark.parametrize("lane", LANES)
def test_traceable_shape_dimensions(cls, field, lane):
    def objective(theta):
        sim = B._base(lane)
        kwargs = dict(center=(.006,) * 3, radius=.0015)
        if cls is D.Cylinder:
            kwargs["height"] = .003
        kwargs[field] = theta
        sim.add_material("body", eps_r=4.)
        sim.add(cls(**kwargs), material="body")
        return jnp.sum(sim.forward(n_steps=40, checkpoint=False, skip_preflight=True).time_series ** 2)
    theta = .003 if field == "height" else .0015
    assert np.isfinite(jax.jit(objective)(theta))
    assert np.isfinite(jax.vmap(objective)(jnp.array([theta, theta + .0001]))).all()
    with pytest.raises(NotImplementedError, match="gradient with respect to a declared shape dimension"):
        jax.grad(objective)(theta)


@pytest.mark.parametrize("entry", ["run", "forward", "run_devices", "forward_devices", "vmap_sweep"])
@pytest.mark.parametrize("transform", ["jit", "grad", "vmap"])
@pytest.mark.parametrize("kind,record,field", [
    ("sphere", "Sphere", "center"), ("port", "_PortEntry", "position"),
    ("precision", "Simulation", "_precision"),
    ("pinned", "PinnedSheet", "normal_axis"), ("ntff", "NTFFBox", "freqs"),
])
def test_structure_at_real_entries(entry, transform, kind, record, field):
    def objective(theta):
        sim = B._base("fwd_uniform")
        if kind == "sphere":
            sim.add(D.Sphere((theta, .006, .006), .0015), material="pec")
        elif kind == "port":
            sim._ports[0] = replace(sim._ports[0], position=(theta, .006, .006))
        elif kind == "precision":
            sim._precision = theta
        elif kind == "pinned":
            sim.add_pinned_sheet(normal_axis=theta, plane_index=6, i_range=(3, 9), j_range=(3, 9))
        else:
            sim.add_ntff_box((.002,) * 3, (.010,) * 3, freqs=theta)
        kwargs = dict(n_steps=2, skip_preflight=True)
        if entry == "vmap_sweep":
            from rfx.vmap_sweep import vmap_material_sweep
            result = vmap_material_sweep(sim, "eps_r", jnp.array([2., 3.]), n_steps=2)
        elif entry.startswith("forward"):
            kwargs["checkpoint"] = False
            if entry.endswith("devices"):
                kwargs.update(distributed=True, devices=jax.devices("cpu")[:2])
            result = sim.forward(**kwargs)
        else:
            kwargs["compute_s_params"] = False
            if entry.endswith("devices"):
                kwargs["devices"] = jax.devices("cpu")[:2]
            result = sim.run(**kwargs)
        return jnp.sum(result.time_series)
    operation = {"jit": jax.jit, "grad": jax.grad, "vmap": jax.vmap}[transform](objective)
    theta = jnp.array([.005, .006]) if transform == "vmap" else jnp.float32(.005)
    with pytest.raises(NotImplementedError) as caught:
        operation(theta)
    assert record in str(caught.value) and field in str(caught.value)
    assert "design_box / eps_override" in str(caught.value) if kind == "sphere" else "declare it concrete" in str(caught.value)


@pytest.mark.parametrize("mode", ["PEC", "surface-impedance"])
def test_thin_thickness_no_effect_refusal(mode):
    def objective(theta):
        sim = B._base("fwd_uniform")
        sim.add_thin_conductor(Box((.006, .003, .003), (.006, .009, .009)),
            sigma_bulk=5.8e7, thickness=theta,
            surface_impedance_f0=5e9 if mode == "surface-impedance" else None)
        return theta
    with pytest.raises(NotImplementedError) as caught:
        jax.jit(objective)(jnp.float32(1e-4))
    assert str(caught.value) == ("ThinConductor.thickness has no effect on the realized model "
        f"in this mode: {mode}; a traced value would have a zero gradient. Declare it concrete.")


def test_physical_dependence_refusal_text():
    with pytest.raises(NotImplementedError) as caught:
        jax.jit(lambda theta: D.refuse_field(D._TFSFEntry, "theta_deg", theta))(jnp.float32(20.))
    assert str(caught.value) == ("_TFSFEntry.theta_deg changes the solved fields, but rfx has no traced "
        "route for it yet (rfx #1961). Declare it concrete.")


# Positive witnesses are inventory theta1 values: a:rlc_L/C/R. The absent
# declaration cells themselves have theta1=0 and cannot supply a positive witness.
RLC_ABSENT_CASES = [
    ("R-only-C", "C", 1.05e-12, dict(R=10.)),
    ("R-only-L", "L", 1.05e-9, dict(R=10.)),
    ("series-RL-C", "C", 1.05e-12, dict(R=10., L=1e-9)),
    ("parallel-C-only-R", "R", 10.5, dict(C=1e-12, topology="parallel")),
]


@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("context", ["C0", "C5"])
@pytest.mark.parametrize("case,field,positive,values", RLC_ABSENT_CASES,
                         ids=[case[0] for case in RLC_ABSENT_CASES])
def test_zero_lumped_value_is_absent(case, field, positive, values, context, lane):
    def objective(theta):
        sim = B._base(lane)
        sim.add_lumped_rlc((.006,) * 3, **(values | {field: theta}))
        if context == "C5":
            sim.add_dft_plane_probe(axis="x", coordinate=.006, n_freqs=3)
        return jnp.sum(sim.forward(n_steps=40, checkpoint=False, skip_preflight=True).time_series ** 2)

    for theta in (0.0, positive):
        concrete = objective(float(theta))
        operand = jnp.float32(theta)
        plain = objective(operand)
        compiled = jax.jit(objective)(operand)
        gradient = jax.grad(objective)(operand)
        assert np.isfinite(np.asarray([concrete, plain, compiled, gradient])).all(), (case, theta, lane)
        if theta == 0.0:
            assert gradient == 0.0, (case, context, lane)
        else:
            assert gradient != 0.0, (case, context, lane)
        np.testing.assert_allclose([plain, compiled], concrete, rtol=1e-6, atol=0)
