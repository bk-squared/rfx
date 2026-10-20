"""All-passive main records retain termination spectra, never a port S matrix."""
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.measurement import port_s_from_record
from tests.contracts.run_forward_census import (
    BUILDERS, SOURCE, base_scene, build_p_waveguide,
)
from tests.contracts.test_graded_port_reference import constant_scene
from tests.unit.nonuniform.test_nu_wire_port_lane_parity import (
    FREQS, PORT_Y, PORT_Z, _build,
)

BINS = np.array([3e9, 7e9, 13e9], np.float32)
REFUSAL = r"Lumped/wire S needs a driven port:.*excite=False.*lumped_port_sparams"


def _scene(mesh="uniform", *, wire=False, illumination="plain", single=False):
    sim = (build_p_waveguide(mesh) if illumination == "waveguide"
           else base_scene(mesh, source=False))
    for x in ((.012,) if single else (.012, .022)):
        sim.add_port((x, .009, .004), "ez", impedance=50.,
                     extent=.002 if wire else None, excite=False)
    if illumination == "plain":
        sim.add_source(SOURCE, "ez", amplitude_kind="current")
    return sim


def _before_step(*args, **kwargs):
    pytest.fail("an explicit all-passive S request reached time stepping")


def test_low_level_all_passive_record_refuses():
    spec = SimpleNamespace(impedance=50., excite=False)
    for wire in (False, True):
        channels = [(jnp.ones(3, dtype=jnp.complex64),) * (5 if wire else 3)]
        with pytest.raises(ValueError, match=REFUSAL):
            port_s_from_record(channels, [spec], wire_mode=wire)


def test_plain_source_refused_before_stepping(monkeypatch):
    monkeypatch.setattr(jax.lax, "scan", _before_step)
    monkeypatch.setattr(jax.lax, "while_loop", _before_step)
    with pytest.raises(ValueError, match=REFUSAL):
        _scene().forward(n_steps=64, port_s11_freqs=BINS, skip_preflight=True)


@pytest.mark.parametrize("mesh,entry,wire", [
    ("uniform", "forward", False), ("uniform", "forward", True),
    ("uniform", "run", True),
    ("graded", "forward", False), ("graded", "forward", True),
    ("graded", "run", True),
])
@pytest.mark.parametrize("illumination", ["plain", "waveguide", "none"])
def test_explicit_main_record_refuses(mesh, entry, wire, illumination, monkeypatch):
    monkeypatch.setattr(jax.lax, "scan", _before_step)
    monkeypatch.setattr(jax.lax, "while_loop", _before_step)
    sim = _scene(mesh, wire=wire, illumination=illumination,
                 single=entry == "run")
    requests = ([{"port_s11_freqs": BINS}] if entry == "forward" else
                [{"compute_s_params": True}, {"s_param_freqs": BINS}])
    for request in requests:
        with pytest.raises(ValueError, match=REFUSAL):
            getattr(sim, entry)(n_steps=64, skip_preflight=True, **request)


def _default(sim, *, devices=None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim.run(n_steps=64, skip_preflight=True, devices=devices)
    refusal = [w for w in caught if "Lumped/wire S needs a driven port" in str(w.message)]
    assert len(refusal) == 1 and refusal[0].category is UserWarning
    assert Path(refusal[0].filename).resolve() == Path(__file__).resolve()
    assert str(refusal[0].message).endswith("; returning no S-parameters.")
    assert result.s_params is None
    assert result.sparam_time_records
    assert np.isfinite(np.asarray(result.time_series)).all()
    return result


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("wire", [False, True])
def test_all_passive_drive_scan_forms_every_column(mesh, wire):
    # Main #1915 drives each declared port, even when all are passive.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _scene(mesh, wire=wire, illumination="none").run(
            n_steps=64, skip_preflight=True, compute_s_params=True, s_param_freqs=BINS)
    assert result.s_params.shape == (2, 2, len(BINS))
    assert np.isfinite(np.asarray(result.s_params)).all()
    assert np.all(np.max(np.abs(np.asarray(result.s_params)), axis=(0, 2)) > 0)
    assert not any("needs a driven port" in str(w.message) for w in caught)
    pairs = result.wire_port_sparams if wire else result.lumped_port_sparams
    assert len(pairs) == 2


def test_graded_all_passive_wire_default_scans_without_refusal():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _scene("graded", wire=True, illumination="none").run(
            n_steps=64, skip_preflight=True)
    assert result.s_params.shape == (2, 2, 50)
    assert np.isfinite(np.asarray(result.s_params)).all()
    assert not any("needs a driven port" in str(w.message) for w in caught)


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("illumination", ["plain", "none"])
def test_default_wire_warns_and_keeps_spectra(mesh, illumination):
    result = _default(_scene(mesh, wire=True, illumination=illumination, single=True))
    assert len(result.wire_port_sparams) == 1
    assert all(np.asarray(a).shape == (50,) for a in result.wire_port_sparams[0][1])
    assert bool(np.any(np.asarray(result.time_series))) == (illumination != "none")


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
@pytest.mark.parametrize("illumination", ["plain", "none"])
def test_direct_runner_default_warns_and_keeps_spectra(mesh, illumination):
    from rfx.runners.nonuniform import run_nonuniform_path
    from rfx.runners.uniform import run_uniform
    sim = _scene(mesh, wire=True, illumination=illumination, single=True)
    kwargs = {}
    if mesh == "uniform":
        grid = sim._build_grid()
        mats, debye, lorentz, pec, shapes, _, kerr = sim._assemble_materials(grid)
        kwargs = dict(grid=grid, base_materials=mats, debye_spec=debye,
                      lorentz_spec=lorentz, pec_mask=pec, pec_shapes=shapes,
                      kerr_chi3=kerr)
    runner = run_uniform if mesh == "uniform" else run_nonuniform_path
    with pytest.warns(UserWarning, match=REFUSAL) as caught:
        result = runner(sim, n_steps=64, **kwargs)
    assert sum("needs a driven port" in str(w.message) for w in caught) == 1
    assert result.s_params is None and result.wire_port_sparams
    assert bool(np.any(np.asarray(result.time_series))) == (illumination != "none")


def test_waveguide_illumination_still_returns_fields_without_s():
    # The graded runner carries this composition. Uniform run refuses it
    # independently of S admission, as the separate boundary test pins.
    result = _default(_scene("graded", wire=True, illumination="waveguide", single=True))
    assert result.wire_port_sparams
    assert np.any(np.asarray(result.time_series))


def test_uniform_waveguide_composition_keeps_its_existing_refusal():
    with pytest.warns(UserWarning, match=REFUSAL):
        with pytest.raises(ValueError, match="Waveguide ports are not supported together"):
            _scene(wire=True, illumination="waveguide", single=True).run(
                n_steps=64, skip_preflight=True)


def test_tfsf_port_combination_is_refused_at_declaration():
    sim = _scene(illumination="none")
    with pytest.raises(ValueError, match="TFSF.*not supported.*lumped ports"):
        sim.add_tfsf_source()
    sim = base_scene(source=False).add_tfsf_source()
    with pytest.raises(ValueError, match="Lumped ports.*not supported.*TFSF"):
        sim.add_port((.012, .009, .004), "ez", impedance=50., excite=False)


def test_tfsf_cannot_admit_a_passive_record():
    from rfx.runners._admission import refuse_plain_sources_s_matrix
    # Public declarations refuse this composition above. Test the admission
    # boundary defensively with a real TFSF entry, without solving that model.
    sim = _scene(illumination="none")
    sim._tfsf = base_scene(source=False).add_tfsf_source()._tfsf
    with pytest.raises(ValueError, match=REFUSAL):
        refuse_plain_sources_s_matrix(sim, main_record=True)
    with pytest.warns(UserWarning, match=REFUSAL) as caught:
        admitted = refuse_plain_sources_s_matrix(sim, main_record=True, implicit=True)
    assert not admitted and len(caught) == 1


@pytest.mark.parametrize("wire", [False, True])
def test_graded_multi_device_default_and_explicit(wire, monkeypatch):
    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _scene("graded", wire=wire).run(
            n_steps=64, skip_preflight=True, devices=devices)
    assert result.s_params is None and result.sparam_time_records
    assert sum("A passive port's S column needs a drive of its own" in str(w.message)
               for w in caught) == int(wire)
    assert not any("Lumped/wire S needs a driven port" in str(w.message) for w in caught)
    monkeypatch.setattr(jax.lax, "scan", _before_step)
    monkeypatch.setattr(jax.lax, "while_loop", _before_step)
    for entry in ("run", "forward"):
        request = (dict(compute_s_params=True, devices=devices) if entry == "run"
                   else dict(port_s11_freqs=BINS, distributed=True, devices=devices))
        error, message = ((NotImplementedError, "A passive port's S column needs a drive")
                          if entry == "run" else (ValueError, REFUSAL))
        with pytest.raises(error, match=message):
            getattr(_scene("graded", wire=wire), entry)(
                n_steps=64, skip_preflight=True, **request)


@pytest.mark.parametrize("wire", [False, True])
def test_uniform_multi_device_explicit_refuses_before_dispatch(wire, monkeypatch):
    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2
    monkeypatch.setattr(jax.lax, "scan", _before_step)
    monkeypatch.setattr(jax.lax, "while_loop", _before_step)
    with pytest.raises(ValueError, match=REFUSAL):
        _scene(wire=wire).forward(
            n_steps=64, skip_preflight=True, port_s11_freqs=BINS,
            distributed=True, devices=devices)


def test_forward_without_s_request_keeps_named_records():
    # forward() has no implicit port-S request; run() owns the defaults above.
    with warnings.catch_warnings(record=True) as caught:
        result = _scene().forward(n_steps=64, skip_preflight=True)
    assert result.s_params is None and result.lumped_port_sparams is None
    assert len(result.sparam_time_records) == 2
    assert all(r.channels == ("V", "I", "V_ref") for r in result.sparam_time_records)
    assert not any("Lumped/wire S needs a driven port" in str(w.message) for w in caught)


# The driven column measured on main 4a90398e2 before product/test edits, using
# test_forward_port_s_matrix::test_driven_column_matches_run's scenes, n_steps=150.
# Compared by value, not bytes: float32 bytes differ between macOS and the Linux gate.
MAIN_DRIVEN_COLUMN = json.loads(
    (Path(__file__).with_name("all_passive_main_driven_column.json")).read_text())


@pytest.mark.parametrize("mesh", ["uniform", "graded"])
def test_driven_and_passive_output_matches_main(mesh):
    if mesh == "uniform":
        sim = BUILDERS["P-lumped2"](mesh)
        run = sim.run(n_steps=150, skip_preflight=True)
    else:
        sim = _build(True, excite=True)
        sim.add_port((12e-3, PORT_Y, PORT_Z), "ez", impedance=50.,
                     extent=3e-3, excite=False)
        run = sim.run(n_steps=150, skip_preflight=True,
                      compute_s_params=True, s_param_freqs=FREQS)
    result = sim.forward(n_steps=150, skip_preflight=True, port_s11_freqs=run.freqs)
    actual = np.asarray(result.s_params)
    assert actual.dtype == np.complex64
    assert np.isfinite(actual[:, 0]).all() and np.isnan(actual[:, 1]).all()
    main = MAIN_DRIVEN_COLUMN[mesh]
    expected = np.asarray(main["col0_re"]) + 1j * np.asarray(main["col0_im"])
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(actual[:, 0], expected.reshape(actual[:, 0].shape),
                               rtol=0, atol=1e-5 * scale)


def test_mixed_uniform_record_selects_the_driven_family():
    sim = constant_scene(False, "wire")
    sim.add_port((6/1024, 3/1024, 2/1024), "ez", impedance=75., excite=False)
    result = sim.forward(n_steps=32, skip_preflight=True, port_s11_freqs=BINS)
    assert result.s_params.shape == (1, 1, 3)
    assert result.lumped_port_sparams and result.wire_port_sparams
    assert result.sparam_read_ids.ids == ("port:0:mode:0",)
    # The value is the driven wire's own reflection, (V_port - Z0 I)/(V_port + Z0 I),
    # not the passive lumped port's load ratio that main selected here.
    (meta, accs), = result.wire_port_sparams
    v_port, i = np.asarray(accs[3]), np.asarray(accs[1])
    expected = (v_port - meta.impedance * i) / (v_port + meta.impedance * i)
    np.testing.assert_allclose(np.asarray(result.s_params)[0, 0], expected, rtol=1e-5)
    assert np.all(np.abs(expected) > 0.9)
