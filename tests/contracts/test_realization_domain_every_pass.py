"""CPML domain rounding has the same admission on prepared and materials solves."""
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.topology import TopologyDesignRegion, topology_optimize


def scene():
    sim = Simulation(freq_max=5e9, domain=(0.015, 0.015, 0.015),
                     boundary="cpml", cpml_layers=8)
    sim.add_port((0.005, 0.0075, 0.0075), "ez")
    sim.add_probe((0.01, 0.0075, 0.0075), "ez")
    return sim


def mixed_materials_forward(sim):
    """Exercise the mixed calculator's real materials-forward call and scan.

    Its drive stage needs a lumped accumulator and a DFT plane; the MSL
    calibration and long-time scoring are outside domain admission.
    """
    from rfx._diagnostic_transport import merge_diagnostics
    from rfx.sparams._mixed_stages import _run_mixed_drive

    sim.add_dft_plane_probe(axis="x", coordinate=0.01, component="ez",
                            freqs=jnp.array([2.5e9]), name="domain_contract")
    grid = sim._build_grid()
    materials, debye, lorentz, pec, *_ = sim._assemble_materials(grid)
    channels = [np.zeros((1, 1, 1), dtype=complex) for _ in range(3)]
    return _run_mixed_drive(
        _diagnostics=(), _mx_pec_sheets=(), _mx_pec_wires=(), box_lw=None,
        debye_spec=debye, flux_faces_lw=(), flux_names_msl=(),
        freqs_arr=np.array([2.5e9]), grid=grid, i_lw=channels[1],
        lorentz_spec=lorentz, magnitude_channel="vi", materials=materials,
        merge_diagnostics=merge_diagnostics, n_freqs_used=1, n_lw=1,
        n_steps=2, pec_mask=pec, plane_msl=None, run_idx=0,
        v_lw=channels[0], variant=sim, vref_lw=channels[2], wire_mode=False,
    )


def uniform_materials_run(sim):
    """Call the uniform materials stage without a pre-solve conductor owner."""
    from rfx.runners.uniform import run_uniform

    grid = sim._build_grid()
    materials, debye, lorentz, pec, *_ = sim._assemble_materials(grid)
    return run_uniform(sim, grid=grid, base_materials=materials,
                       debye_spec=debye, lorentz_spec=lorentz, pec_mask=pec,
                       n_steps=2, compute_s_params=False)


@pytest.mark.parametrize("lane", ("forward", "run_s_params", "topology", "mixed_materials_forward",
                                  "lumped_wire_s_matrix_scan", "uniform_materials_run"))
def test_cpml_domain_admitted_on_every_pass(lane, monkeypatch):
    import rfx.model.realization_verdict as verdict

    # Observe findings even on routes which return only arrays or an optimizer
    # history. This wrapper leaves the production judgement and delivery intact.
    findings = []
    judge = verdict.judge_realization

    def observed_judge(*args, **kwargs):
        result = judge(*args, **kwargs)
        findings.extend(result)
        return result

    monkeypatch.setattr(verdict, "judge_realization", observed_judge)
    monkeypatch.setitem(sys.modules, "optax", None)
    with jax.default_device(jax.devices("cpu")[0]):
        sim = scene()
        assert sim._snap == "strict"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if lane == "forward":
                result = sim.forward(n_steps=2, checkpoint=False)
                assert result.time_series.shape == (2, 1)
            elif lane == "run_s_params":
                # A different S record length enters the public drive scan;
                # the one-port default reads S from the main run's record.
                result = sim.run(compute_s_params=True, s_param_freqs=[2.5e9],
                                 n_steps=2, s_param_n_steps=3)
                assert result.s_params.shape == (1, 1, 1)
            elif lane == "topology":
                region = TopologyDesignRegion(
                    corner_lo=(0.006, 0.006, 0.006),
                    corner_hi=(0.009, 0.009, 0.009),
                    material_bg="air", material_fg="fr4",
                )
                result = topology_optimize(
                    sim, region, lambda run: -jnp.sum(run.time_series ** 2),
                    n_iterations=1, beta_schedule=[(0, 1.0)], verbose=False,
                )
                assert len(result.history) == 1
                assert np.isfinite(result.history).all()
            elif lane == "mixed_materials_forward":
                result = mixed_materials_forward(sim)
                assert result[1]["time_series"].shape == (2, 1)
            elif lane == "lumped_wire_s_matrix_scan":
                from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
                result, _ = compute_lumped_wire_s_matrix_via_scan(
                    sim, [2.5e9], n_steps=2)
                assert result.shape == (1, 1, 1)
            else:
                result = uniform_materials_run(sim)
                assert result.time_series.shape == (2, 1)
    # Any RfxRefusal already fails the call; warnings and retained findings
    # must not silently replace the former domain refusal.
    assert [str(f) for f in findings if "realization domain" in str(f)] == []
    assert [str(w.message) for w in caught if "realization domain" in str(w.message)] == []
