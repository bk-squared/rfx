"""JIT runner tests for the canonical Phase-1 z-slab lane."""

import numpy as np
import jax.numpy as jnp
import pytest


def _make_pec_sim(with_probe=True, with_source=True):
    from rfx import Simulation

    sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="pec", dx=2e-3)
    if with_source:
        sim.add_source(position=(0.02, 0.02, 0.02), component="ez")
    sim.add_refinement(z_range=(0.012, 0.028), ratio=2)
    if with_probe:
        sim.add_probe(position=(0.02, 0.02, 0.02), component="ez")
    return sim


class TestJITBasic:
    def test_jit_pec_runs_without_error(self):
        sim = _make_pec_sim()
        result = sim.run(n_steps=50)
        assert result is not None
        assert result.time_series is not None

    def test_jit_time_series_shape(self):
        sim = _make_pec_sim()
        result = sim.run(n_steps=80)
        ts = np.asarray(result.time_series)
        assert ts.shape == (80, 1)

    def test_jit_produces_nonzero_fields(self):
        sim = _make_pec_sim()
        result = sim.run(n_steps=150)
        ez_max = float(jnp.max(jnp.abs(result.state.ez)))
        assert ez_max > 1e-10


class TestJITEdgeCases:
    def test_jit_no_probe(self):
        sim = _make_pec_sim(with_probe=False)
        result = sim.run(n_steps=30)
        ts = np.asarray(result.time_series)
        assert ts.size == 0 or np.allclose(ts, 0)

    def test_jit_no_source(self):
        sim = _make_pec_sim(with_source=False, with_probe=True)
        result = sim.run(n_steps=30)
        ts = np.asarray(result.time_series).ravel()
        assert np.allclose(ts, 0)

    def test_jit_cpml_is_rejected(self):
        from rfx import Simulation

        sim = Simulation(freq_max=5e9, domain=(0.04, 0.04, 0.04), boundary="cpml", dx=2e-3)
        with pytest.raises(ValueError, match="boundary='pec' only|CPML/UPML coexistence"):
            sim.add_refinement(z_range=(0.012, 0.028), ratio=2)


class TestJITRunnerDirect:
    def test_direct_jit_rejects_partial_xy_config(self):
        from rfx.core.yee import MaterialArrays
        from rfx.grid import Grid
        from rfx.subgridding.face_ops import build_zface_ops
        from rfx.subgridding.jit_runner import run_subgridded_jit
        from rfx.subgridding.sbp_sat_3d import SubgridConfig3D, phase1_3d_dt

        grid_c = Grid(freq_max=5e9, domain=(0.04, 0.04, 0.04), cpml_layers=0)
        nx, ny, nz = grid_c.shape
        dx_c = grid_c.dx
        ratio = 2
        dx_f = dx_c / ratio
        fi_lo, fi_hi = 1, nx - 1
        fj_lo, fj_hi = 0, ny
        fk_lo, fk_hi = 4, nz - 4
        nx_f = (fi_hi - fi_lo) * ratio
        ny_f = (fj_hi - fj_lo) * ratio
        nz_f = (fk_hi - fk_lo) * ratio

        config = SubgridConfig3D(
            nx_c=nx, ny_c=ny, nz_c=nz, dx_c=dx_c,
            fi_lo=fi_lo, fi_hi=fi_hi,
            fj_lo=fj_lo, fj_hi=fj_hi,
            fk_lo=fk_lo, fk_hi=fk_hi,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
            dx_f=dx_f, dt=float(phase1_3d_dt(dx_f)), ratio=ratio, tau=0.5,
            face_ops=build_zface_ops((fi_hi - fi_lo, fj_hi - fj_lo), ratio, dx_c),
        )
        mats_c = MaterialArrays(
            eps_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
            sigma=jnp.zeros((nx, ny, nz), dtype=jnp.float32),
            mu_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
        )
        mats_f = MaterialArrays(
            eps_r=jnp.ones((nx_f, ny_f, nz_f), dtype=jnp.float32),
            sigma=jnp.zeros((nx_f, ny_f, nz_f), dtype=jnp.float32),
            mu_r=jnp.ones((nx_f, ny_f, nz_f), dtype=jnp.float32),
        )

        with pytest.raises(ValueError, match="full-span x"):
            run_subgridded_jit(grid_c, mats_c, mats_f, config, n_steps=1)

    def test_direct_jit_runner_pec(self):
        from rfx.grid import Grid
        from rfx.core.yee import MaterialArrays
        from rfx.subgridding.face_ops import build_zface_ops
        from rfx.subgridding.jit_runner import run_subgridded_jit
        from rfx.subgridding.sbp_sat_3d import SubgridConfig3D, phase1_3d_dt

        grid_c = Grid(freq_max=5e9, domain=(0.04, 0.04, 0.04), cpml_layers=0)
        nx, ny, nz = grid_c.shape
        dx_c = grid_c.dx
        ratio = 2
        dx_f = dx_c / ratio
        fi_lo, fi_hi = 0, nx
        fj_lo, fj_hi = 0, ny
        fk_lo, fk_hi = 4, nz - 4
        nx_f = (fi_hi - fi_lo) * ratio
        ny_f = (fj_hi - fj_lo) * ratio
        nz_f = (fk_hi - fk_lo) * ratio

        dt = phase1_3d_dt(dx_f)

        config = SubgridConfig3D(
            nx_c=nx, ny_c=ny, nz_c=nz, dx_c=dx_c,
            fi_lo=fi_lo, fi_hi=fi_hi,
            fj_lo=fj_lo, fj_hi=fj_hi,
            fk_lo=fk_lo, fk_hi=fk_hi,
            nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
            dx_f=dx_f, dt=float(dt), ratio=ratio, tau=0.5,
            face_ops=build_zface_ops((fi_hi - fi_lo, fj_hi - fj_lo), ratio, dx_c),
        )

        shape_c = (nx, ny, nz)
        shape_f = (nx_f, ny_f, nz_f)
        mats_c = MaterialArrays(
            eps_r=jnp.ones(shape_c, dtype=jnp.float32),
            sigma=jnp.zeros(shape_c, dtype=jnp.float32),
            mu_r=jnp.ones(shape_c, dtype=jnp.float32),
        )
        mats_f = MaterialArrays(
            eps_r=jnp.ones(shape_f, dtype=jnp.float32),
            sigma=jnp.zeros(shape_f, dtype=jnp.float32),
            mu_r=jnp.ones(shape_f, dtype=jnp.float32),
        )

        n_steps = 60
        times = np.arange(n_steps) * dt
        f0 = 3e9
        waveform = np.exp(-((times - 3 / f0) * f0) ** 2) * np.sin(2 * np.pi * f0 * times)
        si, sj, sk = nx_f // 2, ny_f // 2, nz_f // 2
        result = run_subgridded_jit(
            grid_c,
            mats_c,
            mats_f,
            config,
            n_steps,
            sources_f=[(si, sj, sk, "ez", waveform.astype(np.float32))],
            probe_indices_f=[(si, sj, sk)],
            probe_components=["ez"],
        )

        ts = np.asarray(result.time_series).ravel()
        assert np.all(np.isfinite(ts))
        assert np.max(np.abs(ts)) > 1e-10
