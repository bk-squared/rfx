"""The admission matrix follows the actual combined ADE kernel (#1934)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import EPS_0, init_materials, init_state, update_h
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import drude_pole, init_lorentz
from rfx.model.pole_stability import C0, largest_growth
from rfx.stepping.uniform_setup import _update_e_with_optional_dispersion


@pytest.mark.parametrize("wp,de,tau", [(0., 3., 2.), (.25, 3., 2.), (.45, .5, 10.)],
                         ids=["debye", "mixed_stable", "mixed_unstable"])
def test_debye_matrix_matches_actual_fourier_mode(wp, de, tau):
    # One x Fourier mode, Ey/Hz only. The real Nyquist mode has x=4(c dt/dx)^2.
    shape, dx = (8, 1, 1), 1e-3
    dt = .99 * dx / C0
    materials = init_materials(shape)
    periodic = (True,) * 3
    dp = [DebyePole(de, tau * dt)]
    lp = [drude_pole(wp / dt, 0.)] if wp else []
    db = init_debye(dp, materials, dt, periodic=periodic, field_dtype=jnp.complex64)
    lr = init_lorentz(lp, materials, dt, periodic=periodic, field_dtype=jnp.complex64) if lp else None
    mode = jnp.asarray((-1.) ** np.arange(8))[:, None, None]
    empty = init_state(shape, field_dtype=jnp.complex64)
    impedance = C0 * 1.25663706212e-6
    size = 3 + (2 if lr is not None else 0)

    def carry_from_vector(v):
        state = empty._replace(ey=mode * v[0], hz=mode * v[1] / impedance)
        dbs = db[1]._replace(py=mode[None] * v[2] * EPS_0)
        lrs = None if lr is None else lr[1]._replace(
            py=mode[None] * v[3] * EPS_0, py_prev=mode[None] * v[4] * EPS_0)
        return state, dbs, lrs

    def vector_from_carry(carry):
        state, dbs, lrs = carry
        values = [state.ey[0, 0, 0], state.hz[0, 0, 0] * impedance, dbs.py[0, 0, 0, 0] / EPS_0]
        if lrs is not None:
            values += [lrs.py[0, 0, 0, 0] / EPS_0, lrs.py_prev[0, 0, 0, 0] / EPS_0]
        return jnp.asarray(values)

    @jax.jit
    def step(carry, unused):
        state, dbs, lrs = carry
        state = update_h(state, materials, dt, dx, periodic)
        state, dbs, lrs = _update_e_with_optional_dispersion(
            state, materials, dt, dx, debye=(db[0], dbs),
            lorentz=None if lr is None else (lr[0], lrs), periodic=periodic)
        # Full dynamic state norm; P is in eps0 units. Hz is in impedance units.
        values = [state.ey, state.hz * impedance, dbs.py / EPS_0]
        if lrs is not None:
            values += [lrs.py / EPS_0, lrs.py_prev / EPS_0]
        norm = jnp.sqrt(sum(jnp.sum(jnp.abs(v) ** 2) for v in values))
        return (state, dbs, lrs), norm

    # Excite a single eigenmode of the actual kernel, independently of the
    # admission matrix (whose rows are never read here).
    actual = np.stack([np.asarray(vector_from_carry(step(carry_from_vector(v), None)[0]))
                       for v in np.eye(size)], axis=1)
    roots, vectors = np.linalg.eig(actual)
    initial = carry_from_vector(vectors[:, np.argmax(abs(roots))])
    # Sixty steps resolve the root without overflowing float32 on the
    # intentionally unstable mode (rho=1.38, whose squared norm overflows
    # during a 180-step record).
    _, norms = jax.lax.scan(step, initial, None, length=60)
    norms = np.asarray(norms)
    measured = float(np.exp(np.polyfit(np.arange(20, 60), np.log(norms[20:]), 1)[0]))
    x = 4 * (C0 * dt / dx) ** 2
    predicted = largest_growth(1., 0., lp, dt, x, debye=dp, x_min=x)
    print(f"{wp=}: measured={measured:.12g}, matrix={predicted:.12g}, relative={abs(measured/predicted-1):.6g}")
    assert measured == pytest.approx(predicted, rel=1e-3)


@pytest.mark.parametrize("factor", [.9, .98])
@pytest.mark.parametrize("debye", [None, (.5, 10.), (3., 2.), (10., .5), (30., .2)])
def test_ten_stable_debye_drude_cases_stay_admitted(factor, debye):
    from tests.unit.materials.test_pole_stability_admission import BULK_LIMIT, _box, _default_step, _run
    dt = _default_step()
    sim = _box(factor * BULK_LIMIT)
    sim.add_material("m", eps_r=1., lorentz_poles=[drude_pole(factor * BULK_LIMIT / dt, 0.)],
        debye_poles=None if debye is None else [DebyePole(debye[0], debye[1] * dt)])
    trace = np.abs(_run(sim, n_steps=1500)).ravel()
    print(f"{factor=} {debye=}: finite={np.isfinite(trace).all()}, first_peak={trace[:300].max():.9g}, last_peak={trace[-300:].max():.9g}")
    assert np.isfinite(trace).all()


@pytest.mark.parametrize("route", ["uniform", "graded", "graded_override", "compile_budget"])
def test_unstable_combined_medium_reaches_each_judge(route):
    from tests.unit.materials.test_pole_stability_admission import _box, _default_step
    from rfx.model.pole_stability import courant_budget
    dt = _default_step()
    # Debye does not rescue this above-limit Drude medium: the combined
    # matrix gives rho=1.00082051 even with this large polarization.
    sim = _box(.45, eps_inf=4. if route == "graded_override" else 1., graded=route.startswith("graded"))
    dp, lp = [DebyePole(30., .2 * dt)], [drude_pole(.45 / dt, 0.)]
    sim.add_material("m", eps_r=4. if route == "graded_override" else 1., debye_poles=dp, lorentz_poles=lp)
    with pytest.raises(ValueError, match="unstable at this model's time step"):
        if route == "compile_budget":
            grid = sim._build_realized_grid()
            sim._init_dispersion(init_materials(grid.shape), grid.dt, (dp, None), (lp, None), budget=courant_budget(grid))
        elif route == "graded_override":
            grid = sim._build_realized_grid()
            sim.forward(n_steps=5, eps_override=jnp.ones(grid.shape), sigma_override=jnp.zeros(grid.shape),
                        distributed=True, devices=jax.devices("cpu")[:2], skip_preflight=True)
        else:
            sim.realized_geometry()
