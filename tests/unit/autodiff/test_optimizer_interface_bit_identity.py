"""CPU bit identity against verbatim functions from main 04b960c1c.

Only the reference function names differ from that commit. Keep their bodies
unchanged: compare each entry on the same backend in the same process.
"""
from __future__ import annotations

import importlib
import math
from types import SimpleNamespace
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api import Simulation
from rfx.errors import AccuracyWarning
from rfx._internal._precision import resolve_field_dtype
from rfx.optimize import (
    DesignRegion, ProgressiveStage, _adam_update, _value_and_grad_with_fallback,
    optimize, progressive_optimize,
)
from rfx.preflight.realization import release_inspection
from rfx.topology import (
    TopologyDesignRegion, TopologyResult, _DEFAULT_BETA_SCHEDULE, _get_beta,
    apply_density_filter, apply_projection, density_to_material_fields,
    topology_optimize,
)

optimize_module = importlib.import_module("rfx.optimize")


def _reference_adam_multistart_04b960c1c(
    cost_fn: Callable,
    latent_inits,
    *,
    n_iters: int,
    lr: float,
    best_iterate: bool = False,
    step_clamp: float | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps_adam: float = 1e-8,
    verbose: bool = False,
    it_count: list | None = None,
    jit: bool = True,
):
    """Best-of multi-start Adam with optional best-iterate + step-clamp.

    Generic promotion of the MSL open-stub example's ``_multistart_adam``.  The three knobs are all OPT-IN and, with a
    single
    start (``len(latent_inits) == 1``), ``best_iterate=False`` and
    ``step_clamp=None``, this runs the hand-rolled Adam loop with the exact
    same arithmetic as the legacy inline ``optimize()`` loop. With
    ``jit=False`` the path is byte-identical to before this helper existed
    (see ``tests/unit/autodiff/test_optimize_multistart.py`` bit-identity gate).

    Unlike the example, the step-clamp here is GENERIC: it bounds the L2
    norm of the DoF (latent) update vector, rescaling the whole step by
    ``step_clamp / ||step||`` when it overshoots (direction preserved).
    The example clamps a physical LENGTH move through a nonlinear
    ``latent_to_L`` and so needs a bisection; that specialization stays in
    the example.  For a latent-space step the rescale is exact — no
    bisection — and it reduces to ``|Δlatent| <= step_clamp`` in the
    scalar/identity case the example tests exercise.

    Parameters
    ----------
    cost_fn : callable(latent) -> scalar
        Differentiable scalar objective (``jax.value_and_grad`` is taken
        internally).  Pure in ``cost_fn`` so this is testable against a
        synthetic cost with no FDTD forward.
    latent_inits : sequence of latent arrays
        One optimization is run from each init; the best is returned.
        Element 0 is the "primary" start (``optimize()`` uses the caller's
        ``init_latent`` here so ``n_starts=1`` is the legacy single run).
    n_iters, lr : int, float
        Adam iterations per start and learning rate.
    best_iterate : bool
        If True, each start returns (and is scored by) its lowest-loss
        visited iterate rather than its final iterate.  Guards the sharp-
        null overshoot the GPU run caught.
    step_clamp : float or None
        If set (>0), clamp the L2 norm of each latent update to this value.
    it_count : list([int]) or None
        Optional mutable one-element counter threaded from ``optimize()``
        so the ``forward()`` first-call banner fires exactly once.
    jit : bool
        Wrap ``jax.value_and_grad(cost_fn)`` in ``jax.jit`` once per call
        by default; see :func:`optimize`. First-call host-read tracing errors
        fall back to the eager loop with one warning. ``jit=False`` is eager.

    Returns
    -------
    (best_latent, best_loss_history, all_loss_histories, best_start)
        ``best_latent`` is the winning start's final iterate (or its best
        visited iterate when ``best_iterate=True``).  ``best_loss_history``
        is that start's per-iteration loss list.

    Raises
    ------
    RuntimeError
        Only for MULTI-start (``len(latent_inits) > 1``) when EVERY start
        produced a non-finite loss at every iterate — fail-closed rather
        than returning a NaN optimum.  A single start preserves the legacy
        fail-soft contract (warn + return the last finite design).
    """
    if step_clamp is not None and step_clamp <= 0:
        raise ValueError(f"optimize: step_clamp must be > 0, got {step_clamp}")

    grad_fn = _value_and_grad_with_fallback(cost_fn, jit, "optimize")

    def _run_one(init_lat):
        m = jnp.zeros_like(init_lat)
        v = jnp.zeros_like(init_lat)
        latent = init_lat
        loss_history: list[float] = []
        # Only retain visited iterates when best-iterate is requested, so the
        # default path holds no extra design tensors in memory.
        latents: list = [] if best_iterate else None
        last_good = latent  # last design with a confirmed-finite forward+gradient
        for it in range(n_iters):
            loss, grad = grad_fn(latent)

            # NaN/Inf gradient guard (unchanged from the legacy loop): a
            # non-finite gradient silently poisons ``latent`` and every later
            # iteration, so stop fast and RESTORE the last finite design.
            if not (bool(jnp.isfinite(loss)) and bool(jnp.all(jnp.isfinite(grad)))):
                import warnings as _w
                _w.warn(
                    f"optimize: non-finite loss/gradient (NaN or Inf) at iteration "
                    f"{it} — the FDTD forward almost certainly diverged. Common "
                    f"causes: dt above CFL, conformal=True at fine dx (a known NaN), "
                    f"PEC inside the CPML region, or a sub-cell PEC feature. Stopping early and "
                    f"returning the last finite design (from before iteration "
                    f"{it}).", AccuracyWarning,
                    stacklevel=2,
                )
                latent = last_good
                break

            last_good = latent  # this design produced a finite forward+gradient
            if it_count is not None:
                it_count[0] = it + 1
            loss_val = float(loss)
            loss_history.append(loss_val)
            if latents is not None:
                latents.append(latent)

            # Adam update
            step, m, v = _adam_update(
                m, v, grad, it, lr, beta1, beta2, eps_adam, step_clamp,
            )
            latent = latent - step

            if verbose and (it % 10 == 0 or it == n_iters - 1):
                print(f"  iter {it:4d}  loss = {loss_val:.6e}")

        if loss_history:
            if best_iterate:
                best_i = int(np.argmin(np.asarray(loss_history)))
                ret_latent = latents[best_i]
                score = loss_history[best_i]
            else:
                ret_latent = latent
                score = loss_history[-1]
        else:
            # No finite loss was recorded (diverged at iter 0): fail-soft,
            # return the last-good design (== init_lat) with no score.
            ret_latent = last_good
            score = None
        return loss_history, ret_latent, score

    all_histories: list[list[float]] = []
    results = []  # (loss_history, ret_latent, score)
    for init_lat in latent_inits:
        lh, rl, sc = _run_one(init_lat)
        all_histories.append(lh)
        results.append((lh, rl, sc))

    eligible = [
        (i, r) for i, r in enumerate(results)
        if r[2] is not None and math.isfinite(r[2])
    ]
    if eligible:
        best_start, (best_lh, best_lat, _score) = min(
            eligible, key=lambda t: t[1][2]
        )
        return best_lat, best_lh, all_histories, best_start
    if len(latent_inits) == 1:
        # Single-start fail-soft: preserve the legacy warn-and-return-last-good
        # contract (tests/unit/api/test_result_finite_guard.py).
        lh, rl, _sc = results[0]
        return rl, lh, all_histories, 0
    raise RuntimeError(
        f"optimize multi-start: all {len(latent_inits)} starts produced a "
        "non-finite loss at every iterate; no usable optimum."
    )


def _reference_topology_optimize_04b960c1c(
    sim: Simulation,
    design_region: TopologyDesignRegion,
    objective: Callable,
    *,
    n_iterations: int = 100,
    learning_rate: float = 0.01,
    beta_schedule: list[tuple[int, float]] | None = None,
    init_density: jnp.ndarray | None = None,
    verbose: bool = True,
    skip_preflight: bool = False,
    jit: bool = True,
) -> TopologyResult:
    """Run density-based topology optimization.

    Optimizes a continuous density field within the design region to
    minimize an objective function, using gradient information from
    ``jax.value_and_grad`` through the full FDTD pipeline.

    Uses Adam (rfx's own update) for the optimizer.

    Parameters
    ----------
    sim : Simulation
        Base simulation (the design region overrides eps_r in its box).
    design_region : TopologyDesignRegion
        Region and material specification.
    objective : callable(Result) -> scalar
        Objective function to minimize.  Must be JAX-differentiable
        through the simulation.
    n_iterations : int
        Number of optimization iterations.
    learning_rate : float
        Learning rate for Adam optimizer.
    beta_schedule : list of (iteration, beta) or None
        Schedule for increasing projection sharpness.
        Default: [(0, 1), (30, 4), (60, 16), (80, 64)]
    init_density : array or None
        Initial density field.  If None, initialized to 0.5 everywhere
        (uniform mixture of bg and fg materials).
    verbose : bool
        Print progress every 10 iterations.

    jit : bool
        Default True compiles the loss and gradient once across iterations
        and beta values. First-call host-read tracing errors fall back with
        one warning. ``jit=False`` selects the eager loop without that warning.

    Returns
    -------
    TopologyResult
        Contains final density, permittivity, loss history, and beta history.
    """
    resolve_field_dtype(sim._precision)
    sim._require_uniform_mesh("topology_optimize")
    sim._auto_preflight(skip=skip_preflight, context="topology_optimize")
    # Refuse unsupported surface-impedance sheets at entry.
    from rfx.materials.thin_conductor import refuse_f0_sheets as _refuse_f0
    from rfx.model.wall_loss import refuse_wall_loss
    refuse_wall_loss(sim, "topology-optimization")
    _refuse_f0(sim._thin_conductors, "topology-optimization")
    from rfx.runners._admission import admit
    admit(sim, "topology_optimize")
    # Refuse refinement before constructing the optimization forward.
    sim._require_no_refinement_without_a_subgrid("topology_optimize()")

    # Build grid and compute design region indices
    grid = sim._build_grid()
    from rfx._internal._periodic import interval_indices
    lo_idx, hi_idx = map(list, interval_indices(grid, design_region.corner_lo, design_region.corner_hi))

    # Clamp indices to the interior region (exclude CPML padding).
    # Without this, a design region at the domain edge can overlap
    # with CPML cells, causing shape mismatches and incorrect gradients.
    # Per-face pads: a PMC/PEC/periodic face has pad=0 on its
    # side, so the design can touch that reflector.
    pads_lo = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)
    pads_hi = (grid.pad_x_hi, grid.pad_y_hi, grid.pad_z_hi)
    dims = (grid.nx, grid.ny, grid.nz)
    for d in range(3):
        lo_idx[d] = max(lo_idx[d], pads_lo[d])
        hi_idx[d] = min(hi_idx[d], dims[d] - 1 - pads_hi[d])
    lo_idx = tuple(lo_idx)
    hi_idx = tuple(hi_idx)

    design_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))
    if any(s <= 0 for s in design_shape):
        raise ValueError(
            f"Design region is empty after clamping to interior "
            f"(lo_idx={lo_idx}, hi_idx={hi_idx}). Ensure the design "
            f"region does not lie entirely within the CPML boundary."
        )

    # The design arrays reach the solve whole-grid and traced, so the
    # current-moment monitor's guard checks the cells they can change: the
    # design region's bounds. Run the guard here, before tracing the forward.
    design_region_cells = SimpleNamespace(bounds=(
        lo_idx[0], hi_idx[0] + 1, lo_idx[1], hi_idx[1] + 1,
        lo_idx[2], hi_idx[2] + 1))
    from rfx.current_moments import monitor_for_simulation
    monitor_for_simulation(sim, grid, sim._periodic_flags(),
                           overrides={"design_region": design_region_cells})

    from rfx.model.thin_conductors import refuse_design_films
    refuse_design_films(sim, grid, tuple(v for a, b in zip(lo_idx, hi_idx) for v in (a, b + 1)))

    if beta_schedule is None:
        beta_schedule = _DEFAULT_BETA_SCHEDULE

    # Resolve material permittivities and conductivities.
    # _resolve_material checks user-registered materials (sim._materials)
    # first, then falls back to MATERIAL_LIBRARY.
    try:
        mat_bg = sim._resolve_material(design_region.material_bg)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    try:
        mat_fg = sim._resolve_material(design_region.material_fg)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    eps_bg = mat_bg.eps_r
    eps_fg = mat_fg.eps_r
    sigma_bg = mat_bg.sigma
    sigma_fg = mat_fg.sigma
    pec_threshold = sim._PEC_SIGMA_THRESHOLD
    bg_is_pec = sigma_bg >= pec_threshold
    fg_is_pec = sigma_fg >= pec_threshold

    # Compute filter radius in cells
    filt_r = design_region.effective_filter_radius
    filter_radius_cells = None
    if filt_r is not None:
        filter_radius_cells = filt_r / grid.dx

    _topo_pec_sheets: list = []
    _topo_pec_wires: list = []
    base_materials, debye_spec, lorentz_spec, base_pec_mask, *_ = sim._assemble_materials(
        grid, pec_sheets=_topo_pec_sheets, pec_wires=_topo_pec_wires)
    base_mu_r = base_materials.mu_r

    # Initialize density
    if init_density is None:
        init_density = 0.5 * jnp.ones(design_shape, dtype=jnp.float32)

    # We optimize in logit space for unconstrained optimization,
    # then map back via sigmoid to keep density in [0, 1].
    # logit(0.5) = 0
    init_logit = jnp.log(init_density / (1.0 - init_density + 1e-8) + 1e-8)

    # Adam moments in logit space.
    m = v = jnp.zeros_like(init_logit)

    logit = init_logit
    history = []
    beta_history = []

    n_steps = grid.num_timesteps(num_periods=20.0)

    def forward(logit_param, beta):
        """Forward pass: logit -> material fields -> simulation -> objective."""
        rho = jax.nn.sigmoid(logit_param)
        fields = density_to_material_fields(
            rho,
            eps_bg,
            eps_fg,
            filter_radius_cells=filter_radius_cells,
            beta=beta,
            sigma_bg=sigma_bg,
            sigma_fg=sigma_fg,
            pec_bg=bg_is_pec,
            pec_fg=fg_is_pec,
        )

        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        from rfx.model.overrides import apply_material_overrides
        from rfx.core.yee import MaterialArrays
        from rfx.model.occupancy import occupancy_from_design
        window = (slice(si, ei + 1), slice(sj, ej + 1), slice(sk, ek + 1))
        # Retain the existing fresh-container boundary (including absent records).
        cells = MaterialArrays(eps_r=base_materials.eps_r, sigma=base_materials.sigma,
                               mu_r=base_mu_r, sigma_film=base_materials.sigma_film)
        materials = apply_material_overrides(
            cells, eps_override=fields.eps.astype(base_materials.eps_r.dtype),
            sigma_override=fields.sigma.astype(base_materials.sigma.dtype),
            window=(slice(None),) * 3, box_local=window)._replace(mu_r=base_mu_r)
        pec_occupancy = occupancy_from_design(grid.shape, window, fields.pec_occupancy)

        result = sim._forward_from_materials(
            grid,
            materials,
            debye_spec,
            lorentz_spec,
            n_steps=n_steps,
            checkpoint=True,
            pec_mask=base_pec_mask,
            pec_sheets=tuple(_topo_pec_sheets),
            pec_wires=tuple(_topo_pec_wires),
            pec_occupancy=pec_occupancy,
            # The design arrays are whole-grid and traced; the monitor's
            # guard checks the region they can change instead.
            monitor_overrides={"design_region": design_region_cells},
        )
        import inspect
        sig = inspect.signature(objective)
        if 'ntff_box' in sig.parameters:
            return objective(result, ntff_box=result.ntff_box)
        return objective(result)

    grad_fn = _value_and_grad_with_fallback(forward, jit, "topology_optimize")
    last_good = logit  # last design with a confirmed-finite forward+gradient
    for it in range(n_iterations):
        beta = _get_beta(it, beta_schedule)
        beta_history.append(beta)

        loss, grad = grad_fn(logit, jnp.asarray(beta, dtype=jnp.float32))

        # NaN/Inf gradient guard (see optimize() for rationale): a non-finite
        # gradient would silently poison ``logit`` via the Adam update and
        # every later iteration. Stop fast with a cause hint and RESTORE the
        # last design whose forward was finite (the current ``logit`` is the
        # one that just diverged).
        if not (bool(jnp.isfinite(loss)) and bool(jnp.all(jnp.isfinite(grad)))):
            import warnings as _w
            _w.warn(
                f"topology_optimize: non-finite loss/gradient (NaN or Inf) at "
                f"iteration {it} — the FDTD forward almost certainly diverged "
                f"(dt above CFL, conformal=True at fine dx, PEC inside CPML, or "
                f"a sub-cell PEC feature). Stopping early and returning the last "
                f"finite design (from before iteration {it}).", AccuracyWarning,
                stacklevel=2,
            )
            logit = last_good
            break

        last_good = logit  # this design produced a finite forward+gradient
        loss_val = float(loss)
        history.append(loss_val)

        step, m, v = _adam_update(m, v, grad, it, learning_rate, 0.9, 0.999, 1e-8)
        logit = logit - step

        if verbose and (it % 10 == 0 or it == n_iterations - 1):
            rho_raw = jax.nn.sigmoid(logit)
            rho_proj = apply_projection(rho_raw, beta)
            # Binarization: 0 = fully binary, 1 = maximally intermediate
            binarization = float(jnp.mean(4.0 * rho_proj * (1.0 - rho_proj)))
            print(
                f"  iter {it:4d}  loss = {loss_val:.6e}  "
                f"beta = {beta:.1f}  binarization = {binarization:.3f}"
            )

    # Final density and eps
    final_rho = jax.nn.sigmoid(logit)
    final_fields = density_to_material_fields(
        final_rho,
        eps_bg,
        eps_fg,
        filter_radius_cells=filter_radius_cells,
        beta=_get_beta(n_iterations - 1, beta_schedule),
        sigma_bg=sigma_bg,
        sigma_fg=sigma_fg,
        pec_bg=bg_is_pec,
        pec_fg=fg_is_pec,
    )
    final_eps = final_fields.eps

    # Compute projected density for reporting
    rho_filtered = final_rho
    if filter_radius_cells is not None and filter_radius_cells >= 0.5:
        rho_filtered = apply_density_filter(final_rho, filter_radius_cells)
    rho_projected = apply_projection(
        rho_filtered, _get_beta(n_iterations - 1, beta_schedule),
    )

    # The optimization solves through _forward_from_materials, which never reaches
    # _dispatch_plan, so it releases the inspection products its preflight cached, as the
    # calculators that bypass dispatch do.
    release_inspection(sim)
    return TopologyResult(
        density=final_rho,
        density_projected=rho_projected,
        eps_design=final_eps,
        history=history,
        beta_history=beta_history,
        pec_occupancy_design=final_fields.pec_occupancy,
    )


@pytest.fixture(autouse=True)
def _cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _tiny_sim(dx=0.003):
    sim = Simulation(freq_max=5e9, domain=(0.012,) * 3, dx=dx, boundary="pec")
    sim.add_port((0.003, 0.006, 0.006), "ez")
    sim.add_probe((0.009, 0.006, 0.006), "ez")
    return sim


def _tiny_region():
    return DesignRegion((0.006, 0.003, 0.003), (0.009, 0.009, 0.009), (1.0, 4.4))


def _obj(result):
    return -jnp.sum(result.time_series ** 2)


def _equal(actual, reference):
    assert np.array_equal(np.asarray(actual), np.asarray(reference))


def _equal_optimize(actual, reference):
    assert len(actual.loss_history) == len(reference.loss_history) == 3
    _equal(actual.loss_history, reference.loss_history)
    _equal(actual.latent, reference.latent)
    _equal(actual.eps_design, reference.eps_design)
    # A finite, moving design makes equality a meaningful arithmetic witness.
    assert np.all(np.isfinite(actual.latent))
    assert np.any(np.asarray(actual.latent) != 0)


def _compare_multistart(cost, init, **kwargs):
    actual_count, reference_count = [0], [0]
    actual = optimize_module._adam_multistart(cost, [init], it_count=actual_count, **kwargs)
    reference = _reference_adam_multistart_04b960c1c(
        cost, [init], it_count=reference_count, **kwargs)
    for a, b in zip(actual, reference):
        _equal(a, b)
    assert actual_count == reference_count == [kwargs["n_iters"]]
    assert np.any(np.asarray(actual[0]) != np.asarray(init))


def test_multistart_eager_step_clamp():
    init = jnp.asarray([0.5, -0.3, 1.2], dtype=jnp.float32)
    target = jnp.asarray([1.0, -1.0, 0.5], dtype=jnp.float32)
    _compare_multistart(lambda x: jnp.sum((x - target) ** 2), init,
                       n_iters=3, lr=0.1, step_clamp=0.05, jit=False)


def test_multistart_jitted_fdtd():
    sim = _tiny_sim()
    shape = sim._build_grid().shape

    def cost(x):
        eps = 1.0 + 3.4 * jax.nn.sigmoid(x)
        return _obj(sim.forward(eps_override=jnp.broadcast_to(eps, shape),
                                n_steps=20, skip_preflight=True))

    _compare_multistart(cost, jnp.asarray(0.2, dtype=jnp.float32),
                       n_iters=3, lr=0.1, jit=True)


@pytest.mark.parametrize("jit,n_starts", [(True, 1), (False, 1), (True, 2)],
                         ids=["jit", "eager", "multistart"])
def test_optimize(monkeypatch, jit, n_starts):
    kwargs = dict(n_iters=3, n_steps=20, lr=0.2, jit=jit, n_starts=n_starts,
                  seed=7, verbose=False, skip_preflight=True)
    actual = optimize(_tiny_sim(), _tiny_region(), _obj, **kwargs)
    monkeypatch.setattr(optimize_module, "_adam_multistart",
                        _reference_adam_multistart_04b960c1c)
    reference = optimize(_tiny_sim(), _tiny_region(), _obj, **kwargs)
    _equal_optimize(actual, reference)


def test_progressive_two_stages(monkeypatch):
    schedule = [ProgressiveStage(dx=dx, n_iters=3, lr=0.1, n_steps=20)
                for dx in (0.003, 0.002)]
    kwargs = dict(verbose=False, skip_preflight=True, jit=True)
    actual = progressive_optimize(_tiny_sim, _tiny_region(), _obj, schedule, **kwargs)
    monkeypatch.setattr(optimize_module, "_adam_multistart",
                        _reference_adam_multistart_04b960c1c)
    reference = progressive_optimize(_tiny_sim, _tiny_region(), _obj, schedule, **kwargs)
    _equal(actual.loss_history, reference.loss_history)
    _equal(actual.final_latent, reference.final_latent)
    _equal(actual.final_eps_design, reference.final_eps_design)
    assert actual.stage_boundaries == reference.stage_boundaries == [0, 3, 6]
    assert actual.stages[0].latent.shape != actual.stages[1].latent.shape
    for a, b in zip(actual.stages, reference.stages):
        _equal_optimize(a, b)


def test_topology_beta_restart(monkeypatch):
    # topology has no n_steps argument. Limit both real FDTD solves identically;
    # the copied function and all of its setup remain verbatim.
    grid_type = type(_tiny_sim()._build_grid())
    monkeypatch.setattr(grid_type, "num_timesteps", lambda self, num_periods=20.0: 20)
    region = TopologyDesignRegion(_tiny_region().corner_lo, _tiny_region().corner_hi,
                                  material_bg="air", material_fg="fr4")
    kwargs = dict(n_iterations=3, learning_rate=0.05,
                  beta_schedule=[(0, 1.0), (2, 4.0)], verbose=False,
                  skip_preflight=True, jit=True)
    from rfx.optimizers import _Adam
    restart = _Adam.restart
    restarts = []

    def observed_restart(self, state):
        result = restart(self, state)
        assert result is state
        restarts.append(state.it)
        return result

    monkeypatch.setattr(_Adam, "restart", observed_restart)
    actual = topology_optimize(_tiny_sim(), region, _obj, **kwargs)
    reference = _reference_topology_optimize_04b960c1c(_tiny_sim(), region, _obj, **kwargs)
    assert restarts == [2]
    assert actual.beta_history == reference.beta_history == [1.0, 1.0, 4.0]
    assert len(actual.history) == len(reference.history) == 3
    for field in ("history", "density", "density_projected", "eps_design",
                  "pec_occupancy_design"):
        _equal(getattr(actual, field), getattr(reference, field))
    assert np.any(np.asarray(actual.density) != 0.5)
