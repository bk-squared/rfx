"""Host-owned static state for a differentiable forward ring-down result."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rfx.ringdown import RingdownReport


def report_context(plan, result, bins):
    """End the plan's ownership of the realized traced mesh.

    All probe positions were checked against live widths by the run's node
    callbacks. The lazy report only needs the declared nominal mesh. Keep
    the live grid and metrics on the result's children / completion path.

    The context also records, as static host data, which axes were traced
    (``traced_axes``: ``(axis, store)`` pairs) and which port-meta leaves were
    traced. The caller passes the live arrays ``getattr(result.grid, store)``
    of those axes to the result as the child ``cells``; after the transform
    they are the realized widths of that design step, and the report
    rebuilds the realized grid and metas from them.
    """
    import jax
    from rfx.core.jax_utils import is_tracer

    grid = result.grid
    metas = tuple(meta for meta, _ in (result.wire_port_sparams or ()))
    traced_axes = ()
    ports = ()
    meta_traced = None
    with jax.ensure_compile_time_eval():
        if plan.lane == "graded":
            from rfx.model.graded_grid import host_nominal_grid, _profile_to_inv_arrays
            nominal = host_nominal_grid(grid)
            fields = {}
            # host_nominal_grid supplies positions and widths; its inverse
            # metrics still belong to the realized mesh, so rebuild those too.
            for axis, store in zip("xyz", ("dx_arr", "dy_arr", "dz")):
                if is_tracer(getattr(grid, store)):
                    traced_axes += ((axis, store),)
                    fields[f"inv_d{axis}"], fields[f"inv_d{axis}_h"] = _profile_to_inv_arrays(
                        getattr(nominal, store), periodic=axis in grid.periodic_axes)
            if is_tracer(grid.dt):
                fields["dt"] = None
            grid = nominal._replace(**fields) if fields else nominal
            if any(is_tracer(x) for x in jax.tree_util.tree_leaves(metas)):
                from rfx.measurement.graded_port_meta import _build_wp_meta
                ports = tuple(dict(mid_i=m[0], mid_j=m[1], mid_k=m[2], component=m[3],
                                   impedance=m[4], excite=m[7], direction=m[8], live_cells=m[13])
                              for m in metas)
                meta_traced = tuple(is_tracer(x) for x in jax.tree_util.tree_leaves(metas))
                rebuilt = tuple(_build_wp_meta(ports, grid))
                metas = _replace_traced_slots(metas, rebuilt, meta_traced)
    plan.grid = grid
    if is_tracer(plan.dt):
        plan.dt = None
    step = getattr(result, "dt", None)
    return _ForwardReportContext(plan, metas, grid, bins, None if is_tracer(step) else step,
                                 traced_axes=traced_axes, ports=ports, meta_traced=meta_traced)


def _replace_traced_slots(metas, rebuilt, traced):
    """``metas`` with each leaf flagged in ``traced`` taken from ``rebuilt``."""
    import jax
    old, definition = jax.tree_util.tree_flatten(metas)
    new = definition.flatten_up_to(rebuilt)
    return definition.unflatten([n if t else o for o, n, t in zip(old, new, traced, strict=True)])


class _ForwardReportContext:
    """What the lazy report of a ``forward(ringdown=...)`` needs besides arrays."""

    def __init__(self, plan, metas, grid, bins, step=None, *, traced_axes=(), ports=(),
                 meta_traced=None):
        self.plan = plan
        self.metas = tuple(metas)
        self.grid = grid
        self.bins = np.asarray(bins, dtype=np.float64)
        self.step = step            # the solver's own dt (ForwardResult.dt)
        self.solver_step_traced = grid.dt is None
        # static: (axis, store) of each axis whose cells were traced, in the
        # order of the result's ``_cells`` child; the port dicts and the
        # per-leaf traced flags of the metas, to rebuild them on the realized mesh
        self.traced_axes = tuple(traced_axes)
        self.ports = tuple(ports)
        self.meta_traced = meta_traced

    def _realized(self, res):
        """The template grid and metas with the traced cells set to ``res._cells``."""
        import jax.numpy as jnp
        from rfx.model.graded_grid import _profile_to_inv_arrays

        cells = tuple(res._cells)
        if len(cells) != len(self.traced_axes):
            raise ValueError(
                f"ringdown.report: this result carries {len(cells)} realized cell "
                f"arrays, its context expects {len(self.traced_axes)}")
        template = self.grid
        exact_of = {"dx_arr": "dx_arr_f64", "dy_arr": "dy_arr_f64", "dz": "dz_f64"}
        fields = {}
        for (axis, store), live in zip(self.traced_axes, cells):
            dtype = getattr(template, store).dtype
            widths = np.asarray(live).astype(dtype)
            fields[store] = jnp.asarray(widths, dtype=dtype)
            if exact_of[store] in template._fields:
                fields[exact_of[store]] = np.asarray(widths, dtype=np.float64)
            # as the grid builder calls it, on the axis's padded widths
            fields[f"inv_d{axis}"], fields[f"inv_d{axis}_h"] = _profile_to_inv_arrays(
                widths, periodic=axis in template.periodic_axes)
        grid = template._replace(**fields)
        metas = self.metas
        if self.meta_traced is not None:
            from rfx.measurement.graded_port_meta import _build_wp_meta
            metas = _replace_traced_slots(
                metas, tuple(_build_wp_meta(self.ports, grid)), self.meta_traced)
        return grid, tuple(metas)

    def report(self, res) -> RingdownReport:
        from dataclasses import replace
        from types import SimpleNamespace

        from rfx.ringdown import RingdownWitness, CONSISTENCY_BAR, _status_reason

        if self.solver_step_traced:
            raise ValueError(
                "This result's solver step was traced; ringdown.report needs a "
                "concrete forward of the same model. pole() and settling_db "
                "are available.")
        plan, grid, metas = self.plan, self.grid, self.metas
        if self.traced_axes:
            # the run's accumulators were made on the realized mesh of this
            # design step; read them with that mesh's grid and port weights
            import copy
            grid, metas = self._realized(res)
            plan = copy.copy(plan)      # the stored plan keeps the template grid
            plan.grid = grid
        s_plain = np.asarray(res._s_plain)
        if plan.lane == "uniform":
            # run()'s container for the uniform lane's single wire port
            # (rfx/runners/uniform.py): (1, 1, nf) complex64, whatever the
            # accumulator dtype (complex128 under x64)
            s_plain = s_plain.reshape(1, 1, -1).astype(np.complex64)
        result_like = SimpleNamespace(
            wire_port_sparams=tuple(zip(metas, res._accs)), s_params=s_plain,
            grid=grid, freqs=self.bins, dt=self.step)
        S_host, report, nc = plan._host_report(result_like, np.asarray(res._channels), 0)
        from rfx.measurement.port_s import observed_port_s
        pairs = tuple(zip(metas, res._accs))
        S_traced = np.asarray(observed_port_s(res.s_params, pairs)).astype(np.complex128)
        c_now = np.asarray(res._consistency)
        status = np.asarray(res._status).ravel()
        nan_note = ""
        if not np.all(np.isfinite(S_traced)):
            nan_note = "the traced completion is NaN: " + "; ".join(
                f"{w}: {_status_reason(code, c_now, kept_count=count, pole_budget=budget)}"
                for w, code, count, budget in zip(
                    ("[window_start T, T]", "[window_start/2 T, T]"), status,
                    np.asarray(res._pole_counts), np.asarray(res._pole_budgets))
                if int(code) != 0) if np.any(status != 0) else "the traced completion is NaN"
        tol = float(plan.spec.witness_tol)
        rule = ("max |S| difference between the traced completion (the one jax.grad "
                "differentiates, in the working precision) and the float64 completion "
                "run(ringdown=...) gives of the same record")
        if nc is not None:
            w = RingdownWitness("traced", math.nan, tol, False, rule,
                                (nan_note + "; " if nan_note else "") + "no float64 "
                                f"completion to compare: {nc.name} failed on the host",
                                judged=False)
        else:
            S_ref = np.asarray(observed_port_s(S_host, pairs)).astype(np.complex128)
            d = float(np.max(np.abs(S_traced - S_ref)))
            w = RingdownWitness("traced", d, tol, d <= tol, rule, nan_note)
        c = float(c_now)
        wc = RingdownWitness(
            "consistency", c, math.nan if CONSISTENCY_BAR is None else CONSISTENCY_BAR,
            True if CONSISTENCY_BAR is None else c <= CONSISTENCY_BAR,
            "max over the port channels, inside the traced program, of |tail-free "
            "DFT of the rebuilt V_port or I - the run's own accumulator| / its peak",
            "information (not applied)" if CONSISTENCY_BAR is None else "",
            judged=CONSISTENCY_BAR is not None)
        return replace(report, witnesses=report.witnesses + (w, wc))
