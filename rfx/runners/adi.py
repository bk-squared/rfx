"""_run_adi_from_materials moved from rfx/api/_execute.py in Block E1, a pure move; the body is unchanged."""
from __future__ import annotations

from rfx.adi import ADIState2D, run_adi_2d
from rfx.api._spec import ForwardResult, Result
from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.model import conductors as _conductors
from rfx import _grid_metric, _realized
import jax.numpy as jnp


def _run_adi_from_materials(
    self,
    grid: Grid,
    materials: MaterialArrays,
    debye_spec,
    lorentz_spec,
    *,
    diagnostics=(),
    n_steps: int,
    pec_mask: jnp.ndarray | None = None,
    pec_sheets: object = (),
    pec_wires: object = (),
    return_state: bool = True,
    lane: str,
    conformal_pec=None,
):
    """Run the integrated ADI solver path (2D TMz or 3D).

        #931 §1.7: the ADI lane takes the SAME realized PEC edge masks as
        every other lane instead of zeroing E at the occupied cell
        indices.  In 2-D it applies the ``Mz`` plane to Ez; in 3-D the
        three masks componentwise.

        ``lane`` is ``"run_adi"`` from ``run()`` and ``"fwd_adi"`` from the
        ADI route of ``_forward_from_materials``; a declared input that lane
        does not carry is refused before the first step
        (``rfx.runners._admission``).
        """
    import copy

    from rfx._adi_notice import warn_experimental
    warn_experimental()
    from rfx.sources.wire_radius import require_radius_update
    require_radius_update(materials, lane="ADI", unsupported=True)
    self._validate_adi_configuration(materials, debye_spec, lorentz_spec)

    from rfx.boundaries.pec import realized_pec_edge_masks as _rpem_adi
    from rfx.simulation import resolve_periodic as _resolve_periodic_adi
    pec_sheets = tuple(pec_sheets or ())
    pec_wires = tuple(pec_wires or ())
    _adi_periodic = _resolve_periodic_adi(
        grid,
        tuple(a in self._periodic_axes for a in "xyz")
        if self._periodic_axes else None)
    pec_edge_masks = None
    if pec_mask is not None or pec_sheets or pec_wires:
        pec_edge_masks = _rpem_adi(
            pec_mask, sheets=pec_sheets, wires=pec_wires,
            periodic=_adi_periodic)

    # Refuse after realization: a sheet/wire must reach this lane even
    # though it owns no volume cell. This is independent of preflight.
    _conductors.admit_adi_sources(self, grid, pec_edge_masks)

    # Every declared input this lane does not carry is refused here,
    # after the specific refusals above and before the first step.
    # A declared material interface is one (#1373, LANE_GATES).
    from rfx.runners._admission import admit
    admit(
        self,
        lane,
        run_args={"conformal_pec": conformal_pec},
        diagnostics=diagnostics,
    )

    # The arrays this lane steps can differ from the declaration: a
    # concrete eps_override / sigma_override. Read them too, before the
    # absorber's grading is added to sigma; a traced one is not inspected.
    from rfx.adi import adi_material_interface_refusal, adi_traced_override_refusal
    _interface = adi_material_interface_refusal(materials.eps_r, materials.sigma)
    if _interface is not None:
        raise NotImplementedError(_interface)
    # A traced ARRAY cannot be read; a traced scalar is a homogeneous
    # fill (forward(eps_override=<0-d>)) and is broadcast to the grid.
    for _arr in (materials.eps_r, materials.sigma):
        _traced = adi_traced_override_refusal(_arr)
        if _traced is not None:
            raise NotImplementedError(_traced)
    materials = materials._replace(
        eps_r=jnp.broadcast_to(materials.eps_r, grid.shape),
        sigma=jnp.broadcast_to(materials.sigma, grid.shape))

    dt = float(grid.dt * self._adi_cfl_factor)
    from rfx.model.source_coefficients import field_source_samples

    grid_out = copy.copy(grid)
    grid_out.dt = dt

    # ---- 3D path ----
    if self._mode == "3d":
        from rfx.adi import run_adi_3d, ADIState3D

        probes_3d = []
        for pe in self._probes:
            i, j, k = _grid_metric.field_index(grid, pe.position, pe.component)
            probes_3d.append((i, j, k, pe.component))

        eps_r_3d = materials.eps_r
        sigma_3d = materials.sigma

        sources_3d = []
        for pe in self._ports:
            i, j, k = _grid_metric.field_index(grid, pe.position, pe.component)
            waveform = field_source_samples(pe.waveform, n_steps, dt)
            sources_3d.append((i, j, k, pe.component, waveform))

        shape = grid.shape
        if _realized.ACTIVE is not None:
            _realized.sources(grid_out, materials, sources_3d, "adi.sources")
        zeros = jnp.zeros(shape, dtype=jnp.float32)
        ex_f, ey_f, ez_f, hx_f, hy_f, hz_f, probe_data = run_adi_3d(
            zeros, zeros, zeros, zeros, zeros, zeros,
            eps_r_3d, sigma_3d,
            dt, grid.dx, grid.dx, grid.dx,
            n_steps,
            sources=sources_3d,
            probes=probes_3d,
            pec_edge_masks=pec_edge_masks,
        )
        if probe_data is None:
            probe_data = jnp.zeros((n_steps, 0), dtype=jnp.float32)

        if return_state:
            state = ADIState3D(
                ex=ex_f, ey=ey_f, ez=ez_f,
                hx=hx_f, hy=hy_f, hz=hz_f,
                step=jnp.asarray(n_steps, dtype=jnp.int32),
            )
            return Result(
                diagnostics=diagnostics,
                state=state,
                time_series=probe_data,
                s_params=None,
                freqs=None,
                grid=grid_out,
                dt=dt,
                freq_range=(self._freq_max / 10, self._freq_max, self._boundary),
            )
        return ForwardResult(
            diagnostics=diagnostics,
            time_series=probe_data,
            ntff_data=None,
            ntff_box=None,
            grid=grid_out,
            dt=dt,
        )

    # ---- 2D TMz path ----
    probes = []
    for pe in self._probes:
        i, j, _ = _grid_metric.field_index(grid, pe.position, pe.component)
        probes.append((i, j, pe.component))

    # Coefficient derivatives use eps_r units (#1357), including 2-D ADI.
    eps_r_2d = materials.eps_r[:, :, 0]
    sigma_2d = materials.sigma[:, :, 0]

    sources = []
    for pe in self._ports:
        i, j, _ = _grid_metric.field_index(grid, pe.position, pe.component)
        waveform = field_source_samples(pe.waveform, n_steps, dt)
        sources.append((i, j, waveform))

    # The 2-D TMz lane carries only Ez, so its realized PEC mask is
    # the Mz plane (#931 §1.7).
    ez_pec_mask_2d = None
    if pec_edge_masks is not None:
        ez_pec_mask_2d = pec_edge_masks[2][:, :, 0]
    ez0 = jnp.zeros_like(eps_r_2d)
    hx0 = jnp.zeros_like(eps_r_2d)
    hy0 = jnp.zeros_like(eps_r_2d)
    ez_f, hx_f, hy_f, probe_data = run_adi_2d(
        ez0,
        hx0,
        hy0,
        eps_r_2d,
        sigma_2d,
        dt,
        grid.dx,
        grid.dx,
        n_steps,
        sources=sources,
        probes=probes,
        ez_pec_mask=ez_pec_mask_2d,
    )
    if probe_data is None:
        probe_data = jnp.zeros((n_steps, 0), dtype=ez_f.dtype)

    if return_state:
        state = ADIState2D(
            ez=ez_f,
            hx=hx_f,
            hy=hy_f,
            step=jnp.asarray(n_steps, dtype=jnp.int32),
        )
        return Result(
            diagnostics=diagnostics,
            state=state,
            time_series=probe_data,
            s_params=None,
            freqs=None,
            grid=grid_out,
            dt=dt,
            freq_range=(self._freq_max / 10, self._freq_max, self._boundary),
        )

    return ForwardResult(
        diagnostics=diagnostics,
        time_series=probe_data,
        ntff_data=None,
        ntff_box=None,
        grid=grid_out,
        dt=dt,
    )


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly (as rfx/preflight/sources.py
# does). The function above was a ``def`` in ``_ExecuteMixin``'s body, so its
# qualname read ``_ExecuteMixin._run_adi_from_materials``; ``rfx/api/__init__.py``
# rewrites exactly ``<mixin>.<name>`` -> ``Simulation.<name>`` at class
# composition and skips a bare name, so without this line a TypeError would no
# longer name ``Simulation._run_adi_from_materials()``.
# ``tests/contracts/test_execute_split_qualnames.py`` pins it.
# ---------------------------------------------------------------------------
_run_adi_from_materials.__qualname__ = "_ExecuteMixin._run_adi_from_materials"
