"""Pack the prepared graded forward execution without entry-point transport."""
from __future__ import annotations

import jax.numpy as jnp
from rfx.api._dispatch import _declaration_setup_at_trace_time
from rfx.api._spec import ForwardResult


@_declaration_setup_at_trace_time
def _forward_nonuniform_from_materials(
    self,
    *,
    subpixel_smoothing=False,
    diagnostics=(),
    preflight=None,
    conductors=None,
    eps_override: jnp.ndarray | None = None,
    sigma_override: jnp.ndarray | None = None,
    pec_mask_override: jnp.ndarray | None = None,
    pec_occupancy_override: jnp.ndarray | None = None,
    pinned_sheet_layout: tuple | None = None,
    design_box: object | None = None,
    n_steps: int,
    port_s11_freqs: object | None = None,
    checkpoint: bool = True,
    emit_time_series: bool = True,
    checkpoint_every: int | tuple[int, ...] | None = None,
    n_warmup: int = 0,
) -> ForwardResult:
    """Differentiable forward on the non-uniform mesh path.

    Routes through ``run_nonuniform_path`` with optimisation overrides
    applied after material assembly, then repackages the returned
    ``Result`` into the minimal ``ForwardResult`` schema via the shared
    :meth:`_pack_nu_forward_result` lane helper.

    When ``checkpoint`` is True (the default), the NU scan body is
    wrapped in ``jax.checkpoint`` so reverse-mode AD memory scales
    with ``sqrt(n_steps)`` instead of ``n_steps``.
    """
    # Defense-in-depth: the NU runner does not honour stencil_order.
    self._check_stencil_order_supported()
    from rfx.runners.nonuniform import run_nonuniform_path

    # A grid-shaped override is (cells + 1) per axis, because N cells are
    # bounded by N+1 E-nodes — the same shape the uniform path has
    # always wanted. Caller code that sized arrays from the PROFILE length
    # is one sample short, and without this check the mismatch surfaces
    # deep inside XLA as an opaque broadcast TypeError.
    _overrides = (("eps_override", eps_override),
                  ("sigma_override", sigma_override),
                  ("pec_mask_override", pec_mask_override),
                  ("pec_occupancy_override", pec_occupancy_override))
    # Build the grid ONLY when there is an array to check, so the common
    # override-free forward() does not pay for a second grid build
    # (negligible against the solve, but free to skip).
    if any(_a is not None and hasattr(_a, "shape") for _n, _a in _overrides):
        _grid = self._build_nonuniform_grid()
        for _name, _arr in _overrides:
            if _arr is None or not hasattr(_arr, "shape"):
                continue
            if tuple(_arr.shape) == tuple(_grid.shape):
                continue
            raise ValueError(
                f"{_name} has shape {tuple(_arr.shape)} but this "
                f"non-uniform grid is {tuple(_grid.shape)}. Grid-shaped "
                f"arrays are one sample larger per axis than the cell "
                f"counts / profile lengths you requested: N cells are "
                f"bounded by N+1 field nodes. Build the array from "
                f"the grid shape rather than from the profile length."
            )

    result = run_nonuniform_path(
        self,
        subpixel_smoothing=subpixel_smoothing,
        preflight=preflight,
        conductors=conductors,
        diagnostics=diagnostics,
        n_steps=n_steps,
        s_param_freqs=(
            None if port_s11_freqs is None else jnp.asarray(port_s11_freqs, dtype=jnp.float32)
        ),
        eps_override=eps_override,
        sigma_override=sigma_override,
        pec_mask_override=pec_mask_override,
        pec_occupancy_override=pec_occupancy_override,
        pinned_sheet_layout=pinned_sheet_layout,
        design_box=design_box,
        checkpoint=checkpoint,
        emit_time_series=emit_time_series,
        checkpoint_every=checkpoint_every,
        n_warmup=n_warmup,
        lane="fwd_nonuniform",
    )
    return self._pack_nu_forward_result(
        realized_geometry=result.realized_geometry,
        diagnostics=result.diagnostics,
        time_series=result.time_series,
        ntff_data=result.ntff_data,
        ntff_box=result.ntff_box,
        grid=result.grid,
        s_params=getattr(result, "s_params", None),
        freqs=getattr(result, "freqs", None),
        dft_planes=getattr(result, "dft_planes", None),
        flux_monitors=result.flux_monitors,
        waveguide_ports=result.waveguide_ports,
        waveguide_sparams=result.waveguide_sparams,
        **{k: getattr(result, k, None) for k in ("wire_port_sparams", "lumped_port_sparams")},
        sparam_time_records=getattr(result, "sparam_time_records", None),
        sparam_read_ids=getattr(result, "sparam_read_ids", None),
        dft_time_records=getattr(result, "dft_time_records", None),
        dt=getattr(result, "dt", None),
        current_moment_data=getattr(result, "current_moment_data", None),
        current_moment_monitor=getattr(result, "current_moment_monitor", None),
    )


_forward_nonuniform_from_materials.__qualname__ = "_ExecuteMixin._forward_nonuniform_from_materials"
_forward_nonuniform_from_materials.__wrapped__.__qualname__ = "_ExecuteMixin._forward_nonuniform_from_materials"
