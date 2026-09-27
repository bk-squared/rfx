"""How every time-stepping path treats every ``Simulation`` attribute.

A ``Simulation`` is a declaration: materials, conductors, ports, sources,
boundaries, a mesh and observers. Several paths step it in time, and each one
reads only part of the declaration. An input a path does not read is solved as
if it had not been declared, and nothing in the result says so: a Kerr block
comes back linear, a microstrip port launches nothing, a periodic wall is
solved as a metal one. The PI's rule (2026-09-24) is that a physics input a
path does not implement is refused before the first step. This table records,
for every attribute and every path, which of these holds:

``carries``        the path solves it;
``refuses``        the path raises before stepping (``raises`` is a fragment
                   of the message; ``declared`` when the declaration itself is
                   refused);
``falls back``     the path hands the model to another path (named);
``not reachable``  no model with the attribute reaches the path (why);
``ignorable``      bookkeeping or a setting the path does not need (why);
``not yet classified``  only on the calculator columns, see below.

A cell whose behaviour today contradicts its disposition names the open issue
in ``wrong``; its executable check is a strict expected failure until the
issue is closed. For such a cell the disposition is the rule's minimum
(``refuses``) when today's behaviour is a drop, and ``carries`` when the path
solves the input but gets it wrong.

Where the cells are checked
---------------------------
* ``tests/contracts/test_path_disposition.py``: every attribute of a fresh
  ``Simulation`` has a row, every row has a cell on every column, the lane
  columns are the lanes ``_dispatch_plan`` returns plus the declared kernel
  routes, and every function in ``rfx/`` that reaches a time-stepping kernel
  or steps fields is registered.
* ``tests/unit/runners/test_path_disposition_cells.py``: every row that has a
  model there (the physics and observer rows, and the ``_mode`` and
  ``_precision`` settings, which have known-wrong cells). ``refuses`` builds
  the model, then expects a raise naming ``raises`` before any kernel scan
  starts; ``carries`` checks that the attribute changes the result (and,
  where cheap, that the lane agrees with ``sim.run()``); ``falls back`` checks
  the warning and the result of the named path.

The other settings and the bookkeeping rows (``ROW_CLASS``) carry only this
entry. Arguments to ``run()`` and ``forward()`` are not attributes; #1297's
``tests/unit/runners/test_silent_drop_warnings.py`` covers them.

Columns
-------
The first eight lane columns are the lanes ``Simulation._dispatch_plan``
selects: ``run()`` on a uniform, graded, subgridded, ADI and multi-device
model, and ``forward()`` on a uniform, graded and distributed graded model.
One lane token reaches two kernels: ``forward()`` on a ``solver='adi'`` model
is ``fwd_uniform`` to ``_dispatch_plan``, and ``_forward_from_materials``
sends it to the ADI kernel. That route is its own column, ``fwd_adi``
(``KERNEL_ROUTES``). This table replaces the multi-device-only table of #1241
(``tests/unit/runners/test_distributed_admission_refusals.py``); its column
is ``run_distributed``.

Seven more entry points step a ``Simulation`` without ``_dispatch_plan``:
each assembles the model itself and calls a kernel. Their columns are
``not yet classified`` in this table; a follow-up classifies them.

* ``s_matrix_scan``: the lumped/wire S-matrix of ``run(compute_s_params=True)``
  (``compute_lumped_wire_s_matrix_via_scan`` → ``_forward_from_materials``).
* ``mixed_s_matrix``: ``compute_mixed_s_matrix`` → ``_forward_from_materials``.
* ``topology_optimize``: → ``_forward_from_materials``.
* ``waveguide_s_matrix``: ``compute_waveguide_s_matrix``; uniform mesh through
  ``rfx.sources.waveguide_port.extract_*`` → ``rfx.simulation.run``, graded
  mesh through ``run_nonuniform_path``.
* ``coax_calculators``: ``compute_coaxial_line_reflection``,
  ``compute_coaxial_two_port``, ``compute_coax_msl_transition`` →
  ``rfx.simulation.run``.
* ``vmap_sweep_batched``: the batched kernel of ``vmap_material_sweep``
  (its own scan; its sequential fallback calls ``run()``).
* ``material_fit``: ``differentiable_material_fit`` → ``rfx.simulation.run``.

Folded into the lane columns, because they call ``sim.run()`` or
``sim.forward()`` on the caller's model and add no assembly of their own:
``parametric_sweep``, ``ntff_sweep``, ``run_batch`` and
``run_batch_with_manifest``, ``config.runner.run_and_save``,
``convergence_study``, the ``RISUnitCell`` sweeps, the diagnostics smoke run,
``optimize`` (``forward(eps_override=...)``), ``profile_forward``, the
sequential fallback of ``vmap_material_sweep``, ``compute_s_matrix`` (it
dispatches to the calculators above or to ``run()``), ``compute_msl_s_matrix``
(``forward()`` or ``run()`` per drive) and ``run(ringdown=...)``.

Not ``Simulation`` paths, because they take a ``Grid``: ``rfx.run``,
``rfx.run_until_decay``, ``compute_rcs``, ``gpu.benchmark``, the eager
``extract_s_matrix`` / ``extract_s_matrix_wire`` and
``rfx.subgridding.runner.run_subgridded``.
"""

from __future__ import annotations

from typing import NamedTuple

CARRIES = "carries"
REFUSES = "refuses"
FALLS_BACK = "falls back"
NOT_REACHABLE = "not reachable"
IGNORABLE = "ignorable"
UNCLASSIFIED = "not yet classified"
KINDS = (CARRIES, REFUSES, FALLS_BACK, NOT_REACHABLE, IGNORABLE, UNCLASSIFIED)

LANES = (
    "run_uniform", "run_nonuniform", "run_subgridded", "run_adi",
    "run_distributed", "fwd_uniform", "fwd_nonuniform", "fwd_distributed_nu",
    "fwd_adi",
)
# Columns _dispatch_plan reports under another column's lane token.
KERNEL_ROUTES = {
    "fwd_adi": ("fwd_uniform", "solver='adi' sends _forward_from_materials to _run_adi_from_materials"),
}
CALCULATORS = (
    "s_matrix_scan", "mixed_s_matrix", "topology_optimize",
    "waveguide_s_matrix", "coax_calculators", "vmap_sweep_batched",
    "material_fit",
)
PATHS = LANES + CALCULATORS

# A row's class says what it declares. Physics and observer rows have
# executable cells; a setting row has them only where it has a model in the
# cells test (because one of its cells is known wrong).
PHYSICS, OBSERVER, SETTING, BOOKKEEPING = "physics", "observer", "setting", "bookkeeping"


class Cell(NamedTuple):
    kind: str
    note: str = ""
    wrong: str = ""      # open issue while today's behaviour contradicts ``kind``
    to: str = ""         # the path a ``falls back`` cell hands the model to
    raises: str = ""     # a fragment of the refusal's message
    declared: bool = False  # refused when the model is declared, not when it runs


def carries(note="", *, wrong=""):
    return Cell(CARRIES, note, wrong)


def refuses(note="", *, raises="", wrong="", declared=False):
    return Cell(REFUSES, note, wrong, raises=raises, declared=declared)


def falls_back(to, note=""):
    return Cell(FALLS_BACK, note, to=to)


def not_reachable(why):
    return Cell(NOT_REACHABLE, why)


def ignorable(why):
    return Cell(IGNORABLE, why)


def lanes(*, run_uniform, run_nonuniform, run_subgridded, run_adi,
          run_distributed, fwd_uniform, fwd_nonuniform, fwd_distributed_nu,
          fwd_adi):
    """One cell per lane; a missing lane is a TypeError at import."""
    return dict(run_uniform=run_uniform, run_nonuniform=run_nonuniform,
                run_subgridded=run_subgridded, run_adi=run_adi,
                run_distributed=run_distributed, fwd_uniform=fwd_uniform,
                fwd_nonuniform=fwd_nonuniform,
                fwd_distributed_nu=fwd_distributed_nu, fwd_adi=fwd_adi)


# Only a bookkeeping row may give one cell for every path, present and future:
# the completeness test accepts ``"*"`` for an ``ignorable`` cell only.
def every_path(cell):
    return {"*": cell}


# ---------------------------------------------------------------- refusals
# One helper per refusal that recurs; each names a fragment of its message.

def _subgrid(code, note="production validation"):
    return refuses(f"{note} ({code})", raises=f"[{code}]")


def _adi(what, fragment):
    return refuses(f"ADI refuses {what}", raises=fragment)


_SHEET_OWNS_NO_CELL = "a sheet or wire owns no cell, and this lane applies PEC from a cell mask (#931)"
SUBGRID_SHEETS = refuses(_SHEET_OWNS_NO_CELL, raises="does not realize PEC sheets or wires")
DIST_SHEETS = refuses(_SHEET_OWNS_NO_CELL, raises="does not realize declared PEC SHEETS")
DIST_FWD_SHEETS = refuses(_SHEET_OWNS_NO_CELL, raises="forward lane does not realize PEC sheets")
ADI_INTERIOR_PEC = _adi("interior PEC", "adi_interior_pec_unsupported")
ADI_THIN = _adi("thin-conductor corrections", "does not support thin-conductor corrections")
ADI_SOFT_SOURCES = _adi("every port but add_source() soft sources", "supports only add_source()-style soft sources")
ADI_DISPERSIVE = _adi("dispersive materials", "does not support dispersive materials")
ADI_RLC = _adi("lumped RLC", "does not support lumped RLC")
ADI_PORTS = _adi("waveguide and Floquet ports", "does not support waveguide or Floquet ports")
ADI_PER_FACE = refuses("ADI stamps one absorber on all six faces; a per-face layout is refused",
                       raises="supports only a uniform absorber", declared=True)
ADI_GRADED = refuses("ADI requires a uniform mesh", raises="solver='adi' does not support nonuniform",
                     declared=True)
F0 = "surface_impedance_f0) thin conductors are not supported"
DIST_FWD_PORTS = refuses("lumped and wire ports are refused on distributed forward()",
                         raises="Lumped / wire ports (impedance > 0) are not yet supported")
RUN_COAX = refuses("run() refuses add_coaxial_port(); the coax calculators solve it",
                   raises="add_coaxial_port() is not wired into Simulation.run()")
FWD_COAX = refuses("forward() refuses add_coaxial_port()",
                   raises="add_coaxial_port() is not wired into Simulation.forward()")
GRADED_REFINEMENT = refuses("refinement on a graded mesh refused (#1282)",
                            raises="asks for subgridding on a non-uniform mesh")


def _conformal(lane_words):
    return refuses("Boundary(conformal=True) is refused like conformal_pec=True: this lane has no "
                   "Dey-Mittra update (#1297)", raises=f"the {lane_words} does not implement")


CONFORMAL = dict(
    run_nonuniform=_conformal("non-uniform mesh lane"),
    run_subgridded=_conformal("subgridded (SBP-SAT) lane"),
    run_adi=_conformal("ADI (solver='adi') lane"),
    run_distributed=_conformal("distributed multi-device lane"),
    fwd_uniform=_conformal("uniform forward lane"),
    fwd_nonuniform=_conformal("non-uniform forward lane"),
    fwd_distributed_nu=_conformal("distributed non-uniform forward lane"),
    fwd_adi=_conformal("uniform forward lane"),
)


TABLE: dict[str, dict[str, dict[str, Cell]]] = {
    # ------------------------------------------------------------ settings
    "_freq_max": {"": lanes(
        run_uniform=carries("sets the step count from num_periods, the auto mesh and the default S band"),
        run_nonuniform=carries("as run_uniform"),
        run_subgridded=carries("as run_uniform; the count is scaled by the ratio"),
        run_adi=carries("as run_uniform"),
        run_distributed=carries("as run_uniform"),
        fwd_uniform=carries("as run_uniform"),
        fwd_nonuniform=carries("as run_uniform"),
        fwd_distributed_nu=carries("as run_uniform"),
        fwd_adi=carries("as run_uniform"),
    )},
    "_domain": {"": lanes(
        run_uniform=carries("the grid extent"),
        run_nonuniform=carries("the grid extent; a profile must sum to it"),
        run_subgridded=carries("the coarse grid extent"),
        run_adi=carries("the grid extent"),
        run_distributed=carries("the grid extent, padded to split along x"),
        fwd_uniform=carries("the grid extent"),
        fwd_nonuniform=carries("the grid extent"),
        fwd_distributed_nu=carries("the grid extent"),
        fwd_adi=carries("the grid extent"),
    )},
    "_dx": {"": lanes(
        run_uniform=carries("the cell"),
        run_nonuniform=carries("the cell of every axis without a profile"),
        run_subgridded=carries("the coarse cell; the fine cell is dx/ratio"),
        run_adi=carries("the cell"),
        run_distributed=carries("the cell"),
        fwd_uniform=carries("the cell"),
        fwd_nonuniform=carries("the cell of every axis without a profile"),
        fwd_distributed_nu=carries("the cell of every axis without a profile"),
        fwd_adi=carries("the cell"),
    )},
    "_dt_pin": {"": lanes(
        run_uniform=not_reachable("the constructor refuses it without a dx/dy/dz profile, and a "
                                  "profile sends the model to the graded lane"),
        run_nonuniform=carries("the graded grid's time step"),
        run_subgridded=not_reachable("dt= needs a profile, and a refinement on a profiled mesh "
                                     "is refused (#1282)"),
        run_adi=not_reachable("dt= needs a profile, and ADI requires a uniform mesh"),
        run_distributed=carries("the distributed graded grid's time step (same probe as run())"),
        fwd_uniform=not_reachable("dt= needs a profile, which sends the model to the graded lane"),
        fwd_nonuniform=carries("the graded grid's time step"),
        fwd_distributed_nu=carries("the distributed graded grid's time step"),
        fwd_adi=not_reachable("dt= needs a profile, and ADI requires a uniform mesh"),
    )},
    "_dt_min_cell": {"": lanes(
        run_uniform=not_reachable("the constructor refuses it without dt="),
        run_nonuniform=carries("with dt=, read by the graded grid build"),
        run_subgridded=not_reachable("the constructor refuses it without dt=, and dt= needs a profile"),
        run_adi=not_reachable("the constructor refuses it without dt=, and dt= needs a profile"),
        run_distributed=carries("with dt=, read by _build_nonuniform_grid"),
        fwd_uniform=not_reachable("the constructor refuses it without dt="),
        fwd_nonuniform=carries("with dt=, read by the graded grid build"),
        fwd_distributed_nu=carries("with dt=, read by _build_nonuniform_grid"),
        fwd_adi=not_reachable("the constructor refuses it without dt=, and dt= needs a profile"),
    )},
    # Run with precision='mixed', which moves the record 8.5e-2 from float32
    # on the Yee lanes; float64 needs jax x64.
    "_precision": {"": lanes(
        run_uniform=carries("field dtype. Without jax x64, 'float64' runs float32 with JAX's "
                            "truncation warning"),
        run_nonuniform=refuses("non-float32 refused (#630)", raises="(issue #630)"),
        run_subgridded=refuses("non-float32 refused (#630)", raises="(issue #630)"),
        run_adi=refuses("today 'mixed' and 'float64' run float32 without a word; with x64 on, "
                        "'float64' dies in the ADI scan with a TypeError", wrong="#1308"),
        run_distributed=refuses("non-float32 refused (#630)", raises="(issue #630)"),
        fwd_uniform=carries("field dtype"),
        fwd_nonuniform=refuses("non-float32 refused (#630)", raises="(issue #630)"),
        fwd_distributed_nu=refuses("non-float32 refused (#630)", raises="(issue #630)"),
        fwd_adi=refuses("today runs float32 without a word", wrong="#1308"),
    )},
    "_solver": {"": lanes(
        run_uniform=not_reachable("solver='adi' sends run() to run_adi"),
        run_nonuniform=refuses("solver='adi' requires a uniform mesh (_dispatch_plan)"),
        run_subgridded=refuses("run() dispatches solver='adi' before the refinement, and ADI refuses subgridding"),
        run_adi=carries("the lane solver='adi' selects"),
        run_distributed=refuses("solver='adi' does not support distributed execution"),
        fwd_uniform=not_reachable("solver='adi' sends forward() to the fwd_adi route"),
        fwd_nonuniform=refuses("solver='adi' requires a uniform mesh (_dispatch_plan)"),
        fwd_distributed_nu=refuses("solver='adi' requires a uniform mesh (_dispatch_plan)"),
        fwd_adi=carries("the route solver='adi' selects (same probe as run())"),
    )},
    "_adi_cfl_factor": {"": lanes(
        run_uniform=ignorable("setting of solver='adi'"),
        run_nonuniform=ignorable("setting of solver='adi'"),
        run_subgridded=ignorable("setting of solver='adi'"),
        run_adi=carries("the ADI step is the Yee step times this factor"),
        run_distributed=ignorable("setting of solver='adi'"),
        fwd_uniform=ignorable("setting of solver='adi'"),
        fwd_nonuniform=ignorable("setting of solver='adi'"),
        fwd_distributed_nu=ignorable("setting of solver='adi'"),
        fwd_adi=carries("the ADI step is the Yee step times this factor"),
    )},
    "_stencil_order": {"": lanes(
        run_uniform=carries("order 4 on the plain uniform PEC/periodic path"),
        run_nonuniform=refuses("stencil_order=4 refused (_check_stencil_order_supported)"),
        run_subgridded=refuses("stencil_order=4 refused"),
        run_adi=refuses("stencil_order=4 refused"),
        run_distributed=refuses("stencil_order=4 refused"),
        fwd_uniform=carries("order 4 on the plain uniform PEC/periodic path"),
        fwd_nonuniform=refuses("stencil_order=4 refused"),
        fwd_distributed_nu=refuses("stencil_order=4 refused"),
        fwd_adi=refuses("stencil_order=4 refused"),
    )},
    # Run as a 2d_tmz model against the same box in 3d.
    "_mode": {"": lanes(
        run_uniform=carries("3d, 2d_tmz and 2d_tez"),
        run_nonuniform=refuses(
            "today dropped: 2d_tmz on one z cell between PEC walls is solved as that 3-D box "
            "(the record equals mode='3d' bit for bit), which differs from run_uniform's 2-D solve "
            "by 0.34 of the probe peak", wrong="#1340"),
        run_subgridded=_subgrid("z_slab_requires_guarded_boundary",
                                "a slab across a one-cell z domain is never one-sided"),
        run_adi=carries("3d and 2d_tmz; 2d_tez refused. A 3d box one z cell thick dies with an "
                        "IndexError (measured), so its model is three cells thick"),
        run_distributed=carries("2d_tmz matched one device bit for bit (measured)"),
        fwd_uniform=carries("3d, 2d_tmz and 2d_tez"),
        fwd_nonuniform=refuses("today dropped, as run_nonuniform", wrong="#1340"),
        fwd_distributed_nu=refuses("today dropped, as run_nonuniform", wrong="#1340"),
        fwd_adi=carries("as run_adi"),
    )},

    # --------------------------------------------------------- bookkeeping
    "_internal_probe_indices": {"": every_path(ignorable("probe bookkeeping"))},
    "_msl_auto_offset_min": {"": every_path(ignorable("MSL port data, read with the port; the _msl_ports row decides it"))},
    "_msl_auto_probe_spacing": {"": every_path(ignorable("MSL port data, read with the port; the _msl_ports row decides it"))},
    "_msl_auto_probe_lengths": {"": every_path(ignorable("MSL port data, read with the port; the _msl_ports row decides it"))},
    "_boundary_model": {"": every_path(ignorable(
        "derived from the boundary declaration; the _boundary, _boundary_spec, _pec_faces "
        "and _periodic_axes rows decide it"))},

    # ----------------------------------------------------------- materials
    "_materials": {
        "eps": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=carries("inside the production envelope: PEC walls, no CPML, a slab "
                                   "touching one z wall"),
            run_adi=carries(),
            run_distributed=carries("each E edge takes ε from the cell that owns it", wrong="#1303"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries("each E edge takes ε from the cell that owns it", wrong="#1303"),
            fwd_adi=carries(),
        ),
        "sigma": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=carries("inside the production envelope"),
            run_adi=carries("implicit conductivity in the ADI solve"),
            run_distributed=carries("each E edge takes σ from the cell that owns it", wrong="#1303"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries("each E edge takes σ from the cell that owns it", wrong="#1303"),
            fwd_adi=carries("implicit conductivity in the ADI solve"),
        ),
        "mu": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=carries("inside the production envelope"),
            run_adi=refuses("today dropped: the ADI kernel takes ε and σ only", wrong="#1308"),
            run_distributed=carries(),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
            fwd_adi=refuses("today dropped: the ADI kernel takes ε and σ only", wrong="#1308"),
        ),
        **{pole: lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=_subgrid("dispersive_or_nonlinear_material",
                                    "production validation; research/off drop it, see _refinement "
                                    "'relaxed_validation'"),
            run_adi=ADI_DISPERSIVE,
            run_distributed=carries("in a CPML box the two-device field grows without bound", wrong="#1302"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
            fwd_adi=ADI_DISPERSIVE,
        ) for pole in ("debye", "lorentz", "drude")},
        "kerr": lanes(
            run_uniform=carries(),
            run_nonuniform=refuses("today dropped: the graded run() solves the material as linear", wrong="#1309"),
            run_subgridded=_subgrid("dispersive_or_nonlinear_material",
                                    "production validation; research/off drop it"),
            run_adi=refuses("today dropped: the ADI kernel has no χ³ term", wrong="#1308"),
            run_distributed=refuses("Kerr χ³ refused (#1214)", raises="Kerr chi3 (nonlinear) material(s)"),
            fwd_uniform=carries(),
            fwd_nonuniform=refuses("Kerr χ³ refused off the uniform forward lane",
                                   raises="forward() supports Kerr"),
            fwd_distributed_nu=refuses("Kerr χ³ refused off the uniform forward lane",
                                       raises="forward() supports Kerr"),
            fwd_adi=refuses("today dropped: forward()'s Kerr guard lets fwd_uniform through and the "
                            "ADI kernel has no χ³ term", wrong="#1308"),
        ),
    },

    # ------------------------------------------------------------ geometry
    "_geometry": {
        "pec_volume": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=carries(),
            run_adi=ADI_INTERIOR_PEC,
            run_distributed=carries("as a cell mask"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries("as a cell mask"),
            fwd_adi=ADI_INTERIOR_PEC,
        ),
        **{shape: lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=SUBGRID_SHEETS,
            run_adi=ADI_INTERIOR_PEC,
            run_distributed=DIST_SHEETS,
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=DIST_FWD_SHEETS,
            fwd_adi=ADI_INTERIOR_PEC,
        ) for shape in ("pec_sheet", "pec_wire")},
    },
    "_thin_conductors": {
        "lossy_sheet": lanes(
            run_uniform=carries("folded into σ"),
            run_nonuniform=carries("folded into σ"),
            run_subgridded=refuses("today dropped although production validation passes", wrong="#1311"),
            run_adi=ADI_THIN,
            run_distributed=carries("folded into σ, which each E edge takes from the cell that owns it", wrong="#1303"),
            fwd_uniform=carries("folded into σ"),
            fwd_nonuniform=carries("folded into σ"),
            fwd_distributed_nu=carries("folded into σ, which each E edge takes from the cell that owns it", wrong="#1303"),
            fwd_adi=ADI_THIN,
        ),
        "pec_sheet": lanes(
            run_uniform=carries("realized as a PEC sheet"),
            run_nonuniform=carries("realized as a PEC sheet"),
            run_subgridded=SUBGRID_SHEETS,
            run_adi=ADI_THIN,
            run_distributed=DIST_SHEETS,
            fwd_uniform=carries("realized as a PEC sheet"),
            fwd_nonuniform=carries("realized as a PEC sheet"),
            fwd_distributed_nu=DIST_FWD_SHEETS,
            fwd_adi=ADI_THIN,
        ),
        "surface_impedance": lanes(
            run_uniform=carries("the per-step node-thin operator (#677)"),
            run_nonuniform=carries("the per-step node-thin operator"),
            run_subgridded=refuses("surface_impedance_f0 sheets refused", raises=F0),
            run_adi=refuses("surface_impedance_f0 sheets refused", raises=F0),
            run_distributed=refuses("surface_impedance_f0 sheets refused", raises=F0),
            fwd_uniform=carries("the per-step node-thin operator"),
            fwd_nonuniform=carries("the per-step node-thin operator"),
            fwd_distributed_nu=refuses("surface_impedance_f0 sheets refused", raises=F0),
            fwd_adi=refuses("surface_impedance_f0 sheets refused", raises=F0),
        ),
    },
    "_pinned_sheets": {"pec_sheet": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=SUBGRID_SHEETS,
        run_adi=ADI_INTERIOR_PEC,
        run_distributed=DIST_SHEETS,
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=DIST_FWD_SHEETS,
        fwd_adi=ADI_INTERIOR_PEC,
    )},

    # --------------------------------------------------- ports and sources
    "_ports": {
        "source": lanes(
            run_uniform=carries("add_source(): a soft point source"),
            run_nonuniform=carries(),
            run_subgridded=carries("inside the refined slab"),
            run_adi=carries(),
            run_distributed=carries(),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
            fwd_adi=carries(),
        ),
        "amplitude_kind": lanes(
            run_uniform=carries("'current' and 'field' scale the waveform differently (#571)"),
            run_nonuniform=carries(),
            run_subgridded=carries(),
            run_adi=refuses("today ignored: both kinds inject the same waveform", wrong="#1308"),
            run_distributed=carries(),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
            fwd_adi=refuses("today ignored: both kinds inject the same waveform", wrong="#1308"),
        ),
        "lumped_port": lanes(
            run_uniform=carries("drive and 50 Ω load"),
            run_nonuniform=carries("the drive is read in other units, 1/dx² of run_uniform's", wrong="#1266"),
            run_subgridded=carries("inside the refined slab"),
            run_adi=ADI_SOFT_SOURCES,
            run_distributed=carries("single-cell excited ports"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries("the drive is read in other units, 1/dx² of run_uniform's", wrong="#1266"),
            fwd_distributed_nu=DIST_FWD_PORTS,
            fwd_adi=ADI_SOFT_SOURCES,
        ),
        "passive_port": lanes(
            run_uniform=carries("the 50 Ω load of excite=False"),
            run_nonuniform=carries(),
            run_subgridded=carries("inside the refined slab"),
            run_adi=ADI_SOFT_SOURCES,
            run_distributed=refuses("excite=False refused (#1241)", raises="(passive port) is not supported"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=DIST_FWD_PORTS,
            fwd_adi=ADI_SOFT_SOURCES,
        ),
        "wire_port": lanes(
            run_uniform=carries("add_port(extent=...): drive and load along the wire"),
            run_nonuniform=carries("the drive is read in other units, 1/dx² of run_uniform's", wrong="#1266"),
            run_subgridded=carries("inside the refined slab"),
            run_adi=ADI_SOFT_SOURCES,
            run_distributed=refuses("extended ports refused (#1241)", raises="(extended lumped port)"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries("the drive is read in other units, 1/dx² of run_uniform's", wrong="#1266"),
            fwd_distributed_nu=DIST_FWD_PORTS,
            fwd_adi=ADI_SOFT_SOURCES,
        ),
    },
    "_msl_ports": {"msl_port": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses("today dropped although production validation passes", wrong="#1311"),
        run_adi=refuses("a board with a PEC or thin-conductor trace is refused; today a port "
                        "declared without a trace is dropped", wrong="#1308"),
        run_distributed=refuses("MSL ports refused (#1241)", raises="add_msl_port() port(s)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses("today the port launches nothing", wrong="#1285"),
        fwd_adi=refuses("as run_adi: a port declared without a trace is dropped", wrong="#1308"),
    )},
    "_waveguide_ports": {"waveguide_port": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=_subgrid("boundary_terminated_requires_pec_no_cpml",
                                "a waveguide port needs a CPML face, which production validation refuses"),
        run_adi=ADI_PORTS,
        run_distributed=falls_back("run_uniform", "one device, with a warning; an explicit "
                                   "conformal_pec=False is overridden there (#1305)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses("waveguide ports refused",
                                   raises="Waveguide ports are not supported on the distributed forward path"),
        fwd_adi=ADI_PORTS,
    )},
    "_coaxial_ports": {"coax_port": lanes(
        run_uniform=RUN_COAX,
        run_nonuniform=RUN_COAX,
        run_subgridded=RUN_COAX,
        run_adi=RUN_COAX,
        run_distributed=RUN_COAX,
        fwd_uniform=FWD_COAX,
        fwd_nonuniform=FWD_COAX,
        fwd_distributed_nu=FWD_COAX,
        fwd_adi=FWD_COAX,
    )},
    "_floquet_ports": {
        "floquet_port": lanes(
            run_uniform=carries("at normal incidence"),
            run_nonuniform=refuses("today launches nothing on a dx/dy profile; a dz profile is refused", wrong="#1312"),
            run_subgridded=_subgrid("subgrid_overlaps_absorber",
                                    "a Floquet cell absorbs on z, which production validation refuses"),
            run_adi=ADI_PORTS,
            run_distributed=refuses("the periodic axes it sets are refused (#1241)",
                                    raises="periodic / Bloch boundaries are not supported"),
            fwd_uniform=carries("at normal incidence"),
            fwd_nonuniform=refuses("today launches nothing on a dx/dy profile", wrong="#1312"),
            fwd_distributed_nu=refuses("today launches nothing on a dx/dy profile", wrong="#1312"),
            fwd_adi=ADI_PORTS,
        ),
        "scan_angle": lanes(
            run_uniform=refuses("today dropped: 30° gives the record of 0°, the angle never "
                                "reaches the fields", wrong="#1221"),
            run_nonuniform=refuses("today the port launches nothing at any angle", wrong="#1312"),
            run_subgridded=_subgrid("subgrid_overlaps_absorber",
                                    "a Floquet cell absorbs on z, which production validation refuses"),
            run_adi=ADI_PORTS,
            run_distributed=refuses("the periodic axes it sets are refused (#1241)",
                                    raises="periodic / Bloch boundaries are not supported"),
            fwd_uniform=refuses("today dropped: 30° gives the record of 0°", wrong="#1221"),
            fwd_nonuniform=refuses("today the port launches nothing at any angle", wrong="#1312"),
            fwd_distributed_nu=refuses("today the port launches nothing at any angle", wrong="#1312"),
            fwd_adi=ADI_PORTS,
        ),
    },
    "_lumped_rlc": {element: lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=_subgrid("rlc_unvalidated", "production validation; research/off drop it, "
                                "see _refinement 'relaxed_validation'"),
        run_adi=ADI_RLC,
        run_distributed=refuses("lumped RLC refused (#1239)", raises="add_lumped_rlc() element(s)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses("lumped RLC refused", raises="Lumped RLC ports are not yet supported"),
        fwd_adi=ADI_RLC,
    ) for element in ("R", "series_RL")},
    "_tfsf": {"plane_wave": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=_subgrid("subgrid_overlaps_absorber",
                                "a plane wave needs CPML, which production validation refuses"),
        run_adi=_adi("TFSF", "does not support TFSF sources"),
        run_distributed=falls_back("run_uniform", "one device, with a warning; an explicit "
                                   "conformal_pec=False is overridden there (#1305)"),
        fwd_uniform=carries(),
        fwd_nonuniform=refuses("TFSF refused off the uniform forward lane",
                               raises="Differentiable TFSF plane-wave forward is supported only"),
        fwd_distributed_nu=refuses("TFSF refused",
                                   raises="TFSF sources are not supported on the distributed forward path"),
        fwd_adi=_adi("TFSF", "does not support TFSF sources"),
    )},
    "_refinement": {
        "slab": lanes(
            run_uniform=not_reachable("a refinement sends run() on a uniform mesh to run_subgridded"),
            run_nonuniform=GRADED_REFINEMENT,
            run_subgridded=carries("the SBP-SAT fine slab"),
            run_adi=_adi("subgridding", "does not support subgridding"),
            run_distributed=refuses("refinement refused (#1241)", raises="add_refinement() (subgridding)"),
            fwd_uniform=refuses("forward() has no subgridded lane (#1282)", raises="is refused on forward()"),
            fwd_nonuniform=GRADED_REFINEMENT,
            fwd_distributed_nu=GRADED_REFINEMENT,
            fwd_adi=_adi("subgridding", "does not support subgridding"),
        ),
        "relaxed_validation": {
            **{lane: not_reachable("validation modes exist only on the subgridded lane; see 'slab'")
               for lane in ("run_uniform", "run_nonuniform", "run_adi", "run_distributed",
                            "fwd_uniform", "fwd_nonuniform", "fwd_distributed_nu", "fwd_adi")},
            "run_subgridded": refuses(
                "validation='research' and 'off' run the Debye, Lorentz and Drude poles, Kerr "
                "χ³ and lumped RLC that production refuses, and drop them", wrong="#1286"),
        },
    },

    # ---------------------------------------------------------- boundaries
    "_boundary": {
        "cpml": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=_subgrid("subgrid_overlaps_absorber", "no CPML in the guarded envelope"),
            run_adi=carries("a graded conductivity layer, not a CPML (test_realized_boundary.py "
                            "cpml--adi)", wrong="#1221"),
            run_distributed=carries(),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
            fwd_adi=carries("the same conductivity layer as run_adi", wrong="#1221"),
        ),
        "upml": lanes(
            run_uniform=carries(),
            run_nonuniform=refuses("the graded runner implements CPML only",
                                   raises="boundary='upml' does not support the non-uniform run() lane"),
            run_subgridded=_subgrid("subgrid_overlaps_absorber", "no absorber in the guarded envelope"),
            run_adi=refuses("boundary='upml' refused", raises="solver='adi' does not support boundary='upml'",
                            declared=True),
            run_distributed=refuses("boundary='upml' refused",
                                    raises="boundary='upml' does not support distributed execution"),
            fwd_uniform=carries(),
            fwd_nonuniform=refuses("the graded runner implements CPML only",
                                   raises="boundary='upml' does not support the non-uniform forward() lane"),
            fwd_distributed_nu=refuses(
                "the graded runner implements CPML only",
                raises="boundary='upml' does not support the distributed non-uniform forward() lane"),
            fwd_adi=refuses("boundary='upml' refused", raises="solver='adi' does not support boundary='upml'",
                            declared=True),
        ),
    },
    "_pec_faces": {"pec_face": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=_subgrid("boundary_terminated_requires_pec_no_cpml", "CPML on the other faces"),
        run_adi=ADI_PER_FACE,
        run_distributed=carries("the absorber backing differs from one device "
                                "(test_realized_boundary.py pec-zlo--distributed)", wrong="#1221"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=carries(),
        fwd_adi=ADI_PER_FACE,
    )},
    # Magnetic walls on the x faces, electric on y and z.
    "_boundary_spec": {
        "pmc_face": lanes(
            run_uniform=carries("the magnetic wall sits half a cell inside its face "
                                "(test_realized_boundary.py pmc-pec--run)", wrong="#1221"),
            run_nonuniform=carries("half a cell inside (pmc-pec--nonuniform)", wrong="#1221"),
            run_subgridded=refuses("today solved as electric walls although production validation "
                                   "passes", wrong="#1311"),
            run_adi=refuses("today solved as electric walls (pmc-pec--adi)", wrong="#1221"),
            run_distributed=carries("tangential E held at zero on the x faces "
                                    "(pmc-pec--distributed)", wrong="#1221"),
            fwd_uniform=carries("half a cell inside (pmc-pec--forward)", wrong="#1221"),
            fwd_nonuniform=carries("the kernel of run_nonuniform; test_realized_boundary.py does not "
                                   "run this entry"),
            fwd_distributed_nu=carries("test_realized_boundary.py does not run this entry"),
            fwd_adi=refuses("today solved as electric walls, as run_adi", wrong="#1221"),
        ),
        "conformal": lanes(
            run_uniform=carries("Dey-Mittra weights on every PEC shape"),
            **CONFORMAL,
        ),
        "conformal_s_matrix": lanes(
            run_uniform=refuses("today dropped: the lumped-port S-matrix run() returns comes from "
                                "_forward_from_materials, which staircases", wrong="#1299"),
            **CONFORMAL,
        ),
    },
    "_periodic_axes": {"periodic": lanes(
        run_uniform=carries("one cell longer than declared (test_realized_boundary.py "
                            "periodic-xy--run)", wrong="#1221"),
        run_nonuniform=refuses("today solved as PEC walls (periodic-xy--nonuniform)", wrong="#1221"),
        run_subgridded=refuses("today solved as PEC walls although production validation passes",
                               wrong="#1311"),
        run_adi=_adi("periodic axes", "does not support manual periodic axes"),
        run_distributed=refuses("periodic axes refused (#1241)",
                                raises="periodic / Bloch boundaries are not supported"),
        fwd_uniform=carries("one cell longer than declared (periodic-xy--forward)", wrong="#1221"),
        fwd_nonuniform=refuses("today solved as PEC walls", wrong="#1221"),
        fwd_distributed_nu=refuses("today solved as PEC walls", wrong="#1221"),
        fwd_adi=_adi("periodic axes", "does not support manual periodic axes"),
    )},
    "_cpml_layers": {"layers": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=_subgrid("subgrid_overlaps_absorber", "no CPML in the guarded envelope"),
        run_adi=carries("the thickness of ADI's conductivity layer, checked in the σ handed to the kernel"),
        run_distributed=carries(),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=carries(),
        fwd_adi=carries("as run_adi"),
    )},
    "_cpml_kappa_max": {"kappa": lanes(
        run_uniform=carries(),
        run_nonuniform=refuses("today dropped: the graded grid build is never given it", wrong="#1310"),
        run_subgridded=_subgrid("subgrid_overlaps_absorber", "no CPML in the guarded envelope"),
        run_adi=refuses("today dropped: ADI's absorber is not a CPML (cpml--adi)", wrong="#1221"),
        run_distributed=carries(),
        fwd_uniform=carries(),
        fwd_nonuniform=refuses("today dropped", wrong="#1310"),
        fwd_distributed_nu=refuses("today dropped", wrong="#1310"),
        fwd_adi=refuses("today dropped, as run_adi", wrong="#1221"),
    )},
    # An εr 4 block offset by half a cell, 'dual_average' against 'sampled'.
    "_interface_eps": {"dual_average": lanes(
        run_uniform=refuses("today not read: the field is the 'sampled' one", wrong="#1339"),
        run_nonuniform=carries("read by the graded assembly (assemble_interface_eps_nu); no lane to "
                               "compare with"),
        run_subgridded=refuses("today not read", wrong="#1339"),
        run_adi=refuses("today not read", wrong="#1339"),
        run_distributed=refuses("'dual_average' refused",
                                raises="interface_eps='dual_average' is not supported on the distributed lane"),
        fwd_uniform=refuses("today not read", wrong="#1339"),
        fwd_nonuniform=carries("read by the graded assembly"),
        fwd_distributed_nu=refuses("'dual_average' refused",
                                   raises="interface_eps='dual_average' cannot combine with"),
        fwd_adi=refuses("today not read", wrong="#1339"),
    )},

    # -------------------------------------------------------- mesh profiles
    **{profile: {"graded": lanes(
        run_uniform=not_reachable("a dx/dy/dz profile sends the model to the graded lane"),
        run_nonuniform=carries(),
        run_subgridded=GRADED_REFINEMENT,
        run_adi=ADI_GRADED,
        run_distributed=carries("the distributed graded runner, grading up to 5:1"),
        fwd_uniform=not_reachable("a dx/dy/dz profile sends the model to the graded lane"),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=carries(),
        fwd_adi=ADI_GRADED,
    )} for profile in ("_dx_profile", "_dy_profile", "_dz_profile")},

    # ------------------------------------------------------------ observers
    "_probes": {"probe": lanes(
        run_uniform=carries("every effect cell reads it"),
        run_nonuniform=carries(),
        run_subgridded=carries(),
        run_adi=carries(),
        run_distributed=carries(),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=carries(),
        fwd_adi=carries(),
    )},
    "_dft_planes": {"dft_plane": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=_subgrid("dft_plane_unvalidated"),
        run_adi=_adi("DFT planes", "does not support DFT plane probes"),
        run_distributed=refuses("DFT planes refused (#579)",
                                raises="add_dft_plane_probe() is not supported on the distributed multi-device"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses("DFT planes refused (#579)",
                                   raises="add_dft_plane_probe() is not supported on the distributed non-uniform"),
        fwd_adi=_adi("DFT planes", "does not support DFT plane probes"),
    )},
    "_flux_monitors": {"flux": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses("today result.flux_monitors is None", wrong="#1313"),
        run_adi=refuses("today result.flux_monitors is None", wrong="#1313"),
        run_distributed=refuses("flux monitors refused (#1241)", raises="add_flux_monitor() (flux monitors)"),
        fwd_uniform=refuses("today ForwardResult has no flux field", wrong="#1313"),
        fwd_nonuniform=refuses("today ForwardResult has no flux field", wrong="#1313"),
        fwd_distributed_nu=refuses("flux monitors refused",
                                   raises="add_flux_monitor() is not supported on the distributed non-uniform"),
        fwd_adi=refuses("today ForwardResult has no flux field", wrong="#1313"),
    )},
    "_ntff": {"ntff_box": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=carries("inside the refined slab, clear of its artificial interface"),
        run_adi=_adi("NTFF", "does not support NTFF accumulation"),
        run_distributed=refuses("NTFF refused (#1241)", raises="add_ntff_box() (NTFF box)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses("today ntff_data is None", wrong="#1313"),
        fwd_adi=_adi("NTFF", "does not support NTFF accumulation"),
    )},
    # Checked in a CPML box: the monitor refuses PEC domain faces on every lane.
    "_current_moments": {"block_moments": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=refuses("the subgridded runner does not accumulate them",
                               raises="add_current_moment_monitor() is not supported on the subgridded lane"),
        run_adi=refuses("the ADI kernel does not accumulate them",
                        raises="add_current_moment_monitor() is not supported on the ADI lane"),
        run_distributed=refuses("the distributed runner does not accumulate them",
                                raises="add_current_moment_monitor() is not supported on the distributed (v2)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=refuses(
            "the distributed graded kernel does not accumulate them",
            raises="add_current_moment_monitor() is not supported on the distributed non-uniform"),
        fwd_adi=refuses("the ADI kernel does not accumulate them",
                        raises="add_current_moment_monitor() is not supported on the ADI lane"),
    )},
}

ROW_CLASS: dict[str, str] = {
    **{attr: SETTING for attr in (
        "_freq_max", "_domain", "_dx", "_dt_pin", "_dt_min_cell", "_precision",
        "_solver", "_adi_cfl_factor", "_stencil_order", "_mode")},
    **{attr: BOOKKEEPING for attr in (
        "_internal_probe_indices", "_msl_auto_offset_min",
        "_msl_auto_probe_spacing", "_msl_auto_probe_lengths", "_boundary_model")},
    **{attr: OBSERVER for attr in (
        "_probes", "_dft_planes", "_flux_monitors", "_ntff", "_current_moments")},
    **{attr: PHYSICS for attr in (
        "_materials", "_geometry", "_thin_conductors", "_pinned_sheets",
        "_ports", "_msl_ports", "_waveguide_ports", "_coaxial_ports",
        "_floquet_ports", "_lumped_rlc", "_tfsf", "_refinement", "_boundary",
        "_pec_faces", "_boundary_spec", "_periodic_axes", "_cpml_layers",
        "_cpml_kappa_max", "_interface_eps", "_dx_profile", "_dy_profile",
        "_dz_profile")},
}

# Settings with a model in the cells test, because a cell of theirs is known wrong.
EXECUTABLE_SETTINGS = ("_mode", "_precision")


def executable(attr: str) -> bool:
    """Whether the cells test runs this row's cells."""
    return ROW_CLASS[attr] in (PHYSICS, OBSERVER) or attr in EXECUTABLE_SETTINGS


# The calculator columns are classified in a follow-up. Found by reading, not
# run: topology_optimize and differentiable_material_fit discard Kerr χ³ from
# _assemble_materials; topology_optimize, s_matrix_scan and mixed_s_matrix skip
# forward()'s refusals, which may let coax ports and surface-impedance sheets
# through as well as #1299's conformal walls; differentiable_material_fit never
# reads TFSF, waveguide or Floquet ports, per-face boundaries, the solver or
# stencil_order. Known issues on these columns are named in their cells.
CALCULATOR_NOTES: dict[tuple[str, str], str] = {
    ("_boundary_spec", "s_matrix_scan"): "#1299: conformal walls staircased",
    ("_boundary_spec", "mixed_s_matrix"): "#1299: conformal walls staircased",
    ("_boundary_spec", "topology_optimize"): "#1299: conformal walls staircased",
    ("_solver", "waveguide_s_matrix"): "#1300: solver='adi' runs Yee",
    ("_dx", "vmap_sweep_batched"): "#1293: on an auto mesh each value runs its own dt for one step count (sequential path)",
    ("_ports", "material_fit"): "#1290: a wire port's extent and plain sources are dropped",
    ("_msl_ports", "material_fit"): "#1290: never read",
    ("_lumped_rlc", "material_fit"): "#1290: never read",
    **{("_refinement", calc): "refused (#1282)" for calc in (
        "s_matrix_scan", "topology_optimize", "waveguide_s_matrix",
        "vmap_sweep_batched", "material_fit")},
}


def cell(attr: str, feature: str, path: str) -> Cell:
    """The cell for ``path``; a calculator column is ``not yet classified``."""
    row = TABLE[attr][feature]
    if "*" in row:
        return row["*"]
    if path in CALCULATORS:
        return Cell(UNCLASSIFIED, CALCULATOR_NOTES.get((attr, path), ""))
    return row[path]
