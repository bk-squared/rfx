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
``ignorable``      bookkeeping, a setting the path does not need, or an
                   observer a forward() lane returns no field for (why);
``not yet classified``  rejected by the completeness contract.

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

Calculator entry points also assemble a ``Simulation`` and call a kernel.
Their independent admission columns are classified below. Observers retain
each calculator's existing handling, including the waveguide NTFF warning.

* ``s_matrix_scan``: the lumped/wire S-matrix of ``run(compute_s_params=True)``
  (``compute_lumped_wire_s_matrix_via_scan`` → ``_forward_from_materials``).
* ``mixed_s_matrix``: ``compute_mixed_s_matrix`` → ``_forward_from_materials``.
* ``topology_optimize``: → ``_forward_from_materials``.
* ``waveguide_s_matrix``: ``compute_waveguide_s_matrix``; uniform mesh through
  ``rfx.sources.waveguide_port.extract_*`` → ``rfx.simulation.run``, graded
  mesh through ``run_nonuniform_path``.
* ``coaxial_line_reflection``, ``coaxial_two_port``, and
  ``coax_msl_transition`` → ``rfx.simulation.run``. The transition assembles
  registered geometry and an MSL port; the other two reject them.
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
    "waveguide_s_matrix", "coaxial_line_reflection", "coaxial_two_port",
    "coax_msl_transition", "vmap_sweep_batched",
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


# Lane admission (rfx/runners/_admission.py) refuses every declared input a
# lane does not carry. The cells below solved the input as if it were absent
# until admission refused it; ``before`` says how, and the issue that found
# it. The input and lane words are retyped from the admission message, so a
# changed message fails here instead of passing by construction.
PREC = "a precision other than 'float32'"
RUN_U, RUN_NU, RUN_SG, RUN_ADI = "uniform run()", "graded run()", "subgridded run()", "ADI run()"
FWD_U, FWD_NU, FWD_DNU, FWD_ADI = ("uniform forward()", "graded forward()",
                                   "forward(distributed=True)", "ADI forward()")


def admission(what, lane, before):
    return refuses(f"refused by lane admission; before it, {before}",
                   raises=f"{what} is not carried by the {lane} lane")


# A 2-D mode on a graded mesh: _dispatch_plan's own check refuses it on every
# graded lane and names the remedy, a thin 3-D box. Until #1355 a 2d_tmz model
# one cell thick between PEC walls ran there as that 3-D box, whose record
# differs from run_uniform's 2-D solve by 0.34 of the probe peak (#1340).
GRADED_2D = refuses("a 2-D mode is refused on the graded lanes, naming the thin 3-D box "
                    "(mode='3d') as the remedy (#1340)",
                    raises="Build the 2-D problem as a thin 3-D box")
# forward() returns no flux field and does not build the monitors, by design:
# a monitor declared for run() costs forward() nothing and drops no result.
FWD_FLUX = ("forward() returns no flux field and does not build flux monitors, by design "
            "(tests/unit/sparams/test_mixed_port_sparam.py::"
            "test_forward_does_not_pay_for_flux_monitors); the monitor belongs to run()")


# Lane gates: a cell whose lane admits the row for some declarations only,
# decided by the lane's own check (rfx/runners/_admission.py LANE_GATES). The
# cell keeps the disposition of the model the cells file builds for it; the
# gate is listed here, and the contract test holds the two lists together.
GUARDED_LID_NOTE = ("outside production validation's guarded envelope; a lid inside it, "
                    "the absorbing_lid row, is admitted (LANE_GATES)")
GUARDED_LID = ("rfx/subgridding/validation.py _guarded_boundary_production_allowed: a CPML lid "
               "opposite the PEC z face the refined slab touches, closed PEC x/y faces; every "
               "validation mode")
LANE_GATES = {("run_subgridded", row): GUARDED_LID for row in (
    ("_boundary", "cpml"), ("_pec_faces", "pec_face"), ("_cpml_layers", "layers"),
    ("_cpml_kappa_max", "kappa"))}


def _conformal(lane_words):
    return refuses("Boundary(conformal=True) is refused like conformal_pec=True: this lane has no "
                   "Dey-Mittra update (#1297); run(conformal_pec=False) explicitly requests staircase PEC",
                   raises=f"the {lane_words} does not implement")


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
        run_adi=admission(PREC, RUN_ADI, "'mixed' and 'float64' ran float32 without a word; with "
                          "x64 on, 'float64' died in the ADI scan with a TypeError (#1308)"),
        run_distributed=refuses("non-float32 refused (#630)", raises="(issue #630)"),
        fwd_uniform=carries("field dtype"),
        fwd_nonuniform=refuses("non-float32 refused (#630)", raises="(issue #630)"),
        fwd_distributed_nu=refuses("non-float32 refused (#630)", raises="(issue #630)"),
        fwd_adi=admission(PREC, FWD_ADI, "it ran float32 without a word (#1308)"),
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
        run_nonuniform=GRADED_2D,
        run_subgridded=_subgrid("z_slab_requires_guarded_boundary",
                                "a slab across a one-cell z domain is never one-sided"),
        run_adi=carries("3d and 2d_tmz; 2d_tez refused. A 3d box one z cell thick dies with an "
                        "IndexError (measured), so its model is three cells thick"),
        run_distributed=carries("2d_tmz matched one device bit for bit (measured)"),
        fwd_uniform=carries("3d, 2d_tmz and 2d_tez"),
        fwd_nonuniform=GRADED_2D,
        fwd_distributed_nu=GRADED_2D,
        fwd_adi=carries("as run_adi"),
    )},

    # --------------------------------------------------------- bookkeeping
    "_snap": {"": every_path(ignorable(
        "preflight acceptance policy (#1138): run()/forward() gate sheet-size findings "
        "before stepping and realized geometry records the choice; no field update changes"))},
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
            run_distributed=carries("each E edge takes the mean ε of its four cells, as on one device (#1303)"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries("each E edge takes the mean ε of its four cells, as on one device (#1303)"),
            fwd_adi=carries(),
        ),
        "sigma": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=carries("inside the production envelope"),
            run_adi=carries("implicit conductivity in the ADI solve"),
            run_distributed=carries("each E edge takes the mean σ of its four cells, as on one device (#1303)"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries("each E edge takes the mean σ of its four cells, as on one device (#1303)"),
            fwd_adi=carries("implicit conductivity in the ADI solve"),
        ),
        "mu": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=carries("inside the production envelope"),
            run_adi=admission("a magnetic material (mu_r != 1)", RUN_ADI, "the ADI kernel took ε and σ only (#1308)"),
            run_distributed=carries(),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
            fwd_adi=admission("a magnetic material (mu_r != 1)", FWD_ADI, "the ADI kernel took ε and σ only (#1308)"),
        ),
        **{pole: lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=_subgrid("dispersive_or_nonlinear_material",
                                    "production validation; research/off drop it, see _refinement "
                                    "'relaxed_validation'"),
            run_adi=ADI_DISPERSIVE,
            run_distributed=carries("uniform v2 and graded NU ADE; shared forward staging (#1461)"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
            fwd_adi=ADI_DISPERSIVE,
        ) for pole in ("debye", "lorentz", "drude")},
        "kerr": lanes(
            run_uniform=carries(),
            run_nonuniform=admission("a Kerr χ³ material (chi3 != 0)", RUN_NU, "the graded run() solved the material as "
                                     "linear (#1309)"),
            run_subgridded=_subgrid("dispersive_or_nonlinear_material",
                                    "production validation; research/off drop it"),
            run_adi=admission("a Kerr χ³ material (chi3 != 0)", RUN_ADI, "the ADI kernel has no χ³ term (#1308)"),
            run_distributed=refuses("Kerr χ³ refused (#1214)", raises="Kerr chi3 (nonlinear) material(s)"),
            fwd_uniform=carries(),
            fwd_nonuniform=refuses("Kerr χ³ refused off the uniform forward lane",
                                   raises="forward() supports Kerr"),
            fwd_distributed_nu=refuses("Kerr χ³ refused off the uniform forward lane",
                                       raises="forward() supports Kerr"),
            fwd_adi=admission("a Kerr χ³ material (chi3 != 0)", FWD_ADI, "forward()'s Kerr guard let fwd_uniform through "
                              "and the ADI kernel has no χ³ term (#1308)"),
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
            run_subgridded=admission("a lossy thin conductor (add_thin_conductor)", RUN_SG,
                                     "it was dropped although production validation passes (#1311)"),
            run_adi=ADI_THIN,
            run_distributed=carries("folded into σ, which each E edge takes as the mean of its four cells (#1303)"),
            fwd_uniform=carries("folded into σ"),
            fwd_nonuniform=carries("folded into σ"),
            fwd_distributed_nu=carries("folded into σ, which each E edge takes as the mean of its four cells (#1303)"),
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
            run_uniform=carries("add_source() without S; S requests: PLAIN_SOURCE_S_REQUEST"),
            run_nonuniform=carries("without S; S requests: PLAIN_SOURCE_S_REQUEST"),
            run_subgridded=carries("inside the refined slab"),
            run_adi=carries(),
            run_distributed=carries("without S; S requests: PLAIN_SOURCE_S_REQUEST"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
            fwd_adi=carries(),
        ),
        "amplitude_kind": lanes(
            run_uniform=carries("'current' and 'field' scale the waveform differently (#571)"),
            run_nonuniform=carries(),
            run_subgridded=carries(),
            run_adi=refuses("ADI implements only field sources (#1373)", raises="amplitude_kind"),
            run_distributed=carries(),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
            fwd_adi=refuses("ADI implements only field sources (#1373)", raises="amplitude_kind"),
        ),
        "lumped_port": lanes(
            run_uniform=carries("drive and 50 Ω load"),
            run_nonuniform=carries("shared voltage drive agrees with run_uniform"),
            run_subgridded=carries("inside the refined slab"),
            run_adi=ADI_SOFT_SOURCES,
            run_distributed=carries("uniform single-cell excited ports: fields and full S-matrix; "
                                    "owning-cell recordings feed the shared extractor; graded run refuses ports (#1461)"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries("shared voltage drive agrees with run_uniform"),
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
            run_nonuniform=carries("shared voltage drive agrees with run_uniform"),
            run_subgridded=carries("inside the refined slab"),
            run_adi=ADI_SOFT_SOURCES,
            run_distributed=carries("uniform live wire drive/load and whole-port S; planes/radius refused"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries("shared voltage drive agrees with run_uniform"),
            fwd_distributed_nu=DIST_FWD_PORTS,
            fwd_adi=ADI_SOFT_SOURCES,
        ),
    },
    "_msl_ports": {"msl_port": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=admission("a microstrip port (add_msl_port)", RUN_SG,
                                 "it was dropped although production validation passes (#1311)"),
        run_adi=admission("a microstrip port (add_msl_port)", RUN_ADI, "a board with a PEC or thin-conductor trace was refused "
                          "and a port declared without a trace was dropped (#1308)"),
        run_distributed=refuses("MSL ports refused (#1241)", raises="add_msl_port() port(s)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=admission("a microstrip port (add_msl_port)", FWD_DNU, "the port launched nothing (#1285)"),
        fwd_adi=admission("a microstrip port (add_msl_port)", FWD_ADI, "as on run_adi, a port declared without a trace was "
                          "dropped (#1308)"),
    )},
    "_waveguide_ports": {"waveguide_port": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=_subgrid("boundary_terminated_requires_pec_no_cpml",
                                "a waveguide port needs a CPML face, which production validation refuses"),
        run_adi=ADI_PORTS,
        run_distributed=falls_back("run_uniform", "one device, with a warning and every "
                                   "argument the caller gave (#1305)"),
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
            run_nonuniform=admission("a Floquet port (add_floquet_port)", RUN_NU, "it launched nothing on a dx/dy profile; a "
                                     "dz profile was refused (#1312)"),
            run_subgridded=_subgrid("subgrid_overlaps_absorber",
                                    "a Floquet cell absorbs on z, which production validation refuses"),
            run_adi=ADI_PORTS,
            run_distributed=refuses("the periodic axes it sets are refused (#1241)",
                                    raises="periodic / Bloch boundaries are not supported"),
            fwd_uniform=carries("at normal incidence"),
            fwd_nonuniform=admission("a Floquet port (add_floquet_port)", FWD_NU, "it launched nothing on a dx/dy profile "
                                     "(#1312)"),
            fwd_distributed_nu=refuses("the periodic axes it sets are refused (#1350)",
                                       raises="periodic / Bloch boundaries are not supported"),
            fwd_adi=ADI_PORTS,
        ),
        "scan_angle": lanes(
            run_uniform=admission("a Floquet port scanned off normal (scan_theta != 0)", RUN_U, "30° gave the record of 0°: the angle never "
                                  "reached the fields (#1221)"),
            run_nonuniform=admission("a Floquet port scanned off normal (scan_theta != 0)", RUN_NU, "the port launched nothing at any angle "
                                     "(#1312)"),
            run_subgridded=_subgrid("subgrid_overlaps_absorber",
                                    "a Floquet cell absorbs on z, which production validation refuses"),
            run_adi=ADI_PORTS,
            run_distributed=refuses("the periodic axes it sets are refused (#1241)",
                                    raises="periodic / Bloch boundaries are not supported"),
            fwd_uniform=admission("a Floquet port scanned off normal (scan_theta != 0)", FWD_U, "30° gave the record of 0° (#1221)"),
            fwd_nonuniform=admission("a Floquet port scanned off normal (scan_theta != 0)", FWD_NU, "the port launched nothing at any angle "
                                     "(#1312)"),
            fwd_distributed_nu=refuses("the periodic axes it sets are refused (#1350)",
                                       raises="periodic / Bloch boundaries are not supported"),
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
        run_distributed=falls_back("run_uniform", "one device, with a warning and every "
                                   "argument the caller gave (#1305)"),
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
            "run_subgridded": admission(
                "validation='research'/'off' with a dispersive pole, Kerr χ³ or a lumped RLC "
                "element", RUN_SG, "validation='research' and 'off' ran the Debye, Lorentz and "
                "Drude poles, Kerr χ³ and lumped RLC that production refuses, and dropped them "
                "(#1286)"),
        },
    },

    # ---------------------------------------------------------- boundaries
    "_boundary": {
        "cpml": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=_subgrid("subgrid_overlaps_absorber", "a CPML box: " + GUARDED_LID_NOTE),
            run_adi=carries("a graded conductivity layer, not a CPML (test_realized_boundary.py "
                            "cpml--adi)", wrong="#1221"),
            run_distributed=carries("uniform v2 and graded NU slab-aware CPML (#1461)"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
            fwd_adi=carries("the same conductivity layer as run_adi", wrong="#1221"),
        ),
        "upml": lanes(
            run_uniform=carries(),
            run_nonuniform=refuses("the graded runner implements CPML only",
                                   raises="boundary='upml' does not support the non-uniform run() lane"),
            run_subgridded=_subgrid("subgrid_overlaps_absorber",
                                    "no absorber on every face in the guarded envelope; a UPML lid "
                                    "inside it runs as CPML (measured), so it is refused too"),
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
        run_subgridded=_subgrid("boundary_terminated_requires_pec_no_cpml", "CPML on the other faces: " + GUARDED_LID_NOTE),
        run_adi=ADI_PER_FACE,
        # #1235: the distributed absorber takes every face's profile from init_cpml, so the PEC wall
        # carries no absorber backing (test_realized_boundary.py pec-zlo--distributed departures gone).
        run_distributed=carries(),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=carries(),
        fwd_adi=ADI_PER_FACE,
    )},
    # Magnetic walls on the x faces, electric on y and z.
    "_boundary_spec": {
        "pmc_face": lanes(
            run_uniform=carries("odd H image on the declared E-node face (#1221 B3b)"),
            run_nonuniform=carries("graded shared curl carries the declared-face image"),
            run_subgridded=admission("a PMC (magnetic wall) face", RUN_SG, "no magnetic image"),
            run_adi=admission("a PMC (magnetic wall) face", RUN_ADI, "no magnetic image"),
            run_distributed=refuses("distributed_v2 has no declared-face image until B4",
                                    raises="does not implement the declared-face magnetic image"),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=refuses("distributed_nu has no declared-face image until B4",
                                      raises="does not implement the declared-face magnetic image"),
            fwd_adi=admission("a PMC (magnetic wall) face", FWD_ADI, "no magnetic image"),
        ),
        "conformal": lanes(
            run_uniform=carries("Dey-Mittra weights on every PEC shape, including direct run_uniform() by default; conformal_pec=False requests staircase PEC"),
            **CONFORMAL,
        ),
        "conformal_s_matrix": lanes(
            run_uniform=admission("Boundary(conformal=True) or conformal_pec=True with a lumped/wire S-matrix", RUN_U,
                                  "the lumped-port S-matrix run() returns came from "
                                  "_forward_from_materials, which staircases (#1299)"),
            **CONFORMAL,
        ),
        # A closed PEC box with a CPML lid on z_hi, against the all-PEC box.
        "absorbing_lid": lanes(
            run_uniform=carries(),
            run_nonuniform=carries(),
            run_subgridded=carries(
                "the refined slab touches the PEC floor: production validation's guarded envelope, "
                "admitted in every validation mode through LANE_GATES. The lid moves the late field "
                "(#1355 review); whether it absorbs as well as the uniform lane's is not measured "
                "here. A UPML lid is refused by the upml row: this lane runs CPML in its place"),
            run_adi=ADI_PER_FACE,
            run_distributed=carries(),
            fwd_uniform=carries(),
            fwd_nonuniform=carries(),
            fwd_distributed_nu=carries(),
            fwd_adi=ADI_PER_FACE,
        ),
    },
    "_periodic_axes": {"periodic": lanes(
        run_uniform=carries("declared period and wrapped index (#1221 B2)"),
        run_nonuniform=admission("a periodic axis", RUN_NU, "it was solved as PEC walls "
                                 "(periodic-xy--nonuniform, #1221)"),
        run_subgridded=admission("a periodic axis", RUN_SG, "it was solved as PEC walls although "
                                 "production validation passes (#1311)"),
        run_adi=_adi("periodic axes", "does not support manual periodic axes"),
        run_distributed=refuses("periodic axes refused (#1241)",
                                raises="periodic / Bloch boundaries are not supported"),
        fwd_uniform=carries("declared period and wrapped index (#1221 B2)"),
        fwd_nonuniform=admission("a periodic axis", FWD_NU, "it was solved as PEC walls (#1221)"),
        fwd_distributed_nu=refuses("periodic axes refused (#1350)",
                                   raises="periodic / Bloch boundaries are not supported"),
        fwd_adi=_adi("periodic axes", "does not support manual periodic axes"),
    )},
    "_cpml_layers": {"layers": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=_subgrid("subgrid_overlaps_absorber", "a CPML box: " + GUARDED_LID_NOTE),
        run_adi=carries("the thickness of ADI's conductivity layer, checked in the σ handed to the kernel"),
        run_distributed=carries(),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=carries(),
        fwd_adi=carries("as run_adi"),
    )},
    "_cpml_kappa_max": {"kappa": lanes(
        run_uniform=carries("CPML carries kappa; UPML refuses cpml_kappa_max != 1 before stepping"),
        run_nonuniform=admission("cpml_kappa_max != 1", RUN_NU, "the graded grid build was never given it "
                                 "(#1310)"),
        run_subgridded=_subgrid("subgrid_overlaps_absorber", "a CPML box: " + GUARDED_LID_NOTE),
        run_adi=admission("cpml_kappa_max != 1", RUN_ADI, "it was dropped: ADI's absorber is not a CPML "
                          "(cpml--adi, #1221)"),
        run_distributed=carries("uniform only; graded shared staging refuses nondefault kappa (#1461)"),
        fwd_uniform=carries("CPML carries kappa; UPML refuses cpml_kappa_max != 1 before stepping"),
        fwd_nonuniform=admission("cpml_kappa_max != 1", FWD_NU, "it was dropped (#1310)"),
        fwd_distributed_nu=admission("cpml_kappa_max != 1", FWD_DNU, "it was dropped (#1310)"),
        fwd_adi=admission("cpml_kappa_max != 1", FWD_ADI, "it was dropped, as on run_adi (#1221)"),
    )},
    # An εr 4 block offset by half a cell, 'dual_average' against 'sampled'.
    "_interface_eps": {"dual_average": lanes(
        run_uniform=admission("interface_eps='dual_average'", RUN_U, "it was not read: the field was the 'sampled' "
                              "one (#1339)"),
        run_nonuniform=carries("read by the graded assembly (assemble_interface_eps_nu); no lane to "
                               "compare with"),
        run_subgridded=admission("interface_eps='dual_average'", RUN_SG, "it was not read (#1339)"),
        run_adi=admission("interface_eps='dual_average'", RUN_ADI, "it was not read (#1339)"),
        run_distributed=refuses("'dual_average' refused",
                                raises="interface_eps='dual_average' is not supported on the distributed lane"),
        fwd_uniform=admission("interface_eps='dual_average'", FWD_U, "it was not read (#1339)"),
        fwd_nonuniform=carries("read by the graded assembly"),
        fwd_distributed_nu=refuses("'dual_average' refused",
                                   raises="interface_eps='dual_average' cannot combine with"),
        fwd_adi=admission("interface_eps='dual_average'", FWD_ADI, "it was not read (#1339)"),
    )},

    # -------------------------------------------------------- mesh profiles
    **{profile: {"graded": lanes(
        run_uniform=not_reachable("a dx/dy/dz profile sends the model to the graded lane"),
        run_nonuniform=carries(),
        run_subgridded=GRADED_REFINEMENT,
        run_adi=ADI_GRADED,
        run_distributed=carries("shared NU run/forward staging, grading up to 5:1 (#1461)"),
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
        run_subgridded=admission("a flux monitor", RUN_SG, "result.flux_monitors came back None (#1313)"),
        run_adi=admission("a flux monitor", RUN_ADI, "result.flux_monitors came back None (#1313)"),
        run_distributed=refuses("flux monitors refused (#1241)", raises="add_flux_monitor() (flux monitors)"),
        fwd_uniform=ignorable(FWD_FLUX),
        fwd_nonuniform=ignorable(FWD_FLUX),
        fwd_distributed_nu=refuses("flux monitors refused",
                                   raises="add_flux_monitor() is not supported on the distributed non-uniform"),
        fwd_adi=ignorable(FWD_FLUX),
    )},
    "_ntff": {"ntff_box": lanes(
        run_uniform=carries(),
        run_nonuniform=carries(),
        run_subgridded=carries("inside the refined slab, clear of its artificial interface"),
        run_adi=_adi("NTFF", "does not support NTFF accumulation"),
        run_distributed=carries("uniform mesh; owner-partitioned surface record; graded run refuses NTFF (#1461)"),
        fwd_uniform=carries(),
        fwd_nonuniform=carries(),
        fwd_distributed_nu=admission("an NTFF box", FWD_DNU, "ntff_data came back None (#1313)"),
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
        "_msl_auto_probe_spacing", "_msl_auto_probe_lengths", "_boundary_model", "_snap")},
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


# Calculator cells are independent of the production admission lists.
# Each entry below records the assembly/scan inspected for that calculator.
# Observers retain their existing behaviour (leader decision, RESUME1).
CALCULATOR_CELLS = {
    ('_freq_max', ''): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': carries('rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run'),
        'coaxial_two_port': carries('rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_domain', ''): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': carries('rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run'),
        'coaxial_two_port': carries('rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_dx', ''): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': carries('rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run'),
        'coaxial_two_port': carries('rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', wrong="#1293"),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_dt_pin', ''): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises='a pinned time step (dt=)'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a pinned time step (dt=)'),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='a pinned time step (dt=)'),
        'waveguide_s_matrix': refuses('carried only by the graded waveguide builder, through LANE_GATES', raises='a pinned time step (dt=)'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a pinned time step (dt=)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a pinned time step (dt=)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a pinned time step (dt=)'),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises='a pinned time step (dt=)'),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a pinned time step (dt=)'),
    },
    ('_dt_min_cell', ''): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises='dt_min_cell='),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='dt_min_cell='),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='dt_min_cell='),
        'waveguide_s_matrix': refuses('carried only by the graded waveguide builder, through LANE_GATES', raises='dt_min_cell='),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='dt_min_cell='),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='dt_min_cell='),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='dt_min_cell='),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises='dt_min_cell='),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='dt_min_cell='),
    },
    ('_precision', ''): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises="a precision other than 'float32'"),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises="a precision other than 'float32'"),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises="a precision other than 'float32'"),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises="a precision other than 'float32'"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises="a precision other than 'float32'"),
    },
    ('_solver', ''): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises="solver='adi'"),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises="solver='adi'"),
        'topology_optimize': carries("_forward_from_materials dispatches solver=adi to _run_adi_from_materials, which reads adi_cfl_factor; that lane retains its own admission"),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises="solver='adi'"),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises="solver='adi'"),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises="solver='adi'"),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises="solver='adi'"),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises="solver='adi'"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises="solver='adi'"),
    },
    ('_adi_cfl_factor', ''): {
        's_matrix_scan': carries('accepted on Yee as on the time-stepping lanes; ADI uses the multiplier'),
        'mixed_s_matrix': carries('accepted on Yee as on the time-stepping lanes; ADI uses the multiplier'),
        'topology_optimize': carries('accepted on Yee as on the time-stepping lanes; ADI uses the multiplier'),
        'waveguide_s_matrix': carries('accepted on Yee as on the time-stepping lanes; ADI uses the multiplier'),
        'coaxial_line_reflection': carries('accepted on Yee as on the time-stepping lanes; ADI uses the multiplier'),
        'coaxial_two_port': carries('accepted on Yee as on the time-stepping lanes; ADI uses the multiplier'),
        'coax_msl_transition': carries('accepted on Yee as on the time-stepping lanes; ADI uses the multiplier'),
        'vmap_sweep_batched': carries('accepted on Yee as on the time-stepping lanes; ADI uses the multiplier'),
        'material_fit': carries('accepted on Yee as on the time-stepping lanes; ADI uses the multiplier'),
    },
    ('_stencil_order', ''): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='stencil_order=4'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='stencil_order=4'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='stencil_order=4'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='stencil_order=4'),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises='stencil_order=4'),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='stencil_order=4'),
    },
    ('_mode', ''): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises="a 2-D mode (mode='2d_tmz' or '2d_tez')"),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises="a 2-D mode (mode='2d_tmz' or '2d_tez')"),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises="a 2-D mode (mode='2d_tmz' or '2d_tez')"),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_materials', 'eps'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a dielectric (eps_r != 1)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a dielectric (eps_r != 1)'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_materials', 'sigma'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a conductive material (sigma > 0)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a conductive material (sigma > 0)'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_materials', 'mu'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a magnetic material (mu_r != 1)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a magnetic material (mu_r != 1)'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_materials', 'debye'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a Debye pole'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a Debye pole'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a Debye pole'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_materials', 'lorentz'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a Lorentz pole'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a Lorentz pole'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a Lorentz pole'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_materials', 'drude'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a Drude pole'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a Drude pole'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a Drude pole'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_materials', 'kerr'): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises='a Kerr χ³ material (chi3 != 0)'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a Kerr χ³ material (chi3 != 0)'),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='a Kerr χ³ material (chi3 != 0)'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a Kerr χ³ material (chi3 != 0)'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a Kerr χ³ material (chi3 != 0)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a Kerr χ³ material (chi3 != 0)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a Kerr χ³ material (chi3 != 0)'),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises='a Kerr χ³ material (chi3 != 0)'),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a Kerr χ³ material (chi3 != 0)'),
    },
    ('_geometry', 'pec_volume'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a PEC volume'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a PEC volume'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_geometry', 'pec_sheet'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a PEC sheet (a zero-thickness PEC Box)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a PEC sheet (a zero-thickness PEC Box)'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_geometry', 'pec_wire'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a sub-cell PEC wire (PolylineWire)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a sub-cell PEC wire (PolylineWire)'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_thin_conductors', 'lossy_sheet'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a lossy thin conductor (add_thin_conductor)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a lossy thin conductor (add_thin_conductor)'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_thin_conductors', 'pec_sheet'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a PEC thin conductor (add_thin_conductor)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a PEC thin conductor (add_thin_conductor)'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_thin_conductors', 'surface_impedance'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a surface-impedance sheet (surface_impedance_f0)'),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='a surface-impedance sheet (surface_impedance_f0)'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a surface-impedance sheet (surface_impedance_f0)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a surface-impedance sheet (surface_impedance_f0)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a surface-impedance sheet (surface_impedance_f0)'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a surface-impedance sheet (surface_impedance_f0)'),
    },
    ('_pinned_sheets', 'pec_sheet'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a pinned sheet (add_pinned_sheet)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a pinned sheet (add_pinned_sheet)'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_ports', 'source'): {
        's_matrix_scan': carries('source-only declaration; with an impedance port see PLAIN_SOURCE_S_REQUEST'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a soft source (add_source)'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a soft source (add_source)'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a soft source (add_source)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a soft source (add_source)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a soft source (add_source)'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a soft source (add_source)'),
    },
    ('_ports', 'amplitude_kind'): {
        's_matrix_scan': carries('source-only declaration; with an impedance port see PLAIN_SOURCE_S_REQUEST'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises="a soft source with amplitude_kind='current'"),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises="a soft source with amplitude_kind='current'"),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises="a soft source with amplitude_kind='current'"),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises="a soft source with amplitude_kind='current'"),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises="a soft source with amplitude_kind='current'"),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises="a soft source with amplitude_kind='current'"),
    },
    ('_ports', 'lumped_port'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a driven lumped port (add_port)'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a driven lumped port (add_port)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a driven lumped port (add_port)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a driven lumped port (add_port)'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_ports', 'passive_port'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a passive port (add_port(excite=False))'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a passive port (add_port(excite=False))'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a passive port (add_port(excite=False))'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a passive port (add_port(excite=False))'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a passive port (add_port(excite=False))'),
    },
    ('_ports', 'wire_port'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a wire port (add_port(extent=...))'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a wire port (add_port(extent=...))'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a wire port (add_port(extent=...))'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a wire port (add_port(extent=...))'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a wire port (add_port(extent=...))'),
    },
    ('_msl_ports', 'msl_port'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a microstrip port (add_msl_port)'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a microstrip port (add_msl_port)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a microstrip port (add_msl_port)'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a microstrip port (add_msl_port)'),
    },
    ('_waveguide_ports', 'waveguide_port'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a waveguide port (add_waveguide_port)'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a waveguide port (add_waveguide_port)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a waveguide port (add_waveguide_port)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a waveguide port (add_waveguide_port)'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a waveguide port (add_waveguide_port)'),
    },
    ('_coaxial_ports', 'coax_port'): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises='a coaxial port (add_coaxial_port)'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a coaxial port (add_coaxial_port)'),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='a coaxial port (add_coaxial_port)'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a coaxial port (add_coaxial_port)'),
        'coaxial_line_reflection': carries('rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run'),
        'coaxial_two_port': carries('rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a coaxial port (add_coaxial_port)'),
    },
    ('_floquet_ports', 'floquet_port'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a Floquet port (add_floquet_port)'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a Floquet port (add_floquet_port)'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a Floquet port (add_floquet_port)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a Floquet port (add_floquet_port)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a Floquet port (add_floquet_port)'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a Floquet port (add_floquet_port)'),
    },
    ('_floquet_ports', 'scan_angle'): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises='a Floquet port scanned off normal (scan_theta != 0)'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a Floquet port scanned off normal (scan_theta != 0)'),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='a Floquet port scanned off normal (scan_theta != 0)'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a Floquet port scanned off normal (scan_theta != 0)'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a Floquet port scanned off normal (scan_theta != 0)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a Floquet port scanned off normal (scan_theta != 0)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a Floquet port scanned off normal (scan_theta != 0)'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a Floquet port scanned off normal (scan_theta != 0)'),
    },
    ('_lumped_rlc', 'R'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a lumped RLC element (add_lumped_rlc)'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a lumped RLC element (add_lumped_rlc)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a lumped RLC element (add_lumped_rlc)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a lumped RLC element (add_lumped_rlc)'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a lumped RLC element (add_lumped_rlc)'),
    },
    ('_lumped_rlc', 'series_RL'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a series lumped RLC element with an inductance'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a series lumped RLC element with an inductance'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a series lumped RLC element with an inductance'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a series lumped RLC element with an inductance'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a series lumped RLC element with an inductance'),
    },
    ('_tfsf', 'plane_wave'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a TFSF plane-wave source (add_tfsf_source)'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a TFSF plane-wave source (add_tfsf_source)'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a TFSF plane-wave source (add_tfsf_source)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a TFSF plane-wave source (add_tfsf_source)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a TFSF plane-wave source (add_tfsf_source)'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a TFSF plane-wave source (add_tfsf_source)'),
    },
    ('_refinement', 'slab'): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises='a subgrid refinement (add_refinement)'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a subgrid refinement (add_refinement)'),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='a subgrid refinement (add_refinement)'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a subgrid refinement (add_refinement)'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a subgrid refinement (add_refinement)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a subgrid refinement (add_refinement)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a subgrid refinement (add_refinement)'),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises='a subgrid refinement (add_refinement)'),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a subgrid refinement (add_refinement)'),
    },
    ('_refinement', 'relaxed_validation'): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises="validation='research'/'off' with a dispersive pole, Kerr χ³ or a lumped RLC element"),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises="validation='research'/'off' with a dispersive pole, Kerr χ³ or a lumped RLC element"),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises="validation='research'/'off' with a dispersive pole, Kerr χ³ or a lumped RLC element"),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises="validation='research'/'off' with a dispersive pole, Kerr χ³ or a lumped RLC element"),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises="validation='research'/'off' with a dispersive pole, Kerr χ³ or a lumped RLC element"),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises="validation='research'/'off' with a dispersive pole, Kerr χ³ or a lumped RLC element"),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises="validation='research'/'off' with a dispersive pole, Kerr χ³ or a lumped RLC element"),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises="validation='research'/'off' with a dispersive pole, Kerr χ³ or a lumped RLC element"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises="validation='research'/'off' with a dispersive pole, Kerr χ³ or a lumped RLC element"),
    },
    ('_boundary', 'cpml'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': carries('rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run'),
        'coaxial_two_port': carries('rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_boundary', 'upml'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a UPML absorber'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a UPML absorber'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a UPML absorber'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a UPML absorber'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_pec_faces', 'pec_face'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='PEC faces on an absorbing box'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='PEC faces on an absorbing box'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='PEC faces on an absorbing box'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('sim_run resolves the PEC faces from the grid through resolve_wall_faces'),
    },
    ('_boundary_spec', 'pmc_face'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('waveguide_port.init_waveguide_port has no magnetic aperture mode', raises='PMC.*magnetic'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a PMC (magnetic wall) face'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a PMC (magnetic wall) face'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a PMC (magnetic wall) face'),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises='a PMC (magnetic wall) face'),
        'material_fit': carries('sim_run resolves the PMC faces from the grid through resolve_wall_faces'),
    },
    ('_boundary_spec', 'conformal'): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises='Boundary(conformal=True) or conformal_pec=True'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='Boundary(conformal=True) or conformal_pec=True'),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='Boundary(conformal=True) or conformal_pec=True'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='Boundary(conformal=True) or conformal_pec=True'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='Boundary(conformal=True) or conformal_pec=True'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='Boundary(conformal=True) or conformal_pec=True'),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises='Boundary(conformal=True) or conformal_pec=True'),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='Boundary(conformal=True) or conformal_pec=True'),
    },
    ('_boundary_spec', 'conformal_s_matrix'): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises='Boundary(conformal=True) or conformal_pec=True with a lumped/wire S-matrix'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='Boundary(conformal=True) or conformal_pec=True with a lumped/wire S-matrix'),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='Boundary(conformal=True) or conformal_pec=True with a lumped/wire S-matrix'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='Boundary(conformal=True) or conformal_pec=True with a lumped/wire S-matrix'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='Boundary(conformal=True) or conformal_pec=True with a lumped/wire S-matrix'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='Boundary(conformal=True) or conformal_pec=True with a lumped/wire S-matrix'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='Boundary(conformal=True) or conformal_pec=True with a lumped/wire S-matrix'),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises='Boundary(conformal=True) or conformal_pec=True with a lumped/wire S-matrix'),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='Boundary(conformal=True) or conformal_pec=True with a lumped/wire S-matrix'),
    },
    ('_boundary_spec', 'absorbing_lid'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='an absorbing z lid on a closed PEC box'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='an absorbing z lid on a closed PEC box'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='an absorbing z lid on a closed PEC box'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('the grid provides per-face pads and walls to sim_run for the absorbing lid'),
    },
    ('_periodic_axes', 'periodic'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises='a periodic axis'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a periodic axis'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a periodic axis'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a periodic axis'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_cpml_layers', 'layers'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': carries('rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run'),
        'coaxial_two_port': carries('rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_cpml_kappa_max', 'kappa'): {
        's_matrix_scan': carries('rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans'),
        'mixed_s_matrix': carries('rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': carries('rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run'),
        'coaxial_two_port': carries('rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs'),
        'coax_msl_transition': carries('rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_interface_eps', 'dual_average'): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises="interface_eps='dual_average'"),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises="interface_eps='dual_average'"),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises="interface_eps='dual_average'"),
        'waveguide_s_matrix': refuses('admission before the scan; rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards', raises="interface_eps='dual_average'"),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises="interface_eps='dual_average'"),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises="interface_eps='dual_average'"),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises="interface_eps='dual_average'"),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises="interface_eps='dual_average'"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises="interface_eps='dual_average'"),
    },
    ('_dx_profile', 'graded'): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises='a dx_profile (graded x mesh)'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a dx_profile (graded x mesh)'),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='a dx_profile (graded x mesh)'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a dx_profile (graded x mesh)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a dx_profile (graded x mesh)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a dx_profile (graded x mesh)'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a dx_profile (graded x mesh)'),
    },
    ('_dy_profile', 'graded'): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises='a dy_profile (graded y mesh)'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a dy_profile (graded y mesh)'),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='a dy_profile (graded y mesh)'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a dy_profile (graded y mesh)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a dy_profile (graded y mesh)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a dy_profile (graded y mesh)'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a dy_profile (graded y mesh)'),
    },
    ('_dz_profile', 'graded'): {
        's_matrix_scan': refuses('admission before the scan; rfx/probes/sparam_driver.py: assembly and _forward_from_materials device scans', raises='a dz_profile (graded z mesh)'),
        'mixed_s_matrix': refuses('admission before the scan; rfx/sparams/mixed.py: assembly and per-drive _forward_from_materials', raises='a dz_profile (graded z mesh)'),
        'topology_optimize': refuses('admission before the scan; rfx/topology.py: base assembly and the objective forward solve', raises='a dz_profile (graded z mesh)'),
        'waveguide_s_matrix': carries('rfx/sparams/waveguide.py: device extractors; graded run_nonuniform_path, with its existing conditional guards'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a dz_profile (graded z mesh)'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a dz_profile (graded z mesh)'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a dz_profile (graded z mesh)'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': refuses('admission before the scan; rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss', raises='a dz_profile (graded z mesh)'),
    },
    ('_probes', 'probe'): {
        's_matrix_scan': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'mixed_s_matrix': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a point probe'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a point probe'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a point probe'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': carries('rfx/differentiable_material_fit.py: factory assembly, sim_run and probe-spectrum loss'),
    },
    ('_dft_planes', 'dft_plane'): {
        's_matrix_scan': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'mixed_s_matrix': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a DFT plane probe'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a DFT plane probe'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a DFT plane probe'),
        'vmap_sweep_batched': carries('rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn'),
        'material_fit': ignorable('observer: existing calculator handling retained; no observer field returned'),
    },
    ('_flux_monitors', 'flux'): {
        's_matrix_scan': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'mixed_s_matrix': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='a flux monitor'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='a flux monitor'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='a flux monitor'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': ignorable('observer: existing calculator handling retained; no observer field returned'),
    },
    ('_ntff', 'ntff_box'): {
        's_matrix_scan': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'mixed_s_matrix': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': ignorable('observer: warned (#704)'),
        'coaxial_line_reflection': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_line_reflection stamped grid and TEM run', raises='an NTFF box'),
        'coaxial_two_port': refuses('admission before the scan; rfx/sparams/coax.py: compute_coaxial_two_port stamped grid and TEM runs', raises='an NTFF box'),
        'coax_msl_transition': refuses('admission before the scan; rfx/sparams/coax.py: compute_coax_msl_transition registered geometry assembly and TEM/MSL runs', raises='an NTFF box'),
        'vmap_sweep_batched': not_reachable("_build_full_scan_fn returns None or the mesh gate selects sequential run() before the batched entry"),
        'material_fit': ignorable('observer: existing calculator handling retained; no observer field returned'),
    },
    ('_current_moments', 'block_moments'): {
        's_matrix_scan': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'mixed_s_matrix': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'topology_optimize': carries('rfx/topology.py: base assembly and the objective forward solve'),
        'waveguide_s_matrix': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'coaxial_line_reflection': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'coaxial_two_port': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'coax_msl_transition': ignorable('observer: existing calculator handling retained; no observer field returned'),
        'vmap_sweep_batched': refuses('admission before the scan; rfx/vmap_sweep.py: _build_full_scan_fn and _build_vmap_scan_fn', raises='a current-moment monitor'),
        'material_fit': ignorable('observer: existing calculator handling retained; no observer field returned'),
    },
}
for (_attr, _feature), _cells in CALCULATOR_CELLS.items():
    TABLE[_attr][_feature].update(_cells)

LANE_GATES.update({
    ("waveguide_s_matrix", ("_dt_pin", "")): "graded builder reads dt",
    ("waveguide_s_matrix", ("_dt_min_cell", "")): "graded builder reads dt_min_cell",
})


def cell(attr: str, feature: str, path: str) -> Cell:
    """The recorded disposition for one input and one execution path."""
    row = TABLE[attr][feature]
    return row["*"] if "*" in row else row[path]


# Conditional on an S request, including the lane's resolved default. Source
# rows above continue to carry ordinary field runs and forward() unchanged.
PLAIN_SOURCE_S_REQUEST = {
    path: refuses("rfx.runners._admission.refuse_plain_sources_s_matrix; "
                  "tests/unit/sparams/test_plain_source_refusal.py", raises="plain sources")
    for path in ("run_uniform", "run_nonuniform", "run_subgridded", "run_distributed",
                 "s_matrix_scan", "fwd_uniform", "fwd_nonuniform")
}
