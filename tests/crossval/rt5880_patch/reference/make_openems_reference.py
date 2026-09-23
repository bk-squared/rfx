#!/usr/bin/env python
"""Make the RT/Duroid 5880 patch antenna's openEMS reference, with provenance.

NEVER RUN BY CI. The external solver runs by hand, on the cluster, when the case
is created or its geometry changes; this script is the thing that is run.

WHAT IS SIMULATED
-----------------
A rectangular microstrip patch 40.0 mm long (x, the resonant length) and
50.0 mm wide (y, the two radiating edges) on Rogers RT/Duroid 5880 (eps_r 2.2,
tan delta 0.001, h 3.175 mm) over a finite 56 x 66 mm ground plane, fed by a
50 ohm lumped probe that runs from the ground to the patch 9 mm off centre
along x. The TM010 mode puts half a guided wavelength across the 40 mm, made
electrically longer at each radiating edge by the fringing field; near that
frequency the probe sees the patch's radiation resistance and |S11| dips. The
transmission-line model with Hammerstad's fringing extension puts TM010 at
2.4156 GHz. No closed form holds a probe-fed patch on a finite ground to 1 %,
which is why this board is cross-validated against another solver and not
against an oracle.

WHY THE REFERENCE IS BEING REMADE
---------------------------------
The committed record ``validation/crossval/_15_patch_results/openems.json``
carries solver, mesh_res_mm, n_cells, runtime_s, the 181-point arrays, f_dip_hz,
s11_dip_db, max_abs_s11, f_analytic_hz and gain_dbi. It names no openEMS
version, no build, no run, no stop criterion and one mesh, and no reproduce gate
ran with it. It was made by ``validation/crossval/15_patch_antenna_rt5880.py``'s
``run_openems``; the record and that builder entered the repository in the same
commit (f8048051) and the builder has not changed since. It ran with MUR on all
six faces and a 30000-step cap. This script makes the same measurement with the
provenance, the mesh statement and the witnesses a reference needs: openEMS's
own patch-antenna tutorial reproduced first (Stage A), then the board on three
meshes (Stage B).

STAGE A -- the reproduce gate: openEMS's own Simple Patch Antenna tutorial
------------------------------------------------------------------------
``python/Tutorials/Simple_Patch_Antenna.py`` ("(c) 2015-2023 Thorsten Liebig",
"Tested with python 3.10, openEMS v0.0.34+"): a 32 x 40 mm patch (32 mm is the
resonant length, along x) on a 60 x 60 mm, 1.524 mm substrate of eps_r 3.38
(tan delta 1e-3 expressed as a conductivity at 2.45 GHz), a ground under the
whole substrate, a 50 ohm lumped port from the ground to the patch at
x = -6 mm, MUR on all six faces of a 200 x 200 x 150 mm box,
SetGaussExcite(2 GHz, 1 GHz), NrTS 30000 with EndCriteria 1e-4, a lambda/20 mesh
at 3 GHz with the thirds rule at the patch edges.

Which file: the one at openEMS commit ``A_TUTORIAL_OPENEMS_COMMIT``, the openEMS
submodule of openEMS-Project ``A_TUTORIAL_PROJECT_COMMIT`` -- the build the job's
image stamps into RFX_OPENEMS_COMMIT. Its sha256 is ``A_TUTORIAL_SHA256``. The
pinned image ships the same file at ``TUTORIAL_IMAGE_PATH_DEFAULT`` (read
2026-09-23 from the image layer that clones openEMS-Project; same sha256; no
later layer deletes it), and the real run checks it again inside the container
and refuses to measure if it differs.

The tutorial's build block, its lines 28-104, is frozen below verbatim
(``_FROZEN_TUTORIAL_BUILD``). ``_build_stage_a_tutorial``'s body IS that block,
indented by four spaces, with ONE substitution: ``openEMS(NrTS=30000,
EndCriteria=1e-4)`` becomes ``openEMS(**kw)``, so that the smoke pass can run
200 steps. The real pass passes the tutorial's own 30000 / 1e-4, and
``--self-check`` reads those two numbers from the frozen line rather than
trusting the constants. The tutorial's ``from openEMS.physical_constants import
*`` becomes an explicit import of the two names the block uses, C0 and EPS0.
The post-processing is the tutorial's too: CalcPort on linspace(1 GHz, 3 GHz,
401), S11 = uf_ref/uf_inc, Zin = uf_tot/if_tot, the tutorial's own resonance
pick (its line 137: the one bin that is both the minimum and below -10 dB) and
its NF2FF call at that bin. What differs is how the solver is started: through
the shared module's stdout/stderr capture, Run(..., verbose=1, numThreads=8),
after a 200-step smoke pass. The tutorial itself calls Run(Sim_Path,
cleanup=True).

THE TUTORIAL'S DOCUMENTED RESULT, AND THE GATE
----------------------------------------------
The tutorial prints no number. Its documentation page at the same commit
(``python/doc/Tutorials/Simple_Patch_Antenna.rst``) shows two figures,
``images/Simp_Patch_S11.png`` and ``images/Simp_Patch_Zin.png``, committed
2016-09-10 and made from the tutorial as it then was. Read off by pixel
(``STAGE_A_DOCUMENTED`` carries the axis calibration and the pixel positions):
the |S11| dip sits at 2.430 GHz, +-0.005 GHz being the width of the dip's tip on
the image, and is -26.8 dB deep; Re(Zin) peaks at 50.3 ohm at 2.424 GHz.

The script the figures were made from (openEMS c9435905, 2016-09-10) builds
the same patch, substrate, ground, port, excitation, box, boundaries, stop
criteria and mesh resolution. Its mesh-line calls differ in three places,
listed in ``STAGE_A_DOCUMENTED["figure_script_differences"]``. What those
differences, and ten years of solver changes, do to the dip has not been
measured.

The gate: the Stage A |S11| minimum, located with the repository's sub-bin
estimator over the tutorial's whole 1-3 GHz grid, lies within +-2 % of 2.430 GHz
AND is at least 10 dB deep. The -10 dB is the tutorial's own criterion for
calling the minimum a resonance. The 2 % is a proposal, not a derived number:
the reading uncertainty is 0.2 %, and the rest of the window is there for the
unmeasured differences just named. The transmission-line estimate for the
tutorial's board is recorded beside the gate and is not part of it.

THE RECORDED REPRODUCTION IS AN AUDIT TRAIL, NEVER THIS RUN'S GATE
------------------------------------------------------------------
This repository already ran the unmodified tutorial once (VESSL 369367247478,
2026-07-18, on openEMS 7b051bb7 -- four commits behind the pinned build, same
tutorial file): an external 50 ohm decomposition of its saved port data put the
dip at 2.4325 GHz, -33.0 dB, on a 49 x 47 x 45-line mesh, 12046 steps, the
last printed box energy -48.41 dB. ``STAGE_A_RECORDED_REPRODUCTION`` carries it
into every record as provenance for the tutorial, exactly as the MSL notch
filter maker carries its own. It is not what any run is judged by: Stage A runs
the tutorial again in the same invocation and is judged against the documented
figure.

STAGE B -- the RT5880 board, three mesh rungs. THE FULL DELTA LIST
------------------------------------------------------------------
The builder is ``run_openems``'s build block, lines 1087-1125 of
``validation/crossval/15_patch_antenna_rt5880.py`` at commit
``RETIRED_FROZEN_AT_COMMIT`` (the last commit to touch that file), frozen below
verbatim in ``_FROZEN_BUILDER_SLICE`` so the proof survives the script's
removal. ``_build_patch_board_at_rung``'s block IS that slice with the
substitutions in ``B_COPY_SUBSTITUTIONS`` and ``B_RUNG_SUBSTITUTIONS`` applied,
and nothing else; while the script is still on disk ``--self-check`` also proves
the frozen slice IS the script's, character for character. The deltas, in the
record's meta and in the dry run: ``DELTA_LIST`` below.

SANITY GATES ON EVERY REAL PASS
-------------------------------
From the shared module (``tests/crossval/_openems_tutorial_gate.py``), unchanged:
the stdout/stderr scan for unused primitives, off-mesh ports and a cut
excitation; the scale-free excitation and port-trace guard; openEMS's own
"reached before the end-criteria of" warning as a failed gate; |S11| finite and
<= 2; and the passivity witness, fed zeros for S21, so that it bounds
|S11|^2 <= 1.05 over the stage's whole grid. A radiating antenna's |S11|^2 is
below one by the accepted power; that deficit is recorded, not judged.

WHAT IS REPORTED AND NOTHING ELSE
---------------------------------
Per stage: complex S11 and Zin on every bin; the |S11| minimum near TM010 with
the shared sub-bin estimator and its half-grid witness; its depth; the -10 dB
band around it; Zin at the refined frequency and at the bin; the Re(Zin) peak;
max and min |S11|^2; the realized mesh, the PML faces and the smallest cell;
openEMS's timestep, step count, last printed box energy and speed. Stage B also
reports the NF2FF directivity at the resonance, the retired script's own call,
when the dip is below -6 dB. None of the Stage B numbers is gated here.

EXIT CODES
----------
0 every requested stage ran and every gate passed; 1 a gate failed; 2 openEMS is
not importable; 3 a layout/config bug in this script, or an image whose tutorial
differs from the frozen one.

USAGE
-----
    python make_openems_reference.py --self-check
    python make_openems_reference.py --dry-run --stage both
    python make_openems_reference.py --stage both --output <path>.json \\
        --sim-root /tmp/rt5880_patch_openems --threads 8

The job file is ``scripts/vessl_rt5880_patch_openems_reference.yaml``. ``run_id``
in the record is always ``null``: VESSL does not export the run id into the pod,
so the submitter fills it (``~/.claude/rules/vessl-jobs.md``).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# The shared tutorial gate, loaded by PATH, exactly as the Sheen maker loads it:
# on the cluster this runs as a bare file and ``tests`` is not a package.
# ---------------------------------------------------------------------------
_GATE_MODULE_NAME = "_rfx_openems_tutorial_gate"


def _load_tutorial_gate():
    if _GATE_MODULE_NAME in sys.modules:
        return sys.modules[_GATE_MODULE_NAME]
    path = Path(__file__).resolve().parents[2] / "_openems_tutorial_gate.py"
    if not path.is_file():
        raise RuntimeError(
            f"the shared tutorial gate is missing at {path} -- this maker cannot "
            f"run without it"
        )
    spec = importlib.util.spec_from_file_location(_GATE_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_GATE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


_gate = _load_tutorial_gate()
_smooth_estimate = _gate._smooth_estimate
_builder_body_after_kw = _gate._builder_body_after_kw
# max(|S11|^2 + |S21|^2) <= 1 + 0.05, the shared module's. With S21 = 0 that is
# |S11| <= sqrt(1.05) = 1.0247.
PASSIVITY_TOL = _gate.PASSIVITY_TOL


# ---------------------------------------------------------------------------
# THE BOARD (metres). The retired script's own constants, same names, same
# values. --self-check compares every one with the retired script's assignment
# line (frozen below, and read off disk while the file exists).
# ---------------------------------------------------------------------------
C0 = 2.99792458e8
EPS_R = 2.2                 # RT/Duroid 5880
TAN_DELTA = 1.0e-3
H_SUB = 3.175e-3            # 1/8 inch
L_PATCH = 40.0e-3          # resonant length (x)
W_PATCH = 50.0e-3          # radiating width (y)
GP_X = 56.0e-3
GP_Y = 66.0e-3
FEED_OFFSET_X = -9.0e-3    # inset feed, 9 mm off centre along L
N_SUB = 4
F_LO, F_HI = 1.6e9, 3.4e9   # S11 search / sweep band

RETIRED_REL_PATH = "validation/crossval/15_patch_antenna_rt5880.py"
RETIRED_FUNCTION = "run_openems"
RETIRED_FROZEN_AT_COMMIT = "150ed1d846ae395dd3afae479ee0e4caab6f42c4"
RETIRED_FROZEN_LINE_RANGE = (1087, 1125)

# The retired script's assignment lines, verbatim. Each constant above must equal
# the literal on its line, and each line must be a whole line of the script
# while the script exists.
RETIRED_CONSTANT_LINES = {
    "C0": "C0 = 2.99792458e8",
    "EPS_R": "EPS_R = 2.2                 # RT/Duroid 5880",
    "TAN_DELTA": "TAN_DELTA = 1.0e-3",
    "H_SUB": "H_SUB = 3.175e-3            # 1/8 inch",
    "L_PATCH": "L_PATCH = 40.0e-3          # resonant length (x)",
    "W_PATCH": "W_PATCH = 50.0e-3          # radiating width (y)",
    "GP_X": "GP_X = 56.0e-3",
    "GP_Y": "GP_Y = 66.0e-3",
    "FEED_OFFSET_X": "FEED_OFFSET_X = -9.0e-3    # inset feed, 9 mm off centre along L",
    "N_SUB": "N_SUB = 4",
    "F_LO, F_HI": "F_LO, F_HI = 1.6e9, 3.4e9   # S11 search / sweep band",
}

# What the retired, unprovenanced record says about itself, frozen so that the
# numbers the dry run compares against survive the record's removal.
# --self-check re-reads the file while it exists.
RETIRED_RECORD_REL_PATH = "validation/crossval/_15_patch_results/openems.json"
RETIRED_RECORD = {
    "mesh_res_mm": 2.7758560925925924,
    "n_cells": 470988,
    "runtime_s": 46.89117932319641,
    "f_dip_hz": 2330000000.0,
    "s11_dip_db": -20.10299347927542,
    "max_abs_s11": 0.9919735782032139,
    "f_analytic_hz": 2415595433.5060616,
    "gain_dbi": 7.335233974297809,
}


def f_tm010_tl_model(eps_r: float, h: float, length: float, width: float,
                     c0: float = C0) -> tuple:
    """TM010 of a rectangular patch, transmission-line model.

    The retired script's ``f_res_analytic`` (Balanis, Antenna Theory ch. 14):
    Hammerstad's eps_eff and his fringing extension dL at each radiating edge,
    f = c0 / (2 (L + 2 dL) sqrt(eps_eff)). Returns (f, eps_eff, dL).
    """
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / width) ** -0.5
    dl = 0.412 * h * ((eps_eff + 0.3) * (width / h + 0.264)) / \
        ((eps_eff - 0.258) * (width / h + 0.8))
    return c0 / (2 * (length + 2 * dl) * math.sqrt(eps_eff)), eps_eff, dl


F_TM010_BOARD_HZ = f_tm010_tl_model(EPS_R, H_SUB, L_PATCH, W_PATCH)[0]


# ---------------------------------------------------------------------------
# STAGE A: the tutorial. Everything here is the tutorial's own, and the numbers
# the real pass uses are parsed from the frozen text by --self-check.
# ---------------------------------------------------------------------------
A_TUTORIAL = {
    "repo": "thliebig/openEMS",
    "path": "python/Tutorials/Simple_Patch_Antenna.py",
    "attribution": "(c) 2015-2023 Thorsten Liebig <thorsten.liebig@gmx.de>",
    "tested_with": "python 3.10, openEMS v0.0.34+",
    "fetched_via": ("gh api repos/thliebig/openEMS/contents/python/Tutorials/"
                    "Simple_Patch_Antenna.py?ref=<A_TUTORIAL_OPENEMS_COMMIT>"),
    "fetched_on": "2026-09-23",
    "frozen_lines": "28-104 (the build block), plus the post-processing lines "
                    "in _FROZEN_TUTORIAL_POSTPROCESSING",
}
A_TUTORIAL_PROJECT_COMMIT = "5b423bdfe0c84064cf9028167bb759007c33b182"   # openEMS-Project
A_TUTORIAL_OPENEMS_COMMIT = "2000574e8785667a4ad107ad32a304a5034d872a"   # its openEMS submodule
A_TUTORIAL_SHA256 = "33017faa27dc22134fe56964b5cb56623c4043916df01921f27ee54ad1b6a79a"
A_TUTORIAL_IMAGE_CHECK = {
    "image": ("ghcr.io/bk-squared/rfx-openems@sha256:ea8df42dbf1bdcbc93479cfe0618c336"
              "95dc5266363b4873d392d10f813264ff"),
    "linux_amd64_manifest": "sha256:75473e6b3fcbc3f22deb384f6aa27a4453d55680f0c2d22c974bdf5597835bc8",
    "layer": "sha256:76b5697710c3fd381692bbacedefc96a553f0740b99ec04379a1af98b0abec0b "
             "(the RUN git clone --recursive openEMS-Project step)",
    "sha256_in_layer": A_TUTORIAL_SHA256,
    "later_layers_delete_it": False,
    "read_on": "2026-09-23, by fetching the layer from ghcr.io, not by running the image",
}
TUTORIAL_IMAGE_PATH_DEFAULT = "/tmp/openEMS-Project/openEMS/python/Tutorials/Simple_Patch_Antenna.py"

_FROZEN_TUTORIAL_BUILD = "\n".join([
    '# patch width (resonant length) in x-direction',
    'patch_width  = 32 #',
    '# patch length in y-direction',
    'patch_length = 40',
    '',
    '#substrate setup',
    'substrate_epsR   = 3.38',
    'substrate_kappa  = 1e-3 * 2*np.pi*2.45e9 * EPS0*substrate_epsR',
    'substrate_width  = 60',
    'substrate_length = 60',
    'substrate_thickness = 1.524',
    'substrate_cells = 4',
    '',
    '#setup feeding',
    'feed_pos = -6 #feeding position in x-direction',
    'feed_R = 50     #feed resistance',
    '',
    '# size of the simulation box',
    'SimBox = np.array([200, 200, 150])',
    '',
    '# setup FDTD parameter & excitation function',
    'f0 = 2e9 # center frequency',
    'fc = 1e9 # 20 dB corner frequency',
    '',
    '### FDTD setup',
    '## * Limit the simulation to 30k timesteps',
    '## * Define a reduced end criteria of -40dB',
    'FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)',
    'FDTD.SetGaussExcite( f0, fc )',
    "FDTD.SetBoundaryCond( ['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'MUR'] )",
    '',
    '',
    'CSX = ContinuousStructure()',
    'FDTD.SetCSX(CSX)',
    'mesh = CSX.GetGrid()',
    'mesh.SetDeltaUnit(1e-3)',
    'mesh_res = C0/(f0+fc)/1e-3/20',
    '',
    '### Generate properties, primitives and mesh-grid',
    '#initialize the mesh with the "air-box" dimensions',
    "mesh.AddLine('x', [-SimBox[0]/2, SimBox[0]/2])",
    "mesh.AddLine('y', [-SimBox[1]/2, SimBox[1]/2]          )",
    "mesh.AddLine('z', [-SimBox[2]/3, SimBox[2]*2/3]        )",
    '',
    '# create patch',
    "patch = CSX.AddMetal( 'patch' ) # create a perfect electric conductor (PEC)",
    'start = [-patch_width/2, -patch_length/2, substrate_thickness]',
    'stop  = [ patch_width/2 , patch_length/2, substrate_thickness]',
    "patch.AddBox(priority=10, start=start, stop=stop) # add a box-primitive to the metal property 'patch'",
    "FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2)",
    '',
    '# create substrate',
    "substrate = CSX.AddMaterial( 'substrate', epsilon=substrate_epsR, kappa=substrate_kappa)",
    'start = [-substrate_width/2, -substrate_length/2, 0]',
    'stop  = [ substrate_width/2,  substrate_length/2, substrate_thickness]',
    'substrate.AddBox( priority=0, start=start, stop=stop )',
    '',
    '# add extra cells to discretize the substrate thickness',
    "mesh.AddLine('z', np.linspace(0,substrate_thickness,substrate_cells+1))",
    '',
    '# create ground (same size as substrate)',
    "gnd = CSX.AddMetal( 'gnd' ) # create a perfect electric conductor (PEC)",
    'start[2]=0',
    'stop[2] =0',
    'gnd.AddBox(start, stop, priority=10)',
    '',
    "FDTD.AddEdges2Grid(dirs='xy', properties=gnd)",
    '',
    '# apply the excitation & resist as a current source',
    'start = [feed_pos, 0, 0]',
    'stop  = [feed_pos, 0, substrate_thickness]',
    "port = FDTD.AddLumpedPort(1, feed_R, start, stop, 'z', 1.0, priority=5, edges2grid='xy')",
    '',
    "mesh.SmoothMeshLines('all', mesh_res, 1.4)",
    '',
    '# Add the nf2ff recording box',
    'nf2ff = FDTD.CreateNF2FFBox()',
])

# The tutorial's post-processing lines this maker reproduces, verbatim, with
# their line numbers in the pinned file. --self-check parses the grid and the
# far-field call out of them.
_FROZEN_TUTORIAL_POSTPROCESSING = {
    120: 'f = np.linspace(max(1e9,f0-fc),f0+fc,401)',
    121: 'port.CalcPort(Sim_Path, f)',
    122: 's11 = port.uf_ref/port.uf_inc',
    137: 'idx = np.where((s11_dB<-10) & (s11_dB==np.min(s11_dB)))[0]',
    142: '    theta = np.arange(-180.0, 180.0, 2.0)',
    143: '    phi   = [0., 90.]',
    144: '    nf2ff_res = nf2ff.CalcNF2FF(Sim_Path, f_res, theta, phi, center=[0,0,1e-3])',
    158: 'Zin = port.uf_tot/port.if_tot',
}

A_SLICE_FIRST_LINE = "    # patch width (resonant length) in x-direction"
A_SLICE_LAST_LINE = "    nf2ff = FDTD.CreateNF2FFBox()"
A_COPY_SUBSTITUTIONS = [
    ("FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)", "FDTD = openEMS(**kw)"),
]
A_IMPORT_LINE = "    from openEMS.physical_constants import C0, EPS0"
# The names the frozen block reads without defining. The import line above plus
# the builder's two arguments and the module's numpy must supply every one.
A_FREE_NAMES = {"C0", "EPS0", "np", "openEMS", "ContinuousStructure"}

# The tutorial's own numbers, as this maker uses them. --self-check parses each
# from the frozen text above.
A_F0_HZ = 2e9
A_FC_HZ = 1e9
A_REAL_NRTS = 30000
A_REAL_END_CRITERIA = 1e-4
A_N_FREQS = 401
A_UNIT_M = 1e-3
A_PATCH_RESONANT_LENGTH_M = 32e-3
A_PATCH_WIDTH_M = 40e-3
A_SUB_EPS_R = 3.38
A_SUB_H_M = 1.524e-3
A_FARFIELD_THETA = (-180.0, 180.0, 2.0)
A_FARFIELD_PHI = (0.0, 90.0)
A_FARFIELD_CENTER = (0, 0, 1e-3)

F_TM010_TUTORIAL_HZ = f_tm010_tl_model(A_SUB_EPS_R, A_SUB_H_M, A_PATCH_RESONANT_LENGTH_M,
                                       A_PATCH_WIDTH_M)[0]


def stage_a_freqs_hz() -> np.ndarray:
    """The tutorial's own CalcPort grid: linspace(max(1e9, f0-fc), f0+fc, 401)."""
    return np.linspace(max(1e9, A_F0_HZ - A_FC_HZ), A_F0_HZ + A_FC_HZ, A_N_FREQS)


STAGE_A_BAND_HZ = (max(1e9, A_F0_HZ - A_FC_HZ), A_F0_HZ + A_FC_HZ)

# The documented result, read off the documentation's own figure. See the module
# docstring for what the figure is and where it came from.
STAGE_A_DOCUMENTED = {
    "what": ("the |S11| and input-impedance figures on openEMS's documentation page "
             "for this tutorial; the tutorial file itself prints no number"),
    "page": "python/doc/Tutorials/Simple_Patch_Antenna.rst (at A_TUTORIAL_OPENEMS_COMMIT)",
    "s11_figure": "python/doc/Tutorials/images/Simp_Patch_S11.png",
    "s11_figure_sha256": "51c2a4ec50504c91585999455e3f38ae9ac9d09b3d678ea311cb7ed90d7e5973",
    "zin_figure": "python/doc/Tutorials/images/Simp_Patch_Zin.png",
    "zin_figure_sha256": "844772c8650cd9a1fecebe899df8bd317ea3e51a6f66be0202f1554238cd31d6",
    "figures_committed_in": ("b45b615f7edf01c717963a973b9623fe26371021 (2016-09-10, "
                             "'python: massive improvements to documentation'); the "
                             "only commit that touched either file up to "
                             "A_TUTORIAL_OPENEMS_COMMIT"),
    "figure_script": ("python/Tutorials/Simple_Patch_Antenna.py at "
                      "c9435905482aa737cb888a780de262faa3b467a1 (2016-09-10), the "
                      "version committed two minutes before the figures"),
    "figure_script_differences": [
        "x lines: the 2016 script adds x = feed_pos (-6 mm) explicitly with the box "
        "edges; the pinned script adds it through the port's edges2grid='xy', which "
        "also adds y = 0 (the 2016 port call passed edges2grid='all')",
        "ground: the 2016 script passes edges2grid='all' to the ground box; the "
        "pinned script calls FDTD.AddEdges2Grid(dirs='xy', properties=gnd)",
        "patch edges: the 2016 script calls pb.AddEdges2Grid('xy', "
        "metal_edge_res=mesh_res/2) on the box, the pinned script "
        "FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2) "
        "-- the same rule in the newer API",
        "not geometry: Run(verbose=3) vs Run(); pylab vs matplotlib; the Dmax dB fix "
        "of 2023 in the pattern plot",
    ],
    "read_how": ("pixel coordinates of the S11 figure: x axis 1.0 GHz at column 100 and "
                 "3.0 GHz at column 720 (3.23 MHz per pixel); y axis +5 dB at row 60 "
                 "and -30 dB at row 540 (0.073 dB per pixel); grid lines excluded; the "
                 "black curve's lowest pixels are columns 542-545 at rows 495-496. "
                 "Zin figure: same x axis, y axis 60 ohm at row 60 and -20 ohm at row "
                 "540; the black curve's highest pixels are columns 541-542 at row 118"),
    "f_dip_hz": 2.430e9,
    "f_dip_read_half_width_hz": 0.005e9,
    "depth_db": -26.8,
    "minus10db_span_hz": [2.410e9, 2.452e9],
    "re_zin_peak_ohm": 50.3,
    "re_zin_peak_f_hz": 2.424e9,
}
# +-2 % around the documented dip. A proposal: the figure reads to 0.2 %; the
# rest is for the unmeasured differences listed above. See the module docstring.
STAGE_A_WINDOW_REL = 0.02
# The tutorial's own criterion for a resonance (its line 137: s11_dB < -10).
STAGE_A_MAX_DEPTH_DB = -10.0

# The one earlier run of the unmodified tutorial in this repository. Carried into
# every record as the tutorial's audit trail; never a gate (see the docstring).
STAGE_A_RECORDED_REPRODUCTION = {
    "vessl_run": "369367247478",
    "date": "2026-07-18",
    "openems": ("v0.37.0-rc1-2-g7b051bb: openEMS 7b051bb7433facd150294c4e8a3c7452176be238, "
                "4 commits behind A_TUTORIAL_OPENEMS_COMMIT ('fix: avoid double-free of "
                "steady-state engine extension on shutdown', 'fix: match the mode-probe "
                "coordinates to the excitation coordinates', two doc commits); the "
                "tutorial file's git blob is the same, 07d7517c"),
    "csxcad": ("v0.7.0-rc1-1-g5115701, two docstring-only commits behind the pinned "
               "build's CSXCAD e5581710"),
    "how": ("the unmodified tutorial run to completion; S11 from an external lumped-port "
            "decomposition (Z0 = 50 ohm) of its saved port_ut_1 / port_it_1 on "
            "linspace(1, 3.5 GHz, 2001), bare argmin -- not the tutorial's own CalcPort grid "
            "and not this maker's sub-bin estimator"),
    "f_s11_dip_hz": 2.4325e9,
    "s11_dip_db": -33.0,
    "mesh_lines": [49, 47, 45],
    "solver_cells": 103635,
    "dt_s": 1.07987e-12,
    "timesteps": 12046,
    "final_energy_db": -48.41,
    "wall_s": 12.03,
    "speed_mcells_per_s": 103.77,
    "log": ("tests/fixtures/patch_canonical_farfield_e4/"
            "canonical_nf2ff_reverify_369367247478_completed.log, lines 714-764 "
            "(committed in 7fcc7610)"),
    "role": "AUDIT TRAIL for the tutorial; never this run's gate",
}


def stage_a_gate_verdict(f_res_hz: float, depth_db: float) -> dict:
    """The reproduce gate: the documented frequency AND an actual dip.

    Pure arithmetic, so --self-check and the tests can plant curves at it.
    """
    f_doc = STAGE_A_DOCUMENTED["f_dip_hz"]
    lo = f_doc * (1.0 - STAGE_A_WINDOW_REL)
    hi = f_doc * (1.0 + STAGE_A_WINDOW_REL)
    f_ok = bool(lo <= f_res_hz <= hi)
    depth_ok = bool(depth_db <= STAGE_A_MAX_DEPTH_DB)
    return {
        "measured_f_hz": float(f_res_hz),
        "measured_depth_db": float(depth_db),
        "documented_f_hz": float(f_doc),
        "deviation_pct": (float(f_res_hz) - f_doc) / f_doc * 100.0,
        "window_hz": [lo, hi],
        "window_rel": STAGE_A_WINDOW_REL,
        "max_depth_db": STAGE_A_MAX_DEPTH_DB,
        "tl_model_f_hz": float(F_TM010_TUTORIAL_HZ),
        "measured_over_tl_model": float(f_res_hz) / F_TM010_TUTORIAL_HZ,
        "f_ok": f_ok,
        "depth_ok": depth_ok,
        "passed": bool(f_ok and depth_ok),
    }


# ---------------------------------------------------------------------------
# STAGE A builder: the frozen block, indented, with the one declared
# substitution. --self-check derives it from _FROZEN_TUTORIAL_BUILD and compares
# character for character.
# ---------------------------------------------------------------------------
def _build_stage_a_tutorial(ContinuousStructure, openEMS, *,
                            nrts: int | None, end_criteria: float | None):
    """openEMS's Simple Patch Antenna tutorial, verbatim. Returns (FDTD, port, nf2ff)."""
    kw = {}
    if nrts is not None:
        kw["NrTS"] = nrts
    if end_criteria is not None:
        kw["EndCriteria"] = end_criteria
    from openEMS.physical_constants import C0, EPS0
    # patch width (resonant length) in x-direction
    patch_width  = 32 #
    # patch length in y-direction
    patch_length = 40

    #substrate setup
    substrate_epsR   = 3.38
    substrate_kappa  = 1e-3 * 2*np.pi*2.45e9 * EPS0*substrate_epsR
    substrate_width  = 60
    substrate_length = 60
    substrate_thickness = 1.524
    substrate_cells = 4

    #setup feeding
    feed_pos = -6 #feeding position in x-direction
    feed_R = 50     #feed resistance

    # size of the simulation box
    SimBox = np.array([200, 200, 150])

    # setup FDTD parameter & excitation function
    f0 = 2e9 # center frequency
    fc = 1e9 # 20 dB corner frequency

    ### FDTD setup
    ## * Limit the simulation to 30k timesteps
    ## * Define a reduced end criteria of -40dB
    FDTD = openEMS(**kw)
    FDTD.SetGaussExcite( f0, fc )
    FDTD.SetBoundaryCond( ['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'MUR'] )


    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)
    mesh_res = C0/(f0+fc)/1e-3/20

    ### Generate properties, primitives and mesh-grid
    #initialize the mesh with the "air-box" dimensions
    mesh.AddLine('x', [-SimBox[0]/2, SimBox[0]/2])
    mesh.AddLine('y', [-SimBox[1]/2, SimBox[1]/2]          )
    mesh.AddLine('z', [-SimBox[2]/3, SimBox[2]*2/3]        )

    # create patch
    patch = CSX.AddMetal( 'patch' ) # create a perfect electric conductor (PEC)
    start = [-patch_width/2, -patch_length/2, substrate_thickness]
    stop  = [ patch_width/2 , patch_length/2, substrate_thickness]
    patch.AddBox(priority=10, start=start, stop=stop) # add a box-primitive to the metal property 'patch'
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2)

    # create substrate
    substrate = CSX.AddMaterial( 'substrate', epsilon=substrate_epsR, kappa=substrate_kappa)
    start = [-substrate_width/2, -substrate_length/2, 0]
    stop  = [ substrate_width/2,  substrate_length/2, substrate_thickness]
    substrate.AddBox( priority=0, start=start, stop=stop )

    # add extra cells to discretize the substrate thickness
    mesh.AddLine('z', np.linspace(0,substrate_thickness,substrate_cells+1))

    # create ground (same size as substrate)
    gnd = CSX.AddMetal( 'gnd' ) # create a perfect electric conductor (PEC)
    start[2]=0
    stop[2] =0
    gnd.AddBox(start, stop, priority=10)

    FDTD.AddEdges2Grid(dirs='xy', properties=gnd)

    # apply the excitation & resist as a current source
    start = [feed_pos, 0, 0]
    stop  = [feed_pos, 0, substrate_thickness]
    port = FDTD.AddLumpedPort(1, feed_R, start, stop, 'z', 1.0, priority=5, edges2grid='xy')

    mesh.SmoothMeshLines('all', mesh_res, 1.4)

    # Add the nf2ff recording box
    nf2ff = FDTD.CreateNF2FFBox()
    return FDTD, port, nf2ff


# ---------------------------------------------------------------------------
# STAGE B: the retired script's run_openems build block, frozen verbatim.
# ---------------------------------------------------------------------------
_FROZEN_BUILDER_SLICE = "\n".join([
    '    unit = 1e-3',
    '    Lp, Wp = L_PATCH * 1e3, W_PATCH * 1e3',
    '    h = H_SUB * 1e3',
    '    gpx, gpy = GP_X * 1e3, GP_Y * 1e3',
    '    feed = FEED_OFFSET_X * 1e3',
    '    f0, fc = 2.4e9, 1.2e9',
    '',
    '    FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)',
    '    FDTD.SetGaussExcite(f0, fc)',
    "    FDTD.SetBoundaryCond(['MUR'] * 6)",
    '    CSX = ContinuousStructure(); FDTD.SetCSX(CSX)',
    '    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)',
    '    mesh_res = C0 / (f0 + fc) / unit / 30        # ~lambda/30 in air (well-resolved)',
    '    patch_res = 1.2                              # dense lines across patch (converged ref)',
    '',
    '    # air box: generous (openEMS is cheap) -> trustworthy far-field reference',
    '    air = 60.0',
    "    mesh.AddLine('x', [-gpx / 2 - air, gpx / 2 + air])",
    "    mesh.AddLine('y', [-gpy / 2 - air, gpy / 2 + air])",
    "    mesh.AddLine('z', [-40.0, 90.0])",
    '    # explicit fine lines across the patch (+2 mm skirt) so openEMS is at least',
    "    # as converged laterally as rfx's uniform 0.79 mm -> a fair reference.",
    "    mesh.AddLine('x', np.arange(-Lp / 2 - 2, Lp / 2 + 2 + patch_res, patch_res))",
    "    mesh.AddLine('y', np.arange(-Wp / 2 - 2, Wp / 2 + 2 + patch_res, patch_res))",
    '',
    "    patch = CSX.AddMetal('patch')",
    '    patch.AddBox(priority=10, start=[-Lp / 2, -Wp / 2, h], stop=[Lp / 2, Wp / 2, h])',
    "    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=patch_res / 2)",
    '    sub_kappa = 2 * np.pi * 2.4e9 * 8.8541878128e-12 * EPS_R * TAN_DELTA',
    "    substrate = CSX.AddMaterial('sub', epsilon=EPS_R, kappa=sub_kappa)",
    '    substrate.AddBox(priority=0, start=[-gpx / 2, -gpy / 2, 0], stop=[gpx / 2, gpy / 2, h])',
    "    mesh.AddLine('z', np.linspace(0, h, N_SUB + 1))",
    "    gnd = CSX.AddMetal('gnd')",
    '    gnd.AddBox([-gpx / 2, -gpy / 2, 0], [gpx / 2, gpy / 2, 0], priority=10)',
    "    FDTD.AddEdges2Grid(dirs='xy', properties=gnd)",
    "    port = FDTD.AddLumpedPort(1, 50.0, [feed, 0, 0], [feed, 0, h], 'z', 1.0,",
    "                              priority=5, edges2grid='xy')",
    "    mesh.SmoothMeshLines('all', mesh_res, 1.4)",
    '    nf2ff = FDTD.CreateNF2FFBox() if do_gain else None',
])
B_SLICE_FIRST_LINE = "    unit = 1e-3"
B_SLICE_LAST_LINE = "    nf2ff = FDTD.CreateNF2FFBox() if do_gain else None"

# Two substitutions turn the retired block into this board at rung 1.0: the
# stop criteria come from the caller (delta 2) and every face is PML (delta 1).
B_COPY_SUBSTITUTIONS = [
    ("    FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)",
     "    FDTD = openEMS(**kw)"),
    ("    FDTD.SetBoundaryCond(['MUR'] * 6)",
     "    FDTD.SetBoundaryCond(['PML_8'] * 6)"),
]
# Three more make it a rung (delta 3): the air resolution, the patch-region
# resolution (which also scales the thirds-rule offsets, metal_edge_res =
# patch_res / 2) and the substrate's own z cells.
B_RUNG_SUBSTITUTIONS = [
    ("    mesh_res = C0 / (f0 + fc) / unit / 30        # ~lambda/30 in air (well-resolved)",
     "    mesh_res = C0 / (f0 + fc) / unit / 30 * resolution_factor  # ~lambda/30 in air, x rung"),
    ("    patch_res = 1.2                              # dense lines across patch (converged ref)",
     "    patch_res = 1.2 * resolution_factor          # dense lines across patch, x rung"),
    ("    mesh.AddLine('z', np.linspace(0, h, N_SUB + 1))",
     "    mesh.AddLine('z', np.linspace(0, h, substrate_z_cells(resolution_factor) + 1))"),
]


def substrate_z_cells(resolution_factor: float) -> int:
    """Substrate cells at a rung: round(N_SUB / factor) -- 4, 6, 8 at 1, 1/sqrt2, 1/2."""
    return max(1, int(round(N_SUB / resolution_factor)))


def _build_patch_board_at_rung(ContinuousStructure, openEMS, *,
                               nrts: int | None, end_criteria: float | None,
                               resolution_factor: float, do_gain: bool):
    """The RT5880 board at a mesh rung. Returns (FDTD, port, nf2ff or None)."""
    kw = {}
    if nrts is not None:
        kw["NrTS"] = nrts
    if end_criteria is not None:
        kw["EndCriteria"] = end_criteria
    unit = 1e-3
    Lp, Wp = L_PATCH * 1e3, W_PATCH * 1e3
    h = H_SUB * 1e3
    gpx, gpy = GP_X * 1e3, GP_Y * 1e3
    feed = FEED_OFFSET_X * 1e3
    f0, fc = 2.4e9, 1.2e9

    FDTD = openEMS(**kw)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['PML_8'] * 6)
    CSX = ContinuousStructure(); FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)
    mesh_res = C0 / (f0 + fc) / unit / 30 * resolution_factor  # ~lambda/30 in air, x rung
    patch_res = 1.2 * resolution_factor          # dense lines across patch, x rung

    # air box: generous (openEMS is cheap) -> trustworthy far-field reference
    air = 60.0
    mesh.AddLine('x', [-gpx / 2 - air, gpx / 2 + air])
    mesh.AddLine('y', [-gpy / 2 - air, gpy / 2 + air])
    mesh.AddLine('z', [-40.0, 90.0])
    # explicit fine lines across the patch (+2 mm skirt) so openEMS is at least
    # as converged laterally as rfx's uniform 0.79 mm -> a fair reference.
    mesh.AddLine('x', np.arange(-Lp / 2 - 2, Lp / 2 + 2 + patch_res, patch_res))
    mesh.AddLine('y', np.arange(-Wp / 2 - 2, Wp / 2 + 2 + patch_res, patch_res))

    patch = CSX.AddMetal('patch')
    patch.AddBox(priority=10, start=[-Lp / 2, -Wp / 2, h], stop=[Lp / 2, Wp / 2, h])
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=patch_res / 2)
    sub_kappa = 2 * np.pi * 2.4e9 * 8.8541878128e-12 * EPS_R * TAN_DELTA
    substrate = CSX.AddMaterial('sub', epsilon=EPS_R, kappa=sub_kappa)
    substrate.AddBox(priority=0, start=[-gpx / 2, -gpy / 2, 0], stop=[gpx / 2, gpy / 2, h])
    mesh.AddLine('z', np.linspace(0, h, substrate_z_cells(resolution_factor) + 1))
    gnd = CSX.AddMetal('gnd')
    gnd.AddBox([-gpx / 2, -gpy / 2, 0], [gpx / 2, gpy / 2, 0], priority=10)
    FDTD.AddEdges2Grid(dirs='xy', properties=gnd)
    port = FDTD.AddLumpedPort(1, 50.0, [feed, 0, 0], [feed, 0, h], 'z', 1.0,
                              priority=5, edges2grid='xy')
    mesh.SmoothMeshLines('all', mesh_res, 1.4)
    nf2ff = FDTD.CreateNF2FFBox() if do_gain else None
    return FDTD, port, nf2ff


# ---------------------------------------------------------------------------
# Stage B's run parameters.
# ---------------------------------------------------------------------------
B_UNIT_M = 1e-3
B_F0_HZ, B_FC_HZ = 2.4e9, 1.2e9       # the builder's own SetGaussExcite(f0, fc)
B_N_FREQS = 901                        # 2.0 MHz bins over F_LO..F_HI
B_BOUNDARY = ["PML_8"] * 6
B_PML_CELLS = 8
B_REAL_NRTS = None                     # openEMS's own ~1e9 (delta 2)
B_REAL_END_CRITERIA = None             # openEMS's own 1e-5 = -50 dB (delta 2)
RETIRED_NRTS_CAP = 30000
RETIRED_END_CRITERIA_CAP = 1e-4
B_COARSE_RESOLUTION_FACTOR = 1.0
B_MID_RESOLUTION_FACTOR = 1.0 / np.sqrt(2.0)
B_FINE_RESOLUTION_FACTOR = 0.5
STAGE_B_FACTORS = {
    "stage_b_coarse": B_COARSE_RESOLUTION_FACTOR,
    "stage_b_mid": B_MID_RESOLUTION_FACTOR,
    "stage_b_fine": B_FINE_RESOLUTION_FACTOR,
}
STAGE_NAMES = ("stage_a", "stage_b_coarse", "stage_b_mid", "stage_b_fine")
# The retired script's own search window for the dip: 0.80-1.20 x its analytic
# TM010 (its lines 1142-1143), here handed to the shared sub-bin estimator.
B_RESONANCE_BAND_HZ = (0.80 * F_TM010_BOARD_HZ, 1.20 * F_TM010_BOARD_HZ)
B_WITNESS_BAND_HZ = (F_LO, F_HI)
# The retired script's condition for computing the far field at all.
B_FARFIELD_MAX_DEPTH_DB = -6.0
B_FARFIELD_THETA = (-180.0, 180.0, 2.0)
B_FARFIELD_PHI = (0.0, 90.0)
B_FARFIELD_CENTER = (0, 0, 1e-3)
MINUS10_DB = -10.0


def stage_b_freqs_hz() -> np.ndarray:
    return np.linspace(F_LO, F_HI, B_N_FREQS)


DELTA_LIST = [
    "DELTA 1 (boundaries): ['MUR'] * 6 becomes ['PML_8'] * 6 -- an absorbing "
    "layer on every face of the same box (+-60 mm of air beyond the ground in x "
    "and y, z from -40 to +90 mm). PML_8 is the outermost eight cells of each "
    "face, inside that box; the run records where each PML face realized. Why: "
    "the known-issues ledger's Sheen entry -- MUR side walls 3 mm from a "
    "substrate held the Sheen board's box energy flat at about -31 dB for 6 h, "
    "and PML on the same faces ended the run by its energy criterion in 1.4 ns. "
    "Here the substrate stops at the ground's 56 x 66 mm edge, 60 mm inside the "
    "box, so no face touches dielectric.",
    "DELTA 2 (stop criteria): openEMS(NrTS=30000, EndCriteria=1e-4) becomes "
    "openEMS(**kw) with nothing passed on the real pass, so openEMS's own "
    "NrTS ~1e9 and EndCriteria 1e-5 (-50 dB of the box energy's peak) apply and "
    "every real pass ends on its energy criterion; a pass that reaches its step "
    "cap instead is a failed gate. Why: the retired cap was one mesh's cost "
    "choice. At the explicit-line CFL estimate of each rung's timestep "
    "(0.464 / 0.047 / 0.308 ps) 30000 steps are 13.9 / 1.4 / 9.2 ns of "
    "simulated time, against a planned 10 ns of pulse and ring-down. The "
    "smoke pass keeps 200 steps at EndCriteria 0.",
    "DELTA 3 (mesh rungs): three rungs, resolution factors 1.0 (stage_b_coarse), "
    "1/sqrt(2) = 0.70711 (stage_b_mid) and 0.5 (stage_b_fine). The factor "
    "multiplies mesh_res (lambda/30 in air at 3.6 GHz, 2.776 mm at rung 1), "
    "patch_res (the 1.2 mm lines across the patch and its 2 mm skirt; the "
    "thirds-rule offsets at the patch edges follow, since metal_edge_res = "
    "patch_res / 2) and the substrate's own z cells: linspace(0, h, N_SUB + 1) "
    "becomes linspace(0, h, round(4/factor) + 1), so the 3.175 mm board carries "
    "4, 6 and 8 cells (793.75, 529.17, 396.88 um). The air box does not scale. "
    "Rung 1.0 is the retired mesh exactly: the three substituted expressions "
    "evaluate to the retired values at factor 1.",
    "DELTA 4 (frequency grid): the retired CalcPort grid linspace(F_LO, F_HI, "
    "n_freqs) with its default 181 points becomes 901 points over the same "
    "1.6-3.4 GHz, 2.0 MHz per bin. The band sits inside the excitation's 20 dB "
    "corners (1.2-3.6 GHz) and covers 1.8-3.2 GHz; one bin is 0.08 % of 2.4 GHz.",
    "DELTA 5 (resonance estimator): the retired bare argmin of |S11| in dB over "
    "0.80-1.20 x its analytic TM010 becomes the shared sub-bin estimator "
    "(validation/crossval/comparators/spectral_features.py::refined_extremum, "
    "log |S11|) over the same window, with its half-grid witness. The bin and "
    "the refined frequency are both recorded, and so are the -10 dB band "
    "(band_at_level) and Zin = uf_tot / if_tot at the resonance. Nothing is "
    "gated on them.",
    "DELTA 6 (far field): the retired CalcNF2FF call -- theta -180..178 step 2, "
    "phi 0 and 90, center [0, 0, 1e-3], only when the dip is below -6 dB -- is "
    "kept, and made at the refined resonance instead of the bin. The recording "
    "box is created on every rung unless --no-nf2ff. Reported only; a failure "
    "is recorded, not raised. The dump files are measured and then deleted from "
    "the pod's /tmp.",
    "DELTA 7 (how the solver runs): the retired FDTD.Run(sim_path, cleanup=True, "
    "numThreads=8) becomes the shared module's capture, Run(sim_path, "
    "cleanup=True, verbose=1, numThreads=8) with stdout and stderr in a file, "
    "after a 200-step smoke pass, with the shared sanity gates around it.",
    "NOTHING ELSE: the patch, the substrate (eps_r 2.2 with tan delta 1e-3 as a "
    "conductivity at 2.4 GHz), the 56 x 66 mm ground, the 50 ohm lumped port "
    "from z = 0 to z = h at x = -9 mm, SetGaussExcite(2.4e9, 1.2e9), the air box, "
    "the 1.2 mm patch lines, the thirds rule, SmoothMeshLines('all', mesh_res, "
    "1.4) and the mm length unit are the retired builder's, proved so character "
    "for character by --self-check.",
]


# ---------------------------------------------------------------------------
# The copy proofs' plumbing.
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    env = os.environ.get("RFX_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[4]


def _slice(text: str, first: str, last: str, what: str) -> str:
    """``text`` from the line ``first`` to the line ``last``, both included."""
    i = text.find(first)
    if i < 0:
        raise RuntimeError(f"{what}: the anchor {first!r} is not there")
    j = text.rfind(last)
    if j < 0 or j < i:
        raise RuntimeError(f"{what}: the anchor {last!r} is not there")
    return text[i:j + len(last)]


def _retired_text():
    """The retired script off disk, or None once it has been removed."""
    path = _repo_root() / RETIRED_REL_PATH
    return path.read_text() if path.is_file() else None


def _retired_function_source(text: str, name: str) -> str:
    lines = text.split("\n")
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    raise RuntimeError(f"{RETIRED_REL_PATH} has no top-level {name}")


def _copy_proof_b() -> dict:
    """The frozen slice + the declared substitutions == the rung builder's block."""
    import inspect

    mine = _slice(inspect.getsource(_build_patch_board_at_rung), B_SLICE_FIRST_LINE,
                  B_SLICE_LAST_LINE, "_build_patch_board_at_rung")
    derived = _FROZEN_BUILDER_SLICE
    counts = {}
    for old, new in B_COPY_SUBSTITUTIONS + B_RUNG_SUBSTITUTIONS:
        counts[old] = derived.count(old)
        derived = derived.replace(old, new)
    text = _retired_text()
    on_disk = None
    if text is not None:
        on_disk = _slice(_retired_function_source(text, RETIRED_FUNCTION),
                         B_SLICE_FIRST_LINE, B_SLICE_LAST_LINE,
                         f"{RETIRED_REL_PATH}::{RETIRED_FUNCTION}")
    return {
        "mine": mine,
        "derived": derived,
        "matches": derived == mine,
        "counts": counts,
        "retired_on_disk": text is not None,
        "frozen_matches_disk": None if on_disk is None else (on_disk == _FROZEN_BUILDER_SLICE),
        "on_disk": on_disk,
    }


def _indent(block: str) -> str:
    return "\n".join(("    " + ln) if ln else ln for ln in block.split("\n"))


def _copy_proof_a() -> dict:
    """The frozen tutorial block, indented, + one substitution == the Stage A builder's block."""
    body = _builder_body_after_kw(_build_stage_a_tutorial)
    mine = _slice(body, A_SLICE_FIRST_LINE, A_SLICE_LAST_LINE, "_build_stage_a_tutorial")
    derived = _FROZEN_TUTORIAL_BUILD
    counts = {}
    for old, new in A_COPY_SUBSTITUTIONS:
        counts[old] = derived.count(old)
        derived = derived.replace(old, new)
    derived = _indent(derived)
    head = body[:body.find(A_SLICE_FIRST_LINE)] if A_SLICE_FIRST_LINE in body else ""
    return {
        "mine": mine,
        "derived": derived,
        "matches": derived == mine,
        "counts": counts,
        "import_line_present": A_IMPORT_LINE in head.split("\n"),
    }


def _free_names(block: str) -> set:
    """Names the (top-level) block reads that it never assigns."""
    tree = ast.parse(block)
    loaded, stored = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (stored if isinstance(node.ctx, ast.Store) else loaded).add(node.id)
    return loaded - stored


def _frozen_call_kwargs(line: str) -> dict:
    """The literal keyword arguments of the one call on ``line``."""
    call = ast.parse(line.strip()).body[0].value
    return {k.arg: ast.literal_eval(k.value) for k in call.keywords}


def _frozen_assignment(block: str, name: str):
    for node in ast.parse(block).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def _tutorial_source_in_image(path: str | None = None) -> dict:
    """The container's own copy of the tutorial, checked against the frozen one."""
    p = Path(path or os.environ.get("RFX_OPENEMS_TUTORIAL_PATH") or TUTORIAL_IMAGE_PATH_DEFAULT)
    if not p.is_file():
        return {"path": str(p), "present": False,
                "note": "not in this container; the frozen copy's provenance is the "
                        "pinned commit and sha256 alone"}
    data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    return {
        "path": str(p),
        "present": True,
        "sha256": sha,
        "sha256_matches_pinned": sha == A_TUTORIAL_SHA256,
        "frozen_block_found_verbatim": _FROZEN_TUTORIAL_BUILD in data.decode("utf-8", "replace"),
    }


def _tutorial_source_refusal(info: dict):
    """Why a real run must not start with this container's tutorial, or None."""
    if not info.get("present"):
        return None
    why = []
    if not info.get("sha256_matches_pinned"):
        why.append(f"its sha256 is {info.get('sha256')}, not the pinned {A_TUTORIAL_SHA256}")
    if not info.get("frozen_block_found_verbatim"):
        why.append("the frozen build block is not in it verbatim")
    if not why:
        return None
    return f"the image's tutorial at {info['path']} differs from the frozen one: " + "; ".join(why)


# ---------------------------------------------------------------------------
# Pure-numpy mesh plan. No openEMS, no CSXCAD.
#
# Every line the builders add EXPLICITLY is reproduced exactly: the box edges,
# the arange across the patch, the thirds-rule pair at each patch edge (1/3 of
# metal_edge_res inside the metal, 2/3 outside, the rule in openEMS's
# automesh.mesh_hint_from_box), the ground edges, the port's x and y (its
# edges2grid='xy') and the substrate's z lines. Near-coincident lines are merged
# the way CSXCAD merges them (closer than 1e-7 of the mean spacing). What
# CSXCAD's SmoothMeshLines does between them is NOT reproduced: the shared
# ``_smooth_estimate`` subdivides each gap evenly. Checked outside this file
# with CSXCAD's own SmoothMeshLines.py from the pinned image: the same explicit
# lines give the tutorial's 49 x 47 x 45 lines exactly (run 369367247478 printed
# "49x47x45"), where the even estimate gives 46 x 45 x 36; for the three Stage B
# rungs the estimate is within 5 % of CSXCAD's cell count. The timestep estimate
# is the vacuum CFL of the smallest explicit gaps; smoothing can add a smaller
# cell (2.10 mm next to the tutorial's 2.50 mm explicit minimum), and openEMS's
# own dt on the tutorial was 0.87 x that estimate (the calibration line of the
# dry run).
# ---------------------------------------------------------------------------
def _merge_close(lines, rel_tol: float = 1e-7) -> np.ndarray:
    lines = np.unique(np.asarray(lines, dtype=float))
    if lines.size < 3:
        return lines
    tol = float(np.mean(np.diff(lines))) * rel_tol
    keep = [lines[0]]
    for v in lines[1:]:
        if v - keep[-1] >= tol:
            keep.append(v)
    return np.asarray(keep)


def _thirds(lo: float, hi: float, res: float) -> list:
    """openEMS's metal-edge pair at both edges of [lo, hi] (automesh.mesh_hint_from_box)."""
    mer = np.array([-1.0, 2.0]) / 3.0 * res
    if hi - lo <= res:
        return [lo, hi]
    return [lo - mer[0], lo - mer[1], hi + mer[0], hi + mer[1]]


def _explicit_lines_b(resolution_factor: float) -> dict:
    """The rung builder's explicit lines in mm, before SmoothMeshLines."""
    Lp, Wp = L_PATCH * 1e3, W_PATCH * 1e3
    h = H_SUB * 1e3
    gpx, gpy = GP_X * 1e3, GP_Y * 1e3
    feed = FEED_OFFSET_X * 1e3
    patch_res = 1.2 * resolution_factor
    air = 60.0
    x = [-gpx / 2 - air, gpx / 2 + air]
    y = [-gpy / 2 - air, gpy / 2 + air]
    z = [-40.0, 90.0]
    x += list(np.arange(-Lp / 2 - 2, Lp / 2 + 2 + patch_res, patch_res))
    y += list(np.arange(-Wp / 2 - 2, Wp / 2 + 2 + patch_res, patch_res))
    x += _thirds(-Lp / 2, Lp / 2, patch_res / 2)
    y += _thirds(-Wp / 2, Wp / 2, patch_res / 2)
    z += list(np.linspace(0, h, substrate_z_cells(resolution_factor) + 1))
    x += [-gpx / 2, gpx / 2]
    y += [-gpy / 2, gpy / 2]
    x += [feed]
    y += [0.0]
    return {"x": _merge_close(x), "y": _merge_close(y), "z": _merge_close(z)}


def _explicit_lines_a() -> dict:
    """The tutorial's explicit lines in mm, before SmoothMeshLines."""
    mesh_res = C0 / (A_F0_HZ + A_FC_HZ) / 1e-3 / 20
    x = [-100.0, 100.0] + _thirds(-16.0, 16.0, mesh_res / 2) + [-30.0, 30.0] + [-6.0]
    y = [-100.0, 100.0] + _thirds(-20.0, 20.0, mesh_res / 2) + [-30.0, 30.0] + [0.0]
    z = [-50.0, 100.0] + list(np.linspace(0, 1.524, 5))
    return {"x": _merge_close(x), "y": _merge_close(y), "z": _merge_close(z)}


def _smallest_spacing(lines: np.ndarray) -> dict:
    d = np.diff(lines)
    k = int(np.argmin(d))
    return {"mm": float(d[k]), "between_mm": [float(lines[k]), float(lines[k + 1])]}


def _cfl_dt_s(explicit: dict, unit_m: float) -> float:
    """Vacuum CFL limit of the smallest spacing per axis (openEMS automesh's estimate)."""
    inv = sum(float(np.min(np.diff(explicit[ax]))) ** -2 for ax in ("x", "y", "z"))
    return unit_m / (C0 * math.sqrt(inv))


# The cost model. Every number has its source beside it.
PLAN_SPEED_MC_PER_S = 200.0
PLAN_SPEED_SOURCE = (
    "openEMS on 8 threads on remilab-c0 ran 24.8 M cells x 28120 steps in 2848 s "
    "= 245 MC/s (VESSL 369367263789, a Sheen-board mesh rung, 2026-09-23); "
    "planned at 200 MC/s")
PLAN_Q = 10.06
PLAN_Q_SOURCE = ("rfx's ring-down Q of this same board, "
                 "validation/crossval/_15_patch_results/rfx.json::q_harminv = 10.0605 "
                 "at b25df603")
PLAN_END_CRITERIA = 1e-5          # openEMS's library default, delta 2
PLAN_MARGIN = 2.0
PLAN_JOB_BUDGET_S = 21600.0


def _pulse_length_s(fc_hz: float) -> float:
    """openEMS's Gaussian pulse length, 2 * 9 / (2 pi fc) (FDTD/excitation.cpp)."""
    return 2.0 * 9.0 / (2.0 * math.pi * fc_hz)


def _planned_record_s() -> dict:
    t_pulse = _pulse_length_s(B_FC_HZ)
    t_ring = math.log(1.0 / PLAN_END_CRITERIA) * PLAN_Q / (2.0 * math.pi * F_TM010_BOARD_HZ)
    return {"pulse_s": t_pulse, "ring_down_s": t_ring, "sum_s": t_pulse + t_ring,
            "planned_s": PLAN_MARGIN * (t_pulse + t_ring)}


def _nf2ff_dump_estimate_bytes(lines: dict, record_s: float, f_max_hz: float,
                               boundary_cells: int) -> float:
    """E and H, three components each, on the six faces of the box openEMS's
    CreateNF2FFBox puts one line inside each boundary layer, sampled every
    Nyquist/4 steps (openems.cpp, m_OverSampling = 4), 4 bytes a value
    (float32 -- assumed, not read)."""
    n = {ax: max(len(lines[ax]) - 2 * (boundary_cells + 1), 2) - 1 for ax in ("x", "y", "z")}
    faces = 2 * (n["x"] * n["y"] + n["x"] * n["z"] + n["y"] * n["z"])
    samples = record_s * 8.0 * f_max_hz
    return float(faces * 6 * 4 * samples)


def _plan_b(label: str, resolution_factor: float) -> dict:
    ex = _explicit_lines_b(resolution_factor)
    mesh_res = C0 / (B_F0_HZ + B_FC_HZ) / B_UNIT_M / 30 * resolution_factor
    patch_res = 1.2 * resolution_factor
    est = {ax: _smooth_estimate(ex[ax], mesh_res) for ax in ("x", "y", "z")}
    cells = int(np.prod([est[ax].size - 1 for ax in ("x", "y", "z")]))
    dt = _cfl_dt_s(ex, B_UNIT_M)
    rec = _planned_record_s()
    steps = int(math.ceil(rec["planned_s"] / dt))
    cost_s = cells * steps / (PLAN_SPEED_MC_PER_S * 1e6)
    n_sub = substrate_z_cells(resolution_factor)
    pml = {}
    for ax in ("x", "y", "z"):
        ln = est[ax]
        pml[ax] = [float(ln[B_PML_CELLS]), float(ln[-1 - B_PML_CELLS])]
    return {
        "label": label,
        "resolution_factor": float(resolution_factor),
        "mesh_res_mm": float(mesh_res),
        "patch_res_mm": float(patch_res),
        "metal_edge_res_mm": float(patch_res / 2),
        "thirds_rule_offsets_mm": [float(v) for v in np.array([-1.0, 2.0]) / 3.0 * patch_res / 2],
        "substrate_z_cells": n_sub,
        "substrate_z_cells_in_planned_lines": int(np.sum(
            (est["z"] >= -1e-9) & (est["z"] <= H_SUB * 1e3 + 1e-9)) - 1),
        "substrate_z_step_um": float(H_SUB * 1e6 / n_sub),
        "explicit_lines": {ax: int(ex[ax].size) for ax in ("x", "y", "z")},
        "mesh_lines_estimate": {ax: int(est[ax].size) for ax in ("x", "y", "z")},
        "cells_estimate": cells,
        "smallest_explicit_spacing": {ax: _smallest_spacing(ex[ax]) for ax in ("x", "y", "z")},
        "cfl_dt_s": dt,
        "retired_cap_covers_s": RETIRED_NRTS_CAP * dt,
        "planned_record_s": rec["planned_s"],
        "planned_steps": steps,
        "planned_cost_s": cost_s,
        "pml_inner_faces_mm_estimate": pml,
        "nf2ff_dump_bytes_estimate": _nf2ff_dump_estimate_bytes(
            est, rec["planned_s"], B_F0_HZ + B_FC_HZ, B_PML_CELLS),
    }


def _plan_a() -> dict:
    ex = _explicit_lines_a()
    mesh_res = C0 / (A_F0_HZ + A_FC_HZ) / A_UNIT_M / 20
    est = {ax: _smooth_estimate(ex[ax], mesh_res) for ax in ("x", "y", "z")}
    cells = int(np.prod([est[ax].size - 1 for ax in ("x", "y", "z")]))
    dt = _cfl_dt_s(ex, A_UNIT_M)
    steps = A_REAL_NRTS                   # the tutorial's own cap, an upper bound
    return {
        "label": "stage_a",
        "mesh_res_mm": float(mesh_res),
        "explicit_lines": {ax: int(ex[ax].size) for ax in ("x", "y", "z")},
        "mesh_lines_estimate": {ax: int(est[ax].size) for ax in ("x", "y", "z")},
        "cells_estimate": cells,
        "smallest_explicit_spacing": {ax: _smallest_spacing(ex[ax]) for ax in ("x", "y", "z")},
        "cfl_dt_s": dt,
        "cap_covers_s": A_REAL_NRTS * dt,
        "pulse_s": _pulse_length_s(A_FC_HZ),
        "planned_steps": steps,
        "planned_cost_s": cells * steps / (PLAN_SPEED_MC_PER_S * 1e6),
    }


def _stage_plans(stages) -> dict:
    out = {}
    for name in stages:
        out[name] = _plan_a() if name == "stage_a" else _plan_b(name, STAGE_B_FACTORS[name])
    return out


# ---------------------------------------------------------------------------
# Features: reported, never gated here (Stage A's gate reads two of them).
# ---------------------------------------------------------------------------
def _tutorial_pick(freqs_hz, s11) -> dict:
    """The tutorial's own resonance pick, its line 137, verbatim in effect."""
    f = np.asarray(freqs_hz, dtype=float)
    s11_dB = 20.0 * np.log10(np.abs(np.asarray(s11)))
    idx = np.where((s11_dB < -10) & (s11_dB == np.min(s11_dB)))[0]
    rule = _FROZEN_TUTORIAL_POSTPROCESSING[137]
    if not len(idx) == 1:
        return {"f_hz": None, "rule": rule,
                "note": "no single bin is both the minimum and below -10 dB; the tutorial "
                        "prints 'No resonance frequency found' and computes no far field"}
    return {"f_hz": float(f[idx[0]]), "s11_db": float(s11_dB[idx[0]]), "rule": rule}


def _one_port_features(sf, band_hz, *, with_tutorial_pick: bool = False):
    """The |S11| minimum in ``band_hz`` and what goes with it."""
    lo, hi = band_hz[0] / 1e9, band_hz[1] / 1e9

    def features(freqs_hz, s11, zin) -> dict:
        f_ghz = np.asarray(freqs_hz, dtype=float) / 1e9
        mag = np.abs(np.asarray(s11))
        zin = np.asarray(zin, dtype=np.complex128)
        out: dict = {}
        try:
            r = sf.refined_extremum(f_ghz, mag, lo, hi, transform="log")
            res = {
                "bin_f_ghz": float(r["bin_f"]),
                "refined_f_ghz": float(r["refined_f"]),
                "depth_db": float(r["depth_db"]),
                "sub_bin_shift_bins": float(r["sub_bin_shift"]),
                "bin_width_ghz": float(r["bin_width"]),
                "index": int(r["index"]),
                "band_ghz": [lo, hi],
                "estimator": "validation/crossval/comparators/spectral_features.py::"
                             "refined_extremum, transform='log'",
            }
            hg = sf.half_grid_witness(f_ghz, mag, lo, hi, transform="log")
            res["half_grid_witness"] = {
                "refined_ghz": [float(v) for v in hg["refined"]],
                "spread_mhz": float(hg["spread"]) * 1e3,
                "spread_bins": float(hg["spread_bins"]),
            }
            band = sf.band_at_level(f_ghz, mag, MINUS10_DB, r["index"])
            if band is None:
                res["band_minus10db"] = None
            else:
                f_lo, f_hi, n_bins = band
                res["band_minus10db"] = {
                    "f_lo_ghz": float(f_lo), "f_hi_ghz": float(f_hi),
                    "width_mhz": float(f_hi - f_lo) * 1e3,
                    "fractional_pct": float(f_hi - f_lo) / float(r["refined_f"]) * 100.0,
                    "n_bins": int(n_bins),
                    "reaches_grid_edge": bool(f_lo <= f_ghz[0] or f_hi >= f_ghz[-1]),
                    "estimator": "spectral_features.py::band_at_level, linear in dB",
                }
            res["zin_at_refined_ohm"] = {
                "re": float(np.interp(r["refined_f"], f_ghz, zin.real)),
                "im": float(np.interp(r["refined_f"], f_ghz, zin.imag)),
                "how": "linear interpolation of Re and Im between the bracketing bins",
            }
            res["zin_at_bin_ohm"] = {"re": float(zin.real[r["index"]]),
                                     "im": float(zin.imag[r["index"]])}
            in_band = np.where((f_ghz >= lo) & (f_ghz <= hi))[0]
            k = int(in_band[int(np.argmax(zin.real[in_band]))])
            res["re_zin_peak"] = {"f_ghz": float(f_ghz[k]), "ohm": float(zin.real[k])}
            out["resonance"] = res
        except Exception as exc:
            out["resonance"] = {"error": repr(exc)}
        if with_tutorial_pick:
            out["tutorial_pick"] = _tutorial_pick(np.asarray(freqs_hz, dtype=float), s11)
        return out
    return features


# ---------------------------------------------------------------------------
# openEMS's own log lines, read back. The formats: openems.cpp at
# A_TUTORIAL_OPENEMS_COMMIT, and the Sheen branch's capture of a real container
# log (crossval/sheen-lpf-test, run 369367263401), which showed the timestep
# line printing "0.00 s" -- hence dt from the Nyquist line.
# ---------------------------------------------------------------------------
_PROGRESS_LINE_RE = re.compile(
    r"Timestep:\s*(\d+).*?Energy:.*?\(\s*(-?\s*(?:[0-9]*\.?[0-9]+|inf|nan))\s*dB\s*\)",
    re.IGNORECASE)
_NYQUIST_RE = re.compile(
    r"Nyquist\s+rate\s*:\s*([0-9]+)\s*timesteps?\s*@\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*Hz",
    re.IGNORECASE)
_SPEED_RE = re.compile(r"^Speed:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*MCells/s",
                       re.IGNORECASE | re.MULTILINE)
_ITERATIONS_RE = re.compile(r"Time for\s+(\d+)\s+iterations with\s+(\d+)(?:\.\d*)?\s+cells",
                            re.IGNORECASE)
_DT_LINE_RE = re.compile(r"FDTD timestep is:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*s\b",
                         re.IGNORECASE)


def _solver_run_summary(log_text: str, wall_s: float) -> dict:
    """dt, steps, the last box energy openEMS printed, and its speed."""
    out: dict = {"wall_time_s": round(float(wall_s), 1)}
    d = _DT_LINE_RE.search(log_text)
    if d and float(d.group(1)) > 0.0:
        out.update({"dt_s": float(d.group(1)), "dt_s_uncertainty_rel": 0.0,
                    "dt_s_source": "openEMS's own 'FDTD timestep is:' line"})
    m = _NYQUIST_RE.search(log_text)
    if m:
        n, f = int(m.group(1)), float(m.group(2))
        out.update({"nyquist_steps": n, "nyquist_f_hz": f})
        if "dt_s" not in out:
            # The pinned image prints the timestep line as "0.00 s" (the Sheen
            # branch's capture, run 369367263401); N = floor(1 / (2 f dt)).
            out.update({"dt_s": 1.0 / (2.0 * f * n), "dt_s_uncertainty_rel": 1.0 / n,
                        "dt_s_source": "dt = 1 / (2 f N) from openEMS's Nyquist line"})
    trace = []
    for line in log_text.splitlines():
        p = _PROGRESS_LINE_RE.search(line)
        if p:
            try:
                db = float(p.group(2).replace(" ", ""))
            except ValueError:
                continue
            if np.isfinite(db):
                trace.append([int(p.group(1)), db])
    out["energy_db_trace"] = trace
    out["final_energy_db"] = trace[-1][1] if trace else None
    it = _ITERATIONS_RE.search(log_text)
    if it:
        out["timesteps"] = int(it.group(1))
        out["cells_reported_by_solver"] = int(it.group(2))
    else:
        out["timesteps"] = _gate._timesteps_executed(log_text)
    sp = _SPEED_RE.search(log_text)
    out["speed_mcells_per_s"] = float(sp.group(1)) if sp else None
    if out.get("dt_s") and out.get("timesteps"):
        out["record_length_s"] = out["dt_s"] * out["timesteps"]
    return out


# ---------------------------------------------------------------------------
# What the solver built, read back.
# ---------------------------------------------------------------------------
def _realized_geometry(lines_mm, *, patch_x_mm, patch_y_mm, feed_xy_mm, ground_x_mm,
                       ground_y_mm, sub_h_mm, thirds_res_mm, boundary_cells) -> dict:
    if lines_mm is None:
        return {"error": "CSXCAD did not return its grid lines"}
    x, y, z = (np.asarray(lines_mm[a], dtype=float) for a in ("x", "y", "z"))
    out: dict = {"smallest_cell": {a: _smallest_spacing(v) for a, v in (("x", x), ("y", y), ("z", z))}}
    edges = {}
    for name, lines, v in (("patch_x_lo", x, patch_x_mm[0]), ("patch_x_hi", x, patch_x_mm[1]),
                           ("patch_y_lo", y, patch_y_mm[0]), ("patch_y_hi", y, patch_y_mm[1])):
        near, snap = _gate.nearest_line(lines, v)
        edges[name] = {"declared_mm": v, "nearest_line_mm": near, "distance_mm": snap}
    out["patch_edges"] = edges
    mer = np.array([-1.0, 2.0]) / 3.0 * thirds_res_mm
    out["thirds_rule_lines"] = {
        "offsets_mm": [float(v) for v in mer],
        "x_declared_mm": [float(v) for v in _thirds(patch_x_mm[0], patch_x_mm[1], thirds_res_mm)],
        "y_declared_mm": [float(v) for v in _thirds(patch_y_mm[0], patch_y_mm[1], thirds_res_mm)],
    }
    fx, fsx = _gate.nearest_line(x, feed_xy_mm[0])
    fy, fsy = _gate.nearest_line(y, feed_xy_mm[1])
    out["feed"] = {"declared_xy_mm": list(feed_xy_mm), "nearest_x_mm": fx, "dx_mm": fsx,
                   "nearest_y_mm": fy, "dy_mm": fsy}
    g0, s0 = _gate.nearest_line(z, 0.0)
    gh, sh = _gate.nearest_line(z, sub_h_mm)
    out["stack"] = {"ground_z_nearest_mm": g0, "ground_z_distance_mm": s0,
                    "patch_z_nearest_mm": gh, "patch_z_distance_mm": sh,
                    "substrate_z_cells": int(np.sum((z >= -1e-9) & (z <= sub_h_mm + 1e-9)) - 1)}
    if boundary_cells:
        faces = {}
        for a, v, span in (("x", x, ground_x_mm), ("y", y, ground_y_mm), ("z", z, (0.0, sub_h_mm))):
            if v.size > 2 * boundary_cells + 1:
                lo_f, hi_f = float(v[boundary_cells]), float(v[-1 - boundary_cells])
                faces[a] = {"inner_face_lo_mm": lo_f, "inner_face_hi_mm": hi_f,
                            "depth_lo_mm": lo_f - float(v[0]), "depth_hi_mm": float(v[-1]) - hi_f,
                            "clearance_to_board_lo_mm": span[0] - lo_f,
                            "clearance_to_board_hi_mm": hi_f - span[1]}
        out["pml"] = {"cells": boundary_cells, "faces": faces}
    return out


def _nf2ff_dump_bytes(sim_dir: str) -> int:
    total = 0
    for name in os.listdir(sim_dir) if os.path.isdir(sim_dir) else []:
        if name.startswith("nf2ff"):
            total += os.path.getsize(os.path.join(sim_dir, name))
    return total


def _farfield(nf2ff, sim_dir: str, f_hz, theta_spec, phi, center, *, drop_dumps: bool) -> dict:
    """One CalcNF2FF call, reported. A failure is recorded, never raised."""
    theta = np.arange(*theta_spec)
    out = {"f_hz": float(f_hz), "theta_deg": list(theta_spec), "phi_deg": list(phi),
           "center_mm": list(center),
           "box_mm": {"start": [float(v) for v in np.asarray(nf2ff.start)],
                      "stop": [float(v) for v in np.asarray(nf2ff.stop)]}}
    t0 = time.time()
    try:
        res = nf2ff.CalcNF2FF(sim_dir, f_hz, theta, list(phi), center=list(center))
        dmax_lin = float(np.asarray(res.Dmax).ravel()[0])
        out["dmax_linear"] = dmax_lin
        out["dmax_dbi"] = float(10.0 * np.log10(dmax_lin))
    except Exception as exc:
        out["error"] = repr(exc)
    out["calc_wall_time_s"] = round(time.time() - t0, 1)
    out["dump_bytes"] = _nf2ff_dump_bytes(sim_dir)
    if drop_dumps:
        for name in os.listdir(sim_dir):
            if name.startswith("nf2ff") and name.endswith(".h5"):
                try:
                    os.remove(os.path.join(sim_dir, name))
                except OSError:
                    pass
        out["dumps_deleted_after"] = True
    return out


# ---------------------------------------------------------------------------
# The run: one lumped port, the shared module's gates around it. The sequence is
# the shared ``run_stage``'s; that function reads two ports and S21, this board
# has one port.
# ---------------------------------------------------------------------------
def _run_one_port_stage(*, label: str, sim_root: str, threads: int, build, freqs_hz,
                        witness_band_hz, real_nrts, real_end_criteria, realized_fn,
                        meta_extra: dict, features_fn, farfield_fn=None,
                        smoke_nrts: int = 200, smoke_end_criteria: float = 0.0):
    ContinuousStructure, openEMS, _unused_msl = _gate._import_openems()
    sim_dir = os.path.join(sim_root, label)
    smoke_dir = os.path.join(sim_root, label + "_smoke")
    record: dict = {}
    meta: dict = {"stage": label}
    meta.update(meta_extra)

    try:
        smoke, _p, _n = build(ContinuousStructure, openEMS, nrts=smoke_nrts,
                              end_criteria=smoke_end_criteria)
        smoke_log = _gate._run_openems_capturing_stdout(smoke, smoke_dir, threads=threads)
        _gate._scan_stdout_for_bad_patterns(smoke_log, label + "_smoke")

        fdtd, port, nf2ff = build(ContinuousStructure, openEMS, nrts=real_nrts,
                                  end_criteria=real_end_criteria)
        lines = _gate._mesh_lines(fdtd)
        lines_um = _gate.lines_in_um(lines, 1e-3)
        meta["mesh_realized"] = _gate._mesh_realized(
            lines_um, substrate_thickness_um=meta_extra["substrate_thickness_um"])
        meta["geometry_realized"] = realized_fn(lines)

        t0 = time.time()
        real_log = _gate._run_openems_capturing_stdout(fdtd, sim_dir, threads=threads)
        meta["solver_run"] = _solver_run_summary(real_log, time.time() - t0)
        meta["stdout_log_path"] = os.path.join(sim_dir, "_openems_stdout.log")
        meta["smoke_stdout_log_path"] = os.path.join(smoke_dir, "_openems_stdout.log")
        _gate._scan_stdout_for_bad_patterns(real_log, label, check_truncation=True)

        freqs = np.asarray(freqs_hz, dtype=float)
        port.CalcPort(sim_dir, freqs)
        inc_peak, n_samples = _gate._check_excitation_and_trace(port, sim_dir, label)
        meta["excitation_energy_peak"] = inc_peak
        meta["port_trace_samples"] = n_samples
        if _gate._log_indicates_truncation(real_log):
            raise RuntimeError(
                f"[{label}] SANITY GATE 'end criteria reached' FAILED: openEMS's own "
                f"'reached before the end-criteria of' warning is in this real pass's "
                f"captured log -- the run hit its NrTS cap before the field decayed, so "
                f"the spectrum is truncated and no record is written.")

        uf_inc = np.asarray(port.uf_inc, dtype=np.complex128)
        s11 = np.asarray(port.uf_ref, dtype=np.complex128) / uf_inc
        zin = np.asarray(port.uf_tot, dtype=np.complex128) / np.asarray(port.if_tot, dtype=np.complex128)
        power = np.abs(s11) ** 2
        band = (freqs >= witness_band_hz[0]) & (freqs <= witness_band_hz[1])
        record.update({
            "freqs_ghz": (freqs / 1e9).tolist(),
            "s11_re": s11.real.tolist(),
            "s11_im": s11.imag.tolist(),
            "s11_mag": np.abs(s11).tolist(),
            "s11_deg": np.degrees(np.angle(s11)).tolist(),
            "zin_re_ohm": zin.real.tolist(),
            "zin_im_ohm": zin.imag.tolist(),
            "s11_power": power.tolist(),
            "max_s11_power_band": float(np.max(power[band])),
            "min_s11_power_band": float(np.min(power[band])),
            "witness_band_ghz": [witness_band_hz[0] / 1e9, witness_band_hz[1] / 1e9],
            "passivity_tol": 1.0 + PASSIVITY_TOL,
            "z_ref_ohm": float(np.real(np.asarray(port.Z_ref))) if np.ndim(port.Z_ref) == 0 else None,
        })
        print(f"  |S11|^2 over {witness_band_hz[0]/1e9:.2f}-{witness_band_hz[1]/1e9:.2f} GHz: "
              f"max {record['max_s11_power_band']:.4f}, min {record['min_s11_power_band']:.4f} "
              f"(tol {1.0 + PASSIVITY_TOL:.2f})", flush=True)
        meta["openems"] = _gate._openems_version(real_log)
        record.update(features_fn(freqs, s11, zin))
        _gate._non_physical_guard(np.abs(s11), label + "_s11")
        # The shared witness, untouched, fed zeros for S21: it bounds |S11|^2.
        _gate._passivity_witness(s11, np.zeros_like(s11), label,
                                 tol=PASSIVITY_TOL, idx=band)
    except RuntimeError as exc:
        raise _gate.StageFailure(str(exc), stage=label, partial=record, meta=meta) from exc

    if farfield_fn is not None and nf2ff is not None:
        record["farfield"] = farfield_fn(nf2ff, sim_dir, record)
    meta["end_criteria_reached"] = True
    return record, meta


def _run_stage_a(*, sim_root: str, threads: int, sf) -> tuple:
    def build(ContinuousStructure, openEMS, *, nrts, end_criteria):
        return _build_stage_a_tutorial(ContinuousStructure, openEMS, nrts=nrts,
                                       end_criteria=end_criteria)

    def realized(lines):
        mesh_res = C0 / (A_F0_HZ + A_FC_HZ) / 1e-3 / 20
        return _realized_geometry(lines, patch_x_mm=(-16.0, 16.0), patch_y_mm=(-20.0, 20.0),
                                  feed_xy_mm=(-6.0, 0.0), ground_x_mm=(-30.0, 30.0),
                                  ground_y_mm=(-30.0, 30.0), sub_h_mm=1.524,
                                  thirds_res_mm=mesh_res / 2, boundary_cells=0)

    def farfield(nf2ff, sim_dir, record):
        pick = record.get("tutorial_pick") or {}
        if pick.get("f_hz") is None:
            return {"skipped": "the tutorial's own pick found no resonance, so the "
                               "tutorial computes no far field"}
        return _farfield(nf2ff, sim_dir, pick["f_hz"], A_FARFIELD_THETA, A_FARFIELD_PHI,
                         A_FARFIELD_CENTER, drop_dumps=True)

    return _run_one_port_stage(
        label="stage_a", sim_root=sim_root, threads=threads, build=build,
        freqs_hz=stage_a_freqs_hz(), witness_band_hz=STAGE_A_BAND_HZ,
        real_nrts=A_REAL_NRTS, real_end_criteria=A_REAL_END_CRITERIA,
        realized_fn=realized,
        meta_extra={
            "model": "openEMS python/Tutorials/Simple_Patch_Antenna.py, verbatim -- the "
                     "reproduce gate, NOT this case's board",
            "substrate_thickness_um": 1524.0,
            "nrts_declared": A_REAL_NRTS,
            "end_criteria_declared": A_REAL_END_CRITERIA,
            "boundary": ["MUR"] * 6,
            "calcport_grid": f"linspace({STAGE_A_BAND_HZ[0]:g}, {STAGE_A_BAND_HZ[1]:g}, {A_N_FREQS})",
            "plan_estimate": _plan_a(),
        },
        features_fn=_one_port_features(sf, STAGE_A_BAND_HZ, with_tutorial_pick=True),
        farfield_fn=farfield)


def _run_stage_b(*, label: str, sim_root: str, threads: int, resolution_factor: float,
                 sf, do_gain: bool) -> tuple:
    def build(ContinuousStructure, openEMS, *, nrts, end_criteria):
        return _build_patch_board_at_rung(ContinuousStructure, openEMS, nrts=nrts,
                                          end_criteria=end_criteria,
                                          resolution_factor=resolution_factor,
                                          do_gain=do_gain)

    def realized(lines):
        return _realized_geometry(
            lines, patch_x_mm=(-L_PATCH * 5e2, L_PATCH * 5e2),
            patch_y_mm=(-W_PATCH * 5e2, W_PATCH * 5e2),
            feed_xy_mm=(FEED_OFFSET_X * 1e3, 0.0), ground_x_mm=(-GP_X * 5e2, GP_X * 5e2),
            ground_y_mm=(-GP_Y * 5e2, GP_Y * 5e2), sub_h_mm=H_SUB * 1e3,
            thirds_res_mm=1.2 * resolution_factor / 2, boundary_cells=B_PML_CELLS)

    def farfield(nf2ff, sim_dir, record):
        res = record.get("resonance") or {}
        if "refined_f_ghz" not in res:
            return {"skipped": "no resonance estimate"}
        if not res["depth_db"] < B_FARFIELD_MAX_DEPTH_DB:
            return {"skipped": f"the dip is {res['depth_db']:.2f} dB, not below "
                               f"{B_FARFIELD_MAX_DEPTH_DB:g} dB (the retired script's condition)"}
        return _farfield(nf2ff, sim_dir, res["refined_f_ghz"] * 1e9, B_FARFIELD_THETA,
                         B_FARFIELD_PHI, B_FARFIELD_CENTER, drop_dumps=True)

    return _run_one_port_stage(
        label=label, sim_root=sim_root, threads=threads, build=build,
        freqs_hz=stage_b_freqs_hz(), witness_band_hz=B_WITNESS_BAND_HZ,
        real_nrts=B_REAL_NRTS, real_end_criteria=B_REAL_END_CRITERIA,
        realized_fn=realized,
        meta_extra={
            "model": "the RT/Duroid 5880 probe-fed patch",
            "resolution_factor": float(resolution_factor),
            "substrate_thickness_um": H_SUB * 1e6,
            "substrate_z_cells_declared": substrate_z_cells(resolution_factor),
            "nrts_declared": "openEMS library default (~1e9) -- the retired cap of "
                             f"{RETIRED_NRTS_CAP} is not carried",
            "end_criteria_declared": "openEMS library default (1e-5) -- the retired "
                                     f"{RETIRED_END_CRITERIA_CAP} is not carried",
            "boundary": B_BOUNDARY,
            "nf2ff_box_created": bool(do_gain),
            "calcport_grid": f"linspace({F_LO:g}, {F_HI:g}, {B_N_FREQS})",
            "plan_estimate": _plan_b(label, resolution_factor),
        },
        features_fn=_one_port_features(sf, B_RESONANCE_BAND_HZ),
        farfield_fn=farfield if do_gain else None)


def _build_artifact(records: dict, stage_meta: dict, stage_a_gate: dict, stages: list, *,
                    image_tutorial: dict | None = None, do_gain: bool = True,
                    failed_gate: str | None = None) -> dict:
    first = next((m for m in stage_meta.values() if m.get("openems")), None)
    meta = {
        "tool": "openEMS",
        "openems": first["openems"] if first else {"version": None,
                                                   "source": "no stage reached the solver"},
        "rfx_openems_commit": os.environ.get("RFX_OPENEMS_COMMIT"),
        "rfx_openems_image": os.environ.get("RFX_OPENEMS_IMAGE"),
        "rfx_commit": os.environ.get("RFX_COMMIT"),
        "structure": (
            "rectangular microstrip patch 40.0 mm (x, resonant) x 50.0 mm (y) on RT/Duroid "
            "5880 (eps_r 2.2, tan delta 1e-3, h 3.175 mm) over a 56 x 66 mm ground, 50 ohm "
            "lumped probe from ground to patch at x = -9 mm, y = 0; copied from "
            f"{RETIRED_REL_PATH}::{RETIRED_FUNCTION}"),
        "tl_model_tm010_hz": F_TM010_BOARD_HZ,
        "stage_a_is": ("openEMS python/Tutorials/Simple_Patch_Antenna.py, verbatim -- the "
                       "reproduce gate, a different patch on a different substrate; it is "
                       "not a measurement of this board"),
        "stage_a_tutorial": dict(A_TUTORIAL, project_commit=A_TUTORIAL_PROJECT_COMMIT,
                                 openems_commit=A_TUTORIAL_OPENEMS_COMMIT,
                                 sha256=A_TUTORIAL_SHA256, image_check=A_TUTORIAL_IMAGE_CHECK),
        "stage_a_tutorial_in_this_container": image_tutorial,
        "stage_a_documented": STAGE_A_DOCUMENTED,
        "stage_a_recorded_reproduction": STAGE_A_RECORDED_REPRODUCTION,
        "stage_a_gate_rule": (
            f"refined |S11| minimum over the tutorial's {STAGE_A_BAND_HZ[0]/1e9:g}-"
            f"{STAGE_A_BAND_HZ[1]/1e9:g} GHz grid within +-{STAGE_A_WINDOW_REL*100:g} % of the "
            f"documented {STAGE_A_DOCUMENTED['f_dip_hz']/1e9:.3f} GHz AND at most "
            f"{STAGE_A_MAX_DEPTH_DB:g} dB"),
        "stage_a_gate": stage_a_gate,
        "reproduce_gate_ran": "stage_a" in stages,
        "stages_requested": stages,
        "stages": stage_meta,
        "delta_list": DELTA_LIST,
        "copy_proof": {
            "stage_b_frozen_from": f"{RETIRED_REL_PATH} lines "
                                   f"{RETIRED_FROZEN_LINE_RANGE[0]}-{RETIRED_FROZEN_LINE_RANGE[1]} "
                                   f"at {RETIRED_FROZEN_AT_COMMIT}",
            "stage_a_frozen_from": f"{A_TUTORIAL['path']} lines 28-104 at "
                                   f"{A_TUTORIAL_OPENEMS_COMMIT}",
            "checked_by": "--self-check, which the job runs before the solver",
        },
        "boundary": B_BOUNDARY,
        "excitation": "SetGaussExcite(2.4e9, 1.2e9) -- the retired builder's f0, fc",
        "stop_criteria": ("Stage B real passes: openEMS defaults (NrTS ~1e9, EndCriteria "
                          "1e-5). Stage A: the tutorial's NrTS 30000, EndCriteria 1e-4. "
                          "Smoke passes: 200 steps, EndCriteria 0."),
        "calcport_grid": f"Stage B linspace({F_LO:g}, {F_HI:g}, {B_N_FREQS}); Stage A the "
                         f"tutorial's linspace({STAGE_A_BAND_HZ[0]:g}, "
                         f"{STAGE_A_BAND_HZ[1]:g}, {A_N_FREQS})",
        "resonance_band_ghz": [B_RESONANCE_BAND_HZ[0] / 1e9, B_RESONANCE_BAND_HZ[1] / 1e9],
        "passivity_witness": (
            "max |S11|^2 <= 1.05 over each stage's whole grid (the shared "
            "_passivity_witness with S21 = 0). A radiating antenna's |S11|^2 sits below one "
            "by the accepted power; the per-bin values and the band minimum are recorded."),
        "farfield": ("Stage B: CalcNF2FF at the refined resonance when the dip is below "
                     "-6 dB, theta -180..178 step 2, phi 0/90 -- reported only"
                     if do_gain else "Stage B: not computed (--no-nf2ff)"),
        "what_is_not_here": ("no comparison with rfx and no verdict on the board. This "
                             "script writes one solver's result with its provenance."),
        "produced_by": "tests/crossval/rt5880_patch/reference/make_openems_reference.py",
        "ci_runs_this": False,
    }
    artifact: dict = {"meta": meta}
    for name in STAGE_NAMES:
        artifact[name] = records.get(name)
    artifact["run_id"] = None
    artifact["run_id_note"] = (
        "null by design. VESSL does not export VESSL_RUN_ID into the pod, so the job "
        "cannot write its own id; the submitter fills this field.")
    if failed_gate is not None:
        artifact["failed_gate"] = failed_gate
        artifact["meta"]["record_is_partial"] = True
    return artifact


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------
def _fmt_s(seconds: float) -> str:
    return f"{seconds:,.0f} s ({seconds / 60:.1f} min)"


def _print_plan_a(p: dict) -> None:
    m, s = p["mesh_lines_estimate"], p["smallest_explicit_spacing"]
    print(f"  stage_a  (the tutorial's own mesh, lambda/20 at 3 GHz = {p['mesh_res_mm']:.3f} mm)")
    print(f"    mesh lines (estimate)   x {m['x']}  y {m['y']}  z {m['z']}  -> cells ~ "
          f"{p['cells_estimate']:,}")
    print(f"    smallest explicit gap   x {s['x']['mm']:.4f}  y {s['y']['mm']:.4f}  "
          f"z {s['z']['mm']:.4f} mm  -> CFL dt ~ {p['cfl_dt_s']*1e12:.3f} ps")
    print(f"    the tutorial's cap      {A_REAL_NRTS} steps = {p['cap_covers_s']*1e9:.1f} ns "
          f"(pulse {p['pulse_s']*1e9:.2f} ns)")
    print(f"    cost, at the cap        {_fmt_s(p['planned_cost_s'])}")


def _print_plan_b(p: dict) -> None:
    m, s = p["mesh_lines_estimate"], p["smallest_explicit_spacing"]
    print(f"  {p['label']}  (resolution factor {p['resolution_factor']:.5g})")
    print(f"    mesh_res / patch_res    {p['mesh_res_mm']:.4f} / {p['patch_res_mm']:.4f} mm; "
          f"thirds-rule offsets {p['thirds_rule_offsets_mm'][0]:+.4f} / "
          f"{p['thirds_rule_offsets_mm'][1]:+.4f} mm")
    print(f"    substrate z cells       {p['substrate_z_cells']} of "
          f"{p['substrate_z_step_um']:.2f} um across the 3.175 mm board; the planned "
          f"lines hold {p['substrate_z_cells_in_planned_lines']}")
    print(f"    mesh lines (estimate)   x {m['x']}  y {m['y']}  z {m['z']}  -> cells ~ "
          f"{p['cells_estimate']:,}")
    for ax in ("x", "y", "z"):
        a, b = s[ax]["between_mm"]
        print(f"    smallest {ax} gap         {s[ax]['mm']*1e3:8.2f} um between the explicit "
              f"lines {a:+.4f} and {b:+.4f} mm")
    print(f"    CFL dt (estimate)       {p['cfl_dt_s']*1e12:.4f} ps; the retired 30000-step cap "
          f"would cover {p['retired_cap_covers_s']*1e9:.2f} ns")
    f = p["pml_inner_faces_mm_estimate"]
    print(f"    PML_8 inner faces (est) x {f['x'][0]:+.2f}/{f['x'][1]:+.2f}  "
          f"y {f['y'][0]:+.2f}/{f['y'][1]:+.2f}  z {f['z'][0]:+.2f}/{f['z'][1]:+.2f} mm "
          f"(ground edges x +-{GP_X*5e2:.0f}, y +-{GP_Y*5e2:.0f}; patch at z {H_SUB*1e3:.3f})")
    print(f"    planned record          {p['planned_record_s']*1e9:.1f} ns -> "
          f"{p['planned_steps']:,} steps -> {_fmt_s(p['planned_cost_s'])}")
    print(f"    NF2FF dumps (estimate)  {p['nf2ff_dump_bytes_estimate']/1e9:.2f} GB")


def _dry_run(stage: str, do_gain: bool) -> int:
    print("=" * 78)
    print("RT/Duroid 5880 patch antenna -- openEMS reference maker, DRY RUN (no solver)")
    print("=" * 78)
    doc = STAGE_A_DOCUMENTED
    print("STAGE A -- the reproduce gate: openEMS's Simple Patch Antenna tutorial, verbatim")
    print(f"  tutorial        {A_TUTORIAL['repo']} {A_TUTORIAL['path']}")
    print(f"  at              openEMS {A_TUTORIAL_OPENEMS_COMMIT} (submodule of openEMS-Project "
          f"{A_TUTORIAL_PROJECT_COMMIT})")
    print(f"  sha256          {A_TUTORIAL_SHA256}")
    print(f"  attribution     {A_TUTORIAL['attribution']}")
    print(f"  board           32 x 40 mm patch, eps_r 3.38, h 1.524 mm, 60 x 60 mm ground, "
          f"feed x = -6 mm, MUR x6, SetGaussExcite(2e9, 1e9), NrTS {A_REAL_NRTS}, "
          f"EndCriteria {A_REAL_END_CRITERIA:g}")
    print(f"  documented      |S11| dip {doc['f_dip_hz']/1e9:.3f} GHz "
          f"(+-{doc['f_dip_read_half_width_hz']/1e9:.3f}), {doc['depth_db']:.1f} dB; "
          f"Re(Zin) peak {doc['re_zin_peak_ohm']:.1f} ohm at {doc['re_zin_peak_f_hz']/1e9:.3f} GHz "
          f"-- read off {doc['s11_figure']} (sha256 {doc['s11_figure_sha256'][:16]}...)")
    lo, hi = (doc["f_dip_hz"] * (1 - STAGE_A_WINDOW_REL), doc["f_dip_hz"] * (1 + STAGE_A_WINDOW_REL))
    print(f"  gate            refined minimum in {lo/1e9:.4f} .. {hi/1e9:.4f} GHz "
          f"(+-{STAGE_A_WINDOW_REL*100:g} %) AND <= {STAGE_A_MAX_DEPTH_DB:g} dB")
    print(f"  TL model        {F_TM010_TUTORIAL_HZ/1e9:.4f} GHz for the tutorial board "
          f"(documented / model = {doc['f_dip_hz']/F_TM010_TUTORIAL_HZ:.4f}) -- recorded, "
          f"not gated")
    rr = STAGE_A_RECORDED_REPRODUCTION
    print(f"  recorded run    {rr['f_s11_dip_hz']/1e9:.4f} GHz, {rr['s11_dip_db']:.1f} dB "
          f"(VESSL {rr['vessl_run']}, {rr['date']}, openEMS 7b051bb) -- an AUDIT TRAIL for the "
          f"tutorial, never this run's gate")
    print()
    print("STAGE B -- the RT5880 board")
    print(f"  board           {L_PATCH*1e3:.1f} x {W_PATCH*1e3:.1f} mm patch, eps_r {EPS_R}, "
          f"tan d {TAN_DELTA:g}, h {H_SUB*1e3:.3f} mm, ground {GP_X*1e3:.0f} x {GP_Y*1e3:.0f} mm, "
          f"feed x = {FEED_OFFSET_X*1e3:+.1f} mm")
    print(f"  TL model TM010  {F_TM010_BOARD_HZ/1e9:.4f} GHz; resonance window "
          f"{B_RESONANCE_BAND_HZ[0]/1e9:.4f}-{B_RESONANCE_BAND_HZ[1]/1e9:.4f} GHz "
          f"(the retired 0.80-1.20 x)")
    print(f"  grid            linspace({F_LO/1e9:g}, {F_HI/1e9:g} GHz, {B_N_FREQS}) = "
          f"{(F_HI-F_LO)/(B_N_FREQS-1)/1e6:.2f} MHz bins")
    print(f"  boundary        {B_BOUNDARY}")
    print("  stop            openEMS defaults (NrTS ~1e9, EndCriteria 1e-5) on the real "
          "pass; 200 / 0 on the smoke pass")
    print(f"  far field       {'CalcNF2FF at the refined resonance, reported only' if do_gain else 'OFF (--no-nf2ff)'}")
    print("  witness         max |S11|^2 <= 1.05 over the whole grid; the deficit is recorded")
    print()
    print("DELTA LIST -- how Stage B differs from the retired builder it is copied from:")
    for i, line in enumerate(DELTA_LIST, start=1):
        print(f"  [{i}] {line}")
    print()
    order = _gate.stages_for(stage)
    plans = _stage_plans(order)
    print("STAGE PLAN AND COST -- estimates, see the notes at the end:")
    total = 0.0
    for name in order:
        p = plans[name]
        (_print_plan_a if name == "stage_a" else _print_plan_b)(p)
        total += p["planned_cost_s"]
        print()
    rec = _planned_record_s()
    print(f"  planned record, Stage B: pulse {rec['pulse_s']*1e9:.2f} ns + ring-down to "
          f"{10*math.log10(PLAN_END_CRITERIA):.0f} dB {rec['ring_down_s']*1e9:.2f} ns = "
          f"{rec['sum_s']*1e9:.2f} ns, x{PLAN_MARGIN:g} = {rec['planned_s']*1e9:.1f} ns")
    print(f"    Q = {PLAN_Q} from {PLAN_Q_SOURCE}")
    print(f"  speed: {PLAN_SPEED_SOURCE}")
    print(f"  TOTAL (the requested stages, smoke passes excluded): {_fmt_s(total)} against the "
          f"job's {PLAN_JOB_BUDGET_S:,.0f} s -- {total/PLAN_JOB_BUDGET_S*100:.0f} % of it")
    rr = STAGE_A_RECORDED_REPRODUCTION
    dt_ratio = rr["dt_s"] / _plan_a()["cfl_dt_s"]
    worst = sum(plans[n]["planned_cost_s"] for n in order if n != "stage_a") / dt_ratio \
        * (PLAN_SPEED_MC_PER_S / rr["speed_mcells_per_s"])
    print(f"  calibration: on the tutorial (run {rr['vessl_run']}) openEMS printed dt "
          f"{rr['dt_s']*1e12:.5f} ps, {dt_ratio:.3f} x this plan's CFL estimate, and ran "
          f"{rr['speed_mcells_per_s']:.1f} MC/s on {rr['solver_cells']:,} cells. Stage B at that "
          f"dt ratio and that speed: {_fmt_s(worst)} -- {worst/PLAN_JOB_BUDGET_S*100:.0f} % of "
          f"the job")
    if "stage_b_coarse" in plans:
        c = plans["stage_b_coarse"]["cells_estimate"]
        print(f"  calibration: rung 1.0's estimate {c:,} cells; the retired record realized "
              f"{RETIRED_RECORD['n_cells']:,} cells on this mesh in {RETIRED_RECORD['runtime_s']:.1f} s "
              f"(MUR, 30000-step cap, far field on; no openEMS version or step count recorded)")
    print()
    print("WHAT THIS DRY RUN CANNOT TELL YOU: the explicit lines are exact for the builder "
          "as written; every line CSXCAD's SmoothMeshLines adds between them is an "
          "estimate, and the timestep is the vacuum CFL of the smallest explicit gaps, not "
          "a bound -- openEMS computes its own (0.87 x this estimate on the tutorial). The "
          "speed, the ring-down Q and the NF2FF dump size are borrowed or assumed as stated "
          "above. The realized mesh, dt, step count, final energy and speed are written into "
          "the record by the run itself.")
    return 0


# ---------------------------------------------------------------------------
# --self-check
# ---------------------------------------------------------------------------
def _self_check() -> int:
    failures: list = []
    notes: list = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        print(f"  [{'ok ' if ok else 'FAIL'}] {what}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(what)

    print("=" * 78)
    print("RT/Duroid 5880 patch antenna -- openEMS reference maker, SELF-CHECK (pure numpy)")
    print("=" * 78)

    retired = _retired_text()
    retired_lines = set(retired.split("\n")) if retired is not None else set()
    print("the board's constants are the retired script's:")
    mine = {"C0": C0, "EPS_R": EPS_R, "TAN_DELTA": TAN_DELTA, "H_SUB": H_SUB,
            "L_PATCH": L_PATCH, "W_PATCH": W_PATCH, "GP_X": GP_X, "GP_Y": GP_Y,
            "FEED_OFFSET_X": FEED_OFFSET_X, "N_SUB": N_SUB, "F_LO, F_HI": (F_LO, F_HI)}
    for name, line in RETIRED_CONSTANT_LINES.items():
        value = ast.literal_eval(ast.parse(line).body[0].value)
        check(value == mine[name], f"{name} equals the retired line {line.split('#')[0].strip()!r}",
              f"{mine[name]!r}")
        if retired is not None:
            check(line in retired_lines, f"that line is a whole line of {RETIRED_REL_PATH}")
    if retired is None:
        print(f"  [--] {RETIRED_REL_PATH} is gone; the frozen lines are the record")
    res_1 = C0 / (B_F0_HZ + B_FC_HZ) / B_UNIT_M / 30
    check(abs(res_1 - RETIRED_RECORD["mesh_res_mm"]) < 1e-12,
          "rung 1.0's mesh_res is the retired record's own mesh_res_mm",
          f"{res_1!r} vs {RETIRED_RECORD['mesh_res_mm']!r}")
    check(abs(F_TM010_BOARD_HZ - RETIRED_RECORD["f_analytic_hz"]) < 1e-3,
          "the TL model reproduces the retired record's f_analytic_hz",
          f"{F_TM010_BOARD_HZ:.4f} Hz")
    rec_path = _repo_root() / RETIRED_RECORD_REL_PATH
    if rec_path.is_file():
        import json
        on_disk = json.loads(rec_path.read_text())
        check(all(on_disk.get(k) == v for k, v in RETIRED_RECORD.items()),
              f"the frozen RETIRED_RECORD numbers are {RETIRED_RECORD_REL_PATH}'s")
    else:
        print(f"  [--] {RETIRED_RECORD_REL_PATH} is gone; the frozen numbers are the record")

    print("Stage B's grid, stop criteria and rungs:")
    f = stage_b_freqs_hz()
    check(f[0] == F_LO and f[-1] == F_HI and f.size == 901
          and abs((f[1] - f[0]) - 2.0e6) < 1e-3,
          "linspace(1.6, 3.4 GHz, 901): 2.0 MHz bins", f"{(f[1]-f[0])/1e6:.4f} MHz")
    check(f[0] <= 1.8e9 and f[-1] >= 3.2e9, "the grid covers 1.8-3.2 GHz")
    check(B_F0_HZ - B_FC_HZ <= F_LO and F_HI <= B_F0_HZ + B_FC_HZ,
          "the grid sits inside the excitation's 20 dB corners, 1.2-3.6 GHz")
    check(B_REAL_NRTS is None and B_REAL_END_CRITERIA is None,
          "the real pass passes no NrTS/EndCriteria, so openEMS's own defaults apply")
    check(B_BOUNDARY == ["PML_8"] * 6, "PML_8 on all six faces")
    check(B_COARSE_RESOLUTION_FACTOR == 1.0 and abs(B_MID_RESOLUTION_FACTOR - 2 ** -0.5) < 1e-15
          and B_FINE_RESOLUTION_FACTOR == 0.5, "the rungs are 1, 1/sqrt(2), 1/2")
    check([substrate_z_cells(v) for v in STAGE_B_FACTORS.values()] == [4, 6, 8],
          "the substrate carries 4, 6, 8 z cells on the three rungs")
    check(abs(B_RESONANCE_BAND_HZ[0] / F_TM010_BOARD_HZ - 0.80) < 1e-12
          and abs(B_RESONANCE_BAND_HZ[1] / F_TM010_BOARD_HZ - 1.20) < 1e-12,
          "the resonance window is the retired 0.80-1.20 x the TL model",
          f"{B_RESONANCE_BAND_HZ[0]/1e9:.4f}-{B_RESONANCE_BAND_HZ[1]/1e9:.4f} GHz")

    print("the Stage B builder IS the retired run_openems build block:")
    try:
        proof = _copy_proof_b()
        for old, _ in B_COPY_SUBSTITUTIONS + B_RUNG_SUBSTITUTIONS:
            check(proof["counts"][old] == 1, f"the substitution target appears once: "
                  f"{old.strip()[:60]!r}", f"{proof['counts'][old]}")
        check(proof["matches"],
              "_build_patch_board_at_rung's block IS the frozen slice with exactly the "
              "declared copy and rung substitutions applied")
        if not proof["matches"]:
            import difflib
            notes.append("\n".join(difflib.unified_diff(
                proof["derived"].splitlines(), proof["mine"].splitlines(),
                "derived-from-the-frozen-slice", "_build_patch_board_at_rung", lineterm="")))
        if proof["retired_on_disk"]:
            check(proof["frozen_matches_disk"],
                  f"the frozen slice IS {RETIRED_REL_PATH}::{RETIRED_FUNCTION}'s block, "
                  f"character for character")
        else:
            print(f"  [--] {RETIRED_REL_PATH} is gone; the proof runs against the frozen "
                  f"slice from {RETIRED_FROZEN_AT_COMMIT[:10]}")
        mine_b = proof["mine"]
        check("    FDTD.SetGaussExcite(f0, fc)" in mine_b and "    f0, fc = 2.4e9, 1.2e9" in mine_b,
              "the excitation, SetGaussExcite(2.4e9, 1.2e9), is inside the compared block")
        check("NrTS" not in mine_b and "'MUR'" not in mine_b,
              "the builder carries neither the retired cap nor a MUR face")
    except Exception as exc:
        check(False, "the Stage B copy proof runs", repr(exc))

    print("the Stage A builder IS the tutorial's build block:")
    try:
        pa = _copy_proof_a()
        for old, _ in A_COPY_SUBSTITUTIONS:
            check(pa["counts"][old] == 1, f"the substitution target appears once: {old!r}")
        check(pa["matches"],
              "_build_stage_a_tutorial's block IS the frozen tutorial lines 28-104, indented, "
              "with the one declared substitution")
        if not pa["matches"]:
            import difflib
            notes.append("\n".join(difflib.unified_diff(
                pa["derived"].splitlines(), pa["mine"].splitlines(),
                "derived-from-the-frozen-tutorial", "_build_stage_a_tutorial", lineterm="")))
        check(pa["import_line_present"],
              "the builder imports C0 and EPS0 from openEMS.physical_constants before the block")
        free = _free_names(_FROZEN_TUTORIAL_BUILD)
        check(free == A_FREE_NAMES,
              "the block reads no name beyond C0, EPS0, np, openEMS, ContinuousStructure",
              f"{sorted(free)}")
        fdtd_line = next(ln for ln in _FROZEN_TUTORIAL_BUILD.split("\n")
                         if ln.startswith("FDTD = openEMS("))
        kw = _frozen_call_kwargs(fdtd_line)
        check(kw == {"NrTS": A_REAL_NRTS, "EndCriteria": A_REAL_END_CRITERIA},
              "the real pass's NrTS / EndCriteria are read from the tutorial's own line",
              f"{kw}")
        check(_frozen_assignment(_FROZEN_TUTORIAL_BUILD, "f0") == A_F0_HZ
              and _frozen_assignment(_FROZEN_TUTORIAL_BUILD, "fc") == A_FC_HZ,
              "f0 = 2e9, fc = 1e9 are the tutorial's")
        tut = {n: _frozen_assignment(_FROZEN_TUTORIAL_BUILD, n)
              for n in ("patch_width", "patch_length", "substrate_epsR", "substrate_thickness")}
        check(math.isclose(tut["patch_width"] * 1e-3, A_PATCH_RESONANT_LENGTH_M, rel_tol=1e-12)
              and math.isclose(tut["patch_length"] * 1e-3, A_PATCH_WIDTH_M, rel_tol=1e-12)
              and tut["substrate_epsR"] == A_SUB_EPS_R
              and math.isclose(tut["substrate_thickness"] * 1e-3, A_SUB_H_M, rel_tol=1e-12),
              "the TL model reads the tutorial's own 32 x 40 mm, eps_r 3.38, 1.524 mm", f"{tut}")
        grid_line = _FROZEN_TUTORIAL_POSTPROCESSING[120]
        g = ast.parse(grid_line).body[0].value
        check(ast.literal_eval(g.args[2]) == A_N_FREQS
              and ast.unparse(g.args[0]) == "max(1000000000.0, f0 - fc)"
              and ast.unparse(g.args[1]) == "f0 + fc",
              "the CalcPort grid is the tutorial's linspace(max(1e9, f0-fc), f0+fc, 401)",
              f"{stage_a_freqs_hz()[0]/1e9:g}-{stage_a_freqs_hz()[-1]/1e9:g} GHz")
        theta_call = ast.parse(_FROZEN_TUTORIAL_POSTPROCESSING[142].strip()).body[0].value
        theta_args = tuple(ast.literal_eval(a) for a in theta_call.args)
        phi = tuple(ast.literal_eval(ast.parse(_FROZEN_TUTORIAL_POSTPROCESSING[143].strip())
                                     .body[0].value))
        call = ast.parse(_FROZEN_TUTORIAL_POSTPROCESSING[144].strip()).body[0].value
        center = ast.literal_eval(next(k.value for k in call.keywords if k.arg == "center"))
        check(theta_args == A_FARFIELD_THETA and phi == A_FARFIELD_PHI
              and tuple(center) == A_FARFIELD_CENTER,
              "the far-field call is the tutorial's: theta arange(-180, 180, 2), phi 0 / 90, "
              "centre [0, 0, 1e-3]")
        check(A_FARFIELD_THETA == B_FARFIELD_THETA and A_FARFIELD_PHI == B_FARFIELD_PHI
              and A_FARFIELD_CENTER == B_FARFIELD_CENTER,
              "and the retired Stage B builder's call used the same angles and centre")
    except Exception as exc:
        check(False, "the Stage A copy proof runs", repr(exc))

    print("the container check refuses a tutorial that differs from the frozen one:")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        same = Path(tmp) / "a.py"
        same.write_text("# header\n" + _FROZEN_TUTORIAL_BUILD + "\n")
        info = _tutorial_source_in_image(str(same))
        check(info["present"] and info["frozen_block_found_verbatim"]
              and not info["sha256_matches_pinned"]
              and _tutorial_source_refusal(info) is not None,
              "a file that holds the block but is not the pinned file is refused on sha256")
        other = Path(tmp) / "b.py"
        other.write_text(_FROZEN_TUTORIAL_BUILD.replace("feed_pos = -6", "feed_pos = -5"))
        info_b = _tutorial_source_in_image(str(other))
        check(not info_b["frozen_block_found_verbatim"]
              and "not in it verbatim" in (_tutorial_source_refusal(info_b) or ""),
              "a file whose build block moved is refused on the block")
        info_c = _tutorial_source_in_image(str(Path(tmp) / "absent.py"))
        check(not info_c["present"] and _tutorial_source_refusal(info_c) is None,
              "an image without the file is recorded, not refused")

    print("the documented result and the gate:")
    doc = STAGE_A_DOCUMENTED
    check(doc["f_dip_hz"] == 2.430e9 and doc["depth_db"] == -26.8,
          "documented dip 2.430 GHz, -26.8 dB")
    check(abs(F_TM010_TUTORIAL_HZ / 1e9 - 2.5134) < 5e-4,
          "the TL model for the tutorial board is 2.5134 GHz", f"{F_TM010_TUTORIAL_HZ/1e9:.5f}")
    ok = stage_a_gate_verdict(doc["f_dip_hz"], doc["depth_db"])
    check(ok["passed"], "the documented dip itself PASSES")
    tl = stage_a_gate_verdict(F_TM010_TUTORIAL_HZ, -26.8)
    check(not tl["passed"] and not tl["f_ok"] and tl["depth_ok"],
          "a dip at the TL model's 2.513 GHz (3.4 % high) is REJECTED on frequency")
    shallow = stage_a_gate_verdict(doc["f_dip_hz"], -8.0)
    check(not shallow["passed"] and shallow["f_ok"] and not shallow["depth_ok"],
          "a -8 dB dip at the documented frequency is REJECTED on depth")
    check(not stage_a_gate_verdict(float("nan"), -30.0)["passed"],
          "a NaN frequency is REJECTED")
    rr = STAGE_A_RECORDED_REPRODUCTION
    check(stage_a_gate_verdict(rr["f_s11_dip_hz"], rr["s11_dip_db"])["passed"],
          "the recorded run of the unmodified tutorial (audit trail) sits inside the gate",
          f"{rr['f_s11_dip_hz']/1e9:.4f} GHz, "
          f"{(rr['f_s11_dip_hz'] - doc['f_dip_hz']) / doc['f_dip_hz'] * 100:+.2f} % from the figure")
    log_path = _repo_root() / rr["log"].split(",")[0]
    if log_path.is_file():
        log = log_path.read_text()
        check("LITERAL-TUTORIAL reproduce-gate number: S11 dip f_res = 2.4325 GHz, depth -33.0 dB"
              in log and "FDTD simulation size: 49x47x45 --> 103635 FDTD cells" in log
              and "FDTD timestep is: 1.07987e-12 s" in log
              and "Time for 12046 iterations with 103635.00 cells : 12.03 sec" in log
              and "(-48.41dB)" in log and "version v0.37.0-rc1-2-g7b051bb" in log,
              "the recorded numbers are the committed log's own lines")
    else:
        print(f"  [--] {log_path} is gone; the frozen numbers are the record")
    edge_lo = doc["f_dip_hz"] * (1 - STAGE_A_WINDOW_REL)
    check(stage_a_gate_verdict(edge_lo * 1.0000001, -11.0)["passed"]
          and not stage_a_gate_verdict(edge_lo * 0.999, -11.0)["passed"],
          "the window's lower edge is where it says", f"{edge_lo/1e9:.4f} GHz")

    print("the features on planted curves, run through the gate where Stage A would:")
    try:
        sf = _gate.load_spectral_features()
        fa = stage_a_freqs_hz()
        f_planted = 2.4317e9
        # Unloaded Q 38.6: a matched parallel resonance whose -10 dB band is
        # 0.667 f / Q = 42 MHz wide, the documented figure's span.
        x = 2.0 * 38.6 * (fa - f_planted) / f_planted
        gamma = -1j * x / (2.0 + 1j * x) * 0.999 + 0.001      # a matched parallel resonance
        zin = 50.0 * (1 + gamma) / (1 - gamma)
        feats = _one_port_features(sf, STAGE_A_BAND_HZ, with_tutorial_pick=True)(fa, gamma, zin)
        r = feats["resonance"]
        check(abs(r["refined_f_ghz"] * 1e9 - f_planted) < 1.0e6,
              "a planted matched resonance is found within 1 MHz",
              f"{r['refined_f_ghz']:.5f} GHz at {r['depth_db']:.2f} dB")
        check(r["band_minus10db"] is not None and 30.0 < r["band_minus10db"]["width_mhz"] < 60.0,
              "its -10 dB band is found and has the planted width (~42 MHz)",
              f"{(r['band_minus10db'] or {}).get('width_mhz', float('nan')):.2f} MHz")
        check(abs(r["zin_at_refined_ohm"]["re"] - 50.0) < 2.0,
              "Zin at the planted match reads ~50 ohm", f"{r['zin_at_refined_ohm']['re']:.2f}")
        check(feats["tutorial_pick"]["f_hz"] is not None,
              "the tutorial's own pick finds it too")
        check(stage_a_gate_verdict(r["refined_f_ghz"] * 1e9, r["depth_db"])["passed"],
              "and the gate passes it")
        flat = 0.995 * np.exp(-1j * fa / 1e9) * (1 - 0.01 * np.exp(-((fa - 2.43e9) / 5e7) ** 2))
        ff = _one_port_features(sf, STAGE_A_BAND_HZ, with_tutorial_pick=True)(fa, flat, 50 + 0j * fa)
        rf = ff["resonance"]
        check(not stage_a_gate_verdict(rf["refined_f_ghz"] * 1e9, rf["depth_db"])["passed"],
              "a curve with no dip -- a 0.1 dB ripple at the documented frequency -- FAILS the "
              "gate", f"{rf['refined_f_ghz']:.4f} GHz at {rf['depth_db']:.3f} dB")
        check(rf["band_minus10db"] is None and ff["tutorial_pick"]["f_hz"] is None,
              "and has no -10 dB band and no tutorial pick")
        fb = stage_b_freqs_hz()
        gb = -1j * (2 * 10.0 * (fb - 2.33e9) / 2.33e9) / (2.0 + 1j * (2 * 10.0 * (fb - 2.33e9) / 2.33e9))
        rb = _one_port_features(sf, B_RESONANCE_BAND_HZ)(fb, gb, 50 * (1 + gb) / (1 - gb))["resonance"]
        check(abs(rb["refined_f_ghz"] - 2.33) < 1e-3 and "tutorial_pick" not in rb,
              "Stage B's window finds a planted 2.33 GHz dip", f"{rb['refined_f_ghz']:.5f} GHz")
    except Exception as exc:
        check(False, "the shared estimators load by path and run", repr(exc))

    print("the one-port passivity witness bounds |S11|^2 at 1.05:")
    s_ok = np.full(11, 1.02 + 0j)
    s_bad = np.full(11, 1.03 + 0j)
    try:
        _gate._passivity_witness(s_ok, np.zeros_like(s_ok), "planted", tol=PASSIVITY_TOL)
        check(True, "|S11| = 1.02 (|S11|^2 = 1.0404) passes")
    except RuntimeError as exc:
        check(False, "|S11| = 1.02 (|S11|^2 = 1.0404) passes", str(exc))
    try:
        _gate._passivity_witness(s_bad, np.zeros_like(s_bad), "planted", tol=PASSIVITY_TOL)
        check(False, "|S11| = 1.03 (|S11|^2 = 1.0609) is refused")
    except RuntimeError:
        check(True, "|S11| = 1.03 (|S11|^2 = 1.0609) is refused")
    check(_gate.failed_output_path(Path("/tmp/openems_patch.json")).name
          == "openems_patch_FAILED.json", "a failed gate's evidence file is named from the output")

    print("the plan, reported (not gated):")
    for name, p in _stage_plans(list(STAGE_NAMES)).items():
        if name == "stage_a":
            print(f"  [--] stage_a: ~{p['cells_estimate']:,} cells, dt ~{p['cfl_dt_s']*1e12:.3f} ps, "
                  f"<= {_fmt_s(p['planned_cost_s'])} at the tutorial's cap")
        else:
            print(f"  [--] {name}: ~{p['cells_estimate']:,} cells, {p['substrate_z_cells']} "
                  f"substrate cells, smallest x gap {p['smallest_explicit_spacing']['x']['mm']*1e3:.1f} "
                  f"um, dt ~{p['cfl_dt_s']*1e12:.3f} ps, ~{_fmt_s(p['planned_cost_s'])}")

    print()
    print("WHAT THIS SELF-CHECK CANNOT DO: it proves the builders' SOURCE is the retired "
          "script's and the tutorial's, character for character, and checks the arithmetic "
          "around them. It does not run openEMS, so it cannot say where CSXCAD puts the "
          "smoothed lines, what timestep openEMS picks, or that the port and the patch land "
          "where declared; the run records all of that from the realized grid.")
    for note in notes:
        print(note)
    print()
    print(f"SELF-CHECK {'PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=["A", "B", "both"], default="both")
    p.add_argument("--output", default=None, help="where the JSON record is written")
    p.add_argument("--sim-root", default="/tmp/rt5880_patch_openems")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--nf2ff", action=argparse.BooleanOptionalAction, default=True,
                   help="Stage B: record the NF2FF box and report the directivity at the "
                        "resonance (default on; --no-nf2ff turns it off)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args(argv)

    if args.self_check:
        return _self_check()
    if args.dry_run:
        return _dry_run(args.stage, args.nf2ff)

    if not args.output:
        print("ERROR: --output is required for a real run", file=sys.stderr)
        return 3
    if not os.environ.get("RFX_OPENEMS_COMMIT"):
        print("ERROR: RFX_OPENEMS_COMMIT is not set. The record must name the openEMS build "
              "it came from; the job file exports it from the image. Refusing to produce a "
              "reference with no solver provenance.", file=sys.stderr)
        return 3
    try:
        sf = _gate.load_spectral_features()
        if not (_copy_proof_b()["matches"] and _copy_proof_a()["matches"]):
            raise RuntimeError("a builder is not its frozen source -- run --self-check")
    except Exception as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 3

    stages = _gate.stages_for(args.stage)
    image_tutorial = _tutorial_source_in_image() if "stage_a" in stages else None
    if image_tutorial is not None:
        refusal = _tutorial_source_refusal(image_tutorial)
        if refusal:
            print(f"CONFIG ERROR: {refusal}. Refusing to run a reproduce gate on a tutorial "
                  f"that is not the frozen one.", file=sys.stderr)
            return 3
        print(f"tutorial in this container: {image_tutorial}", flush=True)

    try:
        _gate._import_openems()
    except Exception as exc:
        print(f"openEMS IS NOT IMPORTABLE: {exc!r}", file=sys.stderr)
        return 2

    print("=" * 78)
    print("RT/Duroid 5880 patch antenna -- openEMS reference, with provenance")
    print("=" * 78)
    if "stage_a" not in stages:
        print("WARNING: --stage B does not run the reproduce gate. The record carries "
              "stage_a: null and reproduce_gate_ran: false. The job file submits --stage "
              "both.", flush=True)

    records: dict = {name: None for name in STAGE_NAMES}
    stage_meta: dict = {}
    stage_a_gate: dict = {}
    out = Path(args.output)

    def write_failed(msg: str) -> int:
        failed = _build_artifact(records, stage_meta, stage_a_gate, stages,
                                 image_tutorial=image_tutorial, do_gain=args.nf2ff,
                                 failed_gate=msg)
        path = _gate.failed_output_path(out)
        _gate.write_record(failed, path)
        print(f"evidence written to {path}", file=sys.stderr)
        return 1

    for name in stages:
        print(f"--- {name} ---", flush=True)
        try:
            if name == "stage_a":
                record, meta = _run_stage_a(sim_root=args.sim_root, threads=args.threads, sf=sf)
            else:
                record, meta = _run_stage_b(label=name, sim_root=args.sim_root,
                                            threads=args.threads,
                                            resolution_factor=STAGE_B_FACTORS[name], sf=sf,
                                            do_gain=args.nf2ff)
        except _gate.StageFailure as exc:
            print(f"SANITY GATE FAILED [{name}]: {exc}", file=sys.stderr)
            records[name] = exc.partial or None
            stage_meta[name] = exc.meta
            return write_failed(str(exc))
        records[name] = record
        stage_meta[name] = meta
        res = record.get("resonance") or {}
        run = meta.get("solver_run") or {}
        band = res.get("band_minus10db") or {}
        z = res.get("zin_at_refined_ohm") or {}
        print(f"  |S11| minimum {res.get('refined_f_ghz', float('nan')):.5f} GHz (bin "
              f"{res.get('bin_f_ghz', float('nan')):.4f}) at {res.get('depth_db', float('nan')):.2f} dB"
              f" | -10 dB band {band.get('width_mhz', float('nan')):.1f} MHz | Zin "
              f"{z.get('re', float('nan')):.1f}{z.get('im', float('nan')):+.1f}j ohm | "
              f"{run.get('timesteps')} steps, last energy {run.get('final_energy_db')} dB, "
              f"{run.get('wall_time_s')} s", flush=True)
        if name == "stage_a":
            if "refined_f_ghz" not in res:
                return write_failed(f"[stage_a] the resonance estimator produced no frequency: {res!r}")
            stage_a_gate = stage_a_gate_verdict(res["refined_f_ghz"] * 1e9, res["depth_db"])
            record["gate"] = stage_a_gate
            print(f"  reproduce gate: {res['refined_f_ghz']:.5f} GHz vs documented "
                  f"{STAGE_A_DOCUMENTED['f_dip_hz']/1e9:.3f} GHz "
                  f"({stage_a_gate['deviation_pct']:+.2f} %, window +-{STAGE_A_WINDOW_REL*100:g} %) "
                  f"-> {'ok' if stage_a_gate['f_ok'] else 'RED'}; depth {res['depth_db']:.2f} dB "
                  f"<= {STAGE_A_MAX_DEPTH_DB:g} -> {'ok' if stage_a_gate['depth_ok'] else 'RED'} => "
                  f"{'PASSED' if stage_a_gate['passed'] else 'FAILED'}", flush=True)
            if not stage_a_gate["passed"]:
                print("REPRODUCE GATE FAILED: no Stage B record is written.", file=sys.stderr)
                return write_failed("[stage_a] reproduce gate FAILED: "
                                    f"{res['refined_f_ghz']:.5f} GHz, {res['depth_db']:.2f} dB")

    artifact = _build_artifact(records, stage_meta, stage_a_gate, stages,
                               image_tutorial=image_tutorial, do_gain=args.nf2ff)
    _gate.write_record(artifact, out)
    print(f"\n=== written to {out} ===")
    print("run_id is null by design: VESSL does not export the run id into the pod, so the "
          "submitter records it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
