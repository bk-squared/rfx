"""Arm C — the binary heuristic arm, budget-counted in Maxwell solves.

WHAT THIS ARM IS FOR
--------------------
It answers the PLAN's arm-C question — *does the gradient actually buy anything
here?* — and it is written to be able to say NO. The pixelated-filter literature
is the incumbent for this device class, not the challenger: Gomez et al.
(Sci. Rep. 15, 2025) ran a GA over 1 536 pixels in CST and measured −61 dB on a
laser-ablated board; Zhang & Xu (IEEE MWTL 34(1):29–32, 2024) ran 32x32 binary
pixels for a dual-band microstrip filter. Those campaigns cost of order
1 000–2 000 full-wave solves. If a competent binary search on OUR grid, OUR
metric and OUR solver matches the gradient at equal budget, that is a real
finding about this problem class and we need it before building more machinery
on the gradient.

So this file is deliberately NOT a strawman. Every reduction a practitioner
would actually make is implemented, and section "HONEST ACCOUNTING" below lists
exactly which of them the gradient arm does not get.

THE BUDGET IS THE EXPERIMENT
----------------------------
``--budget-solves N`` is a hard stop counted in **Maxwell solves**, never in
iterations, because that is the only unit in which a gradient step (1 forward +
1 backward ~= 2 solves, independent of the variable count) and a heuristic step
(1 forward per candidate, and the count scales with the variable count) are
commensurable. The deliverable is the whole trajectory — solve count, wall
time, best M so far, and the design that achieved it — written to
``<run>_trajectory.jsonl`` after every evaluation, so a killed job still leaves
a curve. ``<run>_checkpoint.{json,npz}`` is written every ``--checkpoint-every``
solves and ``--resume`` continues from it with the RNG stream, the dedup cache
and the search state intact.

Two things are NOT charged to the budget, and the gradient arm gets the same
exemption, so the comparison is unaffected:

  * the **empty-line reference** solve. It is campaign infrastructure, shared
    and cached across every arm, and it is solved once per (dx, window, grid).
  * the **final verification** re-solve of the best design at the 90-period
    window on the 123-point scoring grid. Every arm's headline number is
    quoted from that same independent evaluation.

THE TWO HEURISTICS
------------------
Two, behind one CLI, so the comparison does not rest on one algorithm choice.

``--heuristic dbs`` — **direct binary search** (greedy single-bit flipping with
a randomised sweep order). This is the incumbent local method for pixelated
devices: propose a flip of one pixel, solve, keep it if the score improved,
revert otherwise, and repeat over a random permutation of the pixels until a
full sweep accepts nothing. It is the strongest thing you can do with a
sequential budget when you have no gradient, it has no tuning parameters to get
wrong, and it is exactly the "binary search on the same grid" the PLAN names.
When a sweep converges at the finest block level the incumbent is *shaken*
(a small random multi-bit perturbation, best-so-far retained) rather than left
to burn the remaining budget on a converged sweep.

``--heuristic bpso`` — **binary particle swarm** (Kennedy & Eberhart 1997
sigmoid-transfer discrete PSO), with inertia annealing 0.9 -> 0.4, velocity
clamped at |v| <= 4 (the standard fix for the sigmoid saturating and freezing
the swarm), elitist replacement of the worst particle by gbest, and mutation
restart of the worst quartile after ``--bpso-stall`` stagnant generations.

Why BPSO rather than a compact GA, since the requirement allows either:

  1. **Budget accounting is exact and pre-registerable.** A generation costs
     exactly ``swarm`` solves, so ``generations = budget // swarm`` with no
     remainder logic and no dependence on how many children a variation
     operator happened to produce. With a GA the solve count depends on
     elitism/duplication policy, which is precisely the kind of accounting a
     referee would (rightly) pick at in an equal-budget claim.
  2. **Continuity with our own accepted paper**, which already reports
     budget-matched particle-swarm and GA trailing the gradient by >= 11.6 dB
     on the dielectric taper. Moving the same experiment to binary metal is
     the natural extension, and keeping the swarm family keeps the two
     comparable.
  3. **It is the harder opponent on a warm start.** A seeded swarm keeps the
     classical design as gbest from generation 0 and searches its neighbourhood
     immediately, where a GA's crossover would spend early generations
     destroying it.

  The features that make a GA competitive (mutation, elitism, restart from
  stagnation) are all present above; what is dropped is crossover. That is a
  real difference and it is stated rather than hidden. If arm A wins, "you
  should have run a GA" is a legitimate referee objection and the answer is to
  run one, not to argue.

REDUCTIONS APPLIED (each one documented, each one a CLI flag)
-------------------------------------------------------------
R1. **Coarse-to-fine pixel blocks** (``--blocks 8 4 2 1``). The search variable
    is a BxB block of cells, not a cell. 13 160 binary variables at B=1 becomes
    2 x 12 x 9 = 216 at B=8. Levels are run in order, each seeded from the
    previous level's best (exact, because block boundaries at powers of two
    nest), and a level that converges early hands its unspent budget to the
    next. This is what the pixelated literature actually does — Gomez ran
    1 536 pixels, not 105 960 — and without it neither heuristic can move at
    all inside a few-hundred-solve budget.

R2. **Warm start from the calibrated classical design** (``--init classical``,
    the default; ``--warm-from-armd DIR`` picks arm D's best d3 record and
    rebuilds its mask). The heuristic starts from the incumbent engineering
    answer and only has to improve on it. ``--init empty`` and ``--init
    random`` are the controls, and the headline run should report both, because
    "the heuristic beat the classical design" and "the heuristic could not
    improve on the classical design it was handed" are different findings.

R3. **Early reject without a solve.** Three predicates, all cheap, all
    conservative, all counted in the trajectory so the rejection rate is
    visible:
      - ``duplicate``  — the exact mask has been evaluated before. Reuses the
        stored score at zero solve cost. DBS revisits and swarm particles
        collide, so this is worth 5-15 % of the budget in practice.
      - ``empty``      — no metal at all. Its score is not estimated, it is
        COMPUTED EXACTLY from the empty-line reference (a design with no PEC
        boxes IS the bare fixture, so IL == 0 identically), at zero solve cost.
      - ``overfill``   — fill fraction above ``--max-fill`` (default 0.60). A
        box that is two-thirds metal is a ground plane over the line, and the
        frozen metric already refuses to rank it
        (``DEGENERATE_IL_PASS_MEAN_DB``); this only avoids paying 45 s to be
        told so. It is a genuine restriction on the search space and it is
        logged, so if a run ever presses against it that shows up.

    NOT applied, deliberately: **floating-island pruning**. Metal not connected
    to the trace is a parasitic coupled resonator, which is a legitimate — and
    for skirt sharpness, quite likely a useful — filter element. Deleting
    islands would be the kind of "cleanup" that quietly removes the mechanism
    the arm exists to find. Island cell counts are logged instead.

R4. **Optional symmetry** (``--symmetry none|x|y|xy``), default ``none``.
    ``x`` pairs block ``bi`` with block ``nbx-1-bi`` (mirror along the line);
    ``y`` pairs ``(lo, bj)`` with ``(hi, nby-1-bj)`` (mirror about the trace);
    ``xy`` does both, quartering the dimension.

    **Both mirrors are exact at BLOCK level and only approximate in cells**,
    because the last block along each axis is ragged: 94 cells at B=8 is eleven
    8-cell blocks plus one of 6, and 70 cells at B=4 is seventeen 4-cell blocks
    plus one of 2. The mirrored partner of a full block at one edge is the short
    block at the other, so the realized geometry is off by the remainder at the
    box edge. Making the partition palindromic instead would fix that and break
    the block NESTING that makes the coarse-to-fine hand-off in R1 exact, which
    is the more valuable property. The mismatch vanishes at B=1 and B=2 (2
    divides both 94 and 70).

    Default is ``none`` for a separate and stronger reason: BOTH mirrors exclude
    the classical two-stub design (two different stub lengths at two different
    positions), which is the warm start and the thing to beat.

HONEST ACCOUNTING — what arm C gets that the gradient arm does not
------------------------------------------------------------------
Stated in both directions, because an equal-budget claim is only worth
something if the asymmetries are on the table.

  Arm C gets, arm A/B does not:
    * the CALIBRATED classical warm start (R2), as an exact binary mask, taken
      from arm D's own best swept design. ``armAB_gradient.py`` does offer a
      ``--init stub`` seed, so this is a difference of kind rather than of
      availability: theirs is a GRAY DENSITY field shaped like a stub, which
      the filter and the continuation schedule then blur, and Phase-1's
      stub-seeded arm B was the one that did WORSE than the unseeded arm C.
      Arm C's start is the calibrated design itself, evaluated unquantised at
      solve 1. If the headline comparison uses a warm-started arm C, it must
      also report the ``--init empty`` control, or the claim is about the warm
      start rather than about the heuristic.
    * the dedup cache (R3/duplicate). A gradient step never lands on a
      previously evaluated design exactly, so it cannot benefit.
    * the exact zero-cost score for the empty design (R3/empty).
    * the reduced dimension from blocks (R1). The gradient arm optimizes all
      13 160 variables directly — its per-step cost does not depend on the
      variable count, so coarsening would buy it nothing and it is not given
      it.
    * a search space restricted to binary from the start, so it never pays the
      binarization loss the gradient arm pays at thresholding.

  Arm A/B gets, arm C does not:
    * derivative information: 13 160 partial derivatives per 2 solves, versus
      1 bit of ordering information per 1 solve here.
    * a continuous relaxation, so it can move all variables at once and can
      traverse regions no single bit flip reaches.
    * filter-radius continuation, which is a smoothing of the LANDSCAPE that
      has no discrete analogue.

  Both get identically: the same fixture, the same design box, the same frozen
  metric via ``phase2_calibrate.score_design``, the same descent window and
  descent grid during the search, the same 90-period / 123-point verification
  for the quoted number, the same empty-line reference, and the same etch
  bracket in ``robust_eval`` afterwards.

Run
---
  # headline (GPU): 600 solves, DBS, warm-started from arm D
  python research/metal_to/armC_binary.py --heuristic dbs --budget-solves 600 \
      --periods 45 --run C_dbs_warm

  # the control that makes the warm start reportable
  python research/metal_to/armC_binary.py --heuristic dbs --budget-solves 600 \
      --periods 45 --init empty --run C_dbs_cold

  # swarm at the same budget
  python research/metal_to/armC_binary.py --heuristic bpso --budget-solves 600 \
      --periods 45 --swarm 16 --run C_bpso_warm

  # resume a killed job
  python research/metal_to/armC_binary.py --heuristic dbs --budget-solves 600 \
      --periods 45 --run C_dbs_warm --resume

  # CPU smoke, tiny budget, real solves at a 1-period window (NOT QUOTABLE)
  SMOKE=1 python research/metal_to/armC_binary.py --heuristic dbs \
      --budget-solves 3 --periods 1 --verify-periods 0

  # CPU smoke of the SEARCH MACHINERY only, with the mock solver (NOT PHYSICS)
  SMOKE=1 python research/metal_to/armC_binary.py --heuristic bpso \
      --budget-solves 48 --periods 1 --solver mock --verify-periods 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "validation" / "tmtt_paper"))

import phase2_calibrate as cal  # noqa: E402
import phase2_fixture as fx  # noqa: E402
import robust_eval as rb  # noqa: E402
import score_dualband as sd  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = Path(os.environ.get(
    "OUTPUT_DIR",
    HERE / "out_smoke" / "armC" if SMOKE else HERE / "out_vessl" / "armC"))
OUT.mkdir(parents=True, exist_ok=True)
# Same gating armD uses: a SMOKE empty reference is an unsettled ring-down
# artifact and must never sit where a production run would find it.
CACHE = OUT.parent / "empty_ref"
CACHE.mkdir(parents=True, exist_ok=True)

C0 = 2.998e8
F_LO, F_HI = 5.25e9, 5.775e9

#: Sort key handed to a candidate that was rejected without a solve. Sorts after
#: every real design, including a degenerate one, so a rejected candidate can
#: never become the incumbent.
REJECT_KEY = (True, float("inf"), 0.0)


# ---------------------------------------------------------------------------
# 0. Small shared helpers (same conventions as armD_classical)
# ---------------------------------------------------------------------------
def quarter_wave(f, eps_eff):
    return C0 / (f * np.sqrt(eps_eff)) / 4.0


def _grid_hz(which: str):
    """Search grid vs reporting grid.

    ``descent`` is the 68-point grid ``score_dualband`` documents as "NEVER used
    to report a number"; it exists to make each candidate cheap. ``scoring`` is
    the pre-registered 123-point verification grid and is what the final number
    is quoted from.
    """
    g = sd.descent_grid_mhz() if which == "descent" else sd.scoring_grid_mhz()
    return np.asarray(g, dtype=float) * 1e6


def _sha(mask) -> str:
    """Stable content hash of a two-sided mask — the dedup and provenance key."""
    h = hashlib.sha256()
    for name in fx.SIDES:
        a = np.ascontiguousarray(np.asarray(mask[name], dtype=np.uint8))
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _attached(m: np.ndarray, trace_row: int) -> np.ndarray:
    """4-connected metal reachable from the trace-adjacent row.

    Iterative dilation-and-intersect rather than a recursive flood, so it has no
    stack depth and needs no scipy. Used only for LOGGING — nothing is pruned.
    """
    m = np.asarray(m, dtype=bool)
    seed = np.zeros_like(m)
    seed[:, trace_row] = m[:, trace_row]
    cur = seed
    while True:
        nxt = cur.copy()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        nxt &= m
        if np.array_equal(nxt, cur):
            return cur
        cur = nxt


def _realized(mask, box) -> dict:
    """What the lattice actually made, not what was requested.

    armD_classical's ``_realized`` verbatim (cells / cols / rows / ranges per
    side), plus the three free-form witnesses a pixel design needs and a stub
    design does not: fill fraction, how much metal is actually attached to the
    feed, and how much of it survives one cell of over-etch. Nothing here
    changes the design; it is all record.
    """
    out = {}
    tot = filled = 0
    for name in fx.SIDES:
        side = box.side(name)
        m = np.asarray(mask[name], dtype=np.uint8)
        tot += m.size
        filled += int(m.sum())
        if not m.any():
            out[f"{name}_cells"] = 0
            out[f"{name}_attached_cells"] = 0
            out[f"{name}_island_cells"] = 0
            out[f"{name}_erode1_cells"] = 0
            continue
        cols = np.where(m.any(axis=1))[0]
        rows = np.where(m.any(axis=0))[0]
        att = _attached(m, side.trace_row)
        ero = rb.erode(m, 1, outside=side.etch_outside, connectivity=4)
        out[f"{name}_cells"] = int(m.sum())
        out[f"{name}_cols"] = int(len(cols))
        out[f"{name}_rows"] = int(len(rows))
        out[f"{name}_col_range"] = [int(cols.min()), int(cols.max())]
        out[f"{name}_row_range"] = [int(rows.min()), int(rows.max())]
        out[f"{name}_attached_cells"] = int(att.sum())
        out[f"{name}_island_cells"] = int(m.sum() - att.sum())
        out[f"{name}_erode1_cells"] = int(np.asarray(ero).sum())
    out["fill_cells"] = int(filled)
    out["fill_frac"] = float(filled / tot) if tot else 0.0
    return out


def _record(tag, scored, extra):
    """armD_classical's record convention, unchanged, so the two arms' JSON is
    readable by the same downstream code."""
    d = dict(tag=tag, **extra)
    d.update(scored.to_json() if hasattr(scored, "to_json") else
             json.loads(json.dumps(scored, default=lambda o: getattr(
                 o, "__dict__", str(o)))))
    (OUT / f"{tag}.json").write_text(json.dumps(d, indent=2, default=str))
    return d


# ---------------------------------------------------------------------------
# 1. Block / symmetry coder  (reduction R1 and R4)
# ---------------------------------------------------------------------------
def _edges(n: int, b: int) -> np.ndarray:
    """Block boundaries along one axis: 0, b, 2b, ..., n.

    The last block is ragged when ``b`` does not divide ``n`` (94 cells at b=8
    is eleven 8-cell blocks and one 6-cell block). Boundaries at multiples of a
    power of two NEST across levels, which is what makes the coarse-to-fine
    hand-off in :meth:`Coder.upsample` exact rather than approximate.
    """
    e = list(range(0, n, b))
    if e[-1] != n:
        e.append(n)
    return np.asarray(e, dtype=int)


@dataclass(frozen=True)
class Coder:
    """Maps a free binary vector <-> a two-sided per-cell mask.

    The free vector is what the heuristic searches; the mask is what the ONE
    geometry pathway (``phase2_fixture.boxes_from_mask``) consumes. Every
    candidate any heuristic proposes goes through here, so the block reduction
    and the symmetry constraint cannot leak into the physics.
    """

    nx: int
    ny: int
    block: int
    symmetry: str
    ex: np.ndarray
    ey: np.ndarray
    slot: np.ndarray            # (2, nbx, nby) -> free-vector index
    n_vars: int

    @property
    def nbx(self) -> int:
        return len(self.ex) - 1

    @property
    def nby(self) -> int:
        return len(self.ey) - 1

    #: per-cell block index along each axis, so decode is a fancy-index rather
    #: than a 13 160-iteration Python loop at the finest level
    bx: np.ndarray = field(default=None, repr=False)
    by: np.ndarray = field(default=None, repr=False)
    area: np.ndarray = field(default=None, repr=False)

    def blocks_mask(self, v) -> np.ndarray:
        """Free vector -> (2, nbx, nby) block occupancy."""
        return np.asarray(v, dtype=np.uint8)[self.slot]

    def decode(self, v) -> dict:
        """Free vector -> ``{'lo': (nx,ny), 'hi': (nx,ny)}`` cell mask."""
        bm = self.blocks_mask(v)
        ix = np.ix_(self.bx, self.by)
        return {name: bm[s][ix].astype(np.uint8)
                for s, name in enumerate(fx.SIDES)}

    def encode(self, mask, rule: str = "majority") -> np.ndarray:
        """Cell mask -> free vector, by ``rule`` within each block.

        ``majority`` (the default) is what the coarse-to-fine hand-off uses. It
        is exact there, because every fine block lies wholly inside one coarse
        block and the vote is unanimous, and it is the rule that makes
        ``encode(decode(v)) == v`` hold at every level.

        ``any`` is for SEEDING an arbitrary external design (the classical warm
        start) onto a coarse lattice, and the difference is not academic: the
        D-1 calibrated stub is 2 cells wide, which is a minority of every 8x8
        block it touches, so ``majority`` quantises the whole warm start to
        NOTHING at B=8 and the coarse level would start from an empty box.
        ``any`` keeps the stub's topology and fattens it to the block lattice
        instead — a fatter, lower-impedance stub than the calibrated one, which
        is a real change to the design and is why the caller prints what it did.

        Under a symmetry constraint several blocks share one free variable; the
        vote is then taken over their UNION, which is the only choice that keeps
        the round trip an identity.
        """
        acc = np.zeros(self.n_vars, dtype=np.float64)
        seen = np.zeros(self.n_vars, dtype=np.float64)
        for s, name in enumerate(fx.SIDES):
            m = np.asarray(mask[name], dtype=np.float64)
            sums = np.add.reduceat(np.add.reduceat(m, self.ex[:-1], axis=0),
                                   self.ey[:-1], axis=1)
            np.add.at(acc, self.slot[s], sums)
            np.add.at(seen, self.slot[s], self.area)
        frac = acc / np.maximum(seen, 1.0)
        if rule == "majority":
            return (frac >= 0.5).astype(np.uint8)
        if rule == "any":
            return (frac > 0.0).astype(np.uint8)
        if rule == "all":
            return (frac >= 1.0).astype(np.uint8)
        raise ValueError(f"rule must be majority|any|all, got {rule!r}")

    def sweep_solves(self) -> int:
        """Solves one full DBS sweep of this level costs: one per free variable."""
        return self.n_vars

    def upsample(self, v, finer: "Coder") -> np.ndarray:
        """This level's best -> the next (finer) level's starting vector.

        Exact when both block sizes are powers of two, because every fine block
        lies wholly inside one coarse block, so the majority in :meth:`encode`
        is unanimous.
        """
        return finer.encode(self.decode(v))


def make_coder(box, block: int, symmetry: str) -> Coder:
    nx, ny = box.lo.shape
    if box.lo.shape != box.hi.shape:
        raise RuntimeError("the two design sides have different shapes")
    ex, ey = _edges(nx, block), _edges(ny, block)
    nbx, nby = len(ex) - 1, len(ey) - 1
    slot = np.full((2, nbx, nby), -1, dtype=np.int64)
    nxt = 0

    def claim(cells):
        """Give every (side, bi, bj) in ``cells`` one shared free index."""
        nonlocal nxt
        k = -1
        for s, bi, bj in cells:
            if slot[s, bi, bj] >= 0:
                k = int(slot[s, bi, bj])
        if k < 0:
            k, nxt = nxt, nxt + 1
        for s, bi, bj in cells:
            slot[s, bi, bj] = k
        return k

    if symmetry not in ("none", "x", "y", "xy"):
        raise ValueError(f"symmetry must be none|x|y|xy, got {symmetry!r}")
    for s in range(2):
        for bi in range(nbx):
            for bj in range(nby):
                if slot[s, bi, bj] >= 0:
                    continue
                group = {(s, bi, bj)}
                # Mirrors are applied at BLOCK level; with a ragged last block
                # that is not a cell-exact mirror (see R4 in the module
                # docstring). Iterate to close the group under both.
                for _ in range(3):
                    if symmetry in ("x", "xy"):
                        group |= {(a, nbx - 1 - b, d) for a, b, d in tuple(group)}
                    if symmetry in ("y", "xy"):
                        # mirror about the trace: 'lo' block bj <-> 'hi' block
                        # nby-1-bj, because mask[:, j] ascends in GLOBAL y on
                        # BOTH sides, so the trace-adjacent rows are j = ny-1
                        # on 'lo' and j = 0 on 'hi'.
                        group |= {(1 - a, b, nby - 1 - d) for a, b, d in tuple(group)}
                claim(sorted(group))
    if int(slot.min()) < 0:
        raise RuntimeError("symmetry grouping left an unassigned block")
    bx = np.repeat(np.arange(nbx), np.diff(ex))
    by = np.repeat(np.arange(nby), np.diff(ey))
    area = np.outer(np.diff(ex), np.diff(ey)).astype(float)
    return Coder(nx=nx, ny=ny, block=block, symmetry=symmetry,
                 ex=ex, ey=ey, slot=slot, n_vars=nxt, bx=bx, by=by, area=area)


# ---------------------------------------------------------------------------
# 2. Warm start (reduction R2)
# ---------------------------------------------------------------------------
def _stub_pair_mask(box, sep_cells, l_lo_cells, l_hi_cells, w_cells,
                    two_sided: bool):
    """armD_classical._stub_pair, in CELLS, through the same shared pathway."""
    dx = box.dx
    pad_x = box.hi.pads[0]
    x_c = (0.5 * (box.hi.ix_lo + box.hi.ix_hi) - pad_x) * dx
    stubs = [
        ("lo" if two_sided else "hi", x_c - sep_cells * dx / 2.0,
         w_cells * dx, l_lo_cells * dx),
        ("hi", x_c + sep_cells * dx / 2.0, w_cells * dx, l_hi_cells * dx),
    ]
    return fx.mask_from_stubs(stubs, box)


def classical_warm_start(box, armd_dir: Path | None, defaults: dict) -> tuple:
    """The classical design to start from, and where it came from.

    Prefers arm D's own best ``d3_*`` record so the warm start is the CALIBRATED
    classical design rather than a textbook one — the Phase-1 retraction came
    from comparing against an un-calibrated baseline, and a warm start is a
    comparison in disguise. Falls back to the D-1 calibrated lengths (63 cells
    for 5.25 GHz, 58 for 5.775 GHz; the fit is 0.02-0.11 % per width) when no
    d3 record exists yet, and says which it used.
    """
    src = "d1_calibrated_defaults"
    p = dict(defaults)
    if armd_dir is not None and Path(armd_dir).is_dir():
        best, best_m = None, float("inf")
        for f in sorted(Path(armd_dir).glob("d3_*.json")):
            try:
                d = json.loads(f.read_text())
                m = float(d["result"]["M"])
            except Exception:
                continue
            if m < best_m:
                best, best_m = d, m
        if best is not None:
            p = dict(sep_cells=int(best["sep_cells"]),
                     l_lo_cells=int(best["l_lo_cells"]),
                     l_hi_cells=int(best["l_hi_cells"]),
                     w_cells=int(best["w_cells"]),
                     two_sided=bool(best["two_sided"]))
            src = f"armD:{best.get('tag', '?')} (M={best_m:.2f})"
    return _stub_pair_mask(box, **p), src, p


# ---------------------------------------------------------------------------
# 3. Evaluator — the ONLY place a solve is issued or a budget is spent
# ---------------------------------------------------------------------------
@dataclass
class EvalOut:
    key: tuple
    M: float
    sha: str
    reason: str
    consumed: bool
    scored: object = None
    row: dict = field(default_factory=dict)


class Evaluator:
    """Score a candidate, count the budget, log the trajectory.

    Every heuristic in this file talks to the search problem exclusively through
    :meth:`__call__`. That is what makes "stop exactly at ``--budget-solves``"
    a property of one function instead of a discipline spread over two search
    loops, and it is what makes the reject/cache bookkeeping consistent between
    them.
    """

    def __init__(self, box, freqs_hz, periods, *, budget, out, run,
                 max_fill, cache_dir, solver=None, pre_hook=None,
                 max_evals=None, require_settled_empty=True, thr=sd.SCORE,
                 verbose=True):
        self.box = box
        self.freqs = freqs_hz
        self.periods = float(periods)
        self.budget = int(budget)
        self.out = Path(out)
        self.run = run
        self.max_fill = float(max_fill)
        #: Backstop against a search that spends wall time without spending
        #: budget. Every early-reject costs a decode and a hash but no solve, so
        #: a DBS incumbent parked above ``max_fill`` can thrash indefinitely:
        #: measured at 6 294 rejected evaluations for 6 solves before this cap
        #: existed. The cap is generous (20x the budget) so it never binds on a
        #: healthy run, and the reason it fired is reported.
        self.max_evals = int(max_evals if max_evals is not None else 20 * budget)
        self.stop_reason = None
        self.cache_dir = Path(cache_dir)
        self.solver = solver
        #: called with the mask just before a solve. Exists only so the SMOKE
        #: mock solver, whose ``run(sim, freqs, periods)`` signature never sees
        #: the geometry, can be handed it out of band.
        self.pre_hook = pre_hook
        self.require_settled_empty = require_settled_empty
        self.thr = thr
        self.verbose = verbose

        self.solves = 0
        self.evals = 0
        self.rejected = {"duplicate": 0, "empty": 0, "overfill": 0}
        self.cache: dict = {}                 # sha -> (key, M)
        self.best_key = REJECT_KEY
        self.best_M = float("nan")
        self.best_sha = None
        self.best_mask = None
        self.best_solve = -1
        self.best_scored = None
        self.best_scored_sha = None
        self.improvements: list = []          # (solve, sha, mask), THIS process
        #: solve indices at which the best improved, INCLUDING those carried in
        #: from a checkpoint. ``improvements`` holds masks and therefore only
        #: covers this process; the curve must survive a resume, so it is kept
        #: separately.
        self.improvement_solves: list = []
        self.t0 = time.time()
        self.phase = "init"
        self.block = 0
        self.n_vars = 0
        self.traj_path = self.out / f"{self.run}_trajectory.jsonl"

        # The empty-line reference. Solved once, cached, NOT charged to the
        # budget (campaign infrastructure, identical for every arm).
        if pre_hook is not None:
            pre_hook(None)
        self.empty = cal.empty_reference(
            freqs_hz, self.periods, cache_dir=self.cache_dir, solver=solver,
            verbose=verbose)
        if verbose:
            print(f"[armC] {self.empty.summary()}")
        if self.require_settled_empty and not self.empty.settled:
            raise ValueError(
                f"empty reference at {self.empty.path} is NOT SETTLED "
                f"({self.empty.record.get('settling_worst_db')} dB). Pass "
                f"--allow-unsettled-empty only for a smoke, and label the "
                f"result NOT QUOTABLE.")
        self._empty_design = None             # lazily built, zero solves

    # -- budget ------------------------------------------------------------
    @property
    def exhausted(self) -> bool:
        if self.solves >= self.budget:
            self.stop_reason = self.stop_reason or "budget"
            return True
        if self.evals >= self.max_evals:
            if self.stop_reason is None:
                self.stop_reason = "eval_cap"
                print(f"[armC] !! STOPPING EARLY at {self.solves}/{self.budget} "
                      f"solves: {self.evals} evaluations hit the "
                      f"{self.max_evals} cap, i.e. the search is proposing "
                      f"candidates that are rejected without a solve "
                      f"({self.rejected}). The budget curve is TRUNCATED and "
                      f"must be reported as such.")
            return True
        return False

    # -- the exact, solve-free score of a design with no metal --------------
    def empty_design_score(self):
        """The bare fixture's own score, at zero solve cost.

        This is not an estimate. A mask with no metal adds no PEC boxes, so the
        DUT solve IS the empty-line solve at the same (dx, window, grid), and
        the insertion loss is identically zero. Assembling the frozen metric on
        ``il = 0`` therefore reproduces exactly what a solve would return.
        """
        if self._empty_design is None:
            f = np.asarray(self.empty.freqs_mhz, dtype=int)
            il = np.zeros(f.size, dtype=float)
            rec = self.empty.record
            validity = sd.check_validity(
                rec["settling_db"], rec["passivity_correction"],
                rec["reliable"], f, np.minimum(il, self.thr.r_cap_db),
                thr=self.thr, empty_cal_max_db=self.empty.empty_cal_max_db)
            result = sd.score(
                f, il,
                s11_db=np.asarray(self.empty.s11_db, dtype=float),
                s21_db_abs=np.asarray(self.empty.s21_db, dtype=float),
                thr=self.thr, validity=validity)
            self._empty_design = cal.ScoredDesign(
                result=result, validity=validity, il_db=il, freqs_mhz=f,
                record=rec, empty_key=self.empty.key,
                empty_cal_max_db=self.empty.empty_cal_max_db, n_boxes=0,
                label="empty_design")
        return self._empty_design

    # -- the one entry point -----------------------------------------------
    def __call__(self, mask, *, label="cand", accepted=None) -> EvalOut:
        self.evals += 1
        sha = _sha(mask)
        fill = float(sum(int(np.asarray(mask[n]).sum()) for n in fx.SIDES)
                     / sum(np.asarray(mask[n]).size for n in fx.SIDES))
        t_eval = time.time()

        scored, reason, consumed = None, "solve", False
        if sha in self.cache:
            key, M = self.cache[sha]
            reason = "duplicate"
            self.rejected["duplicate"] += 1
        elif fill <= 0.0:
            scored = self.empty_design_score()
            key, M = sd.rank_key(scored.result), float(scored.result.M)
            reason = "empty"
            self.rejected["empty"] += 1
            self.cache[sha] = (key, M)
        elif fill > self.max_fill:
            key, M = REJECT_KEY, float("nan")
            reason = "overfill"
            self.rejected["overfill"] += 1
            self.cache[sha] = (key, M)
        elif self.exhausted:
            # Never issue solve N+1. The caller is expected to check
            # ``exhausted`` first; this is the backstop that makes the budget a
            # property of the code rather than of every loop that uses it.
            return EvalOut(key=REJECT_KEY, M=float("nan"), sha=sha,
                           reason="budget_exhausted", consumed=False)
        else:
            if self.pre_hook is not None:
                self.pre_hook(mask)
            scored = cal.score_design(
                mask, freqs_hz=self.freqs, num_periods=self.periods,
                cache_dir=self.cache_dir, solver=self.solver,
                label=f"{self.run}:{label}",
                require_settled_empty=self.require_settled_empty,
                empty=self.empty, verbose=False)
            self.solves += 1
            consumed = True
            key, M = sd.rank_key(scored.result), float(scored.result.M)
            self.cache[sha] = (key, M)

        improved = key < self.best_key
        if improved:
            self.best_key, self.best_M, self.best_sha = key, M, sha
            self.best_mask = {n: np.asarray(mask[n], dtype=np.uint8).copy()
                              for n in fx.SIDES}
            self.best_solve = self.solves
            if scored is not None:
                self.best_scored, self.best_scored_sha = scored, sha
            self.improvements.append((self.solves, sha, self.best_mask))
            self.improvement_solves.append(self.solves)

        row = {
            "eval": self.evals, "solve": self.solves, "consumed": consumed,
            "reason": reason, "phase": self.phase, "block": self.block,
            "n_vars": self.n_vars, "label": label,
            "mask_sha": sha, "fill_frac": round(fill, 6),
            "M": (None if M != M else round(M, 4)),
            "best_M": (None if self.best_M != self.best_M
                       else round(self.best_M, 4)),
            "best_solve": self.best_solve,
            "accepted": accepted, "improved": bool(improved),
            "wall_s": round(time.time() - t_eval, 2),
            "cum_wall_s": round(time.time() - self.t0, 1),
        }
        if scored is not None:
            r, v = scored.result, scored.validity
            row.update({
                "S_L": round(r.S_L, 3), "S_U": round(r.S_U, 3),
                "S_G": round(r.S_G, 3), "S_P": round(r.S_P, 3),
                "Omega": round(r.Omega, 3),
                "R_L_raw": round(r.R_L_raw, 2), "R_U_raw": round(r.R_U_raw, 2),
                "IL_gap_max": (None if r.IL_gap_max != r.IL_gap_max
                               else round(r.IL_gap_max, 3)),
                "f_notch_L_MHz": r.f_notch_L_MHz,
                "f_notch_U_MHz": r.f_notch_U_MHz,
                "degenerate": bool(r.degenerate), "valid": bool(v.ok),
                "settling_worst_db": (None if v.settling_worst_db !=
                                      v.settling_worst_db
                                      else round(v.settling_worst_db, 1)),
                "n_boxes": int(scored.n_boxes),
            })
        with self.traj_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if self.verbose and (consumed or improved):
            print(f"[armC] solve {self.solves:>5d}/{self.budget}  "
                  f"{self.phase:<14s} B={self.block} "
                  f"M={'   nan' if M != M else f'{M:6.2f}'}  "
                  f"best={self.best_M:6.2f} @{self.best_solve}  "
                  f"({reason}, {row['cum_wall_s']:.0f}s)", flush=True)
        return EvalOut(key=key, M=M, sha=sha, reason=reason, consumed=consumed,
                       scored=scored, row=row)


# ---------------------------------------------------------------------------
# 4. Heuristic (a) — direct binary search
# ---------------------------------------------------------------------------
def run_dbs(ev: Evaluator, coders, v0, rng, *, shake_bits: int,
            state: dict | None = None, checkpoint=None) -> np.ndarray:
    """Greedy single-bit flipping with a randomised sweep order.

    One "sweep" is a random permutation of the free variables; each is flipped,
    scored, and kept only on strict improvement in ``score_dualband.rank_key``
    (so a degenerate design can never displace a non-degenerate one and Omega
    breaks ties only between designs that already have M = 0). A sweep that
    accepts nothing means the incumbent is 1-flip-optimal at this block level;
    the search then refines to the next level, or — at the finest level —
    SHAKES the incumbent by flipping ``shake_bits`` random variables and
    resumes, keeping the best-so-far. Without the shake a converged DBS burns
    the rest of a large budget re-proving convergence.
    """
    lvl = 0 if state is None else int(state["lvl"])
    v = np.asarray(v0, dtype=np.uint8).copy()
    order = (None if state is None or state.get("order") is None
             else np.asarray(state["order"], dtype=np.int64))
    pos = 0 if state is None else int(state.get("pos", 0))
    sweep = 0 if state is None else int(state.get("sweep", 0))
    accepted_this_sweep = (0 if state is None
                           else int(state.get("accepted_this_sweep", 0)))

    ev.block = coders[lvl].block
    ev.n_vars = coders[lvl].n_vars
    ev.phase = f"dbs_L{lvl}_s{sweep}"
    cur = ev(coders[lvl].decode(v), label=f"L{lvl}_incumbent")
    cur_key = cur.key

    while not ev.exhausted:
        c = coders[lvl]
        if order is None or pos >= order.size:
            if order is not None and accepted_this_sweep == 0:
                # 1-flip-optimal at this level.
                if lvl + 1 < len(coders):
                    v = c.upsample(v, coders[lvl + 1])
                    lvl += 1
                    order, pos, sweep, accepted_this_sweep = None, 0, 0, 0
                    ev.block, ev.n_vars = coders[lvl].block, coders[lvl].n_vars
                    ev.phase = f"dbs_L{lvl}_s0"
                    cur = ev(coders[lvl].decode(v), label=f"L{lvl}_incumbent")
                    cur_key = cur.key
                    continue
                k = min(shake_bits, c.n_vars)
                idx = rng.choice(c.n_vars, size=k, replace=False)
                v[idx] ^= 1
                ev.phase = f"dbs_L{lvl}_shake{sweep}"
                cur = ev(c.decode(v), label=f"L{lvl}_shake")
                cur_key = cur.key
            order = rng.permutation(c.n_vars)
            pos, accepted_this_sweep = 0, 0
            sweep += 1
            ev.phase = f"dbs_L{lvl}_s{sweep}"

        i = int(order[pos])
        pos += 1
        v[i] ^= 1
        trial = ev(c.decode(v), label=f"L{lvl}_flip{i}")
        if trial.reason == "budget_exhausted":
            v[i] ^= 1
            break
        if trial.key < cur_key:
            cur_key = trial.key
            accepted_this_sweep += 1
        else:
            v[i] ^= 1                      # revert; the flip did not help

        if checkpoint is not None:
            checkpoint(dict(lvl=lvl, v=v, order=order, pos=pos, sweep=sweep,
                            accepted_this_sweep=accepted_this_sweep))
    return v


# ---------------------------------------------------------------------------
# 5. Heuristic (b) — binary particle swarm
# ---------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def run_bpso(ev: Evaluator, coders, v0, rng, *, swarm, w_hi, w_lo, c1, c2,
             v_max, stall, mut_p, state: dict | None = None,
             checkpoint=None) -> np.ndarray:
    """Binary PSO (Kennedy & Eberhart 1997) with the standard hardening.

    Position ``x`` in {0,1}^n, velocity ``v`` in R^n:

        v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x),  |v| clipped to v_max
        x_i = 1 with probability sigmoid(v_i)

    ``v_max = 4`` is not decoration: without it sigmoid(v) saturates at 0 or 1
    and the swarm freezes, which is the documented BPSO failure mode and the
    usual reason a published BPSO comparison looks weak. Inertia ``w`` anneals
    linearly from ``w_hi`` to ``w_lo`` across the generations the budget allows,
    the worst particle is replaced by gbest every generation (elitism), and
    after ``stall`` generations without a gbest improvement the worst quartile
    is re-seeded as mutations of gbest at rate ``mut_p`` — the compact-GA
    restart, kept because stagnation is the other way these runs fail.

    A generation costs exactly ``swarm`` solves, so the budget maps to
    generations with no remainder logic. Particle 0 of generation 0 is the warm
    start; the rest are mutations of it, so the swarm begins in the
    neighbourhood of the classical design rather than in random noise.
    """
    lvl = 0 if state is None else int(state["lvl"])
    c = coders[lvl]
    n = c.n_vars

    if state is None:
        x = np.zeros((swarm, n), dtype=np.uint8)
        x[0] = np.asarray(v0, dtype=np.uint8)
        for p in range(1, swarm):
            x[p] = x[0] ^ (rng.random(n) < mut_p * 4.0).astype(np.uint8)
        vel = rng.uniform(-1.0, 1.0, size=(swarm, n))
        pbest = x.copy()
        pbest_key = [REJECT_KEY] * swarm
        gbest = x[0].copy()
        gbest_key = REJECT_KEY
        gen, since = 0, 0
    else:
        x = np.asarray(state["x"], dtype=np.uint8)
        vel = np.asarray(state["vel"], dtype=float)
        pbest = np.asarray(state["pbest"], dtype=np.uint8)
        pbest_key = [tuple(k) for k in state["pbest_key"]]
        gbest = np.asarray(state["gbest"], dtype=np.uint8)
        gbest_key = tuple(state["gbest_key"])
        gen, since = int(state["gen"]), int(state["since"])
        swarm, n = x.shape

    gens_total = max(1, ev.budget // max(1, swarm))
    ev.block, ev.n_vars = c.block, c.n_vars

    # Coarse-to-fine for the swarm: split the generations across block levels,
    # re-seeding each level from the upsampled gbest. Same schedule DBS gets.
    per_level = max(1, gens_total // len(coders))

    while not ev.exhausted:
        want_lvl = min(len(coders) - 1, gen // per_level)
        if want_lvl != lvl:
            gbest_fine = coders[lvl].upsample(gbest, coders[want_lvl])
            lvl = want_lvl
            c = coders[lvl]
            n = c.n_vars
            x = np.zeros((swarm, n), dtype=np.uint8)
            x[0] = gbest_fine
            for p in range(1, swarm):
                x[p] = x[0] ^ (rng.random(n) < mut_p * 4.0).astype(np.uint8)
            vel = rng.uniform(-1.0, 1.0, size=(swarm, n))
            pbest = x.copy()
            pbest_key = [REJECT_KEY] * swarm
            gbest = gbest_fine.copy()
            ev.block, ev.n_vars = c.block, c.n_vars

        ev.phase = f"bpso_L{lvl}_g{gen}"
        gen_best_key, gen_best_p = REJECT_KEY, 0
        gen_worst_key, gen_worst_p = None, 0
        for p in range(swarm):
            if ev.exhausted:
                break
            out = ev(c.decode(x[p]), label=f"L{lvl}_g{gen}_p{p}")
            if out.reason == "budget_exhausted":
                break
            if out.key < pbest_key[p]:
                pbest_key[p] = out.key
                pbest[p] = x[p].copy()
            if out.key < gen_best_key:
                gen_best_key, gen_best_p = out.key, p
            if gen_worst_key is None or out.key > gen_worst_key:
                gen_worst_key, gen_worst_p = out.key, p

        if gen_best_key < gbest_key:
            gbest_key = gen_best_key
            gbest = x[gen_best_p].copy()
            since = 0
        else:
            since += 1

        # elitism: the worst particle is replaced by gbest
        x[gen_worst_p] = gbest.copy()
        vel[gen_worst_p] = 0.0

        if since >= stall:
            k = max(1, swarm // 4)
            worst = np.argsort([kk[1] for kk in pbest_key])[-k:]
            for p in worst:
                x[p] = gbest ^ (rng.random(n) < mut_p).astype(np.uint8)
                vel[p] = rng.uniform(-1.0, 1.0, size=n)
                pbest[p] = x[p].copy()
                pbest_key[p] = REJECT_KEY
            since = 0

        w = w_hi + (w_lo - w_hi) * min(1.0, gen / max(1, gens_total - 1))
        r1 = rng.random((swarm, n))
        r2 = rng.random((swarm, n))
        vel = (w * vel
               + c1 * r1 * (pbest.astype(float) - x.astype(float))
               + c2 * r2 * (gbest.astype(float)[None, :] - x.astype(float)))
        vel = np.clip(vel, -v_max, v_max)
        x = (rng.random((swarm, n)) < _sigmoid(vel)).astype(np.uint8)
        gen += 1

        if checkpoint is not None:
            checkpoint(dict(lvl=lvl, x=x, vel=vel, pbest=pbest,
                            pbest_key=[list(k) for k in pbest_key],
                            gbest=gbest, gbest_key=list(gbest_key),
                            gen=gen, since=since))
    return gbest


# ---------------------------------------------------------------------------
# 6. Mock solver — SMOKE ONLY, and it is not physics
# ---------------------------------------------------------------------------
#: Set by :class:`Evaluator` immediately before a mock solve. The ``solver``
#: contract is ``run(sim, freqs, periods)`` — it never sees the mask — so the
#: mock has to be handed the geometry out of band. This is a smoke-only hack and
#: it is why the mock refuses to run outside SMOKE.
_MOCK_MASK = {"m": None}


def mock_solver(box):
    """A geometry-DEPENDENT fake solver for smoking the SEARCH, not the physics.

    ``phase2_calibrate._synthetic_solver`` fabricates a record that ignores the
    geometry entirely, so every candidate scores the same and a search cannot be
    exercised with it at all. This one reads the mask, treats each contiguous
    run of metal in a column as a shunt open stub of that length, places its
    transmission zero at the measured D-1 law f = K/(L + dL) calibrated so that
    63.5 cells resonates at 5.25 GHz, and sums the stub admittances into
    S21 = 1/(1 + Y Z0/2). It reproduces the qualitative structure the real
    problem has — notch position set by length, depth by count, merging when
    notches are close — which is enough to exercise acceptance, block
    refinement, dedup, checkpoint and resume.

    It is NOT a solver. Every record it produces carries
    ``MOCK_NOT_PHYSICS: True``, the run tag is suffixed ``_MOCK``, and it
    refuses to run unless ``SMOKE=1``.
    """
    if not SMOKE:
        raise RuntimeError(
            "the mock solver is smoke-only: it is a transmission-line caricature, "
            "not a Maxwell solve, and no number it produces may be reported. "
            "Set SMOKE=1 if you are smoking the search machinery.")
    # D-1 law, from the calibration in the campaign record: 63.55 cells -> 5.25
    # GHz and 57.8 cells -> 5.775 GHz. Solve f = K/(L+dL) for (K, dL).
    l1, f1, l2, f2 = 63.55, 5.25e9, 57.8, 5.775e9
    dl = (l1 * f1 - l2 * f2) / (f2 - f1)
    kk = f1 * (l1 + dl)

    def run(sim, freqs_hz, num_periods):
        mask = _MOCK_MASK["m"]
        f = np.asarray(freqs_hz, dtype=np.float64)
        y = np.zeros(f.size, dtype=complex)
        if mask is not None:
            for name in fx.SIDES:
                m = np.asarray(mask[name], dtype=bool)
                side = box.side(name)
                for i in range(m.shape[0]):
                    col = m[i]
                    j = 0
                    while j < col.size:
                        if not col[j]:
                            j += 1
                            continue
                        j0 = j
                        while j < col.size and col[j]:
                            j += 1
                        # length measured OUTWARD from the trace-adjacent row
                        run_cells = j - j0
                        if side.name == "hi":
                            root = j0
                        else:
                            root = col.size - j
                        eff = run_cells + 0.35 * root      # unrooted -> detuned
                        f0 = kk / (eff + dl)
                        q = 30.0 + 6.0 * run_cells
                        y += 0.02 / (1.0 + 1j * q * (f / f0 - f0 / f))
        z0 = 50.0
        s21 = 1.0 / (1.0 + y * z0 / 2.0)
        s11 = 1.0 - s21
        db21 = 20 * np.log10(np.abs(s21) + 1e-30)
        db11 = 20 * np.log10(np.abs(s11) + 1e-30)
        mhz = [int(round(x / 1e6)) for x in f]
        return dict(
            MOCK_NOT_PHYSICS=True,
            num_periods=float(num_periods), n_freqs=int(f.size), n_steps=0,
            record_ns=float("nan"), dft_res_GHz=float("nan"), wall_s=0.0,
            freqs_GHz=[float(x) / 1e9 for x in f], freqs_MHz=mhz,
            s21_db=[float(x) for x in db21], s11_db=[float(x) for x in db11],
            f_min_GHz=float(f[int(np.argmin(db21))] / 1e9),
            depth_min_db=float(np.min(db21)),
            settling_worst_db=-120.0, settled=True,
            reliable_bins=[2 * int(f.size), 2 * int(f.size)],
            passivity_worst=0.0, passivity_projected=False,
            settling_db=[-120.0] * int(f.size),
            reliable=np.ones((2, f.size), dtype=bool).tolist(),
            passivity_correction=[0.0] * int(f.size),
        )
    return run


# ---------------------------------------------------------------------------
# 7. Checkpointing
# ---------------------------------------------------------------------------
def _ckpt_paths(run: str):
    return OUT / f"{run}_checkpoint.json", OUT / f"{run}_checkpoint.npz"


def save_checkpoint(run, ev: Evaluator, rng, search_state: dict, cfg: dict):
    """Full resume state: RNG stream, budget counters, dedup cache, search state.

    Written atomically (temp + replace) because a job killed mid-write would
    otherwise leave a checkpoint that resumes into nonsense.
    """
    pj, pn = _ckpt_paths(run)
    arrays = {}
    plain = {}
    for k, v in search_state.items():
        if isinstance(v, np.ndarray):
            arrays[f"s_{k}"] = v
        else:
            plain[k] = v
    if ev.best_mask is not None:
        for n in fx.SIDES:
            arrays[f"best_{n}"] = ev.best_mask[n]
    doc = {
        "cfg": cfg,
        "rng": rng.bit_generator.state,
        "solves": ev.solves, "evals": ev.evals, "rejected": ev.rejected,
        "best_key": [bool(ev.best_key[0]), float(ev.best_key[1]),
                     float(ev.best_key[2])] if ev.best_key != REJECT_KEY else None,
        "best_M": (None if ev.best_M != ev.best_M else ev.best_M),
        "best_sha": ev.best_sha, "best_solve": ev.best_solve,
        "improvement_solves": list(ev.improvement_solves),
        "cache": {k: [list(v[0]) if not np.isinf(v[0][1]) else None,
                      (None if v[1] != v[1] else v[1])]
                  for k, v in ev.cache.items()},
        "search": plain,
        "search_arrays": sorted(arrays),
        "wall_s": round(time.time() - ev.t0, 1),
    }
    tj = pj.with_name(pj.name + ".tmp")
    # np.savez_compressed APPENDS ".npz" unless the name already ends in it, so
    # the temp file must keep the suffix last or the replace() below chases a
    # file that was never written.
    tn = pn.with_name(pn.stem + ".tmp.npz")
    tj.write_text(json.dumps(doc, indent=2, default=str))
    np.savez_compressed(tn, **arrays)
    tj.replace(pj)
    tn.replace(pn)


def load_checkpoint(run, ev: Evaluator, rng, cfg: dict):
    pj, pn = _ckpt_paths(run)
    if not pj.exists():
        return None
    doc = json.loads(pj.read_text())
    old = doc.get("cfg", {})
    fatal = [k for k in ("heuristic", "periods", "grid", "blocks", "symmetry",
                         "dx_um", "n_freqs", "solver", "max_fill", "init",
                         "swarm")
             if old.get(k) != cfg.get(k)]
    if fatal:
        raise ValueError(
            f"--resume refuses: the checkpoint at {pj} was written with a "
            f"different configuration ({', '.join(fatal)} differ). Resuming "
            f"would splice two different experiments into one budget curve. "
            f"Use a new --run name.")
    rng.bit_generator.state = doc["rng"]
    ev.solves = int(doc["solves"])
    ev.evals = int(doc["evals"])
    ev.rejected.update(doc.get("rejected", {}))
    ev.best_M = (float("nan") if doc.get("best_M") is None else float(doc["best_M"]))
    ev.best_sha = doc.get("best_sha")
    ev.best_solve = int(doc.get("best_solve", -1))
    ev.improvement_solves = list(doc.get("improvement_solves", []))
    bk = doc.get("best_key")
    ev.best_key = REJECT_KEY if bk is None else (bool(bk[0]), float(bk[1]),
                                                 float(bk[2]))
    for k, v in doc.get("cache", {}).items():
        key = REJECT_KEY if v[0] is None else (bool(v[0][0]), float(v[0][1]),
                                               float(v[0][2]))
        ev.cache[k] = (key, float("nan") if v[1] is None else float(v[1]))
    arrays = dict(np.load(pn)) if pn.exists() else {}
    if all(f"best_{n}" in arrays for n in fx.SIDES):
        ev.best_mask = {n: arrays[f"best_{n}"].astype(np.uint8) for n in fx.SIDES}
    state = dict(doc.get("search", {}))
    for k, v in arrays.items():
        if k.startswith("s_"):
            state[k[2:]] = v
    print(f"[armC] RESUMED from {pj}: {ev.solves}/{cfg['budget']} solves "
          f"spent, best M = {ev.best_M:.3f}")
    return state


# ---------------------------------------------------------------------------
# 8. Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Arm C — binary heuristic at a pre-registered Maxwell-solve "
                    "budget.")
    ap.add_argument("--heuristic", choices=("dbs", "bpso"), required=True)
    ap.add_argument("--budget-solves", type=int, required=True,
                    help="HARD stop, counted in Maxwell solves. The empty-line "
                         "reference and the final verification are not charged.")
    ap.add_argument("--periods", type=float, default=45.0,
                    help="descent/search window; 45 is the measured window "
                         "where the two notches separate and the answer stops "
                         "moving (NOTE_stage0_window.md)")
    ap.add_argument("--verify-periods", type=float, default=90.0,
                    help="verification window for the FINAL best design on the "
                         "123-point scoring grid; 0 skips it")
    ap.add_argument("--grid", choices=("descent", "scoring"), default="descent")
    ap.add_argument("--run", default=None, help="output tag; defaults to armC_<heuristic>")
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=25)

    # reductions
    ap.add_argument("--blocks", type=int, nargs="+", default=[8, 4],
                    help="R1 coarse-to-fine block sizes, in cells, coarse "
                         "first. Default [8, 4] because one DBS sweep costs one "
                         "solve per free variable: 216 at B=8, 864 at B=4, "
                         "3 290 at B=2 and 13 160 at B=1. A level the budget "
                         "cannot sweep once contributes almost nothing, and the "
                         "driver prints the arithmetic.")
    ap.add_argument("--symmetry", choices=("none", "x", "y", "xy"),
                    default="none", help="R4; default none because both mirrors "
                                         "exclude the classical warm start")
    ap.add_argument("--init", choices=("classical", "empty", "random", "file"),
                    default="classical", help="R2")
    ap.add_argument("--init-file", default=None,
                    help="npz with 'lo' and 'hi' cell masks, for --init file")
    ap.add_argument("--init-fill", type=float, default=0.15,
                    help="fill fraction for --init random")
    ap.add_argument("--warm-from-armd", default=None,
                    help="arm D output dir; picks its best d3_* record as the "
                         "warm start (falls back to the D-1 calibrated lengths)")
    ap.add_argument("--warm-sep", type=int, default=40)
    ap.add_argument("--warm-llo", type=int, default=63)
    ap.add_argument("--warm-lhi", type=int, default=58)
    ap.add_argument("--warm-w", type=int, default=2)
    ap.add_argument("--warm-one-sided", action="store_true")
    ap.add_argument("--max-evals", type=int, default=None,
                    help="backstop on evaluations that spend no solve; default "
                         "20 x --budget-solves. Firing it truncates the curve "
                         "and says so.")
    ap.add_argument("--max-fill", type=float, default=0.60,
                    help="R3 early reject: fill fraction above this is a ground "
                         "plane over the line, not a filter")

    # dbs
    ap.add_argument("--shake-bits", type=int, default=6)
    # bpso
    ap.add_argument("--swarm", type=int, default=16)
    ap.add_argument("--w-hi", type=float, default=0.9)
    ap.add_argument("--w-lo", type=float, default=0.4)
    ap.add_argument("--c1", type=float, default=2.0)
    ap.add_argument("--c2", type=float, default=2.0)
    ap.add_argument("--v-max", type=float, default=4.0)
    ap.add_argument("--bpso-stall", type=int, default=4)
    ap.add_argument("--mut-p", type=float, default=0.05)

    ap.add_argument("--solver", choices=("fdtd", "mock"), default="fdtd")
    ap.add_argument("--allow-unsettled-empty", action="store_true",
                    help="smoke only; the result is NOT QUOTABLE")
    args = ap.parse_args()

    run = args.run or f"armC_{args.heuristic}"
    if args.solver == "mock":
        run += "_MOCK"

    freqs = _grid_hz(args.grid)
    fixture = fx.build_sim(freqs)
    grid = fixture.sim._build_grid()
    box = fx.design_box(grid)
    eps_eff = fx.EPS_EFF

    coders = [make_coder(box, b, args.symmetry) for b in args.blocks]
    print(f"[armC] grid={tuple(grid.shape)}  box lo={box.lo.nx}x{box.lo.ny} "
          f"hi={box.hi.nx}x{box.hi.ny} = {box.n_vars} cell variables  "
          f"smoke={SMOKE}")
    print(f"[armC] lambda/4 = {quarter_wave(F_LO, eps_eff)*1e3:.3f} / "
          f"{quarter_wave(F_HI, eps_eff)*1e3:.3f} mm")
    print(f"[armC] budget = {args.budget_solves} Maxwell solves "
          f"(= {args.budget_solves/2:.0f} gradient iterations at 1 forward + "
          f"1 backward)")
    for c in coders:
        sw = c.sweep_solves()
        note = ("" if sw <= args.budget_solves else
                "  << BUDGET CANNOT COMPLETE ONE SWEEP AT THIS LEVEL")
        print(f"[armC]   R1 block {c.block:>2d} cells -> {c.nbx} x {c.nby} "
              f"blocks/side, symmetry={c.symmetry} -> {c.n_vars:>6d} free "
              f"variables ({100.0*c.n_vars/box.n_vars:5.1f} % of the cell "
              f"count); one DBS sweep = {sw} solves = "
              f"{args.budget_solves/max(1,sw):.2f} x budget{note}")

    solver, pre_hook = None, None
    # The empty-reference cache key is (fixture geometry, window, freq grid) and
    # deliberately does NOT include which solver produced it, so a mock-solved
    # reference would land on the exact path a real one uses and be silently
    # reused by a production run. The mock therefore gets its own cache dir.
    cache_dir = CACHE
    if args.solver == "mock":
        solver = mock_solver(box)
        cache_dir = OUT.parent / "empty_ref_MOCK"
        cache_dir.mkdir(parents=True, exist_ok=True)

        def _set_mock_mask(mask):      # see mock_solver's docstring
            _MOCK_MASK["m"] = mask
        pre_hook = _set_mock_mask

    ev = Evaluator(box, freqs, args.periods, budget=args.budget_solves,
                   out=OUT, run=run, max_fill=args.max_fill,
                   cache_dir=cache_dir, solver=solver, pre_hook=pre_hook,
                   max_evals=args.max_evals,
                   require_settled_empty=not (args.allow_unsettled_empty or SMOKE),
                   verbose=True)

    # ---- initial design ---------------------------------------------------
    warm_src, warm_p = "n/a", {}
    if args.init == "classical":
        defaults = dict(sep_cells=args.warm_sep, l_lo_cells=args.warm_llo,
                        l_hi_cells=args.warm_lhi, w_cells=args.warm_w,
                        two_sided=not args.warm_one_sided)
        m0, warm_src, warm_p = classical_warm_start(
            box, Path(args.warm_from_armd) if args.warm_from_armd else None,
            defaults)
    elif args.init == "empty":
        m0 = box.empty_mask()
    elif args.init == "file":
        z = np.load(args.init_file)
        m0 = {n: z[n].astype(np.uint8) for n in fx.SIDES}
        warm_src = str(args.init_file)
    else:
        r0 = np.random.default_rng(args.seed)
        m0 = {n: (r0.random(box.side(n).shape) < args.init_fill).astype(np.uint8)
              for n in fx.SIDES}
        warm_src = f"random fill={args.init_fill}"
    g0 = _realized(m0, box)
    print(f"[armC] R2 init = {args.init} ({warm_src}); realized "
          f"{g0['fill_cells']} metal cells, fill {g0['fill_frac']*100:.2f} %")
    if g0["fill_frac"] > args.max_fill:
        raise SystemExit(
            f"[armC] REFUSING to start: the initial design fills "
            f"{g0['fill_frac']*100:.1f} % of the box, above the --max-fill "
            f"early-reject threshold of {args.max_fill*100:.0f} %. Every "
            f"single-bit neighbour of it would also be rejected without a "
            f"solve, so DBS would thrash and the budget curve would be "
            f"meaningless (measured: 6 294 rejected evaluations for 6 solves). "
            f"Lower --init-fill, or raise --max-fill deliberately and say so in "
            f"the record.")

    # Seed the coarsest level. The vote rule is chosen, not fixed:
    #
    #   * "majority" is the right default and is what the coarse-to-fine
    #     hand-off uses everywhere else.
    #   * but the D-1 calibrated stub is 2 cells wide and is a MINORITY of
    #     every 8x8 block it touches, so a majority vote quantises the entire
    #     classical warm start to an EMPTY box. For that case "any" is used
    #     instead: it keeps the stub's topology and fattens it to the block
    #     lattice. That is a different (wider, lower-Z) stub, so both the raw
    #     warm start and its quantisation are evaluated and both go in the
    #     trajectory -- the difference is the quantisation loss the R1
    #     reduction costs, and it belongs in the record rather than absorbed.
    #   * "any" must NOT be used on a dense init: a random 30 %-fill mask has a
    #     metal cell in nearly every 8x8 block, so "any" quantises it to a
    #     SOLID box, which the overfill early-reject then refuses along with
    #     every single-bit neighbour of it. Measured before this was adaptive:
    #     119 rejected evaluations for 1 solve.
    v_maj = coders[0].encode(m0, rule="majority")
    v_any = coders[0].encode(m0, rule="any")
    fill_any = float(np.mean([coders[0].decode(v_any)[n].mean()
                              for n in fx.SIDES]))
    if int(v_maj.sum()) == 0 < int(v_any.sum()) and fill_any <= args.max_fill:
        v0, seed_rule = v_any, "any"
        print(f"[armC]    (a majority vote would have quantised this init to an "
              f"EMPTY box at block {coders[0].block}; using rule='any')")
    else:
        v0, seed_rule = v_maj, "majority"
    m0q = coders[0].decode(v0)
    g0q = _realized(m0q, box)
    print(f"[armC]    quantised to block {coders[0].block} by rule="
          f"'{seed_rule}': {g0q['fill_cells']} metal cells, fill "
          f"{g0q['fill_frac']*100:.2f} % ({int(v0.sum())} of "
          f"{coders[0].n_vars} free variables set)")
    if g0q["fill_frac"] > args.max_fill:
        raise SystemExit(
            f"[armC] REFUSING to start: quantising the init onto the block "
            f"{coders[0].block} lattice fills {g0q['fill_frac']*100:.1f} % of "
            f"the box, above --max-fill {args.max_fill*100:.0f} %. The search "
            f"would start inside the early-reject region and thrash. Use a "
            f"finer first --blocks level or a sparser init.")

    cfg = dict(heuristic=args.heuristic, budget=args.budget_solves,
               periods=args.periods, grid=args.grid, blocks=list(args.blocks),
               symmetry=args.symmetry, init=args.init, seed=args.seed,
               dx_um=box.dx * 1e6, n_freqs=int(freqs.size),
               solver=args.solver, max_fill=args.max_fill,
               swarm=args.swarm, warm_source=warm_src, warm_params=warm_p,
               fixture_version=cal.FIXTURE_VERSION, smoke=SMOKE)

    rng = np.random.default_rng(args.seed)
    state = load_checkpoint(run, ev, rng, cfg) if args.resume else None
    if not args.resume:
        ev.traj_path.write_text("")     # a fresh run starts a fresh curve

    last = {"n": ev.solves}

    def checkpoint(search_state):
        if ev.solves - last["n"] >= args.checkpoint_every or ev.exhausted:
            save_checkpoint(run, ev, rng, search_state, cfg)
            last["n"] = ev.solves

    t_start = time.time()
    # The raw, un-quantised init is evaluated first and charged to the budget.
    # It is the incumbent this arm was handed, so it belongs in the trajectory
    # and in the best-so-far tracking even though the block lattice cannot hold
    # it; a reported "arm C best" that is worse than the design it started from
    # would otherwise be an artifact of R1 rather than a result.
    if not args.resume and not ev.exhausted:
        ev.phase, ev.block, ev.n_vars = "init_raw", 0, box.n_vars
        ev(m0, label="init_raw")

    if args.heuristic == "dbs":
        run_dbs(ev, coders, (state["v"] if state and "v" in state else v0), rng,
                shake_bits=args.shake_bits, state=state, checkpoint=checkpoint)
    else:
        run_bpso(ev, coders, v0, rng, swarm=args.swarm, w_hi=args.w_hi,
                 w_lo=args.w_lo, c1=args.c1, c2=args.c2, v_max=args.v_max,
                 stall=args.bpso_stall, mut_p=args.mut_p, state=state,
                 checkpoint=checkpoint)
    search_wall = time.time() - t_start

    # ---- best design, realized geometry, npz ------------------------------
    if ev.best_mask is None:
        print("[armC] no design was ever evaluated — budget too small")
        return 1
    np.savez_compressed(
        OUT / f"{run}_best.npz",
        **{n: ev.best_mask[n] for n in fx.SIDES},
        improvement_solves=np.asarray([s for s, _, _ in ev.improvements]),
        improvement_masks_lo=np.asarray([m["lo"] for _, _, m in ev.improvements]),
        improvement_masks_hi=np.asarray([m["hi"] for _, _, m in ev.improvements]))

    geom = _realized(ev.best_mask, box)
    # Only write the descent-window record if the ScoredDesign in hand really is
    # the best mask's. After a --resume it is not (the checkpoint stores the
    # mask and the score, not the ScoredDesign object), and writing a stale
    # record under the name "best" is exactly the kind of quiet mislabel this
    # campaign has already been burned by. The verification below re-solves the
    # actual best mask either way.
    stale = ev.best_scored is None or ev.best_scored_sha != ev.best_sha
    if stale:
        print("[armC] descent-window best record NOT written: the best design "
              "was carried in from a checkpoint, so no ScoredDesign for it "
              "exists in this process. The verification record below is solved "
              "from the actual best mask.")
    else:
        _record(f"{run}_best", ev.best_scored,
                dict(mode="armC", heuristic=args.heuristic, window="descent",
                     periods=args.periods, grid=args.grid,
                     budget_solves=args.budget_solves, solves_used=ev.solves,
                     mask_sha=ev.best_sha, best_at_solve=ev.best_solve,
                     search_wall_s=round(search_wall, 1),
                     realized=geom, cfg=cfg))

    # ---- verification: independent long window, scoring grid --------------
    verify, verify_valid = None, None
    if args.verify_periods and args.verify_periods > 0:
        vf = _grid_hz("scoring")
        print(f"[armC] verification: {args.verify_periods:g} periods, "
              f"{vf.size} points (NOT charged to the budget)")
        # The verification window needs its OWN empty reference. Solve it
        # first and hand it over explicitly: score_design would otherwise fetch
        # it after pre_hook had already been pointed at the design mask, and the
        # mock would then "solve" the empty line with the design in place.
        if pre_hook is not None:
            pre_hook(None)
        v_empty = cal.empty_reference(
            vf, args.verify_periods, cache_dir=cache_dir, solver=solver,
            verbose=True)
        if pre_hook is not None:
            pre_hook(ev.best_mask)
        vs = cal.score_design(
            ev.best_mask, freqs_hz=vf, num_periods=args.verify_periods,
            cache_dir=cache_dir, solver=solver, label=f"{run}:verify",
            empty=v_empty,
            require_settled_empty=not (args.allow_unsettled_empty or SMOKE),
            verbose=True)
        verify_valid = bool(vs.validity.ok)
        verify = _record(f"{run}_verify", vs,
                         dict(mode="armC_verify", heuristic=args.heuristic,
                              periods=args.verify_periods, grid="scoring",
                              budget_solves=args.budget_solves,
                              solves_used=ev.solves, mask_sha=ev.best_sha,
                              realized=geom, cfg=cfg))

    # ---- the curve, as a table --------------------------------------------
    curve = list(ev.improvement_solves)
    summary = dict(
        run=run, cfg=cfg,
        budget_solves=args.budget_solves, solves_used=ev.solves,
        evals=ev.evals, rejected=ev.rejected,
        stop_reason=ev.stop_reason, max_evals=ev.max_evals,
        budget_curve_truncated=bool(ev.stop_reason == "eval_cap"),
        cache_hit_rate=round(ev.rejected["duplicate"] / max(1, ev.evals), 4),
        best_M=(None if ev.best_M != ev.best_M else ev.best_M),
        best_at_solve=ev.best_solve, best_sha=ev.best_sha,
        improvement_solves=curve,
        improvement_masks_in_npz=[s for s, _, _ in ev.improvements],
        search_wall_s=round(search_wall, 1),
        wall_per_solve_s=round(search_wall / max(1, ev.solves), 2),
        realized=geom,
        verify_M=(None if verify is None else verify["result"]["M"]),
        verify_valid=verify_valid,
        trajectory=str(ev.traj_path),
        NOT_QUOTABLE=bool(SMOKE or args.solver == "mock"
                          or args.allow_unsettled_empty),
    )
    (OUT / f"{run}_summary.json").write_text(json.dumps(summary, indent=2,
                                                        default=str))
    print("\n" + "=" * 78)
    print(f"[armC] {run}: {ev.solves}/{args.budget_solves} solves, "
          f"{ev.evals} evaluations "
          f"({ev.rejected['duplicate']} dup, {ev.rejected['empty']} empty, "
          f"{ev.rejected['overfill']} overfill rejected without a solve)")
    if ev.stop_reason == "eval_cap":
        print("[armC] *** BUDGET CURVE TRUNCATED *** stopped on the evaluation "
              "cap, not the solve budget")
    print(f"[armC] best M = {ev.best_M:.3f} first reached at solve "
          f"{ev.best_solve}; improvements at solves {curve}")
    if verify is not None:
        print(f"[armC] verification ({args.verify_periods:g} periods, scoring "
              f"grid): M = {verify['result']['M']:.3f}")
    if summary["NOT_QUOTABLE"]:
        print("[armC] *** NOT QUOTABLE *** (smoke / mock solver / unsettled "
              "empty reference)")
    print(f"[armC] wrote {OUT}/{run}_{{trajectory.jsonl,best.json,best.npz,"
          f"summary.json}}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
