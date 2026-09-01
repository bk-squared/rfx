"""#812 falsifier driver for the cv03 dispersion re-gate.

A re-gate that only makes a case pass is cosmetic. This script runs the two
falsifiers pre-declared in
``docs/design_notes/issue812_cv03_dispersion_regate_predeclaration.md`` section
5 and prints, for each, whether the NEW gate (G1, analytic dispersion) fires and
whether the OLD gate (G2, the flux identity) still passes. The contrast is the
finding: G2 cannot see any of these defects.

Each falsifier is a single textual edit applied to a COPY of the crossval
script in a temp directory; the case itself is never modified. F1 reproduces the
audit's own probe verbatim -- it edits the same ``eps_wg = 12.0`` line the audit
swept. F2 touches no declared constant at all: it moves the guide's lower face
up by one cell in the geometry-construction line, so the script still declares
``eps_wg = 12.0`` and ``wg_width = 1.0`` while building a 9-cell guide. F2
exists to refute the reading that G1 merely compares two literals.

Run:
    PYTHONPATH=<repo> python scripts/diagnostics/cv03_flux/regate_falsifiers.py

Runtime: ~10 s per case on CPU, 5 cases.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CASE = os.path.join(REPO, "validation", "crossval", "03_straight_waveguide_flux.py")

# (label, what it models, [(exact old text, exact new text), ...])
FALSIFIERS = [
    ("baseline", "unmodified case (criterion A)", []),
    ("F1 eps=11", "audit sweep point 1 -- the case's own eps_wg line",
     [("eps_wg = 12.0", "eps_wg = 11.0")]),
    ("F1 eps=10", "audit sweep point 2",
     [("eps_wg = 12.0", "eps_wg = 10.0")]),
    ("F1 eps=8", "audit sweep endpoint (its measured n_eff shift: -23.4%)",
     [("eps_wg = 12.0", "eps_wg = 8.0")]),
    # F2: a 9-cell guide built CONSISTENTLY -- the source span follows the
    # narrowed core, so every source still sits in the guide and the only
    # thing that changed is the guide itself. No declared constant moves:
    # the script still says eps_wg = 12.0 and wg_width = 1.0.
    ("F2 d=0.9a", "guide one cell narrow, built consistently; NO constant changed",
     [("wg_y_hi = (OFFSET_Y + wg_width / 2) * a",
       "wg_y_hi = (OFFSET_Y + wg_width / 2) * a - dx"),
      ("for i in range(int(wg_width * resolution)):",
       "for i in range(int(wg_width * resolution) - 1):")]),
    # F2b: the same width error introduced by ONE careless edit, which also
    # pushes the topmost source out of the core. Reported to keep the two
    # mechanisms distinguishable -- here G2 fires too, but on the stray
    # source, not on the width.
    ("F2b 1-edit", "same width error, single careless edit (stray source)",
     [("wg_y_lo = (OFFSET_Y - wg_width / 2) * a",
       "wg_y_lo = (OFFSET_Y - wg_width / 2) * a + dx")]),
    # F3: a solver-flag regression. Reported, not required: whether it moves
    # n_eff at all is a measurement, and this guide is grid-aligned.
    ("F3 nosubpix", "subpixel smoothing disabled (reported, not required)",
     [("res_rfx = sim_rfx.run(n_steps=n_steps, subpixel_smoothing=True)",
       "res_rfx = sim_rfx.run(n_steps=n_steps, subpixel_smoothing=False)")]),
]

_G1 = re.compile(r"max \|n_eff_rfx/n_eff_analytic - 1\| over band: "
                 r"\s*([0-9.]+)%.*?(PASS|FAIL)", re.S)
_G1_RESID = re.compile(r"two-wave fit residual, max over band: ([0-9.]+)\s+(PASS|FAIL)")
_G2 = re.compile(r"rfx band-mean T \[[^\]]*\]:\s*([0-9.]+)\s+(PASS|FAIL)")
_REASON = re.compile(r"^\s+reason: (.*)$", re.M)


def run_one(label, edits, workdir):
    src = open(CASE).read()
    for old, new in edits:
        # The recipe block declares RECIPE_EPS_WG separately; only the
        # simulation parameter must move, exactly as the audit moved it.
        assert src.count(old) == 1, f"{label}: pattern is not unique: {old!r}"
        src = src.replace(old, new)
    path = os.path.join(workdir, "case.py")
    with open(path, "w") as fh:
        fh.write(src)
    env = dict(os.environ, PYTHONPATH=REPO, MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, path], capture_output=True,
                          text=True, env=env, cwd=workdir)
    return proc.returncode, proc.stdout + proc.stderr


def main():
    rows = []
    for label, why, edits in FALSIFIERS:
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(os.path.join(REPO, "validation", "crossval",
                                         "comparators"),
                            os.path.join(td, "comparators"))
            rc, out = run_one(label, edits, td)
        g1 = _G1.search(out)
        g1r = _G1_RESID.search(out)
        g2 = _G2.search(out)
        reason = _REASON.search(out)
        rows.append((label, why, rc,
                     g1.group(1) if g1 else "?", g1.group(2) if g1 else "?",
                     g1r.group(1) if g1r else "?",
                     g2.group(1) if g2 else "?", g2.group(2) if g2 else "?",
                     reason.group(1).strip() if reason else ""))
        print(f"[{label}] exit={rc}")

    print()
    print("=" * 108)
    print(f"{'case':<12} {'exit':>4}  {'G1 dev':>8} {'G1':>5} {'resid':>7}   "
          f"{'G2 <T>':>7} {'G2':>5}   what it models")
    print("-" * 108)
    for label, why, rc, dev, v1, resid, t, v2, reason in rows:
        print(f"{label:<12} {rc:>4}  {dev:>7}% {v1:>5} {resid:>7}   "
              f"{t:>7} {v2:>5}   {why}")
    print("=" * 108)
    for label, why, rc, dev, v1, resid, t, v2, reason in rows:
        if reason:
            print(f"{label}: G1 message -> {reason}")


if __name__ == "__main__":
    main()
