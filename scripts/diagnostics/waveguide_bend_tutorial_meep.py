"""Reproduce-gate producer for validation/crossval/01_waveguide_bend.py.

Runs Meep's own upstream ``bend-flux.py`` tutorial (``python/examples/
bend-flux.py``, ``NanoComp/meep``), UNMODIFIED -- executed verbatim via
``exec()``, not hand-ported -- and records its own output as cv01's
reproduce-gate anchor.

Why unmodified execution, not a port: cv01's OWN existing Meep comparator
leg (``validation/crossval/01_waveguide_bend.py:187-200``) is a hand-port
of this same tutorial and diverges from it in six of ten parameters (cell
size/aspect ratio, waveguide-center placement, source position, nfreq,
flux-region width, and the two-run subtraction algorithm). A transcription
can diverge exactly that way; running the tutorial's own file avoids
reintroducing the same failure mode one layer down.

What the tutorial does NOT publish: its own docs page
(``doc/docs/Python_Tutorials/Basics.md``, "Transmittance Spectrum of a
Waveguide Bend") ends in ``plt.show()`` -- a plot
(``doc/docs/images/Tut-bend-flux.png``), not a printed transmittance value.
So this reproduce-gate cannot check against a published number the way
``validation/crossval/20_msl_phase_referee.py``'s Stage A or
``validation/crossval/21_coax_two_port_referee.py``'s own reproduce-gate do
(both check against a number their own tutorial DOES publish) -- this is a
WEAKER anchor: it records "this Meep, on this exact upstream script,
produced this", which makes future drift attributable, but does not
validate against an external ground truth.

Checks this script performs:
  1. Vendored-file integrity: ``validation/crossval/
     _01_waveguide_bend_upstream/bend-flux.py``'s sha256 (and git blob sha)
     must match what is recorded below and in that directory's
     ``PROVENANCE.md`` -- refuses to run a drifted copy.
  2. PASSIVITY (the real, non-tautological form of "energy conservation").
     NOTE on a proposal correction: asserting ``R + T + loss == 1``
     verbatim, as first proposed, would be a TAUTOLOGY -- the tutorial
     script never measures loss independently, it DEFINES
     ``loss := 1 - R - T`` for its own plot. That form would check nothing
     but this wrapper's own arithmetic. The genuine physical constraint,
     gated here instead, is ``R(f) >= 0``, ``T(f) >= 0``, and
     ``R(f) + T(f) <= 1`` at every frequency (small floating-point
     tolerance) -- a broken run (wrong normalization, a sign error, PML
     leakage into a flux plane) CAN violate this; a correct one cannot.
  3. Qualitative shape (weak, explicitly not a validation): T(f) has an
     interior local minimum (not at either band edge) somewhere in the
     swept band, matching the dip-then-rise shape visible in the published
     plot; R stays within a generous, eyeballed-off-the-plot range (not
     pinned to a value, since none is published).

Usage (VESSL-only -- Meep is not importable in CI/local dev here)::

    python scripts/diagnostics/waveguide_bend_tutorial_meep.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VENDORED_PATH = os.path.join(
    REPO_ROOT, "validation", "crossval", "_01_waveguide_bend_upstream", "bend-flux.py"
)
EXPECTED_GIT_BLOB_SHA = "f56ab6492a3cc55ebc1fc0c682c4981508c51955"
EXPECTED_SHA256 = "1b976439cbdf0695e7f79f7df4299287a08637742100093412fc37c61c414786"

RESULT_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "waveguide_bend_meep_tutorial")
os.makedirs(RESULT_DIR, exist_ok=True)


def _git_blob_sha(data: bytes) -> str:
    """Reproduce `git hash-object`'s blob sha without shelling out to git,
    so this integrity check works even outside a git checkout."""
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _verify_vendored_file() -> str:
    if not os.path.isfile(VENDORED_PATH):
        raise FileNotFoundError(
            f"vendored upstream tutorial missing: {VENDORED_PATH} -- see "
            "validation/crossval/_01_waveguide_bend_upstream/PROVENANCE.md"
        )
    data = open(VENDORED_PATH, "rb").read()
    got_sha256 = hashlib.sha256(data).hexdigest()
    got_blob = _git_blob_sha(data)
    if got_sha256 != EXPECTED_SHA256 or got_blob != EXPECTED_GIT_BLOB_SHA:
        raise RuntimeError(
            f"REFUSING to run: {VENDORED_PATH} has drifted from its recorded "
            f"provenance. Expected sha256={EXPECTED_SHA256} "
            f"git_blob={EXPECTED_GIT_BLOB_SHA}, got sha256={got_sha256} "
            f"git_blob={got_blob}. If Meep's own tutorial changed upstream, "
            "re-fetch it and update both this script and PROVENANCE.md "
            "deliberately -- do not hand-edit the vendored copy."
        )
    return data.decode("utf-8")


def main() -> int:
    t0 = time.time()
    source = _verify_vendored_file()
    print(f"Vendored bend-flux.py verified: sha256={EXPECTED_SHA256[:12]}... "
          f"git_blob={EXPECTED_GIT_BLOB_SHA}")

    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import meep as mp
    except ImportError as exc:
        print(f"[SKIP] Meep not importable ({exc}) -- this script is "
              "VESSL-only (source-built Meep). Exiting 2.", file=sys.stderr)
        raise SystemExit(2)

    print("Running Meep's own bend-flux.py, unmodified, via exec()...")
    # Deliberately executing the vendored, integrity-checked upstream
    # tutorial verbatim -- this IS the point (see module docstring: run it,
    # do not port it). namespace afterward holds the tutorial's own
    # module-level wl/Rs/Ts/nfreq, which the tutorial itself never saves to
    # disk (it only plots them).
    namespace: dict = {"__name__": "__meep_bend_flux_tutorial__", "__file__": VENDORED_PATH}
    code = compile(source, VENDORED_PATH, "exec")
    exec(code, namespace)

    wl = np.asarray(namespace["wl"], dtype=float)
    Rs = np.asarray(namespace["Rs"], dtype=float)
    Ts = np.asarray(namespace["Ts"], dtype=float)
    nfreq = int(namespace["nfreq"])
    assert wl.shape == Rs.shape == Ts.shape == (nfreq,), (
        f"unexpected shapes: wl={wl.shape} Rs={Rs.shape} Ts={Ts.shape} nfreq={nfreq}"
    )
    loss = 1.0 - Rs - Ts  # NOT independently measured -- see module docstring.
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s, {nfreq} frequency points, "
          f"wavelength {float(wl.min()):.2f}-{float(wl.max()):.2f} um")

    # ---- PASSIVITY (the real check; see module docstring for why
    # R+T+loss==1 alone is a tautology and is not gated on here) ----
    tol = 1e-6
    r_ok = bool(np.all(Rs >= -tol))
    t_ok = bool(np.all(Ts >= -tol))
    sum_ok = bool(np.all(Rs + Ts <= 1.0 + tol))
    passivity_ok = r_ok and t_ok and sum_ok
    print(f"  passivity: R>=0 {r_ok}, T>=0 {t_ok}, R+T<=1 {sum_ok} "
          f"(max R+T={float(np.max(Rs + Ts)):.6f}, min R={float(np.min(Rs)):.6f}, "
          f"min T={float(np.min(Ts)):.6f}) -> {'PASS' if passivity_ok else 'FAIL'}")

    # ---- Qualitative shape (weak -- see module docstring) ----
    i_min = int(np.argmin(Ts))
    t_min_interior = bool(0 < i_min < nfreq - 1)
    r_range = float(np.max(Rs) - np.min(Rs))
    # Bound is generous on purpose: eyeballed off the published plot
    # (Tut-bend-flux.png), where reflectance visually spans roughly
    # 0.15-0.27 over the full 5-10um band. This is a check against a
    # grossly wrong run, not a pin -- see module docstring.
    r_flat_ok = bool(r_range <= 0.20)
    shape_ok = t_min_interior and r_flat_ok
    print(f"  shape: T has interior local minimum at wl={float(wl[i_min]):.3f}um "
          f"(index {i_min}/{nfreq}) {t_min_interior}, R range {r_range:.4f} "
          f"<=0.20 {r_flat_ok} -> {'PASS' if shape_ok else 'FAIL (weak check)'}")

    out = dict(
        producer="scripts/diagnostics/waveguide_bend_tutorial_meep.py",
        vendored_file="validation/crossval/_01_waveguide_bend_upstream/bend-flux.py",
        vendored_file_sha256=EXPECTED_SHA256,
        vendored_file_git_blob_sha=EXPECTED_GIT_BLOB_SHA,
        upstream_repo="NanoComp/meep",
        upstream_path="python/examples/bend-flux.py",
        published_number=(
            "NONE -- doc/docs/Python_Tutorials/Basics.md ends in plt.show(), "
            "the only artifact is doc/docs/images/Tut-bend-flux.png (a plot). "
            "See this case's REPRODUCE_GATE_RECORD for what this "
            "reproduce-gate checks instead and why it is weaker than "
            "cv20's/cv21's."
        ),
        meep_version=getattr(mp, "__version__", None),
        wall_s=round(elapsed, 1),
        wl_um=[round(float(v), 6) for v in wl],
        reflectance=[round(float(v), 8) for v in Rs],
        transmittance=[round(float(v), 8) for v in Ts],
        loss=[round(float(v), 8) for v in loss],
        passivity_ok=passivity_ok,
        passivity_detail=dict(
            r_ok=r_ok, t_ok=t_ok, sum_ok=sum_ok,
            max_r_plus_t=float(np.max(Rs + Ts)),
            min_r=float(np.min(Rs)), min_t=float(np.min(Ts)),
        ),
        shape_ok=shape_ok,
        shape_detail=dict(
            t_min_interior=t_min_interior, wl_at_t_min_um=float(wl[i_min]),
            r_range=r_range, r_range_bound=0.20,
        ),
    )
    path = os.path.join(RESULT_DIR, "bend_flux_meep_tutorial.json")
    with open(path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"WROTE {path} (wall {out['wall_s']}s)")
    overall_ok = passivity_ok and shape_ok
    print(f"\noverall: {'PASS' if overall_ok else 'FAIL'} "
          f"(passivity={passivity_ok}, shape={shape_ok})")
    return 0 if passivity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
