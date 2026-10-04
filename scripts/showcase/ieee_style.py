"""IEEE-publication matplotlib style for the showcase engineering figures.

Provenance: a copy of the lab's pubstyle template (figure standard: SciencePlots
science+ieee+no-latex, 400 dpi, no titles), ported 2026-10-05 for issue 1359.

Usage:
    from ieee_style import use_pubstyle, COL_W, DCOL_W   # module name = the repo's copy
    use_pubstyle()
    fig, ax = plt.subplots(figsize=(COL_W, COL_W*0.72))
    ...
    fig.savefig(path)            # DPI + tight bbox come from rcParams

Base style is SciencePlots (``science`` + ``ieee``), which gives Times serif,
thin spines, inward minor ticks on all four sides, and colour-blind-safe
defaults. We use the ``no-latex`` variant (mathtext, not full usetex) so the
module carries no LaTeX/dvipng dependency. On top of that base we keep
these conventions:

Conventions (match an IEEEtran double-column manuscript):
  * NO figure/axes titles -- the LaTeX caption carries the description.
  * mathtext labels for symbols (e.g. $|S_{11}|$, $\\theta$, $\\sigma$).
  * physical units on axes (mm, GHz, dB, deg) -- never internal grid units.
  * tight bounding box; layout owned by constrained_layout (set per figure).
  * colormaps: viridis (scalar magnitude), magma (fields), RdBu_r (signed).
Figures are sized to the single-column width COL_W (3.4 in) or DCOL_W (7.16 in).

Requires: ``pip install SciencePlots`` (registers the 'science'/'ieee' styles).
"""
import matplotlib
matplotlib.use("Agg")  # headless-safe; deliberately overrides any interactive backend
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  -- importing registers the SciencePlots styles

COL_W = 3.4    # IEEE single-column width (in)
DCOL_W = 7.16  # IEEE double-column (full text) width (in)


def use_pubstyle():
    # SciencePlots base look (Times serif, thin spines, inward minor ticks,
    # accessible cycle). no-latex -> mathtext: no LaTeX/dvipng dependency.
    plt.style.use(["science", "ieee", "no-latex"])
    # lineage overrides on top of the SciencePlots base.
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.autolayout": False,      # composites own layout via constrained_layout
        "font.family": "serif",
        "mathtext.default": "regular",
        # readable, consistent sizes (explicit per-call sizes still win).
        # Full-width figure* panels are placed at ~1:1, so these pt values are
        # the on-page sizes; kept generous so multi-panel composites stay as
        # legible as the upscaled single-column schematics.
        "font.size": 10,
        # titles stay banned (docstring; the pre-build check enforces it) --
        # this only keeps a stray title from rendering at an outsized default.
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.6,
        # keep the submitted-lineage colormap discipline (unchanged on purpose)
        "image.cmap": "viridis",
        # grid is applied per-axes in the figure code, not globally
        "axes.grid": False,
    })

# ported 2026-10-05
