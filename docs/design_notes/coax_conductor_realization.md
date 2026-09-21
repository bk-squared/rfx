# How the coax lanes realize their conductors, and what the old way cost

Status: the record behind the change that moved the coaxial pin and outer
conductor from `sigma = PEC_SIGMA` per node to PEC edge masks. Every number here
was measured before the product code moved, by two pre-declared diagnostics whose
readings were written down before they ran (branch `meas/coax-chain-battery`,
`scripts/diagnostics/coax_shell_seal_diagnostic.py`).

## The symptom

The reduced v2.0 chain battery measured, on the through line, a fitted phase
constant well above `omega sqrt(eps_r) / c` and a column power well below one,
both shrinking with the mesh:

| annulus cells | `beta_fit / beta_analytic` | max column power | \|S21\| at 4 GHz | \|S21\| at 12 GHz |
|---|---|---|---|---|
| 4 | 1.1833 | 0.8786 | 0.9368 | 0.5956 |
| 6 | 1.1267 | 0.9929 | 0.9955 | 0.9631 |
| 9 | 1.0791 | 0.9948 | 0.9961 | 0.9278 |

Shrinking with the mesh made it look like ordinary under-resolution, and the
lane's own recommendation — "about four or more annulus cells" — was written
from it. It is not under-resolution. **A homogeneously filled PEC-bounded coax
carries TEM at `beta = omega sqrt(eps)/c` whatever the staircase does to the
cross-section**: the staircase moves `Z_TEM`, not `beta`.

Two independent estimates agreed, so it was not the extractor's fit: the
matrix-pencil `beta` and one from the slope of the unwrapped `S21` phase over the
distance between the reference planes, which never touches that fit, differed by
less than 0.6 % in every arm. The probe planes the extractor assigns matched the
grid's own node coordinates to 0.0000 nm, so it was not a position scale either.

## What it was not: the open ring

The first hypothesis was a leak. As shipped at 4 annulus cells the outer
conductor is **ten fragments touching corner to corner**: the non-conductor
region is 2 components under 4-connectivity but ONE under 8-connectivity, and in
the Yee lattice the tangential E edge runs through every corner contact. (A
4-connected count alone reports that ring as sealed, which is why both are taken.)

Sealing it — three different geometries, all still `sigma` — fixed the power and
left `beta` alone:

| arm | `beta` pencil | `beta` S21 phase | column power |
|---|---|---|---|
| as shipped | 1.1867 | 1.1795 | 0.8786 |
| wall 3 cells thick | 1.2347 | 1.2376 | 0.9982 |
| everything outside the wall solid | 1.2347 | 1.2376 | 0.9980 |
| wall at the declared `b` | 1.1770 | 1.1793 | 1.0001 |

## What it was: the realization

`rfx/core/yee.py` builds `ca`/`cb` from `materials.sigma` and applies them to
`ex`, `ey` and `ez` at the same node index. A sigma-stamped conductor cell
therefore damps exactly its three PLUS-side edges; the edges entering it from the
minus side belong to the neighbouring dielectric node and stay live. The repo's
own volume conductors do not work that way — `realized_pec_edge_masks` is "the
one function that turns conductor geometry into PEC E edges" and puts walls on
BOTH faces with every normal edge between them shorted.

Realizing the same cells that way, changing nothing else:

| arm | geometry | `beta` pencil | `beta` S21 phase | `eps_eff` | column power |
|---|---|---|---|---|---|
| 4, 4 cells | wall at declared `b` | 1.0018 | 1.0000 | 2.1015 | 0.9969 |
| 4, 6 cells | wall at declared `b` | 1.0050 | 0.9992 | 2.1031 | 0.9980 |
| 4, 9 cells | wall at declared `b` | 1.0068 | 0.9986 | 2.1038 | 1.0071 |
| 5, 4 cells | **as shipped** | 1.0008 | 1.0004 | 2.1010 | 0.9998 |

Arm 5 is the control that fixes the attribution: it keeps the as-shipped
geometry, ring still corner-touching and still 8-connected across, and changes
only the realization. Both defects vanish. So the corner contact causes neither;
the realization causes both, and the lattice ownership rule shorts the edges
around a corner contact so the ring is electrically closed whatever the cell mask
looks like.

`Z0` from the 25 and 100 ohm loads under the edge realization: **48.614** and
**48.564 ohm** against the analytic `Z_TEM` on the declared radii, **48.591** —
0.05 % and 0.06 %. Both line constants come back to the declared geometry.

## The second change: the wall's inner radius

`stamp_coaxial_line` set `shell_thickness = min(dz, (b - a)/2)` and put the
wall's inner face at `b - shell_thickness`, so the dielectric annulus — and
hence the line's own characteristic impedance — moved with the mesh:

| dx [um] | annulus cells | dielectric reaches [um] | `Z_TEM` on realized radii [ohm] |
|---|---|---|---|
| 374.74 | 3.79 | 1640.7 | 40.5 |
| 355.00 | 4.00 | 1700.0 | 40.74 |
| 236.67 | 6.00 | 1818.3 | 43.53 |
| 157.78 | 9.00 | 1897.2 | 45.29 |

Against a declared 48.591 ohm. A dx ladder over that geometry refines the LINE as
well as the grid, which is not the convergence study the chain-closure contract's
dx-ladder guard asks for. The wall's inner face is now the declared outer radius
and its thickness is fixed in metres (`SHELL_THICKNESS_M`), so the dielectric
reaches 2041.6 / 2051.5 / 2052.2 / 2052.4 um across that same ladder and the
impedance term the wall contributes is within 0.6 % of the declared value at
every cell size.

## What is still open

The **pin**'s rasterized radius is unchanged by this and is not monotone in dx
(519.7, 622.3, 560.6, 631.6 um at 2, 4, 6 and 9 annulus cells). Feeding it into
a `Z_TEM` estimate gives up to 16.5 % — but the measured `Z0` above is within
0.06 % of the declared value, so that estimate is a property of the proxy (an
outermost cell CENTRE understates where the edge realization puts the wall) and
not of the line. It is recorded here rather than gated.
