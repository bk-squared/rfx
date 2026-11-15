### Fixed — Lorentz fits keep resonances inside the measured frequency span (#PR)

- `fit_lorentz` bounds a free `eps_inf` at 1 and resonances inside the table, with weighted real/loss residuals and starts spread across the span. Tables without a frequency span raise `ValueError`.
- On a constant FR4 table (ε′ = 4.4, tan δ = 0.02, 1–40 GHz), worst tan δ error over 2–20 GHz falls from 48.74 / 30.01% to 10.55 / 2.65% at `n_poles=2 / 3`; worst ε′ error changes from 0.000288 / 0.512% to 1.585 / 1.506%.
- Predicted growth per step on 1 mm cubic cells at 0.99 of the vacuum Courant limit falls from 9.82 / 54.93 to 1; `run()` refused the previous fits.
- On one SciPy build, a last-bit input change moves fitted parameters by less than 0.001%; previously Δε moved by 97% and the low resonance by 82% (0.10 → 0.55 GHz). This does not promise identical poles across SciPy builds.
- Median fit time increases from 0.12 to 0.50 s on a single-resonance table and from 0.63 to 1.89 s on FR4 (five repeats; FR4 pools two- and three-pole fits).
