### Changed — the optimisation loop gains internal MMA and projected-Adam optimisers, not yet offered to users (#PR)

- An internal implementation of Svanberg's method of moving asymptotes (MMA, 2007 note) and a projected Adam now run through the same ask/tell loop as Adam; both stay internal until comparison runs decide whether `optimize()` offers them.
- Adam results remain unchanged bit for bit.
