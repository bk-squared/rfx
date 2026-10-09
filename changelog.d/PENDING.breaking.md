Lossy thin conductors now act on tangential electric edges of their declared node
plane. Results of every lossy-film model change. There is no conduction across
the film, and it no longer changes a cell's permittivity.
Lossy films with `eps_r != 1`, on a PMC face, or normal to a 2-D model's invariant
axis are refused. An f0 sheet sharing active edges with a lossy film and a film
edge inside a design box are also refused.
A `sigma_override` or global sigma sweep replaces volume conductivity while
leaving a declared film in place. UPML and CPML now solve the same film.
the preflight finding dc_film_half_sheet_error (added in #1575) is removed with its cause
