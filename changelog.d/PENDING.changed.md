Under the default `snap="strict"`, a sheet or PEC Box is now refused when
its solved spacing to an electric domain wall (including the PEC floor or
lid) differs from its drawn spacing by more than 1%. A drawn contact that
is solved open is also refused. Absorbing and magnetic faces are excluded.
Use mesh nodes on the planes and a cell size dividing the domain extent,
or set `snap="declared"` to retain the previous warning behavior.
