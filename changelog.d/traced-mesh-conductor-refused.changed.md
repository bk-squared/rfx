### Changed — Refuse a traced mesh that would cut a conductor at the absorber (#PR)

A conductor whose declared extent reaches or ends within one cell of an absorbing
face is now refused before stepping when a mesh axis is traced. Each face uses
its declared edge cell, including windowed nominal profiles. A traced axis with
an absorber needs `nominal_profiles` for any conductor with finite declared extent
on that axis, even one clear of every face. The message asks for the missing
nominal profile or a concrete NumPy profile for the run; a conductor within an
edge cell still requires the concrete profile.
