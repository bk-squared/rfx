### Changed — Refuse a traced mesh that would cut a conductor at the absorber (#PR)

A conductor whose declared bounds reach an absorbing face is now refused before
stepping when a mesh axis is traced. The message names the faces and asks for
that axis's profile as a concrete NumPy array for the run.
