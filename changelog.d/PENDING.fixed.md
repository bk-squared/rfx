On a graded mesh, `forward(pinned_sheet_layout=...)` keeps lumped RLC element
edges free of switched sheet metal. Folded and solved elements now run instead
of being refused. A `ConfigurationWarning` names each released edge when its
layout value is nonzero or traced; concrete zero values produce no release warning.
