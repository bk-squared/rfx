# Agent publication boundary

`scripts/build_public_docs_bundle.py::PUBLIC_AGENT_PAGES` is the explicit
allowlist. These pages ship as `/markdown/agent/<name>.md`, are indexed under
“For coding agents” in `llms.txt`, and appear in `llms-full.txt`.
New agent files do not publish automatically. Numerical solver support paths
are distinct from the lab's session/ownership lanes.

Excluded sources:

- `working-on-rfx.mdx`: internal task ownership, review and session coordination.
- `agent-runbook.mdx`: internal commands, ownership lanes and VESSL configuration.
- `repo-map.mdx`: names a VESSL run and internal PI coordination.
- `recipe-waveguide-sparams.mdx`: names VESSL run evidence; use the public
  sources/ports guide and support matrix instead.
- `gpu-throughput.mdx`: lab hardware benchmark measurements.
- `PUBLICATION.md`: this publication-maintenance note.

README.md links to `working-on-rfx.mdx` and `repo-map.mdx` (line 103 at the
starting commit). No docs/public page links to these excluded sources.
The starting overview links to working-on-rfx and repo-map; the R/T recipe
links to recipe-waveguide-sparams. Those agent links were replaced with published guidance. This file is not a public alternate.
