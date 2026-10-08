### Fixed — Distributed forward material sweeps work with three devices (#PENDING)

- `vmap` over distributed forward with a local epsilon override now keeps the batch axis unpartitioned, including batch size 2 on 3 devices.
- Slab pole staging, rank-helper placement, and retained splitter contracts: no result changes.
