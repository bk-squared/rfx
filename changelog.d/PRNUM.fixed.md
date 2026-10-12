### Fixed — Close finite flux boxes on both meshes (#PRNUM)

- Finite flux boxes now return the same net flux as their position and extent change around a source, using the same rule on uniform and graded meshes.
- In the archived 1 mm-cell measurement, an asymmetric box was 0.8% off and boxes closed by a magnetic face were 1.8–4.7% low; centred boxes were unaffected.
- A plane on a magnetic face now reads zero.
