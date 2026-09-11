# Completed patch integration run

Run 369367260362 tested source e3eb58ab80bb71dcfe54cc62c7e0a3288c3c41a8.
All eight tests passed. The two NPZs are the actual ten-probe estimator inputs.
`run.yaml` and `driver.py` are the as-run provenance, not a reusable output
location: recreate the pinned checkout and select a fresh output directory
before another submission. `files.json` hashes the captured run outputs.
`paired-estimator-replay.json` compares both estimators on those same inputs.
