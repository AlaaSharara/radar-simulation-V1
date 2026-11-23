# Data directory

Generated files only. Keep raw exports out of Git unless they are small and intentionally versioned.

Recommended structure:

- `data/generated/scenes/` — randomized scene JSON snapshots.
- `data/generated/motion/` — labels and pose streams.
- `data/generated/rf/` — path sets and diagnostics.
- `data/generated/adc/` — selected ADC-cube captures.
- `data/generated/dsp/` — detections, point clouds, tracks and maps.
- `data/generated/exports/` — HDF5 or NPZ records with config hash and seed map.
