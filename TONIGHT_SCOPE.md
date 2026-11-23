# Tonight's build — keep the first milestone achievable

## Must work before stopping

1. The default 4 × 5 × 2.7 m room loads from `scene.json`, with furniture, openings and layered surface metadata.
2. The person is a visible, rigged human avatar, not a rendered skeleton, and can enter, walk, sit, stand, lie/sleep and perform at least one contact-aware fall.
3. Different seeds produce visibly different movement sequences. A normal no-fall routine and a fall demo are both available.
4. The RF module produces geometrically valid direct and reflected paths and blocks a path when furniture/body geometry occludes it.
5. The radar module generates the configured TDM-MIMO complex samples and includes at least thermal noise plus one or more impairment toggles.
6. The DSP module produces a real range-Doppler result and radar-derived point cloud from the ADC cube.
7. The four UI panels show the same frame/time and the Scene panel contains the full human avatar.

## Defer if time is running out

- Higher-order reflections beyond the first few paths.
- Sophisticated diffuse scattering or diffraction.
- MUSIC/Capon if the conventional beamforming baseline is not stable.
- Breathing extraction and learned fall classification.
- Dense mesh reconstruction. The 12-channel array is sparse; show point cloud/occupancy with uncertainty instead.
- Large Monte Carlo sweeps; start with fixed seeds and a small smoke test.

## Never defer these safeguards

- Keep actual physics outputs separate from ground-truth coordinates.
- Never replace failed RF/DSP with fabricated point clouds or fake heatmaps.
- Never label the geometric-optics ray panel as a full-wave EM solver.
- Never describe simulated detection scores as physical-hardware accuracy.
- Save the seed, config hash, mode and test log for every demo.
