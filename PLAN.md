# PLAN.md — Project rules and implementation plan

## 1. Project objective

Build a reproducible, physically grounded simulation of a low-cost indoor 60 GHz FMCW/TDM-MIMO radar observing one person in a furnished room. The simulation must generate a changing scene, compute propagation and received radar signals, process the signals, and display synchronized ground truth and radar-derived outputs.

The purpose of tonight's build is a working end-to-end simulation prototype. It is not a claim of perfect physics or validated physical-radar accuracy.

## 2. Primary user-visible result

A browser dashboard divided into four panels:

1. **Scene / ground truth:** 3D room, furniture and a fully visible, adult male-presenting human avatar. It must look like a human character, not a skeleton. Rig bones and the 17-joint motion representation are hidden. Show the action label and ground-truth pose metadata only as optional diagnostic overlays.
2. **RF propagation:** animated wavefront/ray visualization including LOS, reflections, occlusion and path-strength changes. This panel represents the implemented geometric-optics model; it must not imply a full electromagnetic field solution.
3. **Radar point cloud:** detections calculated from the simulated ADC/DSP chain, coloured by radial velocity or SNR and annotated with track IDs. A separate optional ground-truth overlay must be switchable and clearly labelled.
4. **Signal and event diagnostics:** range-Doppler heatmap, range-angle view, activity/event timeline, estimated target height/vertical velocity, track confidence, and a rule-based fall alert.

All panels must display the same `frame_id` and `sim_time_s`. Do not animate the avatar independently from the radar frame.

## 3. Reference hardware and design choice

Use a **pluggable 60 GHz 3-TX / 4-RX ODS-style reference profile**, initially benchmarked against the TI IWR6843ISK-ODS:

- RF band: 60–64 GHz; nominal carrier for calculations: 61.0 GHz.
- 3 TX, 4 RX, TDM-MIMO, 12 virtual channels.
- ODS reference FoV: 120° azimuth and 120° elevation.
- ODS peak antenna gain: use 5 dBi as a reference value only where supported by the TI EVM guide; mark antenna-pattern shape as approximate until measured pattern data is imported.
- RF power / receiver noise / phase-noise values: use the provenance and caveats in `radar_config.yaml`.
- The 4 × 3 virtual array coordinates in the config are a **provisional idealized geometry**, not verified physical coordinates from the board artwork. Keep them replaceable.

Why this choice: it is a documented 60 GHz 3-TX/4-RX architecture with a wide elevation FoV and 2D virtual antenna arrangement, a useful benchmark for a room prototype. Keep the hardware profile behind a configuration/adapter interface so another vendor or a custom PCB can be substituted without rewriting the signal chain. The TI ODS EVM is a technical reference, not a guaranteed lowest-cost purchase choice in Egypt. No hardware purchase is required for this software-only task.

## 4. Default room and sensor placement

- Room: 4.0 m (X) × 5.0 m (Y) × 2.7 m (Z).
- Origin: lower-left/front floor corner; floor is Z=0.
- Door: entrance opening on the Y=0 wall.
- Window: on the opposite Y=5 m wall.
- Furniture: bed, sofa, table, chair and wardrobe, with collision geometry and material IDs.
- Radar: wall mounted near the centre of the entrance wall, at approximately (2.0, 0.08, 2.25) m, boresight toward +Y with 15° downward tilt.
- The room size and object placements must be data-driven; later randomization must support rooms up to 8 × 6 m without code changes.

The avatar starts just outside the door and enters the room. The default room is a starter layout, not a universal room plan.

## 5. Motion and scenario design

- One adult male-presenting avatar, with a configurable height/body-size range.
- Internal pose/motion representation: 17 named joints at 100 Hz. It drives the visible rigged mesh but is never drawn as a skeleton.
- Actions: enter, walk, turn, pause, sit on chair/sofa/bed edge, stand, bend, pick up, kneel, lie down intentionally, sleep, and fall forward/backward/sideways/slump/slow-collapse.
- Randomize path, speed, action order, pauses, body dimensions, fall type, time-to-event and start pose using reproducible subsystem seeds.
- Use A* or an equivalent collision-aware planner for walking paths. Add contact-aware fall trajectories and floor/furniture collision checks.
- Include hard negatives: quick sitting, kneeling, bending, lying down intentionally, dropping onto a sofa, and normal sleep/rest.
- Provide three explicit modes: `normal_daily_routine` (no fall required), `fall_demo` (one specified fall event), and `random_mix` (probabilistic combination). Do not silently label normal lying down as a fall.

## 6. Physics fidelity boundary

The initial engine is **geometric optics / ray-based** and should include:

- Direct LOS paths and geometric occlusion.
- Fresnel reflection coefficients using complex relative permittivity and declared polarization convention.
- First- and optionally second-order specular reflections from walls, floor, ceiling, and furniture.
- Finite material thickness / transmission loss where configured, including the default multi-layer plasterboard/air-cavity wall model.
- Diffuse scattering approximated by seeded stochastic scattering points or scattering lobes.
- Aspect-dependent human point-scatterer RCS priors with uncertainty/sensitivity sweeps.
- Correct path length, delay, phase, amplitude loss, and radial Doppler for each path.

Do not call this a full-wave solver. Diffraction, near-field coupling, detailed human bioelectromagnetics, exact material moisture dependence, coherent multi-bounce target scattering, and calibrated antenna patterns are future improvements unless explicitly implemented and validated.

## 7. Radar signal chain

Implement:

1. TDM-MIMO chirp schedule and per-TX timing.
2. Complex dechirped FMCW beat signal from all active propagation paths.
3. Coherent combination of path contributions per TX/RX pair.
4. Receiver thermal noise at an explicitly stated reference plane and bandwidth.
5. Toggleable imperfections: phase noise approximation, chirp slope/frequency error, TX/RX channel gain/phase mismatch, IQ imbalance, ADC quantization, ADC clipping, TX leakage and timing jitter.
6. ADC cube preserving chronological chirp timing and TX identity.
7. Range FFT, clutter-preserving data path, same-TX Doppler FFT, TDM phase compensation, CFAR, angle estimation, point cloud, clustering and tracking.
8. Keep a separate stationary-return/phase branch for possible breathing displacement. Do not apply a moving-target clutter filter to the only copy of the data.

## 8. Engineering rules (mandatory for every agent)

1. **Read first:** inspect `PLAN.md`, `INTERFACES.md`, `radar_config.yaml`, `scene.json`, `material_db.yaml`, and `motion_config.yaml` before coding.
2. **One task only:** implement only the assigned prompt. Do not redesign unrelated modules.
3. **Single source of truth:** radar waveform, hardware reference, antenna, impairment and expected-derived parameters live in `radar_config.yaml`. Room geometry lives in `scene.json`. Material properties live in `material_db.yaml`. Human motion distributions live in `motion_config.yaml`.
4. **SI units internally:** m, s, Hz, rad, W, V, kg; display units may be converted only at UI boundaries.
5. **Coordinate convention:** use the single world and angle convention from `INTERFACES.md` everywhere.
6. **Determinism:** all random subsystems receive derived independent seeds. Same seed and config must reproduce the same outputs.
7. **No fake physics:** do not hardcode a desired point cloud, Doppler heatmap, fall alert, or expected result. The dashboard must display outputs from the actual preceding stages.
8. **No test gaming:** expected values in tests must come from independent analytical calculations. Do not use the same helper function to compute both result and expected value.
9. **Provenance:** each physical number must be tagged `manufacturer`, `standard`, `literature`, `assumption`, `measured`, or `calibrated`; include a URL/reference and uncertainty/limitations.
10. **Do not invent citations or measured values.** If a value is unavailable, explicitly mark it assumed and sweep it.
11. **Signal conventions:** document real vs complex sampling, dechirp sign, FFT normalization/window gain, Doppler sign, dB reference, and ADC scaling.
12. **Impairments toggle independently** and must have tests demonstrating their expected influence.
13. **Performance:** use vectorized NumPy/SciPy or other documented efficient implementations. Provide a quick-preview mode and a full-frame mode, but do not silently substitute preview output while labelling it full fidelity.
14. **API stability:** follow `INTERFACES.md`. If any change is needed, update the interface doc, producer, consumer and tests together.
15. **Avatar rule:** visible person is a complete skinned/rigged human mesh with a head, torso, arms, hands, legs, feet and clothing or clear human-like materials. A hidden rig may drive animation. Never render the skeleton, bones, capsules or joint markers as the visible person.
16. **Ground-truth separation:** never leak ground-truth avatar points into the radar point cloud. Ground truth and radar-derived outputs must be distinct streams and separately labelled.
17. **No perfect-accuracy claims:** label every result as analytical, simulated, hardware-in-the-loop or measured on physical hardware.
18. **Task close-out:** report changed files, commands executed, tests/results, acceptance checklist, assumptions/limitations, and unresolved blockers.

## 9. Fidelity modes

- `preview`: reduced number of scatterers/rays and faster update for interactive UI; label as preview.
- `full_sim`: all configured chirps, ADC samples, receiver impairments and DSP for the selected frame; label as simulated.
- `analysis_export`: deterministic fixed-seed batch that saves intermediate products/HDF5 for later analysis.

Never compare metrics from different modes without recording the mode and configuration.

## 10. Tonight's acceptance target

A fixed-seed end-to-end demo must:

- Load the default room and material database.
- Display a complete human avatar entering the room and performing a varied sequence.
- Produce deterministic ray paths with LOS, at least one valid reflection, and an occlusion case.
- Generate a complex ADC cube with the correct dimensions and TDM timestamps.
- Pass single-target analytical range and Doppler tests and a receiver-noise statistical test.
- Produce DSP detections and a radar-generated point cloud that changes with the simulated motion.
- Display all four dashboard panels synchronized to one timestamp.
- Save a report documenting tests and known physical-model limitations.

The system is **not** accepted merely because the dashboard looks attractive. It is also not accepted if it displays ground-truth coordinates as radar detections.

## 11. Out of scope for tonight

- A claim of full-wave electromagnetic perfection.
- A trained deep fall-detection model with proven real-world accuracy.
- Verified camera-like dense human surface reconstruction from only 12 virtual channels.
- Physical hardware acquisition, PCB fabrication, regulatory testing or calibration against the actual EVM.
- A guarantee of 95% fall recall or a particular false-alarm rate.

These are later milestones after the first simulation works and measured data is available.
