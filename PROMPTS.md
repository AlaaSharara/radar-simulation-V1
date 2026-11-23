# PROMPTS.md — One task per Antigravity agent session

**Important:** Copy the shared instruction block and only ONE task prompt into each agent session. Run T0 → T7 in order. Do not launch parallel agents that edit overlapping files.

## Shared instruction block — paste at the start of every task

> Read `PLAN.md`, `INTERFACES.md`, `radar_config.yaml`, `scene.json`, `material_db.yaml`, and `motion_config.yaml`. Follow `PLAN.md` exactly. Inspect the existing repository before editing. Execute only the assigned task.
>
> Keep all internal units SI. Follow existing interfaces. Do not invent measured parameters, manufacturer specs, source citations, or test results. Distinguish assumptions from sourced values. Do not hardcode a desired result or use ground-truth coordinates as radar detections. A visible complete human avatar is mandatory in the Scene panel; the internal skeleton/rig must remain invisible. Never claim perfect physics or real-hardware accuracy. If an interface must change, update the contract, producer, consumer and tests together.
>
> At completion report: (1) changed files, (2) commands executed, (3) tests and exact results, (4) acceptance checklist, (5) assumptions/limitations, and (6) unresolved blockers. Never claim a test passed without running it.

---

## T0 — Repository bootstrap and configuration validator

**Goal:** Create the runnable Python/JS project skeleton and validate the radar numbers before implementing simulation physics.

**Owned files:** `pyproject.toml`, `requirements.txt`, `src/radarsim/__init__.py`, `src/radarsim/config.py`, `src/radarsim/rng.py`, `scripts/validate_config.py`, `tests/test_config.py`, `viz/web/package.json`, `viz/web/vite.config.js`, `viz/web/index.html`, `.gitignore`.

**Instructions:**

1. Create the Python package under `src/radarsim/` with module folders `scene`, `motion`, `rf`, `radar`, `dsp`, `viz`, `eval`.
2. Implement strict loading/validation for `radar_config.yaml`, `scene.json`, `material_db.yaml`, `motion_config.yaml`, and `sim_profile.yaml`. Provide a vendor-neutral `RadarProfile` object so hardware-specific assumptions stay in config and another sensor profile can be substituted later. Provide readable error messages for missing/invalid values.
3. Implement independent derived-metric calculation for effective bandwidth, wavelength, range resolution, max unambiguous range, same-TX chirp interval, max unambiguous radial speed and velocity resolution.
4. Compare calculations against `expected_derived` using the configured tolerances. Do not read expected values as calculation inputs.
5. Implement stable independent RNG streams derived from a global seed and stream name.
6. Inspect the pre-created minimal Vite/React package in `viz/web/`; verify dependencies and preserve its working manifest. The actual dashboard is T6.
7. Make `python scripts/validate_config.py` work from the repository root.

**Acceptance:** Config loads; invalid config fails clearly; computed values match tolerances; same seed reproduces stream values; tests pass. Do not build radar physics yet.

---

## T1 — Furnished room geometry and material-aware scene model

**Goal:** Implement the 3D room and all furniture from `scene.json` with one shared geometry source for renderer, collision/navigation and RF.

**Owned files:** `src/radarsim/scene/` and `tests/scene/` only.

**Instructions:**

1. Load the exact scene JSON and material database. Build surfaces for floor, ceiling, layered walls (plasterboard / air cavity / plasterboard), door/window openings, bed, sofa, table, chair and wardrobe. Respect construction-layer thicknesses from `scene.json`.
2. Implement coordinate transforms and bounding/collision geometry using the convention in `INTERFACES.md`.
3. Implement object intersection, wall-boundary and doorway-clearance tests. The door opening must remain navigable.
4. Provide a random-layout generator with seeded dimensions/placements that rejects invalid scenes rather than silently overlapping or clipping objects.
5. Provide material lookup and complex-permittivity helper with the declared sign convention.
6. Export a normalized scene object usable by motion, RF and frontend code. Do not create separate room dimensions inside those modules.
7. Create a simple scene preview artifact or a small geometry summary that can be checked before T2.

**Acceptance:** Default scene loads; furniture objects and openings exist; no object overlap; door approach is clear; 100 seeded randomized layouts are accepted/rejected with explicit reasons; the same scene object is consumed by downstream modules.

---

## T2 — Visible human avatar, navigation, and randomized actions

**Goal:** Make a recognizable human enter the furnished room and perform varied, collision-aware motion. This task must NOT display a skeleton as the person.

**Owned files:** `src/radarsim/motion/`, `assets/avatars/`, `tests/motion/` only. T6 will consume the avatar asset and pose messages.

**Instructions:**

1. Implement a 17-joint internal pose representation at 100 Hz using the names in `INTERFACES.md`.
2. Create or import a complete human avatar mesh with a rig: head, torso, arms, hands, legs, feet and visible human-like surface/clothing. Prefer a local GLB with a documented permissive license. If an acceptable asset cannot be obtained, build a procedural articulated human mesh. Never render bone lines, stick figures, capsules or joint markers as the visible person.
3. Provide a stable avatar asset ID and rig/joint mapping so Three.js can animate the mesh from the pose stream.
4. Generate entry, walk, turn, pause, sit, stand, bend, pick-up, kneel, lie-down, sleep, and five fall styles from `motion_config.yaml`.
5. Use collision-aware A* navigation around furniture. Check the complete body proxy, not only joint points. Support intentional sitting/lying contacts with furniture.
6. Implement contact-aware fall phases: loss of balance, descent/rotation, impact, settling and optional recovery. Do not teleport directly from standing to lying.
7. Implement `normal_daily_routine`, `fall_demo`, and `random_mix`. Keep hard-negative actions distinct from falls.
8. Use independent deterministic RNG streams. Produce `MotionFrame` and body scattering/collision proxies as documented.

**Acceptance:** A browser-ready full-body avatar asset exists; the skeleton is hidden; fixed seeds reproduce the exact sequence; 20+ seeds show different action orders/paths; the actor enters through the doorway; body collisions and contact states are valid; all labels/event timestamps are consistent.

---

## T3 — Room RF propagation and path-set engine

**Goal:** Compute physically grounded propagation paths from each TX through the scene/targets to each RX.

**Owned files:** `src/radarsim/rf/` and `tests/rf/` only.

**Instructions:**

1. Implement direct LOS path generation and geometric occlusion for scene surfaces and human/furniture geometry.
2. Implement first-order specular reflections using image geometry and complex Fresnel TE/TM coefficients. Add second-order reflections behind a config switch and bounded path count.
3. Add transmission through configured finite-thickness materials and multilayer wall stacks only where the material data supports it; use a transfer-matrix or documented layered-interface calculation where implemented. Apply losses exactly once.
4. Add seeded diffuse-scattering approximations for rough surfaces and human body scatterers. Clearly label the model as geometric/stochastic, not full-wave EM.
5. Calculate total path length, delay, phase, amplitude, path loss and Doppler using consistent conventions. Account for TX-to-scatterer and scatterer-to-RX legs.
6. Implement an aspect-dependent point/capsule human RCS prior with uncertainty and seeded variation. Do not claim a universal human RCS.
7. Return `PathSet` exactly as documented in `INTERFACES.md`; include diagnostics and per-path provenance/model flags.

**Acceptance:** Direct path at 3.00 m has delay `2R/c` within numerical tolerance; a wall reflection matches independently computed image geometry; blocked LOS is removed; reflection magnitude is physically bounded for passive materials; same seed is reproducible; at least LOS/reflection/diffuse paths appear in a test scene.

---

## T4 — FMCW/TDM-MIMO front-end and ADC cube

**Goal:** Generate the complex dechirped radar receive data using the selected configurable hardware/waveform profile and all configured non-idealities. Do not hardcode TI-specific constants in the signal-processing code.

**Owned files:** `src/radarsim/radar/` and `tests/radar/` only.

**Instructions:**

1. Implement transmitted linear FMCW chirps, dechirp/mixer convention, per-TX TDM timestamps and per-RX receive channels.
2. Convert each `RFPath` to delayed, attenuated, phase-rotated complex beat signal. Sum paths coherently for each TX/RX channel before noise.
3. Model phase noise, chirp slope/start-frequency error, TX/RX amplitude/phase mismatch, IQ imbalance, ADC quantization/clipping, TX leakage, timing jitter and thermal noise as independently toggleable effects.
4. Set the thermal-noise reference plane and equivalent noise bandwidth explicitly. Use `k*T*B*F`; convert to ADC units using documented gain/impedance/full-scale assumptions. Test statistically, not by matching one exact random sample.
5. Preserve chronological TDM timing and output shape `[loops, tx, rx, adc_samples]` with metadata required by `INTERFACES.md`.
6. Use vectorized operations and provide preview/full-sim modes. Do not fabricate a radar cube from ground-truth XYZ points without signal generation.

**Acceptance:** A stationary point target at 3.00 m appears at the analytically predicted range beat/bin; a +1.00 m/s receding target appears at the expected Doppler sign/bin; TDM same-TX interval is correct; noise meets a stated statistical tolerance; each impairment toggle has an ablation test.

---

## T5 — DSP chain, point cloud, tracks and baseline fall indication

**Goal:** Process the ADC cube into radar-derived detections and diagnostics.

**Owned files:** `src/radarsim/dsp/`, `src/radarsim/eval/` and `tests/dsp/` only.

**Instructions:**

1. Implement range window/FFT and documented normalization/coherent gain.
2. Implement a clutter-suppression branch for movement, but preserve an untouched complex slow-time branch for stationary targets.
3. Implement same-TX Doppler FFT using actual timestamps and TDM-MIMO Doppler phase compensation.
4. Implement configurable OS-CFAR, angle estimation for the configured 4 × 3 provisional virtual array, and clearly label zero-padding as interpolation rather than resolution improvement.
5. Generate radar-only 3D detections with range, azimuth, elevation, radial velocity, SNR/confidence and track ID. Add clustering and a basic tracker.
6. Add a simple radar-derived occupancy/voxel representation only from detections; do not fill missing body areas from the ground-truth avatar.
7. Implement a transparent rule-based fall baseline using temporal height/velocity/stillness features. Normal sitting, kneeling, bending and lying down must be hard negatives.
8. Keep breathing/phase extraction as a separate optional branch, not a mandatory success claim for tonight.
9. Return `DSPFrame` and diagnostics as documented.

**Acceptance:** Synthetic single-target range/Doppler tests pass; detections respond to motion; point cloud has no ground-truth leakage; tracker metrics and fall baseline outputs are reported on fixed synthetic seeds; the 4 × 3 aperture's angular limitations are stated.

---

## T6 — Four-panel synchronized dashboard

**Goal:** Build the web dashboard and control channel. The real scene panel must show a human avatar, not a skeleton.

**Owned files:** `viz/web/` and `src/radarsim/viz/` only.

**Instructions:**

1. Build a React + Three.js dashboard with a responsive 2 × 2 layout and a common frame clock.
2. Panel 1: render the room and complete visible avatar mesh from T2. Animate it from the backend pose/animation stream. Keep bones/joints/capsules invisible. Add orbit/pan/zoom controls.
3. Panel 2: draw animated ray/wave paths for LOS, reflections and diffuse paths; color/style by path type and fade by amplitude/path loss. Add a clear label: `Geometric-optics RF visualization — not full-wave field simulation`.
4. Panel 3: render only radar-derived point cloud/tracks/occupancy. Add a separate toggle for ground-truth comparison; default it off. Label radar observations vs reconstructed voxels distinctly.
5. Panel 4: show range-Doppler heatmap, range-angle view, activity timeline, estimated track height/vertical speed and baseline fall alert.
6. Add play/pause, step, speed, seed selector, scenario mode selector, regenerate and impairment toggles. Controls must send backend control messages and receive actual state changes.
7. Keep all four panels on the same `frame_id` and `sim_time_s`. Do not independently animate the person in the frontend.
8. Handle dropped frames and WebSocket reconnects cleanly.

**Acceptance:** Dashboard opens locally; all four panels update from one backend frame; avatar remains a human mesh; ray display changes with scene/pose; point cloud changes with radar output; controls work; 30-second run sustains at least 10 synchronized updates/s in preview mode on the development machine, with actual measured fps reported.

---

## T7 — End-to-end integration, reproducibility and engineering report

**Goal:** Connect the real implementations, run fixed test scenarios and document what is and is not validated.

**Owned files:** `src/radarsim/main.py`, `src/radarsim/pipeline.py`, `tests/integration/`, `reports/`, and run scripts only.

**Instructions:**

1. Integrate the scene → motion → RF → radar/ADC → DSP → WebSocket pipeline without duplicate physics or hardcoded dashboard data.
2. Add CLI flags for config paths, seed, scenario mode, `preview/full_sim/analysis_export`, and output folder.
3. Run at least three fixed scenarios: normal daily routine, one fall demo (choose and report fall type), and one random-mix seed. Save seed map and config hash.
4. Run unit and integration tests. Report exact failures; do not hide failures or weaken tests to make them pass.
5. Export one HDF5 or NPZ record with timestamps, labels, paths/ADC or selected intermediate products, DSP outputs and metadata. If full ADC export is too large, document the selected window and why.
6. Generate `reports/tonight_run.md` with waveform-derived metrics, frame rate, runtime, tests, plots/screenshots, limitations, and next validation needs.
7. Explicitly separate analytical tests, simulated performance, and any physical measurement (none is expected tonight).

**Acceptance:** A fixed-seed simulation runs end-to-end; all four panels are synchronized; tests pass or documented blockers remain; outputs and configuration are reproducible; the report contains no unsupported real-world accuracy claims.
