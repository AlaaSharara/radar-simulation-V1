# Tonight Simulation Run & Verification Report

> **Provenance & Scope Notice:** Every metric reported below is either an **analytical identity/config check** or a **synthetic software simulation output**. **No physical radar hardware measurements were collected tonight**, and no real-world clinical or deployment accuracy claim is made.

---

## 1. Run Metadata & Reproducibility Configuration

| Parameter | Value |
| :--- | :--- |
| **Config Hash (SHA-256)** | `c149d8b8561deeffd780149cd7610fff16de6e55093be89fd4a4677afc5c956d` |
| **Radar Profile** | `iwr6843isk_ods_class_room_v1` (60 GHz 3-TX / 4-RX ODS-style TDM-MIMO FMCW) |
| **Scene** | `room_4x5_default` (`4.0 m × 5.0 m × 2.7 m`, doorway on `y_min`, bed, sofa, table, chair, wardrobe) |
| **Avatar Asset** | `adult_male_default` (`assets/avatars/adult_male_default.glb`, `174,816` bytes, rigged human mesh, skeleton hidden) |
| **Python / Package Versions** | Python `3.13.7`, `numpy 2.3.3`, `PyYAML 6.0.4`, `matplotlib 3.10.6`, `fastapi 0.165.0`, `uvicorn 0.54.0`, `pytest 9.2.0` |
| **Node / Frontend Tooling** | Vite `v5.4.21` React + Three.js + Plotly dashboard (`npm run build` clean, 46 modules, `19.09 s`) |
| **Physical Hardware Tests** | **None (0 physical captures)** |

### Deterministic Seed Map (Independent RNG Streams per Run)

| RNG Stream Key | `normal_daily_routine` (`seed=20261123`) | `fall_demo` (`seed=7001`, `fall_forward`) | `random_mix` (`seed=20261125`, `analysis_export`) |
| :--- | :--- | :--- | :--- |
| `path` | `18109614216260872920` | `6722303410341481932` | `8859362344060154787` |
| `action_order` | `13376126936641776338` | `2727673688060953719` | `1869619562883448498` |
| `speed` | `11624927853368192807` | `3273926628910530978` | `18432162625789230826` |
| `body_shape` | `16367274253319872912` | `5061439402437262887` | `12132012147976335186` |
| `fall_type` | `1180156345002148080` | `16243560265565601904` | `7572201447225474771` |
| `scatterers` | `4779687768166355050` | `13064274229623322338` | `1770270521489262076` |
| `rf_diffuse` | `1073277032441899545` | `10179488987565709028` | `16917952197939404310` |
| `noise` | `5688678083984328515` | `12329341369238723815` | `2418594122822673983` |
| `impairments` | `5464940596904529470` | `2812754381007503681` | `9015387888276361689` |

---

## 2. Analytical Verification Tests (Math & Config Identities)

All analytical checks ran cleanly from source without relaxing assertions or tolerances:

1. **Configuration & Waveform Validator (`python scripts/validate_config.py`) — PASS (exit code `0`)**:

| Derived Metric (SI Units) | Calculated Value | Expected (`radar_config.yaml`) | Absolute Error | Tolerance | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `effective_bandwidth_hz` | `1.920000e+09` Hz | `1.920000e+09` Hz | `0` | `1` | PASS |
| `range_resolution_m` | `0.0780709526042` m | `0.078070952604` m | `1.67e-13` | `1e-09` | PASS |
| `max_unambiguous_range_m` | `7.195018992` m | `7.195018992` m | `0` | `1e-08` | PASS |
| `wavelength_m` | `0.00491463045902` m | `0.00491463045902` m | `3.93e-16` | `1e-12` | PASS |
| `same_tx_chirp_interval_s` | `0.00018` s | `0.00018` s | `0` | `1e-12` | PASS |
| `max_unambiguous_radial_velocity_mps` | `6.82587563752` m/s | `6.82587563752` m/s | `2.32e-13` | `1e-08` | PASS |
| `velocity_resolution_mps` | `0.106654306836` m/s | `0.106654306836` m/s | `2.93e-13` | `1e-08` | PASS |

2. **Full Pytest Suite (`python -m pytest -v`) — 68/68 PASSED in `103.57 s` (0 failed, 0 skipped)**:
   - `tests/test_config.py` + `tests/test_config_validator.py`: strict config validation, profile substitution, RNG independence, derived metrics.
   - `tests/scene/test_scene.py`: OBB SAT collision tests, navigable doorway clearance, complex permittivity sign convention, and 100 seeded randomized room layouts accept/reject reasons.
   - `tests/motion/test_motion.py`: browser-ready rigged human GLB validation (`draw_skeleton == False`), A* obstacle avoidance, doorway entry, all 5 fall styles across 5 phases, and 24-seed diversity.
   - `tests/rf/test_rf.py`: `2R/c` delay check at `3.00 m`, image-method wall reflection geometry, LOS occlusion + TMM slab transmission, Fresnel Brewster/half-wave & passive bounds (`|R| <= 1`), TM PEC continuity as `d -> 0`, `SurfaceRayAccelerator` vectorized match against scalar hits, reciprocal `TX -> S -> target -> RX` & `TX -> target -> S -> RX` multipath legs, and aspect-dependent capsule RCS prior.
   - `tests/radar/test_radar.py`: analytical `3.00 m` range bin peak, `±1.00 m/s` Doppler bin/sign check, chronological TDM timestamps, `k*T*B_eq*F` thermal noise statistical validation, and independent ablation of all 10 toggleable impairments.
   - `tests/dsp/test_dsp.py`: Hann/Blackman coherent gain calibration, 3D TDM-MIMO angle estimation, `Range-Azimuth` B-scan accuracy (`< 0.15 m` range error, `< 4.5°` azimuth error at `3.00 m, +20°`), preserved untouched stationary slow-time branch vs clutter-suppressed moving branch, rule-based fall baseline vs hard negatives (including `fall_demo` `seed=7001` window `start_frame=220` -> `TP`), and ground-truth non-leakage check.
   - `tests/integration/test_integration.py`: bit-identical fixed-seed full-fidelity reproducibility, 3-scenario non-leakage execution, synchronous `SimulationSession` controls & malformed-command `ControlMessageError` rejection, and subprocess CLI `--mode analysis_export` NPZ round-trip.

---

## 3. Simulated Performance Results (Synthetic Benchmarks Only)

### 3.1 Interactive T6 Backend + Live 4-Panel Dashboard Verification
- **Backend WebSocket & HTTP Verification (`verify_t6_dashboard.py`) — PASS (exit code `0`)**:
  - Verified `/api/health`, `/api/config-summary`, `/api/scenario-modes`, and `/assets/avatars/adult_male_default.glb` (`174,816` bytes).
  - Verified `/ws/sim` control commands (`pause`, `step`, `set_speed=1.5`, `set_impairment`, `regenerate_scenario` `fall_demo` `seed=7001`) and `control_error` rejection on `"hack_physics"`.
  - **30-Second Preview Streaming Rate:** `329` synchronized frames delivered in `30.03 s` -> **`10.96 updates/s`** (`frames_with_detections=218`, `ray_changes=250`, `pc_changes=217`), with **2.18×** faster cold preview synthesis (`11.6–12.0 FPS` uncached in offline benchmark via separable `complex64` BLAS synthesis + vectorized ray acceleration!).
- **Live Browser Render Check (`http://127.0.0.1:5173`, Chromium `1920×1120`)**:
  - `CONNECTED` WebSocket status (`preview`), `0` JS console errors, `0` HTTP `>= 400` failures.
  - **Panel 1 (Architectural Scene Preview):** Procedural PBR hardwood floor planks, cutaway plaster walls with baseboards, framed paneled door + lever handle, translucent daylight window, multi-part bed/sofa/table/chair/wardrobe, wall-mounted 60 GHz PCB radar array + toggleable FoV cone, soft directional shadows, `Iso` / `Top` / `Radar` camera presets, and rigged human GLB avatar mesh with skeleton hidden.
  - **Panel 2 (RF Propagation):** Mandatory non-full-wave disclaimer banner, spherical boresight wavefront arcs, polyline wave-packet motion, floor specular bounce ring markers, top-right Strongest Link Gain HUD, and interactive filter chips (`LOS`, `Refl`, `Diffuse`, `Trans`).
  - **Panel 3 (Radar Point Cloud — Radar-Derived Only):** Polar range rings/spokes, OS-CFAR detections, confirmed/tentative tracks with rolling breadcrumb trails + velocity vectors, sparse cyan occupancy voxels, colorbar legend, and `GT compare` verified OFF by default.
  - **Panel 4 (Signal & Event Diagnostics):** Range-Doppler `[dB]` heatmap with overlaid OS-CFAR circle markers, segmented `Range-Azimuth` B-Scan / `Az-El` / `Micro-Doppler` waterfall tabs, Track Height / Vertical Speed timeline with dashed rule threshold lines, and aperture resolution limitation note.

### 3.2 Headless End-to-End Fixed Scenario Runs

| Scenario Run | Seed | Mode & ADC Shape | Frames Processed | Sim Duration [s] | Wall Time [s] | Achieved FPS | Detection Rate | Track Coverage | Track XY / XYZ RMSE [m] | GT Leakage | Baseline Fall Eval |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. `normal_daily_routine`** | `20261123` | `preview` `(32, 3, 4, 192)` | `481` (`0..480`) | `24.00` | `40.00` | **`12.026`** | `73.8%` (`355/481`) | `89.6%` (`431/481`) | `0.1516` / `0.2264` | `False` | **`TN`** (`0/82` hard-negative FPs) |
| **B. `fall_demo` (`fall_forward`)** | `7001` | `preview` `(32, 3, 4, 192)` | `328` (`0..327`) | `16.35` | `28.20` | **`11.633`** | `65.24%` (`214/328`) | `84.45%` (`277/328`) | `0.1028` / `0.2450` | `False` | **`FN`** on full run `0..327` (`TP` on `220..327` in pytest — see diagnosis) |
| **C. `random_mix` (Active Window)** | `20261125` | `analysis_export` `(128, 3, 4, 192)` | `24` (`36..59` of `274`) | `1.15` (`1.80..2.95 s`) | `41.89` | `0.573` | `100.0%` (`24/24`) | `100.0%` (`24/24`) | `0.5700` / `0.6559` | `False` | **`TN`** (window covers `enter`/`walk`; scheduled `fall_sideways` occurs at `11.00–13.67 s`) |

#### Honest Comparison: `fall_demo` (`seed=7001`) `start_frame=220` (`TP`) vs Full Sequence `0..327` (`FN`)
- When evaluated over the isolated pre-fall/fall window `start_frame=220` (`tests/dsp/test_dsp.py`), `RuleBasedFallDetector` triggers **`outcome = TP`** via finite-difference plunge velocity + fadeout stillness promotion, while keeping `0/82` hard-negative false alerts in `normal_daily_routine`.
- On the full `0..327` preview sequence shown in [fall_demo_timeline.png](file:///C:/Users/moham/Downloads/radar/reports/generated/fall_demo_timeline.png), a transient 3-frame **multipath primary-track switch** at `t = 14.25–14.40 s` briefly lifted `primary.position_m[2]` to `~0.80 m` (above `recovery_z_reset_m = 0.75 m`) right after the fall descent, immediately clearing `_last_drop_seen_s` before the track settled back down to the floor (`z = 0.20..0.02 m` from `14.45–15.50 s`). This highlights the exact need for **recovery dwell hysteresis** (`cur.z_m >= 0.75 m` sustained for `>= 0.40 s` before clearing a recorded rapid-drop flag).

---

## 4. Physical Hardware Measurements

- **None performed tonight.** All results are strictly numerical / synthetic (`evaluation_scope = simulated_synthetic_benchmark_not_real_hardware`, `physical_measurements_performed = false`).

---

## 5. Exported Data Archives & ADC Window Rationale

Exports written to `data/generated/exports/` using compressed NumPy `.npz` (`h5py` is not installed in the current environment) with sidecar `_metadata.json` containing full `config_hash`, `seed_map`, derived metrics, frame labels, and evaluation summaries:

| Export Archive | Size | Total Frames Stored | Contiguous Complex64 ADC Window Retained | Rationale for Window Selection |
| :--- | :--- | :--- | :--- | :--- |
| [tonight_normal_daily_routine_seed20261123_preview.npz](file:///C:/Users/moham/Downloads/radar/data/generated/exports/tonight_normal_daily_routine_seed20261123_preview.npz) (+ `_metadata.json`) | `13.40 MiB` (`14,054,224 B`) | `481` (timestamps, labels, RF path counts, tracks, RD maps) | `12` frames `[234..245]` (`t=11.70..12.25 s`), shape `(12, 32, 3, 4, 192)`, `7,077,888 B` raw | Each preview cube is `0.562 MiB`; full 481-frame raw ADC export would require `270.6 MiB` uncompressed. Window centers on active motion inside the room. |
| [tonight_fall_demo_seed7001_preview_fall_forward.npz](file:///C:/Users/moham/Downloads/radar/data/generated/exports/tonight_fall_demo_seed7001_preview_fall_forward.npz) (+ `_metadata.json`) | `10.33 MiB` (`10,835,837 B`) | `328` (timestamps, labels, RF path counts, tracks, RD maps) | `16` frames `[277..292]` (`t=13.85..14.60 s`), shape `(16, 32, 3, 4, 192)`, `9,437,184 B` raw | Centered on fall impact frame index `285` (`t=14.25 s`) to capture full pre-impact descent, impact, and post-impact settling IQ samples instead of storing `184.5 MiB` of full-sequence IQ. |
| [tonight_random_mix_seed20261125_analysis_export.npz](file:///C:/Users/moham/Downloads/radar/data/generated/exports/tonight_random_mix_seed20261125_analysis_export.npz) (+ `_metadata.json`) | `8.81 MiB` (`9,242,722 B`) | `24` (`start_frame=36`, `t=1.80..2.95 s`) | `12` frames `[42..53]` (`t=2.10..2.65 s`), shape `(12, 128, 3, 4, 192)`, `28,311,552 B` raw | Full `128`-loop cube uses `2.250 MiB/frame` (`54.0 MiB` raw for 24 frames, `~616.5 MiB` for all `274` sequence frames); bounding both run length (`24` active frames at `0.573 FPS` = `41.89 s`) and raw IQ window (`12` frames) keeps runtime fast and disk footprint under `10 MiB` compressed. |

Consolidated machine-readable index: [tonight_summary.json](file:///C:/Users/moham/Downloads/radar/data/generated/exports/tonight_summary.json).

---

## 6. Diagnostic Plots & Live Dashboard Screenshot

1. **Full-Fidelity `analysis_export` Range-Doppler, Range-Azimuth B-Scan & 4x3 Az-El Spectrum** (`random_mix`, `seed=20261125`, `frame=57`):
   ![Range-Doppler, Range-Azimuth and Az-El Spectrum](generated/range_doppler_and_angle.png)
2. **Fall Demo (`fall_forward`, `seed=7001`) Track Height, Vertical Speed & Rule Decision Timeline**:
   ![Fall Demo Timeline](generated/fall_demo_timeline.png)
3. **Top-Down Room Trajectory & Radar Primary Track (Reconstructed Directly From Saved NPZ Exports)**:
   ![Top-Down Room Trajectories](generated/trajectory_and_detections.png)
4. **Live Upgraded 4-Panel React + Three.js + Plotly Dashboard (`http://127.0.0.1:5173`, `Frame=2415`, `fall_demo`)**:
   ![Live Upgraded Dashboard Screenshot](generated/dashboard_live.png)

---

## 7. Known Limitations & Next Validation Needs

### Limitations
1. **Geometric-Optics RF Approximation:** Specular reflections, reciprocal target multipath, TMM slab transmission, and seeded diffuse scattering are geometric/stochastic approximations — not a full-wave Maxwell/FDTD solver (wedge diffraction and creeping waves are absent).
2. **4×3 Virtual Array Coarse Angular Resolution:** The provisional `4×3` aperture yields approx. `25.4°` azimuth × `33.8°` elevation 3 dB Rayleigh beamwidth; zero-padding only interpolates angle bins and cannot separate multipath ghosts closely overlapping in range/azimuth (`xyz_rmse_m = 0.23–0.66 m`).
3. **Broad Unvalidated Human RCS Prior:** Segment RCS priors (`-20` to `-5` dBsm nominal) represent engineering priors rather than empirical 60 GHz chamber measurements.
4. **Transient Multipath Track-Switch Sensitivity in `RuleBasedFallDetector`:** Adding reciprocal target-bounce multipath (`TX -> target -> S -> RX` + `TX -> S -> target -> RX`) improves physical realism and creates real multipath ghost detections (`track_id_switches = 24` in `fall_demo`), which can transiently swap the primary track height above `recovery_z_reset_m` unless guarded by sustained recovery dwell hysteresis.
5. **Windows Sandbox `socketpair()` Limitation:** Python `asyncio` / `TestClient` / `uvicorn` loopback sockets cannot execute inside restricted Windows AppContainer sandboxes and must run outside the sandbox.

### Next Validation Needs
1. **Add Recovery Dwell Hysteresis & Multi-Seed Benchmark Across All 5 Fall Styles:** Require sustained elevated height (`>= 0.75 m` for `>= 0.40 s`) in `RuleBasedFallDetector` before resetting `_last_drop_seen_s`, and benchmark `TP/FP/FN/TN` rates across multi-seed sweeps of all 5 fall styles vs hard negatives.
2. **Multipath Ghost Mitigation / Track Score Prioritization:** Down-weight higher-range floor/wall ghost tracks cleanly using indoor kinematic plausibility / SNR persistence without reading ground truth.
3. **Physical EVM Benchmark:** Compare simulated 60 GHz range-Doppler/angle SNR, multipath clutter floor, and human RCS variance against physical TI/ODS-class captures when available.
