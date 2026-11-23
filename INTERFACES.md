# INTERFACES.md — Shared data contracts

This document is the integration contract. Producers and consumers must use these conventions. Update this file whenever an interface changes.

## 1. Common conventions

- Internal units: metres, seconds, hertz, radians, watts, volts, kilograms, kelvin.
- World axes: `+X` is room width to the right when looking from the entrance; `+Y` points from the entrance wall into the room; `+Z` is upward.
- World origin: front-left floor corner, `(0, 0, 0)`.
- Room bounds: `0 <= x <= width_x_m`, `0 <= y <= depth_y_m`, `0 <= z <= height_z_m`.
- Radar local boresight: local `+Y`; local `+X` points to its right; local `+Z` points upward.
- Azimuth: `atan2(dx, dy)` in the world frame after expressing a target direction in the radar local frame; positive angle is toward radar-local `+X`.
- Elevation: `atan2(dz, sqrt(dx**2 + dy**2))`; positive is above the local horizontal plane.
- Radial velocity: `v_r = dot(v_target - v_radar, unit_vector_radar_to_target)`. Positive means receding from the radar; negative means approaching.
- Use radians and SI units in code. Convert to degrees/metres-per-second only for display labels.
- Use `complex64` or `complex128` for RF samples and document the chosen precision.
- Every frame has monotonically increasing `frame_id` and one authoritative `sim_time_s`.

## 2. Scene schema (`scene.json`)

Top-level keys:

```json
{
  "schema_version": "1.0",
  "scene_id": "room_4x5_default",
  "units": "m",
  "room": {
    "width_x_m": 4.0,
    "depth_y_m": 5.0,
    "height_z_m": 2.7,
    "wall_thickness_m": 0.12,
    "floor_material_id": "concrete",
    "wall_material_id": "plasterboard",
    "ceiling_material_id": "plasterboard"
  },
  "openings": [],
  "objects": [],
  "radar": {}
}
```

Each opening has `id`, `kind` (`door` or `window`), `wall`, `center_m`, `width_m`, `height_m`, `sill_height_m`, `material_id`, and optional `hinge_side` / `swing_direction`. Each furniture object has `id`, `kind`, `position_m` (the south-west/lower floor corner of its local bounding box before yaw), `dimensions_m` `[width_x, depth_y, height_z]`, `yaw_rad`, `material_id` or component material map, and `collision_shape`. Rotation is about the horizontal centre of the footprint. The room `construction` block stores layered walls/floor/ceiling; openings may also have `construction_layers`.

Use the same loaded scene object for rendering, navigation/collision, and RF geometry. Do not maintain separate hand-coded room dimensions in different modules.

## 3. Material database (`material_db.yaml`)

Each material entry includes:

- `epsilon_real` and `epsilon_imag` at the configured center frequency, using the sign convention `epsilon_r = epsilon_real - j*epsilon_imag`.
- `conductivity_s_per_m` only when applicable; do not double-count conductivity and dielectric loss.
- `valid_frequency_hz`, `thickness_m` or `thickness_range_m` when relevant, `roughness_rms_m` or a declared roughness model.
- `polarization_assumption`, `provenance`, `source_url`, `uncertainty`, and `notes`.
- Missing or low-confidence material values must be explicit. Do not turn a placeholder into a claimed measured constant.

Fresnel functions must document TE/TM polarization and incidence-angle conventions.

## 4. Motion frame (`MotionFrame`)

Python-facing logical structure:

```python
MotionFrame(
    frame_id: int,
    sim_time_s: float,
    actor_id: str,
    action_label: str,
    scenario_mode: str,
    fall_event_id: str | None,
    root_position_m: tuple[float, float, float],
    root_yaw_rad: float,
    joints_m: dict[str, tuple[float, float, float]],
    body_capsules: list[BodyCapsule],
    contacts: list[ContactEvent],
    avatar_asset_id: str,
)
```

The 17-joint map uses stable names: `pelvis`, `spine`, `neck`, `head`, `left_shoulder`, `left_elbow`, `left_wrist`, `right_shoulder`, `right_elbow`, `right_wrist`, `left_hip`, `left_knee`, `left_ankle`, `right_hip`, `right_knee`, `right_ankle`, `chest`.

The joint positions drive the hidden rig. `body_capsules` are collision/scattering proxies and are not the visible avatar. The avatar mesh is a distinct render asset.

A `BodyCapsule` includes `segment_id`, `a_m`, `b_m`, `radius_m`, `material_id`, and optional `rcs_prior_dbsm` / uncertainty metadata.

## 5. RF paths (`PathSet`)

A `PathSet` is generated per simulation frame and includes:

```python
PathSet(
    frame_id: int,
    sim_time_s: float,
    paths: list[RFPath],
    diagnostics: dict,
)
```

Each `RFPath` contains:

- `path_id`, `tx_id`, `rx_id`, `target_id` or `surface_id`.
- `path_type`: `los`, `specular_reflection`, `diffuse_scatter`, or `transmitted`.
- `vertices_m`: ordered propagation vertices.
- `tx_leg_length_m`, `rx_leg_length_m`, `total_path_length_m`, `delay_s`.
- `radial_velocity_mps`, `doppler_hz`, `phase_rad`.
- `path_loss_db`, `reflection_coeff_real`, `reflection_coeff_imag`, `rcs_m2` where applicable.
- `amplitude_linear` and `amplitude_phase_rad` or `complex_amplitude` internally. Do not serialize Python complex values directly to JSON.
- `model_flags` and uncertainty/provenance metadata.

For a monostatic direct point target at range `R`, the round-trip delay is `2R/c`. For reflected paths, compute total geometric path length. For moving targets, compute path-length derivative or a documented far-field radial-velocity approximation.

## 6. ADC cube (`ADCCube`)

Canonical in-memory shape:

`complex_samples[loop_index, tx_index, rx_index, adc_sample]`

Expected shape from the initial config: `[128, 3, 4, 192]`.

Required metadata:

- `frame_id`, `sim_time_s`, `config_hash`, `dtype`, `sample_units`, `adc_full_scale`, `adc_bits_effective`.
- `chirp_time_s[loop_index, tx_index]` containing actual transmit timestamps.
- `tx_schedule` and `rx_ids`.
- `sample_rate_hz`, `slope_hz_per_s`, `fc_start_hz`, `chirp_period_s`, `same_tx_interval_s`.
- `noise_reference_plane`, `noise_equivalent_bandwidth_hz`, and enabled impairment names.

The TX axis is a convenient reshaping of the chronological TDM sequence. Preserve timestamps so the Doppler processor uses actual same-TX intervals. A raw chronological buffer may also be retained.

## 7. DSP output (`DSPFrame`)

Each frame contains:

- `frame_id`, `sim_time_s`, `config_hash`, `processing_mode`.
- `detections`: `x_m`, `y_m`, `z_m`, `range_m`, `azimuth_rad`, `elevation_rad`, `radial_velocity_mps`, `snr_db`, `confidence`, `track_id`.
- `tracks`: position/velocity, age, missed-frame count, covariance or confidence.
- `range_doppler`: axis vectors and compressed/log-magnitude matrix or a separate binary plot payload.
- `range_angle`: azimuth/elevation axes and matrix.
- `occupancy`: sparse voxel coordinates, voxel size, probability/confidence and provenance `radar_derived`.
- `features`: estimated height, vertical velocity, track stability and optional micro-Doppler/breathing features.
- `fall_alert`: state, confidence, event time, evidence fields and baseline rule version.

Ground-truth positions are not part of `detections`. They may be included under a separate `ground_truth` object strictly for synchronized display and evaluation.

## 8. WebSocket frame message

Web frontend connects to `ws://localhost:8000/ws/sim`. One JSON message per dashboard update:

```json
{
  "schema_version": "1.0",
  "type": "simulation_frame",
  "frame_id": 120,
  "sim_time_s": 6.0,
  "seed": 20261123,
  "scene_id": "room_4x5_default",
  "ground_truth": {
    "actor_id": "person_01",
    "action_label": "walking",
    "root_position_m": [1.2, 2.1, 0.0],
    "root_yaw_rad": 0.2,
    "joints_m": {},
    "avatar_asset_id": "adult_male_default"
  },
  "rf": {"paths": [], "ray_summary": {}},
  "radar": {"detections": [], "tracks": [], "point_cloud": []},
  "analytics": {
    "fall_alert": {"active": false, "confidence": 0.0, "event_label": "none"},
    "range_doppler": {"range_m": [], "velocity_mps": [], "values_db": []},
    "range_angle": {"azimuth_rad": [], "elevation_rad": [], "values_db": []}
  },
  "provenance": {"mode": "full_sim", "config_hash": "...", "seed_map": {}}
}
```

The example arrays are empty placeholders. The actual server must populate them with the same frame's outputs. If a matrix is too large for JSON, send compact metadata in JSON and a documented binary payload, but keep `frame_id` and `sim_time_s` identical.

## 9. Control messages

Frontend-to-backend messages:

- `{"type":"control","action":"play"}`
- `{"type":"control","action":"pause"}`
- `{"type":"control","action":"step"}`
- `{"type":"control","action":"set_speed","value":1.0}`
- `{"type":"control","action":"set_seed","value":42}`
- `{"type":"control","action":"regenerate_scenario","scenario_mode":"random_mix"}`
- `{"type":"control","action":"set_impairment","name":"phase_noise","enabled":true}`

Validate all control messages and reject unknown keys/actions cleanly.

## 10. HTTP endpoints

- `GET /api/health` → server status and config hash.
- `GET /api/config-summary` → selected radar/scene parameters and derived metrics.
- `GET /api/scenario-modes` → supported scenario modes.
- `WS /ws/sim` → synchronized simulation frames and control messages.

The backend owns simulation time, RNG seeds, and frame IDs. The browser must not independently invent actor motion or radar detections.
