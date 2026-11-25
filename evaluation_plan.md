# Evaluation plan — what to measure and what it proves

## 1. Validation levels

| Level | Evidence | What it proves | What it does not prove |
|---|---|---|---|
| Analytical | Independent formulas and deterministic ideal test cases | Formula/sign/unit consistency | Full simulator realism |
| Software/unit simulation | Synthetic scenes and known synthetic targets | Internal implementation behavior | Match to a physical radar |
| Randomized simulation | Held-out seeds, layouts and impairment settings | Robustness within the chosen simulator model | Real-world generalization |
| Hardware-in-the-loop | Recorded radar ADC/point-cloud data compared with simulator | Similarity for the measured hardware/profile | Universal performance |
| Physical measured validation | Calibrated target/person trials with ground truth | Performance within the tested conditions | Performance in all rooms/users |

## 2. Required tonight metrics

- Derived waveform: effective bandwidth, range resolution, maximum unambiguous range, maximum unambiguous radial velocity, velocity resolution.
- Scene: invalid-layout rejection rate, collisions, door reachability.
- Motion: action diversity by seed, collision/contact violations, fall event timestamps and duration.
- RF: LOS delay, reflected path delay, path-loss/power components, blocked-LOS cases, path-count distribution.
- Front-end: beat-frequency error, range-bin error, Doppler-bin error, noise-power statistical interval, ADC clipping/quantization diagnostics.
- DSP: detection precision/recall on synthetic labels, point-cloud position errors, track continuity, false tracks, baseline fall-event precision/recall and latency.
- Dashboard: measured preview FPS, dropped-frame count, synchronization error across panels.

## 3. Test scenarios

1. **Analytical single reflector:** static target at 3.00 m; known RCS and gains; LOS only.
2. **Moving point target:** target at 3.00 m receding at +1.00 m/s; check range/Doppler sign and bins.
3. **Wall reflection:** simple planar wall; use image geometry for independent path length.
4. **Occluded target:** LOS blocked by a furniture/object surface; reflected path may remain.
5. **Human walk:** actor enters and walks around furniture.
6. **Hard negatives:** quick sit, kneel, bend/pick up, intentional lie down, sleep/roll.
7. **Fall demo:** forward/backward/sideways/slump/slow-collapse in separate seeds.
8. **Random mix:** seeds should create different paths, event order and timing.

## 4. Reporting rules

- Always state number of independent scenarios, seed IDs, total simulated seconds and processing mode.
- Frame-level metrics and event-level metrics must not be mixed.
- A single fall event repeated across hundreds of frames is still one event for event-level recall.
- Report false alarms per simulated hour as well as the underlying count and simulated exposure hours.
- Confidence intervals must reflect independent scenario grouping, not treat adjacent frames as independent trials.
- Simulated false-alarm rates are not real-world false-alarm rates.
- Do not report an accuracy percentage without naming the test set, unit of analysis, metric and sample size.

## 5. Future physical validation checklist

Before claiming that the simulator matches an actual radar:

- Import exact TX/RX antenna element coordinates and measured antenna patterns for the selected board.
- Verify RF output power, receiver noise figure, IF filter/ADC scale, chirp timing and phase-noise data from the correct silicon/board revision.
- Calibrate range with a known reflector at multiple distances.
- Calibrate Doppler using a controlled moving target or translation stage.
- Measure static clutter and leakage in an empty room.
- Record human/dummy radar returns with documented placement and pose; treat human RCS as scenario-dependent.
- Compare range-Doppler, channel phase, angle and detection distributions against physical captures.
- Maintain separate calibration and held-out physical test sessions.
