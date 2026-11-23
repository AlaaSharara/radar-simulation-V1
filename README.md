# mmWave FMCW Radar Simulation & 3D Digital Twin Platform

[![Documentation: Project Specification & Guide](https://img.shields.io/badge/Docs-Bilingual_Spec_%26_Guide_(Ar%2FEn)-blue.svg)](PROJECT_SPECIFICATION_AND_GUIDE.md)
[![Tests: 68 Passing](https://img.shields.io/badge/Tests-68%2F68_Passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE-NOTICE.md)

> [!IMPORTANT]
> **Complete Technical Specification & User Guide (English & Arabic):**  
> For the comprehensive mathematical foundations, physical equations (FMCW, TDM-MIMO, ITU-R P.2040-3, OS-CFAR), system architecture, and team onboarding instructions, refer to **[PROJECT_SPECIFICATION_AND_GUIDE.md](PROJECT_SPECIFICATION_AND_GUIDE.md)** (متوفر باللغتين العربية والإنجليزية).

## What this project is meant to build

A reproducible simulation of a furnished room with one adult male-presenting human avatar entering from the door and performing varied everyday actions. A wall-mounted, 60 GHz FMCW/TDM-MIMO radar model transmits and receives signals. The simulator models room propagation, target scattering, receiver imperfections, ADC sampling, DSP and point-cloud generation. A four-panel browser dashboard shows:

1. **Ground-truth scene:** a complete visible human avatar and furniture. The skeleton is an internal animation/control representation and must never be rendered as the visible person.
2. **RF propagation:** animated direct and reflected ray paths, occlusion and path strength.
3. **Radar output:** radar-generated point cloud, tracks and an explicitly-labelled occupancy reconstruction.
4. **Signal diagnostics:** range-Doppler / range-angle views, activity timeline, tracking information and baseline fall alert.

## Important scientific boundary

No software simulation is “perfect” or automatically equivalent to a physical radar. This starter pack defines a **physically grounded, imperfect, ray-based FMCW simulation**. It includes explicit approximations, uncertainty, impairment toggles and analytical tests. It is not a full-wave Maxwell/FDTD solver, and the assumed antenna layout and human RCS must be calibrated against physical measurements before claiming real-hardware accuracy.

The geometric-optics propagation baseline should include LOS, occlusion, Fresnel specular reflections and stochastic diffuse-scattering approximations. Higher-order reflections may be bounded for runtime. A future full-wave or validated ray-tracing engine can be added behind the same interface.

## Selected reference radar profile

The default reference is **IWR6843ISK-ODS-class**, not a claim that every detail of the commercial board has been recreated. TI specifies a 60–64 GHz device family with 3 TX and 4 RX channels; the ODS EVM has 120° azimuth and 120° elevation FoV. TI's EVM guide describes an ODS antenna arrangement giving 12 virtual positions and an approximate 12–15 m people-detection range. The project's exact antenna coordinates are deliberately marked as a provisional idealized 4 × 3 virtual array until measured board geometry is imported.

The selected waveform is a simulation profile for the room, not a TI factory chirp configuration. Derived values are stored under `expected_derived` in `radar_config.yaml` and checked by `scripts/validate_config.py`.

## First run target

- Room: 4.0 × 5.0 × 2.7 m.
- Radar: near the centre of the entrance wall, approximately 2.25 m high, aimed into the room with a slight downward tilt.
- Motion: deterministic 100 Hz internal motion; radar frames at 20 Hz.
- Front-end: 3-TX / 4-RX TDM-MIMO, 192 ADC samples/chirp, 5 MS/s, 128 chirps per TX per frame.
- Dashboard: four synchronized panels at one simulation timestamp.

See `START_HERE.md` and `TONIGHT_SCOPE.md` first. `PLAN.md`, `INTERFACES.md`, `radar_config.yaml`, and `ARCHITECTURE.md` define the engineering contracts and data flow. See `references.md` for source URLs and what each source supports.
