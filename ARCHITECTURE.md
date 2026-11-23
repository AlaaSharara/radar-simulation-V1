# Architecture — how the pieces fit together

```text
scene.json + material_db.yaml
          |
          v
   3D ROOM / FURNITURE <-------------------
          |                               |
          v                               |
   HUMAN MOTION (100 Hz)                  |
   - visible avatar pose                  |
   - hidden 17-joint rig                  |
   - collision/contact proxies            |
          |                               |
          v                               |
   RF PROPAGATION                         |
   - LOS / occlusion                      |
   - Fresnel reflections                  |
   - diffuse-scatter approximation        |
          |                               |
          v                               |
   FMCW / TDM-MIMO FRONT-END              |
   - per-TX / per-RX beat signals         |
   - impairments + noise                  |
   - ADC cube                             |
          |                               |
          v                               |
   DSP                                    |
   - range / Doppler / angle              |
   - CFAR / tracks / point cloud          |
   - baseline fall alert                  |
          |                               |
          v                               |
   SYNCHRONIZED FRAME MESSAGE             |
          |                               |
          v                               |
   FOUR-PANEL DASHBOARD                   |
   1. Ground-truth human avatar           |
   2. RF paths / reflections              |
   3. Radar-derived point cloud           |
   4. Range-Doppler + analytics           |
```

## Critical separation

- The **Scene panel** shows the true simulated person and room.
- The **RF panel** shows the propagation paths computed from geometry/materials.
- The **Point-cloud panel** shows only detections created by the radar signal-processing chain.
- The **Analytics panel** shows DSP-derived evidence and ground-truth labels separately.

All panels share `frame_id` and `sim_time_s`. The frontend is a viewer/controller, not a second physics engine.
