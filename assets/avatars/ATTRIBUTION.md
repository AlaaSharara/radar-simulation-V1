# Avatar Asset Attribution & License

## Asset: `adult_male_default`

- **Asset ID:** `adult_male_default`
- **Files:**
  - `assets/avatars/adult_male_default.glb` — Binary glTF 2.0 rigged and skinned adult male-presenting human mesh with PBR materials (`human_skin`, `human_hair`, `casual_shirt`, `trousers`, `shoes`) and a hidden 17-joint skinning rig matching `INTERFACES.md`.
  - `assets/avatars/adult_male_default_rig.json` — Joint hierarchy, node indices, neutral bind pose in both simulation world Z-up and glTF Y-up coordinates, axis conversion matrix, and Three.js animation adapter metadata.
- **Origin / Author:** Procedurally generated within this repository by `src/radarsim/motion/avatar.py` (following the documented fallback policy in `assets/avatars/README.md` and `motion_config.yaml: actor.avatar_asset_policy` for reproducible offline builds without third-party licensing ambiguity).
- **Source URL:** Local procedural generator (`src/radarsim/motion/avatar.py`).
- **License:** MIT / Project-internal permissive license (zero third-party assets or external copyright restrictions).
- **Rendering rules:**
  - Render only the skinned clothed human mesh (`adult_male_default_mesh`) in the Scene panel.
  - Keep internal bones, joint markers, and collision/scattering capsules hidden (`draw_skeleton: false`, `hide_bones: true`).
  - Never feed avatar mesh vertices or joint ground truth into the radar point cloud.
