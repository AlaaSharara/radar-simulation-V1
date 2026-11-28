# Room simulation task (furniture RF response + preview window) - Alaa

## Task
"Double check that the furniture responds to the waves like real life, enhance the preview window, any method/library allowed, multi-angle preview encouraged."

## Findings (old model)
- Furniture = solid 6-face boxes; wood slab 50 mm, fabric slab 10 mm.
- 10 mm fabric (eps 1.5) is an anti-reflection layer at 61 GHz: sofa/bed top |R| ~ -56 dB => practically invisible.
- Tables/chairs blocked rays like solid blocks; a seated person's hips were "inside" the sofa box (no LOS in preview mode).
- Bottom faces duplicated the floor; wardrobe was a 50 mm solid instead of hollow 18 mm board.
- Diffuse scatter budget (12 largest surfaces) never selected any furniture.

## Changes
Backend
- src/radarsim/scene/furniture_rf.py (new): panel models for bed, sofa, table, chair, wardrobe (hollow carcass, open underneath, foam upholstery on wood deck). Unknown kinds fall back to the old box.
- material_db.yaml: `chipboard` (ITU-R P.2040-3: eps 2.58, sigma 0.0217 f^0.78) and `foam` (assumption, low confidence).
- rf/engine.py + rf/types.py: separate diffuse budget for furniture (max_diffuse_furniture_surfaces).
- scene/model.py: furniture surfaces in the scene payload carry `rf` (|R| 0/45/70 deg, |T|, roughness).
- scripts/furniture_rf_check.py -> reports/furniture_rf_check.png/.md (4 angles + per-panel table).
- tests: tests/scene/test_furniture_rf.py (new), tests/scene/test_scene.py updated.
Frontend (viz/web)
- ScenePanel.jsx: 8 animated camera presets, Quad view (perspective+top+front+side), Follow camera, auto wall cut-away, RF overlay / RF-only, click-to-inspect panel, labels, PNG screenshot.
- sceneBuilder.js / coords.js / style.css: supporting code; bed mattress top now matches the RF model.

## Limits
- Image-method reflections cannot represent features < first Fresnel zone (legs, handles, slats): omitted.
- Foam values and inner-spring coils are assumptions; no hardware validation.
- Frontend was syntax-checked only (no three.js / vite available offline): run `npm install && npm run dev` and test.
- Pre-existing failures (also on the original zip): test_motion hard-negative test; CLI test needs fastapi.
