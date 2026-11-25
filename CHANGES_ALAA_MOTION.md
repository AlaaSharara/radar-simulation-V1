# Motion enhancement - Alaa (branch: alaa-motion)

Files changed
- src/radarsim/motion/scenario.py   - motion generator
- src/radarsim/motion/navigation.py - planner (open door leaf is now an obstacle)
- src/radarsim/viz/session.py       - random seed per run, new clip at loop wrap, speed>1 skips frames, clip time
- src/radarsim/viz/server.py        - speed>1 keeps stream rate
- viz/web/src/components/ScenePanel.jsx - avatar interpolated at display rate, "clip t/T s" readout
- motion_config.yaml                - walking speed 0.85-1.40 m/s, fall descent 0.5-1.0 s
- check_dur.py, pen_scan.py         - verification scripts (duration/smoothness, furniture + door penetration)

What changed
1. Clip length: every clip is 10-20 s (time-budgeted generator + deterministic retries). Same seed = same clip.
2. Variety: random seed on every dashboard start; a NEW random clip at every loop wrap; an explicit seed
   (CLI --seed / Seed box) keeps that clip looping. Each clip has >= 3 different kinds of action besides 'enter'.
3. Smoothness: trapezoidal walking speed profile (no stop at corners), rate-limited yaw, root speed cap in
   transitions (max accel 150 -> ~6 m/s^2), avatar blended between backend frames in the browser.
4. Collisions: bend/kneel face open floor with extra clearance; lying uses edge-sit -> swing legs up -> lie
   (and the reverse), so legs no longer pass through the bed frame; open door leaf is an obstacle; entry is
   shifted away from the hinge; falls are placed clear of the leaf.
5. Speed menu: >1 now really plays faster (skips motion frames); <1 slows the stream.

Verify
  set PYTHONPATH=src ; python check_dur.py   (duration, smoothness)
  set PYTHONPATH=src ; python pen_scan.py    (penetration scan)
Known: pytest was not run in the authoring environment - run `pytest tests/motion`.
