# Avatar asset instructions

The Scene panel must show a complete, recognizable human avatar — not a skeleton.

Preferred implementation:

1. Use a local rigged GLB human model with a documented license allowing the intended project use.
2. Record the asset name, source URL, author, license and attribution in `assets/avatars/ATTRIBUTION.md`.
3. Map the avatar rig to the 17 joint names in `INTERFACES.md`.
4. Use a Three.js `AnimationMixer` or a documented custom rig adapter to animate walking, sitting, standing, bending, falling and lying/sleeping.
5. Keep the internal bones hidden. Do not show bone lines, stick figures, capsules or joint markers as the visible person.

Fallback if a suitable asset cannot be downloaded: create a procedural skinned humanoid mesh with a head, neck, torso, shoulders, arms, hands, hips, upper/lower legs, feet and clothing/materials. It can be stylized, but it must be visually a person.

The avatar is **ground truth**. It must not be reused as the radar point cloud or occupancy reconstruction.
