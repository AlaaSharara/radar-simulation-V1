import * as THREE from 'three';
import {
  buildRadarPolarGrid,
  buildRealisticRoomGroup,
  worldDirToThreeVec,
  worldToThreeVec,
} from './sceneBuilder.js';

/**
 * Convert simulation world Z-up coordinates [m] (+X right, +Y into room, +Z up)
 * to Three.js / glTF right-handed Y-up coordinates (+X right, +Y up, -Z forward).
 */
export function worldToThree(pt) {
  return worldToThreeVec(pt);
}

export { buildRadarPolarGrid, worldDirToThreeVec };

/**
 * Build reusable 3D room shell, openings, furniture, and radar mount strictly
 * from the backend's normalized `scene` payload (zero hardcoded dimensions).
 */
export function buildRoomGroup(sceneData, { wireframeOnly = false, showFovCone = false } = {}) {
  if (!wireframeOnly) {
    return buildRealisticRoomGroup(sceneData, { showFovCone });
  }

  const group = new THREE.Group();
  if (!sceneData || !sceneData.room) return group;

  const { width_x_m: w, depth_y_m: d, height_z_m: h } = sceneData.room;

  // Room wireframe bounding box (world [0..w, 0..d, 0..h] -> Three [0..w, 0..h, -d..0])
  const boxGeo = new THREE.BoxGeometry(w, h, d);
  const edges = new THREE.EdgesGeometry(boxGeo);
  const wireMat = new THREE.LineBasicMaterial({
    color: 0x475569,
    transparent: true,
    opacity: 0.55,
  });
  const wire = new THREE.LineSegments(edges, wireMat);
  wire.position.set(w / 2, h / 2, -d / 2);
  group.add(wire);

  // Dark reference floor + subtle polar range/azimuth rings
  const floorGeo = new THREE.PlaneGeometry(w, d);
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x0b1220,
    roughness: 0.95,
    metalness: 0.05,
    transparent: true,
    opacity: 0.72,
    side: THREE.DoubleSide,
  });
  const floorMesh = new THREE.Mesh(floorGeo, floorMat);
  floorMesh.rotation.x = -Math.PI / 2;
  floorMesh.position.set(w / 2, 0, -d / 2);
  group.add(floorMesh);
  group.add(buildRadarPolarGrid(sceneData, { maxRangeM: Math.max(4.5, Math.hypot(w, d) * 0.8) }));

  // Subtle translucent furniture wireframes for spatial context
  for (const obj of sceneData.objects || []) {
    const [dx, dy, dz] = obj.dimensions_m;
    const center = worldToThree(obj.box.center_m);
    const geo = new THREE.BoxGeometry(dx, dz, dy);
    const wireObj = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x475569, transparent: true, opacity: 0.52 })
    );
    wireObj.position.copy(center);
    wireObj.rotation.y = obj.yaw_rad || 0;
    group.add(wireObj);
  }

  // Radar sensor marker + boresight arrow
  if (sceneData.radar && sceneData.radar.frame) {
    const rPos = worldToThree(sceneData.radar.frame.position_m);
    const bDir = worldDirToThreeVec(sceneData.radar.frame.boresight_world);

    const sensorMesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.14, 0.10, 0.06),
      new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0x7f1d1d, roughness: 0.35 })
    );
    sensorMesh.position.copy(rPos);
    sensorMesh.quaternion.copy(
      new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), bDir)
    );
    group.add(sensorMesh);

    const arrow = new THREE.ArrowHelper(bDir, rPos, 0.65, 0xef4444, 0.14, 0.08);
    group.add(arrow);
  }

  return group;
}
