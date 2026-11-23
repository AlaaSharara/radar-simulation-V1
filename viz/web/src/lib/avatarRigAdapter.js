import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { worldToThree } from './coords.js';

const JOINT_ORDER = [
  'pelvis',
  'spine',
  'chest',
  'neck',
  'head',
  'left_shoulder',
  'left_elbow',
  'left_wrist',
  'right_shoulder',
  'right_elbow',
  'right_wrist',
  'left_hip',
  'left_knee',
  'left_ankle',
  'right_hip',
  'right_knee',
  'right_ankle',
];

const JOINT_PARENTS = {
  pelvis: null,
  spine: 'pelvis',
  chest: 'spine',
  neck: 'chest',
  head: 'neck',
  left_shoulder: 'chest',
  left_elbow: 'left_shoulder',
  left_wrist: 'left_elbow',
  right_shoulder: 'chest',
  right_elbow: 'right_shoulder',
  right_wrist: 'right_elbow',
  left_hip: 'pelvis',
  left_knee: 'left_hip',
  left_ankle: 'left_knee',
  right_hip: 'pelvis',
  right_knee: 'right_hip',
  right_ankle: 'right_knee',
};

const CHILD_AIM = {
  pelvis: 'spine',
  spine: 'chest',
  chest: 'neck',
  neck: 'head',
  left_shoulder: 'left_elbow',
  left_elbow: 'left_wrist',
  right_shoulder: 'right_elbow',
  right_elbow: 'right_wrist',
  left_hip: 'left_knee',
  left_knee: 'left_ankle',
  right_hip: 'right_knee',
  right_knee: 'right_ankle',
};

const DOWN_Y = new THREE.Vector3(0, -1, 0);

/**
 * Load the rigged adult male human GLB avatar and bind it to an adapter that
 * updates its hidden 17-joint skeleton from backend `MotionFrame.joints_m`.
 * Never renders bones, stick figures, capsules, or joint markers.
 */
export async function loadHumanAvatar(glbUrl) {
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(glbUrl);
  const root = gltf.scene;
  const bones = {};

  root.traverse((obj) => {
    if (obj.isBone) {
      bones[obj.name] = obj;
    }
    if (obj.isSkinnedMesh) {
      obj.frustumCulled = false;
      obj.castShadow = true;
      obj.receiveShadow = true;
    }
  });

  // Reusable scratch objects (zero per-frame GC churn)
  const tmpPos = {};
  for (const name of JOINT_ORDER) tmpPos[name] = new THREE.Vector3();
  const vX = new THREE.Vector3();
  const vY = new THREE.Vector3();
  const vZ = new THREE.Vector3();
  const vDir = new THREE.Vector3();
  const mBasis = new THREE.Matrix4();
  const worldPosMap = {};
  const worldQuatMap = {};
  for (const name of JOINT_ORDER) {
    worldPosMap[name] = new THREE.Vector3();
    worldQuatMap[name] = new THREE.Quaternion();
  }
  const qDelta = new THREE.Quaternion();
  const qParentInv = new THREE.Quaternion();
  const vLocal = new THREE.Vector3();

  function buildOrthonormalQuat(upVec, rightHint, targetQuat) {
    vY.copy(upVec).normalize();
    vZ.crossVectors(rightHint, vY);
    if (vZ.lengthSq() < 1e-8) {
      vZ.set(0, 0, 1);
    } else {
      vZ.normalize();
    }
    vX.crossVectors(vY, vZ).normalize();
    mBasis.makeBasis(vX, vY, vZ);
    targetQuat.setFromRotationMatrix(mBasis);
    return targetQuat;
  }

  function updateFromJoints(jointsM) {
    if (!jointsM || !jointsM.pelvis) return;

    for (const name of JOINT_ORDER) {
      const pt = jointsM[name];
      if (!pt) return;
      tmpPos[name].copy(worldToThree(pt));
      worldPosMap[name].copy(tmpPos[name]);
    }

    // Torso reference right vector (left_hip -> right_hip and left_shoulder -> right_shoulder)
    const hipRight = new THREE.Vector3().subVectors(tmpPos.right_hip, tmpPos.left_hip);
    const shRight = new THREE.Vector3().subVectors(tmpPos.right_shoulder, tmpPos.left_shoulder);

    // 1. Torso spine chain world quaternions
    const pelvisUp = new THREE.Vector3().subVectors(tmpPos.spine, tmpPos.pelvis);
    buildOrthonormalQuat(pelvisUp, hipRight, worldQuatMap.pelvis);

    const spineUp = new THREE.Vector3().subVectors(tmpPos.chest, tmpPos.spine);
    buildOrthonormalQuat(spineUp, shRight, worldQuatMap.spine);

    const chestUp = new THREE.Vector3().subVectors(tmpPos.neck, tmpPos.chest);
    buildOrthonormalQuat(chestUp, shRight, worldQuatMap.chest);

    const neckUp = new THREE.Vector3().subVectors(tmpPos.head, tmpPos.neck);
    buildOrthonormalQuat(neckUp, shRight, worldQuatMap.neck);
    worldQuatMap.head.copy(worldQuatMap.neck);

    // 2. Limb world quaternions (bind pose limbs extend along -Y in torso/pelvis space)
    const limbDefs = [
      ['left_shoulder', 'chest'],
      ['left_elbow', 'left_shoulder'],
      ['right_shoulder', 'chest'],
      ['right_elbow', 'right_shoulder'],
      ['left_hip', 'pelvis'],
      ['left_knee', 'left_hip'],
      ['right_hip', 'pelvis'],
      ['right_knee', 'right_hip'],
    ];
    for (const [jName, refName] of limbDefs) {
      const childName = CHILD_AIM[jName];
      vDir.subVectors(tmpPos[childName], tmpPos[jName]);
      if (vDir.lengthSq() > 1e-8) {
        vDir.normalize();
        const refDown = DOWN_Y.clone().applyQuaternion(worldQuatMap[refName]).normalize();
        qDelta.setFromUnitVectors(refDown, vDir);
        worldQuatMap[jName].copy(qDelta).multiply(worldQuatMap[refName]);
      } else {
        worldQuatMap[jName].copy(worldQuatMap[refName]);
      }
    }

    // Leaf extremities inherit forearm / shin orientation cleanly
    worldQuatMap.left_wrist.copy(worldQuatMap.left_elbow);
    worldQuatMap.right_wrist.copy(worldQuatMap.right_elbow);
    worldQuatMap.left_ankle.copy(worldQuatMap.left_knee);
    worldQuatMap.right_ankle.copy(worldQuatMap.right_knee);

    // 3. Apply in strict parent-to-child hierarchical order to Three.js Bones
    for (const name of JOINT_ORDER) {
      const bone = bones[name];
      if (!bone) continue;
      const parentName = JOINT_PARENTS[name];
      if (!parentName) {
        bone.position.copy(worldPosMap[name]);
        bone.quaternion.copy(worldQuatMap[name]);
      } else {
        qParentInv.copy(worldQuatMap[parentName]).invert();
        vLocal.subVectors(worldPosMap[name], worldPosMap[parentName]).applyQuaternion(qParentInv);
        bone.position.copy(vLocal);
        bone.quaternion.copy(qParentInv).multiply(worldQuatMap[name]);
      }
    }

    root.updateMatrixWorld(true);
  }

  return { root, bones, updateFromJoints };
}
