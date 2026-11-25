import * as THREE from 'three';

/**
 * Convert simulation world Z-up coordinates [m] (+X right, +Y into room, +Z up)
 * to Three.js / glTF right-handed Y-up coordinates (+X right, +Y up, -Z forward).
 */
export function worldToThreeVec(pt) {
  return new THREE.Vector3(pt[0], pt[2], -pt[1]);
}

export function worldDirToThreeVec(dir) {
  return new THREE.Vector3(dir[0], dir[2], -dir[1]).normalize();
}

function makeWoodFloorTexture(widthM, depthM) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#6c5138';
  ctx.fillRect(0, 0, 512, 512);

  const plankRows = 8;
  const plankH = 512 / plankRows;
  for (let r = 0; r < plankRows; r++) {
    const y = r * plankH;
    const offset = (r % 2) * 180 + ((r * 73) % 90);
    const tones = ['#73563c', '#6a4e36', '#7b5e43', '#634932'];
    ctx.fillStyle = tones[r % tones.length];
    ctx.fillRect(0, y, 512, plankH - 1.5);

    // Subtle wood grain streaks
    ctx.strokeStyle = 'rgba(42, 26, 14, 0.16)';
    ctx.lineWidth = 1;
    for (let g = 0; g < 12; g++) {
      const gy = y + 4 + ((g * 5.2) % (plankH - 8));
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.bezierCurveTo(170, gy + ((g % 3) - 1) * 1.4, 340, gy - ((g % 2) - 0.5) * 1.4, 512, gy);
      ctx.stroke();
    }

    // Horizontal bevel seam + vertical plank joints
    ctx.fillStyle = 'rgba(20, 14, 10, 0.42)';
    ctx.fillRect(0, y + plankH - 1.5, 512, 1.5);
    const xJoint = (offset + 256) % 512;
    ctx.fillRect(xJoint, y, 1.5, plankH);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(Math.max(1, widthM / 1.3), Math.max(1, depthM / 1.3));
  return tex;
}

function makePlasterTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#334155';
  ctx.fillRect(0, 0, 256, 256);

  for (let i = 0; i < 2400; i++) {
    const x = (i * 97) % 256;
    const y = (i * 53 + Math.floor(i / 256) * 17) % 256;
    const v = 52 + (i % 18);
    ctx.fillStyle = `rgba(${v}, ${v + 12}, ${v + 28}, 0.22)`;
    ctx.fillRect(x, y, 2, 2);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(4, 2);
  return tex;
}

function makeFabricTexture(baseHex, weaveHex) {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = baseHex;
  ctx.fillRect(0, 0, 128, 128);
  ctx.fillStyle = weaveHex;
  for (let y = 0; y < 128; y += 4) {
    ctx.fillRect(0, y, 128, 1);
  }
  for (let x = 0; x < 128; x += 4) {
    ctx.fillRect(x, 0, 1, 128);
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3, 3);
  return tex;
}

function makeWoodGrainTexture(baseHex, grainHex) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = baseHex;
  ctx.fillRect(0, 0, 256, 256);
  ctx.strokeStyle = grainHex;
  ctx.lineWidth = 1.4;
  for (let y = 2; y < 256; y += 6) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.bezierCurveTo(85, y + 2, 170, y - 2, 256, y);
    ctx.stroke();
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

function makeRadarFaceplateTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 192;
  const ctx = canvas.getContext('2d');

  // Matte RF PCB dark green background + white ceramic radome window
  ctx.fillStyle = '#0f3d2e';
  ctx.fillRect(0, 0, 256, 192);
  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 4;
  ctx.strokeRect(4, 4, 248, 184);

  // White Rogers antenna substrate window
  ctx.fillStyle = '#e2e8f0';
  ctx.fillRect(26, 24, 204, 114);

  // 4 RX patch columns (bottom-left array) + 3 TX elements (stepped ODS layout)
  ctx.fillStyle = '#d97706';
  for (let i = 0; i < 4; i++) {
    const x = 44 + i * 24;
    ctx.fillRect(x, 82, 14, 34);
  }
  // TX0, TX1 (elevated), TX2
  ctx.fillRect(156, 82, 14, 34);
  ctx.fillRect(178, 42, 14, 34);
  ctx.fillRect(200, 82, 14, 34);

  // Label silkscreen
  ctx.fillStyle = '#f8fafc';
  ctx.font = 'bold 16px monospace';
  ctx.fillText('60GHz 3x4 MIMO', 48, 166);
  ctx.fillStyle = '#22c55e';
  ctx.beginPath();
  ctx.arc(28, 160, 5, 0, Math.PI * 2);
  ctx.fill();

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function addBoxPart(parent, w, h, d, x, y, z, material, { castShadow = true, receiveShadow = true } = {}) {
  const geo = new THREE.BoxGeometry(w, h, d);
  const mesh = new THREE.Mesh(geo, material);
  mesh.position.set(x, y, z);
  mesh.castShadow = castShadow;
  mesh.receiveShadow = receiveShadow;
  parent.add(mesh);
  return mesh;
}

function buildFurnitureMesh(obj, sceneData, mats) {
  const [dx, dy, dz] = obj.dimensions_m;
  const g = new THREE.Group();
  g.position.copy(worldToThreeVec(obj.box.center_m));
  g.rotation.y = obj.yaw_rad || 0;

  // Local coordinate extents inside OrientedBox:
  // x in [-dx/2, +dx/2], y in [-dz/2, +dz/2], z in [-dy/2, +dy/2] (note: -z is World +Y)
  const yFloor = -dz / 2;
  const kind = obj.kind;

  if (kind === 'bed') {
    const legH = Math.min(0.10, dz * 0.18);
    const frameH = Math.min(0.16, dz * 0.28);
    const mattH = Math.max(0.14, dz * 0.42);
    const headH = dz;
    const headT = Math.min(0.08, dy * 0.06);

    // 4 short legs
    const legW = 0.06;
    for (const sx of [-dx / 2 + legW, dx / 2 - legW]) {
      for (const sz of [-dy / 2 + legW, dy / 2 - legW]) {
        addBoxPart(g, legW, legH, legW, sx, yFloor + legH / 2, sz, mats.walnut);
      }
    }
    // Wood platform frame
    addBoxPart(g, dx, frameH, dy - headT * 0.5, 0, yFloor + legH + frameH / 2, headT * 0.25, mats.walnut);
    // Headboard at back wall (-z = +Y world)
    addBoxPart(g, dx, headH, headT, 0, yFloor + headH / 2, -dy / 2 + headT / 2, mats.walnut);
    // Quilted mattress
    const mW = dx - 0.06;
    const mD = dy - headT - 0.04;
    const mY = yFloor + legH + frameH + mattH / 2;
    addBoxPart(g, mW, mattH, mD, 0, mY, headT / 2, mats.linen);
    // Folded slate-blue blanket across foot 64%
    const blD = mD * 0.64;
    const blTop = mY + mattH / 2 + 0.015;
    addBoxPart(g, mW + 0.015, 0.03, blD, 0, blTop - 0.01, dy / 2 - blD / 2 - 0.02, mats.bedding);
    // 2 pillows near headboard
    const pCount = dx > 1.0 ? 2 : 1;
    const pW = (mW - 0.14) / pCount;
    for (let i = 0; i < pCount; i++) {
      const px = pCount === 2 ? (i === 0 ? -mW * 0.24 : mW * 0.24) : 0;
      addBoxPart(g, pW, 0.08, 0.30, px, mY + mattH / 2 + 0.04, -dy / 2 + headT + 0.20, mats.pillow);
    }
  } else if (kind === 'sofa') {
    const legH = Math.min(0.08, dz * 0.10);
    const baseH = dz * 0.22;
    const seatH = dz * 0.18;
    const armW = Math.min(0.16, dx * 0.12);
    const backT = Math.min(0.20, dy * 0.26);

    for (const sx of [-dx / 2 + 0.06, dx / 2 - 0.06]) {
      for (const sz of [-dy / 2 + 0.06, dy / 2 - 0.06]) {
        addBoxPart(g, 0.045, legH, 0.045, sx, yFloor + legH / 2, sz, mats.walnut);
      }
    }
    // Lower deck
    addBoxPart(g, dx, baseH, dy, 0, yFloor + legH + baseH / 2, 0, mats.sofaBody);
    // Left & right armrests
    const armH = dz * 0.42;
    addBoxPart(g, armW, armH, dy, -dx / 2 + armW / 2, yFloor + legH + baseH + armH / 2, 0, mats.sofaBody);
    addBoxPart(g, armW, armH, dy, dx / 2 - armW / 2, yFloor + legH + baseH + armH / 2, 0, mats.sofaBody);
    // Backrest (at -z = +Y world)
    const backH = dz - legH - baseH;
    addBoxPart(g, dx - 2 * armW, backH, backT, 0, yFloor + legH + baseH + backH / 2, -dy / 2 + backT / 2, mats.sofaBody);
    // Individual segmented seat & back cushions
    const innerW = dx - 2 * armW - 0.02;
    const seatD = dy - backT - 0.02;
    const cCount = Math.max(2, Math.round(innerW / 0.48));
    const cw = (innerW - (cCount - 1) * 0.015) / cCount;
    for (let i = 0; i < cCount; i++) {
      const cx = -innerW / 2 + cw / 2 + i * (cw + 0.015);
      addBoxPart(g, cw, seatH, seatD, cx, yFloor + legH + baseH + seatH / 2, backT / 2, mats.sofaCushion);
      addBoxPart(g, cw, backH * 0.72, 0.11, cx, yFloor + legH + baseH + seatH + backH * 0.36, -dy / 2 + backT + 0.06, mats.sofaCushion);
    }
  } else if (kind === 'table') {
    const topT = Math.min(0.045, dz * 0.12);
    const apronH = Math.min(0.055, dz * 0.14);
    const legH = dz - topT;
    const legW = 0.045;
    // Tabletop
    addBoxPart(g, dx, topT, dy, 0, dz / 2 - topT / 2, 0, mats.oakFurniture);
    // Apron & 4 legs
    addBoxPart(g, dx - 0.08, apronH, dy - 0.08, 0, dz / 2 - topT - apronH / 2, 0, mats.oakFurniture);
    for (const sx of [-dx / 2 + legW, dx / 2 - legW]) {
      for (const sz of [-dy / 2 + legW, dy / 2 - legW]) {
        addBoxPart(g, legW, legH, legW, sx, yFloor + legH / 2, sz, mats.oakFurniture);
      }
    }
  } else if (kind === 'chair') {
    const seatY = yFloor + dz * 0.46;
    const seatT = 0.045;
    const padT = 0.035;
    const legW = 0.038;
    const legH = dz * 0.46;
    for (const sx of [-dx / 2 + legW, dx / 2 - legW]) {
      for (const sz of [-dy / 2 + legW, dy / 2 - legW]) {
        addBoxPart(g, legW, legH, legW, sx, yFloor + legH / 2, sz, mats.oakFurniture);
      }
    }
    addBoxPart(g, dx - 0.02, seatT, dy - 0.02, 0, seatY, 0, mats.oakFurniture);
    addBoxPart(g, dx - 0.05, padT, dy - 0.05, 0, seatY + (seatT + padT) / 2, 0.01, mats.chairPad);
    // Backrest posts + 3 slats at -z
    const postH = dz * 0.54;
    for (const sx of [-dx / 2 + legW, dx / 2 - legW]) {
      addBoxPart(g, legW, postH, legW, sx, seatY + postH / 2, -dy / 2 + legW, mats.oakFurniture);
    }
    for (const frac of [0.35, 0.65, 0.90]) {
      addBoxPart(g, dx - 0.04, 0.06, 0.025, 0, seatY + postH * frac, -dy / 2 + legW, mats.oakFurniture);
    }
  } else if (kind === 'wardrobe') {
    const plinthH = 0.07;
    const bodyH = dz - plinthH;
    // Determine which side faces into the room (-X if sitting near x_max, etc.)
    const roomW = sceneData?.room?.width_x_m || 4.0;
    const roomD = sceneData?.room?.depth_y_m || 5.0;
    const cxWorld = obj.box.center_m[0];
    const cyWorld = obj.box.center_m[1];
    const frontAlongX = dx < dy;

    addBoxPart(g, dx * 0.94, plinthH, dy * 0.96, 0, yFloor + plinthH / 2, 0, mats.metalDark);
    addBoxPart(g, dx, bodyH, dy, 0, yFloor + plinthH + bodyH / 2, 0, mats.walnut);

    if (frontAlongX) {
      const signX = cxWorld > roomW * 0.5 ? -1 : 1;
      const doorFaceX = signX * (dx / 2 + 0.008);
      const panelW = dy * 0.485;
      addBoxPart(g, 0.014, bodyH - 0.04, panelW, doorFaceX, yFloor + plinthH + bodyH / 2, -panelW / 2 - 0.005, mats.walnutTrim);
      addBoxPart(g, 0.014, bodyH - 0.04, panelW, doorFaceX, yFloor + plinthH + bodyH / 2, panelW / 2 + 0.005, mats.walnutTrim);
      // Brushed metal handles
      addBoxPart(g, 0.02, 0.24, 0.016, doorFaceX + signX * 0.014, 0.05, -0.045, mats.metalLight);
      addBoxPart(g, 0.02, 0.24, 0.016, doorFaceX + signX * 0.014, 0.05, 0.045, mats.metalLight);
    } else {
      const signZ = cyWorld > roomD * 0.5 ? 1 : -1;
      const doorFaceZ = signZ * (dy / 2 + 0.008);
      const panelW = dx * 0.485;
      addBoxPart(g, panelW, bodyH - 0.04, 0.014, -panelW / 2 - 0.005, yFloor + plinthH + bodyH / 2, doorFaceZ, mats.walnutTrim);
      addBoxPart(g, panelW, bodyH - 0.04, 0.014, panelW / 2 + 0.005, yFloor + plinthH + bodyH / 2, doorFaceZ, mats.walnutTrim);
      addBoxPart(g, 0.016, 0.24, 0.02, -0.045, 0.05, doorFaceZ + signZ * 0.014, mats.metalLight);
      addBoxPart(g, 0.016, 0.24, 0.02, 0.045, 0.05, doorFaceZ + signZ * 0.014, mats.metalLight);
    }
  } else {
    addBoxPart(g, dx, dz, dy, 0, 0, 0, mats.oakFurniture);
  }

  return g;
}

function buildRadarSensorAssembly(sceneData, showFovCone) {
  const group = new THREE.Group();
  if (!sceneData?.radar?.frame) return group;

  const rPos = worldToThreeVec(sceneData.radar.frame.position_m);
  const bDir = worldDirToThreeVec(sceneData.radar.frame.boresight_world);

  const housingGroup = new THREE.Group();
  housingGroup.position.copy(rPos);
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), bDir);
  housingGroup.quaternion.copy(q);

  // Dark aluminum sensor chassis + PCB patch antenna faceplate facing +bDir (+Z local)
  const faceTex = makeRadarFaceplateTexture();
  const chassisMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.4, metalness: 0.65 });
  const pcbMat = new THREE.MeshStandardMaterial({ map: faceTex, roughness: 0.35, metalness: 0.25 });
  const chassis = new THREE.Mesh(
    new THREE.BoxGeometry(0.18, 0.13, 0.036),
    [chassisMat, chassisMat, chassisMat, chassisMat, pcbMat, chassisMat]
  );
  chassis.castShadow = true;
  housingGroup.add(chassis);

  // Mounting bracket to wall
  const bracket = new THREE.Mesh(
    new THREE.BoxGeometry(0.07, 0.05, 0.05),
    new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8, roughness: 0.3 })
  );
  bracket.position.set(0, 0, -0.04);
  housingGroup.add(bracket);

  // Subtle boresight vector arrow
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0.025), 0.55, 0xf87171, 0.12, 0.06);
  housingGroup.add(arrow);

  // Toggleable translucent FoV frustum cone (~120 deg total width -> 60 deg half-angle, clipped to 4.2m)
  const fovGroup = new THREE.Group();
  fovGroup.name = 'radar_fov_cone';
  fovGroup.visible = Boolean(showFovCone);
  const rangeFov = 4.2;
  const radiusFov = rangeFov * Math.tan((52 * Math.PI) / 180);
  const coneGeo = new THREE.ConeGeometry(radiusFov, rangeFov, 24, 1, true);
  // Rotate so apex is at origin and opens along local +Z
  coneGeo.translate(0, -rangeFov / 2, 0);
  coneGeo.rotateX(-Math.PI / 2);
  const coneMat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.06,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  fovGroup.add(new THREE.Mesh(coneGeo, coneMat));
  const coneEdges = new THREE.LineSegments(
    new THREE.EdgesGeometry(coneGeo, 24),
    new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.22 })
  );
  fovGroup.add(coneEdges);
  housingGroup.add(fovGroup);

  group.add(housingGroup);
  return group;
}

export function buildRealisticRoomGroup(sceneData, { showFovCone = false } = {}) {
  const group = new THREE.Group();
  if (!sceneData || !sceneData.room) return group;

  const { width_x_m: w, depth_y_m: d, height_z_m: h } = sceneData.room;
  const wallT = sceneData.room.wall_thickness_m || 0.1143;

  // Room wireframe ceiling outline
  const boxGeo = new THREE.BoxGeometry(w, h, d);
  const wire = new THREE.LineSegments(
    new THREE.EdgesGeometry(boxGeo),
    new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.35 })
  );
  wire.position.set(w / 2, h / 2, -d / 2);
  group.add(wire);

  // Procedural PBR materials
  const floorTex = makeWoodFloorTexture(w, d);
  const plasterTex = makePlasterTexture();
  const walnutTex = makeWoodGrainTexture('#5b3a24', 'rgba(32, 16, 8, 0.35)');
  const walnutTrimTex = makeWoodGrainTexture('#68432b', 'rgba(32, 16, 8, 0.25)');
  const oakTex = makeWoodGrainTexture('#a97748', 'rgba(84, 52, 24, 0.28)');

  const mats = {
    floor: new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.68, metalness: 0.04 }),
    wall: new THREE.MeshStandardMaterial({
      map: plasterTex,
      color: 0x9fb0c7,
      roughness: 0.90,
      metalness: 0.02,
      side: THREE.DoubleSide,
    }),
    baseboard: new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.6, metalness: 0.1 }),
    walnut: new THREE.MeshStandardMaterial({ map: walnutTex, roughness: 0.62, metalness: 0.06 }),
    walnutTrim: new THREE.MeshStandardMaterial({ map: walnutTrimTex, roughness: 0.56, metalness: 0.08 }),
    oakFurniture: new THREE.MeshStandardMaterial({ map: oakTex, roughness: 0.58, metalness: 0.05 }),
    linen: new THREE.MeshStandardMaterial({ map: makeFabricTexture('#e2e8f0', 'rgba(148,163,184,0.22)'), roughness: 0.92 }),
    bedding: new THREE.MeshStandardMaterial({ map: makeFabricTexture('#3b6288', 'rgba(30,41,59,0.25)'), roughness: 0.88 }),
    pillow: new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.86 }),
    sofaBody: new THREE.MeshStandardMaterial({ map: makeFabricTexture('#3c4b62', 'rgba(15,23,42,0.28)'), roughness: 0.92 }),
    sofaCushion: new THREE.MeshStandardMaterial({ map: makeFabricTexture('#4f627d', 'rgba(15,23,42,0.20)'), roughness: 0.88 }),
    chairPad: new THREE.MeshStandardMaterial({ map: makeFabricTexture('#64748b', 'rgba(15,23,42,0.22)'), roughness: 0.9 }),
    metalDark: new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.45, metalness: 0.6 }),
    metalLight: new THREE.MeshStandardMaterial({ color: 0xcbd5e1, roughness: 0.28, metalness: 0.85 }),
    rug: new THREE.MeshStandardMaterial({ map: makeFabricTexture('#475569', 'rgba(203,213,225,0.16)'), roughness: 0.96 }),
  };

  // Hardwood floor receiving soft shadows
  const floorMesh = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mats.floor);
  floorMesh.rotation.x = -Math.PI / 2;
  floorMesh.position.set(w / 2, 0, -d / 2);
  floorMesh.receiveShadow = true;
  group.add(floorMesh);

  // Living-zone woven rug underneath coffee table if table exists
  const tableObj = (sceneData.objects || []).find((o) => o.kind === 'table');
  if (tableObj) {
    const tc = worldToThreeVec(tableObj.box.center_m);
    const rugW = Math.min(w * 0.55, tableObj.dimensions_m[0] * 2.1);
    const rugD = Math.min(d * 0.35, tableObj.dimensions_m[1] * 2.4);
    const rugMesh = new THREE.Mesh(new THREE.BoxGeometry(rugW, 0.008, rugD), mats.rug);
    rugMesh.position.set(tc.x, 0.004, tc.z);
    rugMesh.receiveShadow = true;
    group.add(rugMesh);
  }

  // Architectural cutaway plaster walls: back wall (y = d -> z = -d) and left wall (x = 0)
  const backWall = new THREE.Mesh(new THREE.BoxGeometry(w, h, wallT * 0.5), mats.wall);
  backWall.position.set(w / 2, h / 2, -d - wallT * 0.25);
  backWall.receiveShadow = true;
  group.add(backWall);

  const leftWall = new THREE.Mesh(new THREE.BoxGeometry(wallT * 0.5, h, d + wallT * 0.5), mats.wall);
  leftWall.position.set(-wallT * 0.25, h / 2, -d / 2);
  leftWall.receiveShadow = true;
  group.add(leftWall);

  // Subtle architectural baseboards on back & left cutaway walls
  addBoxPart(group, w, 0.09, 0.02, w / 2, 0.045, -d + 0.01, mats.baseboard, { castShadow: false });
  addBoxPart(group, 0.02, 0.09, d, 0.01, 0.045, -d / 2, mats.baseboard, { castShadow: false });

  // Cutaway floor-level curb guides along front (y=0) & right (x=w) walls
  addBoxPart(group, w, 0.04, 0.04, w / 2, 0.02, 0.02, mats.baseboard, { castShadow: false });
  addBoxPart(group, 0.04, 0.04, d, w + 0.02, 0.02, -d / 2, mats.baseboard, { castShadow: false });

  // Architectural openings (door frame + hinged wood leaf + brass lever; window frame + mullions + frosted glass)
  for (const op of sceneData.openings || []) {
    const ap = op.aperture_surface;
    if (ap && ap.vertices_m && ap.vertices_m.length === 4) {
      const av = ap.vertices_m.map(worldToThreeVec);
      const minX = Math.min(...av.map((v) => v.x));
      const maxX = Math.max(...av.map((v) => v.x));
      const minY = Math.min(...av.map((v) => v.y));
      const maxY = Math.max(...av.map((v) => v.y));
      const minZ = Math.min(...av.map((v) => v.z));
      const maxZ = Math.max(...av.map((v) => v.z));
      const cx = 0.5 * (minX + maxX);
      const cy = 0.5 * (minY + maxY);
      const cz = 0.5 * (minZ + maxZ);

      if (op.kind === 'door') {
        const frameThick = 0.055;
        const doorW = Math.max(0.2, op.width_m || 0.9);
        const doorH = Math.max(1.0, op.height_m || 2.1);
        // Header + side jambs around aperture
        addBoxPart(group, doorW + 2 * frameThick, frameThick, wallT * 0.7, cx, maxY + frameThick / 2, cz, mats.walnut);
        addBoxPart(group, frameThick, doorH, wallT * 0.7, minX - frameThick / 2, cy, cz, mats.walnut);
        addBoxPart(group, frameThick, doorH, wallT * 0.7, maxX + frameThick / 2, cy, cz, mats.walnut);
      } else if (op.kind === 'window') {
        const isOnYWall = op.wall === 'y_min' || op.wall === 'y_max';
        const winW = Math.max(0.3, op.width_m || 1.2);
        const winH = Math.max(0.3, op.height_m || 1.2);
        const frameT = 0.055;
        const winGroup = new THREE.Group();
        winGroup.position.set(cx, cy, cz + (op.wall === 'y_max' ? 0.025 : op.wall === 'y_min' ? -0.025 : 0));
        if (!isOnYWall) winGroup.rotation.y = Math.PI / 2;

        // Aluminum mullions & sill
        addBoxPart(winGroup, winW, frameT, 0.07, 0, winH / 2, 0, mats.metalDark, { castShadow: false });
        addBoxPart(winGroup, winW + 0.08, frameT * 1.2, 0.11, 0, -winH / 2, 0, mats.metalDark, { castShadow: false });
        addBoxPart(winGroup, frameT, winH, 0.07, -winW / 2, 0, 0, mats.metalDark, { castShadow: false });
        addBoxPart(winGroup, frameT, winH, 0.07, winW / 2, 0, 0, mats.metalDark, { castShadow: false });
        addBoxPart(winGroup, 0.032, winH - frameT, 0.05, 0, 0, 0, mats.metalDark, { castShadow: false });
        // Daylight window glass glow
        const glassMat = new THREE.MeshPhysicalMaterial({
          color: 0xbae6fd,
          emissive: 0x1e3a5f,
          emissiveIntensity: 0.45,
          transparent: true,
          opacity: 0.42,
          roughness: 0.08,
          metalness: 0.1,
          side: THREE.DoubleSide,
        });
        const glassPane = new THREE.Mesh(new THREE.PlaneGeometry(winW - frameT, winH - frameT), glassMat);
        winGroup.add(glassPane);
        group.add(winGroup);
      }
    }

    // Swinging door leaf + lever handle
    const panel = op.panel_surface;
    if (op.kind === 'door' && panel && panel.vertices_m && panel.vertices_m.length === 4) {
      const verts = panel.vertices_m.map(worldToThreeVec);
      const leafOrigin = verts[0];
      const uVec = new THREE.Vector3().subVectors(verts[1], verts[0]);
      const leafW = uVec.length();
      const leafH = new THREE.Vector3().subVectors(verts[3], verts[0]).length();
      const yaw = Math.atan2(-uVec.z, uVec.x);

      const leafGroup = new THREE.Group();
      leafGroup.position.copy(leafOrigin);
      leafGroup.rotation.y = yaw;
      addBoxPart(leafGroup, leafW, leafH, 0.042, leafW / 2, leafH / 2, 0, mats.walnutTrim);
      // Lever handle near latch edge
      addBoxPart(leafGroup, 0.11, 0.022, 0.09, leafW - 0.11, leafH * 0.46, 0, mats.metalLight);
      group.add(leafGroup);
    }
  }

  // Multi-part realistic furniture from scene objects
  for (const obj of sceneData.objects || []) {
    group.add(buildFurnitureMesh(obj, sceneData, mats));
  }

  // Realistic 60 GHz wall-mounted radar + optional FoV beam frustum
  group.add(buildRadarSensorAssembly(sceneData, showFovCone));

  return group;
}

/**
 * Create subtle radar-centered polar range rings [1m..5m] + +/- 60 deg azimuth spokes on the floor.
 */
export function buildRadarPolarGrid(sceneData, { maxRangeM = 5.0 } = {}) {
  const group = new THREE.Group();
  if (!sceneData?.radar?.frame) return group;

  const rPos = worldToThreeVec(sceneData.radar.frame.position_m);
  const bDir = worldDirToThreeVec(sceneData.radar.frame.boresight_world);
  const flatDir = new THREE.Vector3(bDir.x, 0, bDir.z).normalize();
  const baseYaw = Math.atan2(flatDir.x, flatDir.z);

  const origin = new THREE.Vector3(rPos.x, 0.005, rPos.z);
  const ringMat = new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.48 });

  for (let r = 1.0; r <= maxRangeM + 0.01; r += 1.0) {
    const pts = [];
    for (let deg = -60; deg <= 60; deg += 5) {
      const th = baseYaw + (deg * Math.PI) / 180;
      pts.push(new THREE.Vector3(origin.x + r * Math.sin(th), origin.y, origin.z + r * Math.cos(th)));
    }
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), ringMat));
  }

  for (const deg of [-60, -30, 0, 30, 60]) {
    const th = baseYaw + (deg * Math.PI) / 180;
    const end = new THREE.Vector3(
      origin.x + maxRangeM * Math.sin(th),
      origin.y,
      origin.z + maxRangeM * Math.cos(th)
    );
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([origin, end]), ringMat));
  }

  return group;
}
