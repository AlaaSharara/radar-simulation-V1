import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { buildRoomGroup, worldToThree } from '../lib/coords.js';

function velocityToColor(vr) {
  // Approaching (< 0) -> cyan/blue, stationary (0) -> slate white, receding (> 0) -> amber/red
  const norm = Math.max(-1, Math.min(1, vr / 2.0));
  const c = new THREE.Color();
  if (norm < 0) {
    c.setRGB(0.28 + 0.68 * (1 + norm), 0.72 + 0.28 * (1 + norm), 1.0);
  } else {
    c.setRGB(1.0, 0.58 + 0.42 * (1 - norm), 0.25 + 0.75 * (1 - norm));
  }
  return c;
}

function snrToColor(snrDb) {
  const t = Math.max(0, Math.min(1, (snrDb - 10) / 25));
  const c = new THREE.Color();
  c.setHSL(0.58 - 0.52 * t, 0.92, 0.58);
  return c;
}

/**
 * Panel 3: Radar-only 3D point cloud, tracks + trail, and sparse occupancy voxels.
 * Ground-truth comparison overlay is strictly separate and defaults to OFF.
 */
export default function PointCloudPanel({ sceneData, frame }) {
  const containerRef = useRef(null);
  const dynamicGroupRef = useRef(null);
  const trackHistoryRef = useRef(new Map());
  const lastScenarioKeyRef = useRef('');
  const lastFidRef = useRef(-1);
  const [colorBy, setColorBy] = useState('velocity'); // 'velocity' | 'snr'
  const [showVoxels, setShowVoxels] = useState(true);
  const [showGtOverlay, setShowGtOverlay] = useState(false); // MUST default OFF per PLAN.md & INTERFACES.md

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x09101c);

    const w = sceneData?.room?.width_x_m || 4.0;
    const d = sceneData?.room?.depth_y_m || 5.0;
    const h = sceneData?.room?.height_z_m || 2.7;

    const camera = new THREE.PerspectiveCamera(46, el.clientWidth / Math.max(1, el.clientHeight), 0.05, 60);
    camera.position.set(w * 1.18, h * 1.12, d * 0.42);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.innerHTML = '';
    el.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(w / 2, h * 0.36, -d / 2);
    controls.update();

    scene.add(new THREE.AmbientLight(0xdbeafe, 0.9));
    if (sceneData) {
      scene.add(buildRoomGroup(sceneData, { wireframeOnly: true }));
    }

    const dynGroup = new THREE.Group();
    scene.add(dynGroup);
    dynamicGroupRef.current = dynGroup;

    const ro = new ResizeObserver(() => {
      if (!el) return;
      const cw = Math.max(1, el.clientWidth);
      const ch = Math.max(1, el.clientHeight);
      camera.aspect = cw / ch;
      camera.updateProjectionMatrix();
      renderer.setSize(cw, ch);
    });
    ro.observe(el);

    let rafId = 0;
    function renderLoop() {
      rafId = requestAnimationFrame(renderLoop);
      controls.update();
      renderer.render(scene, camera);
    }
    renderLoop();

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
    };
  }, [sceneData]);

  // Render strictly radar-derived detections, track history trails, and sparse voxels
  useEffect(() => {
    const dyn = dynamicGroupRef.current;
    if (!dyn) return;

    while (dyn.children.length > 0) {
      const child = dyn.children.pop();
      child.geometry?.dispose();
      child.material?.dispose();
    }

    const scenarioKey = `${frame?.seed ?? ''}:${frame?.control_state?.scenario_mode ?? ''}:${frame?.control_state?.mode ?? ''}`;
    if (scenarioKey !== lastScenarioKeyRef.current || (frame && frame.frame_id < lastFidRef.current)) {
      trackHistoryRef.current.clear();
      lastScenarioKeyRef.current = scenarioKey;
    }
    if (frame) {
      lastFidRef.current = frame.frame_id;
    }

    const detections = frame?.radar?.detections || [];
    const tracks = frame?.radar?.tracks || [];
    const occupancy = frame?.radar?.occupancy;

    // 1. Radar observations (OS-CFAR detections) with confidence scaling
    const detGeo = new THREE.SphereGeometry(0.06, 12, 12);
    for (const det of detections) {
      const col = colorBy === 'velocity' ? velocityToColor(det.radial_velocity_mps) : snrToColor(det.snr_db);
      const mat = new THREE.MeshBasicMaterial({ color: col });
      const mesh = new THREE.Mesh(detGeo.clone(), mat);
      const scale = 0.72 + 0.75 * Math.min(1, Math.max(0, det.confidence || 0.5));
      mesh.scale.setScalar(scale);
      mesh.position.copy(worldToThree([det.x_m, det.y_m, det.z_m]));
      dyn.add(mesh);
    }
    detGeo.dispose();

    // 2. Radar-derived sparse occupancy voxels
    if (showVoxels && occupancy && Array.isArray(occupancy.voxels)) {
      const vs = occupancy.voxel_size_m || 0.25;
      const voxGeo = new THREE.BoxGeometry(vs, vs, vs);
      const voxEdges = new THREE.EdgesGeometry(voxGeo);
      for (const v of occupancy.voxels) {
        const pos = worldToThree(v.center_m);
        const fillMat = new THREE.MeshBasicMaterial({
          color: 0x22d3ee,
          transparent: true,
          opacity: Math.max(0.12, Math.min(0.42, (v.probability || 0.3) * 0.45)),
        });
        const box = new THREE.Mesh(voxGeo.clone(), fillMat);
        box.position.copy(pos);
        dyn.add(box);

        const edgeMat = new THREE.LineBasicMaterial({ color: 0x67e8f9, transparent: true, opacity: 0.72 });
        const wire = new THREE.LineSegments(voxEdges.clone(), edgeMat);
        wire.position.copy(pos);
        dyn.add(wire);
      }
      voxGeo.dispose();
      voxEdges.dispose();
    }

    // 3. Radar tracks + rolling breadcrumb trail + velocity vector
    const activeIds = new Set();
    for (const trk of tracks) {
      activeIds.add(trk.track_id);
      const pos = worldToThree(trk.position_m);

      let hist = trackHistoryRef.current.get(trk.track_id);
      if (!hist) {
        hist = [];
        trackHistoryRef.current.set(trk.track_id, hist);
      }
      const prevPt = hist[hist.length - 1];
      if (!prevPt || prevPt.distanceToSquared(pos) > 1e-4) {
        hist.push(pos.clone());
        if (hist.length > 48) hist.shift();
      }
      if (hist.length >= 2) {
        const trailGeo = new THREE.BufferGeometry().setFromPoints(hist);
        const trailMat = new THREE.LineBasicMaterial({
          color: trk.state === 'confirmed' ? 0xfacc15 : 0x94a3b8,
          transparent: true,
          opacity: 0.78,
        });
        dyn.add(new THREE.Line(trailGeo, trailMat));
      }

      const ringGeo = new THREE.OctahedronGeometry(0.10, 0);
      const ringMat = new THREE.MeshBasicMaterial({
        color: trk.state === 'confirmed' ? 0xfacc15 : 0x94a3b8,
        wireframe: true,
      });
      const marker = new THREE.Mesh(ringGeo, ringMat);
      marker.position.copy(pos);
      dyn.add(marker);

      const vel = trk.velocity_mps || [0, 0, 0];
      const v3 = new THREE.Vector3(vel[0], vel[2], -vel[1]);
      const speed = v3.length();
      if (speed > 0.08) {
        const arrow = new THREE.ArrowHelper(
          v3.clone().normalize(),
          pos,
          Math.min(1.2, speed * 0.7),
          0xfacc15,
          0.14,
          0.08
        );
        dyn.add(arrow);
      }
    }

    // Prune stale finished track trails
    for (const tid of Array.from(trackHistoryRef.current.keys())) {
      if (!activeIds.has(tid)) {
        trackHistoryRef.current.delete(tid);
      }
    }

    // 4. Separate OPTIONAL ground-truth comparison marker (strictly OFF by default)
    if (showGtOverlay && frame?.ground_truth?.joints_m?.spine) {
      const gtPos = worldToThree(frame.ground_truth.joints_m.spine);
      const gtGeo = new THREE.SphereGeometry(0.13, 10, 10);
      const gtMat = new THREE.MeshBasicMaterial({ color: 0x4ade80, wireframe: true });
      const gtMarker = new THREE.Mesh(gtGeo, gtMat);
      gtMarker.position.copy(gtPos);
      dyn.add(gtMarker);
    }
  }, [frame, colorBy, showVoxels, showGtOverlay]);

  const detCount = frame?.radar?.detections?.length ?? 0;
  const trackList = frame?.radar?.tracks || [];
  const voxCount = frame?.radar?.occupancy?.voxels?.length ?? 0;

  return (
    <section className="panel">
      <header className="panel-header">
        <div className="panel-title-group">
          <span className="panel-badge badge-radar">PANEL 3</span>
          <h2>Radar Point Cloud (Radar-Derived Only)</h2>
        </div>
        <div className="panel-meta">
          <select
            className="mini-select"
            value={colorBy}
            onChange={(e) => setColorBy(e.target.value)}
            aria-label="Color detections by"
          >
            <option value="velocity">Color: Radial Vel</option>
            <option value="snr">Color: SNR (dB)</option>
          </select>
          <label className="mini-toggle">
            <input type="checkbox" checked={showVoxels} onChange={(e) => setShowVoxels(e.target.checked)} />
            Voxels
          </label>
          <label className="mini-toggle gt-toggle">
            <input type="checkbox" checked={showGtOverlay} onChange={(e) => setShowGtOverlay(e.target.checked)} />
            GT compare
          </label>
          <span className="clock-chip">
            f={frame?.frame_id ?? '--'} | t={frame ? frame.sim_time_s.toFixed(2) : '--'}s
          </span>
        </div>
      </header>
      <div className="panel-viewport">
        <div ref={containerRef} className="canvas-host" />
        <div className="viewport-tag bottom-left">
          <strong>Radar observations:</strong> {detCount} dets, {trackList.length} tracks
          {trackList.length > 0 ? ` [${trackList.map((t) => `T${t.track_id}`).join(', ')}]` : ''} |{' '}
          <strong>Reconstructed voxels:</strong> {voxCount} (sparse)
          {showGtOverlay ? ' | [GT wire sphere ON]' : ''}
        </div>
        <div className="viewport-tag bottom-right pc-colorbar-tag">
          {colorBy === 'velocity' ? (
            <>
              <span className="scale-lbl">-2 m/s</span>
              <span className="scale-bar vel-grad" />
              <span className="scale-lbl">+2 m/s</span>
            </>
          ) : (
            <>
              <span className="scale-lbl">10 dB</span>
              <span className="scale-bar snr-grad" />
              <span className="scale-lbl">35 dB</span>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
