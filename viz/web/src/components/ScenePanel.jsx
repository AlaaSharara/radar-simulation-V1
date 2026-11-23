import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { buildRoomGroup, worldDirToThreeVec, worldToThree } from '../lib/coords.js';
import { loadHumanAvatar } from '../lib/avatarRigAdapter.js';

/**
 * Panel 1: Ground-truth architectural 3D room, furniture, and complete visible skinned human avatar.
 * Internal rig bones, joints, and scattering capsules remain strictly invisible.
 */
export default function ScenePanel({ apiBase, sceneData, frame }) {
  const containerRef = useRef(null);
  const avatarRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const roomGroupRef = useRef(null);
  const [avatarStatus, setAvatarStatus] = useState('loading');
  const [showOverlay, setShowOverlay] = useState(false);
  const [showFovCone, setShowFovCone] = useState(true);

  const applyCameraPreset = useCallback(
    (preset) => {
      const camera = cameraRef.current;
      const controls = controlsRef.current;
      if (!camera || !controls) return;
      const w = sceneData?.room?.width_x_m || 4.0;
      const d = sceneData?.room?.depth_y_m || 5.0;
      const h = sceneData?.room?.height_z_m || 2.7;

      if (preset === 'top') {
        camera.position.set(w * 0.5, Math.max(w, d) * 1.38, -d * 0.5 + 0.05);
        controls.target.set(w * 0.5, 0.3, -d * 0.5);
      } else if (preset === 'radar' && sceneData?.radar?.frame) {
        const rPos = worldToThree(sceneData.radar.frame.position_m);
        const bDir = worldDirToThreeVec(sceneData.radar.frame.boresight_world);
        camera.position.copy(rPos).addScaledVector(bDir, -0.35).add(new THREE.Vector3(0, 0.25, 0));
        controls.target.copy(rPos).addScaledVector(bDir, 2.4);
      } else {
        camera.position.set(w * 1.18, h * 1.12, d * 0.42);
        controls.target.set(w / 2, h * 0.36, -d / 2);
      }
      controls.update();
    },
    [sceneData]
  );

  // Initialize Three.js scene, PBR lighting + shadows, controls, architectural room, and GLB avatar
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a111e);
    scene.fog = new THREE.Fog(0x0a111e, 9, 22);
    sceneRef.current = scene;

    const w = sceneData?.room?.width_x_m || 4.0;
    const d = sceneData?.room?.depth_y_m || 5.0;
    const h = sceneData?.room?.height_z_m || 2.7;

    const camera = new THREE.PerspectiveCamera(46, el.clientWidth / Math.max(1, el.clientHeight), 0.05, 60);
    camera.position.set(w * 1.18, h * 1.12, d * 0.42);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(el.clientWidth, el.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    el.innerHTML = '';
    el.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(w / 2, h * 0.36, -d / 2);
    controls.update();
    controlsRef.current = controls;

    // Studio indoor lighting: warm ceiling key with soft shadows + cool daylight window bounce
    const hemi = new THREE.HemisphereLight(0xe0f2fe, 0x3b2f24, 0.72);
    scene.add(hemi);

    const keyLight = new THREE.DirectionalLight(0xfff5e8, 1.35);
    keyLight.position.set(w * 0.85, h * 1.85, -d * 0.12);
    keyLight.target.position.set(w * 0.5, 0, -d * 0.5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 14;
    const span = Math.max(w, d) * 0.85;
    keyLight.shadow.camera.left = -span;
    keyLight.shadow.camera.right = span;
    keyLight.shadow.camera.top = span;
    keyLight.shadow.camera.bottom = -span;
    keyLight.shadow.bias = -0.0005;
    scene.add(keyLight);
    scene.add(keyLight.target);

    const fillLight = new THREE.DirectionalLight(0x93c5fd, 0.55);
    fillLight.position.set(-1.8, h * 0.9, -d * 0.9);
    scene.add(fillLight);

    // Architectural room & furniture from normalized scene object
    if (sceneData) {
      const roomGrp = buildRoomGroup(sceneData, { wireframeOnly: false, showFovCone });
      roomGroupRef.current = roomGrp;
      scene.add(roomGrp);
    }

    // Load visible rigged adult male human GLB avatar (skeleton remains strictly invisible)
    let cancelled = false;
    const glbUrl = `${apiBase}/assets/avatars/adult_male_default.glb`;
    loadHumanAvatar(glbUrl)
      .then((adapter) => {
        if (cancelled) return;
        adapter.root.traverse((obj) => {
          if (obj.isMesh) {
            obj.castShadow = true;
            obj.receiveShadow = true;
          }
        });
        scene.add(adapter.root);
        avatarRef.current = adapter;
        setAvatarStatus('ready');
      })
      .catch((err) => {
        if (!cancelled) setAvatarStatus(`error: ${err.message}`);
      });

    // Resize observer for responsive 2x2 layout
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
      cancelled = true;
      cancelAnimationFrame(rafId);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      avatarRef.current = null;
      roomGroupRef.current = null;
    };
  }, [apiBase, sceneData]);

  // Toggle radar FoV frustum visibility without rebuilding the entire room
  useEffect(() => {
    const roomGrp = roomGroupRef.current;
    if (!roomGrp) return;
    const cone = roomGrp.getObjectByName('radar_fov_cone');
    if (cone) {
      cone.visible = Boolean(showFovCone);
    }
  }, [showFovCone]);

  // Drive human avatar strictly from backend synchronized frame (never animate independently)
  useEffect(() => {
    const adapter = avatarRef.current;
    const joints = frame?.ground_truth?.joints_m;
    if (adapter && joints) {
      adapter.updateFromJoints(joints);
    }
  }, [frame]);

  const gt = frame?.ground_truth;

  return (
    <section className="panel">
      <header className="panel-header">
        <div className="panel-title-group">
          <span className="panel-badge badge-gt">PANEL 1</span>
          <h2>Architectural Scene Preview</h2>
        </div>
        <div className="panel-meta">
          <div className="seg-group" aria-label="Camera presets">
            <button type="button" className="seg-btn" onClick={() => applyCameraPreset('iso')}>
              Iso
            </button>
            <button type="button" className="seg-btn" onClick={() => applyCameraPreset('top')}>
              Top
            </button>
            <button type="button" className="seg-btn" onClick={() => applyCameraPreset('radar')}>
              Radar
            </button>
          </div>
          <label className="mini-toggle">
            <input
              type="checkbox"
              checked={showFovCone}
              onChange={(e) => setShowFovCone(e.target.checked)}
            />
            FoV cone
          </label>
          <label className="mini-toggle">
            <input
              type="checkbox"
              checked={showOverlay}
              onChange={(e) => setShowOverlay(e.target.checked)}
            />
            Pose meta
          </label>
          <span className="clock-chip">
            f={frame?.frame_id ?? '--'} | t={frame ? frame.sim_time_s.toFixed(2) : '--'}s
          </span>
        </div>
      </header>
      <div className="panel-viewport">
        <div ref={containerRef} className="canvas-host" />
        <div className="viewport-tag bottom-left">
          Avatar: {gt?.avatar_asset_id || 'adult_male_default'} ({avatarStatus}) — rigged mesh, skeleton hidden
        </div>
        {showOverlay && gt && (
          <div className="viewport-overlay top-right">
            <div><strong>Action:</strong> {gt.action_label}{gt.action_phase ? ` (${gt.action_phase})` : ''}</div>
            <div><strong>Mode:</strong> {gt.scenario_mode}</div>
            <div>
              <strong>Root XYZ:</strong>{' '}
              {gt.root_position_m.map((v) => v.toFixed(2)).join(', ')} m
            </div>
            <div><strong>Fall GT:</strong> {gt.fall_event_id || 'none'}</div>
          </div>
        )}
      </div>
    </section>
  );
}
