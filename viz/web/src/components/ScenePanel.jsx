import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  buildObjectLabels,
  buildRfPanelsGroup,
  buildRoomGroup,
  RF_DB_RANGE,
  worldDirToThreeVec,
  worldToThree,
} from '../lib/coords.js';
import { loadHumanAvatar } from '../lib/avatarRigAdapter.js';

const VIEWS = [
  ['iso', 'Iso'],
  ['top', 'Top'],
  ['front', 'Front'],
  ['back', 'Back'],
  ['left', 'Left'],
  ['right', 'Right'],
  ['corner', 'Corner'],
  ['radar', 'Radar'],
];

const TWEEN_MS = 650;
const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);

function roomDims(sceneData) {
  return {
    w: sceneData?.room?.width_x_m || 4.0,
    d: sceneData?.room?.depth_y_m || 5.0,
    h: sceneData?.room?.height_z_m || 2.7,
  };
}

/** Camera position + orbit target (Three.js coords: x right, y up, -z = world +Y) for a named preset. */
function presetPose(name, sceneData) {
  const { w, d, h } = roomDims(sceneData);
  const center = new THREE.Vector3(w / 2, h * 0.36, -d / 2);
  const far = Math.max(w, d) * 1.15;
  switch (name) {
    case 'top':
      return {
        pos: new THREE.Vector3(w * 0.5, Math.max(w, d) * 1.38, -d * 0.5 + 0.05),
        target: new THREE.Vector3(w * 0.5, 0.3, -d * 0.5),
      };
    case 'front': // from the radar wall, looking into the room
      return { pos: new THREE.Vector3(w / 2, h * 0.7, d * 0.6), target: center };
    case 'back': // from behind the far wall, looking back at the radar wall
      return { pos: new THREE.Vector3(w / 2, h * 0.75, -d - d * 0.6), target: center };
    case 'left':
      return { pos: new THREE.Vector3(-far, h * 0.7, -d / 2), target: center };
    case 'right':
      return { pos: new THREE.Vector3(w + far, h * 0.7, -d / 2), target: center };
    case 'corner': // opposite diagonal of 'iso'
      return { pos: new THREE.Vector3(-w * 0.45, h * 1.15, -d * 1.4), target: center };
    case 'radar': {
      if (sceneData?.radar?.frame) {
        const rPos = worldToThree(sceneData.radar.frame.position_m);
        const bDir = worldDirToThreeVec(sceneData.radar.frame.boresight_world);
        return {
          pos: rPos.clone().addScaledVector(bDir, -0.35).add(new THREE.Vector3(0, 0.25, 0)),
          target: rPos.clone().addScaledVector(bDir, 2.4),
        };
      }
      return { pos: new THREE.Vector3(w / 2, h * 0.8, d * 0.2), target: center };
    }
    case 'iso':
    default:
      return { pos: new THREE.Vector3(w * 1.18, h * 1.12, d * 0.42), target: center };
  }
}

function fitOrtho(cam, aspect, halfW0, halfH0) {
  const halfH = Math.max(halfH0, halfW0 / Math.max(0.1, aspect));
  const halfW = halfH * aspect;
  cam.left = -halfW;
  cam.right = halfW;
  cam.top = halfH;
  cam.bottom = -halfH;
  cam.updateProjectionMatrix();
}

/**
 * Panel 1: Ground-truth architectural 3D room, furniture, and complete visible skinned human avatar.
 * Internal rig bones, joints, and scattering capsules remain strictly invisible.
 *
 * Preview features: 8 animated camera presets, 2x2 multi-view (perspective + top + front + side
 * elevations), follow-person camera, automatic wall cut-away, RF-material overlay (furniture RF panels
 * coloured by reflectivity, click a panel to inspect), object labels and PNG screenshot.
 */
export default function ScenePanel({ apiBase, sceneData, frame }) {
  const containerRef = useRef(null);
  const avatarRef = useRef(null);
  const smoothRef = useRef({ prev: null, curr: null, shown: null, t0: 0, interval: 50, snap: true });
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const roomGroupRef = useRef(null);
  const rfGroupRef = useRef(null);
  const labelsRef = useRef(null);
  const furnitureRef = useRef([]);
  const cutawayRef = useRef({ back: [], left: [] });
  const tweenRef = useRef(null);
  const renderOnceRef = useRef(() => {});
  const applyDisplayRef = useRef(() => {});

  const [avatarStatus, setAvatarStatus] = useState('loading');
  const [showOverlay, setShowOverlay] = useState(false);
  const [showFovCone, setShowFovCone] = useState(true);
  const [activeView, setActiveView] = useState('iso');
  const [layout, setLayout] = useState('single'); // 'single' | 'quad'
  const [displayMode, setDisplayMode] = useState('visual'); // 'visual' | 'overlay' | 'rf'
  const [showLabels, setShowLabels] = useState(false);
  const [follow, setFollow] = useState(false);
  const [inspect, setInspect] = useState(null);

  // Latest UI state for the render loop (avoids re-creating the WebGL scene on every click)
  const uiRef = useRef({});
  uiRef.current = { layout, displayMode, showLabels, follow };

  const startTween = useCallback((pos, target, ms = TWEEN_MS) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    tweenRef.current = {
      fromPos: camera.position.clone(),
      fromTarget: controls.target.clone(),
      toPos: pos.clone(),
      toTarget: target.clone(),
      t0: performance.now(),
      dur: ms,
    };
  }, []);

  const applyCameraPreset = useCallback(
    (name) => {
      const { pos, target } = presetPose(name, sceneData);
      setFollow(false);
      setActiveView(name);
      startTween(pos, target);
    },
    [sceneData, startTween]
  );

  const toggleFollow = useCallback(() => {
    setFollow((prev) => {
      const next = !prev;
      if (next) {
        const sm = smoothRef.current;
        const pel = (sm.shown || sm.curr)?.pelvis;
        if (pel) {
          const t = worldToThree([pel[0], pel[1], Math.max(0.7, pel[2])]);
          startTween(new THREE.Vector3(t.x + 1.6, t.y + 1.2, t.z + 1.9), t, 500);
        }
        setActiveView('follow');
      }
      return next;
    });
  }, [startTween]);

  const takeScreenshot = useCallback(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    renderOnceRef.current();
    const url = renderer.domElement.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = `scene_${layout === 'quad' ? 'quad' : activeView}_${Date.now()}.png`;
    a.click();
  }, [activeView, layout]);

  // Initialize Three.js scene, PBR lighting + shadows, controls, architectural room, and GLB avatar
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;

    const scene = new THREE.Scene();
    const fog = new THREE.Fog(0x0a111e, 9, 22);
    scene.background = new THREE.Color(0x0a111e);
    scene.fog = fog;
    sceneRef.current = scene;

    const { w, d, h } = roomDims(sceneData);

    const camera = new THREE.PerspectiveCamera(46, el.clientWidth / Math.max(1, el.clientHeight), 0.05, 60);
    const iso = presetPose('iso', sceneData);
    camera.position.copy(iso.pos);
    cameraRef.current = camera;

    // Orthographic cameras for the multi-view layout: plan, elevation from the radar wall, side elevation
    const topCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 40);
    topCam.up.set(0, 0, -1);
    topCam.position.set(w / 2, h * 3, -d / 2);
    topCam.lookAt(w / 2, 0, -d / 2);
    const frontCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 40);
    frontCam.position.set(w / 2, h / 2, 12);
    frontCam.lookAt(w / 2, h / 2, -d / 2);
    const sideCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 40);
    sideCam.position.set(w + 12, h / 2, -d / 2);
    sideCam.lookAt(w / 2, h / 2, -d / 2);

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
    controls.target.copy(iso.target);
    controls.update();
    controlsRef.current = controls;
    controls.addEventListener('start', () => {
      tweenRef.current = null; // the user took over the camera
      setActiveView('free');
    });

    // Studio indoor lighting: warm ceiling key with soft shadows + cool daylight window bounce
    scene.add(new THREE.HemisphereLight(0xe0f2fe, 0x3b2f24, 0.72));

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
    furnitureRef.current = [];
    cutawayRef.current = { back: [], left: [] };
    if (sceneData) {
      const roomGrp = buildRoomGroup(sceneData, { wireframeOnly: false, showFovCone });
      roomGroupRef.current = roomGrp;
      scene.add(roomGrp);
      roomGrp.traverse((o) => {
        if (o.userData?.isFurniture) furnitureRef.current.push(o);
        const side = o.userData?.cutaway;
        if (side === 'back' || side === 'left') cutawayRef.current[side].push(o);
      });

      const rfGrp = buildRfPanelsGroup(sceneData);
      rfGroupRef.current = rfGrp;
      scene.add(rfGrp);
      const labels = buildObjectLabels(sceneData);
      labelsRef.current = labels;
      scene.add(labels);
    }

    // Display mode (visual / RF overlay / RF only) and labels
    applyDisplayRef.current = () => {
      const { displayMode: mode, showLabels: lbl } = uiRef.current;
      for (const f of furnitureRef.current) f.visible = mode !== 'rf';
      const rf = rfGroupRef.current;
      if (rf) {
        rf.visible = mode !== 'visual';
        for (const m of rf.userData.meshes || []) {
          m.material.opacity = mode === 'rf' ? 0.92 : 0.45;
          m.material.depthWrite = mode === 'rf';
          m.material.needsUpdate = true;
        }
      }
      if (labelsRef.current) labelsRef.current.visible = Boolean(lbl);
    };
    applyDisplayRef.current();

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

    // ---- rendering --------------------------------------------------------------------------
    const setCutaway = (cam) => {
      const hideBack = cam.position.z < -d - 0.05;
      const hideLeft = cam.position.x < -0.05;
      for (const o of cutawayRef.current.back) o.visible = !hideBack;
      for (const o of cutawayRef.current.left) o.visible = !hideLeft;
    };

    const renderViewport = (cam, x, y, vw, vh) => {
      renderer.setViewport(x, y, vw, vh);
      renderer.setScissor(x, y, vw, vh);
      renderer.render(scene, cam);
    };

    const renderFrame = () => {
      const cw = Math.max(1, el.clientWidth);
      const ch = Math.max(1, el.clientHeight);
      if (uiRef.current.layout === 'quad') {
        const hw = Math.floor(cw / 2);
        const hh = Math.floor(ch / 2);
        renderer.setScissorTest(true);
        renderer.setClearColor(0x0a111e, 1);
        // top-left: perspective (orbit) -- WebGL y runs from the bottom
        camera.aspect = hw / Math.max(1, ch - hh);
        camera.updateProjectionMatrix();
        setCutaway(camera);
        scene.fog = fog;
        renderViewport(camera, 0, hh, hw, ch - hh);
        // ortho views: no fog, all walls visible except the ones facing the camera
        scene.fog = null;
        for (const o of cutawayRef.current.back) o.visible = true;
        for (const o of cutawayRef.current.left) o.visible = true;
        fitOrtho(topCam, (cw - hw) / Math.max(1, ch - hh), w / 2 + 0.3, d / 2 + 0.3);
        renderViewport(topCam, hw, hh, cw - hw, ch - hh);
        fitOrtho(frontCam, hw / Math.max(1, hh), w / 2 + 0.3, h / 2 + 0.3);
        renderViewport(frontCam, 0, 0, hw, hh);
        for (const o of cutawayRef.current.left) o.visible = false; // side view looks from +x towards the left wall
        fitOrtho(sideCam, (cw - hw) / Math.max(1, hh), d / 2 + 0.3, h / 2 + 0.3);
        renderViewport(sideCam, hw, 0, cw - hw, hh);
        scene.fog = fog;
        renderer.setScissorTest(false);
      } else {
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, cw, ch);
        camera.aspect = cw / ch;
        camera.updateProjectionMatrix();
        setCutaway(camera);
        renderer.render(scene, camera);
      }
    };
    renderOnceRef.current = renderFrame;

    // Resize observer for responsive 2x2 layout
    const ro = new ResizeObserver(() => {
      renderer.setSize(Math.max(1, el.clientWidth), Math.max(1, el.clientHeight));
    });
    ro.observe(el);

    // Orbit only in the perspective quadrant when the multi-view layout is on
    let dragging = false;
    const dom = renderer.domElement;
    const inPerspectiveQuadrant = (ev) => {
      if (uiRef.current.layout !== 'quad') return true;
      const r = dom.getBoundingClientRect();
      return ev.clientX - r.left < r.width / 2 && ev.clientY - r.top < r.height / 2;
    };
    const onMove = (ev) => {
      if (!dragging) controls.enabled = inPerspectiveQuadrant(ev);
    };
    let downXY = null;
    const onDown = (ev) => {
      dragging = true;
      downXY = [ev.clientX, ev.clientY];
    };
    const raycaster = new THREE.Raycaster();
    const onUp = (ev) => {
      dragging = false;
      if (!downXY || Math.hypot(ev.clientX - downXY[0], ev.clientY - downXY[1]) > 4) return;
      const rf = rfGroupRef.current;
      if (!rf || !rf.visible || !inPerspectiveQuadrant(ev)) {
        setInspect(null);
        return;
      }
      const r = dom.getBoundingClientRect();
      const quad = uiRef.current.layout === 'quad';
      const vw = quad ? r.width / 2 : r.width;
      const vh = quad ? r.height / 2 : r.height;
      const ndc = new THREE.Vector2(((ev.clientX - r.left) / vw) * 2 - 1, -(((ev.clientY - r.top) / vh) * 2 - 1));
      raycaster.setFromCamera(ndc, camera);
      const hit = raycaster.intersectObjects(rf.userData.meshes || [], false)[0];
      setInspect(hit ? hit.object.userData.surface : null);
    };
    dom.addEventListener('pointermove', onMove);
    dom.addEventListener('pointerdown', onDown);
    window.addEventListener('pointerup', onUp);

    let rafId = 0;
    const tmp = new THREE.Vector3();
    function renderLoop() {
      rafId = requestAnimationFrame(renderLoop);

      // Smooth camera preset transitions
      const tw = tweenRef.current;
      if (tw) {
        const a = easeInOut(Math.min(1, (performance.now() - tw.t0) / tw.dur));
        camera.position.lerpVectors(tw.fromPos, tw.toPos, a);
        controls.target.lerpVectors(tw.fromTarget, tw.toTarget, a);
        if (a >= 1) tweenRef.current = null;
      } else if (uiRef.current.follow) {
        // Follow-person camera: glide the orbit target onto the pelvis and carry the camera along
        const pel = (smoothRef.current.shown || smoothRef.current.curr)?.pelvis;
        if (pel) {
          const t = worldToThree([pel[0], pel[1], Math.max(0.7, pel[2])]);
          tmp.copy(t).sub(controls.target).multiplyScalar(0.12);
          controls.target.add(tmp);
          camera.position.add(tmp);
        }
      }
      controls.update();

      // Visual smoothing only: backend frames arrive at ~20 Hz (or slower), so blend the avatar
      // joints between the previous and current backend frame at display rate (~60 fps).
      const sm = smoothRef.current;
      if (avatarRef.current && sm.curr) {
        let joints = sm.curr;
        if (sm.prev && !sm.snap) {
          const a = Math.min(1, (performance.now() - sm.t0) / sm.interval);
          joints = {};
          for (const name of Object.keys(sm.curr)) {
            const c = sm.curr[name];
            const q = sm.prev[name] || c;
            joints[name] = [q[0] + (c[0] - q[0]) * a, q[1] + (c[1] - q[1]) * a, q[2] + (c[2] - q[2]) * a];
          }
        }
        sm.shown = joints;
        avatarRef.current.updateFromJoints(joints);
      }
      renderFrame();
    }
    renderLoop();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      ro.disconnect();
      dom.removeEventListener('pointermove', onMove);
      dom.removeEventListener('pointerdown', onDown);
      window.removeEventListener('pointerup', onUp);
      controls.dispose();
      renderer.dispose();
      avatarRef.current = null;
      roomGroupRef.current = null;
      rfGroupRef.current = null;
      labelsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBase, sceneData]);

  // Display-mode / label changes: no scene rebuild needed
  useEffect(() => {
    applyDisplayRef.current();
    if (displayMode === 'visual') setInspect(null);
  }, [displayMode, showLabels]);

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
    const joints = frame?.ground_truth?.joints_m;
    if (!joints) return;
    const sm = smoothRef.current;
    const now = performance.now();
    if (sm.curr) {
      const dt = now - sm.t0;
      if (dt > 0) sm.interval = Math.min(300, Math.max(30, 0.7 * sm.interval + 0.3 * dt));
      const p0 = sm.shown?.pelvis || sm.curr.pelvis;
      const p1 = joints.pelvis;
      // Large jump (loop wrap / regenerated motion): snap instead of blending across the room
      sm.snap = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 0.6;
      sm.prev = sm.shown || sm.curr;
    } else {
      sm.prev = null;
      sm.snap = true;
    }
    sm.curr = joints;
    sm.t0 = now;
  }, [frame]);

  const gt = frame?.ground_truth;
  const rfOn = displayMode !== 'visual';

  return (
    <section className="panel">
      <header className="panel-header">
        <div className="panel-title-group">
          <span className="panel-badge badge-gt">PANEL 1</span>
          <h2>Architectural Scene Preview</h2>
        </div>
        <div className="panel-meta">
          <label className="mini-toggle">
            <input type="checkbox" checked={showFovCone} onChange={(e) => setShowFovCone(e.target.checked)} />
            FoV cone
          </label>
          <label className="mini-toggle">
            <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} />
            Pose meta
          </label>
          <span className="clock-chip">
            f={frame?.frame_id ?? '--'} | t={frame ? frame.sim_time_s.toFixed(2) : '--'}s
            {frame?.control_state?.clip_duration_s != null &&
              ` | clip ${frame.control_state.clip_time_s}/${frame.control_state.clip_duration_s}s`}
          </span>
        </div>
      </header>
      <div className="panel-viewport">
        <div ref={containerRef} className="canvas-host" />

        <div className="scene-toolbar">
          <div className="seg-group" aria-label="Camera views">
            {VIEWS.map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={`seg-btn${activeView === key && layout === 'single' ? ' active' : ''}`}
                onClick={() => applyCameraPreset(key)}
                title={`${label} view`}
              >
                {label}
              </button>
            ))}
            <button
              type="button"
              className={`seg-btn${follow ? ' active' : ''}`}
              onClick={toggleFollow}
              title="Camera follows the person"
            >
              Follow
            </button>
          </div>
          <div className="seg-group" aria-label="Layout">
            <button type="button" className={`seg-btn${layout === 'single' ? ' active' : ''}`} onClick={() => setLayout('single')}>
              Single
            </button>
            <button
              type="button"
              className={`seg-btn${layout === 'quad' ? ' active' : ''}`}
              onClick={() => setLayout('quad')}
              title="Perspective + top + front + side elevations at once"
            >
              Quad
            </button>
          </div>
          <div className="seg-group" aria-label="Furniture display">
            <button type="button" className={`seg-btn${displayMode === 'visual' ? ' active' : ''}`} onClick={() => setDisplayMode('visual')} title="What a camera would see">
              Visual
            </button>
            <button type="button" className={`seg-btn${displayMode === 'overlay' ? ' active' : ''}`} onClick={() => setDisplayMode('overlay')} title="Visible furniture + RF panels coloured by reflectivity">
              RF overlay
            </button>
            <button type="button" className={`seg-btn${displayMode === 'rf' ? ' active' : ''}`} onClick={() => setDisplayMode('rf')} title="Only the RF panels (what the radar model sees)">
              RF only
            </button>
          </div>
          <div className="seg-group">
            <button type="button" className={`seg-btn${showLabels ? ' active' : ''}`} onClick={() => setShowLabels((v) => !v)} title="Object names">
              Labels
            </button>
            <button type="button" className="seg-btn" onClick={takeScreenshot} title="Save the current view as PNG">
              PNG
            </button>
          </div>
        </div>

        {layout === 'quad' && (
          <div className="quad-labels" aria-hidden="true">
            <span className="quad-label q-tl">Perspective (drag = orbit)</span>
            <span className="quad-label q-tr">Top (plan, +Y up)</span>
            <span className="quad-label q-bl">Elevation from radar wall</span>
            <span className="quad-label q-br">Side elevation</span>
            <div className="quad-cross-h" />
            <div className="quad-cross-v" />
          </div>
        )}

        <div className="viewport-tag bottom-left">
          Avatar: {gt?.avatar_asset_id || 'adult_male_default'} ({avatarStatus}) — rigged mesh, skeleton hidden
        </div>

        {rfOn && (
          <div className="rf-legend">
            <div className="rf-legend-title">Furniture RF reflectivity (|R| @ 45°, 61 GHz)</div>
            <div className="rf-legend-bar" />
            <div className="rf-legend-scale">
              <span>{RF_DB_RANGE[0]} dB</span>
              <span>{RF_DB_RANGE[1]} dB</span>
            </div>
            <div className="rf-legend-note">Click a panel for details</div>
          </div>
        )}

        {rfOn && inspect && (
          <div className="rf-inspect">
            <div>
              <strong>{inspect.surface_id}</strong>
            </div>
            <div>
              {inspect.layers.map((l) => `${l.material_id} ${(l.thickness_m * 1000).toFixed(0)} mm`).join(' + ')}
            </div>
            {inspect.rf && (
              <>
                <div>
                  |R| 0° / 45° / 70°: {inspect.rf.reflection_db['0']} / {inspect.rf.reflection_db['45']} /{' '}
                  {inspect.rf.reflection_db['70']} dB
                </div>
                <div>
                  |T| 45°: {inspect.rf.transmission_db_45} dB &nbsp;·&nbsp; roughness factor: {inspect.rf.roughness_factor_45}
                </div>
              </>
            )}
          </div>
        )}

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
