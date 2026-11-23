import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { buildRoomGroup, worldDirToThreeVec, worldToThree } from '../lib/coords.js';

const PATH_COLORS = {
  los: 0x38bdf8,
  specular_reflection: 0xf59e0b,
  diffuse_scatter: 0xa78bfa,
  transmitted: 0x34d399,
};

function interpolatePolyline(pts, frac) {
  if (!pts || pts.length === 0) return null;
  if (pts.length === 1) return pts[0].clone();
  const segLens = [];
  let total = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const l = pts[i].distanceTo(pts[i + 1]);
    segLens.push(l);
    total += l;
  }
  if (total <= 1e-6) return pts[0].clone();
  let target = Math.max(0, Math.min(1, frac)) * total;
  for (let i = 0; i < segLens.length; i++) {
    const l = segLens[i];
    if (target <= l || i === segLens.length - 1) {
      const alpha = l > 1e-6 ? target / l : 0;
      return new THREE.Vector3().lerpVectors(pts[i], pts[i + 1], Math.min(1, Math.max(0, alpha)));
    }
    target -= l;
  }
  return pts[pts.length - 1].clone();
}

function formatLossDb(val) {
  return typeof val === 'number' ? `-${val.toFixed(1)} dB` : '--';
}

/**
 * Panel 2: Animated geometric-optics RF propagation paths (LOS, reflections,
 * diffuse scatter, transmitted), expanding phase fronts, and bounce markers.
 */
export default function RFPanel({ sceneData, frame }) {
  const containerRef = useRef(null);
  const raysGroupRef = useRef(null);
  const pulseGroupRef = useRef(null);
  const [visibleTypes, setVisibleTypes] = useState({
    los: true,
    specular_reflection: true,
    diffuse_scatter: true,
    transmitted: true,
  });

  function toggleType(ptype) {
    setVisibleTypes((prev) => ({ ...prev, [ptype]: !prev[ptype] }));
  }

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

    const raysGroup = new THREE.Group();
    const pulseGroup = new THREE.Group();
    scene.add(raysGroup);
    scene.add(pulseGroup);
    raysGroupRef.current = raysGroup;
    pulseGroupRef.current = pulseGroup;

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

  // Update ray geometry, bounce rings & expanding phase-front arcs from synchronized frame
  useEffect(() => {
    const raysGroup = raysGroupRef.current;
    const pulseGroup = pulseGroupRef.current;
    if (!raysGroup || !pulseGroup) return;

    while (raysGroup.children.length > 0) {
      const child = raysGroup.children.pop();
      child.geometry?.dispose();
      child.material?.dispose();
    }
    while (pulseGroup.children.length > 0) {
      const child = pulseGroup.children.pop();
      child.geometry?.dispose();
      child.material?.dispose();
    }

    const simT = frame?.sim_time_s || 0;
    const pulsePhase = (simT * 1.15) % 1.0;

    // 1. Expanding spherical boresight phase-front arcs radiating from TX sensor
    if (sceneData?.radar?.frame) {
      const rPos = worldToThree(sceneData.radar.frame.position_m);
      const bDir = worldDirToThreeVec(sceneData.radar.frame.boresight_world);
      const right = new THREE.Vector3().crossVectors(bDir, new THREE.Vector3(0, 1, 0)).normalize();
      const up = new THREE.Vector3().crossVectors(right, bDir).normalize();
      const maxArcR = 3.8;

      for (let wIdx = 0; wIdx < 3; wIdx++) {
        const radius = 0.55 + ((pulsePhase + wIdx / 3.0) % 1.0) * (maxArcR - 0.55);
        const alpha = Math.max(0.05, 0.28 * (1.0 - radius / maxArcR));
        const arcPts = [];
        for (let deg = -50; deg <= 50; deg += 5) {
          const th = (deg * Math.PI) / 180;
          const pt = rPos
            .clone()
            .addScaledVector(bDir, radius * Math.cos(th))
            .addScaledVector(right, radius * Math.sin(th));
          arcPts.push(pt);
        }
        const arcGeo = new THREE.BufferGeometry().setFromPoints(arcPts);
        const arcMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: alpha });
        pulseGroup.add(new THREE.Line(arcGeo, arcMat));

        const elArcPts = [];
        for (let deg = -35; deg <= 35; deg += 5) {
          const ph = (deg * Math.PI) / 180;
          const pt = rPos
            .clone()
            .addScaledVector(bDir, radius * Math.cos(ph))
            .addScaledVector(up, radius * Math.sin(ph));
          elArcPts.push(pt);
        }
        pulseGroup.add(
          new THREE.Line(new THREE.BufferGeometry().setFromPoints(elArcPts), arcMat.clone())
        );
      }
    }

    // 2. Render ray polylines, bounce hotspots, and full-path wave packets
    const paths = frame?.rf?.paths || [];
    const pulseGeo = new THREE.SphereGeometry(0.032, 8, 8);
    const bounceGeo = new THREE.RingGeometry(0.055, 0.095, 16);

    for (let idx = 0; idx < paths.length; idx++) {
      const p = paths[idx];
      if (!visibleTypes[p.path_type]) continue;
      const verts = p.vertices_m;
      if (!verts || verts.length < 2) continue;
      const pts = verts.map(worldToThree);
      const color = PATH_COLORS[p.path_type] || 0x94a3b8;

      // Smooth dynamic opacity scaling from channel path loss [dB]
      const loss = typeof p.path_loss_db === 'number' ? p.path_loss_db : 100;
      const opacity = Math.max(0.18, Math.min(0.95, 1.0 - (loss - 60) / 72));

      const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const lineMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
      raysGroup.add(new THREE.Line(lineGeo, lineMat));

      // Specular bounce ring markers at intermediate wall/floor reflection points
      if (p.path_type === 'specular_reflection' && pts.length >= 3 && idx < 42) {
        const bouncePt = pts[1];
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0xfbbf24,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: Math.min(0.9, opacity + 0.15),
        });
        const ring = new THREE.Mesh(bounceGeo.clone(), ringMat);
        ring.position.copy(bouncePt);
        ring.rotation.x = Math.PI / 2;
        raysGroup.add(ring);
      }

      // Traveling wave packet along complete multi-segment path TX -> bounce/target -> RX
      if (idx < 36) {
        const u = (pulsePhase + idx * 0.11) % 1.0;
        const pos = interpolatePolyline(pts, u);
        if (pos) {
          const dotMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: Math.min(1, opacity + 0.1) });
          const dot = new THREE.Mesh(pulseGeo.clone(), dotMat);
          dot.position.copy(pos);
          pulseGroup.add(dot);
        }
      }
    }
    pulseGeo.dispose();
    bounceGeo.dispose();
  }, [frame, sceneData, visibleTypes]);

  const summary = frame?.rf?.ray_summary || {};
  const counts = summary.counts_by_type || {};
  const occ = summary.occluded_los_count ?? 0;
  const link = summary.link_summary || {};

  return (
    <section className="panel">
      <header className="panel-header">
        <div className="panel-title-group">
          <span className="panel-badge badge-rf">PANEL 2</span>
          <h2>RF Propagation</h2>
        </div>
        <div className="panel-meta">
          <span className="clock-chip">
            f={frame?.frame_id ?? '--'} | t={frame ? frame.sim_time_s.toFixed(2) : '--'}s
          </span>
        </div>
      </header>
      <div className="panel-viewport">
        <div ref={containerRef} className="canvas-host" />
        <div className="viewport-banner top-left">
          Geometric-optics RF visualization — not full-wave field simulation
        </div>
        <div className="viewport-overlay top-right rf-hud">
          <div><strong>Strongest Link Gain:</strong></div>
          <div>LOS: {formatLossDb(link.strongest_los_loss_db)}</div>
          <div>Multipath: {formatLossDb(link.strongest_reflection_loss_db)}</div>
          <div>Diffuse: {formatLossDb(link.strongest_diffuse_loss_db)}</div>
          <div>Occluded rays: {occ}</div>
        </div>
        <div className="viewport-tag bottom-left rf-legend">
          <button
            type="button"
            className={`legend-chip ${visibleTypes.los ? 'active' : ''}`}
            onClick={() => toggleType('los')}
          >
            <span className="dot" style={{ background: '#38bdf8' }} /> LOS ({counts.los ?? 0})
          </button>
          <button
            type="button"
            className={`legend-chip ${visibleTypes.specular_reflection ? 'active' : ''}`}
            onClick={() => toggleType('specular_reflection')}
          >
            <span className="dot" style={{ background: '#f59e0b' }} /> Refl ({counts.specular_reflection ?? 0})
          </button>
          <button
            type="button"
            className={`legend-chip ${visibleTypes.diffuse_scatter ? 'active' : ''}`}
            onClick={() => toggleType('diffuse_scatter')}
          >
            <span className="dot" style={{ background: '#a78bfa' }} /> Diffuse ({counts.diffuse_scatter ?? 0})
          </button>
          <button
            type="button"
            className={`legend-chip ${visibleTypes.transmitted ? 'active' : ''}`}
            onClick={() => toggleType('transmitted')}
          >
            <span className="dot" style={{ background: '#34d399' }} /> Trans ({counts.transmitted ?? 0})
          </button>
        </div>
      </div>
    </section>
  );
}
