import { useCallback, useEffect, useRef, useState } from 'react';

const DEFAULT_API_BASE = 'http://127.0.0.1:8000';
const DEFAULT_WS_URL = 'ws://127.0.0.1:8000/ws/sim';

/**
 * Synchronized backend WebSocket + HTTP hook.
 * - Handles clean exponential-backoff reconnects.
 * - Coalesces bursts via requestAnimationFrame to drop stale frames cleanly
 *   while counting dropped/coalesced frames and measuring true update FPS.
 * - Accumulates rolling radar-only track timeline & Micro-Doppler spectrogram slices.
 */
export function useSimSocket({
  apiBase = DEFAULT_API_BASE,
  wsUrl = DEFAULT_WS_URL,
} = {}) {
  const [configSummary, setConfigSummary] = useState(null);
  const [scenarioModes, setScenarioModes] = useState(null);
  const [frame, setFrame] = useState(null);
  const [controlState, setControlState] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [lastError, setLastError] = useState(null);
  const [measuredFps, setMeasuredFps] = useState(0);
  const [droppedFrames, setDroppedFrames] = useState(0);
  const [timeline, setTimeline] = useState([]);
  const [microDoppler, setMicroDoppler] = useState(null);

  const wsRef = useRef(null);
  const pendingFrameRef = useRef(null);
  const rafRef = useRef(null);
  const lastFrameIdRef = useRef(null);
  const fpsTimestampsRef = useRef([]);
  const reconnectDelayRef = useRef(500);
  const reconnectTimerRef = useRef(null);
  const unmountedRef = useRef(false);

  // Fetch static config summary & scenario modes on startup
  useEffect(() => {
    let active = true;
    async function loadMeta() {
      try {
        const [cfgRes, modesRes] = await Promise.all([
          fetch(`${apiBase}/api/config-summary`),
          fetch(`${apiBase}/api/scenario-modes`),
        ]);
        if (!active) return;
        if (cfgRes.ok) {
          const cfg = await cfgRes.json();
          setConfigSummary(cfg);
          if (cfg.control_state) setControlState(cfg.control_state);
        }
        if (modesRes.ok) {
          setScenarioModes(await modesRes.json());
        }
      } catch (err) {
        if (active) setLastError(`Backend metadata fetch failed: ${err.message}`);
      }
    }
    loadMeta();
    return () => {
      active = false;
    };
  }, [apiBase]);

  // Flush the newest buffered WebSocket frame at most once per animation frame
  const commitPendingFrame = useCallback(() => {
    rafRef.current = null;
    const msg = pendingFrameRef.current;
    pendingFrameRef.current = null;
    if (!msg) return;

    const now = performance.now();
    const stamps = fpsTimestampsRef.current;
    stamps.push(now);
    const cutoff = now - 2000;
    while (stamps.length > 0 && stamps[0] < cutoff) stamps.shift();
    if (stamps.length >= 2) {
      const spanS = (stamps[stamps.length - 1] - stamps[0]) / 1000;
      if (spanS > 0.05) {
        setMeasuredFps(Math.round(((stamps.length - 1) / spanS) * 10) / 10);
      }
    }

    setFrame(msg);
    if (msg.control_state) setControlState(msg.control_state);

    const feat = msg.analytics?.features || {};
    const fallActive = Boolean(msg.analytics?.fall_alert?.active);
    const dets = msg.radar?.detections || [];
    const peakSnr =
      dets.length > 0 ? Math.max(...dets.map((d) => (typeof d.snr_db === 'number' ? d.snr_db : -99))) : null;

    const entry = {
      frame_id: msg.frame_id,
      sim_time_s: msg.sim_time_s,
      action_label: msg.ground_truth?.action_label || 'unknown',
      height_m: typeof feat.estimated_height_m === 'number' ? feat.estimated_height_m : null,
      vz_mps: typeof feat.vertical_velocity_mps === 'number' ? feat.vertical_velocity_mps : 0,
      detections: dets.length,
      peak_snr_db: peakSnr,
      fall_active: fallActive,
    };
    setTimeline((prev) => {
      const next = prev.length >= 180 ? prev.slice(prev.length - 179) : prev.slice();
      next.push(entry);
      return next;
    });

    // Extract 1D Doppler profile slice [N_d] (max over valid range bins) for rolling Micro-Doppler spectrogram
    const rd = msg.analytics?.range_doppler;
    if (rd && Array.isArray(rd.values_db) && Array.isArray(rd.velocity_mps) && rd.values_db.length > 0) {
      const sliceDb = rd.values_db.map((row) => {
        if (!Array.isArray(row) || row.length === 0) return -120;
        let m = -140;
        for (let i = 0; i < row.length; i++) {
          if (row[i] > m) m = row[i];
        }
        return m;
      });
      setMicroDoppler((prev) => {
        const sameShape =
          prev &&
          Array.isArray(prev.velocity_mps) &&
          prev.velocity_mps.length === rd.velocity_mps.length;
        const times = sameShape ? prev.times.slice(-119) : [];
        const cols = sameShape ? prev.cols.slice(-119) : [];
        times.push(msg.sim_time_s);
        cols.push(sliceDb);
        return {
          velocity_mps: rd.velocity_mps,
          times,
          cols,
        };
      });
    }
  }, []);

  useEffect(() => {
    unmountedRef.current = false;

    function connect() {
      if (unmountedRef.current) return;
      setConnectionStatus('connecting');
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectDelayRef.current = 500;
        setConnectionStatus('connected');
        setLastError(null);
      };

      ws.onmessage = (event) => {
        let msg;
        try {
          msg = JSON.parse(event.data);
        } catch {
          return;
        }
        if (msg.type === 'control_ack') {
          if (msg.control_state) setControlState(msg.control_state);
          if (msg.action === 'regenerate_scenario' || msg.action === 'set_seed' || msg.action === 'set_mode') {
            setTimeline([]);
            setMicroDoppler(null);
          }
          setLastError(null);
          return;
        }
        if (msg.type === 'control_error') {
          setLastError(msg.error || 'Control error');
          return;
        }
        if (msg.type === 'simulation_frame') {
          if (lastFrameIdRef.current !== null && msg.frame_id > lastFrameIdRef.current + 1) {
            setDroppedFrames((d) => d + (msg.frame_id - lastFrameIdRef.current - 1));
          }
          if (pendingFrameRef.current !== null) {
            setDroppedFrames((d) => d + 1);
          }
          lastFrameIdRef.current = msg.frame_id;
          pendingFrameRef.current = msg;
          if (rafRef.current === null) {
            rafRef.current = requestAnimationFrame(commitPendingFrame);
          }
        }
      };

      ws.onerror = () => {
        setConnectionStatus('reconnecting');
      };

      ws.onclose = () => {
        if (unmountedRef.current) return;
        setConnectionStatus('reconnecting');
        const delay = reconnectDelayRef.current;
        reconnectDelayRef.current = Math.min(3000, Math.round(delay * 1.5));
        reconnectTimerRef.current = setTimeout(connect, delay);
      };
    }

    connect();

    return () => {
      unmountedRef.current = true;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [wsUrl, commitPendingFrame]);

  const sendControl = useCallback((payload) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'control', ...payload }));
    }
  }, []);

  return {
    apiBase,
    configSummary,
    scenarioModes,
    frame,
    controlState,
    connectionStatus,
    lastError,
    measuredFps,
    droppedFrames,
    timeline,
    microDoppler,
    sendControl,
  };
}
