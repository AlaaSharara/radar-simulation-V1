import React, { useEffect, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';

const PLOT_CONFIG = { displayModeBar: false, responsive: true, staticPlot: false };

// Explicit valid perceptual Plotly colorscales (avoids invalid string aliases)
const RADAR_VIRIDIS = [
  [0.0, '#091124'],
  [0.2, '#1f3a60'],
  [0.45, '#1d7874'],
  [0.72, '#5cc963'],
  [0.9, '#c7e020'],
  [1.0, '#fde725'],
];

const RADAR_INFERNO = [
  [0.0, '#090b16'],
  [0.25, '#3b0f6f'],
  [0.5, '#8c2981'],
  [0.75, '#de4968'],
  [0.92, '#fe9f6d'],
  [1.0, '#fcfdbf'],
];

const BASE_LAYOUT = {
  paper_bgcolor: '#0f172a',
  plot_bgcolor: '#0b1220',
  font: { color: '#cbd5e1', size: 10, family: 'system-ui, sans-serif' },
  margin: { l: 42, r: 28, t: 24, b: 30 },
};

function matrixMax(mat) {
  let m = -Infinity;
  for (let i = 0; i < mat.length; i++) {
    const row = mat[i];
    for (let j = 0; j < row.length; j++) {
      if (row[j] > m) m = row[j];
    }
  }
  return Number.isFinite(m) ? m : -50;
}

/**
 * Panel 4: Pro Radar Signal & Event Diagnostics.
 * - Range-Doppler [dB] with overlaid OS-CFAR detection markers.
 * - Segmented top-right spectral view: Range-Azimuth B-scan | Az-El 4x3 | Micro-Doppler.
 * - Track height / vertical velocity timeline with explicit rule threshold reference lines.
 */
export default function AnalyticsPanel({ frame, timeline, microDoppler }) {
  const rdRef = useRef(null);
  const raRef = useRef(null);
  const tsRef = useRef(null);
  const [angleViewMode, setAngleViewMode] = useState('range_azimuth'); // 'range_azimuth' | 'az_el' | 'micro_doppler'

  // 1. Update Range-Doppler heatmap + overlaid OS-CFAR detections
  useEffect(() => {
    const el = rdRef.current;
    const rd = frame?.analytics?.range_doppler;
    if (!el || !rd || !rd.values_db || rd.values_db.length === 0) return;

    const zMax = matrixMax(rd.values_db);
    const zMin = zMax - 42;
    const dets = frame?.radar?.detections || [];

    const traces = [
      {
        type: 'heatmap',
        x: rd.range_m,
        y: rd.velocity_mps,
        z: rd.values_db,
        colorscale: RADAR_VIRIDIS,
        zmin: zMin,
        zmax: zMax,
        colorbar: { thickness: 8, len: 0.85, tickfont: { size: 8, color: '#94a3b8' } },
      },
    ];
    if (dets.length > 0) {
      traces.push({
        type: 'scatter',
        mode: 'markers',
        name: 'CFAR',
        x: dets.map((d) => d.range_m),
        y: dets.map((d) => d.radial_velocity_mps),
        marker: {
          color: '#ef4444',
          size: 6,
          symbol: 'circle',
          line: { color: '#ffffff', width: 1 },
        },
        showlegend: false,
      });
    }

    Plotly.react(
      el,
      traces,
      {
        ...BASE_LAYOUT,
        title: { text: `Range-Doppler [dB] (${rd.branch || 'moving'})`, font: { size: 11 } },
        xaxis: { title: 'Range [m]', gridcolor: '#1e293b', range: [0, Math.min(7.2, rd.range_m[rd.range_m.length - 1] || 7.2)] },
        yaxis: { title: 'v_r [m/s]', gridcolor: '#1e293b' },
      },
      PLOT_CONFIG
    );
  }, [frame]);

  // 2. Update Top-Right Multi-View Spectrum (Range-Azimuth | Az-El | Micro-Doppler)
  useEffect(() => {
    const el = raRef.current;
    if (!el) return;

    if (angleViewMode === 'micro_doppler') {
      if (!microDoppler || !microDoppler.cols || microDoppler.cols.length === 0) {
        Plotly.react(
          el,
          [],
          {
            ...BASE_LAYOUT,
            title: { text: 'Micro-Doppler Spectrogram (warming...)', font: { size: 11 } },
            xaxis: { title: 'Time [s]', gridcolor: '#1e293b' },
            yaxis: { title: 'v_r [m/s]', gridcolor: '#1e293b' },
          },
          PLOT_CONFIG
        );
        return;
      }
      // Transpose cols[timeIdx][velIdx] -> z[velIdx][timeIdx]
      const nV = microDoppler.velocity_mps.length;
      const nT = microDoppler.cols.length;
      const zMat = [];
      for (let v = 0; v < nV; v++) {
        const row = new Array(nT);
        for (let t = 0; t < nT; t++) {
          row[t] = microDoppler.cols[t][v];
        }
        zMat.push(row);
      }
      const zMax = matrixMax(zMat);
      Plotly.react(
        el,
        [
          {
            type: 'heatmap',
            x: microDoppler.times,
            y: microDoppler.velocity_mps,
            z: zMat,
            colorscale: RADAR_VIRIDIS,
            zmin: zMax - 38,
            zmax: zMax,
            colorbar: { thickness: 8, len: 0.85, tickfont: { size: 8, color: '#94a3b8' } },
          },
        ],
        {
          ...BASE_LAYOUT,
          title: { text: 'Micro-Doppler Waterfall [dB]', font: { size: 11 } },
          xaxis: { title: 'Sim Time [s]', gridcolor: '#1e293b' },
          yaxis: { title: 'v_r [m/s]', gridcolor: '#1e293b' },
        },
        PLOT_CONFIG
      );
      return;
    }

    const ra = frame?.analytics?.range_angle;
    if (!ra) return;
    const azDeg = (ra.azimuth_rad || []).map((r) => Math.round((r * 180) / Math.PI));
    const elDeg = (ra.elevation_rad || []).map((r) => Math.round((r * 180) / Math.PI));
    const dets = frame?.radar?.detections || [];

    if (angleViewMode === 'range_azimuth' && Array.isArray(ra.range_azimuth_db) && Array.isArray(ra.range_m)) {
      const zMax = matrixMax(ra.range_azimuth_db);
      const traces = [
        {
          type: 'heatmap',
          x: azDeg,
          y: ra.range_m,
          z: ra.range_azimuth_db,
          colorscale: RADAR_INFERNO,
          zmin: zMax - 30,
          zmax: zMax,
          colorbar: { thickness: 8, len: 0.85, tickfont: { size: 8, color: '#94a3b8' } },
        },
      ];
      if (dets.length > 0) {
        traces.push({
          type: 'scatter',
          mode: 'markers',
          x: dets.map((d) => (d.azimuth_rad * 180) / Math.PI),
          y: dets.map((d) => d.range_m),
          marker: { color: '#38bdf8', size: 6, symbol: 'cross', line: { color: '#ffffff', width: 1 } },
          showlegend: false,
        });
      }
      Plotly.react(
        el,
        traces,
        {
          ...BASE_LAYOUT,
          title: { text: 'Range-Azimuth B-Scan [dB] (4x3 interp.)', font: { size: 11 } },
          xaxis: { title: 'Azimuth [deg]', gridcolor: '#1e293b' },
          yaxis: { title: 'Range [m]', gridcolor: '#1e293b', range: [0, Math.min(7.2, ra.range_m[ra.range_m.length - 1] || 7.2)] },
        },
        PLOT_CONFIG
      );
      return;
    }

    if (ra.values_db && ra.values_db.length > 0) {
      const zMax = matrixMax(ra.values_db);
      const traces = [
        {
          type: 'heatmap',
          x: azDeg,
          y: elDeg,
          z: ra.values_db,
          colorscale: RADAR_INFERNO,
          zmin: zMax - 24,
          zmax: zMax,
          colorbar: { thickness: 8, len: 0.85, tickfont: { size: 8, color: '#94a3b8' } },
        },
      ];
      if (dets.length > 0) {
        traces.push({
          type: 'scatter',
          mode: 'markers',
          x: dets.map((d) => (d.azimuth_rad * 180) / Math.PI),
          y: dets.map((d) => (d.elevation_rad * 180) / Math.PI),
          marker: { color: '#38bdf8', size: 7, symbol: 'cross' },
          showlegend: false,
        });
      }
      Plotly.react(
        el,
        traces,
        {
          ...BASE_LAYOUT,
          title: { text: 'Azimuth-Elevation [dB] (4x3 interp.)', font: { size: 11 } },
          xaxis: { title: 'Az [deg]', gridcolor: '#1e293b' },
          yaxis: { title: 'El [deg]', gridcolor: '#1e293b' },
        },
        PLOT_CONFIG
      );
    }
  }, [frame, angleViewMode, microDoppler]);

  // 3. Update Height / Vertical Speed / Fall Thresholds Timeline
  useEffect(() => {
    const el = tsRef.current;
    if (!el || !timeline || timeline.length === 0) return;

    const times = timeline.map((p) => p.sim_time_s);
    const heights = timeline.map((p) => p.height_m);
    const vzs = timeline.map((p) => p.vz_mps);
    const alertT = timeline.filter((p) => p.fall_active).map((p) => p.sim_time_s);
    const alertZ = timeline.filter((p) => p.fall_active).map((p) => p.height_m ?? 0.2);

    const traces = [
      {
        type: 'scatter',
        mode: 'lines',
        name: 'Est. height [m]',
        x: times,
        y: heights,
        line: { color: '#38bdf8', width: 2.1 },
      },
      {
        type: 'scatter',
        mode: 'lines',
        name: 'v_z [m/s]',
        x: times,
        y: vzs,
        line: { color: '#f59e0b', width: 1.5, dash: 'dot' },
      },
    ];
    if (alertT.length > 0) {
      traces.push({
        type: 'scatter',
        mode: 'markers',
        name: 'Fall alert',
        x: alertT,
        y: alertZ,
        marker: { color: '#ef4444', size: 7, symbol: 'diamond' },
      });
    }

    const t0 = times[0];
    const t1 = times[times.length - 1];
    Plotly.react(
      el,
      traces,
      {
        ...BASE_LAYOUT,
        margin: { l: 38, r: 12, t: 22, b: 26 },
        title: { text: 'Track Height, Vertical Speed & Rule Thresholds', font: { size: 11 } },
        xaxis: { title: 'Sim Time [s]', gridcolor: '#1e293b' },
        yaxis: { title: 'm / m/s', gridcolor: '#1e293b', range: [-1.5, 2.0] },
        legend: { orientation: 'h', x: 0, y: 1.04, font: { size: 9 } },
        shapes: [
          {
            type: 'line',
            x0: t0,
            x1: t1,
            y0: 0.48,
            y1: 0.48,
            line: { color: 'rgba(56, 189, 248, 0.35)', width: 1, dash: 'dash' },
          },
          {
            type: 'line',
            x0: t0,
            x1: t1,
            y0: -0.55,
            y1: -0.55,
            line: { color: 'rgba(245, 158, 11, 0.40)', width: 1, dash: 'dash' },
          },
        ],
      },
      PLOT_CONFIG
    );
  }, [timeline]);

  useEffect(() => {
    return () => {
      if (rdRef.current) Plotly.purge(rdRef.current);
      if (raRef.current) Plotly.purge(raRef.current);
      if (tsRef.current) Plotly.purge(tsRef.current);
    };
  }, []);

  const fallAlert = frame?.analytics?.fall_alert;
  const feat = frame?.analytics?.features || {};
  const gtAction = frame?.ground_truth?.action_label || '--';
  const gtFall = frame?.ground_truth?.fall_event_id;
  const primaryTrack = frame?.radar?.tracks?.[0];
  const dets = frame?.radar?.detections || [];
  const peakSnr = dets.length > 0 ? Math.max(...dets.map((d) => d.snr_db)) : null;
  const breath = feat.breathing_phase_branch;
  const lim = frame?.analytics?.aperture_limitations;

  const alertClass = fallAlert?.active
    ? 'alert-banner alert-danger'
    : fallAlert?.state === 'monitoring_drop'
      ? 'alert-banner alert-warn'
      : 'alert-banner alert-ok';

  return (
    <section className="panel">
      <header className="panel-header">
        <div className="panel-title-group">
          <span className="panel-badge badge-diag">PANEL 4</span>
          <h2>Signal &amp; Event Diagnostics</h2>
        </div>
        <div className="panel-meta">
          <div className="seg-group" aria-label="Angle/Doppler chart view">
            <button
              type="button"
              className={`seg-btn ${angleViewMode === 'range_azimuth' ? 'active' : ''}`}
              onClick={() => setAngleViewMode('range_azimuth')}
            >
              Range-Azimuth
            </button>
            <button
              type="button"
              className={`seg-btn ${angleViewMode === 'az_el' ? 'active' : ''}`}
              onClick={() => setAngleViewMode('az_el')}
            >
              Az-El
            </button>
            <button
              type="button"
              className={`seg-btn ${angleViewMode === 'micro_doppler' ? 'active' : ''}`}
              onClick={() => setAngleViewMode('micro_doppler')}
            >
              Micro-Doppler
            </button>
          </div>
          <span className="clock-chip">
            f={frame?.frame_id ?? '--'} | t={frame ? frame.sim_time_s.toFixed(2) : '--'}s
          </span>
        </div>
      </header>
      <div className="analytics-body">
        <div className={alertClass}>
          <div className="alert-left">
            <strong>
              {fallAlert?.active
                ? 'FALL ALERT (Baseline Rule)'
                : fallAlert?.state === 'monitoring_drop'
                  ? 'MONITORING RAPID DROP'
                  : 'STATUS: NORMAL'}
            </strong>
            <span>
              Conf: {fallAlert ? `${Math.round((fallAlert.confidence || 0) * 100)}%` : '--'} |{' '}
              Rule: {fallAlert?.rule_version || 'baseline_v1'}
            </span>
          </div>
          <div className="alert-right">
            <span>
              <strong>Track:</strong>{' '}
              {primaryTrack
                ? `T${primaryTrack.track_id} (${primaryTrack.state})`
                : 'none'}
            </span>
            <span>
              <strong>SNR:</strong> {peakSnr !== null ? `${peakSnr.toFixed(1)} dB` : '--'} |{' '}
              <strong>z:</strong> {typeof feat.estimated_height_m === 'number' ? `${feat.estimated_height_m.toFixed(2)}m` : '--'} |{' '}
              <strong>v_z:</strong> {typeof feat.vertical_velocity_mps === 'number' ? `${feat.vertical_velocity_mps.toFixed(2)}m/s` : '--'}
            </span>
            {breath && typeof breath.displacement_mm === 'number' && (
              <span className="muted-kpi">
                Resp(opt): {breath.displacement_mm.toFixed(1)}mm
              </span>
            )}
            <span className="gt-label-sep">
              <strong>GT (eval):</strong> {gtAction}{gtFall ? ` [${gtFall}]` : ''}
            </span>
          </div>
        </div>

        <div className="charts-top">
          <div ref={rdRef} className="chart-box" />
          <div ref={raRef} className="chart-box" />
        </div>
        <div ref={tsRef} className="chart-timeline" />

        {lim && (
          <div className="aperture-note">
            Aperture: {lim.azimuth_elements}×{lim.elevation_elements} virtual array (~{lim.azimuth_beamwidth_3db_deg}° az × {lim.elevation_beamwidth_3db_deg}° el 3 dB beamwidth); zero-padding interpolates bins only.
          </div>
        )}
      </div>
    </section>
  );
}
