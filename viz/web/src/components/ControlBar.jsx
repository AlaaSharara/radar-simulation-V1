import React, { useEffect, useState } from 'react';

const IMPAIRMENT_LABELS = [
  ['phase_noise', 'Phase noise'],
  ['chirp_slope_error', 'Chirp slope error'],
  ['chirp_start_frequency_error', 'Start freq error'],
  ['tx_rx_channel_mismatch', 'TX/RX mismatch'],
  ['iq_imbalance', 'IQ imbalance'],
  ['adc_quantization', 'ADC quantization'],
  ['adc_clipping', 'ADC clipping'],
  ['tx_leakage', 'TX leakage'],
  ['timing_jitter', 'Timing jitter'],
  ['thermal_noise', 'Thermal noise'],
];

export default function ControlBar({
  configSummary,
  scenarioModes,
  frame,
  controlState,
  connectionStatus,
  measuredFps,
  droppedFrames,
  lastError,
  sendControl,
}) {
  const [seedDraft, setSeedDraft] = useState('20261123');
  const [showImpairments, setShowImpairments] = useState(false);

  useEffect(() => {
    if (controlState && typeof controlState.seed === 'number') {
      setSeedDraft(String(controlState.seed));
    }
  }, [controlState?.seed]);

  const playing = controlState?.playing ?? true;
  const speed = controlState?.speed ?? 1.0;
  const scenarioMode = controlState?.scenario_mode ?? 'random_mix';
  const fallStyle = controlState?.fall_style || 'fall_forward';
  const mode = controlState?.mode ?? 'preview';
  const impairments = controlState?.impairments || {};
  const speedValues = configSummary?.sim_profile?.speed_values || [0.25, 0.5, 1.0, 1.5, 2.0];
  const modeList = scenarioModes?.modes
    ? Object.keys(scenarioModes.modes)
    : ['normal_daily_routine', 'fall_demo', 'random_mix'];
  const fallStyles = scenarioModes?.fall_styles || [
    'fall_forward',
    'fall_backward',
    'fall_sideways',
    'fall_slump',
    'fall_slow_collapse',
  ];

  const enabledCount = Object.values(impairments).filter(Boolean).length;

  function applySeed() {
    const parsed = parseInt(seedDraft, 10);
    if (!Number.isNaN(parsed)) {
      sendControl({ action: 'set_seed', value: parsed });
    }
  }

  return (
    <header className="control-bar">
      <div className="brand-group">
        <h1>60 GHz FMCW Room Radar Sim</h1>
        <span className={`conn-pill conn-${connectionStatus}`}>{connectionStatus}</span>
        <span className="mode-pill">
          {frame?.provenance?.fidelity_label || mode}
        </span>
      </div>

      <div className="controls-group">
        <button
          className="ctrl-btn primary"
          onClick={() => sendControl({ action: playing ? 'pause' : 'play' })}
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          className="ctrl-btn"
          onClick={() => sendControl({ action: 'step' })}
        >
          Step
        </button>

        <label className="ctrl-field">
          <span>Speed</span>
          <select
            value={speed}
            onChange={(e) => sendControl({ action: 'set_speed', value: parseFloat(e.target.value) })}
          >
            {speedValues.map((v) => (
              <option key={v} value={v}>
                {v}x
              </option>
            ))}
          </select>
        </label>

        <label className="ctrl-field">
          <span>Seed</span>
          <input
            type="number"
            value={seedDraft}
            onChange={(e) => setSeedDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') applySeed();
            }}
          />
          <button className="ctrl-btn small" onClick={applySeed}>
            Set
          </button>
        </label>

        <label className="ctrl-field">
          <span>Scenario</span>
          <select
            value={scenarioMode}
            onChange={(e) =>
              sendControl({ action: 'regenerate_scenario', scenario_mode: e.target.value })
            }
          >
            {modeList.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>

        {scenarioMode === 'fall_demo' && (
          <label className="ctrl-field">
            <span>Fall</span>
            <select
              value={fallStyle}
              onChange={(e) =>
                sendControl({
                  action: 'regenerate_scenario',
                  scenario_mode: 'fall_demo',
                  value: controlState?.seed ?? 20261123,
                  fall_style: e.target.value,
                })
              }
            >
              {fallStyles.map((fs) => (
                <option key={fs} value={fs}>
                  {fs}
                </option>
              ))}
            </select>
          </label>
        )}

        <button
          className="ctrl-btn"
          onClick={() => {
            const nextSeed = (controlState?.seed ?? 20261123) + 1;
            sendControl({
              action: 'regenerate_scenario',
              scenario_mode: scenarioMode,
              value: nextSeed,
            });
          }}
        >
          Regenerate
        </button>

        <label className="ctrl-field">
          <span>Fidelity</span>
          <select
            value={mode}
            onChange={(e) => sendControl({ action: 'set_mode', mode: e.target.value })}
          >
            <option value="preview">preview</option>
            <option value="full_sim">full_sim</option>
          </select>
        </label>

        <div className="impairments-wrap">
          <button
            className="ctrl-btn"
            onClick={() => setShowImpairments((s) => !s)}
          >
            Impairments ({enabledCount}/10)
          </button>
          {showImpairments && (
            <div className="impairments-popover">
              <div className="popover-title">RF / ADC Impairment Toggles</div>
              {IMPAIRMENT_LABELS.map(([key, label]) => (
                <label key={key} className="impairment-row">
                  <input
                    type="checkbox"
                    checked={Boolean(impairments[key])}
                    onChange={(e) =>
                      sendControl({
                        action: 'set_impairment',
                        name: key,
                        enabled: e.target.checked,
                      })
                    }
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="telemetry-group">
        <span>Frame: <strong>{frame?.frame_id ?? '--'}</strong></span>
        <span>Time: <strong>{frame ? `${frame.sim_time_s.toFixed(2)}s` : '--'}</strong></span>
        <span>FPS: <strong>{measuredFps.toFixed(1)}</strong></span>
        <span>Dropped: <strong>{droppedFrames}</strong></span>
      </div>

      {lastError && <div className="error-strip">{lastError}</div>}
    </header>
  );
}
