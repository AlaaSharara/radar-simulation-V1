import React from 'react';
import { useSimSocket } from './lib/useSimSocket.js';
import ControlBar from './components/ControlBar.jsx';
import ScenePanel from './components/ScenePanel.jsx';
import RFPanel from './components/RFPanel.jsx';
import PointCloudPanel from './components/PointCloudPanel.jsx';
import AnalyticsPanel from './components/AnalyticsPanel.jsx';

export default function App() {
  const {
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
  } = useSimSocket();

  const sceneData = configSummary?.scene || null;

  return (
    <div className="dashboard-root">
      <ControlBar
        configSummary={configSummary}
        scenarioModes={scenarioModes}
        frame={frame}
        controlState={controlState}
        connectionStatus={connectionStatus}
        measuredFps={measuredFps}
        droppedFrames={droppedFrames}
        lastError={lastError}
        sendControl={sendControl}
      />
      <main className="panel-grid">
        <ScenePanel apiBase={apiBase} sceneData={sceneData} frame={frame} />
        <RFPanel sceneData={sceneData} frame={frame} />
        <PointCloudPanel sceneData={sceneData} frame={frame} />
        <AnalyticsPanel frame={frame} timeline={timeline} microDoppler={microDoppler} />
      </main>
    </div>
  );
}
