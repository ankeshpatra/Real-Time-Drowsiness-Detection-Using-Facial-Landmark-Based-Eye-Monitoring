import { useState } from 'react';
import { useDrowsinessDetector } from './hooks/useDrowsinessDetector';
import VideoFeed from './components/VideoFeed';
import MetricsPanel from './components/MetricsPanel';
import AlertBanner from './components/AlertBanner';
import SettingsBar from './components/SettingsBar';
import EventLog from './components/EventLog';
import './App.css';

function getStatusBadgeClass(status: string): string {
  switch (status) {
    case 'awake':
    case 'active':
      return 'safe';
    case 'alert':
      return 'danger';
    case 'no-face':
    case 'offline':
    default:
      return '';
  }
}

function getStatusLabel(status: string): string {
  switch (status) {
    case 'awake':
      return 'Awake';
    case 'active':
      return 'Active';
    case 'alert':
      return 'ALERT';
    case 'no-face':
      return 'No Face';
    case 'offline':
    default:
      return 'Offline';
  }
}

function App() {
  const [state, actions, refs] = useDrowsinessDetector();
  const [earThreshold, setEarThresholdLocal] = useState(0.22);
  const [frameThreshold, setFrameThresholdLocal] = useState(20);

  const handleEarThresholdChange = (value: number) => {
    setEarThresholdLocal(value);
    actions.setEarThreshold(value);
  };

  const handleFrameThresholdChange = (value: number) => {
    setFrameThresholdLocal(value);
    actions.setFrameThreshold(value);
  };

  return (
    <>
      <h2 className="sr-only">
        Real-time driver drowsiness detection system using face mesh eye landmark analysis
      </h2>

      {/* Top Bar */}
      <div className="top-bar" id="top-bar">
        <h1>Driver Drowsiness Monitor</h1>
        <div className="top-bar-right">
          <span className="fps-label">{state.isRunning ? `${state.fps} fps` : '— fps'}</span>
          <span className={`status-badge ${getStatusBadgeClass(state.status)}`} id="status-badge">
            {getStatusLabel(state.status)}
          </span>
        </div>
      </div>

      {/* Main Grid */}
      <div className="main-grid" id="main-grid">
        <VideoFeed
          videoRef={refs.videoRef}
          canvasRef={refs.canvasRef}
          isRunning={state.isRunning}
          ear={state.ear}
          earThreshold={earThreshold}
        />
        <MetricsPanel
          ear={state.ear}
          closedFrames={state.closedFrames}
          alertCount={state.alertCount}
          sessionStart={state.sessionStart}
          earThreshold={earThreshold}
          frameThreshold={frameThreshold}
        />
      </div>

      {/* Alert Banner */}
      <AlertBanner
        active={state.alertActive}
        closedFrames={state.closedFrames}
        onDismiss={actions.dismissAlert}
      />

      {/* Settings Bar */}
      <SettingsBar
        isRunning={state.isRunning}
        onStart={actions.startCamera}
        onStop={actions.stopCamera}
        onEarThresholdChange={handleEarThresholdChange}
        onFrameThresholdChange={handleFrameThresholdChange}
      />

      {/* Event Log */}
      <EventLog logs={state.logs} />
    </>
  );
}

export default App;
