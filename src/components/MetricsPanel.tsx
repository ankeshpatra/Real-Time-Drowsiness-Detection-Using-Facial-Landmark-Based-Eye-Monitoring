import { useState, useEffect } from 'react';
import './MetricsPanel.css';

interface MetricsPanelProps {
  ear: number | null;
  closedFrames: number;
  alertCount: number;
  sessionStart: number | null;
  earThreshold: number;
  frameThreshold: number;
}

export default function MetricsPanel({
  ear,
  closedFrames,
  alertCount,
  sessionStart,
  earThreshold,
  frameThreshold,
}: MetricsPanelProps) {
  const [elapsed, setElapsed] = useState('0:00');

  useEffect(() => {
    if (!sessionStart) {
      setElapsed('0:00');
      return;
    }
    const interval = setInterval(() => {
      const sec = Math.floor((Date.now() - sessionStart) / 1000);
      setElapsed(Math.floor(sec / 60) + ':' + (sec % 60).toString().padStart(2, '0'));
    }, 1000);
    return () => clearInterval(interval);
  }, [sessionStart]);

  const eyesClosed = ear !== null && ear < earThreshold;
  const earClass = ear === null
    ? ''
    : eyesClosed
      ? 'danger-col'
      : ear < earThreshold + 0.03
        ? 'warn-col'
        : 'safe-col';

  const framesLevel =
    closedFrames === 0
      ? 'safe-col'
      : closedFrames < frameThreshold * 0.5
        ? 'warn-col'
        : closedFrames < frameThreshold
          ? 'warn-col'
          : 'danger-col';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div className="card metric-row">
        <div className="metric">
          <div className="lbl">Eye Aspect Ratio</div>
          <div className={`val ${earClass}`} id="m-ear">
            {ear !== null ? ear.toFixed(3) : '—'}
          </div>
        </div>
        <div className="metric">
          <div className="lbl">Eyes Closed (consecutive frames)</div>
          <div className={`val ${framesLevel}`} id="m-frames">{closedFrames}</div>
        </div>
        <div className="metric">
          <div className="lbl">Alerts triggered</div>
          <div className="val" id="m-alerts">{alertCount}</div>
        </div>
        <div className="metric">
          <div className="lbl">Session duration</div>
          <div className="val" id="m-time">{elapsed}</div>
        </div>
      </div>
    </div>
  );
}
