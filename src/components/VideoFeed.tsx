import type { RefObject } from 'react';
import './VideoFeed.css';

interface VideoFeedProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  isRunning: boolean;
  ear: number | null;
  earThreshold: number;
}

export default function VideoFeed({ videoRef, canvasRef, isRunning, ear, earThreshold }: VideoFeedProps) {
  const earPct = ear !== null ? Math.min(100, Math.max(0, (ear / 0.4) * 100)) : 0;
  const eyesClosed = ear !== null && ear < earThreshold;
  const fillColor = eyesClosed
    ? '#E24B4A'
    : ear !== null && ear < earThreshold + 0.03
      ? '#EF9F27'
      : '#1D9E75';

  return (
    <div className="card video-card" id="video-card">
      <div className="video-wrap">
        {!isRunning && (
          <div className="placeholder">Click &quot;Start Camera&quot; to begin monitoring</div>
        )}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{ display: isRunning ? 'block' : 'none' }}
        />
        <canvas ref={canvasRef} className="overlay-canvas" />
      </div>
      <div className="ear-bar-wrap">
        <div className="ear-label">
          <span>Eye Aspect Ratio (EAR)</span>
          <span>{ear !== null ? ear.toFixed(3) : '—'}</span>
        </div>
        <div className="ear-track">
          <div
            className="ear-fill"
            style={{
              width: `${earPct.toFixed(0)}%`,
              background: fillColor,
            }}
          />
        </div>
      </div>
    </div>
  );
}
