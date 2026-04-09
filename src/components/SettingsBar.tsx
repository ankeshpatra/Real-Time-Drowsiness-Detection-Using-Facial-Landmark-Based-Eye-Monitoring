import { useState } from 'react';
import './SettingsBar.css';

interface SettingsBarProps {
  isRunning: boolean;
  onStart: () => void;
  onStop: () => void;
  onEarThresholdChange: (value: number) => void;
  onFrameThresholdChange: (value: number) => void;
}

export default function SettingsBar({
  isRunning,
  onStart,
  onStop,
  onEarThresholdChange,
  onFrameThresholdChange,
}: SettingsBarProps) {
  const [earSlider, setEarSlider] = useState(22);
  const [frameSlider, setFrameSlider] = useState(20);

  const handleEarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = parseInt(e.target.value);
    setEarSlider(raw);
    onEarThresholdChange(raw / 100);
  };

  const handleFrameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = parseInt(e.target.value);
    setFrameSlider(raw);
    onFrameThresholdChange(raw);
  };

  return (
    <div className="card">
      <div className="settings-row">
        <div className="sett">
          <label>EAR threshold</label>
          <input
            type="range"
            id="sl-ear"
            min={15}
            max={40}
            value={earSlider}
            step={1}
            onChange={handleEarChange}
          />
          <span>{(earSlider / 100).toFixed(2)}</span>
        </div>
        <div className="sett">
          <label>Alert after (frames)</label>
          <input
            type="range"
            id="sl-frames"
            min={10}
            max={60}
            value={frameSlider}
            step={1}
            onChange={handleFrameChange}
          />
          <span>{frameSlider}</span>
        </div>
        <div className="controls">
          {!isRunning ? (
            <button className="primary" id="btn-start" onClick={onStart}>
              Start Camera
            </button>
          ) : (
            <button className="stop-btn" id="btn-stop" onClick={onStop}>
              Stop
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
