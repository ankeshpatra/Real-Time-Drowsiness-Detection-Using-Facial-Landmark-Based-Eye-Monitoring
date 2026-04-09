import './AlertBanner.css';

interface AlertBannerProps {
  active: boolean;
  closedFrames: number;
  onDismiss: () => void;
}

export default function AlertBanner({ active, closedFrames, onDismiss }: AlertBannerProps) {
  return (
    <div className={`alert-box${active ? ' show' : ''}`} id="alert-box">
      <svg className="alert-icon" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="16" cy="16" r="15" fill="#FCEBEB" stroke="#E24B4A" strokeWidth="1.5" />
        <path d="M16 8v10M16 22v2" stroke="#A32D2D" strokeWidth="2.5" strokeLinecap="round" />
      </svg>
      <div className="alert-text">
        DROWSINESS DETECTED
        <small>
          Eyes closed for ~{(closedFrames / 20).toFixed(1)}s. Please take a break.
        </small>
      </div>
      <button onClick={onDismiss}>Dismiss</button>
    </div>
  );
}
