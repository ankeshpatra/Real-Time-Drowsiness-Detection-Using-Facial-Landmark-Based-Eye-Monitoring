import type { LogEntry } from '../hooks/useDrowsinessDetector';
import './EventLog.css';

interface EventLogProps {
  logs: LogEntry[];
}

export default function EventLog({ logs }: EventLogProps) {
  return (
    <div className="card log-card" id="log-card">
      <div className="log">
        {logs.map((entry) => (
          <div
            key={entry.id}
            className={`log-entry${entry.level ? ` ev-${entry.level}` : ''}`}
          >
            [{entry.timestamp}] {entry.message}
          </div>
        ))}
      </div>
    </div>
  );
}
