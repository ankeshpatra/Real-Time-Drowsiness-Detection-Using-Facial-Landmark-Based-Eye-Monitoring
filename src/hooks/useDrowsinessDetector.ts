import { useRef, useState, useCallback, useEffect } from 'react';
import { computeAverageEAR, LEFT_EYE_INDICES, RIGHT_EYE_INDICES } from '../utils/ear';
import { beepAlert } from '../utils/audio';

export interface LogEntry {
  id: number;
  timestamp: string;
  message: string;
  level: '' | 'warn' | 'alert';
}

export interface DetectorState {
  /** Current computed EAR value */
  ear: number | null;
  /** Number of consecutive frames with eyes closed */
  closedFrames: number;
  /** Total alerts triggered this session */
  alertCount: number;
  /** Whether the alert banner is currently active */
  alertActive: boolean;
  /** Session start timestamp (ms) or null if stopped */
  sessionStart: number | null;
  /** Current status label */
  status: 'offline' | 'active' | 'awake' | 'no-face' | 'alert';
  /** Current FPS */
  fps: number;
  /** Event log entries */
  logs: LogEntry[];
  /** Whether camera is running */
  isRunning: boolean;
}

export interface DetectorActions {
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  dismissAlert: () => void;
  setEarThreshold: (value: number) => void;
  setFrameThreshold: (value: number) => void;
}

export interface DetectorRefs {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
}

export function useDrowsinessDetector(): [DetectorState, DetectorActions, DetectorRefs] {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Mutable refs for values accessed inside the MediaPipe callback
  const earThreshRef = useRef(0.22);
  const frameThreshRef = useRef(20);
  const closedFramesRef = useRef(0);
  const alertActiveRef = useRef(false);
  const alertCountRef = useRef(0);
  const sessionStartRef = useRef<number | null>(null);
  const fpsTimesRef = useRef<number[]>([]);
  const logIdCounter = useRef(0);

  const faceMeshRef = useRef<FaceMesh | null>(null);
  const mpCameraRef = useRef<Camera | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [state, setState] = useState<DetectorState>({
    ear: null,
    closedFrames: 0,
    alertCount: 0,
    alertActive: false,
    sessionStart: null,
    status: 'offline',
    fps: 0,
    logs: [{ id: 0, timestamp: new Date().toTimeString().slice(0, 8), message: 'System ready. Awaiting camera start.', level: '' }],
    isRunning: false,
  });

  const addLog = useCallback((message: string, level: '' | 'warn' | 'alert' = '') => {
    const ts = new Date().toTimeString().slice(0, 8);
    const id = ++logIdCounter.current;
    setState(prev => ({
      ...prev,
      logs: [{ id, timestamp: ts, message, level }, ...prev.logs].slice(0, 50),
    }));
  }, []);

  const drawEye = useCallback((
    ctx: CanvasRenderingContext2D,
    landmarks: NormalizedLandmark[],
    indices: number[],
    color: string,
    W: number,
    H: number
  ) => {
    ctx.beginPath();
    indices.forEach((id, i) => {
      const px = landmarks[id].x * W;
      const py = landmarks[id].y * H;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.fillStyle = color + '33';
    ctx.fill();
    ctx.stroke();
  }, []);

  const onResults = useCallback((results: FaceMeshResults) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // FPS tracking
    const now = performance.now();
    fpsTimesRef.current.push(now);
    fpsTimesRef.current = fpsTimesRef.current.filter(t => now - t < 1000);
    const fps = fpsTimesRef.current.length;

    // Resize canvas to match video
    canvas.width = video.videoWidth || canvas.offsetWidth;
    canvas.height = video.videoHeight || canvas.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      setState(prev => ({ ...prev, fps, status: 'no-face' }));
      return;
    }

    const lm = results.multiFaceLandmarks[0];
    const W = canvas.width;
    const H = canvas.height;

    // Compute EAR
    const ear = computeAverageEAR(lm);
    const eyesClosed = ear < earThreshRef.current;
    const eyeColor = eyesClosed ? '#E24B4A' : '#1D9E75';

    // Draw eye overlays
    drawEye(ctx, lm, LEFT_EYE_INDICES, eyeColor, W, H);
    drawEye(ctx, lm, RIGHT_EYE_INDICES, eyeColor, W, H);

    // Update closed frame counter
    let newClosedFrames = closedFramesRef.current;
    let newAlertActive = alertActiveRef.current;
    let newAlertCount = alertCountRef.current;
    let newStatus: DetectorState['status'] = 'awake';

    if (eyesClosed) {
      newClosedFrames++;
      if (newClosedFrames === 5) {
        addLog('Eyes closing detected', 'warn');
      }
      if (newClosedFrames >= frameThreshRef.current && !newAlertActive) {
        newAlertActive = true;
        newAlertCount++;
        beepAlert();
        addLog(`ALERT #${newAlertCount}: drowsiness detected (${newClosedFrames} frames closed)`, 'alert');
        newStatus = 'alert';
      }
      if (newAlertActive) {
        newStatus = 'alert';
      }
    } else {
      if (newClosedFrames > 0 && newClosedFrames < frameThreshRef.current) {
        if (newClosedFrames >= 5) {
          addLog(`Eyes reopened after ${newClosedFrames} frames`);
        }
      }
      newClosedFrames = 0;
      if (!newAlertActive) {
        newStatus = 'awake';
      } else {
        newStatus = 'alert';
      }
    }

    // Update refs
    closedFramesRef.current = newClosedFrames;
    alertActiveRef.current = newAlertActive;
    alertCountRef.current = newAlertCount;

    // Batch state update
    setState(prev => ({
      ...prev,
      ear,
      closedFrames: newClosedFrames,
      alertCount: newAlertCount,
      alertActive: newAlertActive,
      status: newStatus,
      fps,
    }));
  }, [addLog, drawEye]);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
      });

      const video = videoRef.current;
      if (!video) return;

      video.srcObject = stream;
      streamRef.current = stream;
      sessionStartRef.current = Date.now();
      closedFramesRef.current = 0;
      alertActiveRef.current = false;
      alertCountRef.current = 0;

      const fm = new FaceMesh({
        locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`,
      });
      fm.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      fm.onResults(onResults);
      faceMeshRef.current = fm;

      const cam = new Camera(video, {
        onFrame: async () => {
          if (faceMeshRef.current) {
            await faceMeshRef.current.send({ image: video });
          }
        },
        width: 640,
        height: 480,
      });
      cam.start();
      mpCameraRef.current = cam;

      setState(prev => ({
        ...prev,
        isRunning: true,
        status: 'active',
        sessionStart: sessionStartRef.current,
        alertCount: 0,
        closedFrames: 0,
        alertActive: false,
        ear: null,
      }));
      addLog('Camera started. Monitoring active.');
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      addLog('Camera error: ' + msg, 'alert');
      alert('Camera access error: ' + msg);
    }
  }, [onResults, addLog]);

  const stopCamera = useCallback(() => {
    if (mpCameraRef.current) {
      mpCameraRef.current.stop();
      mpCameraRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    faceMeshRef.current = null;
    sessionStartRef.current = null;
    closedFramesRef.current = 0;
    alertActiveRef.current = false;

    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    setState(prev => ({
      ...prev,
      isRunning: false,
      status: 'offline',
      sessionStart: null,
      alertActive: false,
      closedFrames: 0,
      ear: null,
      fps: 0,
    }));
    addLog('Camera stopped.');
  }, [addLog]);

  const dismissAlert = useCallback(() => {
    alertActiveRef.current = false;
    closedFramesRef.current = 0;
    setState(prev => ({
      ...prev,
      alertActive: false,
      closedFrames: 0,
    }));
  }, []);

  const setEarThreshold = useCallback((value: number) => {
    earThreshRef.current = value;
  }, []);

  const setFrameThreshold = useCallback((value: number) => {
    frameThreshRef.current = value;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (mpCameraRef.current) mpCameraRef.current.stop();
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    };
  }, []);

  return [
    state,
    { startCamera, stopCamera, dismissAlert, setEarThreshold, setFrameThreshold },
    { videoRef, canvasRef },
  ];
}
