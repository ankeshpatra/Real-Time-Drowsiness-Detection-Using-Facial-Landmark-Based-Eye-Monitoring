/* Type declarations for the MediaPipe global scripts loaded via CDN */

interface NormalizedLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

interface FaceMeshResults {
  multiFaceLandmarks?: NormalizedLandmark[][];
  image: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement;
}

interface FaceMeshOptions {
  maxNumFaces?: number;
  refineLandmarks?: boolean;
  minDetectionConfidence?: number;
  minTrackingConfidence?: number;
}

declare class FaceMesh {
  constructor(config: { locateFile: (file: string) => string });
  setOptions(options: FaceMeshOptions): void;
  onResults(callback: (results: FaceMeshResults) => void): void;
  send(input: { image: HTMLVideoElement }): Promise<void>;
  close(): void;
}

interface CameraOptions {
  onFrame: () => Promise<void>;
  width?: number;
  height?: number;
}

declare class Camera {
  constructor(video: HTMLVideoElement, options: CameraOptions);
  start(): void;
  stop(): void;
}
