/**
 * Eye Aspect Ratio (EAR) calculation utilities.
 *
 * MediaPipe FaceMesh landmark indices for each eye (6-point model):
 *   [outer-corner, upper-1, upper-2, inner-corner, lower-2, lower-1]
 *
 * EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
 */

export const LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380];
export const RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144];

interface Point {
  x: number;
  y: number;
}

function dist(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Compute the Eye Aspect Ratio for a set of 6 landmark points.
 * @param landmarks - Full array of face landmarks from MediaPipe
 * @param indices - The 6 landmark indices for one eye
 * @returns The EAR value (higher = more open)
 */
export function computeEAR(
  landmarks: NormalizedLandmark[],
  indices: number[]
): number {
  const p = (i: number) => landmarks[indices[i]];
  const vert1 = dist(p(1), p(5));
  const vert2 = dist(p(2), p(4));
  const horiz = dist(p(0), p(3));
  return (vert1 + vert2) / (2 * horiz);
}

/**
 * Compute the average EAR of both eyes.
 */
export function computeAverageEAR(landmarks: NormalizedLandmark[]): number {
  const leftEAR = computeEAR(landmarks, LEFT_EYE_INDICES);
  const rightEAR = computeEAR(landmarks, RIGHT_EYE_INDICES);
  return (leftEAR + rightEAR) / 2;
}
