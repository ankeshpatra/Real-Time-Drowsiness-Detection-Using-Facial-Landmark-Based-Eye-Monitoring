/**
 * Audio alert utility — plays a short square-wave beep to wake the driver.
 */

let audioCtx: AudioContext | null = null;

export function beepAlert(): void {
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext)();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.type = 'square';
    osc.frequency.value = 660;
    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.5);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.5);
  } catch {
    // Audio may be blocked by browser autoplay policy — fail silently
  }
}
