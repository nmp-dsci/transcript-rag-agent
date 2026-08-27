/**
 * Raw-PCM capture for streaming voice-to-text.
 *
 * The mic path deliberately avoids MediaRecorder: Safari records AAC/MP4 while
 * Chrome records Opus/WebM, so containerized capture forks the pipeline per
 * browser. An AudioWorklet hands back the same Float32 samples everywhere;
 * converted to 16-bit PCM (linear16) they match what /ws/stt tells Deepgram
 * to expect, at whatever sample rate the AudioContext actually runs.
 */

/** Clamp Float32 samples in [-1, 1] to 16-bit little-endian PCM. */
export function floatTo16BitPcm(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i] ?? 0));
    out[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return out;
}

export const PCM_WORKLET_NAME = 'stt-pcm-capture';

// The processor ships as a Blob URL so the worklet needs no separate bundled
// asset: it forwards each 128-frame render quantum of channel 0 to the main
// thread, where batching and the WebSocket live.
const WORKLET_SOURCE = `
class SttPcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor('${PCM_WORKLET_NAME}', SttPcmCapture);
`;

export function pcmWorkletUrl(): string {
  return URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'application/javascript' }));
}
