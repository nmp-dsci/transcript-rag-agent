import { useCallback, useEffect, useRef, useState } from 'react';

import { PCM_WORKLET_NAME, floatTo16BitPcm, pcmWorkletUrl } from './pcm';

/**
 * Streaming voice-to-text over the app's own /ws/stt relay.
 *
 * Mic → AudioWorklet (raw PCM) → WebSocket → server relay → Deepgram, with
 * transcript JSON coming back as it is spoken. The hook owns the whole
 * session (stream, AudioContext, socket, timers) and exposes only transcript
 * state; the composer decides how spoken text lands in the question box.
 */

export type SttStatus = 'idle' | 'connecting' | 'listening';

export interface TranscriptState {
  /** Segments Deepgram has finalized — they no longer change. */
  committed: string;
  /** The in-flight phrase, replaced wholesale by every interim message. */
  interim: string;
}

const EMPTY_TRANSCRIPT: TranscriptState = { committed: '', interim: '' };

/** Join spoken fragments with single spaces, dropping empty parts. */
export function joinSpeech(...parts: string[]): string {
  return parts
    .map((part) => part.trim())
    .filter(Boolean)
    .join(' ');
}

/**
 * Fold one server transcript message into the running state.
 *
 * Finals append and clear the interim tail (an empty final still clears it —
 * that is how an endpointed pause reads); interims replace the tail only when
 * non-empty, so a silence frame never wipes words already on screen.
 */
export function reduceTranscript(
  state: TranscriptState,
  message: { text?: string; is_final?: boolean },
): TranscriptState {
  const text = (message.text ?? '').trim();
  if (message.is_final) {
    return text
      ? { committed: joinSpeech(state.committed, text), interim: '' }
      : { ...state, interim: '' };
  }
  return text ? { ...state, interim: text } : state;
}

export interface SpeechToText {
  supported: boolean;
  status: SttStatus;
  transcript: TranscriptState;
  error: string | null;
  start: () => void;
  stop: () => void;
  reset: () => void;
}

interface Session {
  ws: WebSocket;
  ctx: AudioContext;
  stream: MediaStream;
  node: AudioWorkletNode;
  capTimer: number;
  flushTimer: number | null;
}

/** Client-side stop before the server's 120s hard cap can ever fire. */
const SESSION_CAP_MS = 110_000;
/** Audio is batched to ~this many seconds per WebSocket frame. */
const BATCH_SECONDS = 0.25;
/** After stop, the socket idles this long so trailing finals still land. */
const STOP_FLUSH_MS = 1_500;

export function useSpeechToText(): SpeechToText {
  const [status, setStatus] = useState<SttStatus>('idle');
  const [transcript, setTranscript] = useState<TranscriptState>(EMPTY_TRANSCRIPT);
  const [error, setError] = useState<string | null>(null);
  const session = useRef<Session | null>(null);
  const connecting = useRef(false);
  const cancelRequested = useRef(false);

  const supported =
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof AudioContext !== 'undefined' &&
    typeof AudioWorkletNode !== 'undefined';

  const teardown = useCallback(() => {
    const current = session.current;
    if (!current) return;
    session.current = null;
    window.clearTimeout(current.capTimer);
    if (current.flushTimer !== null) window.clearTimeout(current.flushTimer);
    current.node.port.onmessage = null;
    for (const track of current.stream.getTracks()) track.stop();
    void current.ctx.close().catch(() => undefined);
    try {
      current.ws.close();
    } catch {
      // already closed
    }
    setStatus('idle');
  }, []);

  useEffect(() => teardown, [teardown]);

  const stop = useCallback(() => {
    const current = session.current;
    if (!current) {
      // Still inside the async setup in start() — flag it so start() tears
      // down whatever it has acquired once it reaches a checkpoint, instead
      // of finishing setup and beginning a session the user already cancelled.
      if (connecting.current) cancelRequested.current = true;
      return;
    }
    if (current.flushTimer !== null) return;
    // Cut the mic at once; hold the socket open briefly so the words still in
    // flight come back as finals before the fold into the question box.
    for (const track of current.stream.getTracks()) track.stop();
    try {
      current.ws.send(JSON.stringify({ type: 'stop' }));
    } catch {
      // socket already gone — teardown below still runs
    }
    current.flushTimer = window.setTimeout(teardown, STOP_FLUSH_MS);
  }, [teardown]);

  const start = useCallback(() => {
    if (session.current || connecting.current || !supported) return;
    connecting.current = true;
    cancelRequested.current = false;
    setError(null);
    setTranscript(EMPTY_TRANSCRIPT);
    setStatus('connecting');
    void (async () => {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true },
        });
      } catch {
        connecting.current = false;
        setError('microphone unavailable — check browser permissions');
        setStatus('idle');
        return;
      }
      if (cancelRequested.current) {
        connecting.current = false;
        cancelRequested.current = false;
        for (const track of stream.getTracks()) track.stop();
        setStatus('idle');
        return;
      }
      try {
        const ctx = new AudioContext();
        await ctx.audioWorklet.addModule(pcmWorkletUrl());
        if (cancelRequested.current) {
          connecting.current = false;
          cancelRequested.current = false;
          for (const track of stream.getTracks()) track.stop();
          void ctx.close().catch(() => undefined);
          setStatus('idle');
          return;
        }
        const source = ctx.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(ctx, PCM_WORKLET_NAME);
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(
          `${protocol}://${window.location.host}/ws/stt?sample_rate=${Math.round(ctx.sampleRate)}`,
        );
        ws.binaryType = 'arraybuffer';

        let pending: Float32Array[] = [];
        let pendingLength = 0;
        node.port.onmessage = (event: MessageEvent<Float32Array>) => {
          if (ws.readyState !== WebSocket.OPEN) return;
          pending.push(event.data);
          pendingLength += event.data.length;
          if (pendingLength < ctx.sampleRate * BATCH_SECONDS) return;
          const merged = new Float32Array(pendingLength);
          let offset = 0;
          for (const block of pending) {
            merged.set(block, offset);
            offset += block.length;
          }
          pending = [];
          pendingLength = 0;
          ws.send(floatTo16BitPcm(merged).buffer);
        };
        source.connect(node);

        ws.onopen = () => setStatus('listening');
        ws.onmessage = (event: MessageEvent<string>) => {
          try {
            const message = JSON.parse(event.data);
            if (message.type === 'transcript') {
              setTranscript((prev) => reduceTranscript(prev, message));
            } else if (message.type === 'error') {
              setError(String(message.message || 'voice input failed'));
            }
          } catch {
            // not JSON — ignore
          }
        };
        ws.onclose = (event) => {
          if (!session.current) return;
          if (event.code === 1008) setError(event.reason || 'voice input unavailable');
          teardown();
        };

        const capTimer = window.setTimeout(stop, SESSION_CAP_MS);
        connecting.current = false;
        session.current = { ws, ctx, stream, node, capTimer, flushTimer: null };
        if (cancelRequested.current) {
          cancelRequested.current = false;
          stop();
        }
      } catch {
        connecting.current = false;
        for (const track of stream.getTracks()) track.stop();
        setError('voice capture failed to start in this browser');
        setStatus('idle');
      }
    })();
  }, [stop, supported, teardown]);

  const reset = useCallback(() => setTranscript(EMPTY_TRANSCRIPT), []);

  return { supported, status, transcript, error, start, stop, reset };
}
