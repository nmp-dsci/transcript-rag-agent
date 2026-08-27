import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { floatTo16BitPcm } from './pcm';
import { joinSpeech, reduceTranscript, useSpeechToText } from './useSpeechToText';

vi.mock('./pcm', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./pcm')>();
  return { ...actual, pcmWorkletUrl: () => 'blob:mock-worklet' };
});

describe('reduceTranscript', () => {
  const empty = { committed: '', interim: '' };

  it('replaces the interim tail on every interim message', () => {
    let state = reduceTranscript(empty, { text: 'what', is_final: false });
    state = reduceTranscript(state, { text: 'what does the', is_final: false });
    expect(state).toEqual({ committed: '', interim: 'what does the' });
  });

  it('commits finals and clears the interim', () => {
    let state = reduceTranscript(empty, { text: 'what does the', is_final: false });
    state = reduceTranscript(state, { text: 'What does the channel say?', is_final: true });
    expect(state).toEqual({ committed: 'What does the channel say?', interim: '' });
  });

  it('accumulates successive finals across pauses', () => {
    let state = reduceTranscript(empty, { text: 'First sentence.', is_final: true });
    state = reduceTranscript(state, { text: 'and then', is_final: false });
    state = reduceTranscript(state, { text: 'Second sentence.', is_final: true });
    expect(state).toEqual({ committed: 'First sentence. Second sentence.', interim: '' });
  });

  it('lets an empty final clear the tail without touching committed text', () => {
    let state = reduceTranscript(empty, { text: 'Kept.', is_final: true });
    state = reduceTranscript(state, { text: 'dangling interim', is_final: false });
    state = reduceTranscript(state, { text: '', is_final: true });
    expect(state).toEqual({ committed: 'Kept.', interim: '' });
  });

  it('ignores empty interims so silence never wipes visible words', () => {
    let state = reduceTranscript(empty, { text: 'still talking', is_final: false });
    state = reduceTranscript(state, { text: '', is_final: false });
    expect(state.interim).toBe('still talking');
  });
});

describe('joinSpeech', () => {
  it('joins non-empty fragments with single spaces', () => {
    expect(joinSpeech('typed text', ' spoken ', '', 'tail')).toBe('typed text spoken tail');
  });

  it('returns empty for all-blank input', () => {
    expect(joinSpeech('', '  ')).toBe('');
  });
});

describe('floatTo16BitPcm', () => {
  it('scales the full float range onto int16 and clamps overshoot', () => {
    const out = floatTo16BitPcm(new Float32Array([0, 1, -1, 2, -2, 0.5]));
    expect(Array.from(out)).toEqual([0, 32767, -32768, 32767, -32768, 16383]);
  });
});

/** Minimal fakes for the browser capture/socket surface, so start()'s async
 * setup can be paused mid-flight to exercise the connect/cancel races. */
class FakeTrack {
  stopped = false;
  stop() {
    this.stopped = true;
  }
}
class FakeStream {
  private tracks = [new FakeTrack(), new FakeTrack()];
  getTracks() {
    return this.tracks;
  }
}
class FakeAudioContext {
  sampleRate = 48000;
  audioWorklet = { addModule: vi.fn(() => Promise.resolve()) };
  closed = false;
  createMediaStreamSource() {
    return { connect: vi.fn() };
  }
  close() {
    this.closed = true;
    return Promise.resolve();
  }
}
class FakeAudioWorkletNode {
  port: { onmessage: ((event: MessageEvent) => void) | null } = { onmessage: null };
}
class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  readyState = 0;
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;
  binaryType = '';
  constructor(public url: string) {
    FakeWebSocket.instances.push(this);
  }
  send() {
    // captured for inspection via instances if ever needed
  }
  close() {
    // no-op: tests assert on state, not on close being called
  }
}

const flush = async () => {
  for (let i = 0; i < 4; i += 1) await Promise.resolve();
};

describe('useSpeechToText connect races', () => {
  const getUserMedia = vi.fn();

  beforeEach(() => {
    FakeWebSocket.instances = [];
    getUserMedia.mockReset();
    Object.defineProperty(navigator, 'mediaDevices', {
      value: { getUserMedia },
      configurable: true,
    });
    vi.stubGlobal('AudioContext', FakeAudioContext);
    vi.stubGlobal('AudioWorkletNode', FakeAudioWorkletNode);
    vi.stubGlobal('WebSocket', FakeWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('tears down the mic and never opens a socket when stop() lands before getUserMedia resolves', async () => {
    let resolveGum!: (stream: FakeStream) => void;
    getUserMedia.mockReturnValue(new Promise((resolve) => (resolveGum = resolve)));
    const { result } = renderHook(() => useSpeechToText());

    act(() => result.current.start());
    expect(result.current.status).toBe('connecting');

    act(() => result.current.stop());

    const stream = new FakeStream();
    await act(async () => {
      resolveGum(stream);
      await flush();
    });

    expect(result.current.status).toBe('idle');
    expect(stream.getTracks().every((track) => track.stopped)).toBe(true);
    expect(FakeWebSocket.instances).toHaveLength(0);
  });

  it('ignores a stale ws.onopen that fires after stop() already began tearing down', async () => {
    getUserMedia.mockResolvedValue(new FakeStream());
    const { result } = renderHook(() => useSpeechToText());

    await act(async () => {
      result.current.start();
      await flush();
    });

    const ws = FakeWebSocket.instances[0];
    expect(ws).toBeDefined();
    expect(result.current.status).toBe('connecting');

    act(() => result.current.stop());
    expect(result.current.status).not.toBe('listening');

    act(() => ws.onopen?.());
    expect(result.current.status).not.toBe('listening');
  });
});
