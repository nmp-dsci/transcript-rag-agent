import { describe, expect, it } from 'vitest';

import { floatTo16BitPcm } from './pcm';
import { joinSpeech, reduceTranscript } from './useSpeechToText';

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
