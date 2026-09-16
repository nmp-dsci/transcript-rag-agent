import { act, renderHook } from '@testing-library/react';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';

import type { SpeechToText, TranscriptState } from './useSpeechToText';
import { useDictation } from './useDictation';

/** A hand-driven stand-in for the real hook: tests move it through the
 *  states the relay would, without a microphone or a socket. */
const fake: { current: SpeechToText } = {
  current: {
    supported: true,
    status: 'idle',
    transcript: { committed: '', interim: '' },
    error: null,
    start: vi.fn(),
    stop: vi.fn(),
    reset: vi.fn(),
  },
};
vi.mock('./useSpeechToText', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./useSpeechToText')>();
  return { ...actual, useSpeechToText: () => fake.current };
});

function set(status: SpeechToText['status'], transcript: TranscriptState) {
  fake.current = { ...fake.current, status, transcript };
}

describe('useDictation', () => {
  it('overlays the live transcript while listening and folds it in once on stop', () => {
    const onSpoken = vi.fn();
    const { result, rerender } = renderHook(() => {
      const [value, setValue] = useState('typed');
      return { value, ...useDictation(value, setValue, onSpoken) };
    });
    expect(result.current.display).toBe('typed');
    expect(result.current.listening).toBe(false);

    set('listening', { committed: 'hello', interim: 'wor' });
    rerender();
    expect(result.current.listening).toBe(true);
    expect(result.current.display).toBe('typed hello wor');
    expect(result.current.value).toBe('typed');

    set('idle', { committed: 'hello', interim: 'world' });
    act(() => rerender());
    expect(result.current.value).toBe('typed hello world');
    expect(onSpoken).toHaveBeenCalledWith('typed hello world');
    expect(fake.current.reset).toHaveBeenCalled();
    // A second idle render must not fold again.
    act(() => rerender());
    expect(onSpoken).toHaveBeenCalledTimes(1);
  });

  it('does nothing on stop when nothing was said', () => {
    const onSpoken = vi.fn();
    const { result, rerender } = renderHook(() => {
      const [value, setValue] = useState('');
      return { value, ...useDictation(value, setValue, onSpoken) };
    });
    set('connecting', { committed: '', interim: '' });
    rerender();
    set('idle', { committed: '', interim: '' });
    act(() => rerender());
    expect(result.current.value).toBe('');
    expect(onSpoken).not.toHaveBeenCalled();
  });

  it('toggle starts when idle and stops when listening', () => {
    set('idle', { committed: '', interim: '' });
    const { result, rerender } = renderHook(() => {
      const [value, setValue] = useState('');
      return useDictation(value, setValue);
    });
    result.current.toggle();
    expect(fake.current.start).toHaveBeenCalled();
    set('listening', { committed: '', interim: '' });
    rerender();
    result.current.toggle();
    expect(fake.current.stop).toHaveBeenCalled();
  });
});
