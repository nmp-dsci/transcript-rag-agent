import { describe, expect, it, vi } from 'vitest';

import { readEvents } from './sse';

function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

describe('readEvents', () => {
  it('dispatches each event to its handler', async () => {
    const answer = vi.fn();
    const done = vi.fn();
    await readEvents(
      streamOf(
        'event: answer\ndata: {"key":"rag_llm"}\n\n',
        'event: done\ndata: {"id":"q-1"}\n\n',
      ),
      { answer, done },
    );
    expect(answer).toHaveBeenCalledWith({ key: 'rag_llm' });
    expect(done).toHaveBeenCalledWith({ id: 'q-1' });
  });

  it('reassembles events split across network chunks', async () => {
    const done = vi.fn();
    await readEvents(streamOf('event: do', 'ne\ndata: {"id":', '"q-2"}\n\n'), { done });
    expect(done).toHaveBeenCalledWith({ id: 'q-2' });
  });

  it('joins multi-line data payloads', async () => {
    const done = vi.fn();
    await readEvents(streamOf('event: done\ndata: {"a":1,\ndata: "b":2}\n\n'), { done });
    expect(done).toHaveBeenCalledWith({ a: 1, b: 2 });
  });

  it('ignores events with no registered handler', async () => {
    const done = vi.fn();
    await readEvents(streamOf('event: noise\ndata: {}\n\nevent: done\ndata: {"id":"q"}\n\n'), {
      done,
    });
    expect(done).toHaveBeenCalledOnce();
  });

  it('survives a malformed payload and keeps reading', async () => {
    const done = vi.fn();
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    await readEvents(streamOf('event: done\ndata: {oops\n\nevent: done\ndata: {"id":"q"}\n\n'), {
      done,
    });
    expect(done).toHaveBeenCalledWith({ id: 'q' });
    spy.mockRestore();
  });
});
