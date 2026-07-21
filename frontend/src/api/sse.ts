/**
 * Reader for the server-sent event streams behind /api/ask and /api/judge.
 *
 * `EventSource` cannot be used: both endpoints are POSTs with a JSON body, so
 * the response body is parsed here instead.
 */

export type SseHandlers = Record<string, (data: never) => void>;

/** Parse one `event:`/`data:` block into its event name and raw payload. */
function parseBlock(block: string): { event: string; data: string } {
  let event = 'message';
  let data = '';
  for (const line of block.split('\n')) {
    if (line.startsWith('event: ')) event = line.slice(7).trim();
    else if (line.startsWith('data: ')) data += line.slice(6);
  }
  return { event, data };
}

export async function readEvents(
  body: ReadableStream<Uint8Array>,
  handlers: Record<string, (data: any) => void>,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split('\n\n');
    // The trailing fragment is an incomplete block; keep it for the next chunk.
    buffer = blocks.pop() ?? '';
    for (const block of blocks) {
      const { event, data } = parseBlock(block);
      const handler = handlers[event];
      if (!handler || !data) continue;
      try {
        handler(JSON.parse(data));
      } catch (error) {
        console.error('Malformed SSE payload', event, error);
      }
    }
  }
}
