import { useEffect, useRef } from 'react';

import { type SpeechToText, joinSpeech, useSpeechToText } from './useSpeechToText';

export interface Dictation {
  speech: SpeechToText;
  /** True from the moment the mic is pressed (connecting counts) until it stops. */
  listening: boolean;
  /** What the box should show: the typed value, plus the live transcript while listening. */
  display: string;
  toggle: () => void;
}

/**
 * Speech landing in a text box, the Chat contract: while listening the box is
 * read-only and shows `value + committed + interim` as an overlay; when the
 * session ends everything spoken is folded into `value` exactly once, and
 * `onSpoken` (if given) sees the folded text — the comment box uses it to add
 * the comment straight away.
 */
export function useDictation(
  value: string,
  setValue: (next: string | ((current: string) => string)) => void,
  onSpoken?: (folded: string) => void,
): Dictation {
  const speech = useSpeechToText();
  const listening = speech.status !== 'idle';
  const wasListening = useRef(false);
  const latest = useRef({ value, onSpoken });
  latest.current = { value, onSpoken };

  useEffect(() => {
    if (wasListening.current && !listening) {
      const spoken = joinSpeech(speech.transcript.committed, speech.transcript.interim);
      if (spoken) {
        const folded = joinSpeech(latest.current.value, spoken);
        setValue(folded);
        latest.current.onSpoken?.(folded);
      }
      speech.reset();
    }
    wasListening.current = listening;
  }, [listening, speech, setValue]);

  const display = listening ? joinSpeech(value, speech.transcript.committed, speech.transcript.interim) : value;
  return {
    speech,
    listening,
    display,
    toggle: () => (listening ? speech.stop() : speech.start()),
  };
}
