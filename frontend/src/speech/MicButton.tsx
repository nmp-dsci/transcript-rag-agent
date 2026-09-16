/** The one voice-input button. Idle breathes green (the invitation), listening
 *  flips to the red stop pulse — the `.micbtn` rules in theme.css. Chat, the
 *  guide composer and the comment box all render this, so the two states are
 *  unmistakable everywhere. */
export function MicButton({
  listening,
  supported,
  disabled = false,
  size = 'md',
  onToggle,
  idleTitle = 'Speak',
}: {
  listening: boolean;
  supported: boolean;
  disabled?: boolean;
  size?: 'md' | 'sm';
  onToggle: () => void;
  idleTitle?: string;
}) {
  const px = size === 'sm' ? 12 : 16;
  return (
    <button
      type="button"
      className={`micbtn${listening ? ' rec' : ''}${size === 'sm' ? ' sm' : ''}`}
      onClick={onToggle}
      disabled={disabled || !supported}
      aria-pressed={listening}
      aria-label={listening ? 'Stop voice input' : 'Start voice input'}
      title={
        !supported
          ? 'Voice input needs a browser with microphone and AudioWorklet support'
          : listening
            ? 'Stop voice input'
            : idleTitle
      }
    >
      {listening ? (
        <svg viewBox="0 0 24 24" width={px - 2} height={px - 2} fill="currentColor" aria-hidden="true">
          <rect x="6" y="6" width="12" height="12" rx="2" />
        </svg>
      ) : (
        <svg
          viewBox="0 0 24 24"
          width={px}
          height={px}
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <rect x="9" y="2" width="6" height="12" rx="3" />
          <path d="M5 10a7 7 0 0 0 14 0" />
          <line x1="12" y1="19" x2="12" y2="22" />
        </svg>
      )}
    </button>
  );
}
