import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';

/** What the bridge script inside a guide page reports and accepts.
 *  Mirrors guides/guide-bridge.js — change both or neither. */
export interface GuideSection {
  id: string;
  label: string;
}

export interface GuideSelection {
  quote: string;
  anchor: string | null;
  sectionId: string | null;
  rect: { top: number; left: number; width: number; height: number } | null;
}

export interface GuideReaderHandle {
  scrollTo: (id: string) => void;
  highlight: (id: string | null) => void;
}

interface Props {
  url: string;
  title: string;
  onReady?: (sections: GuideSection[]) => void;
  onSelection?: (selection: GuideSelection) => void;
}

/** The workbench's current theme, as index.html stamps it before first paint. */
function currentTheme(): 'dark' | 'light' {
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

/**
 * A guide page in a sandboxed iframe.
 *
 * The page is agent-written, so it gets scripts (its own copy buttons and
 * reveals need them) but not our origin: `allow-same-origin` is deliberately
 * absent, which makes every message from it untrusted data the parent
 * validates by source window and shape. The theme is pushed in on load and
 * whenever the workbench toggles, so the document follows the app rather
 * than the OS.
 */
export const GuideReader = forwardRef<GuideReaderHandle, Props>(function GuideReader(
  { url, title, onReady, onSelection },
  ref,
) {
  const frame = useRef<HTMLIFrameElement | null>(null);
  const [loaded, setLoaded] = useState(false);

  const post = (message: Record<string, unknown>) => {
    frame.current?.contentWindow?.postMessage(message, '*');
  };

  useImperativeHandle(ref, () => ({
    scrollTo: (id: string) => post({ type: 'guide:scroll', id }),
    highlight: (id: string | null) => post({ type: 'guide:highlight', id }),
  }));

  // Follow the workbench theme: push on load, then on every toggle.
  useEffect(() => {
    if (!loaded) return;
    post({ type: 'guide:theme', theme: currentTheme() });
    const observer = new MutationObserver(() => {
      post({ type: 'guide:theme', theme: currentTheme() });
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, [loaded, url]);

  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      if (!frame.current || event.source !== frame.current.contentWindow) return;
      const data = event.data;
      if (!data || typeof data !== 'object' || typeof data.type !== 'string') return;
      if (data.type === 'guide:ready') {
        const sections = Array.isArray(data.sections)
          ? data.sections
              .filter((s: unknown) => s && typeof s === 'object')
              .map((s: { id?: unknown; label?: unknown }) => ({
                id: String(s.id ?? ''),
                label: String(s.label ?? s.id ?? ''),
              }))
              .filter((s: GuideSection) => s.id)
          : [];
        onReady?.(sections);
        // The page may have stamped its own theme from the OS before we
        // could push ours — push again now that it is listening.
        post({ type: 'guide:theme', theme: currentTheme() });
      }
      if (data.type === 'guide:selection') {
        onSelection?.({
          quote: typeof data.quote === 'string' ? data.quote : '',
          anchor: typeof data.anchor === 'string' ? data.anchor : null,
          sectionId: typeof data.sectionId === 'string' ? data.sectionId : null,
          rect: data.rect && typeof data.rect === 'object' ? data.rect : null,
        });
      }
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [onReady, onSelection]);

  return (
    <div className="guide-frame">
      <iframe
        ref={frame}
        title={title}
        src={url}
        sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox"
        onLoad={() => setLoaded(true)}
      />
    </div>
  );
});
