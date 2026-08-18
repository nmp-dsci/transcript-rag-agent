/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** PostHog project key, baked in at build time. Blank = analytics off. */
  readonly VITE_POSTHOG_KEY?: string;
  /** PostHog ingestion host; defaults to the US cloud when unset. */
  readonly VITE_POSTHOG_HOST?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
