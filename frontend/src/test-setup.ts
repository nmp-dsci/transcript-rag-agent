import '@testing-library/jest-dom/vitest';

// Node 25 exposes its own partial `localStorage` global that shadows jsdom's,
// leaving an object with no Storage methods. Install a minimal in-memory
// implementation whenever the ambient one is unusable, so anything touching
// persisted state (theme choice, auto-judge preference) is testable.
function installMemoryStorage(): void {
  const store = new Map<string, string>();
  const storage: Storage = {
    get length() {
      return store.size;
    },
    key: (index: number) => [...store.keys()][index] ?? null,
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, String(value)),
    removeItem: (key: string) => void store.delete(key),
    clear: () => store.clear(),
  };
  Object.defineProperty(globalThis, 'localStorage', {
    value: storage,
    configurable: true,
    writable: true,
  });
}

if (typeof globalThis.localStorage?.clear !== 'function') {
  installMemoryStorage();
}
