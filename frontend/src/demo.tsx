/** Demo mode, as the server reports it.
 *
 * The flag is server-enforced (every mutating route 403s); this context only
 * mirrors it so views can hide the controls that would hit those routes
 * instead of rendering buttons that fail. Read it with `useDemo()` — the
 * default is full mode, so nothing changes anywhere until /api/health says
 * `mode: "demo"`.
 */

import { createContext, useContext } from 'react';

export const DemoContext = createContext(false);

export function useDemo(): boolean {
  return useContext(DemoContext);
}
