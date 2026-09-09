/**
 * Test helper shared by the Composer and ChatView suites.
 *
 * The retrieval scope selects live inside a popover behind the Scope chip, so
 * a test that drives them has to open it first. Kept here rather than in
 * either suite so neither has to import the other's module.
 */

import { fireEvent, screen } from '@testing-library/react';

/** Open the scope popover if it is closed. Safe to call repeatedly. */
export function openScope(): void {
  if (screen.queryByLabelText('Channel scope')) return;
  fireEvent.click(screen.getByRole('button', { name: /^Retrieval scope:/ }));
}
