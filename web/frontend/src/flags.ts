/** The parts of the site the server leaves out, from its `HIDE` setting. */

export type Part = 'results' | 'method' | 'trios'

/**
 * Whether the server hides a part of the site.
 * The server writes the hidden parts into the page as the `site-hide` meta tag, read on each call
 * so a test can set it. A page served by the development server has no tag and hides nothing.
 */
export function hidden(part: Part): boolean {
  const content = document.querySelector('meta[name="site-hide"]')?.getAttribute('content') ?? ''
  return content.split(',').includes(part)
}
