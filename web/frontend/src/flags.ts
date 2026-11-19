/** The parts of the site the server turns off, the features of its `config.json`. */

export type Part = 'results' | 'method' | 'trios'

/**
 * Whether a part of the site is turned off. The server writes the features that are off into the
 * page as the `site-hide` meta tag. A page of the development server has no tag and hides nothing.
 */
export function hidden(part: Part): boolean {
  const content = document.querySelector('meta[name="site-hide"]')?.getAttribute('content') ?? ''
  return content.split(',').includes(part)
}
