/** The path the site is served under, taken from the build's `BASE_PATH`. */

/** A base URL of Vite as a prefix without a trailing slash. The root gives the empty string. */
export const prefixOf = (baseUrl: string) => baseUrl.replace(/\/+$/, '')

/** The prefix of this build, `/new` for a site at `/new/` and empty for a site at the root. */
export const BASE = prefixOf(import.meta.env.BASE_URL)

/**
 * A path of the server as the browser requests it.
 * The router adds the prefix to its own links and every other request and link goes through here.
 */
export const served = (path: string, base = BASE) => base + path
