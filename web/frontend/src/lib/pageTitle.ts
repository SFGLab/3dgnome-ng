import meta from '../content/pages.json'

type Page = { title: string }

/** Returns the title of the browser tab for a path of the app. */
export function pageTitle(path: string): string {
  const key = path.replace(/^\/+|\/+$/g, '')
  if (key === '') return meta.pages[''].title
  const page: Page | undefined = (meta.pages as Record<string, Page>)[key] ?? (/^(jobs|p)\/[^/]+$/.test(key) ? meta.job : undefined)
  return `${(page ?? meta.missing).title} | ${meta.site}`
}
