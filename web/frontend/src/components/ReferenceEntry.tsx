import type { Reference } from '../content/references'
import { referenceLink } from '../content/references'

/** One reference: title, authors, venue and its DOI or URL. */
export function ReferenceEntry({ r }: { r: Reference }) {
  const link = referenceLink(r)
  return (
    <article className="reference">
      <p className="ref-title">{r.title}</p>
      <p className="ref-authors">{r.authors}</p>
      <p className="ref-venue">
        {r.venue} ({r.year})
        {link && (
          <a href={link}>{r.doi ? `doi:${r.doi}` : link.replace(/^https:\/\//, '')}</a>
        )}
      </p>
    </article>
  )
}
