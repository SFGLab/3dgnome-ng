import { Fragment } from 'react'

/** Text with citations in square brackets, each number a link to `#ref-<n>`. */
export function CitedText({ children }: { children: string }) {
  const parts = children.split(/(\[\d+(?:, \d+)*\])/)
  return (
    <>
      {parts.map((part, i) => {
        const m = part.match(/^\[(.*)\]$/)
        if (!m) return <Fragment key={i}>{part}</Fragment>
        const ns = m[1].split(', ')
        return (
          <Fragment key={i}>
            [
            {ns.map((n, j) => (
              <Fragment key={n}>
                {j > 0 && ', '}
                <a className="cite-link" href={`#ref-${n}`}>
                  {n}
                </a>
              </Fragment>
            ))}
            ]
          </Fragment>
        )
      })}
    </>
  )
}
