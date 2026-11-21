/** Class names for a component whose stylesheets are CSS modules. */

/**
 * Gives a function that turns class names into the scoped names of the sheets.
 *
 * A name a sheet scopes gets its scoped name and keeps its plain name beside it. The rules of the
 * sheet apply by the scoped name. The plain name stays for the shared sheets of styles/, for the
 * rules of another sheet that refer to this component, for a selector in code and for a test.
 * A name no sheet scopes is left as it is.
 * @param sheets the CSS modules the component uses
 */
export function scoped(...sheets: Record<string, string>[]) {
  function names(given: string): string
  function names(given: string | false | null | undefined): string | undefined
  function names(given: string | false | null | undefined): string | undefined {
    if (!given) return given === '' ? '' : undefined
    const out = new Set<string>()
    for (const name of given.split(/\s+/)) {
      if (!name) continue
      for (const sheet of sheets) if (sheet[name]) out.add(sheet[name])
      out.add(name)
    }
    return [...out].join(' ')
  }
  return names
}
