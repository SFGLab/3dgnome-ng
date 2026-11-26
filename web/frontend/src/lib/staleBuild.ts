const KEY = 'gnome3d-reloaded-at'
/** A second failure within this time is shown and not answered with another reload. */
const WAIT_MS = 30_000

/**
 * Loads the page anew when a script of the build is missing. A tab opened before a deployment asks
 * for the scripts of the build it loaded. The server has those of the new build alone.
 * @param win the window of the app
 * @returns the function that removes the listener
 */
export function reloadOnStaleBuild(win: Window): () => void {
  const onError = (event: Event) => {
    let last = 0
    try {
      last = Number(win.sessionStorage.getItem(KEY) ?? 0)
      if (Date.now() - last < WAIT_MS) return
      win.sessionStorage.setItem(KEY, String(Date.now()))
    } catch {
      // A browser without storage for the tab gets no reload, which could repeat without end.
      return
    }
    event.preventDefault()
    win.location.reload()
  }
  win.addEventListener('vite:preloadError', onError)
  return () => win.removeEventListener('vite:preloadError', onError)
}
