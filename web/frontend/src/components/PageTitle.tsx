import { useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { pageTitle } from '../lib/pageTitle'

/** Sets the title of the browser tab on every change of the page. The server gives the same title
 * with the first page. */
export function PageTitle() {
  const { pathname } = useLocation()
  useEffect(() => {
    document.title = pageTitle(pathname)
  }, [pathname])
  return null
}
