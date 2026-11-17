import { useState } from 'react'
import { Link } from 'react-router-dom'
import { ROUTES } from '../routes'

const KEY = 'gnome3d.privacySeen'

function seen(): boolean {
  try {
    return localStorage.getItem(KEY) === '1'
  } catch {
    // A browser without storage cannot remember the answer, and the notice would return on
    // every page.
    return true
  }
}

/** The privacy notice of a first visit, shown until it is closed once in this browser. */
export function PrivacyNotice() {
  const [open, setOpen] = useState(() => !seen())
  if (!open) return null
  const close = () => {
    try {
      localStorage.setItem(KEY, '1')
    } catch {
      // Closed for this page only.
    }
    setOpen(false)
  }
  return (
    <aside className="privacy-notice" aria-label="Privacy">
      <p>This site sets no cookies and does no tracking. It keeps your job list in your browser, and files that you upload are private and deleted after 14 days.</p>
      <div className="privacy-notice-actions">
        <Link to={ROUTES.privacy} onClick={close}>
          Privacy notice
        </Link>
        <button type="button" onClick={close}>
          OK
        </button>
      </div>
    </aside>
  )
}
