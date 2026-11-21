import { X } from 'lucide-react'
import { type ReactNode, useEffect } from 'react'
import { createPortal } from 'react-dom'
import '../styles/dialog.css'

/**
 * The frame of a dialog. It has the backdrop, the head with the title and the line under it and the
 * Escape key, which closes it. The caller gives the body and the foot.
 * @param id the id of the title, which labels the dialog
 * @param name the class of this dialog beside `dialog`
 * @param closeButton a close button in the head
 * @param inBody the dialog is added to the body of the page, above a page that clips its content
 */
export function Dialog({
  id,
  name,
  title,
  sub,
  onClose,
  closeButton = true,
  inBody = false,
  children,
}: {
  id: string
  name: string
  title: string
  sub: string
  onClose: () => void
  closeButton?: boolean
  inBody?: boolean
  children: ReactNode
}) {
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])
  const frame = (
    <div className="dialog-backdrop">
      <div className={`dialog ${name}`} role="dialog" aria-modal="true" aria-labelledby={id}>
        <header className="dialog-head">
          <div>
            <h2 id={id}>{title}</h2>
            <p className="dialog-sub">{sub}</p>
          </div>
          {closeButton && (
            <button type="button" className="icon-button" aria-label="Close" onClick={onClose}>
              <X size={18} aria-hidden />
            </button>
          )}
        </header>
        {children}
      </div>
    </div>
  )
  return inBody ? createPortal(frame, document.body) : frame
}
