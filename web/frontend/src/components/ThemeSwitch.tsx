import { Monitor, Moon, Sun } from 'lucide-react'
import { useEffect, useState } from 'react'

type Theme = 'system' | 'light' | 'dark'
const KEY = 'gnome3d-theme'
const CHOICES: [Theme, string, typeof Sun][] = [
  ['system', 'System', Monitor],
  ['light', 'Light', Sun],
  ['dark', 'Dark', Moon],
]

function stored(): Theme {
  try {
    const t = localStorage.getItem(KEY)
    return t === 'system' || t === 'dark' ? t : 'light'
  } catch {
    return 'light'
  }
}

/** System, Light or Dark for the whole site. A browser without a choice gets Light. The choice
 * is kept in this browser and index.html applies it before the first paint. */
export function ThemeSwitch() {
  const [theme, setTheme] = useState<Theme>(stored)
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') delete root.dataset.theme
    else root.dataset.theme = theme
    try {
      localStorage.setItem(KEY, theme)
    } catch {
      // A browser that blocks storage keeps the choice for this page only.
    }
  }, [theme])
  return (
    <div className="theme-switch" role="radiogroup" aria-label="Theme">
      {CHOICES.map(([id, label, Icon]) => (
        <button key={id} type="button" role="radio" aria-checked={theme === id} aria-label={label} title={label} onClick={() => setTheme(id)}>
          <Icon size={14} aria-hidden />
        </button>
      ))}
    </div>
  )
}
