import { saveBlob } from '../lib/download'
import * as Slider from '@radix-ui/react-slider'
import { ChevronDown, ChevronRight, Download, Info, RotateCcw, Search, SlidersHorizontal, Upload } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, type Setting, type SettingGroup } from '../api'
import type { FormState } from './form'
import { changedSettings, settingName, toIni } from './settings'
import { scoped } from '../lib/css'
import sheet from './model.module.css'

const cn = scoped(sheet)

const GROUP_NOTES: Record<string, string> = {
  'Distance law': 'One law sets every target distance. The target for a pair without a loop is (s / 1 kb) to the power nu, in bead units.',
}

function clamp(x: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, x))
}

/** A number setting: a box for the value and a slider over its range. */
function NumberSetting({ s, value, onChange }: { s: Setting; value: string; onChange: (v: string) => void }) {
  const lo = s.lo ?? 0
  const hi = s.hi ?? 1
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  const commit = (text: string) => {
    const x = Number(text)
    if (text.trim() !== '' && Number.isFinite(x)) onChange(String(clamp(x, lo, hi)))
    else setDraft(value)
  }
  return (
    <>
      <input
        className={cn('value-box')}
        type="number"
        aria-label={s.key}
        min={lo}
        max={hi}
        step={s.step ?? 'any'}
        value={draft}
        onChange={(e) => {
          setDraft(e.target.value)
          const x = Number(e.target.value)
          if (e.target.value.trim() !== '' && x >= lo && x <= hi) onChange(e.target.value)
        }}
        onBlur={(e) => commit(e.target.value)}
      />
      <Slider.Root
        className={cn('slider')}
        min={lo}
        max={hi}
        step={s.step ?? (hi - lo) / 100}
        value={[clamp(Number(value) || 0, lo, hi)]}
        onValueChange={([x]) => onChange(String(Number(x.toPrecision(6))))}
      >
        <Slider.Track className={cn('slider-track')}>
          <Slider.Range className={cn('slider-range')} />
        </Slider.Track>
        <Slider.Thumb className={cn('slider-thumb')} aria-label={`${s.key} slider`} />
      </Slider.Root>
      <span className={cn('ends')}>
        <span>{lo}</span>
        <span>{hi}</span>
      </span>
    </>
  )
}

function SettingRow({ s, values, set }: { s: Setting; values: Record<string, string>; set: (name: string, v: string | undefined) => void }) {
  const name = settingName(s)
  const value = values[name] ?? s.server
  const changed = value !== s.server
  return (
    <div className={cn(`setting${changed ? ' changed' : ''}`)}>
      <div className={cn('setting-top')}>
        <span className={cn('setting-key')}>{s.key}</span>
        {changed && (
          <button type="button" className="reset" onClick={() => set(name, undefined)}>
            <RotateCcw size={11} aria-hidden />
            {`default ${s.server}`}
          </button>
        )}
        {s.choices && (
          <select className={cn('value-box')} aria-label={s.key} value={value} onChange={(e) => set(name, e.target.value)}>
            {s.choices.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        )}
      </div>
      {!s.choices && <NumberSetting s={s} value={value} onChange={(v) => set(name, v)} />}
      <p className={cn('setting-meaning')}>{s.meaning}</p>
    </div>
  )
}

/** The engine settings a user may change, as a tree of groups with a pane of the chosen group. */
export function SettingsTree({ schema, values, onChange }: { schema: SettingGroup[]; values: Record<string, string>; onChange: (v: Record<string, string>) => void }) {
  const [filter, setFilter] = useState('')
  const [open, setOpen] = useState(schema[0]?.name ?? '')
  const [refused, setRefused] = useState<string[]>([])
  const changes = changedSettings(schema, values)
  const count = Object.keys(changes).length
  const total = schema.reduce((n, g) => n + g.settings.length, 0)
  const f = filter.trim().toLowerCase()
  const matches = (s: Setting) => !f || s.key.toLowerCase().includes(f)
  const set = (name: string, v: string | undefined) => {
    const next = { ...values }
    if (v === undefined) delete next[name]
    else next[name] = v
    onChange(changedSettings(schema, next))
  }
  const shown = f ? schema.flatMap((g) => g.settings.filter(matches)) : (schema.find((g) => g.name === open)?.settings ?? [])

  const upload = async (file: File) => {
    const r = await api.parseIni(file)
    setRefused(r.refused)
    onChange(changedSettings(schema, { ...values, ...r.overrides }))
  }
  const download = () => {
    saveBlob(new Blob([toIni(changes)], { type: 'text/plain' }), 'config_changes.ini')
  }

  return (
    <div className="settings">
      <div className={cn('settings-head')}>
        <span className={cn('settings-title')}>
          <SlidersHorizontal size={14} aria-hidden /> 3D-GNOMEᴺᴳ settings
          {count > 0 && <span className={cn('changed-badge')}>{count} changed</span>}
        </span>
        <span className={cn('settings-actions')}>
          <button type="button" onClick={() => onChange({})}>
            <RotateCcw size={12} aria-hidden /> Reset all
          </button>
          <label>
            <Upload size={12} aria-hidden /> Upload config.ini
            <input type="file" accept=".ini,.txt" onChange={(e) => e.target.files?.[0] && void upload(e.target.files[0])} />
          </label>
          <button type="button" onClick={download} disabled={count === 0}>
            <Download size={12} aria-hidden /> Download
          </button>
        </span>
      </div>
      {refused.length > 0 && (
        <ul className={cn('refused')}>
          {refused.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      )}
      <div className={cn('panes')}>
        <div className={cn('tree-pane')}>
          <label className={cn('filter')}>
            <Search size={12} aria-hidden />
            <input placeholder={`Filter ${total} settings`} value={filter} onChange={(e) => setFilter(e.target.value)} />
          </label>
          <ul role="tree" aria-label="Settings">
            {schema.map((g) => {
              const n = g.settings.filter((s) => changes[settingName(s)] !== undefined).length
              const leaves = g.settings.filter(matches)
              const expanded = f ? leaves.length > 0 : open === g.name
              if (f && !leaves.length) return null
              return (
                <li key={g.name} role="treeitem" aria-expanded={expanded} aria-selected={open === g.name} aria-label={g.name}>
                  <button type="button" className={cn(`node${open === g.name ? ' open' : ''}`)} onClick={() => setOpen(g.name)}>
                    {expanded ? <ChevronDown size={12} aria-hidden /> : <ChevronRight size={12} aria-hidden />}
                    {g.name}
                    {n > 0 && <span className={cn('badge')}>{n}</span>}
                  </button>
                  {expanded && (
                    <ul role="group">
                      {leaves.map((s) => (
                        <li key={settingName(s)} role="treeitem" aria-selected={false} className={cn('leaf')}>
                          {s.key}
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
        <div className={cn('setting-pane')}>
          {!f && (
            <div className={cn('pane-head')}>
              <h3>{open}</h3>
              {GROUP_NOTES[open] && <p>{GROUP_NOTES[open]}</p>}
            </div>
          )}
          {shown.map((s) => (
            <SettingRow key={settingName(s)} s={s} values={values} set={set} />
          ))}
          {shown.length === 0 && (
            <p className={cn('setting-meaning')}>
              <Info size={12} aria-hidden /> No setting matches the filter.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

/** The structures and the job name of step 6. */
export function EnsembleFields({ structures, name, max, onChange }: { structures: number; name: string; max: number; onChange: (p: Partial<FormState>) => void }) {
  const [draft, setDraft] = useState(String(structures))
  useEffect(() => setDraft(String(structures)), [structures])
  return (
    <div className={cn('fields two')}>
      <label className="field">
        <span>Structures</span>
        <input
          type="number"
          min={1}
          max={max}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => onChange({ structures: clamp(Math.round(Number(draft) || 1), 1, max) })}
        />
      </label>
      <label className="field">
        <span>Job name (optional)</span>
        <input value={name} placeholder="e.g. TNFRSF8 with RNAPII loops" onChange={(e) => onChange({ name: e.target.value })} />
      </label>
    </div>
  )
}
