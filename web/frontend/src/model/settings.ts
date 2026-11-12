import type { Setting, SettingGroup } from '../api'

/** The name a change is keyed by, section.key. */
export const settingName = (s: Setting) => `${s.section}.${s.key}`

/** The changes that differ from the server's values, keyed section.key. */
export function changedSettings(schema: SettingGroup[], values: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {}
  for (const s of schema.flatMap((g) => g.settings)) {
    const v = values[settingName(s)]
    if (v !== undefined && v !== s.server) out[settingName(s)] = v
  }
  return out
}

/** The changes as an ini file, one block per section. */
export function toIni(changes: Record<string, string>): string {
  const sections = new Map<string, string[]>()
  for (const [name, value] of Object.entries(changes)) {
    const [section, key] = [name.slice(0, name.indexOf('.')), name.slice(name.indexOf('.') + 1)]
    sections.set(section, [...(sections.get(section) ?? []), `${key} = ${value}`])
  }
  return [...sections].map(([section, lines]) => `[${section}]\n${lines.join('\n')}\n`).join('\n')
}
