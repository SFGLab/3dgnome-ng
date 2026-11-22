import type { Setting, SettingGroup } from '../api'

/** Returns the key of a change, section.key. */
export const settingName = (s: Setting) => `${s.section}.${s.key}`

/** Returns the changes that differ from the values of the server, keyed section.key. */
export function changedSettings(schema: SettingGroup[], values: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {}
  for (const s of schema.flatMap((g) => g.settings)) {
    const v = values[settingName(s)]
    // A value that the chain of the form does not offer is not sent.
    if (v !== undefined && v !== s.server && (!s.choices || s.choices.includes(v))) out[settingName(s)] = v
  }
  return out
}

/** Returns the changes as an ini file, one block per section. */
export function toIni(changes: Record<string, string>): string {
  const sections = new Map<string, string[]>()
  for (const [name, value] of Object.entries(changes)) {
    const [section, key] = [name.slice(0, name.indexOf('.')), name.slice(name.indexOf('.') + 1)]
    sections.set(section, [...(sections.get(section) ?? []), `${key} = ${value}`])
  }
  return [...sections].map(([section, lines]) => `[${section}]\n${lines.join('\n')}\n`).join('\n')
}
