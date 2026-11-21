/** One uploaded file in the own data dialog. */
import type { UploadResult } from '../../api'

/** A file in the dialog: checking, checked, or refused by the server with the way to fix it. */
export type Entry = { key: number; factor: string; name: string; size: number; result?: UploadResult; failure?: string; fix?: string }

export const fromResult = (key: number, factor: string, r: UploadResult): Entry => ({ key, factor, name: r.name, size: r.size, result: r })
export const passed = (e: Entry | null) => !e || !!e.result?.ok
export const pending = (e: Entry | null) => !!e && !e.result && !e.failure
/** The name of the file of an entry as the server has it. */
export const named = (e: Entry | null) => (e ? (e.result?.name ?? e.name) : null)
