import type { UploadResult } from '../api'

/** An uploaded expression table, the samples of it the job takes and the kind of its values. */
export type OwnExpression = { upload: UploadResult; columns: string[]; kind: 'counts' | 'values' }

/**
 * The uploads a job uses in place of the data of a listed cell type and its expression
 * table. On your own cell type loopsFrom and enhancersFrom give a listed cell type whose
 * loops or enhancers it takes, enhancersFrom none for no enhancers.
 */
export type OwnChoice = {
  loops: { factor: string; upload: UploadResult }[]
  anchors: UploadResult | null
  contacts: UploadResult | null
  enhancers: UploadResult | null
  loopsFrom?: string
  enhancersFrom?: string
  /** An expression table with the samples the job takes from it. */
  expression?: OwnExpression | null
}

export const NO_OWN: OwnChoice = { loops: [], anchors: null, contacts: null, enhancers: null }

/**
 * Returns the data of the own data card under the names of the data of a listed cell type, which
 * are each factor of the loops, the contacts and the enhancers. A set taken from a listed cell type
 * has its name.
 */
export function ownData(own: OwnChoice): string[] {
  return [
    ...own.loops.map((l) => l.factor),
    ...(own.loopsFrom ? [`${own.loopsFrom} loops`] : []),
    ...(own.contacts ? ['contacts'] : []),
    ...(own.enhancers ? ['enhancers'] : own.enhancersFrom && own.enhancersFrom !== 'none' ? [`${own.enhancersFrom} enhancers`] : []),
    ...(own.expression ? ['RNA-seq'] : []),
  ]
}

/** Returns the uploads as the JSON field own of POST /api/jobs. */
export function ownField(own: OwnChoice): string {
  return JSON.stringify({
    loops: own.loops.map((l) => ({ factor: l.factor, upload: l.upload.id })),
    anchors: own.anchors?.id ?? null,
    contacts: own.contacts?.id ?? null,
    enhancers: own.enhancers?.id ?? null,
    loops_from: own.loops.length ? undefined : own.loopsFrom,
    enhancers_from: own.enhancers ? undefined : own.enhancersFrom,
  })
}

/** Returns the factor a loops file most likely holds, read from its name and CTCF for the first file. */
export function guessFactor(name: string, index: number): string {
  const n = name.toUpperCase()
  if (/RNAP|POL2|POLII/.test(n)) return 'RNAPII'
  if (n.includes('CTCF')) return 'CTCF'
  return index === 0 ? 'CTCF' : `factor ${index + 1}`
}
