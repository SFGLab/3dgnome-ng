/** The routes of the uploads. */
import { call, posted, query } from './http'
import type { LoopDrawn, UploadKind, UploadResult } from './types'

/** Sends one file to POST /api/uploads, which stores it and checks it against the region. */
export const uploadFile = (kind: UploadKind, file: File, where: { cell: string; region: string; factor?: string }) =>
  call<UploadResult>('/api/uploads', posted({ kind, file, cell: where.cell, region: where.region, factor: where.factor || undefined }))

/** The loops of a loops upload inside a region, none when the server has no answer. */
export const uploadPreview = (id: string, cell: string, region: string) => call<LoopDrawn[]>(`/api/uploads/${id}/preview?${query({ cell, region })}`).catch((): LoopDrawn[] => [])

/** Asks the server to read the region of a remote .hic or .mcool file into a contacts upload. */
export const uploadLink = (url: string, where: { cell: string; region: string }) => call<UploadResult>('/api/uploads/link', posted({ url, cell: where.cell, region: where.region }))
