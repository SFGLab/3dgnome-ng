/** One way to ask the server: the answer as JSON or text, or an ApiError. */
import { served } from '../base'

/**
 * An answer of the server that is not 2xx: its message, its status, the form field it concerns
 * and the fix. Status 0 is a request the server did not answer.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly field?: string,
    readonly fix?: string,
  ) {
    super(message)
  }
}

export const NO_ANSWER = 'The server did not answer.'

async function answer(path: string, init?: RequestInit): Promise<Response> {
  let r: Response
  try {
    r = await fetch(served(path), init)
  } catch {
    throw new ApiError(NO_ANSWER, 0)
  }
  if (!r.ok) {
    const body = await r.json().catch(() => ({}))
    throw new ApiError(body.error ?? `The server answered ${r.status}.`, r.status, body.field, body.fix)
  }
  return r
}

/** The JSON answer of a path of the API. */
export const call = async <T>(path: string, init?: RequestInit): Promise<T> => (await (await answer(path, init)).json().catch(() => ({}))) as T

/** The text answer of a path of the API. */
export const callText = async (path: string): Promise<string> => (await answer(path)).text()

/** A form of fields as the body of a POST. */
export function posted(fields: Record<string, string | Blob | undefined>): RequestInit {
  const body = new FormData()
  for (const [k, v] of Object.entries(fields)) if (v !== undefined) body.append(k, v)
  return { method: 'POST', body }
}

export const query = (params: Record<string, string | number>) => new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString()
