/** A camera state shared by several scenes. The rotation and zoom of one scene is applied to the others. */
export type CameraState = { position: number[]; target: number[]; zoom: number }

export function makeCameraLink() {
  const listeners = new Map<string, (s: CameraState) => void>()
  return {
    publish(from: string, s: CameraState) {
      for (const [id, fn] of listeners) if (id !== from) fn(s)
    },
    subscribe(id: string, fn: (s: CameraState) => void) {
      listeners.set(id, fn)
      return () => listeners.delete(id)
    },
  }
}
export type CameraLink = ReturnType<typeof makeCameraLink>
