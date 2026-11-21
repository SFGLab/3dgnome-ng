/** The share of a view one pixel of wheel travel scales it by, and the most travel one event counts for. */
const PER_PIXEL = 0.0012
const MOST = 80

/**
 * The scale one wheel event asks of a view, above one to see more and below one to see less. It
 * follows the travel of the event. A notch of a mouse wheel scales the view by a tenth. A trackpad
 * sends many small events and each scales it by a little.
 */
export function wheelFactor(e: { deltaY: number; deltaMode?: number }): number {
  // Lines and pages become pixels.
  const px = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1)
  return Math.exp(Math.max(-MOST, Math.min(MOST, px)) * PER_PIXEL)
}
