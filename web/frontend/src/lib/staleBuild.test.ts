import { afterEach, describe, expect, it, vi } from 'vitest'
import { reloadOnStaleBuild } from './staleBuild'

function fakeWindow() {
  const store = new Map<string, string>()
  const target = new EventTarget()
  const reload = vi.fn()
  const win = {
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
    sessionStorage: { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) },
    location: { reload },
  } as unknown as Window
  return { win, reload, fail: () => target.dispatchEvent(new Event('vite:preloadError', { cancelable: true })) }
}

describe('A script of the build that the server does not have', () => {
  afterEach(() => vi.useRealTimers())

  it('loads the page anew once', () => {
    const { win, reload, fail } = fakeWindow()
    reloadOnStaleBuild(win)
    expect(fail()).toBe(false)
    expect(reload).toHaveBeenCalledTimes(1)
  })

  it('does not load the page a second time right after', () => {
    vi.useFakeTimers()
    const { win, reload, fail } = fakeWindow()
    reloadOnStaleBuild(win)
    fail()
    vi.advanceTimersByTime(5_000)
    expect(fail()).toBe(true)
    expect(reload).toHaveBeenCalledTimes(1)
    vi.advanceTimersByTime(60_000)
    fail()
    expect(reload).toHaveBeenCalledTimes(2)
  })

  it('stops after its listener is removed', () => {
    const { win, reload, fail } = fakeWindow()
    reloadOnStaleBuild(win)()
    fail()
    expect(reload).not.toHaveBeenCalled()
  })
})
