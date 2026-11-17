import { expect, test } from 'vitest'
import { BASE, prefixOf, served } from './base'

test('the prefix of a base URL has no trailing slash', () => {
  expect(prefixOf('/')).toBe('')
  expect(prefixOf('/new/')).toBe('/new')
  expect(prefixOf('/a/b/')).toBe('/a/b')
})

test('a server path carries the prefix', () => {
  expect(served('/api/catalog', '/new')).toBe('/new/api/catalog')
  expect(served('/jobs/k/files/ep.tsv', '')).toBe('/jobs/k/files/ep.tsv')
})

test('the test build is served at the root', () => {
  expect(BASE).toBe('')
})
