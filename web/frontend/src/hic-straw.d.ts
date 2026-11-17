/** The part of hic-straw's API the site uses. The package ships no types. */
declare module 'hic-straw' {
  type Span = { chr: string; start: number; end: number }
  export default class Straw {
    constructor(args: { blob?: Blob; url?: string })
    getMetaData(): Promise<unknown>
    getContactRecords(normalization: string, a: Span, b: Span, units: 'BP', binSize: number): Promise<unknown[]>
  }
}
