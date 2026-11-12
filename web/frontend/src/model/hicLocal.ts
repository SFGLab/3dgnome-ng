/** Reads the bin pairs of a region from a local .hic file in the browser, as BEDPE text.
 *
 * @param file the .hic file the user chose
 * @param region the window to read
 * @param res the bin size in bp
 */
export async function readHicWindow(file: File, region: { chrom: string; start: number; end: number }, res: number): Promise<string> {
  void file
  void region
  void res
  throw new Error('This server cannot read a local .hic file yet. Choose Link to a file or BEDPE of the region.')
}
