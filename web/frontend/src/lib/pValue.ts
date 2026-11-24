/**
 * Returns a p value as the text beside a bar of a chart.
 * @param p the p value
 * @param floor the smallest p value the test can state, a value under it is stated as under the floor
 */
export const pValue = (p: number, floor = 0.001) => (p < floor ? `p < ${floor}` : `p = ${p.toFixed(p < 0.1 ? 3 : 2)}`)
