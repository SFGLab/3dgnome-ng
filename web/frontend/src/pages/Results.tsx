import { DocPage } from '../components/DocPage'
import { BarChart, ChartKey, type Bar, type Group } from '../components/BarChart'
import { LineChart, type Line, type Point } from '../components/LineChart'
import { ToolRuns, type ToolRun } from '../components/ToolRuns'
import { hidden } from '../flags'
import { ROUTES } from '../routes'
import { scoped } from '../lib/css'
import { pValue } from '../lib/pValue'
import sheet from './results.module.css'

const cn = scoped(sheet)

/** Returns a bar. `p` is the p value of its test against the bar of 3D-GNOMEᴺᴳ with CTCF loops of the same group. */
const bar = (series: Bar['series'], value: number | null, digits = 2, p?: number): Bar => ({
  series,
  value,
  label: value === null ? 'placeholder' : value.toFixed(digits),
  p: p === undefined ? undefined : pValue(p),
})

/** A p value under 0.001, for a bar whose test gives one. */
const LOW = 1e-4

/**
 * Returns the bars of one cell line: the original models, the chain of one bead per kilobase with CTCF loops and with RNAPII loops added, the chain of loop anchors.
 * @param name the cell line
 * @param values the four values
 * @param rnapiiP the p value of the models with RNAPII loops against those with CTCF loops. The other two are under 0.001 at every resolution.
 */
const agreement = (name: string, [original, ctcf, rnapii, anchors]: number[], rnapiiP: number): Group => ({
  name,
  bars: [bar('original', original, 2, LOW), bar('ctcf', ctcf), bar('rnapii', rnapii, 2, rnapiiP), bar('anchors', anchors, 2, LOW)],
})

/** The agreement with Hi-C per bead pair at each resolution in kilobases, for pairs up to 2 Mb apart. */
const SCC: [number, Group[]][] = [
  [5, [agreement('GM12878', [0.18, 0.32, 0.39, 0.25], LOW), agreement('H1ESC', [0.11, 0.26, 0.29, 0.19], LOW), agreement('HFFc6', [0.13, 0.34, 0.38, 0.26], LOW)]],
  [10, [agreement('GM12878', [0.24, 0.41, 0.47, 0.32], LOW), agreement('H1ESC', [0.16, 0.35, 0.37, 0.28], 0.0042), agreement('HFFc6', [0.18, 0.42, 0.46, 0.34], LOW)]],
  [25, [agreement('GM12878', [0.3, 0.49, 0.51, 0.41], 0.036), agreement('H1ESC', [0.2, 0.43, 0.43, 0.38], 0.5), agreement('HFFc6', [0.21, 0.51, 0.5, 0.44], 0.52)]],
]

const DISTANCE: Group[] = [
  { name: 'GM12878, 17% closer', bars: [bar('ctcf', 4.0, 1), bar('rnapii', 3.3, 1, LOW)] },
  { name: 'H1ESC, 12% closer', bars: [bar('ctcf', 3.9, 1), bar('rnapii', 3.4, 1, LOW)] },
  { name: 'HFFc6, 14% closer', bars: [bar('ctcf', 4.3, 1), bar('rnapii', 3.7, 1, LOW)] },
]

const EXPRESSION: Group[] = [
  { name: 'GM12878 and H1ESC', bars: [bar('original', 0.14, 2, LOW), bar('ctcf', 0.23), bar('rnapii', 0.31, 2, LOW)] },
  { name: 'H1ESC and HFFc6', bars: [bar('original', 0.13, 2, 0.0006), bar('ctcf', 0.2), bar('rnapii', 0.24, 2, LOW)] },
  { name: 'GM12878 and HFFc6', bars: [bar('original', 0.14, 2, LOW), bar('ctcf', 0.24), bar('rnapii', 0.28, 2, 0.001)] },
]

/** Returns the runs of the three tools on one task, each as beads and seconds. A null time is a run that did not finish. */
const runs = (cudammc: [number, number], multimm: [number, number | null], ours: [number, number], note?: string): ToolRun[] => [
  { series: 'cudammc', name: 'cudaMMC', beads: cudammc[0], seconds: cudammc[1] },
  { series: 'multimm', name: 'MultiMM', beads: multimm[0], seconds: multimm[1], note },
  { series: 'ctcf', name: '3D-GNOMEᴺᴳ', beads: ours[0], seconds: ours[1] },
]

const MICROC: Group[] = [
  { name: 'GM12878, 25,783 loops', bars: [bar('original', 0.12, 2, LOW), bar('ctcf', 0.26), bar('rnapii', 0.33, 2, LOW)] },
  { name: 'H1ESC, 10,046 loops', bars: [bar('original', 0.12, 2, LOW), bar('ctcf', 0.39), bar('rnapii', 0.45, 2, LOW)] },
  { name: 'HFFc6, 19,192 loops', bars: [bar('original', 0.1, 2, LOW), bar('ctcf', 0.44), bar('rnapii', 0.52, 2, LOW)] },
]

const MICROC_RNAPII: Group[] = [
  { name: 'GM12878, 2,130 loops', bars: [bar('ctcf', 0.14), bar('rnapii', 0.65, 2, LOW)] },
  { name: 'H1ESC, 586 loops', bars: [bar('ctcf', 0.19), bar('rnapii', 0.67, 2, LOW)] },
  { name: 'HFFc6, 1,318 loops', bars: [bar('ctcf', 0.24), bar('rnapii', 0.7, 2, LOW)] },
]

const TOOLS_MICROC: Group[] = [
  { name: 'GM12878, 902 loops', bars: [bar('cudammc', 0.14, 2, LOW), bar('multimm', 0.25, 2, 0.21), bar('ctcf', 0.29)] },
  { name: 'H1ESC, 385 loops', bars: [bar('cudammc', 0.21, 2, LOW), bar('multimm', 0.16, 2, LOW), bar('ctcf', 0.53)] },
  { name: 'HFFc6, 639 loops', bars: [bar('cudammc', 0.16, 2, LOW), bar('multimm', 0.15, 2, LOW), bar('ctcf', 0.51)] },
]

const TOOLS_MICROC_BEADS: Group[] = [
  { name: 'GM12878', bars: [bar('cudammc', 0.25, 2, LOW), bar('multimm', 1.0, 2, LOW), bar('ctcf', 0.98)] },
  { name: 'H1ESC', bars: [bar('cudammc', 0.26, 2, LOW), bar('multimm', 1.0, 2, LOW), bar('ctcf', 0.95)] },
  { name: 'HFFc6', bars: [bar('cudammc', 0.22, 2, LOW), bar('multimm', 1.0, 2, LOW), bar('ctcf', 0.97)] },
]

/** Returns the bar of another tool. Its bootstrap of 1,000 resamples has no resample with the tool ahead, a p value under 0.002. */
const tool = (series: Bar['series'], value: number): Bar => ({ ...bar(series, value), p: pValue(0.001, 0.002) })

const TOOLS_SCC: Group[] = [
  { name: 'GM12878', bars: [tool('cudammc', 0.27), tool('multimm', 0.3), bar('ctcf', 0.47)] },
  { name: 'H1ESC', bars: [tool('cudammc', 0.17), tool('multimm', 0.16), bar('ctcf', 0.35)] },
  { name: 'HFFc6', bars: [tool('cudammc', 0.2), tool('multimm', 0.18), bar('ctcf', 0.45)] },
]

const RUN_TIME: [string, string, 'seconds' | 'minutes', ToolRun[]][] = [
  ['One structure, 1 Mb', 'chr1:11.5-12.6 Mb.', 'seconds', runs([350, 6.4], [760, 7.4], [760, 9.0])],
  ['Ten structures, 60 Mb', 'chr1:1-60 Mb.', 'minutes', runs([17321, 560.9], [42480, 1207.5], [42480, 634.4])],
  ['One structure, whole chromosome', 'chr1, 248 Mb.', 'minutes', runs([50631, 194.8], [124657, 853.1], [124657, 384.2])],
  ['Three structures at equal bead counts', 'chr1, 248 Mb. cudaMMC was set to about the same number of beads as the other two tools.', 'minutes', runs([126330, 641], [124657, 2528], [124657, 673])],
  ['One structure, whole genome', 'chr1 to chr22.', 'minutes', runs([458445, 1882.1], [1312597, null], [1312597, 2784.0], 'did not complete')],
]

/** One structure of GM12878 chr1 from 10 Mb per tool, as megabases, beads and seconds. The last is the whole chromosome. */
const ONE_STRUCTURE: [Line['series'], string, [number, number, number][]][] = [
  ['cudammc', 'cudaMMC', [[1, 129, 6.5], [2, 448, 7.2], [5, 762, 8.9], [10, 2170, 13.7], [20, 7128, 25.5], [50, 14936, 53.3], [100, 20401, 79.4], [248, 50631, 194.8]]],
  ['multimm', 'MultiMM', [[1, 662, 7.3], [2, 1598, 8.2], [5, 2590, 8.5], [10, 6953, 15.8], [20, 15643, 33.9], [50, 36255, 97.6], [100, 56458, 189.7], [248, 124657, 853.1]]],
  ['ctcf', '3D-GNOMEᴺᴳ', [[1, 662, 9.1], [2, 1598, 13.0], [5, 2590, 15.4], [10, 6953, 22.0], [20, 15643, 45.1], [50, 36255, 88.2], [100, 56458, 124.4], [248, 124657, 384.2]]],
]

const sized = (x: (mb: number, beads: number) => number): Line[] =>
  ONE_STRUCTURE.map(([series, name, rows]) => ({
    series,
    name,
    points: rows.map(([mb, beads, seconds]): Point => ({ x: x(mb, beads), seconds, beads, region: mb === 248 ? 'chr1, 248 Mb' : `${mb} Mb` })),
  }))
const BY_REGION = sized((mb) => mb)
const BY_BEADS = sized((_, beads) => beads)

const SECTIONS: [to: string, label: string][] = [
  ['#agreement', 'Agreement with Hi-C'],
  ['#distance', 'Enhancer to promoter distance'],
  ['#expression', 'Link to expression'],
  ['#trios', 'Nine people, three trios'],
  ['#microc', 'Micro-C loops'],
  ['#tools', 'Other tools'],
  [`${ROUTES.method}#limitations`, 'Limitations'],
  ['#runtime', 'Runtime'],
]

/** The models against measured data and against cudaMMC and MultiMM and the runtime of the three. */
export function Results() {
  const trios = !hidden('trios')
  const sections = SECTIONS.filter(([to]) => (to !== '#trios' || trios) && !(to.startsWith(ROUTES.method) && hidden('method')))
  return (
    <DocPage
      title="Results"
      sections={sections}
      wide
      lead={
        <>
          This page compares 3D-GNOMEᴺᴳ models with measured data and with two other tools. It also shows the change in the models when RNAPII loops are added to the CTCF loops. Unless a chart states another region, the models cover chromosomes 1 to 22 in the cell lines GM12878, H1ESC and HFFc6, with ten structures per chromosome. The 3D-GNOMEᴺᴳ models are built on the chain of one bead per kilobase. Their contact cells read one half of the Hi-C counts and every comparison with Hi-C uses the other half. A structure of this chain keeps a loop with a probability that grows with the strength of the loop, as the Method page describes. The models compared with Hi-C are built this way, as a job of the site is. The charts of the enhancer to promoter distance, of the link to expression and of the Micro-C loops are from models built with every loop in every structure. Such models have the whole effect of a loop on its two ends. Their agreement with Hi-C is lower, by 0.02 to 0.04 with RNAPII loops. The original models are the cudaMMC models of 3D-GNOME 3.0.
        </>
      }
    >
      <ChartKey series={['original', 'ctcf', 'rnapii', 'anchors']} />
      <div id="agreement" className={cn('chart-row three')}>
        {SCC.map(([kb, groups]) => (
          <BarChart
            key={kb}
            title={`Hi-C agreement at ${kb} kb`}
            subtitle={`The stratum adjusted correlation between the contact map of a model and the Hi-C map in bins of ${kb} kb, for pairs up to 2 Mb apart. Contacts are counted per pair of beads.`}
            better="higher"
            groups={groups}
          />
        ))}
      </div>
      <p className={cn('reading-note')}>
        The contact map of a model is the sum of the contacts of its ten structures. The count of a cell of this map is divided by the number of bead pairs of its two bins. Without the division a bin with more beads has more contacts, wherever the beads are. Both chains have beads only on the stretches that loops cover. On these stretches the chain of one bead per kilobase has 25 beads in a bin of 25 kb. The chain of loop anchors has between 1 and 68 beads there, 14 in the median. It has more of them where Hi-C has more reads. The rank correlation between the number of beads of a bin and its Hi-C reads is 0.27 to 0.53 in the three cell lines for this chain and under 0.05 for the other. The undivided count of the chain of loop anchors therefore agrees with Hi-C through the number of beads alone. On GM12878 at 25 kb it scores 0.53 undivided and 0.41 per pair of beads. The chain of one bead per kilobase scores 0.50 and 0.49.
      </p>
      <div className={cn('chart-row')}>
        <div id="distance">
          <BarChart title="Enhancer to promoter distance" subtitle="The median 3D distance between an enhancer and its promoter, in bead units. Only the pairs joined by an RNAPII loop are counted." groups={DISTANCE} />
        </div>
        <div id="expression">
          <BarChart
            title="Link to expression"
            better="higher"
            subtitle="The Spearman correlation between the change in enhancer to promoter distance and the change in gene expression between two cell lines. The sign is flipped and a higher value means a stronger link."
            groups={EXPRESSION}
          />
        </div>
      </div>
      {trios && <div className={cn('chart-row')}>
        <figure id="trios" className="bar-chart" aria-label="Nine people, three trios">
          <figcaption>
            <span className="chart-title">Nine people, three trios</span>
            <span className="chart-sub">The change in the model of each person when RNAPII loops are added, measured against the Hi-C map of that person on chromosomes 1 to 22.</span>
          </figcaption>
          <div className={cn('stat')}>
            <span className={cn('stat-value')}>9 of 9</span>
            <span className={cn('stat-label')}>people with more contact enrichment at RNAPII loops. The rise is 44 to 67% and the median is 53%.</span>
          </div>
        </figure>
      </div>}
      <p className={cn('reading-note')}>
        With RNAPII loops, enhancers are closer to their genes and the link to expression is stronger. Agreement with Hi-C is higher at 5 kb and at 10 kb in every cell line, 0.39 against 0.32 for GM12878 at 5 kb. At 25 kb it is higher for GM12878 and does not change for the other two cell lines. At 10 kb, agreement with Hi-C is higher for pairs under 100 kb apart and lower for pairs 100 to 500 kb apart on all three cell lines.
      </p>
      <p className={cn('reading-note')}>
        The p value beside a bar is that of the difference between the bar and the bar of 3D-GNOMEᴺᴳ with CTCF loops in the same group. Every test is two sided. Agreement with Hi-C is tested with a Wilcoxon signed rank test over the 22 chromosomes. The new models agree better with Hi-C than the original models on every cell line and at every resolution. The chain of one bead per kilobase agrees better than the chain of loop anchors on each of the 22 chromosomes of every cell line at every resolution. The enhancer to promoter distance is tested with a signed rank test over the pairs. The link to expression is tested with a block bootstrap over the genes.{trios && ' The contact enrichment of the nine people is higher with p = 0.004 in an exact signed rank test.'}
      </p>
      <section id="microc">
        <h2>Micro-C loops</h2>
        <p className="intro">
          Micro-C resolves loops that Hi-C at 25 kb does not. Loops were called at 5 kb on each of chromosomes 1 to 22, on the 4DN Micro-C maps of H1ESC and HFFc6 and on the GM12878 Micro-C map of Hong and colleagues, Zenodo record 15303879. A loop is kept only if its signal is strong against its local background. The loops kept on chr1:0-60 Mb also appear in the Hi-C map of the same cell line. A model places a loop if the two ends of the loop are at least a fifth closer in 3D than pairs at the same genomic separation. Each chart shows the share of loops that a model places, minus the share it places after the same loops are moved away from their position. A model that cannot tell a loop from another pair scores zero.
        </p>
        <div className={cn('chart-row')}>
          <BarChart
            title="Micro-C loops placed"
            subtitle="All loops called on the Micro-C map. The share placed, minus the share placed for moved loops."
            better="higher"
            groups={MICROC}
          />
          <BarChart
            title="Micro-C loops with an RNAPII loop only"
            subtitle="The Micro-C loops that match an RNAPII ChIA-PET loop and no CTCF loop. The share placed, minus the share placed for moved loops."
            better="higher"
            groups={MICROC_RNAPII}
          />
        </div>
        <p className={cn('reading-note')}>
          Between 46 and 63 percent of the Micro-C loops have no ChIA-PET loop near them and the models place few of those. With RNAPII loops added, the models place more of the loops that only RNAPII supports, on every cell line. They also place more of the loops that CTCF supports. Both rises have p &lt; 0.001 in a signed rank test over the 22 chromosomes.
        </p>
      </section>
      <section id="tools">
        <h2>Other tools</h2>
        <p className="intro">
          cudaMMC, MultiMM and 3D-GNOMEᴺᴳ were run on the same CTCF loops and the same Hi-C contacts. 3D-GNOMEᴺᴳ ran with the chain of one bead per kilobase. cudaMMC and MultiMM used their default settings. MultiMM was set to the number of beads of the 3D-GNOMEᴺᴳ chain of loop anchors, which is up to a third lower than that of the chain of one bead per kilobase. The contact cells of 3D-GNOMEᴺᴳ read one half of the Hi-C counts and every tool is compared with the other half. The charts of this section that compare the models with Hi-C and Micro-C cover chr1:1-60 Mb.
        </p>
        <ChartKey series={['cudammc', 'multimm', 'ctcf']} label="Key of the tools" />
        <div className={cn('chart-row')}>
          <div>
            <BarChart
              title="Agreement with Hi-C, three tools"
              subtitle="The stratum adjusted correlation with the Hi-C map at 25 kb, on chr1:1-60 Mb. The contact map of a model is the sum of the contacts of its ten structures. Contacts are counted per pair of beads."
              better="higher"
              groups={TOOLS_SCC}
            />
          </div>
          <figure className="bar-chart" aria-label="Comparison over 21 runs">
            <figcaption>
              <span className="chart-title">Comparison over 21 runs</span>
              <span className="chart-sub">Each of the three cell lines was modelled on three regions of 5 Mb, three regions of 20 Mb and chr1:1-60 Mb. A tool counts as better in a run when the 95% bootstrap interval of the difference does not include zero. The stratum adjusted correlation counts contacts per pair of beads.</span>
            </figcaption>
            <div className={cn('stat')}>
              <span className={cn('stat-value')}>21 of 21</span>
              <span className={cn('stat-label')}>runs with a higher Pearson correlation and a higher rank correlation with Hi-C for 3D-GNOMEᴺᴳ than for both other tools</span>
            </div>
            <div className={cn('stat')}>
              <span className={cn('stat-value')}>6 of 21</span>
              <span className={cn('stat-label')}>runs with a higher stratum adjusted correlation for 3D-GNOMEᴺᴳ than for MultiMM. The count against cudaMMC is 13 of 21. Neither tool is better than 3D-GNOMEᴺᴳ in any run.</span>
            </div>
          </figure>
        </div>
        <div className={cn('chart-row')}>
          <BarChart
            title="Micro-C loops placed, three tools"
            subtitle="All loops called on the Micro-C map, chr1:0-60 Mb. The share placed, minus the share placed for moved loops."
            better="higher"
            groups={TOOLS_MICROC}
          />
          <BarChart
            title="Micro-C loops with beads at both ends"
            subtitle="The share of Micro-C loops with model beads in both 5 kb bins of the loop. A model cannot place a loop without them."
            better="higher"
            groups={TOOLS_MICROC_BEADS}
          />
        </div>
        <p className={cn('reading-note')}>
          cudaMMC has beads at both ends of about a quarter of the Micro-C loops and this limits the share it places. MultiMM has beads at every loop. It places nearly as many loops as 3D-GNOMEᴺᴳ on GM12878 and a third as many on the other two cell lines. The share of loops placed is tested by 10,000 resamples of the loops and of the moved loops. The share of loops with beads is tested with an exact sign test over the loops.
        </p>
        <p className={cn('reading-note')}>
          The p value beside the bar of a tool is that of its difference from 3D-GNOMEᴺᴳ in a paired block bootstrap, which resamples the bins of the region 1,000 times. A region of 5 Mb has too few bins for the stratum adjusted correlation to separate the tools. On the nine runs of 20 Mb, 3D-GNOMEᴺᴳ has the higher value against cudaMMC in eight and against MultiMM in seven. By the interval it is better than cudaMMC in four and better than MultiMM in one. On chr1:1-60 Mb it is better than both tools in every cell line.
        </p>
      </section>
      <section id="runtime">
        <h2>Runtime</h2>
        <p className="intro">
          The three tools were timed on the same GM12878 inputs and on one computer with an RTX 3090 and 36 CPU cores. 3D-GNOMEᴺᴳ was timed with its default chain, the chain of loop anchors. At its default settings cudaMMC uses a third to a half of the beads of the other two tools for the same region. Each chart therefore shows three numbers per tool, the number of beads in its model, its runtime and its runtime per 1,000 beads. A tick marks the lowest time in a column.
        </p>
        <div className={cn('chart-row one')}>
          {RUN_TIME.map(([title, sub, unit, toolRuns]) => (
            <ToolRuns key={title} title={title} subtitle={sub} unit={unit} runs={toolRuns} />
          ))}
        </div>
        <p className={cn('reading-note')}>
          Per 1,000 beads, 3D-GNOMEᴺᴳ is the fastest tool for ten structures of 60 Mb, for one structure of the chromosome and for the genome. MultiMM is the fastest tool at 1 Mb. At equal bead counts cudaMMC is 5 percent faster than 3D-GNOMEᴺᴳ.
        </p>
        <p className={cn('reading-note')}>
          3D-GNOMEᴺᴳ models the whole genome in one job. cudaMMC was run on one chromosome at a time and the chart shows the sum of its 22 runs. MultiMM ran out of memory on the 24 GB card at 1,312,597 beads and at half that number. At a quarter of that number it was stopped after 90 minutes.
        </p>
        <h3 className={cn('sub-head')}>Runtime and the size of the region</h3>
        <p className={cn('reading-note')}>
          The two charts below show the growth of the runtime of one structure with the size of the task. Each tool modelled eight regions of chr1, from 1 Mb to the whole chromosome of 248 Mb. Each point is one run. Both charts show the same runs. In the left chart the horizontal axis is the size of the region and the chart shows the time you wait for a region of that size. In the right chart the horizontal axis is the number of beads in the model and the chart compares the tools at the same number of beads. cudaMMC uses fewer beads than the other two tools for the same region. This lowers its line in the left chart and does not change the right chart.
        </p>
        <div className={cn('chart-row')}>
          <LineChart
            title="One structure by region size"
            subtitle="Regions of 1 to 100 Mb that start at chr1:10 Mb and the whole chromosome. Both axes are logarithmic."
            better="lower"
            xLabel="region, Mb"
            xTick={(x) => `${x}`}
            lines={BY_REGION}
          />
          <LineChart
            title="One structure by bead count"
            subtitle="The same runs, with the number of beads in each model on the horizontal axis. Both axes are logarithmic."
            better="lower"
            xLabel="beads"
            xTick={(x) => x.toLocaleString('en-US')}
            lines={BY_BEADS}
          />
        </div>
        <p className={cn('reading-note')}>
          By region size, cudaMMC is the fastest tool from 10 Mb. Below 10 Mb cudaMMC and MultiMM take about the same time. 3D-GNOMEᴺᴳ is faster than MultiMM from about 50 Mb. By bead count, MultiMM is the fastest tool up to about 30,000 beads and 3D-GNOMEᴺᴳ is the fastest above that. cudaMMC takes about the same time as 3D-GNOMEᴺᴳ up to about 7,000 beads and more time above that. Each further structure of the same region costs 3D-GNOMEᴺᴳ less time than the first. One structure of chr1 takes 6.4 minutes and three structures take 11.2 minutes.
        </p>
      </section>
    </DocPage>
  )
}
