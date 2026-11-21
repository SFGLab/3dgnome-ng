import { DocPage } from '../components/DocPage'
import { BarChart, ChartKey, type Bar, type Group } from '../components/BarChart'
import { LineChart, type Line, type Point } from '../components/LineChart'
import { ToolRuns, type ToolRun } from '../components/ToolRuns'
import { hidden } from '../flags'
import { ROUTES } from '../routes'
import { scoped } from '../lib/css'
import sheet from './results.module.css'

const cn = scoped(sheet)

const bar = (series: Bar['series'], value: number | null, digits = 2): Bar => ({
  series,
  value,
  label: value === null ? 'placeholder' : value.toFixed(digits),
})

const SCC: Group[] = [
  { name: 'GM12878', bars: [bar('original', 0.19), bar('ctcf', 0.53)] },
  { name: 'H1ESC', bars: [bar('original', 0.13), bar('ctcf', 0.45)] },
  { name: 'HFFc6', bars: [bar('original', 0.15), bar('ctcf', 0.47)] },
]

const DISTANCE: Group[] = [
  { name: 'GM12878, 19% closer', bars: [bar('ctcf', 4.3, 1), bar('rnapii', 3.5, 1)] },
  { name: 'H1ESC, 24% closer', bars: [bar('ctcf', 4.0, 1), bar('rnapii', 3.1, 1)] },
  { name: 'HFFc6, 17% closer', bars: [bar('ctcf', 4.6, 1), bar('rnapii', 3.8, 1)] },
]

const EXPRESSION: Group[] = [
  { name: 'GM12878 and H1ESC', bars: [bar('original', 0.14), bar('ctcf', 0.19), bar('rnapii', 0.24)] },
  { name: 'H1ESC and HFFc6', bars: [bar('original', 0.14), bar('ctcf', 0.17), bar('rnapii', 0.21)] },
  { name: 'GM12878 and HFFc6', bars: [bar('original', 0.14), bar('ctcf', 0.23), bar('rnapii', 0.27)] },
]

/** The runs of the three tools on one task, each as beads and seconds. A null time is a run that did not finish. */
const runs = (cudammc: [number, number], multimm: [number, number | null], ours: [number, number], note?: string): ToolRun[] => [
  { series: 'cudammc', name: 'cudaMMC', beads: cudammc[0], seconds: cudammc[1] },
  { series: 'multimm', name: 'MultiMM', beads: multimm[0], seconds: multimm[1], note },
  { series: 'ctcf', name: '3D-GNOMEᴺᴳ', beads: ours[0], seconds: ours[1] },
]

const MICROC: Group[] = [
  { name: 'GM12878, 25,783 loops', bars: [bar('original', 0.12), bar('ctcf', 0.23), bar('rnapii', 0.31)] },
  { name: 'H1ESC, 10,046 loops', bars: [bar('original', 0.12), bar('ctcf', 0.35), bar('rnapii', 0.4)] },
  { name: 'HFFc6, 19,192 loops', bars: [bar('original', 0.1), bar('ctcf', 0.39), bar('rnapii', 0.47)] },
]

const MICROC_RNAPII: Group[] = [
  { name: 'GM12878, 2,130 loops', bars: [bar('ctcf', 0.13), bar('rnapii', 0.66)] },
  { name: 'H1ESC, 586 loops', bars: [bar('ctcf', 0.18), bar('rnapii', 0.58)] },
  { name: 'HFFc6, 1,318 loops', bars: [bar('ctcf', 0.23), bar('rnapii', 0.64)] },
]

const TOOLS_MICROC: Group[] = [
  { name: 'GM12878, 902 loops', bars: [bar('cudammc', 0.14), bar('multimm', 0.25), bar('ctcf', 0.26)] },
  { name: 'H1ESC, 385 loops', bars: [bar('cudammc', 0.21), bar('multimm', 0.16), bar('ctcf', 0.38)] },
  { name: 'HFFc6, 639 loops', bars: [bar('cudammc', 0.16), bar('multimm', 0.15), bar('ctcf', 0.45)] },
]

const TOOLS_MICROC_BEADS: Group[] = [
  { name: 'GM12878', bars: [bar('cudammc', 0.25), bar('multimm', 1.0), bar('ctcf', 0.97)] },
  { name: 'H1ESC', bars: [bar('cudammc', 0.26), bar('multimm', 1.0), bar('ctcf', 0.94)] },
  { name: 'HFFc6', bars: [bar('cudammc', 0.22), bar('multimm', 1.0), bar('ctcf', 0.96)] },
]

const TOOLS_SCC: Group[] = [
  { name: 'GM12878', bars: [bar('cudammc', 0.22), bar('multimm', 0.3), bar('ctcf', 0.51)] },
  { name: 'H1ESC', bars: [bar('cudammc', 0.15), bar('multimm', 0.17), bar('ctcf', 0.34)] },
  { name: 'HFFc6', bars: [bar('cudammc', 0.15), bar('multimm', 0.18), bar('ctcf', 0.44)] },
]

const RUN_TIME: [string, string, 'seconds' | 'minutes', ToolRun[]][] = [
  ['One structure, 1 Mb', 'chr1:11.5-12.6 Mb.', 'seconds', runs([350, 6.4], [760, 7.4], [760, 9.0])],
  ['Ten structures, 60 Mb', 'chr1:1-60 Mb.', 'minutes', runs([17321, 560.9], [42480, 1207.5], [42480, 634.4])],
  ['One structure, whole chromosome', 'chr1, 248 Mb.', 'minutes', runs([50631, 194.8], [124657, 853.1], [124657, 384.2])],
  ['Three structures at equal bead counts', 'chr1, 248 Mb. cudaMMC was set to about the same number of beads as the other two tools.', 'minutes', runs([126330, 641], [124657, 2528], [124657, 673])],
  ['One structure, whole genome', 'chr1 to chr22.', 'minutes', runs([458445, 1882.1], [1312597, null], [1312597, 2784.0], 'did not complete')],
]

/** One structure of GM12878 chr1 from 10 Mb per tool: megabases, beads, seconds. The last is the whole chromosome. */
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
          This page compares 3D-GNOMEᴺᴳ models with measured data and with two other tools. It also shows the change in the models when RNAPII loops are added to the CTCF loops. Unless a chart states another region, the models cover chromosomes 1 to 22 in the cell lines GM12878, H1ESC and HFFc6, with ten structures per chromosome. The original models are the cudaMMC models of 3D-GNOME 3.0.
        </>
      }
    >
      <ChartKey />
      <div className={cn('chart-row')}>
        <div id="agreement">
          <BarChart title="Agreement with Hi-C" subtitle="The stratum adjusted correlation between the contact map of a model and the Hi-C map, at 25 kb. The contact map of a model is the sum of the contacts of its ten structures." better="higher" groups={SCC} />
        </div>
        <div id="distance">
          <BarChart title="Enhancer to promoter distance" subtitle="The median 3D distance between an enhancer and its promoter, in bead units. Only the pairs joined by an RNAPII loop are counted." groups={DISTANCE} />
        </div>
      </div>
      <div className={cn('chart-row')}>
        <div id="expression">
          <BarChart
            title="Link to expression"
            better="higher"
            subtitle="The Spearman correlation between the change in enhancer to promoter distance and the change in gene expression between two cell lines. The sign is flipped and a higher value means a stronger link."
            groups={EXPRESSION}
          />
        </div>
        {trios && <figure id="trios" className="bar-chart" aria-label="Nine people, three trios">
          <figcaption>
            <span className="chart-title">Nine people, three trios</span>
            <span className="chart-sub">The change in the model of each person when RNAPII loops are added, measured against the Hi-C map of that person on chromosomes 1 to 22.</span>
          </figcaption>
          <div className={cn('stat')}>
            <span className={cn('stat-value')}>9 of 9</span>
            <span className={cn('stat-label')}>people with more contact enrichment at RNAPII loops. The rise is 44 to 67% and the median is 53%.</span>
          </div>
        </figure>}
      </div>
      <p className={cn('reading-note')}>
        With RNAPII loops, enhancers are closer to their genes and the link to expression is stronger. Agreement with the whole Hi-C map is lower for H1ESC, 0.42 against 0.45. At 10 kb, agreement with Hi-C is higher for pairs under 100 kb apart on GM12878 and HFFc6 and lower for pairs 100 to 500 kb apart on all three cell lines.
      </p>
      <p className={cn('reading-note')}>
        Each difference was tested with a two sided test. The new models agree better with Hi-C than the original models in every cell line, p &lt; 0.001 in a Wilcoxon signed rank test over the 22 chromosomes. With RNAPII loops the enhancer to promoter distance is smaller in every cell line, p &lt; 0.001 in a signed rank test over the pairs. The link to expression is stronger for every pair of cell lines, p = 0.002 or less in a block bootstrap over the genes.{trios && ' The contact enrichment of the nine people is higher with p = 0.004 in an exact signed rank test.'}
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
          cudaMMC, MultiMM and 3D-GNOMEᴺᴳ were run on the same CTCF loops and the same Hi-C contacts. cudaMMC and MultiMM used their default settings and MultiMM was set to the same number of beads as 3D-GNOMEᴺᴳ. The charts of this section that compare the models with Hi-C and Micro-C cover chr1:1-60 Mb.
        </p>
        <ChartKey series={['cudammc', 'multimm', 'ctcf']} label="Key of the tools" />
        <div className={cn('chart-row')}>
          <div>
            <BarChart
              title="Agreement with Hi-C, three tools"
              subtitle="The stratum adjusted correlation with the Hi-C map at 25 kb, on chr1:1-60 Mb. The contact map of a model is the sum of the contacts of its ten structures."
              better="higher"
              groups={TOOLS_SCC}
            />
          </div>
          <figure className="bar-chart" aria-label="Comparison over 21 runs">
            <figcaption>
              <span className="chart-title">Comparison over 21 runs</span>
              <span className="chart-sub">Each of the three cell lines was modelled on three regions of 5 Mb, three regions of 20 Mb and chr1:1-60 Mb. A tool counts as better in a run when the 95% bootstrap interval of the difference does not include zero.</span>
            </figcaption>
            <div className={cn('stat')}>
              <span className={cn('stat-value')}>21 of 21</span>
              <span className={cn('stat-label')}>runs with a higher Pearson correlation and a higher rank correlation with Hi-C for 3D-GNOMEᴺᴳ than for both other tools</span>
            </div>
            <div className={cn('stat')}>
              <span className={cn('stat-value')}>12 of 21</span>
              <span className={cn('stat-label')}>runs with a higher stratum adjusted correlation for 3D-GNOMEᴺᴳ than for MultiMM. The count against cudaMMC is 15 of 21. Neither tool is better than 3D-GNOMEᴺᴳ in any run.</span>
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
          cudaMMC has beads at both ends of about a quarter of the Micro-C loops and this limits the share it places. MultiMM has beads at every loop. It places as many loops as 3D-GNOMEᴺᴳ on GM12878 and fewer on the other two cell lines.
        </p>
        <p className={cn('reading-note')}>
          A region of 5 Mb has too few bins for the stratum adjusted correlation to separate the tools. On the nine runs of 20 Mb, 3D-GNOMEᴺᴳ is better than cudaMMC in all nine and better than MultiMM in seven. On chr1:1-60 Mb it is better than both tools in every cell line.
        </p>
      </section>
      <section id="runtime">
        <h2>Runtime</h2>
        <p className="intro">
          The three tools were timed on the same GM12878 inputs and on one computer with an RTX 3090 and 36 CPU cores. At its default settings cudaMMC uses a third to a half of the beads of the other two tools for the same region. Each chart therefore shows three numbers per tool, the number of beads in its model, its runtime and its runtime per 1,000 beads. A tick marks the lowest time in a column.
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
