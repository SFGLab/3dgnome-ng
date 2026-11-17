import { Link } from 'react-router-dom'
import { BarChart, ChartKey, type Bar, type Group } from '../components/BarChart'
import { LineChart, type Line, type Point } from '../components/LineChart'
import { ToolRuns, type ToolRun } from '../components/ToolRuns'
import { ROUTES } from '../routes'
import '../styles/doc.css'
import './results.css'

const bar = (series: Bar['series'], value: number | null, digits = 2): Bar => ({
  series,
  value,
  label: value === null ? 'placeholder' : value.toFixed(digits),
})

const SCC: Group[] = [
  { name: 'GM12878', bars: [bar('original', 0.22), bar('ctcf', 0.5)] },
  { name: 'H1ESC', bars: [bar('original', 0.15), bar('ctcf', 0.36)] },
  { name: 'HFFc6', bars: [bar('original', 0.15), bar('ctcf', 0.43)] },
]

const DISTANCE: Group[] = [
  { name: 'GM12878, 16% closer', bars: [bar('ctcf', 4.3, 1), bar('rnapii', 3.6, 1)] },
  { name: 'H1ESC, 24% closer', bars: [bar('ctcf', 4.0, 1), bar('rnapii', 3.0, 1)] },
  { name: 'HFFc6, 17% closer', bars: [bar('ctcf', 4.6, 1), bar('rnapii', 3.8, 1)] },
]

const EXPRESSION: Group[] = [
  { name: 'GM12878 and H1ESC', bars: [bar('original', 0.19), bar('ctcf', 0.26), bar('rnapii', 0.28)] },
  { name: 'H1ESC and HFFc6', bars: [bar('original', 0.12), bar('ctcf', 0.18), bar('rnapii', 0.22)] },
  { name: 'GM12878 and HFFc6', bars: [bar('original', 0.25), bar('ctcf', 0.29), bar('rnapii', 0.31)] },
]

/** The runs of the three tools on one task, each as beads and seconds. A null time is a run that did not finish. */
const runs = (cudammc: [number, number], multimm: [number, number | null], ours: [number, number], note?: string): ToolRun[] => [
  { series: 'cudammc', name: 'cudaMMC', beads: cudammc[0], seconds: cudammc[1] },
  { series: 'multimm', name: 'MultiMM', beads: multimm[0], seconds: multimm[1], note },
  { series: 'ctcf', name: '3D-GNOMEᴺᴳ', beads: ours[0], seconds: ours[1] },
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
  ['Three structures at equal bead counts', 'chr1, 248 Mb. cudaMMC is set to as many beads as the other two.', 'minutes', runs([126330, 641], [124657, 2528], [124657, 673])],
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

const SECTIONS = [
  ['#agreement', 'Agreement with Hi-C'],
  ['#distance', 'Enhancer to promoter distance'],
  ['#expression', 'Link to expression'],
  ['#trios', 'Nine people, three trios'],
  ['#tools', 'Other tools'],
  [`${ROUTES.method}#limitations`, 'Limitations'],
  ['#run-time', 'Run time'],
]

/** The models against measured data and against cudaMMC and MultiMM, and the run time of the three. */
export function Results() {
  return (
    <div className="doc-page">
      <aside className="side-panel">
        <p className="eyebrow">On this page</p>
        {SECTIONS.map(([to, label]) =>
          to.startsWith('#') ? (
            <a key={to} href={to}>
              {label}
            </a>
          ) : (
            <Link key={to} to={to}>
              {label}
            </Link>
          ),
        )}
      </aside>
      <div className="reading wide">
        <header className="page-head">
          <h1>Results</h1>
          <p className="lead">
            This page compares the models with measured data and with two other tools, and shows the effect of RNAPII loops. The data are GM12878, H1ESC and HFFc6, chr1:1-60 Mb, ten structures per model. The original models are the cudaMMC models of 3D-GNOME 3.0.
          </p>
        </header>
        <ChartKey />
        <div className="chart-row">
          <div id="agreement">
            <BarChart title="Agreement with Hi-C" subtitle="Stratum adjusted correlation at 25 kb. The contacts of ten structures are summed before the comparison." better="higher" groups={SCC} />
          </div>
          <div id="distance">
            <BarChart title="Enhancer to promoter distance" subtitle="Pairs joined by an RNAPII loop only, median over pairs, bead units." groups={DISTANCE} />
          </div>
        </div>
        <div className="chart-row">
          <div id="expression">
            <BarChart
              title="Link to expression"
              better="higher"
              subtitle="Spearman correlation between the change in 3D distance and the change in expression across cell lines, with the sign flipped. Higher means a stronger link."
              groups={EXPRESSION}
            />
          </div>
          <figure id="trios" className="bar-chart" aria-label="Nine people, three trios">
            <figcaption>
              <span className="chart-title">Nine people, three trios</span>
              <span className="chart-sub">Change with RNAPII loops, measured against the Hi-C of each person.</span>
            </figcaption>
            <div className="stat">
              <span className="stat-value">9 of 9</span>
              <span className="stat-label">contact enrichment at RNAPII loops rises, from 39 to 98%, median 57%</span>
            </div>
            <div className="stat">
              <span className="stat-value">2 of 9</span>
              <span className="stat-label">correlation with the whole Hi-C map rises, from -17 to +29%, median -2%</span>
            </div>
          </figure>
        </div>
        <p className="reading-note">
          With RNAPII loops, enhancers are closer to their genes and the link to expression is stronger. Agreement with the whole Hi-C map is lower.
        </p>
        <section id="tools">
          <h2>Other tools</h2>
          <p className="intro">
            cudaMMC, MultiMM and 3D-GNOMEᴺᴳ on the same CTCF loops and the same Hi-C contacts. cudaMMC and MultiMM run at their default settings, and MultiMM models as many beads as 3D-GNOMEᴺᴳ.
          </p>
          <div className="chart-row">
            <div>
              <BarChart
                title="Agreement with Hi-C, three tools"
                subtitle="chr1:1-60 Mb, stratum adjusted correlation at 25 kb. The contacts of ten structures are summed before the comparison."
                better="higher"
                groups={TOOLS_SCC}
              />
            </div>
            <figure className="bar-chart" aria-label="Twenty one runs with intervals">
              <figcaption>
                <span className="chart-title">Twenty one runs with intervals</span>
                <span className="chart-sub">Three cell lines, three regions of 5 Mb, three of 20 Mb and chr1:1-60 Mb. A lead has a 95% bootstrap interval clear of zero.</span>
              </figcaption>
              <div className="stat">
                <span className="stat-value">21 of 21</span>
                <span className="stat-label">3D-GNOMEᴺᴳ leads both tools on the Pearson and on the rank correlation with Hi-C</span>
              </div>
              <div className="stat">
                <span className="stat-value">12 of 21</span>
                <span className="stat-label">3D-GNOMEᴺᴳ leads MultiMM on the stratum adjusted correlation, 15 of 21 for cudaMMC, and neither tool leads it in any run</span>
              </div>
            </figure>
          </div>
          <p className="reading-note">
            A region of 5 Mb is too short for the stratum adjusted correlation to separate the tools. On the nine runs of 20 Mb 3D-GNOMEᴺᴳ leads cudaMMC on all and MultiMM on seven. On chr1:1-60 Mb it leads both on every cell line.
          </p>
        </section>
        <section id="run-time">
          <h2>Run time</h2>
          <p className="intro">
            The three tools on the same GM12878 inputs and one machine, an RTX 3090 with 36 CPU cores. At its default settings cudaMMC models a region with a third to a half of the beads of the other two. Each chart therefore gives three numbers per tool. They are the beads in its model, its run time and its run time per 1,000 beads. A tick marks the lowest time in a column.
          </p>
          <div className="chart-row one">
            {RUN_TIME.map(([title, sub, unit, toolRuns]) => (
              <ToolRuns key={title} title={title} subtitle={sub} unit={unit} runs={toolRuns} />
            ))}
          </div>
          <p className="reading-note">
            Per 1,000 beads 3D-GNOMEᴺᴳ is the fastest on ten structures of 60 Mb, on one structure of the chromosome and on the genome. MultiMM is the fastest at 1 Mb, and at equal bead counts cudaMMC is 5 percent ahead of 3D-GNOMEᴺᴳ.
          </p>
          <p className="reading-note">
            On the whole genome 3D-GNOMEᴺᴳ runs as one job. cudaMMC ran one chromosome at a time and its 22 runs are added up. MultiMM ran out of memory on the 24 GB card at 1,312,597 beads and at half of that, and at a quarter it was stopped after 90 minutes.
          </p>
          <h3 className="sub-head">Run time against the size of the task</h3>
          <p className="reading-note">
            Each tool built one structure of eight regions of chr1, from 1 Mb to the whole chromosome of 248 Mb, and each point is one of those runs. The two charts hold the same runs. The left chart places a run by the size of its region, and it reads as the wait for a region of that size. The right chart places the same run by the number of beads in its model, and it reads as the cost of a model of that size. A tool with fewer beads per megabase sits lower on the left only.
          </p>
          <div className="chart-row">
            <LineChart
              title="One structure by region size"
              subtitle="Regions of 1 to 100 Mb starting at chr1:10 Mb, and the whole chromosome. Both axes are logarithmic."
              better="lower"
              xLabel="region, Mb"
              xTick={(x) => `${x}`}
              lines={BY_REGION}
            />
            <LineChart
              title="One structure by bead count"
              subtitle="The same runs placed by the bead count of each model. Both axes are logarithmic."
              better="lower"
              xLabel="beads"
              xTick={(x) => x.toLocaleString('en-US')}
              lines={BY_BEADS}
            />
          </div>
          <p className="reading-note">
            By region size cudaMMC is the fastest from 10 Mb, and under that it and MultiMM are level. 3D-GNOMEᴺᴳ is faster than MultiMM from about 50 Mb. By bead count MultiMM is the fastest up to about 30,000 beads and 3D-GNOMEᴺᴳ above that, and cudaMMC is level with 3D-GNOMEᴺᴳ up to about 7,000 beads and slower above. A further structure of the same region costs 3D-GNOMEᴺᴳ less than the first. One structure of chr1 takes 6.4 minutes and three take 11.2.
          </p>
        </section>
      </div>
    </div>
  )
}
