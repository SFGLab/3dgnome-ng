import { Link } from 'react-router-dom'
import { BarChart, ChartKey, type Bar, type Group } from '../components/BarChart'
import { ROUTES } from '../routes'
import '../styles/doc.css'
import './results.css'

const bar = (series: Bar['series'], value: number | null, digits = 2): Bar => ({
  series,
  value,
  label: value === null ? 'placeholder' : value.toFixed(digits),
})

const SCC: Group[] = [
  { name: 'GM12878', bars: [bar('original', 0.22), bar('ctcf', 0.37)] },
  { name: 'H1ESC', bars: [bar('original', 0.15), bar('ctcf', 0.27)] },
  { name: 'HFFc6', bars: [bar('original', 0.15), bar('ctcf', 0.33)] },
]

const DISTANCE: Group[] = [
  { name: 'GM12878, 13% closer', bars: [bar('ctcf', 4.2, 1), bar('rnapii', 3.7, 1)] },
  { name: 'H1ESC, 24% closer', bars: [bar('ctcf', 4.0, 1), bar('rnapii', 3.0, 1)] },
  { name: 'HFFc6, 11% closer', bars: [bar('ctcf', 4.5, 1), bar('rnapii', 4.0, 1)] },
]

const EXPRESSION: Group[] = [
  { name: 'GM12878 and H1ESC', bars: [bar('original', 0.19), bar('ctcf', 0.26), bar('rnapii', 0.29)] },
  { name: 'H1ESC and HFFc6', bars: [bar('original', 0.12), bar('ctcf', 0.19), bar('rnapii', 0.2)] },
  { name: 'GM12878 and HFFc6', bars: [bar('original', 0.25), bar('ctcf', 0.29), bar('rnapii', 0.33)] },
]

const RUN_TIME = [
  ['One structure, 1 Mb', 'chr1:11.5-12.6 Mb, minutes, lower is better.'],
  ['Ten structures, 60 Mb', 'chr1:1-60 Mb, hours, lower is better.'],
  ['One structure, whole chromosome', 'chr1, 248 Mb, hours, lower is better.'],
]
const RUN_TIME_GROUPS: Group[] = [
  { name: 'cudaMMC', bars: [bar('original', null)] },
  { name: '3D-GNOMEᴺᴳ, CTCF loops', bars: [bar('ctcf', null)] },
]

const SECTIONS = [
  ['#agreement', 'Agreement with Hi-C'],
  ['#distance', 'Enhancer to promoter distance'],
  ['#expression', 'Link to expression'],
  ['#trios', 'Nine people, three trios'],
  [`${ROUTES.method}#limitations`, 'Limitations'],
  ['#run-time', 'Run time'],
]

/** The models against measured data, and the run time against cudaMMC. */
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
            This page compares the models with measured data and shows the effect of RNAPII loops. The data are GM12878, H1ESC and HFFc6, chr1:1-60 Mb, ten structures per model. The original models are the cudaMMC models of 3D-GNOME 3.0.
          </p>
        </header>
        <ChartKey />
        <div className="chart-row">
          <div id="agreement">
            <BarChart title="Agreement with Hi-C" subtitle="Stratum adjusted correlation at 25 kb, higher is better." groups={SCC} />
          </div>
          <div id="distance">
            <BarChart title="Enhancer to promoter distance" subtitle="Pairs joined by an RNAPII loop only, median over pairs, bead units." groups={DISTANCE} />
          </div>
        </div>
        <div className="chart-row">
          <div id="expression">
            <BarChart
              title="Link to expression"
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
        <section id="run-time">
          <h2>Run time</h2>
          <p className="intro">
            cudaMMC and 3D-GNOMEᴺᴳ on the same GM12878 inputs and the same RTX 4060 Ti. The values are placeholders until the timing run is done.
          </p>
          <div className="chart-row three">
            {RUN_TIME.map(([title, sub]) => (
              <BarChart key={title} title={title} subtitle={sub} groups={RUN_TIME_GROUPS} />
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
