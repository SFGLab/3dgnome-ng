import type { ReactNode } from 'react'
import { CitedText } from '../components/CitedText'
import { Equation, type EquationName } from '../components/Equation'
import { TermSketch } from '../components/TermSketch'
import { REFERENCES, referenceLink, referenceText } from '../content/references'
import engineBatching from '../assets/engine/engine_batching.svg'
import engineGraph from '../assets/engine/engine_graph.svg'
import methodAnchors from '../assets/method_anchors.png'
import methodChain from '../assets/method_chain.png'
import methodEnsemble from '../assets/method_ensemble.png'
import methodLaw from '../assets/method_law.png'
import '../styles/doc.css'

/** The Method page's references in the order of their numbers [1] to [17]. */
export const METHOD_REFERENCES = [
  'szalaj2016', 'cudammc', 'lieberman2009', 'shi2023', 'multimm', 'byrd1995', 'zhu1997', 'kirkpatrick1983',
  'metropolis1953', 'enhancer3d', 'jax2018', 'sadowski2019', 'hicrep2017', 'kremer1990', 'rosa2008', 'tang2015', 'mirny2011',
]

const SECTIONS = [
  ['inputs', '1. Inputs'],
  ['targets', '2. Blocks and target distances'],
  ['anchors', '3. Anchor phase'],
  ['chain', '4. Chain phase'],
  ['ensemble', '5. Ensemble and output'],
  ['engine', '6. Engine'],
  ['variants', '7. Structural variants'],
  ['validation', '8. Validation'],
  ['limitations', '9. Limitations'],
  ['references', 'References'],
] as const

const INPUTS = [
  ['Loops, one file per factor', 'BEDPE with PET counts', 'Loop springs between anchors. The PET count of a loop is compared with the typical count of loops of the same span and factor.'],
  ['Contact map', 'Hi-C or Micro-C, .mcool or contact pairs', 'The exponent of the distance law and the contact term for pairs without a loop.'],
  ['Genes and enhancers', 'GTF and BED', 'Promoter and enhancer marks and the distance table.'],
  ['Structural variants', 'VCF compressed with bgzip', 'Changes to the loops before modelling.'],
]

type Term = [string, EquationName, string, string]

const ANCHOR_TERMS: Term[] = [
  ['Loop spring', 't_loop_simple', 'anchor pairs joined by a loop, t = d(s, q)', '[1]'],
  ['Background spring', 't_background', 'pairs without a loop under 100 kb apart and far pairs with more Hi-C contacts than expected', 'this work'],
  ['Window size', 't_gyration', 'genomic windows with at least four anchors, n_w the number of anchors, R_g their radius of gyration and R_w the radius from the distance law', 'this work'],
  ['Excluded volume', 't_ev', 'anchors not next to each other', '[5, 14]'],
]

const CHAIN_TERMS: Term[] = [
  ['Bond', 't_bond', 'consecutive beads, t is the bond target', '[1]'],
  ['Bending', 't_bending', 'each pair of consecutive bonds, θ the angle between them', '[1]'],
  ['CTCF orientation', 't_orientation', 'anchors joined by a CTCF loop, o the direction of the motif', '[1]'],
  ['Excluded volume', 't_ev', 'beads not next to each other', '[5, 14]'],
  ['Confinement', 't_confinement', 'beads outside a sphere around the block centre', '[5, 15]'],
]

function Terms({ label, terms, phase }: { label: string; terms: Term[]; phase: 'anchor' | 'chain' }) {
  return (
    <table className="doc-table" aria-label={label}>
      <thead>
        <tr>
          <th>Term</th>
          <th>Before and after</th>
          <th>Energy of one pair or window</th>
          <th>Applies to</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody>
        {terms.map(([name, eq, applies, source]) => (
          <tr key={name}>
            <td>{name}</td>
            <td className="sketch-cell">
              <TermSketch phase={phase} name={name} />
            </td>
            <td>
              <Equation name={eq} />
            </td>
            <td>{applies}</td>
            <td>
              <CitedText>{source}</CitedText>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Section({ id, children }: { id: (typeof SECTIONS)[number][0]; children: ReactNode }) {
  const title = SECTIONS.find(([s]) => s === id)![1]
  return (
    <section id={id}>
      <h2>{title}</h2>
      {children}
    </section>
  )
}

function P({ children }: { children: string }) {
  return (
    <p>
      <CitedText>{children}</CitedText>
    </p>
  )
}

/** One job from its inputs to its output, with the energies and their sources. */
export function Method() {
  return (
    <div className="doc-page">
      <aside className="side-panel">
        <p className="eyebrow">On this page</p>
        {SECTIONS.map(([id, label]) => (
          <a key={id} href={`#${id}`}>
            {label}
          </a>
        ))}
      </aside>
      <div className="reading">
        <header className="page-head">
          <h1>Method</h1>
          <p className="lead">
            <CitedText>
              {'3D-GNOMEᴺᴳ turns chromatin loops and a contact map into an ensemble of 3D structures. It follows the multiscale approach of 3D-GNOME [1, 2]. The loop anchors are placed first and the chromatin between them second. This page follows one job from its inputs to its output.'}
            </CitedText>
          </p>
        </header>

        <Section id="inputs">
          <P>A job reads loops, a contact map and the positions of genes and enhancers. Structural variants are optional. The loops hold the structure together at their anchors. The contact map sets how distance grows with genomic separation and adds contacts between pairs without a loop.</P>
          <table className="doc-table" aria-label="Inputs">
            <thead>
              <tr>
                <th>Input</th>
                <th>Format</th>
                <th>Use</th>
              </tr>
            </thead>
            <tbody>
              {INPUTS.map(([input, format, use]) => (
                <tr key={input}>
                  <td>{input}</td>
                  <td>{format}</td>
                  <td>{use}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <Section id="targets">
          <P>{"Overlapping loops form a block, the computational form of the chromatin contact domains of Tang and colleagues [16] used by 3D-GNOME [1]. The region is split into blocks at points that no loop spans and stretches without loops have no beads. Every pair of anchors then gets a target distance from one law in bead units, one bead being the distance at 1 kb. A pair without a loop has the target b(s), with s the genomic separation, a power law as in polymer models of chromatin [3, 17]. A loop of strength q has the target d(s, q)."}</P>
          <div className="equations">
            <Equation name="background" />
            <Equation name="loop_target" />
          </div>
          <P>{"The exponent nu is fitted at load to the contact probability P(s) of the cell type's Hi-C, which falls as s to the power of minus 3 nu [3, 17]. If the fit fails, the run prints the reason and uses 0.285. The strength q of a loop is its PET count n over the typical count of loops of the same span and factor f. The target shrinks as the PET count grows, as in 3D-GNOME [1] and MultiMM [5]. It saturates at one bead for a strong loop, figure 1. A Hi-C cell converts to a distance the same way, b(s) times observed over expected contacts to the power of minus one third [3]."}</P>
          <div className="equations">
            <Equation name="contact_prob" />
          </div>
          <figure className="figure">
            <img src={methodLaw} alt="Target distance against genomic separation for pairs without a loop and for loops of increasing strength" />
            <figcaption>
              <b>Figure 1</b>Target distance against genomic separation, with nu = 0.34 as measured on GM12878 Hi-C.
            </figcaption>
          </figure>
        </Section>

        <Section id="anchors">
          <P>{'All anchors of a chromosome are placed in one fit. Each structure starts from a random draw of the anchor positions. The draw comes from a Gaussian model that is fitted once per run and its mean pair distances are as close to the target distances as a Gaussian model allows, as in the maximum entropy models of Shi and Thirumalai [4]. L-BFGS-B [6, 7] then minimises the anchor energy below, with L the anchor pairs joined by a loop, B the pairs held at the background and W a set of genomic windows. A loop spring uses k_s when the pair is further apart than its target t and k_q otherwise. The window term keeps each part of the chromosome at the size the distance law gives for its length. The windows start at 100 kb and double in size up to the whole chromosome and each size is placed twice, the second time shifted by half a window. The GPU evaluates the pair terms and their gradient, the optimiser runs on the CPU and the fit stops when the relative improvement per iteration drops below 1e-6.'}</P>
          <div className="equations">
            <Equation name="anchor_energy" />
          </div>
          <Terms label="Anchor energy terms" terms={ANCHOR_TERMS} phase="anchor" />
          <p className="panel-caption">Each drawing shows a few anchors in their order along the genome, before the term acts and after. Blue marks the anchors and the distance or radius the term reads.</p>
        </Section>

        <Section id="chain">
          <P>{"The beads between two anchors start on a compact random coil at the bond targets and the anchors stay where the anchor phase put them. Simulated annealing [8] moves one bead at a time and scores the chain energy below, with C the anchor pairs joined by a CTCF loop and o the direction of the CTCF motif at an anchor. A move that lowers the energy is accepted and a move that raises it is accepted with the probability below, where E' is the energy after the move. The temperature T falls by a constant factor each step [1, 9]. A wall rejects any move that brings two beads closer than the excluded volume radius. After the chains of all blocks are finished, a last pass moves the beads that still overlap, within a block or between two blocks. The pass keeps the anchors fixed."}</P>
          <div className="equations">
            <Equation name="chain_energy" />
            <Equation name="acceptance_plain" />
          </div>
          <Terms label="Chain energy terms" terms={CHAIN_TERMS} phase="chain" />
          <p className="panel-caption">Each drawing shows a short chain, before the term acts and after. Dark beads are anchors and the arrows of the orientation term are the directions of the CTCF motifs.</p>
          <div className="panels">
            {[
              [methodAnchors, 'a', 'Anchor positions after the fit. TNFRSF8 locus, GM12878, CTCF and RNAPII loops.'],
              [methodChain, 'b', 'The full chain of the same structure. Black dots are anchors.'],
              [methodEnsemble, 'c', 'Three structures of the ensemble, superposed on the first.'],
            ].map(([src, key, text]) => (
              <figure key={key} className="figure">
                <img src={src} alt={text} />
                <p className="panel-caption">
                  <b>{key}</b>
                  {text}
                </p>
              </figure>
            ))}
          </div>
          <p className="panel-caption">
            <b>Figure 2</b>a, anchor positions after the anchor phase, TNFRSF8 locus, GM12878, CTCF and RNAPII loops. b, the full chain of the same structure, black dots are anchors. c, three structures of the ensemble superposed on the first.
          </p>
        </Section>

        <Section id="ensemble">
          <P>{'The anchor and chain phases repeat from new random starts, one structure per start. Parts of the region with many loops are similar across structures and parts with few loops vary more. Each structure is saved as an mmCIF file. The job then computes, for every gene, the 3D distance from its promoter to its nearest enhancer within 1 Mb over all structures, as in enhancer3D [10].'}</P>
        </Section>

        <Section id="engine">
          <P>A run is a graph of tasks that grows while it runs, figure 3a. The region task adds one task per chromosome, a chromosome task adds its blocks and each chromosome gets one anchor fit and one chain task per block. Each stage has its own executor, serial, threaded or batched on the GPU. Results are applied in a fixed order and a run with the same settings and seed gives the same structures.</P>
          <P>{'The chain tasks of many blocks run together in JAX [11] launches on the GPU, figure 3b. Each block is padded to the next size on a fixed ladder and blocks with the same energy terms are packed into as few launches as GPU memory allows. Each chain has its own random stream and the grouping of chains into launches does not change the result of a chain. A chain that converges holds its state while the rest of the launch continues. Each annealing step tests 32 proposals against the current state and applies the first accepted one, which keeps the serial Metropolis rule. Excluded volume uses a grid of cells rebuilt every 2,048 proposals. With several GPUs each launch group goes whole to one GPU and the output does not depend on the number of GPUs.'}</P>
          <P>The chain tasks can also run on the CPU cores, one chain per thread, while the anchor fit stays on the GPU. This is faster on a computer with many CPU cores and the server runs its jobs this way.</P>
          <figure className="figure">
            <img src={engineGraph} alt="Task graph of a run on two chromosomes: a region task, one task per chromosome, its blocks, one anchor fit per chromosome on the CPU with the energy on the GPU, the chains of seven blocks in one batched launch and the assembly" />
            <figcaption>
              <b>Figure 3a</b>Task graph of a run on two chromosomes.
            </figcaption>
          </figure>
          <figure className="figure">
            <img src={engineBatching} alt="Nine blocks of 300 to 3600 beads, each padded to a ladder size of 512 to 4096, sorted largest first and packed into two launches padded to 4096 and 1024" />
            <figcaption>
              <b>Figure 3b</b>Bucketing and batching of chain tasks. Each block is padded up to the next size on a fixed ladder, which limits how many shapes the GPU compiles. Blocks with the same energy terms are sorted largest first and packed into as few launches as GPU memory allows and a launch pads its members to the size of the first.
            </figcaption>
          </figure>
        </Section>

        <Section id="variants">
          <P>{'Structural variants change the loops before the anchor phase. The approach comes from Sadowski and colleagues [12] and the rules below extend theirs. Deletions, insertions, inversions and tandem duplications are applied. An anchor that a variant leaves without a loop forms one with the nearest anchor in convergent orientation. The contacts of the Hi-C map move with the positions and both copies of a duplication get the contacts of the duplicated stretch.'}</P>
          <div className="cards two">
            {[
              ['Deletion', 'Anchors inside the deletion are removed with their loops. Positions after it shift by its length.'],
              ['Insertion', 'The inserted sequence is scanned for CTCF motifs. A motif above the score cutoff becomes an anchor and forms a loop with the nearest anchor in convergent orientation.'],
              ['Inversion', 'Anchors inside the inversion flip orientation and lose their loops. Each then forms a loop with the nearest anchor in convergent orientation.'],
              ['Duplication', 'The copy is placed right after the original. Anchors inside the duplication are copied with the loops among them. A loop to an anchor before it stays on the first copy and a loop to an anchor after it moves to the second. Positions after it shift by its length.'],
            ].map(([title, text]) => (
              <div key={title} className="card">
                <h3>{title}</h3>
                <p>{text}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section id="validation">
          <P>{'A change to the engine becomes the default only after it is measured on GM12878, H1ESC and HFFc6. The contacts of the structures of an ensemble are summed into one map and the measurements are the stratum adjusted correlation [13] and the rank correlation of that map with Hi-C at 25 kb, the exponent of distance against separation and the number of overlapping beads. For a cell line with a Micro-C map, loops are called on that map and each loop is checked in the models. The check compares the 3D distance between the two ends of the loop with the distance of pairs at the same genomic separation. A change meant to leave results unchanged is checked by comparing bead coordinates with the previous version. A test against the original 3D-GNOME program compares the distance distributions of both ensembles on a 2 Mb region. The Results page lists the measurements.'}</P>
        </Section>

        <Section id="limitations">
          <div className="cards">
            {[
              ['Population average', 'Loops and contacts are measured over millions of cells. A structure fits that average and is not the conformation of a single cell. Statistics over the ensemble are more reliable than any one structure.'],
              ['Bead units', 'Distances are in bead units. One unit is the target distance between beads 1 kb apart. Bead units are not nanometres.'],
              ['No loops, no beads', 'Stretches without loops have no beads. Genes there are missing from the distance table.'],
            ].map(([title, text]) => (
              <div key={title} className="card">
                <h3>{title}</h3>
                <p>{text}</p>
              </div>
            ))}
          </div>
        </Section>

        <Section id="references">
          <ol className="ref-list" aria-label="References">
            {METHOD_REFERENCES.map((key, i) => {
              const r = REFERENCES.find((x) => x.key === key)!
              const link = referenceLink(r)
              return (
                <li key={key} id={`ref-${i + 1}`}>
                  <span className="n">[{i + 1}]</span>
                  <span>
                    {referenceText(r).replace(/ (doi:\S+|https:\S+)$/, '')}{' '}
                    {link && <a href={link}>{r.doi ? `doi:${r.doi}` : link.replace(/^https:\/\//, '')}</a>}
                  </span>
                </li>
              )
            })}
          </ol>
        </Section>
      </div>
    </div>
  )
}
