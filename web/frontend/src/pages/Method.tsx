import type { ReactNode } from 'react'
import { CitedText } from '../components/CitedText'
import { Equation, type EquationName } from '../components/Equation'
import { TermSketch } from '../components/TermSketch'
import { REFERENCES, referenceLink, referenceText } from '../content/references'
import engineBatching from '../assets/engine/engine_batching.svg'
import engineGraph from '../assets/engine/engine_graph.svg'
import methodAnchorTerms from '../assets/method_anchor_terms.png'
import methodAnchors from '../assets/method_anchors.png'
import methodBlocks from '../assets/method_blocks.png'
import methodChain from '../assets/method_chain.png'
import methodDefault from '../assets/method_default.png'
import methodEnsemble from '../assets/method_ensemble.png'
import methodExponent from '../assets/method_exponent.png'
import methodHic from '../assets/method_hic.png'
import methodLoops from '../assets/method_loops.png'
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
  ['CTCF orientation', 't_orientation', 'anchors joined by a CTCF loop, o the direction of the motif in space', '[1]'],
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

function Figure({ n, src, alt, children }: { n: string; src: string; alt: string; children: string }) {
  return (
    <figure className="figure">
      <img src={src} alt={alt} />
      <figcaption>
        <b>Figure {n}</b>
        {children}
      </figcaption>
    </figure>
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
          <P>{"A job has to produce a 3D position for every bead, yet its inputs contain no positions, only evidence on which parts of the chromosome are close to each other. A loop records that its two anchors were captured together in a ChIA-PET experiment and the Hi-C map counts the contacts between every two stretches of the genome."}</P>
          <P>{"The model goes from this evidence to positions in two steps. It first gives each pair of loop anchors a target distance, the distance the two anchors should have in the finished structure. The target is short for a pair joined by a strong loop and equal to the distance expected from the genomic separation for a pair with no sign of contact. The anchor phase and the chain phase, sections 3 and 4, then move the beads until the distances in the structure are as close to these targets as they can be."}</P>
          <P>{"This section describes the first step. The region is first split into blocks, the parts that loops cover and that the model can therefore build. Three rules then set the targets, one for a pair without a loop, one for a loop and one for a pair with many Hi-C contacts. All distances are in bead units, where one unit is the mean distance between two points of the chromatin fibre 1 kb apart."}</P>

          <h3>Blocks</h3>
          <P>{"The loops decide which parts of the region the model can build. A loop joins two anchors and passes over the chromatin between them. Loops that overlap are grouped into a block, which ends at a position that no loop passes over, figure 1. Blocks are the computational form of the chromatin contact domains of Tang and colleagues [16] and are used here in the same way as in 3D-GNOME [1]. Each block becomes one chain of beads, whereas a stretch between two blocks has no loops, carries no information on its folding and therefore gets no beads."}</P>
          <Figure n="1" src={methodBlocks} alt="Loops of a 6 Mb region drawn as arcs, the number of loops over each position and the four blocks the region is split into at the positions without loops">
            The split of a region into blocks. CTCF loops of GM12878 on chromosome 1, 236.0 to 242.1 Mb. Top, every arc is one loop and a thicker arc has a higher PET count. Middle, the number of loops that pass over each position. Bottom, the four blocks with their anchors as white lines. A block ends at a position where the number of loops falls to zero. The grey stretches have no loops and get no beads.
          </Figure>

          <h3>Distance of a pair without a loop</h3>
          <P>{"Most pairs of anchors have no loop between them, yet such a pair still needs a target, the distance expected for two points of one chromosome that have no specific contact. Two points that are further apart along the genome are on average further apart in space. The model expresses this as a background distance b(s), a power law of the genomic separation s, as in polymer models of chromatin [3, 17]."}</P>
          <div className="equations">
            <Equation name="background" />
          </div>
          <P>{"The exponent ν sets how fast the distance grows with separation. A chain that folds as a random walk has ν = 1/2, while a chain packed as a dense globule has ν = 1/3 [3, 17]. Rather than assume a value, the model measures ν on the Hi-C map of the cell type. Two points of a polymer meet less often when the chain between them fills a larger volume and this volume grows as the cube of the distance between them. The contact probability P(s) therefore falls as s to the power of minus 3ν [3, 17]."}</P>
          <div className="equations">
            <Equation name="contact_prob" />
          </div>
          <P>{"At the start of a job the Hi-C contacts are counted at each separation from 20 kb up to 1 Mb, the range in which nearly all loops fall. The lower limit is two bins of the map if that is more than 20 kb. A straight line fitted to these counts on logarithmic axes has a slope of minus 3ν, figure 2a. With ν known, the law gives the background distance of any pair, figure 2b."}</P>
          <Figure n="2" src={methodExponent} alt="Hi-C contacts of GM12878 against genomic separation with a line fitted between 50 kb and 1 Mb, and the background distance against separation that the fitted exponent gives">
            The fit of the exponent. a, Hi-C contacts of GM12878 on chromosome 1 against genomic separation, in 25 kb bins. The contacts at a separation are divided by the number of bin pairs the chromosome has at that separation. The line is fitted to the black points of the shaded range and gives ν = 0.34. b, the background distance for this exponent. Two anchors 100 kb apart have a target of 4.9 bead units.
          </Figure>

          <h3>The default exponent</h3>
          <P>{"The fit needs enough contacts. If the map has too few of them in this range, or if the counts do not fall with separation as those of a polymer do, the job prints the reason and uses ν = 0.285 instead. This default is itself a measurement, the mean of the exponents of three cell lines, GM12878, H1-hESC and HFFc6. Each was fitted between 20 kb and 1 Mb on a Hi-C map of the 4DN data portal, figure 3. The three values, 0.309, 0.269 and 0.277, are close to each other and a little below 1/3, the value of a dense globule [3]. The exponent of a particular job can differ from them by a few hundredths, depending on the map, its bin size and the chromosome."}</P>
          <Figure n="3" src={methodDefault} alt="Contact probability against genomic separation for GM12878, H1-hESC and HFFc6 with a line fitted to each between 20 kb and 1 Mb, and the three exponents 0.309, 0.269 and 0.277 with their mean 0.285">
            The origin of the default exponent. a, contact probability against genomic separation for three cell lines, from the Hi-C maps 4DNFIQ32RWCQ, 4DNFIHO3CXUQ and 4DNFIDKNBPC3 of the 4DN data portal, chromosomes 1, 2, 5, 11 and 17 in 5 kb bins. The curves are shifted apart. Each line is fitted in the shaded range, 20 kb to 1 Mb. b, the exponent of each cell line. The default, 0.285, is their mean. The dashed line is 1/3.
          </Figure>

          <h3>Distance of a loop</h3>
          <P>{"The two anchors of a loop were captured together by ChIA-PET many times, which is evidence that they are closer in space than a pair without a loop at the same separation. The target of a loop is therefore shorter than the background, the more so the stronger the loop. The natural measure of strength is the PET count n, the number of sequenced read pairs that support the loop. The count alone is not a fair measure, however, since long loops have lower counts than short loops and each factor has its own range of counts. The model therefore divides the count of a loop by the typical count of loops of the same span and the same factor f, figure 4a. This ratio is the strength q of the loop. The typical count is a power law of the span, fitted through the median counts of the loops of the job."}</P>
          <div className="equations">
            <Equation name="loop_target" />
          </div>
          <P>{"A loop with q = 1 is a typical loop of its span, with a target halfway between the background and one bead. A weak loop stays near the background, whereas a very strong loop approaches one bead, the distance at which its two anchors touch, figure 4b. The target thus shrinks as the PET count grows, as it does in 3D-GNOME [1] and MultiMM [5]."}</P>
          <Figure n="4" src={methodLoops} alt="PET count against span for the CTCF loops of GM12878 on chromosome 1 with the typical count as a line, and target distance against genomic separation for pairs without a loop and for loops of increasing strength">
            The strength of a loop and its target. a, PET count against span for the 12,474 CTCF loops of GM12878 on chromosome 1. The black line is the typical count, q = 1. A loop on the dashed line has four times the typical count, q = 4. b, target distance against genomic separation for a pair without a loop and for loops of three strengths, with ν = 0.34.
          </Figure>

          <h3>Distance from the Hi-C map</h3>
          <P>{"Two anchors without a loop can still be close in space, in which case the Hi-C map shows the pair as a cell with more contacts than other cells at the same separation. Each cell has an observed count O, which the model compares with the expected count E(s), the mean count of all cells at the same separation s. The rule behind the exponent fit, that contacts fall as the cube of the distance, then turns the ratio of the two counts into the target t(s) of the pair."}</P>
          <div className="equations">
            <Equation name="hic_target" />
          </div>
          <P>{"A cell with the expected count keeps the background distance, a cell with twice the expected count has 0.79 of it and a cell with eight times the expected count has half of it, figure 5 [3]. Section 3 lists the pairs that get each kind of target and describes how the anchors are placed to meet them."}</P>
          <Figure n="5" src={methodHic} alt="Target distance over background distance against observed over expected Hi-C contacts, and target distance against genomic separation for cells with 1, 2, 4 and 8 times the expected count">
            The target of a Hi-C cell. a, the target as a share of the background distance, against observed over expected contacts. b, target distance against genomic separation for cells with the expected count and with 2, 4 and 8 times the expected count, with ν = 0.34.
          </Figure>
        </Section>

        <Section id="anchors">
          <P>{"The anchor phase places the loop anchors, the points of the chromosome on which the data are most informative. The targets of section 2 cover only a small share of the anchor pairs, however. The region of figure 1 has 53 anchors and 1,378 pairs of anchors. Of these, 46 are joined by a loop and 55 are closer than 100 kb and get the background distance, while the other 1,277 have no target of their own, figure 6a. A structure built from the targets alone would hold its loops and leave nearly everything else to chance."}</P>
          <P>{"The anchor energy therefore combines four terms. Two come from earlier models and two are new in this work, the background spring and the window term. In the energy below L is the set of anchor pairs joined by a loop, B the pairs held at the background and W a set of genomic windows."}</P>
          <div className="equations">
            <Equation name="anchor_energy" />
          </div>

          <h3>The four terms</h3>
          <P>{"The loop springs pull the two anchors of every loop to the loop target, as in 3D-GNOME [1]. A spring uses the constant k_s when the pair is further apart than its target t and k_q otherwise. On their own the loop springs give a structure that is too compact below 100 kb, since they close every loop while nothing keeps the anchors between two loops apart."}</P>
          <P>{"The background spring is the first new term. It holds every pair of anchors closer than 100 kb at the background distance b(s), which keeps the anchors between two loops at the spacing of a chain without contacts. The limit of 100 kb is deliberate. The law gives the mean distance of a pair over many structures and one compact structure cannot have every pair at its mean at once. In a test with the spring on every pair the fit returned a compressed average of all targets, with half the radius of gyration and a lower agreement with Hi-C on three cell lines. A far pair with more Hi-C contacts than expected gets the same spring at its Hi-C target. Such pairs are rare and the region of figure 6 has none."}</P>
          <P>{"The window term is the second new term. It sets the size of the structure at every scale without holding a single pair. For a genomic window it takes the radius of gyration R_g of the anchors inside, which is the root mean square distance of these anchors from their centre. It compares this radius with the radius R_w of a chain that follows the distance law over the same span. The windows start at 100 kb and double in size up to the whole chromosome. Each size is placed twice, the second time shifted by half a window, which puts every border of one placement in the middle of a window of the other, figure 6b. A window with fewer than four anchors is left out."}</P>
          <P>{"Inside a window the anchors stay free to arrange themselves, which lets two distant anchors meet in one structure and stay apart in the next. An earlier version kept distant pairs apart with a repulsion between them. Every far pair then had the same distance in every structure and the ensemble had two to ten times fewer contacts than Hi-C between 100 kb and 1 Mb."}</P>
          <P>{"The excluded volume term keeps two anchors from taking the same place, as in polymer models [5, 14]. The loop springs have a weight of 1 and the three other terms a weight of 0.1. A loop therefore wins wherever it disagrees with another term, while the weaker terms shape everything the loops leave open."}</P>
          <Terms label="Anchor energy terms" terms={ANCHOR_TERMS} phase="anchor" />
          <p className="panel-caption">Each drawing shows a few anchors in their order along the genome, before the term acts and after. Blue marks the anchors and the distance or radius the term reads.</p>
          <Figure n="6" src={methodAnchorTerms} alt="A matrix of the anchor pairs of a 6 Mb region coloured by the kind of target, with loops and background pairs near the diagonal and most pairs without a target, and the genomic windows of the window term from 100 kb to 3.2 Mb drawn over the anchors">
            The terms of the anchor energy on the region of figure 1. a, every pair of the 53 anchors, coloured by its target. Loops and background pairs are close to the diagonal and no pair of two blocks has a target. b, the windows of the window term. Each size has two rows, the second shifted by half a window. A filled window has four or more anchors and enters the energy, an empty one is left out.
          </Figure>

          <h3>The start and the fit</h3>
          <P>{"The anchor energy has many structures of nearly the same value. They differ in the arrangement of the far anchors, which no term holds and an ensemble should contain this variety. Each structure therefore starts from its own random draw of all anchor positions. The draw comes from a Gaussian model that is fitted once per job, with mean pair distances as close to the targets as a Gaussian model allows, as in the maximum entropy models of Shi and Thirumalai [4]."}</P>
          <P>{"L-BFGS-B [6, 7] then lowers the anchor energy from the draw and stops when the relative improvement per iteration falls below 1e-6. The draw decides the arrangement of the distant parts of the chromosome in each structure, while the fit brings the loops and the sizes to their targets. All anchors of a chromosome are fitted together, which lets the windows and the Hi-C targets act across the borders of blocks. The GPU evaluates the pair terms and their gradient and the optimiser runs on the CPU."}</P>
          <P>{"A gradient method is enough for this energy. In tests on real blocks, simulated annealing from ten different starts ended within one percent of the same energy, a sign that the energy has no separate minima for a random search to escape. The gradient method reaches that energy in seconds where annealing took minutes."}</P>
        </Section>

        <Section id="chain">
          <P>{"After the anchor phase only the anchors have positions. The chromatin between two neighbouring anchors is filled with beads, about one per kilobase, none of which is the end of a loop. The chain phase therefore treats each stretch between two anchors as a piece of polymer and asks of it what holds for any chromatin fibre. The fibre is connected, it does not bend sharply, it does not pass through itself and it stays within the volume of its block. One further term uses data, the direction of the CTCF motifs at the anchors. The anchors do not move in this phase, which keeps every loop as the anchor phase placed it."}</P>
          <P>{"In the chain energy below C is the set of anchor pairs joined by a CTCF loop and o the direction of the CTCF motif at an anchor."}</P>
          <div className="equations">
            <Equation name="chain_energy" />
          </div>

          <h3>The five terms</h3>
          <P>{"The bond term keeps consecutive beads at the bond target and the bending term straightens sharp turns, both as in 3D-GNOME [1]. The orientation term acts on the two anchors of a CTCF loop. It turns the chain at each anchor until the two motifs point the same way in space, which gives a hairpin for motifs that face each other along the genome [1]. The excluded volume term pushes apart two beads that are closer than the radius r_0 and the confinement term pulls back a bead that leaves a sphere around the centre of its block [5, 14, 15]."}</P>
          <Terms label="Chain energy terms" terms={CHAIN_TERMS} phase="chain" />
          <p className="panel-caption">Each drawing shows a short chain, before the term acts and after. Dark beads are anchors. The drawing of the orientation term follows 3D-GNOME [1] and shows two loops. On top are the directions of the two CTCF motifs along the genome and below is the loop that brings the two motifs side by side pointing the same way. Motifs that face each other give a hairpin and motifs of one direction give a loop of a full turn.</p>

          <h3>The start and the wall</h3>
          <P>{"Overlapping beads were the main fault of the first chains and two additions of this work remove them. The first is a wall. The excluded volume term is soft and its push fades to zero as two beads approach the distance r_0, which leaves most overlaps in place near that distance. The wall rejects every move that creates an overlap or deepens an existing one, after which the number of overlaps can only fall."}</P>
          <P>{"The second addition is the start. A chain that starts on a straight line between its anchors begins with thousands of overlaps and has to push its way out of them. Each stretch now starts as a compact random coil between its two anchors, with its bonds already at their targets. On a test block the soft term alone ended with 766 overlapping pairs per thousand beads, the wall alone with 92, the coil start alone with 487 and the two together with 43. On three cell lines the two together cut the overlaps within blocks five to seven times and left the size of the structures unchanged."}</P>

          <h3>Annealing</h3>
          <P>{"Simulated annealing [8] moves one bead at a time and scores the chain energy. A move that lowers the energy is accepted. A move that raises it is accepted with the probability below, where E' is the energy after the move. The temperature T falls by a constant factor at each step [1, 9], which makes uphill moves rarer as the chain settles."}</P>
          <div className="equations">
            <Equation name="acceptance_plain" />
          </div>

          <h3>The last pass</h3>
          <P>{"Each block is annealed on its own, which leaves nothing to keep the beads of one block out of another. A last pass therefore looks at the whole chromosome once all chains are finished. It finds the beads that overlap a bead of another block or of their own block and moves only these beads and their two neighbours along the chain, with every anchor fixed. On chromosome 1 of GM12878 the pass lowers the overlaps between blocks from 62 to 14 per thousand beads and the overlaps within blocks from 246 to 37."}</P>
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
            <b>Figure 7</b>a, anchor positions after the anchor phase, TNFRSF8 locus, GM12878, CTCF and RNAPII loops. b, the full chain of the same structure, black dots are anchors. c, three structures of the ensemble superposed on the first.
          </p>
        </Section>

        <Section id="ensemble">
          <P>{'The anchor and chain phases repeat from new random starts, one structure per start. Parts of the region with many loops are similar across structures and parts with few loops vary more. Each structure is saved as an mmCIF file. The job then computes, for every gene, the 3D distance from its promoter to its nearest enhancer within 1 Mb over all structures, as in enhancer3D [10].'}</P>
        </Section>

        <Section id="engine">
          <P>A run is a graph of tasks that grows while it runs, figure 8a. The region task adds one task per chromosome, a chromosome task adds its blocks and each chromosome gets one anchor fit and one chain task per block. Each stage has its own executor, serial, threaded or batched on the GPU. Results are applied in a fixed order and a run with the same settings and seed gives the same structures.</P>
          <P>{'The chain tasks of many blocks run together in JAX [11] launches on the GPU, figure 8b. Each block is padded to the next size on a fixed ladder and blocks with the same energy terms are packed into as few launches as GPU memory allows. Each chain has its own random stream and the grouping of chains into launches does not change the result of a chain. A chain that converges holds its state while the rest of the launch continues. Each annealing step tests 32 proposals against the current state and applies the first accepted one, which keeps the serial Metropolis rule. Excluded volume uses a grid of cells rebuilt every 2,048 proposals. With several GPUs each launch group goes whole to one GPU and the output does not depend on the number of GPUs.'}</P>
          <P>The chain tasks can also run on the CPU cores, one chain per thread, while the anchor fit stays on the GPU. This is faster on a computer with many CPU cores and the server runs its jobs this way.</P>
          <figure className="figure">
            <img src={engineGraph} alt="Task graph of a run on two chromosomes: a region task, one task per chromosome, its blocks, one anchor fit per chromosome on the CPU with the energy on the GPU, the chains of seven blocks in one batched launch and the assembly" />
            <figcaption>
              <b>Figure 8a</b>Task graph of a run on two chromosomes.
            </figcaption>
          </figure>
          <figure className="figure padded">
            <img src={engineBatching} alt="Nine blocks of 300 to 3600 beads, each padded to a ladder size of 512 to 4096, sorted largest first and packed into two launches padded to 4096 and 1024" />
            <figcaption>
              <b>Figure 8b</b>Bucketing and batching of chain tasks. Each block is padded up to the next size on a fixed ladder, which limits how many shapes the GPU compiles. Blocks with the same energy terms are sorted largest first and packed into as few launches as GPU memory allows and a launch pads its members to the size of the first.
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
