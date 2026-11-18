import { BookOpen } from 'lucide-react'
import '../styles/doc.css'
import './help.css'

const FORMATS = [
  ['Loop anchors, BED', 'chr  start  end  orientation', 'Orientation of the CTCF motif, L, R or N. Use N when it is unknown and for RNAPII anchors. Give the orientation on every line or on none. The engine reads the first line to decide whether the file has it.'],
  ['Loops, BEDPE', 'chr1  start1  end1  chr2  start2  end2  count', 'One file per factor, CTCF first. The count is the number of PETs. Loops between chromosomes are skipped, and a loop longer than 1 Mb goes to the coarse contact map.'],
  ['Contacts, BEDPE', 'chr1  start1  end1  chr2  start2  end2  count', 'Hi-C bin pairs or ChIA-PET singletons, with an integer count. The start column gives the position of each end. An uploaded .hic or .mcool map is converted to this format.'],
  ['Genes, GTF', 'standard GTF', 'Gene records give the gene name and the TSS, taken from the strand.'],
  ['Enhancers, BED', 'chr  start  end', 'One enhancer per line, for example from EnhancerAtlas.'],
  ['Segment split, BED', 'chr  position  position', 'Optional. Positions where the coarse segments split.'],
  ['Centromeres, BED', 'chr  start  end', 'One line per chromosome.'],
  ['Structural variants, VCF', 'standard VCF, bgzip', 'See the next section.'],
]

const EXAMPLES = [
  ['anchors.bed', 'chr1  12062800  12064300  R\nchr1  12447900  12451100  L'],
  ['ctcf_loops.bedpe', 'chr1  12062800  12064300  chr1  12447900  12451100  7'],
  ['contacts.bedpe', 'chr1  12050000  12075000  chr1  12425000  12450000  41'],
]

const VCF = `#CHROM  POS       ID    REF  ALT    QUAL  FILTER  INFO                       FORMAT  HG00733
chr1    12101200  del1  N    <DEL>  .     PASS    SVTYPE=DEL;END=12118400    GT      0|1
chr1    12255010  ins1  N    <INS>  .     PASS    SVTYPE=INS;SVLEN=310       GT      1|1
chr1    12390500  inv1  N    <INV>  .     PASS    SVTYPE=INV;END=12402750    GT      1|0`

const SETTINGS = [
  ['target_bp_per_subanchor', '5000', '1000', 'genomic length of one chain segment, in bp'],
  ['polymer_exponent', '0', '0', 'distance law exponent, 0 fits it from the Hi-C'],
  ['background_weight', '0', '0.1', 'spring on pairs without a loop, under 100 kb apart'],
  ['solver', 'mc', 'lbfgs', 'method for the anchor fit'],
  ['hard_wall', 'no', 'yes', 'stops chains passing through each other'],
]

const SECTIONS = [
  ['formats', 'Input formats'],
  ['variants', 'Structural variants'],
  ['reading', 'Reading the results'],
  ['questions', 'Questions'],
  ['running', 'Running gnome3d-ng yourself'],
]

function QA({ q, a }: { q: string; a: string }) {
  return (
    <div className="qa">
      <h3>{q}</h3>
      <p>{a}</p>
    </div>
  )
}

/** Input formats, reading the results and running the engine yourself. */
export function Help() {
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
          <h1>Help</h1>
          <p className="lead">This page covers your own data, the results and running the engine on your own computer.</p>
        </header>

        <section id="formats">
          <h2>Input formats</h2>
          <p className="intro">
            Input files are plain text with one record per line, separated by tabs or spaces, with positions in base pairs on GRCh38. Each cell type on the server has a full set, and an upload replaces one file of the set.
          </p>
          <table className="doc-table" aria-label="Input formats">
            <thead>
              <tr>
                <th>File</th>
                <th>Columns</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {FORMATS.map(([file, columns, notes]) => (
                <tr key={file}>
                  <td>{file}</td>
                  <td>
                    <code>{columns}</code>
                  </td>
                  <td>{notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3 className="example-head">Example, one CTCF loop and its anchors</h3>
          {EXAMPLES.map(([name, text]) => (
            <div key={name} className="example-file">
              <span className="file-name">{name}</span>
              <pre className="code">{text}</pre>
            </div>
          ))}
          <p>
            The 10 column loop files of 3D-GNOME 2.0 and 3.0, with motif orientations and the factor in the loop file, are not read. Move the orientations to the anchors file and write one loop file per factor.
          </p>
          <p>
            To convert a .mcool map yourself, run prep_singletons.py from the gnome3d-ng repository. It writes intra chromosomal bin pairs at a chosen bin size and can thin a deep map.
          </p>
        </section>

        <section id="variants">
          <h2>Structural variants</h2>
          <p>
            Two hosted call sets need no upload. The 1000 Genomes set holds the 3,202 samples of the 30x phased panel, and the HGSVC set holds 35 samples of freeze 4. Choose the set, type a sample name such as HG00096, and pick a haplotype. The 1000 Genomes set covers chromosomes 1 to 22.
          </p>
          <p>
            Or upload a VCF compressed with bgzip. A record needs SVTYPE of DEL, INS, INV or DUP in its INFO field, and END or SVLEN. An insertion uses its ALT sequence when it has one. A duplication is read as tandem, with the copy right after the original. A gene inside it is listed twice, the second time as its second copy. CTCF motifs in that sequence can become new anchors. The result page counts the skipped records of other types. A phased genotype puts each allele on its own haplotype.
          </p>
          <pre className="code">{VCF}</pre>
          <p className="small">An uploaded file and everything built from it are private, reachable only through the job's link, and deleted after 14 days.</p>
        </section>

        <section id="reading">
          <h2>Reading the results</h2>
          <QA q="Bead unit" a="One bead unit is the target distance between two beads 1 kb apart, which is also the bond length. You can compare distances within a job and between jobs of one cell type." />
          <QA q="Genes missing from the table" a="The TSS or the nearest enhancer is more than 5 kb from every bead. Beads exist only in stretches with loops." />
          <QA q="Differences between structures" a="Each structure starts from a different random position. Parts of the region with many loops are similar across structures. Parts with few loops vary more." />
        </section>

        <section id="questions">
          <h2>Questions</h2>
          <QA q="Hi-C instead of ChIA-PET" a="You can call loops from Hi-C and upload them as your own loops, with the Hi-C as the contact background. Loops called from Hi-C are noisier than ChIA-PET loops, and structures built from them are noisier too." />
          <QA q="HiCDiffusion warning" a="HiCDiffusion was trained on GM12878. For other cell types its predicted map is outside its training data." />
          <QA q="My jobs on another computer" a="The list on the My jobs page is kept in the browser. A sync code from that page links the list with your other computers, and each of them then shows the same jobs. Type the code on the other computer, or open the link the page gives. Private jobs join the shared list only if you choose so on making the code." />
          <QA q="Job runtime" a="The runtime grows with the size of the region and the number of structures. A job waits in the queue while the GPUs are busy, and the job page shows its place in the queue." />
        </section>

        <section id="running">
          <h2>Running gnome3d-ng yourself</h2>
          <p>
            The server runs the gnome3d-ng Python package. The package runs on a CPU and uses a GPU when one is present. A run reads one config.ini. The settings reference lists every key with its default, the value on this server and its meaning.
          </p>
          <pre className="code">{'pip install gnome3d-ng\ngnome3d-ng --config config.ini --region chr1:11500000-12600000 -n 10 --out models/'}</pre>
          <table className="doc-table settings" aria-label="Settings">
            <thead>
              <tr>
                <th>Key</th>
                <th>Default</th>
                <th>Server</th>
                <th>Meaning</th>
              </tr>
            </thead>
            <tbody>
              {SETTINGS.map(([key, def, server, meaning]) => (
                <tr key={key}>
                  <td>
                    <code>{key}</code>
                  </td>
                  <td>
                    <code>{def}</code>
                  </td>
                  <td>
                    <code>{server}</code>
                  </td>
                  <td>{meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <a className="settings-link" href="https://github.com/SFGLab/3dgnome-ng/blob/main/SETTINGS.md">
            <BookOpen size={13} aria-hidden /> All 186 settings, from SETTINGS.md
          </a>
        </section>
      </div>
    </div>
  )
}
