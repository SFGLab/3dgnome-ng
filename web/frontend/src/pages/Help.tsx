import { DocPage } from '../components/DocPage'
import { ArrowRight, BookOpen, Download } from 'lucide-react'
import { Link } from 'react-router-dom'
import anchorsFile from '../assets/samples/anchors.bed?url'
import contactsFile from '../assets/samples/contacts.bedpe?url'
import ctcfFile from '../assets/samples/ctcf_loops.bedpe?url'
import enhancersFile from '../assets/samples/enhancers.bed?url'
import expressionFile from '../assets/samples/expression.tsv?url'
import rnapiiFile from '../assets/samples/rnapii_loops.bedpe?url'
import variantsFile from '../assets/samples/variants.vcf.gz?url'
import { hidden } from '../flags'
import { EXAMPLES as EXAMPLE_JOBS } from '../model/examples'
import { useOpenExample } from '../model/useOpenExample'
import { ROUTES } from '../routes'
import { scoped } from '../lib/css'
import sheet from './help.module.css'

const cn = scoped(sheet)

const FORMATS = [
  ['Loop anchors, BED', 'chr  start  end  orientation', 'Orientation of the CTCF motif, L, R or N. Use N when it is unknown and for RNAPII anchors. Give the orientation on every line or on none. The engine reads the first line to decide whether the file has it.'],
  ['Loops, BEDPE', 'chr1  start1  end1  chr2  start2  end2  count', 'One file per factor, CTCF first. The count is the number of PETs. Loops between chromosomes are skipped and a loop longer than 1 Mb goes to the coarse contact map.'],
  ['Contacts, BEDPE', 'chr1  start1  end1  chr2  start2  end2  count', 'Hi-C bin pairs or ChIA-PET singletons, with an integer count. The start column gives the position of each end. An uploaded .hic or .mcool map is converted to this format.'],
  ['Genes, GTF', 'standard GTF', 'Gene records give the gene name and the TSS, taken from the strand.'],
  ['Enhancers, BED', 'chr  start  end', 'One enhancer per line, for example from EnhancerAtlas.'],
  ['Segment split, BED', 'chr  position  position', 'Optional. Positions where the coarse segments split.'],
  ['Centromeres, BED', 'chr  start  end', 'One line per chromosome.'],
  ['Structural variants, VCF', 'standard VCF, bgzip', 'See the next section.'],
  ['Gene expression, table', 'gene  length  sample_1  sample_2', 'Optional. Read counts or normalised values of RNA-seq, tab or comma separated. The length column is optional. See the Gene expression section.'],
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

const EXPRESSION = `gene             length  control_1  control_2  treated_1  treated_2
ENSG00000120949  3667    1840       1722       95         121
ENSG00000116688  7402    311        290        302        335`

const SETTINGS = [
  ['target_bp_per_subanchor', '5000', '1000', 'genomic length of one chain segment, in bp'],
  ['polymer_exponent', '0', '0', 'distance law exponent, 0 fits it from the Hi-C'],
  ['background_weight', '0', '0.1', 'spring on pairs without a loop, under 100 kb apart'],
  ['solver', 'mc', 'lbfgs', 'method for the anchor fit'],
  ['hard_wall', 'no', 'yes', 'stops chains passing through each other'],
]

/** The sample input files: the data of the TNFRSF8 example, chr1:11,500,000-12,600,000 in GM12878, as files to upload. */
const SAMPLES: [string, string, string][] = [
  ['anchors.bed', anchorsFile, 'the loop anchors with their CTCF motif orientation'],
  ['ctcf_loops.bedpe', ctcfFile, '22 CTCF ChIA-PET loops'],
  ['rnapii_loops.bedpe', rnapiiFile, '118 RNAPII ChIA-PET loops'],
  ['contacts.bedpe', contactsFile, 'the Hi-C contacts of the region in 25 kb bins'],
  ['enhancers.bed', enhancersFile, '58 enhancers'],
  ['variants.vcf.gz', variantsFile, 'a deletion, an insertion and an inversion of one made up sample, as a format example'],
  ['expression.tsv', expressionFile, 'read counts of the genes of chromosome 1, two replicates of GM12878 and two of H1ESC'],
]

/** The steps of the tutorial, on the TNFRSF8 example. */
const STEPS = [
  'Open Model a region. In step 1 choose GM12878 and type TNFRSF8 into the region field. The site finds the gene and proposes a region around it. The loops of the region are drawn under the field. Drag the box or its two edges to move the region or to change its length.',
  'In step 2 choose CTCF and RNAPII. The structures then use both loop sets.',
  'Leave step 3 at none. A call set with a sample there gives the structure of one haplotype of that person.',
  'In step 4 keep the Hi-C of the cell type as the contact background.',
  'Step 5 is optional. Choose the ENCODE RNA-seq of GM12878 to see the expression of each gene beside its distances.',
  'In step 6 set the number of structures to 10 for a first look and press Build. The job page opens at once. It shows the stage of the job, the time each stage took and the place of the job in the queue. A job of this size takes about a minute.',
  'On the finished job the Structure tab shows one structure of the ensemble. Type TNFRSF8 into Find a gene. Its promoter is marked A and its nearest enhancer B. A card beside the structure gives the distance from A to B over all structures.',
  'The Genes tab lists every gene of the region with its nearest enhancer and their 3D distance. Sort it by a column or keep the genes that a loop joins to their enhancer.',
  'The Contact map tab draws the contacts of the structures against the Hi-C map of the cell type and gives the agreement of the two.',
  'Press Compare with another job and choose a job of the same region, for example one with CTCF loops alone. The two structures are drawn side by side. The tables and figures under them give the change of every gene.',
  'Download all saves the structures as mmCIF files together with the tables of the job.',
]

/** The parts of a job page, each with the things it shows. */
const GUIDE: [string, string][] = [
  ['Side panel', 'The panel on the left has the state of the job and the time it took, its region, its loops, its variants, its contact background and its expression. The structure field chooses one structure of the ensemble. Medoid is the structure nearest to all the others, Most compact the one with the smallest radius of gyration and Random one picked by chance. The files of the job are listed under it. Rename job gives the job a name in your list. New job from this request opens the form filled with the request of this job.'],
  ['Structure tab', 'The chain is drawn in 3D with promoters in orange and enhancers in green, or coloured by genomic position. Drag to turn it and scroll to zoom. A bead under the pointer shows its position. The loops of the region are drawn above the structure. A click on a loop marks its two anchors and the chain between them. Find a gene marks the promoter of a gene as A and its nearest enhancer as B, with their distance over all structures. Measure gives the distance between any two beads you click. The three buttons on the right reset the view, save it as a PNG and open it on the full screen.'],
  ['Genes tab', 'Each row is a gene of the region with its nearest enhancer, their separation along the genome and their 3D distance, the mean over the structures in bead units. The last column draws the distances of the single structures, a line from the least to the greatest with a mark at the median. A job with expression has a column for the level of each gene. The buttons above the table keep the genes joined to their enhancer by a loop, or by an RNAPII loop alone. The card beside the table has the histogram of the distances and their median.'],
  ['Contact map tab', 'The map has two halves, one above and one below the diagonal. By default the upper half is the Hi-C of the cell type and the lower half the contacts of the structures. Each half can also show the contacts within the median bond length, the HiCDiffusion prediction of a job that has one, or the loops. The map is drawn as observed contacts, as observed over expected, or as the Pearson correlation of its rows. Choose the bin size, zoom with the buttons or the scroll wheel and set the colour range in the card. The card Agreement of the two halves gives the Spearman correlation of bin pairs the same distance apart and the SCC. The camera button saves the map with its axes as a PNG.'],
  ['Loops tab', 'Every loop the engine read is a row with its factor, its two anchors, its span and its PET count. Show in the model marks the loop on the structure.'],
  ['Log tab', 'The log of the engine, with the fitted exponent of the distance law, the stages and their times.'],
  ['Files', 'A job has one mmCIF file for each structure, the table ep.tsv with the enhancer of every gene and their distance, marks.json with the promoters and enhancers marked on the structures. The log of the engine is engine.log. A job with expression has expression.tsv, the samples it used. A job with a HiCDiffusion background has the predicted map as hicdiffusion.bedpe. Download all saves them in one zip file.'],
  ['Comparison of two jobs', 'Compare with another job draws the structure of this job beside the structure of another job of your list or of a pasted job link. Under the structures the tables list the change of the enhancer to promoter distance of every gene, with the p of a Mann-Whitney U test over the structures of the two jobs. The Gene expression section above has the tables and figures of two jobs with expression.'],
]

const SECTIONS: [to: string, label: string][] = [
  ['tutorial', 'Tutorial'],
  ['formats', 'Input formats'],
  ['variants', 'Structural variants'],
  ['expression', 'Gene expression'],
  ['reading', 'Guide to the results'],
  ['questions', 'Questions'],
  ['running', 'Running gnome3d yourself'],
]

function QA({ q, a }: { q: string; a: string }) {
  return (
    <div className={cn('qa')}>
      <h3>{q}</h3>
      <p>{a}</p>
    </div>
  )
}

/** The finished jobs of the examples, as links: the sample output of the site. */
function SampleOutput() {
  const { open, busy, error } = useOpenExample()
  return (
    <>
      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
      <ul className={cn('sample-list')} aria-label="Sample output">
        {EXAMPLE_JOBS.map((e) => (
          <li key={e.id}>
            <button type="button" className={cn('sample-link')} disabled={!!busy} onClick={() => open(e)}>
              {busy === e.id ? 'Opening the job' : e.title} <ArrowRight size={13} aria-hidden />
            </button>
            <span>{`${e.cellLabel}. ${e.description}.`}</span>
          </li>
        ))}
      </ul>
    </>
  )
}

/** A tutorial, the input formats with sample files, the guide to the results and running the engine yourself. */
export function Help() {
  const trios = !hidden('trios')
  return (
    <DocPage title="Help" sections={SECTIONS} lead={<>This page has a tutorial with sample output, the input formats with sample files, a guide to the results and the way to run the engine on your own computer.</>}>

      <section id="tutorial">
        <h2>Tutorial</h2>
        <p className="intro">
          The tutorial builds the TNFRSF8 example from the form to the comparison of two jobs. Each step says what to choose. The two links open the form with the example filled in and the finished jobs of the example.
        </p>
        <p className={cn('tutorial-links')}>
          <Link className={cn('sample-link')} to={`${ROUTES.model}?example=TNFRSF8`}>
            The form with the example filled in <ArrowRight size={13} aria-hidden />
          </Link>
          <Link className={cn('sample-link')} to={ROUTES.examples}>
            All examples <ArrowRight size={13} aria-hidden />
          </Link>
        </p>
        <ol className={cn('steps')}>
          {STEPS.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ol>
        <h3 className={cn('example-head')}>Sample output</h3>
        <p>Each link opens a finished job of the site with every tab of its results. An example that nobody has opened before runs first, which takes about a minute.</p>
        <SampleOutput />
      </section>

      <section id="formats">
        <h2>Input formats</h2>
        <p className="intro">
          Input files are plain text with one record per line, separated by tabs or spaces, with positions in base pairs on GRCh38. Each cell type on the server has a full set and an upload replaces one file of the set.
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
        <h3 className={cn('example-head')}>Example, one CTCF loop and its anchors</h3>
        {EXAMPLES.map(([name, text]) => (
          <div key={name} className="example-file">
            <span className={cn('file-name')}>{name}</span>
            <pre className={cn('code')}>{text}</pre>
          </div>
        ))}
        <p>
          The 10 column loop files of 3D-GNOME 2.0 and 3.0, with motif orientations and the factor in the loop file, are not read. Move the orientations to the anchors file and write one loop file per factor.
        </p>
        <p>
          To convert a .mcool map yourself, run prep_singletons.py from the 3D-GNOMEᴺᴳ repository. It writes intra chromosomal bin pairs at a chosen bin size and can thin a deep map.
        </p>
        <h3 className={cn('example-head')}>Sample files</h3>
        <p>
          The files hold the data of the TNFRSF8 example, chr1:11,500,000-12,600,000 in GM12878. Enter that region in the form and upload the files in Your own data to repeat the example from your own files. The loops and the
          anchors are 4DN and ENCODE ChIA-PET, the contacts 4DN Hi-C, the enhancers EnhancerAtlas 2.0 and the read counts ENCODE RNA-seq. The Cite page has their accessions.
        </p>
        <ul className={cn('sample-list')} aria-label="Sample files">
          {SAMPLES.map(([name, url, about]) => (
            <li key={name}>
              <a className={cn('sample-link mono')} href={url} download={name}>
                <Download size={13} aria-hidden /> {name}
              </a>
              <span>{about}</span>
            </li>
          ))}
        </ul>
      </section>

      <section id="variants">
        <h2>Structural variants</h2>
        <p>
          Three hosted call sets need no upload. The 1000 Genomes set holds the 3,202 samples of the 30x phased panel and the HGSVC set holds 35 samples of freeze 4. Choose the set, type a sample name such as HG00096 and pick a haplotype. The 1000 Genomes set covers chromosomes 1 to 22.
        </p>
        <p>
          The SGDP set holds 277 people of the Simons Genome Diversity Project, named as the project names them, S_Adygei-1 for one. Their structural variants are imputed. The project calls single base variants alone. Each person therefore got the structural variants of the 1000 Genomes haplotypes that match theirs. The set keeps the 35,434 variants with an imputation quality, DR2, of 0.8 or more on chromosomes 1 to 22, between 3,400 and 5,400 a person. For 23 people who are in both projects, imputed without their own genomes in the panel, 98.5% of the genotypes agree with the measured ones, 95% of the measured variant alleles were imputed and 95% of the imputed ones were measured. For variants under 1% in the panel the two shares are 85% and 77%. Those 23 are from populations the panel has. Expect less for a person from a population it lacks.
        </p>
        <p>
          Or upload a VCF compressed with bgzip. A record needs SVTYPE of DEL, INS, INV or DUP in its INFO field and END or SVLEN. An insertion uses its ALT sequence when it has one. A duplication is read as tandem, with the copy right after the original. A gene inside it is listed twice, the second time as its second copy. CTCF motifs in that sequence can become new anchors. The result page counts the skipped records of other types. A phased genotype puts each allele on its own haplotype.
        </p>
        <pre className={cn('code')}>{VCF}</pre>
        <p className={cn('small')}>An uploaded file and everything built from it are private, reachable only through the job's link and deleted after 14 days.</p>
      </section>

      <section id="expression">
        <h2>Gene expression</h2>
        <p>
          Gene expression from RNA-seq is the one kind of data on the site that the structures do not depend on. Step 5 of the form adds it to a job and the step is optional. With it the Genes tab shows the expression of each gene
          beside its distances. The comparison of two jobs then shows the change of expression of each gene beside the change of its distance to the nearest enhancer.
        </p>
        <p>
          The server holds {trios ? 'four' : 'three'} sets. ENCODE has read counts of total RNA-seq for GM12878, H1ESC and HFFc6, with two or three replicates each. A job on a cell type takes the replicates of that cell type. The
          other {trios ? 'three' : 'two'} sets are of people. {trios && 'One has read counts of the nine people of three 1000 Genomes trios, a single library each. '}MAGE has TMM values of 731 people of the 1000 Genomes panel and
          GEUVADIS has RPKM of 462. A job takes the library of a person when the sample chosen in step 3 is in the set.
        </p>
        <p>
          The RNA-seq of every person was measured on a lymphoblastoid cell line, the cell type of GM12878. The form shows a warning when such a set is chosen beside H1ESC or HFFc6. The expression then comes from another cell type
          than the loops do.
        </p>
        <p>
          You can upload a table of your own in place of a hosted set, in the Expression step of the Your own data dialog. The table is tab or comma separated. Its first column holds the gene as an Ensembl id or a gene name. Every other column of numbers is a sample. A column named
          length gives the length of the gene in bp. The output of featureCounts is read as it is. After the upload choose the samples of the job and state that the values are read counts or normalised values.
        </p>
        <pre className={cn('code')}>{EXPRESSION}</pre>
        <p>
          The Genes tab shows one level for each gene, the mean over the samples of the job. Read counts with gene lengths give TPM and read counts without lengths give counts per million. Normalised values are shown in their own unit.
        </p>
        <p>
          Under the two structures the comparison has two tables with every gene of both jobs, one for the 3D distance and one for the expression, each on its own tab. The other job is compared against this one. Each change
          says in which job the value is higher or the distance longer. A row with a p of 0.05 or more is grey. A job measures a gene to its own nearest enhancer. Two cell types have two sets of enhancers. The table states both enhancers of a gene where they differ. The option Enhancers of both jobs only measures every gene to the nearest enhancer that the two jobs share. Beside the tables are three figures. The box plot shows the distance of the selected gene to its nearest enhancer in each structure of the two jobs, with the
          p of a Mann-Whitney U test. The second figure has a point for each gene of the region, with the change of its distance on the x axis and the change of its expression on the y axis, both as log2 of the other job over this
          one. The third is a volcano plot of the test, with the change of expression on the x axis and the adjusted p on the y axis as its negative log10. A click on a point or on a row of a table selects the gene in both
          structures.
        </p>
        <p>
          Two jobs with read counts and two samples or more each are compared with DESeq2, run by PyDESeq2. The test covers the genes with ten reads or more and gives a log2 fold change and an adjusted p for each. Every other pair
          gives a ratio of the two levels with no test. That holds for read counts with one sample on a side and for normalised values of one set. Read counts are not compared with normalised values. The values of two sets are not
          compared with each other.
        </p>
      </section>

      <section id="reading">
        <h2>Guide to the results</h2>
        <p className="intro">A finished job has a side panel and five tabs. The guide goes through them in the order of the page, then the files and the comparison of two jobs.</p>
        {GUIDE.map(([q, a]) => (
          <QA key={q} q={q} a={a} />
        ))}
        <QA q="Bead unit" a="One bead unit is the target distance between two beads 1 kb apart, which is also the bond length. You can compare distances within a job and between jobs of one cell type." />
        <QA q="Saving tables and figures" a="The CSV button beside a table saves all of its rows, also those on its other pages. The PNG button of a figure of the comparison saves the figure with its title. The camera button saves the view of a structure or the contact map as a PNG." />
        <QA q="Expression column" a="A job with gene expression shows one level for each gene in the Genes tab, the mean over its samples. A gene that the expression table lacks has an empty cell." />
        <QA q="Genes missing from the table" a="The TSS or the nearest enhancer is more than 5 kb from every bead. Beads exist only in stretches with loops." />
        <QA q="Differences between structures" a="Each structure starts from a different random position. Parts of the region with many loops are similar across structures. Parts with few loops vary more." />
      </section>

      <section id="questions">
        <h2>Questions</h2>
        <QA q="Hi-C instead of ChIA-PET" a="You can call loops from Hi-C and upload them as your own loops, with the Hi-C as the contact background. Loops called from Hi-C are noisier than ChIA-PET loops and structures built from them are noisier too." />
        <QA q="HiCDiffusion warning" a="HiCDiffusion was trained on GM12878. For other cell types its predicted map is outside its training data." />
        <QA q="My jobs on another computer" a="The list on the My jobs page is kept in the browser. A sync code from that page links the list with your other computers and each of them then shows the same jobs. Type the code on the other computer, or open the link the page gives. Private jobs join the shared list only if you choose so on making the code." />
        <QA q="Job runtime" a="The runtime grows with the size of the region and the number of structures. A job waits in the queue while the GPUs are busy and the job page shows its place in the queue." />
      </section>

      <section id="running">
        <h2>Running gnome3d yourself</h2>
        <p>
          The server runs the gnome3d Python package. The package runs on a CPU and uses a GPU when one is present. A run reads one config.ini. The settings reference lists every key with its default, the value on this server and its meaning. The package and the code of this site are free software under the GNU General Public License, version 3.
        </p>
        <pre className={cn('code')}>{'pip install gnome3d\ngnome3d --config config.ini --region chr1:11500000-12600000 -n 10 --out models/'}</pre>
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
        <a className={cn('settings-link')} href="https://github.com/SFGLab/3dgnome-ng/blob/main/SETTINGS.md">
          <BookOpen size={13} aria-hidden /> All settings, from SETTINGS.md
        </a>
      </section>
    </DocPage>
  )
}
