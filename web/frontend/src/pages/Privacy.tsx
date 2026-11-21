import '../styles/doc.css'
import './privacy.css'

const SECTIONS = [
  ['short', 'In short'],
  ['browser', 'In your browser'],
  ['server', 'On the server'],
  ['genetic', 'Genetic data'],
  ['rights', 'Controller and your rights'],
] as const

const STORED: [string, string, string][] = [
  ['Job requests', 'The region, the cell type, the loops and the settings of a job on hosted data. The link of a job is not secret. An example opens the same job for everyone.', 'no time limit'],
  ['Uploaded files and their jobs', 'A VCF, loops, anchors, contacts or enhancers that you upload and the job made from them. The job is private and reachable only through its link.', '14 days, then deleted'],
  ['Synced job lists', 'The links and the names of the jobs on a list that has a sync code. The links of private jobs are on it only if you chose so.', 'until you ask for removal'],
  ['Links to contact maps', 'The address of a contact map that you give in place of an upload. The server reads the file from that address.', 'with the job'],
  ['Server log', 'The IP address, the time and the address of each request, written by the web servers of the faculty for the operation and the security of the service.', 'set by the faculty'],
]

/** The privacy notice: what the site stores, where, for how long and the rights under the GDPR. */
export function Privacy() {
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
          <h1>Privacy</h1>
          <p className="lead">This notice says what the site stores about you and your data, for how long and what your rights are under the General Data Protection Regulation of the EU, the GDPR.</p>
        </header>

        <section id="short">
          <h2>In short</h2>
          <ul className="privacy-points">
            <li>The site sets no cookies and does no tracking or analytics.</li>
            <li>It asks for no name and no e-mail address and it has no accounts.</li>
            <li>Files that you upload are private and are deleted after 14 days.</li>
            <li>The pages load no fonts, scripts or images from other companies.</li>
          </ul>
        </section>

        <section id="browser">
          <h2>In your browser</h2>
          <p>
            The site keeps three things in the storage of your browser, on your own computer. They are the theme you chose, the list on the My jobs page and the sync code of that list if you made one. The pages need them to work. They stay on your computer and the list goes to the server only
            if you make a sync code. Clearing the site data in your browser removes all three.
          </p>
        </section>

        <section id="server">
          <h2>On the server</h2>
          <table className="privacy-table">
            <thead>
              <tr>
                <th>Data</th>
                <th>Content</th>
                <th>Kept for</th>
              </tr>
            </thead>
            <tbody>
              {STORED.map(([name, content, kept]) => (
                <tr key={name}>
                  <td className="privacy-name">{name}</td>
                  <td>{content}</td>
                  <td>{kept}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p>The data is used to compute the models you ask for and to show them to you. It is not sold, not shared with other parties and not used for anything else.</p>
        </section>

        <section id="genetic">
          <h2>Genetic data</h2>
          <p>
            A VCF of a person holds genetic data, a special category under the GDPR. Upload such a file only if you have the right to process it and prefer a file without a link to an identified person. The site reads the file to build your job and for nothing else and deletes it with the job
            after 14 days.
          </p>
        </section>

        <section id="rights">
          <h2>Controller and your rights</h2>
          <p>
            The controller of the data is Warsaw University of Technology, Politechnika Warszawska, Pl. Politechniki 1, 00-661 Warszawa, Poland. The site is run by the Laboratory of Functional and Structural Genomics at its Faculty of Mathematics and Information Science.
          </p>
          <p>
            You can ask for access to the data that concerns you, for its correction, for its deletion or for a limit on its use. You can object to its use. To have a job or a synced list deleted, send its link or its code. Write to the Data Protection Officer of the university at{' '}
            <a href="mailto:iod@pw.edu.pl">iod@pw.edu.pl</a>.
          </p>
          <p>You can also lodge a complaint with the President of the Personal Data Protection Office of Poland, UODO.</p>
        </section>
      </div>
    </div>
  )
}
