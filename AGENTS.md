# 3D-GNOME NG: agent guide

## The project

3D-GNOME NG predicts the 3D structure of chromosomes from ChIA-PET loops and Hi-C contacts. It returns an ensemble of
bead chains as mmCIF files, one bead per kilobase by default. The Python package is `gnome3d`. Its kernels run on
the CPU through numba and on a GPU through JAX. The site under `web/` runs the same package.

The model began as a port of the Monte Carlo core of 3D-GNOME and is its own code now. It keeps none of the settings
or distance laws of 3D-GNOME. It keeps the ability to reproduce the results of 3D-GNOME, judged by the validation
battery. The old code is under `legacy/` and is not developed.

## Layout

```
gnome3d/            the model
  simulate.py       entry points: simulate, run_region, run_genome
  cli.py            the gnome3d command
  production.py     the production configuration, CANONICAL and settings_for_cell
  settings.py       the loader of every config key
  data.py io.py     input data and file formats
  polymer.py        the distance law and its fits
  hierarchy.py      the cluster tree
  skeleton.py       the inputs of each block and the joint arcs solve
  reconstruct.py    the pipeline driver
  pipeline/         the DAG, the executors and the stages (coarse/, ib/, stitch, relax, holes)
  mc/numba/         CPU kernels and the arcs solver
  mc/jax/           GPU kernels
tests/              pytest suite, unit/ and integration/
web/                the site: gnome3d_web/ (FastAPI), frontend/ (React), scripts/, tests/
slurm/ensemble/     cluster and workstation run scripts and their configs
playground/         research scripts, the validation battery, figures
design/             public design notes and documentation, results under figures/
legacy/             3dnome/ (3D-GNOME, read only), harness/, validation/
data/               input datasets, not in git
docs/               private notes, not in git
SETTINGS.md         every config key, its default, its production value and its meaning
```

`CLAUDE.md` is a symlink to this file. Stage `AGENTS.md` in a commit.

## The model

### Input

A run reads loop anchors, loops per factor, contacts and segment breakpoints. Loops are PET clusters as BEDPE, one
file per factor, CTCF first. Contacts are singletons as BEDPE, Hi-C bin pairs in production. Optional inputs are a
compartment track, an anchor activity track, an anchor contact table and a contact map for the gaps. `gnome3d/data.py`
states the columns.

### Stages

The chromosome is a tree of segments, interaction blocks and anchors. A block is a run of anchors between two
positions that no loop covers.

1. Coarse. Places chromosomes, segments and block centroids by Monte Carlo on contact heatmaps. These stages share one
   random stream and run in a fixed order.
2. Arcs. Places the anchors. Production solves every anchor of a chromosome together with L-BFGS-B from a maximum
   entropy draw. The energy has loop springs, background springs for pairs without a loop, a gyration term over
   genomic windows and excluded volume.
3. Densify. Inserts subanchor beads between consecutive anchors, one per `target_bp_per_subanchor`.
4. Smooth. Monte Carlo on the bead chain of each block with the anchors fixed. The energy has bonds, angles, the CTCF
   motif orientation, excluded volume with a hard wall and confinement.
5. Assembly. Joins the blocks of a chromosome and runs the cross block relaxation. Hole beads and the boundary stitch
   are optional.

`design/features.md` has the record of every term and pass.

### Distance law

Every distance is in bead units. One bead is the distance at `target_bp_per_subanchor`. A pair with no contact is at
`max(1, (s / s0) ^ nu)`. A loop of strength `q` is at `1 + (background - 1) / (1 + q / q_half)`. A heatmap cell is at
the background times observed over expected contact to the minus third. `nu` is fitted on the singletons of the run at
load time. A run that cannot fit it logs the fallback.

### Monte Carlo rule

A move is accepted when it lowers the score, or with probability `exp(-jump_coef * (after / before) / T)`. The rule
divides by the running score. Every energy term must be non negative.

A kernel updates the score by local deltas. The local scores of a term must sum to its full score.

### Executors

Each stage has a setting `mc_executor_<stage>` with the values serial, threaded, batch and auto. Serial and threaded
run numba. Batch runs JAX. A node seeds its random streams from its own seed. Numba results do not depend on the
executor or the thread count. JAX kernels are float32 and are not bit identical across launch widths.

## Rules of the model

- Every physical constant is measured from the input of the run at load time or is a documented choice with its
  provenance beside it. A fallback is logged in a line that is always visible.
- A setting exists only when a run reads it. `SETTINGS.md` has a row for every key.
  `tests/unit/test_settings_doc.py` checks the rows against the loader, the defaults and the production config. Run
  it before a commit that touches settings.
- A feature is opt in while it is measured. It becomes the default when the battery says so and the old path is
  deleted.
- A change meant to be inert is proved inert with the byte exactness gate. A change meant to move results runs the
  battery and is adopted only when it wins there.
- Arms are judged on the pooled ensemble map, by stratum adjusted correlation and Spearman correlation against Hi-C,
  with the distance exponent and the overlap counts beside them.
- The accessibility and nucleus terms were removed on 2026-09-11 after null or harmful measurements. Do not
  reintroduce them without a new idea.
- The arcs annealer, `solver = mc`, is kept beside the solver by decision. Do not remove it.

## Working on the code

Use `.venv/bin/python`. pytest, ruff and pyright are in the `dev` extra.

### Tests

```bash
.venv/bin/pytest                      # every test, about two and a half minutes on a laptop
.venv/bin/pytest -m "not slow"        # without the files that take over a minute
.venv/bin/pytest tests/unit/test_polymer.py     # one file
.venv/bin/pytest web/tests            # the site, with web/requirements.txt and web/requirements-dev.txt installed
```

The tests are plain pytest: a test function, `assert` with a message that says what holds, fixtures for what
several tests share. No test file has a `main()`, a script entry or a helper that records results. `tests/unit/`
has one file per feature. `tests/integration/` runs the whole pipeline on synthetic inputs and on one GM12878
region, which is skipped without `data/`.

A new energy term gets a unit file after `tests/unit/test_terms.py`: a hand built configuration with an energy in
closed form, a test that the local scores sum to the full score and a test that the term is non negative.

Run pytest before a commit that touches `gnome3d/`.

### Byte exactness gate

`playground/parity_dump.py` reconstructs one region and dumps every bead. Run it at the previous commit and on the
working tree with the same executor settings and compare the two files exactly. `--uniform` dumps the uniform chain
on the production config.

```bash
git worktree add --detach /tmp/g3d_head HEAD
python playground/parity_dump.py /tmp/g3d_head head.npz
python playground/parity_dump.py "$PWD"       wip.npz
python playground/parity_dump.py --uniform /tmp/g3d_head head_uniform.npz
python playground/parity_dump.py --uniform "$PWD"       wip_uniform.npz
```

A worktree has its own `data/` with the files that git tracks. The first two runs read those files. The runs with
`--uniform` read `data/` of the directory they are started in, for both trees. A gate that reads the `data/` of each
tree compares two inputs: the worktree lacks the Hi-C singletons and fits another exponent. Compare the line
`polymer law` of the two logs before the bytes.

### Battery

`playground/validation_battery.py` scores structures against a Hi-C map per structure. `playground/ensemble_score.py`
scores the pooled map of an ensemble. `design/validation-metrics.md` defines the metrics.

### Types

`gnome3d/` is checked by pyright in strict mode, `.venv/bin/pyright`. The tree reports errors that predate the
pipeline. A change must not increase the count.

- Use the array aliases of `gnome3d/types.py`, `F32Array`, `F64Array`, `I32Array`, `I64Array` and `BoolArray`, and
  its collection aliases, in place of bare `np.ndarray` and inline `dict[str, list[...]]`.
- Annotate every signature. Parameterise `list`, `dict` and `tuple`.
- `gnome3d/mc/numba/terms.py` wraps `numba.njit` in a typed decorator. Use it for kernels.
- Do not call `typing.cast` inside an `@njit` kernel. Use a `# pyright: ignore[...]` comment on the line.
- A dataclass default uses a factory function with a name, as in `empty_cluster_index_list`.

### Format and lint

```bash
.venv/bin/ruff format gnome3d/ tests/
.venv/bin/ruff check  gnome3d/ tests/ --fix
```

Do not run `ruff format` on `web/` or `playground/`. Those files are not formatted to this width.

### Continuous integration and release

`.github/workflows/tests.yml` runs ruff and pytest on a push to main and on a pull request, on Python 3.11 to 3.14.
No job runs the tests marked slow, `tests/unit/test_batch_seeding.py`. Run them with `pytest -m slow` before a
change of the JAX kernels. `.github/workflows/web.yml` runs the tests of the site with
`pytest web/tests` and the tests, the lint and the build of the frontend when a change touches `web/`, `gnome3d/` or `pyproject.toml`.
`.github/workflows/publish.yml` builds the package and uploads it to PyPI when a GitHub release is published.

The version is the latest git tag, read by setuptools-scm. The tag `v0.3.0` gives the version 0.3.0. A commit after
the tag gets a dev version. `MANIFEST.in` keeps the source distribution to the package.

## Writing

`design/writing-rules.md` has the rules in full. They cover comments, docstrings, descriptions of settings, error
messages, log lines, the text of the site, the notes under `design/`, figures and commit messages. They apply while
the text is written.

- Plain subject, verb and object. Do not open a sentence or a heading with What, Where, How, Who, Whether, When,
  Which or Why.
- No metaphors. Banned words are sits, lives, lands, hears, tells, hands, rides, name as a verb and write for the
  look of a value.
- No ", so", "so that", "because" or "unlike X" to justify a design. The reason goes in the commit message.
- No comma before "and". No capitals for emphasis. Few backticks and few parentheses.
- Comments are few. A comment exists only for a fact that the code and the names do not show. It has no dashes,
  colons or semicolons as sentence punctuation.
- A doc says what the thing is in one or two sentences, as the code is now. A function doc leads with a verb.
- Code, comments and messages under `gnome3d/`, `tests/` and `web/` do not mention 3D-GNOME's code. A design note
  may, and calls it the reference.
- An error message states the one case that happened, the step and the fix.
- The tool is 3D-GNOMEᴺᴳ in published text. `gnome3d` is the package and appears in code, paths and commit messages.

Design goes to `design/` and is committed. A text that is personal, quotes a person or describes infrastructure goes
to `docs/`, which is private. A host, a path on a machine, an account and a deployment record are infrastructure. A
note under `design/` states hardware by its model.

## Records

- `SETTINGS.md` and `gnome3d/production.py` state what holds now.
- `design/features.md` has every feature with its settings, its code and its measurements.
- `design/anchor-placement.md` has the distance law, the placement of anchors and the end passes.
- `design/gap-fill.md` has the gaps, the maximum entropy start, the hole beads and the long range.
- `design/ab-compartments.md` has the compartment term and the steps that build its track.
- `design/kernel-performance.md`, `design/parallel-mc-and-nn-reading.md` and `design/intra-chain-parallelism.md` have
  the performance work.
- `design/algorithm-improvements.md` has the arcs solver.
- `design/rnapii-loops.md`, `design/2026-11-11-rnapii-release.md`, `design/2026-11-12-rnapii-figures.md` and
  `design/microc-loops.md` have the RNAPII work.
- `design/tools-comparison.md`, `design/validation-2026-09.md` and `design/validation-metrics.md` have the validation.
- `design/4dn-data-access.md` has the access to 4DN files.
- `design/website/` has the deployment, the data directory, the settings, the frontend and the jobs of the site.
- `legacy/README.md` has the retired code. `legacy/3dnome/` is read only.
- `data/memory/MEMORY.md` is the index of the project memory, a local folder outside git.
