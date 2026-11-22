# Writing rules

These rules cover every text in the repository that a person reads. They apply while the text is written, not as a
later cleanup pass.

## Scope

The rules apply to these texts.

- Doc comments and inline comments, in Python, TypeScript and shell.
- Descriptions of settings and options, in `SETTINGS.md`, in the help texts of the website's settings and in the help
  of a command line.
- Error messages, log lines and the output of scripts.
- The pages of the website, `README.md`, `AGENTS.md` and the notes under `design/`.
- Titles, labels and legends of figures.
- Bullet points of slides.
- Commit messages.

Readers are researchers and engineers. Many of them are not native English speakers.

## Wording

Use simple, plain engineering English. A sentence has a plain subject, a verb and an object. Start with the thing or
with a verb, as in "Fails unless the map is balanced" or "The anchor that holds the motif".

Do not open with What, Where, How, Who, Whether, When, Which or Why. This applies to sentences, headings,
descriptions and parameter texts.

Do not invert a sentence and leave a preposition at its end. "The file the loader reads the loops from" is wrong.
Write "the file that holds the loops". Do not use clever constructions or a fronted fragment as a sentence.

Do not use metaphors or euphemisms. These words are banned.

- "sits", "lives", "lands". Say "is in" or "is located in".
- "hears", "tells", "told", "hands out", "hands on". Say "receives", "gets", "publishes" or "replies with".
- "rides", "climbs out".
- "name", "names", "named" and "naming" as a verb.
- "write" and "written" for the look of a value. Say "has the form". "Write" is for writing data.

Do not use a ", so" consequence clause, a "because" clause, "so that" or "unlike X" to justify a design. The reason
belongs in the commit message. A clause of this kind stays only when it carries a constraint the next editor would
break, as in "the kernel reads this array in place, so the caller must not resize it".

Do not put a comma before "and". Join the two parts without it. Write two sentences where the joined one reads badly.

Do not use capitals for emphasis. A word in capitals is an acronym or an identifier, not a stressed "not", "is" or
"all". Use backticks only for the identifiers that need them. Use few parentheses. An aside
becomes its own sentence or is dropped.

Do not write a compressed riddle. "A dropped loop stays arcless" needs the code to be understood. Spell out the
concrete case, as in "A loop the draw drops gets no spring in that conformation". Short text is good only when a
reader understands it without the code.

Leave out facts nobody needs. Do not argue against a misreading nobody would have. Use the common term, "contact map"
and not "pairwise frequency matrix".

## Doc comments and inline comments

Do not use dashes, colons or semicolons as sentence punctuation in a doc comment or an inline comment.

Comments are few. A comment or a doc exists only for a fact that the code and the names do not show. A function whose
name and signature say everything has no doc.

A doc says what the thing is and how to use it, usually in one or two sentences. It describes the code as it is now.
It does not hold any of these.

- Design reasons.
- Comparisons to sibling functions or types.
- References to callers or to other features.
- History, such as "now in", "used to" or "moved from".
- A comment left in place of removed code.
- Names of libraries, unless the reader must know the library to use the thing.
- A field by field description of a result that the type already shows.
- A mention of the reference implementation, of its files or of its functions. The model is its own code now.

History and design reasons go in the commit message. Longer reasoning goes in a note under `design/`.

A function doc leads with a verb, as in "Bins the contacts by anchor. Returns the matrix and the bin edges."

An inline comment carries a reason or a constraint that the code cannot show. A comment that restates the line below
it, a visible type or the doc of the called function is removed.

State a fact that holds across a file once, on the module or the class. Do not repeat it on every function or in
every test.

Keep the parameter and return descriptions, the Parameters and Returns blocks of a Python docstring and the `@param`
and `@returns` tags of a TypeScript doc. Reword inside them and do not delete them. Document every parameter of a
function or none.

A style pass is also a correctness pass. Check that a reworded doc is still true. Fix a doc that describes removed
behaviour or a stale module.

## Descriptions of settings and options

A description is one or two sentences. It says what the setting does and gives its default. A measured effect or a
warning that changes the choice of a user may follow in one more sentence.

The behaviour of one setting goes in its row of `SETTINGS.md` or in its help text. It does not get a section of its
own in a page.

## Error messages and log lines

An error message states the one case that happened, mentions the step and gives the fix, in one or two short
sentences. Explanations go in the documentation.

A message that covers several causes is split. Find the place in the code that knows the cause. Give each case its own
message there.

A log line states one event in the same plain wording. A fallback that changes a result is stated in a line that is
always visible, with the value in use.

## Pages and documents

These rules are for the pages of the website, `README.md`, `SETTINGS.md` and the notes under `design/`.

Write flowing paragraphs and address the reader as "you" on a page for users. Join related clauses, as in "A hosted
cell type needs no upload, while your own data is private to the job". Do not chop the text into one short sentence
per line.

Keep each sentence to one idea. Split a chain of "which", "so that" or trailing "-ing" clauses.

Say "does not" or "not" for a fact, not "never".

A heading is a plain noun phrase, such as "Feature flags" or "Contact map".

Use typical, concrete examples, a named gene or a real region, in place of abstract notions.

Stay compact. The behaviour of a field goes in a reference table. The prose adds only what the table cannot say.
Do not state a fact in the introduction and again in its own section.

A page for users does not describe internals, such as the layout of a job's directory, generated files or the order
in which settings resolve. A note under `design/` is the place for those.

Wrap a Markdown file at about 120 columns. A wrapped line must not start with "- ", since it would render as a list
item.

## Figures and slides

A title says what is plotted, on which data and for which comparison, as in "Contact maps of GM12878 chr1:20-25 Mb
from Hi-C, from ten cudaMMC structures and from ten 3D-GNOMEᴺᴳ structures". A title is not a sentence that argues a
conclusion. The conclusion goes in the legend or in the text.

A label uses words. "Higher is better" replaces an arrow. "From 0.22 to 0.37" replaces an arrow between two numbers. A
figure has no text arrows and no drawings made of characters.

A bullet point of a slide is a plain full statement. It is not a punchline.

A figure of a paper draws a comparison only when its test is under 0.05. The supplement draws every comparison and
marks the others ns.

## Names and terms

The tool is 3D-GNOMEᴺᴳ, with a raised NG, in every published text. `gnome3d` is the Python package and appears in
code, paths and commit messages only.

`legacy/3dnome/MC/` is "the reference" or "the reference implementation" in a design note. Do not call it by its
programming language. Code, comments, docstrings and messages under `gnome3d/`, `tests/` and `web/` do not mention it.

## Check before handing over

Reread new text against these rules while writing it. Then search the added lines for these.

- The openers What, Where, How, Who, Whether, When, Which and Why.
- A sentence that ends in "by", "under", "in", "on", "for", "with", "to" or "from".
- The banned words of the list above.
- ", so ", "so that", "because", "unlike" and ", and ".
- A dash, a colon or a semicolon inside a comment.

The text of the website is checked by `web/frontend/src/test/copy.ts` and the help texts of its settings by
`web/tests/test_engine_settings.py`. Comments and docstrings have no automatic check.
