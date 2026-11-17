# Writing rules

These rules apply to the website text, guide pages, doc comments, inline comments, error messages, log lines,
script output and release notes. They came from the maintainer's rules for another project on 2026-11-12.

## Everywhere

Use simple, plain engineering English. Write short, direct statements with a plain subject, verb and object.

Do not open with What, Where, How, Who, Whether, When, Which or Why. This applies to headings, sentences, parameter
texts and descriptions.

Do not invert sentences or leave a preposition at the end, as in "the account blocks act under". Do not use Yoda
phrasing, clever constructions or fronted fragments as sentences.

Do not use metaphors or euphemisms. These words are banned.

- "sits", "lives", "lands". Say "is located in" or "is in".
- "hears", "tells", "hands on", "hand out". Say "receives", "gets", "replies with" or "publishes".
- "climbs out".
- "rides".

Do not use "name", "names" or "naming" as a verb. Use "write" only for writing data or the write permission, not for
how something looks.

Do not use ", so" consequence clauses or "because" clauses. The exception is a clause that carries a rule the next
editor would otherwise break.

Do not use capitals for emphasis. Use few backticks and few parentheses. Leave out facts nobody needs, and do not argue
against a misreading nobody would have. Use the common term, for example "map", not "one output per element".

Avoid wordy text and compressed riddles. Short text is good only when a reader understands it without the code.

## Doc comments and inline comments

Do not use dashes, colons or semicolons. A doc says what the thing is or does, usually in one sentence. It does not
cover design reasons, comparisons to sibling types, callers, links to other features, history, library names or a
field by field description of a result. History and design reasons go in the commit message.

A method doc leads with a verb, as in "Parses X. Returns Y." An inline comment only carries a reason the code cannot
show. A fact shared by every method is stated once on the class. Document every parameter or none, and reword
parameter and return docs rather than deleting them. When you reword a doc, check that the claim is still true.

## Guide pages and website text

Write flowing paragraphs and address the reader as "you". One plain joined clause per sentence is fine, and text
should not be broken into fragments. Say "does not" for facts, not "never". Headings are plain noun phrases, such as
"Block contents". Use named, typical examples. Do not repeat a fact between sections. Behaviour of a single field goes
in the reference table.

A description is one or two sentences that say what the field does and its default. An error message states the one
case that happened, the step, and the fix, in one or two sentences.

## Check before handing over

Reread new text against these rules while writing it. Search the added text for the wh-openers, endings in "by",
"under", "in", "on", "for", "with" or "to", the verbs "names", "write" and "hands", ", so " and "because".
