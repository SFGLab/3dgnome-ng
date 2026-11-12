# Uploads of your own data on the website

The design in web/design/site.pen offers uploads that the web app does not accept yet. Today the form takes one VCF and
nothing else. This plan lists the work that makes the design true.

## Uploads

| Upload | Format | Work |
|---|---|---|
| Loops, one file per factor | seven column BEDPE | Read and check each file, count loops in the region, between chromosomes and over 1 Mb, and write them as the job's cluster files with the factor names in `[data] factors`. Reject a ten column 3D-GNOME 3.0 file with the message in the design. |
| Anchors | BED with an optional orientation | Check that the orientation is on every line or on none. Without an anchors file, build anchors from the loop ends with orientation N. |
| Contacts | .hic, .mcool or seven column BEDPE | Convert .mcool with the code of slurm/ensemble/prep_singletons.py. Convert .hic with hic-straw, a new dependency, to the same bin pairs at 25 kb. Thin a deep map to the production density. |
| Enhancers | BED | Use in place of the cell type's enhancers for marks and the distance table. |
| Structural variants | VCF compressed with bgzip | Accepted today. Add the sample list from the header and the per variant table of changes to the loops shown in the dialog. |

## Checks and messages

Each file is checked on upload and the dialog shows one summary line per file. An error states the line, what was
found, and the fix, as in the design. A job does not start while any file has an error.

## Privacy

Uploaded files and everything built from them stay private to the job and are deleted after the retention period,
as uploaded VCFs are today.

## Tests

web/tests gains one test per upload kind with a small fixture of each format, including a ten column loop file, an
anchors file with orientation on some lines only, and a .mcool and a .hic of a few bins.
