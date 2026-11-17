# Uploads of your own data on the website

The design in web/design/site.pen offers uploads that the web app does not accept yet. Today the form takes one VCF and
nothing else. This plan lists the work that makes the design true.

## Uploads

| Upload | Format | Work |
|---|---|---|
| Loops, one file per factor | seven column BEDPE | Read and check each file, count loops in the region, between chromosomes and over 1 Mb, and write them as the job's cluster files with the factor names in `[data] factors`. Reject a ten column 3D-GNOME 3.0 file with the message in the design. |
| Anchors | BED with an optional orientation | Check that the orientation is on every line or on none. Without an anchors file, build anchors from the loop ends with orientation N. |
| Contacts | a link to .hic or .mcool, a local .hic read in the browser, or a seven column BEDPE of the region | See Large contact maps. |
| Enhancers | BED | Use in place of the cell type's enhancers for marks and the distance table. |
| Structural variants | VCF compressed with bgzip | Accepted today. Add the sample list from the header and the per variant table of changes to the loops shown in the dialog. |

## Large contact maps

A deep .hic or .mcool file is tens of GB. A job reads one window of at most 20 Mb at 25 kb, about 800 bins and a few
MB of bin pairs. The whole file never comes to the server. The dialog in web/design/site.pen, Model a region, your own
contacts, offers three routes.

| Route | Work |
|---|---|
| Link to a file | The worker reads the window's bin pairs at 25 kb over HTTP range requests, .hic with hic-straw, a new dependency, and .mcool with h5py on an fsspec file object, which needs a test before it is offered. A server without range requests is an error that offers the local route. A read has a timeout. |
| Local .hic file | The page reads the window with the hic-straw JavaScript library, as Juicebox web reads a local file, and uploads it as BEDPE. |
| BEDPE of the region | A direct upload. `gnome3d-ng extract`, built from the code of slurm/ensemble/prep_singletons.py, writes it from a local .mcool. |

Every route ends in the same bin pairs. A deep map is thinned to 5,000 contacts per Mb, the density of the hosted maps,
and the distance exponent is fitted on the chromosome of the region. The job keeps only these bin pairs, never the
original file.

A population VCF of several GB takes the link route too, bgzipped with its .tbi or .csi index. htslib reads the header
for the sample list and the records of the window over HTTP.

## Upload handling

The upload handler streams to disk in chunks and stops at a size cap set in the server settings, with the proxy's body
limit set to match. Today app.py reads the whole upload into memory and jobs.py hashes it with read_bytes. Both read in
chunks after this change, or hash the extracted window.

## Window limits

The form limits a region to 20 Mb, `MAX_REGION_MB` in the server settings. A 20 Mb job of 10 structures takes 80 to
90 s alone on the RTX 3090 and needs 0.65 GiB of GPU memory, design/plans/2026-11-12-web-gpu-throughput.md. With HiCDiffusion
as the contact background the limit is 2 Mb, the window HiCDiffusion predicts, and the form refuses a longer region.

## Checks and messages

Each file is checked on upload and the dialog shows one summary line per file. An error states the line, what was
found, and the fix, as in the design. A job does not start while any file has an error.

## Privacy

Uploaded files and everything built from them stay private to the job and are deleted after the retention period,
as uploaded VCFs are today.

## Tests

web/tests gains one test per upload kind with a small fixture of each format, including a ten column loop file, an
anchors file with orientation on some lines only, and a .mcool and a .hic of a few bins. The link route is tested
against a local HTTP server with range requests and one without.
