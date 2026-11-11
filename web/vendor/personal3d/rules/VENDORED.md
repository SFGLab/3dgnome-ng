# Vendored code

Sadowski's rules for rewriting a chromatin interaction network by a person's structural
variants, from the 3D-GNOME 3.0 web server's copy of
bitbucket.org/4dnucleome/spatial_chromatin_architecture, doi 10.5281/zenodo.2837248, under the
BSD 2 clause licence in LICENSE.txt. Ideas 108 and 110 of design/expression-from-structure.md
use it through the personal3d package.

Changes from that copy. threeDgenome.py no longer imports the web server's unused region
parser, and SVintroduce.py, the server's wrapper, is left out. tools.py imports PyVCF and
pybedtools inside the two functions that use them, so the rules load without either. Everything
else is as it was. personal3d/rewrite.py is the adapter, and it replaces the motif call with a
vectorised one of the same output when it runs the rules.
